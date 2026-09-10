"""Protect native forward/head gradients and the opt-in shared-feature gradient scale."""

import copy
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ultralytics.nn.modules.head import Detect


def make_head():
    torch.manual_seed(260829)
    head = Detect(nc=3, reg_max=1, end2end=True, ch=(8, 16, 32))
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    head.max_det = 20
    head.train()
    for m in head.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
    return head


def features():
    return [torch.randn(1, c, s, s, requires_grad=True) for c, s in ((8, 8), (16, 4), (32, 2))]


def measure(head, x):
    outputs = head(x)
    loss = sum(outputs["one2one"][k].square().mean() for k in ("boxes", "scores"))
    params = [p for n, p in head.named_parameters() if "one2one_" in n]
    grads = torch.autograd.grad(loss, x + params, allow_unused=True)
    return outputs, grads


def test_bridge_forward_and_gradient_scaling():
    head = make_head()
    x = features()
    native, native_grads = measure(head, x)
    head.p2_o2o_gradient_alpha = 0.0
    zero, zero_grads = measure(head, x)
    head.p2_o2o_gradient_alpha = 1.0
    full, full_grads = measure(head, x)
    head.p2_o2o_gradient_alpha = 0.1
    bridged, bridge_grads = measure(head, x)
    for branch in ("one2one", "one2many"):
        for key in ("boxes", "scores"):
            assert torch.equal(native[branch][key], zero[branch][key])
            assert torch.equal(native[branch][key], full[branch][key])
            assert torch.equal(native[branch][key], bridged[branch][key])
    assert all(g is None for g in native_grads[:3] + zero_grads[:3])
    for whole, partial in zip(full_grads[:3], bridge_grads[:3]):
        assert partial.abs().sum() > 0
        torch.testing.assert_close(partial, whole * 0.1, atol=1e-7, rtol=1e-5)
    for native_g, zero_g, bridge_g in zip(native_grads[3:], zero_grads[3:], bridge_grads[3:]):
        assert torch.equal(native_g, zero_g)
        assert torch.equal(native_g, bridge_g)


@pytest.mark.parametrize("export", [False, True])
def test_bridge_evaluation_and_export_unchanged(export):
    head = make_head().eval()
    head.export = export
    x = features()
    baseline = head(x)
    head.p2_o2o_gradient_alpha = 0.1
    result = head(x)
    assert torch.equal(baseline if export else baseline[0], result if export else result[0])


def test_bridge_checkpoint_roundtrip():
    head = make_head()
    head.p2_o2o_gradient_alpha = 0.1
    stream = io.BytesIO()
    torch.save(copy.deepcopy(head), stream)
    stream.seek(0)
    loaded = torch.load(stream, weights_only=False)
    assert loaded.p2_o2o_gradient_alpha == 0.1
    assert all(torch.equal(t, loaded.state_dict()[n]) for n, t in head.state_dict().items())
    _, grads = measure(loaded, features())
    assert all(g.abs().sum() > 0 for g in grads[:3])


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_bridge_rejects_invalid_alpha(alpha):
    head = make_head()
    head.p2_o2o_gradient_alpha = alpha
    with pytest.raises(ValueError, match="finite alpha"):
        head(features())


def test_channelwise_gain_summary_and_frozen_digest():
    path = Path(__file__).resolve().parents[1] / "scripts/a1/run_p2_gradient_bridge_pilot.py"
    spec = importlib.util.spec_from_file_location("bridge_runner_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    model = torch.nn.Module()
    model.base = torch.nn.Linear(4, 4)
    model.base.requires_grad_(False)
    model.gain = torch.nn.Parameter(torch.ones(1, 4, 1, 1))
    summary = runner.gain_summary(model)
    assert summary[0]["shape"] == [1, 4, 1, 1]
    assert summary[0]["abs_max"] == 1.0
    assert summary[0]["l2"] == 2.0
    before, keys = runner.frozen_digest(model)
    with torch.no_grad():
        model.gain.add_(1)
    assert runner.frozen_digest(model, keys)[0] == before
    with torch.no_grad():
        model.base.weight.add_(1)
    assert runner.frozen_digest(model, keys)[0] != before
    request = {"params": {"batch": 4}, "p2_bridge": {"train_images": 8}}
    trainer = SimpleNamespace(
        args=SimpleNamespace(batch=4), batch_size=4, train_loader=SimpleNamespace(batch_size=4, dataset=list(range(8)))
    )
    runner.validate_batch_budget(trainer, request)
    trainer.args.batch = 2
    with pytest.raises(RuntimeError, match="batch budget drift"):
        runner.validate_batch_budget(trainer, request)
    trainer.args.batch = 4
    request["p2_bridge"]["train_images"] = 16
    with pytest.raises(RuntimeError, match="dataset size drift"):
        runner.validate_batch_budget(trainer, request)
