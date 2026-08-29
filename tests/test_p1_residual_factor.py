"""Regression tests for the function-preserving A1 P1 residual adapter."""

import json
from types import SimpleNamespace

import pytest
import torch

from scripts.a1.build_p1_residual_factor_initializers import initialize_balanced_router_projections
from scripts.a1.run_p1_bn_frozen import (
    P1_ROUTING_PARAMS,
    R19_EXPLORATION_POLICY,
    R19_GAIN_POLICY,
    R19_ROUTING_SEMANTICS,
    factor_adapter_policy,
    freeze_batch_norm,
    freeze_residual_factor_bases,
    r19_noise_scale,
    read_request,
    routed_module_policy,
)
from ultralytics.nn.modules import C3k2ResidualFactor


def test_dense_adapter_is_exact_at_initialization_and_freezes_base():
    module = C3k2ResidualFactor(64, 64, n=1, moe=False).eval()
    inputs = torch.randn(2, 64, 16, 16)
    with torch.inference_mode():
        expected = module.base(inputs)
        actual = module(inputs)
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(module.gain) == 0
    assert all(not parameter.requires_grad for parameter in module.base.parameters())


def test_moe_adapter_uses_hard_top_k_from_first_step():
    module = C3k2ResidualFactor(64, 64, n=1, moe=True, num_experts=4, top_k=2)
    routers = [item for item in module.factor.modules() if hasattr(item, "progressive_sparsity")]
    assert routers
    assert all(item.progressive_sparsity is False for item in routers)
    assert all(item._current_top_k == 2 for item in routers)
    assert all(item.warmup_steps == 0 for item in routers)
    assert all(item.routing.noise_std == 0.0 for item in routers)
    assert all(item.expert_dropout_rate == 0.0 for item in routers)
    assert module.router_noise_std == 0.0
    assert module.expert_dropout_rate == 0.0
    assert module.p1_gain_lr_scale == 100.0
    assert module.p1_gain_no_warmup is True


def test_zero_expert_dropout_never_samples_a_drop_set(monkeypatch):
    module = C3k2ResidualFactor(64, 64, n=1, moe=True, num_experts=4, top_k=2).train()
    routed = next(item for item in module.factor.modules() if hasattr(item, "expert_dropout_rate"))

    def unexpected_randperm(*_args, **_kwargs):
        raise AssertionError("expert dropout sampled a drop set with rate=0")

    monkeypatch.setattr(torch, "randperm", unexpected_randperm)
    routed(torch.randn(2, routed.in_channels, 8, 8))


def test_noise_free_router_has_matching_train_and_eval_top2():
    module = C3k2ResidualFactor(64, 64, n=1, moe=True, num_experts=4, top_k=2)
    routed = next(item for item in module.factor.modules() if hasattr(item, "expert_dropout_rate"))
    routing = routed.routing
    inputs = torch.randn(4, routed.in_channels, 8, 8)

    routing.train()
    for child in routing.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()
    with torch.no_grad():
        training_indices = routing(inputs)[1]
    routing.eval()
    with torch.no_grad():
        evaluation_indices = routing(inputs)[1]

    assert torch.equal(training_indices, evaluation_indices)


def test_hard_top_k_training_still_advances_the_moe_step_clock():
    module = C3k2ResidualFactor(64, 64, n=1, moe=True, num_experts=4, top_k=2).train()
    routers = [item for item in module.factor.modules() if hasattr(item, "progressive_sparsity")]
    inputs = torch.randn(2, 64, 8, 8)
    with torch.no_grad():
        module(inputs)
    assert all(item._training_step == 1 for item in routers)
    with torch.no_grad():
        module(inputs)
    assert all(item._training_step == 2 for item in routers)


def test_balanced_router_projection_is_private_centered_equal_norm_and_deterministic():
    left = C3k2ResidualFactor(64, 64, n=1, moe=True, num_experts=4, top_k=2)
    right = C3k2ResidualFactor(64, 64, n=1, moe=True, num_experts=4, top_k=2)
    global_rng_before = torch.random.get_rng_state().clone()

    left_reports = initialize_balanced_router_projections(left.factor, base_seed=260829, layer_index=4)
    global_rng_after = torch.random.get_rng_state()
    right_reports = initialize_balanced_router_projections(right.factor, base_seed=260829, layer_index=4)

    assert torch.equal(global_rng_before, global_rng_after)
    assert len(left_reports) == len(right_reports) == 2
    assert [item["weight_sha256"] for item in left_reports] == [item["weight_sha256"] for item in right_reports]
    for report in left_reports:
        assert report["minimum_row_norm"] > 0
        assert report["row_norm_spread"] < 1e-6
        assert report["common_direction_max_abs"] < 1e-7
        assert report["channel_mean_max_abs"] < 1e-7
        assert report["minimum_distinct_row_distance"] > 0
        assert report["off_diagonal_gram_max_error"] < 1e-6
        assert report["observed_entry_rms"] == pytest.approx(0.05, abs=1e-7)


def test_balanced_router_initializer_only_changes_final_router_projections():
    module = C3k2ResidualFactor(64, 64, n=1, moe=True, num_experts=4, top_k=2)
    before = {name: value.detach().clone() for name, value in module.factor.state_dict().items()}
    reports = initialize_balanced_router_projections(module.factor, base_seed=260829, layer_index=4)
    after = module.factor.state_dict()

    changed = {name for name in before if not torch.equal(before[name], after[name])}
    expected = {f"{item['name']}.routing.router.3.weight" for item in reports}
    assert changed == expected
    assert len({item["weight_sha256"] for item in reports}) == len(reports)


def test_runner_reapplies_base_freeze_after_requires_grad_reset():
    model = C3k2ResidualFactor(64, 64, n=1, moe=False)
    for parameter in model.base.parameters():
        parameter.requires_grad = True
    trainer = SimpleNamespace(model=model)
    frozen = freeze_residual_factor_bases(trainer)
    assert frozen == sum(parameter.numel() for parameter in model.base.parameters())
    assert all(not parameter.requires_grad for parameter in model.base.parameters())


def test_runtime_policy_distinguishes_frozen_expert_bn_from_trainable_expert_weights():
    model = C3k2ResidualFactor(64, 64, n=1, moe=True, num_experts=4, top_k=2)
    freeze_batch_norm(SimpleNamespace(model=model))
    routed = routed_module_policy(model)

    assert routed
    assert all(item["expert_batch_norm_parameters"] > 0 for item in routed)
    assert all(item["trainable_expert_batch_norm_parameters"] == 0 for item in routed)
    assert all(
        item["expert_non_batch_norm_parameters"] == item["trainable_expert_non_batch_norm_parameters"]
        for item in routed
    )


@pytest.mark.parametrize("moe", [False, True])
def test_runtime_policy_keeps_factor_weights_and_gain_trainable_while_freezing_bn_and_base(moe):
    model = C3k2ResidualFactor(64, 64, n=1, moe=moe, num_experts=4, top_k=2)
    trainer = SimpleNamespace(model=model)
    freeze_batch_norm(trainer)
    freeze_residual_factor_bases(trainer)
    policy = factor_adapter_policy(model)

    assert len(policy) == 1
    adapter = policy[0]
    assert adapter["trainable_factor_batch_norm_parameters"] == 0
    assert adapter["factor_non_batch_norm_parameters"] == adapter["trainable_factor_non_batch_norm_parameters"]
    assert adapter["trainable_base_parameters"] == 0
    assert adapter["gain_parameters"] == adapter["trainable_gain_parameters"]


def test_runner_rejects_deterministic_routing_policy_drift(tmp_path):
    checkpoint = tmp_path / "initializer.pt"
    data = tmp_path / "coco.yaml"
    checkpoint.touch()
    data.touch()
    request = {
        "skill": "yolo.train",
        "inputs": {"model": str(checkpoint), "data": str(data)},
        "params": {"pretrained": True, **P1_ROUTING_PARAMS},
        "a1_policy": {
            "routing_semantics": "deterministic_hard_top2_from_step_zero",
            "expert_dropout_rate": 0.0,
        },
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    assert read_request(request_path) == request

    request["params"]["moe_noise_std"] = 0.5
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="routing policy drift"):
        read_request(request_path)


def test_r19_noise_schedule_has_locked_hold_and_decay_boundaries():
    policy = R19_EXPLORATION_POLICY
    assert r19_noise_scale(0, policy) == 1.0
    assert r19_noise_scale(625, policy) == 1.0
    assert 0.0 < r19_noise_scale(626, policy) < 1.0
    assert r19_noise_scale(1000, policy) == 0.0
    assert r19_noise_scale(1001, policy) == 0.0


def test_runner_accepts_only_the_exact_r19_exploration_and_gain_policy(tmp_path):
    checkpoint = tmp_path / "initializer.pt"
    data = tmp_path / "coco.yaml"
    checkpoint.touch()
    data.touch()
    params = {"pretrained": True, "seed": 260829, **P1_ROUTING_PARAMS}
    request = {
        "skill": "yolo.train",
        "inputs": {"model": str(checkpoint), "data": str(data)},
        "params": params,
        "a1_policy": {
            "routing_semantics": R19_ROUTING_SEMANTICS,
            "expert_dropout_rate": 0.0,
            "router_exploration": {
                **R19_EXPLORATION_POLICY,
                "base_seed": 260829,
                "enabled": True,
            },
            "factor_gain_optimizer": R19_GAIN_POLICY,
        },
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    assert read_request(request_path) == request

    request["a1_policy"]["router_exploration"]["decay_to_zero_microbatch"] = 999
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="exploration policy drift"):
        read_request(request_path)
