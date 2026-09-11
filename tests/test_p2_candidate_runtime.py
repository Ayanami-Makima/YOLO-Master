"""Regression tests for silently inactive candidate-budget interventions."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from ultralytics import YOLO  # noqa: F401
from ultralytics.utils.loss import E2ELoss

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p1_candidate_runner", ROOT / "scripts/a1/run_p1_bn_frozen.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class DummyLoss:
    def __init__(self, model, tal_topk, tal_topk2=None):
        self.assigner = SimpleNamespace(topk=tal_topk, topk2=tal_topk2, conflict_metric="overlap")


def criterion_model():
    return SimpleNamespace(criterion=E2ELoss(None, loss_fn=DummyLoss))


@pytest.mark.parametrize("topk", [7, 10])
def test_runtime_reads_instantiated_topk(monkeypatch, topk):
    monkeypatch.setenv("A1_E2E_O2O_TAL_TOPK", str(topk))
    monkeypatch.setenv("A1_E2E_O2O_TAL_TOPK2", "1")
    payload = runner.validate_candidate_runtime(criterion_model(), {"a1_policy": {"o2o_tal_topk": topk}})
    assert payload["actual_topk"] == topk
    assert payload["actual_topk2"] == 1
    assert Path(payload["criterion_source"]).resolve() == ROOT / "ultralytics/utils/loss.py"


def test_inactive_intervention_fails_before_training(monkeypatch):
    monkeypatch.setenv("A1_E2E_O2O_TAL_TOPK", "7")
    monkeypatch.setenv("A1_E2E_O2O_TAL_TOPK2", "1")
    with pytest.raises(RuntimeError, match="candidate factor inactive"):
        runner.validate_candidate_runtime(criterion_model(), {"a1_policy": {"o2o_tal_topk": 10}})


def test_wrong_checkout_is_rejected(monkeypatch, tmp_path):
    import ultralytics

    monkeypatch.setattr(ultralytics, "__file__", str(tmp_path / "ultralytics/__init__.py"))
    with pytest.raises(RuntimeError, match="wrong Ultralytics checkout"):
        runner.validate_candidate_runtime(criterion_model(), {"a1_policy": {"o2o_tal_topk": 7}})
