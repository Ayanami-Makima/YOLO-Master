"""Regression tests for keeping CI smoke assets small and self-contained."""

from tests import GENERIC_EXPORT_TASKS, MODELS, TASK_MODEL_DATA, TASK_MODEL_DATA_ALL
from tests.cache_test_assets import COMMON_WEIGHTS, DATASETS, NON_CACHEABLE_DATASETS
from ultralytics.cfg import TASK2DATA


def test_ci_asset_cache_excludes_user_provisioned_full_coco_multitask_data():
    """The generic cache must not try to download the 118k-image production dataset."""
    assert TASK2DATA["multitask"] in NON_CACHEABLE_DATASETS
    assert TASK2DATA["multitask"] not in DATASETS


def test_generic_smoke_matrix_excludes_full_coco_but_engine_matrix_keeps_multitask():
    """Generic lifecycle tests stay light while the dedicated resume test retains coverage."""
    assert "multitask" not in {task for task, _, _ in TASK_MODEL_DATA}
    assert "multitask" in {task for task, _, _ in TASK_MODEL_DATA_ALL}


def test_generic_model_matrix_contains_only_downloadable_checkpoints():
    """The generic tests materialize models under WEIGHTS_DIR and cannot use a YAML architecture path."""
    assert TASK2DATA["multitask"] == "coco-multitask.yaml"
    assert all(model.endswith((".pt", ".ts")) for model in MODELS)


def test_generic_export_matrix_excludes_full_coco_multitask():
    """Slow export/GPU matrices must not treat the production default as a tiny calibration dataset."""
    assert "multitask" not in GENERIC_EXPORT_TASKS


def test_asset_cache_weights_are_downloadable_release_artifacts():
    """Architecture YAMLs are source files, not release assets to pre-cache."""
    assert all(weight.endswith((".pt", ".ts")) for weight in COMMON_WEIGHTS)
