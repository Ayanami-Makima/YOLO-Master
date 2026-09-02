# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, WEIGHTS_DIR, checks

# Shared test constants for model, config, data source, and environment info
MODEL = WEIGHTS_DIR / "path with spaces" / "yolo26n.pt"  # path with spaces to test path handling
CFG = "yolo26n.yaml"
SOURCE = ASSETS / "bus.jpg"
SOURCES_LIST = [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]  # file, dir, and glob patterns
CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()


def _task_model_data(task):
    """Return the shared model/data tuple for one task."""
    return (
        task,
        WEIGHTS_DIR / TASK2MODEL[task] if TASK2MODEL[task].endswith(".pt") else TASK2MODEL[task],
        TASK2DATA[task],
    )


# ``coco-multitask.yaml`` intentionally names a user-provided full COCO setup.
# Generic CLI/Python smoke tests must stay self-contained, while test_engine
# supplies its own temporary COCO-format fixture for the dedicated multitask
# resume case below.
TASK_MODEL_DATA = sorted(
    [
        _task_model_data(task)
        for task in TASKS
        if task != "multitask"
    ]
)  # (task, model, data) tuples
TASK_MODEL_DATA_ALL = sorted(_task_model_data(task) for task in TASKS)
# Generic prediction/export tests materialize entries under WEIGHTS_DIR, so
# they intentionally use downloadable checkpoints rather than YAML-only model
# definitions. The multitask YAML remains covered by its dedicated task tests.
MODELS = sorted([model for model in TASK2MODEL.values() if model.endswith(".pt")] + ["yolo11n-grayscale.pt"])
SOLUTION_ASSETS = {
    "demo_video": "solutions_ci_demo.mp4",
    "crop_video": "decelera_landscape_min.mov",
    "pose_video": "solution_ci_pose_demo.mp4",
    "parking_video": "solution_ci_parking_demo.mp4",
    "vertical_video": "solution_vertical_demo.mp4",
    "parking_areas": "solution_ci_parking_areas.json",
    "parking_model": "solutions_ci_parking_model.pt",
}

__all__ = (
    "CFG",
    "CUDA_DEVICE_COUNT",
    "CUDA_IS_AVAILABLE",
    "MODEL",
    "SOLUTION_ASSETS",
    "SOURCE",
    "SOURCES_LIST",
)
