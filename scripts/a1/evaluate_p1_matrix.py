#!/usr/bin/env python3
"""Collect A1 evidence only after all four locked formal training runs finish.

Validation and ONNX export use the native Agent CLI dispatcher. Prediction and latency use the
same YOLO.predict API as inspect_p1.py / benchmark_p1.py because CLI output does
not expose suppression-kernel calls, raw boxes, or per-call wall-clock samples.
No framework source is patched. The default summary stage never runs a model.

Example (run on the training host, using a new output directory):
    python scripts/a1/evaluate_p1_matrix.py --configs configs/a1/full_recovery \
        --project /path/to/p1_recovery --output /path/to/evidence --stage predict
    # Reuse that output for the other independent stages and final summary.

ONNX evidence includes native fallback declarations, graph inspection, and
sampled FP32 eager-PyTorch/ONNX Runtime CPU agreement. It does not prove full-val
ONNX accuracy or preserved sparse execution. Export never changes a checkpoint.
The summary is evidence for review, never an automatic P1 acceptance decision.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import yaml

CELLS = "abcd"
STAGES = ("validate", "predict", "latency", "export")
LOCKED_SHA = "acce839c7e895d6b179de7f7093fa879e237cc7b"
LANE_CELLS = {"0": "ca", "1": "db"}


def digest(path: Path) -> str:
    """Hash an evidence file without loading it into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path) -> dict:
    """Read one structured evidence document."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    """Atomically write generated status; preserve immutable stage results separately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    """Turn a failed evidence condition into an explicit blocker."""
    if not condition:
        raise ValueError(message)


def checked_file(path: Path, expected_hash: str, label: str) -> str:
    """Require an existing, checksum-locked file."""
    require(path.is_file(), f"missing {label}: {path}")
    actual = digest(path)
    require(bool(expected_hash) and actual == expected_hash, f"{label} SHA-256 mismatch: {path}")
    return actual


def same_path(actual: str, expected: str) -> bool:
    """Compare native-host absolute paths, not path spelling."""
    return Path(actual).resolve() == Path(expected).resolve()


def path_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path is inside one explicit evidence root."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_full_lane_records(project: Path, protocol_hash: str) -> tuple[dict, dict]:
    """Require both formal lanes to have reached their runner-verified terminal state."""
    lanes, records = {}, {}
    for lane, expected_cells in LANE_CELLS.items():
        state_path = project / f"full_lane{lane}_state.json"
        state = read_json(state_path)
        require(state.get("phase") == "full", f"lane {lane}: state is not a formal full run")
        require(str(state.get("lane")) == lane, f"lane {lane}: state lane identity mismatch")
        require(state.get("status") == "completed", f"lane {lane}: formal lane did not complete successfully")
        require(state.get("protocol_sha256") == protocol_hash, f"lane {lane}: state uses another protocol")
        completed = state.get("completed")
        require(isinstance(completed, list), f"lane {lane}: verified records are missing")
        require(
            [record.get("cell") if isinstance(record, dict) else None for record in completed] == list(expected_cells),
            f"lane {lane}: verified cells must be exactly {list(expected_cells)} in runner order",
        )
        state_hash = digest(state_path)
        lanes[lane] = {
            "path": str(state_path),
            "sha256": state_hash,
            "cells": list(expected_cells),
            "status": "completed",
        }
        for record in completed:
            cell = record["cell"]
            require(cell not in records, f"lane {lane}: duplicate runner record for {cell}")
            records[cell] = {"lane": lane, "state_path": state_path, "state_sha256": state_hash, "record": record}
    require(set(records) == set(CELLS), "formal lane records do not cover exactly A/B/C/D")
    return lanes, records


def check_training_manifest(
    identity: dict,
    *,
    cell: str,
    lane: str,
    run_dir: Path,
    protocol_path: Path,
    protocol_hash: str,
    request_path: Path,
    request_hash: str,
    model_path: Path,
    model_hash: str,
    data_path: Path,
    data_hash: str,
    dataset_manifest: Path,
    dataset_manifest_hash: str,
    training: dict,
) -> None:
    """Bind the immutable pre-dispatch manifest to this exact formal cell."""
    require(isinstance(identity, dict), f"{cell}: training manifest is not an object")
    require(identity.get("schema_version") == 1, f"{cell}: unexpected training manifest schema")
    require(identity.get("status") == "prepared", f"{cell}: training manifest was not written before dispatch")
    require(identity.get("recorded_before_dispatch") is True, f"{cell}: manifest timing is not auditable")
    require(identity.get("phase") == "full", f"{cell}: training manifest is not formal full training")
    require(identity.get("cell") == cell, f"{cell}: training manifest cell mismatch")
    require(str(identity.get("lane")) == lane, f"{cell}: training manifest lane mismatch")
    require(same_path(identity["run_dir"], str(run_dir)), f"{cell}: training manifest run directory mismatch")
    require(
        same_path(identity["protocol_path"], str(protocol_path)), f"{cell}: training manifest protocol path mismatch"
    )
    require(identity.get("protocol_sha256") == protocol_hash, f"{cell}: training manifest uses another protocol")
    require(same_path(identity["request_path"], str(request_path)), f"{cell}: training manifest request path mismatch")
    require(identity.get("request_sha256") == request_hash, f"{cell}: training manifest uses another request")
    require(same_path(identity["model"], str(model_path)), f"{cell}: training manifest model mismatch")
    require(identity.get("model_sha256") == model_hash, f"{cell}: training manifest model hash mismatch")
    require(same_path(identity["data_yaml"], str(data_path)), f"{cell}: training manifest data mismatch")
    require(identity.get("data_yaml_sha256") == data_hash, f"{cell}: training manifest data hash mismatch")
    require(same_path(identity["formal_data_yaml"], str(data_path)), f"{cell}: formal data identity mismatch")
    require(identity.get("formal_data_yaml_sha256") == data_hash, f"{cell}: formal data hash mismatch")
    require(
        same_path(identity["dataset_manifest"], str(dataset_manifest)), f"{cell}: dataset manifest identity mismatch"
    )
    require(
        identity.get("dataset_manifest_sha256") == dataset_manifest_hash,
        f"{cell}: dataset manifest hash mismatch",
    )
    require(identity.get("training") == training, f"{cell}: training manifest settings mismatch")


def bind_runner_record(
    binding: dict,
    *,
    cell: str,
    project: Path,
    run_dir: Path,
    request_hash: str,
    identity_path: Path,
    identity_hash: str,
    epochs: int,
    last_epoch: dict,
    weights: dict,
) -> dict:
    """Cross-check current files against the lane runner's verified terminal record."""
    record, lane = binding["record"], binding["lane"]
    require(record.get("cell") == cell, f"{cell}: runner cell mismatch")
    require(same_path(record["run_dir"], str(run_dir)), f"{cell}: runner run directory mismatch")
    require(record.get("epochs") == epochs, f"{cell}: runner epoch count mismatch")
    require(record.get("request_sha256") == request_hash, f"{cell}: runner request hash mismatch")
    require(
        same_path(record["training_manifest"], str(identity_path)), f"{cell}: runner training manifest path mismatch"
    )
    require(record.get("training_manifest_sha256") == identity_hash, f"{cell}: runner training manifest hash mismatch")
    observed_last = record.get("last_epoch_metrics")
    require(isinstance(observed_last, dict), f"{cell}: runner last-epoch metrics are missing")
    normalized_last = {str(key).strip(): str(value).strip() for key, value in observed_last.items()}
    require(normalized_last == last_epoch, f"{cell}: runner last-epoch metrics mismatch")
    checkpoints = record.get("checkpoints")
    require(
        isinstance(checkpoints, dict) and set(checkpoints) == {"best.pt", "last.pt"},
        f"{cell}: runner checkpoint records are incomplete",
    )
    for filename, current in weights.items():
        verified = checkpoints[filename]
        require(isinstance(verified, dict), f"{cell}: malformed runner record for {filename}")
        require(same_path(verified["path"], current["path"]), f"{cell}: runner {filename} path mismatch")
        require(verified.get("sha256") == current["sha256"], f"{cell}: runner {filename} hash mismatch")
        require(verified.get("bytes") == current["bytes"], f"{cell}: runner {filename} size mismatch")
    logs = record.get("logs")
    require(isinstance(logs, list) and len(logs) == 2, f"{cell}: runner must verify exactly stdout/stderr logs")
    require(
        {Path(item.get("path", "")).name for item in logs if isinstance(item, dict)} == {"stdout.log", "stderr.log"},
        f"{cell}: runner log names mismatch",
    )
    log_root = project / "cli_logs" / f"{cell}_full"
    verified_logs = []
    for item in logs:
        require(isinstance(item, dict), f"{cell}: malformed runner log record")
        path = Path(item["path"])
        require(path_within(path, log_root), f"{cell}: runner log escapes its cell directory")
        log_hash = checked_file(path, item.get("sha256"), f"{cell} runner log")
        require(item.get("bytes") == path.stat().st_size, f"{cell}: runner log size mismatch")
        verified_logs.append({"path": str(path), "sha256": log_hash, "bytes": path.stat().st_size})
    return {
        "lane": lane,
        "state_path": str(binding["state_path"]),
        "state_sha256": binding["state_sha256"],
        "logs": verified_logs,
    }


def check_matrix(configs: Path, project: Path) -> dict:
    """Fail closed on missing, preflight, incomplete, altered, or unbound runs.

    This is an artifact gate, not a claim that a checkpoint has useful accuracy.
    Model loading and actual end-to-end behavior are checked in the run stages.
    """
    gate = {"status": "blocked", "blockers": [], "cells": {}}
    try:
        protocol_path = configs / "protocol.json"
        protocol = read_json(protocol_path)
        protocol_hash = digest(protocol_path)
        gate["protocol_sha256"] = protocol_hash
        gate["protocol"] = protocol
        require(protocol.get("locked_sha") == LOCKED_SHA, "unexpected public baseline SHA")
        require(same_path(protocol["run_root"], str(project)), "--project differs from the locked run_root")
        require(protocol.get("core_diff_from_lock") == "empty", "framework baseline diff needs explicit review")
        checked_file(Path(protocol["dataset_manifest"]), protocol["dataset_manifest_sha256"], "dataset manifest")
        checked_file(Path(protocol["data_yaml"]), protocol["data_yaml_sha256"], "data YAML")
        if protocol.get("evaluation_data_yaml"):
            checked_file(
                Path(protocol["evaluation_data_yaml"]),
                protocol["evaluation_data_yaml_sha256"],
                "evaluation data YAML",
            )
        if protocol.get("checkpoint"):
            checked_file(Path(protocol["checkpoint"]), protocol["checkpoint_sha256"], "pretrained checkpoint")
        common = protocol["common_training"]
        require(int(common["epochs"]) > 1, "one-epoch preflight cannot be promoted to formal evidence")
        require(common.get("fraction", 1.0) == 1.0, "formal data must not silently use a training fraction")
        lanes, lane_records = load_full_lane_records(project, protocol_hash)
        gate["lanes"] = lanes
        for cell in CELLS:
            try:
                request_path = configs / f"{cell}_full_request.json"
                request_hash = checked_file(
                    request_path, protocol["requests_sha256"][f"{cell}_full"], f"{cell} full request"
                )
                request = read_json(request_path)
                params, inputs = request["params"], request["inputs"]
                require(request["skill"] == "yolo.train", f"{cell}: not a native training request")
                require(same_path(params["project"], str(project)), f"{cell}: request project mismatch")
                name = params["name"]
                require(Path(name).name == name and "preflight" not in name.lower(), f"{cell}: invalid formal run name")
                require(same_path(inputs["data"], protocol["data_yaml"]), f"{cell}: not the locked formal dataset")
                matrix = protocol["matrix"][cell]
                require(matrix["moe"] == (cell in "cd"), f"{cell}: wrong MoE matrix assignment")
                require(matrix["end2end"] == (cell in "bd"), f"{cell}: wrong end-to-end matrix assignment")
                require(same_path(inputs["model"], matrix["path"]), f"{cell}: model path mismatch")
                checked_file(Path(inputs["model"]), matrix["sha256"], f"{cell} model YAML")
                for key, expected in common.items():
                    require(params.get(key) == expected, f"{cell}: request changed common setting {key}")
                run_dir = project / name
                require(run_dir.resolve().parent == project.resolve(), f"{cell}: run path escapes the project")
                actual_args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
                for key, expected in params.items():
                    actual = actual_args.get(key)
                    equal = str(actual) == str(expected) if key == "device" else actual == expected
                    if key == "project":
                        equal = actual is not None and same_path(str(actual), str(expected))
                    require(equal, f"{cell}: args.yaml changed {key}: {actual!r} != {expected!r}")
                for key in ("model", "data"):
                    require(same_path(str(actual_args[key]), inputs[key]), f"{cell}: args.yaml uses another {key}")
                identity_path = run_dir / "training_manifest.json"
                training_identity = read_json(identity_path)
                identity_hash = digest(identity_path)
                check_training_manifest(
                    training_identity,
                    cell=cell,
                    lane=lane_records[cell]["lane"],
                    run_dir=run_dir,
                    protocol_path=protocol_path,
                    protocol_hash=protocol_hash,
                    request_path=request_path,
                    request_hash=request_hash,
                    model_path=Path(inputs["model"]),
                    model_hash=matrix["sha256"],
                    data_path=Path(inputs["data"]),
                    data_hash=protocol["data_yaml_sha256"],
                    dataset_manifest=Path(protocol["dataset_manifest"]),
                    dataset_manifest_hash=protocol["dataset_manifest_sha256"],
                    training=params,
                )
                with (run_dir / "results.csv").open(encoding="utf-8", newline="") as stream:
                    rows = []
                    for row in csv.DictReader(stream):
                        require(
                            all(isinstance(key, str) and isinstance(value, str) for key, value in row.items()),
                            f"{cell}: malformed or partially written CSV row",
                        )
                        rows.append({key.strip(): value.strip() for key, value in row.items()})
                epochs = [int(row["epoch"]) for row in rows]
                require(epochs == list(range(1, int(params["epochs"]) + 1)), f"{cell}: incomplete/noncontiguous epochs")
                require("metrics/mAP50-95(B)" in rows[-1], f"{cell}: missing detection accuracy column")
                for row in rows:
                    for key, value in row.items():
                        if "loss" in key or key.startswith("metrics/"):
                            require(math.isfinite(float(value)), f"{cell}: non-finite {key} at epoch {row['epoch']}")
                weights = {}
                for filename in ("best.pt", "last.pt"):
                    path = run_dir / "weights" / filename
                    require(path.is_file() and path.stat().st_size > 0, f"{cell}: missing/empty {filename}")
                    weights[filename] = {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
                runner = bind_runner_record(
                    lane_records[cell],
                    cell=cell,
                    project=project,
                    run_dir=run_dir,
                    request_hash=request_hash,
                    identity_path=identity_path,
                    identity_hash=identity_hash,
                    epochs=len(rows),
                    last_epoch=rows[-1],
                    weights=weights,
                )
                gate["cells"][cell] = {
                    "run_dir": str(run_dir),
                    "request_sha256": request_hash,
                    "args_sha256": digest(run_dir / "args.yaml"),
                    "results_sha256": digest(run_dir / "results.csv"),
                    "training_manifest_sha256": identity_hash,
                    "weights": weights,
                    "runner": runner,
                    "epochs": len(rows),
                    "seed": params["seed"],
                    "last_epoch_mAP50_95": float(rows[-1]["metrics/mAP50-95(B)"]),
                    "checkpoint_selection": "best.pt from native trainer; last CSV row is not its validation result",
                }
            except (OSError, ValueError, KeyError, TypeError) as exc:
                gate["blockers"].append(f"{cell}: {exc}")
        require(not gate["blockers"], "all four formal runs must pass the artifact gate")
        gate["status"] = "ready"
        gate["identity"] = {"protocol_sha256": protocol_hash, "lanes": gate["lanes"], "cells": gate["cells"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        gate["blockers"].append(str(exc))
    return gate


def sample_statistics(samples: list[float]) -> dict:
    """Summarize finite raw samples with linearly interpolated quantiles."""
    require(bool(samples) and all(math.isfinite(value) for value in samples), "empty/non-finite samples")
    ordered = sorted(samples)

    def percentile(fraction):
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)

    return {
        "n": len(samples),
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "p50_ms": percentile(0.5),
        "p90_ms": percentile(0.9),
        "p99_ms": percentile(0.99),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def duplicate_boxes(boxes: list[list[float]], confidence: float, iou_threshold: float) -> dict:
    """Count same-class overlaps; a lower-scored box is counted at most once.

    Rows are [x1, y1, x2, y2, confidence, class]. This diagnostic does not modify
    model predictions and does not claim that every overlap is a false positive.
    """
    require(0 <= confidence <= 1 and 0 < iou_threshold <= 1, "invalid duplicate thresholds")
    require(all(len(row) == 6 and all(math.isfinite(value) for value in row) for row in boxes), "invalid box data")
    selected = sorted((row for row in boxes if row[4] >= confidence), key=lambda row: -row[4])
    pairs = 0
    same_class_pairs = 0
    redundant = set()
    for i, left in enumerate(selected):
        for j in range(i + 1, len(selected)):
            right = selected[j]
            if int(left[5]) != int(right[5]):
                continue
            same_class_pairs += 1
            overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
                0.0, min(left[3], right[3]) - max(left[1], right[1])
            )
            union = (
                max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
                + max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
                - overlap
            )
            if union > 0 and overlap / union >= iou_threshold:
                pairs += 1
                redundant.add(j)
    return {
        "boxes": len(selected),
        "duplicate_pairs": pairs,
        "same_class_pairs": same_class_pairs,
        "duplicate_boxes": len(redundant),
        "duplicate_box_rate": len(redundant) / len(selected) if selected else None,
        "duplicate_pair_rate": pairs / same_class_pairs if same_class_pairs else None,
    }


class NMSCallMonitor:
    """Observe real suppression kernels separately from the shared dispatch function.

    non_max_suppression is also called by NMS-free prediction, but its [B,N,6]
    early-return branch only filters scores. Counting that wrapper as NMS would
    be wrong. Warmup is run before this monitor because it uses synthetic NMS.
    """

    def __init__(self, nms_module=None, torchvision_ops=None):
        if nms_module is None:
            from ultralytics.utils import nms as nms_module
        if torchvision_ops is None and "torchvision" in sys.modules:
            torchvision_ops = sys.modules["torchvision"].ops
        self.nms = nms_module
        self.torchvision_ops = torchvision_ops
        self.kernels = Counter()
        self.events = []
        self.stack = ExitStack()
        self.patched_targets = []

    def __enter__(self):
        original = self.nms.non_max_suppression
        signature = inspect.signature(original)

        def dispatch(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs).arguments
            prediction = bound["prediction"]
            if isinstance(prediction, (list, tuple)):
                prediction = prediction[0]
            shape = list(prediction.shape)
            early = shape[-1] == 6 or bool(bound.get("end2end", False))
            self.events.append({"raw_shape": shape, "route": "end2end_score_filter" if early else "nms"})
            return original(*args, **kwargs)

        self.stack.enter_context(patch.object(self.nms, "non_max_suppression", dispatch))
        targets = [(self.nms.TorchNMS, name, f"TorchNMS.{name}") for name in ("nms", "fast_nms", "batched_nms")]
        if self.torchvision_ops is not None:
            targets.append((self.torchvision_ops, "nms", "torchvision.ops.nms"))
        for owner, name, label in targets:
            if not hasattr(owner, name):
                continue
            original_kernel = getattr(owner, name)

            def counted(*args, _original=original_kernel, _label=label, **kwargs):
                self.kernels[_label] += 1
                return _original(*args, **kwargs)

            self.stack.enter_context(patch.object(owner, name, counted))
            self.patched_targets.append(label)
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def report(self, end2end: bool, detections: int) -> dict:
        """Return an inconclusive result when NMS-on has no nonempty predictions."""
        calls = sum(self.kernels.values())
        routes = dict(Counter(event["route"] for event in self.events))
        if not self.events:
            status = "failed"
        elif end2end:
            status = "passed" if calls == 0 and set(routes) == {"end2end_score_filter"} else "failed"
        elif set(routes) != {"nms"}:
            status = "failed"
        else:
            status = "passed" if calls > 0 and detections > 0 else "inconclusive_no_suppression_observed"
        return {
            "status": status,
            "suppression_kernel_calls": calls,
            "kernel_calls": dict(self.kernels),
            "wrapper_calls": len(self.events),
            "wrapper_routes": routes,
            "events": self.events,
            "patched_targets": self.patched_targets,
            "warmup_excluded": True,
            "scope": "actual measured PyTorch predict calls; not validation/export tracing",
        }


def resolve_images(data_yaml: Path, image_list: Path | None, limit: int) -> list[Path]:
    """Choose one reproducible list for every cell; never download an image."""
    if image_list:
        paths = [image_list.resolve()]
    else:
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        root = Path(data.get("path", data_yaml.parent))
        if not root.is_absolute():
            root = data_yaml.parent / root
        entries = data["val"] if isinstance(data["val"], list) else [data["val"]]
        paths = [Path(entry) if Path(entry).is_absolute() else root / entry for entry in entries]
    images = []
    for path in paths:
        if path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = Path(line.strip())
                    images.append((item if item.is_absolute() else path.parent / item).resolve())
        elif path.is_dir():
            images.extend(
                sorted(item.resolve() for item in path.iterdir() if item.suffix.lower() in {".jpg", ".jpeg", ".png"})
            )
        else:
            images.append(path.resolve())
    images = list(dict.fromkeys(images))[:limit]
    require(bool(images) and all(path.is_file() for path in images), "missing/empty selected validation images")
    return images


def load_images(paths: list[Path]):
    """Load images before timing so filesystem I/O is not mistaken for inference."""
    import cv2

    result = []
    for path in paths:
        image = cv2.imread(str(path))
        require(image is not None, f"cannot decode image: {path}")
        result.append(image)
    return result


def prepare_runtime(threads: int, affinity: str | None, *, inspect_cuda: bool = True) -> dict:
    """Use explicit CPU settings and record only non-sensitive environment fields."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import cv2
    import torch
    import ultralytics

    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    if affinity:
        require(hasattr(os, "sched_setaffinity"), "explicit CPU affinity unsupported on this host")
        selected = {int(item) for item in affinity.split(",")}
        require(selected <= os.sched_getaffinity(0), "requested CPU affinity is outside the allowed CPU set")
        os.sched_setaffinity(0, selected)
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "ultralytics_path": ultralytics.__file__,
        "torch_threads": torch.get_num_threads(),
        "opencv_threads": cv2.getNumThreads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "gpu_inspection": "performed" if inspect_cuda else "not_requested_cpu_only_stage",
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if inspect_cuda
        else [],
        "gpu_memory_free_total_bytes_before": [
            list(torch.cuda.mem_get_info(index)) for index in range(torch.cuda.device_count())
        ]
        if inspect_cuda
        else [],
    }


def prediction_params(args, device: str) -> dict:
    """Fix every relevant inference setting across the four cells."""
    return {
        "imgsz": args.imgsz,
        "batch": 1,
        "device": device,
        "conf": args.conf,
        "iou": args.nms_iou,
        "max_det": args.max_det,
        "half": False,
        "rect": False,
        "augment": False,
        "agnostic_nms": False,
        "verbose": False,
        "save": False,
        "save_txt": False,
        "save_crop": False,
        "show": False,
    }


def load_model(gate: dict, cell: str):
    """Verify the trained model's actual head before collecting evidence."""
    from ultralytics import YOLO
    from ultralytics.nn.modules.moe.utils import is_core_moe_block

    model = YOLO(gate["cells"][cell]["weights"]["best.pt"]["path"])
    require(getattr(model, "task", None) == "detect", f"{cell}: checkpoint is not a detection model")
    head = model.model.model[-1]
    one2one = hasattr(head, "one2one_cv2") and hasattr(head, "one2one_cv3")
    routed = [module for module in model.model.modules() if is_core_moe_block(module) and hasattr(module, "routing")]
    require(
        bool(model.model.end2end) == (cell in "bd") and one2one == (cell in "bd"), f"{cell}: checkpoint head mismatch"
    )
    require(bool(routed) == (cell in "cd"), f"{cell}: checkpoint MoE mismatch")
    return model


def run_predict(gate: dict, args, paths: list[Path]) -> dict:
    """Collect raw detections, overlap diagnostics, and real NMS-kernel counts."""
    images = load_images(paths)
    result = {"status": "completed", "cells": {}, "settings": prediction_params(args, args.predict_device)}
    result["duplicate_iou"] = args.duplicate_iou
    for cell in CELLS:
        try:
            model = load_model(gate, cell)
            params = prediction_params(args, args.predict_device)
            model.predict(source=images[0], **params)  # native warmup includes unrelated synthetic NMS
            records = []
            with NMSCallMonitor() as monitor:
                for path, image in zip(paths, images):
                    prediction = model.predict(source=image, **params)[0]
                    boxes = prediction.boxes.data.detach().cpu().tolist()
                    records.append(
                        {
                            "image": str(path),
                            "boxes_xyxy_conf_class": boxes,
                            "duplicates": duplicate_boxes(boxes, args.conf, args.duplicate_iou),
                        }
                    )
                count = sum(item["duplicates"]["boxes"] for item in records)
                nms = monitor.report(cell in "bd", count)
            status = "completed" if nms["status"] == "passed" and count else "inconclusive"
            if nms["status"] == "failed":
                status = "failed"
            result["cells"][cell] = {
                "status": status,
                "images": records,
                "detections": count,
                "nms": nms,
                "images_with_duplicates": sum(item["duplicates"]["duplicate_pairs"] > 0 for item in records),
                "duplicate_boxes": sum(item["duplicates"]["duplicate_boxes"] for item in records),
                "note": "Same-class overlap is a diagnostic, not a ground-truth false-positive label.",
            }
        except Exception as exc:
            result["cells"][cell] = {"status": "failed", "error": str(exc)}
    if any(value["status"] != "completed" for value in result["cells"].values()):
        result["status"] = "partial"
    return result


def run_latency(gate: dict, args, paths: list[Path]) -> dict:
    """Measure native predict wall time and its native component timers, batch=1."""
    import torch

    images = load_images(paths)
    result = {
        "status": "completed",
        "devices": {},
        "warmup": args.warmup,
        "samples": args.samples,
        "scope": "RAM image -> preprocess -> inference -> postprocess -> Results; excludes disk I/O and model loading",
        "measurement_order": "device, then A/B/C/D; identical round-robin image order for each cell",
        "observer_overhead": "NMS monitor disabled during timing",
        "contention": "No other jobs are stopped. Check host/GPU activity before drawing speed conclusions.",
    }
    for device in args.devices:
        result["devices"][device] = {}
        for cell in CELLS:
            try:
                require(
                    device == "cpu" or (device.isdigit() and int(device) < torch.cuda.device_count()),
                    f"unavailable device: {device}",
                )
                model = load_model(gate, cell)
                params = prediction_params(args, device)
                for index in range(args.warmup):
                    model.predict(source=images[index % len(images)], **params)
                samples = []
                for index in range(args.samples):
                    if device != "cpu":
                        torch.cuda.synchronize(int(device))
                    started = time.perf_counter()
                    prediction = model.predict(source=images[index % len(images)], **params)[0]
                    if device != "cpu":
                        torch.cuda.synchronize(int(device))
                    total = (time.perf_counter() - started) * 1000.0
                    samples.append(
                        {
                            "sample": index,
                            "image_index": index % len(images),
                            "total_ms": total,
                            **{
                                f"{key}_ms": float(prediction.speed[key])
                                for key in ("preprocess", "inference", "postprocess")
                            },
                            "detections": len(prediction.boxes),
                        }
                    )
                result["devices"][device][cell] = {
                    "status": "completed",
                    "settings": params,
                    "samples": samples,
                    "statistics": {
                        key: sample_statistics([sample[f"{key}_ms"] for sample in samples])
                        for key in ("total", "preprocess", "inference", "postprocess")
                    },
                }
                del model
                if device != "cpu":
                    torch.cuda.empty_cache()
            except Exception as exc:
                result["devices"][device][cell] = {"status": "failed", "error": str(exc)}
                result["status"] = "partial"
    return result


def parse_native_validation_table(stdout: str) -> dict:
    """Parse exactly one completed native CLI `all` row when structured metrics are absent."""
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    pattern = re.compile(
        rf"^\s*all\s+(\d+)\s+(\d+)\s+({number})\s+({number})\s+({number})\s+({number})\s*$",
        re.MULTILINE,
    )
    matches = pattern.findall(stdout)
    require(len(matches) == 1, "missing or ambiguous completed native validation all-class row")
    images, instances = map(int, matches[0][:2])
    precision, recall, map50, map50_95 = map(float, matches[0][2:])
    values = (precision, recall, map50, map50_95)
    require(images > 0 and instances > 0, "invalid native validation population")
    require(all(math.isfinite(value) and 0 <= value <= 1 for value in values), "invalid native validation metrics")
    return {
        "images": images,
        "instances": instances,
        "metrics": {
            "metrics/precision(B)": precision,
            "metrics/recall(B)": recall,
            "metrics/mAP50(B)": map50,
            "metrics/mAP50-95(B)": map50_95,
        },
        "evaluation": {
            "images": images,
            "instances": instances,
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
        },
    }


def run_validate(gate: dict, args) -> dict:
    """Run native full validation with explicit device, never silently recover to CPU."""
    repo = Path(__file__).resolve().parents[2]
    dispatcher = repo / "agent/scripts/run_yolo_master_skill.py"
    native = args.output / "native_validation"
    native.mkdir(parents=True, exist_ok=False)
    result = {"status": "completed", "cells": {}, "nms_trace": "not instrumented in CLI validation"}
    for cell in CELLS:
        request = {
            "skill": "yolo.val",
            "request_id": f"{cell}_formal_best_validation",
            "runtime": {"cwd": str(repo), "python": sys.executable, "prefer_cli": True},
            "inputs": {
                "model": gate["cells"][cell]["weights"]["best.pt"]["path"],
                "data": gate["protocol"].get("evaluation_data_yaml", gate["protocol"]["data_yaml"]),
                "task": "detect",
            },
            "params": {
                "imgsz": args.imgsz,
                "batch": 1,
                "device": args.predict_device,
                "workers": 0,
                "split": "val",
                "conf": 0.001,
                "iou": args.nms_iou,
                "max_det": args.max_det,
                "half": False,
                "save_json": True,
                "plots": False,
                "verbose": False,
                "project": str(native),
                "name": cell,
                "exist_ok": False,
            },
            "artifacts": {"project": str(native / "manifests"), "name": cell},
            "policy": {"dry_run": False, "async": False},
        }
        request_path = native / f"{cell}_request.json"
        write_json(request_path, request)
        stdout, stderr = native / f"{cell}_stdout.json", native / f"{cell}_stderr.log"
        try:
            with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
                process = subprocess.run(
                    [sys.executable, str(dispatcher), "--request", str(request_path)],
                    cwd=repo,
                    stdout=out,
                    stderr=err,
                    check=False,
                )
            payload = read_json(stdout)
            require(process.returncode == 0 and payload.get("status") == "ok", f"native val failed; see {stdout}")
            require(
                not (payload.get("recovery") or {}).get("recovered"),
                "device fallback invalidates fixed-device validation",
            )
            metrics = payload.get("metrics", {})
            evaluation = payload.get("evaluation", {})
            metric = metrics.get("metrics/mAP50-95(B)")
            if metric is None:
                metric = evaluation.get("map50_95")
            source = "fresh native validation of formal best.pt"
            if metric is None:
                fallback = parse_native_validation_table(payload.get("logs", {}).get("stdout", ""))
                manifest = read_json(Path(gate["protocol"]["dataset_manifest"]))
                expected = manifest.get("splits", {}).get("val2017", {})
                require(
                    fallback["images"] == expected.get("images")
                    and fallback["instances"] == expected.get("boxes"),
                    "native validation table does not match the locked full COCO manifest",
                )
                metrics = fallback["metrics"]
                evaluation = {**evaluation, **fallback["evaluation"]}
                metric = fallback["evaluation"]["map50_95"]
                source = "fresh native validation; strict fallback from completed native CLI all-class row"
            require(
                metric is not None and math.isfinite(float(metric)) and 0 <= float(metric) <= 1, "missing/invalid mAP"
            )
            result["cells"][cell] = {
                "status": "completed",
                "map50_95": float(metric),
                "metrics": metrics,
                "evaluation": evaluation,
                "request": str(request_path),
                "request_sha256": digest(request_path),
                "stdout": str(stdout),
                "stderr": str(stderr),
                "manifest": payload.get("manifest"),
                "source": source,
            }
        except Exception as exc:
            result["cells"][cell] = {
                "status": "failed",
                "error": str(exc),
                "stdout": str(stdout),
                "stderr": str(stderr),
            }
            result["status"] = "partial"
    return result


class ExportBlocked(RuntimeError):
    """Represent a declared unsupported route without rewriting model semantics."""


def export_semantics(preflight: dict, moe: bool) -> dict:
    """Keep declared dispatch fallback separate from measured output agreement."""
    if preflight.get("format") != "onnx" or preflight.get("supported") is not True or preflight.get("errors"):
        raise ExportBlocked(f"native ONNX preflight refuses export: {preflight.get('errors', preflight)}")
    decisions = preflight.get("decisions", [])
    if bool(decisions) != moe:
        raise ExportBlocked("native preflight routed-module inventory does not match the A1 cell")
    for item in decisions:
        if item.get("supported") is not True or item.get("strategy") == "refuse":
            raise ExportBlocked(f"native preflight refuses {item.get('module')}: {item.get('known_error')}")
        if item.get("backend") != "onnx" or item.get("module_family") != "MoE":
            raise ExportBlocked("unexpected routed family/backend needs an explicit export review")
        if item.get("strategy") not in {"dynamic", "dense_fallback", "routing_preserved"}:
            raise ExportBlocked("merged or unknown export strategy is outside this unchanged A1 matrix")
        if not isinstance(item.get("dense_fallback"), bool):
            raise ExportBlocked("native preflight did not declare whether sparse dispatch changes")
        if item["dense_fallback"] != (item["strategy"] == "dense_fallback"):
            raise ExportBlocked("inconsistent native strategy/dense-fallback declaration")
    fallback = any(item["dense_fallback"] for item in decisions)
    return {
        "MoE_semantics": "not_applicable_dense_model"
        if not moe
        else ("declared_dense_expert_execution_with_routing_selection" if fallback else "declared_routing_export"),
        "dense_fallback": fallback,
        "sparse_dispatch_preserved": False if fallback else "not_verified" if moe else "not_applicable",
        "output_equivalence": "not_yet_measured",
        "declarations": decisions,
        "interpretation": (
            "Native ONNX tracing computes all experts before Top-K gather/weighted sum. "
            "Even numerically matching outputs do not preserve sparse skipping or its latency."
            if fallback
            else "Output equivalence must still be measured on the same inputs."
        ),
    }


def parse_onnx_metadata(entries) -> dict:
    """Decode exporter metadata safely: native exporter serializes Python literals."""
    metadata = {}
    for entry in entries:
        if entry.key in metadata:
            raise ExportBlocked(f"duplicate ONNX metadata key: {entry.key}")
        try:
            value = ast.literal_eval(entry.value)
        except (ValueError, SyntaxError):
            value = entry.value
        metadata[entry.key] = value
    return metadata


def walk_onnx_nodes(graph, scope="graph"):
    """Inspect subgraphs and function bodies as well as the top-level graph."""
    for index, node in enumerate(graph.node):
        path = f"{scope}/{node.name or index}"
        yield path, node
        for attribute in node.attribute:
            if attribute.type == 5:  # AttributeProto.GRAPH, avoid importing ONNX for pure tests
                yield from walk_onnx_nodes(attribute.g, f"{path}/{attribute.name}")
            elif attribute.type == 10:  # AttributeProto.GRAPHS
                for number, child in enumerate(attribute.graphs):
                    yield from walk_onnx_nodes(child, f"{path}/{attribute.name}/{number}")


def inspect_onnx_graph(model_proto, preflight: dict, *, cell: str, imgsz: int, max_det: int) -> dict:
    """Verify fixed-shape, explicit head semantics, fallback metadata and NMS ops.

    Absence of ONNX NonMaxSuppression is not sufficient for A/C to be NMS-free:
    those exports intentionally expose raw one-to-many outputs for external NMS.
    """
    semantics = export_semantics(preflight, cell in "cd")
    metadata = parse_onnx_metadata(model_proto.metadata_props)
    expected_end2end = cell in "bd"
    if metadata.get("end2end") is not expected_end2end:
        raise ExportBlocked("ONNX metadata changed or omitted the locked end2end head")
    if metadata.get("batch") != 1 or metadata.get("imgsz") not in ([imgsz, imgsz], (imgsz, imgsz)):
        raise ExportBlocked("ONNX metadata does not declare the fixed batch=1 image size")
    export_args = metadata.get("args", {})
    if not isinstance(export_args, dict) or export_args.get("nms") is not False:
        raise ExportBlocked("ONNX metadata does not explicitly confirm nms=False")
    embedded = metadata.get("mixture_export_preflight")
    if cell in "cd":
        if not isinstance(embedded, dict):
            raise ExportBlocked("MoE ONNX artifact lacks native fallback/preflight metadata")
        embedded_semantics = export_semantics(embedded, True)
        fields = ("module", "module_type", "module_family", "backend", "strategy", "dense_fallback", "supported")

        def declaration_key(item):
            return tuple(str(item.get(key)) for key in fields)

        if sorted(map(declaration_key, embedded["decisions"])) != sorted(map(declaration_key, preflight["decisions"])):
            raise ExportBlocked("ONNX fallback metadata differs from native preflight of the source checkpoint")
        if embedded_semantics["dense_fallback"] != semantics["dense_fallback"]:
            raise ExportBlocked("ONNX artifact changed the declared fallback")
    elif embedded and embedded.get("decisions"):
        raise ExportBlocked("dense control unexpectedly gained routed-module export metadata")

    nodes = list(walk_onnx_nodes(model_proto.graph))
    for function in getattr(model_proto, "functions", []):
        nodes.extend(walk_onnx_nodes(function, f"function/{function.domain}/{function.name}"))
    nms_nodes = [
        {"path": path, "op_type": node.op_type, "domain": node.domain}
        for path, node in nodes
        if node.op_type == "NonMaxSuppression" or "nms" in node.op_type.lower()
    ]
    inputs = list(model_proto.graph.input)
    if len(inputs) != 1:
        raise ExportBlocked("ONNX graph does not expose exactly one input")
    input_type = inputs[0].type.tensor_type
    input_shape = [dim.dim_value or dim.dim_param or None for dim in input_type.shape.dim]
    if input_shape != [1, 3, imgsz, imgsz] or input_type.elem_type != 1:
        raise ExportBlocked(f"ONNX input is not fixed FP32 [1,3,{imgsz},{imgsz}]: {input_shape}")
    if len(model_proto.graph.output) != 1:
        raise ExportBlocked("detect export must expose exactly one output")
    if model_proto.graph.output[0].type.tensor_type.elem_type != 1:
        raise ExportBlocked("ONNX output is not FP32 under the fixed export protocol")
    for initializer in model_proto.graph.initializer:
        if initializer.data_location == 1:  # TensorProto.EXTERNAL
            raise ExportBlocked("external ONNX weight files need a separately verified artifact bundle")
    return {
        "status": "failed" if nms_nodes else "passed",
        "nms_nodes": nms_nodes,
        "NonMaxSuppression_count": sum(node.op_type == "NonMaxSuppression" for _, node in nodes),
        "graph_scan_scope": "main graph, every nested graph and local function body",
        "node_count": len(nodes),
        "op_counts": dict(Counter(node.op_type for _, node in nodes)),
        "input_name": inputs[0].name,
        "input_shape": input_shape,
        "input_dtype": "float32",
        "output_name": model_proto.graph.output[0].name,
        "end2end": expected_end2end,
        "nms_export_argument": False,
        "output_semantics": f"one-to-one [1,{max_det},6], xyxy/confidence/class, no IoU suppression"
        if expected_end2end
        else "one-to-many raw BCN, xywh/class probabilities; external NMS is still required",
        "native_preflight_metadata_checked": True,
        **semantics,
    }


def compare_export_outputs(reference, candidate, *, end2end: bool, atol: float, rtol: float) -> dict:
    """Compare raw outputs, permitting only a one-to-one permutation for tied E2E rows.

    Discrete class IDs must match exactly. Every row is compared, including low
    confidence rows; no NMS or confidence filtering hides export differences.
    """
    import numpy as np

    reference, candidate = np.asarray(reference), np.asarray(candidate)
    result = {
        "status": "failed",
        "reference_shape": list(reference.shape),
        "onnx_shape": list(candidate.shape),
        "atol": atol,
        "rtol": rtol,
        "confidence_filter": None,
        "posthoc_nms": False,
    }
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[0] != 1:
        return {**result, "error": "output shape mismatch"}
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return {**result, "error": "non-finite output"}
    if end2end and reference.shape[-1] != 6:
        return {**result, "error": "end-to-end output must have six columns"}
    raw_error = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    result.update(
        raw_order_max_abs_error=float(raw_error.max(initial=0)),
        raw_order_mean_abs_error=float(raw_error.mean()) if raw_error.size else 0.0,
    )
    direct = bool(np.allclose(reference, candidate, atol=atol, rtol=rtol))
    if not end2end:
        return {
            **result,
            "status": "passed" if direct else "failed",
            "comparison": "fixed anchor order",
            "allclose": direct,
        }

    expected, actual = reference[0], candidate[0]
    if (
        not np.equal(expected[:, 5], np.rint(expected[:, 5])).all()
        or not np.equal(actual[:, 5], np.rint(actual[:, 5])).all()
    ):
        return {**result, "error": "non-integral class IDs in end-to-end output"}
    classes_equal = bool(np.equal(expected[:, 5], actual[:, 5]).all())
    if direct and classes_equal:
        return {**result, "status": "passed", "comparison": "same row order", "allclose": True, "class_ids_equal": True}
    allowed = np.isclose(expected[:, None, :5], actual[None, :, :5], atol=atol, rtol=rtol).all(axis=-1)
    allowed &= expected[:, None, 5] == actual[None, :, 5]
    neighbors = [np.flatnonzero(row).tolist() for row in allowed]
    matched_candidate = [-1] * len(actual)

    def assign(row, seen):
        for column in neighbors[row]:
            if column in seen:
                continue
            seen.add(column)
            if matched_candidate[column] < 0 or assign(matched_candidate[column], seen):
                matched_candidate[column] = row
                return True
        return False

    matched_rows = sum(assign(row, set()) for row in range(len(expected)))
    result.update(
        comparison="class-preserving one-to-one row permutation",
        allclose=direct,
        matched_rows=matched_rows,
        expected_rows=len(expected),
        class_ids_equal=classes_equal,
    )
    if matched_rows != len(expected):
        return result
    permutation = [-1] * len(expected)
    for column, row in enumerate(matched_candidate):
        permutation[row] = column
    aligned_error = np.abs(expected.astype(np.float64) - actual[permutation].astype(np.float64))
    return {
        **result,
        "status": "passed",
        "permutation": permutation,
        "aligned_max_abs_error": float(aligned_error.max(initial=0)),
        "aligned_mean_abs_error": float(aligned_error.mean()) if aligned_error.size else 0.0,
    }


def export_request(checkpoint: Path, destination: Path, cell: str, args) -> dict:
    """Build an explicit CPU-only native export request, preserving every head mode."""
    repo = Path(__file__).resolve().parents[2]
    return {
        "skill": "yolo.export",
        "request_id": f"{cell}_formal_best_onnx",
        "runtime": {"cwd": str(repo), "python": sys.executable, "prefer_cli": True},
        "inputs": {"model": str(checkpoint), "task": "detect"},
        "params": {
            "format": "onnx",
            "imgsz": args.imgsz,
            "batch": 1,
            "device": "cpu",
            "half": False,
            "dynamic": False,
            "simplify": False,
            "opset": 17,
            "nms": False,
            "end2end": cell in "bd",
            "max_det": args.max_det,
            "agnostic_nms": False,
            "pre_export_prune": False,
            "project": str(destination.parent),
            "name": destination.name,
        },
        "artifacts": {"project": str(destination / "agent_manifest"), "name": "export"},
        "policy": {"dry_run": False, "async": False},
    }


def run_export(gate: dict, args, paths: list[Path]) -> dict:
    """Export via native CLI and compare eager PyTorch with ORT CPU on identical tensors."""
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch

        from ultralytics.data.augment import LetterBox
        from ultralytics.utils.export_preflight import export_preflight
    except ImportError as exc:
        return {"status": "blocked", "error": f"missing export dependency (no installation attempted): {exc}"}
    require(args.imgsz == 640, "A1 ONNX agreement is locked to 640x640")
    if "CPUExecutionProvider" not in ort.get_available_providers():
        return {"status": "blocked", "error": "ONNX Runtime CPUExecutionProvider is unavailable"}
    cli_path = shutil.which("yolo", path=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    if cli_path is None:
        return {"status": "blocked", "error": "native yolo CLI unavailable; no bootstrap/install is attempted"}
    native = args.output / "native_export"
    native.mkdir(parents=True, exist_ok=False)
    input_dir = native / "inputs"
    input_dir.mkdir()
    arrays, input_records = [], []
    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
    for index, (path, image) in enumerate(zip(paths, load_images(paths))):
        resized = letterbox(image=image)
        array = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        require(array.shape == (1, 3, 640, 640), "export comparison preprocessing changed fixed input shape")
        tensor_path = input_dir / f"{index:04d}.npy"
        np.save(tensor_path, array, allow_pickle=False)
        arrays.append(array)
        input_records.append(
            {
                "image": str(path),
                "image_sha256": digest(path),
                "input_tensor": str(tensor_path),
                "input_tensor_sha256": digest(tensor_path),
                "shape": list(array.shape),
                "dtype": "float32",
            }
        )
    result = {
        "status": "completed",
        "cells": {},
        "inputs": input_records,
        "dependencies": {"onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__},
        "device": "cpu",
        "provider": "CPUExecutionProvider",
        "batch": 1,
        "imgsz": 640,
        "preprocess": "native LetterBox auto=False, BGR->RGB, BCHW float32 /255; exact tensor shared across cells/backends",
        "agreement_scope": "all raw output elements on the selected input tensors; not full-val mAP agreement",
        "onnx_full_validation": "not_run",
        "sparse_latency_equivalence": "not_claimed",
        "tolerance": {"atol": args.export_atol, "rtol": args.export_rtol},
    }
    repo = Path(__file__).resolve().parents[2]
    dispatcher = repo / "agent/scripts/run_yolo_master_skill.py"
    for cell in CELLS:
        destination = native / cell
        destination.mkdir()
        entry = {"status": "failed", "end2end": cell in "bd", "nms": False, "device": "cpu"}
        try:
            model = load_model(gate, cell)
            model.model.to("cpu").float().eval()
            preflight = export_preflight(model.model, "onnx", strict=False)
            entry["native_preflight"] = preflight
            write_json(destination / "preflight.json", preflight)
            entry.update(export_semantics(preflight, cell in "cd"))
            source = Path(gate["cells"][cell]["weights"]["best.pt"]["path"])
            copied = destination / "model.pt"
            shutil.copy2(source, copied)  # native exporter writes beside its input; keep the training run immutable
            checked_file(copied, gate["cells"][cell]["weights"]["best.pt"]["sha256"], "isolated export checkpoint")
            request = export_request(copied, destination, cell, args)
            request_path = destination / "request.json"
            write_json(request_path, request)
            stdout, stderr = destination / "stdout.json", destination / "stderr.log"
            entry.update(
                request=str(request_path), request_sha256=digest(request_path), stdout=str(stdout), stderr=str(stderr)
            )
            environment = os.environ.copy()
            environment.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", CUDA_VISIBLE_DEVICES="")
            with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
                process = subprocess.run(
                    [sys.executable, str(dispatcher), "--request", str(request_path)],
                    cwd=repo,
                    stdout=out,
                    stderr=err,
                    env=environment,
                    check=False,
                )
            payload = read_json(stdout)
            if process.returncode != 0 or payload.get("status") != "ok":
                text = stdout.read_text(encoding="utf-8") + stderr.read_text(encoding="utf-8")
                if any(
                    marker in text.lower()
                    for marker in ("unsupportedoperator", "not supported", "not implemented", "export preflight failed")
                ):
                    raise ExportBlocked(f"native exporter refuses this route; see {stdout}")
                raise RuntimeError(f"native export failed; see {stdout}")
            require(not (payload.get("recovery") or {}).get("recovered"), "unexpected export device fallback")
            artifact = copied.with_suffix(".onnx")
            require(
                artifact.is_file() and artifact.stat().st_size > 0,
                "native export did not produce the expected ONNX artifact",
            )
            entry["artifact"] = {"path": str(artifact), "sha256": digest(artifact), "bytes": artifact.stat().st_size}
            model_proto = onnx.load(str(artifact), load_external_data=False)
            graph = inspect_onnx_graph(model_proto, preflight, cell=cell, imgsz=args.imgsz, max_det=args.max_det)
            entry["graph"] = graph
            require(graph["status"] == "passed", "ONNX graph contains an NMS operation despite nms=False")
            onnx.checker.check_model(model_proto)
            entry["onnx_checker"] = "passed"
            options = ort.SessionOptions()
            options.intra_op_num_threads = args.threads
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(artifact), sess_options=options, providers=["CPUExecutionProvider"])
            require(session.get_providers() == ["CPUExecutionProvider"], "ORT did not select only CPUExecutionProvider")
            entry["actual_providers"] = session.get_providers()
            head = model.model.model[-1]
            head.max_det = args.max_det
            head.agnostic_nms = False
            head.export = False  # compare normal eager inference, not another forced export/dense-fallback path
            comparisons = []
            for index, array in enumerate(arrays):
                with torch.inference_mode():
                    reference = model.model(torch.from_numpy(array))
                reference = reference[0] if isinstance(reference, tuple) else reference
                require(isinstance(reference, torch.Tensor), "unexpected eager detection output")
                reference = reference.detach().cpu().numpy()
                outputs = session.run([graph["output_name"]], {graph["input_name"]: array})
                candidate = outputs[0]
                expected_shape = (1, args.max_det, 6) if cell in "bd" else (1, 4 + len(model.names), 8400)
                require(
                    reference.shape == expected_shape, f"eager output contradicts the fixed A1 head: {reference.shape}"
                )
                output_path = destination / f"outputs_{index:04d}.npz"
                np.savez_compressed(output_path, pytorch=reference, onnx=candidate)
                comparison = compare_export_outputs(
                    reference, candidate, end2end=cell in "bd", atol=args.export_atol, rtol=args.export_rtol
                )
                comparisons.append(
                    {
                        "input_index": index,
                        "outputs": str(output_path),
                        "outputs_sha256": digest(output_path),
                        **comparison,
                    }
                )
            entry["comparisons"] = comparisons
            all_match = bool(comparisons) and all(item["status"] == "passed" for item in comparisons)
            entry["output_equivalence"] = "matched_on_selected_inputs" if all_match else "failed_on_selected_inputs"
            entry["status"] = "completed" if all_match else "failed"
            checked_file(source, gate["cells"][cell]["weights"]["best.pt"]["sha256"], "unchanged formal checkpoint")
        except ExportBlocked as exc:
            entry.update(status="blocked", error=str(exc))
        except Exception as exc:
            entry.update(status="failed", error=str(exc))
        result["cells"][cell] = entry
        write_json(destination / "evidence.json", entry)
    if any(entry["status"] != "completed" for entry in result["cells"].values()):
        result["status"] = (
            "blocked" if all(entry["status"] == "blocked" for entry in result["cells"].values()) else "partial"
        )
    return result


def factorial_contrasts(values: dict[str, float]) -> dict:
    """Return signed effects in the supplied units; negative latency means faster."""
    dense = values["b"] - values["a"]
    moe = values["d"] - values["c"]
    return {"B_minus_A": dense, "D_minus_C": moe, "interaction_D_minus_C_minus_B_minus_A": moe - dense}


def finite_number(value, label: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    """Require a real finite JSON number within optional inclusive bounds."""
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not a JSON number")
    number = float(value)
    require(math.isfinite(number), f"{label} is non-finite")
    if minimum is not None:
        require(number >= minimum, f"{label} is below {minimum}")
    if maximum is not None:
        require(number <= maximum, f"{label} is above {maximum}")
    return number


def nonnegative_integer(value, label: str) -> int:
    """Require an integer count without accepting JSON booleans."""
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"{label} is not a non-negative integer"
    )
    return value


def completed_cells(evidence: dict, stage: str) -> dict:
    """Return an exact, completed A/B/C/D mapping for a completed stage."""
    cells = evidence.get("cells")
    require(isinstance(cells, dict), f"{stage}: completed evidence has no cell mapping")
    require(set(cells) == set(CELLS), f"{stage}: completed evidence must contain exactly A/B/C/D")
    for cell in CELLS:
        require(isinstance(cells[cell], dict), f"{stage}/{cell}: cell evidence is not an object")
        require(cells[cell].get("status") == "completed", f"{stage}/{cell}: cell is not completed")
    return cells


def validate_validate_evidence(evidence: dict) -> None:
    """Validate the minimum fresh-best-checkpoint metric contract."""
    for cell, entry in completed_cells(evidence, "validate").items():
        finite_number(entry.get("map50_95"), f"validate/{cell}/map50_95", minimum=0, maximum=1)
        require(isinstance(entry.get("metrics"), dict), f"validate/{cell}: metrics are missing")
        require(isinstance(entry.get("evaluation"), dict), f"validate/{cell}: evaluation is missing")
        require(
            entry.get("source") == "fresh native validation of formal best.pt", f"validate/{cell}: wrong metric source"
        )
        require(isinstance(entry.get("request_sha256"), str), f"validate/{cell}: request hash is missing")


def validate_duplicate_record(record: dict, label: str) -> tuple[int, int]:
    """Validate one image's postprocess/overlap counts and return boxes, duplicates."""
    require(isinstance(record, dict), f"{label}: image record is not an object")
    require(isinstance(record.get("image"), str), f"{label}: image path is missing")
    boxes = record.get("boxes_xyxy_conf_class")
    duplicate = record.get("duplicates")
    require(isinstance(boxes, list), f"{label}: boxes are missing")
    require(isinstance(duplicate, dict), f"{label}: duplicate diagnostics are missing")
    selected = nonnegative_integer(duplicate.get("boxes"), f"{label}/boxes")
    duplicate_boxes_count = nonnegative_integer(duplicate.get("duplicate_boxes"), f"{label}/duplicate_boxes")
    pairs = nonnegative_integer(duplicate.get("duplicate_pairs"), f"{label}/duplicate_pairs")
    same_class_pairs = nonnegative_integer(duplicate.get("same_class_pairs"), f"{label}/same_class_pairs")
    require(len(boxes) == selected, f"{label}: detection and diagnostic box counts differ")
    require(duplicate_boxes_count <= selected, f"{label}: duplicate box count exceeds detections")
    require(pairs <= same_class_pairs, f"{label}: duplicate pair count exceeds same-class pairs")
    for key in ("duplicate_box_rate", "duplicate_pair_rate"):
        value = duplicate.get(key)
        if value is not None:
            finite_number(value, f"{label}/{key}", minimum=0, maximum=1)
    return selected, duplicate_boxes_count


def validate_predict_evidence(evidence: dict) -> None:
    """Require complete postprocess records and the expected real NMS route per cell."""
    for cell, entry in completed_cells(evidence, "predict").items():
        records = entry.get("images")
        require(isinstance(records, list) and records, f"predict/{cell}: image records are empty")
        totals = [
            validate_duplicate_record(record, f"predict/{cell}/image-{index}") for index, record in enumerate(records)
        ]
        detections = nonnegative_integer(entry.get("detections"), f"predict/{cell}/detections")
        duplicate_boxes_count = nonnegative_integer(entry.get("duplicate_boxes"), f"predict/{cell}/duplicate_boxes")
        require(
            detections > 0 and detections == sum(item[0] for item in totals),
            f"predict/{cell}: detection total mismatch",
        )
        require(duplicate_boxes_count == sum(item[1] for item in totals), f"predict/{cell}: duplicate total mismatch")
        nms = entry.get("nms")
        require(isinstance(nms, dict) and nms.get("status") == "passed", f"predict/{cell}: NMS trace did not pass")
        calls = nonnegative_integer(nms.get("suppression_kernel_calls"), f"predict/{cell}/suppression_kernel_calls")
        routes = nms.get("wrapper_routes")
        require(isinstance(routes, dict), f"predict/{cell}: wrapper routes are missing")
        expected_route = "end2end_score_filter" if cell in "bd" else "nms"
        require(set(routes) == {expected_route}, f"predict/{cell}: unexpected postprocess route")
        require(
            all(nonnegative_integer(value, f"predict/{cell}/route-count") > 0 for value in routes.values()),
            f"predict/{cell}: empty route count",
        )
        require(
            calls == 0 if cell in "bd" else calls > 0,
            f"predict/{cell}: suppression-kernel evidence contradicts the matrix",
        )


def validate_latency_evidence(evidence: dict) -> None:
    """Require every measured device to contain four complete, fixed-count sample series."""
    expected_samples = nonnegative_integer(evidence.get("samples"), "latency/samples")
    require(expected_samples > 0, "latency: sample count must be positive")
    devices = evidence.get("devices")
    require(isinstance(devices, dict) and devices, "latency: no measured devices")
    for device, cells in devices.items():
        require(isinstance(device, str), "latency: device key is not a string")
        require(isinstance(cells, dict) and set(cells) == set(CELLS), f"latency/{device}: must contain exactly A/B/C/D")
        for cell in CELLS:
            entry = cells[cell]
            require(
                isinstance(entry, dict) and entry.get("status") == "completed",
                f"latency/{device}/{cell}: cell is not completed",
            )
            samples = entry.get("samples")
            require(
                isinstance(samples, list) and len(samples) == expected_samples,
                f"latency/{device}/{cell}: wrong sample count",
            )
            for index, sample in enumerate(samples):
                require(isinstance(sample, dict), f"latency/{device}/{cell}/sample-{index}: not an object")
                require(sample.get("sample") == index, f"latency/{device}/{cell}: noncontiguous sample indices")
                nonnegative_integer(sample.get("image_index"), f"latency/{device}/{cell}/image_index")
                nonnegative_integer(sample.get("detections"), f"latency/{device}/{cell}/detections")
                for component in ("total", "preprocess", "inference", "postprocess"):
                    finite_number(sample.get(f"{component}_ms"), f"latency/{device}/{cell}/{component}_ms", minimum=0)
            statistics_block = entry.get("statistics")
            require(isinstance(statistics_block, dict), f"latency/{device}/{cell}: statistics are missing")
            for component in ("total", "preprocess", "inference", "postprocess"):
                stats = statistics_block.get(component)
                require(isinstance(stats, dict), f"latency/{device}/{cell}/{component}: statistics are missing")
                require(
                    stats.get("n") == expected_samples,
                    f"latency/{device}/{cell}/{component}: statistics count mismatch",
                )
                for key in ("mean_ms", "std_ms", "p50_ms", "p90_ms", "p99_ms", "min_ms", "max_ms"):
                    finite_number(stats.get(key), f"latency/{device}/{cell}/{component}/{key}", minimum=0)


def validate_export_evidence(evidence: dict) -> None:
    """Require graph, checker, CPU runtime and every sampled output comparison to pass."""
    for cell, entry in completed_cells(evidence, "export").items():
        require(entry.get("end2end") is (cell in "bd"), f"export/{cell}: end-to-end mode mismatch")
        require(entry.get("nms") is False, f"export/{cell}: export requested NMS")
        require(isinstance(entry.get("dense_fallback"), bool), f"export/{cell}: fallback declaration is missing")
        artifact = entry.get("artifact")
        require(isinstance(artifact, dict), f"export/{cell}: artifact record is missing")
        require(
            isinstance(artifact.get("path"), str) and isinstance(artifact.get("sha256"), str),
            f"export/{cell}: artifact identity is missing",
        )
        require(
            nonnegative_integer(artifact.get("bytes"), f"export/{cell}/artifact-bytes") > 0,
            f"export/{cell}: artifact is empty",
        )
        graph = entry.get("graph")
        require(isinstance(graph, dict) and graph.get("status") == "passed", f"export/{cell}: graph audit did not pass")
        require(graph.get("nms_nodes") == [], f"export/{cell}: graph contains an NMS node")
        require(entry.get("onnx_checker") == "passed", f"export/{cell}: ONNX checker did not pass")
        require(entry.get("actual_providers") == ["CPUExecutionProvider"], f"export/{cell}: runtime provider mismatch")
        comparisons = entry.get("comparisons")
        require(isinstance(comparisons, list) and comparisons, f"export/{cell}: output comparisons are empty")
        require(
            all(isinstance(item, dict) and item.get("status") == "passed" for item in comparisons),
            f"export/{cell}: an output comparison failed",
        )
        require(
            entry.get("output_equivalence") == "matched_on_selected_inputs",
            f"export/{cell}: output equivalence did not pass",
        )


def validate_stage_evidence(stage: str, evidence: dict) -> None:
    """Fail closed when a stage claims completion without its required evidence."""
    require(isinstance(evidence, dict), f"{stage}: evidence is not an object")
    status = evidence.get("status")
    require(
        status in {"completed", "partial", "blocked", "failed", "inconclusive", "not_implemented"},
        f"{stage}: invalid status",
    )
    if status != "completed":
        return
    validators = {
        "validate": validate_validate_evidence,
        "predict": validate_predict_evidence,
        "latency": validate_latency_evidence,
        "export": validate_export_evidence,
    }
    validators[stage](evidence)


def summarize(gate: dict, output: Path) -> dict:
    """Refuse stale stage results and never equate a numeric target with acceptance."""
    report = {
        "status": "blocked" if gate["status"] != "ready" else "incomplete",
        "artifact_gate": gate,
        "stages": {},
        "acceptance": "not_determined",
        "limitations": [
            "One training seed per cell: call-level timing variation is not training-seed variation.",
            "A completed fixed-budget trial does not demonstrate convergence or P1 acceptance.",
            "Runtime NMS tracing covers PyTorch prediction, not CLI validation. Export has a separate graph audit.",
        ],
    }
    for stage in STAGES:
        path = output / f"{stage}.json"
        if not path.is_file():
            report["stages"][stage] = {"status": "not_run"}
            continue
        try:
            evidence = read_json(path)
            require(
                evidence.get("identity") == gate.get("identity") and "identity" in gate,
                "stale checkpoint/protocol identity",
            )
            validate_stage_evidence(stage, evidence)
            report["stages"][stage] = {"status": evidence["status"], "path": str(path), "sha256": digest(path)}
            if stage == "validate" and evidence["status"] == "completed":
                values = {cell: 100.0 * evidence["cells"][cell]["map50_95"] for cell in CELLS}
                report["accuracy_AP_points"] = {"values": values, **factorial_contrasts(values)}
                report["accuracy_AP_points"]["numeric_drop_at_most_2_points"] = values["d"] - values["c"] >= -2
                if values["c"] <= 2:
                    report["limitations"].append(
                        "Reference C mAP is at most 2 AP points; the drop threshold is vacuous at this accuracy."
                    )
            if stage == "latency":
                report["latency_ms"] = {}
                for device, cells in evidence["devices"].items():
                    if set(cells) != set(CELLS) or any(cells[cell]["status"] != "completed" for cell in CELLS):
                        continue
                    values = {cell: cells[cell]["statistics"]["total"]["mean_ms"] for cell in CELLS}
                    report["latency_ms"][device] = {
                        "values": values,
                        **factorial_contrasts(values),
                        "D_mean_lower_than_C": values["d"] < values["c"],
                        "D_relative_reduction_vs_C_percent": 100.0 * (values["c"] - values["d"]) / values["c"],
                        "significance": "not_tested; inspect raw samples and host contention",
                    }
            if stage == "export":
                report["onnx_export"] = {
                    cell: {
                        "status": evidence.get("cells", {}).get(cell, {}).get("status", "not_run"),
                        "dense_fallback": evidence.get("cells", {}).get(cell, {}).get("dense_fallback", "not_verified"),
                        "output_equivalence": evidence.get("cells", {})
                        .get(cell, {})
                        .get("output_equivalence", "not_verified"),
                        "graph_nms_audit": evidence.get("cells", {})
                        .get(cell, {})
                        .get("graph", {})
                        .get("status", "not_run"),
                    }
                    for cell in CELLS
                }
                report["limitations"].append(
                    "ONNX comparisons are CPU FP32 output checks on selected inputs, not full COCO accuracy or GPU deployment tests."
                )
                if any(row["dense_fallback"] is True for row in report["onnx_export"].values()):
                    report["limitations"].append(
                        "MoE ONNX dense fallback executes all experts; numerical agreement does not preserve sparse execution or latency."
                    )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["stages"][stage] = {"status": "blocked", "error": str(exc), "path": str(path)}
    if report["stages"]["export"]["status"] != "completed":
        report["limitations"].append(
            "ONNX export and agreement evidence is incomplete; missing or failed cells are not passed."
        )
    if gate["status"] == "ready" and all(item["status"] == "completed" for item in report["stages"].values()):
        report["status"] = "collected_pending_review"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "summary", *STAGES), default="summary")
    parser.add_argument("--images", type=Path, help="Explicit local image or text list; defaults to locked val split.")
    parser.add_argument(
        "--limit", type=int, default=16, help="Deterministic image limit for predict/latency/export comparisons."
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--duplicate-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--predict-device", default="0")
    parser.add_argument("--devices", nargs="+", default=["cpu", "0"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpu-affinity", help="Optional comma-separated allowed CPU indices, e.g. 4,5,6,7.")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--export-atol", type=float, default=1e-4, help="Predeclared FP32 output absolute tolerance.")
    parser.add_argument("--export-rtol", type=float, default=1e-3, help="Predeclared FP32 output relative tolerance.")
    args = parser.parse_args(argv)
    args.configs, args.project, args.output = args.configs.resolve(), args.project.resolve(), args.output.resolve()
    require(
        min(args.limit, args.imgsz, args.max_det, args.threads, args.warmup, args.samples) > 0,
        "counts must be positive",
    )
    require(0 <= args.conf <= 1 and 0 < args.nms_iou <= 1 and 0 < args.duplicate_iou <= 1, "invalid thresholds")
    require(
        all(math.isfinite(value) and value >= 0 for value in (args.export_atol, args.export_rtol)),
        "export tolerances must be finite and non-negative",
    )
    gate = check_matrix(args.configs, args.project)
    write_json(args.output / "gate.json", gate)
    if gate["status"] != "ready":
        report = summarize(gate, args.output)
        write_json(args.output / "summary.json", report)
        print(
            json.dumps({"status": "blocked", "blockers": gate["blockers"], "report": str(args.output / "summary.json")})
        )
        return 2
    if args.stage != "summary":
        require(
            args.imgsz == gate["protocol"]["common_training"]["imgsz"], "imgsz differs from locked training protocol"
        )
        stages = STAGES if args.stage == "all" else (args.stage,)
        for stage in stages:
            require(
                not (args.output / f"{stage}.json").exists(), f"stage already has evidence: {stage}; use a new --output"
            )
        try:
            environment = prepare_runtime(args.threads, args.cpu_affinity, inspect_cuda=stages != ("export",))
        except Exception as exc:
            for stage in stages:
                write_json(
                    args.output / f"{stage}.json",
                    {
                        "status": "blocked",
                        "identity": gate["identity"],
                        "error": f"runtime unavailable: {exc}",
                    },
                )
            report = summarize(gate, args.output)
            write_json(args.output / "summary.json", report)
            print(json.dumps({"status": "blocked", "error": str(exc), "report": str(args.output / "summary.json")}))
            return 2
        for stage in stages:
            paths = []
            try:
                if stage in {"predict", "latency", "export"}:
                    paths = resolve_images(
                        Path(gate["protocol"].get("evaluation_data_yaml", gate["protocol"]["data_yaml"])),
                        args.images,
                        args.limit,
                    )
                if stage == "predict":
                    evidence = run_predict(gate, args, paths)
                elif stage == "latency":
                    evidence = run_latency(gate, args, paths)
                elif stage == "validate":
                    evidence = run_validate(gate, args)
                else:
                    evidence = run_export(gate, args, paths)
            except Exception as exc:
                evidence = {"status": "failed", "error": str(exc)}
            evidence.update(
                identity=gate["identity"],
                environment=environment,
                created_at=datetime.now(timezone.utc).isoformat(),
                images=[{"path": str(path), "sha256": digest(path)} for path in paths],
                command=[sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
            )
            write_json(args.output / f"{stage}.json", evidence)
            print(
                json.dumps({"stage": stage, "status": evidence["status"], "output": str(args.output / f"{stage}.json")})
            )
    report = summarize(gate, args.output)
    write_json(args.output / "summary.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "acceptance": report["acceptance"],
                "report": str(args.output / "summary.json"),
            }
        )
    )
    if args.stage in {"summary", "all"}:
        return 0 if report["status"] == "collected_pending_review" else 1
    return 0 if report["stages"][args.stage]["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
