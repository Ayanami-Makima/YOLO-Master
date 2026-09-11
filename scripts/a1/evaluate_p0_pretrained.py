#!/usr/bin/env python3
"""Evaluate the A1 P0 NMS and end-to-end paths from one pretrained checkpoint.

The same checkpoint is evaluated twice. Cell A selects the checkpoint's
one-to-many head and executes conventional NMS. Cell B selects its one-to-one
head and must not execute an IoU suppression kernel. No training is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


CELLS = {"A": False, "B": True}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def build_validation_request(
    *,
    repo: Path,
    checkpoint: Path,
    data: Path,
    output: Path,
    cell: str,
    end2end: bool,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    nms_iou: float,
    max_det: int,
) -> dict:
    """Build a native Agent CLI validation request with an explicit head mode."""
    name = cell.lower()
    return {
        "skill": "yolo.val",
        "request_id": f"a1_p0_pretrained_{name}",
        "runtime": {"cwd": str(repo), "python": sys.executable, "prefer_cli": True},
        "inputs": {"model": str(checkpoint), "data": str(data), "task": "detect"},
        "params": {
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "workers": workers,
            "split": "val",
            "conf": 0.001,
            "iou": nms_iou,
            "max_det": max_det,
            "half": False,
            "save_json": True,
            "plots": False,
            "verbose": False,
            "project": str(output / "native_validation"),
            "name": name,
            "exist_ok": False,
            "end2end": end2end,
        },
        "artifacts": {"project": str(output / "agent_manifests"), "name": f"val_{name}"},
        "policy": {"dry_run": False, "async": False},
    }


def extract_validation(payload: dict) -> dict:
    """Normalize the detector metrics emitted by the structured dispatcher."""
    require(payload.get("status") == "ok", f"dispatcher status is not ok: {payload.get('status')}")
    require(not (payload.get("recovery") or {}).get("recovered"), "device fallback invalidates fixed-device P0")
    metrics = payload.get("metrics") or {}
    evaluation = payload.get("evaluation") or {}

    def value(*keys: str) -> float:
        for key in keys:
            candidate = metrics.get(key, evaluation.get(key))
            if candidate is not None:
                number = float(candidate)
                require(math.isfinite(number), f"metric {key} is not finite")
                return number
        raise RuntimeError(f"missing metric: {keys}")

    return {
        "precision": value("metrics/precision(B)", "precision"),
        "recall": value("metrics/recall(B)", "recall"),
        "map50": value("metrics/mAP50(B)", "map50"),
        "map50_95": value("metrics/mAP50-95(B)", "map50_95"),
        "speed": payload.get("speed") or evaluation.get("speed"),
        "manifest": payload.get("manifest"),
    }


def set_head_mode(model, end2end: bool) -> dict:
    """Select a checkpoint head path and prove that both native branches exist."""
    core = model.model
    head = core.model[-1]
    require(hasattr(head, "one2one_cv2") and hasattr(head, "one2one_cv3"), "checkpoint lacks one-to-one heads")
    core.end2end = end2end
    require(bool(core.end2end) is end2end, "model end2end flag did not update")
    require(bool(head.end2end) is end2end, "head end2end flag did not update")
    return {
        "model_class": type(core).__name__,
        "head_class": type(head).__name__,
        "end2end": end2end,
        "has_one2one_cv2": True,
        "has_one2one_cv3": True,
        "parameters": sum(parameter.numel() for parameter in core.parameters()),
    }


def run_native_validation(args, output: Path) -> dict:
    dispatcher = args.repo / "agent/scripts/run_yolo_master_skill.py"
    require(dispatcher.is_file(), f"missing dispatcher: {dispatcher}")
    result = {"status": "completed", "cells": {}}
    for cell, end2end in CELLS.items():
        request = build_validation_request(
            repo=args.repo,
            checkpoint=args.checkpoint,
            data=args.data,
            output=output,
            cell=cell,
            end2end=end2end,
            imgsz=args.imgsz,
            batch=args.val_batch,
            device=args.val_device,
            workers=args.workers,
            nms_iou=args.nms_iou,
            max_det=args.max_det,
        )
        request_path = output / "requests" / f"{cell.lower()}_val_request.json"
        stdout_path = output / "logs" / f"{cell.lower()}_val_stdout.json"
        stderr_path = output / "logs" / f"{cell.lower()}_val_stderr.log"
        write_json(request_path, request)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.run(
                [sys.executable, str(dispatcher), "--request", str(request_path)],
                cwd=args.repo,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        try:
            payload = json.loads(stdout_path.read_text(encoding="utf-8"))
            require(process.returncode == 0, f"validation process exited {process.returncode}")
            summary = extract_validation(payload)
            result["cells"][cell] = {
                "status": "completed",
                "end2end": end2end,
                **summary,
                "request": str(request_path),
                "request_sha256": sha256(request_path),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
        except Exception as exc:
            result["status"] = "failed"
            result["cells"][cell] = {
                "status": "failed",
                "end2end": end2end,
                "error": str(exc),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
            break
    write_json(output / "validation.json", result)
    require(result["status"] == "completed" and set(result["cells"]) == set(CELLS), "native validation failed")
    return result


def runtime_params(args, device: str) -> dict:
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


def load_runtime_dependencies():
    from ultralytics import YOLO

    try:
        from scripts.a1.evaluate_p1_matrix import (
            NMSCallMonitor,
            duplicate_boxes,
            load_images,
            resolve_images,
            sample_statistics,
        )
    except ModuleNotFoundError:
        from evaluate_p1_matrix import (  # type: ignore[no-redef]
            NMSCallMonitor,
            duplicate_boxes,
            load_images,
            resolve_images,
            sample_statistics,
        )
    return YOLO, NMSCallMonitor, duplicate_boxes, load_images, resolve_images, sample_statistics


def run_predict_and_nms(args, output: Path, paths: list[Path], images: list) -> dict:
    YOLO, NMSCallMonitor, duplicate_boxes, _, _, _ = load_runtime_dependencies()
    result = {"status": "completed", "cells": {}, "images": [str(path) for path in paths]}
    for cell, end2end in CELLS.items():
        model = YOLO(str(args.checkpoint))
        identity = set_head_mode(model, end2end)
        params = runtime_params(args, args.val_device)
        model.predict(source=images[0], **params)
        records = []
        with NMSCallMonitor() as monitor:
            for path, image in zip(paths, images):
                prediction = model.predict(source=image, **params)[0]
                boxes = prediction.boxes.data.detach().cpu().tolist()
                records.append(
                    {
                        "image": str(path),
                        "detections": len(boxes),
                        "duplicates": duplicate_boxes(boxes, args.conf, args.duplicate_iou),
                    }
                )
        detections = sum(record["detections"] for record in records)
        nms = monitor.report(end2end, detections)
        require(detections > 0, f"{cell}: no detections; latency/NMS evidence would be vacuous")
        require(nms["status"] == "passed", f"{cell}: NMS route audit failed: {nms}")
        result["cells"][cell] = {
            "status": "completed",
            "identity": identity,
            "detections": detections,
            "images_with_detections": sum(record["detections"] > 0 for record in records),
            "images_with_duplicate_pairs": sum(record["duplicates"]["duplicate_pairs"] > 0 for record in records),
            "duplicate_boxes": sum(record["duplicates"]["duplicate_boxes"] for record in records),
            "nms": nms,
            "records": records,
        }
    write_json(output / "predict_nms.json", result)
    return result


def run_latency(args, output: Path, images: list) -> dict:
    import torch

    YOLO, _, _, _, _, sample_statistics = load_runtime_dependencies()
    result = {
        "status": "completed",
        "scope": "RAM image -> preprocess -> inference -> postprocess -> Results; batch=1; disk I/O/model load excluded",
        "warmup": args.warmup,
        "samples": args.samples,
        "devices": {},
    }
    for device in args.devices:
        require(device == "cpu" or (device.isdigit() and int(device) < torch.cuda.device_count()), f"bad device {device}")
        result["devices"][device] = {}
        for cell, end2end in CELLS.items():
            model = YOLO(str(args.checkpoint))
            identity = set_head_mode(model, end2end)
            params = runtime_params(args, device)
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
                samples.append(
                    {
                        "sample": index,
                        "image_index": index % len(images),
                        "detections": len(prediction.boxes),
                        "total_ms": (time.perf_counter() - started) * 1000.0,
                        "preprocess_ms": float(prediction.speed["preprocess"]),
                        "inference_ms": float(prediction.speed["inference"]),
                        "postprocess_ms": float(prediction.speed["postprocess"]),
                    }
                )
            require(sum(sample["detections"] for sample in samples) > 0, f"{device}/{cell}: all samples empty")
            result["devices"][device][cell] = {
                "status": "completed",
                "identity": identity,
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
    write_json(output / "latency.json", result)
    return result


def build_report(identity: dict, validation: dict, prediction: dict, latency: dict) -> dict:
    a_metric = validation["cells"]["A"]["map50_95"]
    b_metric = validation["cells"]["B"]["map50_95"]
    latency_delta = {}
    for device, cells in latency["devices"].items():
        a_stats = cells["A"]["statistics"]["total"]
        b_stats = cells["B"]["statistics"]["total"]
        a_ms = a_stats["mean_ms"]
        b_ms = b_stats["mean_ms"]
        latency_delta[device] = {
            "A_mean_ms": a_ms,
            "A_std_ms": a_stats["std_ms"],
            "A_p50_ms": a_stats["p50_ms"],
            "A_p90_ms": a_stats["p90_ms"],
            "B_mean_ms": b_ms,
            "B_std_ms": b_stats["std_ms"],
            "B_p50_ms": b_stats["p50_ms"],
            "B_p90_ms": b_stats["p90_ms"],
            "B_minus_A_ms": b_ms - a_ms,
            "B_vs_A_percent": (b_ms / a_ms - 1.0) * 100.0,
        }
    return {
        "status": "completed",
        "conclusion": "P0 passed: one pretrained checkpoint reproduces standard-NMS and end-to-end NMS-free paths.",
        "identity": identity,
        "accuracy": {
            "A_map50_95": a_metric,
            "B_map50_95": b_metric,
            "B_minus_A_ap_points": (b_metric - a_metric) * 100.0,
            "A_map50": validation["cells"]["A"]["map50"],
            "B_map50": validation["cells"]["B"]["map50"],
            "A_precision": validation["cells"]["A"]["precision"],
            "B_precision": validation["cells"]["B"]["precision"],
            "A_recall": validation["cells"]["A"]["recall"],
            "B_recall": validation["cells"]["B"]["recall"],
        },
        "nms": {
            cell: {
                "detections": prediction["cells"][cell]["detections"],
                "suppression_kernel_calls": prediction["cells"][cell]["nms"]["suppression_kernel_calls"],
                "routes": prediction["cells"][cell]["nms"]["wrapper_routes"],
            }
            for cell in CELLS
        },
        "latency": latency_delta,
        "limitations": [
            "P0 compares two native inference paths in the same pretrained checkpoint; it is not the P1 MoE ablation.",
            "PyTorch runtime NMS tracing is separate from future ONNX graph auditing.",
        ],
    }


def write_markdown_report(path: Path, report: dict) -> None:
    accuracy = report["accuracy"]
    lines = [
        "# A1 P0 pretrained baseline",
        "",
        report["conclusion"],
        "",
        "## Accuracy on COCO val2017",
        "",
        "| Cell | Head path | Precision | Recall | mAP50 | mAP50-95 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| A | one-to-many + NMS | {accuracy['A_precision']:.5f} | {accuracy['A_recall']:.5f} | "
        f"{accuracy['A_map50']:.5f} | {accuracy['A_map50_95']:.5f} |",
        f"| B | one-to-one, NMS-free | {accuracy['B_precision']:.5f} | {accuracy['B_recall']:.5f} | "
        f"{accuracy['B_map50']:.5f} | {accuracy['B_map50_95']:.5f} |",
        "",
        f"B - A mAP50-95: **{accuracy['B_minus_A_ap_points']:+.3f} AP points**.",
        "",
        "## Runtime route audit",
        "",
        "| Cell | Detections | Real suppression kernel calls | Wrapper route |",
        "| --- | ---: | ---: | --- |",
    ]
    for cell in CELLS:
        item = report["nms"][cell]
        lines.append(
            f"| {cell} | {item['detections']} | {item['suppression_kernel_calls']} | "
            f"{json.dumps(item['routes'], ensure_ascii=False)} |"
        )
    lines.extend(
        [
            "",
            "## Batch-1 latency",
            "",
            "| Device | A mean ± std ms | B mean ± std ms | A/B p50 ms | A/B p90 ms | B - A ms | B vs A |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for device, item in report["latency"].items():
        lines.append(
            f"| {device} | {item['A_mean_ms']:.3f} ± {item['A_std_ms']:.3f} | "
            f"{item['B_mean_ms']:.3f} ± {item['B_std_ms']:.3f} | "
            f"{item['A_p50_ms']:.3f}/{item['B_p50_ms']:.3f} | "
            f"{item['A_p90_ms']:.3f}/{item['B_p90_ms']:.3f} | "
            f"{item['B_minus_A_ms']:+.3f} | {item['B_vs_A_percent']:+.2f}% |"
        )
    lines.extend(["", "## Limitations", "", *[f"- {item}" for item in report["limitations"]], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-device", default="0")
    parser.add_argument("--devices", nargs="+", default=["0", "cpu"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--val-batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--duplicate-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.repo = args.repo.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    require(args.checkpoint.is_file(), f"missing checkpoint: {args.checkpoint}")
    require(args.data.is_file(), f"missing data YAML: {args.data}")
    require(not args.output.exists(), f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)

    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"

    from ultralytics import YOLO, __version__ as ultralytics_version

    model = YOLO(str(args.checkpoint))
    identity_a = set_head_mode(model, False)
    identity_b = set_head_mode(model, True)
    identity = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "dataset_manifest": str(args.dataset_manifest.resolve()) if args.dataset_manifest else None,
        "dataset_manifest_sha256": sha256(args.dataset_manifest.resolve()) if args.dataset_manifest else None,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "ultralytics": ultralytics_version,
        "cells": {"A": identity_a, "B": identity_b},
        "protocol": "same checkpoint/data/settings; only end2end=False/True differs",
    }
    write_json(args.output / "identity.json", identity)
    del model

    validation = run_native_validation(args, args.output)
    _, _, _, load_images, resolve_images, _ = load_runtime_dependencies()
    paths = resolve_images(args.data, None, args.limit)
    images = load_images(paths)
    prediction = run_predict_and_nms(args, args.output, paths, images)
    latency = run_latency(args, args.output, images)
    report = build_report(identity, validation, prediction, latency)
    write_json(args.output / "p0_report.json", report)
    write_markdown_report(args.output / "P0_REPORT.md", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
