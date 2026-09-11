#!/usr/bin/env python3
"""Evaluate the P2 segmentation pilots on the fixed COCO val512 subset.

The script is evaluation-only.  It records native box/mask metrics, the head's
NMS-free switch, a small batch=1 forward benchmark, and a one-image prediction
sanity check for each completed cell.  It never changes checkpoint weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def checkpoint_for(protocol: dict, cell: str) -> Path:
    name = f"{cell}_seg_pilot_seed260829_5ep"
    path = Path(protocol["run_root"]) / "train" / name / "weights" / "last.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def scalarize(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--export", action="store_true", help="also attempt ONNX export for each cell")
    args = parser.parse_args()
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(repo))

    import torch

    from ultralytics import YOLO

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data_yaml = Path(protocol["data"]["yaml"])
    data_root = data_yaml.parent
    val_list = data_root / "val2017.txt"
    first_image = None
    for line in val_list.read_text(encoding="utf-8").splitlines():
        if line.strip():
            first_image = (data_root / line.strip()).resolve()
            break
    if first_image is None or not first_image.is_file():
        raise FileNotFoundError(f"no validation image resolved from {val_list}: {first_image}")

    evidence = {
        "schema": "a1-p2-seg-extension-r1-evaluation/v1",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "data": {"yaml": str(data_yaml), "yaml_sha256": sha256(data_yaml), "images": 512},
        "device": str(device),
        "input": [1, 3, 640, 640],
        "warmup": args.warmup,
        "samples": args.samples,
        "runs": {},
    }
    image = torch.zeros((1, 3, 640, 640), dtype=torch.float32, device=device)
    for cell in protocol["configs"]:
        checkpoint = checkpoint_for(protocol, cell)
        yolo = YOLO(str(checkpoint), task="segment")
        model = yolo.model.to(device).float().eval()
        head = model.model[-1]
        entry = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "head_class": type(head).__name__,
            "end2end": bool(getattr(head, "end2end", False)),
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        }
        try:
            metrics = yolo.val(
                data=str(data_yaml),
                split="val",
                imgsz=640,
                batch=1,
                workers=0,
                device=str(device),
                plots=False,
                save_json=False,
                verbose=False,
                project=str(output / "val_runs"),
                name=cell,
                exist_ok=True,
            )
            entry["val_metrics"] = {str(key): scalarize(value) for key, value in metrics.results_dict.items()}
        except Exception as exc:
            entry["val_error"] = f"{type(exc).__name__}: {exc}"

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times = []
        with torch.inference_mode():
            for _ in range(args.samples):
                if device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(torch.cuda.current_stream(device))
                    model(image)
                    end.record(torch.cuda.current_stream(device))
                    end.synchronize()
                    times.append(float(start.elapsed_time(end)))
                else:
                    start = time.perf_counter()
                    model(image)
                    times.append((time.perf_counter() - start) * 1000.0)
        entry["forward_batch1"] = {
            "mean_ms": statistics.mean(times),
            "p50_ms": percentile(times, 0.5),
            "p90_ms": percentile(times, 0.9),
            "throughput_images_per_s": 1000.0 / statistics.mean(times),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        }

        try:
            prediction = yolo.predict(source=str(first_image), imgsz=640, device=str(device), conf=0.001, verbose=False)[0]
            entry["prediction_sanity"] = {
                "image": str(first_image),
                "boxes": int(len(prediction.boxes)) if prediction.boxes is not None else 0,
                "masks": int(len(prediction.masks)) if prediction.masks is not None else 0,
            }
        except Exception as exc:
            entry["prediction_error"] = f"{type(exc).__name__}: {exc}"

        if args.export:
            try:
                export_dir = output / "exports" / cell
                export_dir.mkdir(parents=True, exist_ok=True)
                exported = yolo.export(
                    format="onnx",
                    imgsz=640,
                    batch=1,
                    device=str(device),
                    half=False,
                    dynamic=False,
                    simplify=False,
                    nms=False,
                    opset=17,
                    project=str(export_dir),
                    name=f"{cell}_seg",
                    exist_ok=True,
                )
                entry["onnx_export"] = {"status": "passed", "path": str(exported)}
            except Exception as exc:
                entry["onnx_export"] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
        evidence["runs"][cell] = entry
        del model, yolo
        if device.type == "cuda":
            torch.cuda.empty_cache()

    evidence["status"] = "completed"
    evidence["completed_at"] = datetime.now(timezone.utc).isoformat()
    (output / "evaluation_evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
