#!/usr/bin/env python3
"""Trace one batch-1 forward for r28 A/B/C/D with CPU and CUDA activities."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CELLS = ("a", "b", "c", "d")


def checkpoint(root: Path, seed: str, cell: str) -> Path:
    path = root / f"seed{seed}" / f"{cell}_formal_seed{seed}_15ep" / "weights" / "last.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", default="260829")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--active", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    image = torch.linspace(
        0.0, 1.0, steps=3 * args.image_size * args.image_size,
        dtype=torch.float32, device=device,
    ).reshape(1, 3, args.image_size, args.image_size)
    evidence: dict[str, Any] = {
        "schema": "a1-p1-r28-profiler-trace/v1",
        "status": "running",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "device": str(device), "batch": 1, "image_size": args.image_size,
            "warmup": args.warmup, "active_steps": args.active,
            "scope": "model forward only; CPU+CUDA activities",
        },
        "environment": {
            "python": sys.version, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(device),
        },
        "cells": {},
    }
    for cell in CELLS:
        path = checkpoint(args.checkpoint_root, args.seed, cell)
        model = YOLO(str(path)).model.to(device).float().eval()
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(image)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=True, with_stack=False,
        ) as prof:
            with torch.inference_mode():
                for _ in range(args.active):
                    model(image)
                    prof.step()
        torch.cuda.synchronize(device)
        events = prof.key_averages()
        rows = []
        for event in events:
            rows.append({
                "key": event.key,
                "calls": int(event.count),
                "self_cpu_ms": float(event.self_cpu_time_total) / 1000.0,
                "cpu_total_ms": float(event.cpu_time_total) / 1000.0,
                "self_cuda_ms": float(getattr(event, "self_device_time_total", 0.0)) / 1000.0,
                "cuda_total_ms": float(getattr(event, "device_time_total", 0.0)) / 1000.0,
                "self_cpu_memory_bytes": int(getattr(event, "self_cpu_memory_usage", 0)),
                "self_cuda_memory_bytes": int(getattr(event, "self_device_memory_usage", 0)),
            })
        rows.sort(key=lambda item: item["self_cpu_ms"], reverse=True)
        memcpy = [row for row in rows if "Memcpy" in row["key"] or "copy_" in row["key"] or "to_cpu" in row["key"]]
        sync = [row for row in rows if "Synchronize" in row["key"] or "Item" in row["key"] or "where" in row["key"]]
        evidence["cells"][cell] = {
            "checkpoint": str(path),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "top_cpu_self": rows[:30],
            "memcpy_or_copy_events": memcpy,
            "sync_or_index_events": sync,
            "table": prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30),
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    evidence["status"] = "completed"
    evidence["completed_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with (args.output / "evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": evidence["status"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
