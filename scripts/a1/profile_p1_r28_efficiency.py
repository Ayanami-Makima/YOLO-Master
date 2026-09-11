#!/usr/bin/env python3
"""Profile r28 A/B/C/D inference cost and MoE internal stages on one GPU.

This is an append-only diagnostic profile.  It does not modify checkpoints or
the locked r28 protocol.  The model-forward timer uses CUDA events around the
same finite, spatially varying batch-1 input for every cell.  Forward hooks
measure the nested residual base/factor path and, for MoE cells, router,
shared-expert, selected-expert and residual dispatch/aggregation time.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CELLS = ("a", "b", "c", "d")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] + fraction * (ordered[high] - ordered[low])


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean_ms": 0.0, "std_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "p99_ms": 0.0}
    return {
        "n": len(values),
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


class CudaModuleTimer:
    """Collect CUDA-event durations for selected modules during one forward."""

    def __init__(self, torch_module: Any, device: Any) -> None:
        self.torch = torch_module
        self.device = device
        self.stream = torch_module.cuda.current_stream(device)
        self.handles: list[Any] = []
        self.active = False
        self.starts: dict[str, list[Any]] = {}
        self.events: dict[str, list[tuple[Any, Any]]] = {}

    def register(self, module: Any, label: str) -> None:
        def pre_hook(_module: Any, _inputs: tuple[Any, ...]) -> None:
            if not self.active:
                return
            event = self.torch.cuda.Event(enable_timing=True)
            event.record(self.stream)
            self.starts.setdefault(label, []).append(event)

        def post_hook(_module: Any, _inputs: tuple[Any, ...], _output: Any) -> None:
            if not self.active:
                return
            end = self.torch.cuda.Event(enable_timing=True)
            end.record(self.stream)
            starts = self.starts.get(label, [])
            if not starts:
                raise RuntimeError(f"module timer stack underflow: {label}")
            self.events.setdefault(label, []).append((starts.pop(), end))

        self.handles.append(module.register_forward_pre_hook(pre_hook))
        self.handles.append(module.register_forward_hook(post_hook))

    def attach(self, model: Any) -> dict[str, int]:
        from ultralytics.nn.modules.moe.factor_adapter import C3k2ResidualFactor
        from ultralytics.nn.modules.moe.modules import OptimizedMOEImproved

        counts = {"adapters": 0, "moe_blocks": 0, "experts": 0}
        for name, adapter in model.named_modules():
            if not isinstance(adapter, C3k2ResidualFactor):
                continue
            counts["adapters"] += 1
            self.register(adapter, f"{name}/adapter_total")
            self.register(adapter.base, f"{name}/base")
            self.register(adapter.factor, f"{name}/factor")
            for subname, module in adapter.factor.named_modules():
                if not isinstance(module, OptimizedMOEImproved):
                    continue
                counts["moe_blocks"] += 1
                prefix = f"{name}/factor/{subname}/moe"
                self.register(module, f"{prefix}/total")
                self.register(module.routing, f"{prefix}/router")
                self.register(module.shared_expert, f"{prefix}/shared_expert")
                for expert_index, expert in enumerate(module.experts):
                    counts["experts"] += 1
                    self.register(expert, f"{prefix}/expert[{expert_index}]")
        return counts

    def begin(self) -> None:
        self.active = True
        self.starts = {}
        self.events = {}

    def finish(self) -> dict[str, float]:
        self.active = False
        self.torch.cuda.synchronize(self.device)
        result: dict[str, float] = {}
        for label, pairs in self.events.items():
            result[label] = sum(start.elapsed_time(end) for start, end in pairs)
        if self.starts:
            dangling = {key: len(value) for key, value in self.starts.items() if value}
            if dangling:
                raise RuntimeError(f"module timer stack not empty: {dangling}")
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def aggregate_component_durations(durations: dict[str, float], total_ms: float) -> dict[str, float]:
    def sum_suffix(suffix: str) -> float:
        return sum(value for key, value in durations.items() if key.endswith(suffix))

    adapter_total = sum_suffix("/adapter_total")
    base = sum_suffix("/base")
    factor = sum_suffix("/factor")
    moe_total = sum_suffix("/moe/total")
    router = sum_suffix("/moe/router")
    shared = sum_suffix("/moe/shared_expert")
    experts = sum(value for key, value in durations.items() if "/moe/expert[" in key)
    return {
        "adapter_total_ms": adapter_total,
        "base_ms": base,
        "factor_ms": factor,
        "adapter_residual_combine_ms": max(0.0, adapter_total - base - factor),
        "moe_total_ms": moe_total,
        "router_ms": router,
        "shared_expert_ms": shared,
        "selected_expert_compute_ms": experts,
        "expert_dispatch_aggregate_ms": max(0.0, moe_total - router - shared - experts),
        "non_adapter_model_ms": max(0.0, total_ms - adapter_total),
    }


def discover_checkpoints(root: Path, requested_seeds: list[str] | None) -> dict[str, dict[str, Path]]:
    seeds = requested_seeds or sorted(path.name.removeprefix("seed") for path in root.glob("seed*") if path.is_dir())
    result: dict[str, dict[str, Path]] = {}
    for seed in seeds:
        result[seed] = {}
        for cell in CELLS:
            path = root / f"seed{seed}" / f"{cell}_formal_seed{seed}_15ep" / "weights" / "last.pt"
            if not path.is_file():
                raise FileNotFoundError(f"missing formal checkpoint: {path}")
            result[seed][cell] = path
    return result


def profile_cell(
    checkpoint: Path,
    repo: Path,
    device: Any,
    image_size: int,
    warmup: int,
    samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    sys.path.insert(0, str(repo.resolve()))
    from ultralytics import YOLO

    model = YOLO(str(checkpoint)).model.to(device).float().eval()
    image = torch.linspace(
        0.0,
        1.0,
        steps=3 * image_size * image_size,
        dtype=torch.float32,
        device=device,
    ).reshape(1, 3, image_size, image_size)
    timer = CudaModuleTimer(torch, device)
    hook_counts = timer.attach(model)
    with torch.inference_mode():
        for _ in range(warmup):
            model(image)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = int(torch.cuda.memory_allocated(device))
    reserved_before = int(torch.cuda.memory_reserved(device))

    rows: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for sample_index in range(samples):
                timer.begin()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(torch.cuda.current_stream(device))
                model(image)
                end.record(torch.cuda.current_stream(device))
                durations = timer.finish()
                total_ms = start.elapsed_time(end)
                rows.append(
                    {
                        "sample": sample_index,
                        "model_forward_ms": total_ms,
                        **aggregate_component_durations(durations, total_ms),
                    }
                )
    finally:
        timer.close()

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    component_keys = (
        "adapter_total_ms",
        "base_ms",
        "factor_ms",
        "adapter_residual_combine_ms",
        "moe_total_ms",
        "router_ms",
        "shared_expert_ms",
        "selected_expert_compute_ms",
        "expert_dispatch_aggregate_ms",
        "non_adapter_model_ms",
    )
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "hook_counts": hook_counts,
        "timing": stats([float(row["model_forward_ms"]) for row in rows]),
        "throughput_images_per_s": 1000.0 / statistics.mean(float(row["model_forward_ms"]) for row in rows),
        "components": {
            key: stats([float(row[key]) for row in rows]) for key in component_keys
        },
        "memory": {
            "allocated_before_bytes": allocated_before,
            "reserved_before_bytes": reserved_before,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_extra_allocated_bytes": peak_allocated - allocated_before,
        },
    }
    del model, image
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seeds", nargs="*", default=None)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if not args.repo.is_dir() or not args.checkpoint_root.is_dir():
        raise FileNotFoundError("repo/checkpoint-root missing")
    if args.image_size <= 0 or args.warmup < 1 or args.samples < 2:
        raise ValueError("invalid image-size/warmup/samples")

    sys.path.insert(0, str(args.repo.resolve()))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this diagnostic requires a CUDA device")
    torch.cuda.set_device(device)
    checkpoints = discover_checkpoints(args.checkpoint_root, args.seeds)
    args.output.mkdir(parents=True, exist_ok=False)

    evidence: dict[str, Any] = {
        "schema": "a1-p1-r28-efficiency-profile/v1",
        "status": "running",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "device": str(device),
            "image_size": args.image_size,
            "batch": 1,
            "warmup": args.warmup,
            "samples": args.samples,
            "input": "finite spatially varying float32 linspace [1,3,640,640], shared by every cell",
            "scope": "model forward only; RAM input to DetectionModel output",
            "component_timing": "CUDA events on residual base/factor and nested MoE router/shared expert/experts; dispatch is MoE total minus measured children",
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "repo": str(args.repo.resolve()),
        },
        "cells": {},
    }
    csv_path = args.output / "latency_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_stream:
        writer = csv.DictWriter(
            csv_stream,
            fieldnames=[
                "seed",
                "cell",
                "sample",
                "model_forward_ms",
                "adapter_total_ms",
                "base_ms",
                "factor_ms",
                "adapter_residual_combine_ms",
                "moe_total_ms",
                "router_ms",
                "shared_expert_ms",
                "selected_expert_compute_ms",
                "expert_dispatch_aggregate_ms",
                "non_adapter_model_ms",
            ],
        )
        writer.writeheader()
        for seed, cells in checkpoints.items():
            evidence["cells"][seed] = {}
            for cell, checkpoint in cells.items():
                summary, rows = profile_cell(
                    checkpoint,
                    args.repo,
                    device,
                    args.image_size,
                    args.warmup,
                    args.samples,
                )
                evidence["cells"][seed][cell] = summary
                for row in rows:
                    writer.writerow({"seed": seed, "cell": cell, **row})

    evidence["status"] = "completed"
    evidence["completed_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    evidence["latency_samples_sha256"] = sha256(csv_path)
    with (args.output / "evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": evidence["status"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
