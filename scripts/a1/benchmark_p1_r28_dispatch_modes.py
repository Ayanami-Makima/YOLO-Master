#!/usr/bin/env python3
"""Compare P1 sparse dispatch with the experimental dense-gather baseline."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


def first_tensor(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", default="260829")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules.moe.modules import OptimizedMOEImproved

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    image = torch.linspace(0.0, 1.0, steps=3 * 640 * 640, dtype=torch.float32, device=device).reshape(1, 3, 640, 640)
    evidence = {"schema": "a1-p2-dispatch-mode-benchmark/v1", "status": "running",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "cells": {}}
    for cell in ("c", "d"):
        ckpt = args.checkpoint_root / f"seed{args.seed}" / f"{cell}_formal_seed{args.seed}_15ep" / "weights" / "last.pt"
        modes = {}
        reference = None
        for mode in ("sparse", "dense_gather", "vmap_grouped"):
            model = YOLO(str(ckpt)).model.to(device).float().eval()
            moes = [m for m in model.modules() if isinstance(m, OptimizedMOEImproved)]
            for moe in moes:
                moe.experimental_dispatch_mode = mode
            with torch.inference_mode():
                for _ in range(args.warmup):
                    model(image)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            values = []
            with torch.inference_mode():
                for _ in range(args.samples):
                    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                    start.record(torch.cuda.current_stream(device)); out = model(image); end.record(torch.cuda.current_stream(device)); end.synchronize()
                    values.append(float(start.elapsed_time(end)))
            out_tensor = first_tensor(out)
            if out_tensor is None:
                raise RuntimeError("model output contains no tensor")
            if reference is None:
                reference = out_tensor.detach().float().cpu()
            else:
                max_error = float((out_tensor.detach().float().cpu() - reference).abs().max())
            modes[mode] = {"mean_ms": statistics.mean(values), "p50_ms": sorted(values)[len(values)//2],
                           "p99_ms": sorted(values)[max(0, int(len(values)*0.99)-1)],
                           "throughput_images_per_s": 1000.0 / statistics.mean(values),
                           "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                           "moe_count": len(moes)}
            if mode == "sparse":
                modes[mode]["max_error_vs_sparse"] = 0.0
            else:
                modes[mode]["max_error_vs_sparse"] = max_error
            del model, out; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
        evidence["cells"][cell] = {"checkpoint": str(ckpt), "modes": modes}
    evidence["status"] = "completed"; evidence["completed_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with (args.output / "evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps({"status": evidence["status"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
