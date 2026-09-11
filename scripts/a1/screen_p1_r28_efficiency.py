#!/usr/bin/env python3
"""Inference-only efficiency screen for r28 MoE variants.

The expert-cap variants are engineering proxies: they keep the trained router
and weights but remap routes to the first N experts. They are not claims about
an independently trained N-expert model.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    i = (len(values) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values), "mean_ms": statistics.mean(values),
        "p50_ms": pct(values, 0.5), "p90_ms": pct(values, 0.9),
        "p99_ms": pct(values, 0.99), "min_ms": min(values), "max_ms": max(values),
    }


def configure(model: Any, variant: str) -> dict[str, Any]:
    import torch
    from ultralytics.nn.modules.moe.factor_adapter import C3k2ResidualFactor
    from ultralytics.nn.modules.moe.modules import OptimizedMOEImproved
    adapters = [m for m in model.modules() if isinstance(m, C3k2ResidualFactor)]
    moes = [m for m in model.modules() if isinstance(m, OptimizedMOEImproved)]
    if not moes:
        return {"adapter_count": len(adapters), "moe_count": 0,
                "variant_note": "Dense checkpoint baseline; variant label is ignored"}
    if variant in {"top1", "cap4_top1", "cap2_top1"}:
        for moe in moes:
            moe.top_k = 1
            moe._current_top_k = 1
            moe.routing.top_k = 1
            moe.routing.use_top_k = True
    if variant.startswith("cap4") or variant.startswith("cap2"):
        limit = 4 if variant.startswith("cap4") else 2
        for moe in moes:
            original = moe.routing.forward

            def restricted(x, top_k=None, _original=original, _limit=limit):
                weights, indices, loss = _original(x, top_k=top_k)
                allowed = indices < _limit
                indices = torch.where(allowed, indices, torch.zeros_like(indices))
                weights = weights * allowed.to(weights.dtype)
                denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
                weights = weights / denom
                return weights, indices, loss

            moe.routing.forward = restricted
    if variant == "late_top2":
        # Bypass the first two factor branches; retain the last MoE adapter.
        for adapter in adapters[:-1]:
            adapter.factor = torch.nn.Identity()
            adapter.gain.data.zero_()
    return {"adapter_count": len(adapters), "moe_count": len(moes), "variant_note":
            "cap variants remap routes to first N trained experts; late_top2 bypasses earlier factors"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", default="260829")
    parser.add_argument("--cell", choices=("a", "b", "c", "d"), default="c")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=60)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from ultralytics import YOLO

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    image = torch.linspace(0.0, 1.0, steps=3 * args.image_size * args.image_size,
                           dtype=torch.float32, device=device).reshape(1, 3, args.image_size, args.image_size)
    ckpt = args.checkpoint_root / f"seed{args.seed}" / f"{args.cell}_formal_seed{args.seed}_15ep" / "weights" / "last.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    variants = ("top2", "top1", "cap4_top2", "cap4_top1", "cap2_top1", "late_top2")
    rows: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {
        "schema": "a1-p1-r28-efficiency-screen/v1", "status": "running",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"device": str(device), "cell": args.cell, "seed": args.seed,
                      "batch": 1, "image_size": args.image_size, "warmup": args.warmup,
                      "samples": args.samples, "scope": "model forward only"},
        "checkpoint": str(ckpt), "variants": {},
    }
    for variant in variants:
        model = YOLO(str(ckpt)).model.to(device).float().eval()
        configure_info = configure(model, variant)
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(image)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        times: list[float] = []
        with torch.inference_mode():
            for sample in range(args.samples):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(torch.cuda.current_stream(device))
                model(image)
                end.record(torch.cuda.current_stream(device))
                end.synchronize()
                elapsed = float(start.elapsed_time(end))
                times.append(elapsed)
                rows.append({"variant": variant, "sample": sample, "model_forward_ms": elapsed})
        summary = {**configure_info, "timing": stats(times),
                   "throughput_images_per_s": 1000.0 / statistics.mean(times),
                   "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                   "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}
        evidence["variants"][variant] = summary
        del model
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    evidence["status"] = "completed"
    evidence["completed_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with (args.output / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "sample", "model_forward_ms"])
        writer.writeheader(); writer.writerows(rows)
    with (args.output / "evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps({"status": "completed", "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
