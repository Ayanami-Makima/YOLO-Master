#!/usr/bin/env python3
"""Audit hard Top-2 router load for the P2 segmentation MoE checkpoint."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entropy(values: list[int]) -> float:
    total = sum(values)
    if total <= 0 or len(values) <= 1:
        return 0.0
    return -sum((value / total) * math.log(value / total) for value in values if value) / math.log(len(values))


def gini(values: list[int]) -> float:
    ordered = sorted(float(value) for value in values)
    total = sum(ordered)
    if not ordered or total == 0:
        return 0.0
    n = len(ordered)
    return sum((2 * index - n - 1) * value for index, value in enumerate(ordered, 1)) / (n * total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--conf", type=float, default=0.001)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    images = []
    for line in args.val_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image = Path(line.strip())
        if not image.is_absolute():
            image = args.val_list.parent / image
        images.append(image.resolve())
    if len(images) != 512:
        raise ValueError(f"expected fixed val512, got {len(images)}")
    sys.path.insert(0, str(args.repo.resolve()))
    from ultralytics import YOLO
    from ultralytics.nn.modules.moe.modules import OptimizedMOEImproved

    yolo = YOLO(str(args.checkpoint.resolve()), task="segment")
    yolo.model.to(args.device).float().eval()
    modules = [(name, module) for name, module in yolo.model.named_modules() if isinstance(module, OptimizedMOEImproved)]
    counters = {name: collections.Counter() for name, _ in modules}
    handles = []
    for name, module in modules:
        def hook(_module, _inputs, output, key=name):
            if isinstance(output, (tuple, list)) and len(output) > 1 and hasattr(output[1], "detach"):
                counters[key].update(int(index) for index in output[1].detach().reshape(-1).cpu().tolist())
        handles.append(module.routing.register_forward_hook(hook))
    prediction_count = 0
    mask_count = 0
    try:
        for image in images:
            result = yolo.predict(source=str(image), device=args.device, imgsz=640, conf=args.conf, max_det=300, verbose=False)[0]
            prediction_count += int(len(result.boxes)) if result.boxes is not None else 0
            mask_count += int(len(result.masks)) if result.masks is not None else 0
    finally:
        for handle in handles:
            handle.remove()
    routes = {}
    for name, module in modules:
        experts = int(module.num_experts)
        counts = [int(counters[name][index]) for index in range(experts)]
        total = sum(counts)
        fractions = [count / max(total, 1) for count in counts]
        routes[name] = {
            "num_experts": experts,
            "top_k": int(module.top_k),
            "counts": counts,
            "selection_fractions": fractions,
            "max_selection_fraction": max(fractions, default=0.0),
            "normalized_entropy": entropy(counts),
            "gini": gini(counts),
            "dead_experts": [index for index, count in enumerate(counts) if count == 0],
            "total_selections": total,
        }
    checks = {
        name: {
            "no_dead_experts": not row["dead_experts"],
            "max_selection_fraction_le_0.8": row["max_selection_fraction"] <= 0.8,
            "normalized_entropy_ge_0.5": row["normalized_entropy"] >= 0.5,
        }
        for name, row in routes.items()
    }
    evidence = {
        "schema": "a1-p2-seg-route-audit-r1/v1",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "device": args.device,
        "router_semantics": "deterministic hard Top-2 from step zero",
        "images": len(images),
        "val_list": str(args.val_list.resolve()),
        "val_list_sha256": sha256(args.val_list.resolve()),
        "mean_predictions": prediction_count / len(images),
        "mean_masks": mask_count / len(images),
        "routed_modules": routes,
        "checks": checks,
        "all_checks_passed": all(all(values.values()) for values in checks.values()),
        "note": "Routing thresholds are an explicit diagnostic record for this pilot, not a claim that a single seed proves a formal P2 gain.",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "route_audit.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
