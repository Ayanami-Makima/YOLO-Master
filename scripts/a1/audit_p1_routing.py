#!/usr/bin/env python3
"""Collect expert-selection evidence from the trained P1 C/D checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from evaluate_p1_matrix import check_matrix, digest, load_images, load_model, prediction_params, resolve_images, write_json


def summarize_counts(counts: Counter, num_experts: int, top_k: int) -> dict:
    """Return normalized selection health without inventing unobserved usage."""
    total = sum(counts.values())
    values = [counts[index] for index in range(num_experts)]
    fractions = [value / total if total else 0.0 for value in values]
    entropy = -sum(value * math.log(value) for value in fractions if value > 0)
    normalized = entropy / math.log(num_experts) if num_experts > 1 and total else 0.0
    return {
        "num_experts": num_experts,
        "top_k": top_k,
        "selections": total,
        "counts": values,
        "fractions": fractions,
        "dead_experts_on_sample": [index for index, value in enumerate(values) if value == 0],
        "max_selection_fraction": max(fractions, default=0.0),
        "normalized_entropy": normalized,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    args.configs, args.project, args.output = args.configs.resolve(), args.project.resolve(), args.output.resolve()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    gate = check_matrix(args.configs, args.project)
    if gate["status"] != "ready":
        raise ValueError(f"training evidence gate is not ready: {gate['blockers']}")
    data_yaml = Path(gate["protocol"].get("evaluation_data_yaml", gate["protocol"]["data_yaml"]))
    paths = resolve_images(data_yaml, None, args.limit)
    images = load_images(paths)
    report = {
        "schema_version": 1,
        "status": "completed",
        "protocol_sha256": gate["identity"]["protocol_sha256"],
        "images": [{"path": str(path), "sha256": digest(path)} for path in paths],
        "device": args.device,
        "cells": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "interpretation": "Selection frequency on a fixed sample is a routing diagnostic, not proof of global balance.",
    }
    for cell in "cd":
        model = load_model(gate, cell)
        routed = []
        for name, module in model.model.named_modules():
            routing = getattr(module, "routing", None)
            if routing is not None and hasattr(routing, "num_experts") and hasattr(routing, "top_k"):
                routed.append((f"{name}.routing", routing))
        if not routed:
            raise ValueError(f"{cell}: no routed modules found")
        counters = {name: Counter() for name, _ in routed}
        handles = []
        for name, routing in routed:
            def hook(_module, _inputs, output, *, key=name):
                indices = output[1].detach().cpu().reshape(-1).tolist()
                counters[key].update(int(index) for index in indices)

            handles.append(routing.register_forward_hook(hook))
        params = prediction_params(argparse.Namespace(imgsz=640, conf=0.25, nms_iou=0.7, max_det=300), args.device)
        try:
            model.predict(source=images[0], **params)
            for counter in counters.values():
                counter.clear()
            for image in images:
                model.predict(source=image, **params)
        finally:
            for handle in handles:
                handle.remove()
        modules = {}
        for name, routing in routed:
            modules[name] = summarize_counts(counters[name], int(routing.num_experts), int(routing.top_k))
            expected = len(images) * int(routing.top_k)
            if modules[name]["selections"] != expected:
                raise ValueError(f"{cell}:{name}: expected {expected} selections, got {modules[name]['selections']}")
        report["cells"][cell] = {
            "checkpoint": gate["cells"][cell]["weights"]["best.pt"],
            "routed_modules": len(modules),
            "modules": modules,
        }
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output), "cells": {key: value["routed_modules"] for key, value in report["cells"].items()}}))


if __name__ == "__main__":
    main()
