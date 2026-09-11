#!/usr/bin/env python3
"""Audit routed-expert selections in the P1 factorial C/D preflight checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from audit_p1_routing import summarize_counts
from evaluate_p1_matrix import (
    digest,
    load_images,
    prediction_params,
    resolve_images,
    write_json,
)


def routing_gate(summary: dict) -> dict:
    """Apply a conservative sample-level routing health gate."""
    reasons = []
    if summary["selections"] == 0:
        reasons.append("no selections recorded")
    if summary["selections"] != summary["expected_selections"]:
        reasons.append(f"expected {summary['expected_selections']} selections, got {summary['selections']}")
    if summary["dead_experts_on_sample"]:
        reasons.append(f"dead experts: {summary['dead_experts_on_sample']}")
    if summary["max_image_selection_fraction"] > 0.8:
        reasons.append(f"max image selection fraction {summary['max_image_selection_fraction']:.3f} exceeds 0.8")
    if summary["normalized_entropy"] < 0.5:
        reasons.append(f"normalized entropy {summary['normalized_entropy']:.3f} is below 0.5")
    return {"passed": not reasons, "reasons": reasons}


def status_exit_code(status: str) -> int:
    """Map the persisted gate status to the command-line process exit code."""
    if status not in {"passed", "failed"}:
        raise ValueError(f"unsupported routing audit status: {status}")
    return 0 if status == "passed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", required=True, type=Path)
    parser.add_argument("--d", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    for path in (args.c, args.d, args.data):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)

    from ultralytics import YOLO

    paths = resolve_images(args.data.resolve(), None, args.limit)
    images = load_images(paths)
    report = {
        "schema_version": 2,
        "status": "running",
        "data": str(args.data.resolve()),
        "images": [{"path": str(path), "sha256": digest(path)} for path in paths],
        "device": args.device,
        "cells": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_definitions": {
            "selection_share": "per-expert count / (images * top_k)",
            "image_selection_fraction": "per-expert count / images",
        },
        "gate_thresholds": {
            "dead_experts": 0,
            "max_image_selection_fraction": 0.8,
            "min_normalized_entropy": 0.5,
        },
        "interpretation": "Fixed-sample routing diagnostic; it does not prove population-wide balance.",
    }
    for cell, checkpoint in (("c", args.c.resolve()), ("d", args.d.resolve())):
        model = YOLO(checkpoint, task="detect")
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

            def hook(_module, _inputs, output, *, key=name, counter_map=counters):
                indices = output[1].detach().cpu().reshape(-1).tolist()
                counter_map[key].update(int(index) for index in indices)

            handles.append(routing.register_forward_hook(hook))
        params = prediction_params(
            argparse.Namespace(imgsz=640, conf=0.25, nms_iou=0.7, max_det=300),
            args.device,
        )
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
            summary = summarize_counts(
                counters[name],
                int(routing.num_experts),
                int(routing.top_k),
                len(images),
            )
            summary["gate"] = routing_gate(summary)
            modules[name] = summary
        report["cells"][cell] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "routed_modules": len(modules),
            "passed": all(summary["gate"]["passed"] for summary in modules.values()),
            "modules": modules,
        }
    report["status"] = "passed" if all(cell["passed"] for cell in report["cells"].values()) else "failed"
    write_json(args.output.resolve(), report)
    print(
        json.dumps(
            {"status": report["status"], "output": str(args.output.resolve())},
            ensure_ascii=False,
        )
    )
    return status_exit_code(report["status"])


if __name__ == "__main__":
    raise SystemExit(main())
