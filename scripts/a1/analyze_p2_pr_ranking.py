"""Reconstruct standard AP/PR and descriptive class-wise TP/FP score separation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.a1.summarize_p2_single_side import digest
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class, box_iou, compute_ap


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=lambda x: x.tolist()) + "\n")


def separation(scores, labels):
    """Probability a uniformly chosen TP outranks a FP, with half credit for ties."""
    positive = scores[labels]
    negative = np.sort(scores[~labels])
    if len(positive) == 0 or len(negative) == 0:
        return None
    less = np.searchsorted(negative, positive, side="left")
    less_equal = np.searchsorted(negative, positive, side="right")
    return float(np.mean((less + less_equal) / (2 * len(negative))))


def analyze(rows):
    tp, scores, classes, targets = [], [], [], []
    matcher = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    native = all("native_tp" in r["boxes"]["retained_predictions"] for r in rows)
    for row in rows:
        p = row["boxes"]["retained_predictions"]
        gt = row["boxes"]["gt"]
        pc = torch.tensor(p["classes"], dtype=torch.float32)
        tc = torch.tensor([g["cls"] for g in gt], dtype=torch.float32)
        if native:
            correct = np.asarray(p["native_tp"], dtype=bool).reshape(-1, 10)
        else:
            pb = torch.tensor(p["boxes"], dtype=torch.float32).reshape(-1, 4)
            tb = torch.tensor([g["box"] for g in gt], dtype=torch.float32).reshape(-1, 4)
            correct = BaseValidator.match_predictions(matcher, pc, tc, box_iou(tb, pb)).numpy()
        tp.append(correct)
        scores.append(np.asarray(p["scores"], dtype=np.float32))
        classes.append(pc.numpy())
        targets.append(tc.numpy())
    tp, scores, classes, targets = map(np.concatenate, (tp, scores, classes, targets))
    result = ap_per_class(tp, scores, classes, targets)
    ap, ids = result[5:7]
    x = np.linspace(0, 1, 1000)
    curves, perclass = {}, []
    for ci, cls in enumerate(ids):
        # Preserve ap_per_class's global sorting, including equal-score tie order.
        selected = np.argsort(-scores)
        selected = selected[classes[selected] == cls]
        hits = tp[selected]
        count = int((targets == cls).sum())
        cs = scores[selected]
        pr = []
        for j in range(10):
            if not len(selected):
                pr.append(np.zeros_like(x))
                continue
            total = hits[:, j].cumsum()
            recall = total / count
            precision = total / np.arange(1, len(selected) + 1)
            _, envelope, recall_axis = compute_ap(recall, precision)
            pr.append(np.interp(x, recall_axis, envelope))
        curves[str(int(cls))] = pr
        perclass.append(
            {
                "class": int(cls),
                "gt": count,
                "predictions": len(selected),
                "ap_by_iou": ap[ci],
                "retained_tp_by_iou": hits.sum(axis=0),
                "score_separation_auc": [separation(cs, hits[:, j]) for j in (0, 5, 8)],
                "tp_score_median_iou50": float(np.median(cs[hits[:, 0]])) if hits[:, 0].any() else None,
                "fp_score_median_iou50": float(np.median(cs[~hits[:, 0]])) if (~hits[:, 0]).any() else None,
            }
        )
    return {
        "coordinate_source": "native_validator_tp" if native else "clipped_original_coordinates",
        "map": float(ap.mean()),
        "ap_by_iou": ap.mean(axis=0),
        "classes": perclass,
        "predictions": len(scores),
        "gt": len(targets),
        "retained_recall_micro_by_iou": tp.sum(axis=0) / len(targets),
    }, {
        "recall_axis": x,
        "pr_precision_by_class_iou": curves,
        "confidence_axis": result[10],
        "precision_vs_conf_iou50": result[7],
        "recall_vs_conf_iou50": result[8],
        "class_ids": ids,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=512)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.source / "evidence.json").read_text())
    if original["status"] != "completed":
        raise ValueError("capture incomplete")
    torch.set_num_threads(2)
    evidence = {
        "status": "running",
        "source": str(args.source),
        "source_sha256": digest(args.source / "evidence.json"),
        "script_sha256": digest(Path(__file__)),
        "iou_axis": np.linspace(0.5, 0.95, 10),
        "cells": {},
        "limit": "AP uses native matching, not confidence-greedy diagnostic matching. Score separation is conditional on TP labels and the retained candidate set, not a calibration metric or causal ranking decomposition.",
    }
    for name, cell in original["cells"].items():
        passes = {}
        for tag in ("rect0", "rect1"):
            path = args.source / name / tag / "per_image_inverse_letterbox.json"
            rows = json.loads(path.read_text())
            if len(rows) != args.expected_images or len({r["image"] for r in rows}) != args.expected_images:
                raise ValueError(f"{args.expected_images} unique images required")
            summary, curves = analyze(rows)
            expected = cell["passes"][tag]["metrics"]["metrics/mAP50-95(B)"]
            summary["map_reproduction_error"] = abs(summary["map"] - expected)
            summary["input_sha256"] = digest(path)
            passes[tag] = summary
            write(args.output / f"{name}_{tag}_curves.json", curves)
            print(name, tag, "mAP", summary["map"], "error", summary["map_reproduction_error"], flush=True)
        evidence["cells"][name] = passes
        write(args.output / "evidence.json", evidence)
    evidence["status"] = (
        "completed"
        if all(p["map_reproduction_error"] < 1e-7 for c in evidence["cells"].values() for p in c.values())
        else "metric_reproduction_mismatch"
    )
    write(args.output / "evidence.json", evidence)
    print(evidence["status"])


if __name__ == "__main__":
    main()
