"""Describe single-side GT misses using retained candidates, without assigning causal mechanisms."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

REASONS = (
    "evaluation_match_competition",
    "score_threshold",
    "wrong_class_overlap",
    "localization_threshold",
    "joint_score_localization",
    "wrong_class_nearby",
    "no_near_retained_candidate",
)


def overlaps(boxes, gt):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    gt = np.asarray(gt, dtype=np.float64)
    wh = np.maximum(0, np.minimum(boxes[:, 2:], gt[2:]) - np.maximum(boxes[:, :2], gt[:2]))
    intersection = wh.prod(axis=1)
    area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    gt_area = np.maximum(0, gt[2:] - gt[:2]).prod()
    return intersection / (area + gt_area - intersection + 1e-7)


def diagnose_miss(gt, predictions):
    """Apply a predeclared hierarchy and retain overlapping flags and candidate witnesses."""
    scores = np.asarray(predictions["scores"], dtype=np.float64)
    classes = np.asarray(predictions["classes"], dtype=int)
    iou = overlaps(predictions["boxes"], gt["box"])
    same, high = classes == gt["cls"], scores >= 0.25
    close, near = iou >= 0.5, iou >= 0.1
    flags = {
        "evaluation_match_competition": bool(np.any(same & high & close)),
        "score_threshold": bool(np.any(same & ~high & close)),
        "wrong_class_overlap": bool(np.any(~same & high & close)),
        "localization_threshold": bool(np.any(same & high & near & ~close)),
        "joint_score_localization": bool(np.any(same & ~high & near & ~close)),
        "wrong_class_nearby": bool(np.any(~same & high & near & ~close)),
    }
    reason = next((name for name in REASONS[:-1] if flags[name]), REASONS[-1])

    def witness(mask, rank):
        ids = np.flatnonzero(mask)
        if not len(ids):
            return None
        index = int(ids[np.argmax(rank[ids])])
        return {
            "index": index,
            "box": predictions["boxes"][index],
            "score": float(scores[index]),
            "class": int(classes[index]),
            "iou": float(iou[index]),
        }

    return {
        "reason": reason,
        "flags": flags,
        "same_class_best_iou": witness(same, iou),
        "same_class_best_score_at_iou50": witness(same & close, scores),
        "same_class_high_score_best_iou": witness(same & high, iou),
        "wrong_class_high_score_best_iou": witness(~same & high, iou),
    }


def greedy_gt_ids(row, threshold):
    """Recompute fixed-IoU matches using retained candidates at a selected score threshold."""
    gt = row["boxes"]["gt"]
    pred = row["boxes"]["retained_predictions"]
    scores = np.asarray(pred["scores"])
    matrix = (
        np.stack([overlaps(pred["boxes"], item["box"]) for item in gt], axis=1) if gt else np.zeros((len(scores), 0))
    )
    used = set()
    for i in np.argsort(-scores, kind="stable"):
        if scores[i] < threshold:
            break
        candidates = [j for j, item in enumerate(gt) if j not in used and item["cls"] == int(pred["classes"][i])]
        if not candidates:
            continue
        j = max(candidates, key=lambda j: matrix[i, j])
        if matrix[i, j] >= 0.5:
            used.add(j)
    return used


def keyed(rows):
    result = {row["image"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate image")
    return result


def analyze(square, rect):
    a, b = keyed(square), keyed(rect)
    if a.keys() != b.keys() or len(a) != 512:
        raise ValueError("expected same 512 unique images")
    cases = []
    counts = {tag: Counter() for tag in ("square_only", "rect_only")}
    sizes = {tag: Counter() for tag in counts}
    sweep = {str(t): {"square_tp": 0, "rect_tp": 0} for t in (0.1, 0.2, 0.25, 0.3, 0.5)}
    for image in a:
        left, right = a[image], b[image]
        x, y = left["boxes"]["gt"], right["boxes"]["gt"]
        if len(x) != len(y):
            raise ValueError("GT count differs")
        for row in (left, right):
            actual_ids = greedy_gt_ids(row, 0.25)
            expected_ids = {j for j, item in enumerate(row["boxes"]["gt"]) if item["match"] is not None}
            if actual_ids != expected_ids:
                raise ValueError(f"offline matcher does not reproduce captured matches: {image}")
        for threshold, totals in sweep.items():
            totals["square_tp"] += len(greedy_gt_ids(left, float(threshold)))
            totals["rect_tp"] += len(greedy_gt_ids(right, float(threshold)))
            for tag, row in (("square", left), ("rect", right)):
                count = sum(score >= float(threshold) for score in row["boxes"]["retained_predictions"]["scores"])
                totals[f"{tag}_predictions"] = totals.get(f"{tag}_predictions", 0) + count
                totals[f"{tag}_fp"] = totals[f"{tag}_predictions"] - totals[f"{tag}_tp"]
        for gx, gy in zip(x, y):
            if (
                gx["gt_id"] != gy["gt_id"]
                or gx["cls"] != gy["cls"]
                or max(abs(p - q) for p, q in zip(gx["box"], gy["box"])) > 0.001
            ):
                raise ValueError("GT identity mismatch")
            if (gx["match"] is None) == (gy["match"] is None):
                continue
            direction = "square_only" if gx["match"] else "rect_only"
            missing, detected, target = (right, gx, gy) if gx["match"] else (left, gy, gx)
            diagnosis = diagnose_miss(target, missing["boxes"]["retained_predictions"])
            box = target["box"]
            area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            size = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
            counts[direction][diagnosis["reason"]] += 1
            sizes[direction][size] += 1
            cases.append(
                {
                    "image": image,
                    "gt_id": target["gt_id"],
                    "class": target["cls"],
                    "gt_box": box,
                    "area_px2": area,
                    "size": size,
                    "direction": direction,
                    "detected_match": detected["match"],
                    "missing_side": diagnosis,
                }
            )
    return {
        "images": len(a),
        "gt_total": sum(len(row["boxes"]["gt"]) for row in a.values()),
        "reasons": {tag: {key: counter[key] for key in REASONS} for tag, counter in counts.items()},
        "sizes": sizes,
        "score_sweep_iou50": sweep,
    }, cases


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.source / "evidence.json").read_text())
    if original["status"] != "completed":
        raise ValueError("source audit is incomplete")
    evidence = {
        "status": "running",
        "source": str(args.source),
        "source_evidence_sha256": digest(args.source / "evidence.json"),
        "script_sha256": digest(Path(__file__)),
        "cells": {},
        "reason_priority": list(REASONS),
        "limits": "Descriptive retained-candidate analysis, not TIDE or causal attribution. Native top-300, score floor .001. Overlapping flags retained. Threshold sweep is diagnostic, not a new tuned deployment setting.",
    }
    for name in original["cells"]:
        paths = [args.source / name / tag / "per_image_inverse_letterbox.json" for tag in ("rect0", "rect1")]
        summary, cases = analyze(*(json.loads(path.read_text()) for path in paths))
        evidence["cells"][name] = {"summary": summary, "input_sha256": [digest(path) for path in paths]}
        (args.output / f"{name}_single_side_cases.json").write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n")
    evidence["status"] = "completed"
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
