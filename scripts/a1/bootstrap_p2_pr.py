"""Paired image bootstrap for the native-TP PR/AP summaries."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.a1.analyze_p2_pr_ranking import analyze


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=1000)
    p.add_argument("--seed", type=int, default=260829)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    original = json.loads((args.source / "evidence.json").read_text())
    rng = np.random.default_rng(args.seed)
    out = {"status": "running", "source": str(args.source), "replicates": args.replicates, "seed": args.seed, "cells": {}}
    args.output.mkdir(parents=True)
    for name in original["cells"]:
        paths = {
            t: args.source / name / t / "per_image_inverse_letterbox.json"
            for t in ("rect0", "rect1")
        }
        rows = {t: json.loads(path.read_text()) for t, path in paths.items()}
        if set(r["image"] for r in rows["rect0"]) != set(r["image"] for r in rows["rect1"]):
            raise ValueError("paired image set mismatch")
        # Validators may emit a slightly different order; explicitly align by image.
        aligned = {t: {r["image"]: r for r in values} for t, values in rows.items()}
        images = sorted(aligned["rect0"])
        rows = {t: [aligned[t][image] for image in images] for t in ("rect0", "rect1")}
        deltas = {"map": [], "ap50": [], "ap75": [], "ap90": []}
        n = len(rows["rect0"])
        for _ in range(args.replicates):
            indices = rng.integers(0, n, size=n)
            sampled = {
                t: [rows[t][int(i)] for i in indices] for t in ("rect0", "rect1")
            }
            summaries = {t: analyze(sampled[t])[0] for t in ("rect0", "rect1")}
            for key, idx in (("map", None), ("ap50", 0), ("ap75", 5), ("ap90", 8)):
                if idx is None:
                    value = summaries["rect1"]["map"] - summaries["rect0"]["map"]
                else:
                    value = summaries["rect1"]["ap_by_iou"][idx] - summaries["rect0"]["ap_by_iou"][idx]
                deltas[key].append(float(value))
        out["cells"][name] = {
            metric: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            }
            for metric, values in deltas.items()
        }
        (args.output / "evidence.json").write_text(json.dumps(out, indent=2) + "\n")
        print(name, out["cells"][name], flush=True)
    out["status"] = "completed"
    (args.output / "evidence.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
