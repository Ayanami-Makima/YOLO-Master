"""Aggregate paired gradient audits without treating repeated images as independent samples."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    runs = []
    sources = []
    for seed in (260829, 260830, 260831):
        path = args.input_root / f"evidence_seed{seed}.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        assert run["status"] == "completed" and run["seed"] == seed
        assert run["optimizer_steps"] == 0 and len(run["rows"]) == 64
        assert all(r["forward_max_error"] == 0 for r in run["rows"])
        assert all(not c["state_changed_keys"] for c in run["checkpoints"].values())
        runs.append(run)
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    reference = [(r["cell"], r["image_sha256"]) for r in runs[0]["rows"]]
    assert all([(r["cell"], r["image_sha256"]) for r in run["rows"]] == reference for run in runs)
    per_seed = []
    for run in runs:
        for cell in ("b", "d"):
            rows = [r for r in run["rows"] if r["cell"] == cell]
            item = {"seed": run["seed"], "cell": cell, "images": len(rows), "gt": sum(r["gt"] for r in rows)}
            for group in ("factor", "router"):
                cosines = [r["branches"]["bridge_one2one"]["gradients"][group]["cosine_with_one2many"] for r in rows]
                cosines = [v for v in cosines if v is not None]
                item[group] = {
                    "native_o2o_l2_mean": statistics.mean(
                        r["branches"]["native_one2one"]["gradients"][group]["l2"] for r in rows
                    ),
                    "bridge_o2o_l2_mean": statistics.mean(
                        r["branches"]["bridge_one2one"]["gradients"][group]["l2"] for r in rows
                    ),
                    "cosine_mean": statistics.mean(cosines) if cosines else None,
                    "negative_images": sum(v < 0 for v in cosines),
                    "defined_cosine_images": len(cosines),
                }
            per_seed.append(item)
    summary = {
        "schema": "p2-gradient-bridge-summary/v1",
        "status": "completed",
        "sources": sources,
        "independent_training_seeds": 3,
        "distinct_input_images": 32,
        "image_offset": runs[0].get("offset", 0),
        "paired_seed_cell_image_runs": 192,
        "forward_max_error": 0.0,
        "changed_state_tensors": 0,
        "optimizer_steps": 0,
        "per_seed": per_seed,
        "seed_statistics": {},
        "limitation": "Fixed trained checkpoints and 32 contiguous train images; diagnostic only, no accuracy claim",
    }
    for cell in ("b", "d"):
        summary["seed_statistics"][cell] = {}
        for group in ("factor", "router"):
            vals = [r[group]["cosine_mean"] for r in per_seed if r["cell"] == cell]
            vals = [v for v in vals if v is not None]
            summary["seed_statistics"][cell][group] = {
                "cosine_seed_mean": statistics.mean(vals) if vals else None,
                "cosine_seed_std": statistics.stdev(vals) if len(vals) > 1 else None,
            }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["per_seed"], indent=2))


if __name__ == "__main__":
    main()
