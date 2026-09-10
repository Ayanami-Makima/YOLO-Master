#!/usr/bin/env python3
"""Evaluate the four controlled P2-E r2 checkpoints on the fixed 512-image val set.

This reuses the diagnostic validator from P2-E r1.  It intentionally evaluates
the final ``last.pt`` checkpoint from each pilot run and never modifies a
checkpoint or the training output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the diagnostic validator imports the synchronized repository source,
# not a different editable ultralytics checkout on the remote host.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_p2_e2e_precision_diagnostics_r1 import LOCKED_SHA, git_state, run_cell, sha256


RUNS = (
    "b_gain1p0_seed260829_5ep",
    "b_gain1p2_seed260829_5ep",
    "d_gain1p0_seed260829_5ep",
    "d_gain1p2_seed260829_5ep",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--checkpoint-kind", choices=("last", "best"), default="last")
    args = parser.parse_args()

    evidence_path = args.output / "evidence.json"
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("status") == "completed":
            raise FileExistsError(f"refusing to overwrite completed output: {args.output}")
    else:
        val_list = args.data.parent / "val2017.txt"
        val_lines = [line for line in val_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(val_lines) != 512:
            raise ValueError(f"expected fixed 512-image val2017.txt next to data YAML: {val_list}")
        evidence = {
            "schema": "a1-p2-e2e-loss-r2-eval/v1",
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_ref": LOCKED_SHA,
            "device": args.device,
            "data_yaml": str(args.data),
            "data_yaml_sha256": sha256(args.data),
            "val_list": str(val_list),
            "val_list_sha256": sha256(val_list),
            "images": 512,
            "imgsz": 640,
            "batch": 1,
            "checkpoint_kind": args.checkpoint_kind,
            "runs": {},
        }
    args.output.mkdir(parents=True, exist_ok=True)

    for run_name in RUNS:
        if run_name in evidence["runs"] and evidence["runs"][run_name].get("metrics"):
            print(f"[p2-e2e-r2-eval] resume: {run_name} already completed", flush=True)
            continue
        checkpoint = args.project / run_name / "weights" / f"{args.checkpoint_kind}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(f"[p2-e2e-r2-eval] {run_name}: {checkpoint}", flush=True)
        result = run_cell(checkpoint, args.data, args.output / run_name, args.device)
        result["run_name"] = run_name
        evidence["runs"][run_name] = result
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    evidence["git"] = git_state(Path.cwd())
    evidence["status"] = "completed"
    evidence["completed_at"] = datetime.now(timezone.utc).isoformat()
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
