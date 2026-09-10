#!/usr/bin/env python3
"""Evaluate P2-E r4 conflict-resolution pilots on fixed val512."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_p2_e2e_precision_diagnostics_r1 import LOCKED_SHA, git_state, run_cell, sha256


RUNS = (
    ("b_conflict_overlap_seed260829_5ep", "overlap"),
    ("b_conflict_align_seed260829_5ep", "align"),
    ("d_conflict_overlap_seed260829_5ep", "overlap"),
    ("d_conflict_align_seed260829_5ep", "align"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output / "evidence.json"
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("status") == "completed":
            raise FileExistsError(f"refusing to overwrite completed output: {args.output}")
    else:
        val_list = args.data.parent / "val2017.txt"
        val_lines = [line for line in val_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(val_lines) != 512:
            raise ValueError(f"expected fixed 512-image val2017.txt: {val_list}")
        evidence = {
            "schema": "a1-p2-e2e-conflict-resolution-r4-eval/v1",
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
            "factor": "one_to_one_conflict_resolution_metric",
            "runs": {},
        }

    os.environ["A1_TAL_ASSIGNMENT_AUDIT"] = "1"
    for run_name, metric in RUNS:
        if run_name in evidence["runs"] and evidence["runs"][run_name].get("metrics"):
            print(f"[p2-e2e-r4-eval] resume: {run_name} already completed", flush=True)
            continue
        checkpoint = args.project / run_name / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        os.environ["A1_E2E_O2O_CONFLICT_METRIC"] = metric
        print(f"[p2-e2e-r4-eval] {run_name}: metric={metric} checkpoint={checkpoint}", flush=True)
        result = run_cell(checkpoint, args.data, args.output / run_name, args.device)
        result.update({"run_name": run_name, "conflict_metric": metric})
        evidence["runs"][run_name] = result
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    evidence["git"] = git_state(Path.cwd())
    evidence["status"] = "completed"
    evidence["completed_at"] = datetime.now(timezone.utc).isoformat()
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
