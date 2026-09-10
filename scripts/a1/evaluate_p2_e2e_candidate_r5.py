#!/usr/bin/env python3
"""Evaluate P2-E r5 candidate-budget pilots on fixed val512."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


LOCKED_SHA = "acce839c7e895d6b179de7f7093fa879e237cc7b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: object, default: float = 0.0) -> float:
    if value in {None, "", "None"}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def number(rows: list[dict[str, str]], key: str) -> float:
    values = [as_float(row.get(key)) for row in rows if row.get(key) not in {None, "", "None"}]
    return sum(values) / max(len(values), 1)


def summarize(per_image: Path) -> dict:
    with per_image.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 512:
        raise ValueError(f"expected 512 per-image rows, got {len(rows)} in {per_image}")
    total_gt = sum(int(as_float(row.get("gt_count"))) for row in rows)
    candidate_gt = sum(int(as_float(row.get("assign_assigner_candidate_gt_count"))) for row in rows)
    final_gt = sum(int(as_float(row.get("assign_assigner_final_gt_count"))) for row in rows)
    return {
        "images": len(rows),
        "total_gt": total_gt,
        "candidate_positive_per_image": number(rows, "assign_assigner_candidate_positive_count"),
        "post_conflict_positive_per_image": number(rows, "assign_assigner_post_conflict_positive_count"),
        "final_positive_per_image": number(rows, "assign_assigner_final_positive_count"),
        "candidate_gt_coverage": candidate_gt / max(total_gt, 1),
        "final_gt_coverage": final_gt / max(total_gt, 1),
        "conflict_anchor_per_image": number(rows, "assign_assigner_conflict_anchor_count"),
        "matched_iou_mean": number(rows, "assign_assignment_matched_iou_mean"),
        "cls_loss_mean": number(rows, "assign_cls_loss"),
        "box_loss_mean": number(rows, "assign_box_loss"),
        "dfl_loss_mean": number(rows, "assign_dfl_loss"),
        "prediction_per_image": number(rows, "prediction_count"),
        "fp_iou50_per_image": number(rows, "fp_iou50"),
        "fn_iou50_per_image": number(rows, "fn_iou50"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(repo))
    from run_p2_e2e_precision_diagnostics_r1 import git_state, run_cell

    evidence_path = output / "evidence.json"
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("status") == "completed":
            raise FileExistsError(f"refusing to overwrite completed output: {output}")
    else:
        data = Path(protocol["data"]["yaml"])
        val_list = Path(protocol["data"]["val_list"])
        evidence = {
            "schema": "a1-p2-e2e-candidate-budget-r5-eval/v1",
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_ref": LOCKED_SHA,
            "protocol": str(protocol_path),
            "protocol_sha256": sha256(protocol_path),
            "data_yaml": str(data),
            "data_yaml_sha256": sha256(data),
            "val_list": str(val_list),
            "val_list_sha256": sha256(val_list),
            "images": 512,
            "imgsz": 640,
            "batch": 1,
            "device": args.device,
            "factor": "one_to_one_tal_topk",
            "runs": {},
        }
    for name, spec in protocol["runs"].items():
        if name in evidence["runs"] and evidence["runs"][name].get("metrics"):
            print(f"[p2-e2e-r5-eval] resume: {name}", flush=True)
            continue
        # Ultralytics writes the requested project/name directly under run_root;
        # there is no intermediate ``train`` directory in these requests.
        checkpoint = Path(protocol["run_root"]) / f"{name}_seed260829_5ep" / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        topk = int(spec["topk"])
        os.environ["A1_E2E_O2O_TAL_TOPK"] = str(topk)
        os.environ["A1_E2E_O2O_TAL_TOPK2"] = "1"
        os.environ["A1_TAL_ASSIGNMENT_AUDIT"] = "1"
        print(f"[p2-e2e-r5-eval] {name}: topk={topk} checkpoint={checkpoint}", flush=True)
        result = run_cell(checkpoint, Path(protocol["data"]["yaml"]), output / name, args.device)
        result.update({"run_name": name, "cell": spec["cell"], "topk": topk, "topk2": 1})
        result["assignment_summary"] = summarize(Path(result["per_image_metrics"]))
        evidence["runs"][name] = result
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evidence["git"] = git_state(repo)
    evidence["status"] = "completed"
    evidence["completed_at"] = datetime.now(timezone.utc).isoformat()
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
