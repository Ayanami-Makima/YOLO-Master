"""Uniform last-EMA reevaluation; fixed-confidence errors are descriptive, not TIDE."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.a1.run_p2_e2e_precision_diagnostics_r1 import (
    PrecisionValidatorMixin,
    greedy_matches,
    iou_matrix,
    sha256,
    write_csv,
)
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator


def dump(path, value):
    """Write JSON evidence, including numpy-valued PR curves."""
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist(), allow_nan=False) + "\n")


def errors(pred, target):
    """Class-aware greedy IoU50 matching; mutually exclusive FP hierarchy at conf .25."""
    pred = {k: pred[k][pred["conf"] >= 0.25].detach().cpu() for k in ("bboxes", "conf", "cls")}
    target = {k: target[k].detach().cpu() for k in ("bboxes", "cls")}
    matches, used = greedy_matches(pred, target)
    matched_pred = {p for p, _, _ in matches}
    overlap = iou_matrix(pred["bboxes"], target["bboxes"])
    counts = Counter()
    for pi in range(len(pred["cls"])):
        if pi in matched_pred:
            continue
        same = pred["cls"][pi] == target["cls"]
        if bool(((overlap[pi] >= 0.5) & same).any()):
            counts["duplicate"] += 1
        elif bool((overlap[pi] >= 0.5).any()):
            counts["classification"] += 1
        elif bool(((overlap[pi] >= 0.1) & same).any()):
            counts["localization"] += 1
        elif bool((overlap[pi] >= 0.1).any()):
            counts["classification_localization"] += 1
        else:
            counts["background"] += 1
    return {
        "gt": len(target["cls"]),
        "tp": len(matches),
        "fp": sum(counts.values()),
        "fn": len(target["cls"]) - len(used),
        "matched_iou_sum": sum(v for _, _, v in matches),
        **{
            f"fp_{k}": counts[k]
            for k in ("duplicate", "classification", "localization", "classification_localization", "background")
        },
    }


def gpu_memory(device):
    return int(
        subprocess.check_output(
            ["nvidia-smi", "-i", str(device), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        ).strip()
    )


class Validator(PrecisionValidatorMixin, DetectionValidator):
    """Retain standard low-confidence AP alongside independent fixed-confidence errors."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.error_rows = []

    def update_metrics(self, preds, batch):
        if len(self.error_rows) % 32 == 0 and gpu_memory(self.device.index) > 22528:
            raise RuntimeError("GPU total memory exceeded user 22 GiB ceiling")
        for si, pred in enumerate(preds):
            target = self._prepare_batch(si, batch)
            self.error_rows.append({"image": str(target["im_file"]), **errors(pred, target)})
        super().update_metrics(preds, batch)


def summarize(validator):
    rows = validator.assignment_rows
    assert len(rows) == 512
    for row in rows:
        assert row.get("assigner_branch") == "one2one", row
        assert not any("error" in key for key in row), row
        for key in ("positive_count", "unique_matched_gt_count", "box_loss", "cls_loss", "dfl_loss"):
            assert row.get(key) is not None and np.isfinite(row[key]), (key, row)
        assert not row["positive_count"] or row["assignment_matched_iou_mean"] is not None, row
    totals = {k: sum(r[k] for r in validator.error_rows) for k in validator.error_rows[0] if k != "image"}
    totals["precision"] = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    totals["recall"] = totals["tp"] / max(totals["gt"], 1)
    totals["matched_iou_mean"] = totals["matched_iou_sum"] / max(totals["tp"], 1)
    positives = sum(r["positive_count"] for r in rows)
    assignment = {
        "images": len(rows),
        "foreground_images": sum(r["gt"] > 0 for r in validator.error_rows),
        "positive_count": positives,
        "unique_matched_gt": sum(r["unique_matched_gt_count"] for r in rows),
        "matched_iou_positive_weighted": sum(
            (r["assignment_matched_iou_mean"] or 0) * r["positive_count"] for r in rows
        )
        / max(positives, 1),
        **{f"{k}_all_image_mean": sum(r[k] for r in rows) / len(rows) for k in ("box_loss", "cls_loss", "dfl_loss")},
    }
    return {"conf025_iou050": totals, "assignment": assignment}


def run_cell(request, output, device):
    checkpoint = Path(request["params"]["project"]) / request["params"]["name"] / "weights/last.pt"
    with (checkpoint.parents[1] / "results.csv").open() as handle:
        epochs = list(csv.DictReader(handle))
    assert int(float(epochs[-1]["epoch"])) == 5
    digest = sha256(checkpoint)
    yolo = YOLO(str(checkpoint), task="detect")
    head = yolo.model.model[-1]
    assert head.end2end
    assert getattr(head, "p2_o2o_gradient_alpha", 0.0) == request["p2_bridge"]["alpha"]
    yolo.model.args = get_cfg(overrides=yolo.model.args)
    validator = Validator(
        args={
            "model": str(checkpoint),
            "data": request["inputs"]["data"],
            "split": "val",
            "imgsz": 640,
            "batch": 1,
            "workers": 0,
            "device": str(device),
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "quantize": None,
            "plots": False,
            "verbose": False,
            "save_json": False,
            "save_txt": False,
            "project": str(output),
            "name": "validator",
            "exist_ok": False,
        }
    )
    metrics = validator(model=yolo.model)
    write_csv(output / "per_image_lowconf.csv", validator.precision_rows)
    write_csv(output / "assignment.csv", validator.assignment_rows)
    write_csv(output / "errors_conf025.csv", validator.error_rows)
    dump(output / "pr_curves.json", validator.metrics.curves_results)
    dump(
        output / "per_class_ap.json",
        {"class_ids": validator.metrics.box.ap_class_index, "ap_iou50_to95": validator.metrics.box.all_ap},
    )
    summary = summarize(validator)
    assert digest == sha256(checkpoint), "Checkpoint changed during evaluation"
    image_paths = [r["image"] for r in validator.error_rows]
    assert len(set(image_paths)) == 512
    result = {"checkpoint": str(checkpoint), "sha256": digest, "metrics": metrics, **summary, "images": image_paths}
    dump(output / "result.json", result)
    del validator, yolo
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    os.environ.update(A1_E2E_O2O_TAL_TOPK="7", A1_E2E_O2O_TAL_TOPK2="1", A1_E2E_O2O_CLS_GAIN="1.0")
    torch.set_num_threads(2)
    protocol = json.loads((args.root / "protocol.json").read_text())
    for relative, digest in protocol["source_hashes"].items():
        assert sha256(ROOT / relative) == digest, relative
    requests = []
    for path in protocol["order"]:
        assert sha256(Path(path)) == protocol["requests"][path], path
        request = json.loads(Path(path).read_text())
        for path, digest in request["p2_bridge"]["input_hashes"].items():
            assert sha256(Path(path)) == digest, path
        requests.append(request)
    assert len({r["inputs"]["data"] for r in requests}) == 1
    data = check_det_dataset(requests[0]["inputs"]["data"])
    val_list = Path(data["val"])
    assert len([s for s in val_list.read_text().splitlines() if s.strip()]) == 512
    assert sha256(val_list) == sha256(Path(requests[0]["inputs"]["data"]).parent / "val2017.txt")
    assert gpu_memory(args.device) <= 17408, "Insufficient shared GPU headroom"
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(args.device).total_memory)
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cells": {},
        "protocol_sha256": sha256(args.root / "protocol.json"),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "val_list": str(val_list),
        "val_sha256": sha256(val_list),
        "batch": 1,
        "imgsz": 640,
        "precision": "FP32",
        "map_conf": 0.001,
        "error_conf": 0.25,
        "source_hashes": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (
                Path(__file__),
                ROOT / "scripts/a1/run_p2_e2e_precision_diagnostics_r1.py",
                ROOT / "ultralytics/models/yolo/detect/val.py",
                ROOT / "ultralytics/engine/validator.py",
            )
        },
        "limitations": "Single seed; D alpha0 resumed. FP buckets descriptive, not TIDE causal attribution. No latency claim.",
    }
    dump(args.output / "evidence.json", evidence)
    try:
        for request in requests:
            name = request["params"]["name"]
            print(f"Evaluating {name}", flush=True)
            evidence["cells"][name] = run_cell(request, args.output / name, args.device)
            images = [r["images"] for r in evidence["cells"].values()]
            assert all(x == images[0] for x in images)
            dump(args.output / "evidence.json", evidence)
        evidence["status"] = "completed"
    except Exception:
        evidence["status"] = "failed"
        evidence["error"] = traceback.format_exc()
        raise
    finally:
        evidence["updated_at"] = datetime.now(timezone.utc).isoformat()
        dump(args.output / "evidence.json", evidence)


if __name__ == "__main__":
    main()
