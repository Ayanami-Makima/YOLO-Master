"""Compare native one-to-one assignment and component losses under square/rect padding."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.a1.evaluate_p2_gradient_bridge import dump, errors, gpu_memory
from scripts.a1.run_p2_e2e_precision_diagnostics_r1 import PrecisionValidatorMixin, sha256, write_csv
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine import validator as engine
from ultralytics.models.yolo.detect.val import DetectionValidator


class AssignmentValidator(PrecisionValidatorMixin, DetectionValidator):
    """Native assignment diagnostics plus fixed-confidence prediction errors."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.error_rows = []

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            target = self._prepare_batch(si, batch)
            self.error_rows.append({"image": str(target["im_file"]), **errors(pred, target)})
        super().update_metrics(preds, batch)


def assignment_summary(rows):
    assert len(rows) == 512
    assert all(row.get("assigner_branch") == "one2one" for row in rows)
    assert not any(any("error" in key for key in row) for row in rows)
    positives = sum(row["positive_count"] for row in rows)
    return {
        "images": len(rows),
        "positive_count": positives,
        "unique_matched_gt": sum(row["unique_matched_gt_count"] for row in rows),
        "positive_weighted_iou": sum(
            (row["assignment_matched_iou_mean"] or 0.0) * row["positive_count"] for row in rows
        )
        / max(positives, 1),
        "box_loss_mean": float(np.mean([row["box_loss"] for row in rows])),
        "cls_loss_mean": float(np.mean([row["cls_loss"] for row in rows])),
        "dfl_loss_mean": float(np.mean([row["dfl_loss"] for row in rows])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    protocol = json.loads((args.root / "protocol.json").read_text())
    prior = json.loads((args.root / "reevaluation_r1/evidence.json").read_text())
    requests = [json.loads(Path(path).read_text()) for path in protocol["order"] if "d_alpha" in path]
    assert len(requests) == 2
    assert gpu_memory(args.device) < 16384
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(args.device).total_memory)
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = {
        "status": "running",
        "device": args.device,
        "batch": 1,
        "imgsz": 640,
        "precision": "FP32",
        "fuse": False,
        "cells": {},
        "script_sha256": sha256(Path(__file__)),
        "note": "Read-only native one-to-one assignment/loss diagnostic; no training or checkpoint mutation.",
    }
    dump(args.output / "evidence.json", evidence)
    try:
        for request in requests:
            name = request["params"]["name"]
            checkpoint = Path(prior["cells"][name]["checkpoint"])
            digest = sha256(checkpoint)
            assert digest == prior["cells"][name]["sha256"]
            evidence["cells"][name] = {"checkpoint_sha256": digest, "passes": {}}
            for rect in (False, True):
                tag = f"rect{int(rect)}"
                print(f"ASSIGNMENT_AUDIT {name} {tag}", flush=True)
                yolo = YOLO(str(checkpoint), task="detect")
                if isinstance(getattr(yolo.model, "args", None), dict):
                    yolo.model.args = get_cfg(overrides=yolo.model.args)
                output = args.output / name / tag
                validator = AssignmentValidator(
                    args={
                        "model": str(checkpoint),
                        "data": request["inputs"]["data"],
                        "split": "val",
                        "batch": 1,
                        "rect": rect,
                        "imgsz": 640,
                        "workers": 0,
                        "device": str(args.device),
                        "quantize": None,
                        "conf": 0.001,
                        "iou": 0.7,
                        "max_det": 300,
                        "plots": False,
                        "verbose": False,
                        "save_json": False,
                        "save_txt": False,
                        "project": str(output),
                        "name": "validator",
                        "exist_ok": False,
                    }
                )
                with patch.object(engine, "AutoBackend", partial(engine.AutoBackend, fuse=False)):
                    metrics = validator(model=yolo.model)
                assert (
                    len(validator.precision_rows) == len(validator.assignment_rows) == len(validator.error_rows) == 512
                )
                write_csv(output / "assignment.csv", validator.assignment_rows)
                write_csv(output / "errors_conf025.csv", validator.error_rows)
                result = {
                    "metrics": metrics,
                    "assignment": assignment_summary(validator.assignment_rows),
                    "errors_conf025": {
                        k: sum(row[k] for row in validator.error_rows) for k in validator.error_rows[0] if k != "image"
                    },
                    "checkpoint_sha256": digest,
                    "rect": rect,
                }
                dump(output / "result.json", result)
                evidence["cells"][name]["passes"][tag] = result
                dump(args.output / "evidence.json", evidence)
                del validator, yolo
                gc.collect()
                torch.cuda.empty_cache()
        evidence["status"] = "completed"
    except Exception:
        evidence["status"] = "failed"
        evidence["error"] = traceback.format_exc()
        raise
    finally:
        evidence["updated_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        dump(args.output / "evidence.json", evidence)


if __name__ == "__main__":
    main()
