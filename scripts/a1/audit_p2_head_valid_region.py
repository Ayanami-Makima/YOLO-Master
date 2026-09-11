"""Audit decoded detection-head candidates inside/outside the valid letterbox region."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import traceback
from functools import partial
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.a1.evaluate_p2_gradient_bridge import dump, errors, gpu_memory
from scripts.a1.run_p2_e2e_precision_diagnostics_r1 import sha256
from ultralytics import YOLO
from ultralytics.engine import validator as engine
from ultralytics.models.yolo.detect.val import DetectionValidator


class HeadRegionValidator(DetectionValidator):
    """Capture raw decoded candidate geometry before native postprocessing."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.rows = []
        self.geometry = None

    def preprocess(self, batch):
        image = batch["img"][0]
        oh, ow = batch["ori_shape"][0]
        (rh, rw), (left, top) = batch["ratio_pad"][0]
        h, w = round(oh * rh), round(ow * rw)
        self.geometry = {
            "shape": list(image.shape[-2:]),
            "content_shape": [h, w],
            "offset_xy": [int(left), int(top)],
            "content_sha256": hashlib.sha256(
                image[:, int(top) : int(top) + h, int(left) : int(left) + w].contiguous().numpy().tobytes()
            ).hexdigest(),
        }
        return super().preprocess(batch)

    def postprocess(self, preds):
        raw = preds[0] if isinstance(preds, (tuple, list)) and torch.is_tensor(preds[0]) else preds
        if not torch.is_tensor(raw):
            raise TypeError(f"unexpected prediction type: {type(raw).__name__}")
        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[-1] < 6:
            raise ValueError(f"unexpected prediction shape: {tuple(raw.shape)}")
        boxes = raw[0, :, :4].float()
        if raw.shape[-1] == 6:
            scores = raw[0, :, 4].float()
        else:
            scores = raw[0, :, 4:].float().amax(dim=-1)
        left, top = self.geometry["offset_xy"]
        h, w = self.geometry["content_shape"]
        x0, y0, x1, y1 = left, top, left + w, top + h
        cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
        inside = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
        row = {"geometry": self.geometry, "raw_candidates": len(boxes)}
        for threshold in (0.001, 0.01, 0.25):
            selected = scores >= threshold
            count = int(selected.sum())
            outside = int((selected & ~inside).sum())
            row[f"conf_{threshold:g}"] = {
                "count": count,
                "outside_valid_center": outside,
                "outside_fraction": outside / count if count else 0.0,
                "score_mean": float(scores[selected].mean()) if count else 0.0,
            }
        self.raw_row = row
        return super().postprocess(preds)

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            target = self._prepare_batch(si, batch)
            row = dict(self.raw_row)
            row["image"] = str(target["im_file"])
            row["errors"] = errors(pred, target)
            self.rows.append(row)
        super().update_metrics(preds, batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    protocol = json.loads((args.root / "protocol.json").read_text())
    prior = json.loads((args.root / "reevaluation_r1/evidence.json").read_text())
    requests = [json.loads(Path(path).read_text()) for path in protocol["order"] if "d_alpha" in path]
    if len(requests) != 2:
        raise ValueError("expected D alpha=0 and alpha=0.1 requests")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    if gpu_memory(args.device) >= 16384:
        raise RuntimeError("GPU memory guard exceeded")
    args.output.mkdir(parents=True)
    evidence = {
        "status": "running",
        "device": args.device,
        "batch": 1,
        "imgsz": 640,
        "precision": "FP32",
        "cells": {},
        "script_sha256": sha256(Path(__file__)),
        "note": "Read-only decoded-head valid-region diagnostic; no training or checkpoint mutation.",
    }
    dump(args.output / "evidence.json", evidence)
    try:
        for request in requests:
            name = request["params"]["name"]
            checkpoint = Path(prior["cells"][name]["checkpoint"])
            digest = sha256(checkpoint)
            if digest != prior["cells"][name]["sha256"]:
                raise ValueError(f"checkpoint changed: {name}")
            evidence["cells"][name] = {"checkpoint_sha256": digest, "passes": {}}
            for rect in (False, True):
                tag = f"rect{int(rect)}"
                print(f"HEAD_REGION_AUDIT {name} {tag}", flush=True)
                model = YOLO(str(checkpoint)).model
                output = args.output / name / tag
                validator = HeadRegionValidator(
                    args={
                        "model": str(checkpoint),
                        "data": request["inputs"]["data"],
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
                        "project": str(output),
                        "name": "validator",
                        "exist_ok": False,
                    }
                )
                with patch.object(engine, "AutoBackend", partial(engine.AutoBackend, fuse=False)):
                    metrics = validator(model=model)
                if len(validator.rows) != 512:
                    raise ValueError(f"expected 512 rows, got {len(validator.rows)}")
                dump(output / "per_image_head_region.json", validator.rows)
                evidence["cells"][name]["passes"][tag] = {"metrics": metrics, "rows": validator.rows}
                dump(args.output / "evidence.json", evidence)
                del validator, model
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
