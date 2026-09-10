"""Audit detection-head anchor-grid coverage under square and rectangular padding."""

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

from scripts.a1.evaluate_p2_gradient_bridge import dump, gpu_memory
from scripts.a1.run_p2_e2e_precision_diagnostics_r1 import sha256
from ultralytics import YOLO
from ultralytics.engine import validator as engine
from ultralytics.models.yolo.detect.val import DetectionValidator


class AnchorGridValidator(DetectionValidator):
    """Record anchor-grid shapes and valid-region coverage per image."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.rows = []
        self.geometry = None
        self.head = model.model.model[23]
        self.handle = self.head.register_forward_pre_hook(self.head_hook)

    def preprocess(self, batch):
        image = batch["img"][0]
        oh, ow = batch["ori_shape"][0]
        (rh, rw), (left, top) = batch["ratio_pad"][0]
        h, w = round(oh * rh), round(ow * rw)
        self.geometry = {
            "shape": list(image.shape[-2:]),
            "content_shape": [h, w],
            "offset_xy": [int(left), int(top)],
        }
        return super().preprocess(batch)

    def head_hook(self, _module, inputs):
        stats = []
        left, top = self.geometry["offset_xy"]
        h, w = self.geometry["content_shape"]
        for feature, stride in zip(inputs[0], self.head.stride):
            height, width = feature.shape[-2:]
            stride = float(stride)
            x = (torch.arange(width, device=feature.device) + 0.5) * stride
            y = (torch.arange(height, device=feature.device) + 0.5) * stride
            valid_x = (x >= left) & (x <= left + w)
            valid_y = (y >= top) & (y <= top + h)
            stats.append(
                {
                    "feature_shape": [int(height), int(width)],
                    "stride": stride,
                    "anchor_count": int(height * width),
                    "valid_x_fraction": float(valid_x.float().mean()),
                    "valid_y_fraction": float(valid_y.float().mean()),
                    "valid_anchor_fraction": float(valid_x.float().mean() * valid_y.float().mean()),
                }
            )
        self.grid_stats = stats

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            target = self._prepare_batch(si, batch)
            self.rows.append({"image": str(target["im_file"]), "geometry": self.geometry, "grid": self.grid_stats})
        super().update_metrics(preds, batch)


def run_pass(checkpoint, data, output, device, rect):
    model = YOLO(str(checkpoint)).model
    validator = AnchorGridValidator(
        args={
            "model": str(checkpoint),
            "data": data,
            "batch": 1,
            "rect": rect,
            "imgsz": 640,
            "workers": 0,
            "device": str(device),
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
    rows = validator.rows
    validator.handle.remove()
    del validator, model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, rows


def compare(square, rect):
    right_by_image = {row["image"]: row for row in rect}
    left_ids = {row["image"] for row in square}
    if len(left_ids) != len(square) or len(right_by_image) != len(rect) or left_ids != right_by_image.keys():
        raise ValueError("duplicate or mismatched image IDs")
    result = []
    for left in square:
        right = right_by_image[left["image"]]
        result.append(
            {
                "image": left["image"],
                "same_image": left["image"] == right["image"],
                "grid": [
                    {
                        "shape_square": a["feature_shape"],
                        "shape_rect": b["feature_shape"],
                        "valid_anchor_fraction_square": a["valid_anchor_fraction"],
                        "valid_anchor_fraction_rect": b["valid_anchor_fraction"],
                    }
                    for a, b in zip(left["grid"], right["grid"])
                ],
            }
        )
    summary = {
        "images": len(result),
        "same_images": sum(row["same_image"] for row in result),
        "scales": [
            {
                "mean_square_valid_fraction": float(
                    np.mean([row["grid"][i]["valid_anchor_fraction_square"] for row in result])
                ),
                "mean_rect_valid_fraction": float(
                    np.mean([row["grid"][i]["valid_anchor_fraction_rect"] for row in result])
                ),
                "mean_rect_minus_square": float(
                    np.mean(
                        [
                            row["grid"][i]["valid_anchor_fraction_rect"]
                            - row["grid"][i]["valid_anchor_fraction_square"]
                            for row in result
                        ]
                    )
                ),
            }
            for i in range(3)
        ],
    }
    return summary, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads((args.root / "protocol.json").read_text())
    prior = json.loads((args.root / "reevaluation_r1/evidence.json").read_text())
    requests = [json.loads(Path(path).read_text()) for path in protocol["order"] if "d_alpha" in path]
    if len(requests) != 2:
        raise ValueError("expected D alpha=0 and alpha=0.1 requests")
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
        "note": "Read-only anchor-grid/stride valid-region diagnostic; no checkpoint mutation.",
    }
    dump(args.output / "evidence.json", evidence)
    try:
        for request in requests:
            name = request["params"]["name"]
            checkpoint = Path(prior["cells"][name]["checkpoint"])
            digest = sha256(checkpoint)
            if digest != prior["cells"][name]["sha256"]:
                raise ValueError(f"checkpoint changed: {name}")
            data = request["inputs"]["data"]
            print(f"ANCHOR_GRID_AUDIT {name} square", flush=True)
            square_metrics, square = run_pass(checkpoint, data, args.output / name / "square", args.device, False)
            print(f"ANCHOR_GRID_AUDIT {name} rect", flush=True)
            rect_metrics, rect = run_pass(checkpoint, data, args.output / name / "rect", args.device, True)
            summary, rows = compare(square, rect)
            evidence["cells"][name] = {
                "checkpoint_sha256": digest,
                "metrics": {
                    "square": square_metrics,
                    "rect": rect_metrics,
                },
                "summary": summary,
            }
            dump(args.output / name / "paired_anchor_grid.json", rows)
            dump(args.output / "evidence.json", evidence)
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
