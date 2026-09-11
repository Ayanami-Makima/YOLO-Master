"""Compare post-backbone and detect-head feature statistics under square/rect val padding."""

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

import numpy as np
import torch

from scripts.a1.evaluate_p2_gradient_bridge import dump, errors, gpu_memory
from scripts.a1.run_p2_e2e_precision_diagnostics_r1 import sha256
from ultralytics import YOLO
from ultralytics.engine import validator as engine
from ultralytics.models.yolo.detect.val import DetectionValidator


def tensor_summary(value):
    """Keep channel statistics and shape while avoiding full feature-map dumps."""
    if isinstance(value, (list, tuple)):
        return [tensor_summary(item) for item in value]
    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__}
    tensor = value.detach().float()
    if tensor.ndim < 2:
        return {"shape": list(tensor.shape), "mean": float(tensor.mean()), "std": float(tensor.std())}
    flat = tensor.flatten(2) if tensor.ndim >= 4 else tensor.flatten(1)
    channel_mean = (
        tensor.mean(dim=tuple(range(2, tensor.ndim)))[0].cpu().tolist()
        if tensor.ndim >= 4
        else tensor.mean(0).cpu().tolist()
    )
    channel_std = flat.std(dim=-1)[0].cpu().tolist() if tensor.ndim >= 4 else tensor.std(0).cpu().tolist()
    return {
        "shape": list(tensor.shape),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "norm_per_element": float(tensor.norm() / max(tensor.numel() ** 0.5, 1)),
        "channel_mean": channel_mean,
        "channel_std": channel_std,
    }


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 1.0


class FeatureValidator(DetectionValidator):
    """Capture only read-only layer summaries plus fixed-confidence errors."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.rows = []
        self.current = {}
        self.handles = []
        for index in (4, 6, 8):
            self.handles.append(model.model.model[index].register_forward_hook(self.feature_hook(f"layer{index}")))
        self.handles.append(model.model.model[23].register_forward_pre_hook(self.head_hook))

    def feature_hook(self, name):
        def hook(_module, _inputs, output):
            self.current[name] = tensor_summary(output)

        return hook

    def head_hook(self, _module, inputs):
        self.current["head_input"] = tensor_summary(inputs[0])

    def preprocess(self, batch):
        if self.batch_i % 8 == 0 and gpu_memory(self.device.index) > 22528:
            raise RuntimeError("Shared GPU total exceeded 22 GiB")
        image = batch["img"][0]
        oh, ow = batch["ori_shape"][0]
        (rh, rw), (left, top) = batch["ratio_pad"][0]
        h, w = round(oh * rh), round(ow * rw)
        self.geometry = {
            "shape": list(image.shape[-2:]),
            "content_shape": [h, w],
            "offset_xy": [int(left), int(top)],
            "padding_fraction": 1 - h * w / (image.shape[-2] * image.shape[-1]),
            "content_sha256": hashlib.sha256(
                image[:, int(top) : int(top) + h, int(left) : int(left) + w].contiguous().numpy().tobytes()
            ).hexdigest(),
        }
        self.current = {}
        return super().preprocess(batch)

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            target = self._prepare_batch(si, batch)
            self.rows.append(
                {
                    "image": str(target["im_file"]),
                    "geometry": self.geometry,
                    "errors": errors(pred, target),
                    "features": self.current,
                }
            )
        super().update_metrics(preds, batch)


def compare(square, rect):
    right = {row["image"]: row for row in rect}
    assert len(right) == len(square) == 512
    per_image = []
    for left in square:
        other = right[left["image"]]
        row = {
            "image": left["image"],
            "same_content": left["geometry"]["content_sha256"] == other["geometry"]["content_sha256"],
            "square": left["geometry"],
            "rect": other["geometry"],
            "tp_delta": other["errors"]["tp"] - left["errors"]["tp"],
            "fp_delta": other["errors"]["fp"] - left["errors"]["fp"],
        }
        for name in ("layer4", "layer6", "layer8", "head_input"):
            a, b = left["features"][name], other["features"][name]
            if isinstance(a, list):
                row[name] = [
                    {
                        "cosine_channel_mean": cosine(x["channel_mean"], y["channel_mean"]),
                        "mean_delta": y["mean"] - x["mean"],
                        "std_delta": y["std"] - x["std"],
                    }
                    for x, y in zip(a, b)
                ]
            else:
                row[name] = {
                    "cosine_channel_mean": cosine(a["channel_mean"], b["channel_mean"]),
                    "mean_delta": b["mean"] - a["mean"],
                    "std_delta": b["std"] - a["std"],
                }
        per_image.append(row)
    summary = {
        "images": len(per_image),
        "same_content_images": sum(x["same_content"] for x in per_image),
        "tp_delta_rect_minus_square": sum(x["tp_delta"] for x in per_image),
        "fp_delta_rect_minus_square": sum(x["fp_delta"] for x in per_image),
        "features": {},
    }
    for name in ("layer4", "layer6", "layer8", "head_input"):
        values = [x[name] for x in per_image]
        if isinstance(values[0], list):
            summary["features"][name] = [
                {"mean_cosine": float(np.mean([v[i]["cosine_channel_mean"] for v in values]))}
                for i in range(len(values[0]))
            ]
        else:
            summary["features"][name] = {
                "mean_cosine": float(np.mean([v["cosine_channel_mean"] for v in values])),
                "p10_cosine": float(np.percentile([v["cosine_channel_mean"] for v in values], 10)),
                "mean_delta": float(np.mean([v["mean_delta"] for v in values])),
                "mean_abs_delta": float(np.mean(np.abs([v["mean_delta"] for v in values]))),
            }
    return summary, per_image


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
        "cells": {},
        "script_sha256": sha256(Path(__file__)),
        "note": "Read-only feature/matching mechanism diagnostic; no training or checkpoint mutation.",
    }
    dump(args.output / "evidence.json", evidence)
    try:
        for request in requests:
            name = request["params"]["name"]
            checkpoint = Path(prior["cells"][name]["checkpoint"])
            digest = sha256(checkpoint)
            assert digest == prior["cells"][name]["sha256"]
            passes = {}
            evidence["cells"][name] = {"checkpoint_sha256": digest, "passes": {}}
            for rect in (False, True):
                tag = f"rect{int(rect)}"
                print(f"FEATURE_AUDIT {name} {tag}", flush=True)
                model = YOLO(str(checkpoint)).model
                output = args.output / name / tag
                validator = FeatureValidator(
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
                for handle in validator.handles:
                    handle.remove()
                assert len(validator.rows) == 512
                passes[rect] = validator.rows
                dump(output / "per_image_features.json", validator.rows)
                evidence["cells"][name]["passes"][tag] = {"metrics": metrics}
                dump(args.output / "evidence.json", evidence)
                del validator, model
                gc.collect()
                torch.cuda.empty_cache()
            summary, rows = compare(passes[False], passes[True])
            evidence["cells"][name]["summary"] = summary
            dump(args.output / name / "paired_feature_deltas.json", rows)
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
