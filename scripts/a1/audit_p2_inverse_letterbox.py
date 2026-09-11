"""Compare inverse-letterbox box errors and multi-scale head features."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import traceback
from functools import partial
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.a1.evaluate_p2_gradient_bridge import dump, gpu_memory
from scripts.a1.run_p2_e2e_precision_diagnostics_r1 import sha256
from ultralytics import YOLO
from ultralytics.engine import validator as engine
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou


def head_summary(value):
    result = []
    for tensor in value:
        tensor = tensor.detach().float()
        flat = tensor.flatten(2)
        result.append(
            {
                "shape": list(tensor.shape),
                "mean": float(tensor.mean()),
                "std": float(tensor.std()),
                "norm_per_element": float(tensor.norm() / max(tensor.numel() ** 0.5, 1)),
                "channel_mean": tensor.mean(dim=(0, 2, 3)).cpu().tolist(),
                "channel_std": flat.std(dim=-1)[0].cpu().tolist(),
            }
        )
    return result


def max_error(a, b):
    """Return an explicit empty-safe maximum absolute error."""
    return float((a - b).abs().max()) if a.numel() else 0.0


def inverse_reference(boxes, target):
    """Independent float64 affine inverse, following scale_boxes' scalar-gain contract."""
    out = boxes.detach().double().clone()
    gain = float(target["ratio_pad"][0][0])
    left, top = target["ratio_pad"][1]
    out -= out.new_tensor([left, top, left, top])
    out /= gain
    height, width = target["ori_shape"]
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, height)
    return out


def unique_rows(rows):
    """Reject duplicate IDs instead of silently dropping a row."""
    result = {row["image"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate image IDs")
    return result


def match_summary(pred, target, validator):
    """Save every GT and match by confidence-greedy class-aware IoU >= 0.5."""
    mask = pred["conf"] >= 0.25
    selected = {**pred, "bboxes": pred["bboxes"][mask], "conf": pred["conf"][mask], "cls": pred["cls"][mask]}
    scaled = validator.scale_preds(selected, target)
    gt = ops.scale_boxes(target["imgsz"], target["bboxes"].clone(), target["ori_shape"], ratio_pad=target["ratio_pad"])
    pred_ref = inverse_reference(selected["bboxes"], target)
    gt_ref = inverse_reference(target["bboxes"], target)
    coordinate_error = max(max_error(scaled["bboxes"].double(), pred_ref), max_error(gt.double(), gt_ref))
    if coordinate_error > 0.001:
        raise ValueError(f"inverse affine contract mismatch: {coordinate_error}")
    # Stable GT row indices are checked against class and original-coordinate boxes during pairing.
    rows = [
        {"gt_id": i, "cls": int(c), "box": b.tolist(), "match": None} for i, (c, b) in enumerate(zip(target["cls"], gt))
    ]
    overlaps = box_iou(scaled["bboxes"], gt)
    used = set()
    for i in torch.argsort(scaled["conf"], descending=True, stable=True).tolist():
        candidates = [j for j, row in enumerate(rows) if row["cls"] == int(scaled["cls"][i]) and j not in used]
        if not candidates:
            continue
        j = max(candidates, key=lambda j: float(overlaps[i, j]))
        if float(overlaps[i, j]) < 0.5:
            continue
        used.add(j)
        pbox, gbox = scaled["bboxes"][i], gt[j]
        pwh, gwh = pbox[2:] - pbox[:2], gbox[2:] - gbox[:2]
        delta = ((pbox[:2] + pbox[2:]) - (gbox[:2] + gbox[2:])) / 2
        rows[j]["match"] = {
            "pred_box": pbox.tolist(),
            "confidence": float(scaled["conf"][i]),
            "iou": float(overlaps[i, j]),
            "center_dx_px": float(delta[0]),
            "center_dy_px": float(delta[1]),
            "center_error_norm": float(delta.norm()) / math.hypot(*target["ori_shape"]),
            "width_log_abs": float(torch.log((pwh[0] + 1) / (gwh[0] + 1)).abs()),
            "height_log_abs": float(torch.log((pwh[1] + 1) / (gwh[1] + 1)).abs()),
        }
    retained = validator.scale_preds(pred, target)
    return {
        "pred_conf025": int(mask.sum()),
        "gt": rows,
        "coordinate_max_error_px": coordinate_error,
        "retained_predictions": {
            "native_tp": validator._process_batch(pred, target)["tp"].tolist(),
            "boxes": retained["bboxes"].detach().cpu().tolist(),
            "scores": retained["conf"].detach().cpu().tolist(),
            "classes": retained["cls"].detach().cpu().tolist(),
        },
        "retention": {"conf_floor": 0.001, "max_det": 300, "stage": "native_postprocess_output"},
    }


class InverseLetterboxValidator(DetectionValidator):
    """Audit actual cached anchors after inference and retain same-GT records."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.rows = []
        self.head = model.model.model[23]
        self.handle = self.head.register_forward_hook(self.head_hook)
        self.previous_shape = None

    def preprocess(self, batch):
        if self.batch_i % 16 == 0 and gpu_memory(self.device.index) > 22528:
            raise RuntimeError("shared GPU exceeds 22 GiB")
        return super().preprocess(batch)

    def head_hook(self, module, inputs, output):
        features = inputs[0]
        expected, strides, scales = [], [], []
        ih, iw = features[0].shape[-2:]
        for feature, stride in zip(features, module.stride):
            h, w = feature.shape[-2:]
            value = float(stride)
            # Construct row-major coordinates independently, without make_anchors().
            flat = torch.arange(h * w, device=feature.device)
            expected.append(torch.stack((flat % w + 0.5, flat // w + 0.5)).to(feature.dtype))
            strides.append(torch.full((1, h * w), value, device=feature.device))
            scales.append({"shape": [h, w], "stride": value, "anchor_count": h * w})
            if h * value != ih * float(module.stride[0]) or w * value != iw * float(module.stride[0]):
                raise ValueError("multi-scale dimensions disagree with stride")
        expected = torch.cat(expected, 1)
        expected_stride = torch.cat(strides, 1)
        if module.anchors.shape != expected.shape or module.strides.shape != expected_stride.shape:
            raise ValueError("actual cached anchor/stride shape mismatch")
        anchor_error = max_error(module.anchors, expected)
        stride_error = max_error(module.strides, expected_stride)
        if anchor_error != 0 or stride_error != 0:
            raise ValueError("actual cached anchors differ from independent grid")
        raw = output[1]["one2one"]
        # Independent decode from DFL distances, then compare native decode using actual cache.
        distance = module.dfl(raw["boxes"])
        lt, rb = distance.chunk(2, 1)
        ref = torch.cat((expected.unsqueeze(0) - lt, expected.unsqueeze(0) + rb), 1) * expected_stride
        actual = module._get_decode_boxes(raw)
        decode_error = max_error(actual, ref)
        if decode_error > 0.0001:
            raise ValueError("decode mismatch")
        shape = [list(f.shape[-2:]) for f in features]
        self.grid = {
            "scales": scales,
            "anchor_max_error": anchor_error,
            "stride_max_error": stride_error,
            "decode_max_error": decode_error,
            "shape_transition": self.previous_shape is not None and self.previous_shape != shape,
        }
        self.previous_shape = shape

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            target = self._prepare_batch(si, batch)
            h0, w0 = target["ori_shape"]
            (rh, rw), (left, top) = target["ratio_pad"]
            for scale in self.grid["scales"]:
                h, w = scale["shape"]
                stride = scale["stride"]
                nx = sum(left <= (x + 0.5) * stride < left + round(w0 * rw) for x in range(w))
                ny = sum(top <= (y + 0.5) * stride < top + round(h0 * rh) for y in range(h))
                scale["valid_anchor_count"] = nx * ny
            self.rows.append(
                {
                    "image": str(target["im_file"]),
                    "grid": self.grid,
                    "geometry": {
                        "shape": list(batch["img"][si].shape[-2:]),
                        "ori_shape": list(target["ori_shape"]),
                        "ratio_pad": [list(target["ratio_pad"][0]), list(target["ratio_pad"][1])],
                    },
                    "boxes": match_summary(pred, target, self),
                }
            )
        super().update_metrics(preds, batch)


def compare(square, rect):
    """Pair by image and GT identity; average only common matched targets."""
    left_index, right_index = unique_rows(square), unique_rows(rect)
    if left_index.keys() != right_index.keys():
        raise ValueError("image sets differ")
    counts = {"both": 0, "square_only": 0, "rect_only": 0, "neither": 0}
    rows, deltas = [], []
    identity_error = 0.0
    for image, left in left_index.items():
        right = right_index[image]
        a, b = left["boxes"]["gt"], right["boxes"]["gt"]
        if len(a) != len(b):
            raise ValueError("GT count differs")
        for x, y in zip(a, b):
            err = max(abs(p - q) for p, q in zip(x["box"], y["box"]))
            identity_error = max(identity_error, err)
            if x["gt_id"] != y["gt_id"] or x["cls"] != y["cls"] or err > 0.001:
                raise ValueError(f"GT identity mismatch: {image}, {err}")
            mx, my = x["match"], y["match"]
            category = "both" if mx and my else "square_only" if mx else "rect_only" if my else "neither"
            counts[category] += 1
            diff = (
                {
                    k: my[k] - mx[k]
                    for k in ("iou", "confidence", "center_error_norm", "width_log_abs", "height_log_abs")
                }
                if mx and my
                else None
            )
            if diff is not None:
                deltas.append(diff)
            rows.append(
                {
                    "image": image,
                    "gt_id": x["gt_id"],
                    "cls": x["cls"],
                    "category": category,
                    "square": mx,
                    "rect": my,
                    "delta": diff,
                }
            )
    return {
        "images": len(left_index),
        "gt_total": sum(counts.values()),
        "gt_identity_max_error_px": identity_error,
        "match_conf": 0.25,
        "match_iou": 0.5,
        "counts": counts,
        "common_gt_mean_delta": {k: float(np.mean([d[k] for d in deltas])) for k in deltas[0]} if deltas else None,
        "common_gt_denominator": len(deltas),
        "grid_coordinate_checks": {
            tag: {
                "max_anchor_error": max(r["grid"]["anchor_max_error"] for r in data),
                "max_stride_error": max(r["grid"]["stride_max_error"] for r in data),
                "max_decode_error": max(r["grid"]["decode_max_error"] for r in data),
                "shape_transitions": sum(r["grid"]["shape_transition"] for r in data),
                "max_inverse_error_px": max(r["boxes"]["coordinate_max_error_px"] for r in data),
                "mean_valid_anchor_counts": [
                    float(np.mean([r["grid"]["scales"][i]["valid_anchor_count"] for r in data])) for i in range(3)
                ],
            }
            for tag, data in (("square", square), ("rect", rect))
        },
        "limitation": "Common-GT localization is conditional on matching in both passes; it is not a population causal effect. Affine check validates scale_boxes contract, not independent raw annotation provenance.",
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--data", type=Path, default=None, help="Override validation YAML")
    parser.add_argument("--expected-images", type=int, default=512)
    args = parser.parse_args()
    protocol = json.loads((args.root / "protocol.json").read_text())
    prior = json.loads((args.root / "reevaluation_r1/evidence.json").read_text())
    requests = [json.loads(Path(path).read_text()) for path in protocol["order"] if "d_alpha" in path]
    if len(requests) != 2 or args.output.exists():
        raise ValueError("expected two D requests and a fresh output directory")
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    if gpu_memory(args.device) >= 16384:
        raise RuntimeError("GPU memory guard exceeded")
    args.output.mkdir(parents=True)
    evidence = {
        "schema": "p2-same-gt-grid-audit/v2",
        "status": "running",
        "device": args.device,
        "batch": 1,
        "imgsz": 640,
        "precision": "FP32",
        "cells": {},
        "script_sha256": sha256(Path(__file__)),
        "note": "Read-only inverse-letterbox and multi-scale head diagnostic; no training or checkpoint mutation.",
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
            collected = {}
            for rect in (False, True):
                tag = f"rect{int(rect)}"
                print(f"INVERSE_LETTERBOX_AUDIT {name} {tag}", flush=True)
                model = YOLO(str(checkpoint)).model
                output = args.output / name / tag
                validator = InverseLetterboxValidator(
                    args={
                        "model": str(checkpoint),
                        "data": str(args.data or request["inputs"]["data"]),
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
                if len(validator.rows) != args.expected_images:
                    raise ValueError(f"expected {args.expected_images} rows, got {len(validator.rows)}")
                collected[rect] = validator.rows
                dump(output / "per_image_inverse_letterbox.json", validator.rows)
                evidence["cells"][name]["passes"][tag] = {"metrics": metrics}
                dump(args.output / "evidence.json", evidence)
                validator.handle.remove()
                del validator, model
                gc.collect()
                torch.cuda.empty_cache()
            summary, rows = compare(collected[False], collected[True])
            if sha256(checkpoint) != digest:
                raise ValueError("checkpoint changed during audit")
            evidence["cells"][name]["checkpoint_sha256_after"] = digest
            evidence["cells"][name]["summary_rect_minus_square"] = summary
            dump(args.output / name / "paired_inverse_letterbox_deltas.json", rows)
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
