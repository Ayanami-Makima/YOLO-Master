"""Observe paired-image padding sensitivity of native MoE routes; no intervention or training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.a1.audit_p2_validation_protocol import AuditValidator
from scripts.a1.evaluate_p2_gradient_bridge import dump, gpu_memory
from scripts.a1.run_p2_e2e_precision_diagnostics_r1 import sha256
from ultralytics import YOLO
from ultralytics.engine import validator as engine
from ultralytics.nn.modules.moe.routers import EfficientSpatialRouter


def route_delta(a, b):
    """Compare expert sets separately from weight changes in expert-index coordinates."""
    wa = np.zeros(len(a["logits"]))
    wb = np.zeros(len(b["logits"]))
    wa[a["indices"]] = a["weights"]
    wb[b["indices"]] = b["weights"]
    return {
        "set_changed": set(a["indices"]) != set(b["indices"]),
        "weight_l1": float(np.abs(wa - wb).sum()),
        "logit_l1_mean": float(np.abs(np.array(a["logits"]) - np.array(b["logits"])).mean()),
    }


class RouteValidator(AuditValidator):
    """Read-only hooks record native routing outputs and pre-reduction logit maps."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.handles = []
        self.route_names = []
        for name, router in model.model.named_modules():
            if isinstance(router, EfficientSpatialRouter):
                self.route_names.append(name)
                self.handles.append(router.router.register_forward_hook(partial(self.logits_hook, name)))
                self.handles.append(router.register_forward_hook(partial(self.route_hook, name)))
        assert self.route_names, "No supported routers found"

    def logits_hook(self, name, module, inputs, output):
        logits = output.float().mean((2, 3))[0].detach().cpu()
        self.current_routes[name] = {"logits": logits.tolist(), "logit_map_shape": list(output.shape)}

    def route_hook(self, name, module, inputs, output):
        weights, indices, _ = output
        indices = indices.detach().cpu().reshape(-1)
        weights = weights.detach().cpu().reshape(-1)
        assert len(indices) == 2 and len(set(indices.tolist())) == 2
        assert not module.training
        self.current_routes[name].update(
            indices=indices.tolist(),
            weights=weights.tolist(),
            feature_shape=list(inputs[0].shape),
            input_channel_mean=inputs[0].float().mean((0, 2, 3)).detach().cpu().tolist(),
        )

    def preprocess(self, batch):
        self.current_routes = {}
        image = batch["img"][0]
        oh, ow = batch["ori_shape"][0]
        (rh, rw), (left, top) = batch["ratio_pad"][0]
        h, w = round(oh * rh), round(ow * rw)
        left, top = int(left), int(top)
        content = image[:, top : top + h, left : left + w].contiguous()
        assert tuple(content.shape[-2:]) == (h, w)
        self.geometry = {
            "content_shape": [h, w],
            "offset_xy": [left, top],
            "ratio_pad": batch["ratio_pad"][0],
            "content_sha256": hashlib.sha256(content.numpy().tobytes()).hexdigest(),
            "padding_fraction": 1 - h * w / (image.shape[-2] * image.shape[-1]),
        }
        return super().preprocess(batch)

    def update_metrics(self, preds, batch):
        assert len(preds) == 1 and set(self.current_routes) == set(self.route_names)
        super().update_metrics(preds, batch)
        self.rows[-1].update(geometry=self.geometry, routes=self.current_routes)


def pair_summary(square, rect):
    """Join by image path, not rectangle-sorted loader order; report association, not causality."""
    right = {r["image"]: r for r in rect}
    assert len(right) == len(square) == 512
    pairs = []
    for a in square:
        b = right[a["image"]]
        assert a["gt"] == b["gt"]
        pairs.append(
            {
                "image": a["image"],
                "same_content": a["geometry"]["content_sha256"] == b["geometry"]["content_sha256"],
                "square_padding": a["geometry"]["padding_fraction"],
                "rect_padding": b["geometry"]["padding_fraction"],
                "tp_delta_rect_minus_square": b["tp"] - a["tp"],
                "fp_delta_rect_minus_square": b["fp"] - a["fp"],
                "routes": {name: route_delta(a["routes"][name], b["routes"][name]) for name in a["routes"]},
            }
        )
    layers = {}
    for name in pairs[0]["routes"]:
        layers[name] = {
            "images": 512,
            "set_changed_count": sum(p["routes"][name]["set_changed"] for p in pairs),
            "weight_l1_mean": float(np.mean([p["routes"][name]["weight_l1"] for p in pairs])),
            "logit_l1_mean": float(np.mean([p["routes"][name]["logit_l1_mean"] for p in pairs])),
        }
    grouped = {}
    for changed in (False, True):
        group = [p for p in pairs if any(v["set_changed"] for v in p["routes"].values()) == changed]
        grouped[str(changed)] = {
            "images": len(group),
            "tp_delta": sum(p["tp_delta_rect_minus_square"] for p in group),
            "fp_delta": sum(p["fp_delta_rect_minus_square"] for p in group),
        }
    return {
        "same_content_images": sum(p["same_content"] for p in pairs),
        "layers": layers,
        "any_route_set_changed_groups": grouped,
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    protocol = json.loads((args.root / "protocol.json").read_text())
    prior = json.loads((args.root / "reevaluation_r1/evidence.json").read_text())
    for path, digest in protocol["source_hashes"].items():
        assert sha256(ROOT / path) == digest, path
    requests = [json.loads(Path(p).read_text()) for p in protocol["order"]]
    requests = [r for r in requests if r["params"]["name"].startswith("d_")]
    assert len(requests) == 2
    for r in requests:
        for path, digest in r["p2_bridge"]["input_hashes"].items():
            assert sha256(Path(path)) == digest, path
    assert gpu_memory(args.device) < 16384
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(args.device).total_memory)
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "batch": 1,
        "precision": "FP32",
        "fuse": False,
        "cells": {},
        "script_sha256": sha256(Path(__file__)),
        "note": "Observational padding+geometry probe, not a causal route intervention or proposed fix.",
    }
    dump(args.output / "evidence.json", evidence)
    try:
        for r in requests:
            name = r["params"]["name"]
            checkpoint = Path(prior["cells"][name]["checkpoint"])
            digest = sha256(checkpoint)
            assert digest == prior["cells"][name]["sha256"]
            passes = {}
            evidence["cells"][name] = {"checkpoint_sha256": digest, "passes": {}}
            for rect in (False, True):
                print(f"ROUTE_AUDIT {name} rect={rect}", flush=True)
                model = YOLO(str(checkpoint)).model
                output = args.output / name / str(rect)
                validator = RouteValidator(
                    args={
                        "model": str(checkpoint),
                        "data": r["inputs"]["data"],
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
                dump(output / "per_image_routes.json", validator.rows)
                evidence["cells"][name]["passes"][str(rect)] = {"metrics": metrics, "routers": validator.route_names}
                dump(args.output / "evidence.json", evidence)
                del validator, model
                gc.collect()
                torch.cuda.empty_cache()
            summary = pair_summary(passes[False], passes[True])
            dump(args.output / name / "pairs.json", summary.pop("pairs"))
            evidence["cells"][name]["summary"] = summary
            assert sha256(checkpoint) == digest
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
