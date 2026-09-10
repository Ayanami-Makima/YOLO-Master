"""Test lowest-resolution head feature scale normalization as a read-only counterfactual."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.a1.evaluate_p2_gradient_bridge import dump, gpu_memory
from scripts.a1.run_p2_e2e_precision_diagnostics_r1 import sha256
from ultralytics import YOLO
from ultralytics.engine import validator as engine
from ultralytics.models.yolo.detect.val import DetectionValidator


class ScaleNormValidator(DetectionValidator):
    """Collect square statistics or normalize one head scale on rectangle inputs."""

    def __init__(self, *args, feature_mode="raw", references=None, **kwargs):
        self.feature_mode = feature_mode
        self.references = references if references is not None else {}
        self.current_image = None
        self.feature_rows = {}
        super().__init__(*args, **kwargs)

    def init_metrics(self, model):
        super().init_metrics(model)
        self.handle = model.model.model[23].register_forward_pre_hook(self.head_hook)

    def preprocess(self, batch):
        self.current_image = str(batch["im_file"][0])
        return super().preprocess(batch)

    @staticmethod
    def channel_stats(tensor):
        value = tensor.detach().float()
        return value.mean(dim=(0, 2, 3)), value.std(dim=(0, 2, 3))

    def head_hook(self, _module, inputs):
        features = list(inputs[0])
        stats = [self.channel_stats(feature) for feature in features]
        self.feature_rows[self.current_image] = [
            {"mean": mean.cpu().tolist(), "std": std.cpu().tolist(), "shape": list(feature.shape)}
            for feature, (mean, std) in zip(features, stats)
        ]
        if self.feature_mode != "normalize_rect":
            return None
        reference = self.references.get(self.current_image)
        if reference is None:
            raise KeyError(f"missing square reference for {self.current_image}")
        normalized = []
        for index, (feature, (mean, std)) in enumerate(zip(features, stats)):
            if index != 2:
                normalized.append(feature)
                continue
            target_mean = torch.as_tensor(reference[index]["mean"], device=feature.device, dtype=feature.dtype)
            target_std = torch.as_tensor(reference[index]["std"], device=feature.device, dtype=feature.dtype)
            source_mean = mean.to(device=feature.device, dtype=feature.dtype)
            source_std = std.to(device=feature.device, dtype=feature.dtype).clamp_min(1e-6)
            normalized.append(
                (feature - source_mean.view(1, -1, 1, 1))
                / source_std.view(1, -1, 1, 1)
                * target_std.clamp_min(1e-6).view(1, -1, 1, 1)
                + target_mean.view(1, -1, 1, 1)
            )
        return (normalized,)


def run_pass(checkpoint, data, output, device, rect, mode, references=None):
    model = YOLO(str(checkpoint)).model
    validator = ScaleNormValidator(
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
        },
        feature_mode=mode,
        references=references,
    )
    with patch.object(engine, "AutoBackend", partial(engine.AutoBackend, fuse=False)):
        metrics = validator(model=model)
    if len(validator.feature_rows) != 512:
        raise ValueError(f"expected 512 feature rows, got {len(validator.feature_rows)}")
    features = validator.feature_rows
    validator.handle.remove()
    del validator, model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, features


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
        "normalized_scale": 2,
        "cells": {},
        "script_sha256": sha256(Path(__file__)),
        "note": "Read-only lowest-resolution head feature scale normalization counterfactual; no checkpoint mutation.",
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
            cell = {"checkpoint_sha256": digest, "passes": {}}
            square_dir = args.output / name / "square_reference"
            print(f"SCALE_NORM_AUDIT {name} square_reference", flush=True)
            square_metrics, references = run_pass(checkpoint, data, square_dir, args.device, False, "collect")
            cell["passes"]["square_reference"] = {"metrics": square_metrics}
            dump(args.output / name / "square_reference" / "feature_stats.json", references)
            for tag, mode in (("rect_raw", "raw"), ("rect_norm_lowres", "normalize_rect")):
                print(f"SCALE_NORM_AUDIT {name} {tag}", flush=True)
                metrics, features = run_pass(
                    checkpoint,
                    data,
                    args.output / name / tag,
                    args.device,
                    True,
                    mode,
                    references,
                )
                cell["passes"][tag] = {"metrics": metrics}
                dump(args.output / name / tag / "feature_stats.json", features)
                dump(args.output / "evidence.json", {**evidence, "cells": {**evidence["cells"], name: cell}})
            evidence["cells"][name] = cell
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
