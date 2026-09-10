"""Controlled inference-only test for valid-region masked Router pooling."""

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

import torch
import torch.nn.functional as F

from scripts.a1.audit_p2_padding_routes import RouteValidator, pair_summary
from scripts.a1.evaluate_p2_gradient_bridge import dump, gpu_memory
from scripts.a1.run_p2_e2e_precision_diagnostics_r1 import sha256
from ultralytics import YOLO
from ultralytics.engine import validator as engine
from ultralytics.nn.modules.moe.routers import EfficientSpatialRouter


def masked_forward(router, x, top_k=None):
    """Copy native eval route math, replacing spatial mean with valid-region mean."""
    _, _, h, w = x.shape
    x_in = (
        F.avg_pool2d(x, kernel_size=router.pool_scale, stride=router.pool_scale)
        if h > router.pool_scale and w > router.pool_scale
        else x
    )
    out = router.router(x_in)
    mask = getattr(router, "_a1_valid_mask", None)
    if mask is None or mask.shape[-2:] != out.shape[-2:]:
        raise RuntimeError("masked Router received no valid-region mask")
    logits = (out.float() * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp_min(1.0)
    return router._process_logits(logits, router.noise_std, router.training, top_k=top_k)


class MaskedRouteValidator(RouteValidator):
    """Install only an inference-time masked pooling method on each Router."""

    def __init__(self, *args, masked=False, **kwargs):
        self.masked = masked
        super().__init__(*args, **kwargs)

    def init_metrics(self, model):
        super().init_metrics(model)
        self.mask_handles = []
        if not self.masked:
            return
        for name, router in model.model.named_modules():
            if isinstance(router, EfficientSpatialRouter):
                router.forward = partial(masked_forward, router)
                self.mask_handles.append(router.register_forward_pre_hook(self.set_mask))

    def set_mask(self, router, inputs):
        x = inputs[0]
        _, _, h, w = x.shape
        image_h, image_w = self.geometry["image_shape"]
        top, left = self.geometry["offset_xy"][1], self.geometry["offset_xy"][0]
        content_h, content_w = self.geometry["content_shape"]
        pooled_h = max(h // router.pool_scale, 1) if h > router.pool_scale and w > router.pool_scale else h
        pooled_w = max(w // router.pool_scale, 1) if h > router.pool_scale and w > router.pool_scale else w
        fy0, fy1 = top / image_h * pooled_h, (top + content_h) / image_h * pooled_h
        fx0, fx1 = left / image_w * pooled_w, (left + content_w) / image_w * pooled_w
        y0, y1 = max(0, math.floor(fy0)), min(pooled_h, math.ceil(fy1))
        x0, x1 = max(0, math.floor(fx0)), min(pooled_w, math.ceil(fx1))
        mask = torch.zeros((x.shape[0], 1, pooled_h, pooled_w), device=x.device, dtype=torch.float32)
        y = torch.arange(y0, y1, device=x.device, dtype=torch.float32)
        xcoord = torch.arange(x0, x1, device=x.device, dtype=torch.float32)
        row_weight = (
            torch.minimum(y + 1, torch.tensor(fy1, device=x.device))
            - torch.maximum(y, torch.tensor(fy0, device=x.device))
        ).clamp(0, 1)
        col_weight = (
            torch.minimum(xcoord + 1, torch.tensor(fx1, device=x.device))
            - torch.maximum(xcoord, torch.tensor(fx0, device=x.device))
        ).clamp(0, 1)
        mask[:, :, y0:y1, x0:x1] = row_weight[:, None] * col_weight[None, :]
        router._a1_valid_mask = mask

    def preprocess(self, batch):
        result = super().preprocess(batch)
        self.geometry["image_shape"] = list(result["img"].shape[-2:])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--cell", default="d_alpha0p1_seed260829")
    args = parser.parse_args()
    protocol = json.loads((args.root / "protocol.json").read_text())
    prior = json.loads((args.root / "reevaluation_r1/evidence.json").read_text())
    request = next(json.loads(Path(p).read_text()) for p in protocol["order"] if args.cell in p)
    checkpoint = Path(prior["cells"][args.cell]["checkpoint"])
    assert sha256(checkpoint) == prior["cells"][args.cell]["sha256"]
    assert gpu_memory(args.device) < 16384
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(args.device).total_memory)
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = {
        "status": "running",
        "cell": args.cell,
        "device": args.device,
        "batch": 1,
        "imgsz": 640,
        "precision": "FP32",
        "checkpoint_sha256": sha256(checkpoint),
        "passes": {},
        "note": "Inference-only masked valid-region pooling; no weights or training source changed.",
    }
    dump(args.output / "evidence.json", evidence)
    try:
        pass_rows = {}
        for rect, masked in ((False, False), (True, False), (True, True)):
            tag = f"rect{int(rect)}_masked{int(masked)}"
            print(f"MASKED_ROUTE_AUDIT {tag}", flush=True)
            model = YOLO(str(checkpoint)).model
            output = args.output / tag
            validator = MaskedRouteValidator(
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
                },
                masked=masked,
            )
            with patch.object(engine, "AutoBackend", partial(engine.AutoBackend, fuse=False)):
                metrics = validator(model=model)
            for handle in validator.handles + validator.mask_handles:
                handle.remove()
            assert len(validator.rows) == 512
            pass_rows[(rect, masked)] = validator.rows
            dump(output / "per_image_routes.json", validator.rows)
            evidence["passes"][tag] = {"metrics": metrics, "routers": validator.route_names}
            dump(args.output / "evidence.json", evidence)
            del validator, model
            gc.collect()
            torch.cuda.empty_cache()
        baseline = pass_rows[(False, False)]
        rect = pass_rows[(True, False)]
        masked = pass_rows[(True, True)]
        evidence["comparisons"] = {
            "rect_minus_square": pair_summary(baseline, rect),
            "masked_rect_minus_square": pair_summary(baseline, masked),
            "masked_rect_minus_rect": pair_summary(rect, masked),
        }
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
