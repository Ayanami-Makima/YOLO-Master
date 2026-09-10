"""Evaluation-only batch x rectangular-padding x fusion audit of fixed last checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
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


def variants():
    """Full factorial separates inference batch effects from rectangular batch padding."""
    return [{"batch": b, "rect": r, "fuse": f} for b, r, f in itertools.product((1, 8), (False, True), (False, True))]


class AuditValidator(DetectionValidator):
    """Record actual input geometry and fixed-confidence errors without altering native AP."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.rows = []
        self.shapes = Counter()
        self.bn_count = sum(isinstance(m, torch.nn.BatchNorm2d) for m in model.model.modules())

    def preprocess(self, batch):
        if self.batch_i % 8 == 0 and gpu_memory(self.device.index) > 22528:
            raise RuntimeError("Shared GPU total exceeded 22 GiB")
        result = super().preprocess(batch)
        assert result["img"].dtype == torch.float32
        self.shapes[str(tuple(result["img"].shape))] += 1
        return result

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            target = self._prepare_batch(si, batch)
            self.rows.append(
                {"image": str(target["im_file"]), "shape": list(batch["img"].shape[-2:]), **errors(pred, target)}
            )
        super().update_metrics(preds, batch)


def run(request, variant, output, device):
    checkpoint = Path(request["params"]["project"]) / request["params"]["name"] / "weights/last.pt"
    digest = sha256(checkpoint)
    model = YOLO(str(checkpoint)).model
    assert model.model[-1].end2end
    assert getattr(model.model[-1], "p2_o2o_gradient_alpha", 0) == request["p2_bridge"]["alpha"]
    bn_before = sum(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    args = {
        "model": str(checkpoint),
        "data": request["inputs"]["data"],
        "split": "val",
        "imgsz": 640,
        "batch": variant["batch"],
        "rect": variant["rect"],
        "device": str(device),
        "workers": 0,
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
    validator = AuditValidator(args=args)
    # Scoped backend override only: original validation engine and model files remain unchanged.
    with patch.object(engine, "AutoBackend", partial(engine.AutoBackend, fuse=variant["fuse"])):
        metrics = validator(model=model)
    assert len(validator.rows) == len({r["image"] for r in validator.rows}) == 512
    assert digest == sha256(checkpoint)
    assert validator.bn_count < bn_before if variant["fuse"] else validator.bn_count == bn_before
    if not variant["rect"]:
        assert all(r["shape"] == [640, 640] for r in validator.rows)
    totals = {k: sum(r[k] for r in validator.rows) for k in validator.rows[0] if k not in {"image", "shape"}}
    result = {
        "checkpoint_sha256": digest,
        "variant": variant,
        "metrics": metrics,
        "actual_shapes": dict(validator.shapes),
        "bn_before": bn_before,
        "bn_after": validator.bn_count,
        "actual_rect": bool(validator.dataloader.dataset.rect),
        "errors_conf025": totals,
        "image_set_sha256": hashlib.sha256("\n".join(sorted(r["image"] for r in validator.rows)).encode()).hexdigest(),
    }
    dump(output / "per_image.json", validator.rows)
    dump(output / "result.json", result)
    del validator, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((args.root / "protocol.json").read_text())
    prior = json.loads((args.root / "reevaluation_r1/evidence.json").read_text())
    for path, digest in protocol["source_hashes"].items():
        assert sha256(ROOT / path) == digest, path
    requests = []
    for path in protocol["order"]:
        assert sha256(Path(path)) == protocol["requests"][path]
        request = json.loads(Path(path).read_text())
        for source, digest in request["p2_bridge"]["input_hashes"].items():
            assert sha256(Path(source)) == digest, source
        name = request["params"]["name"]
        assert sha256(Path(prior["cells"][name]["checkpoint"])) == prior["cells"][name]["sha256"]
        requests.append(request)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "runs": len(requests) * len(variants()), "variants": variants()}))
        return
    assert gpu_memory(args.device) < 12288, "Insufficient headroom for 8 GiB allocator cap"
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / torch.cuda.get_device_properties(args.device).total_memory)
    os.environ.update(A1_E2E_O2O_TAL_TOPK="7", A1_E2E_O2O_TAL_TOPK2="1", A1_E2E_O2O_CLS_GAIN="1.0")
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "precision": "FP32",
        "variants": variants(),
        "results": {},
        "protocol_sha256": sha256(args.root / "protocol.json"),
        "script_sha256": sha256(Path(__file__)),
        "note": "Evaluation only. Full factorial. Training reference uses b8/rect/unfused. No latency claim.",
    }
    dump(args.output / "evidence.json", evidence)
    try:
        for variant in variants():
            tag = f"b{variant['batch']}_rect{int(variant['rect'])}_fuse{int(variant['fuse'])}"
            evidence["results"][tag] = {}
            for request in requests:
                name = request["params"]["name"]
                print(f"AUDIT {tag} {name}", flush=True)
                evidence["results"][tag][name] = run(request, variant, args.output / tag / name, args.device)
                dump(args.output / "evidence.json", evidence)
        assert len({v["image_set_sha256"] for group in evidence["results"].values() for v in group.values()}) == 1
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
