#!/usr/bin/env python3
"""Single-batch audit for the P2-E r2 one-to-one classification gain."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def audit(checkpoint: Path, gain: float) -> dict:
    os.environ["A1_E2E_O2O_CLS_GAIN"] = str(gain)
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    import ultralytics

    torch.manual_seed(260829)
    yolo = YOLO(str(checkpoint), task="detect")
    model = yolo.model
    if isinstance(getattr(model, "args", None), dict):
        model.args = get_cfg(overrides=model.args)
    model.train()
    criterion = model.init_criterion()
    native = getattr(criterion, "native_criterion", criterion)
    x = torch.randn(1, 3, 256, 256)
    batch = {
        "batch_idx": torch.zeros(1, dtype=torch.long),
        "cls": torch.zeros(1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    preds = model(x)
    loss, items = criterion(preds, batch)
    scalar = loss.sum() if isinstance(loss, torch.Tensor) else loss
    scalar.backward()
    one2one_grad = 0.0
    one2many_grad = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().abs().sum())
        if "one2one_cv3" in name:
            one2one_grad += value
        elif ".cv3." in name:
            one2many_grad += value
    return {
        "gain_env": gain,
        "criterion_type": type(criterion).__name__,
        "native_type": type(native).__name__,
        "model_end2end": bool(getattr(model, "end2end", False)),
        "ultralytics_file": str(Path(ultralytics.__file__).resolve()),
        "one2one_cls_gain": float(getattr(native.one2one, "cls_gain", -1.0)),
        "loss_vector": [float(v) for v in (items.detach().cpu().reshape(-1) if isinstance(items, torch.Tensor) else items)],
        "loss_total": float(scalar.detach().cpu()),
        "one2one_cv3_grad_l1": one2one_grad,
        "one2many_cv3_grad_l1": one2many_grad,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps([audit(args.checkpoint, value) for value in (1.0, 1.2)], indent=2), flush=True)


if __name__ == "__main__":
    main()
