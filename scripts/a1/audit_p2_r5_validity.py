"""Audit r5 source provenance and paired weights, then probe real-image detection gradients."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r5-root", type=Path, required=True)
    parser.add_argument("--r28-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    probe = """import sys, inspect, json
sys.path[0] = sys.argv[1]
from ultralytics import YOLO
import ultralytics
from ultralytics.utils.loss import E2ELoss
print(json.dumps(dict(module=ultralytics.__file__, loss=inspect.getfile(E2ELoss),
    supports_topk_env='A1_E2E_O2O_TAL_TOPK' in inspect.getsource(E2ELoss))))
"""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    old_import = subprocess.run(
        [sys.executable, "-c", probe, str(REPO / "scripts/a1")],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    import torch
    from run_p1_bn_frozen import enforce_p1_freeze_policy

    import ultralytics
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.loss import E2ELoss

    torch.set_num_threads(2)
    assert Path(ultralytics.__file__).resolve() == REPO / "ultralytics/__init__.py"
    evidence = {
        "schema": "p2-r5-validity-and-gradient-audit/v1",
        "status": "running",
        "device": "cpu",
        "torch_threads": 2,
        "original_script_import_reproduction": json.loads(old_import.stdout.strip().splitlines()[-1]),
        "controlled_module": ultralytics.__file__,
        "controlled_loss": inspect.getfile(E2ELoss),
        "controlled_loss_sha256": digest(inspect.getfile(E2ELoss)),
        "checkpoint_pairs": {},
        "gradients": [],
        "scope": "Fixed first training images, no optimizer steps, separate detection branches, excludes MoE aux loss",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    for cell in ("b", "d"):
        paths = [args.r5_root / f"{cell}_{name}_seed260829_5ep/weights/last.pt" for name in ("control", "candidate10")]
        checkpoints = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
        states = [(c.get("ema") if c.get("ema") is not None else c["model"]).state_dict() for c in checkpoints]
        changed = [k for k in states[0] if not torch.equal(states[0][k], states[1][k])]
        evidence["checkpoint_pairs"][cell] = {
            "paths": list(map(str, paths)),
            "file_sha256": list(map(digest, paths)),
            "state_tensors": len(states[0]),
            "changed_tensors": len(changed),
            "changed_keys": changed,
        }
        del checkpoints, states
    save()
    data = check_det_dataset(str(args.data), autodownload=False)
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 640, "cache": False, "workers": 0})
    dataset = build_yolo_dataset(cfg, data["train"], 1, data, mode="val", rect=False)
    evidence["images"] = [
        {"path": dataset.im_files[i], "sha256": digest(dataset.im_files[i])} for i in range(args.images)
    ]
    for cell in ("b", "d"):
        path = args.r28_root / f"formal/seed260829/{cell}_formal_seed260829_15ep/weights/last.pt"
        model = YOLO(str(path), task="detect").model.float().cpu()
        model.args = get_cfg(overrides=model.args) if isinstance(model.args, dict) else model.args
        for name, p in model.named_parameters():
            p.requires_grad_(int(name.split(".")[1]) in {4, 6, 8, 23})
        model.train()
        trainer = SimpleNamespace(model=model)
        enforce_p1_freeze_policy(trainer)
        assert trainer.p1_frozen_factor_base_parameters == 459232
        names, parameters = zip(*[(n, p) for n, p in model.named_parameters() if p.requires_grad])
        os.environ["A1_E2E_O2O_TAL_TOPK2"] = "1"
        os.environ["A1_E2E_O2O_CLS_GAIN"] = "1.0"
        os.environ["A1_E2E_O2O_CONFLICT_METRIC"] = "overlap"
        for i in range(args.images):
            batch = dataset.collate_fn([dataset[i]])
            batch["img"] = batch["img"].float() / 255
            torch.manual_seed(260829 + i)
            preds = model(batch["img"])
            paired_grad = {}
            for branch, topk in (("one2many", 10), ("one2one", 7), ("one2one", 10)):
                os.environ["A1_E2E_O2O_TAL_TOPK"] = str(topk)
                criterion = model.init_criterion()
                native = getattr(criterion, "native_criterion", criterion)
                loss_fn = getattr(native, branch)
                assert loss_fn.assigner.topk == topk
                loss_fn.assigner.audit_enabled = True
                total, _ = loss_fn.loss(preds[branch], batch)
                grads = torch.autograd.grad(total.sum(), parameters, retain_graph=True, allow_unused=True)
                stats = {}
                for group, select in {
                    "factor": lambda n: int(n.split(".")[1]) in {4, 6, 8},
                    "router": lambda n: "routing" in n or "router" in n,
                    "one2one_head": lambda n: "one2one_" in n,
                    "one2many_head": lambda n: n.startswith("model.23.") and "one2one_" not in n,
                }.items():
                    chosen = [g for n, g in zip(names, grads) if select(n)]
                    stats[group] = {
                        "tensors": len(chosen),
                        "none": sum(g is None for g in chosen),
                        "l1": sum(float(g.abs().sum()) for g in chosen if g is not None),
                    }
                row = {
                    "cell": cell,
                    "image": dataset.im_files[i],
                    "branch": branch,
                    "topk": topk,
                    "loss": float(total.detach().sum()),
                    "gt": len(batch["cls"]),
                    "gradients": stats,
                    "assigner": loss_fn.assigner.last_audit,
                }
                if branch == "one2one":
                    paired_grad[topk] = [g.detach().clone() if g is not None else None for g in grads]
                    if topk == 10:
                        row["gradient_max_difference_vs_7"] = max(
                            (
                                float((a - b).abs().max())
                                for a, b in zip(paired_grad[7], paired_grad[10])
                                if a is not None and b is not None
                            ),
                            default=0.0,
                        )
                evidence["gradients"].append(row)
                del criterion, native, grads, loss_fn, total
            del preds, paired_grad
            save()
            print(f"{cell}: {i + 1}/{args.images}", flush=True)
        del model, parameters
    evidence["status"] = "completed"
    evidence["r5_training_comparison_valid"] = False
    save()
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
