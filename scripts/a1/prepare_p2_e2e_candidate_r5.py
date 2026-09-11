#!/usr/bin/env python3
"""Prepare the P2-E r5 one-to-one candidate-budget pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LOCKED_SHA = "acce839c7e895d6b179de7f7093fa879e237cc7b"
ROUTING_PARAMS = {
    "moe_noise_std": 0.0,
    "moe_router_lr_scale": 1.0,
    "moe_expert_warmup_epochs": 0,
    "moe_dynamic_schedule": "none",
    "moe_map_saturation_enabled": False,
    "moe_balance_loss": 1.0,
    "moe_router_z_loss": 0.1,
    "moe_aux_gain": 1.0,
    "mixture_aux_budget": 3.0,
    "moe_temperature": 1.0,
    "moa_mot_temperature_factor": 1.0,
    "moa_mot_min_temperature": 1.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def common_params(project: Path, name: str) -> dict:
    return {
        "epochs": 5,
        "imgsz": 640,
        "batch": 4,
        "workers": 0,
        "pretrained": True,
        "optimizer": "SGD",
        "lr0": 0.0001,
        "lrf": 0.2,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "nbs": 16,
        "warmup_epochs": 0.5,
        "warmup_bias_lr": 0.0,
        "warmup_momentum": 0.8,
        "freeze": [i for i in range(24) if i not in {4, 6, 8, 23}],
        "amp": False,
        "deterministic": True,
        "seed": 260829,
        "patience": 0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 0,
        "cos_lr": False,
        "cache": False,
        "plots": False,
        "save": True,
        "save_period": 1,
        "val": True,
        "fraction": 1.0,
        "exist_ok": False,
        "device": "1",
        "project": str(project),
        "name": name,
        **ROUTING_PARAMS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--b-init", type=Path, required=True)
    parser.add_argument("--d-init", type=Path, required=True)
    args = parser.parse_args()
    repo, output, run_root, data = (path.resolve() for path in (args.repo, args.output, args.run_root, args.data))
    if output.exists() or run_root.exists():
        raise FileExistsError(f"refusing to overwrite output={output} or run_root={run_root}")
    for path in (data, args.b_init.resolve(), args.d_init.resolve()):
        if not path.is_file():
            raise FileNotFoundError(path)
    if len([line for line in (data.parent / "val2017.txt").read_text(encoding="utf-8").splitlines() if line.strip()]) != 512:
        raise ValueError("candidate r5 requires fixed val512")
    output.mkdir(parents=True, exist_ok=True)
    requests_dir = output / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        "b_control": {"cell": "b", "topk": 7, "initializer": args.b_init.resolve()},
        "b_candidate10": {"cell": "b", "topk": 10, "initializer": args.b_init.resolve()},
        "d_control": {"cell": "d", "topk": 7, "initializer": args.d_init.resolve()},
        "d_candidate10": {"cell": "d", "topk": 10, "initializer": args.d_init.resolve()},
    }
    requests = {}
    for name, spec in runs.items():
        request = {
            "skill": "yolo.train",
            "request_id": f"p2_e2e_candidate_r5_{name}_seed260829_5ep",
            "runtime": {"cwd": str(repo), "python": "/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python", "prefer_cli": False},
            "inputs": {"model": str(spec["initializer"]), "task": "detect", "data": str(data)},
            "params": common_params(run_root, f"{name}_seed260829_5ep"),
            "diagnostics": {"detect_anomaly": False, "failure_report": str(run_root / "train" / f"{name}_seed260829_5ep" / "failure_diagnostics.json")},
            "artifacts": {"project": str(run_root / "agent_manifests"), "name": f"{name}_seed260829_5ep"},
            "policy": {"async": False, "dry_run": False},
            "a1_policy": {
                "freeze_batch_norm": True,
                "freeze_residual_factor_bases": True,
                "formal_restart_from_initializer": True,
                "routing_semantics": "deterministic_hard_top2_from_step_zero",
                "expert_dropout_rate": 0.0,
                "o2o_tal_topk": spec["topk"],
            },
        }
        path = requests_dir / f"{name}.json"
        path.write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        requests[name] = {"path": str(path), "sha256": sha256(path)}

    runs_payload = {
        name: {**spec, "initializer": str(spec["initializer"])}
        for name, spec in runs.items()
    }
    protocol = {
        "schema": "a1-p2-e2e-candidate-budget-r5/v1",
        "status": "prepared",
        "locked_sha": LOCKED_SHA,
        "data": {"yaml": str(data), "yaml_sha256": sha256(data), "train_images": 5000, "val_images": 512, "val_list": str(data.parent / "val2017.txt"), "val_list_sha256": sha256(data.parent / "val2017.txt")},
        "run_root": str(run_root),
        "factor": "one_to_one_candidate_budget",
        "control": "official one-to-one tal_topk=7, tal_topk2=1",
        "treatment": "one-to-one tal_topk=10, tal_topk2=1; only candidate budget changes",
        "budget": "5000 train / 512 val, batch 4, 5 epochs, seed 260829, GPU1, SGD lr0=1e-4, amp=False",
        "initializers": {"b": {"path": str(args.b_init.resolve()), "sha256": sha256(args.b_init.resolve())}, "d": {"path": str(args.d_init.resolve()), "sha256": sha256(args.d_init.resolve())}},
        "runs": runs_payload,
        "requests": requests,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
