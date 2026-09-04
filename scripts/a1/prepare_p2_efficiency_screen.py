"""Prepare a short, late-layer MoE efficiency-screen protocol.

The screen deliberately keeps the pretrained C3k2 path intact and changes only
the residual factor at backbone layer 8.  Dense controls use larger MLP ratios
to approximate the active compute of Top-1/Top-2 MoE.  This is a short pilot,
not a replacement for the locked P1 factorial experiment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

LOCKED_SHA = "acce839c7e895d6b179de7f7093fa879e237cc7b"
SEED = 260829
FACTOR_LAYERS = (4, 6, 8)
TRAIN_LAYERS = {4, 6, 8, 23}
FREEZE = [index for index in range(24) if index not in TRAIN_LAYERS]

VARIANTS = {
    # Dense controls approximate the active routed compute.  The ratio is
    # explicit so the benchmark can report parameter/FLOP differences.
    "late_dense_eq_top1": {"moe": False, "experts": 0, "top_k": 0, "factor_mlp_ratio": 4.0, "control_for": "top1"},
    "late_dense_eq_top2": {"moe": False, "experts": 0, "top_k": 0, "factor_mlp_ratio": 6.0, "control_for": "top2"},
    "late_moe_2e_top1": {"moe": True, "experts": 2, "top_k": 1, "factor_mlp_ratio": 2.0, "control_for": None},
    "late_moe_2e_top2": {"moe": True, "experts": 2, "top_k": 2, "factor_mlp_ratio": 2.0, "control_for": None},
    "late_moe_4e_top1": {"moe": True, "experts": 4, "top_k": 1, "factor_mlp_ratio": 2.0, "control_for": None},
    "late_moe_4e_top2": {"moe": True, "experts": 4, "top_k": 2, "factor_mlp_ratio": 2.0, "control_for": None},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_config(parent: dict, spec: dict) -> dict:
    config = copy.deepcopy(parent)
    config["scale"] = "n"
    config["end2end"] = True
    for layer_index in FACTOR_LAYERS:
        layer = config["backbone"][layer_index]
        if layer[2] != "C3k2":
            raise ValueError(f"layer {layer_index}: expected native C3k2, got {layer[2]}")
        args = layer[3]
        # C3k2 args are [c2, c3k, expansion].  The residual adapter keeps
        # these native base arguments and adds a factor policy afterwards.
        c2, c3k = args[:2]
        expansion = args[2] if len(args) > 2 else 0.5
        layer[2] = "C3k2ResidualFactor"
        layer[3] = [
            c2,
            c3k,
            expansion,
            # The factor is dense at layers 4/6 and only layer 8 is MoE.
            bool(spec["moe"] and layer_index == 8),
            int(spec["experts"] or 4),
            int(spec["top_k"] or 1),
            "dense_mlp",
            False,
            1,
            True,
            float(spec["factor_mlp_ratio"] if layer_index == 8 else 2.0),
        ]
    return config


def common_params(project: Path, name: str, data: Path, epochs: int = 1) -> dict:
    return {
        "epochs": epochs,
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
        "freeze": FREEZE,
        "amp": False,
        "deterministic": True,
        "seed": SEED,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    checkpoint = args.checkpoint.resolve()
    data = args.data.resolve()
    parent_path = repo / "ultralytics/cfg/models/26/yolo26.yaml"
    for path in (checkpoint, data, parent_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() or run_root.exists():
        raise FileExistsError("refusing to overwrite an existing efficiency-screen directory")
    output.mkdir(parents=True)
    run_root.mkdir(parents=True)
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    configs, requests = {}, {}
    for name, spec in VARIANTS.items():
        config_path = output / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(make_config(parent, spec), sort_keys=False, allow_unicode=True), encoding="utf-8")
        configs[name] = {"path": str(config_path), "sha256": sha256(config_path), **spec}
        request_name = f"{name}_seed{SEED}_1ep"
        request = {
            "skill": "yolo.train",
            "request_id": request_name,
            "runtime": {
                "cwd": str(repo),
                "python": "/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python",
                "prefer_cli": False,
            },
            "inputs": {"model": str(run_root / "initializers" / f"{name}.pt"), "task": "detect", "data": str(data)},
            "params": common_params(run_root / "pilot", request_name, data),
            "diagnostics": {"detect_anomaly": False, "failure_report": str(run_root / "pilot" / request_name / "failure_diagnostics.json")},
            "artifacts": {"project": str(run_root / "agent_manifests"), "name": request_name},
            "policy": {"async": False, "dry_run": False},
            "a1_policy": {
                "screen": "late_layer_moe_efficiency",
                "factor_layers": list(FACTOR_LAYERS),
                "moe_layer": 8 if spec["moe"] else None,
                "num_experts": spec["experts"],
                "top_k": spec["top_k"],
                "factor_mlp_ratio": spec["factor_mlp_ratio"],
                "freeze_batch_norm": True,
                "freeze_residual_factor_bases": True,
                "routing_semantics": f"hard_top{spec['top_k']}_from_step_zero" if spec["moe"] else "not_applicable",
            },
        }
        request_path = output / f"{name}_request.json"
        write_json(request_path, request)
        requests[name] = {"path": str(request_path), "sha256": sha256(request_path)}

    protocol = {
        "schema": "a1-p2-late-moe-efficiency-screen/v1",
        "status": "prepared",
        "name": "A1 P2 small efficiency screen: late-layer MoE",
        "locked_sha": LOCKED_SHA,
        "source_checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "parent": {"path": str(parent_path), "sha256": sha256(parent_path)},
        "data": {"path": str(data), "sha256": sha256(data), "train_images": 5000, "val_images": 512},
        "run_root": str(run_root),
        "device": "cuda:1",
        "budget": {"epochs": 1, "batch": 4, "imgsz": 640, "optimizer": "SGD", "lr0": 0.0001, "seed": SEED},
        "variants": configs,
        "requests": requests,
        "dense_controls": {
            "late_dense_eq_top1": {"match": "active Top-1 routed MLP compute", "factor_mlp_ratio": 4.0},
            "late_dense_eq_top2": {"match": "active Top-2 routed MLP compute", "factor_mlp_ratio": 6.0},
        },
        "policy": {
            "only_moe_layer": 8,
            "dense_layers": [4, 6],
            "native_base": "official C3k2 copied from yolo26n.pt and frozen",
            "batch_norm": "all affine parameters and running statistics frozen",
            "preflight": "one epoch, no long training claim",
            "gate": "keep only variants with no meaningful val512 AP loss and acceptable measured latency",
        },
    }
    write_json(output / "protocol.json", protocol)
    (output / "PROTOCOL.md").write_text(
        "# P2 late-layer MoE efficiency screen\n\n"
        "Only backbone layer 8 uses MoE; layers 4/6 retain dense residual factors. "
        "The six variants are two equal-compute Dense controls and four MoE variants "
        "(2/4 experts × Top-1/Top-2). All runs use the native pretrained C3k2 base, "
        "frozen BatchNorm, one pilot epoch, 5000/512 images, batch 4, SGD and seed 260829.\n\n"
        "This is a screening pilot. It cannot replace the locked P1 factorial or support a long-training claim until a variant passes both precision and latency gates.\n",
        encoding="utf-8",
    )
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
