#!/usr/bin/env python3
"""Generate the immutable, three-seed A1 P1 r19 residual-factor protocol."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

import yaml
from prepare_p1_factorial import CELLS, FREEZE, LOCKED_SHA, TRAIN_LAYERS, sha256, write_json
from run_p1_bn_frozen import (
    P1_ROUTING_PARAMS,
    R19_EXPLORATION_POLICY,
    R19_GAIN_POLICY,
    R19_ROUTING_SEMANTICS,
)

SEEDS = (260829, 260830, 260831)
FACTOR_LAYERS = (4, 6, 8)
EXPERTS = {4: 4, 6: 8, 8: 16}
INITIALIZER_SUFFIX = "residual_factor_init"
ROUTER_INITIALIZATION = "native_iid_normal_final_projection_std_0.05"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pilot-data", required=True, type=Path)
    parser.add_argument("--preflight-data", required=True, type=Path)
    parser.add_argument("--experiment-tag", default="r19")
    return parser.parse_args()


def git_output(repo: Path, *args: str) -> str:
    """Return one Git value from the isolated experiment worktree."""
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def resolve_list_path(data_yaml: Path, entry: str) -> Path:
    """Resolve a dataset split list using the same root rules as a YOLO data YAML."""
    candidate = Path(entry)
    if candidate.is_absolute():
        return candidate.resolve()
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(spec.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    return (root / candidate).resolve()


def calibration_images(pilot_data: Path) -> tuple[list[str], Path]:
    """Return the first 512 declared pilot-train images and their source list."""
    spec = yaml.safe_load(pilot_data.read_text(encoding="utf-8"))
    train = spec.get("train")
    if not isinstance(train, str):
        raise TypeError("r19 requires pilot train to be one explicit image-list file")
    source = resolve_list_path(pilot_data, train)
    lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 512:
        raise ValueError(f"r19 BN calibration needs 512 train images, found {len(lines)}")
    resolved = []
    for line in lines[:512]:
        image = Path(line)
        if not image.is_absolute():
            image = source.parent / image
        image = image.resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        resolved.append(str(image))
    return resolved, source


def common_params(*, seed: int, epochs: int, project: Path, name: str) -> dict:
    """Return the exact shared P1 optimization budget for one independent run."""
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
        "seed": seed,
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
        "resume": False,
        "device": "0",
        "project": str(project),
        "name": name,
        **P1_ROUTING_PARAMS,
    }


def training_request(
    *, repo: Path, run_root: Path, data: Path, initializer: Path, seed: int, stage: str, cell: str, epochs: int
) -> dict:
    """Build one immutable request addressed to logical GPU0."""
    name = f"{cell}_{stage}_seed{seed}_{epochs}ep"
    project = run_root / stage / f"seed{seed}"
    return {
        "skill": "yolo.train",
        "request_id": name,
        "runtime": {
            "cwd": str(repo),
            "python": "/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python",
            "prefer_cli": False,
        },
        "inputs": {"model": str(initializer), "task": "detect", "data": str(data)},
        "params": common_params(seed=seed, epochs=epochs, project=project, name=name),
        "a1_policy": {
            "freeze_batch_norm": True,
            "freeze_residual_factor_bases": True,
            "formal_restart_from_initializer": stage == "formal",
            "routing_semantics": R19_ROUTING_SEMANTICS,
            "expert_dropout_rate": 0.0,
            "router_exploration": {
                **R19_EXPLORATION_POLICY,
                "base_seed": seed,
                "enabled": cell in "cd",
            },
            "factor_gain_optimizer": R19_GAIN_POLICY,
            "batch_norm_calibration": "train_only_fixed_512_no_grad_then_frozen",
        },
        "diagnostics": {
            "detect_anomaly": False,
            "failure_report": str(project / name / "failure_diagnostics.json"),
        },
        "policy": {"async": False, "dry_run": False},
    }


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    run_root = args.run_root.resolve()
    checkpoint = args.checkpoint.resolve()
    pilot_data = args.pilot_data.resolve()
    preflight_data = args.preflight_data.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite config directory: {output}")
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {run_root}")
    for path in (checkpoint, pilot_data, preflight_data):
        if not path.is_file():
            raise FileNotFoundError(path)

    parent_path = repo / "ultralytics/cfg/models/26/yolo26.yaml"
    implementation_paths = (
        repo / "ultralytics/engine/trainer.py",
        repo / "ultralytics/nn/modules/moe/routers.py",
        repo / "ultralytics/nn/modules/moe/factor_adapter.py",
        repo / "ultralytics/nn/modules/moe/modules.py",
        repo / "ultralytics/nn/modules/moe/experts.py",
        repo / "ultralytics/nn/modules/moe/__init__.py",
        repo / "ultralytics/nn/modules/__init__.py",
        repo / "ultralytics/nn/tasks.py",
        repo / "scripts/a1/prepare_p1_residual_factor_r19.py",
        repo / "scripts/a1/build_p1_residual_factor_initializers_r19.py",
        repo / "scripts/a1/run_p1_bn_frozen.py",
        repo / "scripts/a1/run_p1_factorial_multiseed_2gpu_r19.py",
        repo / "scripts/a1/audit_p1_checkpoints_r19.py",
        repo / "scripts/a1/audit_p1_routing_r19.py",
        repo / "tests/test_p1_residual_factor.py",
        repo / "tests/test_moe_router_boundaries.py",
        repo / "tests/test_peft_optimizer_policy.py",
    )
    for path in (parent_path, *implementation_paths):
        if not path.is_file():
            raise FileNotFoundError(path)

    images, source_list = calibration_images(pilot_data)
    output.mkdir(parents=True)
    calibration_list = output / "bn_calibration_train512.txt"
    calibration_list.write_text("\n".join(images) + "\n", encoding="utf-8")
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    configs = {}
    for cell, factors in CELLS.items():
        config = copy.deepcopy(parent)
        config["scale"] = "n"
        config["end2end"] = factors["end2end"]
        for layer_index, num_experts in EXPERTS.items():
            layer = config["backbone"][layer_index]
            if layer[2] != "C3k2":
                raise ValueError(f"layer {layer_index}: expected C3k2, got {layer[2]}")
            c2, c3k = layer[3][:2]
            expansion = layer[3][2] if len(layer[3]) > 2 else 0.5
            layer[2] = "C3k2ResidualFactor"
            layer[3] = [c2, c3k, expansion, factors["moe"], num_experts, 2, "dense_mlp"]
        config_path = output / "models" / f"{cell}_matched.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        configs[cell] = {"path": str(config_path), "sha256": sha256(config_path), **factors}

    requests = {"preflight": {}, "routing_probe": {}, "formal": {}}
    for seed in SEEDS:
        seed_key = str(seed)
        for stage, stage_requests in requests.items():
            stage_requests[seed_key] = {}
        for cell in CELLS:
            initializer = run_root / "initializers" / f"seed{seed}" / f"{cell}_{INITIALIZER_SUFFIX}.pt"
            for stage, data, epochs in (("preflight", preflight_data, 1), ("formal", pilot_data, 5)):
                request = training_request(
                    repo=repo,
                    run_root=run_root,
                    data=data,
                    initializer=initializer,
                    seed=seed,
                    stage=stage,
                    cell=cell,
                    epochs=epochs,
                )
                path = output / "requests" / stage / f"seed{seed}" / f"{cell}.json"
                write_json(path, request)
                requests[stage][seed_key][cell] = {"path": str(path), "sha256": sha256(path)}
            if cell in "cd":
                request = training_request(
                    repo=repo,
                    run_root=run_root,
                    data=pilot_data,
                    initializer=initializer,
                    seed=seed,
                    stage="routing_probe",
                    cell=cell,
                    epochs=1,
                )
                path = output / "requests" / "routing_probe" / f"seed{seed}" / f"{cell}.json"
                write_json(path, request)
                requests["routing_probe"][seed_key][cell] = {"path": str(path), "sha256": sha256(path)}

    protocol = {
        "schema_version": 5,
        "name": f"A1 P1 equal-budget 2x2 factorial {args.experiment_tag} residual-factor",
        "experiment_tag": args.experiment_tag,
        "locked_official_baseline_sha": LOCKED_SHA,
        "implementation_head": git_output(repo, "rev-parse", "HEAD"),
        "implementation_branch": git_output(repo, "branch", "--show-current"),
        "source_checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "parent": {"path": str(parent_path), "sha256": sha256(parent_path)},
        "implementation": {
            str(path.relative_to(repo)).replace("\\", "/"): sha256(path) for path in implementation_paths
        },
        "data": {
            "preflight": {"path": str(preflight_data), "sha256": sha256(preflight_data)},
            "pilot": {"path": str(pilot_data), "sha256": sha256(pilot_data)},
        },
        "configs": configs,
        "requests": requests,
        "run_root": str(run_root),
        "seeds": list(SEEDS),
        "train_layers": sorted(TRAIN_LAYERS),
        "freeze": FREEZE,
        "factor_base": "native official P0 C3k2 frozen inside ResidualFactorAdapter",
        "factor_base_expected_parameters": 459232,
        "factor_gain_optimizer": R19_GAIN_POLICY,
        "batch_norm": {
            "calibration_split": "pilot train only",
            "selection": "first 512 paths in the immutable pilot train list",
            "source_list": str(source_list),
            "image_list": str(calibration_list),
            "image_list_sha256": sha256(calibration_list),
            "images": 512,
            "batch": 4,
            "imgsz": 640,
            "augment": False,
            "shuffle": False,
            "grad": False,
            "post_calibration_training_policy": "all BatchNorm affine parameters and running statistics frozen",
        },
        "router_initialization": {
            "scheme": ROUTER_INITIALIZATION,
            "std": 0.05,
            "simplex_override": False,
            "c_d_tensor_parity_within_seed": True,
        },
        "routing": {
            "semantics": R19_ROUTING_SEMANTICS,
            "top_k": 2,
            "progressive_sparsity": False,
            "expert_dropout_rate": 0.0,
            "generic_moe_noise_std": 0.0,
            "train_only_private_exploration": R19_EXPLORATION_POLICY,
            "validation_audit_export_noise_std": 0.0,
        },
        "routing_gate": {
            "sample": "fixed pilot val 512",
            "dead_experts": 0,
            "maximum_image_selection_fraction": 0.8,
            "minimum_normalized_entropy": 0.5,
            "applies_to": "every router in C and D for every seed",
        },
        "routing_gate_scope": "predeclared internal formal-admission gate, not an A1 taskbook numeric requirement",
        "preflight_policy": "all 12 runs start from their seed initializer; discard all weights",
        "routing_probe_policy": "C/D x 3 seeds start independently from original initializers; discard all weights",
        "formal_policy": "start only after every initializer, preflight, and routing gate passes; never reuse probe weights",
        "formal_budget": {
            "train_images": 5000,
            "epochs": 5,
            "physical_batch": 4,
            "nbs": 16,
            "accumulate": 4,
            "optimizer": "SGD",
            "lr0": 0.0001,
            "amp": False,
            "workers": 0,
        },
        "gpu_policy": "two independent one-GPU jobs; never DDP; request-visible device is logical 0",
        "primary_checkpoint": "last.pt",
        "secondary_checkpoint": "best.pt",
        "comparisons": {
            "end2end_dense": "B-A",
            "end2end_moe": "D-C",
            "moe_non_end2end": "C-A",
            "moe_end2end": "D-B",
            "interaction": "(D-C)-(B-A)",
        },
    }
    write_json(output / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
