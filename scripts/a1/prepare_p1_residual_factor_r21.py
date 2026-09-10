#!/usr/bin/env python3
"""Generate the immutable A1 P1 r21 three-seed residual-factor protocol.

r21 preserves r20's MoE training semantics exactly: noisy hard-Top2 dispatch
and clean balance/z-loss inputs.  Its only change from r20 is fail-closed
binding of every process to the isolated worktree actually named by the
protocol.  Model behavior, budgets, and gate thresholds are unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml
from p1_r21_runtime import RUNTIME_ATTESTATION

# isort: split

from p1_r21_integrity import data_list_content_signature

LOCKED_SHA = "acce839c7e895d6b179de7f7093fa879e237cc7b"
OFFICIAL_CHECKPOINT = Path("/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master/yolo26n.pt")
OFFICIAL_CHECKPOINT_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
SEEDS = (260829, 260830, 260831)
CELLS = {
    "a": {"moe": False, "end2end": False},
    "b": {"moe": False, "end2end": True},
    "c": {"moe": True, "end2end": False},
    "d": {"moe": True, "end2end": True},
}
FACTOR_LAYERS = (4, 6, 8)
EXPERTS = {4: 4, 6: 8, 8: 16}
TRAIN_LAYERS = {4, 6, 8, 23}
FREEZE = [index for index in range(24) if index not in TRAIN_LAYERS]
INITIALIZER_SUFFIX = "residual_factor_init"
ROUTING_SEMANTICS = "hard_top2_from_step_zero_private_exploration_clean_aux"
ROUTER_INITIALIZATION = "native_iid_normal_final_projection_std_0.05"
GAIN_POLICY = {"lr": 0.01, "weight_decay": 0.0, "warmup": False}
EXPLORATION_POLICY = {
    "sigma_source": "train512_median_per_image_logit_std_clipped",
    "sigma_min": 0.01,
    "sigma_max": 0.05,
    "hold_through_microbatch": 625,
    "decay_to_zero_microbatch": 1000,
    "private_seed_stride": 10000,
    "evaluation_noise_std": 0.0,
}
CLEAN_AUX_POLICY = {
    "enabled": True,
    "runtime_semantics": "clean_hard_top2_balance_with_noisy_dispatch",
    "dispatch_source": "train_only_private_noisy_logits",
    "dispatch_operator": "hard_top2",
    "balance_probability_source": "clean_logits_softmax",
    "balance_assignment_source": "clean_logits_hard_top2",
    "z_loss_source": "clean_logits",
    "evaluation_source": "clean_logits_hard_top2",
    "adds_parameters": False,
    "changes_inference": False,
}
P1_ROUTING_PARAMS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pilot-data", required=True, type=Path)
    parser.add_argument("--preflight-data", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path("/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python"))
    parser.add_argument("--experiment-tag", default="r21")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def require_locked_baseline_ancestor(repo: Path) -> None:
    """Fail unless the r21 implementation descends from the locked official baseline."""
    commit = subprocess.run(
        ["git", "cat-file", "-e", f"{LOCKED_SHA}^{{commit}}"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0:
        raise ValueError(f"locked official baseline commit is unavailable: {LOCKED_SHA}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", LOCKED_SHA, "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise ValueError(f"HEAD does not descend from locked official baseline {LOCKED_SHA}")


def same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def forbid_r19_artifact(path: Path, label: str) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    if "p1_factorial_r19" in normalized or "yolo-master-r19" in normalized:
        raise ValueError(f"{label} must not reference an r19 artifact: {path}")


def validate_destinations(repo: Path, run_root: Path, output: Path, tag: str) -> None:
    if tag != "r21":
        raise ValueError(f"r21 protocol requires experiment-tag r21, got {tag!r}")
    if repo.name != "YOLO-Master-r21":
        raise ValueError(f"r21 requires isolated worktree YOLO-Master-r21, got {repo}")
    expected_output = repo / "configs/a1/p1_factorial_r21"
    if not same_path(output, expected_output):
        raise ValueError(f"r21 config directory must be {expected_output}, got {output}")
    if run_root.name != "p1_factorial_r21":
        raise ValueError(f"r21 run root must end in p1_factorial_r21, got {run_root}")
    forbid_r19_artifact(repo, "worktree")
    forbid_r19_artifact(run_root, "run root")
    forbid_r19_artifact(output, "config directory")


def resolve_list_path(data_yaml: Path, entry: str) -> Path:
    candidate = Path(entry)
    if candidate.is_absolute():
        return candidate.resolve()
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(spec.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    return (root / candidate).resolve()


def calibration_images(pilot_data: Path) -> tuple[list[str], Path]:
    spec = yaml.safe_load(pilot_data.read_text(encoding="utf-8"))
    train = spec.get("train")
    if not isinstance(train, str):
        raise TypeError("r21 requires pilot train to be one explicit image-list file")
    source = resolve_list_path(pilot_data, train)
    lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 512:
        raise ValueError(f"r21 BN calibration needs 512 train images, found {len(lines)}")
    resolved: list[str] = []
    for line in lines[:512]:
        image = Path(line)
        if not image.is_absolute():
            image = source.parent / image
        image = image.resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        resolved.append(str(image))
    return resolved, source


def data_registry_entry(data_yaml: Path, *, expected_train: int, expected_val: int) -> dict[str, Any]:
    """Bind a dataset YAML and the ordered train/val list files it references."""
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    lists: dict[str, Any] = {}
    for split, expected_count in (("train", expected_train), ("val", expected_val)):
        entry = spec.get(split)
        if not isinstance(entry, str):
            raise TypeError(f"r21 requires {data_yaml} {split} to be one explicit image-list file")
        path = resolve_list_path(data_yaml, entry)
        if not path.is_file():
            raise FileNotFoundError(path)
        count = sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
        if count != expected_count:
            raise ValueError(f"{data_yaml} {split}: expected {expected_count} images, found {count}")
        lists[split] = {
            "path": str(path),
            "sha256": sha256(path),
            "images": count,
            "content": data_list_content_signature(path),
        }
    return {"path": str(data_yaml), "sha256": sha256(data_yaml), "lists": lists}


def common_params(*, seed: int, epochs: int, project: Path, name: str) -> dict[str, Any]:
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


def cell_aux_policy(cell: str) -> dict[str, Any]:
    if cell in "cd":
        return dict(CLEAN_AUX_POLICY)
    return {
        "enabled": False,
        "reason": "dense cell has no router",
        "adds_parameters": False,
        "changes_inference": False,
    }


def training_request(
    *,
    repo: Path,
    python: Path,
    run_root: Path,
    data: Path,
    initializer: Path,
    seed: int,
    stage: str,
    cell: str,
    epochs: int,
    protocol_path: Path,
) -> dict[str, Any]:
    name = f"{cell}_{stage}_seed{seed}_{epochs}ep"
    project = run_root / stage / f"seed{seed}"
    return {
        "skill": "yolo.train",
        "request_id": name,
        "protocol": {"path": str(protocol_path)},
        "runtime": {"cwd": str(repo), "python": str(python), "prefer_cli": False},
        "inputs": {"model": str(initializer), "task": "detect", "data": str(data)},
        "params": common_params(seed=seed, epochs=epochs, project=project, name=name),
        "a1_policy": {
            "freeze_batch_norm": True,
            "freeze_residual_factor_bases": True,
            "formal_restart_from_initializer": stage == "formal",
            "routing_semantics": ROUTING_SEMANTICS,
            "expert_dropout_rate": 0.0,
            "router_exploration": {**EXPLORATION_POLICY, "base_seed": seed, "enabled": cell in "cd"},
            "routing_auxiliary_objective": cell_aux_policy(cell),
            "factor_gain_optimizer": GAIN_POLICY,
            "batch_norm_calibration": "train_only_fixed_512_no_grad_then_frozen",
        },
        "diagnostics": {
            "detect_anomaly": False,
            "failure_report": str(project / name / "failure_diagnostics.json"),
        },
        "policy": {"async": False, "dry_run": False},
    }


def request_count(requests: dict[str, Any]) -> int:
    return sum(len(cells) for stage in requests.values() for cells in stage.values())


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    run_root = args.run_root.resolve()
    checkpoint = args.checkpoint.resolve()
    pilot_data = args.pilot_data.resolve()
    preflight_data = args.preflight_data.resolve()
    python = args.python.resolve()
    validate_destinations(repo, run_root, output, args.experiment_tag)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite config directory: {output}")
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {run_root}")
    for path in (checkpoint, pilot_data, preflight_data, python):
        if not path.is_file():
            raise FileNotFoundError(path)
    forbid_r19_artifact(checkpoint, "source checkpoint")
    if checkpoint != OFFICIAL_CHECKPOINT.resolve():
        raise ValueError(f"r21 must start from the locked official checkpoint {OFFICIAL_CHECKPOINT}, got {checkpoint}")
    if sha256(checkpoint) != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("official yolo26n.pt SHA-256 mismatch")
    branch = git_output(repo, "branch", "--show-current")
    if branch != "a1-p1-r21":
        raise ValueError(f"r21 worktree branch must be a1-p1-r21, got {branch!r}")
    require_locked_baseline_ancestor(repo)
    dirty = git_output(repo, "status", "--porcelain")
    if dirty:
        raise ValueError("implementation worktree must be clean before generating the r21 protocol")

    parent_path = repo / "ultralytics/cfg/models/26/yolo26.yaml"
    moe_runtime_paths = tuple(sorted((repo / "ultralytics/nn/modules/moe").glob("*.py")))
    implementation_paths = (
        repo / "ultralytics/cfg/default.yaml",
        repo / "ultralytics/cfg/__init__.py",
        repo / "ultralytics/engine/model.py",
        repo / "ultralytics/engine/trainer.py",
        repo / "ultralytics/models/yolo/detect/train.py",
        repo / "ultralytics/utils/loss.py",
        repo / "ultralytics/utils/torch_utils.py",
        *moe_runtime_paths,
        repo / "ultralytics/nn/modules/__init__.py",
        repo / "ultralytics/nn/tasks.py",
        repo / "scripts/a1/prepare_p1_residual_factor_r21.py",
        repo / "scripts/a1/p1_r21_integrity.py",
        repo / "scripts/a1/p1_r21_runtime.py",
        repo / "scripts/a1/prepare_p1_factorial.py",
        repo / "scripts/a1/build_p1_residual_factor_initializers_r21.py",
        repo / "scripts/a1/build_p1_residual_factor_initializers.py",
        repo / "scripts/a1/run_p1_bn_frozen_r21.py",
        repo / "scripts/a1/run_p1_factorial_multiseed_2gpu_r21.py",
        repo / "scripts/a1/audit_p1_checkpoints_r21.py",
        repo / "scripts/a1/audit_p1_residual_factor_checkpoints.py",
        repo / "scripts/a1/audit_p1_routing_r21.py",
        repo / "scripts/a1/audit_p1_preflight_routing.py",
        repo / "scripts/a1/audit_p1_routing.py",
        repo / "scripts/a1/evaluate_p1_matrix.py",
        repo / "scripts/a1/audit_p1_residual_activity_r21.py",
        repo / "scripts/a1/audit_p1_formal_admission_r21.py",
        repo / "scripts/a1/verify_p1_r21_migration.py",
        repo / "tests/test_p1_residual_factor.py",
        repo / "tests/test_moe_router_boundaries.py",
        repo / "tests/test_peft_optimizer_policy.py",
        repo / "tests/test_p1_r21_protocol.py",
    )
    for path in (parent_path, *implementation_paths):
        if not path.is_file():
            raise FileNotFoundError(path)

    pilot_registry = data_registry_entry(pilot_data, expected_train=5000, expected_val=512)
    preflight_registry = data_registry_entry(preflight_data, expected_train=256, expected_val=128)
    images, source_list = calibration_images(pilot_data)
    output.mkdir(parents=True)
    calibration_list = output / "bn_calibration_train512.txt"
    calibration_list.write_text("\n".join(images) + "\n", encoding="utf-8")
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    configs: dict[str, Any] = {}
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

    requests: dict[str, Any] = {"preflight": {}, "routing_probe": {}, "formal": {}}
    for seed in SEEDS:
        seed_key = str(seed)
        for stage_payload in requests.values():
            stage_payload[seed_key] = {}
        for cell in CELLS:
            initializer = run_root / "initializers" / f"seed{seed}" / f"{cell}_{INITIALIZER_SUFFIX}.pt"
            forbid_r19_artifact(initializer, f"initializer {seed}/{cell}")
            for stage, data, epochs in (("preflight", preflight_data, 1), ("formal", pilot_data, 5)):
                request = training_request(
                    repo=repo,
                    python=python,
                    run_root=run_root,
                    data=data,
                    initializer=initializer,
                    seed=seed,
                    stage=stage,
                    cell=cell,
                    epochs=epochs,
                    protocol_path=output / "protocol.json",
                )
                path = output / "requests" / stage / f"seed{seed}" / f"{cell}.json"
                write_json(path, request)
                requests[stage][seed_key][cell] = {"path": str(path), "sha256": sha256(path)}
            if cell in "cd":
                request = training_request(
                    repo=repo,
                    python=python,
                    run_root=run_root,
                    data=pilot_data,
                    initializer=initializer,
                    seed=seed,
                    stage="routing_probe",
                    cell=cell,
                    epochs=1,
                    protocol_path=output / "protocol.json",
                )
                path = output / "requests" / "routing_probe" / f"seed{seed}" / f"{cell}.json"
                write_json(path, request)
                requests["routing_probe"][seed_key][cell] = {"path": str(path), "sha256": sha256(path)}
    if request_count(requests) != 30:
        raise AssertionError(f"expected exactly 30 requests, got {request_count(requests)}")

    protocol: dict[str, Any] = {
        "schema_version": 7,
        "name": "A1 P1 equal-budget 2x2 factorial r21 residual-factor",
        "experiment_tag": "r21",
        "locked_official_baseline_sha": LOCKED_SHA,
        "locked_official_baseline_ancestor_verified": True,
        "implementation_head": git_output(repo, "rev-parse", "HEAD"),
        "implementation_branch": branch,
        "runtime_binding": RUNTIME_ATTESTATION,
        "source_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "lineage": "official_yolo26n_only; never any P1 initializer or trained checkpoint",
        },
        "parent": {"path": str(parent_path), "sha256": sha256(parent_path)},
        "implementation": {
            str(path.relative_to(repo)).replace("\\", "/"): sha256(path) for path in implementation_paths
        },
        "data": {
            "preflight": preflight_registry,
            "pilot": pilot_registry,
        },
        "configs": configs,
        "requests": requests,
        "request_count": 30,
        "run_root": str(run_root),
        "seeds": list(SEEDS),
        "train_layers": sorted(TRAIN_LAYERS),
        "freeze": FREEZE,
        "factor_base": "native official P0 C3k2 frozen inside ResidualFactorAdapter",
        "factor_base_expected_parameters": 459232,
        "factor_gain_optimizer": GAIN_POLICY,
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
            "semantics": ROUTING_SEMANTICS,
            "top_k": 2,
            "progressive_sparsity": False,
            "expert_dropout_rate": 0.0,
            "generic_moe_noise_std": 0.0,
            "train_only_private_exploration": EXPLORATION_POLICY,
            "auxiliary_objective": CLEAN_AUX_POLICY,
            "validation_audit_export_noise_std": 0.0,
        },
        "model_change_inherited_from_r20": {
            "scope": "C/D routers only, identically for every layer and seed",
            "change": "balance and z losses consume clean routing tensors; noisy hard-Top2 dispatch is unchanged",
            "a_b_tensor_and_output_invariance_required": True,
            "c_d_policy_parity_required": True,
        },
        "r21_execution_repair": {
            "scope": "all versioned initializer, runner, driver, and audit entrypoints",
            "change": "pin and attest the isolated worktree modules actually loaded by every process",
            "shared_virtualenv_editable_install_changed": False,
            "model_or_routing_semantics_changed_from_r20": False,
            "thresholds_changed": False,
            "budget_changed": False,
            "inference_changed": False,
        },
        "routing_gate": {
            "sample": "fixed pilot val 512",
            "dead_experts": 0,
            "maximum_image_selection_fraction": 0.8,
            "minimum_normalized_entropy": 0.5,
            "applies_to": "every router in C and D for every seed",
        },
        "routing_gate_scope": "predeclared internal formal-admission gate, not an A1 taskbook numeric requirement",
        "residual_activity_gate": {
            "checkpoint": "routing-probe last.pt",
            "sample": "same fixed pilot val 512 as hard-Top2 routing audit",
            "scope": "every layer 4, 6, and 8 in C and D for every seed",
            "metric": "sqrt(sum((gain * factor(base(x)))^2) / sum(base(x)^2)) over all sampled activations",
            "minimum_inclusive": 0.0001,
            "maximum_exclusive": 0.1,
        },
        "preflight_policy": "all 12 runs start from their seed initializer; discard all weights",
        "routing_probe_policy": "C/D x 3 seeds start independently from original r21 initializers; discard all weights",
        "formal_policy": "start only after combined admission passes; each run starts from its original initializer",
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
        "gpu_policy": {
            "mode": "two independent one-GPU jobs; never DDP; request-visible device is logical 0",
            "seed_to_physical_gpu": {"260829": "0", "260830": "1", "260831": "0"},
            "within_seed_order": ["a", "b", "c", "d"],
            "routing_probe_within_seed_order": ["c", "d"],
            "gpu0_seed_order": [260829, 260831],
            "gpu1_seed_order": [260830],
        },
        "formal_admission": {
            "path": str(run_root / "audits/formal_admission.json"),
            "must_bind": [
                "protocol_and_implementation_hashes",
                "official_source_checkpoint",
                "all_three_initializer_manifests",
                "all_12_preflight_checkpoints",
                "all_6_routing_probe_checkpoints",
                "fixed_val512_hard_top2_gate_36_of_36",
                "residual_activity_gate_18_of_18",
                "a_b_invariance",
                "c_d_clean_aux_parity",
                "formal_request_initializer_lineage",
                "gpu_schedule",
                "formal_directory_absent",
            ],
        },
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
