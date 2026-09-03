#!/usr/bin/env python3
"""Generate a function-preserving residual-factor P1 protocol."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml
from prepare_p1_factorial import CELLS, FREEZE, LOCKED_SHA, SEED, TRAIN_LAYERS, sha256, training_request, write_json

BALANCED_ROUTER_SCHEME = "deterministic_data_independent_regular_simplex_final_projection"

P1_MOE_TRAINING_ARGS = {
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
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("configs/a1/p1_factorial_r11"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pilot-data", required=True, type=Path)
    parser.add_argument("--preflight-data", required=True, type=Path)
    parser.add_argument("--experiment-tag", default="r11")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output = (repo / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
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

    suffix = "residual_factor_init"
    parent_path = repo / "ultralytics/cfg/models/26/yolo26.yaml"
    implementation_paths = (
        repo / "scripts/a1/prepare_p1_factorial.py",
        repo / "scripts/a1/prepare_p1_residual_factor.py",
        repo / "ultralytics/nn/modules/moe/factor_adapter.py",
        repo / "ultralytics/nn/modules/moe/modules.py",
        repo / "ultralytics/nn/modules/moe/experts.py",
        repo / "ultralytics/nn/modules/moe/routers.py",
        repo / "ultralytics/nn/modules/moe/loss.py",
        repo / "ultralytics/nn/modules/moe/__init__.py",
        repo / "ultralytics/nn/modules/__init__.py",
        repo / "ultralytics/nn/tasks.py",
        repo / "scripts/a1/build_p1_residual_factor_initializers.py",
        repo / "scripts/a1/run_p1_bn_frozen.py",
        repo / "scripts/a1/run_p1_factorial_preflights.py",
        repo / "scripts/a1/run_p1_factorial_routing_probes_2gpu.py",
        repo / "scripts/a1/run_p1_factorial_formal.py",
        repo / "scripts/a1/audit_p1_residual_factor_checkpoints.py",
        repo / "scripts/a1/audit_p1_preflight_routing.py",
        repo / "scripts/a1/audit_p1_routing.py",
        repo / "tests/test_p1_residual_factor.py",
        repo / "scripts/a1/tests/test_audit_p1_preflight_routing.py",
    )
    for path in (parent_path, *implementation_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    requests = {"preflight": {}, "routing_probe": {}, "formal": {}}
    architectures = {}
    for cell, factors in CELLS.items():
        config = copy.deepcopy(parent)
        config["scale"] = "n"
        config["end2end"] = factors["end2end"]
        experts_by_layer = {4: 4, 6: 8, 8: 16}
        for layer_index, num_experts in experts_by_layer.items():
            layer = config["backbone"][layer_index]
            if layer[2] != "C3k2":
                raise ValueError(f"layer {layer_index}: expected C3k2, got {layer[2]}")
            c2, c3k = layer[3][:2]
            expansion = layer[3][2] if len(layer[3]) > 2 else 0.5
            layer[2] = "C3k2ResidualFactor"
            layer[3] = [c2, c3k, expansion, factors["moe"], num_experts, 2, "dense_mlp"]
        config_path = output / f"{cell}_matched.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        architectures[cell] = {"path": str(config_path), "sha256": sha256(config_path), **factors}
        initializer = run_root / "initializers" / f"{cell}_{suffix}.pt"
        for stage, data, epochs in (("preflight", preflight_data, 1), ("formal", pilot_data, 5)):
            request = training_request(
                cell=cell,
                stage=stage,
                initializer=initializer,
                data=data,
                project=run_root / stage,
                epochs=epochs,
                device="0",
            )
            request["params"].update(P1_MOE_TRAINING_ARGS)
            request["a1_policy"] = {
                "freeze_batch_norm": True,
                "freeze_residual_factor_bases": True,
                "formal_restart_from_initializer": stage == "formal",
                "routing_semantics": "deterministic_hard_top2_from_step_zero",
                "expert_dropout_rate": 0.0,
            }
            request_path = output / f"{cell}_{stage}_request.json"
            write_json(request_path, request)
            requests[stage][cell] = {"path": str(request_path), "sha256": sha256(request_path)}
        if cell in "cd":
            request = training_request(
                cell=cell,
                stage="routing_probe",
                initializer=initializer,
                data=pilot_data,
                project=run_root / "routing_probe",
                epochs=1,
                device="0",
            )
            request["params"].update(P1_MOE_TRAINING_ARGS)
            request["a1_policy"] = {
                "freeze_batch_norm": True,
                "freeze_residual_factor_bases": True,
                "routing_semantics": "deterministic_hard_top2_from_step_zero",
                "expert_dropout_rate": 0.0,
            }
            request_path = output / f"{cell}_routing_probe_request.json"
            write_json(request_path, request)
            requests["routing_probe"][cell] = {"path": str(request_path), "sha256": sha256(request_path)}

    protocol = {
        "schema_version": 4,
        "name": f"A1 P1 equal-budget 2x2 factorial {args.experiment_tag} residual-factor",
        "experiment_tag": args.experiment_tag,
        "locked_sha": LOCKED_SHA,
        "source_checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "parent": {"path": str(parent_path), "sha256": sha256(parent_path)},
        "implementation": {
            str(path.relative_to(repo)).replace("\\", "/"): sha256(path) for path in implementation_paths
        },
        "data": {
            "preflight": {"path": str(preflight_data), "sha256": sha256(preflight_data)},
            "pilot": {"path": str(pilot_data), "sha256": sha256(pilot_data)},
        },
        "configs": architectures,
        "requests": requests,
        "run_root": str(run_root),
        "train_layers": sorted(TRAIN_LAYERS),
        "freeze": FREEZE,
        "factor_base": "native P0 C3k2 frozen inside ResidualFactorAdapter",
        "batch_norm": "all affine parameters and running statistics frozen in every cell",
        "preflight_policy": "one epoch from initializer; discard weights and restart formal runs",
        "routing_probe_policy": "C/D only, one full pilot epoch from initializer; discard weights after route audit",
        "routing_schedule": "hard Top-2 from the first training step; no progressive sparsity warmup",
        "single_behavior_change": {
            "baseline": "r16 iid normal final router projection initialization with standard deviation 0.05",
            "current": "C/D final router projections only use deterministic data-independent regular-simplex directions",
            "unchanged": "A/B initialization and all architecture, data, optimization, routing, freeze, and budget fields",
        },
        "router_initialization": {
            "scope": "C/D residual-factor routers only",
            "scheme": BALANCED_ROUTER_SCHEME,
            "base_seed": SEED,
            "per_router_seed_formula": "base_seed + factor_layer_index * 1000 + router_index",
            "target_entry_rms": 0.05,
            "equal_expert_row_norms": True,
            "expert_common_direction_removed": True,
            "c_d_byte_identical": True,
            "checkpoint_roundtrip": "verify exact expected FP16 serialization and C/D parity after reload",
            "data_source": "none",
            "validation_data_calibration": False,
        },
        "moe_training_policy": {
            **P1_MOE_TRAINING_ARGS,
            "expert_dropout_rate": 0.0,
            "training_audit_inference_routing_parity": True,
        },
        "routing_gate": {
            "sample_images": 512,
            "dead_experts": 0,
            "minimum_normalized_entropy": 0.5,
            "maximum_image_selection_fraction": 0.8,
            "image_selection_fraction_denominator": "number of audited images",
            "selection_share_denominator": "number of images multiplied by Top-K; diagnostic only",
        },
        "routing_gate_scope": "predeclared internal formal-admission validity gate; not an A1 taskbook numeric requirement",
        "formal_budget": "5000 images, batch 4, 5 epochs, identical seed and order",
        "initializer_suffix": suffix,
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
