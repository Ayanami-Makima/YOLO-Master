#!/usr/bin/env python3
"""Audit every seed/cell r23 checkpoint against its original initializer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from p1_r23_runtime import RUNTIME_ATTESTATION, assert_protocol_runtime

# isort: split

from audit_p1_residual_factor_checkpoints import (
    CELLS_BY_STAGE,
    FACTOR_LAYERS,
    audit_log,
    batch_norm_state,
    changed_subset,
    compare_states,
    read_metrics,
    router_report,
    tensor_state,
)
from run_p1_bn_frozen_r23 import (
    R23_CLEAN_AUX_POLICY,
    R23_DENSE_AUX_POLICY,
    R23_GAIN_POLICY,
    R23_ROUTING_SEMANTICS,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=sorted(CELLS_BY_STAGE))
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def runtime_policy(run_dir: Path, *, routed_cell: bool, stage: str, seed: int) -> dict:
    """Verify the persisted r23 freeze, optimizer, and routing policy."""
    path = run_dir / "p1_runtime_policy.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    reasons = []
    if payload.get("routing_semantics") != R23_ROUTING_SEMANTICS:
        reasons.append("routing semantics mismatch")
    expected_aux = R23_CLEAN_AUX_POLICY if routed_cell else R23_DENSE_AUX_POLICY
    if payload.get("routing_auxiliary_objective") != expected_aux:
        reasons.append("routing auxiliary objective mismatch")
    expected_exploration = {
        "sigma_source": "train512_median_per_image_logit_std_clipped",
        "sigma_min": 0.01,
        "sigma_max": 0.05,
        "hold_through_microbatch": 625,
        "decay_to_zero_microbatch": 1000,
        "private_seed_stride": 10000,
        "evaluation_noise_std": 0.0,
        "base_seed": seed,
        "enabled": routed_cell,
    }
    if payload.get("router_exploration") != expected_exploration:
        reasons.append("router exploration policy mismatch")
    expected_microbatches = {"preflight": 64, "routing_probe": 1250, "formal": 6250}[stage]
    expected_scale = 1.0 if stage == "preflight" else 0.0
    if payload.get("microbatch_index") != expected_microbatches:
        reasons.append(f"microbatch count {payload.get('microbatch_index')} != {expected_microbatches}")
    if not math.isclose(float(payload.get("current_noise_scale", -1)), expected_scale, abs_tol=1e-12):
        reasons.append("final private-noise scale mismatch")
    if payload.get("requested") != {
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
    }:
        reasons.append("effective requested mixture parameters drifted")
    moe_values = payload.get("mixture_config", {}).get("values", {}).get("moe", {})
    expected_moe_values = {
        "balance_loss_coeff": 1.0,
        "router_z_loss_coeff": 0.1,
        "noise_std": 0.0,
        "temperature": 1.0,
        "weight_threshold": 0.01,
        "aux_gain": 1.0,
        "aux_budget": 3.0,
    }
    if moe_values != expected_moe_values:
        reasons.append("effective MoE mixture configuration drifted")
    adapters = payload.get("factor_adapters", [])
    if len(adapters) != 3:
        reasons.append(f"expected three factor adapters, got {len(adapters)}")
    for adapter in adapters:
        if adapter.get("trainable_base_parameters") != 0:
            reasons.append(f"trainable base: {adapter.get('name')}")
        if adapter.get("trainable_factor_batch_norm_parameters") != 0:
            reasons.append(f"trainable factor BN: {adapter.get('name')}")
        if adapter.get("factor_non_batch_norm_parameters") != adapter.get(
            "trainable_factor_non_batch_norm_parameters"
        ):
            reasons.append(f"frozen factor non-BN parameter: {adapter.get('name')}")
        if adapter.get("gain_parameters") != adapter.get("trainable_gain_parameters"):
            reasons.append(f"frozen gain: {adapter.get('name')}")
    routed = payload.get("routed_modules", [])
    if len(routed) != (6 if routed_cell else 0):
        reasons.append(f"routed module count {len(routed)}")
    for index, module in enumerate(routed):
        if (
            module.get("top_k") != 2
            or module.get("current_top_k") != 2
            or module.get("progressive_sparsity") is not False
            or module.get("warmup_steps") != 0
            or module.get("expert_dropout_rate") != 0.0
            or not 0.01 <= float(module.get("p1_noise_sigma0", -1)) <= 0.05
        ):
            reasons.append(f"routed policy mismatch: {module.get('name')}")
        if module.get("p1_balance_on_clean_routes") is not True:
            reasons.append(f"router is not balancing clean routes: {module.get('name')}")
        if module.get("routing_aux_semantics") != R23_CLEAN_AUX_POLICY["runtime_semantics"]:
            reasons.append(f"routing-aux semantics mismatch: {module.get('name')}")
        expected_noise_step = min(expected_microbatches, 1000)
        if module.get("p1_noise_seed") != seed + 10000 * (index + 1):
            reasons.append(f"private router seed mismatch: {module.get('name')}")
        if module.get("p1_noise_step") != expected_noise_step:
            reasons.append(f"private router noise-step mismatch: {module.get('name')}")
        expected_final_noise = float(module.get("p1_noise_sigma0", -1)) * expected_scale
        if not math.isclose(float(module.get("noise_std", -1)), expected_final_noise, abs_tol=1e-12):
            reasons.append(f"final router noise mismatch: {module.get('name')}")
        if module.get("trainable_expert_batch_norm_parameters") != 0:
            reasons.append(f"trainable expert BN: {module.get('name')}")
        if module.get("expert_non_batch_norm_parameters") != module.get(
            "trainable_expert_non_batch_norm_parameters"
        ):
            reasons.append(f"frozen expert non-BN parameter: {module.get('name')}")
    gain_groups = [
        group for group in payload.get("optimizer_groups", []) if group.get("param_group") == "residual_gain"
    ]
    if (
        len(gain_groups) != 1
        or not math.isclose(float(gain_groups[0].get("initial_lr", -1)), R23_GAIN_POLICY["lr"], abs_tol=1e-12)
        or float(gain_groups[0].get("weight_decay", -1)) != 0.0
        or gain_groups[0].get("p1_no_warmup") is not True
    ):
        reasons.append(f"gain optimizer policy mismatch: {gain_groups}")
    router_groups = [group for group in payload.get("optimizer_groups", []) if group.get("param_group") == "router"]
    if len(router_groups) != 1 or not math.isclose(
        float(router_groups[0].get("initial_lr", -1)), 0.0001, abs_tol=1e-12
    ):
        reasons.append(f"router optimizer policy mismatch: {router_groups}")
    elif routed_cell and router_groups[0].get("trainable_parameters", 0) <= 0:
        reasons.append("routed cell has no trainable router parameters")
    elif not routed_cell and (
        router_groups[0].get("parameters") != 0 or router_groups[0].get("trainable_parameters") != 0
    ):
        reasons.append(f"dense cell router optimizer group is non-empty: {router_groups}")
    return {"path": str(path), "payload": payload, "reasons": reasons, "passed": path.is_file() and not reasons}


def audit_cell(protocol: dict, stage: str, seed: int, cell: str) -> dict:
    """Audit one exact last.pt without suffix or best-checkpoint discovery."""
    from ultralytics.nn.tasks import load_checkpoint

    request_entry = protocol["requests"][stage][str(seed)][cell]
    request_path = Path(request_entry["path"])
    request_sha = sha256(request_path)
    if request_sha != request_entry.get("sha256"):
        raise ValueError(f"request hash drift for {stage}/{seed}/{cell}: {request_path}")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    initializer_path = Path(request["inputs"]["model"])
    run_dir = Path(request["params"]["project"]) / request["params"]["name"]
    checkpoint_path = run_dir / "weights" / "last.pt"
    if not initializer_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing initializer/checkpoint for {stage}/{seed}/{cell}")
    initializer, _ = load_checkpoint(initializer_path, device="cpu")
    trained, _ = load_checkpoint(checkpoint_path, device="cpu")
    initializer, trained = initializer.float(), trained.float()
    reasons = []
    expected_end2end = cell in "bd"
    report = {
        "request": str(request_path),
        "request_sha256": request_sha,
        "initializer": str(initializer_path),
        "initializer_sha256": sha256(initializer_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "run_dir": str(run_dir),
        "end2end": {
            "expected": expected_end2end,
            "model": bool(trained.end2end),
            "head": bool(trained.model[23].end2end),
        },
        "layers": {},
    }
    if report["end2end"]["model"] != expected_end2end or report["end2end"]["head"] != expected_end2end:
        reasons.append("end2end path mismatch")
    frozen_base_total = 0
    for index in FACTOR_LAYERS:
        before, after = initializer.model[index], trained.model[index]
        base = compare_states(tensor_state(before.base), tensor_state(after.base))
        gain_nonzero = int(torch.count_nonzero(after.gain.detach()))
        frozen_base_total += sum(parameter.numel() for parameter in after.base.parameters())
        routers = router_report(after)
        routing_update = changed_subset(before.factor, after.factor, ".routing.")
        expert_update = changed_subset(before.factor, after.factor, "experts.")
        factor_update = compare_states(tensor_state(before.factor), tensor_state(after.factor))
        layer = {
            "class": type(after).__name__,
            "base_class": type(after.base).__name__,
            "base": base,
            "gain_nonzero": gain_nonzero,
            "gain_max_abs": float(after.gain.detach().abs().max()),
            "routers": routers,
            "routing_update": routing_update,
            "expert_update": expert_update,
            "factor_update": factor_update,
        }
        report["layers"][str(index)] = layer
        if type(after).__name__ != "C3k2ResidualFactor" or type(after.base).__name__ != "C3k2":
            reasons.append(f"layer {index}: adapter/base class mismatch")
        if not base["passed"]:
            reasons.append(f"layer {index}: frozen base changed")
        if gain_nonzero == 0:
            reasons.append(f"layer {index}: gain stayed zero")
        if cell in "cd":
            if len(routers) != 2:
                reasons.append(f"layer {index}: expected two routers")
            if not routing_update["updated"]:
                reasons.append(f"layer {index}: router did not update")
            if not expert_update["updated"]:
                reasons.append(f"layer {index}: experts did not update")
        elif routers:
            reasons.append(f"layer {index}: dense cell contains router")
    report["frozen_factor_base_parameters"] = frozen_base_total
    if frozen_base_total != protocol["factor_base_expected_parameters"]:
        reasons.append(f"frozen base parameter count {frozen_base_total}")
    frozen_layers = {}
    for index in protocol["freeze"]:
        comparison = compare_states(tensor_state(initializer.model[index]), tensor_state(trained.model[index]))
        frozen_layers[str(index)] = comparison
        if not comparison["passed"]:
            reasons.append(f"frozen model layer {index} changed")
    report["frozen_layers"] = frozen_layers
    bn = compare_states(batch_norm_state(initializer), batch_norm_state(trained))
    report["batch_norm"] = bn
    if not bn["passed"]:
        reasons.append("BatchNorm parameters or buffers changed")
    metrics = read_metrics(run_dir)
    report["metrics"] = metrics
    if not metrics["passed"]:
        reasons.append("metrics missing or non-finite")
    log = audit_log(Path(protocol["run_root"]) / stage / f"seed{seed}" / f"{cell}_driver.log")
    report["driver_log"] = log
    if not log["passed"]:
        reasons.append("driver log failed")
    policy = runtime_policy(run_dir, routed_cell=cell in "cd", stage=stage, seed=seed)
    report["runtime_policy"] = policy
    if not policy["passed"]:
        reasons.extend(f"runtime policy: {reason}" for reason in policy["reasons"])
    report["reasons"] = reasons
    report["passed"] = not reasons
    return report


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert_protocol_runtime(protocol)
    if protocol.get("schema_version") != 8 or protocol.get("experiment_tag") != "r23":
        raise ValueError("r23 checkpoint audit requires an r23 schema-8 protocol")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint audit: {output}")
    report = {
        "schema_version": 8,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "runtime_attestation": RUNTIME_ATTESTATION,
        "stage": args.stage,
        "seeds": {},
    }
    for seed in protocol["seeds"]:
        report["seeds"][str(seed)] = {
            cell: audit_cell(protocol, args.stage, seed, cell) for cell in CELLS_BY_STAGE[args.stage]
        }
    report["status"] = (
        "passed" if all(cell["passed"] for seed in report["seeds"].values() for cell in seed.values()) else "failed"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
