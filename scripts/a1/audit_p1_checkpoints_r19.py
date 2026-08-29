#!/usr/bin/env python3
"""Audit every seed/cell r19 checkpoint against its original initializer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
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
from run_p1_bn_frozen import R19_GAIN_POLICY, R19_ROUTING_SEMANTICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=sorted(CELLS_BY_STAGE))
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def runtime_policy(run_dir: Path, *, routed_cell: bool) -> dict:
    """Verify the persisted r19 freeze, optimizer, and routing policy."""
    path = run_dir / "p1_runtime_policy.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    reasons = []
    if payload.get("routing_semantics") != R19_ROUTING_SEMANTICS:
        reasons.append("routing semantics mismatch")
    adapters = payload.get("factor_adapters", [])
    if len(adapters) != 3:
        reasons.append(f"expected three factor adapters, got {len(adapters)}")
    for adapter in adapters:
        if adapter.get("trainable_base_parameters") != 0:
            reasons.append(f"trainable base: {adapter.get('name')}")
        if adapter.get("trainable_factor_batch_norm_parameters") != 0:
            reasons.append(f"trainable factor BN: {adapter.get('name')}")
        if adapter.get("gain_parameters") != adapter.get("trainable_gain_parameters"):
            reasons.append(f"frozen gain: {adapter.get('name')}")
    routed = payload.get("routed_modules", [])
    if len(routed) != (6 if routed_cell else 0):
        reasons.append(f"routed module count {len(routed)}")
    for module in routed:
        if (
            module.get("top_k") != 2
            or module.get("current_top_k") != 2
            or module.get("progressive_sparsity") is not False
            or module.get("warmup_steps") != 0
            or module.get("expert_dropout_rate") != 0.0
            or not 0.01 <= float(module.get("p1_noise_sigma0", -1)) <= 0.05
        ):
            reasons.append(f"routed policy mismatch: {module.get('name')}")
    gain_groups = [
        group for group in payload.get("optimizer_groups", []) if group.get("param_group") == "residual_gain"
    ]
    if (
        len(gain_groups) != 1
        or not math.isclose(float(gain_groups[0].get("initial_lr", -1)), R19_GAIN_POLICY["lr"], abs_tol=1e-12)
        or float(gain_groups[0].get("weight_decay", -1)) != 0.0
        or gain_groups[0].get("p1_no_warmup") is not True
    ):
        reasons.append(f"gain optimizer policy mismatch: {gain_groups}")
    return {"path": str(path), "payload": payload, "reasons": reasons, "passed": path.is_file() and not reasons}


def audit_cell(protocol: dict, stage: str, seed: int, cell: str) -> dict:
    """Audit one exact last.pt without suffix or best-checkpoint discovery."""
    from ultralytics.nn.tasks import load_checkpoint

    request_path = Path(protocol["requests"][stage][str(seed)][cell]["path"])
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
        "initializer": str(initializer_path),
        "checkpoint": str(checkpoint_path),
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
    policy = runtime_policy(run_dir, routed_cell=cell in "cd")
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
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint audit: {output}")
    report = {"schema_version": 5, "protocol": str(protocol_path), "stage": args.stage, "seeds": {}}
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
