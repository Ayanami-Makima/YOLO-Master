#!/usr/bin/env python3
"""Audit P1 residual-factor checkpoints against their locked initializers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import torch
from torch import nn

CELLS_BY_STAGE = {"preflight": "abcd", "routing_probe": "cd", "formal": "abcd"}
FACTOR_LAYERS = (4, 6, 8)
FORBIDDEN_LOG_PATTERNS = {
    "traceback": re.compile(r"traceback", re.IGNORECASE),
    "oom": re.compile(r"cuda out of memory|outofmemoryerror", re.IGNORECASE),
    "nan": re.compile(r"(^|[^a-z])nan([^a-z]|$)", re.IGNORECASE | re.MULTILINE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=sorted(CELLS_BY_STAGE))
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def tensor_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Copy a module state to CPU for exact comparison."""
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def compare_states(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict:
    """Compare two tensor mappings exactly and report the largest numeric error."""
    left_keys, right_keys = set(left), set(right)
    changed = []
    maximum = 0.0
    shape_mismatches = []
    for name in sorted(left_keys & right_keys):
        before, after = left[name], right[name]
        if before.shape != after.shape:
            shape_mismatches.append(name)
            continue
        if not torch.equal(before, after):
            changed.append(name)
            difference = (before.float() - after.float()).abs()
            maximum = max(maximum, float(difference.max()) if difference.numel() else 0.0)
    return {
        "left_only": sorted(left_keys - right_keys),
        "right_only": sorted(right_keys - left_keys),
        "shape_mismatches": shape_mismatches,
        "changed_tensors": len(changed),
        "changed_names": changed,
        "max_abs_error": maximum,
        "passed": not (left_keys ^ right_keys) and not shape_mismatches and not changed,
    }


def batch_norm_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect every BatchNorm parameter and running-statistic tensor."""
    state = {}
    for module_name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for tensor_name, value in module.state_dict().items():
                state[f"{module_name}.{tensor_name}"] = value.detach().cpu()
    return state


def router_report(layer: nn.Module) -> list[dict]:
    """Describe every OptimizedMOEImproved-style router in a factor layer."""
    routers = []
    for name, module in layer.factor.named_modules():
        if hasattr(module, "progressive_sparsity") and hasattr(module, "_current_top_k"):
            routers.append(
                {
                    "name": name,
                    "num_experts": int(module.num_experts),
                    "top_k": int(module.top_k),
                    "current_top_k": int(module._current_top_k),
                    "progressive_sparsity": bool(module.progressive_sparsity),
                    "warmup_steps": int(module.warmup_steps),
                    "noise_std": float(module.routing.noise_std),
                    "expert_dropout_rate": float(module.expert_dropout_rate),
                }
            )
    return routers


def changed_subset(before: nn.Module, after: nn.Module, contains: str) -> dict:
    """Compare only state tensors whose names contain a marker."""
    left = {name: value for name, value in tensor_state(before).items() if contains in name}
    right = {name: value for name, value in tensor_state(after).items() if contains in name}
    result = compare_states(left, right)
    result["tensors"] = len(left)
    result["updated"] = result["changed_tensors"] > 0
    return result


def read_metrics(run_dir: Path) -> dict:
    """Validate that the results CSV has finite loss and accuracy values."""
    results = run_dir / "results.csv"
    rows = list(csv.DictReader(results.open(encoding="utf-8"))) if results.is_file() else []
    required = ("train/box_loss", "train/cls_loss", "train/dfl_loss", "metrics/mAP50-95(B)")
    finite = {}
    last = rows[-1] if rows else {}
    for field in required:
        try:
            finite[field] = math.isfinite(float(last[field]))
        except (KeyError, TypeError, ValueError):
            finite[field] = False
    return {
        "path": str(results),
        "rows": len(rows),
        "last": last,
        "finite": finite,
        "passed": bool(rows) and all(finite.values()),
    }


def audit_log(path: Path) -> dict:
    """Check the driver log for fatal tokens and the frozen-base count."""
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    forbidden = [name for name, pattern in FORBIDDEN_LOG_PATTERNS.items() if pattern.search(text)]
    frozen_count_recorded = re.search(r'"frozen_factor_base_parameters"\s*:\s*459232', text) is not None
    return {
        "path": str(path),
        "exists": path.is_file(),
        "forbidden_tokens": forbidden,
        "frozen_factor_base_parameters_459232": frozen_count_recorded,
        "passed": path.is_file() and not forbidden and frozen_count_recorded,
    }


def audit_runtime_policy(run_dir: Path, routed_cell: bool) -> dict:
    """Verify the effective P1 routing and optimizer policy recorded by the runner."""
    path = run_dir / "p1_runtime_policy.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    expected = {
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
    reasons = []
    if payload.get("requested") != expected:
        reasons.append(f"requested routing policy mismatch: {payload.get('requested')}")
    routed_modules = payload.get("routed_modules", [])
    factor_adapters = payload.get("factor_adapters", [])
    if len(factor_adapters) != 3:
        reasons.append(f"expected three factor adapters, got {len(factor_adapters)}")
    for adapter in factor_adapters:
        if adapter.get("trainable_factor_batch_norm_parameters") != 0:
            reasons.append(f"factor BatchNorm parameters were trainable: {adapter.get('name')}")
        if adapter.get("factor_non_batch_norm_parameters") != adapter.get("trainable_factor_non_batch_norm_parameters"):
            reasons.append(f"factor non-BatchNorm parameters were frozen: {adapter.get('name')}")
        if adapter.get("trainable_base_parameters") != 0:
            reasons.append(f"pretrained factor base parameters were trainable: {adapter.get('name')}")
        if adapter.get("gain_parameters") != adapter.get("trainable_gain_parameters"):
            reasons.append(f"residual gain parameters were frozen: {adapter.get('name')}")
    if routed_cell and len(routed_modules) != 6:
        reasons.append(f"expected six routed modules, got {len(routed_modules)}")
    if not routed_cell and routed_modules:
        reasons.append("dense cell unexpectedly recorded routed modules")
    for module in routed_modules:
        if (
            module.get("top_k") != 2
            or module.get("current_top_k") != 2
            or module.get("progressive_sparsity") is not False
            or module.get("warmup_steps") != 0
            or module.get("noise_std") != 0.0
            or module.get("expert_dropout_rate") != 0.0
        ):
            reasons.append(f"effective routed module policy mismatch: {module}")
        if module.get("trainable_expert_batch_norm_parameters") != 0:
            reasons.append(f"expert BatchNorm parameters were trainable: {module.get('name')}")
        if module.get("expert_non_batch_norm_parameters") != module.get("trainable_expert_non_batch_norm_parameters"):
            reasons.append(f"expert non-BatchNorm parameters were frozen: {module.get('name')}")
    router_groups = [group for group in payload.get("optimizer_groups", []) if group.get("param_group") == "router"]
    if routed_cell and (
        len(router_groups) != 1
        or not math.isclose(float(router_groups[0].get("initial_lr", -1.0)), 0.0001, abs_tol=1e-12)
    ):
        reasons.append(f"router optimizer group does not use lr0=0.0001: {router_groups}")
    return {"path": str(path), "payload": payload, "reasons": reasons, "passed": path.is_file() and not reasons}


def audit_cell(protocol: dict, stage: str, cell: str) -> dict:
    """Audit one trained cell against its original initializer."""
    from ultralytics.nn.tasks import load_checkpoint

    request_path = Path(protocol["requests"][stage][cell]["path"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    initializer_path = Path(request["inputs"]["model"])
    run_dir = Path(request["params"]["project"]) / request["params"]["name"]
    checkpoint_path = run_dir / "weights" / protocol.get("primary_checkpoint", "last.pt")
    initializer, _ = load_checkpoint(initializer_path, device="cpu")
    trained, _ = load_checkpoint(checkpoint_path, device="cpu")
    initializer, trained = initializer.float(), trained.float()

    expected_end2end = cell in "bd"
    cell_report = {
        "request": str(request_path),
        "initializer": str(initializer_path),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "end2end": {
            "expected": expected_end2end,
            "model": bool(trained.end2end),
            "head": bool(trained.model[23].end2end),
        },
        "layers": {},
    }
    reasons = []
    if cell_report["end2end"]["model"] != expected_end2end or cell_report["end2end"]["head"] != expected_end2end:
        reasons.append("end2end path mismatch")

    frozen_base_total = 0
    for index in FACTOR_LAYERS:
        before, after = initializer.model[index], trained.model[index]
        base = compare_states(tensor_state(before.base), tensor_state(after.base))
        gain_nonzero = int(torch.count_nonzero(after.gain.detach()).item())
        frozen_base_total += sum(parameter.numel() for parameter in after.base.parameters())
        routers = router_report(after)
        routing_update = changed_subset(before.factor, after.factor, ".routing.")
        factor_update = compare_states(tensor_state(before.factor), tensor_state(after.factor))
        factor_update["updated"] = factor_update["changed_tensors"] > 0
        expert_update = changed_subset(before.factor, after.factor, "experts.")
        layer_report = {
            "class": type(after).__name__,
            "base_class": type(after.base).__name__,
            "factor_class": type(after.factor).__name__,
            "base": base,
            "gain_nonzero": gain_nonzero,
            "gain_max_abs": float(after.gain.detach().abs().max()),
            "routers": routers,
            "routing_update": routing_update,
            "factor_update": factor_update,
            "expert_update": expert_update,
        }
        cell_report["layers"][str(index)] = layer_report
        if type(after).__name__ != "C3k2ResidualFactor" or type(after.base).__name__ != "C3k2":
            reasons.append(f"layer {index}: residual/base class mismatch")
        if not base["passed"]:
            reasons.append(f"layer {index}: pretrained base changed")
        if gain_nonzero == 0:
            reasons.append(f"layer {index}: gain stayed zero")
        if cell in "cd":
            if len(routers) != 2:
                reasons.append(f"layer {index}: expected two routers, got {len(routers)}")
            for router in routers:
                if (
                    router["progressive_sparsity"]
                    or router["top_k"] != 2
                    or router["current_top_k"] != 2
                    or router["warmup_steps"] != 0
                    or router["noise_std"] != 0.0
                    or router["expert_dropout_rate"] != 0.0
                ):
                    reasons.append(f"layer {index}: deterministic hard Top-2 policy mismatch")
            if not routing_update["updated"]:
                reasons.append(f"layer {index}: routing parameters did not update")
        elif routers:
            reasons.append(f"layer {index}: dense cell unexpectedly contains routers")

    cell_report["frozen_base_parameters"] = frozen_base_total
    if frozen_base_total != 459232:
        reasons.append(f"frozen base parameter count {frozen_base_total} != 459232")

    frozen_layers = {}
    for index in protocol["freeze"]:
        comparison = compare_states(tensor_state(initializer.model[index]), tensor_state(trained.model[index]))
        frozen_layers[str(index)] = comparison
        if not comparison["passed"]:
            reasons.append(f"frozen model layer {index} changed")
    cell_report["frozen_layers"] = frozen_layers

    batch_norm = compare_states(batch_norm_state(initializer), batch_norm_state(trained))
    cell_report["batch_norm"] = batch_norm
    if not batch_norm["passed"]:
        reasons.append("BatchNorm parameters or running statistics changed")

    metrics = read_metrics(run_dir)
    cell_report["metrics"] = metrics
    if not metrics["passed"]:
        reasons.append("missing or non-finite metrics")

    log = audit_log(Path(protocol["run_root"]) / stage / f"{cell}_driver.log")
    cell_report["driver_log"] = log
    if not log["passed"]:
        reasons.append("driver log failed integrity checks")

    runtime_policy = audit_runtime_policy(run_dir, routed_cell=cell in "cd")
    cell_report["runtime_policy"] = runtime_policy
    if not runtime_policy["passed"]:
        reasons.extend(f"runtime policy: {reason}" for reason in runtime_policy["reasons"])

    cell_report["reasons"] = reasons
    cell_report["passed"] = not reasons
    return cell_report


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "protocol": str(protocol_path),
        "stage": args.stage,
        "cells": {},
    }
    for cell in CELLS_BY_STAGE[args.stage]:
        report["cells"][cell] = audit_cell(protocol, args.stage, cell)
    report["status"] = "passed" if all(item["passed"] for item in report["cells"].values()) else "failed"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
