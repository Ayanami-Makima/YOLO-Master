"""Run one short P2 late-layer efficiency-screen pilot with frozen BN/base."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_p1_bn_frozen import (
    P1_ROUTING_PARAMS,
    enforce_p1_freeze_policy,
    read_request,
    runtime_policy_payload,
    write_failure_report,
)
from torch import nn


def routed_modules(model: nn.Module):
    return [
        (name, module)
        for name, module in model.named_modules()
        if hasattr(module, "progressive_sparsity") and hasattr(module, "routing")
    ]


def factor_adapters(model: nn.Module):
    return [
        (name, module)
        for name, module in model.named_modules()
        if hasattr(module, "factor") and hasattr(module, "base") and hasattr(module, "gain")
    ]


def validate_screen_policy(trainer, request: dict) -> None:
    params = request["params"]
    for key, expected in P1_ROUTING_PARAMS.items():
        actual = getattr(trainer.args, key, None)
        if actual != expected:
            raise ValueError(f"effective {key}={actual!r}, expected {expected!r}")
    adapters = factor_adapters(trainer.model)
    if len(adapters) != 3:
        raise ValueError(f"expected three residual factor adapters, got {len(adapters)}")
    for name, adapter in adapters:
        base = list(adapter.base.parameters())
        if any(parameter.requires_grad for parameter in base):
            raise ValueError(f"residual base is trainable: {name}")
        for module in adapter.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and (
                module.training or any(parameter.requires_grad for parameter in module.parameters(recurse=False))
            ):
                raise ValueError(f"BatchNorm is not frozen: {name}")
    policy = request.get("a1_policy", {})
    expected_moe_layer = policy.get("moe_layer")
    expected_top_k = int(policy.get("top_k") or 0)
    expected_experts = int(policy.get("num_experts") or 0)
    routed = routed_modules(trainer.model)
    if bool(expected_moe_layer) != bool(routed):
        raise ValueError(f"expected routed modules={bool(expected_moe_layer)}, got {len(routed)}")
    for name, module in routed:
        if expected_moe_layer is not None and f"model.{expected_moe_layer}" not in name:
            raise ValueError(f"unexpected routed module {name}")
        if int(module.top_k) != expected_top_k or int(module._current_top_k) != expected_top_k:
            raise ValueError(f"top-k drift in {name}: {module.top_k}/{module._current_top_k}")
        if int(module.num_experts) != expected_experts:
            raise ValueError(f"expert-count drift in {name}: {module.num_experts}")
        if module.progressive_sparsity or int(module.warmup_steps) != 0:
            raise ValueError(f"progressive routing is enabled in {name}")
        if float(module.routing.noise_std) != 0.0:
            raise ValueError(f"router noise is enabled in {name}")
    if params.get("freeze") != [index for index in range(24) if index not in {4, 6, 8, 23}]:
        raise ValueError(f"freeze list drift: {params.get('freeze')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    request = read_request(args.request.resolve())
    params = dict(request["params"])
    if args.resume_from is not None:
        resume_checkpoint = args.resume_from.resolve()
        if not resume_checkpoint.is_file() or resume_checkpoint.stat().st_size == 0:
            raise FileNotFoundError(resume_checkpoint)
        request["inputs"]["model"] = str(resume_checkpoint)
        params["resume"] = True
        params["exist_ok"] = True
    report = {
        "schema": "a1-p2-late-moe-efficiency-screen-pilot/v1",
        "request": str(args.request.resolve()),
        "request_id": request.get("request_id"),
        "model": request["inputs"]["model"],
        "data": request["inputs"]["data"],
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    torch.autograd.set_detect_anomaly(bool(request.get("diagnostics", {}).get("detect_anomaly", False)), check_nan=True)
    from ultralytics import YOLO

    model = YOLO(request["inputs"]["model"], task=request["inputs"].get("task", "detect"))

    def prepare_training(trainer) -> None:
        enforce_p1_freeze_policy(trainer)
        trainer.p1_routing_semantics = request.get("a1_policy", {}).get("routing_semantics")
        trainer.p1_router_exploration = None
        trainer.p1_microbatch_index = 0
        path = Path(trainer.save_dir) / "p2_efficiency_screen_runtime_policy_pretrain.json"
        path.write_text(json.dumps(runtime_policy_payload(trainer, request.get("request_id")), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        validate_screen_policy(trainer, request)

    def enforce(trainer) -> None:
        enforce_p1_freeze_policy(trainer)

    model.add_callback("on_pretrain_routine_end", prepare_training)
    model.add_callback("on_train_batch_start", enforce)
    try:
        results = model.train(data=request["inputs"]["data"], **params)
    except BaseException as error:
        failure = write_failure_report(model, request, error)
        print(json.dumps({**report, "status": "failed", "failure_report": str(failure), "traceback": traceback.format_exc()}, indent=2, ensure_ascii=False))
        raise
    trainer = model.trainer
    runtime_path = Path(trainer.save_dir) / "p2_efficiency_screen_runtime_policy.json"
    runtime_path.write_text(json.dumps(runtime_policy_payload(trainer, request.get("request_id")), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report.update({
        "status": "ok",
        "save_dir": str(trainer.save_dir),
        "frozen_bn_count": getattr(trainer, "p1_frozen_bn_count", None),
        "frozen_factor_base_parameters": getattr(trainer, "p1_frozen_factor_base_parameters", None),
        "runtime_policy": str(runtime_path),
        "metrics": dict(getattr(results, "results_dict", {}) or {}),
        "results_csv": str(trainer.csv),
        "best": str(trainer.best),
        "last": str(trainer.last),
    })
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
