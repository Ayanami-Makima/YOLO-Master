#!/usr/bin/env python3
"""Run a P1 training request while keeping every BatchNorm layer frozen."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

import torch
from torch import nn

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

R19_ROUTING_SEMANTICS = "hard_top2_from_step_zero_with_private_first_epoch_exploration"
R19_EXPLORATION_POLICY = {
    "sigma_source": "train512_median_per_image_logit_std_clipped",
    "sigma_min": 0.01,
    "sigma_max": 0.05,
    "hold_through_microbatch": 625,
    "decay_to_zero_microbatch": 1000,
    "private_seed_stride": 10000,
    "evaluation_noise_std": 0.0,
}
R19_GAIN_POLICY = {"lr": 0.01, "weight_decay": 0.0, "warmup": False}


def read_request(path: Path) -> dict:
    """Read and minimally validate a generated yolo.train request."""
    request = json.loads(path.read_text(encoding="utf-8"))
    if request.get("skill") != "yolo.train":
        raise ValueError("request must target yolo.train")
    if request.get("policy", {}).get("dry_run"):
        raise ValueError("request is marked dry-run")
    inputs, params = request.get("inputs", {}), request.get("params", {})
    for key in ("model", "data"):
        if not Path(inputs.get(key, "")).is_file():
            raise ValueError(f"request input is missing: {key}")
    model_path = Path(inputs["model"])
    pretrained = params.get("pretrained")
    if model_path.suffix == ".pt" and pretrained is not True:
        raise ValueError("checkpoint continuation must set pretrained=true so Model.train() retains loaded weights")
    if model_path.suffix in {".yaml", ".yml"} and (not isinstance(pretrained, str) or not Path(pretrained).is_file()):
        raise ValueError("YAML training must provide an existing pretrained checkpoint path")
    policy = request.get("a1_policy", {})
    routing_semantics = policy.get("routing_semantics")
    if routing_semantics in {"deterministic_hard_top2_from_step_zero", R19_ROUTING_SEMANTICS}:
        drift = {key: (params.get(key), value) for key, value in P1_ROUTING_PARAMS.items() if params.get(key) != value}
        if drift:
            raise ValueError(f"P1 deterministic routing policy drift: {drift}")
        if policy.get("expert_dropout_rate") != 0.0:
            raise ValueError("P1 hard Top-2 routing requires expert_dropout_rate=0.0")
    if routing_semantics == R19_ROUTING_SEMANTICS:
        exploration = policy.get("router_exploration")
        if not isinstance(exploration, dict):
            raise ValueError("r19 requires an explicit router_exploration policy")
        expected = {**R19_EXPLORATION_POLICY, "base_seed": params.get("seed")}
        drift = {key: (exploration.get(key), value) for key, value in expected.items() if exploration.get(key) != value}
        if drift:
            raise ValueError(f"r19 router exploration policy drift: {drift}")
        if exploration.get("enabled") not in {True, False}:
            raise ValueError("r19 router_exploration.enabled must be boolean")
        gain_policy = policy.get("factor_gain_optimizer")
        if gain_policy != R19_GAIN_POLICY:
            raise ValueError(f"r19 factor gain optimizer policy drift: {gain_policy!r}")
    return request


def freeze_batch_norm(trainer) -> int:
    """Freeze BatchNorm affine parameters and running statistics in one trainer."""
    count = 0
    for module in trainer.model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad = False
            count += 1
    trainer.p1_frozen_bn_count = count
    return count


def freeze_residual_factor_bases(trainer) -> int:
    """Keep every function-preserving adapter's pretrained path immutable."""
    count = 0
    for module in trainer.model.modules():
        freeze = getattr(module, "freeze_base_parameters", None)
        if callable(freeze):
            count += int(freeze())
    trainer.p1_frozen_factor_base_parameters = count
    return count


def enforce_p1_freeze_policy(trainer) -> None:
    """Reapply all P1 freeze constraints after mode changes and per batch."""
    freeze_batch_norm(trainer)
    freeze_residual_factor_bases(trainer)


def _routed_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return routed MoE cores in stable model traversal order."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if hasattr(module, "progressive_sparsity") and hasattr(module, "routing")
    ]


def configure_r19_exploration(trainer, request: dict) -> None:
    """Install deterministic private RNG streams and calibrated per-router sigma values."""
    policy = request.get("a1_policy", {})
    trainer.p1_routing_semantics = policy.get("routing_semantics")
    trainer.p1_router_exploration = policy.get("router_exploration")
    trainer.p1_microbatch_index = 0
    if trainer.p1_routing_semantics != R19_ROUTING_SEMANTICS:
        return

    exploration = trainer.p1_router_exploration
    routed = _routed_modules(trainer.model)
    enabled = bool(exploration["enabled"])
    if enabled != bool(routed):
        raise ValueError(f"r19 exploration enabled={enabled} but routed module count is {len(routed)}")
    base_seed = int(exploration["base_seed"])
    stride = int(exploration["private_seed_stride"])
    sigma_min = float(exploration["sigma_min"])
    sigma_max = float(exploration["sigma_max"])
    for index, (_, module) in enumerate(routed):
        routing = module.routing
        sigma0 = float(routing.p1_noise_sigma0.item())
        if not sigma_min <= sigma0 <= sigma_max:
            raise ValueError(f"uncalibrated r19 router sigma0={sigma0} outside [{sigma_min}, {sigma_max}]")
        routing.configure_p1_private_noise(base_seed + stride * (index + 1), reset_step=True)
        routing.noise_std = sigma0


def r19_noise_scale(microbatch_index: int, exploration: dict) -> float:
    """Return the locked r19 hold/linear-decay multiplier for one microbatch."""
    step = int(microbatch_index)
    hold = int(exploration["hold_through_microbatch"])
    end = int(exploration["decay_to_zero_microbatch"])
    if step <= hold:
        return 1.0
    if step <= end:
        return max(0.0, (end - step) / (end - hold))
    return 0.0


def enforce_and_schedule_p1_policy(trainer) -> None:
    """Reapply freeze constraints and set the r19 noise for the next train microbatch."""
    enforce_p1_freeze_policy(trainer)
    if getattr(trainer, "p1_routing_semantics", None) != R19_ROUTING_SEMANTICS:
        return
    exploration = trainer.p1_router_exploration
    scale = r19_noise_scale(trainer.p1_microbatch_index, exploration)
    for _, module in _routed_modules(trainer.model):
        module.routing.noise_std = float(module.routing.p1_noise_sigma0.item()) * scale
    trainer.p1_current_noise_scale = scale
    trainer.p1_microbatch_index += 1


def routed_module_policy(model: nn.Module) -> list[dict]:
    """Describe the effective P1 policy of every routed factor module."""
    modules = []
    for name, module in _routed_modules(model):
        routing = module.routing
        expert_parameters = list(module.experts.parameters()) if hasattr(module, "experts") else []
        expert_bn_parameter_ids = {
            id(parameter)
            for child in module.experts.modules()
            if isinstance(child, nn.modules.batchnorm._BatchNorm)
            for parameter in child.parameters(recurse=False)
        }
        expert_bn_parameters = [
            parameter for parameter in expert_parameters if id(parameter) in expert_bn_parameter_ids
        ]
        expert_non_bn_parameters = [
            parameter for parameter in expert_parameters if id(parameter) not in expert_bn_parameter_ids
        ]
        modules.append(
            {
                "name": name,
                "num_experts": int(module.num_experts),
                "top_k": int(module.top_k),
                "current_top_k": int(module._current_top_k),
                "progressive_sparsity": bool(module.progressive_sparsity),
                "warmup_steps": int(module.warmup_steps),
                "noise_std": float(routing.noise_std),
                "p1_noise_sigma0": float(getattr(routing, "p1_noise_sigma0", torch.tensor(0.0)).item()),
                "p1_noise_seed": getattr(routing, "p1_noise_seed", None),
                "p1_noise_step": int(getattr(routing, "p1_noise_step", 0)),
                "expert_dropout_rate": float(module.expert_dropout_rate),
                "expert_parameters": sum(parameter.numel() for parameter in expert_parameters),
                "trainable_expert_parameters": sum(
                    parameter.numel() for parameter in expert_parameters if parameter.requires_grad
                ),
                "expert_batch_norm_parameters": sum(parameter.numel() for parameter in expert_bn_parameters),
                "trainable_expert_batch_norm_parameters": sum(
                    parameter.numel() for parameter in expert_bn_parameters if parameter.requires_grad
                ),
                "expert_non_batch_norm_parameters": sum(parameter.numel() for parameter in expert_non_bn_parameters),
                "trainable_expert_non_batch_norm_parameters": sum(
                    parameter.numel() for parameter in expert_non_bn_parameters if parameter.requires_grad
                ),
            }
        )
    return modules


def factor_adapter_policy(model: nn.Module) -> list[dict]:
    """Describe trainability of every residual factor, gain, base, and nested BatchNorm."""
    adapters = []
    for name, module in model.named_modules():
        if not (hasattr(module, "factor") and hasattr(module, "base") and hasattr(module, "gain")):
            continue
        factor_parameters = list(module.factor.parameters())
        factor_bn_parameter_ids = {
            id(parameter)
            for child in module.factor.modules()
            if isinstance(child, nn.modules.batchnorm._BatchNorm)
            for parameter in child.parameters(recurse=False)
        }
        factor_bn_parameters = [
            parameter for parameter in factor_parameters if id(parameter) in factor_bn_parameter_ids
        ]
        factor_non_bn_parameters = [
            parameter for parameter in factor_parameters if id(parameter) not in factor_bn_parameter_ids
        ]
        base_parameters = list(module.base.parameters())
        adapters.append(
            {
                "name": name,
                "factor_parameters": sum(parameter.numel() for parameter in factor_parameters),
                "factor_batch_norm_parameters": sum(parameter.numel() for parameter in factor_bn_parameters),
                "trainable_factor_batch_norm_parameters": sum(
                    parameter.numel() for parameter in factor_bn_parameters if parameter.requires_grad
                ),
                "factor_non_batch_norm_parameters": sum(parameter.numel() for parameter in factor_non_bn_parameters),
                "trainable_factor_non_batch_norm_parameters": sum(
                    parameter.numel() for parameter in factor_non_bn_parameters if parameter.requires_grad
                ),
                "base_parameters": sum(parameter.numel() for parameter in base_parameters),
                "trainable_base_parameters": sum(
                    parameter.numel() for parameter in base_parameters if parameter.requires_grad
                ),
                "gain_parameters": int(module.gain.numel()),
                "trainable_gain_parameters": int(module.gain.numel()) if module.gain.requires_grad else 0,
                "gain_lr_scale": float(getattr(module, "p1_gain_lr_scale", 1.0)),
                "gain_no_warmup": bool(getattr(module, "p1_gain_no_warmup", False)),
            }
        )
    return adapters


def optimizer_policy(trainer) -> list[dict]:
    """Return the effective learning-rate policy for every optimizer group."""
    groups = []
    for index, group in enumerate(trainer.optimizer.param_groups):
        groups.append(
            {
                "index": index,
                "param_group": group.get("param_group"),
                "parameters": sum(parameter.numel() for parameter in group["params"]),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in group["params"] if parameter.requires_grad
                ),
                "initial_lr": float(group.get("initial_lr", group["lr"])),
                "final_lr": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "p1_no_warmup": bool(group.get("p1_no_warmup", False)),
            }
        )
    return groups


def runtime_policy_payload(trainer, request_id: str | None) -> dict:
    """Build a JSON-safe snapshot of the effective training policy."""
    mixture_config = getattr(trainer, "mixture_config", None)
    if hasattr(mixture_config, "to_dict"):
        mixture_config = mixture_config.to_dict()
    elif not isinstance(mixture_config, (dict, list, str, int, float, bool, type(None))):
        mixture_config = str(mixture_config)
    return {
        "schema_version": 2,
        "request_id": request_id,
        "requested": {key: getattr(trainer.args, key, None) for key in P1_ROUTING_PARAMS},
        "mixture_config": mixture_config,
        "factor_adapters": factor_adapter_policy(trainer.model),
        "routed_modules": routed_module_policy(trainer.model),
        "optimizer_groups": optimizer_policy(trainer),
        "routing_semantics": getattr(trainer, "p1_routing_semantics", None),
        "router_exploration": getattr(trainer, "p1_router_exploration", None),
        "microbatch_index": int(getattr(trainer, "p1_microbatch_index", 0)),
        "current_noise_scale": float(getattr(trainer, "p1_current_noise_scale", 0.0)),
        "args_yaml": str(Path(trainer.save_dir) / "args.yaml"),
        "results_csv": str(trainer.csv),
    }


def validate_runtime_p1_policy(trainer) -> None:
    """Fail before training if effective MoE policy differs from the locked request."""
    if getattr(trainer.args, "moe_dynamic_schedule", None) != "none":
        raise ValueError("P1 requires moe_dynamic_schedule=none")
    for key, expected in P1_ROUTING_PARAMS.items():
        actual = getattr(trainer.args, key, None)
        if actual != expected:
            raise ValueError(f"effective {key}={actual!r}, expected {expected!r}")
    routed = routed_module_policy(trainer.model)
    adapters = factor_adapter_policy(trainer.model)
    if len(adapters) != 3:
        raise ValueError(f"expected three residual factor adapters, got {len(adapters)}")
    for adapter in adapters:
        if adapter["trainable_factor_batch_norm_parameters"] != 0:
            raise ValueError(f"factor BatchNorm parameters are not frozen: {adapter}")
        if adapter["factor_non_batch_norm_parameters"] != adapter["trainable_factor_non_batch_norm_parameters"]:
            raise ValueError(f"factor non-BatchNorm parameters are not trainable: {adapter}")
        if adapter["trainable_base_parameters"] != 0:
            raise ValueError(f"pretrained factor base parameters are trainable: {adapter}")
        if adapter["gain_parameters"] != adapter["trainable_gain_parameters"]:
            raise ValueError(f"residual gain parameters are not trainable: {adapter}")
    r19 = getattr(trainer, "p1_routing_semantics", None) == R19_ROUTING_SEMANTICS
    for module in routed:
        if (
            module["top_k"] != 2
            or module["current_top_k"] != 2
            or module["progressive_sparsity"]
            or module["warmup_steps"] != 0
            or module["expert_dropout_rate"] != 0.0
        ):
            raise ValueError(f"effective routed module policy mismatch: {module}")
        if r19:
            sigma_min = float(trainer.p1_router_exploration["sigma_min"])
            sigma_max = float(trainer.p1_router_exploration["sigma_max"])
            if not sigma_min <= module["p1_noise_sigma0"] <= sigma_max:
                raise ValueError(f"r19 router sigma calibration mismatch: {module}")
            if not math.isclose(module["noise_std"], module["p1_noise_sigma0"], abs_tol=1e-12):
                raise ValueError(f"r19 router did not start at calibrated sigma0: {module}")
            if module["p1_noise_seed"] is None or module["p1_noise_step"] != 0:
                raise ValueError(f"r19 private router RNG was not reset: {module}")
        elif module["noise_std"] != 0.0:
            raise ValueError(f"deterministic P1 router noise is not disabled: {module}")
        if module["trainable_expert_batch_norm_parameters"] != 0:
            raise ValueError(f"MoE expert BatchNorm parameters are not frozen: {module}")
        if module["expert_non_batch_norm_parameters"] != module["trainable_expert_non_batch_norm_parameters"]:
            raise ValueError(f"MoE expert non-BatchNorm parameters are not trainable from epoch zero: {module}")
    if routed:
        router_groups = [group for group in optimizer_policy(trainer) if group["param_group"] == "router"]
        if len(router_groups) != 1 or not math.isclose(router_groups[0]["initial_lr"], 0.0001, abs_tol=1e-12):
            raise ValueError(f"router optimizer group does not use lr0=0.0001: {router_groups}")
    if r19:
        gain_groups = [group for group in optimizer_policy(trainer) if group["param_group"] == "residual_gain"]
        expected_gain_parameters = sum(adapter["gain_parameters"] for adapter in adapters)
        if (
            len(gain_groups) != 1
            or gain_groups[0]["parameters"] != expected_gain_parameters
            or not math.isclose(gain_groups[0]["initial_lr"], R19_GAIN_POLICY["lr"], abs_tol=1e-12)
            or gain_groups[0]["weight_decay"] != R19_GAIN_POLICY["weight_decay"]
            or not gain_groups[0]["p1_no_warmup"]
        ):
            raise ValueError(f"r19 residual gain optimizer group mismatch: {gain_groups}")


def tensor_summary(value) -> dict | str:
    """Return a JSON-safe summary for one batch value."""
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "device": str(value.device),
            "requires_grad": value.requires_grad,
        }
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value), "sample": [str(item) for item in value[:8]]}
    return str(value)


def write_failure_report(model, request: dict, error: BaseException) -> Path:
    """Persist enough state to reproduce a late training failure."""
    params = request["params"]
    diagnostics = request.get("diagnostics", {})
    default_path = Path(params["project"]) / params["name"] / "failure_diagnostics.json"
    output = Path(diagnostics.get("failure_report", default_path)).resolve()
    trainer = getattr(model, "trainer", None)
    batch = getattr(trainer, "batch", {}) if trainer is not None else {}
    pretrain_policy = Path(trainer.save_dir) / "p1_runtime_policy_pretrain.json" if trainer is not None else None
    report = {
        "schema_version": 1,
        "request_id": request.get("request_id"),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "epoch_one_based": getattr(trainer, "epoch", -1) + 1 if trainer is not None else None,
        "runtime_policy_pretrain": str(pretrain_policy)
        if pretrain_policy is not None and pretrain_policy.is_file()
        else None,
        "batch": {key: tensor_summary(value) for key, value in batch.items()} if isinstance(batch, dict) else {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    request = read_request(args.request.resolve())
    params = dict(request["params"])
    report = {
        "schema_version": 1,
        "request": str(args.request.resolve()),
        "request_id": request.get("request_id"),
        "model": request["inputs"]["model"],
        "data": request["inputs"]["data"],
        "freeze": params.get("freeze"),
        "bn_policy": "all BatchNorm affine parameters and running statistics frozen",
        "factor_base_policy": "pretrained bases inside residual factor adapters frozen",
        "detect_anomaly": bool(request.get("diagnostics", {}).get("detect_anomaly", False)),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return

    from ultralytics import YOLO

    torch.autograd.set_detect_anomaly(report["detect_anomaly"], check_nan=True)
    model = YOLO(request["inputs"]["model"], task=request["inputs"].get("task", "detect"))

    def prepare_training(trainer) -> None:
        configure_r19_exploration(trainer, request)
        enforce_p1_freeze_policy(trainer)
        pretrain_policy_path = Path(trainer.save_dir) / "p1_runtime_policy_pretrain.json"
        pretrain_policy_path.write_text(
            json.dumps(runtime_policy_payload(trainer, request.get("request_id")), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        validate_runtime_p1_policy(trainer)

    model.add_callback("on_pretrain_routine_end", prepare_training)
    model.add_callback("on_train_batch_start", enforce_and_schedule_p1_policy)
    try:
        results = model.train(data=request["inputs"]["data"], **params)
    except BaseException as error:
        failure_report = write_failure_report(model, request, error)
        print(json.dumps({**report, "status": "failed", "failure_report": str(failure_report)}, indent=2))
        raise
    trainer = model.trainer
    runtime_policy_path = Path(trainer.save_dir) / "p1_runtime_policy.json"
    runtime_policy = runtime_policy_payload(trainer, request.get("request_id"))
    runtime_policy_path.write_text(json.dumps(runtime_policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report.update(
        {
            "status": "ok",
            "save_dir": str(trainer.save_dir),
            "frozen_bn_count": getattr(trainer, "p1_frozen_bn_count", None),
            "frozen_factor_base_parameters": getattr(trainer, "p1_frozen_factor_base_parameters", None),
            "runtime_policy": str(runtime_policy_path),
            "metrics": dict(getattr(results, "results_dict", {}) or {}),
            "results_csv": str(trainer.csv),
            "best": str(trainer.best),
            "last": str(trainer.last),
        }
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
