#!/usr/bin/env python3
"""Run one locked r26 request with frozen BN/base and clean-aux MoE routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import traceback
from pathlib import Path

import torch
from torch import nn

# isort: split
from p1_r26_runtime import RUNTIME_ATTESTATION, assert_protocol_runtime

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

R26_ROUTING_SEMANTICS = "hard_top2_from_step_zero_private_exploration_clean_aux"
R26_EXPLORATION_POLICY = {
    "sigma_source": "train512_median_per_image_logit_std_clipped",
    "sigma_min": 0.01,
    "sigma_max": 0.05,
    "hold_through_microbatch": 625,
    "decay_to_zero_microbatch": 1000,
    "private_seed_stride": 10000,
    "evaluation_noise_std": 0.0,
}
R26_GAIN_POLICY = {"lr": 0.01, "weight_decay": 0.0, "warmup": False}
R26_CLEAN_AUX_POLICY = {
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
R26_DENSE_AUX_POLICY = {
    "enabled": False,
    "reason": "dense cell has no router",
    "adds_parameters": False,
    "changes_inference": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_request_registration(path: Path, request: dict) -> dict:
    """Bind a request to the immutable protocol and enforce formal admission."""
    protocol_path = Path(request.get("protocol", {}).get("path", "")).resolve()
    if not protocol_path.is_file():
        raise ValueError(f"r26 request has no valid protocol registry: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert_protocol_runtime(protocol)
    if protocol.get("schema_version") != 8 or protocol.get("experiment_tag") != "r26":
        raise ValueError("request protocol is not r26 schema 8")
    repo = Path(request.get("runtime", {}).get("cwd", "")).resolve()
    expected_python = protocol["runtime_binding"]["interpreter"]["executable"]
    if request.get("runtime", {}).get("python") != expected_python:
        raise ValueError("request interpreter differs from the protocol runtime binding")
    if RUNTIME_ATTESTATION["interpreter"]["executable"] != expected_python:
        raise ValueError("executing interpreter differs from the protocol runtime binding")
    if not repo.is_dir():
        raise ValueError(f"request runtime repository is missing: {repo}")
    for relative, expected_sha in protocol.get("implementation", {}).items():
        implementation_path = repo / relative
        if not implementation_path.is_file() or sha256(implementation_path) != expected_sha:
            raise ValueError(f"implementation hash drift: {relative}")
    implementation_head = protocol.get("implementation_head")
    lineage = subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(implementation_head), "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if lineage.returncode != 0:
        raise ValueError("current HEAD does not descend from the protocol implementation commit")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise ValueError("r26 worktree must be clean before executing a request")
    relative_runner = "scripts/a1/run_p1_bn_frozen_r26.py"
    expected_runner_sha = protocol.get("implementation", {}).get(relative_runner)
    if expected_runner_sha != sha256(Path(__file__).resolve()):
        raise ValueError("executing runner differs from the implementation locked by the protocol")
    matches = []
    for stage, seeds in protocol.get("requests", {}).items():
        for seed, cells in seeds.items():
            for cell, entry in cells.items():
                if Path(entry.get("path", "")).resolve() == path:
                    matches.append((stage, seed, cell, entry))
    if len(matches) != 1:
        raise ValueError(f"request must occur exactly once in the protocol registry, found {len(matches)}")
    stage, seed, cell, entry = matches[0]
    request_sha = sha256(path)
    if entry.get("sha256") != request_sha:
        raise ValueError("request bytes differ from the protocol registry")
    formal_flag = request.get("a1_policy", {}).get("formal_restart_from_initializer")
    if formal_flag is not (stage == "formal"):
        raise ValueError(f"request formal flag disagrees with registered stage {stage}")
    data_label = {"preflight": "preflight", "routing_probe": "pilot", "formal": "formal"}[stage]
    data_registry = protocol.get("data", {}).get(data_label, {})
    data_path = Path(data_registry.get("path", "")).resolve()
    if data_path != Path(request.get("inputs", {}).get("data", "")).resolve():
        raise ValueError(f"request data path disagrees with {data_label} registry")
    if not data_path.is_file() or sha256(data_path) != data_registry.get("sha256"):
        raise ValueError(f"{data_label} data YAML hash drift")
    expected_counts = {
        "pilot": {"train": 5000, "val": 512},
        "preflight": {"train": 256, "val": 128},
        "formal": {"train": 20000, "val": 5000},
    }
    for split, expected_count in expected_counts[data_label].items():
        item = data_registry.get("lists", {}).get(split, {})
        list_path = Path(item.get("path", ""))
        if not list_path.is_file() or sha256(list_path) != item.get("sha256"):
            raise ValueError(f"{data_label}/{split} image-list hash drift")
        count = sum(bool(line.strip()) for line in list_path.read_text(encoding="utf-8").splitlines())
        if count != expected_count or item.get("images") != expected_count:
            raise ValueError(f"{data_label}/{split} image count drift")
    initializer = Path(request.get("inputs", {}).get("model", "")).resolve()
    manifest_path = Path(protocol["run_root"]) / "initializers" / f"seed{seed}" / "initialization_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"initializer manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("runtime_attestation") != RUNTIME_ATTESTATION:
        raise ValueError("initializer manifest runtime provenance drift")
    if manifest.get("protocol_sha256") != sha256(protocol_path):
        raise ValueError("initializer manifest belongs to another protocol")
    cell_manifest = manifest.get("cells", {}).get(cell, {})
    if cell_manifest.get("initializer_sha256") != sha256(initializer):
        raise ValueError("initializer bytes differ from the registered manifest")
    admission_record = None
    if stage == "formal":
        admission_path = Path(protocol.get("formal_admission", {}).get("path", "")).resolve()
        if not admission_path.is_file():
            raise ValueError("formal execution requires combined admission evidence")
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        if (
            admission.get("schema_version") != 1
            or admission.get("status") != "passed"
            or admission.get("protocol_sha256") != sha256(protocol_path)
            or admission.get("all_required_gates_passed") is not True
            or admission.get("formal_may_start") is not True
            or admission.get("formal_request_lineage_verified") is not True
            or admission.get("formal_directory_absent_at_admission") is not True
            or admission.get("dependency_hash_graph_verified") is not True
            or admission.get("raw_gate_metrics_recomputed") is not True
            or admission.get("implementation_and_git_lineage_verified") is not True
            or admission.get("counts")
            != {
                "initializer_manifests": 3,
                "preflight_cells": 12,
                "routing_probe_cells": 6,
                "routing_routers": 36,
                "residual_layers": 18,
                "formal_requests": 12,
            }
        ):
            raise ValueError("combined formal admission evidence is invalid")
        admission_record = {"path": str(admission_path), "sha256": sha256(admission_path)}
    return {
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "stage": stage,
        "seed": int(seed),
        "cell": cell,
        "request_sha256": request_sha,
        "initializer_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "runtime_attestation": RUNTIME_ATTESTATION,
        "formal_admission": admission_record,
    }


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
    normalized_model = str(model_path.resolve()).replace("\\", "/").lower()
    if "p1_factorial_r19" in normalized_model or "yolo-master-r19" in normalized_model:
        raise ValueError(f"r26 request must not reference r19 weights: {model_path}")
    pretrained = params.get("pretrained")
    if model_path.suffix == ".pt" and pretrained is not True:
        raise ValueError("checkpoint continuation must set pretrained=true so Model.train() retains loaded weights")
    if model_path.suffix in {".yaml", ".yml"} and (not isinstance(pretrained, str) or not Path(pretrained).is_file()):
        raise ValueError("YAML training must provide an existing pretrained checkpoint path")
    policy = request.get("a1_policy", {})
    routing_semantics = policy.get("routing_semantics")
    if routing_semantics != R26_ROUTING_SEMANTICS:
        raise ValueError(f"r26 routing semantics drift: {routing_semantics!r}")
    drift = {key: (params.get(key), value) for key, value in P1_ROUTING_PARAMS.items() if params.get(key) != value}
    if drift:
        raise ValueError(f"P1 deterministic routing policy drift: {drift}")
    if policy.get("expert_dropout_rate") != 0.0:
        raise ValueError("P1 hard Top-2 routing requires expert_dropout_rate=0.0")
    exploration = policy.get("router_exploration")
    if not isinstance(exploration, dict):
        raise TypeError("r26 requires an explicit router_exploration policy")
    expected = {**R26_EXPLORATION_POLICY, "base_seed": params.get("seed")}
    drift = {key: (exploration.get(key), value) for key, value in expected.items() if exploration.get(key) != value}
    if drift:
        raise ValueError(f"r26 router exploration policy drift: {drift}")
    if exploration.get("enabled") not in {True, False}:
        raise ValueError("r26 router_exploration.enabled must be boolean")
    expected_aux = R26_CLEAN_AUX_POLICY if exploration["enabled"] else R26_DENSE_AUX_POLICY
    if policy.get("routing_auxiliary_objective") != expected_aux:
        raise ValueError(f"r26 routing auxiliary objective drift: {policy.get('routing_auxiliary_objective')!r}")
    gain_policy = policy.get("factor_gain_optimizer")
    if gain_policy != R26_GAIN_POLICY:
        raise ValueError(f"r26 factor gain optimizer policy drift: {gain_policy!r}")
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


def configure_r26_exploration(trainer, request: dict) -> None:
    """Install deterministic private RNG streams and calibrated per-router sigma values."""
    policy = request.get("a1_policy", {})
    trainer.p1_routing_semantics = policy.get("routing_semantics")
    trainer.p1_router_exploration = policy.get("router_exploration")
    trainer.p1_routing_auxiliary_objective = policy.get("routing_auxiliary_objective")
    trainer.p1_microbatch_index = 0
    if trainer.p1_routing_semantics != R26_ROUTING_SEMANTICS:
        return

    exploration = trainer.p1_router_exploration
    routed = _routed_modules(trainer.model)
    enabled = bool(exploration["enabled"])
    if enabled != bool(routed):
        raise ValueError(f"r26 exploration enabled={enabled} but routed module count is {len(routed)}")
    base_seed = int(exploration["base_seed"])
    stride = int(exploration["private_seed_stride"])
    sigma_min = float(exploration["sigma_min"])
    sigma_max = float(exploration["sigma_max"])
    for index, (_, module) in enumerate(routed):
        routing = module.routing
        sigma0 = float(routing.p1_noise_sigma0.item())
        if not sigma_min <= sigma0 <= sigma_max:
            raise ValueError(f"uncalibrated r26 router sigma0={sigma0} outside [{sigma_min}, {sigma_max}]")
        routing.configure_p1_private_noise(base_seed + stride * (index + 1), reset_step=True)
        routing.noise_std = sigma0


def r26_noise_scale(microbatch_index: int, exploration: dict) -> float:
    """Return the locked r26 hold/linear-decay multiplier for one microbatch."""
    step = int(microbatch_index)
    hold = int(exploration["hold_through_microbatch"])
    end = int(exploration["decay_to_zero_microbatch"])
    if step <= hold:
        return 1.0
    if step <= end:
        return max(0.0, (end - step) / (end - hold))
    return 0.0


def enforce_and_schedule_p1_policy(trainer) -> None:
    """Reapply freeze constraints and set the r26 noise for the next train microbatch."""
    enforce_p1_freeze_policy(trainer)
    if getattr(trainer, "p1_routing_semantics", None) != R26_ROUTING_SEMANTICS:
        return
    exploration = trainer.p1_router_exploration
    scale = r26_noise_scale(trainer.p1_microbatch_index, exploration)
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
                "p1_balance_on_clean_routes": bool(
                    getattr(routing, "p1_balance_on_clean_routes", False)
                ),
                "routing_aux_semantics": getattr(
                    module,
                    "routing_aux_semantics",
                    None,
                ),
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
        "runtime_attestation": RUNTIME_ATTESTATION,
        "requested": {key: getattr(trainer.args, key, None) for key in P1_ROUTING_PARAMS},
        "mixture_config": mixture_config,
        "factor_adapters": factor_adapter_policy(trainer.model),
        "routed_modules": routed_module_policy(trainer.model),
        "optimizer_groups": optimizer_policy(trainer),
        "routing_semantics": getattr(trainer, "p1_routing_semantics", None),
        "routing_auxiliary_objective": getattr(
            trainer,
            "p1_routing_auxiliary_objective",
            None,
        ),
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
    r26 = getattr(trainer, "p1_routing_semantics", None) == R26_ROUTING_SEMANTICS
    if not r26:
        raise ValueError(f"runtime routing semantics are not r26: {getattr(trainer, 'p1_routing_semantics', None)!r}")
    requested_aux = getattr(trainer, "p1_routing_auxiliary_objective", None)
    if requested_aux not in (R26_CLEAN_AUX_POLICY, R26_DENSE_AUX_POLICY):
        raise ValueError(f"runtime r26 routing-aux policy drift: {requested_aux!r}")
    clean_aux_enabled = requested_aux == R26_CLEAN_AUX_POLICY
    expected_router_count = 6 if clean_aux_enabled else 0
    if len(routed) != expected_router_count:
        raise ValueError(
            f"r26 clean_aux_enabled={clean_aux_enabled} requires {expected_router_count} routers, got {len(routed)}"
        )
    for module in routed:
        if (
            module["top_k"] != 2
            or module["current_top_k"] != 2
            or module["progressive_sparsity"]
            or module["warmup_steps"] != 0
            or module["expert_dropout_rate"] != 0.0
        ):
            raise ValueError(f"effective routed module policy mismatch: {module}")
        if module["p1_balance_on_clean_routes"] is not True:
            raise ValueError(f"r26 router is not balancing clean routes: {module}")
        if module["routing_aux_semantics"] != R26_CLEAN_AUX_POLICY["runtime_semantics"]:
            raise ValueError(f"r26 routing-aux semantics mismatch: {module}")
        if r26:
            sigma_min = float(trainer.p1_router_exploration["sigma_min"])
            sigma_max = float(trainer.p1_router_exploration["sigma_max"])
            if not sigma_min <= module["p1_noise_sigma0"] <= sigma_max:
                raise ValueError(f"r26 router sigma calibration mismatch: {module}")
            if not math.isclose(module["noise_std"], module["p1_noise_sigma0"], abs_tol=1e-12):
                raise ValueError(f"r26 router did not start at calibrated sigma0: {module}")
            if module["p1_noise_seed"] is None or module["p1_noise_step"] != 0:
                raise ValueError(f"r26 private router RNG was not reset: {module}")
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
    if r26:
        gain_groups = [group for group in optimizer_policy(trainer) if group["param_group"] == "residual_gain"]
        expected_gain_parameters = sum(adapter["gain_parameters"] for adapter in adapters)
        if (
            len(gain_groups) != 1
            or gain_groups[0]["parameters"] != expected_gain_parameters
            or not math.isclose(gain_groups[0]["initial_lr"], R26_GAIN_POLICY["lr"], abs_tol=1e-12)
            or gain_groups[0]["weight_decay"] != R26_GAIN_POLICY["weight_decay"]
            or not gain_groups[0]["p1_no_warmup"]
        ):
            raise ValueError(f"r26 residual gain optimizer group mismatch: {gain_groups}")


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
    request_path = args.request.resolve()
    request = read_request(request_path)
    registration = validate_request_registration(request_path, request)
    params = dict(request["params"])
    report = {
        "schema_version": 1,
        "request": str(args.request.resolve()),
        "registration": registration,
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
        configure_r26_exploration(trainer, request)
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

