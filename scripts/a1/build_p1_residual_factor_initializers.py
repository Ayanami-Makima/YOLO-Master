#!/usr/bin/env python3
"""Build P1 initializers around the native pretrained C3k2 backbone."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

CELLS = "abcd"
FACTOR_LAYERS = (4, 6, 8)
EXPERTS = {4: 4, 6: 8, 8: 16}
BALANCED_ROUTER_SCHEME = "deterministic_data_independent_regular_simplex_final_projection"
ROUTER_SEED_LAYER_STRIDE = 1000


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    """Set every initializer RNG used by the experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def initialize_regular_simplex_projection(
    routed_module: nn.Module, *, seed: int, target_entry_rms: float = 0.05
) -> dict:
    """Initialize one router's final 1x1 projection as a centered regular simplex."""
    router_stack = getattr(getattr(routed_module, "routing", None), "router", None)
    if not isinstance(router_stack, nn.Module):
        raise TypeError(f"cannot resolve router stack for {type(routed_module).__name__}")
    projections = [module for module in router_stack.modules() if isinstance(module, nn.Conv2d)]
    if not projections:
        raise TypeError(f"router stack for {type(routed_module).__name__} has no Conv2d projection")
    projection = projections[-1]
    weight = projection.weight
    if tuple(weight.shape[2:]) != (1, 1):
        raise ValueError(f"router final projection must be 1x1, got {tuple(weight.shape)}")
    num_experts, input_width = weight.shape[:2]
    if input_width < num_experts:
        raise ValueError(
            f"regular-simplex router initialization needs input width >= experts, got {input_width} < {num_experts}"
        )
    if not math.isfinite(target_entry_rms) or target_entry_rms <= 0:
        raise ValueError(f"target_entry_rms must be positive and finite, got {target_entry_rms}")

    # A private CPU generator makes this initialization independent of model-construction RNG use.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    gaussian = torch.randn(input_width, num_experts - 1, generator=generator, dtype=torch.float64)
    gaussian -= gaussian.mean(dim=0, keepdim=True)
    orthogonal, upper = torch.linalg.qr(gaussian, mode="reduced")
    diagonal = torch.diagonal(upper)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    orthogonal *= signs.unsqueeze(0)
    contrasts = torch.zeros(num_experts, num_experts - 1, dtype=torch.float64)
    for column in range(num_experts - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        contrasts[: column + 1, column] = 1.0 / denominator
        contrasts[column + 1, column] = -(column + 1) / denominator
    embedding = contrasts @ orthogonal.transpose(0, 1)
    target_row_norm = target_entry_rms * math.sqrt(input_width)
    simplex = embedding * (target_row_norm / math.sqrt(1.0 - 1.0 / num_experts))
    if not torch.isfinite(simplex).all():
        raise ValueError("regular-simplex router initialization produced non-finite weights")

    with torch.no_grad():
        weight.copy_(simplex.reshape_as(weight).to(device=weight.device, dtype=weight.dtype))
        if projection.bias is not None:
            projection.bias.zero_()

    flat = weight.detach().float().cpu().flatten(1)
    row_norms = flat.norm(dim=1)
    gram = flat @ flat.transpose(0, 1)
    row_distances = torch.cdist(flat, flat)
    row_distances.fill_diagonal_(float("inf"))
    off_diagonal = ~torch.eye(num_experts, dtype=torch.bool)
    expected_off_diagonal = -(target_row_norm**2) / (num_experts - 1)
    return {
        "scheme": BALANCED_ROUTER_SCHEME,
        "seed": seed,
        "num_experts": num_experts,
        "input_width": input_width,
        "target_entry_rms": target_entry_rms,
        "target_row_norm": target_row_norm,
        "observed_entry_rms": float(flat.square().mean().sqrt()),
        "minimum_row_norm": float(row_norms.min()),
        "maximum_row_norm": float(row_norms.max()),
        "row_norm_spread": float(row_norms.max() - row_norms.min()),
        "common_direction_max_abs": float(flat.mean(dim=0).abs().max()),
        "channel_mean_max_abs": float(flat.mean(dim=1).abs().max()),
        "minimum_distinct_row_distance": float(row_distances.min()),
        "off_diagonal_gram_max_error": float((gram[off_diagonal] - expected_off_diagonal).abs().max()),
        "weight_sha256": hashlib.sha256(flat.contiguous().numpy().tobytes()).hexdigest(),
        "expected_fp16_roundtrip_sha256": hashlib.sha256(
            flat.half().float().contiguous().numpy().tobytes()
        ).hexdigest(),
    }


def initialize_balanced_router_projections(
    module: nn.Module, *, base_seed: int, layer_index: int, target_entry_rms: float = 0.05
) -> list[dict]:
    """Initialize every routed core using stable per-layer and per-router private seeds."""
    reports = []
    routed_modules = [
        (name, child)
        for name, child in module.named_modules()
        if hasattr(child, "progressive_sparsity") and hasattr(child, "routing")
    ]
    for router_index, (name, child) in enumerate(routed_modules):
        projection_seed = base_seed + layer_index * ROUTER_SEED_LAYER_STRIDE + router_index
        report = initialize_regular_simplex_projection(
            child,
            seed=projection_seed,
            target_entry_rms=target_entry_rms,
        )
        report.update({"name": name, "router_index": router_index})
        reports.append(report)
    return reports


def router_projection_fingerprints(module: nn.Module) -> list[dict]:
    """Fingerprint routed final projections without changing their tensors."""
    reports = []
    for name, child in module.named_modules():
        if not (hasattr(child, "progressive_sparsity") and hasattr(child, "routing")):
            continue
        router_stack = getattr(child.routing, "router", None)
        projections = [item for item in router_stack.modules() if isinstance(item, nn.Conv2d)]
        if not projections:
            raise TypeError(f"router {name} has no Conv2d projection")
        weight = projections[-1].weight.detach().cpu().contiguous()
        reports.append(
            {
                "name": name,
                "shape": list(weight.shape),
                "dtype": str(weight.dtype),
                "weight_sha256": hashlib.sha256(weight.numpy().tobytes()).hexdigest(),
            }
        )
    return reports


def tensor_leaves(value):
    """Yield tensors from nested native model outputs in a stable order."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from tensor_leaves(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensor_leaves(item)


def equivalence_report(source_model: nn.Module, target_model: nn.Module, inputs: torch.Tensor) -> dict:
    """Compare raw outputs before confidence filtering or NMS."""
    source_model.eval()
    target_model.eval()
    with torch.inference_mode():
        expected = list(tensor_leaves(source_model(inputs)))
        actual = list(tensor_leaves(target_model(inputs)))
    if len(expected) != len(actual):
        return {"passed": False, "reason": f"tensor leaf count {len(expected)} != {len(actual)}"}
    maximum = 0.0
    total_error = 0.0
    total_values = 0
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left.shape != right.shape:
            return {"passed": False, "reason": f"leaf {index} shape {list(left.shape)} != {list(right.shape)}"}
        difference = (left.float() - right.float()).abs()
        maximum = max(maximum, float(difference.max()) if difference.numel() else 0.0)
        total_error += float(difference.sum())
        total_values += difference.numel()
    return {
        "passed": True,
        "tensor_leaves": len(expected),
        "values": total_values,
        "max_abs_error": maximum,
        "mean_abs_error": total_error / max(total_values, 1),
        "input_shape": list(inputs.shape),
    }


def output_channels(module: nn.Module) -> int:
    """Read the output width from one native C3k2 block."""
    cv2 = getattr(module, "cv2", None)
    conv = getattr(cv2, "conv", None)
    channels = getattr(conv, "out_channels", None)
    if not isinstance(channels, int):
        raise TypeError(f"cannot resolve output channels for {type(module).__name__}")
    return channels


def routed_policy(module: nn.Module) -> list[dict]:
    """Describe every routed factor core in a deterministic initializer."""
    report = []
    for name, child in module.named_modules():
        if not (hasattr(child, "progressive_sparsity") and hasattr(child, "routing")):
            continue
        report.append(
            {
                "name": name,
                "num_experts": int(child.num_experts),
                "top_k": int(child.top_k),
                "current_top_k": int(child._current_top_k),
                "progressive_sparsity": bool(child.progressive_sparsity),
                "warmup_steps": int(child.warmup_steps),
                "noise_std": float(child.routing.noise_std),
                "expert_dropout_rate": float(child.expert_dropout_rate),
            }
        )
    return report


def require_deterministic_routing(routers: list[dict], layer_index: int) -> None:
    """Reject a routed initializer that differs from the locked P1 semantics."""
    if len(routers) != 2:
        raise ValueError(f"layer {layer_index}: expected two routed cores, got {len(routers)}")
    for router in routers:
        if (
            router["top_k"] != 2
            or router["current_top_k"] != 2
            or router["progressive_sparsity"]
            or router["warmup_steps"] != 0
            or router["noise_std"] != 0.0
            or router["expert_dropout_rate"] != 0.0
        ):
            raise ValueError(f"layer {layer_index}: deterministic routing policy mismatch: {router}")


def zero_module(module: nn.Module) -> None:
    """Make one auxiliary expert contribute zero in evaluation mode."""
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
        for name, buffer in module.named_buffers():
            buffer.fill_(1 if name.endswith("running_var") else 0)


def transplant_dense_factor(dense: nn.Module, moe: nn.Module) -> dict:
    """Copy all common tensors and clone each dense MLP into every expert."""
    dense_state = dense.state_dict()
    target_state = moe.state_dict()
    compatible = {
        name: value
        for name, value in dense_state.items()
        if name in target_state and value.shape == target_state[name].shape
    }
    moe.load_state_dict(compatible, strict=False)
    experts = 0
    blocks = 0
    for dense_sequence, moe_sequence in zip(dense.m, moe.m):
        for dense_block, moe_block in zip(dense_sequence, moe_sequence):
            dense_mlp_state = dense_block.mlp.state_dict()
            for expert in moe_block.mlp.experts:
                expert.mlp.load_state_dict(dense_mlp_state, strict=True)
                experts += 1
            zero_module(moe_block.mlp.shared_expert)
            blocks += 1
    return {"compatible_tensors": len(compatible), "blocks": blocks, "experts": experts}


def set_end2end(model: nn.Module, enabled: bool) -> None:
    """Select the NMS or native one-to-one inference path consistently."""
    model.end2end = enabled
    head = model.model[-1]
    if not hasattr(head, "end2end"):
        raise TypeError("checkpoint head has no end2end switch")
    head.end2end = enabled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=260829)
    parser.add_argument("--equivalence-tolerance", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    checkpoint = Path(protocol["source_checkpoint"]["path"])
    run_root = Path(protocol["run_root"])
    suffix = protocol.get("initializer_suffix", "residual_factor_init")
    router_initialization = protocol.get("router_initialization")
    if router_initialization is not None:
        if router_initialization.get("scheme") != BALANCED_ROUTER_SCHEME:
            raise ValueError(f"unsupported router initialization policy: {router_initialization}")
        if int(router_initialization.get("base_seed", -1)) != args.seed:
            raise ValueError("initializer seed differs from the locked router initialization policy")
        if router_initialization.get("data_source") != "none":
            raise ValueError("r17 router initialization must remain data-independent")
        target_entry_rms = float(router_initialization.get("target_entry_rms", 0.05))
    else:
        target_entry_rms = 0.05
    if sha256(checkpoint) != protocol["source_checkpoint"]["sha256"]:
        raise ValueError("source checkpoint hash differs from the locked protocol")

    from ultralytics import YOLO
    from ultralytics.nn.tasks import load_checkpoint

    native, _ = load_checkpoint(checkpoint, device="cpu")
    native = native.float()
    native_state = native.state_dict()
    set_seed(args.seed)
    dense_reference = YOLO(protocol["configs"]["a"]["path"], task="detect").model
    dense_factors = {index: copy.deepcopy(dense_reference.model[index].factor) for index in FACTOR_LAYERS}
    output_root = run_root / "initializers"
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite initializer directory: {output_root}")
    output_root.mkdir(parents=True)
    manifest = {
        "schema_version": 3,
        "status": "building",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "policy": "native pretrained C3k2 base frozen; zero-gain dense/MoE residual factors; matched source head",
        "moe_training_policy": protocol.get("moe_training_policy"),
        "router_initialization": router_initialization,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "seed": args.seed,
        "equivalence_tolerance": args.equivalence_tolerance,
        "factor_layers": list(FACTOR_LAYERS),
        "cells": {},
    }
    sample = torch.rand(1, 3, 64, 64)
    moe_projection_fingerprints_before_save = {}
    moe_projection_fingerprints_expected_after_reload = {}
    moe_projection_fingerprints_after_reload = {}
    for cell in CELLS:
        set_seed(args.seed)
        end2end = cell in "bd"
        is_moe = cell in "cd"
        source = copy.deepcopy(native)
        yolo = YOLO(protocol["configs"][cell]["path"], task="detect")
        target = yolo.model
        set_end2end(source, end2end)
        set_end2end(target, end2end)
        target_state = target.state_dict()
        compatible = {
            name: value
            for name, value in native_state.items()
            if name in target_state and target_state[name].shape == value.shape
        }
        incompatible = target.load_state_dict(compatible, strict=False)
        layer_reports = {}
        for layer_index in FACTOR_LAYERS:
            adapter = target.model[layer_index]
            if type(adapter).__name__ != "C3k2ResidualFactor":
                raise TypeError(f"layer {layer_index}: expected C3k2ResidualFactor, got {type(adapter).__name__}")
            source_layer = native.model[layer_index]
            adapter.base.load_state_dict(source_layer.state_dict(), strict=True)
            adapter.freeze_base_parameters()
            channels = output_channels(adapter.base)
            transplant = None
            projection_reports = []
            if is_moe:
                transplant = transplant_dense_factor(dense_factors[layer_index], adapter.factor)
                if router_initialization is not None:
                    projection_reports = initialize_balanced_router_projections(
                        adapter.factor,
                        base_seed=args.seed,
                        layer_index=layer_index,
                        target_entry_rms=target_entry_rms,
                    )
            else:
                adapter.factor.load_state_dict(dense_factors[layer_index].state_dict(), strict=True)
            routers = routed_policy(adapter.factor)
            if is_moe:
                require_deterministic_routing(routers, layer_index)
            elif routers:
                raise ValueError(f"layer {layer_index}: dense initializer unexpectedly contains routers")
            layer_reports[str(layer_index)] = {
                "base": type(adapter.base).__name__,
                "factor": type(adapter.factor).__name__,
                "channels": channels,
                "experts": EXPERTS[layer_index] if is_moe else None,
                "top_k": 2 if is_moe else None,
                "frozen_base_parameters": sum(parameter.numel() for parameter in adapter.base.parameters()),
                "trainable_factor_parameters": sum(parameter.numel() for parameter in adapter.factor.parameters()),
                "gain_nonzero": int(torch.count_nonzero(adapter.gain)),
                "dense_to_moe_transplant": transplant,
                "router_initialization": projection_reports,
                "routers": routers,
            }
        comparison = equivalence_report(source, target, sample)
        comparison["passed"] = bool(
            comparison.get("passed") and comparison.get("max_abs_error", float("inf")) <= args.equivalence_tolerance
        )
        if not comparison["passed"]:
            raise ValueError(f"{cell}: residual initializer equivalence failed: {comparison}")
        output = output_root / f"{cell}_{suffix}.pt"
        yolo.ckpt = {}
        yolo.save(output)
        reloaded, _ = load_checkpoint(output, device="cpu")
        reloaded_comparison = equivalence_report(source, reloaded.float(), sample)
        reloaded_comparison["passed"] = bool(
            reloaded_comparison.get("passed")
            and reloaded_comparison.get("max_abs_error", float("inf")) <= args.equivalence_tolerance
        )
        if not reloaded_comparison["passed"]:
            raise ValueError(f"{cell}: saved initializer equivalence failed: {reloaded_comparison}")
        reloaded_layers = {}
        reloaded_projection_layers = {}
        for layer_index in FACTOR_LAYERS:
            routers = routed_policy(reloaded.model[layer_index].factor)
            if is_moe:
                require_deterministic_routing(routers, layer_index)
            elif routers:
                raise ValueError(f"layer {layer_index}: reloaded dense initializer unexpectedly contains routers")
            reloaded_layers[str(layer_index)] = routers
            reloaded_projection_layers[str(layer_index)] = (
                router_projection_fingerprints(reloaded.model[layer_index].factor) if is_moe else []
            )
        state = target.state_dict()
        manifest["cells"][cell] = {
            "moe": is_moe,
            "end2end": end2end,
            "initializer": str(output),
            "initializer_sha256": sha256(output),
            "parameters": sum(value.numel() for value in state.values()),
            "compatible_source_tensors": len(compatible),
            "missing_tensors_before_base_load": len(incompatible.missing_keys),
            "native_tensors_preserved": sum(
                name in state and state[name].shape == value.shape and torch.equal(state[name], value)
                for name, value in native_state.items()
            ),
            "layers": layer_reports,
            "equivalence_before_save": comparison,
            "equivalence_after_reload": reloaded_comparison,
            "reloaded_routers": reloaded_layers,
            "reloaded_router_projections": reloaded_projection_layers,
        }
        if is_moe and router_initialization is not None:
            before_save = {
                layer: [item["weight_sha256"] for item in layer_reports[layer]["router_initialization"]]
                for layer in map(str, FACTOR_LAYERS)
            }
            expected_after_reload = {
                layer: [
                    item["expected_fp16_roundtrip_sha256"] for item in layer_reports[layer]["router_initialization"]
                ]
                for layer in map(str, FACTOR_LAYERS)
            }
            after_reload = {
                layer: [item["weight_sha256"] for item in reloaded_projection_layers[layer]]
                for layer in map(str, FACTOR_LAYERS)
            }
            if expected_after_reload != after_reload:
                raise ValueError(f"{cell}: router projections differ from the locked FP16 checkpoint round-trip")
            moe_projection_fingerprints_before_save[cell] = before_save
            moe_projection_fingerprints_expected_after_reload[cell] = expected_after_reload
            moe_projection_fingerprints_after_reload[cell] = after_reload
    if router_initialization is not None:
        if moe_projection_fingerprints_before_save.get("c") != moe_projection_fingerprints_before_save.get("d"):
            raise ValueError("C/D balanced router projections are not byte-identical")
        if moe_projection_fingerprints_after_reload.get("c") != moe_projection_fingerprints_after_reload.get("d"):
            raise ValueError("reloaded C/D balanced router projections are not byte-identical")
        manifest["moe_router_projection_parity"] = {
            "passed": True,
            "comparison": "C/D projections match before save and after the locked FP16 checkpoint round-trip",
            "before_save": moe_projection_fingerprints_before_save,
            "expected_after_fp16_roundtrip": moe_projection_fingerprints_expected_after_reload,
            "after_reload": moe_projection_fingerprints_after_reload,
        }
    manifest["status"] = "passed"
    manifest_path = output_root / "initialization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
