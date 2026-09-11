#!/usr/bin/env python3
"""Build one seed's calibrated, function-preserving A1 P1 r20 initializers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from build_p1_residual_factor_initializers import (
    CELLS,
    FACTOR_LAYERS,
    equivalence_report,
    output_channels,
    require_deterministic_routing,
    routed_policy,
    router_projection_fingerprints,
    set_end2end,
    set_seed,
    sha256,
    transplant_dense_factor,
)
from p1_r20_integrity import verify_registered_data_content
from torch import nn

EXPECTED_SEEDS = (260829, 260830, 260831)
OFFICIAL_CHECKPOINT = Path("/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master/yolo26n.pt")
OFFICIAL_CHECKPOINT_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
EXPECTED_CLEAN_AUX_POLICY = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--equivalence-tolerance", type=float, default=0.0)
    return parser.parse_args()


def tensor_hash(tensor: torch.Tensor) -> str:
    """Hash one tensor after making its memory layout stable."""
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def state_hash(module: nn.Module) -> str:
    """Hash a module state dict including names, dtypes, shapes, and values."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def batch_norm_snapshot(model: nn.Module, *, excluded_ids: set[int] | None = None) -> dict[str, dict[str, str]]:
    """Fingerprint BatchNorm running state, optionally excluding selected modules."""
    excluded_ids = excluded_ids or set()
    snapshot = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm) or id(module) in excluded_ids:
            continue
        snapshot[name] = {
            "running_mean": tensor_hash(module.running_mean),
            "running_var": tensor_hash(module.running_var),
            "num_batches_tracked": tensor_hash(module.num_batches_tracked),
        }
    return snapshot


def factor_modules(model: nn.Module) -> list[nn.Module]:
    """Resolve the three new factor branches."""
    return [model.model[index].factor for index in FACTOR_LAYERS]


def factor_batch_norms(model: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    """Return every BatchNorm introduced inside the three factor branches."""
    return [
        child
        for factor in factor_modules(model)
        for child in factor.modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm)
    ]


def routed_cores(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Resolve routed cores in stable model traversal order."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if hasattr(module, "progressive_sparsity") and hasattr(module, "routing")
    ]


def clean_aux_router_report(module: nn.Module, *, expected: bool) -> list[dict]:
    """Attest the r20 clean-aux flag without changing dispatch or weights."""
    reports = []
    for name, core in routed_cores(module):
        routing = core.routing
        enabled = bool(getattr(routing, "p1_balance_on_clean_routes", False))
        if enabled is not expected:
            raise ValueError(f"{name}: clean-aux flag {enabled} != expected {expected}")
        reports.append(
            {
                "name": f"{name}.routing",
                "num_experts": int(routing.num_experts),
                "top_k": int(routing.top_k),
                "p1_balance_on_clean_routes": enabled,
                "routing_aux_semantics": getattr(core, "routing_aux_semantics", None),
            }
        )
        if expected and reports[-1]["routing_aux_semantics"] != EXPECTED_CLEAN_AUX_POLICY["runtime_semantics"]:
            raise ValueError(f"{name}: r20 routing-aux semantics mismatch")
    expected_count = 2 if expected else 0
    if len(reports) != expected_count:
        raise ValueError(f"expected {expected_count} routed cores, got {len(reports)}")
    return reports


def load_image_batch(paths: list[str], imgsz: int) -> torch.Tensor:
    """Load one deterministic, unaugmented letterboxed RGB batch."""
    from ultralytics.data.augment import LetterBox

    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, scale_fill=False, scaleup=True, stride=32)
    images = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"could not decode calibration image: {path}")
        image = letterbox(image=image)
        image = np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))
        images.append(torch.from_numpy(image))
    return torch.stack(images).float().div_(255.0)


def image_batches(paths: list[str], *, batch: int, imgsz: int):
    """Yield fixed image batches without shuffle or augmentation."""
    for start in range(0, len(paths), batch):
        yield load_image_batch(paths[start : start + batch], imgsz)


def expert_batch_norm_groups(core: nn.Module) -> list[list[nn.modules.batchnorm._BatchNorm]]:
    """Return aligned BatchNorm lists for a routed core's cloned experts."""
    return [
        [child for child in expert.modules() if isinstance(child, nn.modules.batchnorm._BatchNorm)]
        for expert in core.experts
    ]


def calibrate_factor_batch_norm(
    model: nn.Module, image_paths: list[str], *, batch: int, imgsz: int, device: torch.device
) -> dict:
    """Calibrate only new factor BN statistics with train images and no gradients."""
    model = model.to(device).float().eval()
    factor_bns = factor_batch_norms(model)
    factor_bn_ids = {id(module) for module in factor_bns}
    nonfactor_before = batch_norm_snapshot(model, excluded_ids=factor_bn_ids)
    factor_before = {
        name: batch_norm_snapshot(factor) for name, factor in zip(map(str, FACTOR_LAYERS), factor_modules(model))
    }
    original_momentum = {id(module): module.momentum for module in factor_bns}
    for module in factor_bns:
        module.reset_running_stats()
        module.momentum = None
        module.train()
        for parameter in module.parameters(recurse=False):
            parameter.requires_grad = False

    # All experts are exact clones at initialization.  Run expert 0 on every
    # calibration batch, keep routed dispatch itself sparse, then copy the
    # prototype statistics to its sibling experts.  This avoids both dead-BN
    # defaults and a dense K=E calibration pass.
    hooks = []
    expert_bn_ids = set()
    core_groups = []
    for _, core in routed_cores(model):
        groups = expert_batch_norm_groups(core)
        if not groups or not groups[0]:
            raise ValueError("r20 expected BatchNorm-bearing cloned experts")
        if any(len(group) != len(groups[0]) for group in groups):
            raise ValueError("r20 expert BatchNorm structures are not aligned")
        core_groups.append(groups)
        for group in groups:
            expert_bn_ids.update(map(id, group))
            for module in group:
                module.eval()

        prototype = core.experts[0]
        prototype_bns = groups[0]

        def calibrate_prototype(_module, inputs, *, prototype=prototype, prototype_bns=prototype_bns):
            for child in prototype_bns:
                child.train()
            prototype(inputs[0])
            for child in prototype_bns:
                child.eval()

        hooks.append(core.register_forward_pre_hook(calibrate_prototype))

    for module in factor_bns:
        if id(module) in expert_bn_ids:
            module.eval()
    with torch.inference_mode():
        for images in image_batches(image_paths, batch=batch, imgsz=imgsz):
            model(images.to(device, non_blocking=False))
    for hook in hooks:
        hook.remove()

    expert_parity = []
    for groups in core_groups:
        prototype = groups[0]
        for siblings in groups[1:]:
            for source, target in zip(prototype, siblings):
                target.running_mean.copy_(source.running_mean)
                target.running_var.copy_(source.running_var)
                target.num_batches_tracked.copy_(source.num_batches_tracked)
        hashes = [
            [tensor_hash(module.running_mean) + tensor_hash(module.running_var) for module in group] for group in groups
        ]
        expert_parity.append(all(value == hashes[0] for value in hashes[1:]))

    for module in factor_bns:
        module.momentum = original_momentum[id(module)]
        module.eval()
        for parameter in module.parameters(recurse=False):
            parameter.requires_grad = False
    nonfactor_after = batch_norm_snapshot(model, excluded_ids=factor_bn_ids)
    if nonfactor_before != nonfactor_after:
        changed = sorted(set(nonfactor_before) | set(nonfactor_after))
        raise ValueError(f"BN calibration changed a non-factor BatchNorm: {changed[:8]}")
    factor_after = {
        name: batch_norm_snapshot(factor) for name, factor in zip(map(str, FACTOR_LAYERS), factor_modules(model))
    }
    changed_factor_bn = sum(
        factor_before[layer][name] != factor_after[layer][name]
        for layer in factor_before
        for name in factor_before[layer]
    )
    if changed_factor_bn == 0:
        raise ValueError("BN calibration did not update any factor running statistics")
    return {
        "factor_batch_norm_count": len(factor_bns),
        "changed_factor_batch_norm_count": changed_factor_bn,
        "nonfactor_batch_norm_unchanged": True,
        "expert_batch_norm_parity": all(expert_parity) if expert_parity else None,
        "batches": len(image_paths) // batch,
        "images": len(image_paths),
        "batch": batch,
        "imgsz": imgsz,
        "grad": False,
        "augment": False,
        "shuffle": False,
    }


def calibrate_router_sigmas(
    model: nn.Module,
    image_paths: list[str],
    *,
    batch: int,
    imgsz: int,
    device: torch.device,
    sigma_min: float,
    sigma_max: float,
) -> list[dict]:
    """Calibrate each router's sigma0 from pre-noise logits on train-only images."""
    model = model.to(device).float().eval()
    routed = routed_cores(model)
    values = {name: [] for name, _ in routed}
    hooks = []
    for name, core in routed:

        def capture(_module, _inputs, output, *, name=name):
            logits = output.float()
            if logits.ndim == 4:
                logits = logits.mean(dim=(2, 3))
            if logits.ndim != 2:
                raise ValueError(f"router {name} emitted unexpected shape {tuple(logits.shape)}")
            values[name].append(logits.std(dim=1, unbiased=False).detach().cpu())

        hooks.append(core.routing.router.register_forward_hook(capture))
    with torch.inference_mode():
        for images in image_batches(image_paths, batch=batch, imgsz=imgsz):
            model(images.to(device, non_blocking=False))
    for hook in hooks:
        hook.remove()

    reports = []
    for name, core in routed:
        per_image = torch.cat(values[name])
        if len(per_image) != len(image_paths) or not torch.isfinite(per_image).all():
            raise ValueError(f"router {name} sigma calibration is incomplete or non-finite")
        raw = float(torch.quantile(per_image, 0.5))
        sigma0 = min(max(raw, sigma_min), sigma_max)
        core.routing.p1_noise_sigma0.fill_(sigma0)
        core.routing.noise_std = 0.0
        reports.append(
            {
                "name": name,
                "images": len(per_image),
                "raw_median_per_image_logit_std": raw,
                "sigma0": sigma0,
                "clipped": sigma0 != raw,
                "minimum": float(per_image.min()),
                "maximum": float(per_image.max()),
            }
        )
    return reports


def clone_factor_state(source: nn.Module, target: nn.Module) -> None:
    """Copy all calibrated factor tensors between paired NMS/E2E cells."""
    for layer_index in FACTOR_LAYERS:
        target.model[layer_index].factor.load_state_dict(source.model[layer_index].factor.state_dict(), strict=True)


def build_models(protocol: dict, seed: int):
    """Construct A/B/C/D around the native source checkpoint for one seed."""
    from ultralytics import YOLO
    from ultralytics.nn.tasks import load_checkpoint

    checkpoint = Path(protocol["source_checkpoint"]["path"])
    native, _ = load_checkpoint(checkpoint, device="cpu")
    native = native.float().eval()
    native_state = native.state_dict()
    set_seed(seed)
    dense_reference = YOLO(protocol["configs"]["a"]["path"], task="detect").model
    dense_factors = {index: copy.deepcopy(dense_reference.model[index].factor) for index in FACTOR_LAYERS}
    wrappers = {}
    sources = {}
    build_reports = {}
    for cell in CELLS:
        set_seed(seed)
        end2end = cell in "bd"
        is_moe = cell in "cd"
        source = copy.deepcopy(native)
        wrapper = YOLO(protocol["configs"][cell]["path"], task="detect")
        target = wrapper.model.float()
        set_end2end(source, end2end)
        set_end2end(target, end2end)
        target_state = target.state_dict()
        compatible = {
            name: value
            for name, value in native_state.items()
            if name in target_state and target_state[name].shape == value.shape
        }
        incompatible = target.load_state_dict(compatible, strict=False)
        layers = {}
        for layer_index in FACTOR_LAYERS:
            adapter = target.model[layer_index]
            if type(adapter).__name__ != "C3k2ResidualFactor":
                raise TypeError(f"layer {layer_index}: got {type(adapter).__name__}")
            adapter.base.load_state_dict(native.model[layer_index].state_dict(), strict=True)
            adapter.freeze_base_parameters()
            if is_moe:
                transplant = transplant_dense_factor(dense_factors[layer_index], adapter.factor)
                require_deterministic_routing(routed_policy(adapter.factor), layer_index)
            else:
                adapter.factor.load_state_dict(dense_factors[layer_index].state_dict(), strict=True)
                transplant = None
            if torch.count_nonzero(adapter.gain):
                raise ValueError(f"{cell} layer {layer_index}: gain is not exactly zero")
            layers[str(layer_index)] = {
                "base": type(adapter.base).__name__,
                "factor": type(adapter.factor).__name__,
                "channels": output_channels(adapter.base),
                "frozen_base_parameters": sum(parameter.numel() for parameter in adapter.base.parameters()),
                "dense_to_moe_transplant": transplant,
                "router_projection": router_projection_fingerprints(adapter.factor) if is_moe else [],
                "r20_clean_aux_routers": clean_aux_router_report(adapter.factor, expected=is_moe),
            }
        wrappers[cell] = wrapper
        sources[cell] = source
        build_reports[cell] = {
            "compatible_source_tensors": len(compatible),
            "missing_tensors_before_base_load": len(incompatible.missing_keys),
            "layers": layers,
        }
    return wrappers, sources, build_reports


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 6 or protocol.get("experiment_tag") != "r20":
        raise ValueError("r20 initializer builder requires an r20 schema-6 protocol")
    if tuple(protocol.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError(f"r20 seed registry drift: {protocol.get('seeds')}")
    if protocol.get("routing", {}).get("auxiliary_objective") != EXPECTED_CLEAN_AUX_POLICY:
        raise ValueError("r20 clean-aux policy drift")
    if args.seed not in protocol["seeds"]:
        raise ValueError(f"seed {args.seed} is not registered in protocol")
    if set(protocol.get("configs", {})) != set(CELLS):
        raise ValueError("r20 model-config registry must contain exactly A/B/C/D")
    for cell in CELLS:
        config = protocol["configs"][cell]
        config_path = Path(config.get("path", ""))
        if not config_path.is_file() or sha256(config_path) != config.get("sha256"):
            raise ValueError(f"r20 model config hash drift: {cell}/{config_path}")
    repo = Path(__file__).resolve().parents[2]
    for relative, expected_sha in protocol.get("implementation", {}).items():
        implementation_path = repo / relative
        if not implementation_path.is_file() or sha256(implementation_path) != expected_sha:
            raise ValueError(f"r20 implementation hash drift: {relative}")
    pilot = protocol.get("data", {}).get("pilot", {})
    pilot_yaml = Path(pilot.get("path", ""))
    if not pilot_yaml.is_file() or sha256(pilot_yaml) != pilot.get("sha256"):
        raise ValueError("r20 pilot YAML hash drift")
    for split, expected_count in (("train", 5000), ("val", 512)):
        item = pilot.get("lists", {}).get(split, {})
        list_path = Path(item.get("path", ""))
        if not list_path.is_file() or sha256(list_path) != item.get("sha256"):
            raise ValueError(f"r20 pilot {split} image-list hash drift")
        count = sum(bool(line.strip()) for line in list_path.read_text(encoding="utf-8").splitlines())
        if count != expected_count or item.get("images") != expected_count:
            raise ValueError(f"r20 pilot {split} image count drift")
    verify_registered_data_content(protocol)
    checkpoint = Path(protocol["source_checkpoint"]["path"])
    normalized_checkpoint = str(checkpoint.resolve()).replace("\\", "/").lower()
    if (
        checkpoint.resolve() != OFFICIAL_CHECKPOINT.resolve()
        or "p1_factorial_r19" in normalized_checkpoint
        or "yolo-master-r19" in normalized_checkpoint
    ):
        raise ValueError(f"r20 must start from official yolo26n.pt, never r19 weights: {checkpoint}")
    if protocol["source_checkpoint"].get("sha256") != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("protocol does not name the locked official yolo26n.pt SHA-256")
    if sha256(checkpoint) != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("source checkpoint hash differs from protocol")
    output_root = Path(protocol["run_root"]) / "initializers" / f"seed{args.seed}"
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite initializer directory: {output_root}")
    output_root.mkdir(parents=True)
    calibration = protocol["batch_norm"]
    image_list = Path(calibration["image_list"])
    if sha256(image_list) != calibration["image_list_sha256"]:
        raise ValueError("BN calibration image-list hash differs from protocol")
    image_paths = [line.strip() for line in image_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(image_paths) != calibration["images"]:
        raise ValueError("BN calibration image count differs from protocol")
    if not torch.cuda.is_available():
        raise RuntimeError("r20 initializer calibration requires CUDA")
    device = torch.device(f"cuda:{args.device}")

    wrappers, sources, build_reports = build_models(protocol, args.seed)
    dense_calibration = calibrate_factor_batch_norm(
        wrappers["a"].model,
        image_paths,
        batch=calibration["batch"],
        imgsz=calibration["imgsz"],
        device=device,
    )
    wrappers["a"].model.to("cpu")
    clone_factor_state(wrappers["a"].model, wrappers["b"].model)
    moe_calibration = calibrate_factor_batch_norm(
        wrappers["c"].model,
        image_paths,
        batch=calibration["batch"],
        imgsz=calibration["imgsz"],
        device=device,
    )
    sigma_policy = protocol["routing"]["train_only_private_exploration"]
    sigma_reports = calibrate_router_sigmas(
        wrappers["c"].model,
        image_paths,
        batch=calibration["batch"],
        imgsz=calibration["imgsz"],
        device=device,
        sigma_min=sigma_policy["sigma_min"],
        sigma_max=sigma_policy["sigma_max"],
    )
    wrappers["c"].model.to("cpu")
    clone_factor_state(wrappers["c"].model, wrappers["d"].model)
    if any(
        not torch.equal(left, right)
        for layer in FACTOR_LAYERS
        for left, right in zip(
            wrappers["c"].model.model[layer].factor.state_dict().values(),
            wrappers["d"].model.model[layer].factor.state_dict().values(),
        )
    ):
        raise ValueError("calibrated C/D factor tensors are not identical")

    manifest = {
        "schema_version": 6,
        "status": "building",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "seed": args.seed,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "helper_dependency": {
            "path": str(Path(__file__).with_name("build_p1_residual_factor_initializers.py").resolve()),
            "sha256": sha256(Path(__file__).with_name("build_p1_residual_factor_initializers.py").resolve()),
        },
        "router_initialization": protocol["router_initialization"],
        "routing_auxiliary_objective": EXPECTED_CLEAN_AUX_POLICY,
        "bn_calibration": {"dense": dense_calibration, "moe": moe_calibration},
        "router_sigma_calibration": sigma_reports,
        "cells": {},
    }
    sample = torch.rand(1, 3, 64, 64)
    reloaded_models = {}
    for cell in CELLS:
        wrapper = wrappers[cell]
        target = wrapper.model.float().eval()
        source = sources[cell].float().eval()
        before = equivalence_report(source, target, sample)
        before["passed"] = bool(
            before.get("passed") and before.get("max_abs_error", float("inf")) <= args.equivalence_tolerance
        )
        if not before["passed"]:
            raise ValueError(f"{cell}: pre-save equivalence failed: {before}")
        frozen_base = sum(
            parameter.numel()
            for layer_index in FACTOR_LAYERS
            for parameter in target.model[layer_index].base.parameters()
        )
        if frozen_base != protocol["factor_base_expected_parameters"]:
            raise ValueError(f"{cell}: frozen base count {frozen_base} != expected")
        trainable_base = sum(
            parameter.numel()
            for layer_index in FACTOR_LAYERS
            for parameter in target.model[layer_index].base.parameters()
            if parameter.requires_grad
        )
        if trainable_base != 0:
            raise ValueError(f"{cell}: {trainable_base} official base parameters remain trainable")
        output = output_root / f"{cell}_residual_factor_init.pt"
        wrapper.ckpt = {}
        wrapper.save(output)
        from ultralytics.nn.tasks import load_checkpoint

        reloaded, _ = load_checkpoint(output, device="cpu")
        reloaded = reloaded.float().eval()
        reloaded_trainable_base = sum(
            parameter.numel()
            for layer_index in FACTOR_LAYERS
            for parameter in reloaded.model[layer_index].base.parameters()
            if parameter.requires_grad
        )
        if reloaded_trainable_base != 0:
            raise ValueError(f"{cell}: reload restored {reloaded_trainable_base} trainable base parameters")
        after = equivalence_report(source, reloaded, sample)
        after["passed"] = bool(
            after.get("passed") and after.get("max_abs_error", float("inf")) <= args.equivalence_tolerance
        )
        if not after["passed"]:
            raise ValueError(f"{cell}: reload equivalence failed: {after}")
        reloaded_models[cell] = reloaded
        manifest["cells"][cell] = {
            **build_reports[cell],
            "moe": cell in "cd",
            "end2end": cell in "bd",
            "initializer": str(output),
            "initializer_sha256": sha256(output),
            "frozen_factor_base_parameters": frozen_base,
            "trainable_factor_base_parameters": trainable_base,
            "reloaded_trainable_factor_base_parameters": reloaded_trainable_base,
            "equivalence_before_save": before,
            "equivalence_after_reload": after,
            "factor_state_sha256": {str(layer): state_hash(target.model[layer].factor) for layer in FACTOR_LAYERS},
            "reloaded_factor_state_sha256": {
                str(layer): state_hash(reloaded.model[layer].factor) for layer in FACTOR_LAYERS
            },
            "router_sigma0": [float(core.routing.p1_noise_sigma0.item()) for _, core in routed_cores(reloaded)],
            "r20_clean_aux_routers": [
                report
                for layer_index in FACTOR_LAYERS
                for report in clean_aux_router_report(
                    reloaded.model[layer_index].factor,
                    expected=cell in "cd",
                )
            ],
        }

    for left, right, label in (("a", "b", "A/B"), ("c", "d", "C/D")):
        for layer_index in FACTOR_LAYERS:
            left_state = reloaded_models[left].model[layer_index].factor.state_dict()
            right_state = reloaded_models[right].model[layer_index].factor.state_dict()
            if left_state.keys() != right_state.keys() or any(
                not torch.equal(left_state[name], right_state[name]) for name in left_state
            ):
                raise ValueError(f"reloaded {label} factor parity failed at layer {layer_index}")
    manifest["paired_factor_tensor_parity_after_reload"] = {"a_b": True, "c_d": True}
    manifest["a_b_dense_clean_aux_not_applicable"] = all(
        not manifest["cells"][cell]["r20_clean_aux_routers"] for cell in "ab"
    )
    manifest["c_d_clean_aux_policy_parity"] = (
        manifest["cells"]["c"]["r20_clean_aux_routers"]
        == manifest["cells"]["d"]["r20_clean_aux_routers"]
    )
    if not manifest["a_b_dense_clean_aux_not_applicable"] or not manifest["c_d_clean_aux_policy_parity"]:
        raise ValueError("r20 dense invariance or C/D clean-aux policy parity failed")
    manifest["status"] = "passed"
    manifest_path = output_root / "initialization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
