#!/usr/bin/env python3
"""Build segmentation initializers with an unchanged pretrained YOLO26 backbone."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from build_p1_residual_factor_initializers import (
    FACTOR_LAYERS,
    initialize_balanced_router_projections,
    require_deterministic_routing,
    routed_policy,
    set_end2end,
    set_seed,
    transplant_dense_factor,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shared_feature_report(source, target, sample: torch.Tensor, indices=(4, 6, 8, 10, 22)) -> dict:
    """Compare the common backbone/neck features, excluding task-specific heads."""
    captured = {}
    handles = []
    for index in indices:
        handles.append(source.model[index].register_forward_hook(lambda _m, _i, output, idx=index: captured.setdefault(("s", idx), output.detach().cpu())))
        handles.append(target.model[index].register_forward_hook(lambda _m, _i, output, idx=index: captured.setdefault(("t", idx), output.detach().cpu())))
    source.eval()
    target.eval()
    with torch.inference_mode():
        source(sample)
        target(sample)
    for handle in handles:
        handle.remove()
    rows = {}
    maximum = 0.0
    for index in indices:
        left, right = captured[("s", index)], captured[("t", index)]
        if left.shape != right.shape:
            return {"passed": False, "reason": f"layer {index} shape {list(left.shape)} != {list(right.shape)}"}
        error = float((left.float() - right.float()).abs().max()) if left.numel() else 0.0
        maximum = max(maximum, error)
        rows[str(index)] = {"shape": list(left.shape), "max_abs_error": error}
    return {"passed": maximum == 0.0, "max_abs_error": maximum, "layers": rows, "input_shape": list(sample.shape)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260829)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    checkpoint = Path(protocol["source_checkpoint"]["path"])
    run_root = Path(protocol["run_root"])
    output_root = run_root / "initializers"
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if sha256(checkpoint) != protocol["source_checkpoint"]["sha256"]:
        raise ValueError("source checkpoint hash differs from protocol")

    from ultralytics import YOLO
    from ultralytics.nn.tasks import load_checkpoint

    native, _ = load_checkpoint(checkpoint, device="cpu")
    native = native.float()
    native_state = native.state_dict()
    dense_reference = YOLO(protocol["configs"]["a"]["path"], task="segment").model
    dense_factors = {index: copy.deepcopy(dense_reference.model[index].factor) for index in FACTOR_LAYERS}
    sample = torch.rand(1, 3, 64, 64)
    manifest = {
        "schema": "a1-p2-seg-initializers-r1/v1", "status": "building", "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path), "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint), "seed": args.seed, "cells": {},
        "shared_feature_layers": list((4, 6, 8, 10, 22)), "equivalence_tolerance": 0.0,
    }
    for cell, factors in protocol["configs"].items():
        set_seed(args.seed)
        source = copy.deepcopy(native)
        yolo = YOLO(factors["path"], task="segment")
        target = yolo.model
        set_end2end(source, bool(factors["end2end"]))
        set_end2end(target, bool(factors["end2end"]))
        target_state = target.state_dict()
        compatible = {name: value for name, value in native_state.items() if name in target_state and target_state[name].shape == value.shape}
        incompatible = target.load_state_dict(compatible, strict=False)
        layer_reports = {}
        for layer_index in FACTOR_LAYERS:
            adapter = target.model[layer_index]
            if type(adapter).__name__ != "C3k2ResidualFactor":
                raise TypeError(f"layer {layer_index}: expected C3k2ResidualFactor, got {type(adapter).__name__}")
            adapter.base.load_state_dict(source.model[layer_index].state_dict(), strict=True)
            adapter.freeze_base_parameters()
            adapter.gain.data.zero_()
            if factors["moe"]:
                transplant = transplant_dense_factor(dense_factors[layer_index], adapter.factor)
                projections = initialize_balanced_router_projections(adapter.factor, base_seed=args.seed, layer_index=layer_index)
            else:
                adapter.factor.load_state_dict(dense_factors[layer_index].state_dict(), strict=True)
                transplant, projections = None, []
            routers = routed_policy(adapter.factor)
            if factors["moe"]:
                require_deterministic_routing(routers, layer_index)
            elif routers:
                raise ValueError(f"layer {layer_index}: dense cell unexpectedly has routers")
            layer_reports[str(layer_index)] = {
                "frozen_base_parameters": sum(p.numel() for p in adapter.base.parameters()),
                "trainable_factor_parameters": sum(p.numel() for p in adapter.factor.parameters()),
                "gain_nonzero": int(torch.count_nonzero(adapter.gain)), "routers": routers,
                "dense_to_moe_transplant": transplant, "router_initialization": projections,
            }
        equivalence = shared_feature_report(source, target, sample)
        if not equivalence["passed"]:
            raise ValueError(f"{cell}: shared feature equivalence failed: {equivalence}")
        output = output_root / f"{cell}_seg_init.pt"
        yolo.ckpt = {}
        yolo.save(output)
        reloaded, _ = load_checkpoint(output, device="cpu")
        reloaded_equivalence = shared_feature_report(source, reloaded.float(), sample)
        if not reloaded_equivalence["passed"]:
            raise ValueError(f"{cell}: reloaded feature equivalence failed: {reloaded_equivalence}")
        manifest["cells"][cell] = {
            "moe": bool(factors["moe"]), "end2end": bool(factors["end2end"]), "initializer": str(output),
            "initializer_sha256": sha256(output), "parameters": sum(value.numel() for value in target.state_dict().values()),
            "compatible_source_tensors": len(compatible), "missing_tensors": len(incompatible.missing_keys),
            "shared_feature_equivalence_before_save": equivalence, "shared_feature_equivalence_after_reload": reloaded_equivalence,
            "layers": layer_reports,
        }
    manifest["status"] = "passed"
    (output_root / "initialization_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
