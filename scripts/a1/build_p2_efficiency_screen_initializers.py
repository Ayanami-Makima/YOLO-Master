"""Build zero-gated, function-preserving initializers for the P2 screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from tensor_leaves(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensor_leaves(item)


def equivalence_report(source, target, inputs: torch.Tensor) -> dict:
    source.eval()
    target.eval()
    with torch.inference_mode():
        expected = list(tensor_leaves(source(inputs)))
        actual = list(tensor_leaves(target(inputs)))
    if len(expected) != len(actual):
        return {"passed": False, "reason": f"tensor leaf count {len(expected)} != {len(actual)}"}
    maximum = 0.0
    total = 0.0
    count = 0
    for left, right in zip(expected, actual):
        if left.shape != right.shape:
            return {"passed": False, "reason": f"shape mismatch {left.shape} != {right.shape}"}
        difference = (left.float() - right.float()).abs()
        maximum = max(maximum, float(difference.max()) if difference.numel() else 0.0)
        total += float(difference.sum())
        count += difference.numel()
    return {"passed": maximum == 0.0, "tensor_leaves": len(expected), "max_abs_error": maximum,
            "mean_abs_error": total / max(count, 1), "input_shape": list(inputs.shape)}


def router_modules(module):
    return [
        (name, child)
        for name, child in module.named_modules()
        if hasattr(child, "progressive_sparsity") and hasattr(child, "routing")
    ]


def initialize_router_projection(module, seed: int) -> dict:
    """Use a deterministic, centered projection without validation-data calibration."""
    reports = []
    for index, (name, child) in enumerate(router_modules(module)):
        stack = getattr(child.routing, "router", None)
        projections = [item for item in stack.modules() if isinstance(item, torch.nn.Conv2d)]
        if not projections:
            raise TypeError(f"router {name} has no Conv2d projection")
        projection = projections[-1]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + index)
        with torch.no_grad():
            weight = torch.randn(projection.weight.shape, generator=generator, dtype=torch.float32)
            weight -= weight.mean(dim=0, keepdim=True)
            weight *= 0.05 / weight.float().square().mean().sqrt().clamp_min(1e-12)
            projection.weight.copy_(weight.to(projection.weight.device, projection.weight.dtype))
            if projection.bias is not None:
                projection.bias.zero_()
        reports.append({"name": name, "num_experts": int(child.num_experts), "top_k": int(child.top_k)})
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260829)
    args = parser.parse_args()
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    checkpoint = Path(protocol["source_checkpoint"]["path"])
    run_root = Path(protocol["run_root"])
    if sha256(checkpoint) != protocol["source_checkpoint"]["sha256"]:
        raise ValueError("source checkpoint hash differs from protocol")
    output_root = run_root / "initializers"
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    sys.path.insert(0, str(repo))
    from ultralytics import YOLO
    from ultralytics.nn.modules.moe.factor_adapter import C3k2ResidualFactor
    from ultralytics.nn.tasks import load_checkpoint

    native, _ = load_checkpoint(checkpoint, device="cpu")
    native = native.float()
    native_state = native.state_dict()
    # A standard ratio-2 dense factor supplies the compatible layer-4/6
    # weights and the per-expert transplant for every ratio-2 MoE target.
    from prepare_p2_efficiency_screen import make_config

    parent = yaml.safe_load(Path(protocol["parent"]["path"]).read_text(encoding="utf-8"))
    dense_reference_path = run_root / "dense_reference_ratio2.yaml"
    dense_reference_path.write_text(
        yaml.safe_dump(
            make_config(parent, {"moe": False, "experts": 0, "top_k": 0, "factor_mlp_ratio": 2.0}),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    dense_reference = YOLO(str(dense_reference_path), task="detect").model
    dense_factors = {index: copy.deepcopy(dense_reference.model[index].factor) for index in (4, 6, 8)}
    sample = torch.rand(1, 3, 64, 64)
    manifest = {
        "schema": "a1-p2-late-moe-efficiency-screen-initializers/v1",
        "status": "building",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "seed": args.seed,
        "variants": {},
    }
    for name, spec in protocol["variants"].items():
        set_seed(args.seed)
        source = copy.deepcopy(native)
        target = YOLO(spec["path"], task="detect").model
        source.end2end = True
        source.model[-1].end2end = True
        target.end2end = True
        target.model[-1].end2end = True
        target_state = target.state_dict()
        compatible = {
            key: value for key, value in native_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        target.load_state_dict(compatible, strict=False)
        layers = {}
        for layer_index in (4, 6, 8):
            adapter = target.model[layer_index]
            if not isinstance(adapter, C3k2ResidualFactor):
                raise TypeError(f"layer {layer_index}: {type(adapter).__name__}")
            adapter.base.load_state_dict(native.model[layer_index].state_dict(), strict=True)
            adapter.freeze_base_parameters()
            transplant = None
            router_report = []
            if layer_index in (4, 6):
                adapter.factor.load_state_dict(dense_factors[layer_index].state_dict(), strict=True)
            elif spec["moe"]:
                dense_factor = dense_factors[layer_index]
                # Copy the dense ratio-2 factor into each expert; the shared
                # expert is zeroed by the existing transplant helper.
                from build_p1_residual_factor_initializers import transplant_dense_factor

                transplant = transplant_dense_factor(dense_factor, adapter.factor)
                router_report = initialize_router_projection(adapter.factor, args.seed + layer_index * 1000)
            else:
                # Equal-compute Dense controls intentionally keep a random
                # factor at layer 8 while their zero gate preserves the native
                # checkpoint output exactly at initialization.
                pass
            layers[str(layer_index)] = {
                "factor_type": type(adapter.factor).__name__,
                "factor_mlp_ratio": float(getattr(adapter, "factor_mlp_ratio", 2.0)),
                "moe": bool(spec["moe"] and layer_index == 8),
                "experts": int(spec["experts"] or 0) if layer_index == 8 else 0,
                "top_k": int(spec["top_k"] or 0) if layer_index == 8 else 0,
                "base_parameters": sum(item.numel() for item in adapter.base.parameters()),
                "factor_parameters": sum(item.numel() for item in adapter.factor.parameters()),
                "gain_nonzero": int(torch.count_nonzero(adapter.gain)),
                "dense_to_moe_transplant": transplant,
                "routers": router_report,
            }
        comparison = equivalence_report(source, target, sample)
        if not comparison["passed"]:
            raise ValueError(f"{name}: initializer is not function-preserving: {comparison}")
        output = output_root / f"{name}.pt"
        target.ckpt = {}
        yolo = YOLO(spec["path"], task="detect")
        yolo.model = target
        yolo.ckpt = {}
        yolo.save(output)
        reloaded, _ = load_checkpoint(output, device="cpu")
        reloaded_comparison = equivalence_report(source, reloaded.float(), sample)
        if not reloaded_comparison["passed"]:
            raise ValueError(f"{name}: reloaded initializer is not function-preserving: {reloaded_comparison}")
        manifest["variants"][name] = {
            "initializer": str(output),
            "initializer_sha256": sha256(output),
            "parameters": sum(value.numel() for value in target.state_dict().values()),
            "compatible_source_tensors": len(compatible),
            "layers": layers,
            "equivalence_before_save": comparison,
            "equivalence_after_reload": reloaded_comparison,
        }
    manifest["status"] = "passed"
    (output_root / "initialization_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
