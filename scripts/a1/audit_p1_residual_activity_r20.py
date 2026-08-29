#!/usr/bin/env python3
"""Audit r20 residual-factor activity on the fixed pilot-validation sample.

This diagnostic treats every routing-probe checkpoint as immutable. It evaluates
the exact C/D ``last.pt`` files registered by the r20 protocol and measures, for
layers 4, 6, and 8,

    sqrt(sum((gain * factor(base(x))) ** 2) / sum(base(x) ** 2)).

Input identity and hashes are checked before any checkpoint is loaded. The
hard-Top2 routing audit is also checked so this script uses exactly the same 512
image paths, per-image hashes, and aggregate image-set hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from run_p1_bn_frozen_r20 import R20_CLEAN_AUX_POLICY

CELLS = ("c", "d")
FACTOR_LAYERS = (4, 6, 8)
EXPECTED_IMAGE_COUNT = 512
EXPECTED_SEED_COUNT = 3
EXPECTED_PROTOCOL_SCHEMA = 6
MINIMUM_INCLUSIVE = 1e-4
MAXIMUM_EXCLUSIVE = 0.1
EXPECTED_METRIC = "sqrt(sum((gain * factor(base(x)))^2) / sum(base(x)^2)) over all sampled activations"


def parse_args() -> argparse.Namespace:
    """Parse explicit, immutable evidence inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--probe-checkpoint-audit", required=True, type=Path)
    parser.add_argument(
        "--routing-audit",
        type=Path,
        help="Hard-Top2 audit to bind the image/checkpoint hashes; defaults under protocol.run_root.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    """Raise immediately when immutable evidence is missing or inconsistent."""
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    """Read one required JSON object."""
    require(path.is_file(), f"missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{label} must contain a JSON object: {path}")
    return payload


def checked_file(path: Path, expected_sha256: str, label: str) -> str:
    """Require one file to match its registered digest."""
    require(path.is_file(), f"missing {label}: {path}")
    actual = sha256(path)
    require(bool(expected_sha256) and actual == expected_sha256, f"{label} SHA-256 mismatch: {path}")
    return actual


def same_path(left: str | Path, right: str | Path) -> bool:
    """Compare absolute filesystem identities rather than path spelling."""
    return Path(left).resolve() == Path(right).resolve()


def validate_protocol(protocol_path: Path, protocol: dict[str, Any]) -> tuple[list[int], Path]:
    """Validate r20 identity, hash-locked inputs, and C/D probe requests."""
    require(protocol.get("schema_version") == EXPECTED_PROTOCOL_SCHEMA, "protocol schema is not r20 schema 6")
    require(protocol.get("experiment_tag") == "r20", "protocol experiment_tag is not r20")
    require(protocol.get("primary_checkpoint") == "last.pt", "r20 primary checkpoint must be last.pt")
    gate = protocol.get("residual_activity_gate", {})
    require(gate.get("checkpoint") == "routing-probe last.pt", "residual checkpoint selection drift")
    require(gate.get("metric") == EXPECTED_METRIC, "residual-activity metric drift")
    require(gate.get("minimum_inclusive") == MINIMUM_INCLUSIVE, "residual minimum threshold drift")
    require(gate.get("maximum_exclusive") == MAXIMUM_EXCLUSIVE, "residual maximum threshold drift")
    seeds = protocol.get("seeds")
    require(
        isinstance(seeds, list)
        and len(seeds) == EXPECTED_SEED_COUNT
        and len(set(seeds)) == EXPECTED_SEED_COUNT
        and all(isinstance(seed, int) for seed in seeds),
        "protocol must register exactly three unique integer seeds",
    )
    probe_requests = protocol.get("requests", {}).get("routing_probe", {})
    require(set(probe_requests) == {str(seed) for seed in seeds}, "routing-probe seed registry differs from protocol")

    pilot = protocol.get("data", {}).get("pilot", {})
    pilot_path = Path(pilot.get("path", "")).resolve()
    checked_file(pilot_path, pilot.get("sha256", ""), "pilot data YAML")
    source = protocol.get("source_checkpoint", {})
    checked_file(Path(source.get("path", "")).resolve(), source.get("sha256", ""), "source checkpoint")
    parent = protocol.get("parent", {})
    parent_path = Path(parent.get("path", "")).resolve()
    checked_file(parent_path, parent.get("sha256", ""), "parent model YAML")

    first_entry = probe_requests[str(seeds[0])][CELLS[0]]
    first_request_path = Path(first_entry["path"]).resolve()
    checked_file(first_request_path, first_entry["sha256"], "first routing-probe request")
    first_request = read_json(first_request_path, "first routing-probe request")
    repo = Path(first_request.get("runtime", {}).get("cwd", "")).resolve()
    require(repo.is_dir(), f"registered r20 repository is missing: {repo}")
    # Only model-runtime files participate in this read-only measurement. Evidence
    # composers/auditors may be versioned after protocol lock without changing a
    # checkpoint's forward semantics, so do not make this audit depend on their
    # source-file hashes.
    runtime_implementation = {
        relative: expected
        for relative, expected in protocol.get("implementation", {}).items()
        if relative.startswith("ultralytics/")
    }
    require(bool(runtime_implementation), "protocol has no hash-locked model-runtime implementation")
    for relative, expected in runtime_implementation.items():
        checked_file(repo / relative, expected, f"locked model runtime {relative}")

    for seed in seeds:
        cells = probe_requests.get(str(seed), {})
        require(set(cells) == set(CELLS), f"routing-probe cells differ for seed {seed}")
        for cell in CELLS:
            entry = cells[cell]
            request_path = Path(entry["path"]).resolve()
            checked_file(request_path, entry.get("sha256", ""), f"routing-probe request {seed}/{cell}")
            request = read_json(request_path, f"routing-probe request {seed}/{cell}")
            params = request.get("params", {})
            require(request.get("skill") == "yolo.train", f"{seed}/{cell}: request is not yolo.train")
            require(params.get("seed") == seed, f"{seed}/{cell}: request seed drift")
            require(params.get("epochs") == 1, f"{seed}/{cell}: routing probe must have one epoch")
            require(params.get("device") == "0", f"{seed}/{cell}: request-visible device must be logical GPU0")
            require(params.get("exist_ok") is False, f"{seed}/{cell}: request must refuse overwrite")
            require(params.get("resume") is False, f"{seed}/{cell}: routing probe must not resume")
    require(Path(protocol_path).resolve().is_file(), "protocol disappeared during validation")
    return seeds, pilot_path


def validate_checkpoint_audit(
    audit_path: Path,
    audit: dict[str, Any],
    protocol_path: Path,
    protocol_sha256: str,
    seeds: list[int],
) -> dict[str, Any]:
    """Require the all-seed routing-probe checkpoint audit to have passed."""
    require(audit.get("schema_version") == EXPECTED_PROTOCOL_SCHEMA, "checkpoint-audit schema is not 6")
    require(audit.get("status") == "passed", "routing-probe checkpoint audit did not pass")
    require(audit.get("stage") == "routing_probe", "checkpoint audit is not for routing_probe")
    require(same_path(audit.get("protocol", ""), protocol_path), "checkpoint audit belongs to another protocol")
    if "protocol_sha256" in audit:
        require(audit["protocol_sha256"] == protocol_sha256, "checkpoint-audit protocol hash mismatch")
    seed_payloads = audit.get("seeds", {})
    require(set(seed_payloads) == {str(seed) for seed in seeds}, "checkpoint-audit seed set drift")
    for seed in seeds:
        cells = seed_payloads[str(seed)]
        require(set(cells) == set(CELLS), f"checkpoint-audit cell set drift for seed {seed}")
        for cell in CELLS:
            require(cells[cell].get("passed") is True, f"checkpoint audit failed for {seed}/{cell}")
    return {"path": str(audit_path), "sha256": sha256(audit_path)}


def resolve_images(data_yaml: Path, limit: int) -> list[Path]:
    """Reproduce the hard-Top2 audit's deterministic validation-image selection."""
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    require(isinstance(data, dict) and "val" in data, f"pilot YAML has no validation split: {data_yaml}")
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    entries = data["val"] if isinstance(data["val"], list) else [data["val"]]
    sources = [Path(entry) if Path(entry).is_absolute() else root / entry for entry in entries]
    images: list[Path] = []
    for source in sources:
        if source.suffix.lower() == ".txt":
            require(source.is_file(), f"validation list is missing: {source}")
            for line in source.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = Path(line.strip())
                    images.append((item if item.is_absolute() else source.parent / item).resolve())
        elif source.is_dir():
            images.extend(
                sorted(
                    item.resolve()
                    for item in source.iterdir()
                    if item.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
            )
        else:
            images.append(source.resolve())
    images = list(dict.fromkeys(images))[:limit]
    require(len(images) == limit, f"expected exactly {limit} validation images, got {len(images)}")
    require(all(path.is_file() for path in images), "one or more fixed validation images are missing")
    return images


def image_evidence(paths: list[Path]) -> tuple[list[dict[str, str]], str]:
    """Hash every selected image using the routing-audit canonical aggregate."""
    records = [{"path": str(path), "sha256": sha256(path)} for path in paths]
    aggregate = hashlib.sha256(
        "".join(f"{item['path']}\0{item['sha256']}\n" for item in records).encode()
    ).hexdigest()
    return records, aggregate


def validate_routing_audit(
    routing_path: Path,
    routing: dict[str, Any],
    protocol_path: Path,
    protocol_sha256: str,
    pilot_path: Path,
    seeds: list[int],
    records: list[dict[str, str]],
    image_set_sha256: str,
) -> dict[str, Any]:
    """Require exact parity with the hard-Top2 audit's image and checkpoint hashes."""
    require(routing.get("schema_version") == EXPECTED_PROTOCOL_SCHEMA, "routing-audit schema is not 6")
    require(same_path(routing.get("protocol", ""), protocol_path), "routing audit belongs to another protocol")
    require(routing.get("protocol_sha256") == protocol_sha256, "routing-audit protocol hash mismatch")
    require(same_path(routing.get("data", ""), pilot_path), "routing audit used another data YAML")
    require(routing.get("images") == records, "fixed image paths or per-image hashes differ from routing audit")
    require(routing.get("image_set_sha256") == image_set_sha256, "fixed image-set hash differs from routing audit")
    seed_payloads = routing.get("seeds", {})
    require(set(seed_payloads) == {str(seed) for seed in seeds}, "routing-audit seed set drift")
    for seed in seeds:
        require(set(seed_payloads[str(seed)]) == set(CELLS), f"routing-audit cell set drift for seed {seed}")
    return {
        "path": str(routing_path),
        "sha256": sha256(routing_path),
        "status": routing.get("status"),
        "image_set_sha256": image_set_sha256,
    }


def load_images(paths: list[Path]) -> list[Any]:
    """Decode the fixed images once, outside every model measurement."""
    import cv2

    images = []
    for path in paths:
        image = cv2.imread(str(path))
        require(image is not None, f"cannot decode fixed validation image: {path}")
        images.append(image)
    return images


def exact_probe_checkpoint(protocol: dict[str, Any], seed: int, cell: str) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve an exact registered last.pt without suffix or best-checkpoint fallback."""
    entry = protocol["requests"]["routing_probe"][str(seed)][cell]
    request_path = Path(entry["path"]).resolve()
    checked_file(request_path, entry["sha256"], f"routing-probe request {seed}/{cell}")
    request = read_json(request_path, f"routing-probe request {seed}/{cell}")
    run_dir = Path(request["params"]["project"]) / request["params"]["name"]
    checkpoint = (run_dir / "weights" / "last.pt").resolve()
    require(checkpoint.is_file(), f"missing routing-probe last.pt for {seed}/{cell}: {checkpoint}")
    return checkpoint, request_path, request


class LayerEnergy:
    """Accumulate base and gated-residual energy for one residual adapter."""

    def __init__(self, layer_index: int, adapter: torch.nn.Module) -> None:
        self.layer_index = layer_index
        self.adapter = adapter
        self.reset()

    def reset(self) -> None:
        """Discard warmup observations."""
        self.base_energy: torch.Tensor | None = None
        self.residual_energy: torch.Tensor | None = None
        self.base_calls = 0
        self.factor_calls = 0
        self.base_elements = 0
        self.residual_elements = 0
        self.base_shape: list[int] | None = None
        self.residual_shape: list[int] | None = None
        self.requires_grad_seen = False

    @staticmethod
    def _energy(value: torch.Tensor) -> torch.Tensor:
        return value.detach().float().square().sum(dtype=torch.float64)

    def capture_base(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        """Capture ``base(x)`` during the native model forward."""
        require(isinstance(output, torch.Tensor), f"layer {self.layer_index}: base output is not a tensor")
        energy = self._energy(output)
        self.base_energy = energy if self.base_energy is None else self.base_energy + energy
        self.base_calls += 1
        self.base_elements += output.numel()
        self.base_shape = list(output.shape)
        self.requires_grad_seen |= output.requires_grad

    def capture_factor(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        """Capture the actually gated ``factor(base(x))`` residual."""
        require(isinstance(output, torch.Tensor), f"layer {self.layer_index}: factor output is not a tensor")
        gain = self.adapter.gain.detach().view(1, -1, 1, 1)
        require(output.ndim == 4 and output.shape[1] == gain.shape[1], f"layer {self.layer_index}: gain shape mismatch")
        residual = output.detach() * gain
        energy = self._energy(residual)
        self.residual_energy = energy if self.residual_energy is None else self.residual_energy + energy
        self.factor_calls += 1
        self.residual_elements += residual.numel()
        self.residual_shape = list(residual.shape)
        self.requires_grad_seen |= output.requires_grad or residual.requires_grad

    def report(self) -> dict[str, Any]:
        """Compute the registered RMS ratio and its inclusive/exclusive gate."""
        require(self.base_calls == EXPECTED_IMAGE_COUNT, f"layer {self.layer_index}: unexpected base call count")
        require(self.factor_calls == EXPECTED_IMAGE_COUNT, f"layer {self.layer_index}: unexpected factor call count")
        require(self.base_calls == self.factor_calls, f"layer {self.layer_index}: base/factor call mismatch")
        require(self.base_elements == self.residual_elements, f"layer {self.layer_index}: activation-size mismatch")
        require(not self.requires_grad_seen, f"layer {self.layer_index}: audit unexpectedly built an autograd graph")
        require(self.base_energy is not None and self.residual_energy is not None, "missing activity accumulators")
        base_energy = float(self.base_energy.item())
        residual_energy = float(self.residual_energy.item())
        require(math.isfinite(base_energy) and base_energy > 0.0, f"layer {self.layer_index}: invalid base energy")
        require(math.isfinite(residual_energy) and residual_energy >= 0.0, f"layer {self.layer_index}: invalid residual energy")
        ratio = math.sqrt(residual_energy / base_energy)
        passed = math.isfinite(ratio) and ratio >= MINIMUM_INCLUSIVE and ratio < MAXIMUM_EXCLUSIVE
        reasons = []
        if not math.isfinite(ratio):
            reasons.append("ratio is not finite")
        if ratio < MINIMUM_INCLUSIVE:
            reasons.append(f"ratio {ratio:.12g} is below inclusive minimum {MINIMUM_INCLUSIVE}")
        if ratio >= MAXIMUM_EXCLUSIVE:
            reasons.append(f"ratio {ratio:.12g} is not below exclusive maximum {MAXIMUM_EXCLUSIVE}")
        return {
            "metric": "sqrt(residual_energy_sum / base_energy_sum)",
            "ratio": ratio,
            "base_energy_sum": base_energy,
            "gated_residual_energy_sum": residual_energy,
            "base_calls": self.base_calls,
            "factor_calls": self.factor_calls,
            "activation_elements": self.base_elements,
            "last_base_shape": self.base_shape,
            "last_residual_shape": self.residual_shape,
            "gate": {
                "minimum_inclusive": MINIMUM_INCLUSIVE,
                "maximum_exclusive": MAXIMUM_EXCLUSIVE,
                "passed": passed,
                "reasons": reasons,
            },
            "passed": passed,
        }


def disable_routing_noise(model: torch.nn.Module) -> list[dict[str, Any]]:
    """Force every routed module to deterministic evaluation semantics in memory."""
    routers = []
    for name, module in model.named_modules():
        routing = getattr(module, "routing", None)
        if routing is None or not hasattr(routing, "num_experts") or not hasattr(routing, "top_k"):
            continue
        require(getattr(routing, "p1_balance_on_clean_routes", False) is True, f"missing clean-aux flag: {name}")
        require(
            getattr(module, "routing_aux_semantics", None) == R20_CLEAN_AUX_POLICY["runtime_semantics"],
            f"routing-aux semantics mismatch: {name}",
        )
        if hasattr(routing, "noise_std"):
            routing.noise_std = 0.0
        configure = getattr(routing, "configure_p1_private_noise", None)
        if callable(configure):
            configure(None, reset_step=True)
        if hasattr(module, "progressive_sparsity"):
            module.progressive_sparsity = False
        if hasattr(module, "_current_top_k"):
            module._current_top_k = int(module.top_k)
        if hasattr(module, "warmup_steps"):
            module.warmup_steps = 0
        if hasattr(module, "expert_dropout_rate"):
            module.expert_dropout_rate = 0.0
        require(getattr(routing, "noise_std", 0.0) == 0.0, f"router noise could not be disabled: {name}")
        require(getattr(routing, "p1_noise_seed", None) is None, f"private router noise remained enabled: {name}")
        routers.append(
            {
                "name": f"{name}.routing",
                "num_experts": int(routing.num_experts),
                "top_k": int(routing.top_k),
                "noise_std": float(getattr(routing, "noise_std", 0.0)),
                "private_noise_seed": getattr(routing, "p1_noise_seed", None),
            }
        )
    require(len(routers) == 6, f"expected six routed modules, got {len(routers)}")
    require(all(router["top_k"] == 2 for router in routers), "one or more routers are not hard Top-2")
    return routers


def prediction_params(device: str) -> dict[str, Any]:
    """Mirror the fixed routing-audit prediction settings."""
    return {
        "imgsz": 640,
        "batch": 1,
        "device": device,
        "conf": 0.25,
        "iou": 0.7,
        "max_det": 300,
        "half": False,
        "rect": False,
        "augment": False,
        "agnostic_nms": False,
        "verbose": False,
        "save": False,
        "save_txt": False,
        "save_crop": False,
        "show": False,
    }


def audit_checkpoint(
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    images: list[Any],
    *,
    seed: int,
    cell: str,
    device: str,
) -> dict[str, Any]:
    """Measure all three residual adapters without mutating the checkpoint."""
    from ultralytics import YOLO

    before_sha256 = checked_file(checkpoint, expected_checkpoint_sha256, f"routing-probe checkpoint {seed}/{cell}")
    yolo = YOLO(checkpoint, task="detect")
    core = yolo.model
    core.eval()
    require(bool(core.end2end) == (cell == "d"), f"{seed}/{cell}: checkpoint end2end path mismatch")
    routers = disable_routing_noise(core)
    accumulators: dict[int, LayerEnergy] = {}
    handles = []
    for layer_index in FACTOR_LAYERS:
        adapter = core.model[layer_index]
        require(type(adapter).__name__ == "C3k2ResidualFactor", f"layer {layer_index}: wrong adapter class")
        require(hasattr(adapter, "base") and hasattr(adapter, "factor") and hasattr(adapter, "gain"), "bad adapter")
        accumulator = LayerEnergy(layer_index, adapter)
        accumulators[layer_index] = accumulator
        handles.append(adapter.base.register_forward_hook(accumulator.capture_base))
        handles.append(adapter.factor.register_forward_hook(accumulator.capture_factor))
    params = prediction_params(device)
    try:
        with torch.no_grad():
            yolo.predict(source=images[0], **params)
            for accumulator in accumulators.values():
                accumulator.reset()
            for image in images:
                yolo.predict(source=image, **params)
    finally:
        for handle in handles:
            handle.remove()
    layers = {str(index): accumulators[index].report() for index in FACTOR_LAYERS}
    after_sha256 = sha256(checkpoint)
    require(after_sha256 == before_sha256, f"checkpoint changed during read-only audit: {checkpoint}")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": before_sha256,
        "checkpoint_sha256_after": after_sha256,
        "checkpoint_unchanged": True,
        "mode": "eval",
        "grad": False,
        "routing_noise_forced_to_zero": True,
        "routers": routers,
        "layers": layers,
        "passed": all(layer["passed"] for layer in layers.values()),
    }


def main() -> int:
    """Validate evidence, audit six checkpoints, and write one immutable report."""
    args = parse_args()
    protocol_path = args.protocol.resolve()
    checkpoint_audit_path = args.probe_checkpoint_audit.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite residual-activity evidence: {output}")

    protocol = read_json(protocol_path, "protocol")
    protocol_sha256 = sha256(protocol_path)
    seeds, pilot_path = validate_protocol(protocol_path, protocol)
    checkpoint_audit = read_json(checkpoint_audit_path, "routing-probe checkpoint audit")
    checkpoint_audit_identity = validate_checkpoint_audit(
        checkpoint_audit_path, checkpoint_audit, protocol_path, protocol_sha256, seeds
    )

    paths = resolve_images(pilot_path, EXPECTED_IMAGE_COUNT)
    image_records, image_set_sha256 = image_evidence(paths)
    routing_path = (
        args.routing_audit.resolve()
        if args.routing_audit
        else (Path(protocol["run_root"]) / "audits" / "routing" / "hard_top2_512.json").resolve()
    )
    routing_audit = read_json(routing_path, "hard-Top2 routing audit")
    routing_identity = validate_routing_audit(
        routing_path,
        routing_audit,
        protocol_path,
        protocol_sha256,
        pilot_path,
        seeds,
        image_records,
        image_set_sha256,
    )

    checkpoint_records: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        checkpoint_records[str(seed)] = {}
        for cell in CELLS:
            checkpoint, request_path, _request = exact_probe_checkpoint(protocol, seed, cell)
            audited = checkpoint_audit["seeds"][str(seed)][cell]
            routed = routing_audit["seeds"][str(seed)][cell]
            require(same_path(audited.get("checkpoint", ""), checkpoint), f"checkpoint-audit path drift for {seed}/{cell}")
            require(same_path(routed.get("checkpoint", ""), checkpoint), f"routing-audit path drift for {seed}/{cell}")
            expected_hash = routed.get("checkpoint_sha256", "")
            checked_file(checkpoint, expected_hash, f"routing-probe checkpoint {seed}/{cell}")
            checkpoint_records[str(seed)][cell] = {
                "checkpoint": checkpoint,
                "request": str(request_path),
                "request_sha256": sha256(request_path),
                "expected_checkpoint_sha256": expected_hash,
            }

    images = load_images(paths)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha256},
        "dependencies": {
            "routing_probe_checkpoint_audit": checkpoint_audit_identity,
            "hard_top2_routing_audit": routing_identity,
        },
        "sample": {
            "description": "same fixed pilot validation 512 used by the hard-Top2 routing audit",
            "data_yaml": str(pilot_path),
            "data_yaml_sha256": sha256(pilot_path),
            "images": image_records,
            "image_set_sha256": image_set_sha256,
            "matches_hard_top2_audit": True,
            "count": len(images),
        },
        "measurement": {
            "metric": EXPECTED_METRIC,
            "mode": "eval",
            "grad": False,
            "device": args.device,
            "batch": 1,
            "imgsz": 640,
            "routing_noise_std": 0.0,
            "checkpoint_write": False,
        },
        "gate_thresholds": {
            "minimum_inclusive": MINIMUM_INCLUSIVE,
            "maximum_exclusive": MAXIMUM_EXCLUSIVE,
            "applies_to": "every layer 4/6/8 in C/D for every registered seed",
        },
        "seeds": {},
    }
    for seed in seeds:
        report["seeds"][str(seed)] = {}
        for cell in CELLS:
            record = checkpoint_records[str(seed)][cell]
            result = audit_checkpoint(
                record["checkpoint"],
                record["expected_checkpoint_sha256"],
                images,
                seed=seed,
                cell=cell,
                device=args.device,
            )
            result["request"] = record["request"]
            result["request_sha256"] = record["request_sha256"]
            report["seeds"][str(seed)][cell] = result
    report["status"] = (
        "passed"
        if all(cell["passed"] for seed_payload in report["seeds"].values() for cell in seed_payload.values())
        else "failed"
    )
    report["formal_activity_gate_passed"] = report["status"] == "passed"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output)}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

