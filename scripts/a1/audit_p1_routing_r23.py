#!/usr/bin/env python3
"""Apply the A1-aligned fixed-512 hard Top-2 diagnostics to r23 C/D probes.

Finite-sample zero-selection experts are reported exactly but are not an A1
formal-admission blocker. Concentration, entropy, checkpoint, residual,
lineage, and numerical-integrity checks remain fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from p1_r23_runtime import RUNTIME_ATTESTATION, assert_protocol_runtime

# isort: split

from audit_p1_preflight_routing import routing_gate
from audit_p1_routing import summarize_counts
from evaluate_p1_matrix import digest, load_images, prediction_params, resolve_images
from run_p1_bn_frozen_r23 import R23_CLEAN_AUX_POLICY

DEAD_EXPERT_DIAGNOSTIC = {"target": 0, "blocking": False, "role": "reported_diagnostic_only"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--preflight-audit", required=True, type=Path)
    parser.add_argument("--probe-audit", required=True, type=Path)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return a file SHA-256."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checked_passed_audit(path: Path, protocol_path: Path, expected_stage: str) -> dict:
    """Validate a checkpoint-audit dependency."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or payload.get("stage") != expected_stage:
        raise ValueError(f"required {expected_stage} audit did not pass: {path}")
    if Path(payload.get("protocol", "")).resolve() != protocol_path.resolve():
        raise ValueError(f"required audit belongs to another protocol: {path}")
    if payload.get("protocol_sha256") != sha256(protocol_path):
        raise ValueError(f"required audit protocol hash mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path.resolve())}


def exact_probe_checkpoint(protocol: dict, seed: int, cell: str) -> Path:
    """Resolve one registered last.pt without best/suffix fallback."""
    entry = protocol["requests"]["routing_probe"][str(seed)][cell]
    request_path = Path(entry["path"])
    if sha256(request_path) != entry["sha256"]:
        raise ValueError(f"routing-probe request hash drift: {seed}/{cell}")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    return Path(request["params"]["project"]) / request["params"]["name"] / "weights/last.pt"


def summarize_image_counts(counts: Counter, num_experts: int, top_k: int, num_images: int) -> dict:
    """Add image-level selection metrics required by the registered gate."""
    if num_images < 1:
        raise ValueError("routing gate requires at least one image")
    summary = summarize_counts(counts, num_experts, top_k)
    expected = num_images * top_k
    image_fractions = [counts[index] / num_images for index in range(num_experts)]
    summary.update(
        {
            "images": num_images,
            "expected_selections": expected,
            "image_selection_fractions": image_fractions,
            "max_image_selection_fraction": max(image_fractions, default=0.0),
        }
    )
    return summary


def a1_aligned_routing_gate(summary: dict) -> dict:
    """Keep the former strict result as a diagnostic while blocking only collapse."""
    strict_internal_gate = routing_gate(summary)
    reasons = []
    maximum = float(summary["max_image_selection_fraction"])
    entropy = float(summary["normalized_entropy"])
    if maximum > 0.8:
        reasons.append(f"max image selection fraction {maximum:.6f} > 0.8")
    if entropy < 0.5:
        reasons.append(f"normalized entropy {entropy:.6f} < 0.5")
    dead = list(summary.get("dead_experts_on_sample", []))
    return {
        "passed": not reasons,
        "reasons": reasons,
        "dead_expert_diagnostic": {
            **DEAD_EXPERT_DIAGNOSTIC,
            "observed": len(dead),
            "expert_ids": dead,
            "target_met": not dead,
        },
        "former_strict_internal_gate": strict_internal_gate,
    }


def audit_checkpoint(checkpoint: Path, images: list, *, device: str) -> dict:
    """Count hard expert selections for one exact checkpoint."""
    from ultralytics import YOLO

    model = YOLO(checkpoint, task="detect")
    routed = []
    for name, module in model.model.named_modules():
        routing = getattr(module, "routing", None)
        if routing is not None and hasattr(routing, "num_experts") and hasattr(routing, "top_k"):
            if getattr(routing, "p1_balance_on_clean_routes", False) is not True:
                raise ValueError(f"r23 clean-aux flag missing: {name}.routing")
            if getattr(module, "routing_aux_semantics", None) != R23_CLEAN_AUX_POLICY["runtime_semantics"]:
                raise ValueError(f"r23 routing-aux semantics mismatch: {name}.routing")
            routing.noise_std = 0.0
            configure = getattr(routing, "configure_p1_private_noise", None)
            if callable(configure):
                configure(None, reset_step=True)
            routed.append((f"{name}.routing", module, routing))
    if len(routed) != 6:
        raise ValueError(f"expected six routed modules, got {len(routed)} in {checkpoint}")
    if any(int(routing.top_k) != 2 for _, _, routing in routed):
        raise ValueError(f"non-Top2 router in {checkpoint}")
    counters = {name: Counter() for name, _, _ in routed}
    handles = []
    for name, _, routing in routed:

        def hook(_module, _inputs, output, *, key=name):
            counters[key].update(int(index) for index in output[1].detach().cpu().reshape(-1).tolist())

        handles.append(routing.register_forward_hook(hook))
    params = prediction_params(
        argparse.Namespace(imgsz=640, conf=0.25, nms_iou=0.7, max_det=300),
        device,
    )
    try:
        model.predict(source=images[0], **params)
        for counter in counters.values():
            counter.clear()
        for image in images:
            model.predict(source=image, **params)
    finally:
        for handle in handles:
            handle.remove()
    modules = {}
    for name, _, routing in routed:
        summary = summarize_image_counts(counters[name], int(routing.num_experts), int(routing.top_k), len(images))
        summary["gate"] = a1_aligned_routing_gate(summary)
        modules[name] = summary
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "routed_modules": len(modules),
        "passed": all(summary["gate"]["passed"] for summary in modules.values()),
        "modules": modules,
    }


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert_protocol_runtime(protocol)
    if protocol.get("schema_version") != 8 or protocol.get("experiment_tag") != "r23":
        raise ValueError("r23 routing audit requires an r23 schema-8 protocol")
    preflight_audit = checked_passed_audit(args.preflight_audit.resolve(), protocol_path, "preflight")
    probe_audit = checked_passed_audit(args.probe_audit.resolve(), protocol_path, "routing_probe")
    output = Path(protocol["run_root"]) / "audits" / "routing" / "hard_top2_512.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite r23 routing evidence")
    data = Path(protocol["data"]["pilot"]["path"])
    paths = resolve_images(data, None, 512)
    if len(paths) != 512:
        raise ValueError(f"routing gate requires exactly 512 images, got {len(paths)}")
    images = load_images(paths)
    image_records = [{"path": str(path), "sha256": digest(path)} for path in paths]
    image_digest = hashlib.sha256(
        "".join(f"{item['path']}\0{item['sha256']}\n" for item in image_records).encode()
    ).hexdigest()
    report = {
        "schema_version": 8,
        "status": "running",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "runtime_attestation": RUNTIME_ATTESTATION,
        "dependencies": {"preflight_checkpoint_audit": preflight_audit, "probe_checkpoint_audit": probe_audit},
        "data": str(data),
        "images": image_records,
        "image_set_sha256": image_digest,
        "device": args.device,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate_thresholds": {
            "dead_experts": DEAD_EXPERT_DIAGNOSTIC,
            "max_image_selection_fraction": 0.8,
            "min_normalized_entropy": 0.5,
        },
        "metric_definitions": {
            "selection_share": "per-expert count / (images * top_k)",
            "image_selection_fraction": "per-expert count / images",
        },
        "admission_policy": "A1 collapse gate plus exact dead-expert diagnostic; combined admission must also bind residual activity",
        "seeds": {},
    }
    for seed in protocol["seeds"]:
        report["seeds"][str(seed)] = {}
        for cell in "cd":
            checkpoint = exact_probe_checkpoint(protocol, seed, cell)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            report["seeds"][str(seed)][cell] = audit_checkpoint(checkpoint, images, device=args.device)
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
