#!/usr/bin/env python3
"""Apply the fixed-512 hard Top-2 gate to all r19 C/D probe checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from audit_p1_preflight_routing import routing_gate
from audit_p1_routing import summarize_counts
from evaluate_p1_matrix import digest, load_images, prediction_params, resolve_images


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
    return {"path": str(path.resolve()), "sha256": sha256(path.resolve())}


def exact_probe_checkpoint(protocol: dict, seed: int, cell: str) -> Path:
    """Resolve one registered last.pt without best/suffix fallback."""
    request_path = Path(protocol["requests"]["routing_probe"][str(seed)][cell]["path"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    return Path(request["params"]["project"]) / request["params"]["name"] / "weights/last.pt"


def audit_checkpoint(checkpoint: Path, images: list, *, device: str) -> dict:
    """Count hard expert selections for one exact checkpoint."""
    from ultralytics import YOLO

    model = YOLO(checkpoint, task="detect")
    routed = []
    for name, module in model.model.named_modules():
        routing = getattr(module, "routing", None)
        if routing is not None and hasattr(routing, "num_experts") and hasattr(routing, "top_k"):
            routing.noise_std = 0.0
            configure = getattr(routing, "configure_p1_private_noise", None)
            if callable(configure):
                configure(None, reset_step=True)
            routed.append((f"{name}.routing", routing))
    if len(routed) != 6:
        raise ValueError(f"expected six routed modules, got {len(routed)} in {checkpoint}")
    if any(int(routing.top_k) != 2 for _, routing in routed):
        raise ValueError(f"non-Top2 router in {checkpoint}")
    counters = {name: Counter() for name, _ in routed}
    handles = []
    for name, routing in routed:

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
    for name, routing in routed:
        summary = summarize_counts(counters[name], int(routing.num_experts), int(routing.top_k), len(images))
        summary["gate"] = routing_gate(summary)
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
    preflight_audit = checked_passed_audit(args.preflight_audit.resolve(), protocol_path, "preflight")
    probe_audit = checked_passed_audit(args.probe_audit.resolve(), protocol_path, "routing_probe")
    output = Path(protocol["run_root"]) / "audits" / "routing" / "hard_top2_512.json"
    admission_path = Path(protocol["run_root"]) / "audits" / "formal_admission.json"
    if output.exists() or admission_path.exists():
        raise FileExistsError("refusing to overwrite r19 routing/admission evidence")
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
        "schema_version": 5,
        "status": "running",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "dependencies": {"preflight_checkpoint_audit": preflight_audit, "probe_checkpoint_audit": probe_audit},
        "data": str(data),
        "images": image_records,
        "image_set_sha256": image_digest,
        "device": args.device,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate_thresholds": {
            "dead_experts": 0,
            "max_image_selection_fraction": 0.8,
            "min_normalized_entropy": 0.5,
        },
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
    if report["status"] == "passed":
        initializer_manifests = {}
        for seed in protocol["seeds"]:
            path = Path(protocol["run_root"]) / "initializers" / f"seed{seed}" / "initialization_manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "passed":
                raise ValueError(f"initializer manifest did not pass: {path}")
            initializer_manifests[str(seed)] = {"path": str(path), "sha256": sha256(path)}
        admission = {
            "schema_version": 5,
            "status": "passed",
            "protocol": str(protocol_path),
            "protocol_sha256": sha256(protocol_path),
            "initializer_manifests": initializer_manifests,
            "preflight_checkpoint_audit": preflight_audit,
            "routing_probe_checkpoint_audit": probe_audit,
            "routing_gate": {"path": str(output), "sha256": sha256(output)},
            "probe_weights_disposition": "discarded; formal requests point to original seed initializers",
            "formal_may_start": True,
        }
        admission_path.write_text(json.dumps(admission, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
