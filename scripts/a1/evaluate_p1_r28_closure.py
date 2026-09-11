"""Collect missing P1 closure evidence from the immutable r28 epoch-15 checkpoints.

This wrapper reuses the already audited prediction, latency, and ONNX routines in
``scripts/a1/evaluate_p1_matrix.py`` while replacing its obsolete r1 training gate
with the recovery-aware r28 checkpoint audit.  It never trains or writes beside a
formal checkpoint.  Every stage is append-only and refuses to overwrite evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

SEEDS = ("260829", "260830", "260831")
CELLS = "abcd"
STAGES = ("prepare", "predict", "latency", "export", "resources", "summary")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def checked_file(path: Path, expected: str, label: str) -> dict:
    require(path.is_file(), f"missing {label}: {path}")
    actual = digest(path)
    require(actual == expected, f"{label} SHA-256 mismatch: {path}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def load_base(repo: Path):
    source = repo / "scripts/a1/evaluate_p1_matrix.py"
    require(source.is_file(), f"missing base evaluator: {source}")
    name = "a1_p1_matrix_evaluator_for_r28_closure"
    spec = importlib.util.spec_from_file_location(name, source)
    require(spec is not None and spec.loader is not None, f"cannot import base evaluator: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_val_images(protocol: dict, limit: int) -> list[Path]:
    formal = protocol["data"]["formal"]
    val = formal["lists"]["val"]
    listing = Path(val["path"])
    checked_file(listing, val["sha256"], "formal val list")
    lines = [line.strip() for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(lines) == val["images"] == 5000, "formal val list count changed")
    images = []
    for line in lines[:limit]:
        item = Path(line)
        if not item.is_absolute():
            item = listing.parent / item
        item = item.resolve()
        require(item.is_file(), f"missing selected val image: {item}")
        images.append(item)
    require(len(images) == limit, "not enough formal val images")
    return images


def verify_runtime_binding(repo: Path, protocol: dict) -> dict:
    results = {}
    for key, record in protocol["runtime_binding"]["modules"].items():
        path = Path(record["path"])
        require(path.resolve().is_relative_to(repo.resolve()), f"runtime module escapes r28 repo: {path}")
        results[key] = checked_file(path, record["sha256"], f"runtime module {key}")
    for relative, expected in protocol["implementation"].items():
        path = repo / relative
        results[f"implementation:{relative}"] = checked_file(path, expected, f"implementation {relative}")
    return results


def checkpoint_manifest(audit: dict) -> dict:
    require(audit.get("status") == "passed" and audit.get("failed_cells") == [], "r28 recovery audit did not pass")
    result = {}
    for seed in SEEDS:
        require(seed in audit["cells"], f"missing audited seed {seed}")
        result[seed] = {}
        for cell in CELLS:
            record = audit["cells"][seed][cell]
            require(record.get("status") == "passed", f"audit failed for seed {seed} cell {cell}")
            identity = record["identities"]["checkpoint"]
            path = Path(identity["path"])
            checked = checked_file(path, identity["actual_sha256"], f"seed {seed} cell {cell} last.pt")
            require(path.name == "last.pt", f"closure must use epoch-15 last.pt: {path}")
            result[seed][cell] = {
                **checked,
                "selection": "formal epoch-15 last.pt",
                "final_metrics": record.get("final_metrics"),
            }
    return result


def current_git(repo: Path) -> dict:
    def run(*items: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *items], text=True).strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "status_porcelain": run("status", "--porcelain"),
    }


def prepare(args, base) -> None:
    require(not args.output.exists(), f"refusing to overwrite closure root: {args.output}")
    repo = args.repo.resolve()
    audit_path = args.audit.resolve()
    source_protocol_path = args.protocol.resolve()
    audit = read_json(audit_path)
    protocol = read_json(source_protocol_path)
    require(Path(protocol["runtime_binding"]["repo_root"]).resolve() == repo, "r28 protocol repo binding mismatch")
    require(protocol.get("experiment_tag") == "r28", "not the locked r28 protocol")
    runtime_files = verify_runtime_binding(repo, protocol)
    checkpoints = checkpoint_manifest(audit)
    images = resolve_val_images(protocol, args.image_limit)
    args.output.mkdir(parents=True)
    tool_copy = args.output / "tools" / Path(__file__).name
    tool_copy.parent.mkdir()
    tool_copy.write_bytes(Path(__file__).read_bytes())
    base_source = Path(base.__file__).resolve()
    closure = {
        "schema": "a1-p1-r28-closure-protocol/v1",
        "status": "prepared",
        "mutates_formal_experiment": False,
        "checkpoint_selection": "epoch-15 last.pt for all 12 formal cells; no best-epoch cherry-picking",
        "repo": str(repo),
        "git": current_git(repo),
        "source_protocol": checked_file(source_protocol_path, digest(source_protocol_path), "r28 protocol"),
        "recovery_aware_audit": checked_file(audit_path, digest(audit_path), "r28 recovery-aware audit"),
        "runtime_files": runtime_files,
        "tools": {
            "closure": checked_file(tool_copy, digest(tool_copy), "copied closure tool"),
            "base_evaluator": checked_file(base_source, digest(base_source), "base evaluator"),
        },
        "checkpoints": checkpoints,
        "formal_val": protocol["data"]["formal"],
        "selected_images": [
            {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size} for path in images
        ],
        "predict": {
            "images": 16,
            "imgsz": 640,
            "batch": 1,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "half": False,
            "real_suppression_kernels_instrumented": True,
        },
        "latency": {
            "images_in_ram": 16,
            "imgsz": 640,
            "batch": 1,
            "conf": 0.25,
            "iou": 0.7,
            "max_det": 300,
            "half": False,
            "warmup": args.warmup,
            "samples": args.samples,
            "threads": args.threads,
            "cpu_affinity": args.affinity,
            "scope": "RAM image -> preprocess -> inference -> postprocess -> Results; excludes disk I/O/model load",
        },
        "export": {
            "images": 4,
            "format": "onnx",
            "imgsz": 640,
            "batch": 1,
            "opset": 17,
            "half": False,
            "dynamic": False,
            "simplify": False,
            "nms": False,
            "atol": args.export_atol,
            "rtol": args.export_rtol,
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
    }
    write_json(args.output / "protocol.json", closure)


def load_closure(args) -> dict:
    path = args.output / "protocol.json"
    require(path.is_file(), f"run prepare first: {path}")
    protocol = read_json(path)
    require(protocol.get("status") == "prepared", "closure protocol is not prepared")
    require(Path(protocol["repo"]).resolve() == args.repo.resolve(), "closure repo changed")
    require(protocol["git"]["status_porcelain"] == "", "r28 repo was dirty when closure protocol was prepared")
    require(current_git(args.repo)["status_porcelain"] == "", "r28 repo is now dirty")
    for seed, cells in protocol["checkpoints"].items():
        for cell, record in cells.items():
            checked_file(Path(record["path"]), record["sha256"], f"seed {seed} cell {cell} checkpoint")
    for record in protocol["selected_images"]:
        checked_file(Path(record["path"]), record["sha256"], "selected image")
    return protocol


def gate_for_seed(protocol: dict, seed: str) -> dict:
    return {
        "cells": {
            cell: {
                "weights": {
                    # The reused evaluator historically calls this slot best.pt; the bound path is explicitly last.pt.
                    "best.pt": {
                        "path": protocol["checkpoints"][seed][cell]["path"],
                        "sha256": protocol["checkpoints"][seed][cell]["sha256"],
                        "bytes": protocol["checkpoints"][seed][cell]["bytes"],
                    }
                }
            }
            for cell in CELLS
        }
    }


def namespace(args, *, output: Path, conf: float, devices: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        imgsz=640,
        conf=conf,
        nms_iou=0.7,
        duplicate_iou=0.7,
        max_det=300,
        predict_device=args.predict_device,
        devices=devices or [args.predict_device],
        warmup=args.warmup,
        samples=args.samples,
        threads=args.threads,
        affinity=args.affinity,
        output=output,
        export_atol=args.export_atol,
        export_rtol=args.export_rtol,
    )


def stage_predict(args, base, protocol: dict) -> None:
    destination = args.output / "predict"
    destination.mkdir(exist_ok=False)
    images = [Path(item["path"]) for item in protocol["selected_images"][:16]]
    evidence = {
        "schema": "a1-p1-r28-predict-evidence/v1",
        "status": "completed",
        "checkpoint_selection": protocol["checkpoint_selection"],
        "environment": base.prepare_runtime(args.threads, None),
        "seeds": {},
    }
    for seed in SEEDS:
        result = base.run_predict(
            gate_for_seed(protocol, seed), namespace(args, output=destination / seed, conf=0.001), images
        )
        evidence["seeds"][seed] = result
        if result.get("status") != "completed":
            evidence["status"] = "partial"
    evidence["checkpoint_hashes_after"] = checkpoint_hashes(protocol)
    write_json(destination / "evidence.json", evidence)


def stage_latency(args, base, protocol: dict) -> None:
    destination = args.output / "latency"
    destination.mkdir(exist_ok=False)
    images = [Path(item["path"]) for item in protocol["selected_images"][:16]]
    devices = args.devices
    evidence = {
        "schema": "a1-p1-r28-latency-evidence/v1",
        "status": "completed",
        "checkpoint_selection": protocol["checkpoint_selection"],
        "environment": base.prepare_runtime(args.threads, args.affinity),
        "devices": devices,
        "seeds": {},
    }
    for seed in SEEDS:
        result = base.run_latency(
            gate_for_seed(protocol, seed), namespace(args, output=destination / seed, conf=0.25, devices=devices), images
        )
        evidence["seeds"][seed] = result
        if result.get("status") != "completed":
            evidence["status"] = "partial"
    evidence["checkpoint_hashes_after"] = checkpoint_hashes(protocol)
    write_json(destination / "evidence.json", evidence)
    write_latency_csv(destination / "latency_samples.csv", evidence)


def stage_export(args, base, protocol: dict) -> None:
    destination = args.output / "export"
    destination.mkdir(exist_ok=False)
    images = [Path(item["path"]) for item in protocol["selected_images"][:4]]
    evidence = {
        "schema": "a1-p1-r28-export-evidence/v1",
        "status": "completed",
        "checkpoint_selection": protocol["checkpoint_selection"],
        "environment": base.prepare_runtime(args.threads, None, inspect_cuda=False),
        "seeds": {},
    }
    for seed in SEEDS:
        seed_dir = destination / seed
        seed_dir.mkdir()
        result = base.run_export(
            gate_for_seed(protocol, seed), namespace(args, output=seed_dir, conf=0.25), images
        )
        evidence["seeds"][seed] = result
        if result.get("status") != "completed":
            evidence["status"] = "partial"
    evidence["checkpoint_hashes_after"] = checkpoint_hashes(protocol)
    write_json(destination / "evidence.json", evidence)


def stage_resources(args, base, protocol: dict) -> None:
    import torch

    from ultralytics.nn.modules.moe.utils import is_core_moe_block
    from ultralytics.utils.torch_utils import get_flops

    destination = args.output / "resources"
    destination.mkdir(exist_ok=False)
    image = base.load_images([Path(protocol["selected_images"][0]["path"])])[0]
    evidence = {
        "schema": "a1-p1-r28-resource-evidence/v1",
        "status": "completed",
        "checkpoint_selection": protocol["checkpoint_selection"],
        "device": args.resource_device,
        "seeds": {},
    }
    for seed in SEEDS:
        evidence["seeds"][seed] = {}
        gate = gate_for_seed(protocol, seed)
        for cell in CELLS:
            entry = {"status": "failed"}
            try:
                model = base.load_model(gate, cell)
                network = model.model
                parameters = sum(parameter.numel() for parameter in network.parameters())
                gradients = sum(parameter.numel() for parameter in network.parameters() if parameter.requires_grad)
                routed = [module for module in network.modules() if is_core_moe_block(module)]
                adapters = [module for module in network.modules() if module.__class__.__name__ == "C3k2ResidualFactor"]
                flops = float(get_flops(network, imgsz=640))
                entry = {
                    "status": "completed",
                    "checkpoint": protocol["checkpoints"][seed][cell],
                    "parameters": parameters,
                    "gradient_parameters_on_reload": gradients,
                    "gflops": flops,
                    "routed_modules": len(routed),
                    "residual_factor_adapters": len(adapters),
                    "end2end": bool(network.end2end),
                }
                if args.resource_device != "none":
                    device = args.resource_device
                    require(device.isdigit() and int(device) < torch.cuda.device_count(), "resource GPU unavailable")
                    params = base.prediction_params(namespace(args, output=destination, conf=0.25), device)
                    for _ in range(10):
                        model.predict(source=image, **params)
                    torch.cuda.synchronize(int(device))
                    torch.cuda.reset_peak_memory_stats(int(device))
                    before_allocated = torch.cuda.memory_allocated(int(device))
                    before_reserved = torch.cuda.memory_reserved(int(device))
                    prediction = model.predict(source=image, **params)[0]
                    torch.cuda.synchronize(int(device))
                    entry["batch1_cuda_memory"] = {
                        "device": device,
                        "allocated_before_bytes": before_allocated,
                        "reserved_before_bytes": before_reserved,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(int(device)),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(int(device)),
                        "detections": len(prediction.boxes),
                        "native_speed_ms": prediction.speed,
                    }
                del model
                if args.resource_device != "none":
                    torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001 - preserve per-cell evidence when one resource probe fails
                entry = {"status": "failed", "error": str(exc)}
                evidence["status"] = "partial"
            evidence["seeds"][seed][cell] = entry
    evidence["checkpoint_hashes_after"] = checkpoint_hashes(protocol)
    write_json(destination / "evidence.json", evidence)


def checkpoint_hashes(protocol: dict) -> dict:
    return {
        seed: {
            cell: digest(Path(record["path"])) for cell, record in cells.items()
        }
        for seed, cells in protocol["checkpoints"].items()
    }


def write_latency_csv(path: Path, evidence: dict) -> None:
    fields = (
        "seed",
        "device",
        "cell",
        "sample",
        "image_index",
        "total_ms",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "detections",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for seed, result in evidence["seeds"].items():
            for device, cells in result["devices"].items():
                for cell, entry in cells.items():
                    for sample in entry.get("samples", []):
                        writer.writerow({"seed": seed, "device": device, "cell": cell, **sample})


def effect_summary(values: dict[str, dict[str, float]]) -> dict:
    per_seed = {}
    for seed, cells in values.items():
        per_seed[seed] = {
            "B_minus_A": cells["b"] - cells["a"],
            "D_minus_C": cells["d"] - cells["c"],
            "interaction": (cells["d"] - cells["c"]) - (cells["b"] - cells["a"]),
        }
    return {
        "per_seed": per_seed,
        "mean": {key: statistics.fmean(item[key] for item in per_seed.values()) for key in next(iter(per_seed.values()))},
        "sample_sd": {
            key: statistics.stdev(item[key] for item in per_seed.values()) for key in next(iter(per_seed.values()))
        },
    }


def stage_summary(args, protocol: dict) -> None:
    destination = args.output / "summary"
    destination.mkdir(exist_ok=False)
    predict = read_json(args.output / "predict/evidence.json")
    latency = read_json(args.output / "latency/evidence.json")
    export = read_json(args.output / "export/evidence.json")
    resources = read_json(args.output / "resources/evidence.json")
    latency_stats = {}
    for device in latency["devices"]:
        means = {
            seed: {
                cell: latency["seeds"][seed]["devices"][device][cell]["statistics"]["total"]["mean"]
                for cell in CELLS
            }
            for seed in SEEDS
        }
        latency_stats[device] = {"means_ms": means, "effects_ms": effect_summary(means)}
    predict_passed = all(
        predict["seeds"][seed]["cells"][cell]["nms"]["status"] == "passed"
        for seed in SEEDS
        for cell in CELLS
    )
    export_statuses = {
        seed: {cell: export["seeds"][seed]["cells"][cell]["status"] for cell in CELLS} for seed in SEEDS
    }
    summary = {
        "schema": "a1-p1-r28-closure-summary/v1",
        "status": "passed"
        if predict_passed
        and latency.get("status") == "completed"
        and export.get("status") == "completed"
        and resources.get("status") == "completed"
        else "completed_with_findings",
        "checkpoint_selection": protocol["checkpoint_selection"],
        "predict_nms_trace_passed_12_of_12": predict_passed,
        "latency": latency_stats,
        "export_statuses": export_statuses,
        "resources_status": resources.get("status"),
        "source_evidence": {
            "predict": checked_file(args.output / "predict/evidence.json", digest(args.output / "predict/evidence.json"), "predict evidence"),
            "latency": checked_file(args.output / "latency/evidence.json", digest(args.output / "latency/evidence.json"), "latency evidence"),
            "latency_samples": checked_file(args.output / "latency/latency_samples.csv", digest(args.output / "latency/latency_samples.csv"), "latency samples"),
            "export": checked_file(args.output / "export/evidence.json", digest(args.output / "export/evidence.json"), "export evidence"),
            "resources": checked_file(args.output / "resources/evidence.json", digest(args.output / "resources/evidence.json"), "resource evidence"),
        },
        "checkpoint_hashes_after_all_stages": checkpoint_hashes(protocol),
    }
    write_json(destination / "result_summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predict-device", default="1")
    parser.add_argument("--devices", nargs="+", default=["cpu", "1"])
    parser.add_argument("--resource-device", default="1")
    parser.add_argument("--image-limit", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--affinity", default="28,29,30,31")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--export-atol", type=float, default=1e-4)
    parser.add_argument("--export-rtol", type=float, default=1e-3)
    args = parser.parse_args()
    require(args.repo.is_dir(), f"missing repo: {args.repo}")
    require(args.audit.is_file(), f"missing audit: {args.audit}")
    require(args.protocol.is_file(), f"missing protocol: {args.protocol}")
    require(args.image_limit >= 16 and args.warmup >= 1 and args.samples >= 2, "invalid sampling budget")
    require(args.threads >= 1 and args.export_atol >= 0 and args.export_rtol >= 0, "invalid numeric setting")
    return args


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")
    base = load_base(repo)
    if args.stage == "prepare":
        prepare(args, base)
        return
    protocol = load_closure(args)
    if args.stage == "predict":
        stage_predict(args, base, protocol)
    elif args.stage == "latency":
        stage_latency(args, base, protocol)
    elif args.stage == "export":
        stage_export(args, base, protocol)
    elif args.stage == "resources":
        stage_resources(args, base, protocol)
    elif args.stage == "summary":
        stage_summary(args, protocol)


if __name__ == "__main__":
    main()
