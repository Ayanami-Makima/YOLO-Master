"""Append-only deterministic resource evidence for the A1 P1 r28 formal checkpoints.

``get_flops()`` in the main runtime accepts an uninitialised tensor.  That is
not suitable for routed models because an accidental NaN is correctly rejected
by the router and ``get_flops()`` silently reports zero.  This utility instead
profiles a fixed, finite 640 x 640 tensor twice per checkpoint and preserves
both repetitions as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

CELLS = ("a", "b", "c", "d")
SEEDS = ("260829", "260830", "260831")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def output_description(value) -> dict:
    """Record only stable, JSON-safe output metadata from a THOP forward."""
    if hasattr(value, "shape"):
        return {"type": type(value).__name__, "shape": list(value.shape)}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "items": [output_description(item) for item in value]}
    return {"type": type(value).__name__}


def profile_once(checkpoint: Path, device: str, image_size: int) -> dict:
    import torch
    from thop import profile

    from ultralytics import YOLO

    model = YOLO(str(checkpoint)).model.to(device).float().eval()
    image = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32, device=device)
    with torch.inference_mode():
        output = model(image)
    # THOP counts multiply-accumulates. Report FLOPs as 2 * MACs explicitly.
    macs, parameters_profiled = profile(model, inputs=(image,), verbose=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_profiled": int(parameters_profiled),
        "macs": float(macs),
        "gflops": float(macs) * 2.0 / 1e9,
        "output": output_description(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    require(args.repo.is_dir(), f"missing repo: {args.repo}")
    require(args.protocol.is_file(), f"missing protocol: {args.protocol}")
    require(args.image_size > 0 and args.repeats >= 2, "invalid profile settings")
    protocol = read_json(args.protocol)
    require(protocol.get("status") == "prepared", "protocol must be prepared")
    destination = args.output / "resources_deterministic_r1"
    destination.mkdir(parents=True, exist_ok=False)

    sys.path.insert(0, str(args.repo.resolve()))
    import torch

    import ultralytics

    if args.device.startswith("cuda"):
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
        torch.cuda.set_device(torch.device(args.device))

    evidence = {
        "schema": "a1-p1-r28-deterministic-resource-evidence/v1",
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_selection": protocol["checkpoint_selection"],
        "method": {
            "counter": "THOP forward-hook profile",
            "reported_unit": "GFLOPs = 2 * THOP MACs / 1e9",
            "input": f"fixed finite all-zero float32 [1,3,{args.image_size},{args.image_size}]",
            "repetitions": args.repeats,
            "interpretation": "fixed-input executed FLOPs; not a claim of data-distribution average or theoretical sparse FLOPs",
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "ultralytics_path": str(Path(ultralytics.__file__).resolve()),
            "device": args.device,
        },
        "seeds": {},
    }
    for seed in SEEDS:
        evidence["seeds"][seed] = {}
        for cell in CELLS:
            record = protocol["checkpoints"][seed][cell]
            checkpoint = Path(record["path"])
            entry = {
                "status": "failed",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256_before": digest(checkpoint),
                "repetitions": [],
            }
            try:
                require(entry["checkpoint_sha256_before"] == record["sha256"], f"checkpoint hash changed: {seed}/{cell}")
                for _ in range(args.repeats):
                    entry["repetitions"].append(profile_once(checkpoint, args.device, args.image_size))
                gflops = [item["gflops"] for item in entry["repetitions"]]
                parameters = [item["parameters"] for item in entry["repetitions"]]
                entry["repeatable"] = all(
                    math.isclose(value, gflops[0], rel_tol=0.0, abs_tol=0.0) for value in gflops[1:]
                ) and len(set(parameters)) == 1
                require(entry["repeatable"], f"non-repeatable profile: {seed}/{cell}: {gflops}")
                entry["parameters"] = parameters[0]
                entry["gflops"] = gflops[0]
                entry["status"] = "completed"
            except Exception as exc:  # noqa: BLE001 - preserve per-cell failure evidence rather than hiding it.
                entry["error"] = f"{type(exc).__name__}: {exc}"
                evidence["status"] = "partial"
            entry["checkpoint_sha256_after"] = digest(checkpoint)
            evidence["seeds"][seed][cell] = entry
    write_json(destination / "evidence.json", evidence)


if __name__ == "__main__":
    main()
