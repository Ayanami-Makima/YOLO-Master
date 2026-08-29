#!/usr/bin/env python3
"""Generate the locked, equal-budget A1 P1 2x2 factorial protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

LOCKED_SHA = "acce839c7e895d6b179de7f7093fa879e237cc7b"
SEED = 260829
CELLS = {
    "a": {"moe": False, "end2end": False},
    "b": {"moe": False, "end2end": True},
    "c": {"moe": True, "end2end": False},
    "d": {"moe": True, "end2end": True},
}
TRAIN_LAYERS = {4, 6, 8, 23}
FREEZE = [index for index in range(24) if index not in TRAIN_LAYERS]


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    """Write deterministic UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def matched_config(parent: dict, moe: bool, end2end: bool, moe_expert_type: str | None = None) -> dict:
    """Create one cell while varying only MoE and end-to-end factors."""
    config = copy.deepcopy(parent)
    config["scale"] = "n"
    config["end2end"] = end2end
    replacements = 0
    for layer in config["backbone"]:
        if layer[2] == "A2C2fMoE":
            if not moe:
                layer[2] = "A2C2f"
                layer[3] = layer[3][:8]
            elif moe_expert_type is not None:
                if len(layer[3]) == 10:
                    layer[3].append(moe_expert_type)
                elif len(layer[3]) == 11:
                    layer[3][-1] = moe_expert_type
                else:
                    raise ValueError(f"unexpected A2C2fMoE argument count: {len(layer[3])}")
            replacements += 1
    if replacements != 3:
        raise ValueError(f"expected three P3/P4/P5 factor blocks, found {replacements}")
    return config


def common_params(device: str, project: Path, name: str, epochs: int) -> dict:
    """Return the exact shared optimization budget for one cell."""
    return {
        "epochs": epochs,
        "imgsz": 640,
        "batch": 4,
        "workers": 0,
        "pretrained": True,
        "optimizer": "SGD",
        "lr0": 0.0001,
        "lrf": 0.2,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "nbs": 16,
        "warmup_epochs": 0.5,
        "warmup_bias_lr": 0.0,
        "warmup_momentum": 0.8,
        "freeze": FREEZE,
        "amp": False,
        "deterministic": True,
        "seed": SEED,
        "patience": 0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 0,
        "cos_lr": False,
        "cache": False,
        "plots": False,
        "save": True,
        "save_period": 1,
        "val": True,
        "fraction": 1.0,
        "exist_ok": False,
        "device": device,
        "project": str(project),
        "name": name,
    }


def training_request(
    *, cell: str, stage: str, initializer: Path, data: Path, project: Path, epochs: int, device: str
) -> dict:
    """Build one BN-frozen training request from a standardized initializer."""
    name = f"{cell}_{stage}_seed{SEED}_{epochs}ep"
    return {
        "skill": "yolo.train",
        "request_id": name,
        "runtime": {
            "cwd": "/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master",
            "python": "/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python",
            "prefer_cli": False,
        },
        "inputs": {"model": str(initializer), "task": "detect", "data": str(data)},
        "params": common_params(device, project, name, epochs),
        "diagnostics": {
            "detect_anomaly": False,
            "failure_report": str(project / name / "failure_diagnostics.json"),
        },
        "artifacts": {"project": str(project / "agent_manifests"), "name": name},
        "policy": {"async": False, "dry_run": False},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("configs/a1/p1_factorial_r9"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pilot-data", type=Path, required=True)
    parser.add_argument("--preflight-data", type=Path, required=True)
    parser.add_argument("--experiment-tag", default="r9")
    parser.add_argument("--initializer-suffix", default="shared_init")
    parser.add_argument("--moe-expert-type")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output = (repo / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    run_root = args.run_root.resolve()
    checkpoint = args.checkpoint.resolve()
    pilot_data = args.pilot_data.resolve()
    preflight_data = args.preflight_data.resolve()
    parent_path = repo / "ultralytics/cfg/models/26/yolo26-master-n.yaml"
    for path in (parent_path, checkpoint, pilot_data, preflight_data):
        if not path.is_file():
            raise FileNotFoundError(path)

    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    configs = {}
    requests = {"preflight": {}, "routing_probe": {}, "formal": {}}
    for cell, factors in CELLS.items():
        config_path = output / f"{cell}_matched.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(
                matched_config(parent, **factors, moe_expert_type=args.moe_expert_type),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        configs[cell] = {"path": str(config_path), "sha256": sha256(config_path), **factors}
        initializer = run_root / "initializers" / f"{cell}_{args.initializer_suffix}.pt"
        for stage, data, epochs in (("preflight", preflight_data, 1), ("formal", pilot_data, 5)):
            request = training_request(
                cell=cell,
                stage=stage,
                initializer=initializer,
                data=data,
                project=run_root / stage,
                epochs=epochs,
                device="0",
            )
            request_path = output / f"{cell}_{stage}_request.json"
            write_json(request_path, request)
            requests[stage][cell] = {"path": str(request_path), "sha256": sha256(request_path)}
        if cell in "cd":
            request = training_request(
                cell=cell,
                stage="routing_probe",
                initializer=initializer,
                data=pilot_data,
                project=run_root / "routing_probe",
                epochs=1,
                device="0",
            )
            request_path = output / f"{cell}_routing_probe_request.json"
            write_json(request_path, request)
            requests["routing_probe"][cell] = {"path": str(request_path), "sha256": sha256(request_path)}

    protocol = {
        "schema_version": 1,
        "name": f"A1 P1 equal-budget 2x2 factorial {args.experiment_tag}",
        "experiment_tag": args.experiment_tag,
        "locked_sha": LOCKED_SHA,
        "source_checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "parent": {"path": str(parent_path), "sha256": sha256(parent_path)},
        "data": {
            "preflight": {"path": str(preflight_data), "sha256": sha256(preflight_data)},
            "pilot": {"path": str(pilot_data), "sha256": sha256(pilot_data)},
        },
        "configs": configs,
        "requests": requests,
        "run_root": str(run_root),
        "train_layers": sorted(TRAIN_LAYERS),
        "freeze": FREEZE,
        "batch_norm": "all affine parameters and running statistics frozen in every cell",
        "preflight_policy": "one epoch from initializer; discard weights and restart formal runs",
        "routing_probe_policy": "C/D only, one full pilot epoch from initializer; discard weights after route audit",
        "formal_budget": "5000 images, batch 4, 5 epochs, identical seed and order",
        "initializer_suffix": args.initializer_suffix,
        "moe_expert_type": args.moe_expert_type or "simple",
        "primary_checkpoint": "last.pt",
        "secondary_checkpoint": "best.pt",
        "comparisons": {
            "end2end_dense": "B-A",
            "end2end_moe": "D-C",
            "moe_non_end2end": "C-A",
            "moe_end2end": "D-B",
            "interaction": "(D-C)-(B-A)",
        },
    }
    write_json(output / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
