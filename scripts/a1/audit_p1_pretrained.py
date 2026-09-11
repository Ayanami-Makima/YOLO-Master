#!/usr/bin/env python3
"""Audit the locked P1 pilot data and common pretrained initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from prepare_coco2017 import sha256


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def class_counts(paths: list[Path]) -> tuple[int, dict[str, int], int]:
    counts = {str(index): 0 for index in range(80)}
    boxes = empty = 0
    for image in paths:
        label = Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")
        rows = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        empty += not rows
        boxes += len(rows)
        for row in rows:
            counts[str(int(row.split()[0]))] += 1
    return boxes, counts, empty


def tensor_digest(state: dict[str, torch.Tensor]) -> str:
    result = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        result.update(key.encode())
        result.update(str(tensor.dtype).encode())
        result.update(bytes(str(tuple(tensor.shape)), "ascii"))
        result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configs, output = args.configs.resolve(), args.output.resolve()
    protocol_path = configs / "protocol.json"
    protocol = read_json(protocol_path)
    problems = []
    for key in ("dataset_manifest", "selection_manifest", "data_yaml", "evaluation_data_yaml", "checkpoint"):
        path = Path(protocol[key])
        expected = protocol[f"{key}_sha256"]
        if not path.is_file() or sha256(path) != expected:
            problems.append(f"locked {key} is missing or changed: {path}")

    selection = read_json(Path(protocol["selection_manifest"]))
    dataset = {"status": "passed", "splits": {}, "train_val_overlap": None}
    identities = {}
    for split in ("train2017", "val2017"):
        entry = selection["lists"][split]
        listing = Path(entry["path"])
        if sha256(listing) != entry["sha256"]:
            problems.append(f"selection list changed: {split}")
        paths = [Path(line).resolve() for line in listing.read_text(encoding="utf-8").splitlines() if line]
        identities[split] = {path.name for path in paths}
        boxes, counts, empty = class_counts(paths)
        observed = {"images": len(paths), "boxes": boxes, "classes": sum(value > 0 for value in counts.values()), "class_box_counts": counts, "empty_labels": empty}
        dataset["splits"][split] = observed
        if observed != selection["splits"][split]:
            problems.append(f"selection statistics mismatch: {split}")
    overlap = identities["train2017"] & identities["val2017"]
    dataset["train_val_overlap"] = len(overlap)
    if overlap:
        problems.append("pilot train/val image identities overlap")

    preflight = protocol["preflight_data"]
    isolation = {"status": "passed", "splits": {}}
    formal_yaml = yaml.safe_load(Path(protocol["data_yaml"]).read_text(encoding="utf-8"))
    preflight_yaml = yaml.safe_load(Path(preflight["data_yaml"]).read_text(encoding="utf-8"))
    for split in ("train2017", "val2017"):
        entry = preflight["lists"][split]
        listing = Path(entry["path"])
        if sha256(listing) != entry["sha256"]:
            problems.append(f"preflight list changed: {split}")
        formal_cache = Path(formal_yaml["path"]) / "labels" / f"{split}.cache"
        preflight_cache = Path(preflight_yaml["path"]) / "labels" / f"{split}.cache"
        isolated = formal_cache.resolve(strict=False) != preflight_cache.resolve(strict=False)
        isolation["splits"][split] = {
            "images": len(listing.read_text(encoding="utf-8").splitlines()),
            "formal_cache": str(formal_cache),
            "preflight_cache": str(preflight_cache),
            "isolated": isolated,
        }
        if not isolated:
            problems.append(f"preflight cache is not isolated: {split}")

    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    source_model = YOLO(protocol["checkpoint"]).model
    source = source_model.state_dict()
    initialization = {
        "status": "passed",
        "checkpoint": protocol["checkpoint"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "source_tensors": len(source),
        "source_parameters_and_buffers": sum(value.numel() for value in source.values()),
        "cells": {},
    }
    for cell in "abcd":
        model_path = Path(protocol["matrix"][cell]["path"])
        if sha256(model_path) != protocol["matrix"][cell]["sha256"]:
            problems.append(f"model config changed: {cell}")
            continue
        random.seed(protocol["common_training"]["seed"])
        np.random.seed(protocol["common_training"]["seed"])
        torch.manual_seed(protocol["common_training"]["seed"])
        target = DetectionModel(cfg=str(model_path), ch=3, nc=80, verbose=False)
        before = target.state_dict()
        compatible = {key: value for key, value in source.items() if key in before and value.shape == before[key].shape}
        target.load(source_model, verbose=False)
        after = target.state_dict()
        mismatched = [key for key, value in compatible.items() if not torch.equal(after[key].cpu(), value.cpu())]
        if mismatched:
            problems.append(f"checkpoint transfer mismatch in {cell}: {mismatched[:3]}")
        compatible_elements = sum(value.numel() for value in compatible.values())
        total_elements = sum(value.numel() for value in after.values())
        initialization["cells"][cell] = {
            "moe": cell in "cd",
            "end2end": cell in "bd",
            "target_tensors": len(after),
            "target_parameters_and_buffers": total_elements,
            "compatible_tensors_transferred": len(compatible),
            "compatible_parameters_and_buffers": compatible_elements,
            "compatible_fraction": compatible_elements / total_elements,
            "transfer_verified": not mismatched,
            "initialized_state_sha256": tensor_digest(after),
        }
    report = {
        "schema_version": 1,
        "status": "failed" if problems else "passed",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "dataset_manifest_sha256": protocol["dataset_manifest_sha256"],
        "dataset": dataset,
        "preflight_cache_isolation": isolation,
        "pretrained_initialization": initialization,
        "problems": problems,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
