#!/usr/bin/env python3
"""Prepare the locked A1 P1 pretrained 2x2 pilot protocol.

The four cells share one checkpoint, optimizer budget, deterministic COCO
subsets and matched model shell.  Only MoE and end-to-end/NMS-free are varied.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

import yaml

from prepare_coco2017 import sha256, write_text_checked

LOCKED_SHA = "acce839c7e895d6b179de7f7093fa879e237cc7b"
SEED = 260829


def matched_config(parent: dict, moe: bool, end2end: bool) -> dict:
    """Return one cell while keeping all non-factor architecture choices fixed."""
    config = copy.deepcopy(parent)
    config["end2end"] = end2end
    config["scale"] = "n"
    replacements = 0
    for layer in config["backbone"]:
        if layer[2] == "A2C2fMoE":
            if not moe:
                layer[2] = "A2C2f"
                layer[3] = layer[3][:8]
            replacements += 1
    if replacements != 3:
        raise ValueError(f"expected P3/P4/P5 factor blocks, found {replacements}")
    return config


def label_classes(data_root: Path, item: str) -> list[int]:
    """Read the native YOLO label classes for one COCO list entry."""
    relative = Path(item.strip())
    parts = list(relative.parts)
    try:
        parts[parts.index("images")] = "labels"
    except ValueError as exc:
        raise ValueError(f"image entry lacks images component: {item}") from exc
    label = (data_root / Path(*parts)).with_suffix(".txt")
    classes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        if line.strip():
            classes.append(int(line.split()[0]))
    return classes


def class_aware_sample(data_root: Path, split: str, count: int, seed: int) -> tuple[list[str], dict]:
    """Cover all 80 classes first, then fill a deterministic random sample."""
    source = [line for line in (data_root / f"{split}.txt").read_text(encoding="utf-8").splitlines() if line]
    if not 1 <= count <= len(source):
        raise ValueError(f"invalid sample size for {split}: {count}")
    order = list(source)
    random.Random(seed).shuffle(order)
    class_cache = {item: label_classes(data_root, item) for item in order}
    uncovered = set(range(80))
    selected = []
    selected_set = set()
    for item in order:
        if uncovered.intersection(class_cache[item]):
            selected.append(item)
            selected_set.add(item)
            uncovered.difference_update(class_cache[item])
            if not uncovered:
                break
    if uncovered:
        raise ValueError(f"{split} cannot cover classes: {sorted(uncovered)}")
    for item in order:
        if len(selected) == count:
            break
        if item not in selected_set:
            selected.append(item)
            selected_set.add(item)
    selected.sort()
    counts = Counter(cls for item in selected for cls in class_cache[item])
    return selected, {
        "images": len(selected),
        "boxes": sum(counts.values()),
        "classes": len(counts),
        "class_box_counts": {str(key): counts[key] for key in range(80)},
        "empty_labels": sum(not class_cache[item] for item in selected),
    }


def create_view(data_root: Path, view: Path, selections: dict[str, list[str]]) -> tuple[Path, dict]:
    """Create a cache-isolated symlink view and immutable image lists."""
    view.mkdir(parents=True, exist_ok=True)
    source_yaml = yaml.safe_load((data_root / "coco2017.yaml").read_text(encoding="utf-8"))
    source_yaml["path"] = str(view)
    lists = {}
    for split, items in selections.items():
        for kind in ("images", "labels"):
            link = view / kind / split
            target = (data_root / kind / split).resolve()
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                if link.resolve() != target:
                    raise ValueError(f"view symlink changed: {link}")
            elif link.exists():
                raise ValueError(f"view path exists and is not a symlink: {link}")
            else:
                link.symlink_to(target, target_is_directory=True)
        paths = [str(view / "images" / split / Path(item).name) for item in items]
        listing = view / f"{split}.txt"
        write_text_checked(listing, "\n".join(paths) + "\n")
        source_yaml["train" if split == "train2017" else "val"] = str(listing)
        lists[split] = {"path": str(listing), "sha256": sha256(listing), "images": len(items)}
    data_yaml = view / "coco.yaml"
    write_text_checked(data_yaml, yaml.safe_dump(source_yaml, sort_keys=False, allow_unicode=True))
    return data_yaml, lists


def request(repo: Path, run_root: Path, model: Path, data: Path, cell: str, phase: str, common: dict) -> dict:
    """Build one structured asynchronous native training request."""
    params = {
        **common,
        "device": "0" if cell in "ac" else "1",
        "project": str(run_root),
        "name": f"{cell}_matched_seed{SEED}_{'preflight' if phase == 'preflight' else str(common['epochs']) + 'ep'}",
    }
    if phase == "preflight":
        params.update(epochs=1, close_mosaic=0, save_period=-1, workers=0)
    return {
        "skill": "yolo.train",
        "request_id": params["name"],
        "runtime": {"cwd": str(repo), "python": str(repo.parent / ".venv/bin/python"), "prefer_cli": True},
        "inputs": {"model": str(model), "task": "detect", "data": str(data)},
        "params": params,
        "artifacts": {"project": str(run_root / "agent_manifests"), "name": params["name"]},
        "policy": {"async": True, "dry_run": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("/data/data2/TuJiajun/COCO2017/coco"))
    parser.add_argument("--run-root", type=Path, default=Path("/data/data2/TuJiajun/A1-smoke-r4/p1_pretrained_r1"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train-images", type=int, default=5000)
    parser.add_argument("--val-images", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    repo, data_root, run_root = args.repo.resolve(), args.data_root.resolve(), args.run_root.resolve()
    output = (args.output_dir or repo / "configs/a1/p1_pretrained").resolve()
    checkpoint = (args.checkpoint or repo / "yolo26n.pt").resolve()
    if min(args.train_images, args.val_images, args.epochs, args.batch) < 1:
        raise ValueError("dataset and training counts must be positive")
    if not checkpoint.is_file():
        raise ValueError(f"missing pretrained checkpoint: {checkpoint}")
    full_manifest = data_root / "dataset_manifest.json"
    ready = json.loads((data_root / "READY.json").read_text(encoding="utf-8"))
    if ready["manifest_sha256"] != sha256(full_manifest):
        raise ValueError("full COCO readiness manifest has changed")
    core_diff = subprocess.check_output(["git", "diff", LOCKED_SHA, "--", "ultralytics"], cwd=repo, text=True)
    if core_diff:
        raise ValueError("framework differs from the locked task-book baseline")
    output.mkdir(parents=True, exist_ok=True)

    selected, stats = {}, {}
    for split, count, offset in (("train2017", args.train_images, 0), ("val2017", args.val_images, 1)):
        selected[split], stats[split] = class_aware_sample(data_root, split, count, SEED + offset)
    pilot_yaml, pilot_lists = create_view(data_root, output / "pilot_data", selected)
    preflight_selected = {
        "train2017": selected["train2017"][:256],
        "val2017": selected["val2017"][:128],
    }
    preflight_yaml, preflight_lists = create_view(data_root, output / "preflight_data", preflight_selected)
    selection_manifest = output / "selection_manifest.json"
    selection_record = {
        "schema_version": 1,
        "seed": SEED,
        "source_dataset_manifest": str(full_manifest),
        "source_dataset_manifest_sha256": sha256(full_manifest),
        "splits": stats,
        "lists": pilot_lists,
        "selection": "greedy 80-class coverage followed by seeded random fill; final lists sorted",
    }
    write_text_checked(selection_manifest, json.dumps(selection_record, indent=2) + "\n")

    parent_path = repo / "ultralytics/cfg/models/26/yolo26-master-n.yaml"
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    common = {
        "epochs": args.epochs,
        "imgsz": 640,
        "batch": args.batch,
        "workers": 0,
        "pretrained": str(checkpoint),
        "optimizer": "SGD",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "nbs": 64,
        "warmup_epochs": 1.0,
        "warmup_bias_lr": 0.0,
        "warmup_momentum": 0.8,
        "amp": False,
        "deterministic": True,
        "seed": SEED,
        "patience": 0,
        "close_mosaic": 1,
        "cos_lr": False,
        "cache": False,
        "plots": False,
        "save": True,
        "save_period": 1,
        "val": True,
        "fraction": 1.0,
        "exist_ok": True,
    }
    protocol = {
        "schema_version": 2,
        "locked_sha": LOCKED_SHA,
        "head_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "core_diff_from_lock": "empty",
        "parent": str(parent_path.relative_to(repo)),
        "parent_sha256": sha256(parent_path),
        "dataset_manifest": str(full_manifest),
        "dataset_manifest_sha256": sha256(full_manifest),
        "selection_manifest": str(selection_manifest),
        "selection_manifest_sha256": sha256(selection_manifest),
        "data_yaml": str(pilot_yaml),
        "data_yaml_sha256": sha256(pilot_yaml),
        "evaluation_data_yaml": str(data_root / "coco2017.yaml"),
        "evaluation_data_yaml_sha256": sha256(data_root / "coco2017.yaml"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "run_root": str(run_root),
        "matrix": {},
        "requests_sha256": {},
        "common_training": common,
        "preflight_data": {"data_yaml": str(preflight_yaml), "data_yaml_sha256": sha256(preflight_yaml), "lists": preflight_lists},
        "scope": "single-seed 5-epoch pretrained P1 pilot; final accuracy is re-evaluated on full COCO val2017",
        "limitations": ["single seed; report uncertainty and do not claim statistical significance"],
    }
    for cell in "abcd":
        model = output / f"{cell}_matched.yaml"
        write_text_checked(model, yaml.safe_dump(matched_config(parent, cell in "cd", cell in "bd"), sort_keys=False))
        protocol["matrix"][cell] = {
            "moe": cell in "cd",
            "end2end": cell in "bd",
            "path": str(model),
            "sha256": sha256(model),
        }
        for phase, data in (("preflight", preflight_yaml), ("full", pilot_yaml)):
            path = output / f"{cell}_{phase}_request.json"
            write_text_checked(path, json.dumps(request(repo, run_root, model, data, cell, phase, common), indent=2) + "\n")
            protocol["requests_sha256"][f"{cell}_{phase}"] = sha256(path)
    write_text_checked(output / "protocol.json", json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
