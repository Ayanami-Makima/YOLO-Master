#!/usr/bin/env python3
"""Prepare a reproducible COCO-seg pilot and A/B/D residual-factor configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from ultralytics.data.converter import coco91_to_coco80_class, merge_multi_segment


LOCKED_SHA = "acce839c7e895d6b179de7f7093fa879e237cc7b"
FACTOR_LAYERS = (4, 6, 8)
EXPERTS = {4: 4, 6: 8, 8: 16}
CELLS = {"a": {"moe": False, "end2end": False}, "b": {"moe": False, "end2end": True}, "d": {"moe": True, "end2end": True}}
NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train",
    7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant", 11: "stop sign", 12: "parking meter",
    13: "bench", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant",
    21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie",
    28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def selected_names(list_path: Path) -> list[str]:
    names = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            names.append(Path(line.strip()).name)
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate images in {list_path}")
    return names


def convert_subset(annotation_path: Path, names: list[str], output_dir: Path, split: str) -> dict:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {item["file_name"]: item for item in data["images"]}
    image_ids = {images[name]["id"] for name in names if name in images}
    missing = sorted(set(names) - set(images))
    if missing:
        raise ValueError(f"{len(missing)} selected images missing from {annotation_path}: {missing[:3]}")
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        if ann["image_id"] in image_ids:
            annotations[ann["image_id"]].append(ann)
    labels_dir = output_dir / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)
    coco80 = coco91_to_coco80_class()
    valid_instances = 0
    skipped_without_polygon = 0
    for name in names:
        item = images[name]
        h, w = item["height"], item["width"]
        lines = []
        seen_boxes = set()
        for ann in annotations[item["id"]]:
            if ann.get("iscrowd", False):
                continue
            seg = ann.get("segmentation")
            if not isinstance(seg, list) or not seg:
                skipped_without_polygon += 1
                continue
            try:
                if len(seg) > 1:
                    merged = merge_multi_segment(seg)
                    polygon = (merged[0] if len(merged) == 1 else np.concatenate(merged, axis=0)).reshape(-1, 2)
                else:
                    polygon = np.asarray(seg[0], dtype=float).reshape(-1, 2)
            except (TypeError, ValueError):
                skipped_without_polygon += 1
                continue
            if len(polygon) < 3:
                skipped_without_polygon += 1
                continue
            box = tuple(round(float(value), 6) for value in ann.get("bbox", []))
            if box in seen_boxes:
                continue
            seen_boxes.add(box)
            cls = coco80[ann["category_id"] - 1]
            points = (polygon / [w, h]).clip(0.0, 1.0).reshape(-1).tolist()
            lines.append(" ".join([str(cls), *(f"{value:.6g}" for value in points)]))
            valid_instances += 1
        (labels_dir / Path(name).with_suffix(".txt").name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {"images": len(names), "valid_instances": valid_instances, "skipped_without_polygon": skipped_without_polygon}


def config_for(parent: dict, *, moe: bool, end2end: bool) -> dict:
    config = copy.deepcopy(parent)
    config["scale"] = "n"
    config["end2end"] = end2end
    for index in FACTOR_LAYERS:
        layer = config["backbone"][index]
        if layer[2] != "C3k2":
            raise ValueError(f"layer {index}: expected C3k2, got {layer[2]}")
        c2, c3k = layer[3][:2]
        expansion = layer[3][2] if len(layer[3]) > 2 else 0.5
        layer[2] = "C3k2ResidualFactor"
        layer[3] = [c2, c3k, expansion, moe, EXPERTS[index], 2, "dense_mlp"]
    return config


def common_params(project: Path, name: str, *, device: str, epochs: int = 5) -> dict:
    return {
        "epochs": epochs, "imgsz": 640, "batch": 4, "workers": 0, "pretrained": True, "optimizer": "SGD",
        "lr0": 0.0001, "lrf": 0.2, "momentum": 0.9, "weight_decay": 0.0005, "nbs": 16, "warmup_epochs": 0.5,
        "warmup_bias_lr": 0.0, "warmup_momentum": 0.8, "freeze": [i for i in range(24) if i not in {4, 6, 8, 23}],
        "amp": False, "deterministic": True, "seed": 260829, "patience": 0, "mosaic": 0.0, "mixup": 0.0,
        "copy_paste": 0.0, "close_mosaic": 0, "cos_lr": False, "cache": False, "plots": False, "save": True,
        "save_period": 1, "val": True, "fraction": 1.0, "exist_ok": False, "device": device,
        "project": str(project), "name": name,
        "moe_noise_std": 0.0, "moe_router_lr_scale": 1.0, "moe_expert_warmup_epochs": 0,
        "moe_dynamic_schedule": "none", "moe_map_saturation_enabled": False, "moe_balance_loss": 1.0,
        "moe_router_z_loss": 0.1, "moe_aux_gain": 1.0, "mixture_aux_budget": 3.0, "moe_temperature": 1.0,
        "moa_mot_temperature_factor": 1.0, "moa_mot_min_temperature": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pilot-data", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    repo, output, run_root = args.repo.resolve(), args.output.resolve(), args.run_root.resolve()
    if output.exists() or run_root.exists():
        raise FileExistsError(f"refusing to overwrite output={output} or run_root={run_root}")
    for path in (args.checkpoint, args.pilot_data, args.annotations):
        if not path.exists():
            raise FileNotFoundError(path)
    source_data = args.pilot_data.resolve()
    data_dir = output / "pilot_data"
    for split in ("train2017", "val2017"):
        source = source_data / f"{split}.txt"
        if not source.is_file():
            raise FileNotFoundError(source)
        names = selected_names(source)
        image_link = data_dir / "images" / split
        image_link.parent.mkdir(parents=True, exist_ok=True)
        image_target = Path("/data/data2/TuJiajun/COCO2017/coco/images") / split
        image_link.symlink_to(image_target, target_is_directory=True)
        stats = convert_subset(args.annotations / f"instances_{split}.json", names, data_dir, split)
        # Ultralytics resolves only entries explicitly prefixed with ./ relative to the list file.
        (data_dir / f"{split}.txt").write_text("\n".join(f"./images/{split}/{name}" for name in names) + "\n", encoding="utf-8")
        if split == "train2017":
            train_stats = stats
        else:
            val_stats = stats
    data_yaml = data_dir / "coco-seg.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(data_dir), "train": "train2017.txt", "val": "val2017.txt", "nc": 80, "names": NAMES}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    parent_path = repo / "ultralytics/cfg/models/26/yolo26-seg.yaml"
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    configs, requests = {}, {}
    for cell, factors in CELLS.items():
        config_path = output / f"{cell}_seg.yaml"
        config_path.write_text(yaml.safe_dump(config_for(parent, **factors), sort_keys=False, allow_unicode=True), encoding="utf-8")
        name = f"{cell}_seg_pilot_seed260829_5ep"
        request = {
            "skill": "yolo.train", "request_id": name,
            "runtime": {"cwd": str(repo), "python": "/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python", "prefer_cli": False},
            "inputs": {"model": str(run_root / "initializers" / f"{cell}_seg_init.pt"), "task": "segment", "data": str(data_yaml)},
            "params": common_params(run_root / "train", name, device="1"),
            "diagnostics": {"detect_anomaly": False, "failure_report": str(run_root / "train" / name / "failure_diagnostics.json")},
            "artifacts": {"project": str(run_root / "agent_manifests"), "name": name}, "policy": {"async": False, "dry_run": False},
            "a1_policy": {"freeze_batch_norm": True, "freeze_residual_factor_bases": True, "formal_restart_from_initializer": True, "routing_semantics": "deterministic_hard_top2_from_step_zero", "expert_dropout_rate": 0.0},
        }
        request_path = output / f"{cell}_request.json"
        request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        configs[cell] = {"path": str(config_path), "sha256": sha256(config_path), **factors}
        requests[cell] = {"path": str(request_path), "sha256": sha256(request_path)}
    protocol = {
        "schema": "a1-p2-seg-extension-r1/v1", "status": "prepared", "locked_sha": LOCKED_SHA,
        "source_checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint.resolve())},
        "parent": {"path": str(parent_path), "sha256": sha256(parent_path)}, "data": {"yaml": str(data_yaml), "yaml_sha256": sha256(data_yaml), "train_stats": train_stats, "val_stats": val_stats},
        "configs": configs, "requests": requests, "run_root": str(run_root), "task": "segment", "pilot_budget": "5000 train / 512 val, batch 4, 5 epochs, seed 260829, GPU1", "train_layers": [4, 6, 8, 23],
        "freeze": "all other model layers; all BatchNorm; residual-factor base", "initializer_policy": "shared pretrained backbone/head tensors where shapes match; zero residual gain; segmentation-specific mask tensors remain newly initialized",
    }
    write_json(output / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
