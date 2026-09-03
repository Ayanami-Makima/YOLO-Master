#!/usr/bin/env python3
"""Create the immutable deterministic COCO train20k/val5k registry for r27."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml

SELECTION_SEED = 260829
TRAIN_IMAGES = 20_000
VAL_IMAGES = 5_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_list(data_yaml: Path, entry: str) -> Path:
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(spec.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    path = Path(entry)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def absolute_images(list_path: Path) -> list[str]:
    images: list[str] = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        image = Path(raw)
        if not image.is_absolute():
            image = list_path.parent / image
        image = image.resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        images.append(str(image))
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-yaml", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source_yaml = args.source_yaml.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite medium-data directory: {output_dir}")
    spec = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    if not isinstance(spec.get("train"), str) or not isinstance(spec.get("val"), str):
        raise TypeError("source COCO YAML must reference one train and one val list")
    source_train = resolve_list(source_yaml, spec["train"])
    source_val = resolve_list(source_yaml, spec["val"])
    train_images = absolute_images(source_train)
    val_images = absolute_images(source_val)
    if len(train_images) != 118_287 or len(val_images) != VAL_IMAGES:
        raise ValueError(f"unexpected source counts: train={len(train_images)} val={len(val_images)}")

    selected = random.Random(SELECTION_SEED).sample(train_images, TRAIN_IMAGES)
    if len(set(selected)) != TRAIN_IMAGES:
        raise AssertionError("train subset contains duplicate paths")

    output_dir.mkdir(parents=True)
    train_list = output_dir / "train2017_20000_seed260829.txt"
    val_list = output_dir / "val2017_5000.txt"
    train_list.write_text("\n".join(selected) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(val_images) + "\n", encoding="utf-8")

    medium_yaml = output_dir / "coco.yaml"
    medium_spec = dict(spec)
    medium_spec["path"] = str(Path(spec["path"]).resolve())
    medium_spec["train"] = str(train_list)
    medium_spec["val"] = str(val_list)
    medium_yaml.write_text(yaml.safe_dump(medium_spec, sort_keys=False, allow_unicode=True), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "experiment_tag": "r27",
        "selection": "random.Random(seed).sample without replacement; emitted order is fixed",
        "seed": SELECTION_SEED,
        "source_yaml": str(source_yaml),
        "source_yaml_sha256": sha256(source_yaml),
        "source_train_list": str(source_train),
        "source_train_list_sha256": sha256(source_train),
        "source_train_images": len(train_images),
        "source_val_list": str(source_val),
        "source_val_list_sha256": sha256(source_val),
        "train_list": str(train_list),
        "train_list_sha256": sha256(train_list),
        "train_images": len(selected),
        "val_list": str(val_list),
        "val_list_sha256": sha256(val_list),
        "validation_images": len(val_images),
        "data_yaml": str(medium_yaml),
        "data_yaml_sha256": sha256(medium_yaml),
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
