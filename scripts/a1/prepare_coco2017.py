#!/usr/bin/env python3
"""Verify official COCO2017 archives and prepare detection-only YOLO labels.

Keep the native converter unchanged. Only instances JSONs are exposed to it,
and its output directory must not exist (convert_coco otherwise increments it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ARCHIVES = {
    "train2017.zip": 19336861798,
    "val2017.zip": 815585330,
    "annotations_trainval2017.zip": 252907541,
}
SPLITS = {"train2017": 118287, "val2017": 5000}


def sha256(path: Path) -> str:
    """Hash a file without loading a large archive into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_checked(path: Path, text: str) -> None:
    """Keep identical existing evidence; refuse to overwrite different content."""
    payload = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing file differs; preserve and inspect first: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def validate_label(path: Path) -> Counter:
    """Validate detection rows using the loader's coordinate tolerance."""
    classes = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        if len(values) != 5 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"invalid detection row: {path}: {line}")
        cls, x, y, width, height = values
        if cls != int(cls) or not 0 <= cls < 80:
            raise ValueError(f"invalid category: {path}: {line}")
        if min(x, y, width, height) < -0.01 or max(x, y, width, height) > 1.01:
            raise ValueError(f"out-of-bounds box: {path}: {line}")
        if width <= 0 or height <= 0:
            raise ValueError(f"non-positive box: {path}: {line}")
        classes[int(cls)] += 1
    return classes


def verify_archive(path: Path, destination: Path) -> dict:
    """Check SHA-256/ZIP CRC, then recover missing or truncated extracted files."""
    expected = ARCHIVES[path.name]
    if path.stat().st_size != expected:
        raise ValueError(f"incomplete archive: {path}")
    print(f"Verifying SHA-256 and all ZIP CRCs: {path.name}", flush=True)
    digest = sha256(path)
    restored = 0
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC failure: {path}: {bad}")
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"archive path escapes destination: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_file() and target.stat().st_size == member.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".recovering")
            with archive.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            temporary.replace(target)
            restored += 1
    return {"bytes": expected, "sha256": digest, "zip_crc": "passed", "restored_files": restored}


def convert_instances(root: Path, stage: Path, converter) -> Path:
    """Expose only bounding-box JSONs; avoid captions and converter path incrementing."""
    source = stage / "instances_only"
    source.mkdir()
    for split in SPLITS:
        name = f"instances_{split}.json"
        (source / name).hardlink_to(root / "annotations" / name)
    output = stage / "converted"
    if output.exists():
        raise ValueError(f"converter output must not exist: {output}")
    converter(labels_dir=str(source), save_dir=str(output), cls91to80=True, use_segments=False)
    return output


def prepare(base: Path) -> dict:
    """Prepare a checked dataset, with READY written only after every required check."""
    import yaml

    from ultralytics.data.converter import coco91_to_coco80_class, convert_coco

    base = base.resolve()
    root = base / "coco"
    root.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1, "dataset_root": str(root), "archives": {}, "splits": {}}
    for name in ARCHIVES:
        destination = root if name.startswith("annotations") else root / "images"
        record["archives"][name] = verify_archive(base / "downloads" / name, destination)

    ids_by_split = {}
    categories = None
    with tempfile.TemporaryDirectory(prefix="a1_coco_convert_", dir=base) as directory:
        converted = convert_instances(root, Path(directory), convert_coco)
        for split, expected_count in SPLITS.items():
            annotation = root / "annotations" / f"instances_{split}.json"
            document = json.loads(annotation.read_text(encoding="utf-8"))
            sorted_categories = sorted(document["categories"], key=lambda item: item["id"])
            if categories is not None and categories != sorted_categories:
                raise ValueError("train/val category mappings differ")
            categories = sorted_categories
            if len(categories) != 80:
                raise ValueError("expected 80 categories")
            mapping = coco91_to_coco80_class()
            if any(mapping[item["id"] - 1] != index for index, item in enumerate(categories)):
                raise ValueError("native converter category mapping differs from COCO category order")
            images = document["images"]
            ids_by_split[split] = {item["id"] for item in images}
            filenames = {item["file_name"] for item in images}
            actual = {path.name for path in (root / "images" / split).glob("*.jpg")}
            if len(images) != expected_count or len(ids_by_split[split]) != expected_count or actual != filenames:
                raise ValueError(f"image counts/identities differ from official JSON: {split}")
            target_labels = root / "labels" / split
            target_labels.mkdir(parents=True, exist_ok=True)
            class_counts = Counter()
            empty = 0
            label_digest = hashlib.sha256()
            for name in sorted(filenames):
                label_name = Path(name).with_suffix(".txt").name
                source = converted / "labels" / split / label_name
                text = source.read_text(encoding="utf-8") if source.exists() else ""
                target = target_labels / label_name
                write_text_checked(target, text)
                counts = validate_label(target)
                class_counts.update(counts)
                empty += not bool(counts)
                label_digest.update(label_name.encode("utf-8") + b"\0" + text.encode("utf-8"))
            if len(list(target_labels.glob("*.txt"))) != expected_count:
                raise ValueError(f"unexpected extra labels: {target_labels}")
            manifest = root / f"{split}.txt"
            lines = [f"./images/{split}/{name}" for name in sorted(filenames)]
            write_text_checked(manifest, "\n".join(lines) + "\n")
            record["splits"][split] = {
                "images": expected_count,
                "labels": expected_count,
                "empty_labels": empty,
                "boxes": sum(class_counts.values()),
                "classes": len(class_counts),
                "class_box_counts": dict(sorted(class_counts.items())),
                "manifest_sha256": sha256(manifest),
                "labels_sha256": label_digest.hexdigest(),
                "annotations_sha256": sha256(annotation),
            }
            print(f"Validated {split}: {expected_count} images, {sum(class_counts.values())} boxes", flush=True)
    if ids_by_split["train2017"] & ids_by_split["val2017"]:
        raise ValueError("train/val overlap")
    data = {
        "path": str(root),
        "train": "train2017.txt",
        "val": "val2017.txt",
        "nc": 80,
        "names": {index: item["name"] for index, item in enumerate(categories)},
    }
    config = root / "coco2017.yaml"
    write_text_checked(config, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    record.update({
        "train_val_overlap": 0,
        "yaml_sha256": sha256(config),
        "conversion": "native convert_coco, instances-only, cls91to80=True, use_segments=False",
        "crowd_policy": "native converter excludes iscrowd annotations; official JSON is preserved",
        "image_decode": "archive CRC and extracted sizes checked; loader decode scan is a separate check",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready",
    })
    (root / "dataset_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (root / "READY.json").write_text(json.dumps({"manifest_sha256": sha256(root / "dataset_manifest.json")}) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("/data/data2/TuJiajun/COCO2017"))
    prepare(parser.parse_args().base)
