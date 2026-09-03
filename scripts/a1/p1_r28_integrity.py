#!/usr/bin/env python3
"""Shared immutable-data integrity helpers for the A1 P1 r28 protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    image_indices = [index for index, part in enumerate(parts) if part == "images"]
    if not image_indices:
        raise ValueError(f"cannot derive YOLO label path from image path: {image}")
    parts[image_indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def data_list_content_signature(list_path: Path) -> dict[str, Any]:
    """Hash ordered image and label bytes without emitting a huge per-file manifest."""
    list_path = list_path.resolve()
    lines = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    digest = hashlib.sha256()
    for line in lines:
        image = Path(line)
        if not image.is_absolute():
            image = list_path.parent / image
        image = image.resolve()
        label = label_path_for_image(image)
        if not image.is_file():
            raise FileNotFoundError(image)
        if not label.is_file():
            raise FileNotFoundError(label)
        image_sha = sha256(image)
        label_sha = sha256(label)
        digest.update(f"{image}\0{image_sha}\0{label}\0{label_sha}\n".encode())
    return {
        "samples": len(lines),
        "image_files": len(lines),
        "label_files": len(lines),
        "ordered_image_label_content_sha256": digest.hexdigest(),
    }


def verify_registered_data_content(protocol: dict[str, Any]) -> None:
    """Verify each dataset at its registered, scale-appropriate integrity level."""
    for label, dataset in protocol.get("data", {}).items():
        for split, item in dataset.get("lists", {}).items():
            path = Path(item["path"])
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if sha256(path) != item.get("sha256") or len(lines) != item.get("images"):
                raise ValueError(f"{label}/{split} ordered image-list drift")
            integrity = item.get("integrity")
            if integrity == "ordered_list_sha256_count_and_image_label_content_sha256":
                actual = data_list_content_signature(path)
                if actual != item.get("content"):
                    raise ValueError(f"{label}/{split} image-or-label content drift")
            elif integrity != "ordered_list_sha256_and_exact_count":
                raise ValueError(f"{label}/{split} unknown integrity policy: {integrity!r}")

