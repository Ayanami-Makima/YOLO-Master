#!/usr/bin/env python3
"""P2-E r1: diagnose the detect End-to-End precision gap on fixed val512.

This stage is evaluation-only. It records ordinary prediction TP/FP/FN and, when the native
criterion is available, the assignment and loss tensors used by the checkpoint's one-to-many or
one-to-one branch. It does not modify model weights, router settings, assigners, or losses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

LOCKED_SHA = "acce839c7e895d6b179de7f7093fa879e237cc7b"
CELLS = ("a", "b", "c", "d")
SEEDS = (260829, 260830, 260831)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise xyxy IoU without changing either input."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))[:, None]
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))[None, :]
    return inter / (area_a + area_b - inter).clamp(min=1e-12)


def greedy_matches(pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor], threshold: float = 0.5):
    """Return confidence-ordered class-aware one-to-one matches and their IoUs."""
    boxes = pred["bboxes"].detach().cpu()
    scores = pred["conf"].detach().cpu()
    classes = pred["cls"].detach().cpu().long()
    gt_boxes = target["bboxes"].detach().cpu()
    gt_classes = target["cls"].detach().cpu().long()
    if boxes.numel() == 0 or gt_boxes.numel() == 0:
        return [], set()
    order = torch.argsort(scores, descending=True).tolist()
    overlaps = iou_matrix(boxes, gt_boxes)
    used = set()
    matches = []
    for pi in order:
        candidates = [gi for gi in range(len(gt_boxes)) if gi not in used and classes[pi] == gt_classes[gi]]
        if not candidates:
            continue
        gi = max(candidates, key=lambda index: float(overlaps[pi, index]))
        value = float(overlaps[pi, gi])
        if value >= threshold:
            used.add(gi)
            matches.append((pi, gi, value))
    return matches, used


def size_bucket(area: float) -> str:
    """COCO area buckets in original-image pixel units."""
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


class PrecisionValidatorMixin:
    """Mixin for DetectionValidator that retains fixed per-image diagnostics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision_rows = []
        self.assignment_rows = []
        self._raw_preds = None
        self.native_model = None

    def init_metrics(self, model):
        result = super().init_metrics(model)
        self.native_model = getattr(model, "model", model)
        self.precision_rows = []
        self.assignment_rows = []
        return result

    def postprocess(self, preds):
        self._raw_preds = preds
        return super().postprocess(preds)

    def _assignment_branch(self, raw):
        """Return the native criterion branch and its matching raw prediction dictionary."""
        criterion = getattr(self.native_model, "criterion", None)
        if criterion is None:
            return None, None, None
        native = getattr(criterion, "native_criterion", criterion)
        raw_dict = raw[1] if isinstance(raw, (list, tuple)) and len(raw) > 1 else raw
        if not isinstance(raw_dict, dict):
            return None, None, None
        if hasattr(native, "one2one"):
            branch_name = "one2one" if self.end2end else "one2many"
            branch = getattr(native, branch_name, None)
            branch_pred = raw_dict.get(branch_name)
        else:
            branch_name = "one2many"
            branch = native
            branch_pred = raw_dict.get("one2many", raw_dict)
        if branch is None or branch_pred is None or not hasattr(branch, "get_assigned_targets_and_loss"):
            return None, None, None
        return branch, branch_pred, branch_name

    @torch.no_grad()
    def _collect_assignment(self, batch, raw, image_index: int):
        """Collect assignment counts, IoUs and component losses for one image."""
        raw_dict = raw[1] if isinstance(raw, (list, tuple)) and len(raw) > 1 else raw
        if getattr(self.native_model, "criterion", None) is None:
            try:
                # Initialises the checkpoint-native criterion without a backward pass. The model
                # remains in eval mode and no parameter or routing state is changed.
                self.native_model.loss(batch, raw_dict)
            except Exception as exc:
                return {"assigner_branch": "unavailable", "criterion_init_error": f"{type(exc).__name__}: {exc}"}
        branch, branch_pred, branch_name = self._assignment_branch(raw)
        if branch is None:
            return {"assigner_branch": "unavailable"}
        try:
            assigned, loss, loss_detached = branch.get_assigned_targets_and_loss(branch_pred, batch)
        except Exception as exc:  # diagnostics must identify unsupported checkpoint APIs explicitly
            return {"assigner_branch": branch_name, "assignment_error": f"{type(exc).__name__}: {exc}"}
        fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor = assigned
        image_fg = fg_mask[image_index]
        indices = image_fg.nonzero(as_tuple=False).flatten()
        gt_indices = target_gt_idx[image_index, indices].detach().cpu().tolist()
        counts = Counter(int(index) for index in gt_indices)
        duplicate_positive = sum(count - 1 for count in counts.values() if count > 1)
        matched_ious = []
        try:
            pred_distri = branch_pred["boxes"].permute(0, 2, 1).contiguous()
            decoded = branch.bbox_decode(anchor_points, pred_distri)
            decoded = decoded * stride_tensor
            pred_boxes = decoded[image_index, indices].detach().cpu()
            target_boxes = target_bboxes[image_index, indices].detach().cpu()
            matched_ious = torch.diag(iou_matrix(pred_boxes, target_boxes)).tolist()
        except Exception:
            matched_ious = []
        loss_values = loss_detached.detach().cpu().reshape(-1).tolist()
        row = {
            "assigner_branch": branch_name,
            "positive_count": int(indices.numel()),
            "unique_matched_gt_count": int(len(counts)),
            "duplicate_positive_count": int(duplicate_positive),
            "assignment_overlap_rate": float(duplicate_positive / max(len(gt_indices), 1)),
            "assignment_matched_iou_mean": float(np.mean(matched_ious)) if matched_ious else None,
            "assignment_matched_iou_p50": float(np.percentile(matched_ious, 50)) if matched_ious else None,
            "assignment_matched_iou_p90": float(np.percentile(matched_ious, 90)) if matched_ious else None,
            "box_loss": float(loss_values[0]) if len(loss_values) > 0 else None,
            "cls_loss": float(loss_values[1]) if len(loss_values) > 1 else None,
            "dfl_loss": float(loss_values[2]) if len(loss_values) > 2 else None,
        }
        return row

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)
            matches, matched_gt = greedy_matches(predn, pbatch, threshold=0.5)
            target_count = int(pbatch["cls"].numel())
            pred_count = int(predn["cls"].numel())
            tp = len(matches)
            raw_gt = batch["bboxes"][batch["batch_idx"] == si].detach().cpu()
            ori_h, ori_w = batch["ori_shape"][si]
            areas = (raw_gt[:, 2] * float(ori_w) * raw_gt[:, 3] * float(ori_h)).tolist()
            size_total = Counter(size_bucket(float(area)) for area in areas)
            size_matched = Counter(size_bucket(float(areas[gi])) for _, gi, _ in matches if gi < len(areas))
            assignment = self._collect_assignment(batch, self._raw_preds, si)
            row = {
                "image": Path(pbatch["im_file"]).name,
                "gt_count": target_count,
                "prediction_count": pred_count,
                "tp_iou50": tp,
                "fp_iou50": int(pred_count - tp),
                "fn_iou50": int(target_count - tp),
                "matched_iou_mean": float(np.mean([value for _, _, value in matches])) if matches else None,
                "matched_iou_p50": float(np.percentile([value for _, _, value in matches], 50)) if matches else None,
                "matched_iou_p90": float(np.percentile([value for _, _, value in matches], 90)) if matches else None,
                "tp_conf_mean": float(np.mean([float(predn["conf"][pi]) for pi, _, _ in matches])) if matches else None,
                "fp_conf_mean": float(np.mean([float(predn["conf"][pi]) for pi in range(pred_count) if pi not in {m[0] for m in matches}])) if pred_count > tp else None,
                "small_gt": int(size_total["small"]),
                "medium_gt": int(size_total["medium"]),
                "large_gt": int(size_total["large"]),
                "small_tp": int(size_matched["small"]),
                "medium_tp": int(size_matched["medium"]),
                "large_tp": int(size_matched["large"]),
            }
            row.update({f"assign_{key}": value for key, value in assignment.items()})
            self.precision_rows.append(row)
            self.assignment_rows.append({"image": row["image"], **assignment})
        return super().update_metrics(preds, batch)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_checkpoint(root: Path, seed: int, cell: str) -> Path:
    exact = root / f"seed{seed}" / f"{cell}_formal_seed{seed}_15ep" / "weights" / "last.pt"
    if exact.is_file():
        return exact
    candidates = sorted(root.glob(f"**/{cell}_formal_seed{seed}_15ep/weights/last.pt"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one checkpoint for seed={seed} cell={cell}, got {candidates}")
    return candidates[0]


def git_state(root: Path) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "--stat"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "diff_stat": diff}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_cell(checkpoint: Path, data_yaml: Path, output: Path, device: str) -> dict:
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator

    class PrecisionValidator(PrecisionValidatorMixin, DetectionValidator):
        pass

    yolo = YOLO(str(checkpoint), task="detect")
    args = {
        "data": str(data_yaml),
        "model": str(checkpoint),
        "split": "val",
        "imgsz": 640,
        "batch": 1,
        "device": device,
        "workers": 0,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "plots": False,
        "verbose": False,
        "save_json": False,
        "save_txt": False,
        "project": str(output),
        "name": "validator",
        "exist_ok": True,
    }
    validator = PrecisionValidator(args=args)
    metrics = validator(model=yolo.model)
    cell_output = output / "validator"
    write_csv(cell_output / "per_image_metrics.csv", validator.precision_rows)
    write_csv(cell_output / "assignment_metrics.csv", validator.assignment_rows)
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "images": len(validator.precision_rows),
        "metrics": {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float, np.floating))},
        "per_image_metrics": str(cell_output / "per_image_metrics.csv"),
        "assignment_metrics": str(cell_output / "assignment_metrics.csv"),
        "assigner_branches": sorted({row.get("assign_assigner_branch") for row in validator.precision_rows}),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--cells", default=",".join(CELLS))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value]
    cells = [value.strip().lower() for value in args.cells.split(",") if value.strip()]
    if set(cells) - set(CELLS):
        raise ValueError(f"invalid cells: {cells}")
    val_list = args.data.parent / "val2017.txt"
    if not val_list.is_file() or len([line for line in val_list.read_text(encoding="utf-8").splitlines() if line.strip()]) != 512:
        raise ValueError(f"expected fixed 512-image val2017.txt next to data YAML: {val_list}")
    evidence = {
        "schema": "a1-p2-e2e-precision-r1/v1",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_ref": LOCKED_SHA,
        "python": sys.version,
        "platform": platform.platform(),
        "device": args.device,
        "data_yaml": str(args.data),
        "data_yaml_sha256": sha256(args.data),
        "val_list": str(val_list),
        "val_list_sha256": sha256(val_list),
        "images": 512,
        "imgsz": 640,
        "batch": 1,
        "cells": {},
    }
    for seed in seeds:
        evidence["cells"][str(seed)] = {}
        for cell in cells:
            checkpoint = find_checkpoint(args.checkpoint_root, seed, cell)
            print(f"[p2-e2e-r1] seed={seed} cell={cell} checkpoint={checkpoint}", flush=True)
            evidence["cells"][str(seed)][cell] = run_cell(
                checkpoint, args.data, args.output / f"seed{seed}" / cell, args.device
            )
            (args.output / "evidence.json").write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    evidence["git"] = git_state(Path.cwd())
    evidence["status"] = "completed"
    evidence["completed_at"] = datetime.now(timezone.utc).isoformat()
    (args.output / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
