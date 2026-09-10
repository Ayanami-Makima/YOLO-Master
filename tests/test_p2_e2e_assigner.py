"""Focused tests for the P2 one-to-one conflict-resolution pilot."""

import pytest
import torch

from ultralytics.utils.tal import TaskAlignedAssigner


def _conflicting_masks():
    # Both GTs select anchor 0; anchor 1 belongs only to GT 0. The two
    # metrics intentionally disagree on which GT should keep anchor 0.
    mask_pos = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])
    overlaps = torch.tensor([[[0.9, 0.2], [0.8, 0.1]]])
    align_metric = torch.tensor([[[0.1, 0.3], [0.8, 0.0]]])
    return mask_pos, overlaps, align_metric


def test_conflict_metric_defaults_to_native_overlap_rule():
    mask_pos, overlaps, align_metric = _conflicting_masks()
    assigner = TaskAlignedAssigner(topk=2)
    target_gt_idx, fg_mask, _ = assigner.select_highest_overlaps(mask_pos, overlaps, 2, align_metric)
    assert target_gt_idx.tolist() == [[0, 0]]
    assert fg_mask.tolist() == [[1.0, 1.0]]


def test_align_conflict_metric_uses_task_aligned_scores():
    mask_pos, overlaps, align_metric = _conflicting_masks()
    assigner = TaskAlignedAssigner(topk=2, conflict_metric="align")
    target_gt_idx, fg_mask, _ = assigner.select_highest_overlaps(mask_pos, overlaps, 2, align_metric)
    assert target_gt_idx.tolist() == [[1, 0]]
    assert fg_mask.tolist() == [[1.0, 1.0]]


def test_conflict_metric_rejects_unknown_value():
    with pytest.raises(ValueError, match="conflict_metric"):
        TaskAlignedAssigner(conflict_metric="unknown")
