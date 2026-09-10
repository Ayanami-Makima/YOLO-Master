"""Regressions for mismatched image order and missing-target bias in P2 diagnostics."""

from copy import deepcopy

import pytest
import torch

from scripts.a1.audit_p2_anchor_grid import compare as compare_grid
from scripts.a1.audit_p2_inverse_letterbox import compare, inverse_reference, unique_rows


def row(image, match):
    return {
        "image": image,
        "boxes": {"gt": [{"gt_id": 0, "cls": 1, "box": [1, 2, 20, 30], "match": match}], "coordinate_max_error_px": 0},
        "grid": {
            "scales": [{"valid_anchor_count": 4}] * 3,
            "anchor_max_error": 0,
            "stride_max_error": 0,
            "decode_max_error": 0,
            "shape_transition": False,
        },
    }


def matched(iou):
    return {"iou": iou, "confidence": 0.8, "center_error_norm": 0.01, "width_log_abs": 0.1, "height_log_abs": 0.1}


def test_reorder_and_missing_gt_are_not_zero_filled():
    left = [row("a", matched(0.8)), row("b", matched(0.9))]
    right = [row("b", None), row("a", matched(0.7))]
    summary, pairs = compare(left, right)
    assert summary["counts"] == {"both": 1, "square_only": 1, "rect_only": 0, "neither": 0}
    assert summary["common_gt_denominator"] == 1
    assert summary["common_gt_mean_delta"]["iou"] == pytest.approx(-0.1)
    assert pairs[1]["delta"] is None


def test_identity_and_duplicate_guard():
    with pytest.raises(ValueError, match="duplicate"):
        unique_rows([row("a", None), row("a", None)])
    left = [row("a", None)]
    right = deepcopy(left)
    right[0]["boxes"]["gt"][0]["cls"] = 2
    with pytest.raises(ValueError, match="GT identity"):
        compare(left, right)
    with pytest.raises(ValueError, match="image sets"):
        compare(left, [row("b", None)])


def test_no_common_matches_stays_null():
    summary, _ = compare([row("a", None)], [row("a", matched(0.8))])
    assert summary["common_gt_mean_delta"] is None
    assert summary["counts"]["rect_only"] == 1


def test_affine_inverse_known_translation_scale_and_clip():
    target = {"ratio_pad": ((2.0, 2.0), (10, 20)), "ori_shape": (100, 200)}
    boxes = torch.tensor([[30.0, 60, 110, 180], [-10, 0, 500, 250]])
    expected = torch.tensor([[10.0, 20, 50, 80], [0, 0, 200, 100]], dtype=torch.float64)
    assert torch.equal(inverse_reference(boxes, target), expected)


def test_grid_join_uses_id_not_position():
    def grid_row(name, fraction):
        return {"image": name, "grid": [{"feature_shape": [2, 3], "valid_anchor_fraction": fraction}] * 3}

    summary, pairs = compare_grid([grid_row("a", 0.2), grid_row("b", 0.3)], [grid_row("b", 0.5), grid_row("a", 0.4)])
    assert summary["same_images"] == 2
    assert pairs[0]["grid"][0]["valid_anchor_fraction_rect"] == 0.4
