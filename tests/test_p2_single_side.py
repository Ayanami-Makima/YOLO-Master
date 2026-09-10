"""Threshold and competition regression tests for descriptive single-side GT analysis."""

import pytest

from scripts.a1.summarize_p2_single_side import diagnose_miss, greedy_gt_ids


@pytest.mark.parametrize(
    "score,box,category,reason",
    [
        (0.2499, [0, 0, 10, 10], 1, "score_threshold"),
        (0.25, [0, 0, 10, 10], 1, "evaluation_match_competition"),
        (0.8, [4, 0, 14, 10], 1, "localization_threshold"),
        (0.8, [0, 0, 10, 10], 2, "wrong_class_overlap"),
        (0.15, [4, 0, 14, 10], 1, "joint_score_localization"),
    ],
)
def test_score_class_and_overlap_boundaries(score, box, category, reason):
    result = diagnose_miss(
        {"cls": 1, "box": [0, 0, 10, 10]}, {"scores": [score], "boxes": [box], "classes": [category]}
    )
    assert result["reason"] == reason


def test_empty_candidates_preserve_missing_evidence():
    result = diagnose_miss({"cls": 1, "box": [0, 0, 10, 10]}, {"scores": [], "boxes": [], "classes": []})
    assert result["reason"] == "no_near_retained_candidate"
    assert result["same_class_best_iou"] is None


def test_one_prediction_cannot_match_two_gt_and_threshold_is_inclusive():
    gt = [{"cls": 1, "box": [0, 0, 10, 10]}, {"cls": 1, "box": [1, 0, 11, 10]}]
    row = {"boxes": {"gt": gt, "retained_predictions": {"scores": [0.25], "classes": [1], "boxes": [[0, 0, 10, 10]]}}}
    assert greedy_gt_ids(row, 0.25) == {0}
    assert greedy_gt_ids(row, 0.251) == set()
