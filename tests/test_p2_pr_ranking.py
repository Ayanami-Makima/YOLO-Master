"""Native TP reuse and tie-aware score-separation checks."""

import numpy as np
import pytest

from scripts.a1.analyze_p2_pr_ranking import analyze, separation


def test_separation_counts_ties_and_missing_groups():
    assert separation(np.array([0.7, 0.7]), np.array([True, False])) == 0.5
    assert separation(np.array([0.8, 0.2]), np.array([True, False])) == 1.0
    assert separation(np.array([0.2, 0.8]), np.array([True, False])) == 0.0
    assert separation(np.array([0.8]), np.array([True])) is None


def test_native_tp_avoids_recomputing_clipped_box_matching():
    row = {
        "boxes": {
            "gt": [{"cls": 0, "box": [0, 0, 10, 10]}],
            "retained_predictions": {
                "boxes": [[100, 100, 110, 110], [120, 120, 130, 130]],
                "scores": [0.9, 0.1],
                "classes": [0, 0],
                "native_tp": [[True] * 10, [False] * 10],
            },
        }
    }
    native, _ = analyze([row])
    assert native["coordinate_source"] == "native_validator_tp"
    assert native["map"] > 0.99
    del row["boxes"]["retained_predictions"]["native_tp"]
    reconstructed, _ = analyze([row])
    assert reconstructed["map"] == pytest.approx(0.0)
