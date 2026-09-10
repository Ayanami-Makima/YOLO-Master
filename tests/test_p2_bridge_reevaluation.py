"""Hand-constructed fixed-confidence diagnostic checks."""

import torch

from scripts.a1.evaluate_p2_gradient_bridge import errors


def test_error_categories():
    target = {"bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "cls": torch.tensor([0.0])}
    pred = {
        "bboxes": torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
                [5.0, 0.0, 15.0, 10.0],
                [5.0, 0.0, 15.0, 10.0],
                [30.0, 30.0, 40.0, 40.0],
            ]
        ),
        "cls": torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        "conf": torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
    }
    result = errors(pred, target)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 5, 0)
    assert all(
        result["fp_" + k] == 1
        for k in ("duplicate", "classification", "localization", "classification_localization", "background")
    )


def test_no_predictions_and_no_targets():
    target = {"bboxes": torch.empty(0, 4), "cls": torch.empty(0)}
    pred = {**target, "conf": torch.empty(0)}
    assert errors(pred, target)["fp"] == 0
    pred = {"bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "cls": torch.tensor([0.0]), "conf": torch.tensor([0.3])}
    assert errors(pred, target)["fp_background"] == 1
    pred["conf"][:] = 0.2
    assert errors(pred, target)["fp"] == 0
