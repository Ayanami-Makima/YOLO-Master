"""Unit tests for the pretrained A1 P0 protocol."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_p0_pretrained.py"
SPEC = importlib.util.spec_from_file_location("evaluate_p0_pretrained", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_validation_requests_differ_only_by_end2end(tmp_path):
    common = {
        "repo": tmp_path,
        "checkpoint": tmp_path / "same.pt",
        "data": tmp_path / "coco.yaml",
        "output": tmp_path / "out",
        "imgsz": 640,
        "batch": 16,
        "device": "0",
        "workers": 0,
        "nms_iou": 0.7,
        "max_det": 300,
    }
    a = MODULE.build_validation_request(cell="A", end2end=False, **common)
    b = MODULE.build_validation_request(cell="B", end2end=True, **common)
    for request in (a, b):
        request["request_id"] = "same"
        request["params"]["name"] = "same"
        request["artifacts"]["name"] = "same"
    assert a["inputs"] == b["inputs"]
    a["params"].pop("end2end")
    b["params"].pop("end2end")
    assert a == b


def test_extract_validation_uses_dispatcher_metrics():
    result = MODULE.extract_validation(
        {
            "status": "ok",
            "metrics": {
                "metrics/precision(B)": 0.7,
                "metrics/recall(B)": 0.6,
                "metrics/mAP50(B)": 0.5,
                "metrics/mAP50-95(B)": 0.4,
            },
            "manifest": "/tmp/manifest.json",
        }
    )
    assert result["map50_95"] == 0.4
    assert result["map50"] == 0.5


def test_set_head_mode_updates_model_and_head():
    head = SimpleNamespace(end2end=True, one2one_cv2=object(), one2one_cv3=object())

    class Core:
        def __init__(self):
            self.model = [head]
            self.parameters = lambda: []

        @property
        def end2end(self):
            return self.model[-1].end2end

        @end2end.setter
        def end2end(self, value):
            self.model[-1].end2end = value

    model = SimpleNamespace(model=Core())
    assert MODULE.set_head_mode(model, False)["end2end"] is False
    assert head.end2end is False
    assert MODULE.set_head_mode(model, True)["end2end"] is True
    assert head.end2end is True


def test_build_report_computes_accuracy_and_latency_effects():
    validation = {
        "cells": {
            "A": {"precision": 0.7, "recall": 0.6, "map50": 0.5, "map50_95": 0.4},
            "B": {"precision": 0.69, "recall": 0.59, "map50": 0.49, "map50_95": 0.385},
        }
    }
    prediction = {
        "cells": {
            "A": {"detections": 10, "nms": {"suppression_kernel_calls": 2, "wrapper_routes": {"nms": 2}}},
            "B": {
                "detections": 9,
                "nms": {"suppression_kernel_calls": 0, "wrapper_routes": {"end2end_score_filter": 2}},
            },
        }
    }
    latency = {
        "devices": {
            "0": {
                "A": {
                    "statistics": {"total": {"mean_ms": 4.0, "std_ms": 0.2, "p50_ms": 3.9, "p90_ms": 4.2}}
                },
                "B": {
                    "statistics": {"total": {"mean_ms": 3.5, "std_ms": 0.1, "p50_ms": 3.4, "p90_ms": 3.7}}
                },
            }
        }
    }
    report = MODULE.build_report({}, validation, prediction, latency)
    assert abs(report["accuracy"]["B_minus_A_ap_points"] + 1.5) < 1e-9
    assert report["latency"]["0"]["B_minus_A_ms"] == -0.5
    assert report["nms"]["B"]["suppression_kernel_calls"] == 0
