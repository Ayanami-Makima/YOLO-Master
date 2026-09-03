#!/usr/bin/env python3
"""Run a short, frozen-budget P2-E one-to-one classification-loss pilot.

The only experimental factor is ``A1_E2E_O2O_CLS_GAIN``. Gain 1.0 is the neutral control;
the treatment value is recorded in the run manifest. This runner reuses the P1 freeze and
hard-Top-2 policy checks and writes a resumable status file beside the training output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from run_p1_bn_frozen import (
    P1_ROUTING_PARAMS,
    R19_EXPLORATION_POLICY,
    configure_r19_exploration,
    enforce_and_schedule_p1_policy,
    enforce_p1_freeze_policy,
    runtime_policy_payload,
    validate_runtime_p1_policy,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_request(args: argparse.Namespace) -> dict:
    freeze = [0, 1, 2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    params = {
        "epochs": args.epochs,
        "imgsz": 640,
        "batch": 4,
        "workers": 0,
        "pretrained": True,
        "optimizer": "SGD",
        "lr0": 0.0001,
        "lrf": 0.2,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "nbs": 16,
        "warmup_epochs": 0.5,
        "warmup_bias_lr": 0.0,
        "warmup_momentum": 0.8,
        "freeze": freeze,
        "amp": False,
        "deterministic": True,
        "seed": args.seed,
        "patience": 0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 0,
        "cos_lr": False,
        "cache": False,
        "plots": False,
        "save": True,
        "save_period": 1,
        "val": True,
        "fraction": 1.0,
        "exist_ok": False,
        "resume": False,
        "device": args.device,
        "project": str(args.project),
        "name": args.name,
        **P1_ROUTING_PARAMS,
    }
    return {
        "schema": "a1-p2-e2e-cls-gain-r2/v1",
        "request_id": f"{args.name}_seed{args.seed}_{args.epochs}ep_gain{args.gain:g}",
        "factor": "one_to_one_classification_loss_gain",
        "gain": args.gain,
        "control_gain": 1.0,
        "inputs": {"model": str(args.model), "data": str(args.data), "task": "detect"},
        "params": params,
        "a1_policy": {
            "freeze_batch_norm": True,
            "freeze_residual_factor_bases": True,
            "routing_semantics": "deterministic_hard_top2_from_step_zero",
            "expert_dropout_rate": 0.0,
            "router_exploration": {**R19_EXPLORATION_POLICY, "base_seed": args.seed, "enabled": False},
            "e2e_o2o_cls_gain": args.gain,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=260829)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--device", default="1")
    args = parser.parse_args()
    if args.gain <= 0:
        raise ValueError("--gain must be positive")
    if not args.model.is_file() or not args.data.is_file():
        raise FileNotFoundError(f"missing model/data: {args.model}, {args.data}")
    save_dir = args.project / args.name
    if save_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing pilot directory: {save_dir}")
    args.project.mkdir(parents=True, exist_ok=True)
    manifest_path = args.project / f"{args.name}.p2_request.json"
    env_path = args.project / f"{args.name}.p2_env.json"
    status_path = args.project / f"{args.name}.status.json"
    request = build_request(args)
    manifest_path.write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    env_path.write_text(
        json.dumps({"A1_E2E_O2O_CLS_GAIN": args.gain, "created_at": utc_now()}, indent=2) + "\n",
        encoding="utf-8",
    )
    os.environ["A1_E2E_O2O_CLS_GAIN"] = str(args.gain)
    from ultralytics import YOLO

    model = YOLO(str(args.model), task="detect")

    def prepare_training(trainer) -> None:
        configure_r19_exploration(trainer, request)
        enforce_p1_freeze_policy(trainer)
        validate_runtime_p1_policy(trainer)
        (Path(trainer.save_dir) / "p1_runtime_policy_pretrain.json").write_text(
            json.dumps(runtime_policy_payload(trainer, request["request_id"]), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    model.add_callback("on_pretrain_routine_end", prepare_training)
    model.add_callback("on_train_batch_start", enforce_and_schedule_p1_policy)
    status = {"schema": "a1-p2-e2e-cls-gain-r2-status/v1", "status": "running", "started_at": utc_now(), "request": request}
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        results = model.train(data=str(args.data), **request["params"])
        trainer = model.trainer
        (Path(trainer.save_dir) / "p1_runtime_policy.json").write_text(
            json.dumps(runtime_policy_payload(trainer, request["request_id"]), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        status.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "save_dir": str(trainer.save_dir),
                "best": str(trainer.best),
                "last": str(trainer.last),
                "metrics": dict(getattr(results, "results_dict", {}) or {}),
            }
        )
    except BaseException as exc:
        status.update({"status": "failed", "completed_at": utc_now(), "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if status.get("save_dir"):
            Path(status["save_dir"]).mkdir(parents=True, exist_ok=True)
            (Path(status["save_dir"]) / "p2_status.json").write_text(
                json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
