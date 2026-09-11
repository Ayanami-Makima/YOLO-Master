#!/usr/bin/env python3
"""Run the C/D routing probes concurrently on two physical GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_p1_factorial_preflights import read_last_metrics, write_json

GPU_BY_CELL = {"c": "0", "d": "1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--runner", type=Path, default=Path("scripts/a1/run_p1_bn_frozen.py"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    repo = protocol_path.parents[3]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    run_root = Path(protocol["run_root"]) / "routing_probe"
    status_path = run_root / "routing_probe_status.json"
    if status_path.exists():
        raise FileExistsError(f"refusing to overwrite routing status: {status_path}")

    status = {
        "schema_version": 2,
        "protocol": str(protocol_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "execution": "concurrent physical GPU0/GPU1; request-visible device remains logical GPU0",
        "gpu_by_cell": GPU_BY_CELL,
        "cells": {},
    }
    processes = {}
    handles = {}
    for cell, physical_gpu in GPU_BY_CELL.items():
        request_path = Path(protocol["requests"]["routing_probe"][cell]["path"])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        params = request.get("params", {})
        if params.get("epochs") != 1 or params.get("device") != "0" or params.get("exist_ok") is not False:
            raise ValueError(f"{cell}: routing probe request violates the locked budget")
        expected_run_dir = Path(params["project"]) / params["name"]
        if expected_run_dir.exists():
            raise FileExistsError(f"refusing to overwrite routing probe: {expected_run_dir}")
        log_path = run_root / f"{cell}_driver.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("x", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = physical_gpu
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            [sys.executable, str(args.runner), "--request", str(request_path)],
            cwd=repo,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        handles[cell] = handle
        processes[cell] = (process, request_path, expected_run_dir, log_path)
        status["cells"][cell] = {
            "status": "running",
            "pid": process.pid,
            "physical_gpu": physical_gpu,
            "logical_device": "0",
            "request": str(request_path),
            "expected_run_dir": str(expected_run_dir),
            "log": str(log_path),
        }
    write_json(status_path, status)

    any_failed = False
    for cell, physical_gpu in GPU_BY_CELL.items():
        process, request_path, run_dir, log_path = processes[cell]
        returncode = process.wait()
        handles[cell].close()
        failure_report = run_dir / "failure_diagnostics.json"
        runtime_policy = run_dir / "p1_runtime_policy.json"
        status["cells"][cell] = {
            "status": "completed" if returncode == 0 else "failed",
            "returncode": returncode,
            "physical_gpu": physical_gpu,
            "logical_device": "0",
            "request": str(request_path),
            "run_dir": str(run_dir),
            "log": str(log_path),
            "failure_report": str(failure_report) if failure_report.exists() else None,
            "runtime_policy": str(runtime_policy) if runtime_policy.exists() else None,
            "metrics": read_last_metrics(run_dir),
        }
        any_failed |= returncode != 0
        write_json(status_path, status)

    status["status"] = "failed" if any_failed else "completed"
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(status_path, status)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    if any_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
