#!/usr/bin/env python3
"""Run the C/D routing preflights sequentially from locked initializers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_p1_factorial_preflights import read_last_metrics, resolve_completed_run_dir, write_json

CELLS = "cd"


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
    status = {
        "schema_version": 1,
        "protocol": str(protocol_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "cells": {},
    }
    write_json(status_path, status)

    for cell in CELLS:
        request_path = Path(protocol["requests"]["routing_probe"][cell]["path"])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        params = request.get("params", {})
        if params.get("epochs") != 1 or params.get("device") != "0" or params.get("exist_ok") is not False:
            raise ValueError(f"{cell}: routing probe must be one epoch, sequential on GPU0, without overwrite")
        expected_run_dir = Path(params["project"]) / params["name"]
        if expected_run_dir.exists():
            raise FileExistsError(f"refusing to overwrite routing probe: {expected_run_dir}")
        log_path = run_root / f"{cell}_driver.log"
        status["current_cell"] = cell
        status["cells"][cell] = {"status": "running", "request": str(request_path), "log": str(log_path)}
        write_json(status_path, status)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, str(args.runner), "--request", str(request_path)],
                cwd=repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        run_dir = resolve_completed_run_dir(expected_run_dir)
        failure_report = run_dir / "failure_diagnostics.json"
        status["cells"][cell] = {
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "request": str(request_path),
            "run_dir": str(run_dir),
            "log": str(log_path),
            "failure_report": str(failure_report) if failure_report.exists() else None,
            "metrics": read_last_metrics(run_dir),
        }
        if result.returncode:
            status["status"] = "failed"
            status["failed_cell"] = cell
            status["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(status_path, status)
            raise SystemExit(result.returncode)
        write_json(status_path, status)

    status.pop("current_cell", None)
    status["status"] = "completed"
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(status_path, status)
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
