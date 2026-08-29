#!/usr/bin/env python3
"""Run equal-budget P1 preflights sequentially and stop at the first failure."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CELLS = "abcd"


def write_json(path: Path, payload: dict) -> None:
    """Write one status snapshot atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_last_metrics(run_dir: Path) -> dict:
    """Read the last metrics row from one Ultralytics results file."""
    results = run_dir / "results.csv"
    rows = list(csv.DictReader(results.open(encoding="utf-8"))) if results.is_file() else []
    if not rows:
        return {}
    row = rows[-1]
    fields = (
        "epoch",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    )
    return {field: row.get(field) for field in fields}


def resolve_completed_run_dir(expected: Path) -> Path:
    """Resolve an Ultralytics-incremented run directory containing results."""
    candidates = [expected, *sorted(expected.parent.glob(f"{expected.name}-*"))]
    completed = [candidate for candidate in candidates if (candidate / "results.csv").is_file()]
    return max(completed, key=lambda path: (path / "results.csv").stat().st_mtime) if completed else expected


def validate_preflight_request(request: dict) -> None:
    """Reject requests that could violate the one-epoch discard-only protocol."""
    if request.get("skill") != "yolo.train":
        raise ValueError("preflight must use yolo.train")
    params = request.get("params", {})
    if params.get("epochs") != 1:
        raise ValueError("preflight must run exactly one epoch")
    if params.get("device") != "0":
        raise ValueError("all factorial preflights must run sequentially on GPU0")
    if params.get("exist_ok") is not False:
        raise ValueError("preflight must refuse to overwrite an existing run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--runner", type=Path, default=Path("scripts/a1/run_p1_bn_frozen.py"))
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    repo = protocol_path.parents[3]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    run_root = Path(protocol["run_root"]) / "preflight"
    status_path = run_root / "preflight_status.json"
    status = {
        "schema_version": 1,
        "protocol": str(protocol_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "cells": {},
    }
    write_json(status_path, status)

    if args.audit_only:
        for cell in CELLS:
            request_path = Path(protocol["requests"]["preflight"][cell]["path"])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            expected = Path(request["params"]["project"]) / request["params"]["name"]
            run_dir = resolve_completed_run_dir(expected)
            metrics = read_last_metrics(run_dir)
            status["cells"][cell] = {
                "status": "completed" if metrics else "missing",
                "request": str(request_path),
                "run_dir": str(run_dir),
                "metrics": metrics,
            }
        status["status"] = (
            "completed" if all(item["status"] == "completed" for item in status["cells"].values()) else "incomplete"
        )
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, status)
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return

    for cell in CELLS:
        request_path = Path(protocol["requests"]["preflight"][cell]["path"])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        validate_preflight_request(request)
        expected_run_dir = Path(request["params"]["project"]) / request["params"]["name"]
        log_path = run_root / f"{cell}_driver.log"
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
        runtime_policy = run_dir / "p1_runtime_policy.json"
        cell_status = {
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "request": str(request_path),
            "run_dir": str(run_dir),
            "log": str(log_path),
            "failure_report": str(failure_report) if failure_report.exists() else None,
            "runtime_policy": str(runtime_policy) if runtime_policy.exists() else None,
            "metrics": read_last_metrics(run_dir),
        }
        status["cells"][cell] = cell_status
        if result.returncode:
            status["status"] = "failed"
            status["failed_cell"] = cell
            status["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(status_path, status)
            raise SystemExit(result.returncode)
        write_json(status_path, status)

    status["status"] = "completed"
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(status_path, status)
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
