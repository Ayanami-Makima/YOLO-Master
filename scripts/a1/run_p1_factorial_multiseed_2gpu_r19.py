#!/usr/bin/env python3
"""Run immutable r19 preflight, routing-probe, or formal waves on two GPUs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PREFORMAL_SCHEDULE = {
    260829: (("a", "0", "b", "1"), ("d", "0", "c", "1")),
    260830: (("c", "0", "d", "1"), ("b", "0", "a", "1")),
    260831: (("a", "0", "c", "1"), ("d", "0", "b", "1")),
}
PROBE_SCHEDULE = {
    260829: (("c", "0", "d", "1"),),
    260830: (("d", "0", "c", "1"),),
    260831: (("c", "0", "d", "1"),),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("preflight", "routing_probe", "formal"))
    parser.add_argument("--runner", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return a file SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    """Atomically update one status file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_metrics(run_dir: Path) -> dict:
    """Read the final exact-run metrics row without suffix discovery."""
    path = run_dir / "results.csv"
    if not path.is_file():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return rows[-1] if rows else {}


def gpu_uuids() -> dict[str, str]:
    """Resolve physical GPU indices to immutable UUIDs."""
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
    )
    return {index.strip(): uuid.strip() for index, uuid in (line.split(",", 1) for line in output.splitlines())}


def request_entry(protocol: dict, stage: str, seed: int, cell: str) -> tuple[Path, dict]:
    """Load and hash-check one registered request."""
    entry = protocol["requests"][stage][str(seed)][cell]
    path = Path(entry["path"])
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"request hash drift: {path}")
    request = json.loads(path.read_text(encoding="utf-8"))
    params = request["params"]
    expected_epochs = 5 if stage == "formal" else 1
    if (
        params.get("epochs") != expected_epochs
        or params.get("seed") != seed
        or params.get("device") != "0"
        or params.get("exist_ok") is not False
        or params.get("resume") is not False
    ):
        raise ValueError(f"{stage}/{seed}/{cell}: request violates locked execution semantics")
    if not Path(request["inputs"]["model"]).is_file():
        raise FileNotFoundError(request["inputs"]["model"])
    return path, request


def formal_admission(protocol: dict, protocol_path: Path) -> dict:
    """Require a separately generated all-seed gate before formal training."""
    path = Path(protocol["run_root"]) / "audits" / "formal_admission.json"
    if not path.is_file():
        raise FileNotFoundError(f"formal admission evidence is missing: {path}")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("status") != "passed" or evidence.get("protocol_sha256") != sha256(protocol_path):
        raise ValueError(f"formal admission evidence did not pass for this protocol: {evidence}")
    return {"path": str(path), "sha256": sha256(path)}


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    first_request_path = Path(protocol["requests"]["preflight"][str(protocol["seeds"][0])]["a"]["path"])
    first_request = json.loads(first_request_path.read_text(encoding="utf-8"))
    repo = Path(first_request["runtime"]["cwd"])
    runner = args.runner.resolve() if args.runner else repo / "scripts/a1/run_p1_bn_frozen.py"
    if not runner.is_file():
        raise FileNotFoundError(runner)
    admission = formal_admission(protocol, protocol_path) if args.stage == "formal" else None
    stage_root = Path(protocol["run_root"]) / args.stage
    status_path = stage_root / f"{args.stage}_status.json"
    if status_path.exists() or stage_root.exists():
        raise FileExistsError(f"refusing to reuse r19 stage directory: {stage_root}")
    status = {
        "schema_version": 5,
        "stage": args.stage,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "formal_admission": admission,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "gpu_uuid_by_physical_index": gpu_uuids(),
        "waves": [],
    }
    write_json(status_path, status)
    schedule = PROBE_SCHEDULE if args.stage == "routing_probe" else PREFORMAL_SCHEDULE
    for seed in protocol["seeds"]:
        for wave_index, wave in enumerate(schedule[seed], start=1):
            assignments = ((wave[0], wave[1]), (wave[2], wave[3]))
            wave_status = {"seed": seed, "wave": wave_index, "status": "running", "jobs": {}}
            processes = {}
            handles = {}
            for cell, physical_gpu in assignments:
                request_path, request = request_entry(protocol, args.stage, seed, cell)
                run_dir = Path(request["params"]["project"]) / request["params"]["name"]
                log_path = stage_root / f"seed{seed}" / f"{cell}_driver.log"
                if run_dir.exists() or log_path.exists():
                    raise FileExistsError(f"refusing to adopt/reuse run artifacts: {run_dir} / {log_path}")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open("x", encoding="utf-8")
                env = os.environ.copy()
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["CUDA_VISIBLE_DEVICES"] = physical_gpu
                env["PYTHONUNBUFFERED"] = "1"
                python = request["runtime"]["python"]
                process = subprocess.Popen(
                    [python, str(runner), "--request", str(request_path)],
                    cwd=repo,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                handles[cell] = handle
                processes[cell] = (process, request_path, request, run_dir, log_path, physical_gpu)
                wave_status["jobs"][cell] = {
                    "status": "running",
                    "pid": process.pid,
                    "physical_gpu": physical_gpu,
                    "physical_gpu_uuid": status["gpu_uuid_by_physical_index"][physical_gpu],
                    "logical_device": "0",
                    "request": str(request_path),
                    "run_dir": str(run_dir),
                    "log": str(log_path),
                }
            status["waves"].append(wave_status)
            write_json(status_path, status)
            failed = False
            for cell, (process, request_path, request, run_dir, log_path, physical_gpu) in processes.items():
                returncode = process.wait()
                handles[cell].close()
                failed |= returncode != 0
                wave_status["jobs"][cell] = {
                    "status": "completed" if returncode == 0 else "failed",
                    "returncode": returncode,
                    "physical_gpu": physical_gpu,
                    "physical_gpu_uuid": status["gpu_uuid_by_physical_index"][physical_gpu],
                    "logical_device": "0",
                    "request": str(request_path),
                    "request_sha256": sha256(request_path),
                    "initializer": request["inputs"]["model"],
                    "initializer_sha256": sha256(Path(request["inputs"]["model"])),
                    "run_dir": str(run_dir),
                    "log": str(log_path),
                    "runtime_policy": str(run_dir / "p1_runtime_policy.json")
                    if (run_dir / "p1_runtime_policy.json").is_file()
                    else None,
                    "last": str(run_dir / "weights/last.pt") if (run_dir / "weights/last.pt").is_file() else None,
                    "metrics": read_metrics(run_dir),
                }
            wave_status["status"] = "failed" if failed else "completed"
            write_json(status_path, status)
            if failed:
                status["status"] = "failed"
                status["finished_at"] = datetime.now(timezone.utc).isoformat()
                write_json(status_path, status)
                raise SystemExit(1)
    status["status"] = "completed"
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(status_path, status)
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
