#!/usr/bin/env python3
"""Run one A1 GPU lane through native async skills with strict artifact checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from prepare_coco2017 import sha256

DEFAULT_PROJECT = Path("/data/data2/TuJiajun/A1-smoke/p1_full")
OOM_MARKERS = (b"reducing to batch=", b"cuda out of memory", b"cuda backend memory error")
REQUIRED_METRICS = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")


def read_json(path: Path) -> dict:
    """Read one complete JSON evidence file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    """Replace a lane-owned status file atomically."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_locked_files(locked: dict[Path, str]) -> None:
    """Reject changed protocol, requests, model configs or dataset evidence."""
    for path, expected in locked.items():
        if sha256(path) != expected:
            raise ValueError(f"locked evidence changed: {path}")


def dataset_file_locks(protocol: dict) -> dict[Path, str]:
    """Bind both the formal data and optional isolated preflight sample lists."""
    locked = {
        Path(protocol["dataset_manifest"]).resolve(): protocol["dataset_manifest_sha256"],
        Path(protocol["data_yaml"]).resolve(): protocol["data_yaml_sha256"],
    }
    for key in ("selection_manifest", "evaluation_data_yaml", "checkpoint"):
        if protocol.get(key):
            locked[Path(protocol[key]).resolve()] = protocol[f"{key}_sha256"]
    preflight = protocol.get("preflight_data")
    if preflight:
        locked[Path(preflight["data_yaml"]).resolve()] = preflight["data_yaml_sha256"]
        for entry in preflight["lists"].values():
            locked[Path(entry["path"]).resolve()] = entry["sha256"]
    return locked


def read_final_payload(path: Path) -> dict | None:
    """Ignore incomplete writes and parse the last complete response in JSONL."""
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 2 * 1024 * 1024))
        text = stream.read().decode("utf-8", errors="replace")
    for line in reversed(text.splitlines(keepends=True)):
        if not line.endswith(("\n", "\r")):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("status") in {"ok", "failed", "blocked", "partial", "running"}:
            return payload
    return None


def process_start_token(pid: int, proc_root: Path = Path("/proc")) -> str | None:
    """Identify a live Linux process, excluding zombies and reused PIDs."""
    try:
        fields = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] in {"Z", "X"} else fields[19]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None


def process_alive(pid: int, token: str | None) -> bool:
    """Do not trust os.kill(pid, 0), which also succeeds for a zombie."""
    return token is not None and process_start_token(pid) == token


def stop_owned_job(job: dict, token: str | None) -> dict:
    """Stop only this lane's still-identical native async process group."""
    pid = int(job["pid"])
    if not process_alive(pid, token):
        return {"attempted": False, "reason": "owned dispatcher is no longer live"}
    if os.getpgid(pid) != pid:
        return {"attempted": False, "reason": "process group identity mismatch; inspect manually"}
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while process_alive(pid, token) and time.monotonic() < deadline:
        time.sleep(0.25)
    if process_alive(pid, token):
        os.killpg(pid, signal.SIGKILL)
    return {"attempted": True, "pid": pid, "reason": "lane failed its execution or protocol checks"}


class LogGuard:
    """Scan all raw log bytes incrementally, including across chunk boundaries."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.offsets: dict[Path, tuple[int, bytes]] = {}

    def paths(self) -> list[Path]:
        return sorted(self.directory.glob("cli-*/*.log"))

    def check(self) -> None:
        for path in self.paths():
            offset, overlap = self.offsets.get(path, (0, b""))
            with path.open("rb") as stream:
                if path.stat().st_size < offset:
                    raise ValueError(f"live log was truncated: {path}")
                stream.seek(offset)
                while chunk := stream.read(65536):
                    combined = overlap + chunk.lower()
                    if any(marker in combined for marker in OOM_MARKERS):
                        raise ValueError(
                            f"CUDA memory failure or automatic batch reduction violates the protocol: {path}"
                        )
                    overlap = combined[-128:]
                self.offsets[path] = (stream.tell(), overlap)


def verify_args(request: dict, actual: dict) -> None:
    """Compare every explicit training option, including model/data/device."""
    expected = {**request["inputs"], **request["params"]}
    for key, value in expected.items():
        if value is None:
            continue
        if key not in actual:
            raise ValueError(f"training did not record explicit {key}")
        observed = actual[key]
        if key in {"model", "data", "project"}:
            matches = isinstance(observed, str) and Path(observed).resolve() == Path(str(value)).resolve()
        elif key == "device":
            matches = str(observed) == str(value)
        else:
            matches = observed == value
        if not matches:
            raise ValueError(f"training changed {key}: {observed!r} != {value!r}")


def read_epoch_rows(path: Path) -> list[dict]:
    """Read complete CSV rows; an in-flight final row is not progress evidence."""
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines.pop()
    return [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(lines)]


def verify_training(request: dict, log_directory: Path) -> dict:
    """Require real completion evidence, not a submitted or terminated process."""
    params = request["params"]
    root = Path(params["project"]) / params["name"]
    verify_args(request, yaml.safe_load((root / "args.yaml").read_text(encoding="utf-8")))
    rows = read_epoch_rows(root / "results.csv")
    expected_epochs = list(range(1, int(params["epochs"]) + 1))
    if [int(row["epoch"]) for row in rows] != expected_epochs:
        raise ValueError(f"incomplete or duplicated epoch budget: {root}")
    for row in rows:
        if any(key not in row for key in REQUIRED_METRICS):
            raise ValueError(f"missing detection metrics: {root}")
        for key, value in row.items():
            if "loss" in key or key.startswith("metrics/"):
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f"non-finite result: {root}: {key}")
    checkpoints = {}
    for checkpoint in ("best.pt", "last.pt"):
        path = root / "weights" / checkpoint
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing or empty {checkpoint}: {root}")
        checkpoints[checkpoint] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    guard = LogGuard(log_directory)
    guard.check()
    paths = guard.paths()
    if {path.name for path in paths} != {"stdout.log", "stderr.log"}:
        raise ValueError(f"missing complete raw CLI logs: {log_directory}")
    return {
        "run_dir": str(root),
        "epochs": len(rows),
        "last_epoch_metrics": rows[-1],
        "checkpoints": checkpoints,
        "logs": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in paths],
    }


def validate_request(request: dict, protocol: dict, cell: str, phase: str, project: Path) -> Path:
    """Keep each cell on its locked GPU, data, model and common training budget."""
    params = request["params"]
    model_path = Path(protocol["matrix"][cell]["path"]).resolve()
    if Path(request["inputs"]["model"]).resolve() != model_path:
        raise ValueError(f"wrong model config for cell {cell}")
    if str(params["device"]) != ("0" if cell in "ac" else "1"):
        raise ValueError(f"wrong GPU for cell {cell}")
    if Path(params["project"]).resolve() != project:
        raise ValueError(f"request uses a different project: {cell}")
    root = (project / params["name"]).resolve()
    if root.parent != project:
        raise ValueError(f"run directory escapes the lane project: {root}")
    expected = dict(protocol["common_training"])
    if phase == "preflight":
        expected.update(epochs=1, close_mosaic=0, save_period=-1, workers=0)
    for key, value in expected.items():
        if params.get(key) != value:
            raise ValueError(f"request changed locked {key}: {cell}")
    if phase == "full" and Path(request["inputs"]["data"]).resolve() != Path(protocol["data_yaml"]).resolve():
        raise ValueError(f"wrong full COCO dataset for cell {cell}")
    if phase == "preflight" and protocol.get("preflight_data"):
        if Path(request["inputs"]["data"]).resolve() != Path(protocol["preflight_data"]["data_yaml"]).resolve():
            raise ValueError(f"wrong isolated preflight dataset for cell {cell}")
    if request.get("skill") != "yolo.train" or request.get("policy", {}).get("async") is not True:
        raise ValueError("lane requests must use native asynchronous yolo.train")
    if request.get("params", {}).get("resume"):
        raise ValueError("resume requires a separately reviewed request and checkpoint lineage")
    return root


def dispatch_training(
    request_path: Path,
    protocol_path: Path,
    protocol_hash: str,
    *,
    cell: str,
    phase: str,
    repo: Path,
    environment: dict,
    on_prepared=None,
) -> tuple[subprocess.CompletedProcess, Path, dict]:
    """Reserve a new run, record its immutable identity, then submit the skill.

    A prepared identity is not proof that training started or succeeded. Existing
    directories are never adopted, even if they contain only an old manifest.
    """
    protocol_path, request_path = protocol_path.resolve(), request_path.resolve()
    protocol, request = read_json(protocol_path), read_json(request_path)
    project = Path(protocol["run_root"]).resolve()
    root = validate_request(request, protocol, cell, phase, project)
    if request["params"].get("exist_ok") is not True:
        raise ValueError("exist_ok=True is required for the newly reserved manifest-only run directory")
    model_path = Path(request["inputs"]["model"]).resolve()
    data_path = Path(request["inputs"]["data"]).resolve()
    request_hash = protocol["requests_sha256"][f"{cell}_{phase}"]
    locked = {
        protocol_path: protocol_hash,
        request_path: request_hash,
        model_path: protocol["matrix"][cell]["sha256"],
        **dataset_file_locks(protocol),
    }
    locked.setdefault(data_path, sha256(data_path))
    check_locked_files(locked)
    identity = {
        "schema_version": 1,
        "status": "prepared",
        "recorded_before_dispatch": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "cell": cell,
        "lane": "0" if cell in "ac" else "1",
        "run_dir": str(root),
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "request_path": str(request_path),
        "request_sha256": request_hash,
        "model": str(model_path),
        "model_sha256": locked[model_path],
        "dataset_manifest": str(Path(protocol["dataset_manifest"]).resolve()),
        "dataset_manifest_sha256": protocol["dataset_manifest_sha256"],
        "data_yaml": str(data_path),
        "data_yaml_sha256": locked[data_path],
        "formal_data_yaml": str(Path(protocol["data_yaml"]).resolve()),
        "formal_data_yaml_sha256": protocol["data_yaml_sha256"],
        "training": request["params"],
    }
    if phase == "preflight" and protocol.get("preflight_data"):
        identity["dataset_lists"] = protocol["preflight_data"]["lists"]
    # mkdir is the reservation boundary; an existence check alone races another launcher.
    root.mkdir(exist_ok=False)
    identity_path = root / "training_manifest.json"
    with identity_path.open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    locked[identity_path] = sha256(identity_path)
    if on_prepared is not None:
        on_prepared(identity_path, identity)
    if set(root.iterdir()) != {identity_path}:
        raise ValueError(f"new run contains unexpected files before dispatch: {root}")
    check_locked_files(locked)
    dispatcher = repo / "agent/scripts/run_yolo_master_skill.py"
    command = [sys.executable, str(dispatcher), "--request", str(request_path)]
    proc = subprocess.run(command, cwd=repo, capture_output=True, text=True, env=environment, check=False)
    return proc, identity_path, identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=("0", "1"), required=True)
    parser.add_argument("--phase", choices=("preflight", "full"), required=True)
    parser.add_argument("--configs", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument(
        "--max-hours", type=float, default=0, help="Optional per-cell limit; zero means no wall-clock limit."
    )
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    if args.max_hours < 0 or args.poll_seconds <= 0:
        parser.error("max-hours must be non-negative and poll-seconds must be positive")
    repo = Path(__file__).resolve().parents[2]
    configs = (args.configs or repo / "configs/a1/full").resolve()
    protocol_path = configs / "protocol.json"
    protocol = read_json(protocol_path)
    project = (args.project or Path(protocol.get("run_root", DEFAULT_PROJECT))).resolve()
    if project != Path(protocol["run_root"]).resolve():
        raise ValueError("project differs from the locked protocol")
    protocol_hash = sha256(protocol_path)
    locked = {
        protocol_path: protocol_hash,
        **dataset_file_locks(protocol),
    }
    audit_path = (args.audit or project / "full_data_model_audit.json").resolve()
    audit = read_json(audit_path)
    if (
        audit.get("status") != "passed"
        or audit.get("protocol_sha256") != protocol_hash
        or audit.get("dataset_manifest_sha256") != protocol["dataset_manifest_sha256"]
    ):
        raise ValueError("model/data audit is missing, failed, or belongs to a different protocol")
    if protocol.get("preflight_data") and audit.get("preflight_cache_isolation", {}).get("status") != "passed":
        raise ValueError("isolated preflight label cache has not passed the data audit")
    locked[audit_path] = sha256(audit_path)
    for cell in "abcd":
        locked[Path(protocol["matrix"][cell]["path"]).resolve()] = protocol["matrix"][cell]["sha256"]
        request_path = configs / f"{cell}_{args.phase}_request.json"
        locked[request_path] = protocol["requests_sha256"][f"{cell}_{args.phase}"]
    check_locked_files(locked)
    project.mkdir(parents=True, exist_ok=True)
    # Training is remote/Linux-only; keep pure verification helpers importable on Windows.
    import fcntl

    lock = (project / f"lane{args.lane}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = project / f"{args.phase}_lane{args.lane}_state.json"
    progress = project / f"{args.phase}_lane{args.lane}_progress.jsonl"
    if state_path.exists() or progress.exists():
        raise ValueError(f"lane evidence already exists; inspect or explicitly plan recovery: {state_path}")
    if args.phase == "full":
        for lane in ("0", "1"):
            gate_path = project / f"preflight_lane{lane}_passed.json"
            gate = read_json(gate_path)
            if gate["protocol_sha256"] != protocol_hash:
                raise ValueError("preflight used a different protocol")
            locked[gate_path] = sha256(gate_path)
    state = {
        "phase": args.phase,
        "lane": args.lane,
        "pid": os.getpid(),
        "completed": [],
        "status": "running",
        "protocol_sha256": protocol_hash,
        "audit": str(audit_path),
        "max_hours": args.max_hours,
    }

    def update(**fields):
        state.update(fields, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json(state_path, state)
        with progress.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(state) + "\n")
        print(json.dumps(state), flush=True)

    environment = os.environ.copy()
    environment.update(
        YOLO_OFFLINE="true",
        YOLO_AUTOINSTALL="false",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
    )
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
    current_job, token = None, None
    try:
        for cell in "ca" if args.lane == "0" else "db":
            check_locked_files(locked)
            request_path = configs / f"{cell}_{args.phase}_request.json"
            request = read_json(request_path)
            root = validate_request(request, protocol, cell, args.phase, project)
            if root.exists():
                raise ValueError(f"run already exists; inspect or resume explicitly: {root}")
            data_path = Path(request["inputs"]["data"]).resolve()
            locked.setdefault(data_path, sha256(data_path))
            log_directory = project / "cli_logs" / f"{cell}_{args.phase}"
            log_directory.mkdir(parents=True, exist_ok=False)
            environment["YOLO_MASTER_CLI_LOG_DIR"] = str(log_directory)

            def prepared(identity_path, identity):
                locked[identity_path] = sha256(identity_path)
                update(
                    cell=cell,
                    job=None,
                    epoch=0,
                    latest_metrics=None,
                    elapsed_seconds=0,
                    cli_log_directory=str(log_directory),
                    cli_logs=[],
                    final_result=None,
                    child_alive=False,
                    launch_state="prepared",
                    training_manifest=str(identity_path),
                    training_manifest_sha256=locked[identity_path],
                    request_sha256=identity["request_sha256"],
                    model_sha256=identity["model_sha256"],
                    dataset_manifest_sha256=identity["dataset_manifest_sha256"],
                )

            proc, identity_path, identity = dispatch_training(
                request_path,
                protocol_path,
                protocol_hash,
                cell=cell,
                phase=args.phase,
                repo=repo,
                environment=environment,
                on_prepared=prepared,
            )
            submission_path = project / f"{cell}_{args.phase}_submission.json"
            submission = json.loads(proc.stdout.strip().splitlines()[-1])
            write_json(submission_path, submission)
            if proc.returncode or submission.get("status") != "running":
                raise ValueError(f"submission failed; see {submission_path}")
            current_job = submission["job"]
            token = process_start_token(int(current_job["pid"]))
            update(
                cell=cell,
                job=current_job,
                epoch=0,
                latest_metrics=None,
                elapsed_seconds=0,
                cli_log_directory=str(log_directory),
                cli_logs=[],
                final_result=None,
                child_alive=token is not None,
                launch_state="submitted",
            )
            started = time.monotonic()
            final_path = Path(current_job["stdout_path"])
            guard = LogGuard(log_directory)
            while True:
                check_locked_files({identity_path: locked[identity_path]})
                guard.check()
                final = read_final_payload(final_path)
                if final and final.get("status") != "running":
                    if final["status"] != "ok":
                        raise RuntimeError(f"skill failed: {final.get('summary')}; see {final_path}")
                    if str(final.get("job", {}).get("device")) != str(request["params"]["device"]):
                        raise ValueError(f"skill changed the selected GPU; see {final_path}")
                    break
                alive = process_alive(int(current_job["pid"]), token)
                if not alive:
                    # The final JSON may have been flushed between the first read and process exit.
                    final = read_final_payload(final_path)
                    if final and final.get("status") == "ok":
                        if str(final.get("job", {}).get("device")) != str(request["params"]["device"]):
                            raise ValueError(f"skill changed the selected GPU; see {final_path}")
                        break
                    raise RuntimeError(f"job exited without a successful final result: {final_path}")
                elapsed = time.monotonic() - started
                if args.max_hours and elapsed > args.max_hours * 3600:
                    raise TimeoutError(f"job exceeded the explicitly configured {args.max_hours}h limit")
                rows = read_epoch_rows(root / "results.csv")
                update(
                    epoch=int(rows[-1]["epoch"]) if rows else 0,
                    latest_metrics=rows[-1] if rows else None,
                    elapsed_seconds=round(elapsed, 1),
                    child_alive=alive,
                    cli_logs=[str(path) for path in guard.paths()],
                )
                time.sleep(args.poll_seconds)
            check_locked_files(locked)
            verified = verify_training(request, log_directory)
            state["completed"].append(
                {
                    "cell": cell,
                    "training_manifest": str(identity_path),
                    "training_manifest_sha256": locked[identity_path],
                    "request_sha256": identity["request_sha256"],
                    **verified,
                }
            )
            update(
                epoch=verified["epochs"],
                latest_metrics=verified["last_epoch_metrics"],
                child_alive=False,
                elapsed_seconds=round(time.monotonic() - started, 1),
                final_result=str(final_path),
                launch_state="verified",
            )
            current_job, token = None, None
        update(status="completed")
        if args.phase == "preflight":
            write_json(
                project / f"preflight_lane{args.lane}_passed.json",
                {
                    "protocol_sha256": protocol_hash,
                    "dataset_manifest_sha256": protocol["dataset_manifest_sha256"],
                    "audit_sha256": locked[audit_path],
                    "results": state["completed"],
                },
            )
    except Exception as exc:
        cleanup = None
        if current_job:
            try:
                cleanup = stop_owned_job(current_job, token)
            except OSError as cleanup_error:
                cleanup = {"attempted": True, "error": str(cleanup_error)}
        update(
            status="failed",
            error=str(exc),
            cleanup=cleanup,
            child_alive=bool(current_job and process_alive(int(current_job["pid"]), token)),
        )
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
