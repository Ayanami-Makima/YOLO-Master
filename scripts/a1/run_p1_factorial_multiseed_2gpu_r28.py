#!/usr/bin/env python3
"""Run immutable r28 stages with at most one independent job per GPU.

A GPU with at least 12 GiB free admits exactly one job. Requests still see
logical device 0 through CUDA_VISIBLE_DEVICES, and every observed capacity and
assignment is recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from p1_r28_runtime import RUNTIME_ATTESTATION, assert_protocol_runtime

# isort: split

from p1_r28_integrity import verify_registered_data_content

SEEDS = (260829, 260830, 260831)
GPU_INDICES = ("0", "1")
ONE_JOB_FREE_MEMORY_MIB = 12000
CLEAN_AUX_POLICY = {
    "enabled": True,
    "runtime_semantics": "clean_hard_top2_balance_with_noisy_dispatch",
    "dispatch_source": "train_only_private_noisy_logits",
    "dispatch_operator": "hard_top2",
    "balance_probability_source": "clean_logits_softmax",
    "balance_assignment_source": "clean_logits_hard_top2",
    "z_loss_source": "clean_logits",
    "evaluation_source": "clean_logits_hard_top2",
    "adds_parameters": False,
    "changes_inference": False,
}


def stage_jobs(stage: str) -> tuple[tuple[int, str], ...]:
    cells = "cd" if stage == "routing_probe" else "abcd"
    return tuple((seed, cell) for seed in SEEDS for cell in cells)


def capacity_for_free_memory(free_mib: int) -> int:
    return int(free_mib >= ONE_JOB_FREE_MEMORY_MIB)


def stage_schedule(
    stage: str, capacities: dict[str, int]
) -> tuple[tuple[tuple[int, str, str], ...], ...]:
    """Assign the ordered seed/cell queue without exceeding per-GPU capacity."""
    pending = list(stage_jobs(stage))
    waves: list[tuple[tuple[int, str, str], ...]] = []
    if sum(capacities.get(index, 0) for index in GPU_INDICES) < 1:
        raise ValueError("neither GPU has enough free memory for one r28 job")
    while pending:
        wave: list[tuple[int, str, str]] = []
        for index in GPU_INDICES:
            for _ in range(capacities.get(index, 0)):
                if pending:
                    seed, cell = pending.pop(0)
                    wave.append((seed, cell, index))
        waves.append(tuple(wave))
    return tuple(waves)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("preflight", "routing_probe", "formal"))
    parser.add_argument("--runner", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalized(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").lower()


def read_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "results.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def gpu_inventory() -> dict[str, dict[str, str]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    inventory = {}
    for line in output.splitlines():
        index, uuid, name, driver, total, used, free = (value.strip() for value in line.split(",", 6))
        inventory[index] = {
            "uuid": uuid,
            "name": name,
            "driver_version": driver,
            "memory_total_mib": int(total),
            "memory_used_mib": int(used),
            "memory_free_mib": int(free),
        }
    return inventory


def gpu_compute_processes() -> list[dict[str, str]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
        text=True,
    )
    records = []
    for line in output.splitlines():
        if line.strip():
            uuid, pid, process_name = (value.strip() for value in line.split(",", 2))
            records.append({"gpu_uuid": uuid, "pid": pid, "process_name": process_name})
    return records


def validate_protocol(protocol: dict[str, Any], protocol_path: Path) -> None:
    require(protocol.get("schema_version") == 8, "r28 runner requires protocol schema 8")
    require(protocol.get("experiment_tag") == "r28", "runner protocol is not r28")
    require(tuple(protocol.get("seeds", ())) == SEEDS, "r28 seed registry drift")
    require(protocol.get("request_count") == 30, "r28 protocol must register exactly 30 requests")
    run_root = Path(protocol["run_root"])
    require(run_root.name == "p1_factorial_medium_r28", f"unexpected run root: {run_root}")
    require("r19" not in normalized(run_root), f"run root references r19: {run_root}")
    require(
        protocol.get("gpu_policy", {}).get("co_location")
        == {
            "authorized": True,
            "authorization": "user explicitly required at most one job per GPU on 2026-08-30",
            "one_job_minimum_free_memory_mib": ONE_JOB_FREE_MEMORY_MIB,
            "maximum_jobs_per_gpu": 1,
            "preexisting_compute_processes": "allowed_and_recorded",
            "oom_policy": "fail_closed_no_resume_no_budget_change",
            "accuracy_caveat": "foreign GPU load may change wall-clock time but not the locked optimization budget",
        },
        "shared-GPU co-location policy drift",
    )
    registered_path = Path(protocol.get("formal_admission", {}).get("path", ""))
    require(
        registered_path.resolve() == (run_root / "audits/formal_admission.json").resolve(),
        "formal admission path drift",
    )
    expected_counts = {
        "pilot": {"train": 5000, "val": 512},
        "preflight": {"train": 256, "val": 128},
        "formal": {"train": 20000, "val": 5000},
    }
    for label, split_counts in expected_counts.items():
        registry = protocol.get("data", {}).get(label, {})
        yaml_path = Path(registry.get("path", ""))
        require(yaml_path.is_file(), f"{label} data YAML is missing: {yaml_path}")
        require(sha256(yaml_path) == registry.get("sha256"), f"{label} data YAML hash drift")
        require(set(registry.get("lists", {})) == set(split_counts), f"{label} list registry drift")
        for split, expected_count in split_counts.items():
            item = registry["lists"][split]
            list_path = Path(item.get("path", ""))
            require(list_path.is_file(), f"{label}/{split} image list is missing: {list_path}")
            require(sha256(list_path) == item.get("sha256"), f"{label}/{split} image-list hash drift")
            count = sum(bool(line.strip()) for line in list_path.read_text(encoding="utf-8").splitlines())
            require(item.get("images") == expected_count == count, f"{label}/{split} image count drift")
    verify_registered_data_content(protocol)
    for stage, expected_cells in (("preflight", set("abcd")), ("routing_probe", set("cd")), ("formal", set("abcd"))):
        seed_map = protocol["requests"].get(stage, {})
        require(set(seed_map) == {str(seed) for seed in SEEDS}, f"{stage}: seed set drift")
        for seed in SEEDS:
            require(set(seed_map[str(seed)]) == expected_cells, f"{stage}/{seed}: cell set drift")
            for cell, entry in seed_map[str(seed)].items():
                request_path = Path(entry["path"])
                require(request_path.is_file(), f"missing request: {request_path}")
                require(sha256(request_path) == entry["sha256"], f"request hash drift: {request_path}")
                require(protocol_path.parent in request_path.resolve().parents, f"request is outside r28 config: {request_path}")
                request = json.loads(request_path.read_text(encoding="utf-8"))
                data_label = {"preflight": "preflight", "routing_probe": "pilot", "formal": "formal"}[stage]
                require(
                    Path(request.get("inputs", {}).get("data", "")).resolve()
                    == Path(protocol["data"][data_label]["path"]).resolve(),
                    f"{stage}/{seed}/{cell}: data path drift",
                )


def locked_runner(protocol: dict[str, Any], repo: Path, override: Path | None) -> tuple[Path, str]:
    """Resolve the only runner admitted by the implementation registry."""
    relative = "scripts/a1/run_p1_bn_frozen_r28.py"
    expected = (repo / relative).resolve()
    require(expected.is_file(), f"locked training runner is missing: {expected}")
    expected_sha = protocol.get("implementation", {}).get(relative)
    require(isinstance(expected_sha, str), f"protocol does not lock {relative}")
    actual_sha = sha256(expected)
    require(actual_sha == expected_sha, f"locked training runner hash drift: {expected}")
    if override is not None:
        require(override.resolve() == expected, f"--runner may only name the locked r28 runner: {expected}")
    return expected, actual_sha


def validate_runtime_repository(protocol: dict[str, Any], repo: Path) -> None:
    """Fail before stage creation if any locked implementation byte or git lineage drifted."""
    for relative, expected_sha in protocol.get("implementation", {}).items():
        path = repo / relative
        require(path.is_file() and sha256(path) == expected_sha, f"implementation hash drift: {relative}")
    implementation_head = protocol.get("implementation_head")
    lineage = subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(implementation_head), "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    require(lineage.returncode == 0, "current HEAD does not descend from the protocol implementation")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    require(not dirty, "r28 worktree must be clean before starting a stage")


def child_environment(repo: Path, physical_gpu: str | None = None) -> dict[str, str]:
    """Build the exact environment shared by runtime preflight and jobs."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    inherited_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo) + (os.pathsep + inherited_pythonpath if inherited_pythonpath else "")
    if physical_gpu is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = physical_gpu
    return env


def validate_child_interpreter(protocol: dict[str, Any], repo: Path, requests: list[dict]) -> dict:
    """Prove the exact registered child Python can import torch and r28."""
    expected_python = protocol.get("runtime_binding", {}).get("interpreter", {}).get("executable")
    python_paths = {request.get("runtime", {}).get("python") for request in requests}
    require(python_paths == {expected_python}, f"stage request interpreter drift: {sorted(map(str, python_paths))}")
    python_path = Path(str(expected_python))
    require(python_path.is_file(), f"registered child interpreter is missing: {python_path}")
    runtime_script = repo / "scripts/a1/p1_r28_runtime.py"
    result = subprocess.run(
        [str(python_path), str(runtime_script)],
        cwd=repo,
        env=child_environment(repo),
        check=False,
        capture_output=True,
        text=True,
    )
    require(result.returncode == 0, f"child runtime preflight failed: {result.stderr.strip()}")
    try:
        actual = json.loads(result.stdout.strip())
    except json.JSONDecodeError as error:
        raise ValueError(f"child runtime preflight did not emit JSON: {result.stdout!r}") from error
    require(actual == RUNTIME_ATTESTATION, "child interpreter runtime attestation drift")
    return {
        "passed": True,
        "python": str(python_path),
        "python_realpath": str(python_path.resolve()),
        "runtime_script": str(runtime_script),
        "runtime_script_sha256": sha256(runtime_script),
        "attestation": actual,
    }


def initializer_attestation(protocol: dict[str, Any], seed: int, cell: str) -> dict[str, Any]:
    run_root = Path(protocol["run_root"])
    path = run_root / "initializers" / f"seed{seed}" / "initialization_manifest.json"
    require(path.is_file(), f"initializer manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema_version") == 8, f"initializer manifest schema drift: {path}")
    require(payload.get("status") == "passed", f"initializer manifest failed: {path}")
    require(
        payload.get("runtime_attestation") == RUNTIME_ATTESTATION,
        f"initializer runtime provenance drift: {path}",
    )
    require(payload.get("protocol_sha256") is not None, f"initializer manifest has no protocol hash: {path}")
    cell_payload = payload.get("cells", {}).get(cell, {})
    require(cell_payload, f"initializer manifest lacks {seed}/{cell}")
    routers = cell_payload.get("r28_clean_aux_routers", [])
    require(len(routers) == (6 if cell in "cd" else 0), f"initializer router count drift: {seed}/{cell}")
    if cell in "cd":
        require(
            all(
                item.get("p1_balance_on_clean_routes") is True
                and item.get("routing_aux_semantics") == CLEAN_AUX_POLICY["runtime_semantics"]
                for item in routers
            ),
            f"initializer clean-aux policy drift: {seed}/{cell}",
        )
    return {"path": str(path), "sha256": sha256(path), "cell": cell_payload}


def request_entry(
    protocol: dict[str, Any], protocol_path: Path, stage: str, seed: int, cell: str
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    entry = protocol["requests"][stage][str(seed)][cell]
    path = Path(entry["path"])
    require(sha256(path) == entry["sha256"], f"request hash drift: {path}")
    request = json.loads(path.read_text(encoding="utf-8"))
    require(
        Path(request.get("protocol", {}).get("path", "")).resolve() == protocol_path,
        f"{stage}/{seed}/{cell}: protocol registration path drift",
    )
    params = request["params"]
    expected_epochs = 15 if stage == "formal" else 1
    require(params.get("epochs") == expected_epochs, f"{stage}/{seed}/{cell}: epoch drift")
    require(params.get("seed") == seed, f"{stage}/{seed}/{cell}: seed drift")
    require(params.get("device") == "0", f"{stage}/{seed}/{cell}: request device must be logical 0")
    require(params.get("exist_ok") is False, f"{stage}/{seed}/{cell}: exist_ok must be false")
    require(params.get("resume") is False, f"{stage}/{seed}/{cell}: resume must be false")
    policy = request.get("a1_policy", {})
    expected_aux = CLEAN_AUX_POLICY if cell in "cd" else {
        "enabled": False,
        "reason": "dense cell has no router",
        "adds_parameters": False,
        "changes_inference": False,
    }
    require(policy.get("routing_auxiliary_objective") == expected_aux, f"{stage}/{seed}/{cell}: aux policy drift")
    require(
        policy.get("formal_restart_from_initializer") is (stage == "formal"),
        f"{stage}/{seed}/{cell}: formal restart flag drift",
    )
    initializer = Path(request["inputs"]["model"]).resolve()
    expected_initializer = (
        Path(protocol["run_root"]) / "initializers" / f"seed{seed}" / f"{cell}_residual_factor_init.pt"
    ).resolve()
    require(initializer == expected_initializer, f"{stage}/{seed}/{cell}: initializer lineage drift")
    require("r19" not in normalized(initializer), f"{stage}/{seed}/{cell}: r19 weight reference")
    require(initializer.is_file(), f"initializer is missing: {initializer}")
    attestation = initializer_attestation(protocol, seed, cell)
    require(
        attestation["cell"].get("initializer_sha256") == sha256(initializer),
        f"{stage}/{seed}/{cell}: initializer hash differs from attestation",
    )
    require(
        bool(attestation["cell"].get("r28_clean_aux_routers")) is (cell in "cd"),
        f"{stage}/{seed}/{cell}: initializer clean-aux manifest drift",
    )
    require(
        attestation["path"] and json.loads(Path(attestation["path"]).read_text(encoding="utf-8"))["protocol_sha256"] == sha256(protocol_path),
        f"{stage}/{seed}/{cell}: initializer attestation protocol mismatch",
    )
    return path, request, attestation


def formal_admission(protocol: dict[str, Any], protocol_path: Path) -> dict[str, str]:
    path = Path(protocol["formal_admission"]["path"])
    require(path.is_file(), f"formal admission evidence is missing: {path}")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    require(evidence.get("schema_version") == 1, "formal admission schema drift")
    require(evidence.get("status") == "passed", f"formal admission did not pass: {path}")
    require(evidence.get("protocol_sha256") == sha256(protocol_path), "formal admission protocol hash mismatch")
    require(evidence.get("formal_request_lineage_verified") is True, "formal request lineage is not admitted")
    require(evidence.get("gpu_schedule_verified") is True, "GPU schedule is not admitted")
    require(evidence.get("formal_directory_absent_at_admission") is True, "formal directory absence not attested")
    require(evidence.get("all_required_gates_passed") is True, "combined formal gates did not all pass")
    require(evidence.get("formal_may_start") is True, "formal admission did not authorize start")
    require(
        evidence.get("counts")
        == {
            "initializer_manifests": 3,
            "preflight_cells": 12,
            "routing_probe_cells": 6,
            "routing_routers": 36,
            "residual_layers": 18,
            "formal_requests": 12,
        },
        "formal admission evidence counts drift",
    )
    require(evidence.get("dependency_hash_graph_verified") is True, "admission dependency graph is unverified")
    require(evidence.get("raw_gate_metrics_recomputed") is True, "admission did not recompute raw gates")
    require(
        evidence.get("implementation_and_git_lineage_verified") is True,
        "admission implementation/git lineage is unverified",
    )
    return {"path": str(path), "sha256": sha256(path)}


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert_protocol_runtime(protocol)
    validate_protocol(protocol, protocol_path)
    first = Path(protocol["requests"]["preflight"][str(SEEDS[0])]["a"]["path"])
    first_request = json.loads(first.read_text(encoding="utf-8"))
    repo = Path(first_request["runtime"]["cwd"]).resolve()
    validate_runtime_repository(protocol, repo)
    runner, runner_sha = locked_runner(protocol, repo, args.runner)
    admission = formal_admission(protocol, protocol_path) if args.stage == "formal" else None
    # Resolve and attest the complete stage before creating any output or
    # launching the first job.  This prevents a late seed/cell drift from
    # producing a partially executed formal matrix.
    stage_requests = []
    for seed, cell in stage_jobs(args.stage):
        _, request, _ = request_entry(protocol, protocol_path, args.stage, seed, cell)
        stage_requests.append(request)
    child_runtime_preflight = validate_child_interpreter(protocol, repo, stage_requests)
    stage_root = Path(protocol["run_root"]) / args.stage
    status_path = stage_root / f"{args.stage}_status.json"
    if stage_root.exists():
        raise FileExistsError(f"refusing to adopt or reuse r28 stage directory: {stage_root}")
    inventory = gpu_inventory()
    require(set(GPU_INDICES) <= set(inventory), "required physical GPU0/GPU1 are unavailable")
    capacities = {
        index: capacity_for_free_memory(inventory[index]["memory_free_mib"]) for index in GPU_INDICES
    }
    schedule = stage_schedule(args.stage, capacities)
    observed_compute_processes = gpu_compute_processes()
    target_uuids = {inventory[index]["uuid"] for index in GPU_INDICES}
    preexisting_compute_processes = [
        process for process in observed_compute_processes if process["gpu_uuid"] in target_uuids
    ]
    require(sum(capacities.values()) >= 1, "both GPUs are below the 12 GiB one-job threshold")
    status: dict[str, Any] = {
        "schema_version": 8,
        "stage": args.stage,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "formal_admission": admission,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "runner": {"path": str(runner), "sha256": runner_sha},
        "runtime_attestation": RUNTIME_ATTESTATION,
        "child_runtime_preflight": child_runtime_preflight,
        "gpu_inventory": inventory,
        "gpu_uuid_by_physical_index": {index: item["uuid"] for index, item in inventory.items()},
        "preexisting_compute_processes": preexisting_compute_processes,
        "shared_gpu_resource_policy": protocol["gpu_policy"]["co_location"],
        "observed_compute_processes_all_gpus": observed_compute_processes,
        "gpu_capacity_by_physical_index": capacities,
        "ordered_jobs": [{"seed": seed, "cell": cell} for seed, cell in stage_jobs(args.stage)],
        "waves": [],
    }
    write_json(status_path, status)
    for wave_index, assignments in enumerate(schedule, start=1):
        wave_status: dict[str, Any] = {"wave": wave_index, "status": "running", "jobs": {}}
        processes: dict[str, tuple[Any, ...]] = {}
        handles: dict[str, Any] = {}
        for seed, cell, physical_gpu in assignments:
            request_path, request, attestation = request_entry(
                protocol, protocol_path, args.stage, seed, cell
            )
            run_dir = Path(request["params"]["project"]) / request["params"]["name"]
            log_path = stage_root / f"seed{seed}" / f"{cell}_driver.log"
            if run_dir.exists() or log_path.exists():
                raise FileExistsError(f"refusing to adopt/reuse artifacts: {run_dir} / {log_path}")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("x", encoding="utf-8")
            env = child_environment(repo, physical_gpu)
            python = request["runtime"]["python"]
            process = subprocess.Popen(
                [python, str(runner), "--request", str(request_path)],
                cwd=repo,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            key = f"seed{seed}_{cell}"
            handles[key] = handle
            processes[key] = (
                process,
                seed,
                cell,
                request_path,
                request,
                attestation,
                run_dir,
                log_path,
                physical_gpu,
            )
            wave_status["jobs"][key] = {
                "status": "running",
                "pid": process.pid,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "seed": seed,
                "cell": cell,
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
        for key, values in processes.items():
            process, seed, cell, request_path, request, attestation, run_dir, log_path, physical_gpu = values
            returncode = process.wait()
            handles[key].close()
            failed |= returncode != 0
            initial_job = wave_status["jobs"][key]
            runtime_policy_path = run_dir / "p1_runtime_policy.json"
            last_path = run_dir / "weights/last.pt"
            wave_status["jobs"][key] = {
                "status": "completed" if returncode == 0 else "failed",
                "returncode": returncode,
                "pid": initial_job["pid"],
                "started_at": initial_job["started_at"],
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "seed": seed,
                "cell": cell,
                "physical_gpu": physical_gpu,
                "physical_gpu_uuid": status["gpu_uuid_by_physical_index"][physical_gpu],
                "logical_device": "0",
                "request": str(request_path),
                "request_sha256": sha256(request_path),
                "initializer": request["inputs"]["model"],
                "initializer_sha256": sha256(Path(request["inputs"]["model"])),
                "initializer_attestation": {"path": attestation["path"], "sha256": attestation["sha256"]},
                "run_dir": str(run_dir),
                "log": str(log_path),
                "log_sha256": sha256(log_path),
                "runtime_policy": str(runtime_policy_path) if runtime_policy_path.is_file() else None,
                "runtime_policy_sha256": sha256(runtime_policy_path) if runtime_policy_path.is_file() else None,
                "last": str(last_path) if last_path.is_file() else None,
                "last_sha256": sha256(last_path) if last_path.is_file() else None,
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

