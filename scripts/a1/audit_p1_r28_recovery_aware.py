#!/usr/bin/env python3
"""Additive, recovery-aware audit for completed r28 formal cells.

This tool never edits checkpoints, runtime policies, driver logs, requests, or
the locked repository. It consumes the immutable r28 checkpoint audit plus the
append-only recovery evidence and writes a separate supplemental verdict.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

FACTOR_LAYERS = (4, 6, 8)
CELLS = "abcd"
EXPECTED_BASE_PARAMETERS = 459232
EXPECTED_EPOCHS = 15
EXPECTED_BATCH = 4
EXPECTED_TRAIN_IMAGES = 20000
EXPECTED_MICROBATCHES = EXPECTED_EPOCHS * math.ceil(
    EXPECTED_TRAIN_IMAGES / EXPECTED_BATCH
)
ROUTING_SEMANTICS = "hard_top2_from_step_zero_private_exploration_clean_aux"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--formal-audit", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def valid_dense_policy(payload: dict, reference: dict) -> tuple[bool, list[str]]:
    reasons = []
    if payload.get("routing_semantics") != ROUTING_SEMANTICS:
        reasons.append("routing semantics mismatch")
    if payload.get("routing_auxiliary_objective") != reference.get(
        "routing_auxiliary_objective"
    ):
        reasons.append("dense routing auxiliary policy mismatch")
    if payload.get("requested") != reference.get("requested"):
        reasons.append("requested mixture policy mismatch")
    if payload.get("mixture_config", {}).get("values", {}).get("moe") != reference.get(
        "mixture_config", {}
    ).get("values", {}).get("moe"):
        reasons.append("MoE mixture policy mismatch")

    exploration = payload.get("router_exploration", {})
    if (
        exploration.get("enabled") is not False
        or payload.get("current_noise_scale") != 0.0
    ):
        reasons.append("dense router exploration unexpectedly active")

    adapters = payload.get("factor_adapters", [])
    if len(adapters) != 3:
        reasons.append(f"expected three factor adapters, got {len(adapters)}")
    for adapter in adapters:
        if adapter.get("trainable_base_parameters") != 0:
            reasons.append(f"trainable base in {adapter.get('name')}")
        if adapter.get("trainable_factor_batch_norm_parameters") != 0:
            reasons.append(f"trainable factor BN in {adapter.get('name')}")
        if adapter.get("factor_non_batch_norm_parameters") != adapter.get(
            "trainable_factor_non_batch_norm_parameters"
        ):
            reasons.append(f"frozen factor non-BN parameters in {adapter.get('name')}")
        if adapter.get("gain_parameters") != adapter.get("trainable_gain_parameters"):
            reasons.append(f"frozen gain in {adapter.get('name')}")
        if (
            adapter.get("gain_lr_scale") != 100.0
            or adapter.get("gain_no_warmup") is not True
        ):
            reasons.append(f"gain schedule mismatch in {adapter.get('name')}")
    if payload.get("routed_modules") != []:
        reasons.append("dense cell unexpectedly contains routed modules")

    groups = payload.get("optimizer_groups", [])
    gain = [group for group in groups if group.get("param_group") == "residual_gain"]
    router = [group for group in groups if group.get("param_group") == "router"]
    bn = [group for group in groups if group.get("param_group") == "bn"]
    if (
        len(gain) != 1
        or not math.isclose(float(gain[0].get("initial_lr", -1)), 0.01, abs_tol=1e-12)
        or gain[0].get("weight_decay") != 0.0
        or gain[0].get("p1_no_warmup") is not True
    ):
        reasons.append("residual-gain optimizer policy mismatch")
    if (
        len(router) != 1
        or not math.isclose(
            float(router[0].get("initial_lr", -1)), 0.0001, abs_tol=1e-12
        )
        or router[0].get("parameters") != 0
        or router[0].get("trainable_parameters") != 0
    ):
        reasons.append("dense router optimizer group mismatch")
    if len(bn) != 1 or bn[0].get("trainable_parameters") != 0:
        reasons.append("BatchNorm optimizer group is trainable")
    return not reasons, reasons


def current_identity(record: dict) -> tuple[bool, dict]:
    identities = {}
    for key in ("request", "initializer", "checkpoint"):
        path = Path(record[key])
        expected = record[f"{key}_sha256"]
        actual = sha256(path) if path.is_file() else None
        identities[key] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
        }
    return all(
        item["actual_sha256"] == item["expected_sha256"] for item in identities.values()
    ), identities


def recovery_evidence(run_root: Path, seed: str, cell: str) -> dict:
    recovery_root = run_root / "audits" / "recovery"
    pre_resume = []
    for path in sorted(recovery_root.glob(f"*_seed{seed}_{cell}_pre_resume.json")):
        payload = load_json(path)
        pre_resume.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "status": payload.get("status"),
                "completed_epochs": payload.get("checkpoint", {}).get(
                    "completed_epochs"
                )
                if isinstance(payload.get("checkpoint"), dict)
                else None,
            }
        )
    starts = []
    for pattern in (
        f"*_seed{seed}_{cell}_pre_start.json",
        f"*_seed{seed}_{cell}_pre_restart.json",
    ):
        for path in sorted(recovery_root.glob(pattern)):
            payload = load_json(path)
            registration = payload.get("registration", {})
            starts.append(
                {
                    "path": str(path),
                    "sha256": sha256(path),
                    "registered_seed": registration.get("seed"),
                    "registered_cell": registration.get("cell"),
                    "request_sha256": registration.get("request_sha256"),
                    "freeze": payload.get("freeze"),
                    "dry_run": payload.get("dry_run"),
                }
            )
    logs = sorted((run_root / "recovery").glob(f"*seed{seed}_{cell}*.log"))
    canonical = run_root / "formal" / f"seed{seed}" / f"{cell}_driver.log"
    if canonical.is_file():
        logs.insert(0, canonical)
    return {
        "pre_resume": pre_resume,
        "pre_start_or_restart": starts,
        "logs": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in logs
        ],
        "pre_resume_all_passed": all(item["status"] == "passed" for item in pre_resume),
        "has_log_evidence": bool(logs),
    }


def policy_snapshots(run_root: Path, seed: str, cell: str, run_dir: Path) -> list[dict]:
    snapshots = []
    final_path = run_dir / "p1_runtime_policy.json"
    if final_path.is_file():
        snapshots.append(
            {
                "source": str(final_path),
                "sha256": sha256(final_path),
                "payload": load_json(final_path),
            }
        )
    recovery_root = run_root / "audits" / "recovery"
    for path in sorted(recovery_root.glob(f"*_seed{seed}_{cell}_resume_started.json")):
        payload = load_json(path).get("runtime_policy")
        if isinstance(payload, dict):
            snapshots.append(
                {"source": str(path), "sha256": sha256(path), "payload": payload}
            )
    return snapshots


def main() -> None:
    args = parse_args()
    auditor_path = Path(__file__).resolve()
    run_root = args.run_root.resolve()
    formal_path = (
        args.formal_audit or run_root / "audits" / "formal_checkpoints.json"
    ).resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite supplemental audit: {output}")

    formal = load_json(formal_path)
    reference_dense = formal["seeds"]["260831"]["a"]["runtime_policy"]["payload"]
    report = {
        "schema": "a1-p1-r28-recovery-aware-formal-audit/v1",
        "kind": "additive_supplement",
        "mutates_experiment": False,
        "auditor": {"path": str(auditor_path), "sha256": sha256(auditor_path)},
        "run_root": str(run_root),
        "formal_audit": {
            "path": str(formal_path),
            "sha256": sha256(formal_path),
            "status": formal.get("status"),
        },
        "protocol": {"path": formal["protocol"], "sha256": formal["protocol_sha256"]},
        "microbatch_accounting": {
            "train_images": EXPECTED_TRAIN_IMAGES,
            "batch": EXPECTED_BATCH,
            "epochs": EXPECTED_EPOCHS,
            "expected_total": EXPECTED_MICROBATCHES,
            "dense_scientific_effect_of_local_counter": "none: dense cells contain no routed modules and exploration is disabled",
        },
        "cells": {},
    }

    for seed, cells in formal["seeds"].items():
        report["cells"][seed] = {}
        for cell in CELLS:
            original = cells[cell]
            identity_ok, identities = current_identity(original)
            layers = original["layers"]
            structural_checks = {
                "identity_hashes_unchanged": identity_ok,
                "end2end_path": original["end2end"]["expected"]
                == original["end2end"]["model"]
                == original["end2end"]["head"],
                "factor_base_parameter_count": original["frozen_factor_base_parameters"]
                == EXPECTED_BASE_PARAMETERS,
                "factor_bases_unchanged": all(
                    layers[str(index)]["base"]["passed"] for index in FACTOR_LAYERS
                ),
                "all_frozen_layers_unchanged": all(
                    item["passed"] for item in original["frozen_layers"].values()
                ),
                "batch_norm_unchanged": original["batch_norm"]["passed"],
                "gains_nonzero": all(
                    layers[str(index)]["gain_nonzero"] > 0 for index in FACTOR_LAYERS
                ),
                "metrics_complete_and_finite": original["metrics"]["passed"]
                and original["metrics"]["rows"] == 15,
                "factor_updated": all(
                    layers[str(index)]["factor_update"]["changed_tensors"] > 0
                    for index in FACTOR_LAYERS
                ),
            }
            if cell in "cd":
                structural_checks["routers_updated"] = all(
                    layers[str(index)]["routing_update"]["updated"]
                    for index in FACTOR_LAYERS
                )
                structural_checks["experts_updated"] = all(
                    layers[str(index)]["expert_update"]["updated"]
                    for index in FACTOR_LAYERS
                )
            else:
                structural_checks["dense_has_no_routers"] = all(
                    not layers[str(index)]["routers"] for index in FACTOR_LAYERS
                )

            request = load_json(Path(original["request"]))
            request_budget_ok = (
                request["params"].get("epochs") == EXPECTED_EPOCHS
                and request["params"].get("batch") == EXPECTED_BATCH
            )
            rows = read_rows(Path(original["metrics"]["path"]))
            epoch_sequence_ok = [int(float(row["epoch"])) for row in rows] == list(
                range(1, EXPECTED_EPOCHS + 1)
            )
            finite_metrics = all(
                math.isfinite(float(row["metrics/mAP50-95(B)"])) for row in rows
            )

            recovery = recovery_evidence(run_root, seed, cell)
            snapshots = policy_snapshots(
                run_root, seed, cell, Path(original["run_dir"])
            )
            if cell in "cd":
                runtime_ok = original["runtime_policy"]["passed"]
                runtime_reasons = original["runtime_policy"]["reasons"]
                valid_sources = (
                    [original["runtime_policy"]["path"]] if runtime_ok else []
                )
            else:
                validations = []
                for snapshot in snapshots:
                    passed, reasons = valid_dense_policy(
                        snapshot["payload"], reference_dense
                    )
                    validations.append(
                        {
                            "source": snapshot["source"],
                            "passed": passed,
                            "reasons": reasons,
                        }
                    )
                runtime_ok = any(item["passed"] for item in validations)
                runtime_reasons = (
                    [] if runtime_ok else ["no valid dense runtime-policy snapshot"]
                )
                valid_sources = [
                    item["source"] for item in validations if item["passed"]
                ]

            checks = {
                **structural_checks,
                "request_budget": request_budget_ok,
                "epoch_sequence_1_to_15": epoch_sequence_ok,
                "all_map_values_finite": finite_metrics,
                "runtime_policy_recovered": runtime_ok,
                "recovery_pre_resume_audits_passed": recovery["pre_resume_all_passed"],
                "log_evidence_present": recovery["has_log_evidence"],
            }
            passed = all(checks.values())
            report["cells"][seed][cell] = {
                "status": "passed" if passed else "failed",
                "checks": checks,
                "identities": identities,
                "original_audit": {
                    "passed": original["passed"],
                    "reasons": original["reasons"],
                },
                "runtime_policy": {
                    "valid_sources": valid_sources,
                    "reasons": runtime_reasons,
                    "snapshot_count": len(snapshots),
                    "final_recorded_microbatch_index": original["runtime_policy"][
                        "payload"
                    ].get("microbatch_index"),
                    "inferred_complete_microbatches": EXPECTED_MICROBATCHES
                    if epoch_sequence_ok
                    else None,
                },
                "recovery": recovery,
                "final_metrics": rows[-1],
            }

    failed = [
        f"{seed}/{cell}"
        for seed, cells in report["cells"].items()
        for cell, record in cells.items()
        if record["status"] != "passed"
    ]
    report["failed_cells"] = failed
    report["status"] = "passed" if not failed else "failed"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": report["status"], "failed_cells": failed, "output": str(output)},
            ensure_ascii=False,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
