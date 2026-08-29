"""Offline consistency tests for the versioned r21 protocol toolchain."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

STAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = STAGE_ROOT / "scripts/a1"
PREPARE = SCRIPT_ROOT / "prepare_p1_residual_factor_r21.py"
BUILD = SCRIPT_ROOT / "build_p1_residual_factor_initializers_r21.py"
RUN_REQUEST = SCRIPT_ROOT / "run_p1_bn_frozen_r21.py"
RUN_MATRIX = SCRIPT_ROOT / "run_p1_factorial_multiseed_2gpu_r21.py"
FORMAL_ADMISSION = SCRIPT_ROOT / "audit_p1_formal_admission_r21.py"
ROUTING_AUDIT = SCRIPT_ROOT / "audit_p1_routing_r21.py"
INTEGRITY = SCRIPT_ROOT / "p1_r21_integrity.py"
RUNTIME = SCRIPT_ROOT / "p1_r21_runtime.py"
SCRIPTS = (
    SCRIPT_ROOT / "audit_p1_checkpoints_r21.py",
    FORMAL_ADMISSION,
    SCRIPT_ROOT / "audit_p1_residual_activity_r21.py",
    ROUTING_AUDIT,
    BUILD,
    INTEGRITY,
    RUNTIME,
    PREPARE,
    RUN_REQUEST,
    RUN_MATRIX,
    SCRIPT_ROOT / "verify_p1_r21_migration.py",
)
ENTRYPOINTS = tuple(path for path in SCRIPTS if path not in {INTEGRITY, RUNTIME})

EXPECTED_CLEAN_AUX = {
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


def assignment(path: Path, name: str):
    """Read one literal top-level assignment without importing project code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment {name} in {path}")


def load_stdlib_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def test_clean_aux_policy_is_byte_for_byte_consistent() -> None:
    assert assignment(PREPARE, "CLEAN_AUX_POLICY") == EXPECTED_CLEAN_AUX
    assert assignment(BUILD, "EXPECTED_CLEAN_AUX_POLICY") == EXPECTED_CLEAN_AUX
    assert assignment(RUN_REQUEST, "R21_CLEAN_AUX_POLICY") == EXPECTED_CLEAN_AUX
    assert assignment(RUN_MATRIX, "CLEAN_AUX_POLICY") == EXPECTED_CLEAN_AUX


def test_seed_block_gpu_schedule_and_order() -> None:
    runner = load_stdlib_script(RUN_MATRIX, "r21_matrix_runner_for_test")
    assert runner.SEEDS == (260829, 260830, 260831)
    assert runner.SEED_GPU == {260829: "0", 260830: "1", 260831: "0"}
    for stage, cells in (("preflight", "abcd"), ("formal", "abcd"), ("routing_probe", "cd")):
        schedule = runner.stage_schedule(stage)
        flattened = [assignment for wave in schedule for assignment in wave]
        for seed in runner.SEEDS:
            seed_jobs = [(cell, gpu) for job_seed, cell, gpu in flattened if job_seed == seed]
            assert seed_jobs == [(cell, runner.SEED_GPU[seed]) for cell in cells]
        assert len(flattened) == len(runner.SEEDS) * len(cells)


def test_prepare_registers_exactly_30_independent_requests() -> None:
    seeds = assignment(PREPARE, "SEEDS")
    cells = assignment(PREPARE, "CELLS")
    assert seeds == (260829, 260830, 260831)
    assert set(cells) == set("abcd")
    assert len(seeds) * (len(cells) + 2 + len(cells)) == 30
    text = PREPARE.read_text(encoding="utf-8")
    assert 'if request_count(requests) != 30:' in text
    assert '"request_count": 30' in text
    assert '"dead_experts": 0' in text
    assert '"maximum_image_selection_fraction": 0.8' in text
    assert '"minimum_normalized_entropy": 0.5' in text
    assert "require_locked_baseline_ancestor(repo)" in text
    assert "OFFICIAL_CHECKPOINT_SHA256" in text
    assert "expected_train=5000, expected_val=512" in text
    assert "expected_train=256, expected_val=128" in text


def test_r21_has_no_unlocked_r19_builder_or_runner_dependency() -> None:
    build_text = BUILD.read_text(encoding="utf-8")
    prepare_text = PREPARE.read_text(encoding="utf-8")
    matrix_text = RUN_MATRIX.read_text(encoding="utf-8")
    assert "from build_p1_residual_factor_initializers_r19" not in build_text
    assert "run_p1_bn_frozen.py\"" not in matrix_text
    assert 'repo / "scripts/a1/run_p1_bn_frozen_r21.py"' in prepare_text
    assert 'repo / "scripts/a1/build_p1_residual_factor_initializers.py"' in prepare_text
    assert '"helper_dependency"' in build_text


def test_runtime_fail_closed_checks_are_present() -> None:
    request_text = RUN_REQUEST.read_text(encoding="utf-8")
    matrix_text = RUN_MATRIX.read_text(encoding="utf-8")
    for required in (
        'expected_router_count = 6 if clean_aux_enabled else 0',
        'module["p1_balance_on_clean_routes"] is not True',
        'module["routing_aux_semantics"] != R21_CLEAN_AUX_POLICY["runtime_semantics"]',
        'policy.get("routing_auxiliary_objective") != expected_aux',
        "validate_request_registration(request_path, request)",
        'if stage == "formal":',
        'admission.get("dependency_hash_graph_verified") is not True',
    ):
        assert required in request_text
    for required in (
        'evidence.get("all_required_gates_passed") is True',
        'evidence.get("formal_request_lineage_verified") is True',
        'evidence.get("formal_directory_absent_at_admission") is True',
        'relative = "scripts/a1/run_p1_bn_frozen_r21.py"',
        "validate_runtime_repository(protocol, repo)",
        "request_entry(protocol, protocol_path, args.stage, seed, cell)",
    ):
        assert required in matrix_text


def test_absolute_entrypoints_pin_the_r21_worktree(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    runtime = subprocess.run(
        [sys.executable, str(RUNTIME)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    attestation = json.loads(runtime.stdout)
    assert Path(attestation["repo_root"]).resolve() == STAGE_ROOT.resolve()
    for item in attestation["modules"].values():
        assert Path(item["path"]).resolve().is_relative_to(STAGE_ROOT.resolve())
        assert len(item["sha256"]) == 64

    for path in ENTRYPOINTS:
        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=tmp_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{path.name}: {result.stderr}"
        assert "from p1_r21_runtime import" in path.read_text(encoding="utf-8")


def test_runtime_rejects_preloaded_foreign_ultralytics(tmp_path: Path) -> None:
    fake_root = tmp_path / "foreign"
    package = fake_root / "ultralytics"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("FOREIGN = True\n", encoding="utf-8")
    code = (
        "import runpy,sys; "
        f"sys.path.insert(0, {str(fake_root)!r}); "
        "import ultralytics; "
        f"runpy.run_path({str(RUNTIME)!r}, run_name='r21_runtime_negative')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_build_manifest_is_schema7_and_attests_clean_aux() -> None:
    text = BUILD.read_text(encoding="utf-8")
    assert '"schema_version": 7' in text
    assert '"routing_auxiliary_objective": EXPECTED_CLEAN_AUX_POLICY' in text
    assert '"c_d_clean_aux_policy_parity"' in text
    assert '"a_b_dense_clean_aux_not_applicable"' in text
    assert "checkpoint.resolve() != OFFICIAL_CHECKPOINT.resolve()" in text
    assert "OFFICIAL_CHECKPOINT_SHA256" in text
    assert "sha256(config_path) != config.get(\"sha256\")" in text
    assert '"trainable_factor_base_parameters": trainable_base' in text


def test_formal_admission_reads_the_routing_reports_dead_expert_field(tmp_path: Path) -> None:
    admission = load_stdlib_script(FORMAL_ADMISSION, "r21_formal_admission_for_routing_test")
    image_paths = []
    for index in range(512):
        path = tmp_path / f"image_{index:03d}.jpg"
        path.write_bytes(b"")
        image_paths.append(path)
    val_list = tmp_path / "val.txt"
    val_list.write_text("\n".join(str(path) for path in image_paths) + "\n", encoding="utf-8")
    data_yaml = tmp_path / "coco.yaml"
    data_yaml.write_text("val: val.txt\n", encoding="utf-8")
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    images = [{"path": str(path), "sha256": admission.sha256(path)} for path in image_paths]
    image_digest = admission.hashlib.sha256(
        "".join(f"{item['path']}\0{item['sha256']}\n" for item in images).encode()
    ).hexdigest()

    def module(num_experts: int) -> dict:
        counts = [1024 // num_experts] * num_experts
        return {
            "num_experts": num_experts,
            "top_k": 2,
            "images": 512,
            "selections": 1024,
            "expected_selections": 1024,
            "counts": counts,
            "gate": {"passed": True, "reasons": []},
            "dead_experts_on_sample": [],
            "max_image_selection_fraction": max(value / 512 for value in counts),
            "normalized_entropy": 1.0,
        }

    modules = {
        **{f"router.4.{index}": module(4) for index in range(2)},
        **{f"router.8.{index}": module(8) for index in range(2)},
        **{f"router.16.{index}": module(16) for index in range(2)},
    }
    payload = {
        "schema_version": 7,
        "data": str(data_yaml),
        "images": images,
        "image_set_sha256": image_digest,
        "gate_thresholds": {
            "dead_experts": 0,
            "max_image_selection_fraction": 0.8,
            "min_normalized_entropy": 0.5,
        },
        "seeds": {
            str(seed): {
                cell: {
                    "passed": True,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": admission.sha256(checkpoint),
                    "modules": copy.deepcopy(modules),
                }
                for cell in "cd"
            }
            for seed in admission.SEEDS
        },
    }
    protocol = {
        "data": {"pilot": {"path": str(data_yaml), "lists": {"val": {"path": str(val_list)}}}}
    }
    probe = {
        "seeds": {
            str(seed): {
                cell: {"checkpoint": str(checkpoint), "checkpoint_sha256": admission.sha256(checkpoint)}
                for cell in "cd"
            }
            for seed in admission.SEEDS
        }
    }
    assert admission.validate_routing(payload, protocol, probe) == 36
    payload["seeds"]["260829"]["c"]["modules"]["router.4.0"]["dead_experts_on_sample"] = [3]
    with pytest.raises(ValueError, match="dead expert"):
        admission.validate_routing(payload, protocol, probe)


def test_routing_audit_uses_current_summary_api() -> None:
    text = ROUTING_AUDIT.read_text(encoding="utf-8")
    assert "summarize_counts(counts, num_experts, top_k)" in text
    assert '"images": num_images' in text
    audit = load_stdlib_script(ROUTING_AUDIT, "r21_routing_summary_for_test")
    summary = audit.summarize_image_counts(Counter({0: 256, 1: 256, 2: 256, 3: 256}), 4, 2, 512)
    assert summary["images"] == 512
    assert summary["selections"] == summary["expected_selections"] == 1024
    assert summary["dead_experts_on_sample"] == []
    assert summary["max_image_selection_fraction"] == 0.5


def test_formal_admission_attests_every_clean_aux_router(tmp_path: Path) -> None:
    admission = load_stdlib_script(FORMAL_ADMISSION, "r21_formal_admission_for_initializer_test")
    protocol_path = tmp_path / "protocol.json"
    protocol = {
        "run_root": str(tmp_path / "run"),
        "source_checkpoint": {"sha256": admission.OFFICIAL_CHECKPOINT_SHA256},
    }
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    protocol_sha = admission.sha256(protocol_path)
    clean_router = {
        "name": "routing",
        "top_k": 2,
        "p1_balance_on_clean_routes": True,
        "routing_aux_semantics": "clean_hard_top2_balance_with_noisy_dispatch",
    }
    for seed in admission.SEEDS:
        cells = {}
        for cell in admission.CELLS:
            cells[cell] = {
                "frozen_factor_base_parameters": 459232,
                "trainable_factor_base_parameters": 0,
                "reloaded_trainable_factor_base_parameters": 0,
                "equivalence_before_save": {"max_abs_error": 0.0},
                "equivalence_after_reload": {"max_abs_error": 0.0},
                "r21_clean_aux_routers": (
                    [] if cell in "ab" else [copy.deepcopy(clean_router) for _ in range(6)]
                ),
            }
        manifest = {
            "schema_version": 7,
            "status": "passed",
            "protocol_sha256": protocol_sha,
            "source_checkpoint_sha256": admission.OFFICIAL_CHECKPOINT_SHA256,
            "runtime_attestation": admission.RUNTIME_ATTESTATION,
            "cells": cells,
            "a_b_dense_clean_aux_not_applicable": True,
            "c_d_clean_aux_policy_parity": True,
            "paired_factor_tensor_parity_after_reload": {"a_b": True, "c_d": True},
        }
        path = tmp_path / "run" / "initializers" / f"seed{seed}" / "initialization_manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(manifest), encoding="utf-8")
    records, parity = admission.validate_initializers(protocol, protocol_path)
    assert len(records) == 3 and parity is True

    broken_path = tmp_path / "run" / "initializers" / "seed260829" / "initialization_manifest.json"
    broken = json.loads(broken_path.read_text(encoding="utf-8"))
    broken["cells"]["c"]["r21_clean_aux_routers"][0]["p1_balance_on_clean_routes"] = False
    broken_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="clean-aux flag false"):
        admission.validate_initializers(protocol, protocol_path)


def test_all_scripts_compile() -> None:
    for path in SCRIPTS:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
