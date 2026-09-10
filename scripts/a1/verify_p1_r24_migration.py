#!/usr/bin/env python3
"""Verify r24 implementation identity, A/B invariance, and C/D policy parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

# isort: split
from p1_r24_runtime import RUNTIME_ATTESTATION, assert_protocol_runtime

# isort: split

from p1_r24_integrity import verify_registered_data_content
from run_p1_bn_frozen_r24 import R24_CLEAN_AUX_POLICY

SEEDS = (260829, 260830, 260831)
LOCKED_SHA = "acce839c7e895d6b179de7f7093fa879e237cc7b"
OFFICIAL_CHECKPOINT_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--r23-protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def state_equal(left: torch.nn.Module, right: torch.nn.Module) -> tuple[bool, list[str]]:
    left_state, right_state = left.state_dict(), right.state_dict()
    differences = []
    if left_state.keys() != right_state.keys():
        return False, ["state_dict key sets differ"]
    for name in left_state:
        if not torch.equal(left_state[name].detach().cpu(), right_state[name].detach().cpu()):
            differences.append(name)
    return not differences, differences


def clean_aux_signature(model: torch.nn.Module) -> list[tuple[str, int, int, bool, str | None]]:
    signature = []
    for name, module in model.named_modules():
        routing = getattr(module, "routing", None)
        if routing is None or not hasattr(routing, "num_experts") or not hasattr(routing, "top_k"):
            continue
        signature.append(
            (
                f"{name}.routing",
                int(routing.num_experts),
                int(routing.top_k),
                bool(getattr(routing, "p1_balance_on_clean_routes", False)),
                getattr(module, "routing_aux_semantics", None),
            )
        )
    return signature


def verify_git_lineage(repo: Path, protocol: dict[str, Any]) -> dict[str, str]:
    """Verify the registered implementation commit and official ancestor relation."""
    require(protocol.get("locked_official_baseline_sha") == LOCKED_SHA, "locked baseline SHA drift")
    implementation_head = protocol.get("implementation_head")
    require(isinstance(implementation_head, str), "protocol has no implementation head")
    for revision in (LOCKED_SHA, implementation_head):
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        require(result.returncode == 0, f"registered git commit is unavailable: {revision}")
    for ancestor, descendant, label in (
        (LOCKED_SHA, implementation_head, "implementation does not descend from locked baseline"),
        (implementation_head, "HEAD", "current HEAD does not descend from registered implementation"),
    ):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        require(result.returncode == 0, label)
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    require(not dirty, "r24 worktree is dirty during migration audit")
    return {"locked_baseline": LOCKED_SHA, "implementation_head": implementation_head}


def verify_data_registry(protocol: dict[str, Any]) -> dict[str, Any]:
    """Recheck the immutable YAML and ordered image-list files."""
    expected = {
        "pilot": {"train": 5000, "val": 512},
        "preflight": {"train": 256, "val": 128},
        "formal": {"train": 118287, "val": 5000},
    }
    result: dict[str, Any] = {}
    for label, splits in expected.items():
        item = protocol.get("data", {}).get(label, {})
        yaml_path = Path(item.get("path", ""))
        require(yaml_path.is_file() and sha256(yaml_path) == item.get("sha256"), f"{label} YAML drift")
        require(set(item.get("lists", {})) == set(splits), f"{label} list registry drift")
        result[label] = {"path": str(yaml_path), "sha256": sha256(yaml_path), "lists": {}}
        for split, expected_count in splits.items():
            registered = item["lists"][split]
            path = Path(registered.get("path", ""))
            require(path.is_file() and sha256(path) == registered.get("sha256"), f"{label}/{split} drift")
            count = sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
            require(count == expected_count == registered.get("images"), f"{label}/{split} count drift")
            result[label]["lists"][split] = {"path": str(path), "sha256": sha256(path), "images": count}
    return result


def main() -> None:
    from ultralytics.nn.tasks import load_checkpoint

    args = parse_args()
    protocol_path = args.protocol.resolve()
    r23_protocol_path = args.r23_protocol.resolve()
    output = args.output.resolve()
    require(not output.exists(), f"refusing to overwrite migration audit: {output}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert_protocol_runtime(protocol)
    r23 = json.loads(r23_protocol_path.read_text(encoding="utf-8"))
    require(protocol.get("schema_version") == 8 and protocol.get("experiment_tag") == "r24", "not r24 schema8")
    require(r23.get("experiment_tag") == "r23", "comparison protocol is not r23")
    require(tuple(protocol.get("seeds", ())) == SEEDS == tuple(r23.get("seeds", ())), "seed set drift")
    first_request = Path(protocol["requests"]["preflight"][str(SEEDS[0])]["a"]["path"])
    repo = Path(json.loads(first_request.read_text(encoding="utf-8"))["runtime"]["cwd"])
    git_lineage = verify_git_lineage(repo, protocol)
    require(
        protocol.get("source_checkpoint", {}).get("sha256") == OFFICIAL_CHECKPOINT_SHA256,
        "official source checkpoint SHA drift",
    )
    data_registry = verify_data_registry(protocol)
    verify_registered_data_content(protocol)
    implementation = {}
    for relative, expected in protocol.get("implementation", {}).items():
        path = repo / relative
        require(path.is_file(), f"missing locked implementation file: {path}")
        actual = sha256(path)
        require(actual == expected, f"implementation hash drift: {relative}")
        implementation[relative] = actual

    # r24 deliberately recalibrates factor BatchNorm on the first 512 full-COCO
    # train paths, so its factor buffers need not be tensor-identical to r23.
    # The model, freeze, optimizer, and routing semantics must remain identical.
    semantic_keys = (
        "train_layers",
        "freeze",
        "factor_base",
        "factor_base_expected_parameters",
        "factor_gain_optimizer",
        "router_initialization",
        "routing",
        "gpu_policy",
        "primary_checkpoint",
        "secondary_checkpoint",
        "comparisons",
    )
    semantic_comparison = {}
    for key in semantic_keys:
        equal = protocol.get(key) == r23.get(key)
        require(equal, f"r24 semantic drift from r23: {key}")
        semantic_comparison[key] = True
    for cell in "abcd":
        current = protocol["configs"][cell]
        previous = r23["configs"][cell]
        for key in ("moe", "end2end", "sha256"):
            require(current.get(key) == previous.get(key), f"r24 config semantic drift: {cell}/{key}")
    for key in ("images", "batch", "imgsz", "augment", "shuffle", "grad", "post_calibration_training_policy"):
        require(protocol["batch_norm"].get(key) == r23["batch_norm"].get(key), f"r24 BN policy drift: {key}")

    comparisons: dict[str, Any] = {}
    c_d_parity: dict[str, Any] = {}
    for seed in SEEDS:
        comparisons[str(seed)] = {}
        manifest_path = Path(protocol["run_root"]) / "initializers" / f"seed{seed}" / "initialization_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("status") == "passed", f"initializer manifest failed: seed{seed}")
        for cell in "ab":
            r24_path = Path(protocol["run_root"]) / "initializers" / f"seed{seed}" / f"{cell}_residual_factor_init.pt"
            require(r24_path.is_file(), f"missing A/B initializer: {seed}/{cell}")
            item = manifest["cells"][cell]
            before = item.get("equivalence_before_save", {}).get("max_abs_error")
            after = item.get("equivalence_after_reload", {}).get("max_abs_error")
            passed = before == 0.0 and after == 0.0
            require(passed, f"r24 A/B invariance failed: {seed}/{cell}")
            comparisons[str(seed)][cell] = {
                "r24": str(r24_path),
                "r24_sha256": sha256(r24_path),
                "equivalence_before_save_max_abs_error": before,
                "equivalence_after_reload_max_abs_error": after,
                "cross_version_tensor_identity_required": False,
                "reason": "factor BatchNorm is independently calibrated on the registered full-COCO train512 sample",
                "passed": passed,
            }
        signatures = {}
        for cell in "cd":
            path = Path(protocol["run_root"]) / "initializers" / f"seed{seed}" / f"{cell}_residual_factor_init.pt"
            model, _ = load_checkpoint(path, device="cpu")
            signatures[cell] = clean_aux_signature(model.float().eval())
            require(len(signatures[cell]) == 6, f"{seed}/{cell}: expected six routers")
            require(
                all(item[3] and item[4] == R24_CLEAN_AUX_POLICY["runtime_semantics"] for item in signatures[cell]),
                f"{seed}/{cell}: clean-aux policy mismatch",
            )
        require(signatures["c"] == signatures["d"], f"{seed}: C/D clean-aux signature mismatch")
        c_d_parity[str(seed)] = {"c": signatures["c"], "d": signatures["d"], "passed": True}

    report = {
        "schema_version": 1,
        "status": "passed",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "runtime_attestation": RUNTIME_ATTESTATION,
        "r23_read_only_reference": {"path": str(r23_protocol_path), "sha256": sha256(r23_protocol_path)},
        "implementation_hashes_verified": True,
        "git_lineage_verified": True,
        "git_lineage": git_lineage,
        "official_checkpoint_verified": True,
        "data_hashes_verified": True,
        "data": data_registry,
        "implementation": implementation,
        "r23_semantics_verified": True,
        "r23_semantic_comparison": semantic_comparison,
        "a_b_invariance_verified": True,
        "a_b": comparisons,
        "c_d_clean_aux_parity_verified": True,
        "c_d": c_d_parity,
        "r23_initializers_or_trained_weights_used_for_training": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": "passed", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

