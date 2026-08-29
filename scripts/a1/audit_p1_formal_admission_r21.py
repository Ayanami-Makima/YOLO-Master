#!/usr/bin/env python3
"""Compose the fail-closed r21 formal-admission evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from p1_r21_runtime import RUNTIME_ATTESTATION, assert_protocol_runtime

# isort: split

from p1_r21_integrity import verify_registered_data_content

SEEDS = (260829, 260830, 260831)
CELLS = "abcd"
OFFICIAL_CHECKPOINT_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
LOCKED_SHA = "acce839c7e895d6b179de7f7093fa879e237cc7b"
CLEAN_AUX_SEMANTICS = "clean_hard_top2_balance_with_noisy_dispatch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--preflight-audit", required=True, type=Path)
    parser.add_argument("--probe-audit", required=True, type=Path)
    parser.add_argument("--routing-audit", required=True, type=Path)
    parser.add_argument("--residual-audit", required=True, type=Path)
    parser.add_argument("--migration-audit", required=True, type=Path)
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


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{label} is not a JSON object")
    return payload


def evidence_protocol_sha(payload: dict[str, Any]) -> str | None:
    direct = payload.get("protocol_sha256")
    if isinstance(direct, str):
        return direct
    protocol = payload.get("protocol")
    if isinstance(protocol, dict):
        value = protocol.get("sha256")
        return value if isinstance(value, str) else None
    return None


def passed_evidence(path: Path, label: str, protocol_sha: str) -> tuple[dict[str, Any], dict[str, str]]:
    payload = read_json(path, label)
    require(payload.get("status") == "passed", f"{label} did not pass: {path}")
    require(evidence_protocol_sha(payload) == protocol_sha, f"{label} protocol hash mismatch")
    return payload, {"path": str(path), "sha256": sha256(path)}


def validate_data_registry(protocol: dict[str, Any]) -> None:
    expected = {"pilot": {"train": 5000, "val": 512}, "preflight": {"train": 256, "val": 128}}
    for label, split_counts in expected.items():
        registry = protocol.get("data", {}).get(label, {})
        yaml_path = Path(registry.get("path", ""))
        require(yaml_path.is_file(), f"missing {label} data YAML")
        require(sha256(yaml_path) == registry.get("sha256"), f"{label} data YAML hash drift")
        require(set(registry.get("lists", {})) == set(split_counts), f"{label} list registry drift")
        for split, expected_count in split_counts.items():
            item = registry["lists"][split]
            path = Path(item.get("path", ""))
            require(path.is_file(), f"missing {label}/{split} image list")
            require(sha256(path) == item.get("sha256"), f"{label}/{split} image-list hash drift")
            count = sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
            require(item.get("images") == expected_count == count, f"{label}/{split} image count drift")


def validate_model_configs(protocol: dict[str, Any]) -> None:
    expected = {
        "a": {"moe": False, "end2end": False},
        "b": {"moe": False, "end2end": True},
        "c": {"moe": True, "end2end": False},
        "d": {"moe": True, "end2end": True},
    }
    require(set(protocol.get("configs", {})) == set(expected), "model config cell registry drift")
    for cell, factors in expected.items():
        item = protocol["configs"][cell]
        path = Path(item.get("path", ""))
        require(path.is_file() and sha256(path) == item.get("sha256"), f"model config hash drift: {cell}")
        require(item.get("moe") is factors["moe"], f"model config MoE factor drift: {cell}")
        require(item.get("end2end") is factors["end2end"], f"model config end2end factor drift: {cell}")


def validate_implementation(protocol: dict[str, Any]) -> None:
    first = Path(protocol["requests"]["preflight"][str(SEEDS[0])]["a"]["path"])
    request = read_json(first, "first preflight request")
    repo = Path(request["runtime"]["cwd"]).resolve()
    for relative, expected_sha in protocol.get("implementation", {}).items():
        path = repo / relative
        require(path.is_file() and sha256(path) == expected_sha, f"implementation hash drift: {relative}")
    locked = protocol.get("locked_official_baseline_sha")
    implementation_head = protocol.get("implementation_head")
    for ancestor, descendant, label in (
        (locked, implementation_head, "implementation does not descend from locked baseline"),
        (implementation_head, "HEAD", "current HEAD does not descend from implementation commit"),
    ):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(ancestor), str(descendant)],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        require(result.returncode == 0, label)
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    require(not dirty, "r21 worktree is dirty at formal admission")


def require_identity(actual: Any, expected: dict[str, str], label: str) -> None:
    require(isinstance(actual, dict), f"{label} dependency is absent")
    require(Path(actual.get("path", "")).resolve() == Path(expected["path"]).resolve(), f"{label} path mismatch")
    require(actual.get("sha256") == expected["sha256"], f"{label} SHA mismatch")


def require_runtime_attestation(payload: dict[str, Any], label: str) -> None:
    require(payload.get("runtime_attestation") == RUNTIME_ATTESTATION, f"{label} runtime provenance drift")


def validate_stage_execution(protocol: dict[str, Any], protocol_path: Path, stage: str, cells: str) -> dict[str, str]:
    run_root = Path(protocol["run_root"])
    path = run_root / stage / f"{stage}_status.json"
    status = read_json(path, f"{stage} execution status")
    require_runtime_attestation(status, f"{stage} execution status")
    require(status.get("schema_version") == 7 and status.get("stage") == stage, f"{stage} status schema drift")
    require(status.get("status") == "completed", f"{stage} matrix did not complete")
    require(status.get("protocol_sha256") == sha256(protocol_path), f"{stage} status protocol drift")
    expected_mapping = {"260829": "0", "260830": "1", "260831": "0"}
    require(status.get("seed_to_physical_gpu") == expected_mapping, f"{stage} GPU mapping drift")
    require(status.get("preexisting_compute_processes") == [], f"{stage} started on occupied GPUs")
    runner = status.get("runner", {})
    relative_runner = "scripts/a1/run_p1_bn_frozen_r21.py"
    require(runner.get("sha256") == protocol["implementation"][relative_runner], f"{stage} runner SHA drift")
    inventory = status.get("gpu_inventory", {})
    require({"0", "1"} <= set(inventory), f"{stage} GPU inventory incomplete")
    require(
        status.get("gpu_uuid_by_physical_index")
        == {index: item.get("uuid") for index, item in inventory.items()},
        f"{stage} GPU UUID registry drift",
    )
    expected_waves = []
    for cell in cells:
        expected_waves.append(((260829, cell, "0"), (260830, cell, "1")))
    for cell in cells:
        expected_waves.append(((260831, cell, "0"),))
    waves = status.get("waves", [])
    require(len(waves) == len(expected_waves), f"{stage} wave count drift")
    for wave, expected_assignments in zip(waves, expected_waves):
        require(wave.get("status") == "completed", f"{stage} wave incomplete")
        jobs = wave.get("jobs", {})
        expected_keys = {f"seed{seed}_{cell}" for seed, cell, _ in expected_assignments}
        require(set(jobs) == expected_keys, f"{stage} wave assignment drift")
        require(len({gpu for _, _, gpu in expected_assignments}) == len(expected_assignments), f"{stage} GPU collision")
        for seed, cell, gpu in expected_assignments:
            job = jobs[f"seed{seed}_{cell}"]
            require(job.get("status") == "completed" and job.get("returncode") == 0, f"{stage}/{seed}/{cell} failed")
            require(job.get("seed") == seed and job.get("cell") == cell, f"{stage} job identity drift")
            require(job.get("physical_gpu") == gpu and job.get("logical_device") == "0", f"{stage} job GPU drift")
            require(job.get("physical_gpu_uuid") == inventory[gpu]["uuid"], f"{stage} job GPU UUID drift")
            entry = protocol["requests"][stage][str(seed)][cell]
            require(Path(job.get("request", "")).resolve() == Path(entry["path"]).resolve(), "stage request path drift")
            require(job.get("request_sha256") == entry["sha256"] == sha256(Path(entry["path"])), "stage request SHA drift")
            for path_key, sha_key in (
                ("initializer", "initializer_sha256"),
                ("last", "last_sha256"),
                ("log", "log_sha256"),
                ("runtime_policy", "runtime_policy_sha256"),
            ):
                artifact = Path(job.get(path_key, ""))
                require(artifact.is_file() and job.get(sha_key) == sha256(artifact), f"{stage} {path_key} SHA drift")
                if path_key == "runtime_policy":
                    require_runtime_attestation(read_json(artifact, f"{stage} runtime policy"), f"{stage} runtime policy")
    return {"path": str(path), "sha256": sha256(path)}


def validate_checkpoint_audit(
    payload: dict[str, Any], stage: str, expected_cells: str, protocol: dict[str, Any]
) -> int:
    require(payload.get("schema_version") == 7, f"{stage} checkpoint audit schema drift")
    require(payload.get("stage") == stage, f"checkpoint audit stage is not {stage}")
    seeds = payload.get("seeds", {})
    require(set(seeds) == {str(seed) for seed in SEEDS}, f"{stage} seed set drift")
    count = 0
    for seed in SEEDS:
        cells = seeds[str(seed)]
        require(set(cells) == set(expected_cells), f"{stage}/{seed} cell set drift")
        for cell in expected_cells:
            item = cells[cell]
            require(item.get("passed") is True, f"{stage}/{seed}/{cell} failed checkpoint gate")
            registered = protocol["requests"][stage][str(seed)][cell]
            request_path = Path(item.get("request", "")).resolve()
            require(request_path == Path(registered["path"]).resolve(), f"{stage}/{seed}/{cell} request path drift")
            require(
                item.get("request_sha256") == registered["sha256"] == sha256(request_path),
                f"{stage}/{seed}/{cell} request SHA drift",
            )
            request = read_json(request_path, f"{stage} request {seed}/{cell}")
            initializer = Path(item.get("initializer", "")).resolve()
            require(initializer == Path(request["inputs"]["model"]).resolve(), "checkpoint initializer path drift")
            require(
                item.get("initializer_sha256") == sha256(initializer),
                f"{stage}/{seed}/{cell} initializer SHA drift",
            )
            checkpoint = Path(item.get("checkpoint", "")).resolve()
            expected_checkpoint = (
                Path(request["params"]["project"]) / request["params"]["name"] / "weights/last.pt"
            ).resolve()
            require(checkpoint == expected_checkpoint, f"{stage}/{seed}/{cell} checkpoint path drift")
            require(
                item.get("checkpoint_sha256") == sha256(checkpoint),
                f"{stage}/{seed}/{cell} checkpoint SHA drift",
            )
            count += 1
    return count


def validate_routing(payload: dict[str, Any], protocol: dict[str, Any], probe: dict[str, Any]) -> int:
    require(payload.get("schema_version") == 7, "routing audit schema drift")
    require(
        payload.get("gate_thresholds")
        == {"dead_experts": 0, "max_image_selection_fraction": 0.8, "min_normalized_entropy": 0.5},
        "routing gate thresholds drift",
    )
    require(Path(payload.get("data", "")).resolve() == Path(protocol["data"]["pilot"]["path"]).resolve(), "routing data drift")
    images = payload.get("images", [])
    require(isinstance(images, list) and len(images) == 512, "routing audit must bind exactly 512 images")
    val_list = Path(protocol["data"]["pilot"]["lists"]["val"]["path"])
    expected_paths = []
    for line in (line.strip() for line in val_list.read_text(encoding="utf-8").splitlines() if line.strip()):
        path = Path(line)
        expected_paths.append((path if path.is_absolute() else val_list.parent / path).resolve())
    require(len(expected_paths) == 512, "pilot validation list is not exactly 512 images")
    for expected_path, record in zip(expected_paths, images):
        path = Path(record.get("path", "")).resolve()
        require(path == expected_path, f"routing image order/path drift: {path}")
        require(path.is_file() and sha256(path) == record.get("sha256"), f"routing image SHA drift: {path}")
    image_digest = hashlib.sha256(
        "".join(f"{item['path']}\0{item['sha256']}\n" for item in images).encode()
    ).hexdigest()
    require(payload.get("image_set_sha256") == image_digest, "routing image-set digest drift")
    count = 0
    for seed in SEEDS:
        cells = payload.get("seeds", {}).get(str(seed), {})
        require(set(cells) == set("cd"), f"routing/{seed} cell set drift")
        for cell in "cd":
            cell_payload = cells[cell]
            require(cell_payload.get("passed") is True, f"routing/{seed}/{cell} failed")
            checkpoint = Path(cell_payload.get("checkpoint", "")).resolve()
            probe_cell = probe["seeds"][str(seed)][cell]
            require(checkpoint == Path(probe_cell["checkpoint"]).resolve(), "routing checkpoint path drift")
            require(
                cell_payload.get("checkpoint_sha256") == probe_cell.get("checkpoint_sha256") == sha256(checkpoint),
                f"routing/{seed}/{cell} checkpoint SHA drift",
            )
            modules = cell_payload.get("modules", {})
            require(len(modules) == 6, f"routing/{seed}/{cell}: expected six routers")
            require(
                sorted(module.get("num_experts") for module in modules.values()) == [4, 4, 8, 8, 16, 16],
                f"routing/{seed}/{cell}: expert topology drift",
            )
            for name, module in modules.items():
                require(module.get("gate", {}).get("passed") is True, f"routing gate failed: {seed}/{cell}/{name}")
                num_experts = module.get("num_experts")
                counts = module.get("counts")
                require(module.get("top_k") == 2 and module.get("images") == 512, f"routing shape drift: {name}")
                require(
                    isinstance(counts, list)
                    and len(counts) == num_experts
                    and all(isinstance(value, int) and value >= 0 for value in counts),
                    f"invalid routing counts: {name}",
                )
                require(sum(counts) == 1024, f"routing selection total drift: {name}")
                require(
                    module.get("selections") == module.get("expected_selections") == 1024,
                    f"serialized routing selection total drift: {name}",
                )
                dead = [index for index, value in enumerate(counts) if value == 0]
                maximum = max(value / 512 for value in counts)
                shares = [value / 1024 for value in counts]
                entropy = -sum(value * math.log(value) for value in shares if value > 0) / math.log(num_experts)
                require(module.get("dead_experts_on_sample") == dead == [], f"dead expert: {seed}/{cell}/{name}")
                require(
                    math.isclose(module.get("max_image_selection_fraction", -1.0), maximum, abs_tol=1e-12),
                    f"selection fraction was not derived from counts: {name}",
                )
                require(
                    math.isclose(module.get("normalized_entropy", -1.0), entropy, abs_tol=1e-12),
                    f"entropy was not derived from counts: {name}",
                )
                require(maximum <= 0.8, f"selection cap failed: {name}")
                require(entropy >= 0.5, f"entropy failed: {name}")
                count += 1
    require(count == 36, f"routing gate count is {count}, expected 36")
    return count


def validate_residual(
    payload: dict[str, Any], protocol: dict[str, Any], routing: dict[str, Any], probe: dict[str, Any]
) -> int:
    require(payload.get("schema_version") == 1, "residual audit schema drift")
    require(payload.get("formal_activity_gate_passed") is True, "residual formal gate flag is false")
    thresholds = payload.get("gate_thresholds", {})
    require(thresholds.get("minimum_inclusive") == 0.0001, "residual minimum drift")
    require(thresholds.get("maximum_exclusive") == 0.1, "residual maximum drift")
    require(
        thresholds.get("applies_to") == "every layer 4/6/8 in C/D for every registered seed",
        "residual scope drift",
    )
    sample = payload.get("sample", {})
    require(Path(sample.get("data_yaml", "")).resolve() == Path(protocol["data"]["pilot"]["path"]).resolve(), "residual data drift")
    require(sample.get("images") == routing.get("images"), "residual and routing fixed images differ")
    require(sample.get("image_set_sha256") == routing.get("image_set_sha256"), "residual image-set digest drift")
    seeds_payload = payload.get("seeds", {})
    require(set(seeds_payload) == {str(seed) for seed in SEEDS}, "residual seed set drift")
    count = 0
    for seed in SEEDS:
        cells = seeds_payload[str(seed)]
        require(set(cells) == set("cd"), f"residual/{seed} cell set drift")
        for cell in "cd":
            item = cells[cell]
            require(item.get("passed") is True, f"residual/{seed}/{cell} failed")
            routing_cell = routing["seeds"][str(seed)][cell]
            probe_cell = probe["seeds"][str(seed)][cell]
            checkpoint = Path(item.get("checkpoint", "")).resolve()
            require(checkpoint == Path(routing_cell["checkpoint"]).resolve(), "residual checkpoint path drift")
            checkpoint_sha = routing_cell["checkpoint_sha256"]
            require(
                item.get("checkpoint_sha256_before")
                == item.get("checkpoint_sha256_after")
                == probe_cell.get("checkpoint_sha256")
                == checkpoint_sha
                == sha256(checkpoint),
                f"residual/{seed}/{cell} checkpoint byte drift",
            )
            require(item.get("checkpoint_unchanged") is True, "residual audit changed checkpoint")
            request_entry = protocol["requests"]["routing_probe"][str(seed)][cell]
            require(Path(item.get("request", "")).resolve() == Path(request_entry["path"]).resolve(), "residual request path drift")
            require(item.get("request_sha256") == request_entry["sha256"] == sha256(Path(request_entry["path"])), "residual request SHA drift")
            require(len(item.get("routers", [])) == 6, f"residual/{seed}/{cell}: router count drift")
            layers = item.get("layers", {})
            require(set(layers) == {"4", "6", "8"}, f"residual/{seed}/{cell} layer set drift")
            for layer, result in layers.items():
                require(result.get("passed") is True, f"residual gate failed: {seed}/{cell}/{layer}")
                base_energy = result.get("base_energy_sum")
                residual_energy = result.get("gated_residual_energy_sum")
                require(
                    isinstance(base_energy, (int, float))
                    and isinstance(residual_energy, (int, float))
                    and math.isfinite(base_energy)
                    and math.isfinite(residual_energy)
                    and base_energy > 0
                    and residual_energy >= 0,
                    f"residual energy is invalid: {seed}/{cell}/{layer}",
                )
                ratio = math.sqrt(residual_energy / base_energy)
                require(math.isclose(result.get("ratio", -1.0), ratio, rel_tol=1e-12, abs_tol=1e-12), "residual ratio derivation drift")
                require(0.0001 <= ratio < 0.1, f"residual ratio gate failed: {seed}/{cell}/{layer}")
                require(result.get("base_calls") == result.get("factor_calls") == 512, "residual sample call count drift")
                count += 1
    require(count == 18, f"residual gate count is {count}, expected 18")
    return count


def validate_initializers(protocol: dict[str, Any], protocol_path: Path) -> tuple[list[dict[str, str]], bool]:
    run_root = Path(protocol["run_root"])
    records: list[dict[str, str]] = []
    all_parity = True
    for seed in SEEDS:
        path = run_root / "initializers" / f"seed{seed}" / "initialization_manifest.json"
        manifest = read_json(path, f"initializer manifest seed{seed}")
        require_runtime_attestation(manifest, f"initializer manifest seed{seed}")
        require(manifest.get("schema_version") == 7 and manifest.get("status") == "passed", f"seed{seed} manifest failed")
        require(manifest.get("protocol_sha256") == sha256(protocol_path), f"seed{seed} manifest protocol drift")
        require(manifest.get("source_checkpoint_sha256") == protocol["source_checkpoint"]["sha256"], "source drift")
        require(manifest.get("source_checkpoint_sha256") == OFFICIAL_CHECKPOINT_SHA256, "official source SHA drift")
        require(set(manifest.get("cells", {})) == set(CELLS), f"seed{seed} initializer cells drift")
        require(manifest.get("a_b_dense_clean_aux_not_applicable") is True, f"seed{seed} A/B aux drift")
        require(manifest.get("c_d_clean_aux_policy_parity") is True, f"seed{seed} C/D aux parity failed")
        all_parity &= manifest.get("paired_factor_tensor_parity_after_reload") == {"a_b": True, "c_d": True}
        for cell in CELLS:
            item = manifest["cells"][cell]
            require(item.get("frozen_factor_base_parameters") == 459232, f"seed{seed}/{cell} base count drift")
            require(item.get("trainable_factor_base_parameters") == 0, f"seed{seed}/{cell} trainable base")
            require(
                item.get("reloaded_trainable_factor_base_parameters") == 0,
                f"seed{seed}/{cell} reload restored trainable base",
            )
            require(item.get("equivalence_before_save", {}).get("max_abs_error") == 0.0, "pre-save mismatch")
            require(item.get("equivalence_after_reload", {}).get("max_abs_error") == 0.0, "reload mismatch")
            clean_aux_routers = item.get("r21_clean_aux_routers")
            if cell in "ab":
                require(clean_aux_routers == [], f"seed{seed}/{cell}: dense clean-aux must be inapplicable")
            else:
                require(
                    isinstance(clean_aux_routers, list) and len(clean_aux_routers) == 6,
                    f"seed{seed}/{cell}: expected six clean-aux routers",
                )
                for router in clean_aux_routers:
                    require(
                        router.get("p1_balance_on_clean_routes") is True,
                        f"seed{seed}/{cell}/{router.get('name')}: clean-aux flag false",
                    )
                    require(
                        router.get("routing_aux_semantics")
                        == CLEAN_AUX_SEMANTICS,
                        f"seed{seed}/{cell}/{router.get('name')}: clean-aux semantics drift",
                    )
                    require(router.get("top_k") == 2, f"seed{seed}/{cell}: routing top-k drift")
        records.append({"path": str(path), "sha256": sha256(path)})
    require(all_parity, "paired initializer tensor parity failed")
    return records, all_parity


def validate_formal_lineage(protocol: dict[str, Any], protocol_path: Path) -> int:
    run_root = Path(protocol["run_root"])
    locked_params = {
        "epochs": 5,
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
        "amp": False,
        "deterministic": True,
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
        "device": "0",
        "moe_noise_std": 0.0,
        "moe_router_lr_scale": 1.0,
        "moe_expert_warmup_epochs": 0,
        "moe_dynamic_schedule": "none",
        "moe_map_saturation_enabled": False,
        "moe_balance_loss": 1.0,
        "moe_router_z_loss": 0.1,
        "moe_aux_gain": 1.0,
        "mixture_aux_budget": 3.0,
        "moe_temperature": 1.0,
        "moa_mot_temperature_factor": 1.0,
        "moa_mot_min_temperature": 1.0,
    }
    count = 0
    for seed in SEEDS:
        manifest_path = run_root / "initializers" / f"seed{seed}" / "initialization_manifest.json"
        manifest = read_json(manifest_path, f"initializer manifest seed{seed}")
        for cell in CELLS:
            entry = protocol["requests"]["formal"][str(seed)][cell]
            request_path = Path(entry["path"])
            require(sha256(request_path) == entry["sha256"], f"formal request hash drift: {seed}/{cell}")
            request = read_json(request_path, f"formal request {seed}/{cell}")
            require(
                Path(request.get("protocol", {}).get("path", "")).resolve() == protocol_path,
                f"formal protocol registration drift: {seed}/{cell}",
            )
            initializer = Path(request["inputs"]["model"]).resolve()
            expected = run_root / "initializers" / f"seed{seed}" / f"{cell}_residual_factor_init.pt"
            require(initializer == expected.resolve(), f"formal initializer path drift: {seed}/{cell}")
            require(sha256(initializer) == manifest["cells"][cell]["initializer_sha256"], "initializer hash drift")
            require(request["a1_policy"].get("formal_restart_from_initializer") is True, "formal restart flag false")
            require(request["a1_policy"].get("freeze_batch_norm") is True, "formal BN freeze flag false")
            require(request["a1_policy"].get("freeze_residual_factor_bases") is True, "formal base freeze flag false")
            expected_aux_enabled = cell in "cd"
            require(
                request["a1_policy"].get("routing_auxiliary_objective", {}).get("enabled")
                is expected_aux_enabled,
                f"formal clean-aux cell policy drift: {seed}/{cell}",
            )
            if expected_aux_enabled:
                require(
                    request["a1_policy"]["routing_auxiliary_objective"].get("runtime_semantics")
                    == CLEAN_AUX_SEMANTICS,
                    f"formal clean-aux semantics drift: {seed}/{cell}",
                )
            params = request["params"]
            for key, expected_value in locked_params.items():
                require(params.get(key) == expected_value, f"formal budget drift {seed}/{cell}/{key}")
            require(params.get("seed") == seed, f"formal seed drift: {seed}/{cell}")
            require(params.get("freeze") == protocol["freeze"], f"formal freeze layer drift: {seed}/{cell}")
            require(
                Path(request["inputs"].get("data", "")).resolve()
                == Path(protocol["data"]["pilot"]["path"]).resolve(),
                f"formal data drift: {seed}/{cell}",
            )
            require(request["inputs"].get("task") == "detect", f"formal task drift: {seed}/{cell}")
            expected_project = run_root / "formal" / f"seed{seed}"
            expected_name = f"{cell}_formal_seed{seed}_5ep"
            require(Path(params.get("project", "")).resolve() == expected_project.resolve(), "formal project drift")
            require(params.get("name") == expected_name, f"formal run name drift: {seed}/{cell}")
            count += 1
    require(count == 12, f"formal lineage count is {count}, expected 12")
    return count


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path, "r21 protocol")
    assert_protocol_runtime(protocol)
    protocol_sha = sha256(protocol_path)
    require(protocol.get("schema_version") == 7 and protocol.get("experiment_tag") == "r21", "not r21 schema7")
    require(protocol.get("locked_official_baseline_sha") == LOCKED_SHA, "locked baseline SHA drift")
    require(protocol.get("locked_official_baseline_ancestor_verified") is True, "baseline ancestor flag false")
    require(
        protocol.get("source_checkpoint", {}).get("sha256") == OFFICIAL_CHECKPOINT_SHA256,
        "protocol source checkpoint is not the locked official yolo26n.pt",
    )
    source_checkpoint = Path(protocol["source_checkpoint"]["path"])
    require(source_checkpoint.is_file() and sha256(source_checkpoint) == OFFICIAL_CHECKPOINT_SHA256, "official checkpoint drift")
    validate_data_registry(protocol)
    verify_registered_data_content(protocol)
    validate_model_configs(protocol)
    validate_implementation(protocol)
    output = args.output.resolve()
    expected_output = Path(protocol["formal_admission"]["path"]).resolve()
    require(output == expected_output, f"formal admission output must be {expected_output}")
    require(not output.exists(), f"refusing to overwrite formal admission: {output}")
    formal_root = Path(protocol["run_root"]) / "formal"
    require(not formal_root.exists(), f"formal directory already exists: {formal_root}")

    preflight, preflight_id = passed_evidence(args.preflight_audit.resolve(), "preflight audit", protocol_sha)
    probe, probe_id = passed_evidence(args.probe_audit.resolve(), "probe audit", protocol_sha)
    routing, routing_id = passed_evidence(args.routing_audit.resolve(), "routing audit", protocol_sha)
    residual, residual_id = passed_evidence(args.residual_audit.resolve(), "residual audit", protocol_sha)
    migration, migration_id = passed_evidence(args.migration_audit.resolve(), "migration audit", protocol_sha)
    for payload, label in (
        (preflight, "preflight checkpoint audit"),
        (probe, "routing-probe checkpoint audit"),
        (routing, "routing audit"),
        (residual, "residual audit"),
        (migration, "migration audit"),
    ):
        require_runtime_attestation(payload, label)
    require(migration.get("a_b_invariance_verified") is True, "A/B invariance not verified")
    require(migration.get("c_d_clean_aux_parity_verified") is True, "C/D clean-aux parity not verified")
    require(migration.get("implementation_hashes_verified") is True, "implementation hashes not verified")
    require(migration.get("git_lineage_verified") is True, "git baseline/implementation lineage not verified")
    require(migration.get("official_checkpoint_verified") is True, "official checkpoint not verified")
    require(migration.get("data_hashes_verified") is True, "dataset lists were not verified")

    require_identity(
        routing.get("dependencies", {}).get("preflight_checkpoint_audit"),
        preflight_id,
        "routing -> preflight checkpoint audit",
    )
    require_identity(
        routing.get("dependencies", {}).get("probe_checkpoint_audit"),
        probe_id,
        "routing -> probe checkpoint audit",
    )
    require_identity(
        residual.get("dependencies", {}).get("routing_probe_checkpoint_audit"),
        probe_id,
        "residual -> probe checkpoint audit",
    )
    require_identity(
        residual.get("dependencies", {}).get("hard_top2_routing_audit"),
        routing_id,
        "residual -> hard-Top2 routing audit",
    )

    initializer_records, initializer_parity = validate_initializers(protocol, protocol_path)
    preflight_count = validate_checkpoint_audit(preflight, "preflight", CELLS, protocol)
    probe_count = validate_checkpoint_audit(probe, "routing_probe", "cd", protocol)
    routing_count = validate_routing(routing, protocol, probe)
    residual_count = validate_residual(residual, protocol, routing, probe)
    formal_count = validate_formal_lineage(protocol, protocol_path)
    preflight_status_id = validate_stage_execution(protocol, protocol_path, "preflight", CELLS)
    probe_status_id = validate_stage_execution(protocol, protocol_path, "routing_probe", "cd")
    expected_gpu = {"260829": "0", "260830": "1", "260831": "0"}
    require(protocol.get("gpu_policy", {}).get("seed_to_physical_gpu") == expected_gpu, "GPU schedule drift")

    report = {
        "schema_version": 1,
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "runtime_attestation": RUNTIME_ATTESTATION,
        "dependencies": {
            "preflight_checkpoint_audit": preflight_id,
            "routing_probe_checkpoint_audit": probe_id,
            "hard_top2_routing_audit": routing_id,
            "residual_activity_audit": residual_id,
            "migration_audit": migration_id,
            "preflight_execution_status": preflight_status_id,
            "routing_probe_execution_status": probe_status_id,
            "initializer_manifests": initializer_records,
        },
        "counts": {
            "initializer_manifests": len(initializer_records),
            "preflight_cells": preflight_count,
            "routing_probe_cells": probe_count,
            "routing_routers": routing_count,
            "residual_layers": residual_count,
            "formal_requests": formal_count,
        },
        "initializer_parity_verified": initializer_parity,
        "official_checkpoint_verified": True,
        "data_hashes_verified": True,
        "implementation_and_git_lineage_verified": True,
        "dependency_hash_graph_verified": True,
        "raw_gate_metrics_recomputed": True,
        "formal_request_lineage_verified": True,
        "gpu_schedule_verified": True,
        "formal_directory_absent_at_admission": True,
        "all_required_gates_passed": True,
        "formal_may_start": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": "passed", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
