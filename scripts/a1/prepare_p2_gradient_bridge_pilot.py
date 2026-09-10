"""Create immutable paired bridge pilot requests, separate from all previous experiments."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from prepare_p2_e2e_candidate_r5 import common_params


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--initializers", type=Path, required=True)
    parser.add_argument("--device", default="1")
    args = parser.parse_args()
    repo, root = args.repo.resolve(), args.root.resolve()
    if root.exists():
        raise FileExistsError(root)
    data = repo / "configs/a1/p1_pretrained/pilot_data/coco.yaml"
    tiny = repo / "configs/a1/p1_pretrained/preflight_data/coco.yaml"
    lists = [data.parent / f"{split}2017.txt" for split in ("train", "val")]
    assert [len(p.read_text().splitlines()) for p in lists] == [5000, 512]
    sources = [
        "ultralytics/nn/modules/head.py",
        "ultralytics/nn/modules/moe/factor_adapter.py",
        "ultralytics/nn/modules/moe/modules.py",
        "ultralytics/nn/modules/moe/routers.py",
        "ultralytics/nn/tasks.py",
        "ultralytics/utils/loss.py",
        "ultralytics/utils/tal.py",
        "ultralytics/engine/trainer.py",
        "ultralytics/engine/model.py",
        "ultralytics/cfg/default.yaml",
        "scripts/a1/run_p1_bn_frozen.py",
        "scripts/a1/run_p2_gradient_bridge_pilot.py",
        "scripts/a1/prepare_p2_e2e_candidate_r5.py",
        "scripts/a1/prepare_p2_gradient_bridge_pilot.py",
        "scripts/a1/run_p2_gradient_bridge_sequence.py",
    ]
    source_hashes = {p: sha256(repo / p) for p in sources}
    root.mkdir(parents=True)
    (root / "requests").mkdir()
    (root / "logs").mkdir()
    order = []
    preflights = []
    for cell in ("b", "d"):
        init = args.initializers / f"{cell}_residual_factor_init.pt"
        for alpha in (0.0, 0.1):
            name = f"{cell}_alpha{str(alpha).replace('.', 'p')}_seed260829"
            request = {
                "skill": "yolo.train",
                "request_id": f"{root.name}_{name}",
                "inputs": {"model": str(init), "data": str(data), "task": "detect"},
                "params": common_params(root / "train", name),
                "policy": {"dry_run": False},
                "a1_policy": {
                    "freeze_batch_norm": True,
                    "freeze_residual_factor_bases": True,
                    "routing_semantics": "deterministic_hard_top2_from_step_zero",
                    "expert_dropout_rate": 0.0,
                    "o2o_tal_topk": 7,
                },
                "p2_bridge": {
                    "alpha": alpha,
                    "source_hashes": source_hashes,
                    "input_hashes": {str(p): sha256(p) for p in (init, data, *lists)},
                    "cuda_allocator_limit_gib": 11,
                    "train_images": 5000,
                },
            }
            request["params"]["device"] = args.device
            path = root / "requests" / f"{name}.json"
            path.write_text(json.dumps(request, indent=2) + "\n")
            order.append(str(path))
            preflight = copy.deepcopy(request)
            preflight["request_id"] += "_preflight"
            preflight["inputs"]["data"] = str(tiny)
            preflight["params"].update(epochs=2, project=str(root / "preflight"), device=args.device)
            preflight["p2_bridge"]["input_hashes"][str(tiny)] = sha256(tiny)
            tiny_train = tiny.parent / "train2017.txt"
            tiny_val = tiny.parent / "val2017.txt"
            preflight["p2_bridge"]["train_images"] = len(tiny_train.read_text().splitlines())
            preflight["p2_bridge"]["input_hashes"].update({str(p): sha256(p) for p in (tiny_train, tiny_val)})
            preflight_path = root / "requests" / f"{name}_preflight.json"
            preflight_path.write_text(json.dumps(preflight, indent=2) + "\n")
            preflights.append(str(preflight_path))
    protocol = {
        "schema": "p2-gradient-bridge-pilot/v1",
        "status": "prepared",
        "root": str(root),
        "device": args.device,
        "preflights": preflights,
        "order": order,
        "source_hashes": source_hashes,
        "requests": {p: sha256(Path(p)) for p in preflights + order},
        "data": {"train": 5000, "val": 512},
        "epochs": 5,
        "seed": 260829,
        "comparison": "B/D x alpha=0/0.1, restart each cell from original initializer, no preflight weight reuse",
        "checkpoint_selection": "last epoch EMA for primary paired comparison; best secondary only",
        "selection_for_three_seeds": {"d_map_delta_min": 0.001, "d_recall_not_down": True, "b_map_delta_min": -0.001},
        "selection_note": "Internal budget screen, not A1 acceptance or statistical significance; export still required",
        "max_concurrent_training": 1,
        "gpu_total_limit_mib": 22528,
        "gpu_start_limit_mib": 10240,
        "resume_note": "SSH disconnect survives; restart from last saved epoch only, not bitwise uninterrupted equivalence",
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(str(root / "protocol.json"))


if __name__ == "__main__":
    main()
