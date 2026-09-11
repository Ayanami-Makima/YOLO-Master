"""Audit disjoint samples, real preflight resumes and test evidence before pilot launch."""

import argparse
import hashlib
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original-audit", type=Path, required=True)
    parser.add_argument("--holdout-audit", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "launch_gate.json"
    if output.exists():
        raise FileExistsError(output)
    protocol_path = args.root / "protocol.json"
    protocol = read(protocol_path)
    repo = Path(__file__).resolve().parents[2]
    for p, digest in protocol["source_hashes"].items():
        assert sha256(repo / p) == digest, p
    test_path = args.root / "test_results.xml"
    suites = ET.parse(test_path).getroot().findall("testsuite")
    assert suites and sum(int(s.get("tests", 0)) for s in suites) >= 24
    assert all(int(s.get(k, 0)) == 0 for s in suites for k in ("errors", "failures", "skipped"))
    rows = []
    fingerprints = set()
    for seed in (260829, 260830, 260831):
        original = read(args.original_audit / f"evidence_seed{seed}.json")
        path = args.holdout_audit / f"evidence_seed{seed}.json"
        holdout = read(path)
        assert holdout["status"] == "completed" and holdout["offset"] == 128
        assert holdout["optimizer_steps"] == 0 and len(holdout["rows"]) == 64
        assert not {r["image_sha256"] for r in original["rows"]} & {r["image_sha256"] for r in holdout["rows"]}
        fingerprints.add(tuple((r["cell"], r["image_sha256"]) for r in holdout["rows"]))
        assert all(not c["state_changed_keys"] for c in holdout["checkpoints"].values())
        for row in holdout["rows"]:
            assert row["forward_max_error"] == 0.0
            for group in ("factor", "router") if row["cell"] == "d" else ("factor",):
                assert row["branches"]["native_one2one"]["gradients"][group]["l2"] == 0
                assert row["branches"]["bridge_one2one"]["gradients"][group]["l2"] > 0
        d_cosines = [
            r["branches"]["bridge_one2one"]["gradients"]["router"]["cosine_with_one2many"]
            for r in holdout["rows"]
            if r["cell"] == "d"
        ]
        assert statistics.mean(d_cosines) > 0, "sample-direction interpretation changed; inspect before training"
        rows.append(
            {
                "seed": seed,
                "router_cosine_mean": statistics.mean(d_cosines),
                "negative_images": sum(v < 0 for v in d_cosines),
                "source_sha256": sha256(path),
            }
        )
    assert len(fingerprints) == 1
    cells = []
    for path in protocol["preflights"]:
        path = Path(path)
        assert sha256(path) == protocol["requests"][str(path)]
        request = read(path)
        run = Path(request["params"]["project"]) / request["params"]["name"]
        completed = read(run / "completed.json")
        runtime = read(run / "p2_bridge_runtime.json")
        first = read(run / "p2_first_forward.json")
        epochs = [json.loads(line) for line in (run / "p2_epoch_audit.jsonl").read_text().splitlines()]
        alpha = request["p2_bridge"]["alpha"]
        assert completed["status"] == "completed" and completed["alpha"] == alpha
        assert runtime["resume"] and runtime["start_epoch"] == 1 and runtime["alpha"] == alpha
        assert runtime["frozen_factor_base_parameters"] == 459232
        assert runtime["batch"] == request["params"]["batch"] == 4
        assert runtime["train_images"] == request["p2_bridge"]["train_images"]
        assert runtime["batches_per_epoch"] == (runtime["train_images"] + 3) // 4
        assert (runtime["actual_topk"], runtime["actual_topk2"]) == (7, 1)
        assert runtime["source_hashes"] == protocol["source_hashes"]
        assert first["training"] and all(v == bool(alpha) for v in first["one2one_feature_requires_grad"])
        assert [e["epoch"] for e in epochs] == [1, 2]
        assert all(e["frozen_unchanged"] for e in epochs)
        assert len(epochs[-1]["gains"]) == 3 and all(g["abs_max"] > 0 for g in epochs[-1]["gains"])
        cells.append(
            {
                "cell": run.name,
                "alpha": alpha,
                "resumed_epoch": 2,
                "frozen_unchanged_per_segment": True,
                "runtime_sha256": sha256(run / "p2_bridge_runtime.json"),
            }
        )
    gate = {
        "status": "passed",
        "protocol_sha256": sha256(protocol_path),
        "test_results_sha256": sha256(test_path),
        "gate_script_sha256": sha256(Path(__file__)),
        "holdout": rows,
        "preflight": cells,
        "scope": "Correctness and sample-sensitivity checks only; no accuracy efficacy or export-backend claim",
        "resume_scope": "Actual optimizer/epoch resume checked; numerical equivalence to uninterrupted training not claimed",
    }
    output.write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
