"""Append-only operational export-parity audit for P1 r28 end-to-end outputs.

The primary closure intentionally compares every raw E2E TopK row, including
very-low-confidence filler rows. This companion audit does not replace that
strict result. It checks whether the exported and eager outputs agree after
the same confidence threshold used by the locked native predict path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

CELLS = ("b", "d")
SEEDS = ("260829", "260830", "260831")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def load_comparator(repo: Path):
    path = repo / "scripts/a1/evaluate_p1_matrix.py"
    spec = importlib.util.spec_from_file_location("p1_matrix_threshold_comparator", path)
    require(spec is not None and spec.loader is not None, f"cannot load comparator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare_export_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--export-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.001)
    args = parser.parse_args()
    require(args.repo.is_dir(), f"missing repo: {args.repo}")
    require(args.export_evidence.is_file(), f"missing export evidence: {args.export_evidence}")
    require(0 <= args.confidence <= 1, "confidence must be within [0, 1]")

    import numpy as np

    compare = load_comparator(args.repo.resolve())
    strict = read_json(args.export_evidence)
    tolerance = strict["seeds"][SEEDS[0]]["cells"][CELLS[0]]["comparisons"][0]
    atol, rtol = float(tolerance["atol"]), float(tolerance["rtol"])
    destination = args.output / "export_threshold_semantics_r1"
    destination.mkdir(parents=True, exist_ok=False)
    evidence = {
        "schema": "a1-p1-r28-e2e-export-threshold-evidence/v1",
        "status": "completed",
        "scope": "E2E B/D only; companion to, not replacement for, strict all-row raw export parity",
        "confidence": args.confidence,
        "atol": atol,
        "rtol": rtol,
        "strict_export_evidence": {"path": str(args.export_evidence), "sha256": digest(args.export_evidence)},
        "seeds": {},
    }
    for seed in SEEDS:
        evidence["seeds"][seed] = {}
        for cell in CELLS:
            strict_cell = strict["seeds"][seed]["cells"][cell]
            entry = {"status": "completed", "strict_raw_status": strict_cell["status"], "inputs": []}
            for comparison in strict_cell["comparisons"]:
                arrays = np.load(comparison["outputs"], allow_pickle=False)
                eager, onnx = arrays["pytorch"], arrays["onnx"]
                require(eager.shape == onnx.shape and eager.shape[0] == 1 and eager.shape[-1] == 6, "invalid E2E arrays")
                eager_filtered = eager[:, eager[0, :, 4] >= args.confidence, :]
                onnx_filtered = onnx[:, onnx[0, :, 4] >= args.confidence, :]
                result = compare(eager_filtered, onnx_filtered, end2end=True, atol=atol, rtol=rtol)
                input_entry = {
                    "input_index": comparison["input_index"],
                    "outputs": comparison["outputs"],
                    "outputs_sha256": digest(Path(comparison["outputs"])),
                    "eager_rows_at_or_above_confidence": int(eager_filtered.shape[1]),
                    "onnx_rows_at_or_above_confidence": int(onnx_filtered.shape[1]),
                    "comparison": result,
                }
                entry["inputs"].append(input_entry)
                if result["status"] != "passed":
                    entry["status"] = "failed"
                    evidence["status"] = "partial"
            evidence["seeds"][seed][cell] = entry
    write_json(destination / "evidence.json", evidence)


if __name__ == "__main__":
    main()
