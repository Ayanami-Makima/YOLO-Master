"""Evaluate P2 efficiency-screen pilots on the fixed 512-image validation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    repo = args.repo.resolve()
    protocol = json.loads(args.protocol.resolve().read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(repo))
    from run_p2_e2e_precision_diagnostics_r1 import run_cell

    evidence = {
        "schema": "a1-p2-late-moe-efficiency-screen-eval/v1",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(args.protocol.resolve()),
        "data": protocol["data"],
        "device": args.device,
        "images": 512,
        "runs": {},
    }
    for name in protocol["variants"]:
        checkpoint = Path(protocol["run_root"]) / "pilot" / f"{name}_seed260829_1ep" / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result = run_cell(checkpoint, Path(protocol["data"]["path"]), output / name, args.device)
        result["variant"] = name
        result["checkpoint_sha256"] = sha256(checkpoint)
        evidence["runs"][name] = result
    evidence["status"] = "completed"
    evidence["completed_at"] = datetime.now(timezone.utc).isoformat()
    (output / "evaluation_evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
