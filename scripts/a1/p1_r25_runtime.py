#!/usr/bin/env python3
"""Fail-closed runtime binding for the isolated A1 P1 r25 worktree."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT_TEXT = str(REPO_ROOT)

# An absolute script path makes Python put ``scripts/a1`` rather than the
# repository root at sys.path[0].  The shared virtualenv is an editable install
# of another checkout, so pin this worktree before importing Ultralytics.
if not sys.path or sys.path[0] != _REPO_ROOT_TEXT:
    sys.path.insert(0, _REPO_ROOT_TEXT)


def _module_source(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    source = getattr(module, "__file__", None)
    if not source:
        raise RuntimeError(f"r25 runtime module has no source path: {module_name}")
    return Path(source).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_r25_runtime() -> dict[str, str]:
    """Require critical imports to resolve inside this exact worktree."""
    expected = {
        "ultralytics": (REPO_ROOT / "ultralytics/__init__.py").resolve(),
        "ultralytics.nn.tasks": (REPO_ROOT / "ultralytics/nn/tasks.py").resolve(),
        "ultralytics.nn.modules.moe.factor_adapter": (
            REPO_ROOT / "ultralytics/nn/modules/moe/factor_adapter.py"
        ).resolve(),
        "ultralytics.nn.modules.moe.modules": (
            REPO_ROOT / "ultralytics/nn/modules/moe/modules.py"
        ).resolve(),
        "ultralytics.nn.modules.moe.routers": (
            REPO_ROOT / "ultralytics/nn/modules/moe/routers.py"
        ).resolve(),
    }
    actual = {}
    for name, expected_path in expected.items():
        actual_path = _module_source(name)
        if actual_path != expected_path:
            mismatch = {name: {"expected": str(expected_path), "actual": str(actual_path)}}
            raise RuntimeError(f"r25 runtime repository mismatch: {json.dumps(mismatch, sort_keys=True)}")
        actual[name] = actual_path
    return {name: str(path) for name, path in actual.items()}


RUNTIME_SOURCES = assert_r25_runtime()
_torch = importlib.import_module("torch")
_torch_path = _module_source("torch")
RUNTIME_ATTESTATION = {
    "repo_root": str(REPO_ROOT),
    "interpreter": {
        "executable": str(Path(sys.executable).absolute()),
        "prefix": str(Path(sys.prefix).absolute()),
        "base_prefix": str(Path(sys.base_prefix).absolute()),
        "version": platform.python_version(),
        "virtualenv": sys.prefix != sys.base_prefix,
    },
    "torch": {
        "path": str(_torch_path),
        "sha256": _sha256(_torch_path),
        "version": str(_torch.__version__),
        "cuda_version": str(_torch.version.cuda),
    },
    "modules": {
        name: {"path": path, "sha256": _sha256(Path(path))}
        for name, path in sorted(RUNTIME_SOURCES.items())
    },
}


def assert_protocol_runtime(protocol: dict) -> None:
    """Bind a protocol to the modules that this process actually loaded."""
    if protocol.get("runtime_binding") != RUNTIME_ATTESTATION:
        raise RuntimeError("r25 protocol/runtime module provenance mismatch")


if __name__ == "__main__":
    print(json.dumps(RUNTIME_ATTESTATION, sort_keys=True))

