"""Benchmark the completed P2 efficiency-screen pilots on one GPU."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def set_end2end(model, enabled: bool) -> None:
    model.end2end = enabled
    head = model.model[-1]
    if not hasattr(head, "end2end"):
        raise TypeError("checkpoint head has no end2end switch")
    head.end2end = enabled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    repo = args.repo.resolve()
    protocol = json.loads(args.protocol.resolve().read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(repo))
    import torch

    from ultralytics import YOLO

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    image = torch.linspace(0.0, 1.0, steps=3 * 640 * 640, dtype=torch.float32, device=device).reshape(1, 3, 640, 640)
    evidence = {
        "schema": "a1-p2-late-moe-efficiency-screen-benchmark/v1",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(args.protocol.resolve()),
        "device": str(device),
        "warmup": args.warmup,
        "samples": args.samples,
        "input": [1, 3, 640, 640],
        "variants": {},
    }
    sample_rows = []
    for name, spec in protocol["variants"].items():
        checkpoint = Path(protocol["run_root"]) / "pilot" / f"{name}_seed260829_1ep" / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = YOLO(str(checkpoint), task="detect").model.to(device).float().eval()
        variant_entry = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "moe": bool(spec["moe"]),
            "experts": int(spec["experts"]),
            "top_k": int(spec["top_k"]),
            "factor_mlp_ratio": float(spec["factor_mlp_ratio"]),
            "modes": {},
        }
        for mode, enabled in (("e2e", True), ("nms", False)):
            set_end2end(model, enabled)
            with torch.inference_mode():
                for _ in range(args.warmup):
                    model(image)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            times = []
            with torch.inference_mode():
                for sample in range(args.samples):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(torch.cuda.current_stream(device))
                    model(image)
                    end.record(torch.cuda.current_stream(device))
                    end.synchronize()
                    elapsed = float(start.elapsed_time(end))
                    times.append(elapsed)
                    sample_rows.append({"variant": name, "mode": mode, "sample": sample, "model_forward_ms": elapsed})
            variant_entry["modes"][mode] = {
                "mean_ms": statistics.mean(times),
                "p50_ms": percentile(times, 0.50),
                "p90_ms": percentile(times, 0.90),
                "p99_ms": percentile(times, 0.99),
                "throughput_images_per_s": 1000.0 / statistics.mean(times),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        # THOP is an auxiliary static profile.  Sparse dynamic FLOPs may not
        # equal runtime executed FLOPs, so failures are recorded, not hidden.
        try:
            from thop import profile

            set_end2end(model, True)
            macs, profiled_parameters = profile(model, inputs=(image,), verbose=False)
            variant_entry["thop_e2e"] = {
                "macs": float(macs),
                "gflops": float(macs) * 2.0 / 1e9,
                "parameters_profiled": int(profiled_parameters),
                "note": "static THOP profile; dynamic sparse routing interpretation is auxiliary",
            }
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            variant_entry["thop_e2e"] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
        evidence["variants"][name] = variant_entry
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    evidence["status"] = "completed"
    evidence["completed_at"] = datetime.now(timezone.utc).isoformat()
    with (output / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "mode", "sample", "model_forward_ms"])
        writer.writeheader()
        writer.writerows(sample_rows)
    (output / "benchmark_evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
