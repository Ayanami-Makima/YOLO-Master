#!/usr/bin/env python3
"""P2-B r1: fixed val512 routing-load and duplicate-output diagnostics."""
from __future__ import annotations

import argparse, collections, hashlib, json, math, sys
from datetime import datetime, timezone
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def entropy(values):
    total = sum(values)
    if total <= 0: return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log(p) for p in probs) / math.log(len(values)) if len(values) > 1 else 0.0

def gini(values):
    xs = sorted(float(v) for v in values); n = len(xs); s = sum(xs)
    if n == 0 or s == 0: return 0.0
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * s)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2,bx2)-max(ax1,bx1)) * max(0.0, min(ay2,by2)-max(ay1,by1))
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1); area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    return inter / max(area_a + area_b - inter, 1e-12)

def duplicate_rate(result):
    boxes = result.boxes
    if boxes is None or len(boxes) < 2: return 0.0, 0
    xyxy = boxes.xyxy.detach().cpu().tolist(); cls = boxes.cls.detach().cpu().tolist()
    dup = 0
    for i in range(len(xyxy)):
        for j in range(i + 1, len(xyxy)):
            if int(cls[i]) == int(cls[j]) and iou(xyxy[i], xyxy[j]) >= 0.9: dup += 1
    return dup / max(len(xyxy), 1), len(xyxy)

def audit_checkpoint(checkpoint: Path, images: list[Path], device: str):
    from ultralytics import YOLO
    from ultralytics.nn.modules.moe.modules import OptimizedMOEImproved
    model = YOLO(str(checkpoint), task="detect")
    model.model.to(device).float().eval()
    modules = [(name, m) for name, m in model.model.named_modules() if isinstance(m, OptimizedMOEImproved)]
    counters = {name: collections.Counter() for name, _ in modules}
    hooks = []
    for name, module in modules:
        def hook(_m, _inputs, output, key=name):
            indices = output[1].detach().reshape(-1).cpu().tolist()
            counters[key].update(int(i) for i in indices)
        hooks.append(module.routing.register_forward_hook(hook))
    duplicate_sum = 0.0; candidate_sum = 0; candidate_n = 0
    try:
        for path in images:
            results = model.predict(source=str(path), device=device, imgsz=640, conf=0.25, iou=0.7, max_det=300, verbose=False)
            dr, n = duplicate_rate(results[0]); duplicate_sum += dr; candidate_sum += n; candidate_n += 1
    finally:
        for h in hooks: h.remove()
    route = {}
    for name, module in modules:
        e = int(module.num_experts); counts = [int(counters[name][i]) for i in range(e)]
        total = sum(counts); fractions = [c / max(total, 1) for c in counts]
        route[name] = {"num_experts": e, "top_k": int(module.top_k), "counts": counts,
                       "selection_fractions": fractions, "max_selection_fraction": max(fractions, default=0.0),
                       "normalized_entropy": entropy(counts), "gini": gini(counts),
                       "dead_experts": [i for i, c in enumerate(counts) if c == 0], "total_selections": total}
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "routed_modules": route,
            "images": len(images), "mean_duplicate_rate": duplicate_sum / max(candidate_n, 1),
            "mean_candidates": candidate_sum / max(candidate_n, 1)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint-root", type=Path, required=True); p.add_argument("--val-list", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--device", default="cuda:0")
    a=p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    images=[Path(x.strip()) for x in a.val_list.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(images) != 512: raise ValueError(f"expected 512 images, got {len(images)}")
    out={"schema":"a1-p2-mechanism-r1/v1","status":"running","created_at":datetime.now(timezone.utc).isoformat(),"protocol":{"device":a.device,"images":512,"imgsz":640,"router":"eval noise-free hard Top-2"},"val_list":str(a.val_list),"val_list_sha256":sha256(a.val_list),"seeds":{}}
    for seed in ("260829","260830","260831"):
        out["seeds"][seed]={}
        for cell in ("c","d"):
            ck=a.checkpoint_root/f"seed{seed}"/f"{cell}_formal_seed{seed}_15ep"/"weights"/"last.pt"
            if not ck.is_file(): raise FileNotFoundError(ck)
            out["seeds"][seed][cell]=audit_checkpoint(ck,images,a.device)
    out["status"]="completed"; out["completed_at"]=datetime.now(timezone.utc).isoformat()
    (a.output/"evidence.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":out["status"],"output":str(a.output)},ensure_ascii=False))
if __name__ == "__main__": main()
