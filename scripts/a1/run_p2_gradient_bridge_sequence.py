"""Serial detached pilot launcher; preserve checkpoints and never touch other GPU processes."""

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def memory_used(device):
    result = subprocess.check_output(
        ["nvidia-smi", "-i", str(device), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return int(result.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "train"), required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    root = Path(protocol["root"])
    lock = (root / "sequence.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.stage == "train":
        for path in protocol["preflights"]:
            request = json.loads(Path(path).read_text())
            run = Path(request["params"]["project"]) / request["params"]["name"]
            if not (run / "completed.json").exists():
                raise RuntimeError(f"preflight incomplete: {run}")
        gate = json.loads((root / "launch_gate.json").read_text())
        if (
            gate["status"] != "passed"
            or gate["protocol_sha256"] != hashlib.sha256(args.protocol.read_bytes()).hexdigest()
        ):
            raise RuntimeError("missing or mismatched launch gate")
    paths = protocol["preflights"] if args.stage == "preflight" else protocol["order"]
    state = {"stage": args.stage, "status": "starting", "pid": __import__("os").getpid()}

    def save(**kwargs):
        state.update(kwargs, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        (root / f"{args.stage}_status.json").write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(state), flush=True)

    for path in paths:
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol["requests"][str(path)]:
            raise RuntimeError(f"request drift: {path}")
        request = json.loads(path.read_text())
        run = Path(request["params"]["project"]) / request["params"]["name"]
        if (run / "completed.json").exists():
            save(status="skipping_completed", cell=run.name)
            continue
        attempt = 0
        while True:
            attempt += 1
            while memory_used(protocol["device"]) > protocol["gpu_start_limit_mib"]:
                save(status="waiting_memory", cell=run.name)
                time.sleep(30)
            last = run / "weights/last.pt"
            command = [
                sys.executable,
                str(Path(__file__).with_name("run_p2_gradient_bridge_pilot.py")),
                "--request",
                str(path),
            ]
            if last.exists():
                command.append("--resume")
            elif args.stage == "preflight":
                command.append("--stop-after-checkpoint")
            log_path = root / "logs" / f"{args.stage}_{run.name}.log"
            with log_path.open("a") as log:
                log.write(f"\nATTEMPT {attempt} {time.strftime('%FT%T%z')} resume={last.exists()}\n")
                log.flush()
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                save(status="running", cell=run.name, child_pid=proc.pid, log=str(log_path), attempt=attempt)
                memory_stop = False
                while proc.poll() is None:
                    time.sleep(15)
                    if proc.poll() is None and memory_used(protocol["device"]) > protocol["gpu_total_limit_mib"]:
                        proc.terminate()
                        memory_stop = True
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
            if proc.returncode == 0 and (run / "completed.json").exists():
                save(status="cell_completed", cell=run.name)
                break
            if proc.returncode == 75 and args.stage == "preflight" and last.exists():
                save(status="planned_resume_probe", cell=run.name)
                continue
            tail = log_path.read_text(errors="replace")[-16000:]
            transient = memory_stop or proc.returncode in (-9, -15) or "CUDA out of memory" in tail
            if transient and last.exists():
                save(status="retrying_saved_epoch", cell=run.name, returncode=proc.returncode)
                time.sleep(60)
                continue
            save(status="failed", cell=run.name, returncode=proc.returncode, has_checkpoint=last.exists())
            raise RuntimeError("cell failed; preserved outputs, no initializer restart or next-cell launch")
    save(status="completed", child_pid=None)


if __name__ == "__main__":
    main()
