"""Regression tests for long-running P1 logs, stale jobs and protocol checks."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "agent"))
sys.path.insert(0, str(REPO / "scripts/a1"))

import run_p1_lane as lane  # noqa: E402
from runtime.cli import executor  # noqa: E402


class LiveLogTests(unittest.TestCase):
    def test_logs_are_visible_before_exit_and_response_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            log_root = Path(directory)
            code = (
                "import sys,time;print('early',flush=True);"
                "print('early-error',file=sys.stderr,flush=True);time.sleep(1.5);"
                "print('x'*90000);print('late',flush=True)"
            )
            with ExitStack() as contexts:
                contexts.enter_context(patch.dict(os.environ, {"YOLO_MASTER_CLI_LOG_DIR": str(log_root)}))
                contexts.enter_context(
                    patch.object(executor, "ensure_yolo_cli", return_value=(sys.executable, {"status": "available"}))
                )
                pool = contexts.enter_context(ThreadPoolExecutor(max_workers=1))
                future = pool.submit(executor.run_cli, ["-c", code])
                deadline = time.monotonic() + 5
                live = []
                while time.monotonic() < deadline:
                    live = list(log_root.glob("cli-*/stdout.log"))
                    if live and "early" in live[0].read_text():
                        break
                    time.sleep(0.02)
                self.assertTrue(live)
                self.assertIn("early", live[0].read_text())
                self.assertFalse(future.done(), "logs must be readable before the CLI exits")
                result = future.result(timeout=5)
            self.assertEqual(result["returncode"], 0)
            self.assertTrue(result["streams_are_tails"])
            self.assertLessEqual(len(result["stdout"].encode()), executor.CLI_LOG_TAIL_BYTES)
            self.assertNotIn("early", result["stdout"])
            self.assertIn("late", result["stdout"])
            self.assertIn("early", Path(result["stdout_path"]).read_text())
            self.assertIn("early-error", Path(result["stderr_path"]).read_text())
            self.assertEqual(executor.cli_logs(result)["stdout_path"], result["stdout_path"])
            self.assertEqual(executor.cli_attempt_record(result)["stderr_path"], result["stderr_path"])

    def test_default_cli_capture_is_unchanged(self):
        with ExitStack() as contexts:
            contexts.enter_context(patch.dict(os.environ, {"YOLO_MASTER_CLI_LOG_DIR": ""}))
            contexts.enter_context(
                patch.object(executor, "ensure_yolo_cli", return_value=(sys.executable, {"status": "available"}))
            )
            result = executor.run_cli(["-c", "print('normal')"])
        self.assertEqual(result["stdout"].strip(), "normal")
        self.assertNotIn("stdout_path", result)

    def test_each_cli_attempt_preserves_a_distinct_log_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExitStack() as contexts:
                contexts.enter_context(patch.dict(os.environ, {"YOLO_MASTER_CLI_LOG_DIR": directory}))
                contexts.enter_context(
                    patch.object(executor, "ensure_yolo_cli", return_value=(sys.executable, {"status": "available"}))
                )
                first = executor.run_cli(["-c", "print('first')"])
                second = executor.run_cli(["-c", "print('second')"])
            self.assertNotEqual(first["stdout_path"], second["stdout_path"])
            self.assertEqual(Path(first["stdout_path"]).read_text().strip(), "first")


class LaneVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name).resolve()
        self.root = self.project / "c_run"
        (self.root / "weights").mkdir(parents=True)
        self.request = {
            "skill": "yolo.train",
            "inputs": {
                "model": str(self.project / "c.yaml"),
                "data": str(self.project / "data.yaml"),
                "task": "detect",
            },
            "params": {
                "project": str(self.project),
                "name": "c_run",
                "epochs": 3,
                "batch": 4,
                "workers": 0,
                "imgsz": 640,
                "optimizer": "MuSGD",
                "seed": 0,
                "pretrained": False,
                "device": "0",
                "nbs": 64,
            },
            "policy": {"async": True},
        }
        self.actual = {**self.request["inputs"], **self.request["params"]}
        (self.root / "args.yaml").write_text(yaml.safe_dump(self.actual), encoding="utf-8")
        self.write_epochs([1, 2, 3])
        for name in ("best.pt", "last.pt"):
            (self.root / "weights" / name).write_bytes(b"test checkpoint")
        self.log_directory = self.project / "cli_logs"
        self.log_path = self.log_directory / "cli-test"
        self.log_path.mkdir(parents=True)
        (self.log_path / "stdout.log").write_text("training finished\n", encoding="utf-8")
        (self.log_path / "stderr.log").touch()

    def write_epochs(self, epochs):
        columns = ["epoch", "train/box_loss", *lane.REQUIRED_METRICS]
        with (self.root / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for epoch in epochs:
                writer.writerow({key: epoch if key == "epoch" else 0.2 for key in columns})

    def test_complete_budget_with_raw_logs_passes(self):
        result = lane.verify_training(self.request, self.log_directory)
        self.assertEqual(result["epochs"], 3)
        self.assertEqual(len(result["logs"]), 2)
        self.assertIn("sha256", result["checkpoints"]["last.pt"])

    def test_missing_or_duplicate_epochs_fail_even_if_last_is_correct(self):
        for epochs in ([1, 3], [1, 2, 2, 3]):
            with self.subTest(epochs=epochs):
                self.write_epochs(epochs)
                with self.assertRaisesRegex(ValueError, "epoch budget"):
                    lane.verify_training(self.request, self.log_directory)

    def test_explicit_runtime_and_input_changes_are_rejected(self):
        changes = {"workers": 4, "device": "1", "batch": 2, "nbs": 16, "data": "wrong.yaml", "model": "wrong.yaml"}
        for key, value in changes.items():
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, f"changed {key}"):
                    lane.verify_args(self.request, {**self.actual, key: value})

    def test_request_rejects_budget_gpu_data_or_directory_drift(self):
        protocol = {
            "matrix": {"c": {"path": self.request["inputs"]["model"]}},
            "common_training": {
                key: value for key, value in self.request["params"].items() if key not in {"project", "name", "device"}
            },
            "data_yaml": self.request["inputs"]["data"],
        }
        self.assertEqual(lane.validate_request(self.request, protocol, "c", "full", self.project), self.root)
        for section, key, value in (
            ("params", "workers", 4),
            ("params", "device", "1"),
            ("inputs", "data", "wrong.yaml"),
            ("params", "name", "../escape"),
        ):
            with self.subTest(key=key):
                changed = json.loads(json.dumps(self.request))
                changed[section][key] = value
                with self.assertRaises(ValueError):
                    lane.validate_request(changed, protocol, "c", "full", self.project)

    def test_oom_outside_bounded_tail_is_still_rejected(self):
        (self.log_path / "stdout.log").write_text("Reducing to batch=2\n" + "progress\n" * 20000, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "batch reduction"):
            lane.verify_training(self.request, self.log_directory)

    def test_incremental_log_scan_detects_split_oom_marker(self):
        path = self.log_path / "stdout.log"
        path.write_text("CUDA out of mem", encoding="utf-8")
        guard = lane.LogGuard(self.log_directory)
        guard.check()
        with path.open("a", encoding="utf-8") as stream:
            stream.write("ory with batch=4\n")
        with self.assertRaisesRegex(ValueError, "CUDA memory failure"):
            guard.check()

    def test_nonfinite_metrics_fail(self):
        path = self.root / "results.csv"
        path.write_text(path.read_text().replace("0.2", "nan", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            lane.verify_training(self.request, self.log_directory)

    def test_incomplete_csv_row_is_not_counted_as_an_epoch(self):
        self.write_epochs([1, 2])
        with (self.root / "results.csv").open("a", encoding="utf-8") as stream:
            stream.write("3,0.2,0.2")
        self.assertEqual(len(lane.read_epoch_rows(self.root / "results.csv")), 2)

    def test_final_jsonl_ignores_incomplete_response(self):
        path = self.project / "stdout.jsonl"
        path.write_text('startup diagnostic\n{"status":"ok"', encoding="utf-8")
        self.assertIsNone(lane.read_final_payload(path))
        path.write_text(
            "startup diagnostic\n" + json.dumps({"status": "ok", "job": {"device": "0"}}) + "\n", encoding="utf-8"
        )
        self.assertEqual(lane.read_final_payload(path)["status"], "ok")

    def test_locked_file_changes_are_rejected(self):
        path = self.project / "protocol.json"
        path.write_text("original", encoding="utf-8")
        locked = {path: lane.sha256(path)}
        lane.check_locked_files(locked)
        path.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            lane.check_locked_files(locked)

    def test_zombie_is_not_a_live_process(self):
        process = self.project / "123"
        process.mkdir()
        fields = ["S", *(["0"] * 18), "10001"]
        (process / "stat").write_text("123 (worker (test)) " + " ".join(fields), encoding="utf-8")
        self.assertEqual(lane.process_start_token(123, self.project), "10001")
        fields[0] = "Z"
        (process / "stat").write_text("123 (worker (test)) " + " ".join(fields), encoding="utf-8")
        self.assertIsNone(lane.process_start_token(123, self.project))


class TrainingIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.configs, self.project = self.base / "configs", self.base / "runs"
        self.configs.mkdir()
        self.project.mkdir()
        self.root = self.project / "c_formal"
        self.model, self.data, self.manifest = self.base / "c.yaml", self.base / "coco.yaml", self.base / "dataset.json"
        for path, content in ((self.model, "model"), (self.data, "dataset"), (self.manifest, '{"status":"ready"}')):
            path.write_text(content, encoding="utf-8")
        self.common = {
            "epochs": 3,
            "batch": 4,
            "workers": 0,
            "imgsz": 640,
            "seed": 0,
            "optimizer": "MuSGD",
            "pretrained": False,
            "nbs": 64,
            "exist_ok": True,
        }
        self.request = {
            "skill": "yolo.train",
            "policy": {"async": True},
            "inputs": {"model": str(self.model), "data": str(self.data), "task": "detect"},
            "params": {**self.common, "name": self.root.name, "project": str(self.project), "device": "0"},
        }
        self.request_path, self.protocol_path = self.configs / "c_full_request.json", self.configs / "protocol.json"
        self.protocol = {
            "run_root": str(self.project),
            "common_training": self.common,
            "matrix": {"c": {"path": str(self.model), "sha256": lane.sha256(self.model)}},
            "dataset_manifest": str(self.manifest),
            "dataset_manifest_sha256": lane.sha256(self.manifest),
            "data_yaml": str(self.data),
            "data_yaml_sha256": lane.sha256(self.data),
            "requests_sha256": {},
        }
        self.save_protocol()

    def save_protocol(self, phase="full"):
        self.request_path = self.configs / f"c_{phase}_request.json"
        lane.write_json(self.request_path, self.request)
        self.protocol["requests_sha256"][f"c_{phase}"] = lane.sha256(self.request_path)
        lane.write_json(self.protocol_path, self.protocol)
        self.protocol_hash = lane.sha256(self.protocol_path)

    def submit(self, **kwargs):
        return lane.dispatch_training(
            self.request_path,
            self.protocol_path,
            self.protocol_hash,
            cell="c",
            phase=kwargs.pop("phase", "full"),
            repo=REPO,
            environment={},
            **kwargs,
        )

    def isolated_preflight(self):
        yaml_path = self.configs / "preflight.yaml"
        yaml_path.write_text("isolated preflight", encoding="utf-8")
        lists = {}
        for split, count in (("train2017", 256), ("val2017", 128)):
            path = self.configs / f"{split}.txt"
            path.write_text(f"{split}-sample\n", encoding="utf-8")
            lists[split] = {"path": str(path), "sha256": lane.sha256(path), "images": count}
        self.protocol["preflight_data"] = {
            "data_yaml": str(yaml_path),
            "data_yaml_sha256": lane.sha256(yaml_path),
            "lists": lists,
        }
        self.request["inputs"]["data"] = str(yaml_path)
        self.request["params"].update(epochs=1, close_mosaic=0, save_period=-1)
        self.save_protocol("preflight")

    def test_identity_is_durable_and_complete_before_dispatch(self):
        events = []

        def prepared(path, identity):
            events.append("prepared")
            self.assertTrue(path.is_file())
            self.assertEqual(lane.read_json(path), identity)

        def execute(command, **kwargs):
            events.append("dispatch")
            self.assertEqual({path.name for path in self.root.iterdir()}, {"training_manifest.json"})
            identity = lane.read_json(self.root / "training_manifest.json")
            self.assertEqual(identity["protocol_sha256"], self.protocol_hash)
            self.assertEqual(identity["request_sha256"], lane.sha256(self.request_path))
            self.assertEqual(identity["model_sha256"], lane.sha256(self.model))
            self.assertEqual(identity["dataset_manifest_sha256"], lane.sha256(self.manifest))
            self.assertEqual(identity["data_yaml_sha256"], lane.sha256(self.data))
            self.assertEqual((identity["phase"], identity["cell"]), ("full", "c"))
            self.assertTrue(identity["recorded_before_dispatch"])
            self.assertEqual(identity["status"], "prepared")
            self.assertTrue(identity["started_at"].endswith("+00:00"))
            return subprocess.CompletedProcess(command, 0, '{"status":"running"}\n', "")

        with patch.object(lane.subprocess, "run", side_effect=execute) as dispatch:
            _, identity_path, identity = self.submit(on_prepared=prepared)
        self.assertEqual(events, ["prepared", "dispatch"])
        dispatch.assert_called_once()
        self.assertEqual(lane.read_json(identity_path), identity)

    def test_existing_evidence_is_not_adopted_or_overwritten(self):
        self.root.mkdir()
        identity_path = self.root / "training_manifest.json"
        identity_path.write_text("historical evidence", encoding="utf-8")
        (self.root / "results.csv").write_text("old results", encoding="utf-8")
        with patch.object(lane.subprocess, "run") as dispatch:
            with self.assertRaises(FileExistsError):
                self.submit()
        dispatch.assert_not_called()
        self.assertEqual(identity_path.read_text(), "historical evidence")
        self.assertEqual((self.root / "results.csv").read_text(), "old results")

    def test_even_an_existing_empty_directory_is_not_adopted(self):
        self.root.mkdir()
        with patch.object(lane.subprocess, "run") as dispatch:
            with self.assertRaises(FileExistsError):
                self.submit()
        dispatch.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_extra_file_before_dispatch_blocks_submission(self):
        def intrusion(path, identity):
            (path.parent / "results.csv").write_text("unexpected", encoding="utf-8")

        with patch.object(lane.subprocess, "run") as dispatch:
            with self.assertRaisesRegex(ValueError, "unexpected files"):
                self.submit(on_prepared=intrusion)
        dispatch.assert_not_called()

    def test_changed_identity_before_dispatch_blocks_submission(self):
        def tamper(path, identity):
            path.write_text('{"cell":"d"}', encoding="utf-8")

        with patch.object(lane.subprocess, "run") as dispatch:
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                self.submit(on_prepared=tamper)
        dispatch.assert_not_called()

    def test_request_tampering_blocks_before_reserving_a_run(self):
        with self.request_path.open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with patch.object(lane.subprocess, "run") as dispatch:
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                self.submit()
        dispatch.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_exist_ok_must_allow_only_the_newly_reserved_directory(self):
        self.request["params"]["exist_ok"] = self.common["exist_ok"] = False
        self.save_protocol()
        with patch.object(lane.subprocess, "run") as dispatch:
            with self.assertRaisesRegex(ValueError, "exist_ok=True"):
                self.submit()
        dispatch.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_preflight_manifest_identifies_its_actual_isolated_data(self):
        self.isolated_preflight()
        with patch.object(lane.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            _, _, identity = self.submit(phase="preflight")
        self.assertEqual(identity["data_yaml_sha256"], self.protocol["preflight_data"]["data_yaml_sha256"])
        self.assertNotEqual(identity["data_yaml_sha256"], identity["formal_data_yaml_sha256"])
        self.assertEqual(identity["dataset_lists"], self.protocol["preflight_data"]["lists"])

    def test_preflight_sample_list_tampering_is_locked(self):
        self.isolated_preflight()
        path = Path(self.protocol["preflight_data"]["lists"]["train2017"]["path"])
        path.write_text("changed sample\n", encoding="utf-8")
        with patch.object(lane.subprocess, "run") as dispatch:
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                self.submit(phase="preflight")
        dispatch.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_preflight_request_cannot_point_back_to_formal_data(self):
        self.isolated_preflight()
        self.request["inputs"]["data"] = str(self.data)
        self.save_protocol("preflight")
        with patch.object(lane.subprocess, "run") as dispatch:
            with self.assertRaisesRegex(ValueError, "isolated preflight dataset"):
                self.submit(phase="preflight")
        dispatch.assert_not_called()
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
