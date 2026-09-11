"""Exercise formal-run gates and measurement math without loading a YOLO model."""

import contextlib
import copy
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_p1_matrix.py"
SPEC = importlib.util.spec_from_file_location("evaluate_p1_matrix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FormalGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.configs = self.root / "configs"
        self.project = self.root / "runs"
        self.configs.mkdir()
        self.project.mkdir()
        self.data = self.root / "coco.yaml"
        self.data.write_text("path: coco\ntrain: train.txt\nval: val.txt\n", encoding="utf-8")
        self.manifest = self.root / "dataset_manifest.json"
        MODULE.write_json(self.manifest, {"status": "ready"})
        self.common = {
            "epochs": 3,
            "imgsz": 640,
            "batch": 4,
            "seed": 0,
            "optimizer": "MuSGD",
            "pretrained": False,
            "fraction": 1.0,
            "workers": 0,
        }
        self.protocol = {
            "locked_sha": MODULE.LOCKED_SHA,
            "run_root": str(self.project),
            "core_diff_from_lock": "empty",
            "dataset_manifest": str(self.manifest),
            "dataset_manifest_sha256": MODULE.digest(self.manifest),
            "data_yaml": str(self.data),
            "data_yaml_sha256": MODULE.digest(self.data),
            "common_training": self.common,
            "matrix": {},
            "requests_sha256": {},
        }
        self.requests = {}
        self.log_paths = {}
        for cell in MODULE.CELLS:
            model = self.configs / f"{cell}_matched.yaml"
            model.write_text(f"end2end: {str(cell in 'bd').lower()}\n", encoding="utf-8")
            self.protocol["matrix"][cell] = {
                "path": str(model),
                "sha256": MODULE.digest(model),
                "moe": cell in "cd",
                "end2end": cell in "bd",
            }
            request = {
                "skill": "yolo.train",
                "inputs": {"model": str(model), "data": str(self.data)},
                "params": {
                    **self.common,
                    "name": f"{cell}_matched_seed0_3ep",
                    "project": str(self.project),
                    "device": "0" if cell in "ac" else "1",
                },
            }
            self.requests[cell] = request
            request_path = self.configs / f"{cell}_full_request.json"
            MODULE.write_json(request_path, request)
            self.protocol["requests_sha256"][f"{cell}_full"] = MODULE.digest(request_path)
            run_dir = self.project / request["params"]["name"]
            (run_dir / "weights").mkdir(parents=True)
            (run_dir / "args.yaml").write_text(
                yaml.safe_dump({**request["params"], **request["inputs"]}), encoding="utf-8"
            )
            (run_dir / "results.csv").write_text(
                "epoch,train/box_loss,metrics/mAP50-95(B)\n1,2.5,0.1\n2,2.0,0.15\n3,1.5,0.2\n", encoding="utf-8"
            )
            # Opaque fixture bytes only test the artifact gate, never model loading.
            for filename in ("best.pt", "last.pt"):
                (run_dir / "weights" / filename).write_bytes(f"fixture-{cell}-{filename}".encode())
            log_dir = self.project / "cli_logs" / f"{cell}_full" / "cli-fixture"
            log_dir.mkdir(parents=True)
            self.log_paths[cell] = []
            for filename in ("stdout.log", "stderr.log"):
                path = log_dir / filename
                path.write_text(f"verified {cell} {filename}\n", encoding="utf-8")
                self.log_paths[cell].append(path)
        self.protocol_path = self.configs / "protocol.json"
        MODULE.write_json(self.protocol_path, self.protocol)
        self.protocol_hash = MODULE.digest(self.protocol_path)
        for cell in MODULE.CELLS:
            MODULE.write_json(self.run_dir(cell) / "training_manifest.json", self.training_manifest(cell))
        self.write_lane_states()

    def run_dir(self, cell):
        return self.project / self.requests[cell]["params"]["name"]

    def request_path(self, cell):
        return self.configs / f"{cell}_full_request.json"

    def training_manifest(self, cell):
        request = self.requests[cell]
        return {
            "schema_version": 1,
            "status": "prepared",
            "recorded_before_dispatch": True,
            "phase": "full",
            "cell": cell,
            "lane": "0" if cell in "ac" else "1",
            "run_dir": str(self.run_dir(cell)),
            "protocol_path": str(self.protocol_path),
            "protocol_sha256": self.protocol_hash,
            "request_path": str(self.request_path(cell)),
            "request_sha256": self.protocol["requests_sha256"][f"{cell}_full"],
            "model": request["inputs"]["model"],
            "model_sha256": self.protocol["matrix"][cell]["sha256"],
            "dataset_manifest": str(self.manifest),
            "dataset_manifest_sha256": self.protocol["dataset_manifest_sha256"],
            "data_yaml": str(self.data),
            "data_yaml_sha256": self.protocol["data_yaml_sha256"],
            "formal_data_yaml": str(self.data),
            "formal_data_yaml_sha256": self.protocol["data_yaml_sha256"],
            "training": request["params"],
        }

    def runner_record(self, cell):
        run_dir = self.run_dir(cell)
        identity = run_dir / "training_manifest.json"
        checkpoints = {}
        for filename in ("best.pt", "last.pt"):
            path = run_dir / "weights" / filename
            checkpoints[filename] = {
                "path": str(path),
                "sha256": MODULE.digest(path),
                "bytes": path.stat().st_size,
            }
        logs = [
            {"path": str(path), "sha256": MODULE.digest(path), "bytes": path.stat().st_size}
            for path in self.log_paths[cell]
        ]
        return {
            "cell": cell,
            "training_manifest": str(identity),
            "training_manifest_sha256": MODULE.digest(identity),
            "request_sha256": self.protocol["requests_sha256"][f"{cell}_full"],
            "run_dir": str(run_dir),
            "epochs": self.common["epochs"],
            "last_epoch_metrics": {
                "epoch": "3",
                "train/box_loss": "1.5",
                "metrics/mAP50-95(B)": "0.2",
            },
            "checkpoints": checkpoints,
            "logs": logs,
        }

    def lane_path(self, lane):
        return self.project / f"full_lane{lane}_state.json"

    def write_lane_states(self):
        for lane, cells in MODULE.LANE_CELLS.items():
            MODULE.write_json(
                self.lane_path(lane),
                {
                    "phase": "full",
                    "lane": lane,
                    "status": "completed",
                    "protocol_sha256": self.protocol_hash,
                    "completed": [self.runner_record(cell) for cell in cells],
                    "child_alive": False,
                    "launch_state": "verified",
                },
            )

    def test_complete_four_cell_gate_binds_checkpoints_and_protocol(self):
        gate = MODULE.check_matrix(self.configs, self.project)
        self.assertEqual(gate["status"], "ready", gate["blockers"])
        self.assertEqual(set(gate["cells"]), set(MODULE.CELLS))
        self.assertEqual(gate["identity"]["protocol_sha256"], MODULE.digest(self.configs / "protocol.json"))
        self.assertEqual(set(gate["identity"]["lanes"]), {"0", "1"})
        self.assertEqual(gate["cells"]["d"]["runner"]["lane"], "1")
        self.assertIn("not its validation", gate["cells"]["a"]["checkpoint_selection"])

    def test_last_epoch_alone_does_not_prove_complete_training(self):
        path = self.run_dir("d") / "results.csv"
        path.write_text("epoch,train/box_loss,metrics/mAP50-95(B)\n3,1.5,0.2\n", encoding="utf-8")
        gate = MODULE.check_matrix(self.configs, self.project)
        self.assertEqual(gate["status"], "blocked")
        self.assertTrue(any("noncontiguous" in reason for reason in gate["blockers"]))

    def test_checksum_drift_blocks_before_any_runtime_is_imported(self):
        with (self.configs / "b_full_request.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
        gate = MODULE.check_matrix(self.configs, self.project)
        self.assertEqual(gate["status"], "blocked")
        self.assertTrue(any("request SHA-256 mismatch" in reason for reason in gate["blockers"]))

    def test_model_config_and_dataset_tampering_are_rejected(self):
        for path in (self.configs / "a_matched.yaml", self.data, self.manifest):
            with self.subTest(path=path.name):
                before = path.read_bytes()
                path.write_bytes(before + b" ")
                self.assertEqual(MODULE.check_matrix(self.configs, self.project)["status"], "blocked")
                path.write_bytes(before)

    def test_unbound_training_identity_is_rejected(self):
        identity = self.training_manifest("c")
        identity["protocol_sha256"] = "wrong"
        MODULE.write_json(self.run_dir("c") / "training_manifest.json", identity)
        gate = MODULE.check_matrix(self.configs, self.project)
        self.assertEqual(gate["status"], "blocked")
        self.assertTrue(any("another protocol" in reason for reason in gate["blockers"]))

    def test_missing_failed_or_forged_lane_state_is_rejected(self):
        path = self.lane_path("1")
        original = MODULE.read_json(path)
        cases = {
            "missing": None,
            "failed": {**original, "status": "failed"},
            "wrong_protocol": {**original, "protocol_sha256": "forged"},
            "forged_cells": {**original, "completed": original["completed"][:1]},
        }
        for name, state in cases.items():
            with self.subTest(name=name):
                if state is None:
                    path.unlink()
                else:
                    MODULE.write_json(path, state)
                gate = MODULE.check_matrix(self.configs, self.project)
                self.assertEqual(gate["status"], "blocked")
                MODULE.write_json(path, original)

    def test_training_manifest_full_cell_run_model_and_data_identity_are_bound(self):
        path = self.run_dir("a") / "training_manifest.json"
        original = MODULE.read_json(path)
        cases = {
            "phase": ("phase", "preflight", "not formal full training"),
            "cell": ("cell", "b", "cell mismatch"),
            "run_dir": ("run_dir", str(self.root / "forged-run"), "run directory mismatch"),
            "model": ("model", str(self.root / "forged-model.yaml"), "model mismatch"),
            "data": ("data_yaml", str(self.root / "forged-data.yaml"), "data mismatch"),
        }
        for name, (key, value, message) in cases.items():
            with self.subTest(name=name):
                forged = copy.deepcopy(original)
                forged[key] = value
                MODULE.write_json(path, forged)
                gate = MODULE.check_matrix(self.configs, self.project)
                self.assertEqual(gate["status"], "blocked")
                self.assertTrue(any(message in reason for reason in gate["blockers"]), gate["blockers"])
                MODULE.write_json(path, original)

    def test_runner_manifest_checkpoint_and_log_hashes_are_bound(self):
        path = self.lane_path("0")
        original = MODULE.read_json(path)

        def manifest_hash(state):
            state["completed"][0]["training_manifest_sha256"] = "0" * 64

        def checkpoint_hash(state):
            state["completed"][0]["checkpoints"]["best.pt"]["sha256"] = "1" * 64

        def log_hash(state):
            state["completed"][0]["logs"][0]["sha256"] = "2" * 64

        for name, mutate, message in (
            ("manifest", manifest_hash, "training manifest hash mismatch"),
            ("checkpoint", checkpoint_hash, "best.pt hash mismatch"),
            ("log", log_hash, "runner log SHA-256 mismatch"),
        ):
            with self.subTest(name=name):
                forged = copy.deepcopy(original)
                mutate(forged)
                MODULE.write_json(path, forged)
                gate = MODULE.check_matrix(self.configs, self.project)
                self.assertEqual(gate["status"], "blocked")
                self.assertTrue(any(message in reason for reason in gate["blockers"]), gate["blockers"])
                MODULE.write_json(path, original)

    def test_one_epoch_cannot_be_promoted_from_preflight(self):
        self.protocol["common_training"]["epochs"] = 1
        MODULE.write_json(self.configs / "protocol.json", self.protocol)
        gate = MODULE.check_matrix(self.configs, self.project)
        self.assertEqual(gate["status"], "blocked")
        self.assertTrue(any("preflight" in reason for reason in gate["blockers"]))

    def test_optional_full_evaluation_yaml_is_checksum_locked(self):
        evaluation = self.root / "full_coco.yaml"
        evaluation.write_text("path: full-coco\nval: val2017.txt\n", encoding="utf-8")
        self.protocol["evaluation_data_yaml"] = str(evaluation)
        self.protocol["evaluation_data_yaml_sha256"] = MODULE.digest(evaluation)
        MODULE.write_json(self.configs / "protocol.json", self.protocol)
        evaluation.write_text("path: changed\nval: val2017.txt\n", encoding="utf-8")
        gate = MODULE.check_matrix(self.configs, self.project)
        self.assertEqual(gate["status"], "blocked")
        self.assertTrue(any("evaluation data YAML" in reason for reason in gate["blockers"]))

    def test_missing_checkpoint_main_returns_blocked_without_running_stages(self):
        (self.run_dir("d") / "weights" / "best.pt").unlink()
        output = self.root / "evidence"
        with contextlib.redirect_stdout(io.StringIO()):
            code = MODULE.main(
                [
                    "--configs",
                    str(self.configs),
                    "--project",
                    str(self.project),
                    "--output",
                    str(output),
                    "--stage",
                    "all",
                ]
            )
        self.assertEqual(code, 2)
        report = MODULE.read_json(output / "summary.json")
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["stages"]["validate"]["status"], "not_run")
        self.assertEqual(report["acceptance"], "not_determined")

    def test_summary_rejects_stale_checkpoints_and_never_infers_best_map_from_csv(self):
        gate = MODULE.check_matrix(self.configs, self.project)
        output = self.root / "evidence"
        clean = MODULE.summarize(gate, output)
        self.assertNotIn("accuracy_AP_points", clean)
        MODULE.write_json(
            output / "validate.json",
            {
                "identity": {"protocol_sha256": "stale"},
                "status": "completed",
                "cells": {cell: {"map50_95": 0.5} for cell in MODULE.CELLS},
            },
        )
        report = MODULE.summarize(gate, output)
        self.assertEqual(report["stages"]["validate"]["status"], "blocked")
        self.assertNotIn("accuracy_AP_points", report)

    def test_completed_stage_without_its_required_payload_is_rejected(self):
        gate = MODULE.check_matrix(self.configs, self.project)
        for stage in MODULE.STAGES:
            with self.subTest(stage=stage):
                output = self.root / f"malformed_{stage}"
                MODULE.write_json(
                    output / f"{stage}.json",
                    {
                        "identity": gate["identity"],
                        "status": "completed",
                        "cells": {cell: {"status": "completed"} for cell in MODULE.CELLS},
                    },
                )
                report = MODULE.summarize(gate, output)
                self.assertEqual(report["stages"][stage]["status"], "blocked")
                self.assertIn(stage, report["stages"][stage]["error"])

    def test_incomplete_summary_returns_nonzero(self):
        output = self.root / "incomplete_evidence"
        with contextlib.redirect_stdout(io.StringIO()):
            code = MODULE.main(
                [
                    "--configs",
                    str(self.configs),
                    "--project",
                    str(self.project),
                    "--output",
                    str(output),
                    "--stage",
                    "summary",
                ]
            )
        self.assertEqual(code, 1)
        report = MODULE.read_json(output / "summary.json")
        self.assertEqual(report["status"], "incomplete")
        self.assertTrue(all(item["status"] == "not_run" for item in report["stages"].values()))


class DiagnosticsTests(unittest.TestCase):
    def test_native_validation_table_fallback_is_strict_and_supports_scientific_notation(self):
        stdout = """\n Class Images Instances Box(P R mAP50 mAP50-95)\n all 5000 36335 0.000808 0.0194 0.000148 3.6e-05\n"""
        result = MODULE.parse_native_validation_table(stdout)
        self.assertEqual(result["images"], 5000)
        self.assertEqual(result["instances"], 36335)
        self.assertEqual(result["evaluation"]["map50_95"], 3.6e-05)
        with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
            MODULE.parse_native_validation_table(stdout + stdout)
        with self.assertRaisesRegex(ValueError, "invalid native validation metrics"):
            MODULE.parse_native_validation_table("all 5000 36335 0.1 0.2 0.3 1.2\n")

    def test_overlap_is_same_class_confidence_filtered_and_counts_each_lower_box_once(self):
        boxes = [
            [0, 0, 10, 10, 0.9, 0],
            [0, 0, 10, 10, 0.8, 0],
            [0, 0, 10, 10, 0.7, 0],
            [0, 0, 10, 10, 0.9, 1],
            [0, 0, 10, 10, 0.1, 0],
            [30, 30, 40, 40, 0.9, 0],
        ]
        counts = MODULE.duplicate_boxes(boxes, confidence=0.25, iou_threshold=0.7)
        self.assertEqual(counts["boxes"], 5)
        self.assertEqual(counts["duplicate_pairs"], 3)
        self.assertEqual(counts["duplicate_boxes"], 2)
        self.assertAlmostEqual(counts["duplicate_box_rate"], 0.4)
        self.assertIsNone(MODULE.duplicate_boxes([], 0.25, 0.7)["duplicate_box_rate"])
        with self.assertRaises(ValueError):
            MODULE.duplicate_boxes([[0, 0, 1, 1, float("nan"), 0]], 0.25, 0.7)

    def test_percentiles_and_factorial_signs_are_not_mixed(self):
        stats = MODULE.sample_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(stats["n"], 5)
        self.assertAlmostEqual(stats["mean_ms"], 3.0)
        self.assertAlmostEqual(stats["p90_ms"], 4.6)
        self.assertAlmostEqual(stats["p99_ms"], 4.96)
        contrast = MODULE.factorial_contrasts({"a": 10.0, "b": 8.0, "c": 20.0, "d": 19.0})
        self.assertEqual(contrast["B_minus_A"], -2.0)
        self.assertEqual(contrast["D_minus_C"], -1.0)
        self.assertEqual(contrast["interaction_D_minus_C_minus_B_minus_A"], 1.0)

    def fake_nms(self):
        class Kernels:
            @staticmethod
            def nms(*args, **kwargs):
                return [0]

        def non_max_suppression(prediction, end2end=False):
            if prediction.shape[-1] == 6 or end2end:
                return ["score_filter"]
            return Kernels.nms()

        return SimpleNamespace(TorchNMS=Kernels, non_max_suppression=non_max_suppression)

    def test_shared_dispatch_is_not_counted_as_real_nms_for_end2end(self):
        module = self.fake_nms()
        original = module.non_max_suppression
        with MODULE.NMSCallMonitor(module, SimpleNamespace()) as monitor:
            module.non_max_suppression(SimpleNamespace(shape=(1, 300, 6)), end2end=True)
            report = monitor.report(end2end=True, detections=1)
        self.assertEqual(report["wrapper_calls"], 1)
        self.assertEqual(report["suppression_kernel_calls"], 0)
        self.assertEqual(report["status"], "passed")
        self.assertIs(module.non_max_suppression, original)

    def test_real_kernel_call_is_observed_and_empty_nms_on_is_inconclusive(self):
        module = self.fake_nms()
        with MODULE.NMSCallMonitor(module, SimpleNamespace()) as monitor:
            module.non_max_suppression(SimpleNamespace(shape=(1, 84, 8400)))
            report = monitor.report(end2end=False, detections=1)
        self.assertEqual(report["suppression_kernel_calls"], 1)
        self.assertEqual(report["status"], "passed")
        with MODULE.NMSCallMonitor(module, SimpleNamespace()) as empty:
            empty.events.append({"raw_shape": [1, 84, 8400], "route": "nms"})
            self.assertTrue(empty.report(False, 0)["status"].startswith("inconclusive"))

    def test_hooks_are_restored_if_prediction_raises(self):
        module = self.fake_nms()
        original = module.TorchNMS.nms
        with self.assertRaisesRegex(RuntimeError, "prediction error"):
            with MODULE.NMSCallMonitor(module, SimpleNamespace()):
                raise RuntimeError("prediction error")
        self.assertIs(module.TorchNMS.nms, original)

    def test_relative_image_lists_resolve_against_list_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"path fixture only")
            listing = root / "val.txt"
            listing.write_text("./image.jpg\n./image.jpg\n", encoding="utf-8")
            self.assertEqual(MODULE.resolve_images(root / "unused.yaml", listing, 16), [image.resolve()])


class ExportEvidenceTests(unittest.TestCase):
    def preflight(self, moe):
        return {
            "format": "onnx",
            "supported": True,
            "errors": [],
            "decisions": [
                {
                    "module": "model.4.m.0.0.mlp",
                    "module_type": "OptimizedMOEImproved",
                    "module_family": "MoE",
                    "backend": "onnx",
                    "supported": True,
                    "strategy": "dense_fallback",
                    "dense_fallback": True,
                }
            ]
            if moe
            else [],
        }

    def value_info(self, name, shape):
        return SimpleNamespace(
            name=name,
            type=SimpleNamespace(
                tensor_type=SimpleNamespace(
                    elem_type=1,
                    shape=SimpleNamespace(dim=[SimpleNamespace(dim_value=value, dim_param="") for value in shape]),
                )
            ),
        )

    def model_proto(self, cell):
        preflight = self.preflight(cell in "cd")
        metadata = {"end2end": cell in "bd", "batch": 1, "imgsz": [640, 640], "args": {"nms": False}}
        if cell in "cd":
            metadata["mixture_export_preflight"] = copy.deepcopy(preflight)
        graph = SimpleNamespace(
            node=[],
            initializer=[],
            input=[self.value_info("images", [1, 3, 640, 640])],
            output=[self.value_info("output0", [1, 300, 6] if cell in "bd" else [1, 84, 8400])],
        )
        model = SimpleNamespace(
            graph=graph,
            functions=[],
            metadata_props=[SimpleNamespace(key=key, value=repr(value)) for key, value in metadata.items()],
        )
        return model, preflight

    def test_dense_fallback_is_declared_without_claiming_sparse_preservation(self):
        result = MODULE.export_semantics(self.preflight(True), True)
        self.assertIs(result["dense_fallback"], True)
        self.assertIs(result["sparse_dispatch_preserved"], False)
        self.assertEqual(result["output_equivalence"], "not_yet_measured")
        unsupported = self.preflight(True)
        unsupported.update(supported=False, errors=["test unsupported route"])
        with self.assertRaises(MODULE.ExportBlocked):
            MODULE.export_semantics(unsupported, True)

    def test_nms_inside_subgraph_and_function_body_is_detected(self):
        model, preflight = self.model_proto("b")
        hidden_nms = SimpleNamespace(name="hidden", op_type="NonMaxSuppression", domain="", attribute=[])
        subgraph = SimpleNamespace(node=[hidden_nms])
        model.graph.node = [
            SimpleNamespace(
                name="branch",
                op_type="If",
                domain="",
                attribute=[SimpleNamespace(name="then_branch", type=5, g=subgraph)],
            )
        ]
        model.functions = [SimpleNamespace(name="local", domain="test", node=[hidden_nms])]
        result = MODULE.inspect_onnx_graph(model, preflight, cell="b", imgsz=640, max_det=300)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["NonMaxSuppression_count"], 2)
        self.assertEqual(len(result["nms_nodes"]), 2)

    def test_raw_dense_export_without_nms_is_not_called_nms_free(self):
        model, preflight = self.model_proto("a")
        result = MODULE.inspect_onnx_graph(model, preflight, cell="a", imgsz=640, max_det=300)
        self.assertEqual(result["status"], "passed")
        self.assertIs(result["end2end"], False)
        self.assertIn("external NMS is still required", result["output_semantics"])

    def test_head_downgrade_or_missing_nms_declaration_is_blocked(self):
        for key, value in (("end2end", "False"), ("args", "{}"), ("batch", "2")):
            with self.subTest(key=key):
                model, preflight = self.model_proto("d")
                next(entry for entry in model.metadata_props if entry.key == key).value = value
                with self.assertRaises(MODULE.ExportBlocked):
                    MODULE.inspect_onnx_graph(model, preflight, cell="d", imgsz=640, max_det=300)

    def test_moe_metadata_must_agree_with_source_preflight(self):
        model, preflight = self.model_proto("c")
        embedded = copy.deepcopy(preflight)
        embedded["decisions"][0].update(strategy="dynamic", dense_fallback=False)
        entry = next(item for item in model.metadata_props if item.key == "mixture_export_preflight")
        entry.value = repr(embedded)
        with self.assertRaisesRegex(MODULE.ExportBlocked, "differs from native preflight"):
            MODULE.inspect_onnx_graph(model, preflight, cell="c", imgsz=640, max_det=300)

    def test_metadata_literals_are_not_evaluated_as_code_and_duplicates_are_rejected(self):
        expression = "__import__('builtins').globals()"
        entry = SimpleNamespace(key="untrusted", value=expression)
        self.assertEqual(MODULE.parse_onnx_metadata([entry])["untrusted"], expression)
        with self.assertRaisesRegex(MODULE.ExportBlocked, "duplicate"):
            MODULE.parse_onnx_metadata([entry, entry])

    def test_e2e_row_permutation_requires_complete_class_preserving_matching(self):
        import numpy as np

        reference = np.array([[[0, 0, 10, 10, 0.8, 0], [20, 20, 30, 30, 0.8, 1]]], dtype=np.float32)
        reordered = reference[:, ::-1].copy()
        result = MODULE.compare_export_outputs(reference, reordered, end2end=True, atol=1e-4, rtol=1e-3)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["permutation"], [1, 0])
        reordered[0, 0, 5] = 2
        self.assertEqual(
            MODULE.compare_export_outputs(
                reference,
                reordered,
                end2end=True,
                atol=1e-4,
                rtol=1e-3,
            )["status"],
            "failed",
        )

    def test_duplicate_candidate_cannot_satisfy_two_distinct_reference_rows(self):
        import numpy as np

        reference = np.array([[[0, 0, 10, 10, 0.8, 0], [20, 20, 30, 30, 0.8, 0]]], dtype=np.float32)
        duplicated = np.repeat(reference[:, :1], 2, axis=1)
        result = MODULE.compare_export_outputs(reference, duplicated, end2end=True, atol=1e-4, rtol=1e-3)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["matched_rows"], 1)

    def test_nonfinite_or_shape_mismatched_outputs_do_not_pass(self):
        import numpy as np

        reference = np.zeros((1, 84, 4), dtype=np.float32)
        for candidate in (np.full_like(reference, np.nan), reference[:, :, :3], np.ones_like(reference)):
            with self.subTest(shape=candidate.shape):
                self.assertEqual(
                    MODULE.compare_export_outputs(
                        reference,
                        candidate,
                        end2end=False,
                        atol=1e-4,
                        rtol=1e-3,
                    )["status"],
                    "failed",
                )

    def test_export_request_preserves_each_head_and_uses_cpu_without_pruning(self):
        args = SimpleNamespace(imgsz=640, max_det=300)
        for cell in MODULE.CELLS:
            request = MODULE.export_request(Path("isolated/model.pt"), Path("isolated"), cell, args)
            self.assertEqual(request["skill"], "yolo.export")
            self.assertEqual(request["params"]["device"], "cpu")
            self.assertIs(request["params"]["end2end"], cell in "bd")
            self.assertIs(request["params"]["nms"], False)
            self.assertIs(request["params"]["pre_export_prune"], False)

    def test_export_stage_requests_cpu_only_runtime_and_keeps_refusal_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = {
                "status": "ready",
                "identity": {"protocol_sha256": "fixture"},
                "protocol": {"common_training": {"imgsz": 640}, "data_yaml": "unused.yaml"},
            }
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(MODULE, "check_matrix", return_value=gate))
                runtime = stack.enter_context(patch.object(MODULE, "prepare_runtime", return_value={}))
                stack.enter_context(patch.object(MODULE, "resolve_images", return_value=[]))
                export = stack.enter_context(
                    patch.object(
                        MODULE,
                        "run_export",
                        return_value={
                            "status": "blocked",
                            "error": "fixture unsupported route",
                        },
                    )
                )
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                code = MODULE.main(
                    [
                        "--configs",
                        str(root / "configs"),
                        "--project",
                        str(root / "project"),
                        "--output",
                        str(root / "evidence"),
                        "--stage",
                        "export",
                    ]
                )
            self.assertEqual(code, 1)
            runtime.assert_called_once_with(4, None, inspect_cuda=False)
            export.assert_called_once()
            report = MODULE.read_json(root / "evidence" / "summary.json")
            self.assertEqual(report["stages"]["export"]["status"], "blocked")
            self.assertEqual(report["acceptance"], "not_determined")


if __name__ == "__main__":
    unittest.main()
