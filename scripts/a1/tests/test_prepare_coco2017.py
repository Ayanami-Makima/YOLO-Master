"""Regression checks for the COCO preparation failures found during A1 P1."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "prepare_coco2017.py"
SPEC = importlib.util.spec_from_file_location("prepare_coco2017", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PreparationTests(unittest.TestCase):
    def test_only_instances_reach_converter_and_output_does_not_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations"
            annotations.mkdir()
            for name in ("instances_train2017", "instances_val2017", "captions_train2017", "person_keypoints_val2017"):
                (annotations / f"{name}.json").write_text("{}", encoding="utf-8")
            stage = root / "stage"
            stage.mkdir()

            def converter(**kwargs):
                self.assertEqual(
                    {path.name for path in Path(kwargs["labels_dir"]).glob("*.json")},
                    {"instances_train2017.json", "instances_val2017.json"},
                )
                self.assertFalse(Path(kwargs["save_dir"]).exists())
                self.assertTrue(kwargs["cls91to80"])

            MODULE.convert_instances(root, stage, converter)

    def test_existing_different_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            MODULE.write_text_checked(path, "original")
            MODULE.write_text_checked(path, "original")
            with self.assertRaises(ValueError):
                MODULE.write_text_checked(path, "replacement")
            self.assertEqual(path.read_text(), "original")

    def test_invalid_boxes_and_categories_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            for line in ("80 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2", "0 0.5 0.5 0 0.2"):
                path.write_text(line, encoding="utf-8")
                with self.assertRaises(ValueError):
                    MODULE.validate_label(path)
            path.write_text("0 0.5 0.5 0.2 0.2\n79 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            self.assertEqual(dict(MODULE.validate_label(path)), {0: 1, 79: 1})


if __name__ == "__main__":
    unittest.main()
