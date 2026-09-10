"""Test deterministic P1 pretrained protocol helpers without training."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_p1_pretrained import SEED, class_aware_sample, matched_config, request


class PretrainedProtocolTests(unittest.TestCase):
    def test_class_aware_sample_is_deterministic_and_covers_every_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            listing = []
            for index in range(100):
                name = f"{index:012d}"
                listing.append(f"./images/train2017/{name}.jpg")
                label = root / "labels" / "train2017" / f"{name}.txt"
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text(f"{index % 80} 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            (root / "train2017.txt").write_text("\n".join(listing) + "\n", encoding="utf-8")
            first, stats = class_aware_sample(root, "train2017", 90, SEED)
            second, _ = class_aware_sample(root, "train2017", 90, SEED)
            self.assertEqual(first, second)
            self.assertEqual(stats["images"], 90)
            self.assertEqual(stats["classes"], 80)
            self.assertTrue(all(stats["class_box_counts"][str(index)] > 0 for index in range(80)))

    def test_only_the_two_factors_change(self):
        parent = {
            "end2end": True,
            "backbone": [[-1, 1, "A2C2fMoE", [1, 2, 3, 4, 5, 6, 7, 8, 4, 1]] for _ in range(3)],
            "head": [[-1, 1, "Detect", [80]]],
        }
        original = copy.deepcopy(parent)
        dense = matched_config(parent, False, False)
        sparse = matched_config(parent, True, True)
        self.assertEqual(parent, original)
        self.assertEqual([layer[2] for layer in dense["backbone"]], ["A2C2f"] * 3)
        self.assertEqual([layer[3] for layer in dense["backbone"]], [[1, 2, 3, 4, 5, 6, 7, 8]] * 3)
        self.assertEqual([layer[2] for layer in sparse["backbone"]], ["A2C2fMoE"] * 3)
        self.assertIs(dense["end2end"], False)
        self.assertIs(sparse["end2end"], True)

    def test_request_keeps_explicit_checkpoint_in_common_params(self):
        common = {"epochs": 5, "pretrained": "/locked/yolo26n.pt", "close_mosaic": 1, "save_period": 1, "workers": 0}
        value = request(Path("/repo"), Path("/runs"), Path("/model.yaml"), Path("/data.yaml"), "d", "preflight", common)
        self.assertEqual(value["params"]["pretrained"], "/locked/yolo26n.pt")
        self.assertEqual(value["params"]["epochs"], 1)
        self.assertEqual(value["params"]["device"], "1")
        self.assertTrue(value["policy"]["async"])


if __name__ == "__main__":
    unittest.main()
