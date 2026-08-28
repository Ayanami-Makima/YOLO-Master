"""Test P1 routing evidence aggregation."""

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "audit_p1_routing.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("audit_p1_routing", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RoutingSummaryTests(unittest.TestCase):
    def test_counts_report_dead_experts_and_normalized_entropy(self):
        result = MODULE.summarize_counts(Counter({0: 2, 1: 2}), num_experts=4, top_k=2)
        self.assertEqual(result["selections"], 4)
        self.assertEqual(result["fractions"], [0.5, 0.5, 0.0, 0.0])
        self.assertEqual(result["dead_experts_on_sample"], [2, 3])
        self.assertAlmostEqual(result["normalized_entropy"], 0.5)

    def test_empty_counts_remain_explicit(self):
        result = MODULE.summarize_counts(Counter(), num_experts=4, top_k=2)
        self.assertEqual(result["selections"], 0)
        self.assertEqual(result["normalized_entropy"], 0.0)
        self.assertEqual(result["dead_experts_on_sample"], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
