"""Regression tests for the immutable E9 scoring consumer."""
from __future__ import annotations

import unittest

from analyze_e9 import five_gate_summary


def _row(output: float | None, x: float = 0.0, raw: float = 0.5, oracle: float = 1.0) -> dict:
    if output is None:
        aggregation = {"return_kind": "no_return", "output": None}
    else:
        aggregation = {"return_kind": "unique_largest", "output": output}
    return {
        "aggregation": aggregation,
        "baseline_estimates": {"X": x, "raw_XW": raw, "oracle": oracle},
    }


class E9ScoringTests(unittest.TestCase):
    def test_return_rate_is_unconditional(self) -> None:
        result = five_gate_summary({"rows": [_row(1.0), _row(None)]})
        self.assertEqual(result["metrics"]["replicates"], 2)
        self.assertEqual(result["metrics"]["unique_returns"], 1)
        self.assertEqual(result["metrics"]["return_rate"], 0.5)
        self.assertFalse(result["checks"]["return_rate"])
        self.assertFalse(result["all_frozen_empirical_benchmark_gates_pass"])

    def test_all_six_checks_are_required(self) -> None:
        payload = {"rows": [_row(1.0 + offset) for offset in ([-0.01, 0.01] * 10)]}
        result = five_gate_summary(payload)
        self.assertEqual(
            set(result["checks"]),
            {"mae", "p90", "return_rate", "beats_X", "beats_raw_XW", "oracle_excess"},
        )
        self.assertTrue(all(result["checks"].values()))
        self.assertTrue(result["all_frozen_empirical_benchmark_gates_pass"])


if __name__ == "__main__":
    unittest.main()
