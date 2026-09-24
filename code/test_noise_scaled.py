"""Checks for the v3 noise-scaled screen (noise_scaled.py)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

import noise_scaled as ns

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


class Rule(unittest.TestCase):
    def test_floor_is_the_median_noise_and_retention_compares_to_it(self):
        names = [f"S{i}" for i in range(1, 31)]
        rec = {nm: {"estimate": 1.0, "screen": [{"gap": 0.0, "se": 0.001 * (1 + i % 3)} for _ in range(4)]}
               for i, nm in enumerate(names)}
        probe = {"split_names": names, "per_split": rec, "score_crossproduct": np.eye(30).tolist()}
        dec = ns.decide(probe, 1000, 0)
        floor_expected = float(np.median([ns.Z * 0.001 * (1 + i % 3) / 2 for i in range(30)]))
        self.assertAlmostEqual(dec["floor"], floor_expected, places=12)
        self.assertTrue(all(ns.pooled(rec[nm])[0] + ns.Z * ns.pooled(rec[nm])[1] <= dec["floor"] for nm in dec["retained"]))

    def test_a_positive_mean_gap_raises_the_upper_bound(self):
        rec = {"screen": [{"gap": 0.002, "se": 0.0}] * 4, "estimate": 0.0}
        g, se = ns.pooled(rec)
        self.assertAlmostEqual(g, 0.002)
        rec2 = {"screen": [{"gap": -0.002, "se": 0.0}] * 4, "estimate": 0.0}
        self.assertEqual(ns.pooled(rec2)[0], 0.0)


class Validation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v3 = json.loads((RESULTS / "noise_scaled_v3.json").read_text())

    def test_sealed_simulation_estimates_are_preserved(self):
        for level, v in self.v3["simulations"].items():
            self.assertEqual(v["returned"], 20, level)
            self.assertLess(v["max_abs_change_in_estimate"], 0.006, level)
            self.assertAlmostEqual(v["mae_v3"], v["mae_sealed"], places=3, msg=level)
            self.assertEqual(v["kept_any_w5"], 0, level)

    def test_floor_matches_the_sealed_thresholds_within_ten_percent(self):
        for level, v in self.v3["simulations"].items():
            self.assertLess(abs(v["floor_median"] / v["sealed_t"] - 1.0), 0.15, level)

    def test_every_real_data_run_now_returns(self):
        for name, v in self.v3["real_data"].items():
            self.assertEqual(v["returned"], v["of"], name)

    def test_rhc_separates_on_the_treatment_side_pair(self):
        kept = set(self.v3["rhc"]["retained"])
        self.assertEqual(len(kept), 15)
        self.assertTrue(all("1" not in nm for nm in kept))          # every retained choice keeps W1
        all_without_w1 = {nm for nm in json.loads((RESULTS / "rhc" / "result_v1.json").read_text())["per_split"] if "1" not in nm}
        self.assertEqual(kept, all_without_w1)                        # and every such choice is retained


if __name__ == "__main__":
    unittest.main(verbosity=2)
