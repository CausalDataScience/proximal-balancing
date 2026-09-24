"""Checks for the RHC run, against memo/2026-09-22-rhc-real-data-prereg-v1.md.

    python3 -B test_rhc.py
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

import rhc_data as rd
import run_family_v2 as rf
import run_rhc as rr

HERE = Path(__file__).resolve().parent
PREREG = HERE.parent / "memo" / "2026-09-22-rhc-real-data-prereg-v1.md"


class Data(unittest.TestCase):
    def test_file_matches_the_recorded_hash(self):
        self.assertEqual(rf.sha256(rd.CSV), rd.SHA256)

    def test_shape_and_arms_match_the_published_description(self):
        d = rd.load()
        self.assertEqual(d["n"], 5_735)
        self.assertEqual(d["treated"], 2_184)
        self.assertEqual(d["n"] - d["treated"], 3_551)

    def test_every_block_has_its_two_named_coordinates(self):
        d = rd.load()
        for j, block in enumerate(rd.BLOCKS):
            self.assertEqual(d["view"][f"W{j + 1}"].shape, (5_735, len(block)))
        self.assertEqual(len(rd.PROXY), 10)
        self.assertEqual(len(set(rd.PROXY)), 10)

    def test_the_two_pairs_cui_et_al_name_are_the_first_two_blocks(self):
        self.assertEqual(rd.BLOCKS[0], ("pafi1", "paco21"))
        self.assertEqual(rd.BLOCKS[1], ("ph1", "hema1"))

    def test_no_proxy_coordinate_appears_among_the_covariates(self):
        d = rd.load()
        for name in d["x_names"]:
            self.assertNotIn(name.split("=")[0], rd.PROXY)

    def test_nothing_dropped_is_used(self):
        used = set(rd.CONTINUOUS_X) | set(rd.CATEGORICAL_X) | set(rd.PROXY)
        self.assertEqual(used & set(rd.DROPPED), set())

    def test_the_view_is_finite_and_has_only_learner_keys(self):
        d = rd.load()
        self.assertEqual(set(d["view"]), {"X", "W1", "W2", "W3", "W4", "W5", "A", "Y"})
        for key, value in d["view"].items():
            self.assertTrue(np.all(np.isfinite(value)), key)

    def test_treatment_is_binary_and_outcome_is_within_thirty_days(self):
        d = rd.load()
        self.assertEqual(set(np.unique(d["view"]["A"])), {0.0, 1.0})
        self.assertGreaterEqual(d["view"]["Y"].min(), 0.0)
        self.assertLessEqual(d["view"]["Y"].max(), 30.0)

    def test_categorical_encoding_drops_one_level_so_there_is_no_constant_direction(self):
        block, levels = rd._one_hot(["a", "b", "b", "c"])
        self.assertEqual(block.shape, (4, 2))
        self.assertEqual(levels, ["b", "c"])
        self.assertFalse(np.all(block.sum(axis=1) == 1.0))


class Protocol(unittest.TestCase):
    def test_the_sealed_files_are_untouched(self):
        sealed = json.loads((HERE.parent / "results" / "family_v2" / "scm4"
                             / "confirm_protocol_v1.json").read_text())["source_sha256"]
        for name in rr.SEALED:
            self.assertEqual(rf.sha256(HERE / name), sealed[name], name)

    def test_the_level_asks_for_five_blocks_of_two_and_rank_two(self):
        level = rd.level(61)
        self.assertEqual(level.d_blocks, (2, 2, 2, 2, 2))
        self.assertEqual(level.k, 2)

    def test_the_grid_is_the_one_the_preregistration_fixes(self):
        text = PREREG.read_text()
        self.assertIn("0.5,\\ 1,\\ 2,\\ 4,\\ 8,\\ 16,\\ 32", text)
        self.assertEqual(rr.GRID, [0.5e-4, 1e-4, 2e-4, 4e-4, 8e-4, 16e-4, 32e-4])

    def test_the_gate_interval_is_three_published_standard_errors(self):
        lo = rd.GATE["target"] - 3.0 * rd.GATE["se"]
        hi = rd.GATE["target"] + 3.0 * rd.GATE["se"]
        self.assertAlmostEqual(lo, -2.13, places=2)
        self.assertAlmostEqual(hi, -0.21, places=2)

    def test_the_seed_is_the_recorded_one_and_is_new(self):
        self.assertEqual(rd.SEED, 98_100_000)
        self.assertIn("98_100_000", PREREG.read_text())

    def test_results_are_write_once(self):
        if rr.OUT.exists():
            with self.assertRaises(SystemExit):
                rr.main()



class Pooled(unittest.TestCase):
    """The v2 rule of rhc_pooled.py."""

    def test_pooled_upper_is_mean_gap_plus_z_times_se_of_the_mean(self):
        import rhc_pooled as rp
        rots = [{"screen": {"gap": g, "se": s}} for g, s in ((0.002, 0.001), (-0.001, 0.001), (0.0, 0.002), (0.003, 0.001))]
        expected = max(0.001, 0.0) + rp.Z * np.sqrt(0.001 ** 2 + 0.001 ** 2 + 0.002 ** 2 + 0.001 ** 2) / 4
        self.assertAlmostEqual(rp.pooled_upper(rots), expected, places=12)

    def test_negative_mean_gap_is_clipped_to_zero(self):
        import rhc_pooled as rp
        rots = [{"screen": {"gap": -0.01, "se": 0.001}} for _ in range(4)]
        self.assertAlmostEqual(rp.pooled_upper(rots), rp.Z * 0.001 / 2, places=12)

    def test_z_is_the_sealed_bonferroni_level(self):
        import rhc_pooled as rp
        from scipy.stats import norm
        self.assertAlmostEqual(rp.Z, norm.ppf(1 - 0.05 / 30), places=12)

    def test_pooled_rule_leaves_every_sealed_simulation_estimate_unchanged(self):
        import rhc_pooled as rp
        v = rp.validate_on_simulations()
        for level, rules in v.items():
            self.assertEqual(rules["pooled"]["returned"], 20, level)
            self.assertAlmostEqual(rules["pooled"]["mae"], rules["sealed"]["mae"], places=3, msg=level)
            self.assertEqual(rules["pooled"]["data_sets_keeping_a_w5_choice"], 0, level)

    def test_v2_result_is_write_once(self):
        import rhc_pooled as rp
        if rp.OUT.exists():
            with self.assertRaises(SystemExit):
                rp.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
