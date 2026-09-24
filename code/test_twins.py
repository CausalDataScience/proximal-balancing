"""Checks for the Twins benchmark run, against memo/2026-09-22-twins-real-outcome-prereg-v1.md."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

import family_v2_dgp as g
import run_family_v2 as rf
import run_twins as rt
import twins_data as td

HERE = Path(__file__).resolve().parent
PREREG = HERE.parent / "memo" / "2026-09-22-twins-real-outcome-prereg-v1.md"


class Real(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real = td.load_real()

    def test_subset_size_and_effect_match_the_preregistration(self):
        self.assertEqual(self.real["n"], 11_984)
        self.assertAlmostEqual(self.real["tau"], -0.0252, places=4)
        self.assertEqual(self.real["X"].shape[1], 42)

    def test_the_hidden_confounder_is_not_among_the_covariates(self):
        self.assertNotIn(td.HIDDEN, self.real["x_names"])
        for name in td.DROP:
            self.assertNotIn(name, self.real["x_names"])

    def test_potential_outcomes_are_binary_and_both_observed(self):
        for key in ("Y0", "Y1"):
            self.assertEqual(set(np.unique(self.real[key])), {0.0, 1.0})
            self.assertEqual(len(self.real[key]), self.real["n"])

    def test_covariates_are_standardised_and_finite(self):
        x = self.real["X"]
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertTrue(np.allclose(x.mean(0), 0.0, atol=1e-8))


class Generated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real = td.load_real()
        cls.d = td.generate(cls.real, 1)

    def test_only_treatment_and_proxies_are_generated(self):
        d = self.d
        self.assertTrue(np.array_equal(d["X"], self.real["X"]))
        self.assertTrue(np.all((d["A"] == 1) * (d["Y"] == self.real["Y1"]) + (d["A"] == 0) * (d["Y"] == self.real["Y0"])))

    def test_mechanism_is_scm1(self):
        self.assertEqual(g.TREAT["U"], 1.5); self.assertEqual(g.TREAT["I"], 2.0)
        self.assertEqual(g.PROXY_SD, 0.85); self.assertEqual(g.BAD_I, -3.0)
        d = self.d
        self.assertEqual(d["W5"].shape, (self.real["n"], 2))
        # W4 carries -3 I: its correlation with I is negative and strong; W1 carries none
        self.assertLess(np.corrcoef(d["W4"][:, 0], d["I_eval_only"])[0, 1], -0.8)
        self.assertLess(abs(np.corrcoef(d["W1"][:, 0], d["I_eval_only"])[0, 1]), 0.05)

    def test_replicates_differ_only_through_their_seed(self):
        e = td.generate(self.real, 2)
        self.assertFalse(np.array_equal(e["A"], self.d["A"]))
        self.assertTrue(np.array_equal(td.generate(self.real, 1)["A"], self.d["A"]))

    def test_confounding_is_present(self):
        self.assertGreater(abs(self.d["naive"] - self.real["tau"]), 0.05)
        self.assertGreater(abs(np.corrcoef(self.d["A"], self.d["U_eval_only"])[0, 1]), 0.2)


class Protocol(unittest.TestCase):
    def test_the_sealed_files_are_untouched(self):
        sealed = json.loads((HERE.parent / "results" / "family_v2" / "scm4"
                             / "confirm_protocol_v1.json").read_text())["source_sha256"]
        for name in rt.SEALED:
            self.assertEqual(rf.sha256(HERE / name), sealed[name], name)

    def test_threshold_and_grid_are_the_preregistered_ones(self):
        self.assertEqual(rt.THRESHOLD, 0.0015)
        self.assertEqual(rt.GRID, [0.0005, 0.001, 0.002, 0.004])
        self.assertIn("1.5\\times10^{-3}", PREREG.read_text())

    def test_development_replicate_is_excluded_from_confirmation(self):
        self.assertNotIn(0, list(rt.CONFIRM))
        self.assertEqual(list(rt.CONFIRM), list(range(1, 11)))

    def test_shards_are_write_once(self):
        for r in rt.CONFIRM:
            if rt.shard_path(r).exists():
                self.assertEqual(rt.run_replicate(r)["status"], "exists")


if __name__ == "__main__":
    unittest.main(verbosity=2)
