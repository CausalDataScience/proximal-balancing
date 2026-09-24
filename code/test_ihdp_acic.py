"""Checks for the IHDP and ACIC runs, against memo/2026-09-22-ihdp-acic-prereg-v1.md."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

import family_v2_dgp as g
import hidden_proxy_recipe as hp
import ihdp_acic_data as ia
import run_family_v2 as rf
import run_ihdp_acic as ri

HERE = Path(__file__).resolve().parent
PREREG = HERE.parent / "memo" / "2026-09-22-ihdp-acic-prereg-v1.md"


class Recipe(unittest.TestCase):
    def test_mechanism_matches_sealed_scm1(self):
        self.assertEqual((g.TREAT["U"], g.TREAT["I"], g.PROXY_SD, g.BAD_I), (1.5, 2.0, 0.85, -3.0))
        self.assertEqual((hp.LINK_LO, hp.LINK_HI, hp.X_COEF_SD), (0.1, 0.8, 0.1))

    def test_build_leaves_covariates_and_outcomes_alone(self):
        rng = np.random.default_rng(0)
        x, u = rng.normal(size=(300, 4)), rng.normal(size=300)
        y0, y1 = rng.normal(size=300), rng.normal(size=300) + 2.0
        d = hp.build(x, u, y0, y1, 11, 12)
        self.assertTrue(np.array_equal(d["X"], x))
        self.assertAlmostEqual(d["tau"], float((y1 - y0).mean()), places=12)
        self.assertTrue(np.all((d["A"] == 1) * (d["Y"] == y1) + (d["A"] == 0) * (d["Y"] == y0)))

    def test_w4_carries_the_treatment_only_cause_and_w1_does_not(self):
        d = ia.ihdp(2)
        self.assertLess(np.corrcoef(d["W4"][:, 0], d["I_eval_only"])[0, 1], -0.8)
        self.assertLess(abs(np.corrcoef(d["W1"][:, 0], d["I_eval_only"])[0, 1]), 0.12)

    def test_overlap_is_bounded_by_the_link(self):
        for name in ("ihdp", "acic"):
            d = ia.DATASETS[name]["load"](2)
            self.assertGreaterEqual(d["propensity"].min(), hp.LINK_LO)
            self.assertLessEqual(d["propensity"].max(), hp.LINK_LO + hp.LINK_HI)


class Data(unittest.TestCase):
    def test_ihdp_shape_and_hidden_column(self):
        d = ia.ihdp(2)
        self.assertEqual((d["n"], d["d_x"]), (747, 24))
        self.assertEqual(ia.IHDP_HIDDEN, 5)

    def test_acic_shape_and_hidden_column(self):
        d = ia.acic(2)
        self.assertEqual(d["n"], 4802)
        self.assertEqual(ia.ACIC_HIDDEN, "x_44")

    def test_potential_outcomes_are_the_benchmark_s_own(self):
        a = ia._ihdp_arrays()
        t, yf, ycf = a["t"][:, 2], a["yf"][:, 2], a["ycf"][:, 2]
        d = ia.ihdp(2)
        self.assertAlmostEqual(d["tau"], float((np.where(t == 1, yf, ycf) - np.where(t == 1, ycf, yf)).mean()), places=12)

    def test_replicates_are_reproducible_and_distinct(self):
        self.assertTrue(np.array_equal(ia.acic(2)["A"], ia.acic(2)["A"]))
        self.assertFalse(np.array_equal(ia.acic(2)["A"], ia.acic(3)["A"]))


class Protocol(unittest.TestCase):
    def test_the_sealed_files_are_untouched(self):
        sealed = json.loads((HERE.parent / "results" / "family_v2" / "scm4"
                             / "confirm_protocol_v1.json").read_text())["source_sha256"]
        for name in ("family_v2_dgp.py", "family_v2_probe.py", "family_v2_competitors.py",
                     "probe_structured_scm.py"):
            self.assertEqual(rf.sha256(HERE / name), sealed[name], name)

    def test_the_development_replicate_is_excluded(self):
        self.assertEqual(ia.DEVELOPMENT, 1)
        self.assertNotIn(1, list(ia.IHDP_REPS))
        self.assertNotIn(1, list(ia.ACIC_REPS))
        self.assertEqual(list(ia.IHDP_REPS), list(range(2, 12)))
        self.assertEqual(list(ia.ACIC_REPS), list(range(2, 11)))

    def test_the_preregistration_records_the_constants(self):
        text = PREREG.read_text()
        for token in ("98,400,000", "98,500,000", "x_44", "1.5"):
            self.assertIn(token, text)

    def test_shards_are_write_once(self):
        for name in ("ihdp", "acic"):
            for r in ia.DATASETS[name]["reps"]:
                if ri.shard_path(name, r).exists():
                    self.assertEqual(ri.run_replicate(name, r)["status"], "exists")


if __name__ == "__main__":
    unittest.main(verbosity=2)
