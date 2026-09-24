"""Checks for the planted-effect RHC run, against memo/2026-09-22-rhc-planted-prereg-v1.md."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

import rhc_data as rd
import rhc_planted_data as rp
import run_family_v2 as rf
import run_rhc_planted as rrp

HERE = Path(__file__).resolve().parent
PREREG = HERE.parent / "memo" / "2026-09-22-rhc-planted-prereg-v1.md"


class Design(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real = rp.load_real()
        cls.d = rp.generate(cls.real, 1)

    def test_the_hidden_score_is_removed_from_the_covariates(self):
        self.assertNotIn(rp.HIDDEN, self.real["x_names"])
        self.assertEqual(self.real["X"].shape, (5_735, 54))

    def test_every_proxy_block_is_the_raw_clinical_measurement(self):
        import csv
        with rd.CSV.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for k, block in enumerate(rd.BLOCKS):
            for c, name in enumerate(block):
                self.assertTrue(np.array_equal(self.real["blocks"][f"W{k + 1}"][:, c],
                                               np.array([float(r[name]) for r in rows])), name)

    def test_the_effect_is_planted_exactly(self):
        d, real = self.d, self.real
        self.assertTrue(np.array_equal(d["Y"], real["Y_real"] + rp.DELTA * d["A"]))
        self.assertEqual(d["tau"], rp.DELTA)
        self.assertEqual(rp.DELTA, -1.0)

    def test_only_the_treatment_changes_between_replicates(self):
        e = rp.generate(self.real, 2)
        self.assertFalse(np.array_equal(e["A"], self.d["A"]))
        self.assertTrue(np.array_equal(e["X"], self.d["X"]))
        for k in ("W1", "W2", "W3", "W4", "W5"):
            self.assertTrue(np.array_equal(e[k], self.d[k]))

    def test_confounding_is_present_and_overlap_holds(self):
        self.assertGreater(abs(np.corrcoef(self.d["A"], self.d["U_eval_only"])[0, 1]), 0.3)
        self.assertGreater(abs(self.d["naive"] - rp.DELTA), 1.0)
        self.assertGreaterEqual(self.d["propensity"].min(), rp.LINK_LO)
        self.assertLessEqual(self.d["propensity"].max(), rp.LINK_LO + rp.LINK_HI)

    def test_replicates_are_reproducible(self):
        self.assertTrue(np.array_equal(rp.generate(self.real, 1)["A"], self.d["A"]))


class Protocol(unittest.TestCase):
    def test_the_sealed_files_are_untouched(self):
        sealed = json.loads((HERE.parent / "results" / "family_v2" / "scm4"
                             / "confirm_protocol_v1.json").read_text())["source_sha256"]
        for name in ("family_v2_dgp.py", "family_v2_probe.py", "family_v2_competitors.py",
                     "probe_structured_scm.py"):
            self.assertEqual(rf.sha256(HERE / name), sealed[name], name)

    def test_the_preregistered_constants_are_the_ones_in_the_code(self):
        text = PREREG.read_text()
        self.assertIn("aps1", text)
        self.assertIn("98,300,000", text)
        self.assertEqual(rp.U_COEF, 1.5)
        self.assertEqual(rp.REPLICATE_SEED, 98_300_000)
        self.assertEqual(list(rrp.CONFIRM), list(range(1, 11)))

    def test_development_replicate_is_excluded(self):
        self.assertNotIn(0, list(rrp.CONFIRM))

    def test_shards_are_write_once(self):
        for r in rrp.CONFIRM:
            if rrp.shard_path(r).exists():
                self.assertEqual(rrp.run_replicate(r)["status"], "exists")


if __name__ == "__main__":
    unittest.main(verbosity=2)
