"""The independent scorer: it must reproduce the runners' own summaries and refuse malformed results."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import summarize_minimal_suite as sm

E12 = "../results/e12_scm1_ncurve_confirm_v2.json"
E11 = "../results/e11_v2_exact_confirm_v2.json"


def need(path: str) -> None:
    """Skip rather than error when a confirmatory result has not been produced yet."""
    if not Path(path).exists():
        raise unittest.SkipTest(f"{path} has not been written yet")


def payload(path):
    return json.loads(Path(path).read_text())


class TestLoadGuards(unittest.TestCase):
    def _write(self, body):
        d = tempfile.mkdtemp()
        p = Path(d) / "r.json"
        p.write_text(json.dumps(body))
        return str(p)

    def test_an_incomplete_run_is_refused(self):
        with self.assertRaises(RuntimeError):
            sm.load(self._write({"complete": False, "cells": []}))

    def test_a_repeated_design_point_is_refused(self):
        body = {"complete": True, "cells": [{"data_seed": 1, "n_total": 100, "complete": True},
                                            {"data_seed": 1, "n_total": 100, "complete": True}]}
        with self.assertRaises(RuntimeError):
            sm.load(self._write(body))

    def test_the_same_seed_at_two_sample_sizes_is_allowed(self):
        body = {"complete": True, "cells": [{"data_seed": 1, "n_total": 100, "complete": True},
                                            {"data_seed": 1, "n_total": 200, "complete": True}]}
        self.assertEqual(len(sm.load(self._write(body))["cells"]), 2)

    def test_an_incomplete_cell_is_refused(self):
        body = {"complete": True, "cells": [{"data_seed": 1, "complete": False}]}
        with self.assertRaises(RuntimeError):
            sm.load(self._write(body))


class TestE12MatchesTheRunner(unittest.TestCase):
    """Recomputing from the stored cells must land on the runner's own numbers to the last digit."""

    @classmethod
    def setUpClass(cls):
        need(E12)
        cls.data = payload(E12)
        cls.protocol = json.loads(Path(cls.data["protocol"]["path"]).read_text())
        cls.out = sm.score_e12(cls.data, cls.protocol)

    def test_location_and_spread_agree_with_the_stored_summary(self):
        stored = self.data["summary"]["by_sample_size"]
        for row in self.out["rows"]:
            s = stored[str(row["n"])]
            self.assertAlmostEqual(row["mean_abs_error"], s["mae"], places=12)
            self.assertAlmostEqual(row["p90_abs_error"], s["p90"], places=12)
            self.assertAlmostEqual(row["return_rate"], s["return_rate"], places=12)

    def test_the_gate_statistic_is_the_mean_not_the_median(self):
        row = self.out["rows"][0]
        self.assertNotAlmostEqual(row["mean_abs_error"], row["median_abs_error"], places=3)
        self.assertAlmostEqual(row["mean_abs_error"],
                               self.data["summary"]["by_sample_size"]["1500"]["mae"], places=12)

    def test_a_tied_largest_component_does_not_count_as_a_return(self):
        """A tie is scored as no return.  The nested-prefix confirmation happens to contain none, so one
        is injected into a copy of the payload rather than relying on the fixture."""
        import copy
        payload = copy.deepcopy(self.data)
        victim = next(q for q in payload["cells"] if q["n_total"] == 6000)
        victim["return_kind"], victim["estimate"] = "tied_largest", None
        out = sm.score_e12(payload, self.protocol)
        row = next(r for r in out["rows"] if r["n"] == 6000)
        self.assertLess(row["return_rate"], 1.0)
        self.assertEqual(row["cells"], 20)

    def test_the_verdict_is_reproduced(self):
        self.assertEqual(self.out["verdict"], self.data["verdict"])


class TestE11MatchesTheTheorem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        need(E11)
        cls.data = payload(E11)
        cls.protocol = json.loads(Path(cls.data["protocol"]["path"]).read_text())
        cls.out = sm.score_e11_v2(cls.data, cls.protocol)

    def test_the_bound_rebuilt_from_the_hypergeometric_law_matches_the_stored_one(self):
        for row in self.out["rows"]:
            self.assertLess(row["bound_difference"], 1e-12)

    def test_the_bound_holds_in_every_outer_state(self):
        for row in self.out["rows"]:
            self.assertGreaterEqual(row["success_lower_99"], row["B"])

    def test_the_separation_condition_is_met(self):
        for row in self.out["rows"]:
            self.assertLess(4 * row["rho"], row["separation_g"])

    def test_a_tampered_bound_is_caught(self):
        cell = dict(self.data["cells"][0])
        cell["theorem_B"] = cell["theorem_B"] + 0.01
        rebuilt = sm.recompute_bound(cell, self.protocol)
        self.assertGreater(rebuilt["difference"], 1e-3)


class TestScoreArtifact(unittest.TestCase):
    """The verdict lives in one sealed artifact, not in the result files the runners write."""

    def test_the_artifact_records_both_verdicts_and_flags_a_disagreement(self):
        path = "../results/scm6_highdim_confirm_v1.json"
        out = sm.score_scm6(sm.load(path),
                            json.loads(Path(json.loads(Path(path).read_text())
                                            ["protocol"]["path"]).read_text()))
        e = sm.entry("scm6", path, out)
        self.assertEqual(e["verdict"], "FAIL")
        self.assertEqual(e["verdict_stored_in_result"], "NOT_EVALUATED")
        self.assertFalse(e["stored_verdict_usable"])
        self.assertTrue(e["verdicts_agree"], "an unevaluated stored verdict is not a disagreement")
        forged = dict(e, verdict_stored_in_result="PASS")
        forged["verdicts_agree"] = forged["verdict_stored_in_result"] == forged["verdict"]
        self.assertFalse(forged["verdicts_agree"])

    def test_the_artifact_notes_a_protocol_that_predates_sealing(self):
        path = "../results/scm6_highdim_dev_v1.json"
        out = sm.score_scm6(sm.load(path),
                            json.loads(Path(json.loads(Path(path).read_text())
                                            ["protocol"]["path"]).read_text()))
        self.assertFalse(sm.entry("scm6", path, out)["protocol_sealed"])

    def test_the_written_artifact_is_sealed_and_write_once(self):
        import minimal_suite_common as c
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "scores.json")
            entries = [{"experiment_id": "X", "phase": "development", "verdict": "FAIL",
                        "verdicts_agree": True, "protocol_sealed": True,
                        "result_path": "a", "protocol_path": "b"}]
            sm.write_scores(path, entries)
            written = json.loads(Path(path).read_text())
            self.assertEqual(c.canonical_digest(written), written[c.DIGEST_FIELD])
            self.assertEqual(written["summary"], {"X development": "FAIL"})
            with self.assertRaises(FileExistsError):
                sm.write_scores(path, entries)


class TestPairedContrast(unittest.TestCase):
    def test_a_clearly_better_method_gives_a_negative_upper_end(self):
        rng = np.random.default_rng(4)
        better = np.abs(rng.normal(0.02, 0.005, 40))
        worse = np.abs(rng.normal(0.40, 0.050, 40))
        self.assertLess(sm.paired_upper(better, worse, 4, 0.05), 0.0)

    def test_the_bonferroni_correction_widens_the_interval(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(0.10, 0.02, 30), rng.normal(0.11, 0.02, 30)
        self.assertLess(sm.paired_upper(a, b, 1, 0.05), sm.paired_upper(a, b, 8, 0.05))


if __name__ == "__main__":
    unittest.main()
