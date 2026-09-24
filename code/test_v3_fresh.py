"""Checks for the v3 prospective confirmation: the scorer's integrity gates, the frozen protocol, and the
explicit-identifier runner (memo/2026-09-22-v3-confirmation-prd-v1.md, sections 4, 5 and 9)."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np

import noise_scaled as ns
import v3_fresh_scorer as sc

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def a_shard() -> dict:
    """A real shard, used as the base for the corruption tests."""
    return json.loads((RESULTS / "twins" / "replicate_1.json").read_text())


def manifest_for(shard: dict, replicate: int) -> dict:
    return {"dataset": "twins", "seed": shard["seed"], "source_sha256": shard["source_sha256"]}


class Integrity(unittest.TestCase):
    def setUp(self):
        self.shard = a_shard()
        self.expected = manifest_for(self.shard, 1)

    def accepts(self, shard):
        sc.check_shard(shard, "twins", 1, self.expected)

    def rejects(self, shard, fragment):
        with self.assertRaises(sc.Rejected) as cm:
            sc.check_shard(shard, "twins", 1, self.expected)
        self.assertIn(fragment, str(cm.exception))

    def test_a_real_shard_passes(self):
        self.accepts(self.shard)

    def test_wrong_replicate_is_rejected(self):
        with self.assertRaises(sc.Rejected):
            sc.check_shard(self.shard, "twins", 2, self.expected)

    def test_wrong_seed_is_rejected(self):
        self.expected = dict(self.expected, seed=self.expected["seed"] + 1)
        self.rejects(self.shard, "seed")

    def test_changed_source_hash_is_rejected(self):
        bad = dict(self.expected)
        key = next(iter(bad["source_sha256"]))
        bad["source_sha256"] = dict(bad["source_sha256"], **{key: "0" * 64})
        self.expected = bad
        self.rejects(self.shard, "source hash")

    def test_non_finite_gap_is_rejected(self):
        s = copy.deepcopy(self.shard)
        nm = s["probe"]["split_names"][0]
        s["probe"]["per_split"][nm]["screen"][0]["gap"] = float("nan")
        self.rejects(s, "not finite")

    def test_negative_standard_error_is_rejected(self):
        s = copy.deepcopy(self.shard)
        nm = s["probe"]["split_names"][0]
        s["probe"]["per_split"][nm]["screen"][0]["se"] = -1e-6
        self.rejects(s, "negative standard error")

    def test_failed_overlap_is_rejected(self):
        s = copy.deepcopy(self.shard)
        nm = s["probe"]["split_names"][1]
        s["probe"]["per_split"][nm]["screen"][2]["overlap"] = False
        self.rejects(s, "overlap")

    def test_missing_rotation_is_rejected(self):
        s = copy.deepcopy(self.shard)
        nm = s["probe"]["split_names"][0]
        s["probe"]["per_split"][nm]["screen"] = s["probe"]["per_split"][nm]["screen"][:3]
        self.rejects(s, "rotations")

    def test_duplicate_or_short_split_list_is_rejected(self):
        s = copy.deepcopy(self.shard)
        s["probe"]["split_names"] = s["probe"]["split_names"][:29]
        self.rejects(s, "distinct split names")

    def test_bad_score_matrix_is_rejected(self):
        s = copy.deepcopy(self.shard)
        s["probe"]["score_crossproduct"] = [row[:29] for row in s["probe"]["score_crossproduct"]]
        self.rejects(s, "score cross-product")

    def test_missing_comparator_is_rejected(self):
        s = copy.deepcopy(self.shard)
        del s["baselines"]["oracle"]
        self.rejects(s, "comparator oracle")


class Scoring(unittest.TestCase):
    def test_a_missing_shard_makes_the_set_incomplete_and_reads_no_gate(self):
        shard = a_shard()
        manifest = {"replicates": [1, 2], "expected": {"1": manifest_for(shard, 1), "2": manifest_for(shard, 2)}}
        paths = {1: RESULTS / "twins" / "replicate_1.json", 2: RESULTS / "twins" / "replicate_999.json"}
        out = sc.score("twins", manifest, paths)
        self.assertEqual(out["status"], "incomplete")
        self.assertNotIn("gates", out)
        self.assertIn("shard missing", out["rejected"][2])

    def test_the_denominator_is_the_designated_count(self):
        ids = [1, 2, 3]
        manifest = {"replicates": ids,
                    "expected": {str(r): manifest_for(json.loads((RESULTS / "twins" / f"replicate_{r}.json").read_text()), r)
                                 for r in ids}}
        out = sc.score("twins", manifest, sc.shard_paths_for("twins", ids))
        self.assertEqual(out["status"], "complete")
        self.assertEqual(out["return_denominator"], len(ids))
        self.assertEqual(out["gates"]["return_rate"]["of"], len(ids))

    def test_tpr_is_reported_for_twins_and_withheld_for_the_real_proxy_designs(self):
        self.assertTrue(sc.HAS_VALIDITY_TRUTH["twins"])
        self.assertFalse(sc.HAS_VALIDITY_TRUTH["rhc_planted"])
        self.assertFalse(sc.HAS_VALIDITY_TRUTH["rhc"])

    def test_paired_bound_is_not_computable_below_two_pairs(self):
        ids = [1]
        manifest = {"replicates": ids,
                    "expected": {"1": manifest_for(a_shard(), 1)}}
        out = sc.score("twins", manifest, sc.shard_paths_for("twins", ids))
        for key, v in out["paired_upper_on_both_returned"].items():
            self.assertLessEqual(v["k"], 1)
            self.assertIsNone(v["upper"], key)


class Gates(unittest.TestCase):
    def test_the_two_gates_are_the_preregistered_ones(self):
        self.assertEqual(sc.GATE_RETURN_MIN, 8)
        self.assertEqual(sc.GATE_MAE_MULTIPLE, 1.5)

    def test_the_limits_come_from_the_stored_reference(self):
        ref = json.loads(sc.REFERENCE.read_text())["real_data"]
        for name, expected in (("twins", 0.009557489874420181), ("rhc_planted", 0.45349871582366097),
                               ("ihdp", 0.33618081767248315)):
            self.assertEqual(ref[name]["mae"], expected, name)

    def test_the_threshold_rule_is_the_median_noise_term(self):
        self.assertAlmostEqual(ns.Z, 2.9352, places=3)


class Protocol(unittest.TestCase):
    def test_the_freeze_refuses_when_a_designated_replicate_already_exists(self):
        """The freshness guarantee is that the freeze happens before any designated shard exists.  Once the
        run is done the shards are there by design, so the invariant to test is the refusal itself."""
        import freeze_v3_protocol as fz
        self.assertTrue(fz.OUT.exists(), "the protocol must be frozen before this is meaningful")
        with self.assertRaises(SystemExit):
            fz.main()                                  # refuses: the protocol file is already write-once
        protocol = json.loads(fz.OUT.read_text())
        self.assertTrue(protocol["verified_before_freezing"]["designated_replicates_have_no_prior_shards"])
        self.assertEqual(protocol["verified_before_freezing"]["checked"],
                         {d: p["replicates"] for d, p in fz.PLAN.items()})

    def test_the_correction_is_neutral(self):
        import v3_fresh_scorer as s2
        if not s2.PROTOCOL_V2.exists():
            self.skipTest("no correction was needed")
        v1 = json.loads(s2.PROTOCOL_V1.read_text()); v2 = json.loads(s2.PROTOCOL_V2.read_text())
        for key in ("rule", "z", "gates", "reference_file", "protocol_sha256_of_prd"):
            self.assertEqual(v1[key], v2[key], key)
        for name in v1["manifests"]:
            for key in ("replicates", "reference_conditional_mae", "gate_limit", "gate_return_minimum",
                        "benchmark_input_sha256"):
                self.assertEqual(v1["manifests"][name][key], v2["manifests"][name][key], (name, key))
            for r in v1["manifests"][name]["expected"]:
                self.assertEqual(v1["manifests"][name]["expected"][r]["seed"],
                                 v2["manifests"][name]["expected"][r]["seed"], (name, r))
        self.assertEqual(v2["correction"]["removed"], ["twins_data.py"])

    def test_the_plan_matches_the_preregistration(self):
        import freeze_v3_protocol as fz
        self.assertEqual(fz.PLAN["twins"]["replicates"], list(range(11, 21)))
        self.assertEqual(fz.PLAN["rhc_planted"]["replicates"], list(range(11, 21)))
        self.assertEqual(fz.PLAN["ihdp"]["replicates"], list(range(12, 22)))
        self.assertNotIn("acic", fz.PLAN)

    def test_the_runner_refuses_identifiers_outside_the_protocol(self):
        import run_v3_fresh as rv
        self.assertTrue(rv.PROTOCOL.name.endswith("v3_confirmation_protocol.json"))
        with self.assertRaises(ValueError):
            rv.run_one("acic", 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
