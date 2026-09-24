"""Checks for the replicate extension driver (memo/2026-09-23-replicate-extension-prd-v2.md)."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

import numpy as np

import run_extension_v1 as ex

HERE = Path(__file__).resolve().parent


class Plan(unittest.TestCase):
    def setUp(self):
        self.tasks = ex.planned_tasks()

    def test_counts_match_the_prd(self):
        count = lambda e: sum(t["exp"] == e for t in self.tasks)
        self.assertEqual({e: count(e) for e in ("E1", "E2", "E3", "E5", "E6", "E8")},
                         {"E1": 240, "E2": 80, "E3": 80, "E5": 90, "E6": 79, "E8": 80})
        self.assertEqual(sum(ex.kspc_child(t, Path("x")) is not None for t in self.tasks), 489)

    def test_identifiers_start_after_the_sealed_ones(self):
        self.assertEqual(min(t["rep"] for t in self.tasks if t["exp"] in ("E1", "E2")), 20)
        self.assertEqual({e: min(t["id"] for t in self.tasks if t["exp"] == e) for e in ("E3", "E5", "E6", "E8")},
                         {"E3": 21, "E5": 21, "E6": 22, "E8": 21})
        self.assertEqual(max(t["id"] for t in self.tasks if t["exp"] == "E6"), 100)

    def test_e1_e2_seeds_are_new_and_distinct(self):
        seeds = [t["seed"] for t in self.tasks if t["exp"] in ("E1", "E2")]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual((min(seeds), max(seeds)), (99_720_000, 100_099_000))
        constants = {int(c.replace("_", "")) for f in HERE.glob("*.py") if f.name != "test_extension_v1.py"
                     and f.name != "run_extension_v1.py"
                     for c in re.findall(r"\b9\d_\d{3}_\d{3}\b|\b1\d\d_\d{3}_\d{3}\b", f.read_text())}
        self.assertFalse([c for c in constants if min(seeds) - 1_000 <= c <= max(seeds) + 1_000])
        sealed = {s for s in range(97_200_000, 97_240_000, 1_000)} | {s for s in range(97_400_000, 97_420_000, 1_000)}
        self.assertFalse(sealed & set(seeds))

    def test_order_puts_the_longest_first(self):
        self.assertTrue(all(t.get("level") == "SCM-3" for t in self.tasks[:80]))
        self.assertTrue(all(t["exp"] == "E2" for t in self.tasks[80:160]))

    def test_every_output_is_a_distinct_file_under_the_extension_folder(self):
        paths = [ex.shard_of(t, ex.EXT) for t in self.tasks]
        paths += [ex.shard_of(c, ex.EXT) for t in self.tasks if (c := ex.kspc_child(t, Path("x"))) is not None]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(ex.EXT in p.parents for p in paths))
        sealed_roots = [p.resolve() for p in ex.SEALED.values()]
        self.assertFalse([p for p in paths if any(r in p.resolve().parents for r in sealed_roots)])

    def test_kspc_rows_point_at_their_own_data_set(self):
        t = next(t for t in self.tasks if t["exp"] == "E5")
        child = ex.kspc_child(t, ex.shard_of(t, ex.EXT))
        self.assertEqual((child["of"], child["name"], child["replicate"]), ("E5", "twins", t["id"]))
        self.assertEqual(child["sealed"], str(ex.shard_of(t, ex.EXT)))
        self.assertIsNone(ex.kspc_child({"exp": "E8", "id": 21}, Path("x")))


class Integrity(unittest.TestCase):
    def test_sources_match_the_frozen_records(self):
        checks = ex.check_sources()
        self.assertFalse({e: c["unexplained"] for e, c in checks.items() if c["unexplained"]})

    def test_ihdp_has_the_realizations_the_plan_uses(self):
        z = np.load(HERE.parent / "materials" / "real_world_data" / "ihdp" / "ihdp_npci_1-100.train.npz")
        self.assertEqual(z["mu1"].shape[1], 100)

    def test_diff_ignores_bookkeeping_and_catches_a_number(self):
        a = {"timing_s": {"total": 1}, "probe": {"estimate": 1.0, "retained": ["S1"]}}
        b = {"timing_s": {"total": 2}, "probe": {"estimate": 1.0, "retained": ["S1"]}}
        self.assertEqual(ex.diff(a, b), [])
        b["probe"]["estimate"] = 1.0 + 1e-15
        self.assertEqual(len(ex.diff(a, b)), 1)

    def test_reproduction_never_writes_to_the_extension_folder(self):
        with self.assertRaises(SystemExit):
            ex.main(["run_extension_v1.py", "reproduce", "--root", str(ex.EXT)])


class SummaryReproducesTheFrozenIndex(unittest.TestCase):
    """The sealed block of the extension summary must give the frozen index's numbers
    (EXPERIMENT_INDEX.md, rounded as printed there)."""

    @classmethod
    def setUpClass(cls):
        import summarise_extension_v1 as se
        cls.s = se.summary()["experiments"]

    def check(self, exp, printed, places):
        for key, value in printed.items():
            got = self.s[exp][key]["sealed"]["mae"]
            self.assertAlmostEqual(round(got, places), value, places=places, msg=f"{exp} {key}")

    def test_e1_e2(self):
        self.check("SCM-1", {"probe": 0.020, "cevae": 0.030, "p2sls_given": 0.031, "park_clean": 0.033, "raw": 0.271,
                             "x": 1.442, "p2sls_swapped": 5.186, "park_contaminated": 5.258, "kspc_clean": 0.875,
                             "kspc_contaminated": 1.881, "oracle": 0.003}, 3)
        self.check("SCM-3", {"probe": 0.014, "cevae": 0.409, "p2sls_given": 0.120, "kspc_clean": 0.813}, 3)
        self.check("SCM-4", {"probe": 0.018, "pixel": 0.281, "cevae": 0.047, "kspc_contaminated": 1.807}, 3)

    def test_e5_e6(self):
        self.check("E5", {"probe": 0.0077, "p2sls_given_roles": 0.0156, "raw": 0.0207, "cevae": 0.0894,
                          "coca_ate": 0.7318, "kspc_clean": 0.0099, "oracle": 0.0074}, 4)
        self.check("E6", {"cevae": 0.1828, "probe": 0.3054, "coca_ate": 5.9838, "kspc_contaminated": 1.2499}, 4)

    def test_e8_and_e3(self):
        self.check("E8", {"PROBE, significance": 0.054, "raw": 0.104, "naive": 0.617}, 3)
        self.assertAlmostEqual(round(self.s["E3"]["probe_v3"]["sealed"]["mean"], 2), -1.20)
        self.assertEqual(self.s["E3"]["probe_v3"]["sealed"]["returned"], 20)


if __name__ == "__main__":
    unittest.main()
