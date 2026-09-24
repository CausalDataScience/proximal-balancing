"""Checks of the SCM-4 image stage (spec memo/2026-09-20-scm4-shapes3d-reader-spec-v1.md).

Every check runs without the Shapes3D file: the bank is replaced by a small array of stand-in images whose
pixels encode the orientation level exactly.  Defects are planted on purpose, and a check fails if the gate or
the validator does not see them.  Seeds come from the test namespace 98,700,000-98,799,999 and from
family_v2_dgp.TEST_SEED.  Run: python3 -B test_scm4_images.py
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import family_v2_dgp as g
import family_v2_images as im
import run_scm4_images as run

T0 = 98_700_000


def stand_in_bank(seed: int = T0):
    """One uint8 image per bank row, with a bright band whose column is the row's orientation level."""
    rng = np.random.default_rng(seed)
    rows = np.arange(im.N_IMAGES)[::397]                       # a stride coprime with 15, so every level appears
    levels = im.factor_index(rows)[:, im.ORIENTATION]
    img = rng.integers(0, 40, size=(len(rows), 64, 64, 3), dtype=np.uint8)
    for i, level in enumerate(levels):
        img[i, :, 4 + int(level) * 4:7 + int(level) * 4, :] = 255
    return rows, levels, img


def perfect_reader(levels: np.ndarray):
    """A stand-in for the frozen reader: returns the true level of each row index into the stand-in bank."""
    class Reader:
        def eval(self):
            return self

    reader = Reader()
    table = np.asarray(levels, dtype=np.float64)

    def fake_raw(net, images, rows, dev, chunk=2000):
        return table[np.asarray(rows).ravel()]

    return reader, fake_raw


class Recording(unittest.TestCase):
    def test_population_sd_matches_the_data(self):
        data = g.generate(g.LEVELS["SCM-1"], 200_000, T0 + 1)
        for key, sd in im.population_sd().items():
            np.testing.assert_allclose(data[key].std(0), sd, rtol=0.02)

    def test_levels_are_in_range_and_ends_are_clipped(self):
        sd = np.array([2.0])
        w = np.array([[-100.0], [-6.0], [0.0], [6.0], [100.0]])
        self.assertEqual(im.record(w, sd).ravel().tolist(), [0, 0, 7, 14, 14])
        data = g.generate(g.LEVELS["SCM-1"], 20_000, T0 + 2)
        levels = im.record(data["W5"], im.population_sd()["W5"])
        self.assertTrue(((levels >= 0) & (levels < im.LEVELS)).all())
        self.assertGreater(len(np.unique(levels)), 10)

    def test_image_draw_reads_the_level_and_its_own_stream_only(self):
        rows, levels, _ = stand_in_bank()
        groups = im.bank_by_level(rows)
        want = np.array([0, 3, 14, 7, 7])
        got = im.draw_images(want, groups, np.random.default_rng(T0 + 3))
        self.assertTrue(np.array_equal(im.factor_index(got)[:, im.ORIENTATION], want))
        again = im.draw_images(want.copy(), groups, np.random.default_rng(T0 + 3))
        self.assertTrue(np.array_equal(got, again))
        other = im.draw_images(want, groups, np.random.default_rng(T0 + 4))
        self.assertFalse(np.array_equal(got, other))

    def test_split_roles_are_disjoint(self):
        rows = im.split_rows()
        self.assertEqual({k: len(v) for k, v in rows.items()}, im.SPLIT)
        allrows = np.concatenate(list(rows.values()))
        self.assertEqual(len(np.unique(allrows)), im.N_IMAGES)
        for name, r in rows.items():
            counts = np.bincount(im.factor_index(r)[:, im.ORIENTATION], minlength=im.LEVELS)
            self.assertTrue((counts > 0).all(), name)

    def test_factor_index_layout(self):
        self.assertEqual(im.factor_index(np.array([0, 1, 15, im.N_IMAGES - 1])).tolist(),
                         [[0] * 6, [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [9, 9, 9, 7, 3, 14]])


class ReaderGate(unittest.TestCase):
    def table(self, pred, truth, groups=None):
        return im.report(np.asarray(pred, dtype=float), np.asarray(truth), groups or {})

    def test_perfect_reading_passes(self):
        truth = np.tile(np.arange(im.LEVELS), 400)
        rep = self.table(truth + 0.05, truth, {"shape": truth % 4})
        self.assertTrue(im.gate(rep)["pass"])
        self.assertEqual(rep["overall"]["far"], 0)

    def test_a_single_nonfinite_reading_fails_the_gate(self):
        truth = np.tile(np.arange(im.LEVELS), 400)
        pred = truth.astype(float)
        pred[17] = np.nan
        rep = self.table(pred, truth)
        self.assertEqual(rep["overall"]["nonfinite"], 1)
        self.assertFalse(im.gate(rep)["pass"])
        self.assertIn("overall nonfinite", [f[0] for f in im.gate(rep)["failing"]])
        self.assertIsNone(im.propose_exclusion(rep))

    def test_errors_hidden_in_one_factor_value_are_caught(self):
        truth = np.tile(np.arange(im.LEVELS), 400)
        shape = np.arange(len(truth)) % 4
        pred = truth + 3.0 * (shape == 2)
        rep = self.table(pred, truth, {"shape": shape})
        self.assertFalse(im.gate(rep)["pass"])
        self.assertEqual(im.propose_exclusion(rep), {"shape": [2]})
        rep_ok = self.table(pred[shape != 2], truth[shape != 2], {"shape": shape[shape != 2]})
        self.assertTrue(im.gate(rep_ok)["pass"])

    def test_a_failing_level_cannot_be_excluded(self):
        truth = np.tile(np.arange(im.LEVELS), 400)
        pred = truth + 3.0 * (truth == 7)
        rep = self.table(pred, truth, {"shape": np.arange(len(truth)) % 4})
        self.assertFalse(im.gate(rep)["pass"])
        self.assertIsNone(im.propose_exclusion(rep))

    def test_systematic_shift_fails_through_the_bias_rule(self):
        truth = np.tile(np.arange(im.LEVELS), 400)
        rep = self.table(truth + 0.4, truth)
        self.assertLess(rep["overall"]["far"], 2)
        self.assertFalse(im.gate(rep)["pass"])

    def test_the_number_scored_is_the_number_the_learner_gets(self):
        rows, levels, images = stand_in_bank()
        reader, fake_raw = perfect_reader(levels)
        original = im.raw_read
        try:
            im.raw_read = lambda *a, **k: np.array([-3.0, 7.0, 99.0, np.nan])
            got = im.reading(reader, images, np.arange(4), None)
        finally:
            im.raw_read = original
        self.assertEqual(got[:3].tolist(), [0.0, 7.0, 14.0])
        self.assertTrue(np.isnan(got[3]))

    def test_reader_trains_and_reads_stand_in_images(self):
        rows, levels, images = stand_in_bank(T0 + 10)
        dev = torch.device("cpu")
        train, val = np.arange(len(rows))[: len(rows) - 200], np.arange(len(rows))[len(rows) - 200:]
        net, info = im.fit(images, levels, train, val, dev, seed=T0 + 11, epochs=2, batch=64, log=lambda *_: None)
        pred = im.reading(net, images, val, dev)
        self.assertLess(np.sqrt(np.mean((pred - levels[val]) ** 2)), 3.0)
        self.assertIn(info["epoch"], (1, 2))

    def test_training_stops_at_the_time_cap(self):
        rows, levels, images = stand_in_bank(T0 + 12)
        with self.assertRaises(TimeoutError):
            im.fit(images, levels, np.arange(len(rows)), np.arange(50), torch.device("cpu"),
                   seed=T0 + 13, epochs=2, batch=8, max_seconds=0.0, log=lambda *_: None)


def good_result() -> dict:
    names = sorted(run.EXPECTED_SPLITS)
    return {"split_names": names, "return_kind": "unique_largest", "estimate": 1.01, "retained": names[:8],
            "records": {n: {"estimate": 1.0, "rotations": [
                {"rotation": r, "theta": 1.0, "gap_fit": 0.0,
                 "screen": {"gap": 0.0, "se": 1e-4, "upper": 2e-4, "overlap": True, "pass": True}}
                for r in range(4)]} for n in names}}


class Validator(unittest.TestCase):
    def test_a_complete_result_passes(self):
        self.assertEqual(run.validate(good_result()), [])

    def test_duplicate_split_names_are_caught(self):
        out = good_result()
        out["split_names"] = ["S1"] * 30
        self.assertTrue(any("split ids" in p for p in run.validate(out)))

    def test_repeated_rotation_numbers_are_caught(self):
        out = good_result()
        out["records"]["S1"]["rotations"] = [dict(out["records"]["S1"]["rotations"][0]) for _ in range(4)]
        self.assertIn("S1: rotations are not 0-3 exactly", run.validate(out))

    def test_a_missing_field_is_recorded_not_raised(self):
        out = good_result()
        del out["records"]["S2"]["rotations"][1]["theta"]
        del out["records"]["S3"]["rotations"][0]["screen"]["se"]
        problems = run.validate(out)
        self.assertTrue(any("missing ['theta']" in p for p in problems))
        self.assertTrue(any("screen.se" in p for p in problems))

    def test_nonfinite_and_negative_standard_errors_are_caught(self):
        out = good_result()
        out["records"]["S4"]["rotations"][2]["theta"] = float("nan")
        out["records"]["S12"]["rotations"][0]["screen"]["se"] = -1e-6
        problems = run.validate(out)
        self.assertIn("S4 rotation 2: non-finite value", problems)
        self.assertIn("S12 rotation 0: negative standard error", problems)

    def test_no_return_and_missing_records_are_caught(self):
        out = good_result()
        out["return_kind"], out["estimate"] = "no_return", None
        self.assertIn("return kind no_return", run.validate(out))
        self.assertIn("no finite estimate", run.validate(out))
        self.assertEqual(run.validate({}), ["no split records"])

    def test_run_one_records_a_crash_instead_of_raising(self):
        out = run.run_one({"A": np.zeros(3)}, T0 + 20)
        self.assertFalse(out["valid"])
        self.assertEqual(out["problems"], ["exception"])
        self.assertIn("Traceback", out["traceback"])


class Verdict(unittest.TestCase):
    def ok(self, **over):
        base = {"valid": True, "problems": [], "estimate": 1.01, "return_kind": "unique_largest",
                "retained": ["S1", "S2", "S4"]}
        return base | over

    def test_pass_and_the_three_ways_to_fail(self):
        self.assertTrue(run.verdict(self.ok(), self.ok(estimate=1.02))["pass"])
        self.assertFalse(run.verdict(self.ok(), self.ok(estimate=1.04))["pass"])
        self.assertFalse(run.verdict(self.ok(estimate=0.85), self.ok(estimate=0.855))["pass"])
        self.assertFalse(run.verdict(self.ok(), self.ok(retained=["S1", "S3", "S12"]))["pass"])
        self.assertTrue(run.verdict(self.ok(), self.ok(retained=["S1", "S2"]))["pass"])
        self.assertFalse(run.verdict(self.ok(), {"valid": False, "problems": ["exception"]})["pass"])


class FrozenReaderGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.saved = im.BANK_DIR
        im.BANK_DIR = self.dir
        for name in ("reader_frozen.pt", "reader_frozen.json", "split_rows.npz", "manifest.json"):
            (self.dir / name).write_text(name)

    def tearDown(self):
        im.BANK_DIR = self.saved
        self.tmp.cleanup()

    def write_final(self, passed: bool, digests=None):
        payload = {"table": {"overall": {"n": 1}}, "gate": {"pass": passed, "failing": [] if passed else [["x", 1, 2]]},
                   "excluded": {}, "digests": digests if digests is not None else run.current_digests()}
        (self.dir / "reader_final_check.json").write_text(json.dumps(payload))

    def test_missing_failed_or_stale_final_check_all_stop_the_run(self):
        with self.assertRaises(SystemExit) as cm:
            run.load_frozen_reader()
        self.assertIn("no final check", str(cm.exception))
        self.write_final(False)
        with self.assertRaises(SystemExit) as cm:
            run.load_frozen_reader()
        self.assertIn("failed its final check", str(cm.exception))
        self.write_final(True)
        (self.dir / "reader_frozen.pt").write_text("a different reader")
        with self.assertRaises(SystemExit) as cm:
            run.load_frozen_reader()
        self.assertIn("files changed", str(cm.exception))


class RankTest(unittest.TestCase):
    def diffs(self, cnn, noise):
        return {"cnn": cnn} | {f"noise_{k + 1}": v for k, v in enumerate(noise)}

    def test_reader_that_always_moves_the_answer_most_is_rejected(self):
        out = run.rank_verdict([self.diffs(0.05, [0.01, 0.02, 0.015, 0.03, 0.005])] * 3)
        self.assertEqual(out["rank_sum"], 18.0)
        self.assertFalse(out["pass"])

    def test_reader_like_noise_passes_and_ties_are_averaged(self):
        out = run.rank_verdict([self.diffs(0.01, [0.01, 0.02, 0.015, 0.03, 0.005]),
                                self.diffs(0.001, [0.01, 0.02, 0.015, 0.03, 0.005]),
                                self.diffs(0.04, [0.01, 0.02, 0.015, 0.03, 0.005])])
        self.assertEqual(out["reader_ranks"], [2.5, 1.0, 6.0])
        self.assertTrue(out["pass"])

    def test_the_stated_calibration_is_exact(self):
        import itertools
        sums = [sum(r) for r in itertools.product(range(1, 7), repeat=3)]
        self.assertEqual(sum(s >= run.RANK_SUM_REJECT for s in sums), 10)
        self.assertEqual(len(sums), 216)

    def test_missing_noise_draws_or_nonfinite_values_fail(self):
        self.assertFalse(run.rank_verdict([self.diffs(0.01, [0.01, 0.02])] * 3)["pass"])
        self.assertFalse(run.rank_verdict([self.diffs(float("nan"), [0.01, 0.02, 0.015, 0.03, 0.005])] * 3)["pass"])

    def test_seed_conditions(self):
        ok = {"valid": True, "problems": [], "estimate": 1.0, "retained": ["S1", "S2"]}
        runs = {"exact": ok, "cnn": dict(ok, estimate=1.03)} | {f"noise_{k}": dict(ok, estimate=1.0 + 0.01 * k)
                                                                for k in range(1, 6)}
        cond = run.seed_conditions(runs)
        self.assertTrue(cond["pass"])
        self.assertAlmostEqual(cond["abs_diff_from_exact"]["cnn"], 0.03)
        runs["noise_3"] = {"valid": False, "problems": ["exception"]}
        self.assertFalse(run.seed_conditions(runs)["pass"])          # an invalid reference run fails the seed
        runs["noise_3"] = dict(ok)
        runs["cnn"] = dict(ok, estimate=0.85)
        self.assertFalse(run.seed_conditions(runs)["pass"])          # the reader input misses the truth

    def test_noise_reference_is_clipped_like_a_reading(self):
        levels = {"W1": np.array([[0], [14], [7]])}
        view = run.noisy_view({"W1": None, "A": 1}, levels, 5.0, T0 + 50, 1)
        self.assertTrue(((view["W1"] >= 0) & (view["W1"] <= 14)).all())
        again = run.noisy_view({"W1": None, "A": 1}, levels, 5.0, T0 + 50, 1)
        self.assertTrue(np.array_equal(view["W1"], again["W1"]))
        other = run.noisy_view({"W1": None, "A": 1}, levels, 5.0, T0 + 50, 2)
        self.assertFalse(np.array_equal(view["W1"], other["W1"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
