"""Checks of the SCM-4 stage runner, its pixel CNN comparator and its summariser.

The pixel CNN is exercised on stand-in images, so these checks need neither the Shapes3D file nor the frozen
reader.  Seeds come from the test namespace 98,600,000-98,699,999.  Run: python3 -B test_scm4_run.py
"""
from __future__ import annotations

import time
import unittest

import numpy as np
import torch

import family_v2_dgp as g
import family_v2_images as im
import run_family_v2 as rf
import run_scm4 as rs
import scm4_pixel_cnn as px
import summarize_scm4 as ss

T0 = 98_600_000


class Protocol(unittest.TestCase):
    def test_seeds_are_new_and_disjoint(self):
        dev = [t["seed"] for t in rs.tasks_for("dev")]
        conf = [t["seed"] for t in rs.tasks_for("confirm")]
        self.assertEqual((len(dev), len(conf)), (3, 20))
        self.assertFalse(set(dev) & set(conf))
        used = {t["seed"] for stage in ("dev", "confirm") for t in rf.tasks_for(stage)}
        self.assertFalse((set(dev) | set(conf)) & used)
        for s in dev + conf:                              # outside the test and rotation ranges as well
            self.assertFalse(g.TEST_SEED <= s < g.TEST_SEED + 100_000)
            self.assertNotEqual(s // 100_000, g.ROTATION_SEED // 100_000)

    def test_fold_seed_matches_the_sealed_replay(self):
        self.assertEqual(rs.FOLD_SEED_OFFSET, 7)          # summarize_family_v2.replay recomputes the radius with +7

    def test_sources_exist_and_include_the_reader_code(self):
        for name in rs.SOURCES:
            self.assertTrue((rs.HERE / name).exists(), name)
        self.assertIn("family_v2_images.py", rs.SOURCES)
        self.assertIn("scm4_pixel_cnn.py", rs.SOURCES)

    def test_a_threshold_above_the_cap_or_missing_is_refused(self):
        for bad in (5e-3, 0.0, None, -1e-3):
            with self.assertRaises(SystemExit) as cm:
                rs.freeze("confirm", bad)
            self.assertIn("2.0e-3 cap", str(cm.exception))


def stand_in(n_rows: int, seed: int):
    """A bank of stand-in images whose bright band encodes a level, plus six image indices per row."""
    rng = np.random.default_rng(seed)
    bank_levels = np.arange(60) % im.LEVELS
    bank = rng.integers(0, 40, size=(60, 64, 64, 3), dtype=np.uint8)
    for i, lv in enumerate(bank_levels):
        bank[i, :, 4 + int(lv) * 4:7 + int(lv) * 4, :] = 255
    src = rng.integers(0, 60, size=(n_rows, 6))
    signal = bank_levels[src].mean(1) - 7.0
    x = rng.standard_normal((n_rows, 2))
    a = (rng.random(n_rows) < 1 / (1 + np.exp(-0.4 * signal))).astype(float)
    y = 1.0 * a - 0.5 * signal + 0.3 * rng.standard_normal(n_rows)
    return bank, src, x, a, y


class PixelCnn(unittest.TestCase):
    def setUp(self):
        self.saved = dict(px.CONFIG)
        torch.manual_seed(T0)
        self.reader_state = {k: v.detach().clone() for k, v in im.build().state_dict().items()}

    def tearDown(self):
        px.CONFIG.clear()
        px.CONFIG.update(self.saved)

    def test_first_projection_row_starts_as_the_reader_output(self):
        net = px.PixelNet(self.reader_state, 2, 6)
        reader = im.build()
        reader.load_state_dict(self.reader_state)
        pixels = torch.rand(5, 3, 64, 64)
        with torch.no_grad():
            want = reader(pixels).squeeze(1)
            got = net.project(net.tower(pixels))[:, 0]
        self.assertTrue(torch.allclose(want, got, atol=1e-6))

    def test_it_trains_and_returns_a_finite_estimate(self):
        bank, src, x, a, y = stand_in(240, T0 + 1)
        px.CONFIG.update(epochs=1, batch_rows=32, eval_rows=60)
        out = px.estimate(bank, src, x, a, y, self.reader_state, T0 + 2, torch.device("cpu"))
        self.assertTrue(out["finite"])
        self.assertTrue(np.isfinite(out["ate"]))
        self.assertEqual(len(out["rotations"]), 4)
        for r in out["rotations"]:
            self.assertGreaterEqual(r["outside_clip_before"], 0.0)

    def test_the_time_cap_stops_it_and_is_reported(self):
        bank, src, x, a, y = stand_in(240, T0 + 3)
        px.CONFIG.update(epochs=1, batch_rows=32, max_seconds=0.0)
        out = px.estimate(bank, src, x, a, y, self.reader_state, T0 + 4, torch.device("cpu"))
        self.assertFalse(out["finite"])
        self.assertIsNone(out["ate"])
        self.assertIn("time cap", out["reason"])

    def test_pixels_are_the_bank_images_in_row_order(self):
        bank, src, *_ = stand_in(10, T0 + 5)
        got = px._pixels(bank, src[:3], torch.device("cpu"))
        want = im.to_input(torch.as_tensor(bank[src[:3].ravel()]))
        self.assertTrue(torch.equal(got, want))


def fake_shard(seed: int, pixel, problems=None) -> dict:
    return {"task": {"seed": seed}, "pixel_cnn": {"ate": pixel}, "probe_problems": problems or [],
            "baselines": {k: 0.5 for k in ("naive", "X", "raw", "summary", "oracle")},
            "proximal": {k: 1.0 for k in ss.sm.PROXIMAL}, "cevae": {"ate": 0.9},
            "reading": {"W1": {"n": 10, "nonfinite": 0, "adjacent": 1, "far": 0}}}


class Summary(unittest.TestCase):
    def test_pixel_cnn_joins_the_comparators_and_missing_values_are_counted(self):
        table = ss.comparators([fake_shard(1, 0.7), fake_shard(2, None), fake_shard(3, 0.6)])
        self.assertEqual(table["pixel_cnn"]["finite"], 2)
        self.assertAlmostEqual(table["pixel_cnn"]["mae"], 0.35)
        self.assertIn("cevae", table)

    def test_reading_totals_add_up(self):
        self.assertEqual(ss.reading_totals([fake_shard(1, 0.7), fake_shard(2, 0.7)]),
                         {"n": 20, "nonfinite": 0, "adjacent": 2, "far": 0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
