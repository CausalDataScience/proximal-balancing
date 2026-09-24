"""Checks of the SCM-4e runner and summariser (spec section 14).

Seeds come from the test namespace 98,500,000-98,599,999.  Run: python3 -B test_scm4e.py
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
import family_v2_probe as pr
import run_family_v2 as rf
import run_scm4 as rs
import run_scm4e as rse
import summarize_scm4e as sse

T0 = 98_500_000


def stand_in_bank(n: int = 90, seed: int = T0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(n, 64, 64, 3), dtype=np.uint8)


class Embedding(unittest.TestCase):
    def test_embed_is_the_last_hidden_layer_in_the_order_asked_for(self):
        torch.manual_seed(T0 + 1)
        net, bank = im.build(), stand_in_bank()
        rows = np.array([5, 2, 2, 80, 0, 5])
        got = rse.embed(net, bank, rows, torch.device("cpu"))
        self.assertEqual(got.shape, (6, rse.EMBED_DIM))
        tower = torch.nn.Sequential(*list(net.children())[:-1])
        with torch.no_grad():
            want = tower(im.to_input(torch.as_tensor(bank[rows]))).numpy()
        np.testing.assert_allclose(got, want, atol=1e-6)
        self.assertTrue(np.array_equal(got[1], got[2]))

    def test_the_network_output_is_linear_in_the_embedding(self):
        """The decoded angle of SCM-4 is one fixed linear read-out of what SCM-4e hands over, so SCM-4e gives the
        learner at least that much, and does not tell it which read-out it is."""
        torch.manual_seed(T0 + 2)
        net, bank = im.build(), stand_in_bank()
        rows = np.arange(40)
        emb = rse.embed(net, bank, rows, torch.device("cpu"))
        last = list(net.children())[-1]
        decoded = emb @ last.weight.detach().numpy().astype(np.float64).T + last.bias.detach().numpy()
        with torch.no_grad():
            want = net(im.to_input(torch.as_tensor(bank[rows]))).numpy()
        np.testing.assert_allclose(decoded, want, atol=1e-5)


class Layout(unittest.TestCase):
    def test_block_layout_and_new_seeds(self):
        self.assertEqual(rse.LEVEL.d_blocks, (256, 256, 256, 256, 512))
        self.assertEqual((rse.LEVEL.k, rse.FOLD_SEED_OFFSET, rse.IMAGE_STREAM_OFFSET), (2, 7, rs.IMAGE_STREAM_OFFSET))
        mine = {t["seed"] for stage in ("dev", "confirm") for t in rse.tasks_for(stage)}
        earlier = {t["seed"] for stage in ("dev", "confirm") for t in rf.tasks_for(stage) + rs.tasks_for(stage)}
        self.assertEqual(len(mine), 23)
        self.assertFalse(mine & earlier)

    def test_sources_exist(self):
        for name in rse.SOURCES:
            self.assertTrue((rse.HERE / name).exists(), name)

    def test_same_seed_gives_the_same_images_as_scm4(self):
        data = g.generate(g.LEVELS["SCM-1"], 400, T0 + 10)
        groups = [np.arange(k * 50, k * 50 + 50) for k in range(im.LEVELS)]
        _, src = rse.draw_sources(data, groups, T0 + 10)
        saved = im.reading
        try:
            im.reading = lambda net, images, rows, dev, chunk=2000: np.zeros(len(np.asarray(rows).ravel()))
            _, _, src4, _ = rs.image_view(data, None, None, None, groups, T0 + 10)
        finally:
            im.reading = saved
        self.assertTrue(np.array_equal(np.column_stack([src[k] for k in src]), src4))

    def test_the_sealed_learner_accepts_embedding_sized_blocks(self):
        rng = np.random.default_rng(T0 + 20)
        n = 1_600
        data = g.generate(g.LEVELS["SCM-1"], n, T0 + 21)
        view = g.learner_view(data)
        for key in ("W1", "W2", "W3", "W4", "W5"):
            cols = [np.outer(view[key][:, c], rng.standard_normal(256)) + 0.05 * rng.standard_normal((n, 256))
                    for c in range(view[key].shape[1])]
            view[key] = np.column_stack(cols)
        out = pr.run_dataset(view, rse.LEVEL, 1.0, T0 + 22)
        self.assertEqual(len(out["split_names"]), 30)
        self.assertEqual(out["level"], "SCM-4e")


class Merge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.saved = (dict(rse.SHARDS), dict(rse.PIXEL_SHARDS))
        rse.SHARDS["dev"], rse.PIXEL_SHARDS["dev"] = root / "dev", root / "dev_pixel"
        for d in (rse.SHARDS["dev"], rse.PIXEL_SHARDS["dev"]):
            d.mkdir()
        for seed in (1, 2):
            (rse.SHARDS["dev"] / f"SCM-4e_seed{seed}.json").write_text(json.dumps(
                {"protocol_sha256": "p", "task": {"seed": seed},
                 "embedding_check": {"W1": {"components_kept": 4, "level_r2_from_kept_components": [0.99]}}}))
        (rse.PIXEL_SHARDS["dev"] / "SCM-4e_seed1.json").write_text(json.dumps(
            {"protocol_sha256": "p", "task": {"seed": 1}, "pixel_cnn": {"finite": True, "ate": 0.7}}))

    def tearDown(self):
        rse.SHARDS.update(self.saved[0]); rse.PIXEL_SHARDS.update(self.saved[1])
        self.tmp.cleanup()

    def test_a_missing_pixel_shard_is_scored_as_missing(self):
        shards = sse.merged("dev")
        self.assertEqual(shards[0]["pixel_cnn"]["ate"], 0.7)
        self.assertIsNone(shards[1]["pixel_cnn"]["ate"])
        self.assertEqual(sse.embedding_summary(shards)["components_kept_min_max"], [4, 4])

    def test_a_pixel_shard_from_another_protocol_is_refused(self):
        (rse.PIXEL_SHARDS["dev"] / "SCM-4e_seed2.json").write_text(json.dumps(
            {"protocol_sha256": "other", "task": {"seed": 2}, "pixel_cnn": {"finite": True, "ate": 0.7}}))
        with self.assertRaises(SystemExit):
            sse.merged("dev")


if __name__ == "__main__":
    unittest.main(verbosity=2)
