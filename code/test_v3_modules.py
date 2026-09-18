"""The three modules the v3 confirmations rest on, and the properties each must have."""
import unittest

import numpy as np
import torch

import learned_target as lt
import minimal_suite_common as c
import probe_full_target as ft
import probe_highdim_w as hd
import probe_image_w as im
import probe_pixel_critic as pc
import scm6_population as sp

DEV = ft.device()


class Scm6Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = hd.generate(5000, 94000000)
        view = hd.observed_view(data, include_outcome=True)
        sl = lambda i: {k: ([b[i] for b in v] if k == "H" else v[i]) for k, v in view.items()}
        emb = hd.fit_embedding(sl(np.arange(1200)), sl(np.arange(1200, 1500)))
        cls.phi, cls.scr = sl(np.arange(1500, 3000)), sl(np.arange(3000, 5000))
        cls.k_phi, cls.k_scr = hd.apply_embedding(emb, cls.phi), hd.apply_embedding(emb, cls.scr)
        ft.RESTARTS, ft.MAX_EPOCHS = 1, 40


class TestResidualCriticContainment(Scm6Fixture):
    def test_zero_branch_reproduces_the_base_exactly(self):
        z = self.k_phi[:, :3]
        base = ft.fit_base(z, self.phi["A"], 1, DEV)
        aug = ft.AugmentedCritic(base["model"], 3, 4).to(DEV).eval()
        zt = torch.as_tensor(base["scaler"](z), dtype=torch.float32, device=DEV)
        tt = torch.randn(len(z), 4, device=DEV)
        with torch.no_grad():
            np.testing.assert_allclose(aug(zt, tt).cpu().numpy(), base["model"](zt).cpu().numpy(), atol=1e-6)

    def test_augmented_validation_risk_never_exceeds_the_base(self):
        z, t = hd.representation(self.k_phi, self.phi, (1,), 0.2245), hd.heldout(self.phi, (1,))
        base = ft.fit_base(z, self.phi["A"], 2, DEV)
        aug = ft.fit_augmented(base, z, t, self.phi["A"], 2, DEV)
        self.assertTrue(ft.nesting_check(base, aug, z, t, self.phi["A"], DEV)["nested"])

    def test_output_stays_inside_the_declared_bounds(self):
        z = self.k_phi[:, :3]
        base = ft.fit_base(z, self.phi["A"], 3, DEV)
        with torch.no_grad():
            p = base["model"](torch.as_tensor(base["scaler"](z), dtype=torch.float32, device=DEV)).cpu().numpy()
        self.assertGreaterEqual(p.min(), ft.P_LO)
        self.assertLessEqual(p.max(), ft.P_HI)


class TestFullTargetScreen(Scm6Fixture):
    def _screen(self, split, r):
        z, t = hd.representation(self.k_phi, self.phi, split, r), hd.heldout(self.phi, split)
        zs, ts = hd.representation(self.k_scr, self.scr, split, r), hd.heldout(self.scr, split)
        base = ft.fit_base(z, self.phi["A"], 5, DEV)
        aug = ft.fit_augmented(base, z, t, self.phi["A"], 5, DEV)
        return ft.screen_full_target(base, aug, zs, ts, self.scr["A"], 1e-3, 0.05, 30, DEV)

    def test_the_bound_is_never_below_the_clipped_gap(self):
        s = self._screen((1, 4), 1.0)
        self.assertGreaterEqual(s["upper"], max(s["gap"], 0.0))

    def test_an_infeasible_split_is_rejected(self):
        self.assertFalse(self._screen((1, 4), 1.0)["pass"])

    def test_a_negative_gap_alone_does_not_pass(self):
        """A large negative point estimate leaves U = z SE, which passes only if the precision is there."""
        s = self._screen((1,), 1.0)
        if s["gap"] < 0:
            self.assertAlmostEqual(s["upper"], s["z"] * s["se"], places=12)

    def test_the_target_is_the_full_block(self):
        self.assertEqual(hd.heldout(self.phi, (1,)).shape[1], 1 + hd.DIM)
        self.assertEqual(hd.heldout(self.phi, (0, 1)).shape[1], 3 + 2 * hd.DIM)


class TestLearnedTarget(unittest.TestCase):
    def test_recovers_the_population_target_of_the_exact_representation(self):
        """With the true channel, theta at the root is 1 and at a wrong coefficient it is not."""
        big = hd.generate(60000, 424242)
        view = hd.observed_view(big, include_outcome=True)
        k = big["K_eval_only"]
        z = hd.representation(k, view, (1,), 0.2245)
        at_root = lt.learned_target(z, view["A"], view["Y"], 1, DEV)
        self.assertLess(abs(at_root["theta"] - 1.0), 0.05, at_root)
        self.assertLess(abs(at_root["theta"] - at_root["theta_linear"]), 0.05)
        # Far from balance the regression is harder and the gate never looks there; what matters is
        # that the evaluator reports a target far from 1, in the direction the population predicts.
        z = hd.representation(k, view, (1,), 1.0)
        off = lt.learned_target(z, view["A"], view["Y"], 1, DEV)
        self.assertGreater(off["theta"], 1.5, off)
        self.assertLess(abs(off["theta"] - off["theta_linear"]), 0.1)


class TestPixelCritic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lat = im.latents(200, 95000000)
        cls.images = im.block(lat, 1, np.arange(48)).astype(np.float32)

    def test_reflection_preserves_mass_and_both_centroids(self):
        flipped = pc.reflect_about_centroid(self.images)
        coords = np.arange(float(im.CANVAS))
        for a, b in zip(self.images, flipped):
            self.assertAlmostEqual(a.sum(), b.sum(), places=4)
            self.assertAlmostEqual((a.sum(0) * coords).sum() / a.sum(), (b.sum(0) * coords).sum() / b.sum(), places=2)
            self.assertAlmostEqual((a.sum(1) * coords).sum() / a.sum(), (b.sum(1) * coords).sum() / b.sum(), places=2)
        # mass is normalised to one over a few hundred pixels, so intensities are small; the change is
        # judged against the image's own peak
        rel = np.abs(self.images - flipped).max(axis=(1, 2)) / self.images.max(axis=(1, 2))
        self.assertGreater(np.median(rel), 0.2, "the arrangement must actually change")

    def test_an_untrained_trunk_sees_arrangement(self):
        torch.manual_seed(0)
        out = pc.discrimination_check(pc.PixelTrunk().to(DEV).eval(), self.images, DEV)
        self.assertTrue(out["pass"], out)

    def test_every_pixel_reaches_the_feature(self):
        """Perturbing one far-corner pixel changes the feature; global pooling would not erase it either,
        but a centroid-only reader would respond only through the centroid shift, which is tiny."""
        torch.manual_seed(0)
        trunk = pc.PixelTrunk().to(DEV).eval()
        img = self.images[:1].copy()
        bumped = img.copy(); bumped[0, 2, 2] += 0.5
        with torch.no_grad():
            f0 = trunk(torch.as_tensor(img, device=DEV)).cpu().numpy()
            f1 = trunk(torch.as_tensor(bumped, device=DEV)).cpu().numpy()
        self.assertGreater(np.abs(f0 - f1).max(), 1e-6)

    def test_image_target_carries_block_identity_and_mask(self):
        feats = np.zeros((10, im.N_BLOCKS, pc.FEATURE_DIM))
        numeric = {"X": np.zeros(10), "I": np.zeros(10), "C": np.zeros(10)}
        t = pc.image_target(feats, numeric, (1, 3))
        self.assertEqual(t.shape[1], 1 + 2 * pc.FEATURE_DIM + im.N_BLOCKS)
        np.testing.assert_array_equal(t[0, -im.N_BLOCKS:], [0, 1, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()


class TestComponentLabels(unittest.TestCase):
    def test_exploded_labels_are_joined_back_to_split_ids(self):
        import probe_structured_scm as ps
        import run_scm6_v3 as r6
        import run_scm7_v3 as r7
        labels = [c.split_id(s) for s in [(1,), (2,), (4,), (1, 2)]]
        agg = ps.aggregate(labels, np.array([1.01, 0.99, -0.44, 1.02]), 0.05)
        for mod in (r6, r7):
            got = mod.component_labels(agg)
            self.assertEqual(sorted(got), ["S1", "S12", "S2"])
            self.assertTrue(all(g in labels for g in got))
        self.assertEqual(r6.component_labels({"members": []}), [])
        self.assertEqual(r6.component_labels({"return_kind": "no_return"}), [])


class TestEvaluatorCache(unittest.TestCase):
    """The cache must reproduce a direct encoding exactly and re-encode only when the encoder changes.

    This is a property of the cache, not of the encoder, so the encoder's calibration gate is lowered
    inside the test: at fixture size the gate fails closed and would leave nothing to compare.
    """

    def test_cached_channels_equal_a_direct_encoding_and_are_reused_per_encoder(self):
        import run_scm7_v3 as r7
        saved = dict(im.TRAIN)
        try:
            im.TRAIN.update({"epochs": 3, "early_stopping": False, "scale_min": 1e-12})
            r7.FROZEN["learned_target_evaluator"] = {"n": 600, "seed_rule": "424242 + data seed",
                                                     "chunk": 250}
            lat = im.latents(500, 95000000)
            enc = im.fit_encoder(lat, np.arange(400), np.arange(400, 500), seed=95000000)
            self.assertIsNone(enc["fail_closed"], enc["fail_closed"])
            cache = r7.EvaluatorCache(95000000)
            direct = im.apply_encoder(enc, cache.big, np.arange(600))
            # chunked and whole-sample encodings differ by float32 reduction order on the GPU, about
            # one part in 1e5; anything beyond that would be a cache defect
            np.testing.assert_allclose(cache.channels(enc), direct, rtol=1e-4, atol=1e-3)
            self.assertIs(cache.channels(enc), cache.channels(enc), "same encoder must reuse the encoding")
            enc2 = im.fit_encoder(lat, np.arange(400), np.arange(400, 500), seed=95000001)
            self.assertIsNone(enc2["fail_closed"], enc2["fail_closed"])
            self.assertIsNot(cache.channels(enc2), direct, "a new encoder must re-encode")
            self.assertGreater(np.abs(cache.channels(enc2) - direct).max(), 0.0)
        finally:
            im.TRAIN.clear(); im.TRAIN.update(saved)
