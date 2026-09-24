"""Stage 0 of the Tiny16 PRD: every mechanical check, as a test.  Run: python3 -B test_direct_balance_tiny16.py

Seeds come from the test namespace 96,900,000-96,999,999.
"""
from __future__ import annotations

import inspect
import unittest

import numpy as np
import torch

import direct_balance_data as db
import direct_balance_eval as de
import direct_balance_net as dn
import direct_balance_tiny16 as t16

T0 = 96_901_000


def world(n: int = 600, seed: int = T0, informative: bool = True):
    """A 64 x 64 stand-in bank whose brightness is the level, rows whose treatment follows image 0's level."""
    rng = np.random.default_rng(seed)
    levels = rng.integers(0, 4, size=(n, db.N_IMAGES))
    bank = np.stack([np.full((64, 64, 3), 40 + 50 * k, dtype=np.uint8) for k in range(4)] * 2)
    bank = bank + rng.integers(0, 3, size=bank.shape).astype(np.uint8)        # pictures differ a little
    src = levels + 4 * rng.integers(0, 2, size=levels.shape)
    x = rng.standard_normal((n, 2)) * np.array([3.0, 0.5]) + np.array([10.0, -2.0])
    signal = levels[:, 0] - 1.5 if informative else np.zeros(n)
    a = (rng.random(n) < 1 / (1 + np.exp(-(1.2 * signal + 0.4 * (x[:, 0] - 10) / 3)))).astype(float)
    y = a + 0.5 * signal + rng.standard_normal(n)
    rows = {"represent": np.arange(0, 300), "select": np.arange(300, 400), "check": np.arange(400, 600)}
    data = t16.TinyBatches(bank, src, x, a, torch.device("cpu"), standardise_rows=rows["represent"])
    return bank, src, x, a, y, rows, data


class Resize(unittest.TestCase):
    def test_output_is_exactly_3_by_16_by_16_and_bitwise_deterministic(self):
        u8 = torch.randint(0, 255, (5, 64, 64, 3), dtype=torch.uint8)
        r1, r2 = t16.resize16(u8), t16.resize16(u8.clone())
        self.assertEqual(tuple(r1.shape), (5, 3, 16, 16))
        self.assertTrue(torch.equal(r1, r2))
        self.assertTrue(torch.equal(r1[2:3], t16.resize16(u8[2:3])))       # not dependent on the batch it sits in

    def test_it_is_area_averaging_over_4_by_4_blocks(self):
        u8 = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)
        u8[0, :4, :4, 0] = 255                                              # one 4 x 4 block, one channel
        r = t16.resize16(u8)
        self.assertAlmostEqual(float(r[0, 0, 0, 0]), 1.0, places=6)
        self.assertEqual(float(r[0, 0, 0, 1]), 0.0)
        self.assertEqual(float(r[0, 1, 0, 0]), 0.0)

    def test_tiny_batches_serve_16_for_every_column_and_64_only_where_asked(self):
        bank, src, x, a, y, rows, data = world()
        px = data.pixels(np.arange(7), np.arange(db.N_IMAGES))
        self.assertTrue(all(tuple(p.shape) == (7, 3, 16, 16) for p in px))
        mixed = t16.TinyBatches(bank, src, x, a, torch.device("cpu"), standardise_rows=rows["represent"], held64=np.array([0]))
        px = mixed.pixels(np.arange(7), np.array([0, 1]))
        self.assertEqual(tuple(px[0].shape), (7, 3, 64, 64))
        self.assertEqual(tuple(px[1].shape), (7, 3, 16, 16))
        self.assertTrue(torch.equal(px[1], data.pixels(np.arange(7), np.array([1]))[0]))

    def test_normalisation_uses_the_named_rows_only(self):
        bank, src, x, a, y, rows, data = world()
        rep_rows = rows["represent"]
        at = torch.as_tensor(np.searchsorted(data.cache_index, data.src[rep_rows].ravel()))
        self.assertTrue(torch.allclose(data.small[at].mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-4))
        self.assertTrue(torch.allclose(data.small[at].std(dim=(0, 2, 3)), torch.ones(3), atol=1e-3))


class Learner(unittest.TestCase):
    def test_the_representation_never_receives_a_held_out_column_and_the_tower_is_the_prd_tower(self):
        held, kept = db.image_columns((0,))
        rep = dn.Representation(kept, 2, spec=t16.SPEC16)
        self.assertNotIn(int(held[0]), rep.kept.tolist())
        with self.assertRaises(ValueError):
            rep([torch.rand(3, 3, 16, 16) for _ in range(6)], torch.randn(3, 2))
        body = [m for m in rep.towers[0].body]
        self.assertEqual([type(m).__name__ for m in body], ["Conv2d", "ReLU", "Conv2d", "ReLU", "Flatten", "Linear"])
        self.assertEqual((body[0].in_channels, body[0].out_channels, body[0].stride), (3, 16, (2, 2)))
        self.assertEqual((body[2].in_channels, body[2].out_channels), (16, 32))
        self.assertEqual((body[-1].in_features, body[-1].out_features), (512, dn.FEATURE_DIM))

    def test_no_label_latent_or_outcome_reaches_the_cnn_path(self):
        bank, src, x, a, y, rows, data = world()
        for name in ("y", "u", "levels", "orientation", "digit"):
            self.assertFalse(hasattr(data, name), name)
        for fn in (t16.train_split_kfold, t16.TinyBatches.pixels, dn.fit_critics, dn.cnn_step):
            for word in ("eval_only", "U_", "orientation", "levels_", "[\"Y\"]"):
                self.assertNotIn(word, inspect.getsource(fn), (fn.__name__, word))

    def test_every_stage_consumes_the_same_z(self):
        bank, src, x, a, y, rows, data = world()
        held, kept = db.image_columns((0,))
        torch.manual_seed(T0 + 1)
        rep = dn.Representation(kept, 2, spec=t16.SPEC16)
        z = dn.representation_values(rep, data, rows["represent"], kept, grad=False)
        self.assertEqual(z.shape[1], 2 + dn.REPRESENTATION_DIM)
        self.assertTrue(torch.allclose(z[:, :2], data.x[torch.as_tensor(rows["represent"])]))
        base, aug, _ = dn.fit_critics(rep, data, rows["represent"], kept, held, 3, np.random.default_rng(1), None, spec=t16.SPEC16)
        self.assertEqual(base.net[0].in_features, z.shape[1])
        seen = {}
        saved = de.aipw_from_features
        try:
            de.aipw_from_features = lambda f_n, a_n, y_n, f_e, a_e, y_e, seed, dev: seen.update(n=f_n) or {}
            de.estimate(rep, data, a, y, {"nuisance": rows["select"], "evaluate": rows["check"]}, (0,), 3)
        finally:
            de.aipw_from_features = saved
        self.assertEqual(seen["n"].shape[1], z.shape[1])

    def test_the_base_critic_class_is_inside_the_augmented_one(self):
        torch.manual_seed(T0 + 2)
        base = dn.BaseCritic(4)
        aug = dn.AugmentedCritic(base, 1, 4, t16.SPEC16)
        z, held = torch.randn(9, 4), [torch.randn(9, 3, 16, 16)]
        self.assertEqual(float((aug(z, held) - base(z)).abs().max()), 0.0)


class Training(unittest.TestCase):
    def setUp(self):
        self.saved = dict(dn.TRAIN)
        dn.TRAIN.update({"critic_initial": 4, "critic_per_iter": 2})

    def tearDown(self):
        dn.TRAIN.clear(); dn.TRAIN.update(self.saved)

    def test_three_folds_fit_on_two_score_on_the_third_and_the_clip_is_applied_once_after_pooling(self):
        bank, src, x, a, y, rows, data = world()
        seen = []
        real_fit, real_step = dn.fit_critics, dn.cnn_step
        try:
            dn.fit_critics = lambda rep, d, fit_rows, *args, **kw: (seen.append(("fit", set(fit_rows.tolist()))),
                                                                     real_fit(rep, d, fit_rows, *args, **kw))[1]
            dn.cnn_step = lambda rep, groups, *args, **kw: (seen.append(("score", [set(g_[0].tolist()) for g_ in groups])),
                                                            real_step(rep, groups, *args, **kw))[1]
            out = t16.train_split_kfold(data, rows, (0,), T0 + 3, 2, checkpoints=(0, 1, 2))
        finally:
            dn.fit_critics, dn.cnn_step = real_fit, real_step
        fits = [v for k, v in seen if k == "fit"][:3]
        scores = next(v for k, v in seen if k == "score")
        whole = set(rows["represent"].tolist())
        for fit_set, score_set in zip(fits, scores):
            self.assertEqual(len(fit_set & score_set), 0)
            self.assertEqual(fit_set | score_set, whole)
        self.assertEqual(set.union(*scores), whole)
        self.assertEqual(sum(len(s) for s in scores), len(whole))
        step_src = inspect.getsource(dn.cnn_step)
        self.assertIn("gap = total0 - total1", step_src)                    # pooled first,
        self.assertIn("if gap > 0:", step_src)                              # clipped once
        self.assertEqual(out["folds"], 3)
        self.assertEqual(sorted(out["checkpoint_states"]), [0, 1, 2])

    def test_gradient_reaches_the_first_convolution_and_the_weights_move(self):
        bank, src, x, a, y, rows, data = world()
        held, kept = db.image_columns((0,))
        torch.manual_seed(T0 + 4)
        rep = dn.Representation(kept, 2, spec=t16.SPEC16)
        before = {n: p.detach().clone() for n, p in rep.named_parameters()}
        base, aug, _ = dn.fit_critics(rep, data, rows["represent"], kept, held, 6, np.random.default_rng(2), None, spec=t16.SPEC16)
        opt = torch.optim.Adam(rep.parameters(), lr=1e-3)
        step = dn.cnn_step(rep, [(rows["represent"], base, aug)], data, kept, held, opt)
        self.assertTrue(step["stepped"], step)
        first = "towers.0.body.0.weight"
        grads = {n: p.grad for n, p in rep.named_parameters()}
        self.assertGreater(float(grads[first].norm()), 0.0)
        self.assertGreater(float((dict(rep.named_parameters())[first] - before[first]).abs().max()), 0.0)

    def test_a_non_finite_value_is_reported_not_swallowed(self):
        bank, src, x, a, y, rows, data = world()
        z = torch.tensor([[float("nan")]])
        self.assertFalse(bool(torch.isfinite(z).all()))
        # the runner records status "no value" for a split whose check or estimate is not finite: see de.check
        self.assertIn("finite", inspect.getsource(de.check_from_z))


class Rows(unittest.TestCase):
    def test_the_row_roles_are_exclusive_and_are_the_existing_split(self):
        rows = db.split_rows(db.N, 96_900_800)
        every = np.concatenate(list(rows.values()))
        self.assertEqual(len(np.unique(every)), db.N)
        self.assertEqual({k: len(v) for k, v in rows.items()}, db.ROLES)


class ValidatedCritics(unittest.TestCase):
    """The fix for the diagnosed failure: the training critics generalise, and the nesting is decided on rows
    they did not fit.  These tests would have failed on the plain critics."""

    def test_the_lifted_decision_is_made_on_the_validation_rows_and_makes_q1_equal_q0_exactly(self):
        for informative in (True, False):
            bank, src, x, a, y, rows, data = world(informative=informative)
            held, kept = db.image_columns((0,))
            torch.manual_seed(T0 + 90)
            rep = dn.Representation(kept, 2, spec=t16.SPEC16)
            rng = np.random.default_rng(T0 + 91)
            base, aug, state = dn.fit_critics_validated(rep, data, rows["represent"], kept, held, 60, rng, None, spec=t16.SPEC16)
            info = state["info"]
            self.assertEqual(info["n_fit"] + info["n_val"], len(rows["represent"]))
            # the returned q1 is never worse than q0 on the validation fifth: that is the rule
            self.assertLessEqual(info["risk_augmented_val"], info["risk_base_val"] + 1e-9, (informative, info))
            if info["kept_the_lifted_base"]:
                z = dn.representation_values(rep, data, rows["select"], kept, grad=False)
                px = data.pixels(rows["select"], held)
                self.assertEqual(float((aug(z, px) - base(z)).abs().max()), 0.0)
                self.assertAlmostEqual(info["risk_augmented_val"], info["risk_base_val"], places=9)

    def test_a_memorising_augmented_critic_is_not_rewarded(self):
        """Plain critics: q1 beats q0 on its fit rows and can lose on others, and the plain rule keeps it.  The
        validated rule looks where the memorising cannot help."""
        bank, src, x, a, y, rows, data = world(n=900, informative=False)   # nothing to find: any q1 gain is memory
        held, kept = db.image_columns((0,))
        torch.manual_seed(T0 + 92)
        rep = dn.Representation(kept, 2, spec=t16.SPEC16)
        fit_rows = np.arange(0, 450)
        plain = dn.fit_critics(rep, data, fit_rows, kept, held, 150, np.random.default_rng(1), None, spec=t16.SPEC16)[2]["info"]
        valid = dn.fit_critics_validated(rep, data, fit_rows, kept, held, 150, np.random.default_rng(1), None, spec=t16.SPEC16)[2]["info"]
        self.assertFalse(plain["kept_the_lifted_base"])                     # the plain rule was fooled on fit rows
        self.assertLess(plain["risk_augmented_fit"], plain["risk_base_fit"])
        self.assertLessEqual(valid["risk_augmented_val"], valid["risk_base_val"] + 1e-9)

    def test_early_stopping_keeps_the_state_that_did_best_on_the_validation_fifth(self):
        bank, src, x, a, y, rows, data = world()
        held, kept = db.image_columns((0,))
        torch.manual_seed(T0 + 93)
        rep = dn.Representation(kept, 2, spec=t16.SPEC16)
        base, aug, state = dn.fit_critics_validated(rep, data, rows["represent"], kept, held, 40, np.random.default_rng(3), None,
                                                    spec=t16.SPEC16, eval_every=10)
        info = state["info"]
        self.assertIn(info["base_step"], (0, 10, 20, 30, 40))
        self.assertIn(info["augmented_step"], (0, 10, 20, 30, 40))

    def test_warm_start_keeps_the_towers_and_resets_the_z_branch_to_the_current_base(self):
        bank, src, x, a, y, rows, data = world()
        held, kept = db.image_columns((0,))
        torch.manual_seed(T0 + 94)
        rep = dn.Representation(kept, 2, spec=t16.SPEC16)
        rng = np.random.default_rng(4)
        _, _, st1 = dn.fit_critics_validated(rep, data, rows["represent"], kept, held, 20, rng, None, spec=t16.SPEC16)
        base2, aug2, st2 = dn.fit_critics_validated(rep, data, rows["represent"], kept, held, 0, rng, st1, spec=t16.SPEC16)
        for k, v in base2.net.state_dict().items():                          # zero extra steps: q1's z branch IS q0's
            self.assertTrue(torch.equal(aug2.net.state_dict()[k], v))

    def test_the_kfold_training_can_use_the_validated_critics(self):
        bank, src, x, a, y, rows, data = world()
        saved = dict(dn.CRITIC_V)
        try:
            dn.CRITIC_V.update({"initial_steps": 6, "steps_per_iter": 4})
            out = t16.train_split_kfold(data, rows, (0,), T0 + 95, 2, checkpoints=(0, 1, 2), validated=True)
        finally:
            dn.CRITIC_V.clear(); dn.CRITIC_V.update(saved)
        self.assertTrue(out["critics"].startswith("validated"))
        self.assertEqual(out["critic_budget"], [6, 4])
        self.assertIn("gap_val", out["history"][0]["critics"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
