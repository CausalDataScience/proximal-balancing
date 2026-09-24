"""The thirteen implementation checks of the autoencoder warm-start PRD, section 11.
Run: python3 -B test_direct_balance_ae_warmstart.py
"""
from __future__ import annotations

import inspect
import unittest

import numpy as np
import torch

import direct_balance_ae_warmstart as ae
import direct_balance_data as db
import direct_balance_eval as de
import direct_balance_net as dn
import direct_balance_tiny16 as t16
import direct_balance_v3 as v3
from test_direct_balance_tiny16 import world

T0 = 96_902_000


def small(n: int = 600, seed: int = T0):
    bank, src, x, a, y, _, _ = world(n=n, seed=seed)
    half, three = n // 2, (n * 2) // 3
    rows = {"represent": np.arange(0, half), "select": np.arange(half, three), "check": np.arange(three, n)}
    data = t16.TinyBatches(bank, src, x, a, torch.device("cpu"), standardise_rows=rows["represent"])
    held, kept = db.image_columns((0,))
    return bank, src, x, a, y, rows, data, held, kept


class One_Rows(unittest.TestCase):
    def test_the_thirty_thousand_rows_keep_their_roles_and_hashes(self):
        rows = db.split_rows(db.N, ae.DATA_SEED)
        self.assertEqual({k: len(v) for k, v in rows.items()}, db.ROLES)
        every = np.concatenate(list(rows.values()))
        self.assertEqual(len(np.unique(every)), db.N)
        again = db.split_rows(db.N, ae.DATA_SEED)
        for k in rows:
            self.assertTrue(np.array_equal(rows[k], again[k]))


class Two_AeIds(unittest.TestCase):
    def test_the_autoencoder_images_are_split_by_id_and_never_leave_the_representation_rows(self):
        bank, src, x, a, y, rows, data, held, kept = small()
        ids = ae.image_id_split(src, rows["represent"])
        self.assertEqual(len(np.intersect1d(ids["fit"], ids["val"])), 0)
        self.assertEqual(len(ids["fit"]) + len(ids["val"]), len(ids["all"]))
        inside = np.unique(src[rows["represent"]].ravel())
        self.assertEqual(len(np.setdiff1d(ids["all"], inside)), 0)
        self.assertEqual(len(ids["fit"]), int(np.floor(0.8 * len(ids["all"]))))


class Three_NoLabels(unittest.TestCase):
    def test_the_autoencoder_loader_and_the_representation_see_no_label_and_no_held_out_pixel(self):
        src = inspect.getsource(ae.train_autoencoder) + inspect.getsource(ae.image_id_split)
        for word in ("eval_only", "orientation", "levels", '["Y"]', '["A"]', "u_r2"):
            self.assertNotIn(word, src, word)
        self.assertIn("tiny.small", inspect.getsource(ae.train_autoencoder))
        held, kept = db.image_columns((0,))
        rep = dn.Representation(kept, 2, spec=t16.SPEC16)
        self.assertNotIn(int(held[0]), rep.kept.tolist())
        with self.assertRaises(ValueError):
            rep([torch.rand(2, 3, 16, 16) for _ in range(6)], torch.randn(2, 2))


class Four_Shapes(unittest.TestCase):
    def test_the_encoder_matches_the_tiny16_tower_and_the_decoder_returns_finite_pixels(self):
        torch.manual_seed(T0 + 4)
        enc, dec = dn.ImageTower(t16.SPEC16), ae.decoder()
        tower = dn.Representation(np.arange(5), 2, spec=t16.SPEC16).towers[0]
        self.assertEqual(sorted(enc.state_dict()), sorted(tower.state_dict()))
        for k in enc.state_dict():
            self.assertEqual(enc.state_dict()[k].shape, tower.state_dict()[k].shape)
        x = torch.randn(4, 3, 16, 16)
        z = enc(x)
        self.assertEqual(tuple(z.shape), (4, ae.AE["latent"]))
        out = dec(z)
        self.assertEqual(tuple(out.shape), (4, 3, 16, 16))
        self.assertTrue(torch.isfinite(out).all())
        self.assertFalse(any(isinstance(m, torch.nn.Sigmoid) for m in dec))


class Five_PairedInit(unittest.TestCase):
    def test_the_two_arms_share_a_head_and_the_ae_towers_are_independent_copies(self):
        held, kept = db.image_columns((0,))
        torch.manual_seed(T0 + 50)
        enc_state = {k: v.clone() for k, v in dn.ImageTower(t16.SPEC16).state_dict().items()}
        r = ae.build_representation(kept, 2, torch.device("cpu"), "R", None)
        w = ae.build_representation(kept, 2, torch.device("cpu"), "AE", enc_state)
        self.assertEqual(ae.sha_state(r.head.state_dict()), ae.sha_state(w.head.state_dict()))
        for tower in w.towers:
            self.assertEqual(ae.sha_state(tower.state_dict()), ae.sha_state(enc_state))
        self.assertNotEqual(ae.sha_state(r.towers[0].state_dict()), ae.sha_state(enc_state))
        with torch.no_grad():
            w.towers[0].body[0].weight.add_(1.0)                      # independent storage
        self.assertNotEqual(ae.sha_state(w.towers[0].state_dict()), ae.sha_state(w.towers[1].state_dict()))
        for p in list(w.towers[0].parameters()) + list(w.head.parameters()):
            self.assertTrue(p.requires_grad)
        with self.assertRaises(RuntimeError):
            ae.build_representation(kept, 2, torch.device("cpu"), "AE", None)


class Six_Rng(unittest.TestCase):
    def test_the_shared_draws_do_not_depend_on_the_arm_or_on_the_autoencoder(self):
        a1 = ae.seed_for("critic_init", 2, 7)
        torch.manual_seed(12345); torch.randn(1000)                    # burn the global state
        np.random.default_rng(999).integers(0, 10, 1000)
        self.assertEqual(a1, ae.seed_for("critic_init", 2, 7))
        self.assertNotEqual(a1, ae.seed_for("critic_init", 2, 8))
        self.assertNotEqual(a1, ae.seed_for("critic_batch", 2, 7))
        self.assertEqual(len(set(ae.PURPOSE.values())), len(ae.PURPOSE))
        b1 = np.random.default_rng(ae.seed_for("critic_batch", 1, 3)).integers(0, 100, 20)
        b2 = np.random.default_rng(ae.seed_for("critic_batch", 1, 3)).integers(0, 100, 20)
        self.assertTrue(np.array_equal(b1, b2))


class Seven_Clip(unittest.TestCase):
    def test_the_clip_is_applied_once_after_pooling_the_folds(self):
        src = inspect.getsource(v3.pooled_objective)
        self.assertIn("signed = (total0 - total1) / n_all", src)
        self.assertIn("torch.clamp(signed, min=0.0)", src)
        self.assertNotIn("clamp(part", src)
        bank, s_, x, a, y, rows, data, held, kept = small()
        plan = ae.fold_plan({"represent": rows["represent"], "select": rows["select"]})
        torch.manual_seed(T0 + 70)
        rep = dn.Representation(kept, 2, spec=t16.SPEC16)
        pairs = [ae.fit_critics(rep, v3.Normaliser(None, None), data, plan["fit_sets"][i], plan["val_sets"][i],
                                kept, held, 6, i, 0, None, None)[:2] for i in range(plan["folds"])]
        groups, index_of = ae.build_groups(rep, data, plan, pairs, kept, held, plan["represent"])
        z = dn.representation_values(rep, data, plan["represent"], kept, grad=False)
        norm = v3.Normaliser.fit(z, enabled=True)
        clipped, info = v3.pooled_objective(z, norm, groups, data, index_of)
        # the same number, computed directly from the three folds' risks
        n = sum(f["n"] for f in info["folds"])
        by_hand = sum(f["n"] * (f["risk_base"] - f["risk_augmented"]) for f in info["folds"]) / n
        self.assertAlmostEqual(by_hand, info["signed_gap"], places=6)
        self.assertAlmostEqual(float(clipped), max(info["signed_gap"], 0.0), places=9)


class Eight_Gradient(unittest.TestCase):
    def test_the_gradient_through_the_dynamic_statistics_matches_finite_differences(self):
        torch.set_default_dtype(torch.float64)
        try:
            bank, s_, x, a, y, rows, data, held, kept = small(seed=T0 + 80)   # `world` fixes its own row sizes
            data.x = data.x.double(); data.a = data.a.double(); data.small = data.small.double()
            plan = ae.fold_plan({"represent": rows["represent"][:120], "select": rows["select"]})
            torch.manual_seed(T0 + 81)
            rep = dn.Representation(kept, 2, spec=t16.SPEC16).double()
            pairs = [ae.fit_critics(rep, v3.Normaliser(None, None), data, plan["fit_sets"][i], plan["val_sets"][i],
                                    kept, held, 20, i, 0, None, None)[:2] for i in range(plan["folds"])]
            groups, index_of = ae.build_groups(rep, data, plan, pairs, kept, held, plan["represent"])
            z = dn.representation_values(rep, data, plan["represent"], kept, grad=False).clone().requires_grad_(True)
            loss, info = v3.pooled_objective(z, v3.Normaliser.fit(z, enabled=True), groups, data, index_of)
            self.assertGreater(info["signed_gap"], 0.0)
            loss.backward()
            g = z.grad.clone()
            eps = 1e-6
            rng = np.random.default_rng(T0 + 82)
            for _ in range(4):
                i, j = int(rng.integers(0, z.shape[0])), int(rng.integers(2, 4))
                out = []
                for sign in (+1, -1):
                    zp = z.detach().clone(); zp[i, j] += sign * eps
                    out.append(float(v3.pooled_objective(zp, v3.Normaliser.fit(zp, enabled=True), groups, data, index_of)[0]))
                fd = (out[0] - out[1]) / (2 * eps)
                self.assertLess(abs(fd - float(g[i, j])), 1e-5 + 1e-3 * abs(fd), (i, j, fd, float(g[i, j])))
        finally:
            torch.set_default_dtype(torch.float32)


class Nine_Lifecycle(unittest.TestCase):
    def test_the_statistics_returned_are_the_post_step_ones_and_a_skip_leaves_them_alone(self):
        bank, s_, x, a, y, rows, data, held, kept = small()
        plan = ae.fold_plan({"represent": rows["represent"], "select": rows["select"]})
        torch.manual_seed(T0 + 90)
        rep = dn.Representation(kept, 2, spec=t16.SPEC16)
        pairs = [ae.fit_critics(rep, v3.Normaliser(None, None), data, plan["fit_sets"][i], plan["val_sets"][i],
                                kept, held, 8, i, 0, None, None)[:2] for i in range(plan["folds"])]
        with torch.no_grad():
            before = v3.Normaliser.fit(dn.representation_values(rep, data, plan["represent"], kept, grad=False), enabled=True)
        opt = torch.optim.Adam(rep.parameters(), lr=1e-2)
        step, norm_next = ae.update(rep, data, plan, pairs, kept, held, opt, before.frozen())
        with torch.no_grad():
            truth = v3.Normaliser.fit(dn.representation_values(rep, data, plan["represent"], kept, grad=False), enabled=True)
        self.assertEqual(norm_next.record()["mean"], truth.record()["mean"])      # post-step, not pre-step
        self.assertEqual(step["normaliser_after"], norm_next.record())
        if step["stepped"]:
            self.assertNotEqual(step["normaliser_used"], step["normaliser_after"])
        # a skip must leave the statistics untouched
        b2 = dn.BaseCritic(4); a2 = dn.AugmentedCritic(b2, len(held), 4, t16.SPEC16)
        with torch.no_grad():
            a2.extra[-1].bias.fill_(6.0)
        step2, norm2 = ae.update(rep, data, plan, [(b2, a2)] * plan["folds"], kept, held, opt, norm_next)
        self.assertFalse(step2["stepped"])
        self.assertEqual(step2["normaliser_used"], step2["normaliser_after"])


class Ten_Carry(unittest.TestCase):
    def test_carrying_the_critics_keeps_them_the_same_function_of_the_raw_h(self):
        torch.manual_seed(T0 + 100)
        base = dn.BaseCritic(4)
        aug = dn.AugmentedCritic(base, 1, 4, t16.SPEC16)
        with torch.no_grad():
            aug.extra[-1].weight.normal_(std=0.5); aug.extra[-1].bias.normal_()
        raw = torch.randn(30, 4) * torch.tensor([1.0, 1.0, 4.0, 0.3]) + torch.tensor([0.0, 0.0, -3.0, 2.0])
        old = v3.Normaliser(torch.tensor([-2.0, 1.5]), torch.tensor([3.0, 0.4]))
        new = v3.Normaliser(torch.tensor([0.5, -1.0]), torch.tensor([1.1, 2.2]))
        px = [torch.randn(30, 3, 16, 16)]
        with torch.no_grad():
            want = (base(old.apply(raw)).clone(), aug(old.apply(raw), px).clone())
        v3.retune_critic(base, old, new); v3.retune_critic(aug, old, new)
        with torch.no_grad():
            got = (base(new.apply(raw)), aug(new.apply(raw), px))
        for w_, g_ in zip(want, got):
            self.assertLess(float((w_ - g_).abs().max()), 1e-5)


class Eleven_ActualUpdates(unittest.TestCase):
    def test_only_real_updates_are_numbered_and_a_missing_one_is_not_reached(self):
        self.assertEqual(ae.ACTUAL_UPDATE_ORDINALS, (1, 3, 7))
        out = {"arm": "T", "plan": ae.fold_plan({"represent": np.arange(300), "select": np.arange(300, 400)}),
               "diag": {}}
        bank, s_, x, a, y, rows, data, held, kept = small()
        got = ae.actual_update_diagnostics(out, data, rows, (0,), kept, held, 2, torch.device("cpu"),
                                           ae.Deadline(60))
        self.assertEqual([g["status"] for g in got], ["not_reached"] * 3)
        self.assertEqual([g["actual_update"] for g in got], list(ae.ACTUAL_UPDATE_ORDINALS))
        src = inspect.getsource(ae.run_arm)
        self.assertIn("if step[\"stepped\"]:", src)
        self.assertIn("actual += 1", src)


class Twelve_Lock(unittest.TestCase):
    def test_the_selection_never_reads_the_check_rows_the_effect_or_the_latent(self):
        src = inspect.getsource(ae.select)
        for word in ('rows["check"]', 'rows["evaluate"]', 'rows["nuisance"]', "eval_only", "theta",
                     "de.check(", "de.estimate("):
            self.assertNotIn(word, src, word)
        self.assertIn('rows["select"]', src)
        self.assertIn('rows["represent"]', src)


class Thirteen_FailClosed(unittest.TestCase):
    def test_a_time_cap_and_a_missing_encoder_are_reported_rather_than_swallowed(self):
        d = ae.Deadline(-1.0)
        self.assertTrue(d.over())
        bank, s_, x, a, y, rows, data, held, kept = small()
        ids = ae.image_id_split(s_, rows["represent"])
        out = ae.train_autoencoder(data, ids, ae.Deadline(-1.0))
        self.assertEqual(out["epochs_completed"], 0)
        self.assertIsNotNone(out["stopped_early"])
        self.assertIn("val_mse", out["history"][0])
        with self.assertRaises(RuntimeError):
            ae.build_representation(kept, 2, torch.device("cpu"), "AE", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
