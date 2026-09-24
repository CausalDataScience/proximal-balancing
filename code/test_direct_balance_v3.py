"""Checks of the v3 module: fixed critic folds, the exact whole-fold gradient, and the standardisation that
carries the critics with it.  Run: python3 -B test_direct_balance_v3.py
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

import direct_balance_data as db
import direct_balance_net as dn
import direct_balance_tiny16 as t16
import direct_balance_v3 as v3
from test_direct_balance_tiny16 import world

T0 = 96_901_500


def setup(n: int = 900, seed: int = T0, informative: bool = True):
    bank, src, x, a, y, _, data = world(n=n, seed=seed, informative=informative)
    rows = {"represent": np.arange(0, int(n * 0.6)), "select": np.arange(int(n * 0.6), int(n * 0.8)),
            "check": np.arange(int(n * 0.8), n)}
    data = t16.TinyBatches(bank, src, x, a, torch.device("cpu"), standardise_rows=rows["represent"])
    held, kept = db.image_columns((0,))
    return data, rows, held, kept


class Folds(unittest.TestCase):
    def test_every_fold_has_disjoint_fit_validation_and_scoring_rows_that_cover_the_fold(self):
        data, rows, held, kept = setup()
        plan = v3.fold_plan(rows, T0 + 1)
        rep = v3.plan_report(plan)
        self.assertEqual(rep["leaks"], [])
        self.assertEqual(sum(rep["sizes"]["score"]), len(rows["represent"]))
        for i in range(plan["folds"]):
            self.assertEqual(len(np.intersect1d(plan["fit_sets"][i], plan["val_sets"][i])), 0)
            self.assertEqual(len(np.intersect1d(plan["fit_sets"][i], plan["score_sets"][i])), 0)

    def test_the_plan_is_drawn_once_and_does_not_move_between_iterations(self):
        data, rows, held, kept = setup()
        a, b = v3.fold_plan(rows, T0 + 2), v3.fold_plan(rows, T0 + 2)
        for i in range(a["folds"]):
            self.assertTrue(np.array_equal(a["val_sets"][i], b["val_sets"][i]))
        self.assertFalse(np.array_equal(a["val_sets"][0], v3.fold_plan(rows, T0 + 3)["val_sets"][0]))

    def test_a_critic_is_never_trained_on_its_own_validation_rows(self):
        """The v2 bug: validation rows were redrawn while the weights were carried forward."""
        data, rows, held, kept = setup()
        plan = v3.fold_plan(rows, T0 + 4)
        torch.manual_seed(T0 + 5)
        rep = dn.Representation(kept, 2, spec=t16.SPEC16)
        rng = np.random.default_rng(T0 + 6)
        seen = []
        real = dn.brier
        state = None
        for _ in range(3):                                   # three refits, as a run would do
            _, _, state = v3.fit_critics_v3(rep, v3.Normaliser(None, None), data, plan["fit_sets"][0],
                                            plan["val_sets"][0], kept, held, 4, rng, state, t16.SPEC16)
            seen.append(set(plan["val_sets"][0].tolist()))
        self.assertEqual(len(set.union(*seen) & set(plan["fit_sets"][0].tolist())), 0)
        self.assertEqual(seen[0], seen[-1])


class Gradient(unittest.TestCase):
    def test_the_precomputed_held_features_reproduce_the_augmented_critic_exactly(self):
        data, rows, held, kept = setup()
        torch.manual_seed(T0 + 10)
        rep = dn.Representation(kept, 2, spec=t16.SPEC16)
        base = dn.BaseCritic(4)
        aug = dn.AugmentedCritic(base, len(held), 4, t16.SPEC16)
        with torch.no_grad():
            aug.extra[-1].weight.normal_(std=0.3); aug.extra[-1].bias.normal_()
        r = rows["represent"][:64]
        z = dn.representation_values(rep, data, r, kept, grad=False)
        feats = v3._held_features(aug, data, r, held)
        with torch.no_grad():
            direct = aug(z, data.pixels(r, held))
            viafeat = dn.link(aug.net(z).squeeze(1) + aug.extra(torch.cat([feats, z], dim=1)).squeeze(1))
        self.assertTrue(torch.allclose(direct, viafeat, atol=1e-6))

    def test_the_two_stage_gradient_equals_full_autograd(self):
        data, rows, held, kept = setup()
        plan = v3.fold_plan(rows, T0 + 11)
        torch.manual_seed(T0 + 12)
        rep = dn.Representation(kept, 2, spec=t16.SPEC16)
        rng = np.random.default_rng(T0 + 13)
        pairs = [v3.fit_critics_v3(rep, v3.Normaliser(None, None), data, plan["fit_sets"][i], plan["val_sets"][i],
                                   kept, held, 12, rng, None, t16.SPEC16)[:2] for i in range(plan["folds"])]
        for standardise in (False, True):
            # full autograd: one graph over every row
            rep.zero_grad(set_to_none=True)
            rep.train()
            idx = torch.as_tensor(plan["represent"])
            z = rep(data.pixels(plan["represent"], kept), data.x[idx])
            norm = v3.Normaliser.fit(z, enabled=standardise)
            pos = {int(r): i for i, r in enumerate(plan["represent"])}
            groups, index_of = [], {}
            for (b, a_), score in zip(pairs, plan["score_sets"]):
                g = {"rows": score, "base": b, "aug": a_, "feats": v3._held_features(a_, data, score, held)}
                groups.append(g); index_of[id(g)] = torch.as_tensor([pos[int(r)] for r in score])
            loss, info = v3.pooled_objective(z, norm, groups, data, index_of)
            self.assertGreater(info["signed_gap"], 0.0)
            loss.backward()
            want = {n: p.grad.clone() for n, p in rep.named_parameters()}
            rep.zero_grad(set_to_none=True)
            opt = torch.optim.SGD(rep.parameters(), lr=0.0)
            out, _ = v3.update(rep, data, plan, pairs, kept, held, opt, standardise, batch=128)
            self.assertTrue(out["stepped"])
            for n, p in rep.named_parameters():
                self.assertTrue(torch.allclose(p.grad, want[n], atol=1e-5, rtol=1e-3), (standardise, n))

    def test_no_step_when_the_pooled_gap_is_not_positive(self):
        data, rows, held, kept = setup(informative=False)
        plan = v3.fold_plan(rows, T0 + 14)
        torch.manual_seed(T0 + 15)
        rep = dn.Representation(kept, 2, spec=t16.SPEC16)
        base = dn.BaseCritic(4); aug = dn.AugmentedCritic(base, len(held), 4, t16.SPEC16)
        with torch.no_grad():                                  # a deliberately useless augmented critic
            aug.extra[-1].bias.fill_(5.0)
        pairs = [(base, aug)] * plan["folds"]
        before = {n: p.detach().clone() for n, p in rep.named_parameters()}
        opt = torch.optim.Adam(rep.parameters(), lr=1e-2)
        out, _ = v3.update(rep, data, plan, pairs, kept, held, opt, False)
        self.assertLessEqual(out["signed_gap"], 0.0)
        self.assertFalse(out["stepped"])
        for n, p in rep.named_parameters():
            self.assertTrue(torch.equal(p, before[n]))


class Standardisation(unittest.TestCase):
    def test_retuning_keeps_the_critic_the_same_function_of_the_raw_representation(self):
        torch.manual_seed(T0 + 20)
        base = dn.BaseCritic(4)
        aug = dn.AugmentedCritic(base, 1, 4, t16.SPEC16)
        with torch.no_grad():
            aug.extra[-1].weight.normal_(std=0.4); aug.extra[-1].bias.normal_()
        raw = torch.randn(40, 4) * torch.tensor([1.0, 1.0, 3.0, 0.2]) + torch.tensor([0.0, 0.0, 2.0, -1.0])
        old = v3.Normaliser(torch.tensor([1.0, -0.5]), torch.tensor([2.0, 0.5]))
        new = v3.Normaliser(torch.tensor([2.5, 0.3]), torch.tensor([0.7, 3.0]))
        px = [torch.randn(40, 3, 16, 16)]
        with torch.no_grad():
            want_b, want_a = base(old.apply(raw)), aug(old.apply(raw), px)
        v3.retune_critic(base, old, new); v3.retune_critic(aug, old, new)
        with torch.no_grad():
            got_b, got_a = base(new.apply(raw)), aug(new.apply(raw), px)
        self.assertTrue(torch.allclose(want_b, got_b, atol=1e-5), float((want_b - got_b).abs().max()))
        self.assertTrue(torch.allclose(want_a, got_a, atol=1e-5), float((want_a - got_a).abs().max()))

    def test_the_statistics_are_in_the_gradient(self):
        z = (torch.randn(200, 4) * 3.0).requires_grad_(True)
        norm = v3.Normaliser.fit(z, enabled=True)
        self.assertTrue(norm.mu.requires_grad and norm.sd.requires_grad)
        norm.apply(z).pow(2).sum().backward()
        self.assertIsNotNone(z.grad)
        frozen = v3.Normaliser.fit(z.detach(), enabled=True)
        self.assertFalse(frozen.mu.requires_grad)

    def test_standardising_puts_h_on_unit_scale_and_the_identity_leaves_z_alone(self):
        z = torch.cat([torch.randn(300, 2), torch.randn(300, 2) * torch.tensor([9.0, 0.0004]) + 5.0], dim=1)
        out = v3.Normaliser.fit(z, enabled=True).apply(z)
        self.assertTrue(torch.allclose(out[:, :2], z[:, :2]))
        self.assertLess(abs(float(out[:, 2].std(unbiased=False)) - 1.0), 1e-6)   # the fit uses the population sd
        self.assertLess(float(out[:, 3].std()), 1.0)                       # the floor caps a degenerate column
        self.assertTrue(torch.equal(v3.Normaliser.fit(z, enabled=False).apply(z), z))

    def test_the_evaluator_sees_the_checkpoints_own_statistics(self):
        data, rows, held, kept = setup()
        torch.manual_seed(T0 + 25)
        rep = dn.Representation(kept, 2, spec=t16.SPEC16)
        stats = {"identity": False, "mean": [1.0, -2.0], "sd": [4.0, 0.5]}
        wrapped = v3.load_normalised({k: v.clone() for k, v in rep.state_dict().items()}, kept, 2,
                                     torch.device("cpu"), t16.SPEC16, stats)
        r = rows["check"][:32]
        raw = dn.representation_values(rep, data, r, kept, grad=False)
        got = dn.representation_values(wrapped, data, r, kept, grad=False)
        want = torch.cat([raw[:, :2], (raw[:, 2:] - torch.tensor(stats["mean"])) / torch.tensor(stats["sd"])], dim=1)
        self.assertTrue(torch.allclose(got, want, atol=1e-6))
        plain = v3.load_normalised({k: v.clone() for k, v in rep.state_dict().items()}, kept, 2,
                                   torch.device("cpu"), t16.SPEC16, {"identity": True})
        self.assertTrue(torch.allclose(dn.representation_values(plain, data, r, kept, grad=False), raw, atol=1e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
