"""Checks of the kernel-critic module, required by the last-attempt PRD before any CNN is trained.

Seeds come from the test namespace 96,900,000-96,999,999.  Run: python3 -B test_direct_balance_kernel.py
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch

import direct_balance_data as db
import direct_balance_kernel as dk
import direct_balance_net as dn
from test_direct_balance import TINY, tiny_problem

T0 = 96_900_900
torch.set_default_dtype(torch.float32)


def small_world(n: int = 240, seed: int = T0, informative: bool = True):
    """A tiny image problem, its Batches on the CPU, a representation, the held-out pixel projection."""
    bank, src, x, a, _ = tiny_problem(n=n, seed=seed, informative=informative)
    dev = torch.device("cpu")
    data = dn.Batches(bank, src, x, a, dev, standardise_rows=np.arange(n))
    held, kept = db.image_columns((0,))
    torch.manual_seed(seed + 1)
    rep = dn.Representation(kept, 2, spec=TINY)
    width = dk.pixel_width(data, np.arange(n), held, seed + 2)
    feats = dk.Features(4, width["n_pix"], width["sigma_w"], seed + 3, dev, d=16)
    pw = feats.pixel_projection(data, np.arange(n), held)
    a_t = torch.as_tensor(a, dtype=torch.float64)
    return data, rep, feats, pw, a_t, held, kept


class Nesting(unittest.TestCase):
    def test_zero_extra_coefficients_reproduce_the_base_critic_exactly(self):
        data, rep, feats, pw, a, held, kept = small_world()
        z = dk.representation_all(rep, data, np.arange(240), kept)
        mu, sd = dk.h_stats(z)
        zn = dk.normalise(z, mu, sd)
        fit = dk.critics(zn, pw, a, feats, nest=False)
        lifted = torch.cat([fit["c0"], torch.zeros(feats.d, dtype=torch.float64)])
        self.assertTrue(torch.allclose(fit["f1"] @ lifted, fit["f0"] @ fit["c0"], atol=1e-12))

    def test_the_augmented_critic_is_never_worse_than_the_base_on_its_fit_rows(self):
        for informative in (True, False):
            data, rep, feats, pw, a, held, kept = small_world(informative=informative)
            z = dk.representation_all(rep, data, np.arange(240), kept)
            mu, sd = dk.h_stats(z)
            fit = dk.critics(dk.normalise(z, mu, sd), pw, a, feats)
            self.assertLessEqual(float(fit["r1"]), float(fit["r0"]) + 1e-12, informative)
            if fit["info"]["lifted_base_used"]:
                self.assertAlmostEqual(float(fit["r1"]), float(fit["r0"]), places=12)


class Objective(unittest.TestCase):
    def test_the_clip_is_applied_once_to_the_whole_fold_gap(self):
        data, rep, feats, pw, a, held, kept = small_world()
        z = dk.representation_all(rep, data, np.arange(240), kept)
        loss, info = dk.training_loss(z, pw, a, feats)
        self.assertEqual(float(loss), max(info["signed_gap"], 0.0))
        # per-batch clipping would be a different number whenever some batches are negative and some positive
        mu, sd = dk.h_stats(z)
        zn = dk.normalise(z, mu, sd)
        fit = dk.critics(zn, pw, a, feats)
        per_row = ((a - fit["q0"]) ** 2 - (a - fit["q1"]) ** 2)
        batched = float(sum(torch.clamp(per_row[s:s + 60].mean(), min=0.0) for s in range(0, 240, 60)) / 4)
        self.assertGreaterEqual(batched, float(loss) - 1e-12)     # batch-wise clipping can only be larger

    def test_autograd_through_the_solve_matches_finite_differences(self):
        data, rep, feats, pw, a, held, kept = small_world()
        z = dk.representation_all(rep, data, np.arange(240), kept).clone().requires_grad_(True)
        loss, info = dk.training_loss(z, pw, a, feats)
        self.assertGreater(info["signed_gap"], 0.0, "the untrained representation must leave a positive gap")
        loss.backward()
        g = z.grad.clone()
        rng = np.random.default_rng(T0 + 9)
        eps = 1e-6
        for _ in range(6):
            i, j = int(rng.integers(0, 240)), int(rng.integers(2, 4))
            zp, zm = z.detach().clone(), z.detach().clone()
            zp[i, j] += eps; zm[i, j] -= eps
            fd = (float(dk.training_loss(zp, pw, a, feats)[0]) - float(dk.training_loss(zm, pw, a, feats)[0])) / (2 * eps)
            self.assertLess(abs(fd - float(g[i, j])), 1e-6 + 1e-4 * abs(fd), (i, j, fd, float(g[i, j])))

    def test_the_two_stage_gradient_equals_the_full_autograd_gradient(self):
        data, rep, feats, pw, a, held, kept = small_world()
        full = dk.full_autograd_step_gradients(rep, data, np.arange(240), kept, pw, a, feats)
        rep.zero_grad(set_to_none=True)
        opt = torch.optim.SGD(rep.parameters(), lr=0.0)             # a zero step: the gradients stay comparable
        out = dk.two_stage_step(rep, data, np.arange(240), kept, pw, a, feats, opt, batch=64)
        self.assertTrue(out["stepped"])
        for name, p in rep.named_parameters():
            self.assertTrue(torch.allclose(p.grad.double(), full[name].double(), atol=1e-6, rtol=1e-4), name)

    def test_the_gradient_reaches_the_first_convolution_and_the_weights_move(self):
        data, rep, feats, pw, a, held, kept = small_world()
        before = {n: p.detach().clone() for n, p in rep.named_parameters()}
        opt = torch.optim.Adam(rep.parameters(), lr=1e-3)
        out = dk.two_stage_step(rep, data, np.arange(240), kept, pw, a, feats, opt)
        self.assertTrue(out["stepped"])
        self.assertGreater(out["first_conv_grad_norm"], 0.0)
        first = [n for n in before if n.endswith("towers.0.body.0.weight")][0]
        self.assertGreater(float((dict(rep.named_parameters())[first] - before[first]).abs().max()), 0.0)

    def test_no_step_and_no_gradient_when_the_fold_gap_is_not_positive(self):
        data, rep, feats, pw, a, held, kept = small_world(informative=False)
        z = dk.representation_all(rep, data, np.arange(240), kept)
        _, info = dk.training_loss(z, pw, a, feats)
        if info["signed_gap"] > 0:
            self.skipTest("this draw left a positive gap; the rule is exercised by the other tests")
        opt = torch.optim.Adam(rep.parameters(), lr=1e-3)
        before = {n: p.detach().clone() for n, p in rep.named_parameters()}
        out = dk.two_stage_step(rep, data, np.arange(240), kept, pw, a, feats, opt)
        self.assertFalse(out["stepped"])
        for n, p in rep.named_parameters():
            self.assertTrue(torch.equal(p, before[n]))


class Structure(unittest.TestCase):
    def test_the_representation_never_receives_a_held_out_column(self):
        held, kept = db.image_columns((0,))
        rep = dn.Representation(kept, 2, spec=TINY)
        self.assertEqual(rep.kept.tolist(), kept.tolist())
        self.assertNotIn(int(held[0]), rep.kept.tolist())
        with self.assertRaises(ValueError):
            rep([torch.rand(3, 3, 8, 8) for _ in range(6)], torch.randn(3, 2))

    def test_the_cnn_path_reads_no_label_and_no_latent(self):
        data, rep, feats, pw, a, held, kept = small_world()
        for name in ("y", "u", "levels", "orientation", "digit"):
            self.assertFalse(hasattr(data, name), name)
        import inspect
        for fn in (dk.two_stage_step, dk.training_loss, dk.critics, dn.Batches.pixels):
            src = inspect.getsource(fn)
            for word in ("eval_only", "U_", "orientation", "levels"):
                self.assertNotIn(word, src, (fn.__name__, word))

    def test_the_pixel_width_is_a_positive_finite_median_and_the_features_are_fixed_by_their_seed(self):
        data, rep, feats, pw, a, held, kept = small_world()
        width = dk.pixel_width(data, np.arange(240), held, T0 + 2)
        self.assertTrue(width["finite"] and width["sigma_w"] > 0)
        again = dk.Features(4, width["n_pix"], width["sigma_w"], T0 + 3, torch.device("cpu"), d=16)
        self.assertEqual(again.record()["omega_z_sha256"], feats.record()["omega_z_sha256"])
        with self.assertRaises(RuntimeError):
            dk.Features(4, width["n_pix"], 0.0, T0 + 3, torch.device("cpu"), d=16)

    def test_the_kernel_check_tells_a_planted_imbalance_from_none(self):
        found = {}
        for informative in (True, False):
            data, rep, feats, pw, a, held, kept = small_world(n=1_200, informative=informative)
            z = dk.representation_all(rep, data, np.arange(1_200), kept)
            mu, sd = dk.h_stats(z)
            found[informative] = dk.kernel_check(z, pw, a, feats, mu, sd, 1.5e-3)
        self.assertGreater(found[True]["gap"], 5 * max(found[False]["gap"], 1e-4))
        self.assertTrue(found[True]["finite"] and found[False]["finite"])
        self.assertEqual(len(found[True]["rows_d"]), 1_200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
