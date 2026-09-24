"""Tests for SCM family v2 (PRD memo/2026-09-18-family-v2-role-blind-dimension-prd-v1.md, section 11).

Seeds come from the test namespace 97,800,000-97,899,999 only.  Run: python3 -B test_family_v2.py
"""
from __future__ import annotations

import math
import unittest

import numpy as np

import family_v2_competitors as fc
import family_v2_dgp as g
import family_v2_population as fp
import family_v2_probe as pr

T0 = g.TEST_SEED


class Generator(unittest.TestCase):
    def test_rotations_are_orthogonal_fixed_and_level_specific(self):
        digests = set()
        for level in g.LEVELS.values():
            q = g.rotations(level)
            for key, m in q.items():
                self.assertTrue(np.allclose(m @ m.T, np.eye(m.shape[0]), atol=1e-12), key)
            self.assertEqual(g.rotation_digest(level), g.rotation_digest(level))
            digests.add(g.rotation_digest(level))
        self.assertEqual(len(digests), len(g.LEVELS))

    def test_shapes_and_learner_view(self):
        for level in g.LEVELS.values():
            data = g.generate(level, 200, T0 + level.index)
            self.assertEqual(data["X"].shape, (200, level.d_x))
            for j, d in enumerate(level.d_blocks):
                self.assertEqual(data[f"W{j + 1}"].shape, (200, d))
            view = g.learner_view(data)
            self.assertEqual(set(view), set(g.LEARNER_KEYS))
            self.assertFalse(any("eval" in k for k in view))
            self.assertEqual(level.dim, sum(view[k].shape[1] for k in view if k not in ("A", "Y")))

    def test_evaluator_inverse_recovers_informative_coordinates(self):
        for level in g.LEVELS.values():
            data = g.generate(level, 300, T0 + 10 + level.index)
            np.testing.assert_allclose(g.informative(level, data, "X"),
                                       np.column_stack([data["X1_eval_only"], data["C_eval_only"]]), atol=1e-10)
            np.testing.assert_allclose(g.informative(level, data, "W5"),
                                       np.column_stack([data["M0_eval_only"], data["I_eval_only"]]), atol=1e-10)
            np.testing.assert_allclose(g.informative(level, data, "W4")[:, 0], data["M4_eval_only"], atol=1e-10)

    def test_same_seed_same_data_and_levels_share_latents(self):
        a = g.generate(g.LEVELS["SCM-1"], 100, T0 + 20)
        b = g.generate(g.LEVELS["SCM-1"], 100, T0 + 20)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        c = g.generate(g.LEVELS["SCM-3"], 100, T0 + 20)
        for key in ("U_eval_only", "A", "Y", "M4_eval_only"):
            np.testing.assert_array_equal(a[key], c[key])


class Population(unittest.TestCase):
    def test_reference_values_match_monte_carlo(self):
        ref = fp.reference_values()
        data = g.generate(g.LEVELS["SCM-1"], 400_000, T0 + 30)
        a, y = data["A"], data["Y"]
        naive = y[a == 1].mean() - y[a == 0].mean()
        se = math.sqrt(y[a == 1].var() / (a == 1).sum() + y[a == 0].var() / (a == 0).sum())
        self.assertLess(abs(naive - ref["naive"]), 4 * se)
        # oracle regression on (A, U, X_1, C) recovers tau
        design = np.column_stack([np.ones_like(a), a, data["U_eval_only"], data["X1_eval_only"], data["C_eval_only"]])
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        self.assertLess(abs(coef[1] - 1.0), 0.01)
        self.assertAlmostEqual(ref["oracle (X, U)"], 1.0, places=10)

    def test_reproduces_manuscript_population_numbers(self):
        rep = fp.reproduce_manuscript()
        for label, v in rep.items():
            self.assertLess(abs(v["tau"] - v["expected"]), 5e-4, label)

    def test_balance_residual_zero_iff_discrepancy_zero(self):
        smp = fp.sample_balanced((0,), 2, 5, T0 + 40)
        self.assertGreater(len(smp["taus"]), 0)
        x_rows, v, t = fp.split_rows((0,))
        z = fp.z_of(x_rows, v, smp["b"][0])
        self.assertLess(fp.discrepancy(z, t), 1e-14)
        z_bad = fp.z_of(x_rows, v, np.eye(2, v.shape[0]))
        self.assertGreater(fp.discrepancy(z_bad, t), 1e-6)
        self.assertGreater(np.max(np.abs(fp.balance_residual(z_bad, t))), 1e-6)

    def test_clean_split_balanced_points_target_tau(self):
        for split in ((0,), (0, 1), (0, 1, 2)):
            for rank in (1, 2):
                smp = fp.sample_balanced(split, rank, 6, T0 + 50 + rank)
                self.assertGreater(len(smp["taus"]), 0, (split, rank))
                self.assertLess(max(abs(tv - 1.0) for tv in smp["taus"]), 1e-8, (split, rank))

    def test_c_inside_w5_breaks_the_clean_split(self):
        smp = fp.sample_balanced((0,), 2, 20, T0 + 60, c_in_x=False)
        self.assertGreater(len(smp["taus"]), 0)
        self.assertGreater(max(abs(tv - 1.0) for tv in smp["taus"]), 0.01)


class Variant(unittest.TestCase):
    def test_scm1c_layout(self):
        lv = g.level("SCM-1c")
        data = g.generate(lv, 300, T0 + 70)
        self.assertEqual(data["X"].shape, (300, 3))
        for j in range(g.N_BLOCKS):
            self.assertEqual(data[f"W{j + 1}"].shape, (300, 1))
        np.testing.assert_allclose(g.informative(lv, data, "X"),
                                   np.column_stack([data["X1_eval_only"], data["C_eval_only"], data["I_eval_only"]]),
                                   atol=1e-10)
        self.assertEqual(g.bad_i(lv), 0.0)
        resid_1c = data["M4_eval_only"] - data["U_eval_only"]
        self.assertLess(abs(np.corrcoef(resid_1c, data["I_eval_only"])[0, 1]), 0.2)
        main = g.generate(g.LEVELS["SCM-1"], 300, T0 + 70)
        resid_main = main["M4_eval_only"] - main["U_eval_only"]
        self.assertLess(np.corrcoef(resid_main, main["I_eval_only"])[0, 1], -0.9)


class Learner(unittest.TestCase):
    def test_folds_partition_and_roles_differ(self):
        f = pr.folds(1003, T0 + 80)
        allrows = np.concatenate(f)
        self.assertEqual(sorted(allrows.tolist()), list(range(1003)))
        for rot in range(4):
            self.assertEqual(sorted(pr.roles(rot).values()), [0, 1, 2, 3])

    def test_envelope_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(T0 + 90)
        n, dx, q, dt, k = 400, 2, 5, 2, 2
        x, v, t = rng.standard_normal((n, dx)), rng.standard_normal((n, q)), rng.standard_normal((n, dt))
        a = (rng.random(n) < 0.3 + 0.4 * (v[:, 0] + t[:, 0] > 0)).astype(float)
        ones = np.ones((n, 1))

        def gap(flat):
            b, _ = pr._normalise(flat.reshape(k, q))
            fit = pr.nested_critics(np.column_stack([ones, x, v @ b.T]), t, a)
            return fit["r0"] - fit["r1"]

        def analytic(flat):
            b, norms = pr._normalise(flat.reshape(k, q))
            f0 = np.column_stack([ones, x, v @ b.T])
            fit = pr.nested_critics(f0, t, a)
            zi = slice(1 + dx, 1 + dx + k)
            grads = []
            for f, c in ((f0, fit["c0"]), (np.column_stack([f0, t]), fit["c1"])):
                eta = f @ c
                r = (pr.link(eta) - a) * pr.dlink(eta)
                grads.append(np.outer(c[zi], (2.0 / n) * (v.T @ r)))
            gb = grads[0] - grads[1]
            return ((gb - np.sum(gb * b, axis=1, keepdims=True) * b) / norms).ravel()

        b0 = rng.standard_normal(k * q)
        num = np.array([(gap(b0 + 1e-5 * e) - gap(b0 - 1e-5 * e)) / 2e-5 for e in np.eye(k * q)])
        self.assertLess(np.linalg.norm(analytic(b0) - num) / np.linalg.norm(num), 1e-3)

    def test_nested_critics_never_negative(self):
        rng = np.random.default_rng(T0 + 91)
        f0 = np.column_stack([np.ones(300), rng.standard_normal((300, 3))])
        t = rng.standard_normal((300, 2))
        a = (rng.random(300) < 0.5).astype(float)
        fit = pr.nested_critics(f0, t, a)
        self.assertLessEqual(fit["r1"], fit["r0"] + 1e-15)


class Competitors(unittest.TestCase):
    def test_p2sls_exact_linear_case(self):
        rng = np.random.default_rng(T0 + 100)
        n = 4000
        u = rng.standard_normal(n)
        a = (rng.random(n) < 1 / (1 + np.exp(-u))).astype(float)
        x = rng.standard_normal((n, 1))
        y = 1.7 * a - 2.0 * u + 0.5 * x[:, 0]
        view = {"X": x, "W4": u[:, None].copy(), "W2": u[:, None].copy(), "A": a, "Y": y}
        self.assertAlmostEqual(fc.p2sls(view, "W4", "W2"), 1.7, places=8)

    def test_park_spc_exact_linear_bridge(self):
        rng = np.random.default_rng(T0 + 101)
        n = 3000
        w = rng.standard_normal(n)
        x = rng.standard_normal((n, 1))
        a = (rng.random(n) < 0.4).astype(float)
        y = 1.0 + 2.0 * w + 0.5 * x[:, 0] + 2.0 * a
        view = {"X": x, "W1": w[:, None], "A": a, "Y": y}
        self.assertAlmostEqual(fc.park_spc(view, "W1"), 2.0, places=8)

    def test_cevae_runs(self):
        data = g.generate(g.LEVELS["SCM-1"], 400, T0 + 102)
        out = fc.cevae_ate(g.learner_view(data), T0 + 103, num_epochs=1, num_samples=5)
        self.assertTrue(np.isfinite(out["ate"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
