"""Tests for the positive-control audit machinery (PRD v4, section 6)."""
from __future__ import annotations

import itertools
import re
import unittest
from pathlib import Path

import numpy as np

import positive_control_scm as pc
from positive_control_scm import Design

# the frozen design, rebuilt from the preflight solution
DESIGN = Design(
    b_c=(1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
    b_t=(0.0, 0.0, 0.0, -0.6, -3.433554687499998, 1.0, 1.0),
    sigma=(0.85,) * 7, kappa=2.0, alpha_c=1.0, alpha_t=-1.8041666666666671,
    outcome_u=2.5, outcome_noise=0.3,
)
J = 7


def beta_on(weights: dict[int, float], split: tuple[int, ...]) -> np.ndarray:
    kept = pc.kept_blocks(J, split)
    v = np.array([weights.get(j, 0.0) for j in kept], dtype=float)
    return v / np.linalg.norm(v)


class LibraryTests(unittest.TestCase):
    def test_size_and_sign_canonical(self):
        """With the support cap at one, |B_d| = d."""
        for d in range(1, 7):
            lib = pc.library(d)
            self.assertEqual(pc.MAX_SUPPORT, 1)
            self.assertEqual(len(lib), d)
            for beta in lib:
                self.assertLessEqual(int((beta != 0).sum()), pc.MAX_SUPPORT)
            for beta in lib:
                self.assertAlmostEqual(float(np.linalg.norm(beta)), 1.0, places=12)
                first = next(x for x in beta if x != 0)
                self.assertGreater(first, 0)

    def test_closed_under_permutation(self):
        d = 4
        lib = {tuple(np.round(b, 12)) for b in pc.library(d)}
        for order in itertools.permutations(range(d)):
            for beta in pc.library(d):
                moved = beta[list(order)]
                canonical = moved if next(x for x in moved if x != 0) > 0 else -moved
                self.assertIn(tuple(np.round(canonical, 12)), lib)


class BalanceTests(unittest.TestCase):
    def test_algebraic_criterion_matches_the_definition(self):
        """D_S^2 = 0 exactly where B_S gamma = 0, and is positive elsewhere."""
        split = (0,)
        balanced = beta_on({3: 1.0}, split)
        self.assertLess(pc.balance_residual(DESIGN, split, balanced), 1e-10)
        self.assertLess(pc.discrepancy_numeric(DESIGN, split, balanced), 1e-12)
        for beta in pc.library(J - 1)[:40]:
            residual = pc.balance_residual(DESIGN, split, beta)
            numeric = pc.discrepancy_numeric(DESIGN, split, beta)
            if residual > 1e-6:
                self.assertGreater(numeric, 1e-6)
            if numeric < 1e-12:
                self.assertLess(residual, 1e-8)

    def test_wrong_family_is_balanced_at_its_designated_element(self):
        for split, weights in (((5,), {4: 1.0}), ((6,), {4: 1.0}), ((5, 6), {4: 1.0})):
            beta = beta_on(weights, split)
            self.assertLess(pc.balance_residual(DESIGN, split, beta), 1e-10)
            self.assertLess(pc.discrepancy_numeric(DESIGN, split, beta), 1e-12)

    def test_target_family_targets_tau_exactly(self):
        for size in (1, 2, 3):
            for split in itertools.combinations((0, 1, 2), size):
                beta = beta_on({3: 1.0}, split)
                self.assertLess(pc.balance_residual(DESIGN, split, beta), 1e-10)
                self.assertAlmostEqual(pc.theta(DESIGN, split, beta), 1.0, places=10)

    def test_wrong_family_does_not_target_tau(self):
        beta = beta_on({4: 1.0}, (5,))
        self.assertGreater(abs(pc.theta(DESIGN, (5,), beta) - 1.0), 0.5)


class SelectionMarginTests(unittest.TestCase):
    def test_population_minimiser_is_isolated(self):
        """The second best library element must sit well above the balance point.

        A finite learning sample can only recover the population minimiser when this
        margin dominates the in-sample overfitting floor, so the cap on the library
        support is part of the design, not a convenience.
        """
        for split in ((0,), (5,), (0, 1), (0, 1, 2)):
            values = np.sort([pc.discrepancy_numeric(DESIGN, split, b)
                              for b in pc.library(J - len(split))])
            self.assertLess(values[0], 1e-12)
            self.assertGreater(values[1], 1e-3)


class OrcLabelTests(unittest.TestCase):
    def test_rowspan_rule(self):
        self.assertTrue(pc.orc_valid(DESIGN, (0,)))          # pure U_c row
        self.assertTrue(pc.orc_valid(DESIGN, (3, 4)))        # two non-proportional mixed rows
        self.assertFalse(pc.orc_valid(DESIGN, (3,)))         # a single mixed row
        self.assertFalse(pc.orc_valid(DESIGN, (5,)))         # pure U_t row
        self.assertFalse(pc.orc_valid(DESIGN, (5, 6)))       # two proportional U_t rows


class RoleBlindnessTests(unittest.TestCase):
    def test_permutation_equivariance(self):
        order = (3, 0, 6, 1, 5, 2, 4)
        moved = DESIGN.permuted(order)
        inverse = {order[i]: i for i in range(J)}
        for split in ((0,), (5,), (0, 1), (3, 4), (0, 5, 6)):
            image = tuple(sorted(inverse[j] for j in split))
            lib_a = pc.library(J - len(split))
            best_a = min(pc.balance_residual(DESIGN, split, b) for b in lib_a)
            best_b = min(pc.balance_residual(moved, image, b) for b in lib_a)
            self.assertAlmostEqual(best_a, best_b, places=12)

    def test_no_block_role_symbols_in_the_module(self):
        text = Path(pc.__file__).read_text().lower()
        for forbidden in ("cause_block", "confounder_block", "clean_block",
                          "contaminated", "bad_block", "role_of_block"):
            self.assertNotIn(forbidden, text)


class SamplingTests(unittest.TestCase):
    def test_exact_nuisance_matches_sampling(self):
        split, beta = (0,), beta_on({3: 1.0}, (0,))
        data = pc.generate(DESIGN, 200_000, 8_100_777)
        e, m0, m1 = pc.exact_nuisance(DESIGN, split, beta, data)
        self.assertLess(abs(float(e.mean() - data["A"].mean())), 0.01)
        for arm, fitted in ((1, m1), (0, m0)):
            mask = data["A"] == arm
            residual = (data["Y"][mask] - fitted[mask]).mean()
            self.assertLess(abs(float(residual)), 0.02)

    def test_aipw_scores_centre_on_theta(self):
        split, beta = (5,), beta_on({4: 1.0}, (5,))
        data = pc.generate(DESIGN, 400_000, 8_100_778)
        scores = pc.aipw_scores(DESIGN, split, beta, data)
        target = pc.theta(DESIGN, split, beta)
        se = scores.std(ddof=1) / np.sqrt(len(scores))
        self.assertLess(abs(float(scores.mean()) - target), 4 * se)


if __name__ == "__main__":
    unittest.main()
