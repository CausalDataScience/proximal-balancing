"""The population facts SCM-6 rests on, computed without fitting anything.

These are the claims the finite-sample run is measured against: which splits can balance, what they
target when they do, and that the adversarial split balances exactly while targeting the wrong value.
"""
import unittest

import numpy as np

import minimal_suite_common as c
import scm6_population as sp

FEASIBLE = {"S1": 0.2245, "S2": 0.2245, "S3": 0.2245, "S12": 0.3004, "S13": 0.3004,
            "S23": 0.3004, "S123": 0.4540, "S4": -0.6502}
BY_ID = {c.split_id(s): s for s in c.oriented_splits()}


class TestBalanceRoots(unittest.TestCase):
    def test_exactly_eight_splits_can_balance_inside_the_search_grid(self):
        found = {c.split_id(s): sp.operative_root(s) for s in c.oriented_splits()}
        reachable = {k: v for k, v in found.items() if v is not None}
        self.assertEqual(set(reachable), set(FEASIBLE))
        for name, root in reachable.items():
            self.assertAlmostEqual(root, FEASIBLE[name], places=3, msg=name)

    def test_the_equation_has_two_roots_so_an_endpoint_bracket_finds_none(self):
        """The residual is a rational function of r whose sign changes twice in the scan window.

        A bracket that only tests the endpoints sees the same sign at both and reports no root.  That
        happened during development and is why the scan-then-bisect form is used.
        """
        roots = sp.balance_roots((1,))
        self.assertEqual(len(roots), 2)
        lo = sp.balance_residual((1,), -6.0)[0]
        hi = sp.balance_residual((1,), 6.0)[0]
        self.assertGreater(lo * hi, 0.0)

    def test_holding_out_block_zero_leaves_no_coefficient_to_choose(self):
        for split in c.oriented_splits():
            if 0 in split:
                self.assertIsNone(sp.operative_root(split), c.split_id(split))

    def test_the_discrepancy_is_machine_zero_at_every_balancing_root(self):
        for name, r in FEASIBLE.items():
            root = sp.operative_root(BY_ID[name])
            d2 = sp.residual_discrepancy(BY_ID[name], root, n=50000)["d2_res"]
            self.assertLess(d2, 1e-20, f"{name} should balance exactly")


class TestWhatTheBalancingSplitsTarget(unittest.TestCase):
    def test_seven_of_the_eight_balancing_splits_hit_the_true_effect(self):
        hits = {name: sp.adjusted_effect(BY_ID[name], sp.operative_root(BY_ID[name]), n=200000)["tau_Z"]
                for name in FEASIBLE if name != "S4"}
        for name, tau in hits.items():
            self.assertAlmostEqual(tau, 1.0, places=3, msg=name)

    def test_the_contaminated_singleton_balances_exactly_and_targets_the_wrong_value(self):
        """S4 holds out the block carrying the treatment-only cause.

        For a clean block K_j = U + noise with the noise absent from Z, so zero conditional covariance
        with K_j is zero conditional covariance with U.  For K_4 = U + noise - 0.6 I the same equation
        mixes two channels, and solving it leaves confounding in place.
        """
        split = BY_ID["S4"]
        root = sp.operative_root(split)
        self.assertLess(sp.residual_discrepancy(split, root, n=50000)["d2_res"], 1e-20)
        tau = sp.adjusted_effect(split, root, n=200000)["tau_Z"]
        self.assertAlmostEqual(tau, -0.4432, places=3)

    def test_the_plurality_and_the_separation_support_the_theorem(self):
        good = [n for n in FEASIBLE if n != "S4"]
        self.assertEqual(len(good), 7)
        self.assertEqual(len(FEASIBLE) - len(good), 1)
        gap = abs(1.0 - (-0.4432))
        self.assertGreater(gap, 4 * 0.36, "separation must exceed four radii for the theorem to bite")


class TestSelectedCoefficientIsNotTheNaiveCancellation(unittest.TestCase):
    def test_cancelling_the_contaminating_block_is_a_different_and_wrong_answer(self):
        """The naive value removes -0.6 I from the kept average; the balancing value does not equal it.

        Holding out block 1 keeps blocks 0, 2, 3 and 4, so cancellation would give +0.6/4 = +0.15.
        The balancing root is +0.2245, because the held-out block sharpens U and I together.
        """
        naive = 0.6 / 4
        self.assertAlmostEqual(naive, 0.15, places=6)
        self.assertNotAlmostEqual(sp.operative_root((1,)), naive, places=2)
        residual_at_naive = abs(float(sp.balance_residual((1,), naive)[0]))
        self.assertGreater(residual_at_naive, 0.1)


if __name__ == "__main__":
    unittest.main()
