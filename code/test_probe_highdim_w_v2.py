"""The repaired selection and screening steps, and the properties each one is supposed to have."""
import unittest

import numpy as np

import minimal_suite_common as c
import probe_highdim_w as hd
import probe_highdim_w_v2 as v2
import scm6_population as sp


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = hd.generate(6000, 94000000)
        view = hd.observed_view(data, include_outcome=True)
        sl = lambda i: {k: ([b[i] for b in v] if k == "H" else v[i]) for k, v in view.items()}
        emb = hd.fit_embedding(sl(np.arange(1200)), sl(np.arange(1200, 1500)))
        cls.phi = sl(np.arange(1500, 3000))
        cls.screen = sl(np.arange(3000, 4500))
        cls.k_phi = hd.apply_embedding(emb, cls.phi)
        cls.k_screen = hd.apply_embedding(emb, cls.screen)


class TestTargetProjection(Fixture):
    def test_the_projection_cuts_the_target_to_the_declared_width(self):
        for split, extra in (((1,), 0), ((1, 2), 0), ((0,), 2), ((0, 1), 2)):
            proj = v2.fit_target_projection(self.phi, split, 2, 5)
            wide = hd.heldout(self.phi, split)
            narrow = v2.projected_heldout(self.phi, split, proj)
            self.assertEqual(wide.shape[1], 1 + extra + hd.DIM * len(split))
            self.assertEqual(narrow.shape[1], 1 + extra + 2 * len(split))

    def test_no_projection_returns_the_full_target(self):
        np.testing.assert_array_equal(v2.projected_heldout(self.phi, (1,), None),
                                      hd.heldout(self.phi, (1,)))

    def test_the_leading_direction_recovers_the_informative_coordinate(self):
        """Each block is one informative coordinate plus independent noise of smaller variance."""
        proj = v2.fit_target_projection(self.phi, (1,), 1, 5)
        scores = self.phi["H"][1] @ proj["directions"][1][:, 0]
        self.assertGreater(abs(float(np.corrcoef(scores, self.k_phi[:, 1])[0, 1])), 0.9)

    def test_the_projection_is_reproducible_and_digested(self):
        a = v2.fit_target_projection(self.phi, (1, 2), 1, 5)
        b = v2.fit_target_projection(self.phi, (1, 2), 1, 5)
        self.assertEqual(a["digest"], b["digest"])
        self.assertNotEqual(a["digest"], v2.fit_target_projection(self.phi, (1, 2), 2, 5)["digest"])


class TestScreenIsNotFailOpen(Fixture):
    """The v1 rule clipped a risk difference at zero, so a negative value passed automatically.

    Here the statistic is a mean of squares, so it cannot be negative and cannot buy a pass that way.
    """

    def test_the_statistic_is_never_negative(self):
        proj = v2.fit_target_projection(self.phi, (1,), 1, 5)
        for r in (-1.0, 0.0, 0.2245, 1.0):
            stat = v2.squared_discrepancy(self.phi, self.k_phi, self.screen, self.k_screen,
                                          (1,), r, 1e-2, proj)
            self.assertGreaterEqual(stat["d2"], 0.0)

    def test_a_split_far_from_balance_is_rejected(self):
        proj = v2.fit_target_projection(self.phi, (1,), 1, 5)
        far = v2.screen_v2(self.phi, self.k_phi, self.screen, self.k_screen, (1,), -1.0, 1e-2,
                           1e-3, proj)
        self.assertFalse(far["pass"])

    def test_a_balancing_split_at_its_own_root_is_retained(self):
        root = sp.operative_root((1,))
        proj = v2.fit_target_projection(self.phi, (1,), 1, 5)
        near = v2.screen_v2(self.phi, self.k_phi, self.screen, self.k_screen, (1,), root, 1e-2,
                            1e-3, proj)
        self.assertTrue(near["pass"], near)

    def test_the_wide_target_inflates_the_statistic_far_above_the_signal(self):
        """Documents why the target is compressed: at the exact root the truth is machine zero."""
        root = sp.operative_root((1,))
        wide = v2.squared_discrepancy(self.phi, self.k_phi, self.screen, self.k_screen,
                                      (1,), root, 1e-2, None)["d2"]
        narrow = v2.squared_discrepancy(self.phi, self.k_phi, self.screen, self.k_screen,
                                        (1,), root, 1e-2,
                                        v2.fit_target_projection(self.phi, (1,), 1, 5))["d2"]
        self.assertGreater(wide, 100 * narrow)
        self.assertLess(sp.residual_discrepancy((1,), root, n=20000)["d2_res"], 1e-20)


class TestCrossFittedSelection(Fixture):
    def test_holding_out_block_zero_leaves_one_candidate(self):
        sel = v2.select_r_crossfit(self.phi, self.k_phi, (0,), 1e-2, 5)
        self.assertIsNone(sel["r"])
        self.assertEqual(len(sel["objective_values"]), 1)

    def test_every_grid_point_is_scored_and_the_argmin_is_returned(self):
        grid = (-0.5, 0.0, 0.25, 0.5)
        proj = v2.fit_target_projection(self.phi, (1,), 1, 5)
        sel = v2.select_r_crossfit(self.phi, self.k_phi, (1,), 1e-2, 5, grid=grid, projection=proj)
        self.assertEqual(len(sel["objective_values"]), len(grid))
        self.assertEqual(sel["r"], grid[int(np.argmin(sel["objective_values"]))])

    def test_the_argmin_lands_near_the_balancing_root(self):
        grid = tuple(np.round(np.arange(-1.0, 1.001, 0.05), 4))
        proj = v2.fit_target_projection(self.phi, (1,), 1, 5)
        sel = v2.select_r_crossfit(self.phi, self.k_phi, (1,), 1e-2, 5, grid=grid, projection=proj)
        self.assertLess(abs(sel["r"] - sp.operative_root((1,))), 0.15)

    def test_cross_fitting_scores_every_row_out_of_sample(self):
        parts = v2.crossfit_folds(100, 5, 3)
        self.assertEqual(len(parts), 5)
        np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(100))


if __name__ == "__main__":
    unittest.main()
