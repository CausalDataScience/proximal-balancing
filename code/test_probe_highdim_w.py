"""The SCM-6 generator, its embedding, and the containment the nested gap relies on."""
import unittest

import numpy as np

import minimal_suite_common as c
import probe_highdim_w as hd


class TestOrthogonalBlocks(unittest.TestCase):
    def test_each_block_matrix_is_orthogonal(self):
        for j in range(5):
            q = hd.orthogonal_block(j)
            np.testing.assert_allclose(q.T @ q, np.eye(hd.DIM), atol=1e-10)

    def test_blocks_are_reproducible_and_distinct(self):
        np.testing.assert_array_equal(hd.orthogonal_block(2), hd.orthogonal_block(2))
        self.assertEqual(hd.orthogonal_digest(2), hd.orthogonal_digest(2))
        self.assertNotEqual(hd.orthogonal_digest(2), hd.orthogonal_digest(3))

    def test_the_channel_is_recovered_exactly_at_the_population_direction(self):
        report = hd.identity_check(n=20000, seed=99)
        self.assertLess(report["max_inverse_error"], 1e-9)


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.data = hd.generate(4000, 21)

    def test_the_observed_view_hides_every_evaluator_only_field(self):
        view = hd.observed_view(self.data, include_outcome=True)
        for hidden in ("U_eval_only", "K_eval_only", "p_eval_only"):
            self.assertNotIn(hidden, view)
        self.assertEqual(set(view), {"X", "I", "C", "A", "Y", "H"})

    def test_raw_proxy_width_is_the_declared_one(self):
        self.assertEqual(len(self.data["H"]), 5)
        self.assertEqual(self.data["H"][0].shape[1], hd.DIM)
        self.assertEqual(2 + 5 * hd.DIM, 1282)

    def test_only_the_last_block_carries_the_contaminating_cause(self):
        k, i = self.data["K_eval_only"], self.data["I"]
        clean = [abs(np.corrcoef(k[:, j] - self.data["U_eval_only"], i)[0, 1]) for j in range(4)]
        bad = abs(np.corrcoef(k[:, 4] - self.data["U_eval_only"], i)[0, 1])
        self.assertLess(max(clean), 0.08)
        self.assertGreater(bad, 0.4)

    def test_propensity_stays_inside_the_structural_overlap_range(self):
        p = self.data["p_eval_only"]
        self.assertGreaterEqual(p.min(), 0.1)
        self.assertLessEqual(p.max(), 0.9)


class TestEmbedding(unittest.TestCase):
    def test_the_embedding_tracks_the_true_channel(self):
        data = hd.generate(6000, 31)
        view = hd.observed_view(data)
        sl = lambda i: {k: ([b[i] for b in v] if k == "H" else v[i]) for k, v in view.items()}
        emb = hd.fit_embedding(sl(np.arange(1200)), sl(np.arange(1200, 1500)))
        self.assertIsNone(emb["fail_closed"])
        corr = hd.channel_correlation(emb, data, np.arange(1500, 6000))
        self.assertGreater(float(np.median(corr)), 0.90)

    def test_a_block_with_no_signal_fails_closed(self):
        data = hd.generate(2000, 41)
        data["H"] = list(data["H"])
        data["H"][3] = np.zeros_like(data["H"][3])
        view = hd.observed_view(data)
        sl = lambda i: {k: ([b[i] for b in v] if k == "H" else v[i]) for k, v in view.items()}
        emb = hd.fit_embedding(sl(np.arange(1200)), sl(np.arange(1200, 1500)))
        self.assertIsNotNone(emb["fail_closed"])
        self.assertIn("block 3", emb["fail_closed"])


class TestNestedContainment(unittest.TestCase):
    """The base critic padded with zeros must be feasible for the augmented problem at equal cost.

    That is what makes the gap an estimate of a non-negative quantity.  It holds only when both critics
    carry the same ridge penalty, so this fixes the rule rather than leaving it to a tuning choice.
    """

    def setUp(self):
        data = hd.generate(4000, 51)
        view = hd.observed_view(data)
        sl = lambda i: {k: ([b[i] for b in v] if k == "H" else v[i]) for k, v in view.items()}
        emb = hd.fit_embedding(sl(np.arange(1200)), sl(np.arange(1200, 1500)))
        self.rows = sl(np.arange(1500, 3000))
        self.k_hat = hd.apply_embedding(emb, self.rows)

    def test_both_critics_receive_one_shared_penalty(self):
        pen = hd.split_penalties(self.rows, self.k_hat, (1,), (1e-3, 1e-2, 1e-1), 7)
        self.assertEqual(pen["base"], pen["augmented"])

    def test_the_in_sample_gap_is_not_negative_under_the_shared_penalty(self):
        for split in [(1,), (4,), (1, 2)]:
            pen = hd.split_penalties(self.rows, self.k_hat, split, (1e-2,), 7)
            sel = hd.select_r(self.rows, self.k_hat, split, pen, grid=(0.0, 0.5))
            self.assertGreaterEqual(sel["signed_gap"], c.GAP_FLOOR, split)

    def test_holding_out_block_zero_removes_the_coefficient_entirely(self):
        pen = hd.split_penalties(self.rows, self.k_hat, (0,), (1e-2,), 7)
        sel = hd.select_r(self.rows, self.k_hat, (0,), pen)
        self.assertIsNone(sel["r"])
        z = hd.representation(self.k_hat, self.rows, (0,), None)
        self.assertEqual(z.shape[1], 2, "I and C are not observable when block 0 is held out")


class TestScaler(unittest.TestCase):
    """A ridge penalty is not scale invariant and r changes the scale of one coordinate.

    Without standardisation the penalised objective falls as |r| grows for reasons unrelated to balance.
    """

    def test_scaling_is_fitted_once_and_then_frozen(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(500, 3)) * np.array([1.0, 50.0, 0.01])
        scaler = hd.Scaler(a)
        np.testing.assert_allclose(scaler(a).mean(0), 0.0, atol=1e-10)
        np.testing.assert_allclose(scaler(a).std(0), 1.0, atol=1e-10)
        b = rng.normal(size=(200, 3)) * np.array([1.0, 50.0, 0.01]) + 7.0
        self.assertFalse(np.allclose(scaler(b).mean(0), 0.0),
                         "the scaler must not refit itself on new rows")


if __name__ == "__main__":
    unittest.main()
