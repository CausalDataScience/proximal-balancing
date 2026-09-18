"""The SCM-7 renderer: constant mass, an exact channel, and nuisances that carry no information."""
import unittest

import numpy as np

import probe_image_w as im


class TestBank(unittest.TestCase):
    def test_the_template_bank_is_pinned_by_its_digest(self):
        self.assertEqual(im.bank()["sha256"], im.MNIST_SHA256)

    def test_every_template_is_l1_normalised(self):
        mass = im.bank()["templates"].sum(axis=(1, 2))
        np.testing.assert_allclose(mass, 1.0, atol=1e-12)

    def test_class_index_covers_all_ten_digits(self):
        idx = im.bank()["class_index"]
        self.assertEqual(len(idx), 10)
        self.assertEqual(sum(len(v) for v in idx), 60000)


class TestRendererGates(unittest.TestCase):
    """The tolerances were frozen in the module before any pilot and are checked here as written."""

    @classmethod
    def setUpClass(cls):
        cls.report = im.exact_channel_check(n=1500, seed=2468)

    def test_total_mass_is_conserved(self):
        self.assertLessEqual(self.report["worst"]["mass_error"], im.TOLERANCES["mass_error_max"])

    def test_the_first_moment_lands_where_the_channel_puts_it(self):
        self.assertLessEqual(self.report["worst"]["centroid_error"],
                             im.TOLERANCES["centroid_error_max"])

    def test_the_channel_inverts_exactly_from_the_image(self):
        self.assertLessEqual(self.report["worst"]["inverse_k_error"],
                             im.TOLERANCES["inverse_k_error_max"])

    def test_nothing_is_lost_at_the_canvas_edge(self):
        self.assertGreaterEqual(self.report["worst"]["boundary_margin"],
                                im.TOLERANCES["boundary_margin_min"])

    def test_the_tanh_tail_stays_away_from_saturation(self):
        self.assertLessEqual(self.report["worst"]["tanh_max"],
                             im.TOLERANCES["tanh_saturation_max"])

    def test_the_gate_verdict_is_reported_as_an_exact_channel(self):
        self.assertTrue(self.report["exact_channel"])
        self.assertTrue(all(self.report["gates"].values()))


class TestNuisancesCarryNothing(unittest.TestCase):
    def test_the_vertical_shift_is_independent_of_the_channel(self):
        lat = im.latents(6000, 777)
        for j in range(im.N_BLOCKS):
            r = abs(np.corrcoef(lat["style"][:, j], lat["K_eval_only"][:, j])[0, 1])
            self.assertLess(r, 0.05, f"block {j}")

    def test_the_digit_class_is_drawn_independently_of_the_channel(self):
        lat = im.latents(8000, 888)
        for j in range(im.N_BLOCKS):
            means = [lat["K_eval_only"][lat["digit"][:, j] == d, j].mean() for d in range(10)]
            self.assertLess(float(np.std(means)), 0.12, f"block {j}")

    def test_the_vertical_shift_does_not_move_the_horizontal_centroid(self):
        lat = im.latents(400, 999)
        k = lat["K_eval_only"][:, 0]
        base = im.render(k, lat["digit"][:, 0], lat["template"][:, 0], np.zeros(400))
        moved = im.render(k, lat["digit"][:, 0], lat["template"][:, 0], np.full(400, 7.5))
        np.testing.assert_allclose(im.horizontal_centroid(base), im.horizontal_centroid(moved),
                                   atol=1e-9)

    def test_a_changed_channel_moves_the_centroid_as_the_renderer_promises(self):
        lat = im.latents(300, 1234)
        d, t = lat["digit"][:, 0], lat["template"][:, 0]
        s = lat["style"][:, 0]
        for value in (-1.5, 0.0, 2.0):
            images = im.render(np.full(300, value), d, t, s)
            target = im.CENTRE + im.SHIFT_SCALE * np.tanh(value / im.CHANNEL_SCALE)
            np.testing.assert_allclose(im.horizontal_centroid(images), target, atol=1e-9)


class TestRendererRefusesToCrop(unittest.TestCase):
    def test_a_template_pushed_off_the_canvas_raises_instead_of_clipping(self):
        lat = im.latents(50, 4321)
        with self.assertRaises(RuntimeError):
            im.render(lat["K_eval_only"][:, 0], lat["digit"][:, 0], lat["template"][:, 0],
                      np.full(50, 500.0))


if __name__ == "__main__":
    unittest.main()
