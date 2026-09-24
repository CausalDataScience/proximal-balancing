"""Regression checks of the multi-image suite (PRDs memo/2026-09-20-*-multi-image-prd-v1.md, section 15).

Seeds come from the test namespace 98,800,000-98,899,999 only.  Run: python3 -B test_multi_image_suite.py
"""
from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

import multi_image_banks as banks
import multi_image_probe as mp
import multi_image_scm as scm
from sim5_cnn_algorithm1 import aggregate, aipw_scores

T0 = 98_800_000


class Scalar(unittest.TestCase):
    def test_reference_functionals_match_the_prd(self):
        for name, value in scm.reference_values().items():
            self.assertLess(abs(value - scm.REFERENCE[name]), scm.REFERENCE_TOL, name)

    def test_state_table_is_a_probability_table_with_overlap(self):
        t = scm.state_table()
        self.assertEqual(len(t["p"]), 128)
        self.assertAlmostEqual(float(t["p"].sum()), 1.0, places=14)
        self.assertGreaterEqual(scm.overlap_range()[0], 0.05 - 1e-12)
        self.assertLessEqual(scm.overlap_range()[1], 0.95 + 1e-12)

    def test_sampler_matches_the_exact_table(self):
        d = scm.sample(400_000, T0 + 1)
        ev, a, y = d["evaluator"], d["learner"]["A"], d["learner"]["Y"]
        naive = y[a == 1].mean() - y[a == 0].mean()
        se = np.sqrt(y[a == 1].var() / (a == 1).sum() + y[a == 0].var() / (a == 0).sum())
        self.assertLess(abs(naive - scm.REFERENCE["unadjusted"]), 4 * se)
        for key, p in (("V1", scm.P_V1), ("V2", scm.P_V2)):
            self.assertLess(abs((ev[key] == ev["Uc"]).mean() - p), 0.004)
        self.assertLess(abs((d["learner"]["X"][:, 0] == ev["Uc"]).mean() - scm.P_X1), 0.004)
        self.assertTrue(np.array_equal(ev["V3"], ((ev["Uc"] == 1) | (ev["Ut"] == 1)).astype(int)))
        self.assertTrue(set(np.unique(ev["M"])) == {-1, 1})
        self.assertLess(abs(a.mean() - 0.5), 0.004)

    def test_learner_fields_carry_nothing_privileged(self):
        d = scm.sample(100, T0 + 2)
        self.assertEqual(set(d["learner"]), {"row_id", "X", "A", "Y"})

    def test_witness(self):
        w = scm.witness_report()
        self.assertAlmostEqual(w["target"], 1.0, places=12)
        for entry in w["splits"].values():
            self.assertLess(entry["balance_gap"], 1e-20)
            self.assertLess(entry["heldout_channel_gap"], 1e-20)
        for det in w["completeness_det_by_cell"]:
            self.assertAlmostEqual(det, 0.4, places=12)
        self.assertTrue(w["outcome_free_of_Ut"])

    def test_census_counts(self):
        self.assertEqual(len(scm.all_splits()), 14)
        self.assertEqual(sum(1 for _ in scm.set_partitions(4)), 15)
        self.assertEqual(sum(1 for _ in scm.set_partitions(8)), 4140)


class Banks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = banks.MnistBank()

    def test_kernel_reads_the_sign_alone(self):
        rng = np.random.default_rng(T0 + 10)
        sign = 2 * rng.integers(0, 2, 5000) - 1
        rows = self.bank.draw(sign, np.random.default_rng(T0 + 11))
        self.assertTrue(np.array_equal(self.bank.group(rows), sign))
        # the same signs and the same stream give the same rows whatever else the data set holds
        again = self.bank.draw(sign.copy(), np.random.default_rng(T0 + 11))
        self.assertTrue(np.array_equal(rows, again))
        digits = self.bank._labels[rows]
        for d in range(10):
            self.assertLess(abs((digits == d).mean() - 0.1), 0.02)

    def test_block_streams_are_separate(self):
        d = scm.sample(2000, T0 + 12)
        m = d["evaluator"]["M"]
        src = banks.draw_blocks(self.bank.draw, m, T0 + 13)
        changed = m.copy()
        changed[:, 3] = -changed[:, 3]
        src2 = banks.draw_blocks(self.bank.draw, changed, T0 + 13)
        self.assertTrue(np.array_equal(src[:, :3], src2[:, :3]))
        for j in range(scm.J):
            self.assertTrue(np.array_equal(self.bank.group(src[:, j]), m[:, j]))

    def test_learner_pixels_invert_to_the_source_bytes(self):
        rows = np.arange(0, 60_000, 997)
        back = np.rint(self.bank.images(rows)[:, 0] * 255.0).astype(np.uint8)
        self.assertTrue(np.array_equal(back, self.bank.pixels[rows]))
        self.assertFalse(self.bank.pixels.flags.writeable)

    def test_no_image_is_shared_by_the_two_groups(self):
        audit = self.bank.collision_audit()
        self.assertTrue(audit["exact_decoding_certificate"], audit["duplicates_across_groups"][:3])

    def test_collision_detector_sees_a_planted_collision(self):
        flat = np.arange(12, dtype=np.uint8).reshape(4, 3)
        flat[3] = flat[0]
        self.assertFalse(banks._collisions(flat, np.array([-1, -1, 1, 1]))["exact_decoding_certificate"])
        self.assertTrue(banks._collisions(flat, np.array([-1, 1, 1, -1]))["exact_decoding_certificate"])

    def test_shapes3d_index_arithmetic(self):
        rng = np.random.default_rng(T0 + 14)
        sign = 2 * rng.integers(0, 2, 4000) - 1
        rows = banks.shapes3d_draw(sign, rng)
        self.assertTrue(np.array_equal(banks.shapes3d_group(rows), sign))
        self.assertTrue(((rows >= 0) & (rows < banks.SHAPES3D_ROWS)).all())
        self.assertEqual(int(banks.shapes3d_row(np.array([[9, 9, 9, 7, 3, 14]]))[0]), banks.SHAPES3D_ROWS - 1)
        self.assertEqual(int(banks.shapes3d_row(np.array([[0, 0, 0, 0, 0, 1]]))[0]), 1)


def fixture(n: int = 256, seed: int = T0 + 20):
    d = scm.sample(n, seed)
    bank = banks.MnistBank()
    src = banks.draw_blocks(bank.draw, d["evaluator"]["M"], seed + 1)
    dev = torch.device("cpu")
    view = mp.BalanceView(d["learner"]["X"], d["learner"]["A"], src, torch.as_tensor(np.array(bank.pixels)), dev)
    return d, view, mp.roles(n, seed + 2)


class Learner(unittest.TestCase):
    def test_roles_partition_the_rows(self):
        role = mp.roles(24_000, T0 + 30)
        self.assertEqual({k: len(v) for k, v in role.items()}, mp.ROLES)
        allrows = np.concatenate(list(role.values()))
        self.assertEqual(len(np.unique(allrows)), 24_000)

    def test_balance_view_has_no_outcome(self):
        _, view, _ = fixture()
        self.assertFalse(any("y" == name.lower() for name in vars(view)))

    def test_encoder_and_critic_touch_disjoint_blocks(self):
        _, view, role = fixture()
        view.log = True
        sp = mp.Split((2, 4), view.device)
        enc, q0, q1 = mp.Encoder(1), mp.BaseCritic(), mp.AugmentedCritic(1)
        z = mp.encode_all(enc, view, role["Dselect"], sp)
        mp.predict(q0, q1, view, role["Dselect"], z, sp)
        self.assertEqual({blocks for who, blocks in view.served if who == "encoder"}, {(1, 3)})
        self.assertEqual({blocks for who, blocks in view.served if who == "critic"}, {(2, 4)})

    def test_heldout_pixels_cannot_move_the_representation(self):
        _, view, role = fixture()
        sp = mp.Split((2,), view.device)
        enc = mp.Encoder(1)
        z = mp.encode_all(enc, view, role["Dselect"], sp)
        other = copy.copy(view)
        other.src = view.src.clone()
        other.src[:, 1] = torch.flip(view.src[:, 1], dims=[0])
        self.assertTrue(torch.equal(z, mp.encode_all(enc, other, role["Dselect"], sp)))

    def test_lifted_critic_is_the_base_function_and_shares_nothing(self):
        _, view, role = fixture()
        sp = mp.Split((1, 3), view.device)
        torch.manual_seed(T0 + 31)
        enc, q0, q1 = mp.Encoder(1), mp.BaseCritic(), mp.AugmentedCritic(1)
        for p in q1.residual[-1].parameters():
            torch.nn.init.normal_(p)
        z = mp.encode_all(enc, view, role["Dselect"], sp)
        p0, p1 = mp.predict(q0, q1, view, role["Dselect"], z, sp)
        self.assertGreater(np.abs(p0 - p1).max(), 1e-4)
        q1.load_state_dict(q1.lifted_state(q0))
        p0, p1 = mp.predict(q0, q1, view, role["Dselect"], z, sp)
        self.assertTrue(np.array_equal(p0, p1))
        mine = {id(p) for p in q1.parameters()}
        self.assertFalse(mine & {id(p) for p in q0.parameters()})

    def test_search_moves_the_trunk_and_keeps_the_initial_checkpoint(self):
        """With the augmented critic's own state in force and enough steps to overfit the few fit rows, the fit
        gap is positive, so the clipped loss has a gradient and the encoder, trunk included, has to move."""
        _, view, role = fixture()
        saved, chooser = dict(mp.SEARCH), mp.select_augmented
        mp.SEARCH.update(initial_critic_steps=80, outer=2, critic_steps=5, encoder_steps=3)
        mp.select_augmented = lambda q0, q1, *rest: (copy.deepcopy(q1), "own")
        try:
            out = mp.search(view, role, (1,), 1, T0 + 32)
        finally:
            mp.SEARCH.update(saved)
            mp.select_augmented = chooser
        self.assertEqual(len(out["history"]), 3)
        self.assertEqual(mp.state_digest(out["untrained"]), out["initial_digest"])
        self.assertEqual(out["history"][0]["parameter_delta"], 0.0)
        self.assertGreater(out["history"][0]["fit"]["signed_gap"], 0.0)
        self.assertGreater(max(out["history"][1]["encoder_grad_norms"]), 0.0)
        self.assertGreater(out["history"][-1]["parameter_delta"], 0.0)
        self.assertGreater(out["history"][-1]["trunk_delta"], 0.0)

    def test_zero_gap_gives_the_encoder_no_gradient(self):
        """When the lifted base is the selected augmented critic the two risks are one function of the encoder,
        so the encoder does not move.  The PRD asks for this case to be recorded, not repaired."""
        _, view, role = fixture()
        saved, chooser = dict(mp.SEARCH), mp.select_augmented

        def lifted(q0, q1, *rest):
            out = copy.deepcopy(q1)
            out.load_state_dict(q1.lifted_state(q0))
            return out, "lifted"

        mp.SEARCH.update(initial_critic_steps=4, outer=2, critic_steps=2, encoder_steps=2)
        mp.select_augmented = lifted
        try:
            out = mp.search(view, role, (2,), 1, T0 + 34)
        finally:
            mp.SEARCH.update(saved)
            mp.select_augmented = chooser
        self.assertEqual(out["history"][-1]["parameter_delta"], 0.0)
        self.assertEqual(max(out["history"][1]["encoder_grad_norms"]), 0.0)
        self.assertFalse(out["selected_was_updated"])

    def test_audit_is_fail_closed_and_stores_row_differences(self):
        _, view, role = fixture()
        saved = dict(mp.AUDIT)
        mp.AUDIT.update(steps=4, every=2)
        try:
            out = mp.audit(view, role, (2,), mp.Encoder(1), 1, T0 + 33)
        finally:
            mp.AUDIT.update(saved)
        self.assertEqual(len(out["audit_d"]), len(role["Daudit"]))
        self.assertAlmostEqual(float(out["audit_d"].mean()), out["Daudit"]["signed_gap"], places=12)
        self.assertEqual(out["screen"]["pass"], out["Daudit"]["clipped_gap"] <= mp.AUDIT["threshold"])


class Estimation(unittest.TestCase):
    def test_aipw_recovers_tau_with_true_nuisances(self):
        d = scm.sample(200_000, T0 + 40)
        ev, a, y = d["evaluator"], d["learner"]["A"], d["learner"]["Y"]
        scores = aipw_scores(ev["e"], ev["mu0"], ev["mu0"] + scm.TAU, a, y)
        self.assertLess(abs(scores.mean() - 1.0), 4 * scores.std(ddof=1) / np.sqrt(len(scores)))

    def test_oracle_features_give_tau(self):
        d = scm.sample(24_000, T0 + 41)
        ev, lr = d["evaluator"], d["learner"]
        feat = np.column_stack([lr["X"], ev["Uc"], ev["Ut"]])
        out = mp.aipw_from_features(feat, lr["A"], lr["Y"], mp.roles(24_000, T0 + 42), T0 + 43)
        self.assertTrue(out["finite"])
        self.assertLess(abs(out["theta"] - 1.0), 0.06)
        self.assertGreaterEqual(out["overlap"]["clipped_range"][0], mp.ETA)

    def test_nonfinite_features_fail_closed(self):
        feat = np.full((24_000, 2), np.nan)
        out = mp.aipw_from_features(feat, np.zeros(24_000), np.zeros(24_000), mp.roles(24_000, T0 + 44), T0 + 45)
        self.assertFalse(out["finite"])

    def test_graph_rules(self):
        self.assertEqual(aggregate([], [], 0.1)["return_kind"], "no_return")
        self.assertEqual(aggregate(["a"], [np.nan], 0.1)["return_kind"], "no_return")
        self.assertEqual(aggregate(["a", "b"], [0.0, 1.0], 0.1)["return_kind"], "tied_largest")
        edge = aggregate(["a", "b", "c"], [1.0, 1.2, 2.0], 0.1)          # |1.0 - 1.2| = 2 rho links
        self.assertEqual((edge["return_kind"], edge["output"]), ("unique_largest", 1.1))
        chain = aggregate(["a", "b", "c", "d"], [1.0, 1.15, 1.3, 3.0], 0.1)
        self.assertEqual(chain["members"], [["a", "b", "c"]])
        picked = mp.combine({"S2": 1.0, "S4": 1.3, "S24": 1.02, "S1": 0.4},
                            {"S2": True, "S4": True, "S24": True, "S1": False})
        self.assertAlmostEqual(picked["output"], 1.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
