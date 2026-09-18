#!/usr/bin/env python3
"""Focused implementation-contract tests for Sim-5 v2 Revision 3."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import sim5_digit_proxy as sim
import run_sim5_digit_proxy as runner


class Sim5DigitProxyTests(unittest.TestCase):
    def test_quick_orchestration_runs_all_stages_without_latent_input(self):
        torch.set_num_threads(1)
        result = runner.fixed_split_smoke(811, quick=True, bank=self.tiny_bank(),
                                          budget_override={"max_epochs": 1, "patience": 1})
        self.assertTrue(result["complete"])
        self.assertTrue(result["diagnostic_quick"])
        self.assertTrue(result["implementation_pass"])
        self.assertEqual(result["sizes"]["E"], 400)
        self.assertEqual(result["restarts"][0]["head_update_count"], 100)
        self.assertEqual(result["selected"]["restart_id"], 0)
        self.assertNotEqual(result["audits"]["warm"]["seed"], result["audits"]["final"]["seed"])
        json.dumps(result, allow_nan=False)

    @staticmethod
    def fit_test_warm(view, retained, train_idx, val_idx, seed, budget):
        bank = sim.fit_block_encoder_bank(view, train_idx, val_idx, seed=seed + 10000, budget=budget)
        return sim.fit_warm_fusion(view, bank, retained, train_idx, val_idx, seed=seed, budget=budget)

    def test_shared_bank_is_fitted_once_and_frozen_across_split_warmstarts(self):
        torch.set_num_threads(1)
        data = sim.generate_latents(24, seed=711)
        images = sim.render_images(data, seed=712, bank=self.tiny_bank())
        view = sim.learner_view({**{k: data[k] for k in ("X", "A", "Y")},
                                 "images": images, "row_id": np.arange(24)})
        tr, va, tiny = np.arange(18), np.arange(18, 24), {"max_epochs": 1}
        bank = sim.fit_block_encoder_bank(view, tr, va, seed=713, budget=tiny)
        self.assertEqual(bank["block_fit_count"], 5)
        for retained in ([1, 2, 3, 4], [0, 2, 3, 4]):
            warm = sim.fit_warm_fusion(view, bank, retained, tr, va, seed=714, budget=tiny)
            for encoder, j in zip(warm["model"].encoders, retained):
                self.assertEqual(sim.state_digest(encoder), bank["metadata"][j]["digest"])
                self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
            self.assertGreater(warm["representation_parameter_change_l2"], 0.)
        for j in range(5):
            self.assertEqual(sim.state_digest(bank["encoders"][j]), bank["metadata"][j]["digest"])

    def test_aipw_formula_recovers_exact_unit_effect_without_residuals(self):
        a = np.array([0., 1., 1., 0.])
        m0 = np.array([-.4, .2, .8, -2.])
        m1 = m0 + 1
        y = np.where(a == 1, m1, m0)
        scores = sim.aipw_scores(a, y, np.array([.1, .4, .6, .9]), m0, m1)
        self.assertTrue(np.array_equal(scores, np.ones(4)))

    def test_nuisance_and_evaluation_keep_phi_frozen_and_e_out_of_fit(self):
        torch.set_num_threads(1)
        data = sim.generate_latents(48, seed=611)
        images = sim.render_images(data, seed=612, bank=self.tiny_bank())
        view = sim.learner_view({**{k: data[k] for k in ("X", "A", "Y")},
                                 "images": images, "row_id": np.arange(48)})
        warm = self.fit_test_warm(view, [1, 2, 3, 4], np.arange(12), np.arange(12, 16),
                                           seed=613, budget={"max_epochs": 1})
        digest = sim.state_digest(warm["model"])
        nu = sim.fit_nuisance(warm["model"], view, warm["standardizer"], np.arange(16, 36),
                              seed=614, budget={"max_epochs": 1})
        result = sim.aipw_evaluate(warm["model"], view, warm["standardizer"], nu, np.arange(36, 48))
        self.assertEqual(sim.state_digest(warm["model"]), digest)
        self.assertTrue(np.isfinite(result["theta"]))
        self.assertTrue(result["propensity"]["structural_range_pass"])
        self.assertEqual(result["n_e"], 12)
        self.assertFalse(set(result["eval_row_ids"]) & set(nu["fit_row_ids"] + nu["val_row_ids"]))
        with self.assertRaises(ValueError):
            sim.aipw_evaluate(warm["model"], view, warm["standardizer"], nu, np.arange(20, 40))

    def test_selection_ignores_audit_labels_and_audit_refits_are_independent(self):
        torch.set_num_threads(1)
        data = sim.generate_latents(32, seed=511)
        images = sim.render_images(data, seed=512, bank=self.tiny_bank())
        view = sim.learner_view({**{k: data[k] for k in ("X", "A", "Y")},
                                 "images": images, "row_id": np.arange(32)})
        tr, va, se, au = np.arange(16), np.arange(16, 22), np.arange(22, 27), np.arange(27, 32)
        tiny = {"max_epochs": 1}
        warm = self.fit_test_warm(view, [1, 2, 3, 4], tr, va, seed=513, budget=tiny)
        balanced = sim.balance_updates(warm, view, tr, va, seed=514, critic_budget=tiny)
        selected = sim.select_checkpoint(balanced, view, tr, va, se, seed=515, budget=tiny)
        changed = dict(view)
        changed["A"] = view["A"].copy()
        changed["A"][au] = 1 - changed["A"][au]
        again = sim.select_checkpoint(balanced, changed, tr, va, se, seed=515, budget=tiny)
        self.assertEqual(selected["selected"], again["selected"])
        audited = sim.audit_warm_final(selected, view, tr, va, au, seed=516, budget=tiny)
        self.assertNotEqual(audited["warm"]["seed"], audited["final"]["seed"])
        self.assertEqual(audited["warm"]["state_sha256"], warm["warm_digest"])
        self.assertEqual(audited["selected_state_sha256"], sim.state_digest(selected["model"]))
        self.assertTrue(audited["selection_unchanged"])
        self.assertTrue(audited["final"]["overlap"]["structural_range_pass"])

    def test_overlap_statistics_use_bounded_probability_output(self):
        p = sim.bounded_probability(torch.tensor([-10., 0., 10.]))
        out = sim.overlap_diagnostics(p)
        self.assertAlmostEqual(out["min"], float(p.min()))
        self.assertAlmostEqual(out["q50"], .5)
        self.assertAlmostEqual(out["fraction_outside_0p1_0p9"], 2 / 3)
        self.assertTrue(out["structural_range_pass"])

    def test_balance_records_100_real_updates_and_keeps_encoders_frozen(self):
        torch.set_num_threads(1)
        data = sim.generate_latents(24, seed=411)
        images = sim.render_images(data, seed=412, bank=self.tiny_bank())
        view = sim.learner_view({**{k: data[k] for k in ("X", "A", "Y")},
                                 "images": images, "row_id": np.arange(24)})
        warm = self.fit_test_warm(view, [1, 2, 3, 4], np.arange(18), np.arange(18, 24),
                                           seed=413, budget={"max_epochs": 1})
        result = sim.balance_updates(warm, view, np.arange(18), np.arange(18, 24), seed=414,
                                     critic_budget={"max_epochs": 1})
        self.assertEqual(result["head_update_count"], 100)
        self.assertEqual(len(result["refreshes"]), 21)
        self.assertEqual(sorted(result["checkpoints"]), [0, 25, 50, 100])
        self.assertEqual(sim.state_digest(warm["model"].encoders),
                         sim.state_digest(result["model"].encoders))
        self.assertEqual(result["checkpoints"][0]["state_sha256"], warm["warm_digest"])
        for count, checkpoint in result["checkpoints"].items():
            self.assertEqual(checkpoint["head_update_count"], count)
            self.assertEqual(checkpoint["state_sha256"], sim.state_digest(checkpoint["state"]))
        if any(u["signed_gap"] > 0 and u["gradient_l2"] > 0 for u in result["updates"]):
            self.assertGreater(result["fusion_parameter_change_l2"], 0.)

    def test_warmstart_moves_representation_and_saves_exact_inference_checkpoint(self):
        data = sim.generate_latents(24, seed=311)
        images = sim.render_images(data, seed=312, bank=self.tiny_bank())
        view = sim.learner_view({**{k: data[k] for k in ("X", "A", "Y")},
                                 "images": images, "row_id": np.arange(24)})
        result = self.fit_test_warm(view, [1, 2, 3, 4], np.arange(18), np.arange(18, 24),
                                              seed=313, budget={"max_epochs": 2, "batch_size": 6})
        self.assertTrue(np.isfinite(result["fit"]["val_loss"]))
        self.assertGreater(result["representation_parameter_change_l2"], 0.)
        self.assertEqual(result["fit"]["optimizer_updates"], 6)
        self.assertEqual(result["warm_digest"], sim.state_digest(result["model"]))
        self.assertEqual(result["warm_digest"], sim.state_digest(result["warm_state"]))
        with torch.no_grad():
            z = result["model"](sim.to_tensor(result["standardizer"].x(view["X"])[:, None]),
                                 [sim.to_tensor(view["blocks"][j]) for j in [1, 2, 3, 4]])
        self.assertEqual(tuple(z.shape), (24, 9))

    def test_standardizer_fits_training_rows_only(self):
        x = np.array([1., 3., 1000., 2000.])
        y = np.array([2., 6., -100., -200.])
        st = sim.Standardizer().fit(x, y, np.array([0, 1]))
        self.assertEqual(st.x_mean, 2.)
        self.assertEqual(st.y_mean, 4.)
        self.assertTrue(np.allclose(st.x(x[:2]), [-1., 1.]))

    def test_critic_lift_is_exact_but_parameters_are_independent(self):
        torch.manual_seed(17)
        base = sim.BaseCritic()
        aug = sim.AugmentedCritic(base, 16)
        z, w = torch.randn(12, 9), torch.randn(12, 16)
        self.assertTrue(torch.equal(aug(z, w), aug.base(z)))
        self.assertTrue(torch.equal(aug(z, w), base(z)))
        self.assertTrue(set(p.data_ptr() for p in base.parameters()).isdisjoint(
            set(p.data_ptr() for p in aug.parameters())))
        self.assertGreaterEqual(float(base(z).detach().min()), .05)
        self.assertLessEqual(float(base(z).detach().max()), .95)

    def test_paired_gap_keeps_signed_value_and_paired_standard_error(self):
        a = np.array([0., 1., 0., 1.])
        p0, p1 = np.array([.1, .9, .2, .8]), np.array([.9, .1, .8, .2])
        out = sim.paired_gap(a, p0, p1)
        d = (a - p0)**2 - (a - p1)**2
        self.assertAlmostEqual(out["signed_gap"], d.mean())
        self.assertAlmostEqual(out["se"], d.std(ddof=1) / 2)
        self.assertEqual(out["clipped_gap"], 0.)
        self.assertTrue(out["reliability_fail"])

    def test_fit_predictor_records_real_updates_and_rejects_overlap(self):
        torch.manual_seed(18)
        model = sim.BaseCritic(2)
        x = torch.randn(16, 2)
        a = (x[:, 0] > 0).float()
        result = sim.fit_predictor(model, (x,), a, np.arange(12), np.arange(12, 16),
                                   seed=19, max_epochs=2, batch_size=4)
        self.assertEqual(result["optimizer_updates"], 6)
        self.assertEqual(result["state_sha256"], sim.state_digest(model))
        with self.assertRaises(ValueError):
            sim.fit_predictor(model, (x,), a, np.arange(12), np.arange(10, 16), seed=19)

    @staticmethod
    def tiny_bank():
        images = np.zeros((9, 28, 28), dtype=np.float32)
        for j in range(9):
            images[j, 6:22, 8 + j:10 + j] = 1
        return {"images": images, "labels": np.arange(1, 10)}

    def test_render_is_deterministic_bounded_and_covers_learner_blocks(self):
        data = sim.generate_latents(12, seed=71)
        images = sim.render_images(data, seed=72, bank=self.tiny_bank())
        again = sim.render_images(data, seed=72, bank=self.tiny_bank())
        self.assertTrue(np.array_equal(images, again))
        self.assertEqual(images.shape, (12, 36, 36))
        self.assertGreaterEqual(float(images.min()), 0.)
        self.assertLessEqual(float(images.max()), 1.)
        view = sim.learner_view({**{k: data[k] for k in ("X", "A", "Y")},
                                 "images": images, "row_id": np.arange(12)})
        self.assertEqual([v.shape[1] for v in view["blocks"]], list(sim.BLOCK_DIMS))

    def test_marker_latent_changes_only_marker_with_same_rendering_seed(self):
        low = sim.generate_latents(8, seed=91)
        high = dict(low)
        low["U_t_eval_only"] = np.full(8, -5.)
        high["U_t_eval_only"] = np.full(8, 5.)
        a = sim.render_images(low, seed=92, bank=self.tiny_bank())
        b = sim.render_images(high, seed=92, bank=self.tiny_bank())
        marker = sim.block_masks()[-1]
        self.assertTrue(np.array_equal(a[:, ~marker], b[:, ~marker]))
        self.assertGreater(float(b[:, marker].mean() - a[:, marker].mean()), .70)

    def test_five_anonymous_blocks_partition_every_pixel_once(self):
        masks = sim.block_masks()
        self.assertEqual([int(m.sum()) for m in masks], [324, 324, 324, 308, 16])
        self.assertTrue(np.array_equal(np.sum(masks, axis=0), np.ones((36, 36), dtype=int)))
        self.assertEqual(len(sim.all_splits()), 30)

    def test_outer_rows_are_disjoint_and_have_revision3_sizes(self):
        rows = sim.outer_rows(sim.FULL_SIZES, seed=17)
        self.assertEqual({k: len(v) for k, v in rows.items()}, sim.FULL_SIZES)
        joined = np.concatenate(list(rows.values()))
        self.assertEqual(len(joined), len(np.unique(joined)))
        self.assertEqual(len(joined), 48_000)

    def test_generator_is_deterministic_and_structurally_overlapped(self):
        a = sim.generate_latents(200, seed=9)
        b = sim.generate_latents(200, seed=9)
        for key in ("X", "A", "Y", "p_eval_only", "U_c_eval_only", "U_t_eval_only"):
            self.assertTrue(np.array_equal(a[key], b[key]))
        self.assertGreaterEqual(float(a["p_eval_only"].min()), 0.1)
        self.assertLessEqual(float(a["p_eval_only"].max()), 0.9)

    def test_learner_view_rejects_latent_keys_and_phi_accepts_only_xw(self):
        data = sim.generate_latents(24, seed=4)
        images = np.zeros((24, 36, 36), dtype=np.float32)
        with self.assertRaises(ValueError):
            sim.learner_view(data | {"images": images})
        view = sim.learner_view({"X": data["X"], "A": data["A"], "Y": data["Y"],
                                 "images": images, "row_id": np.arange(24)})
        model = sim.SplitRepresentation([1, 2, 3, 4])
        z = model(sim.to_tensor(view["X"][:, None]),
                  [sim.to_tensor(view["blocks"][j]) for j in (1, 2, 3, 4)])
        self.assertEqual(tuple(z.shape), (24, 9))

    def test_revision3_schedule_has_only_real_update_checkpoints(self):
        self.assertEqual(sim.HEAD_UPDATES, 100)
        self.assertEqual(sim.CRITIC_REFRESH_EVERY, 5)
        self.assertEqual(sim.CHECKPOINT_UPDATES, (0, 25, 50, 100))
        events = sim.training_event_schedule()
        self.assertEqual([e["head_update"] for e in events if e["checkpoint"]], [0, 25, 50, 100])
        self.assertEqual(sum(e["critic_refresh"] for e in events), 21)

    def test_warm_and_final_audits_use_independent_refits(self):
        seeds = sim.audit_refit_seeds(1_234)
        self.assertNotEqual(seeds["warm"], seeds["final"])
        self.assertEqual(len(set(seeds.values())), 2)

    def test_atomic_result_refuses_overwrite_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "result.json"
            sim.write_json_atomic(path, {"complete": True, "value": 1.0})
            self.assertEqual(json.loads(path.read_text())["value"], 1.0)
            with self.assertRaises(FileExistsError):
                sim.write_json_atomic(path, {"complete": True})
            with self.assertRaises(ValueError):
                sim.write_json_atomic(Path(td) / "bad.json", {"x": float("nan")})


if __name__ == "__main__":
    unittest.main()
