"""Checks of the direct image balancing data, recording rule and row split
(PRD memo/2026-09-21-direct-image-balancing-prd-v1.md).

Seeds come from the test namespace 96,900,000-96,999,999.  Run: python3 -B test_direct_balance.py
"""
from __future__ import annotations

import unittest

import numpy as np

import direct_balance_data as db
import direct_balance_p0 as p0
import family_v2_dgp as g
import family_v2_images as im
import family_v2_probe as pr
import run_family_v2 as rf
import run_scm4 as rs
import run_scm4e as rse

T0 = 96_900_000


class Recording(unittest.TestCase):
    def test_at_fifteen_levels_it_is_the_sealed_rule_itself(self):
        data = g.generate(g.LEVELS[db.BASE_LEVEL], 2_000, T0 + 1)
        sds = db.population_sd()
        for key in db.BLOCK_KEYS:
            np.testing.assert_array_equal(db.record(data[key], sds[key], im.LEVELS),
                                          im.record(data[key], sds[key]))

    def test_every_level_is_inside_the_scale_and_the_ends_absorb_the_tails(self):
        sd = np.array([1.0])
        w = np.array([[-99.0], [0.0], [99.0]])
        for levels in db.BENCHMARKS.values():
            got = db.record(w, sd, levels)
            self.assertEqual(got[0, 0], 0)
            self.assertEqual(got[2, 0], levels - 1)
            # with an even number of levels there is no centre level, so the centre lands next to the midpoint
            self.assertLessEqual(abs(got[1, 0] - (levels - 1) / 2), 0.5)
            self.assertTrue(((got >= 0) & (got <= levels - 1)).all())

    def test_a_real_data_set_uses_every_level_at_both_resolutions(self):
        data = g.generate(g.LEVELS[db.BASE_LEVEL], 20_000, T0 + 2)
        for levels in db.BENCHMARKS.values():
            for key, lv in db.recorded(data, levels).items():
                for c in range(lv.shape[1]):
                    self.assertEqual(len(np.unique(lv[:, c])), levels, f"{key} column {c} at L{levels}")

    def test_numbers_replace_the_blocks_and_keep_the_evaluator_keys(self):
        data = g.generate(g.LEVELS[db.BASE_LEVEL], 500, T0 + 3)
        out = db.as_numbers(data, 15)
        for key in db.BLOCK_KEYS:
            self.assertEqual(out[key].shape, data[key].shape)
            self.assertFalse(np.array_equal(out[key], data[key]))
            self.assertTrue(np.array_equal(out[key], np.rint(out[key])))
        for key in ("X", "A", "Y", "U_eval_only", "p_eval_only"):
            self.assertTrue(np.array_equal(out[key], data[key]), key)
        self.assertTrue(np.array_equal(g.learner_view(out)["W1"], out["W1"]))

    def test_a_learner_view_of_the_numbers_carries_nothing_marked_evaluator_only(self):
        data = g.generate(g.LEVELS[db.BASE_LEVEL], 200, T0 + 4)
        view = g.learner_view(db.as_numbers(data, 10))
        self.assertFalse([k for k in view if k.endswith("_eval_only")])
        self.assertEqual(set(view), {"X", "A", "Y", *db.BLOCK_KEYS})


class RowSplit(unittest.TestCase):
    def test_the_five_row_sets_are_disjoint_cover_everything_and_have_the_stated_sizes(self):
        rows = db.split_rows(db.N, T0 + 10)
        self.assertEqual({k: len(v) for k, v in rows.items()}, db.ROLES)
        every = np.concatenate(list(rows.values()))
        self.assertEqual(len(np.unique(every)), db.N)
        for role, r in rows.items():
            self.assertTrue(np.array_equal(r, np.sort(r)), role)

    def test_it_is_the_same_on_a_rerun_and_different_for_another_data_set(self):
        self.assertTrue(np.array_equal(db.split_rows(db.N, T0 + 11)["check"],
                                       db.split_rows(db.N, T0 + 11)["check"]))
        self.assertFalse(np.array_equal(db.split_rows(db.N, T0 + 11)["check"],
                                        db.split_rows(db.N, T0 + 12)["check"]))

    def test_a_different_n_is_refused_rather_than_silently_reshaped(self):
        with self.assertRaises(ValueError):
            db.split_rows(24_000, T0 + 13)


class Blocks(unittest.TestCase):
    def test_the_candidate_splits_are_the_sealed_thirty(self):
        self.assertEqual(db.all_splits(), pr.all_splits())
        self.assertEqual(len(db.all_splits()), 30)
        self.assertEqual([db.split_name(s) for s in db.all_splits()][:3], ["S1", "S2", "S3"])

    def test_the_two_images_of_the_fifth_block_always_move_together(self):
        for split in db.all_splits():
            held, kept = db.image_columns(split)
            self.assertEqual(sorted(held.tolist() + kept.tolist()), list(range(db.N_IMAGES)))
            self.assertEqual(len(set(held.tolist()) & set(kept.tolist())), 0)
            together = {4, 5}
            self.assertIn(together & set(held.tolist()), ({}, set(), together))
            if 4 in held:
                self.assertIn(5, held)
        self.assertEqual(db.image_columns((4,))[0].tolist(), [4, 5])
        self.assertEqual(db.image_columns((0,))[1].tolist(), [1, 2, 3, 4, 5])

    def test_no_split_holds_out_every_image_or_none(self):
        for split in db.all_splits():
            held, kept = db.image_columns(split)
            self.assertGreater(len(held), 0)
            self.assertGreater(len(kept), 0)


class Layout(unittest.TestCase):
    def test_the_seeds_are_a_range_no_earlier_run_used(self):
        mine = {t["seed"] for t in p0.tasks()}
        earlier = {t["seed"] for stage in ("dev", "confirm")
                   for t in rf.tasks_for(stage) + rs.tasks_for(stage) + rse.tasks_for(stage)}
        earlier |= {g.TEST_SEED + off for off in range(0, 30_000, 1_000)}
        self.assertFalse(mine & earlier)
        self.assertEqual(len(mine), p0.REPS)

    def test_both_benchmarks_are_scored_on_the_same_data_sets(self):
        by = {}
        for t in p0.tasks():
            by.setdefault(t["benchmark"], []).append(t["seed"])
        self.assertEqual(len(by), 2)
        self.assertEqual(by["shapes3d"], by["mnist"])
        self.assertNotEqual(p0.shard_path(p0.tasks()[0]), p0.shard_path(p0.tasks()[-1]))

    def test_sources_exist(self):
        for name in p0.SOURCES:
            self.assertTrue((p0.HERE / name).exists(), name)
        self.assertTrue(p0.PRD.exists())

    def test_the_sealed_files_are_untouched(self):
        import json
        sealed = json.loads(rs.PROTOCOLS["confirm"].read_text())["source_sha256"]
        self.assertEqual([f for f, h in sealed.items() if rf.sha256(p0.HERE / f) != h], [])


class Resolution(unittest.TestCase):
    def test_finer_recording_keeps_more_of_the_coordinate(self):
        data = g.generate(g.LEVELS[db.BASE_LEVEL], 5_000, T0 + 20)
        r2 = {}
        for levels in (10, 15):
            got = p0.resolution(data, levels)
            r2[levels] = min(c["r2_of_the_exact_coordinate"] for cols in got.values() for c in cols)
            self.assertEqual(got["W5"][0]["levels_used"], levels)
        self.assertGreater(r2[15], r2[10])
        self.assertGreater(r2[10], 0.9)



# ----------------------------------------------------------------------------- the wiring of Z and of the critics

import torch  # noqa: E402

import direct_balance_eval as de  # noqa: E402
import direct_balance_net as dn  # noqa: E402

TINY = {"channels": 3, "size": 8}                      # an 8 x 8 image: one convolution, then the 4 x 4 map


def tiny_problem(n: int = 600, seed: int = T0 + 50, informative: bool = True):
    """A bank of 8 x 8 images whose brightness is the level, and rows whose treatment follows the level of image 0.
    Small enough for the CPU, shaped like the real thing: six image columns, five blocks, X of width two."""
    rng = np.random.default_rng(seed)
    levels = rng.integers(0, 4, size=(n, db.N_IMAGES))
    bank = np.stack([np.full((8, 8, 3), 40 + 50 * k, dtype=np.uint8) for k in range(4)])
    bank = np.concatenate([bank, bank])                 # two pictures per level
    src = levels + 4 * rng.integers(0, 2, size=levels.shape)
    x = rng.standard_normal((n, 2)) * np.array([3.0, 0.5]) + np.array([10.0, -2.0])
    signal = levels[:, 0] - 1.5 if informative else np.zeros(n)
    a = (rng.random(n) < 1 / (1 + np.exp(-(1.2 * signal + 0.4 * (x[:, 0] - 10) / 3)))).astype(float)
    y = a + 0.5 * signal + rng.standard_normal(n)
    return bank, src, x, a, y


class OneZ(unittest.TestCase):
    def test_z_is_x_joined_with_h_and_x_arrives_untouched(self):
        torch.manual_seed(T0 + 51)
        rep = dn.Representation(np.arange(5), x_dim=2, spec=TINY)
        x = torch.randn(7, 2)
        z = rep([torch.rand(7, 3, 8, 8) for _ in range(5)], x)
        self.assertEqual(z.shape[1], 2 + dn.REPRESENTATION_DIM)
        self.assertTrue(torch.equal(z[:, :2], x))

    def test_every_stage_consumes_the_same_z(self):
        """The bug this guards against: critics and the check saw h, the estimate saw (X, h)."""
        bank, src, x, a, y = tiny_problem()
        dev = torch.device("cpu")
        fit_rows = np.arange(300)
        data = dn.Batches(bank, src, x, a, dev, standardise_rows=fit_rows)
        split = (0,)
        held, kept = db.image_columns(split)
        torch.manual_seed(T0 + 52)
        rep = dn.Representation(kept, 2, spec=TINY)
        z = dn.representation_values(rep, data, fit_rows, kept, grad=False)
        width = 2 + dn.REPRESENTATION_DIM
        self.assertEqual(z.shape[1], width)
        np.testing.assert_allclose(z[:, :2].numpy(), ((x - data.x_mu) / data.x_sd)[fit_rows], atol=1e-5)
        base, aug, _ = dn.fit_critics(rep, data, fit_rows, kept, held, 5, np.random.default_rng(1), None, spec=TINY)
        self.assertEqual(base.net[0].in_features, width)
        self.assertEqual(aug.net[0].in_features, width)
        seen = {}
        saved = de.aipw_from_features
        try:
            de.aipw_from_features = lambda f_n, a_n, y_n, f_e, a_e, y_e, seed, dev: seen.update(n=f_n, e=f_e) or {}
            de.estimate(rep, data, a, y, {"nuisance": np.arange(300, 450), "evaluate": np.arange(450, 600)}, split, 3)
        finally:
            de.aipw_from_features = saved
        self.assertEqual(seen["n"].shape[1], width)
        want = dn.representation_values(rep, data, np.arange(300, 450), kept, grad=False).numpy()
        np.testing.assert_allclose(seen["n"], want, atol=1e-6)

    def test_one_scaling_of_x_is_fixed_on_the_rows_named_and_used_for_all_rows(self):
        bank, src, x, a, _ = tiny_problem()
        data = dn.Batches(bank, src, x, a, torch.device("cpu"), standardise_rows=np.arange(100))
        np.testing.assert_allclose(data.x_mu, x[:100].mean(0))
        got = data.x.numpy()
        np.testing.assert_allclose(got[:100].mean(0), np.zeros(2), atol=1e-5)
        self.assertGreater(abs(got[100:].mean(0)).max(), 1e-4)       # the other rows were not used to centre it


class Nesting(unittest.TestCase):
    def test_the_returned_augmented_critic_is_never_worse_than_the_base_on_the_rows_it_was_fitted_on(self):
        for informative in (True, False):
            bank, src, x, a, _ = tiny_problem(informative=informative)
            data = dn.Batches(bank, src, x, a, torch.device("cpu"), standardise_rows=np.arange(400))
            held, kept = db.image_columns((0,))
            torch.manual_seed(T0 + 60)
            rep = dn.Representation(kept, 2, spec=TINY)
            rows, state = np.arange(400), None
            for round_ in range(3):                                   # cold start, then two warm starts
                base, aug, state = dn.fit_critics(rep, data, rows, kept, held, 8, np.random.default_rng(round_),
                                                  state, spec=TINY)
                info = state["info"]
                got = dn.signed_gap(rep, base, aug, data, rows, kept, held)
                self.assertGreaterEqual(got["signed_gap"], -1e-6, (informative, round_, info))
                if info["kept_the_lifted_base"]:
                    self.assertAlmostEqual(got["signed_gap"], 0.0, places=6)

    def test_keeping_the_lifted_base_makes_q1_equal_q0_exactly(self):
        torch.manual_seed(T0 + 61)
        base = dn.BaseCritic(4)
        aug = dn.AugmentedCritic(base, 1, 4, TINY)
        with torch.no_grad():
            aug.extra[-1].weight.normal_()
        z, held = torch.randn(9, 4), [torch.rand(9, 3, 8, 8)]
        self.assertGreater(float((aug(z, held) - base(z)).abs().max()), 1e-4)
        aug.net.load_state_dict(base.net.state_dict())
        torch.nn.init.zeros_(aug.extra[-1].weight); torch.nn.init.zeros_(aug.extra[-1].bias)
        self.assertEqual(float((aug(z, held) - base(z)).abs().max()), 0.0)


class FreshCheck(unittest.TestCase):
    def test_it_finds_an_imbalance_that_is_there_and_not_one_that_is_not(self):
        dev = torch.device("cpu")
        found = {}
        for informative in (True, False):
            bank, src, x, a, _ = tiny_problem(n=1_600, informative=informative)
            data = dn.Batches(bank, src, x, a, dev, standardise_rows=np.arange(800))
            _, kept = db.image_columns((0,))
            torch.manual_seed(T0 + 70)
            rep = dn.Representation(kept, 2, spec=TINY).eval()        # untrained: it has removed nothing
            found[informative] = de.check(rep, data, np.arange(800, 1_600), (0,), T0 + 71, TINY)
        self.assertGreater(found[True]["gap"], 5 * max(found[False]["gap"], 1e-4))
        self.assertGreater(found[True]["upper"], 1.5e-3)
        self.assertTrue(found[True]["finite"] and found[False]["finite"])


class Schemes(unittest.TestCase):
    def test_cross_fitting_scores_each_half_with_the_critics_of_the_other_half(self):
        bank, src, x, a, _ = tiny_problem()
        data = dn.Batches(bank, src, x, a, torch.device("cpu"), standardise_rows=np.arange(400))
        rows = {"represent": np.arange(400), "select": np.arange(400, 600)}
        saved = dict(dn.TRAIN)
        seen = []
        real_fit, real_step = dn.fit_critics, dn.cnn_step
        try:
            dn.TRAIN.update({"critic_initial": 3, "outer_iters": 2, "critic_per_iter": 2, "checkpoints": (0, 1, 2)})
            dn.fit_critics = lambda rep, d, fit_rows, *args, **kw: (seen.append(("fit", set(fit_rows.tolist()))),
                                                                     real_fit(rep, d, fit_rows, *args, **kw))[1]
            dn.cnn_step = lambda rep, groups, *args, **kw: (seen.append(("score", [set(g_[0].tolist()) for g_ in groups])),
                                                            real_step(rep, groups, *args, **kw))[1]
            out = dn.train_split(data, rows, (0,), T0 + 80, 2, spec=TINY, crossfit=True)
        finally:
            dn.TRAIN.clear(); dn.TRAIN.update(saved)
            dn.fit_critics, dn.cnn_step = real_fit, real_step
        fits = [v for k, v in seen if k == "fit"]
        h1, h2 = fits[0], fits[1]
        self.assertEqual(len(h1 & h2), 0)
        self.assertEqual(h1 | h2, set(range(400)))
        first_score = next(v for k, v in seen if k == "score")
        self.assertEqual(first_score, [h2, h1])                      # critics of half one score half two
        self.assertEqual(out["scheme"], "cross-fitted halves")
        self.assertEqual(sorted(out["checkpoint_states"]), [0, 1, 2])

    def test_the_whole_fold_scheme_is_one_group_on_the_fold_itself(self):
        bank, src, x, a, _ = tiny_problem()
        data = dn.Batches(bank, src, x, a, torch.device("cpu"), standardise_rows=np.arange(400))
        rows = {"represent": np.arange(400), "select": np.arange(400, 600)}
        saved = dict(dn.TRAIN)
        try:
            dn.TRAIN.update({"critic_initial": 3, "outer_iters": 1, "critic_per_iter": 2, "checkpoints": (0, 1)})
            out = dn.train_split(data, rows, (0,), T0 + 81, 2, spec=TINY)
        finally:
            dn.TRAIN.clear(); dn.TRAIN.update(saved)
        self.assertEqual(out["scheme"], "whole fold")
        self.assertEqual(len(out["history"][0]["critics"]), 1)

    def test_fresh_selection_reads_no_outcome_and_no_latent_and_breaks_ties_towards_the_earliest(self):
        bank, src, x, a, _ = tiny_problem()
        data = dn.Batches(bank, src, x, a, torch.device("cpu"), standardise_rows=np.arange(400))
        self.assertFalse(hasattr(data, "y"))                         # the selector is never handed Y or U
        rows = {"represent": np.arange(400), "select": np.arange(400, 600)}
        _, kept = db.image_columns((0,))
        torch.manual_seed(T0 + 82)
        state = {k: v.clone() for k, v in dn.Representation(kept, 2, spec=TINY).state_dict().items()}
        got = de.select_with_fresh_critics({0: state, 5: state}, data, rows, (0,), 2, T0 + 83, TINY)
        self.assertEqual(got["chosen"], 0)                           # identical weights, identical gap: earliest
        self.assertEqual(set(got["table"][0]), {"gap", "se", "clipped_gap", "risk_base", "risk_augmented", "critics"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
