"""Checks of the SCM-4e v2 encoder, association front end, pixel CNN and runner
(PRD memo/2026-09-21-scm4e-v2-labelfree-association-prd-v1.md).

Seeds come from the test namespace 98,600,000-98,699,999.  Run: python3 -B test_scm4e_v2.py
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import family_v2_dgp as g
import family_v2_images as im
import family_v2_probe as pr
import run_family_v2 as rf
import run_scm4 as rs
import run_scm4e as rse
import run_scm4e_v2 as rv
import scm4e_encoder as ec
import scm4e_frontend as fe
import scm4e_pixel_cnn as pxe
import summarize_scm4e_v2 as sv

T0 = 98_600_000
KEYS = [f"W{j + 1}" for j in range(g.N_BLOCKS)]


DIM, FIT = 6, 1_200                      # one image's embedding width, and the rows a front end is fitted on


NUISANCE_DIRECTIONS, NUISANCE_SD = 3, 5.0


def planted_view(n: int = 2_400, dim: int = DIM, seed: int = T0, dead: tuple = ()) -> dict:
    """An image embedding in miniature.  Each image has three large directions drawn fresh every time and
    correlated with nothing, which is what floor hue, wall hue and object hue are, and one small direction that
    carries a latent coordinate, which is what the angle is.  Variance points at the first kind and association
    points at the second.  W5 is two images carrying two different coordinates, and a block named in `dead`
    carries none."""
    rng = np.random.default_rng(seed)
    u, v = rng.standard_normal(n), rng.standard_normal(n)
    view = {"A": (u + v + rng.standard_normal(n) > 0).astype(float), "Y": u + rng.standard_normal(n),
            "X": np.column_stack([u + rng.standard_normal(n), v + rng.standard_normal(n)])}
    carried = {"W1": [u], "W2": [u], "W3": [u], "W4": [u - 1.5 * v], "W5": [u, v]}   # as in SCM-1, W4 mixes both
    for key in KEYS:
        images = []
        for latent in carried[key]:
            basis = np.linalg.qr(rng.standard_normal((dim, dim)))[0]
            colour = rng.standard_normal((n, NUISANCE_DIRECTIONS)) * NUISANCE_SD @ basis[:, :NUISANCE_DIRECTIONS].T
            carrier = np.zeros(n) if key in dead else latent + 0.5 * rng.standard_normal(n)
            images.append(colour + np.outer(carrier, basis[:, NUISANCE_DIRECTIONS])
                          + 0.5 * rng.standard_normal((n, dim)))
        view[key] = np.column_stack(images)
    return view


class FrontEnd(unittest.TestCase):
    def test_it_keeps_the_shared_direction_that_variance_would_miss(self):
        view = planted_view()
        rows = np.arange(FIT)
        prep = fe.AssociationPrep(view, rows, DIM)
        self.assertEqual(prep.widths(), [1, 1, 1, 1, 2])
        for key in KEYS:
            self.assertFalse(prep.report[key]["nothing_above_threshold"], key)
        u_like = view["X"][:, 0]
        kept = prep.block(view, np.arange(len(view["A"])), 0)[:, 0]
        self.assertGreater(abs(np.corrcoef(kept, u_like)[0, 1]), 0.3)
        sealed = pr.Prep(view, rows)                       # the sealed rule on the same block, for contrast
        best = max(abs(np.corrcoef(sealed.block(view, np.arange(len(view["A"])), 0)[:, c], u_like)[0, 1])
                   for c in range(sealed.widths()[0]))
        self.assertGreater(abs(np.corrcoef(kept, u_like)[0, 1]), best)

    def test_a_block_with_no_shared_direction_reports_nothing_above_the_floor(self):
        view = planted_view(dead=("W2",))
        prep = fe.AssociationPrep(view, np.arange(FIT), DIM)
        self.assertTrue(prep.report["W2"]["nothing_above_threshold"])
        self.assertFalse(prep.report["W1"]["nothing_above_threshold"])
        self.assertEqual(prep.widths()[1], 1)              # one direction is still handed over, never zero

    def test_the_cap_is_one_direction_per_image(self):
        view = planted_view()
        shared = [view["X"][:, :1], view["X"][:, 1:], view["W2"][:, :1], view["W3"][:, :1]]
        view["W1"] = np.column_stack(shared + [view["W1"]])      # four separate directions the rest can see
        prep = fe.AssociationPrep(view, np.arange(FIT), DIM)
        self.assertEqual(prep.report["W1"]["images"], 1)
        self.assertGreater(prep.report["W1"]["above_threshold"], 1)
        self.assertTrue(prep.report["W1"]["hit_the_image_cap"])
        self.assertEqual(prep.widths()[0], 1)

    def test_the_noise_floor_is_the_same_on_a_rerun_and_moves_with_the_rows(self):
        view = planted_view()
        a = fe.AssociationPrep(view, np.arange(FIT), DIM).report
        b = fe.AssociationPrep(view, np.arange(FIT)[::-1], DIM).report        # same rows, other order
        c = fe.AssociationPrep(view, np.arange(300, 300 + FIT), DIM).report
        self.assertEqual(a["W1"]["noise_floor"], b["W1"]["noise_floor"])
        self.assertNotEqual(a["W1"]["noise_floor"], c["W1"]["noise_floor"])

    def test_it_is_fitted_on_its_rows_only_and_applies_to_any_rows(self):
        view = planted_view()
        rows = np.arange(FIT)
        prep = fe.AssociationPrep(view, rows, DIM)
        moved = dict(view)
        moved["W1"] = view["W1"].copy()
        moved["W1"][FIT:] += 7.0                            # rows outside the fit change nothing about the map
        self.assertTrue(np.allclose(prep.block(view, rows, 0), fe.AssociationPrep(moved, rows, DIM).block(moved, rows, 0)))
        out = prep.block(view, np.arange(FIT, 2 * FIT), 0)
        self.assertEqual(out.shape, (FIT, 1))
        self.assertTrue(np.isfinite(out).all())

    def test_the_fitted_variates_are_standardised_on_the_fold(self):
        view = planted_view()
        rows = np.arange(FIT)
        prep = fe.AssociationPrep(view, rows, DIM)
        got = prep.block(view, rows, 4)
        np.testing.assert_allclose(got.std(0), np.ones(got.shape[1]), atol=1e-6)
        np.testing.assert_allclose(prep.x(view, rows).mean(0), np.zeros(2), atol=1e-9)


class Swap(unittest.TestCase):
    def test_the_slot_is_restored_even_when_the_body_raises(self):
        sealed = pr.Prep
        with self.assertRaises(ValueError):
            with fe.installed(DIM):
                self.assertIsNot(pr.Prep, sealed)
                raise ValueError("stop")
        self.assertIs(pr.Prep, sealed)

    def test_a_second_swap_is_refused(self):
        sealed = pr.Prep
        with fe.installed(DIM):
            with self.assertRaises(RuntimeError):
                with fe.installed(DIM):
                    pass
        self.assertIs(pr.Prep, sealed)

    def test_the_sealed_learner_runs_under_the_swap_and_sees_the_narrow_blocks(self):
        view = planted_view(n=4 * FIT)                  # four folds, so the front end is fitted on FIT rows
        with fe.installed(DIM):
            out = pr.run_dataset(view, g.Level("test", 5, 2, (DIM,) * 4 + (2 * DIM,), 2), 1.0, T0 + 3, keep_b=True)
        self.assertEqual(len(out["split_names"]), 30)
        widths = out["records"]["S1"]["rotations"][0]["prep_widths"]
        self.assertEqual(widths, [1, 1, 1, 1, 2])

    def test_variate_view_narrows_every_block(self):
        view = planted_view()
        out, report = fe.variate_view(view, DIM)
        self.assertEqual([out[k].shape[1] for k in KEYS], [1, 1, 1, 1, 2])
        self.assertEqual(len(out["A"]), len(view["A"]))
        self.assertEqual(set(report), set(KEYS))
        self.assertTrue(np.array_equal(out["X"], view["X"]))


class Encoder(unittest.TestCase):
    def test_the_two_halves_fit_together_and_the_latent_is_the_declared_width(self):
        torch.manual_seed(T0 + 10)
        enc, dec = ec.build()
        x = torch.rand(4, 3, 64, 64)
        z = enc(x)
        self.assertEqual(tuple(z.shape), (4, ec.LATENT))
        self.assertEqual(tuple(dec(z).shape), tuple(x.shape))
        self.assertEqual(list(enc.children())[-1].out_features, ec.LATENT)

    def test_embed_keeps_the_order_and_repeats_of_the_rows_asked_for(self):
        torch.manual_seed(T0 + 11)
        enc, _ = ec.build()
        bank = np.random.default_rng(T0 + 12).integers(0, 255, size=(40, 64, 64, 3), dtype=np.uint8)
        rows = np.array([9, 3, 3, 31, 0, 9])
        got = ec.embed(enc, bank, rows, torch.device("cpu"))
        with torch.no_grad():
            want = enc(im.to_input(torch.as_tensor(bank[rows]))).numpy()
        np.testing.assert_allclose(got, want, atol=1e-6)
        self.assertTrue(np.array_equal(got[1], got[2]))
        self.assertTrue(np.array_equal(got[0], got[5]))

    def test_the_training_images_cover_every_factor_value_and_avoid_the_causal_bank(self):
        rows = ec.training_rows()
        self.assertEqual(len(rows), ec.ENCODER["images"])
        split = np.load(im.BANK_DIR / "split_rows.npz")
        self.assertEqual(len(np.intersect1d(rows, split["causal_bank"])), 0)
        self.assertEqual(len(np.intersect1d(rows, split["reader_selection"])), 0)
        facts = im.factor_index(rows)
        self.assertEqual(tuple(len(np.unique(facts[:, k])) for k in range(6)), im.SIZES)

    def test_the_check_passes_on_an_embedding_that_carries_the_angle_and_fails_on_one_that_does_not(self):
        rows = np.load(im.BANK_DIR / "split_rows.npz")["reader_selection"]
        level = im.factor_index(rows)[:, im.ORIENTATION].astype(float)
        rng = np.random.default_rng(T0 + 20)
        saved = ec.embed
        try:
            for carries, expected in ((True, True), (False, False)):
                base = np.column_stack([level + rng.standard_normal(len(rows)) * (0.5 if carries else 40.0),
                                        rng.standard_normal((len(rows), ec.LATENT - 1))])
                ec.embed = lambda *a, _z=base, **k: _z
                out = ec.check(None, None, None)
                self.assertIs(out["pass"], expected, out["level_r2"])
                self.assertEqual(out["scored_rows"] + out["fit_rows"], len(rows))
        finally:
            ec.embed = saved

    def test_a_non_finite_embedding_fails_the_check_before_anything_is_scored(self):
        saved = ec.embed
        try:
            rows = len(np.load(im.BANK_DIR / "split_rows.npz")["reader_selection"])
            bad = np.zeros((rows, ec.LATENT)); bad[5, 0] = np.nan
            ec.embed = lambda *a, _z=bad, **k: _z
            out = ec.check(None, None, None)
        finally:
            ec.embed = saved
        self.assertFalse(out["pass"])
        self.assertEqual(out["finite_rows"], rows - 1)

    def test_load_refuses_a_failed_check_and_weights_that_do_not_match_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = (ec.WEIGHTS, ec.RECORD, ec.CHECK_FILE)
            try:
                ec.WEIGHTS, ec.RECORD, ec.CHECK_FILE = (Path(tmp) / n for n in
                                                        ("w.pt", "r.json", "c.json"))
                enc, _ = ec.build()
                torch.save(enc.state_dict(), ec.WEIGHTS)
                digest = im.file_sha256(ec.WEIGHTS)
                source = im.file_sha256(Path(ec.__file__).resolve())
                ec.RECORD.write_text(json.dumps({"weights_sha256": digest, "trained_by": {"sha256": "another file"}}))
                ec.CHECK_FILE.write_text(json.dumps({"pass": True, "encoder_sha256": digest}))
                with self.assertRaises(SystemExit):                  # the module is sealed by the weights it trained
                    ec.load(torch.device("cpu"))
                ec.RECORD.write_text(json.dumps({"weights_sha256": digest, "trained_by": {"sha256": source}}))
                ec.CHECK_FILE.write_text(json.dumps({"pass": False, "encoder_sha256": digest}))
                with self.assertRaises(SystemExit):
                    ec.load(torch.device("cpu"))
                ec.CHECK_FILE.write_text(json.dumps({"pass": True, "encoder_sha256": "other"}))
                with self.assertRaises(SystemExit):
                    ec.load(torch.device("cpu"))
                ec.CHECK_FILE.write_text(json.dumps({"pass": True, "encoder_sha256": digest}))
                got, _, digests = ec.load(torch.device("cpu"))
                self.assertEqual(digests["weights_sha256"], digest)
                self.assertEqual(list(got.children())[-1].out_features, ec.LATENT)
            finally:
                ec.WEIGHTS, ec.RECORD, ec.CHECK_FILE = saved


class PixelNet(unittest.TestCase):
    def test_it_starts_holding_exactly_the_embedding_the_front_end_works_on(self):
        torch.manual_seed(T0 + 30)
        enc, _ = ec.build()
        state = {k: v.detach().clone() for k, v in enc.state_dict().items()}
        net = pxe.EncoderPixelNet(state, x_dim=2, n_images=6)
        x = torch.rand(3, 3, 64, 64)
        with torch.no_grad():
            self.assertTrue(torch.allclose(net.project(net.tower(x)), enc(x), atol=1e-6))
        self.assertTrue(any(n.startswith("tower.") for n, _ in net.named_parameters()))
        self.assertEqual(net.body[0].in_features, 6 * ec.LATENT + 2)

    def test_the_sealed_class_is_restored_after_an_estimate(self):
        import scm4_pixel_cnn as px
        sealed = px.PixelNet
        saved = px.estimate
        try:
            px.estimate = lambda *a, **k: {"finite": True, "ate": 1.0, "config": px.CONFIG}
            out = pxe.estimate(None, None, None, None, None, {}, 1, None)
        finally:
            px.estimate = saved
        self.assertIs(px.PixelNet, sealed)
        self.assertEqual(out["config"]["feature_dim"], ec.LATENT)


class Layout(unittest.TestCase):
    def test_block_layout_and_seeds_no_earlier_run_used(self):
        self.assertEqual(rv.LEVEL.d_blocks, (ec.LATENT,) * 4 + (2 * ec.LATENT,))
        self.assertEqual((rv.LEVEL.k, rv.FOLD_SEED_OFFSET, rv.IMAGE_STREAM_OFFSET), (2, 7, rs.IMAGE_STREAM_OFFSET))
        mine = {t["seed"] for stage in ("dev", "confirm") for t in rv.tasks_for(stage)} | set(rv.PRECHECK_SEEDS)
        earlier = {t["seed"] for stage in ("dev", "confirm")
                   for t in rf.tasks_for(stage) + rs.tasks_for(stage) + rse.tasks_for(stage)}
        earlier |= {g.TEST_SEED + off for off in range(0, 13_000, 1_000)}      # every test seed used so far
        self.assertEqual(len(mine), 26)
        self.assertFalse(mine & earlier)

    def test_a_precheck_opens_the_development_seeds_only_if_it_passed_on_this_code_and_this_encoder(self):
        saved = rv.current_digests
        try:
            rv.current_digests = lambda: {ec.WEIGHTS.name: "w"}
            good = {"pass": True, "source_sha256": {f: rf.sha256(rv.HERE / f) for f in rv.SOURCES},
                    "encoder_digests": {"weights_sha256": "w"}, "prd_sha256": rf.sha256(rv.PRD)}
            self.assertEqual(rv.precheck_is_stale(good), [])
            self.assertTrue(rv.precheck_is_stale(dict(good, **{"pass": False})))
            self.assertTrue(rv.precheck_is_stale(dict(good, encoder_digests={"weights_sha256": "other"})))
            self.assertTrue(rv.precheck_is_stale(dict(good, prd_sha256="other")))
            edited = dict(good, source_sha256=dict(good["source_sha256"], **{"scm4e_frontend.py": "edited since"}))
            self.assertTrue(rv.precheck_is_stale(edited))
            fewer = dict(good, source_sha256={k: v for k, v in good["source_sha256"].items() if k != "run_scm4e_v2.py"})
            self.assertTrue(rv.precheck_is_stale(fewer))
        finally:
            rv.current_digests = saved

    def test_sources_exist_and_include_every_new_file(self):
        for name in rv.SOURCES:
            self.assertTrue((rv.HERE / name).exists(), name)
        for name in ("scm4e_encoder.py", "scm4e_frontend.py", "scm4e_pixel_cnn.py",
                     "run_scm4e_v2.py", "summarize_scm4e_v2.py"):
            self.assertIn(name, rv.SOURCES)

    def test_the_sealed_files_are_untouched_by_this_experiment(self):
        """Their digests are recorded in the SCM-4 protocols; if this experiment edited one, those break."""
        sealed = json.loads(rs.PROTOCOLS["confirm"].read_text())["source_sha256"]
        bad = [f for f, h in sealed.items() if rf.sha256(rv.HERE / f) != h]
        self.assertEqual(bad, [])

    def test_the_same_seed_gives_the_same_six_images_as_scm4(self):
        data = g.generate(g.LEVELS["SCM-1"], 400, T0 + 40)
        groups = [np.arange(k * 397, k * 397 + 397) for k in range(im.LEVELS)]
        _, src = rv.draw_sources(data, groups, T0 + 40)
        _, src_e = rse.draw_sources(data, groups, T0 + 40)
        self.assertTrue(np.array_equal(np.column_stack([src[k] for k in src]),
                                       np.column_stack([src_e[k] for k in src_e])))


class Merge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.saved = (dict(rv.SHARDS), dict(rv.PIXEL_SHARDS))
        rv.SHARDS["dev"], rv.PIXEL_SHARDS["dev"] = root / "dev", root / "dev_pixel"
        for d in (rv.SHARDS["dev"], rv.PIXEL_SHARDS["dev"]):
            d.mkdir()
        block = {k: {"canonical_correlations": [0.7, 0.2], "noise_floor": 0.2, "threshold": 0.3,
                     "above_threshold": 1, "images": 1 if k != "W5" else 2, "kept": 1 if k != "W5" else 2,
                     "hit_the_image_cap": False, "nothing_above_threshold": False,
                     "level_r2_from_kept_directions": [0.93]} for k in KEYS}
        for seed in (1, 2):
            (rv.SHARDS["dev"] / f"SCM-4e-v2_seed{seed}.json").write_text(json.dumps(
                {"protocol_sha256": "p", "task": {"seed": seed}, "front_end": {"rotation0": block},
                 "cevae": {"ate": 0.6, "variates": {"ate": 0.6}, "full_embedding": {"ate": 0.5}}}))
        (rv.PIXEL_SHARDS["dev"] / "SCM-4e-v2_seed1.json").write_text(json.dumps(
            {"protocol_sha256": "p", "task": {"seed": 1}, "pixel_cnn": {"finite": True, "ate": 0.7}}))

    def tearDown(self):
        rv.SHARDS.update(self.saved[0]); rv.PIXEL_SHARDS.update(self.saved[1])
        self.tmp.cleanup()

    def test_a_missing_pixel_shard_is_scored_as_missing(self):
        shards = sv.merged("dev")
        self.assertEqual(shards[0]["pixel_cnn"]["ate"], 0.7)
        self.assertIsNone(shards[1]["pixel_cnn"]["ate"])

    def test_a_pixel_shard_from_another_protocol_is_refused(self):
        (rv.PIXEL_SHARDS["dev"] / "SCM-4e-v2_seed2.json").write_text(json.dumps(
            {"protocol_sha256": "other", "task": {"seed": 2}, "pixel_cnn": {"finite": True, "ate": 0.7}}))
        with self.assertRaises(SystemExit):
            sv.merged("dev")

    def test_the_front_end_and_cevae_summaries_read_every_shard(self):
        shards = sv.merged("dev")
        got = sv.front_end_summary(shards)
        self.assertEqual(got["kept_min_max"], [1, 2])
        self.assertEqual(got["kept_one_direction_per_image"], {"yes": 10, "of": 10})
        self.assertEqual(got["nothing_above_threshold"], 0)
        rows = sv.cevae_rows(shards)
        self.assertEqual(rows["variates"]["mean"], 0.6)
        self.assertEqual(rows["full_embedding"]["mean"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
