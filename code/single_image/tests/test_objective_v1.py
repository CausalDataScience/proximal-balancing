import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from single_image import level_a, nets, objective_v1 as ob, pipeline_v2 as pv, reader as rd, regions as rg
from single_image import run_objective_v1 as cli


def _fake_mode(n=100, seed=7, width=8):
    rng = np.random.default_rng(seed)
    image = rng.normal(size=(n, 64, 64)).astype(np.float32)
    frame = pv.ImageFrame(image, image, rng.normal(size=(n, 1)).astype(np.float32),
                          [rng.normal(size=(n, 8)) for _ in range(5)], rng.normal(size=(n, 5, 16)).astype(np.float32))
    bank = rd.ReaderBank([rd.RegionReaderNet(8) for _ in range(5)], 8, (0., 1.))
    fit = {"mae": {"model": SimpleNamespace(trunk=nets.ConvTrunk(width)), "stats": (0., 1.)}, "bank": bank}
    mode = pv.Mode("corrected", fit, frame, (0., 1.), (0., 1.), .55, {})
    roles = {key: np.arange(k * 20, (k + 1) * 20) for k, key in enumerate(("R", "D", "C", "P", "S"))}
    data = SimpleNamespace(A=(rng.random(n) < .5).astype(float))
    return mode, roles, data


def _args(**kw):
    base = dict(lr=2e-4, mu=ob.MU, critic_epochs=1, raw_epochs=1, rounds=2, encoder_steps=2, o3_steps=40,
                kci_batch=512, kci_eps=1e-2)
    base.update(kw)
    return SimpleNamespace(**base)


def test_generalised_fit_reproduces_level_a_brier_fit_exactly():
    rng = np.random.default_rng(1)
    n = 300
    x = torch.as_tensor(rng.normal(size=(n, 4)), dtype=torch.float32)
    a = (rng.random(n) < .4).astype(float)
    base = torch.zeros(n)
    train, val = np.arange(60, n), np.arange(60)
    nets.seed_all(3); m1 = level_a._ProbabilityResidual(4)
    nets.seed_all(3); m2 = level_a._ProbabilityResidual(4)
    r1 = level_a._fit_probability(m1, x, base, a, train, val, None, 5, 11)
    r2 = ob.fit_probability(m2, x, base, a, train, val, None, 5, 11, loss="brier", batch=128)
    assert pv.tensor_state_sha(m1) == pv.tensor_state_sha(m2)
    assert r1["selected_epoch"] == r2["selected_epoch"] and r1["selected_val_brier"] == r2["selected_val_risk"]


def test_ladder_bundle_with_brier_matches_decisive_critic_bundle():
    rng = np.random.default_rng(2)
    n = 120
    crop = rg.region_crops(rng.normal(size=(n, 64, 64)).astype(np.float32), 2)
    z = rng.normal(size=(n, 3)).astype(np.float32)
    t = rng.normal(size=(n, 8))
    a = (rng.random(n) < .5).astype(float)
    old = pv.CriticBundle(z, t, crop, a, 5, 2, 2)
    new = ob.LadderBundle(z, t, crop, a, 5, 2, 2, loss="brier", batch=128)
    p_old, p_new = old.predict(z, t, crop), new.predict(z, t, crop)
    for x, y in zip(p_old, p_new):
        assert np.array_equal(x, y)
    p_tex = new.predict_with_texture(z, t, rd.raw_texture_features(crop), crop, chunk=50)
    for x, y in zip(p_new, p_tex):
        assert np.allclose(x, y, atol=1e-7)
    logloss = ob.LadderBundle(z, t, crop, a, 5, 2, 2, loss="logloss")
    assert logloss.report["q0"]["loss"] == "logloss"


def _explicit_kci(z, a, kt, eps):
    def gram(x):
        d2 = ((x[:, None] - x[None]) ** 2).sum(-1)
        iu = np.triu_indices(len(x), 1)
        values = np.sort(d2[iu])
        return np.exp(-d2 / (2 * values[(len(values) - 1) // 2]))   # torch.median: lower median
    m = len(a)
    h = np.eye(m) - 1. / m
    kz = h @ gram(z) @ h
    r = eps * np.linalg.inv(kz + eps * np.eye(m))
    ka = np.outer(h @ a, h @ a)
    return np.trace(r @ kt @ r @ r @ ka @ r) / m ** 2


def test_kci_loss_equals_explicit_trace_and_is_differentiable_in_z():
    rng = np.random.default_rng(3)
    m = 41
    z = rng.normal(size=(m, 3))
    a = (rng.random(m) < .5).astype(float)
    f = torch.as_tensor(rng.normal(size=(m, 10)))
    kt = ob.center(ob.gaussian_kernel(f, ob.median_sq_distance(f)))
    zt = torch.as_tensor(z).requires_grad_(True)
    value, h2 = ob.kci_loss(zt, a, kt, 1e-2)
    assert float(value.detach()) == pytest.approx(_explicit_kci(z, a, kt.numpy(), 1e-2), rel=1e-8)
    value.backward()
    assert torch.isfinite(zt.grad).all() and zt.grad.abs().sum() > 0
    assert h2 > 0


def test_kci_is_small_when_z_explains_a_and_large_when_t_carries_the_signal():
    rng = np.random.default_rng(4)
    m = 300
    u = rng.normal(size=m)
    t = u + .1 * rng.normal(size=m)
    a = (u + .3 * rng.normal(size=m) > 0).astype(float)
    feats = torch.as_tensor(np.column_stack([t, rng.normal(size=m)]))
    kt = ob.center(ob.gaussian_kernel(feats, ob.median_sq_distance(feats)))
    good = torch.as_tensor(np.column_stack([u, rng.normal(size=(m, 2)) * .01]))
    bad = torch.as_tensor(rng.normal(size=(m, 3)))
    assert float(ob.kci_loss(good, a, kt)[0]) < .2 * float(ob.kci_loss(bad, a, kt)[0])


def test_overlap_penalty_is_a_smooth_hinge_on_the_overlap_interval():
    q = torch.tensor([.02, .05, .3, .5, .7, .95, .98], requires_grad=True)
    pen = ob.overlap_softplus(q)
    vals = pen.detach()
    assert float(vals[2:5].max()) < 1e-10
    assert float(vals[0]) == pytest.approx(.03, abs=2e-3) and float(vals[-1]) == pytest.approx(.03, abs=2e-3)
    pen.sum().backward()
    assert q.grad[0] < -.9 and q.grad[-1] > .9 and abs(float(q.grad[3])) < 1e-10


def test_logistic_q0_torch_matches_numpy_and_reports_tail():
    rng = np.random.default_rng(5)
    z = rng.normal(size=(500, 3))
    a = (z[:, 0] + rng.normal(size=500) > 0).astype(float)
    fit = ob.LogisticQ0(z, a)
    assert np.allclose(fit.torch_prob(torch.as_tensor(z, dtype=torch.float32)).numpy(), fit.predict(z), atol=1e-5)
    assert 0 <= fit.report["fraction_outside_05_95"] <= 1


def test_stop_rule_counts_non_improvements_and_selects_lowest():
    rule = ob.StopRule(patience=3, relative=.01)
    values = [1., .9, .895, .893, .892]
    stops = [rule.update(v, i, f"state{i}") for i, v in enumerate(values)]
    assert stops == [False, False, False, False, True]
    assert rule.best_index == 4 and rule.best_state == "state4"
    rule = ob.StopRule()
    assert not rule.update(-.001, 0, None) and not rule.update(-.01, 1, None)


def test_brier_and_logloss_terms():
    q0, q1, a = np.array([.3, .6]), np.array([.2, .9]), np.array([0., 1.])
    assert np.allclose(ob.brier_terms(q0, q1, a), [.09 - .04, .16 - .01])
    expected = [-math.log(.7) + math.log(.8), -math.log(.6) + math.log(.9)]
    assert np.allclose(ob.ce_np(q0, a) - ob.ce_np(q1, a), expected)
    t = [torch.as_tensor(x) for x in (q0, q1, a)]
    assert np.allclose((ob.ce_torch(t[0], t[2]) - ob.ce_torch(t[1], t[2])).numpy(), expected)


def test_t_features_use_held_out_pixels_and_are_standardised_on_R():
    mode, roles, _ = _fake_mode()
    tf = ob.TFeatures(mode, mode.frame, roles["R"], 1, seed=3)
    f_r = tf.transform(mode, mode.frame, roles["R"]).numpy()
    assert f_r.shape[1] == tf.report["dimension"] <= 256 + 6
    assert np.allclose(f_r.mean(0), 0, atol=1e-8) and np.allclose(f_r.std(0), 1, atol=1e-6)
    changed = copy.deepcopy(mode.frame)
    changed.raw[:, rg.masks()[0] > 0] += 5.   # region 1 is not the held-out region 2
    assert np.allclose(tf.transform(mode, changed, roles["D"]).numpy(), tf.transform(mode, mode.frame, roles["D"]).numpy())
    assert tf.kernel(torch.as_tensor(f_r[:7])).shape == (7, 7)


def test_gap_and_kci_arms_refit_select_on_C_and_are_deterministic():
    mode, roles, data = _fake_mode()
    cfg = SimpleNamespace(seed=11)
    shas = []
    for objective in ("brier", "logloss"):
        enc = ob.base_encoder(mode, 1, cfg)
        rep = ob.train_gap_arm(mode, data, roles, enc, 1, objective, 21, _args(), lambda _: None, "t")
        assert rep["rounds_completed"] == 2 and len(rep["history"]) == 3
        assert rep["selected_round"] in (0, 1, 2) and not rep["L_keep"]
        assert rep["selected_encoder_sha256"] == ob.state_sha(enc.model.state_dict())
        shas.append(rep["selected_encoder_sha256"])
    tf = ob.TFeatures(mode, mode.frame, roles["R"], 1, seed=3)
    runs = []
    for _ in range(2):
        enc = ob.base_encoder(mode, 1, cfg)
        rep = ob.train_kci_arm(mode, data, roles, enc, 1, tf, 31, _args(), lambda _: None, "k")
        runs.append(rep)
    assert runs[0]["selected_encoder_sha256"] == runs[1]["selected_encoder_sha256"]
    assert [h["step"] for h in runs[0]["history"]] == [0, 20, 40] and runs[0]["minibatch"] == 20
    assert not runs[0]["inner_predictor_refit_every_chunk"]


def test_warm_start_regresses_on_R_only_and_reports_fit():
    mode, roles, _ = _fake_mode()
    seen = []

    class Target:
        report = {"B_sha256": "x" * 64}

        def encode(self, frame, rows):
            seen.append(np.asarray(rows).copy())
            img = frame.processed[rows]
            return np.column_stack([frame.reference[rows], img[:, :32].mean((1, 2)), img[:, 32:].mean((1, 2))])

    enc, rep = ob.warm_start(mode, SimpleNamespace(seed=11), 1, Target(), roles["R"], 5, 2, 1e-3, 3, lambda _: None)
    assert len(seen) == 1 and np.array_equal(seen[0], roles["R"])
    assert rep["rows"] == "R images only" and len(rep["R_val_R2_per_coordinate"]) == 2
    assert rep["selected_R_val_mse"] <= rep["initial_R_val_mse"]


def test_fresh_d2_is_cross_fitted_out_of_fold(monkeypatch):
    n = 10
    fits = []

    class Stub:
        def __init__(self, z, t, crop, a, seed, *args, **kw):
            self.r = len(fits); fits.append(z[:, 0].copy())
            self.report = {k: {"selected_val_risk": .2, "selected_epoch": 1, "epochs_run": 2}
                           for k in ("q0", "reader", "raw")}

        def predict_with_texture(self, z, t, texture, crop):
            q0 = np.full(len(z), .5)
            return q0, q0 + .1 * (self.r + 1), q0

    monkeypatch.setattr(ob, "LadderBundle", Stub)
    fresh = SimpleNamespace(n=n, A=np.r_[np.zeros(5), np.ones(5)], halves=(np.arange(5), np.arange(5, n)),
                            p_LI=np.full(n, .5),
                            encode=lambda encoder: np.column_stack([np.arange(n)] * 3).astype(float),
                            inputs=lambda mode, split: (np.zeros((n, 2, 4, 4)), np.zeros((n, 6)), np.zeros((n, 8))))
    enc = SimpleNamespace()
    args = SimpleNamespace(fresh_fit_seed=1, fresh_epochs=1, fresh_raw_epochs=1, fresh_batch=8, fresh_patience=1,
                           fresh_raw_patience=1)
    out = ob.fresh_d2(None, enc, 0, fresh, args, lambda _: None, "m")
    assert np.array_equal(fits[0], np.arange(5)) and np.array_equal(fits[1], np.arange(5, 10))
    assert out["replicate_D2_out_of_fold"] == pytest.approx([.01, .04])
    assert out["D2_fresh"] == pytest.approx(.025) and out["second_replicate_difference"] == pytest.approx(-.03)
    assert out["replicate_disagreement_mean_sq_delta"] == pytest.approx(.01)


def _merged(seed, theta_shift, d2_arm, la_d2=.002, s5_gap=.01, init_sets=None):
    rng = np.random.default_rng(seed)
    level, models = {}, {}
    base_scores = rng.normal(0., 1., 400)
    base_scores = base_scores - base_scores.mean() + 1.08 + .001 * seed
    for s in range(1, 6):
        level[f"S{s}"] = {"D2_fresh": {"D2_fresh": la_d2}, "effect_E": {"candidate_estimate": float(base_scores.mean())},
                          "audit_S": {"gap": .009 if s == 5 else .001}, "E_scores": base_scores.tolist()}
        for arm, inits in (init_sets or ob.ARMS).items():
            for init in inits:
                scores = base_scores + theta_shift[arm] + rng.normal(0, .01, 400)
                models[f"{arm}_{init}_S{s}"] = {
                    "arm": arm, "init": init, "split": s, "D2_fresh": {"D2_fresh": d2_arm[arm]},
                    "effect_E": {"candidate_estimate": float(scores.mean())}, "E_scores": scores.tolist(),
                    "audit_S": {"gap": s5_gap if s == 5 else .002}}
    return {"level_A": level, "models": models}


def test_decision_table_rules_and_branches():
    shift = {"O1": .2, "O2": .2, "O3": -.02}
    d2 = {"O1": .01, "O2": .01, "O3": .0025}
    one = ob.decision_table({1: _merged(1, shift, d2)})
    rows = one["rows"]
    assert rows["O3"]["R1_balance_pass"] and not rows["O1"]["R1_balance_pass"]
    assert rows["O3"]["R2_accuracy_pass"] and not rows["O1"]["R2_accuracy_pass"]
    assert rows["O3"]["R3_stability_pass"] is None and "pending" in rows["O3"]["R3_status"]
    assert all(rows[k]["R4_device_pass"] for k in rows if "/" in k)
    assert any("provisional" in note for note in one["branches"]["notes"])
    two = ob.decision_table({1: _merged(1, shift, d2), 2: _merged(2, shift, d2)})
    assert two["level_A"]["theta_range_over_seeds_mean_S1_S4"] is not None
    bad = ob.decision_table({1: _merged(1, {"O1": .3, "O2": .3, "O3": .3}, d2, s5_gap=.0)})
    fired = " ".join(bad["branches"]["fired"])
    assert "KEEP_READER" in fired and "R4_FAIL" in fired


def test_cli_refuses_existing_part_before_learning(tmp_path, monkeypatch):
    stem = cli.stem(.3, 20261009)
    occupied = tmp_path / f"{stem}.part-X.json"
    occupied.write_text("unchanged")
    monkeypatch.setattr(ob, "run_part", lambda *a, **k: pytest.fail("must refuse before learning"))
    with pytest.raises(SystemExit):
        cli.main(["--tag", "X", "--out-dir", str(tmp_path), "--arms", "O3"])
    assert occupied.read_text() == "unchanged" and not (tmp_path / f"{stem}.part-X.log").exists()
    args = cli.parser().parse_args([])
    assert (args.n, args.sigma, args.seed, args.rounds, args.encoder_steps, args.o3_steps, args.kci_batch,
            args.kci_eps, args.mc) == (48000, .3, 20261009, 30, 20, 600, 512, 1e-2, 100000)
    assert cli.OUTPUT.name == "objective_v1" and cli.OUTPUT.parent.name == "single_image"


def test_chunked_fresh_sample_regenerates_identical_images_and_encodings():
    from single_image import phantom as ph, decisive_v2 as dv
    mode, _, _ = _fake_mode()
    cfg = ph.PhantomConfig(n=10, sigma=.3, seed=5, image_only=True)
    fresh = ob.FreshSample(mode, cfg, 300, 777, lambda _: None, chunk=100)
    assert fresh.A.shape == (300,) and [len(h) for h in fresh.halves] == [150, 150]
    second = dv.bounded_generate(ph.with_changes(cfg, n=100, seed=777 + 200))
    assert np.array_equal(fresh.A[200:], second.A)
    enc = ob.base_encoder(mode, 1, SimpleNamespace(seed=3))
    z = fresh.encode(enc)
    full = pv.ImageFrame(second.images, pv.corrected_images(second.images, fresh.offset[200:]),
                         fresh.reference[200:], [p[200:] for p in fresh.psi], None)
    assert np.array_equal(z[200:], enc.encode(full, np.arange(100)))
    crops, texture, psi = fresh.inputs(mode, 1)
    assert crops.shape[0] == texture.shape[0] == psi.shape[0] == 300
    assert np.array_equal(crops[200:], mode.crop(full, np.arange(100), 1))
