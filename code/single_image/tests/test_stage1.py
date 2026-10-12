import inspect

import numpy as np
import pytest
import torch

from single_image import level_a, mae, phantom as ph, reader as rd, run_phase1, stage1, summaries as sm

FORBIDDEN = {"a", "y", "A", "Y", "treatment", "outcome", "view", "data", "U", "u", "latent"}


@pytest.fixture(scope="module")
def tiny():
    return ph.generate(ph.PhantomConfig(n=400, seed=11))


def test_summaries_frozen_after_fit(tiny):
    s = sm.FixedSummaries()
    with pytest.raises(RuntimeError):
        s.transform(tiny.images)
    s.fit(tiny.images[:200])
    before = (s.thresholds.copy(), s.center.copy(), s.scale.copy(), s.fingerprint())
    out = s.transform(tiny.images[200:])
    assert out.shape == (200, 5, sm.D_CHI) and np.all(np.isfinite(out))
    assert np.array_equal(before[0], s.thresholds) and np.array_equal(before[1], s.center)
    assert np.array_equal(before[2], s.scale) and before[3] == s.fingerprint()
    with pytest.raises(RuntimeError):
        s.fit(tiny.images[200:])
    with pytest.raises(ValueError):
        s.center[0, 0] = 1.0                                 # parameters are read-only arrays
    fit_rows = s.transform(tiny.images[:200])
    assert np.allclose(fit_rows.reshape(-1, sm.D_CHI).mean(0), 0.0, atol=1e-4)


def test_stage1_entry_points_take_images_only():
    for fn in (stage1.run_stage1, rd.train_readers, mae.train_mae, sm.FixedSummaries.fit):
        names = set(inspect.signature(fn).parameters)
        assert not (names & FORBIDDEN), (fn.__name__, names & FORBIDDEN)
    with pytest.raises(TypeError):
        stage1.run_stage1({"images": np.zeros((2, 64, 64)), "A": np.zeros(2)}, 0, torch.device("cpu"))
    with pytest.raises(TypeError):
        rd.train_readers({"A": 1}, sm.FixedSummaries(), 0, torch.device("cpu"))


def test_runner_passes_only_fold_r_images(monkeypatch, tmp_path):
    seen = {}

    class Stop(Exception):
        pass

    def spy(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        raise Stop

    monkeypatch.setattr(run_phase1.stage1, "run_stage1", spy)
    with pytest.raises(Stop):
        run_phase1.main(["--n", "400", "--quick", "--out-dir", str(tmp_path), "--device", "cpu"])
    data = ph.generate(ph.PhantomConfig(n=400, seed=0))
    R = run_phase1.folds.phase1_groups(400, 0)["R"]
    assert len(seen["args"]) == 1 and isinstance(seen["args"][0], np.ndarray)
    assert np.array_equal(seen["args"][0], data.images[R])
    assert set(seen["kwargs"]) <= {"seed", "device", "reader_epochs", "mae_epochs", "n_boot", "log"}


def test_stage1_runs_and_is_deterministic(tiny):
    kw = dict(seed=3, device=torch.device("cpu"), reader_epochs=2, mae_epochs=1, n_boot=20, log=lambda *a: None)
    a = stage1.run_stage1(tiny.images[:300], **kw)
    b = stage1.run_stage1(tiny.images[:300], **kw)
    pa, pb = a["bank"].transform(tiny.images[300:]), b["bank"].transform(tiny.images[300:])
    assert all(np.array_equal(x, y) for x, y in zip(pa, pb))
    assert all(x.shape == (100, a["bank"].d_psi) for x in pa)
    assert a["factor_R"]["n_tetrads"] == 15 and len(a["factor_R"]["omega_fa"]) == 5
    assert np.isfinite(a["mae"]["val_masked_mse"])


def test_new_probability_head_does_not_force_overlap():
    head = level_a._ProbabilityResidual(1)
    with torch.no_grad():
        head.head[-1].bias.fill_(5.)
        q = head(torch.zeros(20, 1), torch.zeros(20)).numpy()
    report = level_a.probability_diagnostics(q, np.arange(20) % 2)
    assert np.all((q > .95) & (q < 1.))
    assert report["fraction_outside_05_95"] == 1.
    assert not report["empirical_all_inside_05_95"]
    assert not report["population_overlap_certified"]


def test_fixed_raw_texture_is_image_only_brightness_and_outside_mask_invariant():
    import inspect
    rng = np.random.default_rng(30)
    crop = np.zeros((5, 2, 24, 24), dtype=np.float32)
    crop[:, 1, 2:22, 2:22] = 1.
    crop[:, 0] = rng.normal(size=(5, 24, 24)) * crop[:, 1]
    f = rd.raw_texture_features(crop)
    brighter = crop.copy(); brighter[:, 0] += 3. * brighter[:, 1]
    outside = crop.copy(); outside[:, 0][outside[:, 1] == 0.] = 1000.
    assert np.allclose(f, rd.raw_texture_features(brighter), atol=2e-6, rtol=2e-6)
    assert np.array_equal(f, rd.raw_texture_features(outside))
    constant = crop.copy(); constant[:, 0] = 10. * constant[:, 1]
    assert np.allclose(rd.raw_texture_features(constant), 0., atol=1e-9)
    assert f.shape == (5, len(rd.RAW_TEXTURE_FEATURES)) and np.all(np.isfinite(f))
    assert not (set(inspect.signature(rd.raw_texture_features).parameters) & FORBIDDEN)


def test_fresh_critics_share_Z_rows_q0_and_keep_scoring_independent():
    rng = np.random.default_rng(23)
    z, t = rng.normal(size=(120, 1)), rng.normal(size=(120, 1))
    i = (rng.random(120) < .3).astype(float)
    a = (z[:, 0] + 2 * i + rng.normal(size=120) > 0).astype(float)
    report = level_a.fresh_critic_contrast(z, t, a, oracle_i=i, epochs=2, raw_epochs=1, seed=23)
    assert report["common_Z_rows_q0"] and not report["scoring_used_for_checkpoint_selection"]
    assert set(report["contrasts"]) == {"reader", "oracle_I_positive_control"}
    for fold in report["folds"]:
        assert fold["n_train"] + fold["n_internal_val"] + fold["n_score"] == len(a)
        assert fold["train_rows_sha256"] != fold["scoring_rows_sha256"]
        for fit in fold["augmented_fits"].values():
            assert fit["selected_val_brier"] <= fit["initial_val_brier"]
    for rec in report["contrasts"].values():
        assert rec["certified_D2_upper"] is None and rec["epsilon1_squared"] is None
        assert not rec["causal_balance_certified"]


def test_true_L_oracle_identity_is_reference_only_and_has_sampling_residual():
    rng = np.random.default_rng(24)
    z, t = rng.normal(size=(120, 1)), rng.normal(size=(120, 1))
    # With T ~ N(0,1), E[Phi(Z+T) | Z] = Phi(Z/sqrt(2)).
    e = __import__("scipy").special.ndtr(z[:, 0] / np.sqrt(2.))
    p = __import__("scipy").special.ndtr(z[:, 0] + t[:, 0])
    a = (rng.random(120) < p).astype(float)
    report = level_a.fresh_critic_contrast(z, t, a, epochs=2, raw_epochs=1, seed=24,
                                          reference_e=e, reference_p=p)
    rec = report["contrasts"]["reader"]
    assert np.isclose(rec["reference_D2_scoring_mean"], np.mean((p - e) ** 2))
    assert rec["epsilon0_squared"] >= 0. and rec["epsilon1_squared"] >= 0.
    assert "identity_sampling_residual" in rec
    assert abs(rec["identity_sampling_residual"]) <= 6 * rec["identity_sampling_residual_se"]
    assert rec["certified_D2_upper"] is None
