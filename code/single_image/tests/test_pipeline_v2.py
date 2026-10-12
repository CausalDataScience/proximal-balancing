import math

import numpy as np
import pytest
import torch
from scipy.stats import norm

from single_image import level_a, phantom as ph, pipeline_v2 as pv, regions as rg, run_pipeline_v2


def test_fixed_body_reference_cannot_copy_held_region_into_retained_view():
    data = ph.generate(ph.PhantomConfig(n=400, seed=50, image_only=True))
    ref_mask = pv.body_reference_mask()
    assert not np.any(ref_mask & rg.masks().any(0))
    for split in range(5):
        changed = data.images[:2].copy()
        changed[:, rg.masks()[split].astype(bool)] += 20.
        off0, off1 = pv.body_offset(data.images[:2]), pv.body_offset(changed)
        assert np.array_equal(off0, off1)
        retained0 = rg.holdout_view(pv.corrected_images(data.images[:2], off0), (split + 1,))
        retained1 = rg.holdout_view(pv.corrected_images(changed, off1), (split + 1,))
        assert np.array_equal(retained0, retained1)
    assert data.X.shape == (400, 0) and data.config.treat_x == data.config.out_x == 0.


def test_reference_corrects_additive_brightness_but_keeps_reference_coordinate():
    rng = np.random.default_rng(60)
    images = rng.normal(size=(4, 64, 64)).astype(np.float32)
    shifted = images + np.array([.1, -.2, .3, .4], dtype=np.float32)[:, None, None]
    off, shifted_off = pv.body_offset(images), pv.body_offset(shifted)
    assert not np.array_equal(off, shifted_off)
    assert np.allclose(pv.corrected_images(images, off), pv.corrected_images(shifted, shifted_off), atol=1e-6)


def test_overlap_penalty_is_zero_inside_and_does_not_bound_probability_head():
    q = torch.tensor([.01, .05, .5, .95, .99], requires_grad=True)
    penalty = pv.overlap_hinge(q)
    assert torch.equal(penalty[1:4], torch.zeros(3))
    assert torch.allclose(penalty[[0, 4]], torch.full((2,), .04**2))
    penalty.sum().backward()
    assert q.grad[0] < 0 and q.grad[-1] > 0


def test_finite_screen_does_not_certify_small_empirical_gap_from_reference_epsilon():
    rec = pv.finite_screen(.002, .001, 3600, .04, 5000)
    assert rec["normal_empirical_upper"] == pytest.approx(.002 + norm.ppf(1 - .05 / 12) * .001)
    assert rec["finite_risk_radius"] == pytest.approx(math.sqrt(2 * math.log(480) / 3600))
    assert rec["b1_conservative_upper"] > .04
    assert rec["cutoffs"]["0.01"]["empirical_gap_pass"]
    assert not any(x["finite_discrepancy_pass"] for x in rec["cutoffs"].values())
    assert not rec["uniform_overlap_certified"] and not rec["causal_identification_verified"]


def test_nested_probability_inference_matches_fit_clamp_and_scoring_cannot_refit():
    rng = np.random.default_rng(70)
    z = rng.normal(size=(100, 3)).astype(np.float32)
    t = rng.normal(size=(100, 8)).astype(np.float32)
    crop = np.zeros((100, 2, *rg.crop_size()), dtype=np.float32)
    crop[:, 1, 3:-3, 3:-3] = 1.
    crop[:, 0] = rng.normal(size=crop[:, 0].shape) * crop[:, 1]
    a = (z[:, 0] + rng.normal(size=100) > 0).astype(float)
    bundle = pv.CriticBundle(z, t, crop, a, seed=70, epochs=2, raw_epochs=2)
    before = [pv.tensor_state_sha(model) for model in (bundle.q0, bundle.reader, bundle.raw)]
    q0, q1, qr = bundle.predict(z, t, crop)
    # Exact fitting-time chain, including the extreme-probability numerical clamp.
    arrays = (z, np.column_stack([z, t]), np.column_stack([z, t, pv.rd.raw_texture_features(crop)]))
    features = [torch.as_tensor((arr - mu) / sd, dtype=torch.float32)
                for arr, (mu, sd) in zip(arrays, bundle.scales)]
    fit_q0 = level_a._predict_probability(bundle.q0, features[0], torch.zeros(100))
    fit_qr = level_a._predict_probability(bundle.reader, features[1], level_a._logit(fit_q0))
    fit_q1 = level_a._predict_probability(bundle.raw, features[2], level_a._logit(fit_qr), torch.as_tensor(crop))
    assert np.allclose(q0, fit_q0) and np.allclose(qr, fit_qr) and np.allclose(q1, fit_q1)
    bundle.predict(z * 20, t, crop)
    assert before == [pv.tensor_state_sha(model) for model in (bundle.q0, bundle.reader, bundle.raw)]
    assert not bundle.report["scoring_rows_used"]
    assert set(bundle.train).isdisjoint(bundle.val)
    assert all(p.grad is None for model in (bundle.q0, bundle.reader, bundle.raw) for p in model.parameters())


def test_new_cli_is_image_only_and_refuses_existing_output_before_learning(tmp_path, monkeypatch):
    args = run_pipeline_v2.parser().parse_args([])
    assert args.n == 24000 and args.encoder_steps == 20 and args.rounds == 12
    occupied = tmp_path / "original.json"
    occupied.write_text("preserve")
    monkeypatch.setattr(pv, "run_local", lambda *a: pytest.fail("must refuse before training"))
    with pytest.raises(SystemExit):
        run_pipeline_v2.main(["--out-json", str(occupied), "--log", str(tmp_path / "new.log")])
    assert occupied.read_text() == "preserve" and not (tmp_path / "new.log").exists()


def test_new_encoder_must_be_refitted_before_C_and_final_S(monkeypatch):
    from types import SimpleNamespace
    from single_image import nets
    nets.setup_torch(2)
    rng = np.random.default_rng(80)
    n = 60
    image = rng.normal(size=(n, 64, 64)).astype(np.float32)
    frame = pv.ImageFrame(image, image, rng.normal(size=(n, 1)).astype(np.float32),
                          [rng.normal(size=(n, 8)).astype(np.float32) for _ in range(5)],
                          rng.normal(size=(n, 5, 16)).astype(np.float32))
    mode = pv.Mode("corrected", {"mae": {"model": SimpleNamespace(trunk=nets.ConvTrunk(width=8)),
                                        "stats": (0., 1.)}}, frame, (0., 1.), (0., 1.), .55, {})
    roles = {name: np.arange(i * 12, (i + 1) * 12) for i, name in enumerate(("D", "C", "P", "S", "E"))}
    data = SimpleNamespace(A=(rng.random(n) < .5).astype(float))
    fits, evaluations = [], []

    class FrozenPredictors:
        def __init__(self, encoder):
            self.fitted_encoder_hash = pv.tensor_state_sha(encoder.model)

        def tensor_predict(self, z, t, texture, crop):
            q0 = torch.sigmoid(z[:, 1])
            q1 = torch.sigmoid(z[:, 1] + .1 * t[:, 0])
            return q0, q1, q1

    def fit(mode, encoder, frame, rows, a, *args):
        bundle = FrozenPredictors(encoder)
        fits.append((bundle.fitted_encoder_hash, rows.copy()))
        return bundle

    def score(mode, encoder, bundle, frame, rows, a, independent=True):
        assert bundle.fitted_encoder_hash == pv.tensor_state_sha(encoder.model), "stale critic at checkpoint"
        evaluations.append((rows.copy(), independent))
        return {"gap": .02, "q0": {"fraction_outside_05_95": .02}}

    monkeypatch.setattr(pv, "fit_for_rows", fit)
    monkeypatch.setattr(pv, "evaluate_rows", score)
    _, _, report, _ = pv.train_alternating(mode, data, roles, seed=80, rounds=1, encoder_steps=20,
                                           epochs=2, raw_epochs=2, log=lambda _: None)
    assert len(fits) == 3 and len(evaluations) == 4
    assert fits[0][0] != fits[1][0] and fits[1][0] == fits[2][0]
    for (rows, independent), role, expected_independent in zip(evaluations, ("D", "C", "P", "S"),
                                                               (False, True, False, True)):
        assert np.array_equal(rows, roles[role]) and independent == expected_independent
    assert np.array_equal(fits[2][1], roles["P"])
    assert report["encoder_steps_total"] == 20 and report["capped"]
    assert not report["suitable_C_checkpoint_exists"]


def test_known_oracle_uses_E_without_claiming_P_nuisance_fitting():
    scores = np.array([.8, 1., 1.2])
    prop = np.array([.01, .5, .99])
    fitted = pv._effect_summary(scores, 1., "P fitted diagnostic", prop)
    oracle = pv._effect_summary(scores, 1., "known oracle diagnostic", prop, p_fitted=False)
    assert fitted["P_nuisance_fit"] and fitted["E_score"]
    assert not oracle["P_nuisance_fit"] and not oracle["P_fit_E_score"] and oracle["E_score"]
    for key in ("candidate_estimate", "sampling_SE", "normal_sampling_interval", "propensity_tail_fraction"):
        assert oracle[key] == fitted[key]
    assert oracle["causal_confidence_interval"] is None and oracle["returned_causal_estimate"] is None
