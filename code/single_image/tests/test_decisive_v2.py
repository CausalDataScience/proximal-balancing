import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.special import ndtr

from single_image import decisive_v2 as dv, folds, nets, oracle, phantom as ph, pipeline_v2 as pv, run_decisive_v2 as cli


def test_bounded_generator_preserves_image_latents_and_potential_outcomes():
    cfg = ph.PhantomConfig(n=1200, sigma=.3, seed=91, image_only=True)
    old = ph.generate(cfg)
    new = dv.bounded_generate(cfg)
    for key in ("images", "U", "L", "G", "I", "Y0", "Y1"):
        assert np.array_equal(getattr(old, key), getattr(new, key))
    expected = .1 + .8 * ndtr(1.5 * new.U + new.I)
    draw = np.random.default_rng([cfg.seed, 47]).random(cfg.n)
    assert np.array_equal(new.A, (draw < expected).astype(float))
    assert np.array_equal(new.Y, new.Y0 + new.A)
    assert np.array_equal(new.propensity, expected)
    assert new.X.shape == (cfg.n, 0) and cfg.treat_x == cfg.out_x == 0
    assert np.all((new.propensity >= .1) & (new.propensity <= .9))
    assert "link" not in ph.config_dict(cfg)  # Keep the legacy replay guard unchanged.
    with pytest.raises(ValueError):
        dv.bounded_generate(ph.PhantomConfig(n=10, image_only=False))


def test_bounded_gaussian_posterior_uses_new_link_for_truth_controls():
    cfg = ph.PhantomConfig(n=4, sigma=.3, image_only=True)
    m = np.array([-10., -.3, .7, 10.])
    v = .09 / 4.09
    expected = .1 + .8 * ndtr((1.5 * m + 1.) / math.sqrt(1 + 2.25 * v))
    assert np.allclose(dv.bounded_posterior(m, v, np.ones(4), cfg), expected)
    averaged = .1 + .8 * (.7 * ndtr(1.5 * m / math.sqrt(1 + 2.25 * v)) +
                          .3 * ndtr((1.5 * m + 1.) / math.sqrt(1 + 2.25 * v)))
    assert np.allclose(dv.bounded_posterior(m, v, None, cfg), averaged)


def test_empirical_bernstein_rescales_range_two_and_counts_primary_family():
    term = math.log(4 * 45 / .05)
    assert dv.eb_radius(.01, 7200, 2.) == pytest.approx(
        math.sqrt(2 * .01 * term / 7200) + 14 * term / (3 * 7199))
    assert dv.eb_radius(0., 100000, 1.) == pytest.approx(7 * term / (3 * 99999))
    assert dv.eb_radius(.01, 7200, 2., 135) > dv.eb_radius(.01, 7200, 2., 45)
    with pytest.raises(ValueError):
        dv.eb_radius(.01, 1, 2.)


def test_finite_gate_contains_actual_reference_error_and_never_forces_causal_return():
    audit = {"gap": -.001, "se": .0005}
    error = {"n": 100000, "mean": .012, "variance_ddof1": .0001}
    rec = dv.finite_bound(audit, error, 7200)
    assert rec["b1_conservative_upper"] > .012
    assert not rec["cutoffs"]["0.01"]["finite_discrepancy_pass"]
    assert rec["cutoffs"]["0.01"]["normal_empirical_gap_pass"]
    assert not rec["near_Bayes_accuracy_certified"] and rec["causal_confidence_interval"] is None


def test_streaming_moments_and_paired_sampling_keep_patient_covariance():
    values = np.array([.1, .2, .4, .8])
    moments = dv.Moments()
    moments.add(values[:2]); moments.add(values[2:])
    assert moments.mean == pytest.approx(values.mean())
    assert moments.variance == pytest.approx(values.var(ddof=1))
    first = np.array([1., 2., 3., 4.])
    second = first + .2
    pair = dv.paired_sampling(first, second)
    assert pair["candidate_difference"] == pytest.approx(-.2)
    assert pair["sampling_SE"] < 1e-15
    assert pair["n_E"] == 4


def test_pilot_roles_exact_and_random_branch_reads_R_mae_only():
    groups = folds.six_groups(48000, 20261009)
    roles = {"R": groups["R"], **folds.roles(groups["groups"], 0)}
    assert {key: len(v) for key, v in roles.items()} == {"R": 12000, "D": 7200, "C": 7200,
                                                        "P": 7200, "S": 7200, "E": 7200}
    assert len(np.unique(np.concatenate(list(roles.values())))) == 48000
    mode = SimpleNamespace(fit={"mae": {"val_masked_mse": .2, "reference_mean_image_mse": .18}})
    assert dv.random_branch_needed(mode)
    mode.fit["mae"]["val_masked_mse"] = .17
    assert not dv.random_branch_needed(mode)


def test_matching_bounded_reference_does_not_demand_device_rejection_at_primary():
    cfg = ph.PhantomConfig(n=10, sigma=.3, image_only=True)
    result = dv.reference_sanity(cfg, n=20000, seed=93)
    clean, device = result["splits"]["S1"], result["splits"]["S5"]
    assert clean["D2_reference"] < .001
    assert .007 < device["D2_reference"] < .01
    assert device["reference_primary_budget_contains_D2"]
    assert all(rec["sampling_identity_sanity_pass_5SE"] for rec in result["splits"].values())
    assert clean["population_adjustment_target_MC"] < 1.05
    assert "floor" in result["scope"]


def test_split_five_random_encoder_refit_precedes_C_and_tail_is_diagnostic(monkeypatch):
    nets.setup_torch(2)
    rng = np.random.default_rng(94)
    n = 60
    image = rng.normal(size=(n, 64, 64)).astype(np.float32)
    frame = pv.ImageFrame(image, image, rng.normal(size=(n, 1)).astype(np.float32),
        [np.full((n, 8), k, dtype=np.float32) for k in range(5)],
        rng.normal(size=(n, 5, 16)).astype(np.float32))
    mode = pv.Mode("corrected", {"mae": {"model": SimpleNamespace(trunk=nets.ConvTrunk(8)),
                                       "stats": (0., 1.)}}, frame, (0., 1.), (0., 1.), .55, {})
    roles = {key: np.arange(k * 12, (k + 1) * 12) for k, key in enumerate(("D", "C", "P", "S", "E"))}
    data = SimpleNamespace(A=(rng.random(n) < .5).astype(float))
    fits = []

    class Frozen:
        def __init__(self, encoder):
            self.hash = pv.tensor_state_sha(encoder.model)

        def tensor_predict(self, z, t, texture, crop):
            assert torch.all(t == 4), "heldout predictor must receive split-five features"
            return torch.sigmoid(z[:, 1]), torch.sigmoid(z[:, 1] + .1), torch.sigmoid(z[:, 1] + .1)

    def fit(mode, encoder, frame, rows, a, *args):
        assert encoder.split == 4 and encoder.init == "random"
        fitted = Frozen(encoder); fits.append(fitted.hash)
        return fitted

    def score(mode, encoder, bundle, frame, rows, a, independent=True):
        assert bundle.hash == pv.tensor_state_sha(encoder.model), "new phi must be refitted before score"
        return {"gap": .002, "q0": {"fraction_outside_05_95": .8}}

    monkeypatch.setattr(pv, "fit_for_rows", fit)
    monkeypatch.setattr(pv, "evaluate_rows", score)
    _, _, report, _ = pv.train_alternating(mode, data, roles, 94, 1, 20, 2, 2, lambda _: None,
                                           split=4, init="random", enforce_tail_gate=False)
    assert len(fits) == 3 and fits[0] != fits[1] == fits[2]
    assert report["suitable_C_checkpoint_exists"] and not report["C_tail_gate_enforced"]
    assert report["split"] == 5 and report["capped"]


def test_true_I_control_is_unavailable_outside_held_device_region():
    assert not pv.oracle_i_positive_control(None, SimpleNamespace(split=0), None, None, None, None, 0, 1)["run"]


def test_new_cli_preserves_family_defaults_and_refuses_overwrite(tmp_path, monkeypatch):
    args = cli.parser().parse_args(["--conditional-random"])
    assert args.n == 48000 and args.family_size == 45 and args.mc == 100000
    assert args.max_seconds == 21600 and args.rss_limit_mib == 12288
    occupied = tmp_path / "original.json"; occupied.write_text("unchanged")
    monkeypatch.setattr(dv, "run_pilot", lambda *args: pytest.fail("must refuse before learning"))
    with pytest.raises(SystemExit):
        cli.main(["--out-json", str(occupied), "--log", str(tmp_path / "new.log")])
    assert occupied.read_text() == "unchanged" and not (tmp_path / "new.log").exists()


def test_external_source_change_fails_closed(monkeypatch):
    monkeypatch.setattr(cli, "source_hashes", lambda: {"source.py": "changed"})
    with pytest.raises(RuntimeError, match="source changed"):
        cli.assert_frozen({"source.py": "original"}, {}, "protocol")


def test_atomic_checkpoint_writes_only_requested_artifact(tmp_path):
    output = tmp_path / "result.json"
    cli.atomic_json(output, {"array": np.array([1, 2]), "complete": False})
    import json
    assert json.loads(output.read_text()) == {"array": [1, 2], "complete": False}

