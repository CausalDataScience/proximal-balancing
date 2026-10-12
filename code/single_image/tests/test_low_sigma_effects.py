import copy
from types import SimpleNamespace

import numpy as np
import pytest

from single_image import phantom as ph, pipeline_v2 as pv, run_low_sigma_effects as runner


def audit_fixture():
    return {"Z_sha256": "z", "q0_prediction_sha256": "q0", "q1_prediction_sha256": "q1",
            "patient_rows_sha256": "s", "gap": .002, "se": .001, "B2": .003,
            "q0": {"brier": .12}, "q1": {"brier": .118}}


@pytest.mark.parametrize("change", ("Z_sha256", "gap"))
def test_reference_mismatch_fails_closed(change):
    expected = audit_fixture()
    assert runner.assert_reproduction(copy.deepcopy(expected), expected)["matched"]
    observed = copy.deepcopy(expected)
    observed[change] = "different" if change.endswith("sha256") else .004
    with pytest.raises(RuntimeError, match="reproduction failed"):
        runner.assert_reproduction(observed, expected)


def test_low_sigma_rejected_gate_still_calculates_effect_candidates(monkeypatch):
    args = runner.parser().parse_args([])
    cfg = ph.PhantomConfig(n=args.n, sigma=.3, seed=args.seed, image_only=True)
    reference = {"config": {"seed": 20261009}, "low_sigma_control": {
        "config": ph.config_dict(cfg), "audit_S1": audit_fixture()},
        "provenance": {"arguments": vars(args)}}
    data = SimpleNamespace(images=np.zeros((args.n, 2, 2)), X=np.empty((args.n, 0)), A=np.zeros(args.n))
    mode = SimpleNamespace(frame=None, report={})
    encoder = SimpleNamespace(report={})
    bundle = object()
    monkeypatch.setattr(runner.ph, "generate", lambda config: data)
    monkeypatch.setattr(pv, "prepare_mode", lambda *a, **k: mode)
    monkeypatch.setattr(pv, "LinearImageEncoder", lambda *a, **k: encoder)
    monkeypatch.setattr(pv, "fit_for_rows", lambda *a, **k: bundle)
    monkeypatch.setattr(pv, "evaluate_rows", lambda *a, **k: audit_fixture())
    def reject(models, *unused):
        models[0][-1]["qualification_failures"] = {"finite_gate": True, "empirical_tail": True}
        models[0][-1]["returned_causal_estimate"] = None
    monkeypatch.setattr(pv, "add_fresh_bounds", reject)
    called = []
    def effects(mode_arg, data_arg, roles, baseline, critic, trained, trained_critic, *unused):
        assert mode_arg is mode and baseline is encoder and critic is bundle
        assert trained is None and trained_critic is None
        assert len(roles["P"]) == len(roles["E"]) == 900
        assert not np.intersect1d(roles["P"], roles["E"]).size
        called.append(True)
        return {"baseline_S1": {"candidate_estimate": 1.2, "returned_causal_estimate": None}}
    monkeypatch.setattr(pv, "effects", effects)
    result = runner.run_experiment(args, reference, lambda message: None)
    assert called and result["same_E_effect_diagnostics"]["baseline_S1"]["candidate_estimate"] == 1.2
    assert "trained_S1" not in result["same_E_effect_diagnostics"]
    assert result["returned_causal_estimate"] is None
    assert not result["protocol"]["low_sigma_neural_training_run"]


def test_effect_helper_omits_absent_neural_pair_and_preserves_main_order(monkeypatch):
    n = 40
    u = np.linspace(-.5, .5, n)
    data = SimpleNamespace(A=np.arange(n) % 2, Y=np.arange(n) % 2 + u, U=u,
                          L=np.repeat(u[:, None], 5, axis=1), tau=1.,
                          config=ph.PhantomConfig(n=n, sigma=.3, image_only=True))
    frame = SimpleNamespace(reference=np.zeros((n, 1)), psi=[np.zeros((n, 1)) for _ in range(5)])
    mode = SimpleNamespace(frame=frame, crop=lambda frame, rows, split: np.zeros((len(rows), 2, 8, 8)))
    encoder = SimpleNamespace(encode=lambda frame, rows: frame.reference[rows])
    class Predictor:
        def __init__(self, *args):
            pass
        def predict(self, z, *args):
            return np.full(len(z), .5), np.full(len(z), .5), np.full(len(z), .5)
    monkeypatch.setattr(pv, "CriticBundle", Predictor)
    roles = {"P": np.arange(20), "E": np.arange(20, 40)}
    low = pv.effects(mode, data, roles, encoder, Predictor(), None, None, 1, 1, lambda text: None)
    assert list(low)[:2] == ["baseline_S1", "full_image_reader_adjustment"]
    assert "trained_S1" not in low
    assert low["oracle_U_evaluation_control"]["P_nuisance_fit"] is False
    main = pv.effects(mode, data, roles, encoder, Predictor(), encoder, Predictor(), 1, 1, lambda text: None)
    assert list(main)[:3] == ["baseline_S1", "trained_S1", "full_image_reader_adjustment"]
    for key, value in low.items():
        assert main[key] == value
    with pytest.raises(ValueError, match="both be present"):
        pv.effects(mode, data, roles, encoder, Predictor(), encoder, None, 1, 1, lambda text: None)


def test_existing_output_rejected_before_learning(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("original")
    def forbidden(*args, **kwargs):
        raise AssertionError("learning must not start")
    monkeypatch.setattr(runner, "run_experiment", forbidden)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        runner.main(["--out-json", str(output), "--log", str(tmp_path / "new.log")])
    assert output.read_text() == "original"
