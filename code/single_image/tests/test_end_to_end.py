import json
import math

from single_image import level_a, oracle, phantom as ph, run_phase1


def test_tiny_end_to_end(tmp_path):
    res = run_phase1.run(run_phase1.parse(["--n", "2000", "--quick", "--out-dir", str(tmp_path), "--seed", "0"]))
    files = list(tmp_path.glob("phase1_gaussian_n2000_sigma0.85_seed0.json"))
    assert len(files) == 1
    d = json.loads(files[0].read_text())
    assert d["folds"] == {**d["folds"], "R": 500, "rest": 1500}
    for k in ("naive", "aipw_X", "aipw_X_U", "aipw_X_readers", "aipw_X_true_L"):
        assert math.isfinite(d["estimates"][k])
    assert abs(d["estimates"]["aipw_X_U"] - 1.0) < 0.3
    assert d["estimates"]["naive"] > 1.5                          # confounded upward
    for nm in ("level_A_readers", "oracle_reader_L"):
        assert len(d["level_A"][nm]["per_split"]) == 30
    assert (tmp_path / "weights").is_dir()
    assert res["timing_s"]["total"] > 0


def test_all_rotation_screen_is_not_pooled_screen():
    rec = {"per_split": {"S5": {"pooled_upper": .01872855,
            "screen": [{"upper": upper, "overlap": True}
                       for upper in (.01939, .02114, .01973, .02460)]}}}
    assert level_a.rethreshold(rec, .02, "pooled")["S5_retained"]
    assert not level_a.rethreshold(rec, .02, "all_rotations")["S5_retained"]
    assert "no causal/overlap certification" in level_a.rethreshold(rec, .02)["scope"]


def test_reference_and_prediction_ceiling_are_not_effect_certification():
    cfg = ph.PhantomConfig(n=100, image_only=True)
    res = oracle.population(cfg, n_mc=500, seed=31)
    assert "not learned B" in res["representation_scope"]
    assert "not a bound on all ATE" in res["ceiling_scope"]
    assert "adjustment_allL" in res["targets"] and "aipw_X_allL" not in res["targets"]
    assert res["closed_form_section_3_6"]["S1"]["D2_posterior_reference_approx"] >= 0.


def test_control_labels_and_saved_reader_S5_rethreshold():
    old = json.loads((run_phase1.DEFAULT_OUT / "phase1_gaussian_n24000_sigma0.85_seed0.json").read_text())
    reader = old["level_A"]["level_A_readers"]
    oracle_i = old["level_A"]["oracle_reader_L_device_in_W5"]
    assert level_a.rethreshold(reader, .01)["n_retained"] == 5
    assert level_a.rethreshold(reader, .01)["S5_retained"]
    assert not level_a.rethreshold(oracle_i, .01)["S5_retained"]
    assert oracle_i["per_split"]["S5"]["pooled_gap"] > 3 * reader["per_split"]["S5"]["pooled_gap"]


def test_existing_result_is_rejected_before_training(monkeypatch, tmp_path):
    path = tmp_path / "phase1_gaussian_n400_sigma0.85_seed0.json"
    path.write_text('{"original": true}')
    def unexpected(*args, **kwargs):
        raise AssertionError("existing output must stop before phantom/training")
    monkeypatch.setattr(run_phase1.ph, "generate", unexpected)
    import pytest
    with pytest.raises(SystemExit, match="before training"):
        run_phase1.run(run_phase1.parse(["--n", "400", "--out-dir", str(tmp_path)]))
    assert path.read_text() == '{"original": true}'
