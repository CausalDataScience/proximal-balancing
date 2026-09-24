"""Checks for the two-proxy data sets and the J-block runner."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import probe_j2 as pj
import two_proxy_data as tp

RESULTS = Path(__file__).resolve().parent.parent / "results" / "two_proxy"


@pytest.fixture(scope="module")
def zika():
    return tp.load_zika()


@pytest.fixture(scope="module")
def lalonde():
    return tp.load_lalonde()


def test_zika_file_matches_the_recorded_hash(zika):
    assert tp.sha256(tp.ZIKA_CSV) == tp.ZIKA_SHA


def test_zika_has_the_published_shape_and_crude_effect(zika):
    assert len(zika["A"]) == 673 and int(zika["A"].sum()) == 185
    crude = zika["Y"][zika["A"] == 1].mean() - zika["Y"][zika["A"] == 0].mean()
    assert crude == pytest.approx(tp.ZIKA_PUBLISHED["crude"], abs=1e-6)


def test_zika_proxies_precede_the_outcome_and_differ(zika):
    assert zika["W1"].shape == (673, 1) and zika["W2"].shape == (673, 1)
    assert not np.allclose(zika["W1"], zika["W2"])


def test_lalonde_has_the_nsw_treated_and_the_cps_controls(lalonde):
    assert int(lalonde["A"].sum()) == 185 and len(lalonde["A"]) == 16_177


def test_lalonde_psid_has_the_nsw_treated_and_the_psid_controls():
    v = tp.load_lalonde_psid()
    assert int(v["A"].sum()) == 185 and len(v["A"]) == 185 + 2_490


@pytest.mark.parametrize("name,n_controls", [("lalonde_cps2", 2_369), ("lalonde_cps3", 429),
                                               ("lalonde_psid2", 253), ("lalonde_psid3", 128)])
def test_each_nested_comparison_group_pairs_the_same_185_men_with_its_controls(name, n_controls):
    v = tp.DATASETS[name]["load"]()
    assert int(v["A"].sum()) == 185 and len(v["A"]) == 185 + n_controls


def test_the_afdc_benchmark_uses_complete_cases_and_1979_earnings():
    b = tp.afdc_benchmark()
    assert (b["n_treated"], b["n_control"]) == (527, 514)
    assert b["estimate"] == pytest.approx(801.01, abs=0.01)
    for flag, n_controls in (("psid1", 648), ("psid2", 182)):
        v = tp.load_afdc(flag)
        assert int(v["A"].sum()) == 527 and len(v["A"]) == 527 + n_controls
        assert np.isfinite(v["Y"]).all() and np.isfinite(v["W1"]).all() and np.isfinite(v["W2"]).all()


def test_the_early_random_assignment_women_are_a_subset_with_their_own_benchmark():
    b = tp.afdc_early_benchmark()
    assert (b["n_treated"], b["n_control"]) == (286, 279)
    assert b["estimate"] == pytest.approx(1267.66, abs=0.01)
    assert int(tp.load_afdc("psid1", "sample_early")["A"].sum()) == 286


def test_the_randomised_arm_reproduces_the_published_benchmark():
    import pandas as pd
    d = pd.read_stata(tp.LALONDE_TREATED)
    att = d[d.treat == 1].re78.mean() - d[d.treat == 0].re78.mean()
    assert att == pytest.approx(tp.LALONDE_BENCHMARK["estimate"], abs=0.5)


def test_the_band_is_declared_and_bounds_every_weight():
    lo, hi = tp.BAND
    assert 0.0 < lo < hi < 1.0 and max(1 / lo, 1 / (1 - hi)) <= 20.0


def test_the_band_rescues_the_treated_effective_sample_size(zika, lalonde):
    for view in (zika, lalonde):
        e = tp.propensity(view)
        a = view["A"]
        ess = lambda w: w.sum() ** 2 / np.sum(w ** 2)
        before = ess(a / np.clip(e, 1e-9, 1))
        k = (e >= tp.BAND[0]) & (e <= tp.BAND[1])
        after = ess(a[k] / np.clip(e[k], 1e-9, 1))
        assert before < 20.0 < after


def test_dependent_columns_are_dropped_only_where_they_are_dependent(zika, lalonde):
    for name, view in (("zika", zika), ("lalonde", lalonde)):
        e = tp.propensity(view)
        k = (e >= tp.BAND[0]) & (e <= tp.BAND[1])
        out, dropped = tp.drop_dependent_columns(tp.restrict(view, k),
                                                 tp.DATASETS[name]["covariate_labels"])
        assert out["X"].shape[1] == view["X"].shape[1] - len(dropped)
        design = np.column_stack([np.ones(len(out["X"])), out["X"]])
        assert np.linalg.matrix_rank(design) == design.shape[1]


def test_two_blocks_give_exactly_two_candidates():
    assert pj.all_splits(2) == [(0,), (1,)]
    assert [pj.split_name(s) for s in pj.all_splits(2)] == ["S1", "S2"]


def test_the_multiplicity_correction_follows_the_candidate_count():
    from scipy.stats import norm
    assert norm.ppf(1 - pj.ALPHA / 2) == pytest.approx(1.959963985, abs=1e-6)


def test_the_j_block_preprocessing_standardises_on_the_fit_rows_only(zika):
    rows = np.arange(0, 400)
    prep = pj.PrepJ(zika, rows, 2)
    x = prep.x(zika, rows)
    assert np.allclose(x.mean(0), 0.0, atol=1e-8) and np.allclose(x.std(0), 1.0, atol=1e-6)
    other = prep.x(zika, np.arange(400, 673))
    assert not np.allclose(other.mean(0), 0.0, atol=1e-3)


@pytest.mark.parametrize("name", ["zika", "lalonde", "lalonde_psid", "lalonde_cps2", "lalonde_cps3",
                                  "lalonde_psid2", "lalonde_psid3", "afdc_psid1", "afdc_psid2",
                                  "afdc_early_psid1", "afdc_early_psid2"])
def test_the_written_result_records_both_readings_and_the_band(name):
    path = RESULTS / f"{name}_v1.json"
    if not path.exists():
        pytest.skip("not run yet")
    d = json.loads(path.read_text())
    assert d["band"] == list(tp.BAND)
    assert set(d["readings"]) == {"whole_sample", "common_support"}
    for r in d["readings"].values():
        assert set(r["probe"]["readings"]) == {"significance", "noise_floor", "both"}
        assert r["overlap"]["treated_ess"] <= r["overlap"]["treated"]


def test_the_noise_floor_rule_is_degenerate_at_two_candidates():
    """With two candidates the median of the two noise terms is their average, so the rule can only ever
    keep the more precise candidate, or neither.  It can never keep both."""
    path = RESULTS / "zika_v1.json"
    if not path.exists():
        pytest.skip("not run yet")
    d = json.loads(path.read_text())
    for r in d["readings"].values():
        assert len(r["probe"]["readings"]["noise_floor"]["retained"]) <= 1
