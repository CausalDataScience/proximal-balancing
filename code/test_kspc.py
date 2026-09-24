"""Checks for the KSPC wrapper and the KSPC benchmark design."""
from __future__ import annotations

import numpy as np
import pytest

import kspc_design as kd
import kspc_wrapper as kw


def test_the_design_has_exact_effect_one_and_bounded_overlap():
    d = kd.generate(1)
    assert d["tau"] == 1.0
    assert 0.15 < d["propensity"].min() and d["propensity"].max() < 0.85
    y0 = np.sin(np.pi * d["U_eval_only"] / 2) - 0.3
    assert np.allclose(d["Y"], y0 + d["A"])                      # deterministic, effect exactly 1


def test_the_image_block_is_the_digit_of_u():
    d = kd.generate(2)
    assert d["W2"].shape == (kd.N, 784) and d["digits"].min() >= 0 and d["digits"].max() <= 9
    assert np.array_equal(d["digits"], np.minimum(np.floor(5 * d["U_eval_only"] + 5).astype(int), 9))


def test_the_bandwidth_rule_only_acts_when_the_median_is_zero():
    k = kw._kf.GaussianKernel()
    k.fit(np.array([[0.0], [0.0], [0.0], [1.0]]))
    assert k.sigma == pytest.approx(1.0)
    k.fit(np.array([[0.0], [1.0], [2.0], [3.0]]))
    assert k.sigma == pytest.approx(np.median([0, 1, 4, 9, 1, 0, 1, 4, 4, 1, 0, 1, 9, 4, 1, 0]))


def test_kspc_returns_a_finite_effect_on_a_binary_treatment():
    d = kd.generate(3)
    sub = slice(0, 400)
    out = kw.kspc_effect(d["A"][sub], d["W1"][sub], d["Y"][sub], method="skpv", seed=0)
    assert np.isfinite(out["ate"])


def test_the_covariate_version_reduces_to_the_authors_estimator():
    """PRD v2 gate 1 as a regression test: with X constant the kernel on X is 1."""
    d = kd.generate(1)
    a, w, y = d["A"][:300], d["W1"][:300], d["Y"][:300]
    for method in ("skpv", "spmmr"):
        ref = kw.kspc_effect(a, w, y, method=method, seed=3)
        cov = kw.kspc_effect(a, w, y, method=method, seed=3, x=np.zeros((300, 1)))
        assert abs(ref["ate"] - cov["ate"]) < 1e-8
