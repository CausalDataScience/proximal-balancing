"""RHC with a planted effect: real covariates, a real hidden confounder, real proxy measurements, real
outcome variation, and an effect that is known exactly
(pre-registration memo/2026-09-22-rhc-planted-prereg-v1.md).

The ten blood-test measures of the first 24 hours are genuine, error-prone readings of the patient's
physiological state.  This module hides one real summary of that state, the APACHE III acute physiology
score, generates the treatment from it, and adds a known constant to the treated outcome.  Everything the
learner sees except the treatment column is a real measurement.
"""
from __future__ import annotations

import csv

import numpy as np
from scipy.stats import norm

import rhc_data as rd

HIDDEN = "aps1"                 # APACHE III acute physiology score, a real severity summary
DELTA = -1.0                    # the planted effect, so tau = DELTA exactly for every unit
U_COEF = 1.5                    # SCM-1's coefficient on U in the treatment index
X_COEF_SD = 0.1                 # CEVAE's w_o ~ N(0, 0.1^2) on the standardised covariates
X_COEF_SEED = 98_300_000        # one draw of w, fixed for every replicate
REPLICATE_SEED = 98_300_000     # replicate r uses REPLICATE_SEED + 1000 r
LINK_LO, LINK_HI = 0.1, 0.8     # P(A=1) = LINK_LO + LINK_HI * Phi(index), the sealed family's link


def load_real() -> dict:
    """The real part: X without the hidden score, the hidden score, the five real proxy blocks, real Y."""
    base = rd.load()
    with rd.CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    j = base["x_names"].index(HIDDEN)
    x = np.delete(base["view"]["X"], j, axis=1)
    names = [c for c in base["x_names"] if c != HIDDEN]
    raw = np.array([float(r[HIDDEN]) for r in rows])
    u = (raw - raw.mean()) / raw.std()
    blocks = {f"W{k + 1}": np.column_stack([np.array([float(r[c]) for r in rows]) for c in block])
              for k, block in enumerate(rd.BLOCKS)}
    return {"X": x, "x_names": names, "U": u, "U_raw": raw, "blocks": blocks,
            "Y_real": np.array([float(r[rd.OUTCOME]) for r in rows]), "n": len(rows), "tau": DELTA}


def x_coefficients(d_x: int) -> np.ndarray:
    return np.random.default_rng(X_COEF_SEED).normal(0.0, X_COEF_SD, size=d_x)


def generate(real: dict, replicate: int) -> dict:
    """One replicate: a fresh treatment draw on the fixed real covariates, proxies and outcome."""
    x = real["X"]
    xs = (x - x.mean(0)) / (x.std(0) + 1e-12)
    index = U_COEF * real["U"] + xs @ x_coefficients(x.shape[1])
    p = LINK_LO + LINK_HI * norm.cdf(index)
    rng = np.random.default_rng(REPLICATE_SEED + 1000 * replicate)
    a = (rng.random(real["n"]) < p).astype(float)
    y = real["Y_real"] + DELTA * a
    view = {"X": x, **real["blocks"], "A": a, "Y": y}
    return {**view, "U_eval_only": real["U"], "propensity": p, "tau": DELTA,
            "naive": float(y[a == 1].mean() - y[a == 0].mean()), "replicate": replicate}


def level(d_x: int):
    import family_v2_dgp as g
    return g.Level(name="RHC-planted", index=0, d_x=d_x, d_blocks=tuple(len(b) for b in rd.BLOCKS),
                   k=2, variant="real")
