"""One recipe that turns a benchmark with observed covariates and known potential outcomes into a proxy
problem, shared by Twins, IHDP and ACIC so that the three are built by the same code.

Every one of these benchmarks has all its covariates observed, so none of them is a proxy problem as
distributed.  The recipe makes one: it hides a real covariate that drives the outcome surface, generates
noisy measurements of it by the sealed SCM-1 mechanism, and generates the treatment from the hidden
covariate.  The outcome is the benchmark's own, so the effect stays exactly known.

    U            one real covariate, standardised, removed from X
    M_j          U + 0.85 eps_j for j = 0..3,  M_4 = U - 3.0 I + 0.85 eps_4
    W_1..W_4     M_1..M_4          W_5 = Q_5 [M_0, I]        (SCM-1's coefficients and rotation)
    A            Bernoulli(0.1 + 0.8 Phi(1.5 U + 2.0 I + w' Xtilde)),  w ~ N(0, 0.1^2) drawn once
    Y            A Y(1) + (1 - A) Y(0), the benchmark's own potential outcomes
    tau          mean(Y(1) - Y(0)), exact

What is real: the covariates, the hidden confounder, and both potential outcomes.
What is generated: the treatment-only cause I, the treatment A, and the five proxy blocks W.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

import family_v2_dgp as g

X_COEF_SD = 0.1                 # CEVAE's w_o ~ N(0, 0.1^2) on the standardised covariates
LINK_LO, LINK_HI = 0.1, 0.8     # the sealed family's link


def x_coefficients(d_x: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, X_COEF_SD, size=d_x)


def build(x: np.ndarray, u_raw: np.ndarray, y0: np.ndarray, y1: np.ndarray,
          seed: int, x_coef_seed: int) -> dict:
    """One data set from real (x, u_raw, y0, y1) and one treatment draw."""
    n = len(u_raw)
    u = (u_raw - u_raw.mean()) / (u_raw.std() + 1e-12)
    xs = (x - x.mean(0)) / (x.std(0) + 1e-12)
    rng = np.random.default_rng(seed)
    i_ = rng.standard_normal(n)
    eps = rng.standard_normal((n, 5))
    m = np.column_stack([u + g.PROXY_SD * eps[:, j] for j in range(4)]
                        + [u + g.BAD_I * i_ + g.PROXY_SD * eps[:, 4]])
    index = g.TREAT["U"] * u + g.TREAT["I"] * i_ + xs @ x_coefficients(x.shape[1], x_coef_seed)
    p = LINK_LO + LINK_HI * norm.cdf(index)
    a = (rng.random(n) < p).astype(float)
    y = a * y1 + (1.0 - a) * y0
    q5 = g.rotations(g.LEVELS["SCM-1"])["W5"]
    w5 = np.column_stack([m[:, 0], i_]) @ q5.T
    view = {"X": x, "W1": m[:, [1]], "W2": m[:, [2]], "W3": m[:, [3]], "W4": m[:, [4]], "W5": w5,
            "A": a, "Y": y}
    return {**view, "U_eval_only": u, "I_eval_only": i_, "propensity": p,
            "tau": float((y1 - y0).mean()), "naive": float(y[a == 1].mean() - y[a == 0].mean())}


def level(d_x: int, name: str):
    return g.Level(name=name, index=0, d_x=d_x, d_blocks=(1, 1, 1, 1, 2), k=2, variant="real")


def block_information(d: dict) -> dict:
    """R^2 of each block for the hidden confounder, the diagnostic that explained the RHC decline."""
    from sklearn.linear_model import LinearRegression
    u = d["U_eval_only"]
    return {k: float(LinearRegression().fit(d[k], u).score(d[k], u)) for k in ("W1", "W2", "W3", "W4", "W5")}
