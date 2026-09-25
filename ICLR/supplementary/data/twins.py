"""The Twins benchmark as a PROBE data set: real covariates, a real hidden confounder, real outcomes with
both potential outcomes observed, and SCM-1's treatment and proxy mechanism laid over them.

The benchmark is the one Louizos et al. (2017, CEVAE) built from the US twin births of 1989-1991: same-sex
twin pairs with both birth weights under 2,000 g, treatment "being the heavier twin", outcome first-year
mortality.  Because both twins' mortality is recorded, both potential outcomes of every pair are observed
and the average treatment effect is an exact sample quantity, not an assumption.  CEVAE hides the
gestational age (gestat10) and lets it drive a simulated treatment assignment; we do the same.

What is real:      the covariates X, the hidden confounder U = gestat10, and Y(0), Y(1), hence tau.
What is generated: the treatment-only cause I, the treatment A given (U, I, X), and the proxy blocks W,
                   by the SCM-1 mechanism exactly (scm_family), so the five blocks carry the same roles as
                   in the simulations: W1-W3 clean, W4 contaminated by I, W5 = (M0, I).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import scm_family as g

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "raw" / "twins"                  # default place of the three CSV files
FILES = {"X": "twin_pairs_X_3years_samesex.csv", "T": "twin_pairs_T_3years_samesex.csv",
         "Y": "twin_pairs_Y_3years_samesex.csv"}
WEIGHT_CAP = 2000.0                       # CEVAE's subset: both twins under 2,000 g
HIDDEN = "gestat10"                       # the real confounder, withheld from every learner
DROP = ("Unnamed: 0.1", "Unnamed: 0", "infant_id_0", "infant_id_1", "bord_0", "bord_1",
        "brstate", "stoccfipb", "mplbir", "birmon", "data_year", HIDDEN)
X_COEF_SD = 0.1                           # CEVAE's w_o ~ N(0, 0.1^2) on the standardised covariates
X_COEF_SEED = 98_200_000                  # one draw of w_o, fixed for every replicate
REPLICATE_SEED = 98_210_000               # replicate r uses REPLICATE_SEED + 1000 r


def load_real() -> dict:
    """The real part, loaded once: X, the hidden U, and the two potential outcomes."""
    x = pd.read_csv(ROOT / FILES["X"])
    t = pd.read_csv(ROOT / FILES["T"])
    y = pd.read_csv(ROOT / FILES["Y"])
    keep = (t["dbirwt_0"] < WEIGHT_CAP) & (t["dbirwt_1"] < WEIGHT_CAP)
    x, y = x.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
    u_raw = x[HIDDEN].astype(float).to_numpy()
    cols = [c for c in x.columns if c not in DROP]
    xm = x[cols].astype(float).to_numpy()
    if np.isnan(xm).any() or np.isnan(u_raw).any():
        # the public file carries a handful of blanks; fill each column with its median, and say so
        med = np.nanmedian(xm, axis=0)
        filled = int(np.isnan(xm).sum())
        xm = np.where(np.isnan(xm), med[None, :], xm)
        u_raw = np.where(np.isnan(u_raw), np.nanmedian(u_raw), u_raw)
    else:
        filled = 0
    xs = (xm - xm.mean(0)) / (xm.std(0) + 1e-12)
    u = (u_raw - u_raw.mean()) / u_raw.std()
    return {"X": xs, "x_names": cols, "U": u, "U_raw": u_raw,
            "Y0": y["mort_0"].astype(float).to_numpy(), "Y1": y["mort_1"].astype(float).to_numpy(),
            "n": int(keep.sum()), "filled_blanks": filled,
            "tau": float((y["mort_1"] - y["mort_0"]).mean())}


def x_coefficients(d_x: int) -> np.ndarray:
    return np.random.default_rng(X_COEF_SEED).normal(0.0, X_COEF_SD, size=d_x)


def generate(real: dict, replicate: int) -> dict:
    """One replicate: fresh I, noises and A on the fixed real (X, U, Y0, Y1)."""
    n, u = real["n"], real["U"]
    rng = np.random.default_rng(REPLICATE_SEED + 1000 * replicate)
    i_ = rng.standard_normal(n)
    eps = rng.standard_normal((n, 5))
    m = np.column_stack([u + g.PROXY_SD * eps[:, j] for j in range(4)] + [u + g.BAD_I * i_ + g.PROXY_SD * eps[:, 4]])
    index = g.TREAT["U"] * u + g.TREAT["I"] * i_ + real["X"] @ x_coefficients(real["X"].shape[1])
    p = 0.1 + 0.8 * norm.cdf(index)
    a = (rng.random(n) < p).astype(float)
    y = a * real["Y1"] + (1.0 - a) * real["Y0"]
    level = g.LEVELS["SCM-1"]
    q5 = g.rotations(level)["W5"]
    w5 = np.column_stack([m[:, 0], i_]) @ q5.T if q5.shape == (2, 2) else np.column_stack([m[:, 0], i_])
    view = {"X": real["X"], "W1": m[:, [1]], "W2": m[:, [2]], "W3": m[:, [3]], "W4": m[:, [4]], "W5": w5,
            "A": a, "Y": y}
    return {**view, "U_eval_only": u, "I_eval_only": i_, "propensity": p,
            "naive": float(y[a == 1].mean() - y[a == 0].mean()), "tau": real["tau"], "replicate": replicate}


def level(d_x: int):
    return g.Level(name="TWINS", index=0, d_x=d_x, d_blocks=(1, 1, 1, 1, 2), k=2, variant="real")
