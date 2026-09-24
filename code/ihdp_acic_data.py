"""IHDP and ACIC 2016 as proxy problems, through the shared recipe in hidden_proxy_recipe.

IHDP   Hill (2011) over the covariates of the Infant Health and Development Program, in the npz pair
       distributed by Johansson et al. (2016).  n = 747 per replication (672 train + 75 test, pooled),
       25 covariates, 100 replications.  yf and ycf give both potential outcomes.
ACIC   the 2016 data challenge over the Collaborative Perinatal Project covariates.  n = 4,802,
       58 covariates, 10 realizations.  y0 and y1 are given directly.

Hidden confounder: the covariate with the strongest correlation to the control surface mu0 in the first
replication, chosen once and fixed.  IHDP: column 5 (|corr| 0.858).  ACIC: x_44 (|corr| 0.689).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import hidden_proxy_recipe as hp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "materials" / "real_world_data"
IHDP_DIR, ACIC_DIR = ROOT / "ihdp", ROOT / "acic2016"

IHDP_HIDDEN = 5                 # the column index, fixed before any replication was scored
ACIC_HIDDEN = "x_44"
IHDP_SEED, ACIC_SEED = 98_400_000, 98_500_000      # replicate r uses SEED + 1000 r
IHDP_X_COEF_SEED, ACIC_X_COEF_SEED = 98_400_777, 98_500_777
IHDP_REPS, ACIC_REPS = range(2, 12), range(2, 11)  # replication/realization 1 was the development check


def _ihdp_arrays() -> dict:
    tr = np.load(IHDP_DIR / "ihdp_npci_1-100.train.npz")
    te = np.load(IHDP_DIR / "ihdp_npci_1-100.test.npz")
    out = {}
    for key in ("x", "t", "yf", "ycf", "mu0", "mu1"):
        out[key] = np.concatenate([tr[key], te[key]], axis=0)
    return out


def ihdp(replication: int) -> dict:
    """One IHDP replication as a proxy problem.  Y(1) and Y(0) come from yf and ycf under the
    benchmark's own treatment, so both are the benchmark's realised potential outcomes."""
    a = _ihdp_arrays()
    k = replication
    x, t = a["x"][:, :, k], a["t"][:, k]
    yf, ycf = a["yf"][:, k], a["ycf"][:, k]
    y1 = np.where(t == 1, yf, ycf)
    y0 = np.where(t == 1, ycf, yf)
    u_raw = x[:, IHDP_HIDDEN]
    x_rest = np.delete(x, IHDP_HIDDEN, axis=1)
    d = hp.build(x_rest, u_raw, y0, y1, IHDP_SEED + 1000 * replication, IHDP_X_COEF_SEED)
    return {**d, "replication": replication, "n": len(t), "d_x": x_rest.shape[1],
            "tau_mu": float((a["mu1"][:, k] - a["mu0"][:, k]).mean())}


def _acic_covariates() -> tuple[np.ndarray, list[str]]:
    x = pd.read_csv(ACIC_DIR / "x.csv")
    dummies = pd.get_dummies(x, drop_first=True).astype(float)
    return dummies.values, list(dummies.columns)


def acic(realization: int) -> dict:
    """One ACIC 2016 realization as a proxy problem."""
    xm, names = _acic_covariates()
    z = pd.read_csv(ACIC_DIR / f"zymu_{realization}.csv")
    j = names.index(ACIC_HIDDEN)
    u_raw, x_rest = xm[:, j], np.delete(xm, j, axis=1)
    d = hp.build(x_rest, u_raw, z["y0"].to_numpy(), z["y1"].to_numpy(),
                 ACIC_SEED + 1000 * realization, ACIC_X_COEF_SEED)
    return {**d, "replication": realization, "n": len(z), "d_x": x_rest.shape[1],
            "tau_mu": float((z["mu1"] - z["mu0"]).mean())}


DATASETS = {"ihdp": {"load": ihdp, "reps": IHDP_REPS, "seed": IHDP_SEED, "hidden": f"column {IHDP_HIDDEN}"},
            "acic": {"load": acic, "reps": ACIC_REPS, "seed": ACIC_SEED, "hidden": ACIC_HIDDEN}}
DEVELOPMENT = 1   # opened for the design check; excluded from every confirmation
