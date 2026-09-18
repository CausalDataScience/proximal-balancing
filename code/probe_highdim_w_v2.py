"""Two replacements for the SCM-6 selection and screening steps, kept separate from the v1 estimators.

The v1 objective is the nested Brier risk difference evaluated on the rows that fitted both critics.  Its
bias is the augmented critic's excess error, which on 1,500 rows against a 257-column target is larger
than the whole range of the population discrepancy, so the minimiser moves to the edge of the grid.  Two
distinct repairs are needed because selection and screening want different things from the estimate.

Selection needs the argmin to sit where the population objective sits.  Only the shape matters, so a
cross-fitted risk difference is enough: fitting on one part and scoring on another removes the leading
same-sample term from both critics.

Screening needs a quantity that cannot pass a candidate merely because a critic generalised badly.  The
cross-fitted risk difference is negative almost everywhere here, so under the clipped rule
max(gap, 0) <= 2 se it passes automatically and the screen becomes fail-open.  The squared difference of
the two fitted propensities is non-negative by construction.  It is not unbiased and it is not a
conservative bound: writing q and e for the two population propensities and delta_q, delta_e for the
fitted errors,

    E[(q_hat - e_hat)^2] = D^2 + E[(delta_q - delta_e)^2] + 2 E[(q - e)(delta_q - delta_e)],

so the second term inflates it and the third has no fixed sign.  Whether it is usable is an empirical
question about this design, which is why the development harness measures it against the exactly known
population curve rather than assuming anything.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

import minimal_suite_common as c
import probe_highdim_w as hd


def fit_target_projection(rows: dict, split: tuple[int, ...], components: int,
                         seed: int) -> dict:
    """Leading principal directions of each held-out block, fitted on D_phi rows only.

    The augmented critic's width is what drives the excess-error term.  Each block is one informative
    coordinate plus independent noise, so its leading principal directions carry the information while
    cutting the column count from 257 to a handful.  The directions are estimated from the observed block
    alone and never look at the generating rotation, so this is a choice a practitioner can make.
    """
    directions = {}
    for j in split:
        block = rows["H"][j]
        centred = block - block.mean(0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        directions[j] = vt[:components].T
    return {"directions": directions, "components": components, "split": split,
            "digest": c.array_sha256(np.concatenate([directions[j].ravel() for j in split]))}


def projected_heldout(rows: dict, split: tuple[int, ...], projection: dict | None) -> np.ndarray:
    """T_S, either in full or compressed onto the frozen principal directions."""
    if projection is None:
        return hd.heldout(rows, split)
    cols = [rows["X"][:, None]]
    for j in split:
        if j == 0:
            cols.extend([rows["I"][:, None], rows["C"][:, None]])
        cols.append(rows["H"][j] @ projection["directions"][j])
    return np.column_stack(cols)


def _fit_pair(z_tr, t_tr, a_tr, penalty):
    """Base and augmented critics on the same rows under one shared penalty, base lifted as the start."""
    b0 = hd.brier_ridge(z_tr, a_tr, penalty)
    lifted = np.concatenate([b0, np.zeros(t_tr.shape[1])])
    b1 = hd.brier_ridge(np.column_stack([z_tr, t_tr]), a_tr, penalty,
                        starts=[lifted, np.zeros(len(lifted))])
    return b0, b1


def _propensity(beta, design):
    return 0.1 + 0.8 * norm.cdf(np.column_stack([np.ones(len(design)), design]) @ beta)


def crossfit_folds(n: int, folds: int, seed: int) -> list[np.ndarray]:
    return np.array_split(np.random.default_rng(seed).permutation(n), folds)


def crossfit_gap(rows: dict, k_hat: np.ndarray, split: tuple[int, ...], r: float | None,
                 penalty: float, folds: int, seed: int, projection: dict | None = None) -> float:
    """Risk difference with every row scored by critics that did not see it."""
    z_raw = hd.representation(k_hat, rows, split, r)
    t_raw = projected_heldout(rows, split, projection)
    a = rows["A"]
    parts = crossfit_folds(len(a), folds, seed)
    total = []
    for f in range(folds):
        te = parts[f]
        tr = np.concatenate([parts[g] for g in range(folds) if g != f])
        zs, ts = hd.Scaler(z_raw[tr]), hd.Scaler(t_raw[tr])
        z_tr, z_te, t_tr, t_te = zs(z_raw[tr]), zs(z_raw[te]), ts(t_raw[tr]), ts(t_raw[te])
        b0, b1 = _fit_pair(z_tr, t_tr, a[tr], penalty)
        total.append(np.mean(hd.brier_rows(b0, z_te, a[te])
                             - hd.brier_rows(b1, np.column_stack([z_te, t_te]), a[te])))
    return float(np.mean(total))


def select_r_crossfit(rows: dict, k_hat: np.ndarray, split: tuple[int, ...], penalty: float,
                      seed: int, grid: tuple = c.GRID, folds: int = 5,
                      projection: dict | None = None) -> dict:
    """Line 2 of Algorithm 1 with the objective cross-fitted inside D_phi."""
    candidates = [None] if 0 in split else list(grid)
    values = [crossfit_gap(rows, k_hat, split, r, penalty, folds, seed, projection)
              for r in candidates]
    best = int(np.argmin(values))
    ties = [i for i, v in enumerate(values) if v <= values[best] + c.TIE_TOL]
    if len(ties) > 1:
        best = min(ties, key=lambda i: (abs(candidates[i] or 0.0), candidates[i] or 0.0))
    return {"r": candidates[best], "objective": values[best], "objective_values": values,
            "objective_vector_sha256": c.array_sha256(np.asarray(values)),
            "objective_min_count": len(ties), "folds": folds, "penalty": penalty,
            "target_components": None if projection is None else projection["components"]}


def squared_discrepancy(fit_rows: dict, k_fit: np.ndarray, screen_rows: dict, k_screen: np.ndarray,
                        split: tuple[int, ...], r: float | None, penalty: float,
                        projection: dict | None = None) -> dict:
    """Mean squared difference of the two fitted propensities, on rows neither critic was fitted on.

    Non-negative by construction, so a badly generalising critic can no longer buy a pass.  The standard
    error is the usual one for a mean over the screening rows.
    """
    z_fit = hd.representation(k_fit, fit_rows, split, r)
    t_fit = projected_heldout(fit_rows, split, projection)
    zs, ts = hd.Scaler(z_fit), hd.Scaler(t_fit)
    b0, b1 = _fit_pair(zs(z_fit), ts(t_fit), fit_rows["A"], penalty)
    z_s = zs(hd.representation(k_screen, screen_rows, split, r))
    t_s = ts(projected_heldout(screen_rows, split, projection))
    q = _propensity(b1, np.column_stack([z_s, t_s]))
    e = _propensity(b0, z_s)
    d = (q - e) ** 2
    return {"d2": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "p_min": float(min(q.min(), e.min())), "p_max": float(max(q.max(), e.max())),
            "overlap_ok": bool(e.min() >= c.OVERLAP_RANGE[0] and e.max() <= c.OVERLAP_RANGE[1])}


def screen_v2(fit_rows: dict, k_fit: np.ndarray, screen_rows: dict, k_screen: np.ndarray,
              split: tuple[int, ...], r: float | None, penalty: float, threshold: float,
              projection: dict | None = None) -> dict:
    """Retain the split when the squared discrepancy is below the threshold and overlap holds.

    The threshold is an absolute one on a non-negative statistic, so there is no clipping and no way for a
    large negative number to be read as agreement.
    """
    stat = squared_discrepancy(fit_rows, k_fit, screen_rows, k_screen, split, r, penalty, projection)
    return stat | {"threshold": threshold,
                   "pass": bool(stat["d2"] <= threshold and stat["overlap_ok"])}
