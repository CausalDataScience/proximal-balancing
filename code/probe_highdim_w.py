"""SCM-6: the same causal structure with each proxy block observed as a 256-dimensional vector.

The question is whether a representation can still be learned from observed data when W is high dimensional.
Each block's scalar channel K_j is mixed into 256 coordinates by a fixed orthogonal matrix together with
independent nuisance directions:

    H_j = Q_j [ K_j ; 0.5 xi_j ] in R^256,     W_0 = (I, C, H_0),   W_j = H_j for j = 1..4,

so the raw proxy is 2 + 5 * 256 = 1282 dimensional.  Because Q_j is orthogonal and xi_j is independent of
everything causal, the population identity

    q_j = Cov(H_j, X) / ||Cov(H_j, X)||_2 = Q_j e_1,      q_j^T H_j = K_j

holds, which says the direction carrying the channel is identified by observable second moments alone.  That
identity is an evaluator check.  The learner never receives Q_j or K_j; it estimates the direction from the
embedding rows and calibrates its scale on separate rows.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.stats import norm

import minimal_suite_common as c

DIM = 256
NUISANCE_AMPLITUDE = 0.5
Q_SEED_BASE = 620260912
PROXY_SD = 0.85
BAD_I = -0.6
TREATMENT_I = 2.5
OUTCOME_U = 2.5
FAIL_CLOSED = {"direction_norm_min": 1e-8, "scale_min": 1e-3}


def orthogonal_block(j: int) -> np.ndarray:
    """Q_j from a QR decomposition with the sign of R's diagonal fixed positive, so it is reproducible."""
    rng = np.random.Generator(np.random.PCG64DXSM(Q_SEED_BASE + j))
    q, r = np.linalg.qr(rng.standard_normal((DIM, DIM)))
    return q * np.sign(np.diag(r))


def orthogonal_digest(j: int) -> str:
    q = orthogonal_block(j)
    return hashlib.sha256(np.ascontiguousarray(q, dtype="<f8").tobytes()).hexdigest()


def generate(n: int, seed: int) -> dict:
    """The shared latent SCM, rendered with 256-dimensional blocks.  Evaluator-only fields are marked."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 9))
    u, i_cause, c_sev, e_x = z[:, :4].T
    x = u + 2.0 * e_x
    k = u[:, None] + PROXY_SD * z[:, 4:9]
    k[:, 4] += BAD_I * i_cause
    index = u + TREATMENT_I * i_cause + 0.2 * x + 0.3 * c_sev
    p = 0.1 + 0.8 * norm.cdf(index)
    a = rng.binomial(1, p).astype(float)
    y = a - OUTCOME_U * u + 0.2 * x - 0.5 * c_sev + 0.3 * rng.standard_normal(n)
    blocks = []
    for j in range(5):
        xi = rng.standard_normal((n, DIM - 1))
        carrier = np.column_stack([k[:, j], NUISANCE_AMPLITUDE * xi])
        blocks.append(carrier @ orthogonal_block(j).T)
    return {"X": x, "I": i_cause, "C": c_sev, "A": a, "Y": y, "H": blocks,
            "U_eval_only": u, "K_eval_only": k, "p_eval_only": p}


def observed_view(data: dict, include_outcome: bool = False) -> dict:
    """Strip every evaluator-only field before any learner is called."""
    keys = ["X", "I", "C", "A"] + (["Y"] if include_outcome else [])
    return {k: data[k] for k in keys} | {"H": data["H"]}


def fit_embedding(fit_rows: dict, cal_rows: dict) -> dict:
    """Estimate the channel direction on the fit rows and its scale on separate calibration rows.

    The direction is the normalised empirical covariance between the block and X, which is the sample version
    of the population identity above.  The scale is then fixed on rows the direction never saw, so the two
    estimates do not share information.  Any degenerate direction or scale fails closed.
    """
    out = {"directions": [], "centres": [], "scales": [], "fail_closed": None}
    for j in range(5):
        h_fit = fit_rows["H"][j]
        cov = (h_fit - h_fit.mean(0)).T @ (fit_rows["X"] - fit_rows["X"].mean()) / len(h_fit)
        norm_ = float(np.linalg.norm(cov))
        if not np.isfinite(norm_) or norm_ <= FAIL_CLOSED["direction_norm_min"]:
            out["fail_closed"] = f"block {j}: direction norm {norm_:.3e}"
            return out
        q_hat = cov / norm_
        p_cal = cal_rows["H"][j] @ q_hat
        scale = float(np.cov(p_cal, cal_rows["X"], ddof=0)[0, 1])
        if not np.isfinite(scale) or abs(scale) <= FAIL_CLOSED["scale_min"]:
            out["fail_closed"] = f"block {j}: calibration scale {scale:.3e}"
            return out
        out["directions"].append(q_hat)
        out["centres"].append(float(p_cal.mean()))
        out["scales"].append(scale)
    return out


def apply_embedding(emb: dict, rows: dict) -> np.ndarray:
    """K-hat for every block, on any set of rows, using the frozen direction, centre and scale."""
    return np.column_stack([(rows["H"][j] @ emb["directions"][j] - emb["centres"][j]) / emb["scales"][j]
                            for j in range(5)])


def channel_correlation(emb: dict, data: dict, rows: np.ndarray | None = None) -> list[float]:
    """Evaluator-only: how well K-hat tracks the true K, per block."""
    view = {"H": [h if rows is None else h[rows] for h in data["H"]]}
    k_hat = apply_embedding(emb, view)
    k_true = data["K_eval_only"] if rows is None else data["K_eval_only"][rows]
    return [float(np.corrcoef(k_hat[:, j], k_true[:, j])[0, 1]) for j in range(5)]


def identity_check(n: int = 200000, seed: int = 424242) -> dict:
    """Evaluator-only: confirm q_j^T H_j = K_j at the population direction Q_j e_1."""
    d = generate(n, seed)
    errs = []
    for j in range(5):
        q = orthogonal_block(j)[:, 0]
        errs.append(float(np.abs(d["H"][j] @ q - d["K_eval_only"][:, j]).max()))
    cov_dirs = []
    for j in range(5):
        h = d["H"][j]
        cov = (h - h.mean(0)).T @ (d["X"] - d["X"].mean()) / n
        cov_dirs.append(float(np.abs(cov / np.linalg.norm(cov) - orthogonal_block(j)[:, 0]).max()))
    return {"max_inverse_error": max(errs), "per_block_inverse_error": errs,
            "max_direction_error_at_n": max(cov_dirs), "n": n}


# ----------------------------------------------------------------------------- critics and selection
def brier_ridge(z: np.ndarray, a: np.ndarray, penalty: float, starts: list | None = None) -> np.ndarray:
    """Bounded-probit critic p = 0.1 + 0.8 Phi(intercept + z beta), fitted by penalised Brier risk.

    The bound matches the generator's own treatment probability range, so the critic cannot chase values the
    design never produces.  The gradient is analytic and the penalty leaves the intercept free.
    """
    from scipy.optimize import minimize
    design = np.column_stack([np.ones(len(z)), z])
    mask = np.ones(design.shape[1]); mask[0] = 0.0

    def obj(beta):
        eta = design @ beta
        p = 0.1 + 0.8 * norm.cdf(eta)
        resid = p - a
        grad = design.T @ (2.0 * resid * 0.8 * norm.pdf(eta)) / len(a) + 2.0 * penalty * mask * beta
        return float(np.mean(resid ** 2) + penalty * float(np.sum((mask * beta) ** 2))), grad

    best, best_val = None, np.inf
    for s in (starts or [np.zeros(design.shape[1])]):
        r = minimize(obj, s, jac=True, method="L-BFGS-B")
        if r.fun < best_val:
            best, best_val = r.x, r.fun
    return best


def brier_rows(beta: np.ndarray, z: np.ndarray, a: np.ndarray) -> np.ndarray:
    p = 0.1 + 0.8 * norm.cdf(np.column_stack([np.ones(len(z)), z]) @ beta)
    return (p - a) ** 2


def representation(k_hat: np.ndarray, rows: dict, split: tuple[int, ...], r: float | None) -> np.ndarray:
    """Z_{S,r} = (X, C, mean of the kept channels + r I), or (X, mean) when block 0 is held out.

    When block 0 is held out, I and C are not observable, and no other block may stand in for them.
    """
    kept = [j for j in range(5) if j not in split]
    mean_k = k_hat[:, kept].mean(axis=1)
    if 0 in split:
        return np.column_stack([rows["X"], mean_k])
    return np.column_stack([rows["X"], rows["C"], mean_k + r * rows["I"]])


def heldout(rows: dict, split: tuple[int, ...]) -> np.ndarray:
    """T_S = (X, W_S): the held-out blocks in full, 256 dimensions each, plus X."""
    cols = [rows["X"][:, None]]
    for j in split:
        if j == 0:
            cols.extend([rows["I"][:, None], rows["C"][:, None]])
        cols.append(rows["H"][j])
    return np.column_stack(cols)


class Scaler:
    """Column centring and scaling fitted on one set of rows and then frozen.

    A ridge penalty is not scale invariant, and the coefficient r changes the scale of the third
    representation coordinate.  Without this step the penalised objective falls monotonically as |r| grows
    for a reason that has nothing to do with balance, and the minimiser sits on the edge of the grid.  That
    was observed directly before this was added.  Standardising first makes the penalty act the same way at
    every r, so the objective compares representations rather than magnitudes.
    """

    def __init__(self, x: np.ndarray) -> None:
        self.mu = x.mean(0)
        self.sd = np.maximum(x.std(0), 1e-8)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sd


def choose_penalty(design_fn, a: np.ndarray, grid: tuple, seed: int) -> float:
    """Three-fold selection of the ridge penalty inside D_phi only, on the base critic's own design.

    Only one penalty is chosen, and it is then used by both critics; see split_penalties for why a separate
    penalty for the augmented critic is not permitted.  The screen rows and the evaluation rows never enter
    this choice.
    """
    design = design_fn()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(a))
    folds = np.array_split(order, 3)
    best, best_risk = grid[0], np.inf
    for pen in grid:
        risk = 0.0
        for f in range(3):
            te = folds[f]
            tr = np.concatenate([folds[g] for g in range(3) if g != f])
            beta = brier_ridge(design[tr], a[tr], pen)
            risk += float(np.mean(brier_rows(beta, design[te], a[te])))
        if risk < best_risk:
            best, best_risk = pen, risk
    return best


def select_r(rows: dict, k_hat: np.ndarray, split: tuple[int, ...], penalties: dict,
             grid: tuple = c.GRID) -> dict:
    """Line 2 of Algorithm 1 on D_phi: minimise the nested Brier gap over the frozen r grid.

    Both critics carry the same ridge penalty, and that is forced rather than chosen.  Containment of the
    base class inside the augmented one is what makes the gap an estimate of a non-negative discrepancy: the
    lifted base solution, with zeros on the held-out coordinates, must be feasible for the augmented problem
    at the same objective value.  A heavier penalty on the augmented critic breaks that and produces negative
    gaps, which was observed directly when the two were tuned separately.
    """
    candidates = [None] if 0 in split else list(grid)
    target_raw = heldout(rows, split)
    t_scaler = Scaler(target_raw)
    target = t_scaler(target_raw)
    best = None
    objective = []
    for r in candidates:
        z_raw = representation(k_hat, rows, split, r)
        z_scaler = Scaler(z_raw)
        z = z_scaler(z_raw)
        b0 = brier_ridge(z, rows["A"], penalties["base"])
        lifted = np.concatenate([b0, np.zeros(target.shape[1])])
        b1 = brier_ridge(np.column_stack([z, target]), rows["A"], penalties["base"],
                         starts=[lifted, np.zeros(len(lifted))])
        gap = float(np.mean(brier_rows(b0, z, rows["A"])
                            - brier_rows(b1, np.column_stack([z, target]), rows["A"])))
        if gap < c.GAP_FLOOR:
            raise RuntimeError(f"nested Brier gap below the floor: {gap}")
        objective.append(gap)
        cand = {"r": r, "signed_gap": gap, "objective": max(gap, 0.0), "beta0": b0, "beta1": b1,
                "z_scaler": z_scaler, "t_scaler": t_scaler}
        if best is None or cand["objective"] < best["objective"] - c.TIE_TOL or (
                abs(cand["objective"] - best["objective"]) <= c.TIE_TOL
                and (abs(r or 0.0), r or 0.0) < (abs(best["r"] or 0.0), best["r"] or 0.0)):
            best = cand
    best["objective_vector_sha256"] = c.array_sha256(np.asarray(objective))
    best["objective_min_count"] = int(sum(1 for v in objective if max(v, 0.0) <= best["objective"] + c.TIE_TOL))
    return best


def split_penalties(rows: dict, k_hat: np.ndarray, split: tuple[int, ...], grid: tuple,
                    seed: int) -> dict:
    """The single ridge penalty this split will use, chosen at the reference coefficient r = 0.

    Both critics must carry the same penalty.  The base critic's solution, padded with zeros on the
    held-out coordinates, is feasible for the augmented problem and attains the same penalised objective
    only when the penalty is identical, and that containment is what forces the nested gap to estimate a
    non-negative discrepancy.  Tuning the augmented critic separately was tried and produced a gap of
    -3.3e-4 on a split where the true discrepancy is non-negative.
    """
    ref = None if 0 in split else 0.0
    z_raw = representation(k_hat, rows, split, ref)
    z = Scaler(z_raw)(z_raw)
    shared = choose_penalty(lambda: z, rows["A"], grid, seed)
    return {"base": shared, "augmented": shared, "reference_r": ref,
            "rule": "one penalty for both critics, required for nested containment"}


def screen(rows: dict, k_hat: np.ndarray, split: tuple[int, ...], selected: dict) -> dict:
    """An independent screen on S: the clipped gap must not exceed twice its own standard error.

    Failing to reject a zero discrepancy is not a proof of balance.  It is the operational screen the
    algorithm specifies, and it is evaluated with the representation and critics held fixed.
    """
    z = selected["z_scaler"](representation(k_hat, rows, split, selected["r"]))
    target = selected["t_scaler"](heldout(rows, split))
    d = (brier_rows(selected["beta0"], z, rows["A"])
         - brier_rows(selected["beta1"], np.column_stack([z, target]), rows["A"]))
    gap, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    p = 0.1 + 0.8 * norm.cdf(np.column_stack([np.ones(len(z)), z]) @ selected["beta0"])
    overlap = bool(np.isfinite(p).all() and p.min() >= c.OVERLAP_RANGE[0] and p.max() <= c.OVERLAP_RANGE[1])
    return {"gap": gap, "se": se, "overlap_ok": overlap, "p_min": float(p.min()), "p_max": float(p.max()),
            "pass": bool(max(gap, 0.0) <= 2.0 * se and overlap)}


# ----------------------------------------------------------------------------- nuisance, AIPW, baselines
def aipw_from_features(z_n, a_n, y_n, z_e, a_e, y_e) -> np.ndarray:
    """Honest AIPW on a supplied feature map: nuisances from N, scores on E, with the propensity clipped."""
    from scipy.optimize import minimize
    zn = Scaler(z_n)
    zs_n, zs_e = zn(z_n), zn(z_e)
    dn = np.column_stack([np.ones(len(zs_n)), zs_n])
    de = np.column_stack([np.ones(len(zs_e)), zs_e])

    def obj(b):
        eta = dn @ b
        p = 0.1 + 0.8 * norm.cdf(eta)
        res = p - a_n
        return float(np.mean(res ** 2)), dn.T @ (2.0 * res * 0.8 * norm.pdf(eta)) / len(a_n)

    beta = minimize(obj, np.zeros(dn.shape[1]), jac=True, method="L-BFGS-B").x
    p_e = np.clip(0.1 + 0.8 * norm.cdf(de @ beta), c.OVERLAP_RANGE[0], c.OVERLAP_RANGE[1])
    mu = []
    for arm in (0, 1):
        m = a_n == arm
        coef = np.linalg.lstsq(dn[m], y_n[m], rcond=None)[0]
        mu.append(de @ coef)
    return mu[1] - mu[0] + a_e * (y_e - mu[1]) / p_e - (1 - a_e) * (y_e - mu[0]) / (1 - p_e)


def blockwise_pca(fit_rows: dict, n_components: int = 1) -> list[np.ndarray]:
    """Leading principal directions of each block, fitted on the embedding rows only."""
    out = []
    for j in range(5):
        h = fit_rows["H"][j]
        hc = h - h.mean(0)
        _, _, vt = np.linalg.svd(hc, full_matrices=False)
        out.append(vt[:n_components].T)
    return out


def baseline_features(rows: dict, emb: dict | None, pcs: list | None) -> dict:
    """The comparators of section 7.5, all built from observed data on the same rows."""
    x = rows["X"][:, None]
    feats = {"X": x}
    raw = np.column_stack([x, rows["I"][:, None], rows["C"][:, None]] + rows["H"])
    feats["raw_ridge"] = raw
    if emb is not None:
        feats["learned_projection"] = np.column_stack([x, rows["I"][:, None], rows["C"][:, None],
                                                       apply_embedding(emb, rows)])
    if pcs is not None:
        feats["blockwise_pca"] = np.column_stack([x, rows["I"][:, None], rows["C"][:, None]]
                                                 + [rows["H"][j] @ pcs[j] for j in range(5)])
    return feats
