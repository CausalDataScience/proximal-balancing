"""PROBE on SCM family v2: a role-blind linear representation learned by the nested Brier gap.

For one data set (learner view only) and one split S of the five blocks:

    features   X standardised; every block with more than two coordinates reduced on the D fold to its
               principal components above the Marchenko-Pastur edge (at most P; no A, no Y, no role)
    Z_S        (X, V_{-S} B^T),  B in R^{k x q} with unit rows, V_{-S} the retained block features
    objective  G(B) = R_0(B) - R_1(B), the penalised Brier risks of two nested bounded-probit critics on D:
               the base critic reads (1, X, Z), the augmented one also reads the held-out features T_S.
               The augmented fit starts from the lifted base fit, so G >= 0 by construction.
    gradient   envelope theorem: critics at their optimum, chain rule through Z = V B^T and the row norm
    screen     fresh nested critics cross-fitted on the two halves of the S fold; upper = max(gap, 0) +
               z_{1 - alpha/N} SE < t over the pooled out-of-fold differences, and overlap holds
    estimate   bounded-probit propensity and arm-wise ridge outcome on N, honest AIPW on E

Four rotations of four folds; a split is retained when it passes in all four; its estimate is the mean over
rotations.  Aggregation links retained estimates within 2 rho and returns the largest component's median,
with rho from the sealed common_radius rule.  Nothing here reads a latent, a coordinate name or a block role.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import norm

import family_v2_dgp as g
import probe_structured_scm as ps

P_FEATURES = 4
LINK_LO, LINK_HI = 0.1, 0.9
RIDGE = 1e-4
ALPHA = 0.05
N_SPLITS = 30
OUTER_ITERS = 60
RESTARTS = 2
CLIP = (0.05, 0.95)


_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def link(eta: np.ndarray) -> np.ndarray:
    return LINK_LO + (LINK_HI - LINK_LO) * ndtr(eta)


def dlink(eta: np.ndarray) -> np.ndarray:
    return (LINK_HI - LINK_LO) * _INV_SQRT_2PI * np.exp(-0.5 * eta * eta)


# ----------------------------------------------------------------------------- folds and features

def folds(n: int, seed: int) -> list[np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    return np.array_split(perm, 4)


def roles(rotation: int) -> dict[str, int]:
    return {"D": rotation % 4, "S": (rotation + 1) % 4, "N": (rotation + 2) % 4, "E": (rotation + 3) % 4}


MP_MARGIN = 1.1


def spike_count(eigenvalues: np.ndarray, n: int, d: int) -> int:
    """Principal components above the Marchenko-Pastur bulk edge, with the noise level set by the median
    eigenvalue.  Unsupervised and role-blind: it reads only the block's own sample covariance."""
    sigma2 = float(np.median(eigenvalues))
    edge = sigma2 * (1.0 + math.sqrt(d / n)) ** 2
    return int(min(P_FEATURES, max(1, np.sum(eigenvalues > MP_MARGIN * edge))))


class Prep:
    """Unsupervised preprocessing fitted on the D fold: standardise X; reduce each block with more than two
    coordinates to its principal components above the Marchenko-Pastur edge (at most P); standardise."""

    def __init__(self, view: dict[str, np.ndarray], rows: np.ndarray):
        x = view["X"][rows]
        self.x_mu, self.x_sd = x.mean(0), x.std(0) + 1e-12
        self.blocks = []
        for j in range(g.N_BLOCKS):
            w = view[f"W{j + 1}"][rows]
            mu = w.mean(0)
            if w.shape[1] > 2:
                _, sv, vt = np.linalg.svd(w - mu, full_matrices=False)
                eig = sv ** 2 / len(rows)
                proj = vt[:spike_count(eig, len(rows), w.shape[1])].T
            else:
                proj = np.eye(w.shape[1])
            scores = (w - mu) @ proj
            self.blocks.append((mu, proj, scores.std(0) + 1e-12))

    def x(self, view, rows):
        return (view["X"][rows] - self.x_mu) / self.x_sd

    def block(self, view, rows, j):
        mu, proj, sd = self.blocks[j]
        return ((view[f"W{j + 1}"][rows] - mu) @ proj) / sd

    def widths(self) -> list[int]:
        return [b[1].shape[1] for b in self.blocks]


# ----------------------------------------------------------------------------- critics

def _risk_grad(c: np.ndarray, f: np.ndarray, a: np.ndarray, lam: float):
    eta = f @ c
    q = link(eta)
    r = q - a
    pen = c.copy()
    pen[0] = 0.0
    risk = float(np.mean(r ** 2) + lam * (pen @ pen))
    grad = (2.0 / len(a)) * (f.T @ (r * dlink(eta))) + 2.0 * lam * pen
    return risk, grad


def fit_critic(f: np.ndarray, a: np.ndarray, start: np.ndarray | None, lam: float = RIDGE):
    x0 = np.zeros(f.shape[1]) if start is None else start
    out = minimize(_risk_grad, x0, args=(f, a, lam), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-9})
    return out.x, float(out.fun)


def nested_critics(f0: np.ndarray, t: np.ndarray, a: np.ndarray, warm: dict | None = None):
    """Base on f0, augmented on (f0, t); the augmented fit keeps the lifted base if it cannot improve on it."""
    c0, r0 = fit_critic(f0, a, None if warm is None else warm.get("c0"))
    f1 = np.column_stack([f0, t])
    lifted = np.concatenate([c0, np.zeros(t.shape[1])])
    start = lifted if warm is None or warm.get("c1") is None else warm["c1"]
    c1, r1 = fit_critic(f1, a, start)
    lifted_risk = _risk_grad(lifted, f1, a, RIDGE)[0]
    if lifted_risk < r1:
        c1, r1 = lifted, lifted_risk
    return {"c0": c0, "r0": r0, "c1": c1, "r1": r1}


# ----------------------------------------------------------------------------- representation learning

def _normalise(b: np.ndarray):
    norms = np.linalg.norm(b, axis=1, keepdims=True) + 1e-15
    return b / norms, norms


def learn_representation(x: np.ndarray, v: np.ndarray, t: np.ndarray, a: np.ndarray, k: int,
                         seed: int) -> dict:
    """Minimise G(B) over unit-row B from a principal-direction start and a random start; keep the lower."""
    q = v.shape[1]
    kk = min(k, q)
    ones = np.ones((len(a), 1))
    _, _, vt = np.linalg.svd(v - v.mean(0), full_matrices=False)
    starts = [vt[:kk].copy(), np.random.default_rng(seed).standard_normal((kk, q))]
    best = None
    for s_idx, b_start in enumerate(starts[:RESTARTS]):
        warm = {}

        def objective(flat):
            b_raw = flat.reshape(kk, q)
            b, norms = _normalise(b_raw)
            z = v @ b.T
            f0 = np.column_stack([ones, x, z])
            fit = nested_critics(f0, t, a, warm)
            warm.update(c0=fit["c0"], c1=fit["c1"])
            zi = slice(1 + x.shape[1], 1 + x.shape[1] + kk)
            grads = []
            for f, c in ((f0, fit["c0"]), (np.column_stack([f0, t]), fit["c1"])):
                eta = f @ c
                r = (link(eta) - a) * dlink(eta)
                grads.append(np.outer(c[zi], (2.0 / len(a)) * (v.T @ r)))
            g_bn = grads[0] - grads[1]
            proj = g_bn - np.sum(g_bn * b, axis=1, keepdims=True) * b
            return fit["r0"] - fit["r1"], (proj / norms).ravel()

        out = minimize(objective, b_start.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": OUTER_ITERS, "ftol": 1e-14, "gtol": 1e-12})
        b, _ = _normalise(out.x.reshape(kk, q))
        gap = float(out.fun)
        if best is None or gap < best["gap_fit"]:
            best = {"b": b, "gap_fit": gap, "start": s_idx, "iterations": int(out.nit)}
    return best


# ----------------------------------------------------------------------------- screen, nuisance, AIPW

def screen(x_s, v_s, t_s, a_s, b, tolerance: float) -> dict:
    """Honest screen on the S fold with fresh critics.

    The critics fitted on D during the search can have the held-out coefficients driven to exactly zero by
    the search itself, which would make any test on new rows degenerate.  So both critics are refitted on one
    half of S and scored on the other, in both directions, and the out-of-fold squared-error differences are
    pooled.  Overlap: every base prediction lies in [0.05, 0.95]; with the bounded link this always holds,
    and it is recorded rather than assumed."""
    ones = np.ones((len(a_s), 1))
    f0 = np.column_stack([ones, x_s, v_s @ b.T])
    f1 = np.column_stack([f0, t_s])
    half = np.arange(len(a_s)) % 2 == 0
    d = np.empty(len(a_s))
    q0_all = np.empty(len(a_s))
    for fit_rows, eval_rows in ((half, ~half), (~half, half)):
        crit = nested_critics(f0[fit_rows], t_s[fit_rows], a_s[fit_rows])
        q0 = link(f0[eval_rows] @ crit["c0"])
        q1 = link(f1[eval_rows] @ crit["c1"])
        d[eval_rows] = (q0 - a_s[eval_rows]) ** 2 - (q1 - a_s[eval_rows]) ** 2
        q0_all[eval_rows] = q0
    gap, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    z = float(norm.ppf(1.0 - ALPHA / N_SPLITS))
    upper = max(gap, 0.0) + z * se
    overlap = bool(np.all((q0_all >= CLIP[0]) & (q0_all <= CLIP[1])))
    return {"gap": gap, "se": se, "upper": upper, "overlap": overlap,
            "pass": bool(upper < tolerance and overlap)}


def ridge_ols(f: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    pen = lam * len(y) * np.eye(f.shape[1])
    pen[0, 0] = 0.0
    return np.linalg.solve(f.T @ f + pen, f.T @ y)


def aipw(f_n, a_n, y_n, f_e, a_e, y_e, lam: float = RIDGE) -> np.ndarray:
    """Honest AIPW scores on E with nuisances fitted on N; features include the intercept column."""
    c, _ = fit_critic(f_n, a_n, None, lam)
    e = np.clip(link(f_e @ c), *CLIP)
    m1 = f_e @ ridge_ols(f_n[a_n == 1], y_n[a_n == 1], lam)
    m0 = f_e @ ridge_ols(f_n[a_n == 0], y_n[a_n == 0], lam)
    return m1 - m0 + a_e * (y_e - m1) / e - (1 - a_e) * (y_e - m0) / (1 - e)


# ----------------------------------------------------------------------------- one data set

def all_splits() -> list[tuple[int, ...]]:
    import itertools
    return [s for r in range(1, g.N_BLOCKS) for s in itertools.combinations(range(g.N_BLOCKS), r)]


def split_name(split) -> str:
    return "S" + "".join(str(j + 1) for j in split)


def run_dataset(view: dict[str, np.ndarray], level: g.Level, tolerance: float, seed: int,
                keep_b: bool = False) -> dict:
    """Algorithm 1 on one data set.  Returns per-split, per-rotation records and the aggregate."""
    t_start = time.time()
    n = len(view["A"])
    fold_rows = folds(n, seed)
    a_all, y_all = view["A"], view["Y"]
    splits = all_splits()
    scores = np.zeros((n, len(splits)))
    records = {split_name(s): {"rotations": []} for s in splits}
    for rot in range(4):
        rl = roles(rot)
        rows = {k: fold_rows[v] for k, v in rl.items()}
        prep = Prep(view, rows["D"])
        feats = {role: {"x": prep.x(view, r), "blocks": [prep.block(view, r, j) for j in range(g.N_BLOCKS)]}
                 for role, r in rows.items()}
        a = {role: a_all[r] for role, r in rows.items()}
        y = {role: y_all[r] for role, r in rows.items()}
        for s_idx, split in enumerate(splits):
            kept = [j for j in range(g.N_BLOCKS) if j not in split]
            cat = lambda role, idx: np.column_stack([feats[role]["blocks"][j] for j in idx])
            learned = learn_representation(feats["D"]["x"], cat("D", kept), cat("D", split), a["D"], level.k,
                                           seed + 1_000 * rot + s_idx)
            b = learned["b"]
            scr = screen(feats["S"]["x"], cat("S", kept), cat("S", split), a["S"], b, tolerance)
            f = {role: np.column_stack([np.ones(len(a[role])), feats[role]["x"], cat(role, kept) @ b.T])
                 for role in ("N", "E")}
            psi = aipw(f["N"], a["N"], y["N"], f["E"], a["E"], y["E"])
            scores[rows["E"], s_idx] = psi
            rec = {"rotation": rot, "gap_fit": learned["gap_fit"], "start": learned["start"],
                   "iterations": learned["iterations"], "screen": scr, "theta": float(psi.mean())}
            if keep_b:
                rec["b"] = b.tolist()
                rec["prep_widths"] = prep.widths()
            records[split_name(split)]["rotations"].append(rec)
    names = [split_name(s) for s in splits]
    for s_idx, name in enumerate(names):
        rots = records[name]["rotations"]
        records[name]["pass_all"] = all(r["screen"]["pass"] for r in rots)
        records[name]["estimate"] = float(np.mean([r["theta"] for r in rots]))
    retained = [i for i, nme in enumerate(names) if records[nme]["pass_all"]]
    if retained:
        rho = float(ps.common_radius(scores[:, retained], seed + 5))
        agg = ps.aggregate([splits[i] for i in retained],
                           np.array([records[names[i]]["estimate"] for i in retained]), rho)
    else:
        rho, agg = None, {"return_kind": "no_return", "output": None, "members": [], "component_sizes": []}
    centred = scores - scores.mean(0)
    return {"level": level.name, "n": n, "tolerance": tolerance, "records": records,
            "retained": [names[i] for i in retained], "rho": rho,
            "return_kind": agg["return_kind"], "estimate": agg["output"],
            "members": agg.get("members"), "component_sizes": agg.get("component_sizes"),
            "score_crossproduct": (centred.T @ centred).tolist(), "score_means": scores.mean(0).tolist(),
            "split_names": names, "seconds": round(time.time() - t_start, 1)}
