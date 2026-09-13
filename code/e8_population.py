"""E8, population half: exact D_res^2(r) and tau_{Z_r} for the SCM family, with no sampling.

Section 2.8 of memo/2026-09-12-scm-family-and-experiment-plan.md asks how much the Brier discrepancy moves
as the representation coefficient r leaves its clean root, and how much causal error that buys.  This module
answers the population side of that question in closed form, so the finite-sample half has a fixed target.

Why closed form is available.  Every variable of SCM-1 to SCM-4 is a fixed linear combination of nine
independent standard normals, and the treatment link is a probit, so for any Gaussian conditioning set W

    E[Phi(V) | W] = Phi( m / sqrt(1 + s^2) ),      m = E[V|W],  s^2 = Var(V|W)

exactly, and Stein's identity gives E[U Phi(V) | W] in closed form as well.  Only the outer expectation over
the conditioning set is left, and it is an expectation of a smooth function of two jointly Gaussian linear
forms, so a two-dimensional Gauss-Hermite rule evaluates it to machine precision.  SCM-5 applies sinh to the
measurements; sinh is strictly increasing, so the representation built from the rank-Gaussian transform
recovers the same K channel and the same population quantities as SCM-1.  That identity is asserted by a
test rather than assumed here, and any learned nonlinear arm must be reported as a Monte Carlo diagnostic
instead of an exact one.

Definitions used below.  U is the latent confounder, I and C are observed inside the proxy blocks, X is the
observed covariate, A the treatment and Y the outcome.  K_j is the measurement channel of block j.  For a
held-out split S the representation is Z_r = (X, C, mean of the kept K + r I) and the held-out variable is
T_S, which contains (I, C, K_0) when block 0 is held out and K_j otherwise.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

import probe_structured_scm as m

# noise coordinates of `generate_role`: z[:, :4] = U, I, C, eps_X and z[:, 4:9] = e_0..e_4
NOISE = ("U", "I", "C", "eX", "e0", "e1", "e2", "e3", "e4")
NIDX = {n: i for i, n in enumerate(NOISE)}
GH_NODES = 48


def row(**coef) -> np.ndarray:
    v = np.zeros(len(NOISE))
    for k, c in coef.items():
        v[NIDX[k]] = c
    return v


def linear_maps(spec: m.SCMSpec) -> dict[str, np.ndarray]:
    """Coefficient rows over the noise vector for every quantity the population formulas need."""
    sd = np.asarray(spec.proxy_sd, dtype=float)
    if sd.ndim == 0:
        sd = np.full(5, float(sd))
    maps = {"U": row(U=1), "I": row(I=1), "C": row(C=1), "X": row(U=1, eX=2.0)}
    for j in range(5):
        maps[f"K{j}"] = row(U=1, **{f"e{j}": sd[j]})
    maps["K4"] = maps["K4"] + spec.bad_i * maps["I"]          # block 4 carries the contamination
    maps["V"] = (maps["U"] + spec.treatment_i * maps["I"] + 0.2 * maps["X"] + 0.3 * maps["C"])
    return maps


def z_rows(maps: dict[str, np.ndarray], split: tuple[int, ...], r: float) -> np.ndarray:
    """Z_r = (X, C, mean of the kept measurements + r I); block 0 must be kept for the typed I channel."""
    if 0 in split:
        raise ValueError("the typed representation needs block 0 among the kept blocks")
    kept = [j for j in range(5) if j not in split]
    mean_k = np.mean([maps[f"K{j}"] for j in kept], axis=0)
    return np.vstack([maps["X"], maps["C"], mean_k + r * maps["I"]])


def t_rows(maps: dict[str, np.ndarray], split: tuple[int, ...]) -> np.ndarray:
    cols = []
    for j in split:
        cols.extend([maps["I"], maps["C"], maps["K0"]] if j == 0 else [maps[f"K{j}"]])
    return np.vstack(cols)


def conditional(target: np.ndarray, given: np.ndarray) -> tuple[np.ndarray, float]:
    """Coefficient row of E[target | given] and the conditional variance, through an orthonormal basis.

    The basis keeps the answer invariant to repeated or linearly dependent conditioning coordinates, which
    happens here because C sits in both Z and T.
    """
    if given.shape[0] == 0:
        return np.zeros(len(NOISE)), float(target @ target)
    _, sv, vt = np.linalg.svd(given, full_matrices=False)
    basis = vt[sv > 1e-10 * max(sv[0], 1.0)]
    coef = basis @ target
    return coef @ basis, max(float(target @ target - coef @ coef), 0.0)


def _gh2(cov: np.ndarray, nodes: int = GH_NODES):
    """Nodes and weights for a mean-zero bivariate Gaussian with the given covariance."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    chol = np.linalg.cholesky(cov + 1e-14 * np.eye(2))
    g1, g2 = np.meshgrid(x, x, indexing="ij")
    pts = chol @ np.vstack([g1.ravel(), g2.ravel()])
    return pts, np.outer(w, w).ravel()


def treatment_probability(mean: np.ndarray, resid_var: float) -> np.ndarray:
    """P(A=1 | W) = 0.1 + 0.8 Phi(m / sqrt(1 + s^2)), exact for the probit link."""
    return 0.1 + 0.8 * norm.cdf(mean / math.sqrt(1.0 + resid_var))


def residual_discrepancy(spec: m.SCMSpec, split: tuple[int, ...], r: float) -> float:
    """D_res^2 = E[{P(A=1|T_S, Z_r) - P(A=1|Z_r)}^2]."""
    maps = linear_maps(spec)
    Z = z_rows(maps, split, r)
    ZT = np.vstack([Z, t_rows(maps, split)])
    m0, s0 = conditional(maps["V"], Z)
    m1, s1 = conditional(maps["V"], ZT)
    cov = np.array([[m0 @ m0, m0 @ m1], [m0 @ m1, m1 @ m1]])
    pts, w = _gh2(cov)
    return float((((treatment_probability(pts[1], s1) - treatment_probability(pts[0], s0)) ** 2) * w).sum())


def adjusted_effect(spec: m.SCMSpec, split: tuple[int, ...], r: float) -> float:
    """tau_{Z_r} = E[ E(Y|A=1,Z_r) - E(Y|A=0,Z_r) ], the population value this representation targets.

    X and C are coordinates of Z, and the outcome error is mean zero and independent of A given everything,
    so only the latent term survives:

        E(Y|A=a,Z) = a(1 + c_e E[U|A=a,Z]) - c_U E[U|A=a,Z] + 0.2 X - 0.5 C.

    With (U, V) bivariate Gaussian given Z, Stein's identity gives

        E[U P(A=1|U,V) | Z] = 0.1 mu_U + 0.8 { mu_U Phi(mu_V / sqrt(1+s_V^2))
                                               + Cov(U,V) phi(mu_V / sqrt(1+s_V^2)) / sqrt(1+s_V^2) },

    and E[U | A=0, Z] follows from E[U|Z] = p E[U|A=1,Z] + (1-p) E[U|A=0,Z].  Balance means tau_Z = 1.
    """
    maps = linear_maps(spec)
    Z = z_rows(maps, split, r)
    mu_u_row, _ = conditional(maps["U"], Z)
    mu_v_row, s_v2 = conditional(maps["V"], Z)
    # residual covariance of U and V given Z
    ru, rv = maps["U"] - mu_u_row, maps["V"] - mu_v_row
    cov_uv = float(ru @ rv)
    cov = np.array([[mu_u_row @ mu_u_row, mu_u_row @ mu_v_row],
                    [mu_u_row @ mu_v_row, mu_v_row @ mu_v_row]])
    pts, w = _gh2(cov)
    mu_u, mu_v = pts[0], pts[1]
    root = math.sqrt(1.0 + s_v2)
    scaled = mu_v / root
    p = 0.1 + 0.8 * norm.cdf(scaled)
    e_u_a1 = (0.1 * mu_u + 0.8 * (mu_u * norm.cdf(scaled) + cov_uv * norm.pdf(scaled) / root)) / p
    e_u_a0 = (mu_u - p * e_u_a1) / (1.0 - p)
    contrast = (1.0 + spec.effect_u * e_u_a1) - spec.outcome_u * (e_u_a1 - e_u_a0)
    return float((contrast * w).sum())


def clean_roots(spec: m.SCMSpec, split: tuple[int, ...], lo: float = -4.0, hi: float = 4.0,
                scan: int = 801, tol: float = 1e-13) -> list[float]:
    """Every r at which the held-out block is balanced, refined from a scan of D_res^2.

    The balance condition is quadratic in r, so the discrepancy has two zeros, not one.  A unimodal search
    returns whichever it happens to bracket, which is why an earlier version of this function reported the
    far root for one split and the near root for another.  Both are returned, sorted.
    """
    grid = np.linspace(lo, hi, scan)
    values = np.array([residual_discrepancy(spec, split, r) for r in grid])
    roots = []
    for i in range(1, scan - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            a, b = grid[i - 1], grid[i + 1]
            phi = (math.sqrt(5.0) - 1.0) / 2.0
            c, d = b - phi * (b - a), a + phi * (b - a)
            for _ in range(200):
                if residual_discrepancy(spec, split, c) < residual_discrepancy(spec, split, d):
                    b, d = d, c
                    c = b - phi * (b - a)
                else:
                    a, c = c, d
                    d = a + phi * (b - a)
                if b - a < 1e-12:
                    break
            r = 0.5 * (a + b)
            if residual_discrepancy(spec, split, r) < tol and all(abs(r - q) > 1e-6 for q in roots):
                roots.append(float(r))
    return sorted(roots)


# The implementation searches r on np.linspace(-1, 1, 101), so only a root inside that interval is reachable.
GRID_LO, GRID_HI = -1.0, 1.0


def operative_root(spec: m.SCMSpec, split: tuple[int, ...]) -> dict:
    """The balancing r the implementation can actually reach, alongside every root the design admits."""
    roots = clean_roots(spec, split)
    inside = [r for r in roots if GRID_LO <= r <= GRID_HI]
    return {"roots": roots, "inside_grid": inside,
            "r_star": inside[0] if inside else (roots[0] if roots else None),
            "reachable": bool(inside)}


def conditional_arm_means(spec: m.SCMSpec, Z_rows: np.ndarray, z_values: np.ndarray) -> dict:
    """Exact e_S(z), m_0(z) and m_1(z) for a linear representation, evaluated at supplied rows.

    `Z_rows` gives the representation as coefficient rows over the noise vector; `z_values` are the realised
    coordinates of that representation, one row per unit.  Everything below is exact for this family.

    Given Z, the vector (U, C, X, V) is jointly Gaussian, so with V the probit index,

        P(A=1 | Z=z) = 0.1 + 0.8 Phi( mu_V(z) / sqrt(1 + s_V^2) ),

    and for any linear W, Stein's identity gives

        E[W P(A=1|V) | Z=z] = 0.1 mu_W(z)
                              + 0.8 { mu_W(z) Phi(k) + Cov(W,V|Z) phi(k) / sqrt(1+s_V^2) },  k = mu_V/sqrt(1+s_V^2),

    from which E[W | A=1, Z=z] follows by division and E[W | A=0, Z=z] from the tower property.  The outcome
    mean in each arm is then

        m_a(z) = a (1 + c_e E[U|A=a,Z=z]) - c_U E[U|A=a,Z=z] + 0.2 E[X|A=a,Z=z] - 0.5 E[C|A=a,Z=z].
    """
    maps = linear_maps(spec)
    # coefficient rows of the conditional means given Z, and the realised values of those means
    def mean_of(name: str) -> tuple[np.ndarray, float]:
        row_, var = conditional(maps[name], Z_rows)
        return row_, var

    beta = np.linalg.lstsq(Z_rows.T, np.eye(len(NOISE)), rcond=None)[0]   # unused; kept explicit below
    # express each conditional mean as a linear function of the representation coordinates
    gram = Z_rows @ Z_rows.T
    def coords(name: str) -> np.ndarray:
        return np.linalg.lstsq(gram, Z_rows @ maps[name], rcond=None)[0]

    mu = {name: z_values @ coords(name) for name in ("U", "C", "X", "V")}
    _, s_v2 = mean_of("V")
    root = math.sqrt(1.0 + s_v2)
    k = mu["V"] / root
    p = 0.1 + 0.8 * norm.cdf(k)
    out = {"e": p}
    arm = {}
    for name in ("U", "C", "X"):
        row_, _ = mean_of(name)
        cov_wv = float((maps[name] - row_) @ (maps["V"] - mean_of("V")[0]))
        e_w_a1 = (0.1 * mu[name] + 0.8 * (mu[name] * norm.cdf(k) + cov_wv * norm.pdf(k) / root)) / p
        arm[name] = (e_w_a1, (mu[name] - p * e_w_a1) / (1.0 - p))
    for a in (0, 1):
        u = arm["U"][1 - a] if a == 0 else arm["U"][0]
        c = arm["C"][1 - a] if a == 0 else arm["C"][0]
        x = arm["X"][1 - a] if a == 0 else arm["X"][0]
        out[f"m{a}"] = a * (1.0 + spec.effect_u * u) - spec.outcome_u * u + 0.2 * x - 0.5 * c
    return out
