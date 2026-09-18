"""Positive-control SCM for the PROBE Algorithm 1 audit (PRD v4).

Observed data are O = (X, W, A, Y) only.  The latent vector U = (U_c, U_t) is never
observed; the simulator and the oracle evaluator are the only holders.  No function in
this module gives a learner the loadings, the latent components, or any block role.

    X_1 = U_c + kappa eta_1
    X_2 = eta_2
    W_j = b_jc U_c + b_jt U_t + sigma_j eps_j,        j = 0 .. J-1
    P(A = 1 | U, X) = 0.1 + 0.8 Phi(L),    L = alpha_c U_c + alpha_t U_t + 0.2 X_1 + 0.3 X_2
    Y = A - c U_c + 0.2 X_1 - 0.5 X_2 + s eps_Y,      so tau = 1

Everything is a fixed linear combination of independent standard normals, and the
treatment link is a probit, so the population quantities below are closed form up to a
two-dimensional Gauss-Hermite rule.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.stats import norm

GH_ORDERS = (48, 96, 144)
BALANCE_TOL = 1e-10


@dataclass(frozen=True)
class Design:
    """Coefficients of one SCM.  Nothing here is visible to a learner."""

    b_c: tuple[float, ...]
    b_t: tuple[float, ...]
    sigma: tuple[float, ...]
    kappa: float = 2.0
    alpha_c: float = 1.0
    alpha_t: float = 1.0
    outcome_u: float = 2.5
    outcome_noise: float = 0.3

    @property
    def J(self) -> int:
        return len(self.b_c)

    @property
    def names(self) -> tuple[str, ...]:
        return ("Uc", "Ut", "eta1", "eta2") + tuple(f"e{j}" for j in range(self.J))

    def row(self, **coef: float) -> np.ndarray:
        index = {name: i for i, name in enumerate(self.names)}
        v = np.zeros(len(self.names))
        for key, value in coef.items():
            v[index[key]] = value
        return v

    @property
    def maps(self) -> dict[str, np.ndarray]:
        m = {
            "Uc": self.row(Uc=1.0),
            "Ut": self.row(Ut=1.0),
            "X1": self.row(Uc=1.0, eta1=self.kappa),
            "X2": self.row(eta2=1.0),
        }
        for j in range(self.J):
            m[f"W{j}"] = self.row(Uc=self.b_c[j], Ut=self.b_t[j], **{f"e{j}": self.sigma[j]})
        m["G"] = self.alpha_c * m["Uc"] + self.alpha_t * m["Ut"]
        m["L"] = m["G"] + 0.2 * m["X1"] + 0.3 * m["X2"]
        return m

    def permuted(self, order: tuple[int, ...]) -> "Design":
        """Relabel the blocks.  Used only by the role-blindness test."""
        return replace(
            self,
            b_c=tuple(self.b_c[j] for j in order),
            b_t=tuple(self.b_t[j] for j in order),
            sigma=tuple(self.sigma[j] for j in order),
        )


# ----------------------------------------------------------------------------- splits and library

def all_splits(J: int) -> tuple[tuple[int, ...], ...]:
    """The 2^J - 2 proper nonempty held-out sets of Algorithm 1."""
    return tuple(s for k in range(1, J) for s in itertools.combinations(range(J), k))


MAX_SUPPORT = 1


def library(d: int) -> tuple[np.ndarray, ...]:
    """The prespecified representation class on d retained blocks.

        B_d = { v / ||v||_2 : v in {-1,0,1}^d, 1 <= ||v||_0 <= MAX_SUPPORT } / {beta ~ -beta}

    so a channel uses at most MAX_SUPPORT retained blocks with a sign.  With the cap at
    one the class is "use a single retained block as the channel" and |B_d| = d.  The class is closed under coordinate permutation and sign flip and
    refers to no block role.  beta and -beta generate the same sigma-algebra, so they are
    identified.

    The support cap is what keeps the population minimiser isolated: a denser library puts
    non-balancing elements arbitrarily close to the balance surface, and then no feasible
    learning sample can separate them.
    """
    out = []
    for v in itertools.product((-1, 0, 1), repeat=d):
        support = sum(x != 0 for x in v)
        if support == 0 or support > MAX_SUPPORT:
            continue
        if next(x for x in v if x) < 0:                     # canonical sign
            continue
        arr = np.array(v, dtype=float)
        out.append(arr / np.linalg.norm(arr))
    return tuple(out)


def kept_blocks(J: int, split: tuple[int, ...]) -> list[int]:
    return [j for j in range(J) if j not in split]


def channel_row(design: Design, split: tuple[int, ...], beta: np.ndarray) -> np.ndarray:
    maps = design.maps
    kept = kept_blocks(design.J, split)
    return sum(b * maps[f"W{j}"] for b, j in zip(beta, kept))


def z_rows(design: Design, split: tuple[int, ...], beta: np.ndarray) -> np.ndarray:
    maps = design.maps
    return np.vstack([maps["X1"], maps["X2"], channel_row(design, split, beta)])


# ----------------------------------------------------------------------------- linear-Gaussian algebra

def conditional(target: np.ndarray, given: np.ndarray) -> tuple[np.ndarray, float]:
    """Coefficient row of E[target | given] and the conditional variance.

    An orthonormal basis of the conditioning rows keeps this invariant to repeated or
    linearly dependent conditioning coordinates.
    """
    if given.shape[0] == 0:
        return np.zeros_like(target), float(target @ target)
    _, sv, vt = np.linalg.svd(given, full_matrices=False)
    basis = vt[sv > 1e-10 * max(sv[0], 1.0)]
    coef = basis @ target
    return coef @ basis, max(float(target @ target - coef @ coef), 0.0)


def gamma(design: Design, split: tuple[int, ...], beta: np.ndarray) -> np.ndarray:
    """(Cov(G, U_c | Z), Cov(G, U_t | Z)), the two partial covariances that decide balance.

    X_2 is independent of everything else, so conditioning on Z = (X_1, X_2, C) and on
    (X_1, C) give the same value.
    """
    maps = design.maps
    Z = z_rows(design, split, beta)
    res_g = maps["G"] - conditional(maps["G"], Z)[0]
    res_c = maps["Uc"] - conditional(maps["Uc"], Z)[0]
    res_t = maps["Ut"] - conditional(maps["Ut"], Z)[0]
    return np.array([float(res_g @ res_c), float(res_g @ res_t)])


def loading_matrix(design: Design, split: tuple[int, ...]) -> np.ndarray:
    return np.array([[design.b_c[j], design.b_t[j]] for j in split], dtype=float)


def balance_residual(design: Design, split: tuple[int, ...], beta: np.ndarray) -> float:
    """||B_S gamma(beta)||_inf.  Zero exactly when D_S^2(beta) = 0.

    In this jointly normal probit design the held-out block adds nothing to the treatment
    probability given Z precisely when its loading row annihilates gamma, because the
    held-out idiosyncratic noise is independent of Z.
    """
    return float(np.max(np.abs(loading_matrix(design, split) @ gamma(design, split, beta))))


def orc_valid(design: Design, split: tuple[int, ...], tol: float = 1e-10) -> bool:
    """Outcome-relevant completeness label: e_c = (1, 0) lies in rowspan(B_S).

    The outcome depends on the latent only through U_c, so the held-out channel has to
    determine that direction.  Evaluation label; never shown to a learner.
    """
    B = loading_matrix(design, split)
    _, sv, vt = np.linalg.svd(B, full_matrices=False)
    basis = vt[sv > tol * max(sv[0], 1.0)] if sv.size and sv[0] > 0 else np.zeros((0, 2))
    e_c = np.array([1.0, 0.0])
    residual = e_c - basis.T @ (basis @ e_c) if basis.shape[0] else e_c
    return bool(np.linalg.norm(residual) <= 1e-8)


# ----------------------------------------------------------------------------- population target value

def _gh2(cov: np.ndarray, order: int):
    """Nodes and weights for a mean-zero bivariate normal with the given covariance.

    A symmetric eigendecomposition is used rather than a Cholesky factor: the two
    conditional means become nearly collinear in some designs, and the factor then fails
    while the eigen route stays well defined.
    """
    x, w = np.polynomial.hermite_e.hermegauss(order)
    w = w / w.sum()
    values, vectors = np.linalg.eigh((cov + cov.T) / 2.0)
    root = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
    g1, g2 = np.meshgrid(x, x, indexing="ij")
    return root @ np.vstack([g1.ravel(), g2.ravel()]), np.outer(w, w).ravel()


def _arm_means(mu_u: np.ndarray, mu_l: np.ndarray, var_l: float, cov_ul: float):
    """E[U_c | A = a, Z] for both arms, exact for the bounded probit link via Stein."""
    root = math.sqrt(1.0 + var_l)
    scaled = mu_l / root
    p = 0.1 + 0.8 * norm.cdf(scaled)
    e_one = (0.1 * mu_u + 0.8 * (mu_u * norm.cdf(scaled) + cov_ul * norm.pdf(scaled) / root)) / p
    e_zero = (mu_u - p * e_one) / (1.0 - p)
    return p, e_one, e_zero


def theta(design: Design, split: tuple[int, ...], beta: np.ndarray, order: int = GH_ORDERS[-1]) -> float:
    """tau_Z for this representation: E[E(Y | A=1, Z) - E(Y | A=0, Z)].

    Y = A - c U_c + 0.2 X_1 - 0.5 X_2 + noise and X is inside Z, so only the latent term
    survives the contrast.
    """
    maps = design.maps
    Z = z_rows(design, split, beta)
    mu_u_row = conditional(maps["Uc"], Z)[0]
    mu_l_row, var_l = conditional(maps["L"], Z)
    cov_ul = float((maps["Uc"] - mu_u_row) @ (maps["L"] - mu_l_row))
    cov = np.array([[mu_u_row @ mu_u_row, mu_u_row @ mu_l_row],
                    [mu_u_row @ mu_l_row, mu_l_row @ mu_l_row]])
    pts, weights = _gh2(cov, order)
    _, e_one, e_zero = _arm_means(pts[0], pts[1], var_l, cov_ul)
    return float(((1.0 - design.outcome_u * (e_one - e_zero)) * weights).sum())


def discrepancy_numeric(design: Design, split: tuple[int, ...], beta: np.ndarray,
                        order: int = GH_ORDERS[0]) -> float:
    """D_S^2(beta) = E[{P(A=1 | W_S, Z) - P(A=1 | Z)}^2], by direct integration.

    This is the definition.  `balance_residual` decides whether it is zero algebraically;
    this function exists so that a test can confirm the two agree.
    """
    maps = design.maps
    Z = z_rows(design, split, beta)
    ZT = np.vstack([Z] + [maps[f"W{j}"] for j in split])
    m0, v0 = conditional(maps["L"], Z)
    m1, v1 = conditional(maps["L"], ZT)
    cov = np.array([[m0 @ m0, m0 @ m1], [m0 @ m1, m1 @ m1]])
    pts, weights = _gh2(cov, order)
    p0 = 0.1 + 0.8 * norm.cdf(pts[0] / math.sqrt(1.0 + v0))
    p1 = 0.1 + 0.8 * norm.cdf(pts[1] / math.sqrt(1.0 + v1))
    return float((((p1 - p0) ** 2) * weights).sum())


def theta_with_residual(design: Design, split: tuple[int, ...], beta: np.ndarray) -> tuple[float, float]:
    values = [theta(design, split, beta, order) for order in GH_ORDERS]
    residual = max(abs(values[1] - values[0]), abs(values[2] - values[1]))
    return values[-1], residual


# ----------------------------------------------------------------------------- sampling and exact nuisance

def generate(design: Design, n: int, seed: int) -> dict[str, np.ndarray]:
    """One data set.  Fields ending in _latent are for the simulator and oracle only."""
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((n, 4 + design.J))
    u_c, u_t, eta1, eta2 = draws[:, 0], draws[:, 1], draws[:, 2], draws[:, 3]
    x1 = u_c + design.kappa * eta1
    x2 = eta2
    w = np.column_stack([
        design.b_c[j] * u_c + design.b_t[j] * u_t + design.sigma[j] * draws[:, 4 + j]
        for j in range(design.J)
    ])
    ell = design.alpha_c * u_c + design.alpha_t * u_t + 0.2 * x1 + 0.3 * x2
    p = 0.1 + 0.8 * norm.cdf(ell)
    a = rng.binomial(1, p).astype(float)
    y = a - design.outcome_u * u_c + 0.2 * x1 - 0.5 * x2 + design.outcome_noise * rng.standard_normal(n)
    return {"X1": x1, "X2": x2, "W": w, "A": a, "Y": y,
            "Uc_latent": u_c, "Ut_latent": u_t, "p_latent": p}


def observed_view(data: dict[str, np.ndarray], *, include_outcome: bool = False) -> dict[str, np.ndarray]:
    keys = ["X1", "X2", "W", "A"] + (["Y"] if include_outcome else [])
    return {k: data[k] for k in keys}


def representation(data: dict[str, np.ndarray], split: tuple[int, ...], beta: np.ndarray, J: int) -> np.ndarray:
    kept = kept_blocks(J, split)
    return np.column_stack([data["X1"], data["X2"], data["W"][:, kept] @ beta])


def exact_nuisance(design: Design, split: tuple[int, ...], beta: np.ndarray,
                   data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """e(z), m_0(z), m_1(z) in closed form under the true law.

    The confirmation uses these, so the AIPW nuisance error term of the manuscript radius
    is exactly zero.
    """
    maps = design.maps
    Z = z_rows(design, split, beta)
    mu_u_row = conditional(maps["Uc"], Z)[0]
    mu_l_row, var_l = conditional(maps["L"], Z)
    cov_ul = float((maps["Uc"] - mu_u_row) @ (maps["L"] - mu_l_row))
    basis = np.vstack([maps["X1"], maps["X2"], channel_row(design, split, beta)])
    # express the two conditional means as linear functions of the realised Z columns
    coef_u = np.linalg.lstsq(basis.T, mu_u_row, rcond=None)[0]
    coef_l = np.linalg.lstsq(basis.T, mu_l_row, rcond=None)[0]
    z = representation(data, split, beta, design.J)
    mu_u, mu_l = z @ coef_u, z @ coef_l
    p, e_one, e_zero = _arm_means(mu_u, mu_l, var_l, cov_ul)
    m_one = 1.0 - design.outcome_u * e_one + 0.2 * data["X1"] - 0.5 * data["X2"]
    m_zero = -design.outcome_u * e_zero + 0.2 * data["X1"] - 0.5 * data["X2"]
    return p, m_zero, m_one


def aggregate(labels: list[tuple[int, ...]], estimates: np.ndarray, rho: float) -> dict:
    """Algorithm 1 lines 5 and 6: link within 2 rho, return the largest component's median.

    Several components of equal largest size return a candidate set rather than a point.
    """
    if len(estimates) == 0:
        return {"return_kind": "no_return", "output": None, "candidates": [], "members": []}
    parent = list(range(len(estimates)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(estimates)):
        for j in range(i + 1, len(estimates)):
            if abs(estimates[i] - estimates[j]) <= 2.0 * rho:
                parent[find(i)] = find(j)
    components: dict[int, list[int]] = {}
    for i in range(len(estimates)):
        components.setdefault(find(i), []).append(i)
    largest_size = max(map(len, components.values()))
    largest = [v for v in components.values() if len(v) == largest_size]
    candidates = [float(np.median(estimates[idx])) for idx in largest]
    return {"return_kind": "unique_largest" if len(largest) == 1 else "tied_largest",
            "output": candidates[0] if len(candidates) == 1 else None,
            "candidates": candidates,
            "members": [[list(labels[i]) for i in idx] for idx in largest],
            "component_sizes": sorted((len(v) for v in components.values()), reverse=True)}


# ----------------------------------------------------------------------------- the manuscript screen

def _brier_risk_and_gradient(coef: np.ndarray, design_matrix: np.ndarray, a: np.ndarray):
    eta = design_matrix @ coef
    p = 0.1 + 0.8 * norm.cdf(eta)
    resid = p - a
    risk = float(np.mean(resid ** 2))
    grad = (2.0 * 0.8 / len(a)) * (design_matrix.T @ (resid * norm.pdf(eta)))
    return risk, grad


def brier_fit(features: np.ndarray, a: np.ndarray, starts: list[np.ndarray] | None = None):
    """Least-Brier probit-linear critic for P(A = 1 | features).

    The link is the bounded probit of the design, 0.1 + 0.8 Phi(eta), so the critic class
    contains the truth whenever the features span the treatment index.
    """
    from scipy.optimize import minimize

    X = np.column_stack([np.ones(len(a)), features])
    if starts is None:
        starts = [np.zeros(X.shape[1])]
    best = None
    for start in starts:
        out = minimize(lambda c: _brier_risk_and_gradient(c, X, a), np.asarray(start, dtype=float),
                       jac=True, method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10})
        if best is None or out.fun < best.fun:
            best = out
    return best.x, float(best.fun)


def screen_statistic(design: Design, split: tuple[int, ...], beta: np.ndarray,
                     data: dict[str, np.ndarray]) -> dict:
    """The manuscript objective: max{R_0 - R_1, 0} for nested probit-linear Brier critics.

    The augmented class contains the base class, which is enforced numerically by starting
    the augmented fit from the fitted base coefficients with zero weight on the held-out
    block.  The statistic is therefore non-negative by construction.
    """
    z = representation(data, split, beta, design.J)
    held = data["W"][:, list(split)]
    coef0, risk0 = brier_fit(z, data["A"])
    lifted = np.concatenate([coef0, np.zeros(held.shape[1])])
    coef1, risk1 = brier_fit(np.column_stack([z, held]), data["A"],
                             starts=[lifted, np.zeros(len(lifted))])
    return {"risk_base": risk0, "risk_augmented": risk1,
            "objective": max(risk0 - risk1, 0.0), "signed_gap": risk0 - risk1}


def select_and_screen(design: Design, split: tuple[int, ...], data: dict[str, np.ndarray],
                      threshold: float) -> dict:
    """Algorithm 1 lines 2 to 4 for one split, over the whole library."""
    lib = library(design.J - len(split))
    values = []
    for beta in lib:
        values.append(screen_statistic(design, split, beta, data)["objective"])
    values = np.array(values)
    order = np.argsort(values)
    best = int(order[0])
    runner_up = float(values[order[1]]) if len(order) > 1 else float("inf")
    return {"beta": lib[best].tolist(), "beta_index": best,
            "objective": float(values[best]), "selection_margin": runner_up - float(values[best]),
            "retained": bool(values[best] <= threshold), "library_size": len(lib)}


def aipw_scores(design: Design, split: tuple[int, ...], beta: np.ndarray,
                data: dict[str, np.ndarray]) -> np.ndarray:
    e, m_zero, m_one = exact_nuisance(design, split, beta, data)
    a, y = data["A"], data["Y"]
    return m_one - m_zero + a * (y - m_one) / e - (1.0 - a) * (y - m_zero) / (1.0 - e)
