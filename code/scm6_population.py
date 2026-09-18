"""Population residual discrepancy for the SCM-6 representation family.

Every proxy block is an orthogonal rotation of one informative coordinate and independent noise, so
conditioning on the block is conditioning on that coordinate: sigma(W_j) = sigma(K_j) joined with noise
that is independent of the latents.  The propensity index is jointly Gaussian with the latents and the
link is probit, so both conditional propensities are exact, and

    E[Phi(V)] = Phi(m / sqrt(1 + s^2))   for   V ~ N(m, s^2)

turns each one into a closed-form function of the conditioning variables.  Only the outer expectation of
the squared difference is taken by simulation, and it involves no fitting of any kind.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

import minimal_suite_common as c
import probe_highdim_w as hd

LATENTS = ("U", "I", "C", "eX", "x0", "x1", "x2", "x3", "x4")
INDEX = {"U": 1.0, "I": hd.TREATMENT_I, "X": 0.2, "C": 0.3}


def _loadings(split: tuple[int, ...], r: float,
              channel_noise: float = 0.0) -> tuple[np.ndarray, list[np.ndarray]]:
    """Rows of the conditioning variables in the latent basis, for Z alone and for (T_S, Z).

    `channel_noise` is the standard deviation of independent error added to each recovered channel, which
    is what an imperfect encoder leaves behind.  It enters the representation, not the held-out target,
    because the target is the raw block and the encoder only touches the kept side.  Setting it to zero
    gives the exact-channel population that SCM-6's linear embedding is close to.
    """
    e = {name: np.eye(len(LATENTS))[i] for i, name in enumerate(LATENTS)}
    if channel_noise > 0.0:
        e = {name: np.concatenate([v, np.zeros(5)]) for name, v in e.items()}
        extra = [np.concatenate([np.zeros(len(LATENTS)), np.eye(5)[j]]) for j in range(5)]
    x = e["U"] + 2.0 * e["eX"]
    k = [e["U"] + hd.PROXY_SD * e[f"x{j}"] for j in range(5)]
    k[4] = k[4] + hd.BAD_I * e["I"]
    kept = [j for j in range(5) if j not in split]
    index = INDEX["U"] * e["U"] + INDEX["I"] * e["I"] + INDEX["X"] * x + INDEX["C"] * e["C"]
    k_hat = list(k)
    if channel_noise > 0.0:
        k_hat = [k[j] + channel_noise * extra[j] for j in range(5)]
    mean_kept = sum(k_hat[j] for j in kept) / len(kept)
    if 0 in split:
        z = [x, mean_kept]
    else:
        z = [x, e["C"], mean_kept + r * e["I"]]
    extra = [k[j] for j in split if j != 0] + ([e["I"], e["C"]] if 0 in split else [])
    return index, [np.asarray(z), np.asarray(z + extra)]


def _conditional(index: np.ndarray, design: np.ndarray, draws: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact conditional mean and variance of the index given a set of jointly Gaussian variables."""
    sigma = design @ design.T
    cross = design @ index
    coef = np.linalg.lstsq(sigma, cross, rcond=None)[0]
    var = float(index @ index - cross @ coef)
    return draws @ design.T @ coef, max(var, 0.0)


def residual_discrepancy(split: tuple[int, ...], r: float, n: int = 400000,
                         seed: int = 2026091301) -> dict:
    """D^2_res for one split at one coefficient, and the two conditional propensities behind it."""
    index, designs = _loadings(split, r)
    draws = np.random.default_rng(seed).standard_normal((n, len(LATENTS)))
    probs = []
    for design in designs:
        mean, var = _conditional(index, design, draws)
        probs.append(0.1 + 0.8 * norm.cdf(mean / np.sqrt(1.0 + var)))
    gap = probs[1] - probs[0]
    return {"split": split, "r": r, "d2_res": float(np.mean(gap ** 2)),
            "mean_abs_gap": float(np.mean(np.abs(gap))),
            "base_brier": float(np.mean(probs[0] * (1 - probs[0]))),
            "augmented_brier": float(np.mean(probs[1] * (1 - probs[1])))}


def balance_residual(split: tuple[int, ...], r: float, channel_noise: float = 0.0) -> np.ndarray:
    """Conditional covariance of the propensity index with each extra variable of T_S, given Z.

    Everything here is jointly Gaussian, so zero conditional covariance is conditional independence, and
    conditional independence of the index from T_S given Z is exactly D^2_res = 0.  Cancelling the
    contaminating block out of the kept average is NOT the same condition and does not solve it: the extra
    block sharpens U and I together, and the coefficient has to balance both channels at once.
    """
    index, designs = _loadings(split, r, channel_noise)
    z, full = designs
    extra = full[len(z):]
    if len(extra) == 0:
        return np.zeros(0)
    coef = np.linalg.lstsq(z @ z.T, z @ index, rcond=None)[0]
    return extra @ index - extra @ z.T @ coef


def balance_roots(split: tuple[int, ...], scan: tuple[float, float] = (-6.0, 6.0),
                  points: int = 24001, tol: float = 1e-13,
                  channel_noise: float = 0.0) -> list[float]:
    """Every coefficient in the scan window at which all components of the balance residual vanish.

    The residual is a rational function of r and it is not monotone: it changes sign more than once, so a
    single bracket on the endpoints finds nothing.  The window is scanned for sign changes and each one is
    bisected, and a candidate is kept only when every component vanishes there at once, which matters for
    splits whose held-out blocks are not all of the same type.
    """
    grid = np.linspace(scan[0], scan[1], points)
    values = np.array([balance_residual(split, float(r), channel_noise) for r in grid])
    if values.shape[1] == 0:
        return []
    lead = values[:, 0]
    roots = []
    for i in np.flatnonzero(np.sign(lead[:-1]) * np.sign(lead[1:]) < 0):
        lo, hi = float(grid[i]), float(grid[i + 1])
        f = lambda r: float(balance_residual(split, r, channel_noise)[0])
        for _ in range(200):
            if hi - lo < tol:
                break
            mid = 0.5 * (lo + hi)
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        root = 0.5 * (lo + hi)
        if np.abs(balance_residual(split, root, channel_noise)).max() < 1e-6:
            roots.append(root)
    return roots


def operative_root(split: tuple[int, ...], grid: tuple = c.GRID,
                   channel_noise: float = 0.0) -> float | None:
    """The balancing coefficient the search can actually reach, meaning the one inside the frozen r grid.

    None when block 0 is held out, because the representation then carries no I coordinate at all, and None
    when no balancing coefficient lies inside the grid.
    """
    if 0 in split:
        return None
    inside = [r for r in balance_roots(split, channel_noise=channel_noise)
              if grid[0] <= r <= grid[-1]]
    return None if not inside else min(inside, key=abs)


def curve(split: tuple[int, ...], grid: np.ndarray, n: int = 400000) -> list[dict]:
    return [residual_discrepancy(split, float(r), n) for r in grid]


def _residual_terms(index: np.ndarray, target: np.ndarray, design: np.ndarray,
                    draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Conditional means of the index and of a target, plus the two constants the probit tilt needs."""
    sigma = design @ design.T
    solve = lambda v: np.linalg.lstsq(sigma, design @ v, rcond=None)[0]
    b_index, b_target = solve(index), solve(target)
    var_index = max(float(index @ index - (design @ index) @ b_index), 0.0)
    cov = float(target @ index - (design @ target) @ b_index)
    return draws @ design.T @ b_index, draws @ design.T @ b_target, var_index, cov


def adjusted_effect(split: tuple[int, ...], r: float | None, n: int = 400000,
                    seed: int = 2026091302) -> dict:
    """Population effect obtained by adjusting for Z alone, in closed form up to the outer average.

    Y is linear in A, U, X and C, so the adjusted effect is 1 minus the outcome loadings times the arm
    difference of each latent.  Each arm mean follows from the Gaussian-probit identities
    E[Phi(V)] = Phi(mu / sqrt(1 + s^2)) and E[g Phi(V)] = mu_g Phi(t) + cov phi(t) / sqrt(1 + s^2),
    so no propensity or outcome model is ever fitted.
    """
    index, designs = _loadings(split, r if r is not None else 0.0)
    design = designs[0]
    draws = np.random.default_rng(seed).standard_normal((n, len(LATENTS)))
    e = {name: np.eye(len(LATENTS))[i] for i, name in enumerate(LATENTS)}
    targets = {"U": (-hd.OUTCOME_U, e["U"]), "X": (0.2, e["U"] + 2.0 * e["eX"]), "C": (-0.5, e["C"])}
    tau, parts = 1.0, {}
    for name, (weight, vec) in targets.items():
        mu_v, mu_g, var_v, cov = _residual_terms(index, vec, design, draws)
        scaled = mu_v / np.sqrt(1.0 + var_v)
        p1 = 0.1 + 0.8 * norm.cdf(scaled)
        joint = 0.1 * mu_g + 0.8 * (mu_g * norm.cdf(scaled)
                                    + cov * norm.pdf(scaled) / np.sqrt(1.0 + var_v))
        diff = float(np.mean(joint / p1 - (mu_g - joint) / (1.0 - p1)))
        parts[name] = weight * diff
        tau += weight * diff
    return {"split": split, "r": r, "tau_Z": tau, "contributions": parts,
            "overlap": [float(np.min(0.1 + 0.8 * norm.cdf(
                _residual_terms(index, e["U"], design, draws)[0]
                / np.sqrt(1.0 + _residual_terms(index, e["U"], design, draws)[2])))),
                        float(np.max(0.1 + 0.8 * norm.cdf(
                            _residual_terms(index, e["U"], design, draws)[0]
                            / np.sqrt(1.0 + _residual_terms(index, e["U"], design, draws)[2]))))]}
