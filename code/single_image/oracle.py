"""Population ("oracle") quantities of the phantom (plan sections 3.6 and 10.2), evaluation only.

The phantom's latent part is linear Gaussian, so the oracle predictors need no fitting: with K the set of
lobes whose true burden L_k is available, U | L_K ~ N(m_K, v_K), Omega_K = |K| / sigma^2,
v_K = 1 / (1 + Omega_K), m_K = v_K sum_{k in K} L_k / sigma^2, and

    P(A = 1 | L_K, X, I) = Phi((1.5 m_K + 1.0 I + 0.2 X) / sqrt(1 + 1.5^2 v_K)),

averaged over I ~ Bern(0.3) when I is not seen.  The oracle-reader representation for split S is
Z_S = (X, m_{-S}) (and I when it is visible in a retained block): m_{-S} is the posterior mean, so no other
linear function of the retained burdens carries additional treatment information.  This is a fixed
posterior-mean reference, not an infimum over encoders and not the learned rank-2 representation.
For every split we compute, by Monte Carlo over 10^6 fresh draws
(Gauss-Hermite with 40 nodes for the integral over U):

    D^2_S   = E[{P(A=1 | T_S, Z_S) - P(A=1 | Z_S)}^2]                residual treatment discrepancy
    bias_S  = tau_{Z_S} - tau = E_Z[E(h(U) | A=1, Z) - E(h(U) | A=0, Z)]

in two scenarios: "no_device" (block k is L_k; what the oracle-reader PROBE run sees) and "device_visible"
(block 5 is (L_5, I)).  Also the population targets of the comparators (naive, X only, X and all L, X and
U), and the working-model closed forms of section 3.6 (which ignore I and X) for later comparison.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.special import ndtr

from .phantom import PhantomConfig, outcome_shape

N_MC = 1_000_000
CHUNK = 100_000
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)
GH_W = GH_W / math.sqrt(2.0 * math.pi)
J = 5


def _draws(cfg: PhantomConfig, n: int, seed: int) -> dict:
    rng = np.random.default_rng([seed, 31])
    u = rng.standard_normal(n)
    L = u[:, None] + cfg.sigma * rng.standard_normal((n, J))
    i = (rng.random(n) < cfg.p_device).astype(float)
    x = rng.standard_normal(n)
    return {"U": u, "L": L, "I": i, "X": x}


def _posterior(L: np.ndarray, keep: tuple[int, ...], sigma: float) -> tuple[np.ndarray, float]:
    if not keep:
        return np.zeros(L.shape[0]), 1.0
    om = len(keep) / sigma ** 2
    v = 1.0 / (1.0 + om)
    return v * L[:, list(keep)].sum(1) / sigma ** 2, v


def _p_given_post(m, v, x, i, cfg: PhantomConfig) -> np.ndarray:
    """P(A=1 | U ~ N(m, v), X = x, I = i); i is an array (known) or None (averaged over the prior)."""
    s = math.sqrt(1.0 + cfg.treat_u ** 2 * v)
    lin = cfg.treat_u * m + cfg.treat_x * x
    if i is None:
        return (1 - cfg.p_device) * ndtr(lin / s) + cfg.p_device * ndtr((lin + cfg.treat_i) / s)
    return ndtr((lin + cfg.treat_i * i) / s)


def _arm_means(m, v, x, i, cfg: PhantomConfig) -> tuple[np.ndarray, np.ndarray]:
    """E(h(U) | A = 1, Z) and E(h(U) | A = 0, Z) with U | Z ~ N(m, v)."""
    u = m[:, None] + math.sqrt(v) * GH_X[None, :]
    lin = cfg.treat_u * u + cfg.treat_x * x[:, None]
    if i is None:
        pu = (1 - cfg.p_device) * ndtr(lin) + cfg.p_device * ndtr(lin + cfg.treat_i)
    else:
        pu = ndtr(lin + cfg.treat_i * i[:, None])
    h = outcome_shape(u, cfg)
    p1 = pu @ GH_W
    return (h * pu) @ GH_W / p1, (h * (1 - pu)) @ GH_W / (1 - p1)


def all_splits() -> list[tuple[int, ...]]:
    return [s for r in range(1, J) for s in itertools.combinations(range(J), r)]


def split_name(split) -> str:
    return "S" + "".join(str(j + 1) for j in split)


def population(cfg: PhantomConfig, n_mc: int = N_MC, seed: int = 99) -> dict:
    d = _draws(cfg, n_mc, seed)
    acc = {}

    def add(key, val):
        acc[key] = acc.get(key, 0.0) + float(val)

    for c0 in range(0, n_mc, CHUNK):
        sl = slice(c0, c0 + CHUNK)
        L, x, i = d["L"][sl], d["X"][sl], d["I"][sl]
        m_all, v_all = _posterior(L, tuple(range(J)), cfg.sigma)
        for split in all_splits():
            keep = tuple(j for j in range(J) if j not in split)
            m_k, v_k = _posterior(L, keep, cfg.sigma)
            name = split_name(split)
            # no_device: blocks are L_k; I is never seen
            p = _p_given_post(m_all, v_all, x, None, cfg)
            e = _p_given_post(m_k, v_k, x, None, cfg)
            add(("no_device", name, "D2"), np.sum((p - e) ** 2))
            a1, a0 = _arm_means(m_k, v_k, x, None, cfg)
            add(("no_device", name, "bias"), np.sum(a1 - a0))
            add(("no_device", name, "outside_05_95"), np.sum((e < 0.05) | (e > 0.95)))
            # device_visible: block 5 is (L_5, I); I is in Z when block 5 is retained, in T when held out
            i_in_z = 4 not in split
            p = _p_given_post(m_all, v_all, x, i, cfg)
            e = _p_given_post(m_k, v_k, x, i if i_in_z else None, cfg)
            add(("device_visible", name, "D2"), np.sum((p - e) ** 2))
            a1, a0 = _arm_means(m_k, v_k, x, i if i_in_z else None, cfg)
            add(("device_visible", name, "bias"), np.sum(a1 - a0))
        # comparator targets
        z0 = np.zeros(len(x))
        a1, a0 = _arm_means(z0, 1.0, x, None, cfg)
        add(("targets", "aipw_X", "bias"), np.sum(a1 - a0))
        a1, a0 = _arm_means(m_all, v_all, x, None, cfg)
        add(("targets", "aipw_X_allL", "bias"), np.sum(a1 - a0))
        a1, a0 = _arm_means(m_all, v_all, x, i, cfg)
        add(("targets", "aipw_X_allL_I", "bias"), np.sum(a1 - a0))
    out = {"n_mc": n_mc, "seed": seed, "tau": cfg.tau, "splits": {}, "targets": {},
           "feature_mode": "image_only" if cfg.image_only else "legacy_X",
           "representation_scope": "fixed true-L posterior-mean reference; not learned B or an encoder floor",
           "target_scope": "population ordinary-adjustment targets; finite fitted AIPW can differ",
           "severity_R2_ceiling": J / (J + cfg.sigma ** 2),
           "ceiling_scope": "prediction of U from L; not a bound on all ATE estimators"}
    for (scen, name, q), val in acc.items():
        if scen == "targets":
            out["targets"][name] = cfg.tau + val / n_mc
        else:
            out["splits"].setdefault(scen, {}).setdefault(name, {})[q] = val / n_mc
    # naive difference in means and (X, U) adjustment
    rng = np.random.default_rng([seed, 32])
    u, x, i = d["U"], d["X"], d["I"]
    a = (cfg.treat_u * u + cfg.treat_i * i + cfg.treat_x * x + rng.standard_normal(n_mc) > 0)
    y0 = outcome_shape(u, cfg) + cfg.out_x * x
    out["targets"]["naive"] = float(cfg.tau + y0[a].mean() - y0[~a].mean())
    out["targets"]["aipw_X_U"] = cfg.tau
    out["propensity"] = {"mean_P_A": float(a.mean()),
                         "frac_true_propensity_outside_05_95": float(np.mean(
                             (lambda pp: (pp < 0.05) | (pp > 0.95))(ndtr(cfg.treat_u * u + cfg.treat_i * i
                                                                         + cfg.treat_x * x))))}
    out["device_discrepancy_memo_approx"] = cfg.treat_i ** 2 * (1 / (2 * math.pi)) * cfg.p_device * (1 - cfg.p_device)
    out["omega_true_per_block"] = 1.0 / cfg.sigma ** 2
    out["closed_form_section_3_6"] = {split_name(s): closed_form_36(cfg, len(s)) for s in all_splits()
                                      if len(s) == 1}
    if cfg.image_only:
        aliases = {"aipw_X": "adjustment_no_feature", "aipw_X_allL": "adjustment_allL",
                   "aipw_X_allL_I": "adjustment_allL_I", "aipw_X_U": "adjustment_U"}
        out["targets"] = {aliases.get(k, k): v for k, v in out["targets"].items()}
    return out


def closed_form_36(cfg: PhantomConfig, size_s: int) -> dict:
    """Section 3.6 working model: probit A = 1{gamma U + eps_A > 0} (no I, no X), beta = 1, equal lambda = 1,
    sigma_k = sigma.  Posterior-mean reference D^2 (delta-method approximation) and the exact bias.
    This is neither the device phantom truth nor a global encoder lower bound."""
    g = cfg.treat_u
    om_k = 1.0 / cfg.sigma ** 2
    om_s, om_all = size_s * om_k, J * om_k
    om_rest = om_all - om_s
    v_rest = 1.0 / (1.0 + om_rest)
    v_all = 1.0 / (1.0 + om_all)
    s_rest2, s_all2 = 1.0 + g * g * v_rest, 1.0 + g * g * v_all
    c2 = g * g * (1.0 - v_rest) / s_rest2                      # Var(t_{-S}), t = gamma m_{-S} / s_{-S}
    e_phi2 = 1.0 / (2.0 * math.pi * math.sqrt(1.0 + 2.0 * c2))
    d2 = g * g / s_all2 * om_s / ((1.0 + om_all) * (1.0 + om_rest)) * e_phi2
    t = math.sqrt(c2) * GH_X
    phi = np.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
    ratio = float((phi / (ndtr(t) * ndtr(-t))) @ GH_W)
    bias = g * v_rest / math.sqrt(s_rest2) * ratio
    return {"D2_posterior_reference_approx": d2, "bias": bias, "omega_S": om_s, "omega_rest": om_rest,
            "note": "working model of section 3.6; ignores I and X, so it is a reference, not the phantom truth"}
