#!/usr/bin/env python3
"""Structural causal models for the PROBE experiments.

Every SCM draws (U, X, W, A, Y) with a binary treatment A, outcome Y, observed covariates X in R^{d_x}, a proxy
W in R^{d_w} split into blocks, and an unobserved confounder U.  The population average treatment effect is
TRUE_ATE = 1 in every SCM (constant effect), so ATE and ATT coincide.

Common skeleton (all SCMs except s2 and s3 draw U ~ N(0,1); s3 draws a finite latent state):
    X = b_x U + eps_X                                  (d_x coordinates, unit-norm loading b_x)
    W_k = g_k(U) + block-specific noise                (blocks conditionally independent given U)
    P(A = 1 | U, X) = c + (1 - 2c) expit(u_treat U + 0.6 a_x'X)
    Y = TRUE_ATE * A + 0.7 y_x'X + 0.3 (yt_x'X)^2 + u_out U + eps_Y
The five main SCMs use the Simpson configuration u_treat = 1.2, u_out = -3 (s9: 1.5, -4) and c = 0.05, under
which the naive and the X-adjusted contrasts have the wrong sign while the oracle (X, U) adjustment recovers 1.

  s1  linear Gaussian proxy channel W = s*b_w*U + noise, nonlinear outcome.
  s2  nonlinear monotone proxy channel and nonlinear treatment/outcome (SCM2 of the earlier study).
  s3  finite latent state U in {0,...,K-1}; each block is a noisy K-bit one-hot code of U (bits flipped with
      probability p_flip).  The regime of Kuroki and Pearl (2014): finite U, discrete proxies.  A whole code
      block is needed as the held-out set: the channel matrix of one bit has rank 2 < K.
  s4  J scalar blocks W_k = g_k(U) + sigma*noise with injective monotone g_k; J is a parameter (used for the
      block-count sweep: exact balance is approached only as J grows).
  s5  partially measured latent U = (U1, U2): W measures U1 only, while A and Y depend on both, so no
      representation of (X, W) removes the U2 confounding and the held-out audit cannot see it.
  s6  s1 plus block 0 affecting treatment directly: held-out sets containing block 0 are invalid.
  s7  s1 plus block 3 affecting the outcome directly: all held-out sets remain valid for PROBE, but block 3
      is invalid as a proximal outcome proxy.
  s8  s1 with a steep treatment index and propensities in [0.02, 0.98]: limited overlap.
  s9  instrument-contaminated proxy: the coordinates of block 0 drive treatment directly and carry no
      information about U, while the other blocks are weak proxies of U.  Adjusting on all of W amplifies the
      residual confounding bias (bias amplification); block 0 is invalid as a held-out block.
  s10 many weak proxies: d_w coordinates with small loadings on U in n_blocks blocks.  The aggregate is
      informative but nuisance learners on the raw (X, W) struggle at moderate sample sizes.
  s11 decoder proxy: W_j = A2_j' tanh(a1 U + b1) + B_j N_j + sigma*eps_j with random decoder weights and
      block-specific nuisance factors N_j that affect W_j only.  With nuis_mode="block" (default) the block
      noise B_j N_j + sigma*eps_j is Gaussian and independent of (X, W_{-j}, U), so the block is an injective
      decoder with additive noise (Proposition 1(b)); nuis_mode="shared" reproduces the earlier construction
      W = A2' tanh(A1 [U, N] + b1) + noise with one nuisance vector shared by all blocks.
  s12 mixed-modality proxy: four blocks of block_size coordinates each; continuous linear (Prop. 1(b)),
      Poisson counts with a log-linear rate in U (Prop. 1(c)), binary codes with a logistic link (not covered:
      a finite-valued block of a continuous U is not complete on its own), and a nonlinear tanh channel
      with Gaussian noise (Prop. 1(b)).

Dimensions: every SCM accepts d_x (default 20) and d_w; block widths follow from d_w and n_blocks (s3: bits per
code k = d_w / n_blocks; s4: one coordinate per block; s12: four modality blocks of width d_w / 4).
All SCMs share the seed discipline of the pipeline: `generate(name, n, key, coef, **params)`.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

D_X = 20  # default covariate dimension (override with the d_x parameter)
TRUE_ATE = 1.0

SCM_NAMES = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12"]
MAIN_SCMS = ["s10", "s12", "s3", "s9", "s11"]  # paper order; the decoder SCM comes last

SIMPSON = dict(u_treat=1.2, u_out=-3.0, prop_floor=0.05)

DEFAULTS = {
    "s1": dict(d_w=10, n_blocks=5, proxy_strength=2.5),
    "s2": dict(d_w=10, n_blocks=5, proxy_strength=2.5),
    "s3": dict(k=8, n_blocks=10, p_flip=0.25, **SIMPSON),
    "s4": dict(n_blocks=8, sigma=1.0),
    "s5": dict(d_w=10, n_blocks=5, proxy_strength=2.5),
    "s6": dict(d_w=10, n_blocks=5, proxy_strength=2.5, treat_coef=0.9),
    "s7": dict(d_w=10, n_blocks=5, proxy_strength=2.5, outcome_coef=0.8),
    "s8": dict(d_w=10, n_blocks=5, proxy_strength=2.5, steepness=3.0),
    "s9": dict(d_w=10, n_blocks=5, proxy_strength=1.5, instrument_coef=2.0, u_treat=1.5, u_out=-4.0, prop_floor=0.05),
    "s10": dict(d_w=200, n_blocks=10, proxy_strength=0.35, **SIMPSON),
    "s11": dict(d_w=300, n_blocks=10, d_nuis=20, hidden=32, sigma=1.5, nuis_scale=2.0, nuis_mode="block", **SIMPSON),
    "s12": dict(d_w=160, n_blocks=4, proxy_strength=0.35, **SIMPSON),
}

DESCRIPTIONS = {
    "s1": "linear Gaussian proxy, nonlinear outcome",
    "s2": "nonlinear monotone proxy channel",
    "s3": "finite latent state, noisy one-hot code blocks (Kuroki-Pearl regime)",
    "s4": "J injective nonlinear scalar blocks (block-count sweep)",
    "s5": "partially measured latent: audit blind to U2",
    "s6": "block 0 affects treatment: invalid held-out block",
    "s7": "block 3 affects outcome: remaining block may affect Y",
    "s8": "steep treatment index: limited overlap",
    "s9": "instrument-contaminated proxy: block 0 drives A only, weak U-proxies elsewhere (bias amplification)",
    "s10": "many weak proxies: small loadings spread over many coordinates",
    "s11": "nonlinear decoder of U with block-specific nuisance factors: image-like proxy",
    "s12": "mixed-modality proxy: continuous, Poisson counts, binary codes, nonlinear channel",
}


def resolve(name: str, **params) -> dict:
    """Merge SCM defaults with overrides and derive the block layout from (d_w, n_blocks)."""
    if name not in DEFAULTS:
        raise ValueError(f"unknown SCM {name}")
    p = {**DEFAULTS[name], **params}
    p.setdefault("d_x", D_X)
    if name == "s3":
        if "d_w" in params:
            if p["d_w"] % p["n_blocks"]:
                raise ValueError("s3 needs d_w divisible by n_blocks (one K-bit code per block)")
            p["k"] = p["d_w"] // p["n_blocks"]
        p["d_w"] = p["k"] * p["n_blocks"]
    elif name == "s4":
        if "d_w" in params:
            p["n_blocks"] = p["d_w"]
        p["d_w"] = p["n_blocks"]
    elif name == "s12":
        if p["n_blocks"] != 4:
            raise ValueError("s12 has exactly four modality blocks")
        if "block_size" in params and "d_w" not in params:
            p["d_w"] = 4 * p["block_size"]
        if p["d_w"] % 4:
            raise ValueError("s12 needs d_w divisible by 4")
        p["block_size"] = p["d_w"] // 4
    else:
        if p["d_w"] % p["n_blocks"]:
            raise ValueError(f"{name} needs d_w divisible by n_blocks")
    for key in ("d_x", "d_w", "n_blocks"):
        p[key] = int(p[key])
    return p


def _rng(key: tuple[int, ...]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(key))


def _unit(g: np.random.Generator, d: int) -> np.ndarray:
    v = g.standard_normal(d)
    return v / np.linalg.norm(v)


def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def blocks(name: str, **params) -> list[np.ndarray]:
    p = resolve(name, **params)
    size = p["d_w"] // p["n_blocks"]
    return [np.arange(b * size, (b + 1) * size) for b in range(p["n_blocks"])]


def coefficients(name: str, coef_key: tuple[int, ...], **params) -> dict[str, np.ndarray]:
    """Structural coefficients drawn once per SCM from `coef_key` (shared across replicates)."""
    p = resolve(name, **params)
    d_x = p["d_x"]
    g = _rng(coef_key)
    c = {"b_x": _unit(g, d_x), "a_x": _unit(g, d_x), "y_x": _unit(g, d_x), "yt_x": _unit(g, d_x)}
    if name in ("s1", "s5", "s6", "s7", "s8", "s2", "s9", "s10"):
        d_w = p["d_w"]
        c["b_w"] = p["proxy_strength"] * g.choice(np.array([-1.0, 1.0]), size=d_w) * g.uniform(0.35, 1.0, size=d_w)
    if name == "s2":
        g2 = _rng(coef_key + (2,))
        c.update({"c_x": _unit(g2, d_x), "c_w": np.sign(c["b_w"]) * g2.uniform(0.15, 0.35, size=p["d_w"]),
                  "a2": _unit(g2, d_x), "a3": _unit(g2, d_x), "a4": _unit(g2, d_x), "y3": _unit(g2, d_x), "y4": _unit(g2, d_x)})
    if name == "s3":
        c["mu"] = np.linspace(-1.5, 1.5, p["k"])  # category values of the latent state
    if name == "s4":
        J = p["n_blocks"]
        c["alpha"] = g.uniform(0.8, 1.2, size=J) * g.choice(np.array([-1.0, 1.0]), size=J)
        c["beta"] = g.uniform(0.3, 0.8, size=J)
    if name == "s5":
        c["a_u2"], c["y_u2"] = 0.8, 1.5
    if name == "s11":
        g3 = _rng(coef_key + (3,))
        H, d_w, d_nuis = p["hidden"], p["d_w"], p["d_nuis"]
        # shared-nuisance decoder (earlier construction)
        c["dec_A1"] = g3.standard_normal((1 + d_nuis, H)) / math.sqrt(1 + d_nuis)
        c["dec_A1"][0, :] *= 2.0  # give U a strong share of the hidden units
        c["dec_b1"] = 0.3 * g3.standard_normal(H)
        c["dec_A2"] = g3.standard_normal((H, d_w)) / math.sqrt(H)
        # block-specific additive nuisance (default): first layer sees U only
        g5 = _rng(coef_key + (5,))
        c["dec_a1"] = g5.standard_normal(H)
        c["nuis_B"] = g5.standard_normal((p["n_blocks"], d_nuis, d_w // p["n_blocks"])) / math.sqrt(d_nuis)
    if name == "s12":
        g4 = _rng(coef_key + (4,)); m = p["block_size"]
        c["m_lin"] = p["proxy_strength"] * g4.choice(np.array([-1.0, 1.0]), size=m) * g4.uniform(0.6, 1.4, size=m)
        c["m_pois"] = 0.5 * g4.choice(np.array([-1.0, 1.0]), size=m) * g4.uniform(0.6, 1.4, size=m)
        c["m_bin"] = 1.2 * g4.choice(np.array([-1.0, 1.0]), size=m) * g4.uniform(0.6, 1.4, size=m)
        c["m_tanh"] = p["proxy_strength"] * g4.choice(np.array([-1.0, 1.0]), size=m) * g4.uniform(0.6, 1.4, size=m)
    return c


def invalid_heldout_blocks(name: str, **params) -> set[int]:
    """Blocks whose inclusion in the held-out set violates the common held-out channel condition (Assumption 2)."""
    return {0} if name in ("s6", "s9") else set()


def heldout_set_valid(name: str, S, **params) -> bool:
    """True when the held-out set S satisfies Assumption 2 and the sufficient conditions of Proposition 1.

    s6, s9: any set containing block 0 violates Assumption 2 (block 0 drives treatment).
    s12: the binary-code block alone (S = {2}) is not complete for a continuous U; any set that also contains a
         linear, Poisson, or tanh block is complete because a complete sub-block makes the whole set complete.
    """
    S = set(int(b) for b in S)
    if S & invalid_heldout_blocks(name, **params):
        return False
    if name == "s12" and S == {2}:
        return False
    return True


def oracle_adjustment(name: str, data: dict[str, np.ndarray]) -> np.ndarray:
    """Oracle adjustment variables (X plus the latent confounder) for the oracle baseline."""
    if name == "s5":
        return np.column_stack([data["X"], data["U"], data["U2"]])
    return np.column_stack([data["X"], data["U"]])


def decoder_mean(name: str, coef: dict[str, np.ndarray], u: np.ndarray, **params) -> np.ndarray:
    """Noise-free proxy mean E[W | U = u] for the SCMs whose channel is 'injective map plus independent noise'."""
    p = resolve(name, **params)
    u = np.asarray(u, dtype=float)
    if name == "s11" and p["nuis_mode"] == "block":
        return np.tanh(u[:, None] * coef["dec_a1"] + coef["dec_b1"]) @ coef["dec_A2"]
    if name in ("s1", "s5", "s6", "s7", "s8", "s9", "s10"):
        return u[:, None] * coef["b_w"]
    if name == "s4":
        return coef["alpha"][None, :] * u[:, None] + coef["beta"][None, :] * np.tanh(u)[:, None]
    raise ValueError(f"no closed-form decoder mean for {name}")


def check_orc(name: str, coef: dict[str, np.ndarray], **params) -> dict:
    """Numerical check of Proposition 1(ii) per block for the main SCMs.

    Linear and decoder blocks: minimum secant ratio min ||g(u) - g(u')|| / |u - u'| over a grid on [-4, 4]
    (positive means the noise-free block mean is injective on the grid).  s3: rank of the channel matrix of one
    K-bit code block (must equal K).  s12: per-modality status.
    """
    p = resolve(name, **params)
    blk = blocks(name, **params)
    out: dict = {"scm": name}
    if name == "s3":
        K, q = p["k"], p["p_flip"]
        codes = np.array(list(itertools.product([0, 1], repeat=K)), dtype=float)  # 2^K x K
        onehot = np.eye(K)
        mism = (codes[:, None, :] != onehot[None, :, :]).sum(2)  # 2^K x K mismatches
        channel = (q ** mism) * ((1 - q) ** (K - mism))
        out.update({"k": K, "p_flip": q, "channel_rank": int(np.linalg.matrix_rank(channel)), "single_bit_rank": 2 if K > 1 else 1})
        return out
    if name == "s11" and p["nuis_mode"] == "shared":
        out["status"] = "not covered: the shared nuisance vector makes the blocks dependent given (X, U), so Proposition 1(i) fails"
        return out
    if name == "s12":
        out["blocks"] = {"linear": "Prop 1(b), loading nonzero: %s" % bool(np.all(coef["m_lin"] != 0)),
                         "poisson": "Prop 1(c), rate map injective: %s" % bool(np.any(coef["m_pois"] != 0)),
                         "binary": "not covered (finite-valued block of a continuous U); valid only inside a larger set",
                         "tanh": "Prop 1(b), loading nonzero: %s" % bool(np.any(coef["m_tanh"] != 0))}
        return out
    grid = np.linspace(-4.0, 4.0, 401)
    G = decoder_mean(name, coef, grid, **params)
    ratios = {}
    for j, cols in enumerate(blk):
        if name in ("s6", "s9") and j == 0:
            ratios[j] = None
            continue
        gj = G[:, cols]
        diff = np.linalg.norm(gj[:, None, :] - gj[None, :, :], axis=2)
        du = np.abs(grid[:, None] - grid[None, :])
        mask = du > 1e-9
        ratios[j] = float((diff[mask] / du[mask]).min())
    out["min_secant_ratio_per_block"] = ratios
    return out


def generate(name: str, n: int, key: tuple[int, ...], coef: dict[str, np.ndarray], **params) -> dict[str, np.ndarray]:
    p = resolve(name, **params)
    d_x = p["d_x"]
    ru, rx, rw, ra, ry, r2 = [np.random.default_rng(c) for c in np.random.SeedSequence(key).spawn(6)]
    eps_x = rx.standard_normal((n, d_x)); tu = ra.uniform(size=n); eps_y = ry.standard_normal(n)
    blk = blocks(name, **params)
    out: dict[str, np.ndarray] = {}

    if name == "s3":
        k = p["k"]; u_cat = ru.integers(0, k, size=n); u = coef["mu"][u_cat]
        x = u[:, None] * coef["b_x"] + eps_x
        onehot = np.eye(k)[u_cat]
        w = np.concatenate([np.where(rw.uniform(size=(n, k)) < p["p_flip"], 1 - onehot, onehot) for _ in range(p["n_blocks"])], 1)
        index = p.get("u_treat", 0.8) * u + 0.6 * (x @ coef["a_x"])
        y0 = 0.7 * (x @ coef["y_x"]) + 0.3 * (x @ coef["yt_x"]) ** 2 + p.get("u_out", 1.5) * u + eps_y
        out["U_cat"] = u_cat
    elif name == "s4":
        u = ru.standard_normal(n)
        x = u[:, None] * coef["b_x"] + eps_x
        J = p["n_blocks"]; eps_w = rw.standard_normal((n, J))
        w = coef["alpha"][None, :] * u[:, None] + coef["beta"][None, :] * np.tanh(u)[:, None] + p["sigma"] * eps_w
        index = 0.8 * u + 0.6 * (x @ coef["a_x"])
        y0 = 0.7 * (x @ coef["y_x"]) + 0.3 * (x @ coef["yt_x"]) ** 2 + 1.5 * u + eps_y
    elif name == "s11":
        u = ru.standard_normal(n)
        x = u[:, None] * coef["b_x"] + eps_x
        if p["nuis_mode"] == "shared":
            nuis = r2.standard_normal((n, p["d_nuis"]))
            hid = np.tanh(np.column_stack([u, nuis]) @ coef["dec_A1"] + coef["dec_b1"])
            w = hid @ coef["dec_A2"] + p["sigma"] * rw.standard_normal((n, p["d_w"]))
            out["N"] = nuis
        else:  # block-specific additive nuisance factors: W_j = f_j(U) + B_j N_j + sigma eps_j
            w = decoder_mean(name, coef, u, **params) + p["sigma"] * rw.standard_normal((n, p["d_w"]))
            nuis = r2.standard_normal((n, p["n_blocks"], p["d_nuis"]))
            for j, cols in enumerate(blk):
                w[:, cols] += p["nuis_scale"] * (nuis[:, j, :] @ coef["nuis_B"][j])
            out["N"] = nuis.reshape(n, -1)
        index = p["u_treat"] * u + 0.6 * (x @ coef["a_x"])
        y0 = 0.7 * (x @ coef["y_x"]) + 0.3 * (x @ coef["yt_x"]) ** 2 + p["u_out"] * u + eps_y
    elif name == "s12":
        u = ru.standard_normal(n); m = p["block_size"]
        x = u[:, None] * coef["b_x"] + eps_x
        w_lin = u[:, None] * coef["m_lin"] + rw.standard_normal((n, m))
        w_pois = r2.poisson(np.exp(0.3 + u[:, None] * coef["m_pois"])).astype(float)
        w_bin = (rw.uniform(size=(n, m)) < expit(u[:, None] * coef["m_bin"])).astype(float)
        w_tanh = np.tanh(u[:, None] * coef["m_tanh"]) + 0.5 * rw.standard_normal((n, m))
        w = np.column_stack([w_lin, w_pois, w_bin, w_tanh])
        index = p["u_treat"] * u + 0.6 * (x @ coef["a_x"])
        y0 = 0.7 * (x @ coef["y_x"]) + 0.3 * (x @ coef["yt_x"]) ** 2 + p["u_out"] * u + eps_y
    elif name == "s2":
        u = ru.standard_normal(n); eps_w = rw.standard_normal((n, p["d_w"]))
        x = u[:, None] * coef["b_x"] + 0.35 * ((u ** 2 - 1.0) / math.sqrt(2.0))[:, None] * coef["c_x"] + eps_x
        w = u[:, None] * coef["b_w"] + np.tanh(u)[:, None] * coef["c_w"] + eps_w
        index = 0.75 * u + 0.45 * np.tanh(x @ coef["a_x"]) + 0.25 * np.sin(x @ coef["a2"]) + 0.20 * (x @ coef["a3"]) * (x @ coef["a4"])
        y0 = 0.5 * np.sin(x @ coef["y_x"]) + 0.25 * (x @ coef["yt_x"]) ** 2 + 0.30 * np.tanh(x @ coef["y3"]) * (x @ coef["y4"]) + 1.2 * u + 0.4 * (u ** 2 - 1.0) + eps_y
    else:  # s1, s5, s6, s7, s8, s9, s10 share the linear-proxy skeleton
        u = ru.standard_normal(n); eps_w = rw.standard_normal((n, p["d_w"]))
        x = u[:, None] * coef["b_x"] + eps_x
        w = u[:, None] * coef["b_w"] + eps_w
        u_treat, u_out = p.get("u_treat", 0.8), p.get("u_out", 1.5)
        index = u_treat * u + 0.6 * (x @ coef["a_x"])
        y0 = 0.7 * (x @ coef["y_x"]) + 0.3 * (x @ coef["yt_x"]) ** 2 + u_out * u + eps_y
        if name == "s9":  # block 0 = pure treatment drivers (no U information)
            w[:, blk[0]] = r2.standard_normal((n, len(blk[0])))
            index = index + p["instrument_coef"] * w[:, blk[0]].sum(1)
        if name == "s5":
            u2 = r2.standard_normal(n)
            index = index + coef["a_u2"] * u2
            y0 = y0 + coef["y_u2"] * u2
            out["U2"] = u2
        if name == "s6":
            index = index + p["treat_coef"] * w[:, blk[0]].mean(1)
        if name == "s7":  # sign-aligned so the direct effect reinforces rather than cancels the U effect
            y0 = y0 + p["outcome_coef"] * (np.sign(coef["b_w"])[None, blk[3]] * w[:, blk[3]]).mean(1)
        if name == "s8":
            index = p["steepness"] * index

    floor = p.get("prop_floor", 0.02 if name == "s8" else 0.1)
    propensity = floor + (1 - 2 * floor) * expit(index)
    a = (tu < propensity).astype(float)
    out.update({"U": u, "X": x, "W": w, "A": a, "Y": y0 + TRUE_ATE * a, "propensity": propensity})
    return out
