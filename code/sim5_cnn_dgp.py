"""Sim5 reference SCM: one MNIST image per unit, a fixed 200-state bank, 64 spatial blocks.

Specification: memo/2026-09-18-sim5-cnn-implementation-prd-v1.md, section 3.  The latent state is
U = (D, H), a digit and one of its first 20 training exemplars.  Only the generator and the evaluator
see U; the learner view is (X, W, A, Y).

Population quantities (E tau_U, P(A=1), the naive contrast, the X-adjustment target) are computed by
exact enumeration over the 200 states and Gauss-Hermite quadrature over the Gaussian covariates.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import norm

MNIST_PATH = Path(__file__).resolve().parents[1] / "materials" / "benchmarks" / "mnist" / "mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
PER_DIGIT = 20
N_STATES = 10 * PER_DIGIT
SIDE = 28
GRID = 8                                   # 8 x 8 = 64 blocks
C_OUTCOME = 4.2996324990                   # PRD value; population_summary() re-solves it
NOISE_HALF_WIDTH = 0.05
LATENT_KEYS = ("U_eval_only", "D_eval_only", "H_eval_only", "p_eval_only", "tau_u_eval_only")


@lru_cache(maxsize=1)
def bank() -> dict:
    """The first 20 training images of each digit, in source order, scaled to [0, 1]."""
    digest = hashlib.sha256(MNIST_PATH.read_bytes()).hexdigest()
    if digest != MNIST_SHA256:
        raise RuntimeError(f"MNIST digest {digest} differs from the frozen {MNIST_SHA256}")
    raw = np.load(MNIST_PATH)
    x, y = raw["x_train"], raw["y_train"]
    index = np.concatenate([np.flatnonzero(y == d)[:PER_DIGIT] for d in range(10)])
    templates = x[index].astype(np.float64) / 255.0
    return {"templates": templates, "digit": np.repeat(np.arange(10), PER_DIGIT),
            "exemplar": np.tile(np.arange(PER_DIGIT), 10), "source_index": index,
            "template_sha256": [hashlib.sha256(t.tobytes()).hexdigest() for t in x[index]],
            "mnist_sha256": digest}


@lru_cache(maxsize=1)
def states() -> dict:
    """Per-state severity r, style s, treatment index G and effect tau_u (PRD 3.2-3.3)."""
    b = bank()
    d = np.arange(10)
    f = d - 4.5 + 2.0 * np.sin(np.pi * d / 5.0)
    fc = f - f.mean()
    r_digit = fc / np.abs(fc).max()
    ink = b["templates"].mean(axis=(1, 2))
    ink_bar = ink.reshape(10, PER_DIGIT).mean(axis=1)
    dev = ink - ink_bar[b["digit"]]
    s = dev / np.abs(dev).max()
    r = r_digit[b["digit"]]
    g = 1.2 * r + 0.8 * s + 0.4 * r * s
    tau_u = 1.0 + 0.25 * r + 0.15 * s
    return {"r": r, "s": s, "G": g, "tau_u": tau_u, "r_digit": r_digit, "ink": ink}


@lru_cache(maxsize=1)
def block_masks() -> np.ndarray:
    """64 disjoint boolean masks, block k = (row piece k // 8, column piece k % 8)."""
    pieces = np.array_split(np.arange(SIDE), GRID)
    masks = np.zeros((GRID * GRID, SIDE, SIDE), dtype=bool)
    for i, rows in enumerate(pieces):
        for j, cols in enumerate(pieces):
            masks[GRID * i + j][np.ix_(rows, cols)] = True
    return masks


def propensity(g: np.ndarray, x1: np.ndarray, x3: np.ndarray) -> np.ndarray:
    return 0.1 + 0.8 * norm.cdf(g + 0.3 * x3 + 0.2 * np.tanh(x1))


def h_x(x1, x2, x3, x4):
    return 0.3 * np.tanh(x1) - 0.2 * np.tanh(x2) + 0.3 * x3 + 0.2 * np.tanh(x4)


def generate(n: int, seed: int, c: float = C_OUTCOME, images: bool = True) -> dict:
    """n i.i.d. units.  Keys ending in _eval_only are withheld from every learner.

    Each component draws from its own child stream, so images=False (for large population checks)
    leaves every other variable unchanged.
    """
    st, b = states(), bank()
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    u = streams[0].integers(0, N_STATES, size=n)
    e1, e2, e4 = streams[1].standard_normal((3, n))
    x3 = streams[1].choice(np.array([-1.0, 1.0]), size=n)
    r, s, g, tau = st["r"][u], st["s"][u], st["G"][u], st["tau_u"][u]
    x1, x2, x4 = 0.6 * r + e1, 0.6 * s + e2, np.tanh(r * s) + e4
    p = propensity(g, x1, x3)
    a = (streams[2].random(n) < p).astype(np.float64)
    ey = streams[3].standard_normal(n)
    y = a * tau - c * g + h_x(x1, x2, x3, x4) + (0.3 + 0.2 * np.abs(s)) * ey
    w = None
    if images:
        xi = streams[4].uniform(-NOISE_HALF_WIDTH, NOISE_HALF_WIDTH, size=(n, SIDE, SIDE))
        w = (0.1 + 0.8 * (b["templates"][u] + xi)).astype(np.float32)
    return {"X": np.column_stack([x1, x2, x3, x4]), "W": w, "A": a, "Y": y,
            "U_eval_only": u, "D_eval_only": b["digit"][u], "H_eval_only": b["exemplar"][u],
            "p_eval_only": p, "tau_u_eval_only": tau}


def learner_view(data: dict) -> dict:
    return {k: data[k] for k in ("X", "W", "A", "Y")}


def _gh(order: int):
    t, w = np.polynomial.hermite_e.hermegauss(order)
    return t, w / w.sum()


def population_summary(order: int = 24, c: float | None = None) -> dict:
    """E tau_U, P(A=1), the naive contrast, c solving naive = -1/2, and the X-adjustment target."""
    st = states()
    r, s, g, tau = st["r"], st["s"], st["G"], st["tau_u"]
    pi = np.full(N_STATES, 1.0 / N_STATES)
    t, wt = _gh(order)
    x3s = np.array([-1.0, 1.0])

    # P(A=1 | u) and E[(0.3 tanh X1 + 0.3 X3) e | u] by quadrature over eps1 and the two X3 values
    x1 = 0.6 * r[:, None] + t[None, :]
    p1_u = np.zeros(N_STATES); he1_u = np.zeros(N_STATES); he0_u = np.zeros(N_STATES)
    for x3 in x3s:
        e = propensity(g[:, None], x1, x3)
        p1_u += 0.5 * (e * wt).sum(1)
        part = 0.3 * np.tanh(x1) + 0.3 * x3
        he1_u += 0.5 * (part * e * wt).sum(1)
        he0_u += 0.5 * (part * (1 - e) * wt).sum(1)
    rest_u = (-0.2 * np.tanh(0.6 * s[:, None] + t[None, :]) * wt).sum(1) + \
             (0.2 * np.tanh(np.tanh(r * s)[:, None] + t[None, :]) * wt).sum(1)
    pa1 = pi @ p1_u
    ey1_free = pi @ (tau * p1_u + he1_u + rest_u * p1_u)
    ey0_free = pi @ (he0_u + rest_u * (1 - p1_u))
    k1, k0 = pi @ (g * p1_u), pi @ (g * (1 - p1_u))
    a0 = ey1_free / pa1 - ey0_free / (1 - pa1)
    a1 = k1 / pa1 - k0 / (1 - pa1)
    c_solved = (a0 + 0.5) / a1
    c_used = c_solved if c is None else c
    naive = a0 - c_used * a1

    # theta_X: outer expectation over each true state's covariates, posterior over all 200 states
    t3, w3 = t, wt
    e1, e2, e4 = np.meshgrid(t3, t3, t3, indexing="ij")
    wgrid = (w3[:, None, None] * w3[None, :, None] * w3[None, None, :]).ravel()
    e1, e2, e4 = e1.ravel(), e2.ravel(), e4.ravel()
    m1r, m2r, m4r = 0.6 * r, 0.6 * s, np.tanh(r * s)
    theta_x = 0.0
    for ustar in range(N_STATES):
        x1v, x2v, x4v = m1r[ustar] + e1, m2r[ustar] + e2, m4r[ustar] + e4
        loglik = -0.5 * ((x1v[:, None] - m1r[None]) ** 2 + (x2v[:, None] - m2r[None]) ** 2
                         + (x4v[:, None] - m4r[None]) ** 2)
        loglik -= loglik.max(axis=1, keepdims=True)
        lik = np.exp(loglik)
        for x3 in x3s:
            e = propensity(g[None, :], x1v[:, None], x3)
            w1 = lik * e; w1 /= w1.sum(1, keepdims=True)
            w0 = lik * (1 - e); w0 /= w0.sum(1, keepdims=True)
            diff = w1 @ tau - c_used * ((w1 - w0) @ g)
            theta_x += pi[ustar] * 0.5 * (diff * wgrid).sum()
    return {"order": order, "E_tau_u": float(pi @ tau), "P_A1": float(pa1), "c_solved": float(c_solved),
            "c_used": float(c_used), "naive": float(naive), "theta_X": float(theta_x),
            "E_r": float(pi @ r), "E_s": float(pi @ s)}
