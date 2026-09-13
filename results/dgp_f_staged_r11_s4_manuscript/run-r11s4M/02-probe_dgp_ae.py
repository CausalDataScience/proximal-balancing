#!/usr/bin/env python3
"""Data-generating processes A to E of the fixed experiment plan.

Specification: memo/2026-09-11-dgp-a-e-experiment-plan.md, Sections 2 to 7.  Nothing here is tunable at run
time except the sample size, the replicate index and the phase; every structural constant is fixed by the plan.

Common structure (plan 2.2 and 2.3).  For each unit,

    U, eps_C, I1, I2, eps_Y ~ N(0,1) independent,   eps_X ~ N(0, I_20),
    C   = 0.5 U + sqrt(0.75) eps_C                       (Var C = 1, Corr(U,C) = 0.5)
    X   = b_x U + eps_X
    g(X)= 0.7 y_x'X + 0.3 (yt_x'X)^2
    e   = 0.05 + 0.9 expit(1.5 U + 0.6 a_x'X + 2 (I1 + I2) + 0.6 C)      (A, C, D, E)
    A   ~ Bernoulli(e)
    Y(a)= a + g(X) - 4 U - 0.8 C + eps_Y                                  (A, C, D, E)

DGP B replaces the treatment index and the outcome by its own polynomial versions (plan 4.3).

Block 0 is (I1, I2, C) in every DGP: two direct causes of treatment that do not enter the outcome, and the
observed baseline severity, which is correlated with the hidden U and is a cause of both A and Y.  Block 0 is
never pruned by the main algorithm; knowing its identity is a privileged diagnostic only (plan 2.4).

A generated sample is a dict with
    U, C, I1, I2, X, A, Y, Y0, Y1, propensity  and  blocks, a list of five block dicts.
Each block dict has kind "tabular" (key v), "image" (key img) or "image_numeric" (keys img and num).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

ROOT_SEED = 190909
NAMESPACE = {"coefficient": 1, "data": 2, "unit_split": 5, "encoder": 6, "critic": 7,
             "nuisance": 8, "mnist_pool": 10, "multiplier": 11, "subset_ablation": 12}
DGP_ID = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
PHASE_ID = {"pilot": 0, "confirmatory": 1}
TRUE_ATE = 1.0
D_X = 20
N_BLOCKS = 5
MNIST_PATH = Path(__file__).resolve().parents[1] / "materials" / "benchmarks" / "mnist" / "mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
MNIST_BYTES = 11490434

B_GOOD_EXPECTED = np.array([
    1.1489038010649661, 1.7705451453361223, -1.4216890020446680, -0.7855101753703675,
    -1.5159414732646240, -1.6443659023490418, 1.3794377203214340, -1.2844937715884432])

# DGP B (plan 4.1), DGP C (plan 5.1)
B_BETA = np.array([0.15, 0.25, 0.35, 0.45])
B_DELTA = np.array([-0.6, -0.2, 0.2, 0.6])
C_LAMBDA = np.array([0.35, -0.35, 0.50, -0.50])
C_KAPPA = np.array([0.50, 0.50, -0.35, -0.35])
E_CORRUPT = 0.25


def seed_key(namespace: str, *parts: int) -> tuple[int, ...]:
    return (ROOT_SEED, NAMESPACE[namespace], *parts)


def run_key(namespace: str, phase: str, dgp: str, n: int, replicate: int, rotation: int = 0,
            subset_code: int = 0, restart: int = 0) -> tuple[int, ...]:
    """Plan 15.1: SeedSequence((root, namespace, phase, dgp, n, replicate, rotation, subset, restart))."""
    return (ROOT_SEED, NAMESPACE[namespace], PHASE_ID[phase], DGP_ID[dgp], int(n), int(replicate),
            int(rotation), int(subset_code), int(restart))


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _unit(v):
    return v / np.linalg.norm(v)


def coefficients() -> dict:
    """Plan 2.1.  Drawn once, independent of phase, DGP, sample size and replicate."""
    g = np.random.default_rng(np.random.SeedSequence(seed_key("coefficient")))
    c = {k: _unit(g.standard_normal(D_X)) for k in ("b_x", "a_x", "y_x", "yt_x")}
    b_w = 2.0 * g.choice(np.array([-1.0, 1.0]), size=10) * g.uniform(0.35, 1.0, size=10)
    c["b_w"] = b_w
    c["b_good"] = b_w[2:]
    if not np.allclose(c["b_good"], B_GOOD_EXPECTED, atol=1e-12):
        raise AssertionError("b_good does not match the value fixed in plan 2.1")
    for k in ("b_x", "a_x", "y_x", "yt_x"):
        if abs(np.linalg.norm(c[k]) - 1.0) > 1e-12:
            raise AssertionError(f"{k} is not a unit vector")
    return c


# DGP F (design of 2026-09-11, second review).  X is scalar here and every block is three-dimensional.
F_X_NOISE = 2.0
F_INDEX = {"U": 1.0, "I": 3.0, "X": 0.2, "C": 0.3}
F_OUTCOME = {"U": -4.0, "X": 0.2, "C": -0.5}
F_C_NOISE = 0.35


def f_balance_coefficient(k: int) -> float:
    """The exact balancing coefficient of design F: the smaller root of r^2 - 3 r + 1/k = 0, where k is the
    number of latent measurements left in the representation input.  Used only to verify the design and to
    build the verification representation; it is never given to a learner."""
    if k < 1:
        raise ValueError("design F always keeps at least one measurement when block 0 feeds the representation")
    return (3.0 - math.sqrt(9.0 - 4.0 / k)) / 2.0


def f_exact_representation(data: dict, S) -> np.ndarray:
    """Z = (X, C, mean of the available latent measurements + r* I) for design F.

    Requires block 0 in the representation input, since I and M_0 live there.  Returns None-free arrays only
    for feasible configurations; callers check `balance_feasible` first.
    """
    S = set(int(b) for b in S)
    if 0 in S:
        raise ValueError("design F needs block 0 in the representation input for the exact representation")
    kept = [0] + [j for j in range(1, 5) if j not in S]
    m = np.column_stack([data["M"][:, j] for j in kept]).mean(1)
    r = f_balance_coefficient(len(kept))
    return np.column_stack([data["X"][:, 0], data["C"], m + r * data["I1"]]).astype(np.float32)


def _h2(u):
    return (u ** 2 - 1.0) / math.sqrt(2.0)


def _h3(u):
    return (u ** 3 - 3.0 * u) / math.sqrt(6.0)


# ----------------------------------------------------------------------------- DGP D templates
def d_templates() -> np.ndarray:
    """Plan 6.1.  Three orthonormal 32x32 templates per good block, from Gaussian bumps by Gram-Schmidt."""
    p = np.arange(32)
    xs = 2.0 * (p + 0.5) / 32.0 - 1.0
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    grid = np.stack([gx, gy], axis=-1)                      # (32, 32, 2), index [p, q]
    out = np.zeros((4, 3, 32, 32))
    s_t = 0.18
    for j in range(1, 5):
        alpha = j * math.pi / 10.0
        c1 = 0.35 * np.array([math.cos(alpha), math.sin(alpha)])
        centers = [c1, -c1, 0.35 * np.array([-math.sin(alpha), math.cos(alpha)])]
        raw = [np.exp(-((grid - c) ** 2).sum(-1) / (2.0 * s_t ** 2)) for c in centers]
        basis = []
        for v in raw:                                       # modified Gram-Schmidt
            w = v.copy()
            for b in basis:
                w = w - (b * w).sum() * b
            w = w / np.linalg.norm(w)
            basis.append(w)
        out[j - 1] = np.stack(basis)
    gram = np.einsum("jrpq,jspq->jrs", out, out)
    if np.abs(gram - np.eye(3)[None]).max() >= 1e-10:
        raise AssertionError("DGP D templates are not orthonormal to 1e-10")
    return out


# ----------------------------------------------------------------------------- MNIST pool
_MNIST_POOL_CACHE: dict | None = None


def mnist_pool() -> dict:
    """Plan 7.2.  1000 images per class from the MNIST training split, chosen once by a seeded permutation of
    the ascending original indices.  Images are converted to float32 in [0,1] and not augmented."""
    global _MNIST_POOL_CACHE
    if _MNIST_POOL_CACHE is not None:
        return _MNIST_POOL_CACHE
    raw = MNIST_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != MNIST_SHA256 or len(raw) != MNIST_BYTES:
        raise AssertionError("MNIST source file does not match the manifest digest fixed in plan 7.2")
    d = np.load(MNIST_PATH)
    x, y = d["x_train"], d["y_train"].astype(int)
    g = np.random.default_rng(np.random.SeedSequence(seed_key("mnist_pool")))
    images, indices = [], []
    for k in range(10):
        idx_k = np.sort(np.where(y == k)[0])
        pick = idx_k[g.permutation(len(idx_k))[:1000]]
        indices.append(pick)
        images.append((x[pick].astype(np.float32) / 255.0))
    pool = {"images": np.stack(images), "indices": np.stack(indices)}       # (10, 1000, 28, 28)
    flat = pool["images"].reshape(10 * 1000, -1)
    order = np.lexsort(flat.T)
    srt = flat[order]
    dup = np.where((srt[1:] == srt[:-1]).all(1))[0]
    collisions = []
    for i in dup:
        a, b = order[i], order[i + 1]
        if a // 1000 != b // 1000:
            collisions.append((int(a), int(b)))
    pool["cross_class_duplicates"] = collisions
    pool["digest"] = hashlib.sha256(pool["images"].tobytes()).hexdigest()
    _MNIST_POOL_CACHE = pool
    return pool


def e_channel_matrix() -> np.ndarray:
    """Plan 7.5/14.2.  The 10 x 10 corrupted-label channel at q = 0.25."""
    m = np.full((10, 10), E_CORRUPT / 9.0)
    np.fill_diagonal(m, 1.0 - E_CORRUPT)
    return m


# ----------------------------------------------------------------------------- generation
def _generate_f(n: int, ss, coef: dict) -> dict:
    """Design F: a scalar covariate, five three-dimensional blocks, and a representation that balances exactly.

        U, I, C, eps_X, e_0..e_4, zeta_j, N_j, eps_Y  independent standard normals
        X   = U + 2 eps_X                                   (scalar)
        M_j = U + e_j,  j = 0..4
        W_0 = (I, C, M_0)
        W_j = (M_j, C + 0.35 zeta_j, N_j),  j = 1..4
        A   ~ Bernoulli[0.05 + 0.9 expit(U + 3 I + 0.2 X + 0.3 C)]
        Y(a)= a - 4 U + 0.2 X - 0.5 C + eps_Y

    Block 0 carries the treatment-only cause I together with the observed common cause C and one measurement,
    so it must feed the representation rather than be audited.
    """
    ru, ri, rc, rx, rm, rn, ry, ra = [np.random.default_rng(s) for s in ss.spawn(8)]
    u, i_cause, c_sev = ru.standard_normal(n), ri.standard_normal(n), rc.standard_normal(n)
    x = (u + F_X_NOISE * rx.standard_normal(n))[:, None]
    m = u[:, None] + rm.standard_normal((n, 5))
    index = F_INDEX["U"] * u + F_INDEX["I"] * i_cause + F_INDEX["X"] * x[:, 0] + F_INDEX["C"] * c_sev
    propensity = 0.05 + 0.9 * expit(index)
    a = (ra.uniform(size=n) < propensity).astype(np.float64)
    y0 = F_OUTCOME["U"] * u + F_OUTCOME["X"] * x[:, 0] + F_OUTCOME["C"] * c_sev + ry.standard_normal(n)
    blocks = [{"kind": "tabular", "name": "W0_cause_severity_measurement",
               "v": np.column_stack([i_cause, c_sev, m[:, 0]]).astype(np.float32)}]
    for j in range(1, 5):
        blocks.append({"kind": "tabular", "name": f"W{j}",
                       "v": np.column_stack([m[:, j], c_sev + F_C_NOISE * rn.standard_normal(n),
                                             rn.standard_normal(n)]).astype(np.float32)})
    return {"dgp": "F", "n": n, "U": u, "C": c_sev, "I1": i_cause, "I2": np.zeros(n), "M": m,
            "X": x.astype(np.float32), "A": a, "Y": np.where(a == 1.0, y0 + TRUE_ATE, y0),
            "Y0": y0, "Y1": y0 + TRUE_ATE, "propensity": propensity, "blocks": blocks}


def generate(dgp: str, n: int, replicate: int, phase: str = "confirmatory", coef: dict | None = None) -> dict:
    if dgp not in DGP_ID:
        raise ValueError(f"unknown DGP {dgp}")
    coef = coef or coefficients()
    ss = np.random.SeedSequence(run_key("data", phase, dgp, n, replicate))
    if dgp == "F":
        out = _generate_f(n, ss, coef)
        out.update({"replicate": replicate, "phase": phase})
        return out
    (ru, rc, rx, ri, ry, ra, rw, rk) = [np.random.default_rng(s) for s in ss.spawn(8)]

    if dgp == "E":
        k_class = rk.integers(0, 10, size=n)
        u = (k_class - 4.5) / math.sqrt(8.25)
    else:
        k_class = None
        u = ru.standard_normal(n)
    c_sev = 0.5 * u + math.sqrt(0.75) * rc.standard_normal(n)
    x = u[:, None] * coef["b_x"] + rx.standard_normal((n, D_X))
    i1, i2 = ri.standard_normal(n), ri.standard_normal(n)
    gx = 0.7 * (x @ coef["y_x"]) + 0.3 * (x @ coef["yt_x"]) ** 2

    if dgp == "B":                                            # plan 4.3
        index = 1.4 * u + 0.5 * _h2(u) + 0.6 * (x @ coef["a_x"]) + 2.0 * (i1 + i2) + 0.6 * c_sev
        y0 = gx - 3.5 * u - 0.8 * _h2(u) - 0.8 * c_sev + ry.standard_normal(n)
    else:                                                     # plan 2.3
        index = 1.5 * u + 0.6 * (x @ coef["a_x"]) + 2.0 * (i1 + i2) + 0.6 * c_sev
        y0 = gx - 4.0 * u - 0.8 * c_sev + ry.standard_normal(n)
    propensity = 0.05 + 0.9 * expit(index)
    a = (ra.uniform(size=n) < propensity).astype(np.float64)
    y1 = y0 + TRUE_ATE

    bg = coef["b_good"]
    blocks: list[dict] = [{"kind": "tabular", "v": np.column_stack([i1, i2, c_sev]).astype(np.float32),
                           "name": "W0_instruments_and_severity"}]
    if dgp == "A":                                            # plan 3.1
        for j in range(4):
            e = rw.standard_normal((n, 3))
            v = np.column_stack([bg[2 * j] * u + e[:, 0], bg[2 * j + 1] * u + e[:, 1], c_sev + 0.35 * e[:, 2]])
            blocks.append({"kind": "tabular", "v": v.astype(np.float32), "name": f"W{j+1}"})
    elif dgp == "B":                                          # plan 4.2
        for j in range(4):
            e = rw.standard_normal((n, 3))
            beta, delta = B_BETA[j], B_DELTA[j]
            c1 = 1.4 * (u + beta * _h3(u)) / math.sqrt(1.0 + beta ** 2) + e[:, 0]
            c2 = 1.4 * (_h2(u) + delta * u) / math.sqrt(1.0 + delta ** 2) + e[:, 1]
            blocks.append({"kind": "tabular", "name": f"W{j+1}",
                           "v": np.column_stack([c1, c2, c_sev + 0.35 * e[:, 2]]).astype(np.float32)})
    elif dgp == "C":                                          # plan 5.1
        rate = 4.0 * np.exp(u[:, None] * C_LAMBDA[None, :] + c_sev[:, None] * C_KAPPA[None, :])
        for j in range(4):
            counts = rw.poisson(rate).astype(np.float32)
            blocks.append({"kind": "tabular", "v": counts, "name": f"W{j+1}", "counts": True})
    elif dgp == "D":                                          # plan 6.2
        tmpl = d_templates()
        for j in range(4):
            e = rw.standard_normal((n, 3))
            h = np.stack([bg[2 * j] * u + e[:, 0], bg[2 * j + 1] * u + e[:, 1], c_sev + 0.35 * e[:, 2]], 1)
            img = 0.7 * np.einsum("nr,rpq->npq", h, tmpl[j]) + 0.1 * rw.standard_normal((n, 32, 32))
            blocks.append({"kind": "image", "img": img.astype(np.float32)[:, None], "name": f"W{j+1}",
                           "payload": h.astype(np.float32)})
    else:                                                     # plan 7.3
        pool = mnist_pool()
        for j in range(4):
            keep = rk.uniform(size=n) >= E_CORRUPT
            other = rk.integers(0, 9, size=n)
            k_tilde = np.where(keep, k_class, (k_class + 1 + other) % 10)
            pick = rk.integers(0, 1000, size=n)
            img = pool["images"][k_tilde, pick]
            num = (c_sev + 0.35 * rw.standard_normal(n)).astype(np.float32)[:, None]
            blocks.append({"kind": "image_numeric", "img": img[:, None], "num": num, "name": f"W{j+1}",
                           "corrupted_class": k_tilde.astype(np.int64)})

    out = {"dgp": dgp, "n": n, "replicate": replicate, "phase": phase, "U": u, "C": c_sev, "I1": i1, "I2": i2,
           "X": x.astype(np.float32), "A": a, "Y": np.where(a == 1.0, y1, y0), "Y0": y0, "Y1": y1,
           "propensity": propensity, "blocks": blocks}
    if k_class is not None:
        out["K"] = k_class
    return out


# ----------------------------------------------------------------------------- adjustment sets
def block_matrix(block: dict) -> np.ndarray:
    """Flatten one block into a matrix, for the tabular raw baseline and for provenance."""
    if block["kind"] == "tabular":
        return block["v"]
    if block["kind"] == "image":
        return block["img"].reshape(len(block["img"]), -1)
    return np.column_stack([block["img"].reshape(len(block["img"]), -1), block["num"]])


def raw_matrix(data: dict) -> np.ndarray:
    return np.column_stack([data["X"]] + [block_matrix(b) for b in data["blocks"]])


def oracle_matrix(data: dict) -> np.ndarray:
    return np.column_stack([data["X"], data["U"], data["C"]])


def xc_matrix(data: dict) -> np.ndarray:
    return np.column_stack([data["X"], data["C"]])


def balance_feasible(dgp: str, S) -> bool:
    """Can any representation built from the remaining blocks balance the held-out set exactly?

    Holding out block 0 always fails, since that block carries a direct cause of treatment.  In DGP A to E a
    configuration that also holds out every measurement block leaves the representation with no information
    about the latent, so no function of the remaining input can balance the held-out measurements either.
    This is stricter than `structural_valid` and is the right target for scoring the screen.
    """
    S = set(int(b) for b in S)
    if 0 in S:
        return False
    if dgp == "F":
        # Block 0 of design F carries the measurement M_0 as well as I and C, so the representation input
        # always keeps at least one latent measurement and the balancing coefficient exists for k >= 1.
        return True
    # In designs A to E block 0 holds no latent measurement, so holding out every measurement block leaves the
    # representation with nothing to balance with.
    return bool({1, 2, 3, 4} - S)


def structural_valid(S) -> bool:
    """A held-out set is structurally valid when it does not contain block 0 (plan 2.4, 8.7).

    Block 0 holds the two instruments and the observed severity, so holding it out violates the common
    held-out channel condition.  Used only to score the screen, never by the algorithm.
    """
    return 0 not in set(int(b) for b in S)


def all_subsets() -> list[tuple[int, ...]]:
    """Plan 8.1: the 30 nonempty proper subsets of {0,...,4}, enumerated in a fixed order."""
    out = []
    for code in range(1, 2 ** N_BLOCKS - 1):
        out.append(tuple(b for b in range(N_BLOCKS) if code >> b & 1))
    return out


def subset_code(S) -> int:
    code = 0
    for b in S:
        code |= 1 << int(b)
    return code


def preprocess_counts(train_v: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Plan 5.2: log1p then standardize with statistics from the training split."""
    lt = np.log1p(train_v)
    mu, sd = lt.mean(0), np.maximum(lt.std(0), 1e-6)
    return ((np.log1p(v) - mu) / sd).astype(np.float32)
