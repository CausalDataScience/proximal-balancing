"""The KSPC benchmark (Xu and Gretton, arXiv 2308.04585), adapted to a binary treatment so that PROBE and
the AIPW comparators can run on it.  Fixed before any run, 2026-09-22 22:30 CDT.

From the authors' generator (`src/data/generate_synthetic.py`, commit a5283d62), unchanged:
    U ~ Unif(-1, 1)
    W1 = exp(U) + N(0, 0.1^2)                         their low-dimensional proxy
    W2 = an MNIST image of the digit floor(5U + 5)    their high-dimensional proxy (U = 1 gives 10, set to 9)
    Y  = sin(pi U / 2) + A^2 - 0.3                    deterministic, their identifying condition
Changed, and labelled everywhere as an adaptation:
    A  ~ Bernoulli(0.1 + 0.8 (1 + erf(U)) / 2)        their index erf(U), made binary with overlap [0.16, 0.84]
With a binary A, A^2 = A, so the average effect is exactly f(1) - f(0) = 1.
    No covariates, as in the paper.  n = 3,000 per replicate, replicates 1 to 20, seed 99_300_000 + r.
    MNIST images are the local training set `materials/benchmarks/mnist/mnist.npz`, pixels scaled to [0, 1];
    each unit draws one training image of its digit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import erf

HERE = Path(__file__).resolve().parent
MNIST = HERE.parent / "materials" / "benchmarks" / "mnist" / "mnist.npz"
N = 3_000
REPLICATES = tuple(range(1, 21))
SEED = 99_300_000
TAU = 1.0

_images: dict[int, np.ndarray] | None = None


def mnist_by_digit() -> dict[int, np.ndarray]:
    global _images
    if _images is None:
        d = np.load(MNIST)
        x = d["x_train"].reshape(len(d["x_train"]), -1).astype(np.float32) / 255.0
        _images = {k: x[d["y_train"] == k] for k in range(10)}
    return _images


def generate(r: int) -> dict:
    rng = np.random.default_rng(SEED + r)
    u = rng.uniform(-1.0, 1.0, N)
    p = 0.1 + 0.8 * (1.0 + erf(u)) / 2.0
    a = (rng.uniform(size=N) < p).astype(float)
    y = np.sin(np.pi * u / 2.0) + a - 0.3
    w1 = (np.exp(u) + rng.normal(0.0, 0.1, N))[:, None]
    digits = np.minimum(np.floor(5 * u + 5).astype(int), 9)
    pool = mnist_by_digit()
    w2 = np.stack([pool[k][rng.integers(len(pool[k]))] for k in digits]).astype(float)
    return {"X": np.zeros((N, 0)), "W1": w1, "W2": w2, "A": a, "Y": y, "U_eval_only": u,
            "propensity": p, "tau": TAU, "digits": digits}
