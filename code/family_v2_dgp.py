"""SCM family v2: the role-blind scalar and vector family (PRD memo/2026-09-18-family-v2-role-blind-dimension-prd-v1.md).

The learner sees (X, W, A, Y) and nothing else.  No coordinate carries a name.  The latent confounder U is a
scalar, and every observed block is a fixed random rotation of its informative coordinates stacked on pure
noise coordinates, so a coordinate of W_5 is neither "the treatment cause" nor "a measurement": it is a mix.

Informative coordinates (never shown to a learner):

    X_1 = U + 2 eta                       C                         (both inside X)
    M_j = U + 0.85 eps_j,  j = 0..3       M_4 = U - 3.0 I + 0.85 eps_4
    W_1, W_2, W_3 carry M_1, M_2, M_3     W_4 carries M_4           W_5 carries (M_0, I)

    P(A = 1 | U, I, C, X_1) = 0.1 + 0.8 Phi(1.5 U + 2.0 I + 0.2 X_1 + 0.3 C)
    Y(a) = a - 2.5 U + 0.2 X_1 - 0.5 C + 0.3 eps_Y,       so tau = 1

U, I, C, eta and every eps are independent standard normals; every pure noise coordinate is N(0, 0.5^2).
The observed confounder C sits in X, and X always enters the representation, so an outcome-relevant
coordinate can never be dropped by a representation that balances the held-out block (Revision 1.1 of the
PRD records why C left W_5).  The three levels share every coefficient and differ only in how many noise
coordinates hide the informative ones, so their population values and split census are the same objects.
The coefficients are PRD v2's set B (Revision R4): the treatment index weights U and I at 1.5 and 2.0 and
the contaminated block carries -3.0 I, so that every split mixing a clean block with W_4 has a population
imbalance of at least 1.0e-3 while all seven clean-only splits still admit exact balance.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

KAPPA = 2.0
PROXY_SD = 0.85
BAD_I = -3.0
TREAT = {"U": 1.5, "I": 2.0, "X1": 0.2, "C": 0.3}
OUTCOME = {"U": -2.5, "X1": 0.2, "C": -0.5}
OUTCOME_NOISE = 0.3
NOISE_SD = 0.5
TAU = 1.0
N_BLOCKS = 5
BLOCK_INFO = (("M1",), ("M2",), ("M3",), ("M4",), ("M0", "I"))   # informative content of W_1 .. W_5
X_INFO = ("X1", "C")
ROTATION_SEED = 97_000_000
DEV_SEED = 97_100_000
CONFIRM_SEED = 97_200_000
TEST_SEED = 97_800_000


@dataclass(frozen=True)
class Level:
    name: str
    index: int
    d_x: int
    d_blocks: tuple[int, ...]
    k: int
    variant: str = "main"

    @property
    def dim(self) -> int:
        return self.d_x + sum(self.d_blocks)


LEVELS = {
    "SCM-1": Level("SCM-1", 0, 2, (1, 1, 1, 1, 2), 2),
    "SCM-2": Level("SCM-2", 1, 8, (16, 16, 16, 16, 16), 2),
    "SCM-3": Level("SCM-3", 2, 32, (256, 256, 256, 256, 256), 2),
}

# SCM-1c (PRD section 7): the CEVAE diagnostic.  W_4 loses its I contamination, I moves from W_5 into X, and
# every block becomes a pure measurement of U.  Treatment and outcome equations are unchanged.
VARIANTS = {"SCM-1c": Level("SCM-1c", 3, 3, (1, 1, 1, 1, 1), 2, "1c")}
BLOCK_INFO_1C = (("M1",), ("M2",), ("M3",), ("M4",), ("M0",))
X_INFO_1C = ("X1", "C", "I")


def x_info(level: "Level") -> tuple[str, ...]:
    return X_INFO_1C if level.variant == "1c" else X_INFO


def block_info(level: "Level") -> tuple[tuple[str, ...], ...]:
    return BLOCK_INFO_1C if level.variant == "1c" else BLOCK_INFO


def bad_i(level: "Level") -> float:
    return 0.0 if level.variant == "1c" else BAD_I


def level(name: str) -> "Level":
    return LEVELS[name] if name in LEVELS else VARIANTS[name]


def _haar(d: int, rng: np.random.Generator) -> np.ndarray:
    """A Haar-distributed orthogonal matrix (QR of a Gaussian matrix with the sign fix)."""
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))[None, :]


def rotations(level: Level) -> dict[str, np.ndarray]:
    """The fixed rotation of each observed block, drawn once per level and never per data set."""
    ss = np.random.SeedSequence(ROTATION_SEED + level.index).spawn(1 + N_BLOCKS)
    out = {"X": _haar(level.d_x, np.random.default_rng(ss[0]))}
    for j, d in enumerate(level.d_blocks):
        out[f"W{j + 1}"] = _haar(d, np.random.default_rng(ss[1 + j]))
    return out


def rotation_digest(level: Level) -> str:
    h = hashlib.sha256()
    for key, q in rotations(level).items():
        h.update(key.encode())
        h.update(np.ascontiguousarray(q).tobytes())
    return h.hexdigest()


def latents(n: int, rng: np.random.Generator, bad: float | None = None) -> dict[str, np.ndarray]:
    z = rng.standard_normal((n, 9))
    u, i, c, eta = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    eps = z[:, 4:9]
    m = u[:, None] + PROXY_SD * eps
    m[:, 4] += (BAD_I if bad is None else bad) * i
    return {"U": u, "I": i, "C": c, "X1": u + KAPPA * eta,
            "M0": m[:, 0], "M1": m[:, 1], "M2": m[:, 2], "M3": m[:, 3], "M4": m[:, 4]}


def propensity(lat: dict[str, np.ndarray]) -> np.ndarray:
    index = TREAT["U"] * lat["U"] + TREAT["I"] * lat["I"] + TREAT["X1"] * lat["X1"] + TREAT["C"] * lat["C"]
    return 0.1 + 0.8 * norm.cdf(index)


def outcome_mean(lat: dict[str, np.ndarray]) -> np.ndarray:
    """E[Y(0) | U, X_1, C]; Y(1) adds tau."""
    return OUTCOME["U"] * lat["U"] + OUTCOME["X1"] * lat["X1"] + OUTCOME["C"] * lat["C"]


def _observe(info: np.ndarray, d: int, q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, c = info.shape
    stacked = np.concatenate([info, NOISE_SD * rng.standard_normal((n, d - c))], axis=1) if d > c else info
    return stacked @ q.T


def generate(level: Level, n: int, seed: int) -> dict[str, np.ndarray]:
    """One data set.  Keys ending in _eval_only are for the evaluator and must never reach a learner."""
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    lat = latents(n, streams[0], bad_i(level))
    q = rotations(level)
    blocks = block_info(level)
    x = _observe(np.column_stack([lat[k] for k in x_info(level)]), level.d_x, q["X"], streams[1])
    w = [_observe(np.column_stack([lat[k] for k in blocks[j]]), level.d_blocks[j], q[f"W{j + 1}"], streams[1])
         for j in range(N_BLOCKS)]
    p = propensity(lat)
    a = (streams[2].random(n) < p).astype(float)
    mu0 = outcome_mean(lat)
    y = mu0 + TAU * a + OUTCOME_NOISE * streams[3].standard_normal(n)
    out = {"X": x, "A": a, "Y": y}
    for j in range(N_BLOCKS):
        out[f"W{j + 1}"] = w[j]
    for key, value in lat.items():
        out[f"{key}_eval_only"] = value
    out["p_eval_only"] = p
    out["mu0_eval_only"] = mu0
    return out


LEARNER_KEYS = ("X", "W1", "W2", "W3", "W4", "W5", "A", "Y")


def learner_view(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: data[k] for k in LEARNER_KEYS}


def informative(level: Level, data: dict[str, np.ndarray], block: str) -> np.ndarray:
    """Evaluator-only inverse: the informative coordinates of an observed block, by undoing its rotation."""
    q = rotations(level)[block]
    names = x_info(level) if block == "X" else block_info(level)[int(block[1:]) - 1]
    return (data[block] @ q)[:, :len(names)]
