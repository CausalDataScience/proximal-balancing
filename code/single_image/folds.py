"""Patient groups (plan section 5).  One image per patient here, so a patient is a row and "all pieces of
a patient in one group" holds by construction.

    phase1_groups   R (image-only, 25%) and the rest; level A runs the paper's own four rotating folds
                    (family_v2_probe.folds) inside the rest
    six_groups      R plus five equal groups that rotate through the roles D, C, P, S, E (stages 2-4)
"""
from __future__ import annotations

import numpy as np

R_FRACTION = 0.25
ROLES = ("D", "C", "P", "S", "E")


def _perm(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng([seed, 5]).permutation(n)


def phase1_groups(n: int, seed: int, r_fraction: float = R_FRACTION) -> dict[str, np.ndarray]:
    perm = _perm(n, seed)
    n_r = int(round(r_fraction * n))
    return {"R": np.sort(perm[:n_r]), "rest": np.sort(perm[n_r:])}


def six_groups(n: int, seed: int, r_fraction: float = R_FRACTION) -> dict[str, object]:
    """R is the same set as in phase1_groups for the same (n, seed); the rest is cut into five groups."""
    g = phase1_groups(n, seed, r_fraction)
    rest = np.random.default_rng([seed, 9]).permutation(g["rest"])
    return {"R": g["R"], "groups": [np.sort(x) for x in np.array_split(rest, len(ROLES))]}


def roles(groups: list[np.ndarray], rotation: int) -> dict[str, np.ndarray]:
    """Rotation r gives role ROLES[i] to group (i + r) mod 5; R never rotates."""
    return {role: groups[(i + rotation) % len(ROLES)] for i, role in enumerate(ROLES)}
