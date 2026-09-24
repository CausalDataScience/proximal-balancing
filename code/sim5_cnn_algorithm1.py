"""Algorithm 1 pieces for the Sim5 CNN run that do not involve a network: the uniform split draw, the AIPW
score, the operational radius and the radius graph with its return rule (PRD sections 8 and 10).

A split is a 64-bit word; bit k set means block k is held out.  The universe is every nonempty proper subset.
"""
from __future__ import annotations

import numpy as np

J = 64
FULL = (1 << J) - 1


def draw_splits(m: int, seed: int) -> list[int]:
    """m distinct words uniform over the 2^64 - 2 nonempty proper subsets (rejection of 0, all-ones, repeats)."""
    bitgen = np.random.PCG64(seed)
    out, seen = [], set()
    while len(out) < m:
        word = int(bitgen.random_raw())
        if word == 0 or word == FULL or word in seen:
            continue
        seen.add(word)
        out.append(word)
    return out


def held_out_blocks(word: int) -> list[int]:
    return [k for k in range(J) if word >> k & 1]


def keep_mask(word: int, block_masks: np.ndarray) -> np.ndarray:
    """28 x 28 boolean mask of the retained pixels for split `word`."""
    held = np.zeros(block_masks.shape[1:], dtype=bool)
    for k in held_out_blocks(word):
        held |= block_masks[k]
    return ~held


def aipw_scores(e: np.ndarray, m0: np.ndarray, m1: np.ndarray, a: np.ndarray, y: np.ndarray) -> np.ndarray:
    return m1 - m0 + a * (y - m1) / e - (1 - a) * (y - m0) / (1 - e)


def operational_radius(sds: list[float], n_e: int, delta: float) -> float:
    """rho = max_S sd_S sqrt(R / (n_E delta)); a diagnostic radius over the R retained splits, not a certificate."""
    if not sds:
        return float("inf")
    return float(max(sds) * np.sqrt(len(sds) / (n_e * delta)))


def aggregate(labels: list, estimates, rho: float) -> dict:
    """Link within 2 rho (boundary included), return the median of the unique largest component.

    No retained split, a non-finite estimate or radius, or a tie for the largest component is no single return;
    a tie still reports every tied component's median as the candidate set.
    """
    est = np.asarray(estimates, dtype=float)
    if len(est) == 0:
        return {"return_kind": "no_return", "reason": "R=0", "output": None, "candidates": [], "members": []}
    if not np.all(np.isfinite(est)) or not np.isfinite(rho):
        return {"return_kind": "no_return", "reason": "non-finite", "output": None, "candidates": [], "members": []}
    parent = list(range(len(est)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(est)):
        for j in range(i + 1, len(est)):
            if abs(est[i] - est[j]) <= 2.0 * rho:
                parent[find(i)] = find(j)
    comps: dict[int, list[int]] = {}
    for i in range(len(est)):
        comps.setdefault(find(i), []).append(i)
    size = max(len(v) for v in comps.values())
    largest = [sorted(v) for v in comps.values() if len(v) == size]
    cands = [float(np.median(est[v])) for v in largest]
    unique = len(largest) == 1
    return {"return_kind": "unique_largest" if unique else "tied_largest", "reason": None,
            "output": cands[0] if unique else None, "candidates": cands,
            "members": [[labels[i] for i in v] for v in largest],
            "component_sizes": sorted((len(v) for v in comps.values()), reverse=True)}
