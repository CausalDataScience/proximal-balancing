"""Aggregation step of PROBE: the common linking radius and the largest agreeing component.

    common_radius   rho, the 95th percentile of max_S |xi_S| over 5,000 draws xi ~ N(0, Sigma), where Sigma is
                    the centred cross-product of the evaluation-fold scores of the candidate splits divided by n^2
    aggregate       link two splits when their estimates differ by at most 2 rho, form connected components
                    (single linkage), and return the median of the unique largest component; a tie for the
                    largest component returns no estimate

Both functions and the number of draws are copied unchanged from the module that held them in the code used
for the paper's experiments.
"""
from __future__ import annotations

import numpy as np

BOOTSTRAP_DRAWS = 5_000


def common_radius(score_matrix: np.ndarray, seed: int) -> float:
    n = score_matrix.shape[0]
    centered = score_matrix - score_matrix.mean(axis=0)
    covariance = centered.T @ centered / (n * n)
    draws = np.random.default_rng(seed).multivariate_normal(
        np.zeros(score_matrix.shape[1]), covariance, size=BOOTSTRAP_DRAWS
    )
    return float(np.quantile(np.max(np.abs(draws), axis=1), 0.95))


def aggregate(labels: list[tuple[int, ...]], estimates: np.ndarray, rho: float) -> dict:
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
    return {
        "return_kind": "unique_largest" if len(largest) == 1 else "tied_largest",
        "output": candidates[0] if len(candidates) == 1 else None,
        "candidates": candidates,
        "members": [[list(labels[i]) for i in idx] for idx in largest],
        "component_sizes": sorted((len(v) for v in components.values()), reverse=True),
    }
