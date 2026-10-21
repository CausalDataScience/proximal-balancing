"""PROBE (family_v2_probe) with any number of blocks, for blocks found by discover_blocks.merge_coupled (2026-10-19,
memo section 43).  The sealed modules are not edited; configure(J) sets, in the running process:
    the number of blocks J and of held-out sets 2^J - 2 (the screen's and the v3 rule's Bonferroni level)
    the per-block reduction: a block is reduced to its principal components above the Marchenko-Pastur edge only
    when it has more coordinates than the P_FEATURES = 4 features the reduction may keep; a smaller block is kept
    whole, as blocks of one or two coordinates always were.  For every block split used in the paper (blocks of 1,
    2, 16, 256 coordinates) this is the sealed rule; it differs only for blocks of three or four coordinates.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

import family_v2_dgp as g
import family_v2_probe as pr
import noise_scaled as ns
import rhc_pooled as rp

_SEALED_PREP = pr.Prep


class SmallBlocksWhole(_SEALED_PREP):
    def __init__(self, view: dict[str, np.ndarray], rows: np.ndarray):
        x = view["X"][rows]
        self.x_mu, self.x_sd = x.mean(0), x.std(0) + 1e-12
        self.blocks = []
        for j in range(g.N_BLOCKS):
            w = view[f"W{j + 1}"][rows]
            mu = w.mean(0)
            if w.shape[1] > pr.P_FEATURES:
                _, sv, vt = np.linalg.svd(w - mu, full_matrices=False)
                proj = vt[:pr.spike_count(sv ** 2 / len(rows), len(rows), w.shape[1])].T
            else:
                proj = np.eye(w.shape[1])
            self.blocks.append((mu, proj, ((w - mu) @ proj).std(0) + 1e-12))


def configure(n_blocks: int) -> None:
    g.N_BLOCKS = n_blocks
    pr.N_SPLITS = 2 ** n_blocks - 2
    z = float(norm.ppf(1.0 - pr.ALPHA / pr.N_SPLITS))
    rp.Z = z
    ns.Z = z
    pr.Prep = SmallBlocksWhole


_SEALED_Z = rp.Z


def reset() -> None:
    """Back to the sealed configuration: five blocks, 30 held-out sets, the sealed reduction."""
    g.N_BLOCKS = 5
    pr.N_SPLITS = 30
    rp.Z = _SEALED_Z
    ns.Z = _SEALED_Z
    pr.Prep = _SEALED_PREP
