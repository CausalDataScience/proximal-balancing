"""SCM-4e v2: the association front end
(PRD memo/2026-09-21-scm4e-v2-labelfree-association-prd-v1.md, sections 3.4 and 3.5).

The sealed front end keeps the directions of a block that carry the most variance.  In an image embedding that
keeps the wrong thing: floor hue, wall hue, object hue, scale and shape are drawn afresh for every image, so they
are large and they are independent of everything else in the problem.  The angle is the only visual attribute
that carries a recorded coordinate, and a recorded coordinate is correlated with the other blocks through the
latent state.  So this front end keeps the directions of a block that are correlated with the rest of what the
learner holds, and lets variance alone decide nothing.

Measured on the label-free embedding: the four directions the sealed rule keeps recover the angle with R^2 0.006,
and the one direction this rule keeps recovers it with R^2 0.934 to 0.939.

What it reads is the fold it is fitted on and nothing else: X, A and the blocks' embeddings.  Not Y, not any
image label, not the role of a block, not which split is being scored.  All thirty splits of a rotation share one
fitted front end, exactly as they share one sealed `Prep`.
"""
from __future__ import annotations

import contextlib
import hashlib

import numpy as np

import family_v2_dgp as g
import family_v2_probe as pr

THRESHOLD_MULTIPLE = 1.5          # a direction must beat this multiple of the block's own permutation floor
PERMUTATIONS = 5                  # draws of the floor; the largest of them is the floor
RANK_TOLERANCE = 1e-8             # relative singular value below which a whitened direction is dropped


def _standardise(a: np.ndarray, rows: np.ndarray):
    mu, sd = a[rows].mean(0), a[rows].std(0) + 1e-12
    return mu, sd


def _whiten(z: np.ndarray):
    """Columns with no correlation left and unit variance, plus the map that sends new rows to the same place."""
    n = len(z)
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    keep = s > RANK_TOLERANCE * (s[0] if s.size else 1.0)
    return u[:, keep] * np.sqrt(n), (vt[keep].T / s[keep]) * np.sqrt(n)


def _floor_seed(rows: np.ndarray, j: int) -> int:
    """From the rows the front end is fitted on and the block number, so a rerun gives the same floor."""
    digest = hashlib.sha256(np.sort(np.asarray(rows)).astype(np.int64).tobytes() + bytes([j]))
    return int.from_bytes(digest.digest()[:8], "little")


class AssociationPrep:
    """Stands in for the sealed `Prep`: standardise X, and reduce each block to its canonical variates against
    the rest.  `x`, `block` and `widths` are the three methods the sealed learner calls."""

    def __init__(self, view: dict[str, np.ndarray], rows: np.ndarray, embed_dim: int) -> None:
        rows = np.sort(np.asarray(rows))        # the fit depends on which rows, never on the order they arrive in
        self.embed_dim = embed_dim
        self.x_mu, self.x_sd = _standardise(view["X"], rows)
        keys = [f"W{j + 1}" for j in range(g.N_BLOCKS)]
        stats = {k: _standardise(view[k], rows) for k in keys}
        std = {k: (view[k][rows] - stats[k][0]) / stats[k][1] for k in keys}
        x_std = (view["X"][rows] - self.x_mu) / self.x_sd
        a = view["A"][rows].astype(np.float64)[:, None]
        a_std = (a - a.mean()) / (a.std() + 1e-12)
        self.blocks, self.report = [], {}
        for j, key in enumerate(keys):
            n_images = max(1, view[key].shape[1] // embed_dim)
            rest = np.column_stack([x_std, a_std] + [std[o] for o in keys if o != key])
            ue, te = _whiten(std[key])
            ur, _ = _whiten(rest)
            cross = ue.T @ ur / len(rows)
            left, rho, _ = np.linalg.svd(cross, full_matrices=False)
            floors = []
            for p in range(PERMUTATIONS):
                order = np.random.default_rng(_floor_seed(rows, j) + p).permutation(len(rows))
                floors.append(float(np.linalg.svd(ue[order].T @ ur / len(rows), compute_uv=False)[0]))
            floor = max(floors)
            above = int(np.sum(rho >= THRESHOLD_MULTIPLE * floor))
            keep = max(1, min(n_images, above))
            proj = te @ left[:, :keep]
            scores = std[key] @ proj
            self.blocks.append((stats[key][0], stats[key][1], proj, scores.std(0) + 1e-12))
            self.report[key] = {"canonical_correlations": [round(float(v), 4) for v in rho[:4]],
                                "noise_floor": round(floor, 4),
                                "threshold": round(THRESHOLD_MULTIPLE * floor, 4),
                                "above_threshold": above, "images": n_images, "kept": keep,
                                "hit_the_image_cap": above > n_images, "nothing_above_threshold": above == 0}

    def x(self, view, rows):
        return (view["X"][rows] - self.x_mu) / self.x_sd

    def block(self, view, rows, j):
        mu, sd, proj, scale = self.blocks[j]
        return (((view[f"W{j + 1}"][rows] - mu) / sd) @ proj) / scale

    def widths(self) -> list[int]:
        return [b[2].shape[1] for b in self.blocks]


@contextlib.contextmanager
def installed(embed_dim: int):
    """Put `AssociationPrep` in the sealed learner's `Prep` slot for the duration, and always put it back.

    Nothing in `family_v2_probe.py` or `family_v2_competitors.py` is edited; the swap lives here and lasts only
    as long as the block.  The sealed code calls `Prep(view, rows)` with two positional arguments, so that is the
    shape of what goes in the slot.
    """
    sealed = pr.Prep
    if getattr(sealed, "__name__", "") != "Prep":
        raise RuntimeError("the Prep slot does not hold the sealed class; refusing to swap")
    pr.Prep = lambda view, rows: AssociationPrep(view, rows, embed_dim)
    try:
        yield
    finally:
        pr.Prep = sealed


def variate_view(view: dict[str, np.ndarray], embed_dim: int, rows: np.ndarray | None = None) -> tuple[dict, dict]:
    """The same view with every block replaced by its canonical variates, fitted on `rows` (all rows by default).

    This is what the methods without folds receive: proximal 2SLS, Park's single proxy control, and the CEVAE row
    that is given the same directions PROBE works on.
    """
    rows = np.arange(len(view["A"])) if rows is None else np.asarray(rows)
    prep = AssociationPrep(view, rows, embed_dim)
    every = np.arange(len(view["A"]))
    out = dict(view)
    for j in range(g.N_BLOCKS):
        out[f"W{j + 1}"] = prep.block(view, every, j)
    return out, prep.report
