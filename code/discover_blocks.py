"""Block discovery from the proxy alone (2026-10-19, after memo section 43: PROBE with coordinate-level held-out sets fails
because one coordinate is too weak a check, and uniform random subsets almost never avoid the invalid coordinates).

The blocks of a proxy are groups of coordinates that measure the same thing.  After the factor that all coordinates
share is removed, coordinates of one group still covary through what is specific to that group, and coordinates of
different groups do not.  discover_blocks clusters the coordinates of W by these residual correlations; it reads W
only, never A or Y, so it can run on all rows like the unsupervised preprocessing of PROBE.

    residual (first version): cluster |R - lambda_1 v_1 v_1^T|; works when a coordinate carries much of its group's
             signal, but drowns in noise when the signal of a group is spread over hundreds of coordinates.
    loading (default): keep the K eigencomponents of R above the Marchenko-Pastur edge (1 + sqrt(m / n))^2, where
             the signal lives; the loading row of a coordinate is its position in that K-dimensional space.  The
             coordinates of one group measure the same thing in different proportions, so their loading rows point in
             one direction; spectral clustering of the |cosine| between loading rows (raised to a power to sharpen it)
             into J groups.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2


def affinity(w: np.ndarray, method: str = "loading", power: float = 4.0) -> tuple[np.ndarray, int]:
    z = (w - w.mean(0)) / (w.std(0) + 1e-12)
    n, m = z.shape
    r = z.T @ z / n
    lam, vec = np.linalg.eigh(r)
    if method == "residual":
        r1 = r - lam[-1] * np.outer(vec[:, -1], vec[:, -1])
        aff = np.abs(r1); k = 1
    else:
        k = max(1, int((lam > (1 + np.sqrt(m / n)) ** 2).sum()))
        load = vec[:, -k:] * np.sqrt(lam[-k:])
        unit = load / (np.linalg.norm(load, axis=1, keepdims=True) + 1e-12)
        aff = np.abs(unit @ unit.T) ** power
    np.fill_diagonal(aff, 0.0)
    return aff, k


def discover_blocks(w: np.ndarray, n_blocks: int, seed: int = 0, restarts: int = 20, method: str = "loading") -> list[np.ndarray]:
    if w.shape[1] == n_blocks:
        return [np.array([j]) for j in range(n_blocks)]
    aff, _ = affinity(w, method)
    deg = aff.sum(1) + 1e-12
    norm = aff / np.sqrt(np.outer(deg, deg))
    _, ev = np.linalg.eigh(norm)
    emb = ev[:, -n_blocks:]
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    best, best_inertia = None, np.inf
    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        cent, lab = kmeans2(emb, n_blocks, minit="++", seed=int(rng.integers(2 ** 31)))
        if len(set(lab.tolist())) < n_blocks:
            continue
        inertia = float(((emb - cent[lab]) ** 2).sum())
        if inertia < best_inertia:
            best, best_inertia = lab, inertia
    if best is None:
        raise RuntimeError("spectral clustering did not find the requested number of non-empty groups")
    groups = [np.flatnonzero(best == c) for c in range(n_blocks)]
    return sorted(groups, key=lambda idx: int(idx.min()))


def loading_rows(w: np.ndarray) -> np.ndarray:
    """Loading rows of the coordinates in the signal subspace (the K eigencomponents above the Marchenko-Pastur edge)."""
    z = (w - w.mean(0)) / (w.std(0) + 1e-12)
    n, m = z.shape
    lam, vec = np.linalg.eigh(z.T @ z / n)
    k = max(1, int((lam > (1 + np.sqrt(m / n)) ** 2).sum()))
    return vec[:, -k:] * np.sqrt(lam[-k:])


def refine_by_subspace(w: np.ndarray, groups: list[np.ndarray], max_dim: int = 2, rel: float = 0.3,
                       iters: int = 20, penalty: float | None = None) -> list[np.ndarray]:
    """k-subspaces refinement of a partition.  A group of coordinates that measure the same quantities spans a
    low-dimensional subspace of loading space: a line when they measure one quantity, a plane when they mix two (as
    the coordinates of a block that carries two values do).  Each group gets the subspace of its loading rows (weighted
    by their length, dimension 1 or 2: 2 when the second singular value is at least `rel` of the first); every
    coordinate moves to the group whose subspace is closest in angle to its loading row; repeat until stable.

    With `penalty` = c (the likelihood-ratio form, 2026-10-20): a coordinate moves to the group with the smallest
    absolute residual ||l - P_g l||^2 plus c * s2 * dim(g), s2 the noise of a loading row per dimension (median over
    coordinates of the residual to their own group divided by K - dim).  A plane has one dimension more than a line
    and so always absorbs one more dimension of noise; c = 3.84, the 95% point of chi-square with one degree of
    freedom, asks the plane to beat a line by more than that noise before it takes a coordinate.  Weak coordinates,
    whose direction is mostly noise, then stop drifting into planes (memo section 43, SCM-2 confirmation)."""
    load = loading_rows(w)
    max_dim = min(max_dim, load.shape[1] - 1)       # in a K-dimensional signal space a K-dimensional "subspace" is
    if max_dim < 1:                                  # the whole space and would absorb every coordinate
        return sorted(groups, key=lambda idx: int(idx.min()))
    norm = np.linalg.norm(load, axis=1) + 1e-12
    unit = load / norm[:, None]
    lab = np.empty(len(load), dtype=int)
    for g, idx in enumerate(groups):
        lab[idx] = g
    for _ in range(iters):
        bases = []
        for g in range(len(groups)):
            rows = load[lab == g]
            if len(rows) == 0:
                bases.append(None); continue
            _, s, vt = np.linalg.svd(rows, full_matrices=False)
            d = 1 if len(s) < 2 or s[1] < rel * s[0] else min(max_dim, len(s))
            bases.append(vt[:d])
        resid = np.full((len(load), len(groups)), np.inf)
        dims = np.zeros(len(groups))
        for g, b in enumerate(bases):
            if b is not None:
                dims[g] = len(b)
                if penalty is None:
                    resid[:, g] = 1 - ((unit @ b.T) ** 2).sum(1)      # 1 - cos^2 of the angle to the subspace
                else:
                    resid[:, g] = (load ** 2).sum(1) - ((load @ b.T) ** 2).sum(1)
        if penalty is not None:
            own = resid[np.arange(len(load)), lab]
            s2 = float(np.median(own / (load.shape[1] - dims[lab])))
            resid = resid + penalty * s2 * dims[None, :]
        new = resid.argmin(1)
        if np.array_equal(new, lab):
            break
        lab = new
    out = [np.flatnonzero(lab == g) for g in range(len(groups))]
    return sorted([o for o in out if len(o)], key=lambda idx: int(idx.min()))


def discover_blocks_refined(w: np.ndarray, n_blocks: int, seed: int = 0, penalty: float | None = None) -> list[np.ndarray]:
    """discover_blocks (spectral clustering of loading directions), then k-subspaces refinement."""
    if w.shape[1] == n_blocks:
        return [np.array([j]) for j in range(n_blocks)]
    return refine_by_subspace(w, discover_blocks(w, n_blocks, seed=seed), penalty=penalty)


COUPLING = 0.5      # the 6-coordinate proxies of SCM-1, Twins, IHDP and ACIC: one pair of groups at 0.65-0.68, every
                    # other pair below 0.28; SCM-2, SCM-3 and RHC: every pair below 0.28 (2026-10-19, memo section 43)


def partial_correlations(w: np.ndarray) -> np.ndarray:
    """|partial correlation| of every pair of coordinates given all the other coordinates, from the inverse
    correlation matrix."""
    z = (w - w.mean(0)) / (w.std(0) + 1e-12)
    p = np.linalg.pinv(z.T @ z / len(z))
    pc = np.abs(p) / np.sqrt(np.outer(np.diag(p), np.diag(p)))
    np.fill_diagonal(pc, 0.0)
    return pc


def merge_coupled(w: np.ndarray, groups: list[np.ndarray], threshold: float = COUPLING) -> list[np.ndarray]:
    """Merge groups that hold a pair of coordinates with partial correlation above `threshold`.  Such a pair is
    nearly redundant given everything else, so a held-out set that takes one and keeps the other is a weak check
    that can pass while biased (memo section 43: on Twins the splits holding out one coordinate of W_5 passed with
    a bias of -0.020).  Coordinates coupled this tightly are therefore always held out together.  This can leave
    fewer groups than requested."""
    pc = partial_correlations(w)
    parent = list(range(len(groups)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            if pc[np.ix_(groups[a], groups[b])].max() > threshold:
                parent[find(a)] = find(b)
    merged: dict[int, list[int]] = {}
    for a, idx in enumerate(groups):
        merged.setdefault(find(a), []).extend(idx.tolist())
    return sorted([np.array(sorted(v)) for v in merged.values()], key=lambda idx: int(idx.min()))


def discover_blocks_merged(w: np.ndarray, n_blocks: int, seed: int = 0, penalty: float | None = None) -> list[np.ndarray]:
    """discover_blocks_refined, then merge_coupled."""
    return merge_coupled(w, discover_blocks_refined(w, n_blocks, seed=seed, penalty=penalty))
