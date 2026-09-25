"""The noise-floor screen rule, used for Twins, IHDP, ACIC and RHC, and aggregation from a stored cross-product.

`core.run_dataset` keeps, for every split and rotation, the screen gap and its standard error.  The rule pools
the four rotations of a split, whose screen folds are disjoint,

    gap_S = max(mean_r gap_{S,r}, 0),        SE_S = sqrt(sum_r se_{S,r}^2) / 4,        z = Phi^{-1}(1 - 0.05 / 30),

sets the threshold to the data set's own noise level,

    floor = median over the candidate splits of z * SE_S,

and retains S when gap_S + z * SE_S <= floor.  Nothing is refitted.  The retained splits are then aggregated as in
`core.run_dataset`: the linking radius comes from the Gaussian multiplier bootstrap, here computed from the stored
cross-product of the evaluation-fold scores (the same formula as `aggregation.common_radius`, with the same seed
offset), and the estimate is the median of the unique largest agreeing component.

`aggregate_kept` is copied from the module that pooled the rotations; `screens_of`, `pooled` and `decide` from
the module that introduced the noise floor.  `decide` is unchanged except that it no longer reports true and
false positive rates, which read the known valid splits of the synthetic family and are not part of PROBE.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from . import aggregation as ps
from . import core as pr

Z = float(norm.ppf(1.0 - pr.ALPHA / pr.N_SPLITS))
BOOTSTRAP_SEED_OFFSET = 5


def aggregate_kept(probe: dict, keep: list[int], n: int, seed: int) -> dict:
    """The sealed aggregation on a retained set, with the common radius recomputed from the stored
    cross-product exactly as the sealed run computes it from the score matrix."""
    names, splits = probe["split_names"], pr.all_splits()
    if not keep:
        return {"return_kind": "no_return", "output": None, "rho": None, "component_sizes": [], "se_member": None}
    cov = np.array(probe["score_crossproduct"])[np.ix_(keep, keep)] / (n * n)
    draws = np.random.default_rng(seed + BOOTSTRAP_SEED_OFFSET).multivariate_normal(
        np.zeros(len(keep)), cov, size=ps.BOOTSTRAP_DRAWS)
    rho = float(np.quantile(np.max(np.abs(draws), axis=1), 0.95))
    values = np.array([probe["records"][names[i]]["estimate"] for i in keep])
    agg = ps.aggregate([splits[i] for i in keep], values, rho)
    return {"return_kind": agg["return_kind"], "output": agg["output"], "rho": rho,
            "component_sizes": agg.get("component_sizes"),
            "se_member": float(np.sqrt(np.median(np.diag(cov))))}


def screens_of(record: dict) -> list[dict]:
    """The four per-rotation screens of one held-out choice, in either shard layout."""
    return record["screen"] if "screen" in record else [r["screen"] for r in record["rotations"]]


def pooled(record: dict) -> tuple[float, float]:
    g = np.array([s["gap"] for s in screens_of(record)])
    se = np.array([s["se"] for s in screens_of(record)])
    return max(float(g.mean()), 0.0), float(np.sqrt((se ** 2).sum()) / len(se))


def decide(probe: dict, n: int, seed: int) -> dict:
    """The noise-floor decision on one run: floor, retained set, and the aggregation of it."""
    records = probe["records"] if "records" in probe else probe["per_split"]
    names = probe["split_names"]
    stats = {nm: pooled(records[nm]) for nm in names}
    floor = float(np.median([Z * se for _, se in stats.values()]))
    keep = [i for i, nm in enumerate(names) if stats[nm][0] + Z * stats[nm][1] <= floor]
    like = {"split_names": names, "score_crossproduct": probe["score_crossproduct"],
            "records": {nm: {"estimate": records[nm]["estimate"]} for nm in names}}
    agg = aggregate_kept(like, keep, n, seed)
    kept = {names[i] for i in keep}
    return {"floor": floor, "n_retained": len(keep), "retained": sorted(kept), **agg}
