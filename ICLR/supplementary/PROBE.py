"""PROBE, PROxy Blockwise Exclusion: the entry point (Algorithm 1 of the paper).

Input: n units with covariates X, a proxy split into J prespecified blocks W1, ..., WJ, a binary treatment A and an
outcome Y.  Output: an estimate of the average treatment effect, or a declared failure.

For every split S, a nonempty proper subset of the blocks that is held out (T = 2^J - 2 splits in all), PROBE
    learns    a rank-k linear encoder Z = (X, V_{-S} B^T) of the kept blocks on the representation fold by
              minimising the empirical Brier discrepancy: the Brier risk of a base critic that predicts A from Z
              minus that of an augmented critic that predicts A from (X, W_S, Z)
    screens   the split with freshly fitted critics on a separate screen fold (see the rules below)
    estimates the effect with the honest AIPW score: a bounded-probit propensity clipped to [0.05, 0.95] and
              arm-wise ridge outcome regressions fitted on the nuisance fold, evaluated on the evaluation fold
and the four folds are rotated so that each serves once in every role; a split's estimate is the mean over the
four rotations.  The retained splits are linked when their estimates differ by at most 2 rho, with rho from a
Gaussian multiplier bootstrap, and PROBE returns the median of the unique largest connected component.  A tie
for the largest component, or no retained split, is a declared failure: the estimate is None.

Screen rules (the choices of the paper):
    threshold     five blocks.  Keep S when, in every one of the four rotations, max(gap, 0) + z SE < t and every
                  base-critic prediction on the screen fold lies in [0.05, 0.95]; z = Phi^{-1}(1 - 0.05 / 30).
                  Used for SCM-1 to SCM-4 with t = 1.5, 1.8, 1.6 and 1.7 x 10^-3 at n = 24,000.  These values
                  were calibrated at that sample size; at a much smaller n this rule usually keeps no split.
    noise_floor   any J.  Pool the four rotations, gap = max(mean gap, 0) and SE = sqrt(sum SE_r^2) / 4, and keep S
                  when gap + z SE is at most the median of z SE over the candidate splits.  Used for Twins, IHDP,
                  ACIC and RHC.
    significance  J other than five.  Keep S when the pooled mean gap - z SE <= 0, z = Phi^{-1}(1 - 0.05 / T).
                  Used for the two-block kernel benchmark.

Split budget.  Algorithm 1 draws m of the T splits uniformly without replacement, independently of the data.
Every experiment in the paper used m = T, the default.  With m < T the m splits are drawn with `split_seed`, and
only they enter the screen and the aggregation.  All T splits are still fitted, with the same seeds, so a split's
record does not depend on m, and the running time does not shrink with m.

Python:
    from PROBE import probe
    result = probe({"X": X, "W1": W1, ..., "W5": W5, "A": A, "Y": Y}, rule="noise_floor", seed=7)
    result["estimate"]            # a float, or None when PROBE declares failure

Command line, on an .npz file with arrays X, W1, ..., WJ, A, Y (for example from data/make_dataset.py):
    python PROBE.py data.npz --rule noise_floor
    python PROBE.py data.npz --rule threshold --threshold 0.0015 --m 10 --split-seed 3 --out result.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import scm_family                                   # noqa: E402  (Level, the run configuration)
from probe_lib import aggregation, core, j_blocks, pooled_screen   # noqa: E402

RULES = {"five_blocks": ("threshold", "noise_floor"), "other": ("significance", "noise_floor")}


def block_count(view: dict) -> int:
    j = 0
    while f"W{j + 1}" in view:
        j += 1
    return j


def as_view(view: dict) -> dict[str, np.ndarray]:
    """X, W1..WJ as two-dimensional float arrays and A, Y as vectors; X may have zero columns."""
    n = len(np.asarray(view["A"]))
    out = {"A": np.asarray(view["A"], dtype=float).ravel(), "Y": np.asarray(view["Y"], dtype=float).ravel()}
    x = np.asarray(view["X"], dtype=float) if "X" in view else np.zeros((n, 0))
    out["X"] = x if x.ndim == 2 else x.reshape(n, -1)
    for j in range(block_count(view)):
        w = np.asarray(view[f"W{j + 1}"], dtype=float)
        out[f"W{j + 1}"] = w if w.ndim == 2 else w.reshape(n, -1)
    if not set(np.unique(out["A"])) <= {0.0, 1.0}:
        raise ValueError("A must be binary, coded 0 and 1")
    if len(out["Y"]) != n or any(len(v) != n for v in out.values()):
        raise ValueError("every array must have one row per unit")
    return out


def draw_splits(T: int, m: int | None, split_seed: int) -> list[int]:
    """Line 1 of Algorithm 1: indices of m of the T splits, uniformly without replacement, from `split_seed` alone."""
    if m is None or m >= T:
        return list(range(T))
    if m < 1:
        raise ValueError("the split budget m must be at least 1")
    order = np.random.default_rng(split_seed).permutation(T)
    return sorted(int(i) for i in order[:m])


def aggregate_subset(names, splits, crossproduct, estimates, keep: list[int], n: int, seed: int) -> dict:
    """The aggregation on the retained splits of a budget draw.  rho is computed from the stored cross-product of
    the evaluation-fold scores exactly as pooled_screen.aggregate_kept computes it (same formula, same seed offset,
    same number of draws), and aggregation.aggregate links, picks the largest component and takes its median."""
    if not keep:
        return {"return_kind": "no_return", "output": None, "rho": None, "component_sizes": [], "retained": []}
    cov = np.array(crossproduct)[np.ix_(keep, keep)] / (n * n)
    draws = np.random.default_rng(seed + pooled_screen.BOOTSTRAP_SEED_OFFSET).multivariate_normal(
        np.zeros(len(keep)), cov, size=aggregation.BOOTSTRAP_DRAWS)
    rho = float(np.quantile(np.max(np.abs(draws), axis=1), 0.95))
    agg = aggregation.aggregate([splits[i] for i in keep], np.array([estimates[i] for i in keep]), rho)
    return {"return_kind": agg["return_kind"], "output": agg["output"], "rho": rho,
            "component_sizes": agg.get("component_sizes"), "retained": [names[i] for i in keep]}


def _five_blocks(view, rule, threshold, seed, k, sampled, n) -> tuple[dict, dict, dict]:
    widths = tuple(view[f"W{j + 1}"].shape[1] for j in range(5))
    level = scm_family.Level(name="PROBE", index=0, d_x=view["X"].shape[1], d_blocks=widths, k=k, variant="real")
    tolerance = threshold if rule == "threshold" else 0.0      # the noise-floor rule does not read it
    res = core.run_dataset(view, level, tolerance, seed)
    names, splits = res["split_names"], core.all_splits()
    table = {}
    for nm in names:
        rec = res["records"][nm]
        gap, se = pooled_screen.pooled(rec)
        table[nm] = {"estimate": rec["estimate"], "pooled_gap": gap, "pooled_se": se,
                     "passes_threshold_in_all_rotations": rec["pass_all"] if rule == "threshold" else None,
                     "overlap_all_rotations": all(r["screen"]["overlap"] for r in rec["rotations"])}
    full = len(sampled) == len(names)
    if rule == "threshold":
        if full:                                                # the paper's configuration: core's own output
            out = {"return_kind": res["return_kind"], "output": res["estimate"], "rho": res["rho"],
                   "component_sizes": res["component_sizes"], "retained": res["retained"]}
        else:
            keep = [i for i in sampled if res["records"][names[i]]["pass_all"]]
            out = aggregate_subset(names, splits, res["score_crossproduct"],
                                   [res["records"][nm]["estimate"] for nm in names], keep, n, seed)
    else:
        if full:                                                # the paper's configuration: pooled_screen.decide
            dec = pooled_screen.decide(res, n, seed)
            out = {k_: dec[k_] for k_ in ("return_kind", "output", "rho", "component_sizes", "retained", "floor")}
        else:
            stats = {i: pooled_screen.pooled(res["records"][names[i]]) for i in sampled}
            floor = float(np.median([pooled_screen.Z * se for _, se in stats.values()]))
            keep = [i for i in sampled if stats[i][0] + pooled_screen.Z * stats[i][1] <= floor]
            out = aggregate_subset(names, splits, res["score_crossproduct"],
                                   [res["records"][nm]["estimate"] for nm in names], keep, n, seed)
            out["floor"] = floor
    return out, table, res


def _other_blocks(view, rule, seed, k, sampled, n, J) -> tuple[dict, dict, dict]:
    res = j_blocks.run(view, k, seed, n_blocks=J)
    names, splits, z = res["split_names"], j_blocks.all_splits(J), res["z"]
    recs = res["records"]
    table = {nm: {"estimate": recs[nm]["estimate"], "pooled_gap": max(recs[nm]["pooled_gap"], 0.0),
                  "pooled_se": recs[nm]["pooled_se"], "mean_gap": recs[nm]["pooled_gap"],
                  "overlap_all_rotations": recs[nm]["overlap_all_rotations"]} for nm in names}
    if len(sampled) == len(names):                              # the paper's configuration: the readings of run
        reading = res["readings"][rule]
        out = {k_: reading.get(k_) for k_ in ("return_kind", "output", "rho", "retained")}
        if rule == "noise_floor":
            out["floor"] = res["noise_floor_threshold"]
    else:
        if rule == "significance":
            keep = [i for i in sampled if recs[names[i]]["pooled_gap"] - z * recs[names[i]]["pooled_se"] <= 0.0]
            floor = None
        else:
            floor = float(np.median([z * recs[names[i]]["pooled_se"] for i in sampled]))
            keep = [i for i in sampled if res["per_candidate_upper"][names[i]] <= floor]
        out = aggregate_subset(names, splits, res["score_crossproduct"], [recs[nm]["estimate"] for nm in names],
                               keep, n, seed)
        if floor is not None:
            out["floor"] = floor
    return out, table, res


def probe(view: dict, rule: str = "noise_floor", threshold: float | None = None, seed: int = 0, k: int = 2,
          m: int | None = None, split_seed: int = 0, return_records: bool = False) -> dict:
    """Run PROBE on one data set.

    view        dict of arrays: X (n x d, d may be 0), W1, ..., WJ (n x d_j each, J >= 2), A (0/1), Y
    rule        "threshold" (J = 5, needs `threshold`), "noise_floor" (any J) or "significance" (J other than 5)
    threshold   the screen threshold t of the threshold rule
    seed        the seed of the fold split, the encoder starts (seed + 1000 rotation + split index) and the
                bootstrap of rho (seed + 5)
    k           the rank of the encoder's map of the kept blocks (2 in the paper)
    m           the split budget of Algorithm 1; None means every split, as in the paper's experiments
    split_seed  the seed of the budget draw when m < T
    """
    t0 = time.time()
    view = as_view(view)
    n, J = len(view["A"]), block_count(view)
    if J < 2:
        raise ValueError("PROBE needs at least two proxy blocks, W1 and W2")
    allowed = RULES["five_blocks"] if J == 5 else RULES["other"]
    if rule not in allowed:
        raise ValueError(f"with J = {J} blocks the rule must be one of {allowed}")
    if rule == "threshold" and threshold is None:
        raise ValueError("the threshold rule needs a threshold")
    T = 2 ** J - 2
    sampled = draw_splits(T, m, split_seed)
    if J == 5:
        out, table, res = _five_blocks(view, rule, threshold, seed, k, sampled, n)
    else:
        out, table, res = _other_blocks(view, rule, seed, k, sampled, n, J)
    names = res["split_names"]
    result = {"estimate": out["output"], "return_kind": out["return_kind"],
              "rule": rule, "threshold": threshold if rule == "threshold" else None, "floor": out.get("floor"),
              "retained": sorted(out["retained"] or []), "rho": out["rho"],
              "component_sizes": out.get("component_sizes"),
              "n": n, "J": J, "T": T, "m": len(sampled), "splits_searched": [names[i] for i in sampled],
              "seed": seed, "split_seed": split_seed if len(sampled) < T else None, "k": k,
              "per_split": table, "seconds": round(time.time() - t0, 1)}
    if return_records:
        result["records"] = res
    return result


def _jsonable(x):
    if isinstance(x, dict):
        return {str(key): _jsonable(v) for key, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run PROBE on one data set stored as an .npz file with arrays "
                                             "X, W1, ..., WJ, A and Y.")
    ap.add_argument("data", help="path of the .npz file")
    ap.add_argument("--rule", choices=("threshold", "noise_floor", "significance"), default="noise_floor")
    ap.add_argument("--threshold", type=float, default=None, help="screen threshold t of the threshold rule")
    ap.add_argument("--seed", type=int, default=None,
                    help="PROBE's seed; default: the file's probe_seed if present, otherwise 0")
    ap.add_argument("--k", type=int, default=2, help="rank of the encoder map (2 in the paper)")
    ap.add_argument("--m", type=int, default=None, help="split budget m; default: all T splits, as in the paper")
    ap.add_argument("--split-seed", type=int, default=0, help="seed of the budget draw when m < T")
    ap.add_argument("--out", default=None, help="write the full result as JSON here")
    ap.add_argument("--records", action="store_true", help="include every rotation-level record in --out")
    args = ap.parse_args(argv)

    with np.load(args.data, allow_pickle=False) as f:
        arrays = {key: f[key] for key in f.files}
    view = {key: arrays[key] for key in arrays if key in ("X", "A", "Y") or
            (key.startswith("W") and key[1:].isdigit())}
    seed = args.seed if args.seed is not None else int(arrays["probe_seed"]) if "probe_seed" in arrays else 0
    result = probe(view, rule=args.rule, threshold=args.threshold, seed=seed, k=args.k, m=args.m,
                   split_seed=args.split_seed, return_records=args.records)

    print(f"PROBE on {args.data}: n = {result['n']}, J = {result['J']} blocks, {result['m']} of T = {result['T']} "
          f"splits searched, rule {result['rule']}, seed {result['seed']}, {result['seconds']} s")
    print(f"  retained {len(result['retained'])}: {' '.join(result['retained']) or 'none'}")
    if result["rho"] is not None:
        print(f"  linking radius rho = {result['rho']:.6f}; component sizes {result['component_sizes']}")
    if result["estimate"] is None:
        print(f"  no estimate: {result['return_kind']} (PROBE declares failure)")
    else:
        print(f"  estimate of the average treatment effect: {result['estimate']:.6f}")
    if "tau" in arrays and np.isfinite(arrays["tau"]):
        print(f"  true effect stored in the file (not used by PROBE): {float(arrays['tau']):.6f}")
    if args.out:
        Path(args.out).write_text(json.dumps(_jsonable(result), indent=1) + "\n")
        print(f"  full result written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
