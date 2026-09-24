"""RHC v2: the screen evidence of the four rotations pooled into one decision
(memo/2026-09-22-rhc-real-data-results-v1.md, section 8, written after the v1 diagnostic).

The sealed rule retains a held-out choice only if every one of its four rotations clears the threshold on
its own.  The four screen folds are disjoint samples and the hypothesis is one, so v2 pools them: the mean
gap and the standard error of that mean,

    upper_pooled = max(mean_r gap_r, 0) + z * sqrt(sum_r se_r^2) / 4,      z = Phi^{-1}(1 - 0.05 / 30),

and retains the choice when upper_pooled <= t.  Nothing is refitted.  The gap_r and se_r are the ones the
sealed run computed; v2 only changes how the four are combined.

The rule is validated first on the 80 sealed simulation data sets, where the truth is known, and is applied
to the RHC records only after that.  Both results are written here, write-once.

    python3 -B rhc_pooled.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

import family_v2_probe as pr
import probe_structured_scm as ps
import rhc_data as rd
import run_family_v2 as rf
import run_rhc as rr

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FULL = RESULTS / "rhc" / "diagnostic_v1_full_records.json"
OUT = RESULTS / "rhc" / "result_v2_pooled.json"
Z = float(norm.ppf(1.0 - pr.ALPHA / pr.N_SPLITS))
TARGETS = ("S1", "S2", "S3", "S12", "S13", "S23", "S123")
SEALED_T = {"SCM-1": 0.0015, "SCM-2": 0.0018, "SCM-3": 0.0016, "SCM-4": 0.0017}
BOOTSTRAP_SEED_OFFSET = 5


def pooled_upper(rotations: list[dict]) -> float:
    gaps = np.array([r["screen"]["gap"] for r in rotations])
    ses = np.array([r["screen"]["se"] for r in rotations])
    return max(float(gaps.mean()), 0.0) + Z * float(np.sqrt((ses ** 2).sum()) / len(ses))


def sealed_upper(rotations: list[dict]) -> float:
    return max(r["screen"]["upper"] for r in rotations)


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


def shards(level: str) -> list[dict]:
    folder = RESULTS / "family_v2" / ("scm4/confirm" if level == "SCM-4" else "confirm")
    return [json.loads(p.read_text()) for p in sorted(folder.glob(f"{level}_seed*.json"))]


def validate_on_simulations() -> dict:
    """Both rules on the sealed confirmations at their sealed thresholds: what pooling changes and what it
    leaves alone."""
    out = {}
    for level, t in SEALED_T.items():
        out[level] = {}
        for rule_name, upper in (("sealed", sealed_upper), ("pooled", pooled_upper)):
            errors, tpr, fpr, kept_w5 = [], [], [], 0
            for shard in shards(level):
                probe, names = shard["probe"], shard["probe"]["split_names"]
                keep = [i for i, nm in enumerate(names) if upper(probe["records"][nm]["rotations"]) <= t]
                agg = aggregate_kept(probe, keep, shard["n"], shard["task"]["seed"])
                if agg["output"] is not None:
                    errors.append(abs(agg["output"] - 1.0))
                kept = {names[i] for i in keep}
                tpr.append(len(kept & set(TARGETS)) / len(TARGETS))
                fpr.append(len(kept - set(TARGETS)) / (len(names) - len(TARGETS)))
                kept_w5 += any("5" in nm for nm in kept)
            out[level][rule_name] = {"t": t, "returned": len(errors), "mae": float(np.mean(errors)),
                                     "p90": float(np.quantile(errors, 0.9)), "tpr": float(np.mean(tpr)),
                                     "fpr": float(np.mean(fpr)), "data_sets_keeping_a_w5_choice": kept_w5}
    return out


def rhc_path(probe: dict) -> list[dict]:
    names = probe["split_names"]
    rows = []
    for t in rr.GRID:
        keep = [i for i, nm in enumerate(names) if pooled_upper(probe["records"][nm]["rotations"]) <= t]
        agg = aggregate_kept(probe, keep, rd.N, rd.SEED)
        rows.append({"t": t, "n_retained": len(keep), "retained": [names[i] for i in keep], **agg})
    return rows


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; results are write-once")
    validation = validate_on_simulations()
    for level, rules in validation.items():
        for rule, v in rules.items():
            print(f"{level} {rule:6s}: returned {v['returned']}/20 MAE {v['mae']:.4f} p90 {v['p90']:.4f} "
                  f"TPR {v['tpr']:.3f} FPR {v['fpr']:.3f} W5 kept in {v['data_sets_keeping_a_w5_choice']}")
    probe = json.loads(FULL.read_text())["probe"]
    names = probe["split_names"]
    per_split = {nm: {"pooled_upper": pooled_upper(probe["records"][nm]["rotations"]),
                      "sealed_upper": sealed_upper(probe["records"][nm]["rotations"]),
                      "mean_gap": float(np.mean([r["screen"]["gap"] for r in probe["records"][nm]["rotations"]])),
                      "estimate": probe["records"][nm]["estimate"]} for nm in names}
    path = rhc_path(probe)
    payload = {"artifact": "rhc_probe_result_v2_pooled", "generated_utc": rf.now(),
               "rule": "retain S if max(mean_r gap_r, 0) + z * sqrt(sum_r se_r^2)/4 <= t, z = Phi^{-1}(1 - 0.05/30)",
               "z": Z, "why": "v1 diagnostic: the held-out choice W2 had gap ~ 0 in every rotation and failed only on "
                              "the per-rotation noise term z*se at fold size 1,434; the four screen folds are disjoint, "
                              "so their evidence is pooled",
               "adopted_after_seeing": "the v1 RHC diagnostic (results/rhc/diagnostic_v1_full_records.json)",
               "validated_before_reading_rhc_on": "the 80 sealed simulation data sets, below",
               "simulation_validation": validation,
               "source_sha256": {f: rf.sha256(HERE / f) for f in rr.SOURCES + ["rhc_pooled.py"]},
               "full_records_sha256": rf.sha256(FULL),
               "per_split": per_split, "path_over_threshold_grid": path,
               "published_comparison": rd.PUBLISHED}
    with open(OUT, "x") as handle:
        handle.write(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(f"written {OUT}")
    for row in path:
        print(f"  t = {row['t']:.1e}: retained {row['n_retained']} {row['retained']} -> {row['return_kind']} {row['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
