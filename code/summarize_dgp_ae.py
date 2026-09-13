#!/usr/bin/env python3
"""Summarize DGP A to E runs (plan sections 11 and 16.4).

Reports, per DGP and sample size:
  return behaviour        no-return, single-survivor, unique-largest, tied-largest rates
  unconditional           acceptable-return rate (single return within 0.10 of the truth)
  conditional             mean absolute error over single-return replicates, marked as conditional
  paired contrasts        G^X, G^raw, G^orc with the Bonferroni-t upper bound over the 15-test family
  screen behaviour        true and false positive rates against structural validity
  set-valued              distance from the truth to the returned set, its diameter, and whether the truth
                          lies in its convex hull
The headline table is only produced when every cell has its full complement of replicates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

TRUE = 1.0
FAMILY_SIZE = 15          # plan 11.1: five DGPs times three contrasts
LEVEL = 0.05
TARGET_BAND = 0.10


def load(paths: list[str]) -> list[dict]:
    """Load runs and refuse to treat the same cell from two files as independent replicates.

    Two result files can hold the same (phase, DGP, n, replicate) under different code versions.  Counting
    both would inflate the replicate count and shrink every interval, so the duplicate is an error.
    """
    runs, seen = [], {}
    for p in paths:
        payload = json.loads(Path(p).read_text())
        for r in payload["runs"]:
            key = (r.get("phase"), r["dgp"], r["n"], r["replicate"])
            if key in seen:
                raise SystemExit(f"duplicate cell {key} in {p} and {seen[key]}; summarize one code version "
                                 "at a time, or renumber the replicates")
            seen[key] = p
            r["_source"] = p
            r["_complete"] = payload.get("complete", False)
            runs.append(r)
    return runs


def contrasts(rec: dict) -> dict | None:
    """Plan 1.3.  Defined only for replicates that returned a single value."""
    out = rec["algorithm1"]["output"]
    if out is None:
        return None
    comp = rec["comparators"]
    L = lambda v: abs(v - TRUE)
    lp = L(out)
    return {"L_probe": lp, "L_X": L(comp["X"]), "L_raw": L(comp["raw"]), "L_oracle": L(comp["oracle"]),
            "L_XC": L(comp["XC"]), "L_naive": L(comp["naive"]),
            "G_X": lp - 0.5 * L(comp["X"]), "G_raw": lp - 0.5 * L(comp["raw"]),
            "G_orc": lp - L(comp["oracle"]) - TARGET_BAND}


def bonferroni_upper(values: np.ndarray) -> dict:
    r = len(values)
    if r < 2:
        return {"mean": float(values.mean()) if r else float("nan"), "n": r, "upper": float("nan")}
    m, se = float(values.mean()), float(values.std(ddof=1) / math.sqrt(r))
    crit = tdist.ppf(1 - LEVEL / FAMILY_SIZE, r - 1)
    return {"mean": m, "se": se, "n": r, "upper": m + crit * se,
            "upper_unadjusted": m + tdist.ppf(1 - LEVEL, r - 1) * se}


def screen_rates(rec: dict, key: str = "balance_feasible") -> dict:
    """True and false positive rates of the screen against the target set.

    The default target is the set of configurations for which an exactly balancing representation can exist,
    which is stricter than "does not hold out block 0": a configuration that holds out every measurement block
    leaves nothing to balance with.
    """
    tp = fp = tn = fn = 0
    for rot in rec["rotations"]:
        for c in rot["candidates"]:
            if c.get(key, c["structural_valid"]):
                tp += c["screen_pass"]; fn += not c["screen_pass"]
            else:
                fp += c["screen_pass"]; tn += not c["screen_pass"]
    return {"tpr": tp / max(1, tp + fn), "fpr": fp / max(1, fp + tn), "n_valid": tp + fn, "n_invalid": fp + tn}


def summarize(runs: list[dict]) -> dict:
    cells: dict[tuple, list[dict]] = {}
    for r in runs:
        cells.setdefault((r["dgp"], r["n"]), []).append(r)
    out = {}
    for (d, n), recs in sorted(cells.items()):
        kinds = [r["algorithm1"]["return_kind"] for r in recs]
        singles = [r for r in recs if r["algorithm1"]["output"] is not None]
        con = [contrasts(r) for r in singles]
        # plan 11.2: the denominator is every replicate, not only those that returned a single value
        acceptable = [bool(r["algorithm1"]["output"] is not None and abs(r["algorithm1"]["output"] - TRUE) <= TARGET_BAND)
                      for r in recs]
        cell = {"replicates": len(recs), "single_return_rate": len(singles) / len(recs),
                "no_return_rate": kinds.count("no_return") / len(recs),
                "single_survivor_rate": kinds.count("single_survivor") / len(recs),
                "unique_largest_rate": kinds.count("unique_largest") / len(recs),
                "tied_largest_rate": kinds.count("tied_largest") / len(recs),
                "acceptable_return_rate": float(np.mean(acceptable)),
                "screen": {k: float(np.mean([screen_rates(r)[k] for r in recs])) for k in ("tpr", "fpr")},
                "retained_mean": float(np.mean([len(r["retained"]) for r in recs])),
                "retained_valid_mean": float(np.mean([len(r["retained_valid"]) for r in recs])),
                "rho_mean": float(np.mean([r["algorithm1"]["rho"] for r in recs])),
                "seconds_mean": float(np.mean([r["seconds"] for r in recs]))}
        if con:
            cell["conditional_mae"] = {k: float(np.mean([c[k] for c in con]))
                                       for k in ("L_probe", "L_X", "L_XC", "L_raw", "L_oracle", "L_naive")}
            cell["paired"] = {k: bonferroni_upper(np.array([c[k] for c in con]))
                              for k in ("G_X", "G_raw", "G_orc")}
            cell["criterion"] = {
                "mae_at_most_0.10": cell["conditional_mae"]["L_probe"] <= TARGET_BAND,
                "half_of_X_and_raw": (cell["conditional_mae"]["L_probe"] <= 0.5 * cell["conditional_mae"]["L_X"]
                                      and cell["conditional_mae"]["L_probe"] <= 0.5 * cell["conditional_mae"]["L_raw"]),
                "single_return_rate_at_least_0.95": cell["single_return_rate"] >= 0.95,
                "paired_upper_bounds_below_zero": all(cell["paired"][k]["upper"] < 0 for k in cell["paired"])}
            cell["criterion"]["all_four"] = all(cell["criterion"].values())
        # candidate-level aggregate over every replicate and every rotation, so a table can never be built
        # from one rotation of one replicate by accident
        cand = [c for r in recs for rot in r["rotations"] for c in rot["candidates"]]
        cell["candidates"] = {}
        for label, keep in (("feasible", True), ("infeasible", False)):
            sub = [c for c in cand if bool(c.get("balance_feasible", c["structural_valid"])) is keep]
            if not sub:
                continue
            cell["candidates"][label] = {
                "records": len(sub),
                "refit_gap_audit": float(np.mean([c.get("refit_gap_audit", float("nan")) for c in sub])),
                "training_val_gap": float(np.mean([c.get("val_gap", float("nan")) for c in sub])),
                "theta": float(np.mean([c["theta"] for c in sub])),
                "screen_pass_rate": float(np.mean([c["screen_pass"] for c in sub])),
                "encoder_updates": float(np.mean([c.get("encoder_updates", float("nan")) for c in sub]))}
        cell["candidate_aggregate_scope"] = f"{len(recs)} replicates x {len(recs[0]['rotations'])} rotations"
        ties = [r for r in recs if r["algorithm1"]["return_kind"] == "tied_largest"]
        if ties:
            sets = [np.array(r["algorithm1"]["candidates"]) for r in ties]
            cell["set_valued"] = {"n": len(ties),
                                  "distance_to_truth": float(np.mean([np.abs(s - TRUE).min() for s in sets])),
                                  "diameter": float(np.mean([s.max() - s.min() for s in sets])),
                                  "truth_in_hull": float(np.mean([(s.min() <= TRUE <= s.max()) for s in sets]))}
        out[f"{d}_n{n}"] = cell
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--expect-replicates", type=int, default=0,
                    help="plan 16.4: only emit the headline table when every cell has this many replicates")
    args = ap.parse_args()
    runs = load(args.runs)
    summary = summarize(runs)
    phases = sorted({r.get("phase", "unknown") for r in runs})
    files_complete = all(r["_complete"] for r in runs)
    blockers = []
    if not args.expect_replicates:
        blockers.append("no --expect-replicates given")
    else:
        for k, v in summary.items():
            if v["replicates"] < args.expect_replicates:
                blockers.append(f"{k} has {v['replicates']} of {args.expect_replicates} replicates")
    if not files_complete:
        blockers.append("at least one source file is marked incomplete")
    if len(phases) != 1:
        blockers.append(f"runs mix phases {phases}")
    elif phases[0] != "confirmatory":
        blockers.append(f"phase is {phases[0]}, not confirmatory")
    # plan 16.4: a headline table is produced only when every cell is complete and confirmatory
    payload = {"summary": summary, "phases": phases, "headline_ready": not blockers,
               "headline_blockers": blockers}
    print(json.dumps(payload, indent=1, default=float))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=1, default=float))
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dgp", "n", "replicate", "return_kind", "probe", "X", "XC", "raw", "oracle", "naive",
                        "retained", "retained_valid", "rho", "seconds"])
            for r in runs:
                a = r["algorithm1"]
                w.writerow([r["dgp"], r["n"], r["replicate"], a["return_kind"], a["output"],
                            r["comparators"]["X"], r["comparators"]["XC"], r["comparators"]["raw"],
                            r["comparators"]["oracle"], r["comparators"]["naive"], len(r["retained"]),
                            len(r["retained_valid"]), a["rho"], r["seconds"]])


if __name__ == "__main__":
    main()
