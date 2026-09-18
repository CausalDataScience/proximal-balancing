"""E12: the SCM-1 sample-size curve.

The question is narrow.  A pass at n = 6000 could be one lucky point, so this asks whether the error falls as
the sample grows, on fresh seeds, with the smaller samples nested inside the larger ones so that only the
amount of data changes.  It is not a proof of asymptotic consistency, and full monotonicity through the
middle points is reported but not required.

The error is also split into the part the representation and the aggregation are responsible for and the part
the estimator is responsible for:

    tau_hat - tau = (theta_C - tau) + (tau_hat - theta_C),

where theta_C is the median population adjustment value over the selected largest component.  Those
population values come from quadrature on the structural equations, not from the data.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

import e8_population as e8
import minimal_suite_common as c
import probe_structured_scm as ps

SOURCES = ["minimal_suite_common.py", "run_e12_scm1_ncurve.py", "probe_structured_scm.py",
           "e8_population.py"]
PROTOCOL_PATH = "../results/e12_scm1_ncurve_protocol_v2.json"
OUT_PATH = "../results/e12_scm1_ncurve_confirm_v2.json"
FROZEN = {
    "protocol_id": "e12-scm1-sample-size-curve-v1",
    "experiment_id": "E12", "mode": "operational_rotate4", "phase": "confirmation",
    "sampling": "nested prefixes: one draw per seed at the largest size, every smaller size is its prefix",
    "supersedes": "the v1 protocol, which regenerated each role at every sample size",
    "supersede_reason": "regenerating at a fixed seed does not nest. U, X, I, C and the proxies agree on "
                        "the prefix, but A consumes a variable number of stream values and the outcome "
                        "noise then starts at a different offset, so A and Y differ from the first unit. "
                        "The v1 curve therefore compared independent datasets, not one growing sample.",
    "scm": "SCM-1", "learner": "restricted_parametric_same_D_brier_rotate4",
    "sample_sizes": [1500, 3000, 6000, 12000],
    "rows_per_fold": {"1500": 375, "3000": 750, "6000": 1500, "12000": 3000},
    "data_seeds": [91000000 + 1000 * r for r in range(20)],
    "baselines": ["naive", "X", "raw_XW", "oracle"],
    "gates": {"return_rate_min": 0.95, "mae_max": 0.05, "p90_max": 0.10,
              "oracle_excess_upper_max": 0.05, "bonferroni_families": 4, "alpha": 0.05},
    "note": ("the existing n = 6000 confirmation is external corroboration only; this curve computes its own "
             "n = 6000 cell on fresh seeds"),
}


def population_theta(split: tuple[int, ...], r: float | None, spec) -> float:
    """theta_S of the selected map, by quadrature on the structural equations."""
    maps = e8.linear_maps(spec)
    if 0 in split:
        kept = [j for j in range(5) if j not in split]
        Z = np.vstack([maps["X"], np.mean([maps[f"K{j}"] for j in kept], axis=0)])
        import e11_theorem_audit as legacy
        return legacy._adjusted_effect_rows(spec, Z)
    return e8.adjusted_effect(spec, split, r)


def freeze_protocol() -> dict:
    p = Path(PROTOCOL_PATH)
    if p.exists():
        return json.loads(p.read_text())
    proto = dict(FROZEN)
    proto["dgp"] = vars(ps.SCM_FAMILY["SCM-1"])
    proto["source_snapshot"] = c.source_snapshot(SOURCES)
    proto["frozen_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    proto["runs"] = [{"experiment_id": "E12", "mode": "operational_rotate4", "phase": "confirmation",
                      "sample_sizes": FROZEN["sample_sizes"], "data_seeds": FROZEN["data_seeds"]}]
    return c.seal_protocol(p, proto)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(FROZEN["data_seeds"]))
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()
    proto = freeze_protocol()
    seeds = FROZEN["data_seeds"][:args.seeds]
    c.check_protocol(proto, {"experiment_id": "E12", "mode": "operational_rotate4",
                             "phase": "confirmation", "sample_sizes": FROZEN["sample_sizes"],
                             "data_seeds": FROZEN["data_seeds"]}, PROTOCOL_PATH, SOURCES)
    spec = ps.SCM_FAMILY["SCM-1"]
    root = c.result_root("E12", "operational_rotate4", "confirmation", PROTOCOL_PATH, SOURCES)
    t0 = time.time()
    print(f"E12 | SCM-1 | n in {FROZEN['sample_sizes']} | {len(seeds)} seeds, nested prefixes")
    largest = max(FROZEN["sample_sizes"])
    # One draw per seed at the largest size.  Every smaller size is a prefix of it, so a seed's curve
    # follows a single growing sample rather than four unrelated datasets.
    banks = {seed: [ps.generate_role(spec, largest // 4, seed + j) for j in range(4)] for seed in seeds}
    for n_total in FROZEN["sample_sizes"]:
        for seed in seeds:
            # the protocol's learner is the restricted parametric Brier one; the module default is a
            # different estimator and would silently change what this experiment measures
            rep = ps.run_rotating_replicate(
                spec, n_total, seed, learner="brier",
                folds=[ps.slice_role(f, n_total // 4) for f in banks[seed]])
            agg = rep["aggregation"]
            unique = agg["return_kind"] == "unique_largest"
            theta_c = None
            if unique and agg.get("members"):
                members = [tuple(s) for s in agg["members"][0]]
                rs = {}
                for fit in rep.get("split_fits", []):
                    sp = tuple(fit["split"]) if "split" in fit else None
                    if sp in members:
                        rot = [q["brier"]["r"] for q in fit.get("rotations", [])
                               if (q.get("brier") or {}).get("r") is not None]
                        if rot:
                            rs[sp] = float(np.median(rot))
                vals = [population_theta(sp, rs.get(sp), spec) for sp in members if sp in rs or 0 in sp]
                theta_c = float(np.median(vals)) if vals else None
            cell = {"data_seed": seed, "n_total": n_total, "rows_per_fold": n_total // 4,
                    "return_kind": agg["return_kind"],
                    "estimate": agg["output"] if unique else None,
                    "abs_error": abs(agg["output"] - 1.0) if unique else None,
                    "theta_component_median": theta_c,
                    "representation_error": (abs(theta_c - 1.0) if theta_c is not None else None),
                    "estimation_error": (abs(agg["output"] - theta_c)
                                         if unique and theta_c is not None else None),
                    "baselines": rep.get("baseline_estimates", {}),
                    "rho": rep.get("rho"), "complete": True}
            root["cells"].append(cell)
        done = [q for q in root["cells"] if q["n_total"] == n_total and q["abs_error"] is not None]
        print(f"  n={n_total:6d}: {len(done)}/{len(seeds)} unique, "
              f"MAE {np.mean([q['abs_error'] for q in done]):.4f} at {time.time()-t0:.0f}s")
    root["summary"] = summarise(root["cells"], seeds)
    root["verdict"] = "PASS" if root["summary"]["all_gates_pass"] else "FAIL"
    root["complete"] = True
    c.write_json_new(args.out, root)
    print(f"\nverdict: {root['verdict']}  written to {args.out}")
    return 0


def summarise(cells: list[dict], seeds: list[int]) -> dict:
    g = FROZEN["gates"]
    q = tdist.ppf(1.0 - g["alpha"] / g["bonferroni_families"], len(seeds) - 1)
    by_n = {}
    for n_total in FROZEN["sample_sizes"]:
        sel = [x for x in cells if x["n_total"] == n_total]
        ok = [x for x in sel if x["abs_error"] is not None]
        err = np.array([x["abs_error"] for x in ok])
        by_n[str(n_total)] = {
            "cells": len(sel), "unique_returns": len(ok), "return_rate": len(ok) / max(len(sel), 1),
            "mae": float(err.mean()) if len(err) else None,
            "p90": float(np.quantile(err, 0.9)) if len(err) else None,
            "median_representation_error": float(np.median([x["representation_error"] for x in ok
                                                            if x["representation_error"] is not None]))
            if any(x["representation_error"] is not None for x in ok) else None,
            "median_estimation_error": float(np.median([x["estimation_error"] for x in ok
                                                        if x["estimation_error"] is not None]))
            if any(x["estimation_error"] is not None for x in ok) else None,
            "baseline_mae": {k: float(np.mean([abs(x["baselines"][k] - 1.0) for x in sel
                                               if k in x["baselines"]]))
                             for k in FROZEN["baselines"] if any(k in x["baselines"] for x in sel)},
        }

    def paired_upper(n_total: int, base: str) -> float:
        sel = [x for x in cells if x["n_total"] == n_total and x["abs_error"] is not None
               and base in x["baselines"]]
        d = np.array([x["abs_error"] - abs(x["baselines"][base] - 1.0) for x in sel])
        return float(d.mean() + q * d.std(ddof=1) / math.sqrt(len(d)))

    big = FROZEN["sample_sizes"][-1]
    small = FROZEN["sample_sizes"][0]
    pair = {b: paired_upper(big, b) for b in ("X", "raw_XW", "oracle")}
    by_seed = {}
    for x in cells:
        if x["abs_error"] is not None:
            by_seed.setdefault(x["data_seed"], {})[x["n_total"]] = x["abs_error"]
        shrink = [v[big] - v[small] for v in by_seed.values() if big in v and small in v]
    shrink = np.array([v[big] - v[small] for v in by_seed.values() if big in v and small in v])
    shrink_upper = float(shrink.mean() + q * shrink.std(ddof=1) / math.sqrt(len(shrink))) if len(shrink) > 1 \
        else float("inf")
    b = by_n[str(big)]
    gates = {
        "return_rate": b["return_rate"] >= g["return_rate_min"],
        "mae": b["mae"] is not None and b["mae"] <= g["mae_max"],
        "p90": b["p90"] is not None and b["p90"] <= g["p90_max"],
        "beats_X": pair["X"] < 0.0,
        "beats_raw": pair["raw_XW"] < 0.0,
        "oracle_excess": pair["oracle"] <= g["oracle_excess_upper_max"],
        "error_falls_from_smallest_to_largest": shrink_upper < 0.0,
    }
    return {"by_sample_size": by_n, "paired_upper_at_largest": pair,
            "shrinkage_upper": shrink_upper, "bonferroni_t": float(q), "gates": gates,
            "all_gates_pass": all(gates.values())}


if __name__ == "__main__":
    raise SystemExit(main())
