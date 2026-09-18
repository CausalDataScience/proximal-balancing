"""SCM-6 v2, development only: do the repaired selection and screen recover the known population facts?

This opens no confirmation seeds.  The population truth for this design is known exactly, so the three
things that have to work can be measured directly rather than inferred from an ATE number:

  1. root recovery, whether the selected coefficient lands near the balancing root of its split,
  2. screen behaviour, the true and false positive rates against the exactly known feasible set,
  3. ATE error, whether the pipeline's returned effect approaches the truth.

Only if all three recover is there a case for freezing a confirmatory protocol.  The gates below are the
ones that decision rests on and they are stated before the run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import minimal_suite_common as c
import probe_highdim_w as hd
import probe_highdim_w_v2 as v2
import scm6_population as sp
import run_scm6_highdim_w as v1

SOURCES = ["minimal_suite_common.py", "probe_highdim_w.py", "probe_highdim_w_v2.py",
           "scm6_population.py", "run_scm6_v2_development.py"]
DEV_SEEDS = [94000000 + 1000 * r for r in range(5)]
TOTAL_N = 12000
PENALTY_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
# Rows inside the D fold.  The encoder is far better than it needs to be at 1,200 rows, reaching a
# channel correlation of 0.982 against a gate of 0.90, while the r search is the step that is short of
# data.  Moving 600 rows across costs the encoder 0.982 -> 0.965, still well clear of its gate, and buys
# the search a 40 percent larger sample.  Measured on development seeds before this run.
EMB_FIT, EMB_CAL = 600, 300
TARGET_COMPONENTS = 1
# The population curve separates the two families cleanly: a balance-feasible split reaches a discrepancy
# of at most 3.0e-5 at its own root, while the best any infeasible split can do is 7.7e-3.  The threshold
# is the round number nearest the geometric midpoint of that window, fixed before the multi-seed run.
POPULATION_WINDOW = (2.992e-5, 7.724e-3)
SCREEN_THRESHOLD = 1e-3
ROOT_TOLERANCE = 0.10
GATES = {"root_recovery_rate_min": 0.80, "screen_tpr_min": 0.80, "screen_fpr_max": 0.20,
         "mean_abs_error_max": 0.05}


def truth() -> dict:
    """Exact population facts: the balancing root of each split, and what each split targets there."""
    out = {}
    for split in c.oriented_splits():
        root = sp.operative_root(split)
        feasible = root is not None
        tau = sp.adjusted_effect(split, root, n=200000)["tau_Z"] if feasible else None
        out[c.split_id(split)] = {"split": split, "root": root, "feasible": feasible,
                                  "tau_at_root": tau,
                                  "targets_truth": bool(feasible and abs(tau - 1.0) < 1e-3)}
    return out


def one_replicate(seed: int, facts: dict) -> dict:
    t0 = time.time()
    data = hd.generate(TOTAL_N, seed)
    view = hd.observed_view(data, include_outcome=True)
    folds = c.four_folds(TOTAL_N, seed + 1)
    splits = c.oriented_splits()
    rows_out, scores = [], {sp_: [] for sp_ in splits}
    passes = {sp_: 0 for sp_ in splits}
    base_scores: dict[str, list] = {}
    for rot in c.rotations(folds):
        c.assert_roles_disjoint(rot.roles)
        d_idx = rot.roles["D"]
        emb = hd.fit_embedding(v1.slice_rows(view, d_idx[:EMB_FIT]),
                               v1.slice_rows(view, d_idx[EMB_FIT:EMB_FIT + EMB_CAL]))
        if emb["fail_closed"]:
            return {"data_seed": seed, "return_kind": "no_return", "estimate": None,
                    "fail_closed": emb["fail_closed"], "complete": True,
                    "timing_seconds": time.time() - t0}
        phi = v1.slice_rows(view, d_idx[EMB_FIT + EMB_CAL:])
        s_rows, n_rows, e_rows = (v1.slice_rows(view, rot.roles[k]) for k in ("S", "N", "E"))
        k_phi, k_s = hd.apply_embedding(emb, phi), hd.apply_embedding(emb, s_rows)
        k_n, k_e = hd.apply_embedding(emb, n_rows), hd.apply_embedding(emb, e_rows)
        # Comparators on the same N and E rows.  None of these touch a random stream, so adding them
        # leaves every PROBE number in this run identical to the run without them.
        pcs = hd.blockwise_pca(v1.slice_rows(view, d_idx[:EMB_FIT]))
        feat_n, feat_e = hd.baseline_features(n_rows, emb, pcs), hd.baseline_features(e_rows, emb, pcs)
        for name in feat_n:
            base_scores.setdefault(name, []).append(hd.aipw_from_features(
                feat_n[name], n_rows["A"], n_rows["Y"], feat_e[name], e_rows["A"], e_rows["Y"]))
        base_scores.setdefault("oracle", []).append(hd.aipw_from_features(
            np.column_stack([n_rows["X"], data["U_eval_only"][rot.roles["N"]], n_rows["C"]]),
            n_rows["A"], n_rows["Y"],
            np.column_stack([e_rows["X"], data["U_eval_only"][rot.roles["E"]], e_rows["C"]]),
            e_rows["A"], e_rows["Y"]))
        for sp_ in splits:
            sid = c.split_id(sp_)
            oseed = c.optimizer_seed("scm6-v2-development", seed, rot.index, sid, 0)
            proj = v2.fit_target_projection(phi, sp_, TARGET_COMPONENTS, oseed)
            pen = hd.split_penalties(phi, k_phi, sp_, PENALTY_GRID, oseed)["base"]
            sel = v2.select_r_crossfit(phi, k_phi, sp_, pen, oseed, projection=proj)
            scr = v2.screen_v2(phi, k_phi, s_rows, k_s, sp_, sel["r"], pen, SCREEN_THRESHOLD, proj)
            fact = facts[sid]
            rows_out.append({
                "rotation": rot.index, "split_id": sid, "r": sel["r"],
                "objective": sel["objective"], "objective_min_count": sel["objective_min_count"],
                "screen_d2": scr["d2"], "screen_se": scr["se"], "screen_pass": scr["pass"],
                "overlap": [scr["p_min"], scr["p_max"]], "target_digest": proj["digest"],
                "evaluator_only": {"balance_root": fact["root"], "feasible": fact["feasible"],
                                   "targets_truth": fact["targets_truth"],
                                   "root_error": (None if fact["root"] is None or sel["r"] is None
                                                  else abs(sel["r"] - fact["root"]))}})
            if scr["pass"]:
                passes[sp_] += 1
                scores[sp_].append(hd.aipw_from_features(
                    hd.representation(k_n, n_rows, sp_, sel["r"]), n_rows["A"], n_rows["Y"],
                    hd.representation(k_e, e_rows, sp_, sel["r"]), e_rows["A"], e_rows["Y"]))

    retained = [sp_ for sp_ in splits if passes[sp_] == 4]
    est = {c.split_id(sp_): float(np.concatenate(scores[sp_]).mean()) for sp_ in retained}
    if retained:
        mat = np.column_stack([np.concatenate(scores[sp_]) for sp_ in retained])
        rho = float(v1.ps_common_radius(mat, seed + 5))
        agg = v1.hd_aggregate([c.split_id(sp_) for sp_ in retained],
                              np.array([est[c.split_id(sp_)] for sp_ in retained]), rho)
    else:
        rho, agg = None, {"return_kind": "no_return", "output": None, "members": []}
    baselines = {k: float(np.concatenate(v).mean()) for k, v in base_scores.items()}
    baselines["naive"] = float(view["Y"][view["A"] == 1].mean() - view["Y"][view["A"] == 0].mean())
    return {"data_seed": seed, "rows": rows_out, "baselines": baselines,
            "retained_splits": [c.split_id(s) for s in retained],
            "candidate_estimates": est, "rho": rho, "return_kind": agg["return_kind"],
            "estimate": agg["output"], "complete": True, "timing_seconds": time.time() - t0}


def score(cells: list[dict], facts: dict) -> dict:
    """The three criteria, each measured against the exactly known population facts."""
    rows = [r for cell in cells for r in cell.get("rows", [])]
    with_root = [r for r in rows if r["evaluator_only"]["root_error"] is not None]
    recovered = [r for r in with_root if r["evaluator_only"]["root_error"] <= ROOT_TOLERANCE]
    feasible = [r for r in rows if r["evaluator_only"]["feasible"]]
    infeasible = [r for r in rows if not r["evaluator_only"]["feasible"]]
    tpr = float(np.mean([r["screen_pass"] for r in feasible])) if feasible else float("nan")
    fpr = float(np.mean([r["screen_pass"] for r in infeasible])) if infeasible else float("nan")
    returned = [cell for cell in cells if cell["estimate"] is not None]
    err = np.array([abs(cell["estimate"] - 1.0) for cell in returned])
    out = {
        "selections_with_a_reachable_root": len(with_root),
        "root_recovery_rate": len(recovered) / len(with_root) if with_root else float("nan"),
        "root_error_median": float(np.median([r["evaluator_only"]["root_error"] for r in with_root]))
        if with_root else float("nan"),
        "screen_tpr": tpr, "screen_fpr": fpr,
        "screen_feasible_rows": len(feasible), "screen_infeasible_rows": len(infeasible),
        "return_rate": len(returned) / len(cells),
        "mean_abs_error": float(np.mean(err)) if len(err) else float("nan"),
        "median_abs_error": float(np.median(err)) if len(err) else float("nan"),
        "mean_estimate": float(np.mean([cell["estimate"] for cell in returned])) if returned
        else float("nan"),
    }
    out["gates"] = {
        "root_recovery": out["root_recovery_rate"] >= GATES["root_recovery_rate_min"],
        "screen_tpr": tpr >= GATES["screen_tpr_min"],
        "screen_fpr": fpr <= GATES["screen_fpr_max"],
        "ate_error": out["mean_abs_error"] <= GATES["mean_abs_error_max"],
    }
    out["verdict"] = "PASS" if all(out["gates"].values()) else "FAIL"
    out["confirmation_may_open"] = out["verdict"] == "PASS"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(DEV_SEEDS))
    ap.add_argument("--out", default="../results/scm6_v2_development_v1.json")
    args = ap.parse_args()
    facts = truth()
    print(f"SCM-6 v2 | development only | n={TOTAL_N} | {args.seeds} seeds | "
          f"screen threshold {SCREEN_THRESHOLD:g} | root tolerance {ROOT_TOLERANCE} | "
          f"target {TARGET_COMPONENTS} component(s) | D fold {EMB_FIT}+{EMB_CAL}+{3000-EMB_FIT-EMB_CAL}")
    print(f"  population truth: {sum(f['feasible'] for f in facts.values())} of {len(facts)} splits "
          f"balance, {sum(f['targets_truth'] for f in facts.values())} of those target 1")
    root = {"schema_version": c.SCHEMA_VERSION, "experiment_id": "SCM-6-v2", "phase": "development",
            "mode": "operational_rotate4", "total_n": TOTAL_N, "seeds": DEV_SEEDS[:args.seeds],
            "screen_threshold": SCREEN_THRESHOLD, "root_tolerance": ROOT_TOLERANCE,
            "target_components": TARGET_COMPONENTS, "population_window": list(POPULATION_WINDOW),
            "emb_fit": EMB_FIT, "emb_cal": EMB_CAL, "d_phi": 3000 - EMB_FIT - EMB_CAL,
            "iteration": 3,
            "iteration_reason": "iteration 1 used 1,200 encoder rows and 1,500 for the r search, and "
                                "missed the root recovery gate at 0.706 while the screen was perfect "
                                "and the effect error passed. The selected r was correctly centred on "
                                "every split, within 0.05 of the root, so the shortfall was spread "
                                "rather than bias. 600 rows moved from the encoder to the search. "
                                "Iteration 3 is iteration 2 with comparators added; no PROBE number "
                                "changes because the comparators draw nothing from a random stream.",
            "selection": "cross-fitted nested Brier gap, argmin over the frozen r grid",
            "screen": "mean squared difference of the two fitted propensities on independent rows, "
                      "which is non-negative so a badly generalising critic cannot buy a pass",
            "penalty_grid": list(PENALTY_GRID), "gates": GATES,
            "population_truth": {k: {kk: vv for kk, vv in v.items() if kk != "split"}
                                 for k, v in facts.items()},
            "source_snapshot": c.source_snapshot(SOURCES), "environment": c.environment(),
            "cells": [], "complete": False}
    for seed in DEV_SEEDS[:args.seeds]:
        cell = one_replicate(seed, facts)
        root["cells"].append(cell)
        got = [r for r in cell.get("rows", []) if r["evaluator_only"]["root_error"] is not None]
        near = sum(r["evaluator_only"]["root_error"] <= ROOT_TOLERANCE for r in got)
        print(f"  seed {seed}: {cell['return_kind']} est={cell['estimate']} "
              f"retained={len(cell.get('retained_splits', []))} "
              f"root recovery {near}/{len(got)} ({cell['timing_seconds']:.0f}s)")
    root["summary"] = score(root["cells"], facts)
    root["complete"] = True
    c.write_json_new(args.out, root)
    s = root["summary"]
    print(f"\nroot recovery   {s['root_recovery_rate']:.3f}  (gate >= {GATES['root_recovery_rate_min']})"
          f"   median |r - r*| {s['root_error_median']:.4f}")
    print(f"screen TPR      {s['screen_tpr']:.3f}  (gate >= {GATES['screen_tpr_min']})")
    print(f"screen FPR      {s['screen_fpr']:.3f}  (gate <= {GATES['screen_fpr_max']})")
    print(f"mean |error|    {s['mean_abs_error']:.4f}  (gate <= {GATES['mean_abs_error_max']})"
          f"   mean estimate {s['mean_estimate']:.4f}")
    print(f"verdict {s['verdict']} | confirmation may open: {s['confirmation_may_open']}")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
