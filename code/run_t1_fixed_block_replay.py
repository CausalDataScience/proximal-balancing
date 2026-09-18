"""Theorem 1 on SCM-1: prespecified held-out blocks, evaluated on the fits of the sealed E12 confirmation.

Theorem `thm:exact-proxy-identification` is a statement about one prespecified block.  The E12 record cannot
show it directly: the rotating run computes a split's AIPW score only in rotations where that split passes the
balance screen, and it stores only the aggregated output.  This script replays the E12 confirmation cells at
n = 12,000 with the same seeds, folds and Brier selection, checks that the replay reproduces every stored
output, and then evaluates each prespecified held-out choice in all four rotations at the r that rotation's
representation fold selected.  There is no screen, no search and no refit: every r is the sealed run's choice.

Fixed before any fixed-block estimate was computed:
  held-out choices  the eight with a population balancing root: W1 W2 W3 W4 W1W2 W1W3 W2W3 W1W2W3
  figure            W1 (clean) and W4 (contaminated), with adjustment for X as the reference
  sample            n = 12,000, the largest E12 size, all 20 sealed seeds
Block names follow the manuscript: W1-W3 clean, W4 contaminated, W5 = (I, C, M_0).  Code block j = 1..4 is
W_j and code block 0 is W5; none of the eight choices contains it.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import e8_population as e8
import minimal_suite_common as c
import probe_structured_scm as ps

E12_PROTOCOL = "../results/e12_scm1_ncurve_protocol_v2.json"
E12_RESULT = "../results/e12_scm1_ncurve_confirm_v2.json"
OUT_PATH = "../results/t1_fixed_block_scm1_v1.json"
E12_SOURCES = ["minimal_suite_common.py", "run_e12_scm1_ncurve.py", "probe_structured_scm.py", "e8_population.py"]
SOURCES = E12_SOURCES + ["run_t1_fixed_block_replay.py"]
N_TOTAL = 12000
CHOICES = [(1,), (2,), (3,), (4,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
REPLAY_TOL = 1e-9


def name(split: tuple[int, ...]) -> str:
    return "".join(f"W{j if j else 5}" for j in split)


def replay(seed: int) -> dict:
    spec = ps.SCM_FAMILY["SCM-1"]
    t0 = time.time()
    # exactly the E12 folds: one draw per fold at 3,000 rows, which n = 12,000 uses whole
    folds = [ps.slice_role(ps.generate_role(spec, N_TOTAL // 4, seed + j), N_TOTAL // 4) for j in range(4)]
    rep = ps.run_rotating_replicate(spec, N_TOTAL, seed, learner="brier", folds=folds)
    fits = {tuple(f["split"]): f for f in rep["split_fits"]}
    views = []
    for rotation in range(4):
        d_i, _, n_i, e_i = [(rotation + j) % 4 for j in range(4)]
        transform = ps.Transform.fit(folds[d_i], spec.observation)
        views.append((ps.observed_view(folds[n_i], transform, include_outcome=True),
                      ps.observed_view(folds[e_i], transform, include_outcome=True)))
    blocks, worst = {}, 0.0
    for split in CHOICES:
        rows = fits[split]["rotations"]
        scores, rotations = [], []
        for rotation, (nuisance, evaluation) in enumerate(views):
            r = rows[rotation]["brier"]["r"]
            s = ps.aipw_scores(nuisance, evaluation, split, r)
            if rows[rotation].get("estimate") is not None:
                worst = max(worst, abs(float(s.mean()) - rows[rotation]["estimate"]))
            scores.append(s)
            rotations.append({"rotation": rotation, "r": float(r), "screen_pass": bool(rows[rotation]["screen"]["pass"]),
                              "estimate": float(s.mean()), "population_theta": e8.adjusted_effect(spec, split, r)})
        pooled = np.concatenate(scores)
        blocks[name(split)] = {"estimate": float(pooled.mean()),
                               "se": float(pooled.std(ddof=1) / math.sqrt(len(pooled))),
                               "passed_screen_all_four": bool(fits[split]["pass_all_four"]),
                               "rotations": rotations}
    return {"data_seed": seed, "n_total": N_TOTAL, "probe_output": rep["aggregation"]["output"],
            "return_kind": rep["aggregation"]["return_kind"], "baselines": rep["baseline_estimates"],
            "rotation_estimate_replay_max_abs_diff": worst, "fixed_blocks": blocks,
            "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=None, help="first k sealed seeds; a smoke run writes nothing")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()
    proto = json.loads(Path(E12_PROTOCOL).read_text())
    if c.canonical_digest(proto) != proto[c.DIGEST_FIELD]:
        raise SystemExit("the E12 protocol does not match its own digest")
    snap = c.source_snapshot(E12_SOURCES)
    drift = sorted(f for f, d in proto["source_snapshot"]["files"].items() if snap["files"].get(f) != d)
    stored = {q["data_seed"]: q for q in json.loads(Path(E12_RESULT).read_text())["cells"] if q["n_total"] == N_TOTAL}
    seeds = proto["data_seeds"][: args.seeds]
    print(f"T1 fixed-block replay | SCM-1 | n = {N_TOTAL} | {len(seeds)} sealed E12 seeds | "
          f"E12 protocol {proto[c.DIGEST_FIELD][:12]} | source drift since seal: {drift or 'none'}", flush=True)
    t0, cells = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(replay, s) for s in seeds]
        for fut in as_completed(futures):
            cell = fut.result()
            ref = stored[cell["data_seed"]]
            diffs = [abs(cell["baselines"][k] - ref["baselines"][k]) for k in ref["baselines"]]
            if (ref["estimate"] is None) != (cell["probe_output"] is None):
                diffs.append(math.inf)
            elif ref["estimate"] is not None:
                diffs.append(abs(cell["probe_output"] - ref["estimate"]))
            cell["stored_output_replay_max_abs_diff"] = max(diffs)
            cells.append(cell)
            fb = cell["fixed_blocks"]
            print(f"  seed {cell['data_seed']}: replay diff {max(diffs):.1e}/{cell['rotation_estimate_replay_max_abs_diff']:.1e}"
                  f" | W1 {fb['W1']['estimate']:+.3f} W4 {fb['W4']['estimate']:+.3f} X {cell['baselines']['X']:+.3f}"
                  f" | {cell['seconds']}s | {len(cells)}/{len(seeds)} at {time.time() - t0:.0f}s", flush=True)
    cells.sort(key=lambda q: q["data_seed"])
    worst = max(max(q["stored_output_replay_max_abs_diff"], q["rotation_estimate_replay_max_abs_diff"]) for q in cells)
    if worst > REPLAY_TOL:
        raise SystemExit(f"the replay does not reproduce the sealed E12 run (max diff {worst:.2e}); nothing written")
    spec = ps.SCM_FAMILY["SCM-1"]
    population, summary = {}, {}
    for split in CHOICES:
        roots = [float(r) for r in e8.clean_roots(spec, split)]
        inside = [r for r in roots if -1.0 <= r <= 1.0]
        population[name(split)] = {"roots": roots, "roots_inside_grid": inside,
                                   "tau_at_roots_inside_grid": [e8.adjusted_effect(spec, split, r) for r in inside],
                                   "d_res_at_roots_inside_grid": [e8.residual_discrepancy(spec, split, r) for r in inside]}
    for split in CHOICES:
        k = name(split)
        est = np.array([q["fixed_blocks"][k]["estimate"] for q in cells])
        theta = np.array([np.mean([r["population_theta"] for r in q["fixed_blocks"][k]["rotations"]]) for q in cells])
        summary[k] = {"mean": float(est.mean()), "sd": float(est.std(ddof=1)), "median": float(np.median(est)),
                      "mae_vs_tau": float(np.mean(np.abs(est - 1.0))),
                      "mean_abs_estimate_minus_selected_theta": float(np.mean(np.abs(est - theta))),
                      "population_value": population[k]["tau_at_roots_inside_grid"],
                      "screen_pass_all_four_count": int(sum(q["fixed_blocks"][k]["passed_screen_all_four"] for q in cells))}
    for b in ("X", "raw_XW", "oracle", "naive"):
        v = np.array([q["baselines"][b] for q in cells])
        summary[f"baseline_{b}"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                    "mae_vs_tau": float(np.mean(np.abs(v - 1.0)))}
    for k, v in summary.items():
        extra = f"  population {['%+.4f' % t for t in v['population_value']]}  screen 4/4 in {v['screen_pass_all_four_count']}" \
            if "population_value" in v else ""
        print(f"  {k:16s} mean {v['mean']:+.4f}  sd {v['sd']:.4f}  MAE vs tau {v['mae_vs_tau']:.4f}{extra}")
    if args.seeds is not None:
        print("smoke run; nothing written")
        return 0
    c.write_json_new(args.out, {
        "schema_version": 1, "experiment_id": "T1-fixed-block-replay", "scm": "SCM-1",
        "phase": "evaluation_of_sealed_confirmation_fits",
        "question": "Theorem thm:exact-proxy-identification for one prespecified held-out block",
        "decided_before_computing": {"choices": [name(s) for s in CHOICES], "figure": ["W1", "W4", "baseline X"],
                                     "n_total": N_TOTAL, "seeds": "all 20 sealed E12 seeds"},
        "e12_protocol": {"path": E12_PROTOCOL, c.DIGEST_FIELD: proto[c.DIGEST_FIELD]},
        "e12_result_sha256": c.file_sha256(E12_RESULT), "e12_source_drift_since_seal": drift,
        "source_snapshot": c.source_snapshot(SOURCES), "environment": c.environment(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "replay_tolerance": REPLAY_TOL, "replay_max_abs_diff": worst,
        "population": population, "summary": summary, "cells": cells})
    print(f"written to {args.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
