"""Theorem 1 on the SCM family: prespecified held-out blocks, evaluated on the fits of the sealed family confirmation.

The restricted Brier rotate-4 confirmation (protocol restricted-brier-rotate4-g-confirm-v2) stored, for every split
and rotation, the r its representation fold selected, but an AIPW estimate only in rotations whose screen passed.
This script regenerates the same folds, checks them against the stored fold digests, recomputes every stored
rotation estimate exactly, and then evaluates each prespecified held-out choice in all four rotations at its stored
r.  No representation is selected or refitted; the AIPW nuisances are the same deterministic fits the run used.

Fixed before any fixed-block estimate was computed:
  settings          SCM-1 to SCM-5, all 20 sealed seeds each, n = 6,000
  held-out choices  the eight with a population balancing root: W1 W2 W3 W4 W1W2 W1W3 W2W3 W1W2W3
  reported          W1 (clean) and W4 (contaminated), with adjustment for X and for (X, W)
Block names follow the manuscript: code block j = 1..4 is W_j, code block 0 is W5.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import e8_population as e8
import minimal_suite_common as c
import probe_structured_scm as ps

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
PROTOCOL = RESULTS / "probe_structured_scm_brier_g_confirm_protocol_v2.json"
STORED = {f"SCM-{k}": RESULTS / f"probe_structured_scm_brier_g_confirm_v2_g{k}.json" for k in range(1, 6)}
OUT_PATH = RESULTS / "t1_fixed_block_scm_family_v1.json"
SOURCES = ["probe_structured_scm.py", "e8_population.py", "minimal_suite_common.py", "run_t1_fixed_block_family.py"]
CHOICES = [(1,), (2,), (3,), (4,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
TOL = 1e-9


def name(split: tuple[int, ...]) -> str:
    return "".join(f"W{j if j else 5}" for j in split)


def evaluate(scm: str) -> dict:
    stored = json.loads(STORED[scm].read_text())
    spec = ps.SCM_FAMILY[scm]
    if stored["settings"] != [ps.SCM_FAMILY_NOTES[scm]["legacy_key"]]:
        raise SystemExit(f"{STORED[scm].name} holds {stored['settings']}, not {scm}")
    cells, worst = [], 0.0
    for rep, row in enumerate(stored["rows"]):
        seed_base = stored["seed_start"] + 10 * rep
        if row["fold_seeds"] != [seed_base + j for j in range(4)]:
            raise SystemExit(f"{scm} replicate {rep}: fold seeds do not follow the runner's rule")
        folds = [ps.generate_role(spec, row["fold_n"], seed_base + j) for j in range(4)]
        if [ps.role_sha256(f) for f in folds] != row["fold_sha256"]:
            raise SystemExit(f"{scm} replicate {rep}: regenerated folds differ from the stored digests")
        fits = {tuple(f["split"]): f for f in row["split_fits"]}
        views = []
        for rotation in range(4):
            d_i, s_i, n_i, e_i = [(rotation + j) % 4 for j in range(4)]
            if row["rotations"][rotation]["fold_roles"] != {"D": d_i, "S": s_i, "N": n_i, "E": e_i}:
                raise SystemExit(f"{scm} replicate {rep}: rotation {rotation} roles differ from the runner's rule")
            transform = ps.Transform.fit(folds[d_i], spec.observation)
            views.append((ps.observed_view(folds[n_i], transform, include_outcome=True),
                          ps.observed_view(folds[e_i], transform, include_outcome=True)))
        blocks = {}
        for split in CHOICES:
            rots = fits[split]["rotations"]
            scores, per = [], []
            for rotation, (nuisance, evaluation) in enumerate(views):
                r = rots[rotation]["brier"]["r"]
                s = ps.aipw_scores(nuisance, evaluation, split, r)
                if rots[rotation].get("estimate") is not None:
                    worst = max(worst, abs(float(s.mean()) - rots[rotation]["estimate"]))
                scores.append(s)
                per.append({"rotation": rotation, "r": float(r), "screen_pass": bool(rots[rotation]["screen"]["pass"]),
                            "estimate": float(s.mean())})
            pooled = np.concatenate(scores)
            blocks[name(split)] = {"estimate": float(pooled.mean()),
                                   "se": float(pooled.std(ddof=1) / math.sqrt(len(pooled))),
                                   "passed_screen_all_four": bool(fits[split]["pass_all_four"]), "rotations": per}
        cells.append({"scm": scm, "replicate": rep, "seed_base": seed_base, "n_total": row["total_n"],
                      "probe_output": row["aggregation"]["output"], "baselines": row["baseline_estimates"],
                      "fixed_blocks": blocks})
    population = {}
    for split in CHOICES:
        try:
            roots = [float(r) for r in e8.clean_roots(spec, split) if -1.0 <= r <= 1.0]
            population[name(split)] = {"roots_inside_grid": roots,
                                       "tau_at_roots": [e8.adjusted_effect(spec, split, r) for r in roots]}
        except Exception as exc:  # the quadrature covers the linear observation maps only
            population[name(split)] = {"unavailable": f"{type(exc).__name__}: {exc}"[:200]}
    return {"scm": scm, "cells": cells, "stored_estimate_max_abs_diff": worst, "population": population}


def main() -> int:
    proto = json.loads(PROTOCOL.read_text())
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=5, mp_context=mp.get_context("spawn")) as pool:
        parts = list(pool.map(evaluate, list(STORED)))
    worst = max(p["stored_estimate_max_abs_diff"] for p in parts)
    print(f"folds match stored digests in all 100 replicates | stored rotation estimates reproduced, max diff {worst:.1e}")
    if worst > TOL:
        raise SystemExit("stored estimates were not reproduced; nothing written")
    summary = {}
    for p in parts:
        cs, out = p["cells"], {}
        for k in ("W1", "W4", "W2", "W3", "W1W2", "W1W3", "W2W3", "W1W2W3"):
            v = np.array([q["fixed_blocks"][k]["estimate"] for q in cs])
            out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "mae_vs_tau": float(np.mean(np.abs(v - 1))),
                      "screen_pass_all_four_count": int(sum(q["fixed_blocks"][k]["passed_screen_all_four"] for q in cs))}
        for b in ("X", "raw_XW", "oracle", "naive"):
            v = np.array([q["baselines"][b] for q in cs])
            out[f"baseline_{b}"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                    "mae_vs_tau": float(np.mean(np.abs(v - 1)))}
        summary[p["scm"]] = out
        pop4 = p["population"]["W4"].get("tau_at_roots")
        print(f"  {p['scm']}: X {out['baseline_X']['mean']:+.3f} | (X,W) {out['baseline_raw_XW']['mean']:+.3f} | "
              f"W1 {out['W1']['mean']:+.3f} (MAE {out['W1']['mae_vs_tau']:.3f}) | W4 {out['W4']['mean']:+.3f} "
              f"(population {['%+.3f' % t for t in pop4] if pop4 else 'n/a'}) | other clean MAE "
              f"{min(out[k]['mae_vs_tau'] for k in ('W2','W3','W1W2','W1W3','W2W3','W1W2W3')):.3f}-"
              f"{max(out[k]['mae_vs_tau'] for k in ('W2','W3','W1W2','W1W3','W2W3','W1W2W3')):.3f}")
    c.write_json_new(OUT_PATH, {
        "schema_version": 1, "experiment_id": "T1-fixed-block-family", "phase": "evaluation_of_sealed_confirmation_fits",
        "question": "Theorem thm:exact-proxy-identification for one prespecified held-out block, SCM-1 to SCM-5",
        "decided_before_computing": {"settings": list(STORED), "choices": [name(s) for s in CHOICES],
                                     "reported": ["W1", "W4", "baseline X", "baseline (X,W)"], "n_total": 6000},
        "protocol": {"path": str(PROTOCOL.relative_to(HERE.parent)), "protocol_id": proto["protocol_id"],
                     "sha256": c.file_sha256(PROTOCOL)},
        "stored_results_sha256": {k: c.file_sha256(v) for k, v in STORED.items()},
        "source_snapshot": c.source_snapshot(SOURCES), "environment": c.environment(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stored_estimate_tolerance": TOL, "stored_estimate_max_abs_diff": worst,
        "summary": summary, "population": {p["scm"]: p["population"] for p in parts},
        "cells": [q for p in parts for q in p["cells"]]})
    print(f"written to {OUT_PATH.relative_to(HERE.parent)} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
