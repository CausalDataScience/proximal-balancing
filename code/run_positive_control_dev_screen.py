"""Development study for the two open numbers of the confirmation: n_D and the screen
threshold t (PRD v4, section 4 step 2).

Nothing else is chosen here.  The DGP is already frozen by the preflight.  Seeds come
from the labelled development block 8_200_000 and are never reused by a confirmation.

At each ladder rung the study reports only what the reviewer asked for:
  * the retention rate of the nine intended population candidates,
  * the pass rate of the other 117 splits,
  * whether the plurality margin stays positive on the realised retained set.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import positive_control_scm as pc
from positive_control_scm import Design

RESULTS = Path(__file__).resolve().parents[1] / "results" / "positive_control"
PREFLIGHT = RESULTS / "preflight_v1.json"
OUTPUT = RESULTS / "dev_screen_ladder_v1.json"
OUTPUT_TOP = RESULTS / "dev_screen_top_v1.json"
DEV_SEED_START = 8_200_000
LADDER = (2_000, 5_000, 10_000, 20_000, 50_000)
THRESHOLDS = (5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3)
TAU = 1.0


def frozen_design() -> tuple[Design, dict]:
    payload = json.loads(PREFLIGHT.read_text())
    d = payload["design"]
    design = Design(b_c=tuple(d["b_c"]), b_t=tuple(d["b_t"]), sigma=tuple(d["sigma"]),
                    kappa=d["kappa"], alpha_c=d["alpha_c"], alpha_t=d["alpha_t"],
                    outcome_u=d["outcome_u"], outcome_noise=d["outcome_noise"])
    return design, payload


_WORKER: dict = {}


def _init(design: Design, n_d: int, seed: int) -> None:
    _WORKER["design"] = design
    _WORKER["data"] = pc.generate(design, n_d, seed)


def _one_split(split: tuple[int, ...]) -> dict:
    design, data = _WORKER["design"], _WORKER["data"]
    out = pc.select_and_screen(design, split, data, threshold=float("inf"))
    return {"split": list(split), "objective": out["objective"], "beta": out["beta"],
            "selection_margin": out["selection_margin"]}


def sweep(design: Design, n_d: int, seed: int, workers: int) -> list[dict]:
    splits = pc.all_splits(design.J)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(design, n_d, seed)) as pool:
        return pool.map(_one_split, splits, chunksize=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ladder", default=",".join(str(x) for x in LADDER))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    ladder = tuple(int(x) for x in args.ladder.split(","))

    design, preflight = frozen_design()
    candidates = {tuple(s) for s in preflight["candidates"]["target_splits"]}
    wrong = {tuple(s) for c in preflight["candidates"]["wrong_classes"] for s in c["splits"]}
    population_theta = {}
    for row in preflight["rows"]:
        if row["balanced_count"]:
            population_theta[tuple(row["split"])] = row["theta"][0]
    intended = candidates | wrong
    population_beta = {tuple(r["split"]): r["balanced_beta"][0] for r in preflight["rows"]
                       if r["balanced_count"]}

    rungs = []
    for n_d in ladder:
        per_rep = []
        for rep in range(args.replicates):
            seed = DEV_SEED_START + 1000 * rep
            rows = sweep(design, n_d, seed, args.workers)
            by_split = {tuple(r["split"]): r for r in rows}
            beta_hits = 0
            theta_gap = 0.0
            for s in intended:
                if np.allclose(by_split[s]["beta"], population_beta[s], atol=1e-9):
                    beta_hits += 1
                else:
                    chosen = pc.theta(design, s, np.array(by_split[s]["beta"]))
                    theta_gap = max(theta_gap, abs(chosen - population_theta[s]))
            entry = {"seed": seed, "beta_recovered": beta_hits, "intended": len(intended),
                     "theta_gap_when_missed": theta_gap,
                     "objective_intended_max": max(by_split[s]["objective"] for s in intended),
                     "objective_other_min": min(by_split[s]["objective"] for s in by_split
                                                if s not in intended),
                     "thresholds": {}}
            for t in THRESHOLDS:
                retained = [s for s in by_split if by_split[s]["objective"] <= t]
                kept_target = [s for s in retained if s in candidates]
                kept_wrong = [s for s in retained if s in wrong]
                kept_other = [s for s in retained if s not in intended]
                n = len(retained)
                if n:
                    values = [population_theta.get(s) for s in retained]
                    known = [v for v in values if v is not None]
                    p_tau = len(kept_target) / n
                    counts: dict[float, int] = {}
                    for v in known:
                        if abs(v - TAU) > 1e-8:
                            counts[round(v, 8)] = counts.get(round(v, 8), 0) + 1
                    p_star = max((c / n for c in counts.values()), default=0.0)
                else:
                    p_tau = p_star = 0.0
                entry["thresholds"][f"{t:g}"] = {
                    "retained": n, "target": len(kept_target), "wrong": len(kept_wrong),
                    "unintended": len(kept_other), "p_tau": p_tau, "p_star": p_star,
                    "delta": p_tau - p_star,
                }
            per_rep.append(entry)
        rungs.append({"n_D": n_d, "replicates": per_rep})
        print(f"\nn_D = {n_d:,}   (beta recovered {np.mean([e['beta_recovered'] for e in per_rep]):.1f}/{len(intended)}"
              f", intended objective max {max(e['objective_intended_max'] for e in per_rep):.2e}"
              f", other objective min {min(e['objective_other_min'] for e in per_rep):.2e})")
        print(f"{'t':>8s} {'retained':>9s} {'target':>7s} {'wrong':>6s} {'unintended':>11s} {'Delta':>7s}")
        for t in THRESHOLDS:
            key = f"{t:g}"
            rows_t = [e["thresholds"][key] for e in per_rep]
            print(f"{key:>8s} {np.mean([r['retained'] for r in rows_t]):9.1f} "
                  f"{np.mean([r['target'] for r in rows_t]):7.1f} {np.mean([r['wrong'] for r in rows_t]):6.1f} "
                  f"{np.mean([r['unintended'] for r in rows_t]):11.1f} "
                  f"{np.mean([r['delta'] for r in rows_t]):7.3f}")

    payload = {"complete": True,
               "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
               "purpose": "development only: choose n_D and the screen threshold t",
               "preflight": str(PREFLIGHT), "dev_seed_start": DEV_SEED_START,
               "replicates": args.replicates, "thresholds": list(THRESHOLDS), "rungs": rungs}
    Path(args.output).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
