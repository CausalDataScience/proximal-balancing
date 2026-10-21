"""Parallel driver for run_random_subsets.py (DeltaAI, plan 1 of memo section 43, 2026-10-19).

Same design and output as run_random_subsets.py, but the 4 rotations x (up to) 30 held-out sets of one data set run
as separate units on a pool of worker processes, so that no single data set holds the allocation for long.  Data
sets run one after another in the order of the task file (round robin over the five conditions, replicate by
replicate), each result file is written as soon as its data set finishes, existing files are skipped, and no data set
starts after the deadline.

Usage: run_random_subsets_parallel.py TASKS.json OUTDIR --workers 70 [--deadline EPOCH] [--n N] [--budget B]
TASKS.json: a list of {"mode", "level", "seed", "tolerance", "designed_estimate"}.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import family_v2_dgp as g  # noqa: E402
import family_v2_probe as pr  # noqa: E402
import probe_structured_scm as ps  # noqa: E402
import run_random_subsets as rs  # noqa: E402

_D: dict = {}       # the current data set, set in the parent before the pool forks


def _unit(job: tuple[int, int]) -> tuple[int, int, bool, float, np.ndarray]:
    rot, s_idx = job
    d = _D
    split = d["splits"][s_idx]
    held = list(split); kept = [j for j in range(d["m"]) if j not in split]
    x, w, a, y = d["x"][rot], d["w"][rot], d["a"][rot], d["y"][rot]
    learned = pr.learn_representation(x["D"], w["D"][:, kept], w["D"][:, held], a["D"], d["k"], d["fold_seed"] + 1_000 * rot + s_idx)
    b = learned["b"]
    scr = pr.screen(x["S"], w["S"][:, kept], w["S"][:, held], a["S"], b, d["tol"])
    f = {r: np.column_stack([np.ones(len(a[r])), x[r], w[r][:, kept] @ b.T]) for r in ("N", "E")}
    psi = pr.aipw(f["N"], a["N"], y["N"], f["E"], a["E"], y["E"])
    return rot, s_idx, bool(scr["pass"]), float(psi.mean()), psi


def run_task(task: dict, workers: int, n: int, budget: int) -> dict:
    t0 = time.time()
    lv = g.level(task["level"]); seed = task["seed"]; fold_seed = seed + 7
    view = g.learner_view(g.generate(lv, n, seed))
    origin = np.concatenate([np.full(view[f"W{j + 1}"].shape[1], j + 1) for j in range(g.N_BLOCKS)])
    rs.BUDGET = budget
    splits = rs.sample_splits(len(origin), task["mode"], np.random.default_rng(seed + 13))
    w_all = np.column_stack([view[f"W{j + 1}"] for j in range(g.N_BLOCKS)])
    fold_rows = pr.folds(n, fold_seed)
    xs, ws, as_, ys, rows_by_rot = [], [], [], [], []
    for rot in range(4):
        rows = {k: fold_rows[v] for k, v in pr.roles(rot).items()}
        dr = rows["D"]
        mx, sx = view["X"][dr].mean(0), view["X"][dr].std(0) + 1e-12
        mw, sw = w_all[dr].mean(0), w_all[dr].std(0) + 1e-12
        xs.append({r: (view["X"][i] - mx) / sx for r, i in rows.items()})
        ws.append({r: (w_all[i] - mw) / sw for r, i in rows.items()})
        as_.append({r: view["A"][i] for r, i in rows.items()})
        ys.append({r: view["Y"][i] for r, i in rows.items()})
        rows_by_rot.append(rows)
    _D.clear()
    _D.update({"splits": splits, "m": int(len(origin)), "x": xs, "w": ws, "a": as_, "y": ys, "k": lv.k,
               "fold_seed": fold_seed, "tol": task["tolerance"]})
    jobs = [(rot, s) for rot in range(4) for s in range(len(splits))]
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
        out = pool.map(_unit, jobs, chunksize=1)
    scores = np.zeros((n, len(splits)))
    rot_recs = [[None] * 4 for _ in splits]
    for rot, s_idx, ok, theta, psi in out:
        scores[rows_by_rot[rot]["E"], s_idx] = psi
        rot_recs[s_idx][rot] = {"pass": ok, "theta": theta}
    recs = []
    for split, rr in zip(splits, rot_recs):
        counts = {f"W{b}": int((origin[list(split)] == b).sum()) for b in range(1, g.N_BLOCKS + 1)}
        recs.append({"held_out": list(split), "size": len(split), "from": counts,
                     "clean": counts["W4"] == 0 and counts["W5"] == 0,
                     "pass_all": all(r["pass"] for r in rr), "estimate": float(np.mean([r["theta"] for r in rr]))})
    retained = [i for i, r in enumerate(recs) if r["pass_all"]]
    if retained:
        rho = float(ps.common_radius(scores[:, retained], fold_seed + 5))
        agg = ps.aggregate([splits[i] for i in retained], np.array([recs[i]["estimate"] for i in retained]), rho)
    else:
        rho, agg = None, {"return_kind": "no_return", "output": None}
    return {**task, "m": int(len(origin)), "n": n, "budget": budget, "splits": recs, "retained": retained, "rho": rho,
            "estimate": agg["output"], "return_kind": agg["return_kind"],
            "component_sizes": agg.get("component_sizes"), "seconds": round(time.time() - t0, 1), "workers": workers}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks"); ap.add_argument("outdir")
    ap.add_argument("--workers", type=int, default=70); ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--n", type=int, default=24_000); ap.add_argument("--budget", type=int, default=30)
    a = ap.parse_args()
    tasks = json.loads(Path(a.tasks).read_text())
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        path = out / f"{t['mode']}_{t['level']}_seed{t['seed']}.json"
        if path.exists():
            continue
        if a.deadline is not None and time.time() > a.deadline:
            print("deadline reached; stopping before", path.name, flush=True); break
        res = run_task(t, a.workers, a.n, a.budget)
        path.write_text(json.dumps(res))
        sp = res["splits"]; kept = [sp[i] for i in res["retained"]]
        est = res["estimate"]
        print(f"{t['mode']:7s} {t['level']} seed {t['seed']}: clean {sum(s['clean'] for s in sp)}/{len(sp)}, retained {len(kept)} "
              f"({sum(s['clean'] for s in kept)} clean), PROBE {'none' if est is None else f'{est:.3f}'}, designed "
              f"{t['designed_estimate']:.3f}, {res['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
