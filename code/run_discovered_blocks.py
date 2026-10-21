"""PROBE on SCM-1 to SCM-3 with blocks discovered from the proxy alone (2026-10-19, memo section 43, the fix).

For each data set: the same data, fold seed and sealed tolerance as the extension run E1; the blocks are not taken
from the design but found by discover_blocks.discover_blocks from W only (no A, no Y), with J = 5 groups; PROBE
(family_v2_probe.run_dataset, Algorithm 1 over the 30 splits of the 5 discovered blocks) is unchanged.  The evaluator
records how many coordinates of each designed block every discovered block holds.  Data sets run in parallel, one per
worker process, each writing its result file when it ends; existing files are skipped; none starts after the deadline.

With --merged: discover_blocks.discover_blocks_merged (refined, then coupled groups merged) and PROBE through
probe_any_blocks.configure(J).

Usage: run_discovered_blocks.py TASKS.json OUTDIR --workers 60 [--deadline EPOCH] [--n N] [--refined | --merged]
TASKS.json: a list of {"level", "seed", "tolerance", "designed_estimate"}.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import family_v2_dgp as g  # noqa: E402
import family_v2_probe as pr  # noqa: E402
import noise_scaled as ns  # noqa: E402
import probe_any_blocks as pab  # noqa: E402
from discover_blocks import discover_blocks, discover_blocks_merged, discover_blocks_refined  # noqa: E402

ARGS: dict = {}


def run_task(task: dict) -> str:
    out_path = Path(ARGS["outdir"]) / f"{ARGS['tag']}_{task['level']}_seed{task['seed']}.json"
    if out_path.exists():
        return f"exists {out_path.name}"
    if ARGS["deadline"] is not None and time.time() > ARGS["deadline"]:
        return f"skipped (deadline) {out_path.name}"
    t0 = time.time()
    if ARGS["merged"]:
        pab.configure(5)            # the generator reads g.N_BLOCKS; a worker may still hold the J of its last task
    lv = g.level(task["level"]); seed = task["seed"]
    view = g.learner_view(g.generate(lv, ARGS["n"], seed))
    w = np.column_stack([view[f"W{j + 1}"] for j in range(5)])
    origin = np.concatenate([np.full(view[f"W{j + 1}"].shape[1], j + 1) for j in range(5)])
    if ARGS["merged"]:
        groups = discover_blocks_merged(w, 5, seed=seed, penalty=ARGS["penalty"])
        pab.configure(len(groups))
    else:
        groups = (discover_blocks_refined if ARGS["refined"] else discover_blocks)(w, g.N_BLOCKS, seed=seed)
    t_disc = time.time() - t0
    dview = {k: view[k] for k in ("X", "A", "Y")}
    comp = []
    for j, idx in enumerate(groups):
        dview[f"W{j + 1}"] = w[:, idx]
        counts = {f"W{b}": int((origin[idx] == b).sum()) for b in range(1, 6)}
        comp.append({"columns": idx.tolist(), "from": counts, "size": int(len(idx)),
                     "has_W4": counts["W4"] > 0, "has_W5": counts["W5"] > 0})
    res = pr.run_dataset(dview, lv, task["tolerance"], seed + 7)
    rec = {**task, "n": ARGS["n"], "blocks": comp, "discovery_seconds": round(t_disc, 1),
           "estimate": res["estimate"], "return_kind": res["return_kind"], "retained": res["retained"],
           "members": res["members"], "component_sizes": res["component_sizes"], "rho": res["rho"],
           "split_estimates": {k: v["estimate"] for k, v in res["records"].items()},
           "split_pass": {k: v["pass_all"] for k, v in res["records"].items()}, "seconds": round(time.time() - t0, 1)}
    # the screen statistics, so that the v3 rule (noise_scaled.decide) can be applied as on the benchmarks (2026-10-20)
    rec["probe"] = {"per_split": {nm: {"estimate": r["estimate"], "screen": [x["screen"] for x in r["rotations"]]}
                                  for nm, r in res["records"].items()},
                    "split_names": res["split_names"], "score_crossproduct": res["score_crossproduct"]}
    v3 = ns.decide(rec["probe"], ARGS["n"], seed)
    rec["v3"] = {k: v3[k] for k in ("floor", "n_retained", "retained", "output")}
    out_path.write_text(json.dumps(rec))
    est = rec["estimate"]
    return (f"{ARGS['tag']} {task['level']} seed {seed}: blocks {['/'.join(str(c['from'][f'W{b}']) for b in range(1, 6)) for c in comp]}, "
            f"retained {len(res['retained'])}, PROBE {'none' if est is None else f'{est:.3f}'}, designed {task['designed_estimate']:.3f}, "
            f"{rec['seconds']} s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks"); ap.add_argument("outdir")
    ap.add_argument("--workers", type=int, default=60); ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--n", type=int, default=24_000)
    ap.add_argument("--refined", action="store_true", help="refine the spectral partition by k-subspaces (discover_blocks_refined)")
    ap.add_argument("--merged", action="store_true", help="refined, then merge coupled groups (discover_blocks_merged)")
    ap.add_argument("--penalty", type=float, default=None, help="with --merged: likelihood-ratio refinement (discover_blocks.refine_by_subspace)")
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    ARGS.update({"outdir": a.outdir, "deadline": a.deadline, "n": a.n, "refined": a.refined, "merged": a.merged, "penalty": a.penalty,
                 "tag": ("lrt" if a.penalty is not None else "merged") if a.merged else ("refined" if a.refined else "discovered")})
    tasks = json.loads(Path(a.tasks).read_text())
    with mp.get_context("fork").Pool(min(a.workers, len(tasks))) as pool:
        for line in pool.imap_unordered(run_task, tasks):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
