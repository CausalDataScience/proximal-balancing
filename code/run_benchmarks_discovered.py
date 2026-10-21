"""PROBE on Twins, IHDP and ACIC with blocks discovered from the proxy alone (2026-10-19, memo section 43).

Each replicate is built exactly as the runs behind the figure built it (run_twins.run_replicate for Twins,
run_ihdp_acic.run_replicate for IHDP and ACIC): the same real covariates and potential outcomes, the same hidden
covariate, the same generated proxies and treatment, the same fold seed and the same screen threshold.  The only
change is the block split: the six proxy coordinates are pooled and split into five blocks by
discover_blocks.discover_blocks_refined, which reads W only.  PROBE (family_v2_probe.run_dataset) is unchanged,
and the reported estimate is the v3 noise-scaled decision (noise_scaled.decide) that the figure applies to the
stored runs.  Replicates run in parallel, one per worker; each result file is written when it ends; existing files
are skipped; none starts after the deadline.

With --merged the blocks are discover_blocks.discover_blocks_merged (coupled groups merged, so J can be below 5)
and PROBE runs through probe_any_blocks.configure(J).

Usage: run_benchmarks_discovered.py TASKS.json OUTDIR --workers 40 [--deadline EPOCH] [--merged]
TASKS.json: a list of {"bench": "twins" | "ihdp" | "acic", "replicate": r}.
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
import family_v2_probe as pr  # noqa: E402
import hidden_proxy_recipe as hp  # noqa: E402
import ihdp_acic_data as ia  # noqa: E402
import noise_scaled as ns  # noqa: E402
import probe_any_blocks as pab  # noqa: E402
import run_twins as rtw  # noqa: E402
import twins_data as td  # noqa: E402
from discover_blocks import discover_blocks_merged, discover_blocks_refined  # noqa: E402

ARGS: dict = {}
KEYS = ("X", "W1", "W2", "W3", "W4", "W5", "A", "Y")


def build(bench: str, r: int):
    if bench == "twins":
        real = td.load_real()
        d = td.generate(real, r)
        return d, td.REPLICATE_SEED + 1000 * r + rtw.FOLD_SEED_OFFSET, td.level(real["X"].shape[1]), real["n"], real["tau"]
    d = ia.DATASETS[bench]["load"](r)
    return d, ia.DATASETS[bench]["seed"] + 1000 * r + rtw.FOLD_SEED_OFFSET, hp.level(d["d_x"], bench.upper()), d["n"], d["tau"]


def run_task(task: dict) -> str:
    bench, r = task["bench"], task["replicate"]
    out_path = Path(ARGS["outdir"]) / f"{ARGS['tag']}_{bench}_replicate_{r}.json"
    if out_path.exists():
        return f"exists {out_path.name}"
    if ARGS["deadline"] is not None and time.time() > ARGS["deadline"]:
        return f"skipped (deadline) {out_path.name}"
    t0 = time.time()
    if ARGS["merged"]:
        pab.configure(5)            # the proxy recipe reads g.N_BLOCKS; a worker may still hold the J of its last task
    d, seed, level, n, tau = build(bench, r)
    w = np.column_stack([d[f"W{j + 1}"] for j in range(5)])
    origin = np.concatenate([np.full(d[f"W{j + 1}"].shape[1], j + 1) for j in range(5)])
    groups = (discover_blocks_merged if ARGS["merged"] else discover_blocks_refined)(w, 5, seed=seed)
    if ARGS["merged"]:
        pab.configure(len(groups))
    view = {k: d[k] for k in ("X", "A", "Y")}
    comp = []
    for j, idx in enumerate(groups):
        view[f"W{j + 1}"] = w[:, idx]
        comp.append({"columns": idx.tolist(), "from": {f"W{b}": int((origin[idx] == b).sum()) for b in range(1, 6)}})
    probe = pr.run_dataset(view, level, rtw.THRESHOLD, seed)
    stored = {"per_split": {nm: {"estimate": rec["estimate"], "screen": [x["screen"] for x in rec["rotations"]]}
                            for nm, rec in probe["records"].items()},
              "split_names": probe["split_names"], "score_crossproduct": probe["score_crossproduct"]}
    v3 = ns.decide(stored, n, seed)
    rec = {**task, "n": n, "tau": tau, "seed": seed, "blocks": comp, "probe": stored,
           "v3": {k: v3[k] for k in ("floor", "n_retained", "retained", "output")},
           "sealed_rule": {"estimate": probe["estimate"], "retained": probe["retained"]},
           "seconds": round(time.time() - t0, 1)}
    out_path.write_text(json.dumps(rec))
    est = v3["output"]
    return (f"{bench} {r}: J {len(comp)}, blocks {['/'.join(str(c['from'][f'W{b}']) for b in range(1, 6)) for c in comp]}, "
            f"v3 kept {v3['n_retained']}, PROBE {'none' if est is None else f'{est - tau:+.4f} (error)'}, {rec['seconds']} s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks"); ap.add_argument("outdir")
    ap.add_argument("--workers", type=int, default=40); ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--merged", action="store_true")
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    ARGS.update({"outdir": a.outdir, "deadline": a.deadline, "merged": a.merged, "tag": "merged" if a.merged else "refined"})
    tasks = json.loads(Path(a.tasks).read_text())
    with mp.get_context("fork").Pool(min(a.workers, len(tasks))) as pool:
        for line in pool.imap_unordered(run_task, tasks):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
