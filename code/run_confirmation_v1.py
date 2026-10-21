"""Confirmation of block discovery on data never used before (2026-10-20, memo section 43).

The rule is fixed before this run: discover_blocks.discover_blocks_merged with COUPLING = 0.5 and
probe_any_blocks.configure(J), exactly as in job 3419470.  The data are new:
    Twins replicates 111 to 210 (the figure uses 11 to 110), built by twins_data.generate as before
    SCM-1 to SCM-3, 20 data sets each, seeds 777x0000 + 1000 i (no result file holds a seed 777xxxxx)
For every data set, PROBE runs twice on the same data, fold seed and threshold: with the designed blocks (the sealed
configuration) and with the merged discovered blocks.  Twins also gets the comparison methods of the figure that need
no neural network (run_twins.baselines: X, raw, oracle; family_v2_competitors.proximal_rows: 2SLS; KSPC given W1 and
given W4 as in run_kspc_comparator.run_task, on its 3,000-row subsample); CEVAE is left out for time (its Twins MAE
was 0.074, ten times PROBE's).  Twins and IHDP-style benchmarks use the v3 decision,
SCM the sealed tolerance, as in the figure.  Tasks run in parallel, one per worker; each result file is written when
its task ends; existing files are skipped; none starts after the deadline.

Usage: run_confirmation_v1.py TASKS.json OUTDIR --workers 70 [--deadline EPOCH]
TASKS.json: a list of {"kind": "bench", "bench": "twins" | "ihdp" | "acic", "replicate": r}
                   or {"kind": "scm", "level": "SCM-1", "seed": s, "tolerance": t}.
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
import family_v2_competitors as fc  # noqa: E402
import family_v2_dgp as g  # noqa: E402
import family_v2_probe as pr  # noqa: E402
import noise_scaled as ns  # noqa: E402
import probe_any_blocks as pab  # noqa: E402
import run_benchmarks_discovered as rb  # noqa: E402
import run_kspc_comparator as rk  # noqa: E402
import run_twins as rtw  # noqa: E402
from discover_blocks import discover_blocks_merged  # noqa: E402

ARGS: dict = {}


def stored(probe: dict) -> dict:
    return {"per_split": {nm: {"estimate": rec["estimate"], "screen": [x["screen"] for x in rec["rotations"]]}
                          for nm, rec in probe["records"].items()},
            "split_names": probe["split_names"], "score_crossproduct": probe["score_crossproduct"]}


def merged_view(view: dict) -> tuple[dict, list[dict]]:
    """The merged discovered blocks of view's proxy, and pab configured for their number."""
    w = np.column_stack([view[f"W{j + 1}"] for j in range(5)])
    origin = np.concatenate([np.full(view[f"W{j + 1}"].shape[1], j + 1) for j in range(5)])
    groups = discover_blocks_merged(w, 5, seed=ARGS["seed"])
    pab.configure(len(groups))
    out = {k: view[k] for k in ("X", "A", "Y")}
    comp = []
    for j, idx in enumerate(groups):
        out[f"W{j + 1}"] = w[:, idx]
        comp.append({"size": int(len(idx)), "from": {f"W{b}": int((origin[idx] == b).sum()) for b in range(1, 6)}})
    return out, comp


def run_task(task: dict) -> str:
    name = (f"confirm_{task['bench']}_replicate_{task['replicate']}" if task["kind"] == "bench"
            else f"confirm_{task['level']}_seed{task['seed']}")
    out_path = Path(ARGS["outdir"]) / f"{name}.json"
    if out_path.exists():
        return f"exists {out_path.name}"
    if ARGS["deadline"] is not None and time.time() > ARGS["deadline"]:
        return f"skipped (deadline) {out_path.name}"
    t0 = time.time()
    pab.reset()                         # the generators and the designed run need the sealed configuration
    if task["kind"] == "bench":
        d, seed, level, n, tau = rb.build(task["bench"], task["replicate"])
        view = {k: d[k] for k in ("X", "W1", "W2", "W3", "W4", "W5", "A", "Y")}
        base = rtw.baselines(view, d["U_eval_only"], seed)
        prox = fc.proximal_rows(view)
        import kspc_wrapper as kw
        rows = np.arange(n); sub_seed = rk.SUB_SEED + task["replicate"]
        if n > rk.N_SUB:
            rows = np.sort(np.random.default_rng(sub_seed).choice(n, rk.N_SUB, replace=False))
        kspc = {b: kw.kspc_effect(view["A"][rows], view[b][rows], view["Y"][rows], method="skpv", seed=sub_seed,
                                  x=view["X"][rows])["ate"] for b in ("W1", "W4")}
        designed = stored(pr.run_dataset(view, level, rtw.THRESHOLD, seed))
        v3_designed = ns.decide(designed, n, seed)
        ARGS["seed"] = seed
        mview, comp = merged_view(view)
        merged = stored(pr.run_dataset(mview, level, rtw.THRESHOLD, seed))
        v3_merged = ns.decide(merged, n, seed)
        pab.reset()
        keep = ("floor", "n_retained", "retained", "output")
        rec = {**task, "n": n, "tau": tau, "seed": seed, "baselines": base, "proximal": prox, "kspc": kspc, "blocks": comp,
               "designed": {"probe": designed, "v3": {k: v3_designed[k] for k in keep}},
               "merged": {"probe": merged, "v3": {k: v3_merged[k] for k in keep}}}
        est = {"designed": v3_designed["output"], "merged": v3_merged["output"]}
        truth = tau
    else:
        lv = g.level(task["level"]); seed = task["seed"]
        view = g.learner_view(g.generate(lv, 24_000, seed))
        res_d = pr.run_dataset(view, lv, task["tolerance"], seed + 7)
        st_d = stored(res_d); v3_d = ns.decide(st_d, 24_000, seed)
        ARGS["seed"] = seed
        mview, comp = merged_view(view)
        res_m = pr.run_dataset(mview, lv, task["tolerance"], seed + 7)
        st_m = stored(res_m); v3_m = ns.decide(st_m, 24_000, seed)
        pab.reset()
        rec = {**task, "n": 24_000, "blocks": comp,
               "designed": {"estimate": res_d["estimate"], "retained": res_d["retained"],
                            "split_estimates": {k: v["estimate"] for k, v in res_d["records"].items()},
                            "probe": st_d, "v3": {k: v3_d[k] for k in ("floor", "n_retained", "retained", "output")}},
               "merged": {"estimate": res_m["estimate"], "retained": res_m["retained"],
                          "split_estimates": {k: v["estimate"] for k, v in res_m["records"].items()},
                          "probe": st_m, "v3": {k: v3_m[k] for k in ("floor", "n_retained", "retained", "output")}}}
        est = {"designed": res_d["estimate"], "merged": res_m["estimate"]}
        truth = 1.0
    rec["seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(rec))
    err = {k: (None if v is None else round(v - truth, 4)) for k, v in est.items()}
    return f"{name}: J {len(comp)}, error designed {err['designed']}, merged {err['merged']}, {rec['seconds']} s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks"); ap.add_argument("outdir")
    ap.add_argument("--workers", type=int, default=70); ap.add_argument("--deadline", type=float, default=None)
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    ARGS.update({"outdir": a.outdir, "deadline": a.deadline})
    tasks = json.loads(Path(a.tasks).read_text())
    with mp.get_context("fork").Pool(min(a.workers, len(tasks))) as pool:
        for line in pool.imap_unordered(run_task, tasks):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
