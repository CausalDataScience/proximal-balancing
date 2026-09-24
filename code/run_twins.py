"""PROBE and the comparators on the Twins benchmark, replicates 1-10, exactly as
memo/2026-09-22-twins-real-outcome-prereg-v1.md fixes them.

    python3 -B run_twins.py run [--workers 3] [--replicates 1-10]
    python3 -B run_twins.py summarise

Each replicate writes one write-once shard results/twins/replicate_r.json; the summary is write-once too.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "3")

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "twins"
PREREG = HERE.parent / "memo" / "2026-09-22-twins-real-outcome-prereg-v1.md"
SOURCES = ["family_v2_dgp.py", "family_v2_probe.py", "family_v2_competitors.py", "probe_structured_scm.py",
           "twins_data.py", "rhc_pooled.py", "run_twins.py"]
SEALED = ["family_v2_dgp.py", "family_v2_probe.py", "family_v2_competitors.py", "probe_structured_scm.py"]
THRESHOLD = 0.0015                                       # SCM-1's sealed value
GRID = [0.0005, 0.001, 0.002, 0.004]                     # sensitivity, fixed in the pre-registration
FOLD_SEED_OFFSET = 7
CONFIRM = range(1, 11)
TARGETS = ("S1", "S2", "S3", "S12", "S13", "S23", "S123")


def shard_path(r: int) -> Path:
    return RESULTS / f"replicate_{r}.json"


def baselines(view: dict, u: np.ndarray, seed: int) -> dict:
    """naive, X only, X and all of W, and the oracle (X, U), with the sealed AIPW on the sealed rotations."""
    import family_v2_probe as pr
    n = len(view["A"])
    fold_rows = pr.folds(n, seed)
    sums = {"X": 0.0, "raw": 0.0, "oracle": 0.0}
    for rot in range(4):
        rl = pr.roles(rot)
        rn, re = fold_rows[rl["N"]], fold_rows[rl["E"]]
        prep = pr.Prep(view, rn)
        on, oe = np.ones((len(rn), 1)), np.ones((len(re), 1))
        w_all = lambda rows: np.column_stack([view[f"W{j + 1}"][rows] for j in range(5)])
        mu, sd = w_all(rn).mean(0), w_all(rn).std(0) + 1e-12
        feats = {"X": (np.column_stack([on, prep.x(view, rn)]), np.column_stack([oe, prep.x(view, re)])),
                 "raw": (np.column_stack([on, prep.x(view, rn), (w_all(rn) - mu) / sd]),
                         np.column_stack([oe, prep.x(view, re), (w_all(re) - mu) / sd])),
                 "oracle": (np.column_stack([on, prep.x(view, rn), u[rn]]), np.column_stack([oe, prep.x(view, re), u[re]]))}
        for k, (fn, fe) in feats.items():
            psi = pr.aipw(fn, view["A"][rn], view["Y"][rn], fe, view["A"][re], view["Y"][re])
            sums[k] += float(psi.mean()) / 4.0
    a, y = view["A"], view["Y"]
    sums["naive"] = float(y[a == 1].mean() - y[a == 0].mean())
    return sums


def probe_rules(probe: dict, n: int, seed: int) -> dict:
    """The sealed per-rotation rule and the pooled rule, at the sealed threshold and over the grid."""
    import rhc_pooled as rp
    names = probe["split_names"]
    out = {}
    for rule_name, upper in (("sealed", rp.sealed_upper), ("pooled", rp.pooled_upper)):
        out[rule_name] = {}
        for t in [THRESHOLD] + GRID:
            keep = [i for i, nm in enumerate(names) if upper(probe["records"][nm]["rotations"]) <= t]
            agg = rp.aggregate_kept(probe, keep, n, seed)
            kept = {names[i] for i in keep}
            out[rule_name][f"{t:.4f}"] = {"retained": sorted(kept), "n_retained": len(keep), **agg,
                                          "tpr": len(kept & set(TARGETS)) / len(TARGETS),
                                          "fpr": len(kept - set(TARGETS)) / (len(names) - len(TARGETS)),
                                          "kept_w4": "S4" in kept, "kept_any_w5": any("5" in nm for nm in kept)}
    return out


def run_replicate(r: int) -> dict:
    import family_v2_competitors as fc
    import family_v2_probe as pr
    import run_family_v2 as rf
    import twins_data as td

    out_path = shard_path(r)
    if out_path.exists():
        return {"replicate": r, "status": "exists"}
    t_all = time.time()
    real = td.load_real()
    d = td.generate(real, r)
    view = {k: d[k] for k in ("X", "W1", "W2", "W3", "W4", "W5", "A", "Y")}
    seed = td.REPLICATE_SEED + 1000 * r + FOLD_SEED_OFFSET
    timing = {}
    t0 = time.time(); base = baselines(view, d["U_eval_only"], seed); timing["baselines"] = round(time.time() - t0, 1)
    t0 = time.time(); prox = fc.proximal_rows(view); timing["proximal"] = round(time.time() - t0, 1)
    t0 = time.time(); probe = pr.run_dataset(view, td.level(real["X"].shape[1]), THRESHOLD, seed)
    timing["probe"] = round(time.time() - t0, 1)
    t0 = time.time(); cevae = fc.cevae_ate(view, seed); timing["cevae"] = round(time.time() - t0, 1)
    shard = {"replicate": r, "n": real["n"], "d_x": real["X"].shape[1], "tau": real["tau"], "seed": seed,
             "naive": d["naive"], "propensity_mean": float(d["propensity"].mean()),
             "baselines": base, "proximal": prox, "cevae": cevae,
             "probe": {"sealed_run": {"return_kind": probe["return_kind"], "estimate": probe["estimate"],
                                      "retained": probe["retained"], "rho": probe["rho"],
                                      "component_sizes": probe["component_sizes"]},
                       "per_split": {nm: {"estimate": rec["estimate"],
                                          "screen": [r_["screen"] for r_ in rec["rotations"]],
                                          "theta": [r_["theta"] for r_ in rec["rotations"]]}
                                     for nm, rec in probe["records"].items()},
                       "rules": probe_rules(probe, real["n"], seed),
                       "score_crossproduct": probe["score_crossproduct"], "split_names": probe["split_names"],
                       "seconds": probe["seconds"]},
             "timing_s": {**timing, "total": round(time.time() - t_all, 1)},
             "prereg_sha256": rf.sha256(PREREG),
             "source_sha256": {f: rf.sha256(HERE / f) for f in SOURCES}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(out_path, "x") as handle:
        handle.write(json.dumps(shard, indent=1, allow_nan=False) + "\n")
    p = shard["probe"]["rules"]["pooled"][f"{THRESHOLD:.4f}"]
    return {"replicate": r, "status": "written", "seconds": shard["timing_s"]["total"],
            "probe_sealed": probe["estimate"], "probe_pooled": p["output"], "retained_pooled": p["retained"],
            "naive": d["naive"], "X": base["X"], "raw": base["raw"], "oracle": base["oracle"],
            "cevae": cevae["ate"], "p2sls": prox["p2sls_given_roles"], "park": prox["park_spc_clean_proxy"]}


def run(workers: int, reps: list[int]) -> int:
    import multiprocessing
    todo = [r for r in reps if not shard_path(r).exists()]
    print(f"Twins: {len(todo)} replicates with {workers} workers", flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_replicate, r): r for r in todo}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                res = fut.result()
                print(f"  replicate {r}: {res['status']} {res.get('seconds')} s | tau -0.0252 | "
                      f"PROBE sealed {res.get('probe_sealed')} pooled {res.get('probe_pooled')} {res.get('retained_pooled')} | "
                      f"naive {res.get('naive'):+.4f} X {res.get('X'):+.4f} raw {res.get('raw'):+.4f} "
                      f"oracle {res.get('oracle'):+.4f} CEVAE {res.get('cevae'):+.4f} P2SLS {res.get('p2sls'):+.4f} "
                      f"Park {res.get('park'):+.4f}" if res["status"] == "written" else f"  replicate {r}: exists",
                      flush=True)
            except Exception:
                failures += 1
                print(f"  replicate {r}: FAILED\n{traceback.format_exc()}", flush=True)
    return 1 if failures else 0


def summarise() -> int:
    import run_family_v2 as rf
    out_path = RESULTS / "summary_v1.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; summaries are write-once")
    shards = [json.loads(shard_path(r).read_text()) for r in CONFIRM if shard_path(r).exists()]
    if len(shards) != len(CONFIRM):
        raise SystemExit(f"{len(shards)} shards of {len(CONFIRM)}; nothing is summarised early")
    tau = shards[0]["tau"]
    key = f"{THRESHOLD:.4f}"
    rows = {"probe_sealed": [s["probe"]["sealed_run"]["estimate"] for s in shards],
            "probe_pooled": [s["probe"]["rules"]["pooled"][key]["output"] for s in shards],
            "naive": [s["naive"] for s in shards], "X": [s["baselines"]["X"] for s in shards],
            "raw": [s["baselines"]["raw"] for s in shards], "oracle": [s["baselines"]["oracle"] for s in shards],
            "cevae": [s["cevae"]["ate"] for s in shards],
            "p2sls_given_roles": [s["proximal"]["p2sls_given_roles"] for s in shards],
            "p2sls_swapped_roles": [s["proximal"]["p2sls_swapped_roles"] for s in shards],
            "park_spc_clean_proxy": [s["proximal"]["park_spc_clean_proxy"] for s in shards],
            "park_spc_contaminated_proxy": [s["proximal"]["park_spc_contaminated_proxy"] for s in shards]}

    def stats(values):
        ok = [v for v in values if v is not None]
        err = [abs(v - tau) for v in ok]
        return {"values": ok, "returned": len(ok), "mean": float(np.mean(ok)) if ok else None,
                "mae": float(np.mean(err)) if err else None, "p90": float(np.quantile(err, 0.9)) if err else None}

    def paired_upper(a, b):
        d = np.array([abs(x - tau) - abs(y - tau) for x, y in zip(a, b) if x is not None and y is not None])
        from scipy.stats import t as tdist
        return float(d.mean() + tdist.ppf(0.95, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))

    summary = {"artifact": "twins_summary", "generated_utc": rf.now(), "tau": tau, "n": shards[0]["n"],
               "replicates": [s["replicate"] for s in shards], "threshold": THRESHOLD,
               "methods": {k: stats(v) for k, v in rows.items()},
               "paired_upper_pooled_minus": {k: paired_upper(rows["probe_pooled"], rows[k]) for k in ("X", "raw", "cevae", "oracle")},
               "screen_pooled": {"tpr": float(np.mean([s["probe"]["rules"]["pooled"][key]["tpr"] for s in shards])),
                                 "fpr": float(np.mean([s["probe"]["rules"]["pooled"][key]["fpr"] for s in shards])),
                                 "kept_w4": int(sum(s["probe"]["rules"]["pooled"][key]["kept_w4"] for s in shards)),
                                 "kept_any_w5": int(sum(s["probe"]["rules"]["pooled"][key]["kept_any_w5"] for s in shards)),
                                 "return_kinds": [s["probe"]["rules"]["pooled"][key]["return_kind"] for s in shards]},
               "screen_sealed": {"tpr": float(np.mean([s["probe"]["rules"]["sealed"][key]["tpr"] for s in shards])),
                                 "fpr": float(np.mean([s["probe"]["rules"]["sealed"][key]["fpr"] for s in shards])),
                                 "return_kinds": [s["probe"]["rules"]["sealed"][key]["return_kind"] for s in shards]},
               "grid_pooled": {f"{t:.4f}": stats([s["probe"]["rules"]["pooled"][f"{t:.4f}"]["output"] for s in shards])
                               for t in GRID},
               "prereg_sha256": rf.sha256(PREREG), "shard_sha256": {shard_path(r).name: rf.sha256(shard_path(r)) for r in CONFIRM}}
    with open(out_path, "x") as handle:
        handle.write(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    for k, v in summary["methods"].items():
        print(f"{k:28s} returned {v['returned']:2d}/10  mean {v['mean']:+.4f}  MAE {v['mae']:.4f}  p90 {v['p90']:.4f}")
    print("paired upper (pooled PROBE minus):", {k: round(v, 4) for k, v in summary["paired_upper_pooled_minus"].items()})
    print("screen pooled:", summary["screen_pooled"]); print("screen sealed:", summary["screen_sealed"])
    print(f"written {out_path}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "summarise"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--replicates", default="1-10")
    args = ap.parse_args(argv[1:])
    if args.command == "summarise":
        return summarise()
    lo, hi = (int(v) for v in args.replicates.split("-"))
    return run(args.workers, list(range(lo, hi + 1)))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
