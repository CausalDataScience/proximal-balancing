"""PROBE and the comparators on IHDP and ACIC 2016, exactly as
memo/2026-09-22-ihdp-acic-prereg-v1.md fixes them.

    python3 -B run_ihdp_acic.py run ihdp [--workers 2]
    python3 -B run_ihdp_acic.py summarise ihdp

The scoring helpers come from run_twins so that Twins, RHC-planted, IHDP and ACIC are all scored by the
same code.  Replication 1 of each was the development check and is excluded.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import hidden_proxy_recipe as hp
import ihdp_acic_data as ia
import run_twins as rtw

HERE = Path(__file__).resolve().parent
PREREG = HERE.parent / "memo" / "2026-09-22-ihdp-acic-prereg-v1.md"
SOURCES = ["family_v2_dgp.py", "family_v2_probe.py", "family_v2_competitors.py", "probe_structured_scm.py",
           "hidden_proxy_recipe.py", "ihdp_acic_data.py", "rhc_pooled.py", "run_twins.py", "run_ihdp_acic.py"]
KEYS = ("X", "W1", "W2", "W3", "W4", "W5", "A", "Y")


def results_dir(name: str) -> Path:
    return HERE.parent / "results" / name


def shard_path(name: str, r: int) -> Path:
    return results_dir(name) / f"replicate_{r}.json"


def run_replicate(name: str, r: int) -> dict:
    import family_v2_competitors as fc
    import family_v2_probe as pr
    import run_family_v2 as rf

    out_path = shard_path(name, r)
    if out_path.exists():
        return {"replicate": r, "status": "exists"}
    t_all = time.time()
    d = ia.DATASETS[name]["load"](r)
    view = {k: d[k] for k in KEYS}
    seed = ia.DATASETS[name]["seed"] + 1000 * r + rtw.FOLD_SEED_OFFSET
    timing = {}
    t0 = time.time(); base = rtw.baselines(view, d["U_eval_only"], seed); timing["baselines"] = round(time.time() - t0, 1)
    t0 = time.time(); prox = fc.proximal_rows(view); timing["proximal"] = round(time.time() - t0, 1)
    t0 = time.time(); probe = pr.run_dataset(view, hp.level(d["d_x"], name.upper()), rtw.THRESHOLD, seed)
    timing["probe"] = round(time.time() - t0, 1)
    t0 = time.time(); cevae = fc.cevae_ate(view, seed); timing["cevae"] = round(time.time() - t0, 1)
    shard = {"dataset": name, "replicate": r, "n": d["n"], "d_x": d["d_x"], "tau": d["tau"], "tau_mu": d["tau_mu"],
             "seed": seed, "naive": d["naive"], "propensity_mean": float(d["propensity"].mean()),
             "block_information": hp.block_information(d),
             "baselines": base, "proximal": prox, "cevae": cevae,
             "probe": {"sealed_run": {"return_kind": probe["return_kind"], "estimate": probe["estimate"],
                                      "retained": probe["retained"], "rho": probe["rho"],
                                      "component_sizes": probe["component_sizes"]},
                       "per_split": {nm: {"estimate": rec["estimate"],
                                          "screen": [x["screen"] for x in rec["rotations"]]}
                                     for nm, rec in probe["records"].items()},
                       "rules": rtw.probe_rules(probe, d["n"], seed),
                       "score_crossproduct": probe["score_crossproduct"], "split_names": probe["split_names"],
                       "seconds": probe["seconds"]},
             "timing_s": {**timing, "total": round(time.time() - t_all, 1)},
             "prereg_sha256": rf.sha256(PREREG),
             "source_sha256": {f: rf.sha256(HERE / f) for f in SOURCES}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "x") as handle:
        handle.write(json.dumps(shard, indent=1, allow_nan=False) + "\n")
    p = shard["probe"]["rules"]["pooled"][f"{rtw.THRESHOLD:.4f}"]
    return {"replicate": r, "status": "written", "seconds": shard["timing_s"]["total"], "tau": d["tau"],
            "probe_pooled": p["output"], "n_retained": p["n_retained"], "naive": d["naive"],
            "X": base["X"], "raw": base["raw"], "oracle": base["oracle"], "cevae": cevae["ate"]}


def run(name: str, workers: int) -> int:
    import multiprocessing
    reps = list(ia.DATASETS[name]["reps"])
    todo = [r for r in reps if not shard_path(name, r).exists()]
    print(f"{name}: {len(todo)} replicates with {workers} workers", flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_replicate, name, r): r for r in todo}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                res = fut.result()
                if res["status"] == "written":
                    shown = "none" if res["probe_pooled"] is None else f"{res['probe_pooled']:+.4f}"
                    print(f"  {name} {r}: {res['seconds']} s | tau {res['tau']:+.4f} | PROBE {shown} "
                          f"({res['n_retained']} kept) | naive {res['naive']:+.4f} X {res['X']:+.4f} "
                          f"raw {res['raw']:+.4f} oracle {res['oracle']:+.4f} CEVAE {res['cevae']:+.4f}", flush=True)
            except Exception:
                failures += 1
                print(f"  {name} {r}: FAILED\n{traceback.format_exc()}", flush=True)
    return 1 if failures else 0


def summarise(name: str) -> int:
    import run_family_v2 as rf
    from scipy.stats import t as tdist
    out_path = results_dir(name) / "summary_v1.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; summaries are write-once")
    reps = list(ia.DATASETS[name]["reps"])
    shards = [json.loads(shard_path(name, r).read_text()) for r in reps if shard_path(name, r).exists()]
    if len(shards) != len(reps):
        raise SystemExit(f"{len(shards)} shards of {len(reps)}; nothing is summarised early")
    key = f"{rtw.THRESHOLD:.4f}"
    taus = [s["tau"] for s in shards]
    rows = {"probe_sealed": [s["probe"]["sealed_run"]["estimate"] for s in shards],
            "probe_pooled": [s["probe"]["rules"]["pooled"][key]["output"] for s in shards],
            "naive": [s["naive"] for s in shards], "X": [s["baselines"]["X"] for s in shards],
            "raw": [s["baselines"]["raw"] for s in shards], "oracle": [s["baselines"]["oracle"] for s in shards],
            "cevae": [s["cevae"]["ate"] for s in shards],
            "p2sls_given_roles": [s["proximal"]["p2sls_given_roles"] for s in shards],
            "p2sls_swapped_roles": [s["proximal"]["p2sls_swapped_roles"] for s in shards],
            "park_spc_clean_proxy": [s["proximal"]["park_spc_clean_proxy"] for s in shards]}

    def stats(values):
        pairs = [(v, t) for v, t in zip(values, taus) if v is not None]
        err = [abs(v - t) for v, t in pairs]
        return {"values": [v for v, _ in pairs], "returned": len(pairs),
                "mean": float(np.mean([v for v, _ in pairs])) if pairs else None,
                "mae": float(np.mean(err)) if err else None,
                "p90": float(np.quantile(err, 0.9)) if err else None}

    def paired_upper(a, b):
        dd = np.array([abs(x - t) - abs(y - t) for x, y, t in zip(a, b, taus) if x is not None and y is not None])
        if len(dd) < 2:
            return None
        return float(dd.mean() + tdist.ppf(0.95, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd)))

    summary = {"artifact": f"{name}_summary", "generated_utc": rf.now(), "dataset": name,
               "hidden_confounder": ia.DATASETS[name]["hidden"], "n": shards[0]["n"],
               "taus": taus, "tau_mean": float(np.mean(taus)),
               "replicates": [s["replicate"] for s in shards], "threshold": rtw.THRESHOLD,
               "development_replicate_excluded": ia.DEVELOPMENT,
               "block_information_mean": {k: float(np.mean([s["block_information"][k] for s in shards]))
                                          for k in ("W1", "W2", "W3", "W4", "W5")},
               "methods": {k: stats(v) for k, v in rows.items()},
               "paired_upper_pooled_minus": {k: paired_upper(rows["probe_pooled"], rows[k])
                                             for k in ("X", "raw", "cevae", "oracle")},
               "screen_pooled": {"tpr": float(np.mean([s["probe"]["rules"]["pooled"][key]["tpr"] for s in shards])),
                                 "fpr": float(np.mean([s["probe"]["rules"]["pooled"][key]["fpr"] for s in shards])),
                                 "kept_w4": int(sum(s["probe"]["rules"]["pooled"][key]["kept_w4"] for s in shards)),
                                 "kept_any_w5": int(sum(s["probe"]["rules"]["pooled"][key]["kept_any_w5"] for s in shards)),
                                 "n_retained": [s["probe"]["rules"]["pooled"][key]["n_retained"] for s in shards]},
               "screen_sealed": {"tpr": float(np.mean([s["probe"]["rules"]["sealed"][key]["tpr"] for s in shards])),
                                 "fpr": float(np.mean([s["probe"]["rules"]["sealed"][key]["fpr"] for s in shards])),
                                 "n_retained": [s["probe"]["rules"]["sealed"][key]["n_retained"] for s in shards]},
               "prereg_sha256": rf.sha256(PREREG),
               "shard_sha256": {shard_path(name, r).name: rf.sha256(shard_path(name, r)) for r in reps}}
    with open(out_path, "x") as handle:
        handle.write(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    print(f"{name}: tau mean {summary['tau_mean']:+.4f}, n = {summary['n']}, {len(shards)} replicates")
    for k, v in summary["methods"].items():
        if v["returned"]:
            print(f"  {k:24s} returned {v['returned']:2d}/{len(reps)}  mean {v['mean']:+.4f}  MAE {v['mae']:.4f}  p90 {v['p90']:.4f}")
        else:
            print(f"  {k:24s} returned  0/{len(reps)}")
    print("  paired upper:", {k: (None if v is None else round(v, 4)) for k, v in summary["paired_upper_pooled_minus"].items()})
    print("  screen pooled:", {k: summary["screen_pooled"][k] for k in ("tpr", "fpr", "kept_w4", "kept_any_w5")})
    print(f"  written {out_path}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "summarise"])
    ap.add_argument("dataset", choices=sorted(ia.DATASETS))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv[1:])
    return summarise(args.dataset) if args.command == "summarise" else run(args.dataset, args.workers)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
