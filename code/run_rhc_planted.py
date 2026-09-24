"""PROBE and the comparators on the planted-effect RHC design, replicates 1-10, exactly as
memo/2026-09-22-rhc-planted-prereg-v1.md fixes them.

    python3 -B run_rhc_planted.py run [--workers 3]
    python3 -B run_rhc_planted.py summarise

The runner is run_twins.py's, with the data module and the output directory swapped; the helpers
`baselines`, `probe_rules` and the summary arithmetic are imported from it so the two experiments are
scored by exactly the same code.
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

import rhc_planted_data as rp
import run_twins as rtw

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "rhc_planted"
PREREG = HERE.parent / "memo" / "2026-09-22-rhc-planted-prereg-v1.md"
SOURCES = ["family_v2_dgp.py", "family_v2_probe.py", "family_v2_competitors.py", "probe_structured_scm.py",
           "rhc_data.py", "rhc_planted_data.py", "rhc_pooled.py", "run_twins.py", "run_rhc_planted.py"]
CONFIRM = range(1, 11)
BLOCK_KEYS = ("W1", "W2", "W3", "W4", "W5")


def shard_path(r: int) -> Path:
    return RESULTS / f"replicate_{r}.json"


def run_replicate(r: int) -> dict:
    import family_v2_competitors as fc
    import family_v2_probe as pr
    import run_family_v2 as rf

    out_path = shard_path(r)
    if out_path.exists():
        return {"replicate": r, "status": "exists"}
    t_all = time.time()
    real = rp.load_real()
    d = rp.generate(real, r)
    view = {k: d[k] for k in ("X",) + BLOCK_KEYS + ("A", "Y")}
    seed = rp.REPLICATE_SEED + 1000 * r + rtw.FOLD_SEED_OFFSET
    timing = {}
    t0 = time.time(); base = rtw.baselines(view, d["U_eval_only"], seed); timing["baselines"] = round(time.time() - t0, 1)
    t0 = time.time(); prox = fc.proximal_rows(view); timing["proximal"] = round(time.time() - t0, 1)
    t0 = time.time(); probe = pr.run_dataset(view, rp.level(real["X"].shape[1]), rtw.THRESHOLD, seed)
    timing["probe"] = round(time.time() - t0, 1)
    t0 = time.time(); cevae = fc.cevae_ate(view, seed); timing["cevae"] = round(time.time() - t0, 1)
    shard = {"replicate": r, "n": real["n"], "d_x": real["X"].shape[1], "tau": real["tau"], "seed": seed,
             "naive": d["naive"], "propensity_mean": float(d["propensity"].mean()),
             "baselines": base, "proximal": prox, "cevae": cevae,
             "probe": {"sealed_run": {"return_kind": probe["return_kind"], "estimate": probe["estimate"],
                                      "retained": probe["retained"], "rho": probe["rho"],
                                      "component_sizes": probe["component_sizes"]},
                       "per_split": {nm: {"estimate": rec["estimate"],
                                          "screen": [x["screen"] for x in rec["rotations"]]}
                                     for nm, rec in probe["records"].items()},
                       "rules": rtw.probe_rules(probe, real["n"], seed),
                       "score_crossproduct": probe["score_crossproduct"], "split_names": probe["split_names"],
                       "seconds": probe["seconds"]},
             "timing_s": {**timing, "total": round(time.time() - t_all, 1)},
             "prereg_sha256": rf.sha256(PREREG),
             "source_sha256": {f: rf.sha256(HERE / f) for f in SOURCES}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(out_path, "x") as handle:
        handle.write(json.dumps(shard, indent=1, allow_nan=False) + "\n")
    p = shard["probe"]["rules"]["pooled"][f"{rtw.THRESHOLD:.4f}"]
    return {"replicate": r, "status": "written", "seconds": shard["timing_s"]["total"],
            "probe_pooled": p["output"], "n_retained": p["n_retained"], "naive": d["naive"],
            "X": base["X"], "raw": base["raw"], "oracle": base["oracle"], "cevae": cevae["ate"]}


def run(workers: int) -> int:
    import multiprocessing
    todo = [r for r in CONFIRM if not shard_path(r).exists()]
    print(f"RHC-planted: {len(todo)} replicates with {workers} workers, tau = {rp.DELTA}", flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_replicate, r): r for r in todo}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                res = fut.result()
                if res["status"] == "written":
                    shown = "none" if res["probe_pooled"] is None else f"{res['probe_pooled']:+.4f}"
                    print(f"  replicate {r}: {res['seconds']} s | PROBE {shown} "
                          f"({res['n_retained']} retained) | naive {res['naive']:+.3f} X {res['X']:+.3f} "
                          f"raw {res['raw']:+.3f} oracle {res['oracle']:+.3f} CEVAE {res['cevae']:+.3f}", flush=True)
            except Exception:
                failures += 1
                print(f"  replicate {r}: FAILED\n{traceback.format_exc()}", flush=True)
    return 1 if failures else 0


def summarise() -> int:
    import run_family_v2 as rf
    from scipy.stats import t as tdist
    out_path = RESULTS / "summary_v1.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; summaries are write-once")
    shards = [json.loads(shard_path(r).read_text()) for r in CONFIRM if shard_path(r).exists()]
    if len(shards) != len(CONFIRM):
        raise SystemExit(f"{len(shards)} shards of {len(CONFIRM)}; nothing is summarised early")
    tau = shards[0]["tau"]
    key = f"{rtw.THRESHOLD:.4f}"
    rows = {"probe_sealed": [s["probe"]["sealed_run"]["estimate"] for s in shards],
            "probe_pooled": [s["probe"]["rules"]["pooled"][key]["output"] for s in shards],
            "naive": [s["naive"] for s in shards], "X": [s["baselines"]["X"] for s in shards],
            "raw": [s["baselines"]["raw"] for s in shards], "oracle": [s["baselines"]["oracle"] for s in shards],
            "cevae": [s["cevae"]["ate"] for s in shards],
            "p2sls_given_roles": [s["proximal"]["p2sls_given_roles"] for s in shards],
            "p2sls_swapped_roles": [s["proximal"]["p2sls_swapped_roles"] for s in shards],
            "park_spc_clean_proxy": [s["proximal"]["park_spc_clean_proxy"] for s in shards]}

    def stats(values):
        ok = [v for v in values if v is not None]
        err = [abs(v - tau) for v in ok]
        return {"values": ok, "returned": len(ok), "mean": float(np.mean(ok)) if ok else None,
                "mae": float(np.mean(err)) if err else None, "p90": float(np.quantile(err, 0.9)) if err else None}

    def paired_upper(a, b):
        """The prespecified one-sided bound over the replicates where both methods returned; None when
        fewer than two pairs survive, which happens when the screen declines on most replicates."""
        dd = np.array([abs(x - tau) - abs(y - tau) for x, y in zip(a, b) if x is not None and y is not None])
        if len(dd) < 2:
            return None
        return float(dd.mean() + tdist.ppf(0.95, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd)))

    summary = {"artifact": "rhc_planted_summary", "generated_utc": rf.now(), "tau": tau, "n": shards[0]["n"],
               "hidden_confounder": rp.HIDDEN, "delta": rp.DELTA,
               "replicates": [s["replicate"] for s in shards], "threshold": rtw.THRESHOLD,
               "methods": {k: stats(v) for k, v in rows.items()},
               "paired_upper_pooled_minus": {k: paired_upper(rows["probe_pooled"], rows[k])
                                             for k in ("X", "raw", "cevae", "oracle")},
               "screen_pooled": {"n_retained": [s["probe"]["rules"]["pooled"][key]["n_retained"] for s in shards],
                                 "return_kinds": [s["probe"]["rules"]["pooled"][key]["return_kind"] for s in shards]},
               "screen_sealed": {"n_retained": [s["probe"]["rules"]["sealed"][key]["n_retained"] for s in shards],
                                 "return_kinds": [s["probe"]["rules"]["sealed"][key]["return_kind"] for s in shards]},
               "grid_pooled": {f"{t:.4f}": stats([s["probe"]["rules"]["pooled"][f"{t:.4f}"]["output"] for s in shards])
                               for t in rtw.GRID},
               "no_planted_invalid_block": "every block is a real measurement of the same hidden score, so the "
                                           "screen's discrimination is not tested here (pre-registration section 4)",
               "prereg_sha256": rf.sha256(PREREG),
               "shard_sha256": {shard_path(r).name: rf.sha256(shard_path(r)) for r in CONFIRM}}
    with open(out_path, "x") as handle:
        handle.write(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    for k, v in summary["methods"].items():
        if v["returned"]:
            print(f"{k:24s} returned {v['returned']:2d}/10  mean {v['mean']:+.4f}  MAE {v['mae']:.4f}  p90 {v['p90']:.4f}")
        else:
            print(f"{k:24s} returned  0/10")
    print("paired upper (pooled PROBE minus):", {k: (None if v is None else round(v, 4)) for k, v in summary["paired_upper_pooled_minus"].items()})
    print("retained per replicate (pooled):", summary["screen_pooled"]["n_retained"])
    print(f"written {out_path}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "summarise"])
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv[1:])
    return summarise() if args.command == "summarise" else run(args.workers)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
