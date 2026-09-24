"""E3, observed RHC: the whole analysis repeated on 20 bootstrap resamples of the 5,735 patients.

Fixed before any resample was drawn (2026-09-23, at the author's request for 20 points on E3):
    B = 20 resamples, rows drawn with replacement, seed 99_600_000 + b for b = 1..20, which also seeds the
        cross-fitting folds of that resample.
    PROBE: the path that produced E3's reported number, `probe_family.run_dataset` at the run's threshold
        followed by the v3 rule `noise_scaled.decide`.
    Comparators: the difference in means; AIPW adjusting for X, the function E3's reproduction gate used;
        AIPW adjusting for X and all ten proxies; proximal 2SLS with the roles Cui et al. assign, the first
        block (pafi1, paco21) on the treatment side and the second (ph1, hema1) on the outcome side.

What a point means here: RHC has no known answer, so a bootstrap point shows how much each method's number
moves under resampling of the same patients.  It is not a Monte Carlo replicate of a known design.  A
resample can put copies of one patient in different cross-fitting folds, which makes each point somewhat
optimistic; the spread is descriptive.  E3's frozen files are not touched; this writes new ones.

    python3 -B rhc_bootstrap.py run [--workers 2]
    python3 -B rhc_bootstrap.py summarise
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "rhc_bootstrap"
B = 20
SEED = 99_600_000
SOURCES = ("rhc_bootstrap.py", "rhc_data.py", "run_rhc.py", "noise_scaled.py", "family_v2_probe.py",
           "probe_structured_scm.py", "family_v2_competitors.py", "run_two_proxy.py", "probe_j2.py")


def run_one(b: int) -> dict:
    try:
        os.nice(10)
    except OSError:
        pass
    import family_v2_competitors as fc
    import family_v2_probe as pr
    import noise_scaled as ns
    import rhc_data as rd
    import run_family_v2 as rf
    import run_rhc as rr
    import run_two_proxy as rtp

    out = OUT / f"boot_{b}.json"
    if out.exists():
        return {"b": b, "status": "exists"}
    t0 = time.time()
    data = rd.load()
    view, n = data["view"], data["n"]
    seed = SEED + b
    idx = np.random.default_rng(seed).integers(0, n, n)
    vb = {k: v[idx] for k, v in view.items()}
    probe = pr.run_dataset(vb, rd.level(data["d_x"]), rr.GRID[2], seed)
    decision = ns.decide(probe, n, seed)
    a, y = vb["A"], vb["Y"]
    comps = {"naive": float(y[a == 1].mean() - y[a == 0].mean()),
             "adjust_X": rr.x_only_aipw(vb, seed)["estimate"],
             "adjust_X_and_all_W": rtp.baselines(vb, seed, 5)["raw"],
             "p2sls_cui_roles": float(fc.p2sls(vb, "W1", "W2"))}
    rec = {"b": b, "seed": seed, "n": int(n), "probe_v3": {k: decision[k] for k in ("output", "return_kind",
                                                                                   "n_retained", "retained")},
           "comparators": comps, "seconds": round(time.time() - t0, 1), "generated_utc": rf.now(),
           "source_sha256": {f: rf.sha256(HERE / f) for f in SOURCES}}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as handle:
        handle.write(json.dumps(rec, indent=1, allow_nan=False) + "\n")
    return {"b": b, "status": "written", "probe": decision["output"], "seconds": rec["seconds"]}


def run(workers: int) -> int:
    import multiprocessing
    todo = [b for b in range(1, B + 1) if not (OUT / f"boot_{b}.json").exists()]
    print(f"{len(todo)} resamples, {workers} workers", flush=True)
    failed = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_one, b): b for b in todo}
        for f in as_completed(futures):
            try:
                print(f"  {f.result()}", flush=True)
            except Exception as exc:
                failed.append(futures[f])
                print(f"  FAILED b={futures[f]}: {type(exc).__name__}: {exc}", flush=True)
    print(f"done; failed {failed or 'none'}", flush=True)
    return 1 if failed else 0


def summarise() -> int:
    out = OUT / "summary_v1.json"
    if out.exists():
        raise SystemExit(f"{out} exists; write-once")
    rows = [json.loads((OUT / f"boot_{b}.json").read_text()) for b in range(1, B + 1)]
    probe = [r["probe_v3"]["output"] for r in rows]
    got = [p for p in probe if p is not None]
    table = {"probe_v3": {"returned": len(got), "of": B, "mean": float(np.mean(got)), "sd": float(np.std(got, ddof=1)),
                          "q05": float(np.quantile(got, 0.05)), "q95": float(np.quantile(got, 0.95))}}
    for k in rows[0]["comparators"]:
        v = np.array([r["comparators"][k] for r in rows])
        table[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "q05": float(np.quantile(v, 0.05)),
                    "q95": float(np.quantile(v, 0.95))}
    out.write_text(json.dumps({"B": B, "seed": SEED, "methods": table}, indent=1) + "\n")
    for k, v in table.items():
        print(f"{k:22s} mean {v['mean']:+.3f}  sd {v['sd']:.3f}  90% ({v['q05']:+.3f}, {v['q95']:+.3f})"
              + (f"  returned {v['returned']}/{B}" if "returned" in v else ""))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("run", "summarise"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    raise SystemExit(run(args.workers) if args.command == "run" else summarise())
