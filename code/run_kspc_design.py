"""Every method on the KSPC benchmark adapted to a binary treatment (kspc_design.py), replicates 1 to 20.

Per replicate, one write-once shard in results/kspc_design/:
    PROBE on the two blocks (J = 2; the significance rule is primary at two candidates, as for Zika and
        LaLonde, because the v3 rule degenerates there; all three readings are kept)
    naive, adjust for X (there is no X, so this is the intercept-only AIPW), adjust for all of W
    proximal 2SLS in both orientations, COCA as an ATE from each block
    KSPC, both of the authors' estimators (SKPV and SPMMR), given W1 with their low-dimensional
        configuration and given W2 with their MNIST configuration
    oracle, the AIPW that adjusts for U itself

    python3 -B run_kspc_design.py run [--workers 2]
    python3 -B run_kspc_design.py summarise
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "kspc_design"
K = 2
SOURCES = ("kspc_design.py", "run_kspc_design.py", "kspc_wrapper.py", "probe_j2.py", "family_v2_probe.py",
           "probe_structured_scm.py", "family_v2_competitors.py", "park_ate.py", "run_two_proxy.py")


def oracle(view: dict, u: np.ndarray, seed: int) -> float:
    """AIPW adjusting for U, on the same four rotations as PROBE."""
    import family_v2_competitors as fc
    import family_v2_probe as pr
    n = len(view["A"])
    fold_rows = pr.folds(n, seed)
    total = 0.0
    for rot in range(4):
        rl = pr.roles(rot)
        rn, re = fold_rows[rl["N"]], fold_rows[rl["E"]]
        fn = np.column_stack([np.ones(len(rn)), u[rn]])
        fe = np.column_stack([np.ones(len(re)), u[re]])
        total += float(fc.aipw_cv(fn, view["A"][rn], view["Y"][rn], fe, view["A"][re], view["Y"][re],
                                  seed + 97 * rot).mean()) / 4.0
    return total


def run_replicate(r: int) -> dict:
    import family_v2_competitors as fc
    import kspc_design as kd
    import kspc_wrapper as kw
    import park_ate as pa
    import probe_j2 as pj
    import run_family_v2 as rf
    import run_two_proxy as rtp

    out_path = RESULTS / f"replicate_{r}.json"
    if out_path.exists():
        return {"replicate": r, "status": "exists"}
    t0 = time.time()
    d = kd.generate(r)
    view = {k: d[k] for k in ("X", "W1", "W2", "A", "Y")}
    seed = kd.SEED + 1_000 * r
    timing = {}
    t = time.time(); probe = pj.run(view, K, seed, n_blocks=2); timing["probe"] = round(time.time() - t, 1)
    t = time.time(); base = rtp.baselines(view, seed, 2); timing["baselines"] = round(time.time() - t, 1)
    comps = {"naive": base["naive"], "X": base["X"], "raw": base["raw"],
             "oracle": oracle(view, d["U_eval_only"], seed),
             "p2sls_W1_treats_W2_outcome": fc.p2sls(view, "W1", "W2"),
             "p2sls_W2_treats_W1_outcome": fc.p2sls(view, "W2", "W1"),
             "coca_W1_ate": pa.park_parts(view, "W1")["ate"], "coca_W2_ate": pa.park_parts(view, "W2")["ate"]}
    t = time.time()
    kspc = {}
    for method in ("skpv", "spmmr"):
        for block, image in (("W1", False), ("W2", True)):
            kspc[f"{method}_{block}"] = kw.kspc_effect(view["A"], view[block], view["Y"], method=method,
                                                       image=image, seed=seed)
    timing["kspc"] = round(time.time() - t, 1)
    shard = {"replicate": r, "n": int(len(view["A"])), "tau": d["tau"], "seed": seed,
             "probe": {k: v for k, v in probe.items() if k != "score_crossproduct"},
             "score_crossproduct": probe["score_crossproduct"],
             "comparators": comps, "kspc": kspc,
             "timing_s": {**timing, "total": round(time.time() - t0, 1)},
             "source_sha256": {f: rf.sha256(HERE / f) for f in SOURCES}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(out_path, "x") as handle:
        handle.write(json.dumps(shard, indent=1, allow_nan=False) + "\n")
    return {"replicate": r, "status": "written", "seconds": shard["timing_s"]["total"],
            "probe": probe["readings"]["significance"]["output"], "raw": comps["raw"],
            "kspc_skpv_W1": kspc["skpv_W1"]["ate"]}


def _one(r: int) -> dict:
    try:
        os.nice(10)
    except OSError:
        pass
    return run_replicate(r)


def run(workers: int) -> int:
    import multiprocessing
    import kspc_design as kd
    todo = [r for r in kd.REPLICATES if not (RESULTS / f"replicate_{r}.json").exists()]
    print(f"KSPC design: {len(todo)} replicates with {workers} workers", flush=True)
    failed = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_one, r): r for r in todo}
        for f in as_completed(futures):
            r = futures[f]
            try:
                res = f.result()
                print(f"  replicate {r}: {res}", flush=True)
            except Exception as exc:
                failed.append(r)
                print(f"  replicate {r}: FAILED {type(exc).__name__}: {exc}", flush=True)
    print(f"done; failed {failed or 'none'}", flush=True)
    return 1 if failed else 0


def summarise() -> int:
    import kspc_design as kd
    rows = [json.loads((RESULTS / f"replicate_{r}.json").read_text()) for r in kd.REPLICATES]
    tau = kd.TAU
    table = {}
    for name, get in (("PROBE, significance", lambda s: s["probe"]["readings"]["significance"]["output"]),
                      ("PROBE, v3", lambda s: s["probe"]["readings"]["noise_floor"]["output"]),
                      *[(k, (lambda s, k=k: s["comparators"][k])) for k in rows[0]["comparators"]],
                      *[(f"KSPC {k}", (lambda s, k=k: s["kspc"][k]["ate"])) for k in rows[0]["kspc"]]):
        vals = [get(s) for s in rows]
        got = [v for v in vals if v is not None]
        table[name] = {"returned": len(got), "mae": float(np.mean([abs(v - tau) for v in got])) if got else None,
                       "mean": float(np.mean(got)) if got else None}
    out = RESULTS / "summary_v1.json"
    if out.exists():
        raise SystemExit(f"{out} exists; write-once")
    out.write_text(json.dumps({"tau": tau, "replicates": list(kd.REPLICATES), "methods": table}, indent=1) + "\n")
    for name, v in sorted(table.items(), key=lambda kv: (kv[1]["mae"] is None, kv[1]["mae"])):
        mae = "none" if v["mae"] is None else f"{v['mae']:.4f}"
        print(f"{name:32s} returned {v['returned']:2d}/20  mean {v['mean']:+.4f}  MAE {mae}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("run", "summarise"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    raise SystemExit(run(args.workers) if args.command == "run" else summarise())
