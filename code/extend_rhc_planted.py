"""More replicates of the planted RHC design, run with the frozen code and scored with the frozen rule.

Why: the prospective check used replicates 11-20, ten points per method, and ten points cannot separate
PROBE (mean absolute error 0.351) from adjusting for X and all of W (0.305).  This adds replicates 21-50.

What is fixed before anything runs:
    EXTENSION = 21..50, thirty replicates.  The number was set before any of them was drawn and is not
    changed by what they show; every one is reported.
    The code: every source file must hash to the value the frozen protocol recorded for replicates 11-20,
    so the new shards come from exactly the code that produced the old ones.  The script refuses otherwise.
    The rule: the v3 noise-scaled rule, applied through `noise_scaled.decide`, unchanged.

What is not claimed: replicates 21-50 were not named in the frozen protocol.  They were run after 11-20 had
been scored, at the user's request for more points.  They are reported as an extension, separately from
the designated ten and pooled with them, never in place of them.

    python3 -B extend_rhc_planted.py run [--workers 2]
    python3 -B extend_rhc_planted.py score
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "rhc_planted"
PROTOCOL = HERE.parent / "results" / "v3_confirmation_protocol_v2.json"
OUT = RESULTS / "extension_v1.json"
DESIGNATED = tuple(range(11, 21))
EXTENSION = tuple(range(21, 51))
METHODS = ("X", "raw", "p2sls_given_roles", "p2sls_swapped_roles", "coca", "cevae", "probe", "oracle")


def frozen_expectation() -> dict:
    manifest = json.loads(PROTOCOL.read_text())["manifests"]["rhc_planted"]
    return manifest["expected"][str(DESIGNATED[0])]["source_sha256"]


def assert_same_code() -> None:
    expected = frozen_expectation()
    differ = [f for f, h in expected.items() if hashlib.sha256((HERE / f).read_bytes()).hexdigest() != h]
    if differ:
        raise SystemExit(f"these files differ from the frozen protocol, so the extension would not be the same "
                         f"experiment: {differ}")


def _one(r: int) -> dict:
    try:
        os.nice(10)
    except OSError:
        pass
    import run_rhc_planted as m
    return m.run_replicate(r)


def run(workers: int) -> int:
    import multiprocessing
    assert_same_code()
    todo = [r for r in EXTENSION if not (RESULTS / f"replicate_{r}.json").exists()]
    print(f"planted RHC extension: {len(todo)} of {len(EXTENSION)} to run with {workers} workers", flush=True)
    failed = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_one, r): r for r in todo}
        for f in as_completed(futures):
            r = futures[f]
            try:
                res = f.result()
                print(f"  replicate {r}: {res['status']} {res.get('seconds', '')}", flush=True)
            except Exception as exc:                       # an execution failure is recorded, never hidden
                failed.append(r)
                print(f"  replicate {r}: FAILED {type(exc).__name__}: {exc}", flush=True)
    print(f"done; failed {failed or 'none'}", flush=True)
    return 1 if failed else 0


def score() -> int:
    import coca_ate_all as ca
    import noise_scaled as ns
    import park_ate as pa
    import rhc_planted_data as rp
    import run_family_v2 as rf
    import run_twins as rtw
    import v3_fresh_scorer as vs

    if OUT.exists():
        raise SystemExit(f"{OUT} exists; write-once")
    assert_same_code()
    hashes = frozen_expectation()
    ids = DESIGNATED + EXTENSION
    views = ca.views_for("rhc_planted", ids)
    rows, rejected = [], {}
    for r in ids:
        path = RESULTS / f"replicate_{r}.json"
        if not path.exists():
            rejected[r] = "shard missing"
            continue
        shard = json.loads(path.read_text())
        expected = {"dataset": "rhc_planted", "seed": rp.REPLICATE_SEED + 1000 * r + rtw.FOLD_SEED_OFFSET,
                    "source_sha256": hashes}
        try:
            vs.check_shard(shard, "rhc_planted", r, expected)
        except vs.Rejected as exc:
            rejected[r] = str(exc)
            continue
        decision = ns.decide(shard["probe"], shard["n"], shard["seed"])
        view, tau = views[r]
        comps = {k: vs.comparator(shard, k) for k in vs.COMPARATORS}
        comps["coca"] = pa.park_parts(view, "W1")["ate"]
        rows.append({"replicate": r, "set": "designated" if r in DESIGNATED else "extension",
                     "tau": shard["tau"], "output": decision["output"], "return_kind": decision["return_kind"],
                     "n_retained": decision["n_retained"], "comparators": comps})
    if rejected:
        print(f"INCOMPLETE: {rejected}")
        return 1

    def summary(which: set[int]) -> dict:
        sel = [x for x in rows if x["replicate"] in which]
        out = {"n": len(sel), "returned": sum(x["output"] is not None for x in sel)}
        for k in METHODS:
            if k == "probe":
                errs = [abs(x["output"] - x["tau"]) for x in sel if x["output"] is not None]
            else:
                errs = [abs(x["comparators"][k] - x["tau"]) for x in sel]
            out[k] = float(np.mean(errs)) if errs else None
        return out

    payload = {"artifact": "rhc_planted_extension", "generated_utc": rf.now(),
               "designated": list(DESIGNATED), "extension": list(EXTENSION),
               "fixed_before_running": "the thirty identifiers 21-50; every one is reported",
               "not_preregistered": "21-50 were not named in the frozen protocol and were run after 11-20 had "
                                    "been scored; the code hashes to the frozen protocol and the rule is v3 "
                                    "unchanged",
               "rule": "noise_scaled.decide (v3)", "coca": "park_ate.park_parts on the W1 block, as an ATE",
               "replicates": rows,
               "mae": {"designated_11_20": summary(set(DESIGNATED)), "extension_21_50": summary(set(EXTENSION)),
                       "pooled_11_50": summary(set(ids))},
               "source_sha256": {f: rf.sha256(HERE / f) for f in
                                 ("extend_rhc_planted.py", "noise_scaled.py", "v3_fresh_scorer.py", "park_ate.py",
                                  "coca_ate_all.py")},
               "frozen_code_sha256": hashes}
    with open(OUT, "x") as handle:
        handle.write(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    for key, s in payload["mae"].items():
        line = "  ".join(f"{k} {s[k]:.3f}" for k in METHODS if s[k] is not None)
        print(f"{key:18s} n={s['n']:2d} returned {s['returned']:2d}  {line}")
    print(f"written {OUT}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("run", "score"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv[1:])
    return run(args.workers) if args.command == "run" else score()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
