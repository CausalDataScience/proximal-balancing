"""P0 of the direct image balancing experiment: the numeric contrast, before any image and any GPU
(PRD memo/2026-09-21-direct-image-balancing-prd-v1.md, section 8, stage P0).

The question is the cheap one: if a learner read every image perfectly, would the problem still be solvable at
this recording resolution?  So the images are replaced by the exact recorded levels, 15 for Shapes3D and 10 for
MNIST, and the existing numeric machinery runs on them.  A benchmark that fails here fails for a reason that has
nothing to do with a CNN, and no GPU time is spent on it.

    python3 -B direct_balance_p0.py run [--workers 3]     the six tasks (two benchmarks by three data sets)
    python3 -B direct_balance_p0.py score                 the entry decision

What is recorded per data set:

    naive              no adjustment at all; the sign flip this design is built to produce
    X                  adjust for X only; what the proxy is supposed to add
    raw                adjust for X and all six recorded levels
    summary            adjust for X and the sealed front end's summary of the blocks
    oracle             adjust for the latent confounder itself (evaluator only)
    probe              the sealed balancing pipeline on the recorded levels: what it returns and how far off

`probe` is the sealed `run_dataset` with its own four rotations.  P0 asks whether the measurement system still
carries the effect, so it uses the machinery that is already trusted rather than the new one being built.
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

import direct_balance_data as db
import family_v2_dgp as g

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "direct_balance" / "p0"
PRD = HERE.parent / "memo" / "2026-09-21-direct-image-balancing-prd-v1.md"
SOURCES = ["family_v2_dgp.py", "family_v2_probe.py", "family_v2_competitors.py", "family_v2_images.py",
           "probe_structured_scm.py", "direct_balance_data.py", "direct_balance_p0.py"]
P0_SEED = 96_100_000                       # a range no earlier run has used
REPS = 3
TOLERANCE = 1.5e-3                         # the development threshold of PRD section 6
FOLD_SEED_OFFSET = 7                       # the sealed replay's offset, so `probe` reproduces exactly
GATE = {"worst_abs_error": 0.10}
SUMMARY_FILE = RESULTS / "p0_summary_v1.json"


def tasks() -> list[dict]:
    """The same three data sets at both resolutions, so the only difference between the two benchmarks here is
    how finely the coordinate is recorded: 15 levels or 10."""
    return [{"benchmark": name, "levels": levels, "rep": rep, "seed": P0_SEED + 1_000 * rep}
            for name, levels in db.BENCHMARKS.items() for rep in range(REPS)]


def shard_path(task: dict) -> Path:
    return RESULTS / f"{task['benchmark']}_L{task['levels']}_seed{task['seed']}.json"


def resolution(data: dict, levels: int) -> dict:
    """How much of each observed coordinate survives the recording, for the evaluator's notes only."""
    rec = db.recorded(data, levels)
    out = {}
    for key, lv in rec.items():
        exact, got = data[key], lv.astype(np.float64)
        out[key] = [{"levels_used": int(len(np.unique(got[:, c]))),
                     "r2_of_the_exact_coordinate": round(float(np.corrcoef(exact[:, c], got[:, c])[0, 1] ** 2), 4),
                     "share_at_an_end_level": round(float(np.mean((got[:, c] == 0) | (got[:, c] == levels - 1))), 4)}
                    for c in range(lv.shape[1])]
    return out


def run_task(task: dict) -> dict:
    import family_v2_competitors as fc
    import family_v2_probe as pr
    import summarize_family_v2 as sm

    out_path = shard_path(task)
    if out_path.exists():
        return {"task": task, "status": "exists"}
    t_all = time.time()
    seed, levels = task["seed"], task["levels"]
    data = g.generate(g.LEVELS[db.BASE_LEVEL], db.N, seed)
    numbers = db.as_numbers(data, levels)
    fold_seed = seed + FOLD_SEED_OFFSET
    shard = {"task": task, "n": db.N, "true_effect": g.TAU, "tolerance": TOLERANCE,
             "resolution": resolution(data, levels), "timing_s": {}}
    t0 = time.time()
    shard["baselines"] = fc.aipw_baselines(numbers, fold_seed)
    shard["timing_s"]["baselines"] = round(time.time() - t0, 1)
    t0 = time.time()
    res = pr.run_dataset(g.learner_view(numbers), g.LEVELS[db.BASE_LEVEL], TOLERANCE, fold_seed)
    shard["timing_s"]["probe"] = round(time.time() - t0, 1)
    rep = sm.replay({"probe": res, "task": {"seed": seed}}, TOLERANCE)
    rep.update(sm.mechanism(rep))
    shard["probe"] = {"return_kind": rep["return_kind"], "estimate": rep["estimate"],
                      "retained": rep["retained"], "members": rep["members"],
                      "component_sizes": rep["component_sizes"], "tpr": rep["tpr"], "fpr": rep["fpr"],
                      "purity": rep["purity"],
                      "candidate_estimates": {name: res["records"][name]["estimate"] for name in sm.TARGETS},
                      "max_upper_balanceable": max(sm.split_max_upper(res["records"][n])[0] for n in sm.TARGETS),
                      "min_upper_other": min(sm.split_max_upper(rec)[0] for name, rec in res["records"].items()
                                             if name not in sm.TARGETS)}
    shard["row_split_check"] = {role: int(len(rows)) for role, rows in db.split_rows(db.N, seed).items()}
    shard["timing_s"]["total"] = round(time.time() - t_all, 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(shard, indent=1, allow_nan=False) + "\n")
    tmp.rename(out_path)
    return {"task": task, "status": "written", "seconds": shard["timing_s"]["total"],
            "estimate": shard["probe"]["estimate"], "return_kind": shard["probe"]["return_kind"],
            "naive": shard["baselines"]["naive"], "X": shard["baselines"]["X"], "raw": shard["baselines"]["raw"]}


def run(workers: int) -> int:
    import multiprocessing
    todo = [t for t in tasks() if not shard_path(t).exists()]
    print(f"P0: {len(todo)} tasks with {workers} workers", flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_task, t): t for t in todo}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                r = fut.result()
                print(f"  {t['benchmark']} L{t['levels']} seed {t['seed']}: {r['status']} "
                      f"{r.get('seconds')} s | PROBE {r.get('estimate')} ({r.get('return_kind')}) | "
                      f"naive {r.get('naive')} X {r.get('X')} raw {r.get('raw')}", flush=True)
            except Exception:
                failures += 1
                print(f"  {t['benchmark']} L{t['levels']} seed {t['seed']}: FAILED\n{traceback.format_exc()}",
                      flush=True)
    print(f"P0 finished with {failures} failures", flush=True)
    return 1 if failures else 0


def score() -> int:
    import run_family_v2 as rf
    if SUMMARY_FILE.exists():
        raise SystemExit(f"{SUMMARY_FILE} exists; summaries are write-once")
    shards = [json.loads(shard_path(t).read_text()) for t in tasks() if shard_path(t).exists()]
    if len(shards) != len(tasks()):
        raise SystemExit(f"{len(shards)} shards, {len(tasks())} expected; nothing is scored early")
    out = {"artifact": "direct_balance_p0_summary", "generated_utc": rf.now(), "n": db.N, "reps": REPS,
           "tolerance": TOLERANCE, "prd_sha256": rf.sha256(PRD),
           "source_sha256": {f: rf.sha256(HERE / f) for f in SOURCES}, "benchmarks": {}}
    for name, levels in db.BENCHMARKS.items():
        mine = [sh for sh in shards if sh["task"]["benchmark"] == name]
        est = [sh["probe"]["estimate"] for sh in mine]
        returned = [e for e in est if e is not None]
        errs = [abs(e - g.TAU) for e in returned]
        rows = {key: [sh["baselines"][key] for sh in mine] for key in ("naive", "X", "raw", "summary", "oracle")}
        gate = {"all_returned": len(returned) == len(mine),
                "worst_abs_error": bool(errs) and max(errs) <= GATE["worst_abs_error"]}
        # The PRD's entry bar for P0 is exactly those two (section 8).  Everything below is reported, never gated.
        seen = {"naive_sign_is_wrong": all(v < 0 for v in rows["naive"]),
                "X_alone_is_far_off": all(abs(v - g.TAU) > 0.5 for v in rows["X"]),
                "oracle_is_close": all(abs(v - g.TAU) <= 0.05 for v in rows["oracle"])}
        out["benchmarks"][name] = {
            "levels": levels, "estimates": est, "abs_errors": [round(e, 4) for e in errs],
            "worst_abs_error": round(max(errs), 4) if errs else None,
            "return_kinds": [sh["probe"]["return_kind"] for sh in mine],
            "component_sizes": [sh["probe"]["component_sizes"] for sh in mine],
            "tpr": [sh["probe"]["tpr"] for sh in mine], "fpr": [sh["probe"]["fpr"] for sh in mine],
            "comparators": {k: {"values": [round(x, 4) for x in v],
                                "mae": round(float(np.mean(np.abs(np.array(v) - g.TAU))), 4)}
                            for k, v in rows.items()},
            "resolution_r2_min": round(min(c["r2_of_the_exact_coordinate"]
                                           for sh in mine for cols in sh["resolution"].values() for c in cols), 4),
            "seconds_total": round(sum(sh["timing_s"]["total"] for sh in mine)),
            "gate": gate, "also_seen": seen, "pass": all(gate.values())}
    out["requirement"] = GATE
    out["notes"] = {
        "gate": "the PRD section 8 entry bar only: every data set returns, and every absolute error is at most 0.10",
        "min_upper_other_in_the_shards": "computed against the seven target splits, so it counts S4 -- a split that "
                                         "balances but does not identify -- as 'other'.  It is not the sealed "
                                         "threshold rule's 'other' set and no decision here reads it.",
        "scorer_corrected_after_the_shards_were_written": "the three shapes3d shards already existed when the two "
                                                          "extra gates above were demoted to reported diagnostics; "
                                                          "run_task was not touched, so every shard is unchanged."}
    out["pass"] = all(v["pass"] for v in out["benchmarks"].values())
    out["may_use_gpu_for"] = [k for k, v in out["benchmarks"].items() if v["pass"]]
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    for name, v in out["benchmarks"].items():
        print(f"{name} (L{v['levels']}): estimates {v['estimates']} | worst |error| {v['worst_abs_error']} | "
              f"naive {v['comparators']['naive']['values']} X MAE {v['comparators']['X']['mae']} "
              f"raw MAE {v['comparators']['raw']['mae']} oracle MAE {v['comparators']['oracle']['mae']} | "
              f"also {v['also_seen']} | "
              f"{'PASS' if v['pass'] else 'FAIL ' + str([k for k, ok in v['gate'].items() if not ok])}")
    print(f"written {SUMMARY_FILE}\nGPU may be spent on: {out['may_use_gpu_for']}")
    return 0 if out["pass"] else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "score"])
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv[1:])
    return run(args.workers) if args.command == "run" else score()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
