"""KSPC with covariates (Appendix A, our implementation; SKPV only) on the final experiments
(memo/2026-09-22-kspc-comparator-prd-v2.md, sections 3 to 5 and the amendment in section 9).

Every data set is rebuilt from its frozen generator and seed; no sealed file is read for anything but the
integrity check or written.  Integrity: the two COCA values each sealed shard stores, on the clean block W1
and the contaminated block W4, are recomputed on the rebuilt data and must agree to 1e-8, which pins X, A, Y,
W1 and W4 to the sealed data except for a constant added to Y, which COCA cannot see.  That gap is closed
separately by kspc_comparator_integrity.py: every generator file each frozen protocol hashed still hashes
the same, so the same seed rebuilds exactly the sealed data.  A task that fails the check is recorded and
not scored.

    python3 -B run_kspc_comparator.py run E1 E3 E5 E6 E7 [--workers 1]
    python3 -B run_kspc_comparator.py run E2 [--workers 1]
    python3 -B run_kspc_comparator.py summarise
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
RESULTS = HERE.parent / "results"
OUT = RESULTS / "kspc_comparator"
N_SUB = 3_000
SUB_SEED = 99_400_000
LABEL = "KSPC with covariates, Appendix A, our implementation; SKPV only, decided after gate 2"
SOURCES = ("run_kspc_comparator.py", "kspc_wrapper.py", "family_v2_dgp.py", "family_v2_competitors.py",
           "run_scm4.py", "family_v2_images.py", "coca_ate_all.py", "rhc_data.py")


def tasks(exp: str) -> list[dict]:
    import glob
    if exp in ("E1", "E2"):
        pattern = ("family_v2/confirm/{}_seed*.json" if exp == "E1" else "family_v2/scm4/confirm/{}_seed*.json")
        levels = ("SCM-1", "SCM-2", "SCM-3") if exp == "E1" else ("SCM-4",)
        out = []
        for lv in levels:
            for f in sorted(glob.glob(str(RESULTS / pattern.format(lv)))):
                s = json.loads(Path(f).read_text())
                out.append({"exp": exp, "name": lv, "replicate": int(s["task"]["rep"]), "seed": int(s["task"]["seed"]),
                            "tau": 1.0, "sealed": f})
        return out
    if exp == "E3":
        return [{"exp": "E3", "name": "rhc", "replicate": 0, "seed": None, "tau": None, "sealed": None}]
    name, reps = {"E5": ("twins", range(11, 21)), "E6": ("ihdp", range(12, 22)), "E7": ("acic", range(2, 11))}[exp]
    return [{"exp": exp, "name": name, "replicate": r, "seed": None, "tau": None,
             "sealed": str(RESULTS / name / f"replicate_{r}.json")} for r in reps]


_SCM4 = None


def build_view(task: dict) -> tuple[dict, float | None]:
    import family_v2_dgp as g
    if task["exp"] == "E1":
        s = json.loads(Path(task["sealed"]).read_text())
        n = int(s.get("n") or s["probe"]["n"])
        return g.learner_view(g.generate(g.LEVELS[task["name"]], n, task["seed"])), 1.0
    if task["exp"] == "E2":
        global _SCM4
        import torch
        torch.set_num_threads(1)
        import family_v2_images as im
        import run_scm4 as r4
        import run_scm4_images as ri
        if _SCM4 is None:
            net, dev, excluded, _ = ri.load_frozen_reader()
            images = im.load_images()
            groups = im.bank_by_level(np.load(im.BANK_DIR / "split_rows.npz")["causal_bank"], excluded)
            _SCM4 = (net, dev, images, groups)
        net, dev, images, groups = _SCM4
        data = g.generate(g.LEVELS[r4.BASE_LEVEL], r4.N, task["seed"])
        view = r4.image_view(data, net, images, dev, groups, task["seed"])[0]
        return view, 1.0
    if task["exp"] == "E3":
        import rhc_data as rd
        return rd.load()["view"], None
    import coca_ate_all as ca
    view, tau = ca.views_for(task["name"], [task["replicate"]])[task["replicate"]]
    return view, float(tau)


def integrity(view: dict, task: dict) -> dict:
    if task["sealed"] is None:
        return {"checked": False, "why": "no sealed shard; the RHC file is checked against its sha256 on load"}
    import family_v2_competitors as fc
    sealed = json.loads(Path(task["sealed"]).read_text())["proximal"]
    out = {"checked": True}
    for block, key in (("W1", "park_spc_clean_proxy"), ("W4", "park_spc_contaminated_proxy")):
        mine, theirs = float(fc.park_spc(view, block)), float(sealed[key])
        out[block] = {"recomputed": mine, "sealed": theirs}
        if abs(mine - theirs) > 1e-8 * max(1.0, abs(theirs)):
            raise RuntimeError(f"integrity: COCA on {block} is {mine}, the sealed shard says {theirs}")
    return out


def run_task(task: dict) -> dict:
    import kspc_wrapper as kw
    import run_family_v2 as rf
    out_path = OUT / task["exp"] / f"{task['name']}_replicate_{task['replicate']}.json"
    if out_path.exists():
        return {**task, "status": "exists"}
    t0 = time.time()
    view, tau = build_view(task)
    check = integrity(view, task)
    n = len(view["A"])
    rows = np.arange(n)
    sub_seed = SUB_SEED + task["replicate"]
    if n > N_SUB:
        rows = np.sort(np.random.default_rng(sub_seed).choice(n, N_SUB, replace=False))
    blocks = ("W1", "W2", "W3", "W4", "W5") if task["exp"] == "E3" else ("W1", "W4")
    kspc = {}
    for block in blocks:
        res = kw.kspc_effect(view["A"][rows], view[block][rows], view["Y"][rows], method="skpv", seed=sub_seed,
                             x=view["X"][rows])
        kspc[block] = {k: res[k] for k in ("ate", "f0", "f1")}
    shard = {"task": task, "label": LABEL, "tau": tau, "n_full": int(n), "n_used": int(len(rows)),
             "subsample_seed": sub_seed if n > N_SUB else None, "integrity": check, "kspc_skpv_with_x": kspc,
             "timing_s": round(time.time() - t0, 1), "generated_utc": rf.now(),
             "source_sha256": {f: rf.sha256(HERE / f) for f in SOURCES}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "x") as handle:
        handle.write(json.dumps(shard, indent=1, allow_nan=False) + "\n")
    return {**task, "status": "written", "seconds": shard["timing_s"],
            "ate": {b: round(v["ate"], 4) for b, v in kspc.items()}}


def _one(task: dict) -> dict:
    try:
        os.nice(10)
    except OSError:
        pass
    return run_task(task)


def run(exps: list[str], workers: int) -> int:
    import multiprocessing
    todo = [t for e in exps for t in tasks(e)
            if not (OUT / t["exp"] / f"{t['name']}_replicate_{t['replicate']}.json").exists()]
    print(f"{len(todo)} tasks, {workers} workers", flush=True)
    failed = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_one, t): t for t in todo}
        for f in as_completed(futures):
            t = futures[f]
            try:
                print(f"  {f.result()}", flush=True)
            except Exception as exc:
                failed.append((t["exp"], t["name"], t["replicate"], f"{type(exc).__name__}: {exc}"))
                print(f"  FAILED {t['exp']} {t['name']} {t['replicate']}: {type(exc).__name__}: {exc}", flush=True)
    print(f"done; failed {failed or 'none'}", flush=True)
    return 1 if failed else 0


def summarise() -> int:
    out_path = OUT / "summary_v1.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; write-once")
    summary = {"label": LABEL, "experiments": {}}
    for exp in ("E1", "E2", "E3", "E5", "E6", "E7"):
        shards = [json.loads(p.read_text()) for p in sorted((OUT / exp).glob("*.json"))]
        expected = len(tasks(exp))
        by_name: dict[str, list] = {}
        for s in shards:
            by_name.setdefault(s["task"]["name"], []).append(s)
        block = {}
        for name, rows in by_name.items():
            if exp == "E3":
                block[name] = {b: rows[0]["kspc_skpv_with_x"][b]["ate"] for b in rows[0]["kspc_skpv_with_x"]}
                continue
            block[name] = {b: {"mae": float(np.mean([abs(r["kspc_skpv_with_x"][b]["ate"] - r["tau"]) for r in rows])),
                               "mean_error": float(np.mean([r["kspc_skpv_with_x"][b]["ate"] - r["tau"] for r in rows])),
                               "n": len(rows)}
                           for b in ("W1", "W4")}
        summary["experiments"][exp] = {"complete": len(shards) == expected, "tasks": len(shards),
                                       "expected": expected, "results": block}
        print(exp, json.dumps(block)[:400])
    with open(out_path, "x") as handle:
        handle.write(json.dumps(summary, indent=1) + "\n")
    print(f"written {out_path}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("run", "summarise"))
    ap.add_argument("experiments", nargs="*")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    raise SystemExit(run(args.experiments, args.workers) if args.command == "run" else summarise())
