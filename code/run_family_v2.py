"""Stage runner for SCM family v2 (PRD memo/2026-09-18-family-v2-role-blind-dimension-prd-v1.md, v2).

    python3 run_family_v2.py freeze-dev              write results/family_v2/dev_protocol_v1.json (write-once)
    python3 run_family_v2.py dev [--workers 6]       run every development task of that protocol
    python3 run_family_v2.py freeze-confirm T1 T2 T3 write results/family_v2/confirm_protocol_v1.json with the
                                                     per-level screen thresholds chosen on development data
    python3 run_family_v2.py confirm [--workers 6]   run every confirmation task of that protocol

A task is one (level, data set).  It runs PROBE, the AIPW baselines, proximal 2SLS, Park SPC and CEVAE on the
same rows and writes one shard, atomically and never over an existing file.  Before any task runs, the
source files, coefficients and rotation digests are compared with the protocol and the run stops on any
difference.  Development shards keep every split's rotation-level screen statistics and estimates, so the
threshold and the aggregate are replayed afterwards by summarize_family_v2.py rather than chosen here.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import family_v2_dgp as g

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "family_v2"
PRD = HERE.parent / "memo" / "2026-09-18-family-v2-role-blind-dimension-prd-v1.md"
SOURCES = ["family_v2_dgp.py", "family_v2_population.py", "family_v2_probe.py", "family_v2_competitors.py",
           "probe_structured_scm.py", "positive_control_scm.py", "run_family_v2.py"]
N = 24_000
MAIN_LEVELS = ("SCM-1", "SCM-2", "SCM-3")
DEV_REPS = 3
CONFIRM_REPS = 20
PROTOCOLS = {"dev": RESULTS / "dev_protocol_v1.json", "confirm": RESULTS / "confirm_protocol_v1.json"}
SHARDS = {"dev": RESULTS / "dev", "confirm": RESULTS / "confirm"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def design_snapshot() -> dict:
    import family_v2_probe as pr
    import family_v2_competitors as fc
    levels = {name: {"index": lv.index, "d_x": lv.d_x, "d_blocks": list(lv.d_blocks), "k": lv.k,
                     "variant": lv.variant, "rotation_sha256": g.rotation_digest(lv)}
              for name, lv in list(g.LEVELS.items()) + list(g.VARIANTS.items())}
    return {"coefficients": {"treat": g.TREAT, "bad_i": g.BAD_I, "proxy_sd": g.PROXY_SD, "noise_sd": g.NOISE_SD,
                             "outcome": g.OUTCOME, "outcome_noise": g.OUTCOME_NOISE, "kappa": g.KAPPA, "tau": g.TAU},
            "levels": levels,
            "probe": {"p_features_max": pr.P_FEATURES, "mp_margin": pr.MP_MARGIN, "link": [pr.LINK_LO, pr.LINK_HI],
                      "ridge": pr.RIDGE, "alpha": pr.ALPHA, "n_splits": pr.N_SPLITS, "outer_iters": pr.OUTER_ITERS,
                      "restarts": pr.RESTARTS, "clip": list(pr.CLIP), "fold_seed_offset": 7},
            "competitors": {"ridge_grid": list(fc.RIDGE_GRID), "cevae": fc.CEVAE_DEFAULTS, "cevae_seed_offset": 11}}


def tasks_for(stage: str) -> list[dict]:
    out = []
    if stage == "dev":
        for name in MAIN_LEVELS:
            lv = g.LEVELS[name]
            for rep in range(DEV_REPS):
                out.append({"level": name, "rep": rep, "seed": g.DEV_SEED + 1_000 * lv.index + rep, "probe": True})
        lv = g.VARIANTS["SCM-1c"]
        for rep in range(DEV_REPS):         # PROBE stopped on SCM-1c by its P0 (no split admits balance)
            out.append({"level": "SCM-1c", "rep": rep, "seed": g.DEV_SEED + 1_000 * lv.index + rep, "probe": False})
    else:
        for name in MAIN_LEVELS:
            lv = g.LEVELS[name]
            for rep in range(CONFIRM_REPS):
                out.append({"level": name, "rep": rep,
                            "seed": g.CONFIRM_SEED + 10_000 * lv.index + 1_000 * rep, "probe": True})
    return out


def freeze(stage: str, thresholds: dict | None = None) -> int:
    path = PROTOCOLS[stage]
    if path.exists():
        raise SystemExit(f"{path} exists; protocols are write-once")
    if stage == "confirm" and not PROTOCOLS["dev"].exists():
        raise SystemExit("the development protocol must exist first")
    payload = {"protocol_id": f"family-v2-{stage}-v1", "stage": stage, "frozen_at_utc": now(),
               "prd": {"path": str(PRD.relative_to(HERE.parent)), "sha256": sha256(PRD)},
               "n": N, "design": design_snapshot(), "tasks": tasks_for(stage),
               "source_sha256": {f: sha256(HERE / f) for f in SOURCES},
               "screen_thresholds": thresholds,
               "notes": {"dev": "thresholds are chosen after development by the PRD section 5 rule; shards keep "
                                "every rotation-level screen statistic so the aggregate is replayed",
                         "confirm": "thresholds, sources and design are frozen; nothing is chosen on these data"}[stage]}
    if stage == "confirm":
        payload["dev_protocol_sha256"] = sha256(PROTOCOLS["dev"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    tmp.rename(path)
    print(f"frozen {path} ({len(payload['tasks'])} tasks), sha256 {sha256(path)}")
    return 0


def check_protocol(stage: str) -> dict:
    path = PROTOCOLS[stage]
    if not path.exists():
        raise SystemExit(f"no protocol at {path}")
    proto = json.loads(path.read_text())
    bad = [f for f, h in proto["source_sha256"].items() if sha256(HERE / f) != h]
    if bad:
        raise SystemExit(f"source files differ from the frozen protocol: {bad}")
    if json.dumps(design_snapshot(), sort_keys=True) != json.dumps(proto["design"], sort_keys=True):
        raise SystemExit("the design in the code differs from the frozen protocol")
    return proto


def shard_path(stage: str, task: dict) -> Path:
    return SHARDS[stage] / f"{task['level']}_seed{task['seed']}.json"


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def run_task(stage: str, task: dict, proto_sha: str, tolerance: float | None) -> dict:
    import torch
    torch.set_num_threads(1)
    import family_v2_competitors as fc
    import family_v2_population as fp
    import family_v2_probe as pr
    out_path = shard_path(stage, task)
    if out_path.exists():
        return {"task": task, "status": "exists"}
    t_all = time.time()
    lv = g.level(task["level"])
    data = g.generate(lv, N, task["seed"])
    view = g.learner_view(data)
    fold_seed = task["seed"] + 7
    shard = {"protocol_sha256": proto_sha, "stage": stage, "task": task, "n": N, "generated_utc": now(),
             "timing_s": {}}
    if task["probe"]:
        t0 = time.time()
        res = pr.run_dataset(view, lv, 1.0 if tolerance is None else tolerance, fold_seed, keep_b=True)
        shard["timing_s"]["probe"] = round(time.time() - t0, 1)
        t0 = time.time()
        fold_rows = pr.folds(N, fold_seed)
        preps = {rot: pr.Prep(view, fold_rows[pr.roles(rot)["D"]]) for rot in range(4)}
        for name, rec in res["records"].items():
            split = tuple(int(c) - 1 for c in name[1:])
            for r in rec["rotations"]:
                lt = fp.learned_target(lv, preps[r["rotation"]], np.array(r["b"]), split)
                r["learned_tau"], r["learned_d2"] = lt["tau_z"], lt["d2"]
        shard["timing_s"]["learned_target"] = round(time.time() - t0, 1)
        shard["probe"] = res
    t0 = time.time()
    shard["baselines"] = fc.aipw_baselines(data, fold_seed) if lv.variant == "main" else _baselines_1c(data, fold_seed)
    shard["timing_s"]["baselines"] = round(time.time() - t0, 1)
    if lv.variant == "main":
        t0 = time.time()
        shard["proximal"] = fc.proximal_rows(view)
        shard["timing_s"]["proximal"] = round(time.time() - t0, 2)
    t0 = time.time()
    cev = fc.cevae_ate(view, task["seed"] + 11)
    shard["cevae"] = {"ate": cev["ate"], "final_loss": cev["final_loss"]}
    shard["timing_s"]["cevae"] = round(time.time() - t0, 1)
    if stage == "dev" and task["rep"] == 0 and lv.variant == "main" and lv.name == "SCM-1":
        t0 = time.time()
        shard["cevae_exact_proxy_check"] = _cevae_exact_proxy(data, task["seed"] + 13)
        shard["timing_s"]["cevae_exact_proxy_check"] = round(time.time() - t0, 1)
    shard["timing_s"]["total"] = round(time.time() - t_all, 1)
    shard = _jsonable(shard)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(shard, allow_nan=False) + "\n")
    tmp.rename(out_path)
    return {"task": task, "status": "written", "seconds": shard["timing_s"]["total"],
            "estimate": shard.get("probe", {}).get("estimate"), "cevae": shard["cevae"]["ate"]}


def _baselines_1c(data: dict, seed: int) -> dict:
    """SCM-1c has the same learner view layout, so the AIPW baselines apply; the oracle columns are the same."""
    import family_v2_competitors as fc
    return fc.aipw_baselines(data, seed)


def _cevae_exact_proxy(data: dict, seed: int) -> dict:
    """Implementation check, not an experiment: replace W_1 by U itself.  With an exact proxy of the confounder,
    a correct CEVAE should land near tau; if it does not, the implementation is suspect."""
    import family_v2_competitors as fc
    view = g.learner_view(data)
    view = dict(view, W1=data["U_eval_only"][:, None].copy())
    return {"ate": fc.cevae_ate(view, seed)["ate"], "note": "W1 := U exactly"}


def run(stage: str, workers: int, only: str | None) -> int:
    proto = check_protocol(stage)
    proto_sha = sha256(PROTOCOLS[stage])
    if stage == "confirm" and not proto.get("screen_thresholds"):
        raise SystemExit("the confirmation protocol has no screen thresholds")
    todo = [t for t in proto["tasks"] if (only is None or t["level"] == only) and not shard_path(stage, t).exists()]
    print(f"{stage}: {len(todo)} tasks to run with {workers} workers (protocol {proto_sha[:12]})", flush=True)
    thresholds = proto.get("screen_thresholds") or {}
    failures = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_task, stage, t, proto_sha, thresholds.get(t["level"])): t for t in todo}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                r = fut.result()
                print(f"  {t['level']} seed {t['seed']}: {r['status']} {r.get('seconds')} s "
                      f"PROBE {r.get('estimate')} CEVAE {r.get('cevae')}", flush=True)
            except Exception:
                failures += 1
                print(f"  {t['level']} seed {t['seed']}: FAILED\n{traceback.format_exc()}", flush=True)
    print(f"{stage} finished with {failures} failures", flush=True)
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["freeze-dev", "dev", "freeze-confirm", "confirm"])
    ap.add_argument("thresholds", nargs="*", type=float)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None)
    args = ap.parse_args(argv[1:])
    if args.command == "freeze-dev":
        return freeze("dev")
    if args.command == "freeze-confirm":
        if len(args.thresholds) != len(MAIN_LEVELS):
            raise SystemExit("give one threshold per main level, in the order SCM-1 SCM-2 SCM-3")
        return freeze("confirm", dict(zip(MAIN_LEVELS, args.thresholds)))
    return run(args.command, args.workers, args.only)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
