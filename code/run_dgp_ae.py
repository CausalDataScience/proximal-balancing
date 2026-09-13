#!/usr/bin/env python3
"""Runner for DGP A to E (plan sections 8 to 12).

One invocation runs a list of (DGP, n, replicate) cells and writes one JSON record per replicate.

  python3 run_dgp_ae.py --dgp A --n 6000 --replicates 10000 10001 --phase pilot --jobs 5 --out ../results/x.json

Each replicate record holds every candidate's audit gap, screens, per-rotation estimate, the pooled estimate,
the linking radius, the component structure, the comparator values, and timing.  Nothing is averaged away.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import probe_brier_neural as pbn
import probe_dgp_ae as dgp

RESULTS = Path(__file__).resolve().parents[1] / "results"
COMPARATOR_SEED = {"X": 101, "XC": 202, "raw": 303, "oracle": 404}


def folds(n: int, dgp_name: str, n_size: int, replicate: int, phase: str) -> list[np.ndarray]:
    g = np.random.default_rng(np.random.SeedSequence(dgp.run_key("unit_split", phase, dgp_name, n_size, replicate)))
    perm = g.permutation(n)
    return [np.sort(p) for p in np.array_split(perm, 3)]


def d_subsplit(d_idx: np.ndarray, dgp_name: str, n_size: int, replicate: int, rotation: int, phase: str):
    """Plan 8.3: 50/25/25 fit, val, audit inside D."""
    g = np.random.default_rng(np.random.SeedSequence(
        dgp.run_key("encoder", phase, dgp_name, n_size, replicate, rotation) + (999,)))
    perm = d_idx[g.permutation(len(d_idx))]
    a, b = int(0.5 * len(perm)), int(0.75 * len(perm))
    return perm[:a], perm[a:b], perm[b:]


_CTX: dict = {}


def _worker_init(threads: int, data: dict, st: dict) -> None:
    """The pool uses the spawn start method: on macOS a forked child crashes when torch has already
    initialised its Objective-C graph runtime in the parent.  The sample and the standardizers are therefore
    pickled once per worker, not once per task."""
    torch.set_num_threads(threads)
    _CTX["data"], _CTX["st"] = data, st


def _candidate_job(payload):
    """Reads the sample and the standardizers from the forked context so the dataset is never pickled."""
    (S, d_fit, d_val, d_audit, n_idx, e_idx, key_enc, key_nuis) = payload
    data, st = _CTX["data"], _CTX["st"]
    t0 = time.time()
    fitted = pbn.train_representation(data, S, d_fit, d_val, st, key_enc)
    audit = pbn.audit_gap(data, S, fitted, d_audit, st)
    refit = pbn.refit_critics(data, S, fitted, d_fit, d_val, d_audit, st, key_nuis + 31)
    z_n, z_e = pbn.embed(data, fitted, n_idx, st), pbn.embed(data, fitted, e_idx, st)
    nuis = pbn.fit_nuisances(z_n, data["A"][n_idx], data["Y"][n_idx], key_nuis)
    ov = pbn.overlap_pass(nuis, z_n)
    est = pbn.aipw_scores(nuis, z_e, data["A"][e_idx], data["Y"][e_idx])
    rec = {"S": list(S), "subset_code": dgp.subset_code(S), "structural_valid": dgp.structural_valid(S),
           "balance_feasible": dgp.balance_feasible(data["dgp"], S),
           **audit, **refit, **ov, "theta": est["theta"], "sigma": est["sigma"], "clip_rate": est["clip_rate"],
           "val_gap": fitted["val_gap"], "best_epoch": fitted["epoch"], "restart": fitted["restart"],
           "val_overlap_violation": fitted["val_overlap_violation"],
           "encoder_updates": fitted["encoder_updates"], "critic_updates": fitted["critic_updates"],
           "restart_eligible": fitted["any_restart_eligible"],
           # Amendment of 2026-09-11: selection uses the audit computed with the representation frozen and
           # both critics refitted.  The plan's training-critic audit is kept in the record as `d2_raw` and
           # `discrepancy_pass`; it was shown to invert the signal.
           "refit_threshold": 0.02 + 2 * refit["refit_se_audit"],
           "refit_pass": bool(refit["refit_gap_audit"] <= 0.02 + 2 * refit["refit_se_audit"]),
           "training_ok": bool(fitted["epoch"] >= 0 and fitted["encoder_updates"] >= pbn.MIN_ENCODER_UPDATES),
           "fit_gap": fitted["fit_gap"],
           **pbn.screen_decision(refit["refit_gap_audit"], refit["refit_se_audit"], ov["overlap_pass"],
                                 fitted["any_restart_eligible"], fitted["epoch"], fitted["encoder_updates"]),
           "screen_pass_plan": bool(audit["discrepancy_pass"] and ov["overlap_pass"] and fitted["any_restart_eligible"]),
           "seconds": time.time() - t0}
    return rec, est["psi"]


def comparator_features(data: dict, kind: str) -> np.ndarray:
    if kind == "X":
        return data["X"]
    if kind == "XC":
        return dgp.xc_matrix(data)
    if kind == "raw":
        return dgp.raw_matrix(data)
    if kind == "oracle":
        return dgp.oracle_matrix(data)
    raise ValueError(kind)


def run_replicate(dgp_name: str, n: int, replicate: int, phase: str, args) -> dict:
    t_start = time.time()
    coef = dgp.coefficients()
    data = dgp.generate(dgp_name, n, replicate, phase, coef)
    f = folds(n, dgp_name, n, replicate, phase)
    subsets = dgp.all_subsets()
    if args.subset_limit:
        subsets = subsets[:args.subset_limit]
    rotations = range(3) if not args.single_rotation else range(1)

    per_rot: list[dict] = []
    psi_pooled: dict[int, np.ndarray] = {}
    comp_psi: dict[str, np.ndarray] = {k: np.zeros(n) for k in ("X", "XC", "raw", "oracle")}
    comp_seen = np.zeros(n, dtype=bool)
    naive_vals = []

    for k in rotations:
        d_idx, n_idx, e_idx = f[k], f[(k + 1) % 3], f[(k + 2) % 3]
        d_fit, d_val, d_audit = d_subsplit(d_idx, dgp_name, n, replicate, k, phase)
        st = pbn.build_standardizers(data, d_fit)
        jobs = []
        for S in subsets:
            code = dgp.subset_code(S)
            jobs.append((S, d_fit, d_val, d_audit, n_idx, e_idx,
                         dgp.run_key("encoder", phase, dgp_name, n, replicate, k, code),
                         int(np.random.SeedSequence(dgp.run_key("nuisance", phase, dgp_name, n, replicate, k, code)).generate_state(1)[0] % (2 ** 31))))
        _CTX["data"], _CTX["st"] = data, st
        if args.jobs > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.jobs, initializer=_worker_init, initargs=(args.threads, data, st)) as pool:
                results = pool.map(_candidate_job, jobs, chunksize=1)
        else:
            results = [_candidate_job(j) for j in jobs]
        recs = []
        for (rec, psi), S in zip(results, subsets):
            rec["rotation"] = k
            recs.append(rec)
            psi_pooled.setdefault(dgp.subset_code(S), np.full(n, np.nan))[e_idx] = psi
        comps = {}
        image_dgp = any(b["kind"] != "tabular" for b in data["blocks"])
        for kind in ("X", "XC", "raw", "oracle"):
            # A fixed integer per comparator: Python string hashing is salted per process, so hash(kind)
            # would change the seed between runs of the same configuration.
            seed = int(np.random.SeedSequence(dgp.run_key("nuisance", phase, dgp_name, n, replicate, k, 0)
                                              + (COMPARATOR_SEED[kind],)).generate_state(1)[0] % (2 ** 31))
            if kind == "raw" and image_dgp:
                # plan 9.2: the image raw baseline learns its own CNN end to end, with the same architecture,
                # restart count and epoch cap as the representation.
                nuis = pbn.fit_nuisances_blocks(data, n_idx, st, seed)
                est = pbn.aipw_scores_blocks(nuis, data, e_idx)
            else:
                feat = comparator_features(data, kind)
                nuis = pbn.fit_nuisances(feat[n_idx], data["A"][n_idx], data["Y"][n_idx], seed)
                est = pbn.aipw_scores(nuis, feat[e_idx], data["A"][e_idx], data["Y"][e_idx])
            comps[kind] = {"theta": est["theta"], "clip_rate": est["clip_rate"]}
            comp_psi[kind][e_idx] = est["psi"]
        comp_seen[e_idx] = True
        ye, ae = data["Y"][e_idx], data["A"][e_idx]
        naive_vals.append(float(ye[ae == 1].mean() - ye[ae == 0].mean()))
        per_rot.append({"rotation": k, "candidates": recs, "comparators": comps})

    pass_counts = {}
    for rot in per_rot:
        for rec in rot["candidates"]:
            pass_counts.setdefault(rec["subset_code"], []).append(rec["screen_pass"])
    retained = [c for c, v in pass_counts.items() if len(v) == len(list(rotations)) and all(v)]
    retained.sort()

    thetas = np.array([np.nanmean(psi_pooled[c]) for c in retained]) if retained else np.zeros(0)
    if retained:
        mat = np.stack([psi_pooled[c] for c in retained])
        rho = pbn.linking_radius(mat, thetas, int(np.random.SeedSequence(
            dgp.run_key("multiplier", phase, dgp_name, n, replicate)).generate_state(1)[0] % (2 ** 31)))
    else:
        rho = 0.0
    agg = pbn.aggregate(retained, thetas, rho)
    sens = {f"b={b}": pbn.aggregate(retained, thetas, rho, extra=b)["output"] for b in (0.025, 0.05, 0.10)}

    comparators = {k: float(comp_psi[k][comp_seen].mean()) for k in comp_psi}
    comparators["naive"] = float(np.mean(naive_vals))

    out = {"dgp": dgp_name, "n": n, "replicate": replicate, "phase": phase,
           "rotations": per_rot, "retained": retained,
           "retained_valid": [c for c in retained if dgp.structural_valid([b for b in range(5) if c >> b & 1])],
           "retained_feasible": [c for c in retained if dgp.balance_feasible(dgp_name, [b for b in range(5) if c >> b & 1])],
           "thetas": thetas.tolist(), "algorithm1": agg, "linking_sensitivity": sens,
           "comparators": comparators, "seconds": time.time() - t_start,
           "n_subsets": len(subsets), "n_rotations": len(list(rotations))}
    return out


def environment() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    except Exception:
        commit, dirty = "unknown", True
    import scipy, sklearn
    coef = dgp.coefficients()
    return {"git_commit": commit, "git_dirty": dirty, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "torch": torch.__version__, "platform": platform.platform(),
            "coefficients": {k: coef[k].tolist() for k in ("b_x", "a_x", "y_x", "yt_x", "b_w")},
            "mnist_pool_digest": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", nargs="+", default=["A"], choices=list(dgp.DGP_ID))
    ap.add_argument("--n", type=int, nargs="+", default=[6000])
    ap.add_argument("--replicates", type=int, nargs="+", required=True)
    ap.add_argument("--phase", default="pilot", choices=["pilot", "confirmatory"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--subset-limit", type=int, default=0, help="debug only: use the first k of the 30 subsets")
    ap.add_argument("--single-rotation", action="store_true", help="theory-aligned diagnostic: rotation 0 only")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    env = environment()
    if "E" in args.dgp:
        env["mnist_pool_digest"] = dgp.mnist_pool()["digest"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    payload = {"plan": "memo/2026-09-11-dgp-a-e-experiment-plan.md", "environment": env,
               "config": vars(args), "runs": runs, "complete": False}
    for name in args.dgp:
        for n in args.n:
            for rep in args.replicates:
                t0 = time.time()
                rec = run_replicate(name, n, rep, args.phase, args)
                runs.append(rec)
                a = rec["algorithm1"]
                print(f"[{name} n={n} rep={rep}] retained {len(rec['retained'])}/{rec['n_subsets']} "
                      f"({len(rec['retained_feasible'])} balance-feasible) | {a['return_kind']} "
                      f"out={a['output']} rho={a['rho']:.3f} | raw={rec['comparators']['raw']:+.3f} "
                      f"oracle={rec['comparators']['oracle']:+.3f} X={rec['comparators']['X']:+.3f} "
                      f"| {time.time()-t0:.0f}s", flush=True)
                tmp = out_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(payload, indent=1, default=float))
                tmp.replace(out_path)
    payload["complete"] = True
    out_path.write_text(json.dumps(payload, indent=1, default=float))
    print("saved", out_path)


if __name__ == "__main__":
    main()
