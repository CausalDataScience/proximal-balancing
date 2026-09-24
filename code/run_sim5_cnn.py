"""Stage runner for the Sim5 CNN operational arm (PRD memo/2026-09-18-sim5-cnn-implementation-prd-v1.md).

  python run_sim5_cnn.py --stage smoke     tiny end-to-end check, writes results/sim5_cnn/smoke_v1.json
  python run_sim5_cnn.py --stage freeze    writes the dev protocol (constants, seeds, source hashes)
  python run_sim5_cnn.py --stage dev       one development dataset under the frozen dev protocol

Three methods are compared on the same rows: the raw-image CNN, PROBE with the warm representation (every step
of the pipeline except balance learning), and PROBE with balance learning.  X-only AIPW and the mean and median
of the screened estimates are recorded as ablations.  Truth is tau = 1.  The run does not certify theorem
conditions (user decision, 2026-09-18).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import sim5_cnn_algorithm1 as a1
import sim5_cnn_dgp as g
import sim5_cnn_models as nm

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "sim5_cnn"
SOURCES = ("sim5_cnn_dgp.py", "sim5_cnn_models.py", "sim5_cnn_algorithm1.py", "run_sim5_cnn.py")

CONFIGS = {
    "smoke": {"n_D": 600, "n_N": 600, "n_E": 2000, "m": 3, "warm_epochs": 3, "balance_updates": 10,
              "checkpoints": [0, 5, 10], "raw_restarts": 1, "raw_pre_epochs": 2, "raw_fine_epochs": 2,
              "data_seed": 71_900_000, "split_seed": 71_900_001, "algo_seed": 71_900_002},
    "dev": {"n_D": 6000, "n_N": 6000, "n_E": 20000, "m": 128, "warm_epochs": 60, "balance_updates": 100,
            "checkpoints": [0, 25, 50, 75, 100], "raw_restarts": 3, "raw_pre_epochs": 30, "raw_fine_epochs": 60,
            "data_seed": 71_000_000, "split_seed": 71_000_001, "algo_seed": 71_000_002},
}
COMMON = {"t": 0.002, "delta": 0.05, "eta": nm.ETA, "code_dim": nm.CODE_DIM, "d_val_frac": 0.2, "n_val_frac": 0.2,
          "refresh_every": 5, "refresh_epochs": 3, "balance_lr": 1e-3, "tau": 1.0}


def sha(name):
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def roles(cfg, seed):
    n_d, n_n, n_e = cfg["n_D"], cfg["n_N"], cfg["n_E"]
    rng = np.random.default_rng(seed)
    d = np.arange(n_d); nn_ = np.arange(n_d, n_d + n_n); e = np.arange(n_d + n_n, n_d + n_n + n_e)
    dp, np_ = rng.permutation(d), rng.permutation(nn_)
    k_d, k_n = int(round(n_d * (1 - COMMON["d_val_frac"]))), int(round(n_n * (1 - COMMON["n_val_frac"])))
    r = {"D": d, "N": nn_, "E": e, "D_fit": np.sort(dp[:k_d]), "D_val": np.sort(dp[k_d:]),
         "N_fit": np.sort(np_[:k_n]), "N_val": np.sort(np_[k_n:])}
    assert not (set(d) & set(nn_)) and not (set(d) & set(e)) and not (set(nn_) & set(e))
    return r


def aipw_summary(e, m0, m1, a, y):
    s = a1.aipw_scores(e, m0, m1, a, y)
    return float(s.mean()), float(s.std(ddof=1))


def arm_result(splits, key, t, n_e, delta):
    kept = [s for s in splits if s[key]["clipped_gap"] <= t]
    est = [s[key]["theta"] for s in kept]
    rho = a1.operational_radius([s[key]["sd"] for s in kept], n_e, delta)
    agg = a1.aggregate([s["word"] for s in kept], est, rho)
    return {"R": len(kept), "rho": rho if np.isfinite(rho) else None,
            "return_kind": agg["return_kind"], "no_return_reason": agg["reason"], "estimate": agg["output"],
            "candidates": agg["candidates"], "component_sizes": agg.get("component_sizes", []),
            "screened_mean": float(np.mean(est)) if est else None,
            "screened_median": float(np.median(est)) if est else None,
            "all_splits_median": float(np.median([s[key]["theta"] for s in splits]))}


def run(stage: str, cfg: dict, protocol: dict | None):
    t_start = time.time()
    torch.set_num_threads(torch.get_num_threads())
    n = cfg["n_D"] + cfg["n_N"] + cfg["n_E"]
    data = g.generate(n, cfg["data_seed"])
    view = g.learner_view(data)                                   # the learner sees X, W, A, Y only
    W = torch.from_numpy(view["W"]); X = torch.from_numpy(view["X"]).float()
    A = torch.from_numpy(view["A"]).float(); Y = torch.from_numpy(view["Y"]).float()
    r = roles(cfg, cfg["algo_seed"])
    tD = {k: torch.from_numpy(v) for k, v in r.items()}
    d_fit_local = torch.from_numpy(np.searchsorted(r["D"], r["D_fit"]))
    d_val_local = torch.from_numpy(np.searchsorted(r["D"], r["D_val"]))
    masks = g.block_masks()
    words = a1.draw_splits(cfg["m"], cfg["split_seed"])
    keeps = [torch.from_numpy(a1.keep_mask(wd, masks)) for wd in words]
    warm_masks = torch.stack(keeps + [~k for k in keeps])
    ledger = {"probe": {"cnn_backprop_images": 0, "cnn_forward_images": 0},
              "raw": {"cnn_backprop_images": 0, "cnn_forward_images": 0}}
    timing = {}

    t0 = time.time()
    warm, warm_info = nm.warm_start(W, X, A, Y, warm_masks, tD["D_fit"], tD["D_val"], cfg["algo_seed"] + 1,
                                    max_epochs=cfg["warm_epochs"], ledger=ledger["probe"])
    timing["warm_start_s"] = time.time() - t0
    for p in warm.parameters():
        p.requires_grad_(False)

    a_np, y_np = view["A"], view["Y"]
    e_rows, n_rows = r["E"], r["N"]
    n_fit_local = torch.from_numpy(np.searchsorted(r["N"], r["N_fit"]))
    n_val_local = torch.from_numpy(np.searchsorted(r["N"], r["N_val"]))
    splits = []
    t0 = time.time()
    for i, (wd, keep) in enumerate(zip(words, keeps)):
        ts = time.time()
        f_keep = nm.trunk_features(warm.trunk, W, keep, ledger["probe"])
        h_d = nm.trunk_features(warm.trunk, W[tD["D"]], ~keep, ledger["probe"])
        xd, ad = X[tD["D"]], A[tD["D"]]
        final_head, binfo = nm.balance_head(warm.head, f_keep[tD["D"]], None, h_d, ad, xd, d_fit_local, d_val_local,
                                            cfg["algo_seed"] + 10_000 * (i + 1), updates=cfg["balance_updates"],
                                            refresh=COMMON["refresh_every"], refresh_epochs=COMMON["refresh_epochs"],
                                            lr=COMMON["balance_lr"], checkpoints=tuple(cfg["checkpoints"]))
        rec = {"index": i, "word": str(wd), "held_out_blocks": a1.held_out_blocks(wd),
               "balance": {k: v for k, v in binfo.items() if k != "fit_gap_trace"},
               "fit_gap_first_last": [binfo["fit_gap_trace"][0], binfo["fit_gap_trace"][-1]]}
        ck = {c["update"]: c for c in binfo["checkpoints"]}
        for arm, head in (("warm", warm.head), ("final", final_head)):
            with torch.no_grad():
                z_all = torch.cat([X, head(f_keep, X)], 1)
            pred = nm.fit_nuisance(z_all[tD["N"]], A[tD["N"]], Y[tD["N"]], n_fit_local, n_val_local,
                                   cfg["algo_seed"] + 10_000 * (i + 1) + (1 if arm == "warm" else 2))
            e, m0, m1, raw_p = pred(z_all[tD["E"]])
            theta, sd = aipw_summary(e, m0, m1, a_np[e_rows], y_np[e_rows])
            gsrc = ck[0] if arm == "warm" else ck[binfo["selected_update"]]
            rec[arm] = {"theta": theta, "sd": sd, "abs_error": abs(theta - COMMON["tau"]),
                        "gap": gsrc["gap"], "clipped_gap": gsrc["clipped"], "gap_se": gsrc["se"],
                        "e_min": float(e.min()), "e_max": float(e.max()),
                        "e_unclipped_outside_0p1_0p9": float(np.mean((raw_p < 0.1) | (raw_p > 0.9)))}
        rec["seconds"] = time.time() - ts
        splits.append(rec)
        if i < 2 or (i + 1) % 16 == 0:
            print(f"  split {i + 1}/{len(words)}  {rec['seconds']:.1f}s  warm {rec['warm']['theta']:+.3f} "
                  f"(gap {rec['warm']['gap']:+.4f})  final {rec['final']['theta']:+.3f} (gap {rec['final']['gap']:+.4f}, "
                  f"upd {binfo['selected_update']})", flush=True)
    timing["splits_s"] = time.time() - t0

    t0 = time.time()
    preds = []
    for k in range(cfg["raw_restarts"]):
        pr = nm.fit_raw(W, X, A, Y, tD["D_fit"], tD["D_val"], tD["N_fit"], tD["N_val"], cfg["algo_seed"] + 900_000 + k,
                        pre_epochs=cfg["raw_pre_epochs"], fine_epochs=cfg["raw_fine_epochs"], ledger=ledger["raw"])
        preds.append(pr(W[tD["E"]], X[tD["E"]]))
    e, m0, m1 = (np.mean([p[j] for p in preds], axis=0) for j in range(3))
    raw_theta, raw_sd = aipw_summary(e, m0, m1, a_np[e_rows], y_np[e_rows])
    timing["raw_s"] = time.time() - t0

    px = nm.fit_nuisance(X[tD["N"]], A[tD["N"]], Y[tD["N"]], n_fit_local, n_val_local, cfg["algo_seed"] + 950_000)
    e, m0, m1, _ = px(X[tD["E"]])
    x_theta, x_sd = aipw_summary(e, m0, m1, a_np[e_rows], y_np[e_rows])

    t, delta, n_e = COMMON["t"], COMMON["delta"], cfg["n_E"]
    out = {
        "stage": stage, "complete": True,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol_sha256": None if protocol is None else hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "config": {**cfg, **COMMON}, "source_sha256": {s: sha(s) for s in SOURCES},
        "rows": {k: [int(v.min()), int(v.max()), int(len(v))] for k, v in r.items()},
        "split_words": [str(wd) for wd in words],
        "warm_start": warm_info, "ledger": ledger, "timing_s": {**timing, "total": time.time() - t_start},
        "methods": {
            "raw_cnn": {"estimate": raw_theta, "abs_error": abs(raw_theta - 1), "sd": raw_sd},
            "x_only": {"estimate": x_theta, "abs_error": abs(x_theta - 1), "sd": x_sd},
            "probe_warm": arm_result(splits, "warm", t, n_e, delta),
            "probe_final": arm_result(splits, "final", t, n_e, delta),
        },
        "splits": splits,
    }
    for k in ("probe_warm", "probe_final"):
        est = out["methods"][k]["estimate"]
        out["methods"][k]["abs_error"] = None if est is None else abs(est - 1)
    write_json(OUT / f"{stage}_v1.json", out)
    return out


def protocol_for(stage):
    return {"prd": "memo/2026-09-18-sim5-cnn-implementation-prd-v1.md", "stage": stage,
            "scope": "operational empirical arm; theorem conditions not certified (user decision 2026-09-18)",
            "config": {**CONFIGS[stage], **COMMON}, "source_sha256": {s: sha(s) for s in SOURCES},
            "mnist_sha256": g.MNIST_SHA256, "c_outcome": g.C_OUTCOME,
            "methods": ["raw_cnn", "probe_warm", "probe_final", "x_only (ablation)",
                        "screened mean/median (ablation)"],
            "question": "from the same warm representation, does balance learning reduce |tau_hat - 1|?",
            "stop_rule": "first development result is reported as is; no rescue tuning (PRD decision 15)",
            "timing_probe": {"seed": 71_800_000, "printed": "wall times only, no estimates",
                             "warm_start_s_30_epochs": 43.8, "split_s": [12.1, 13.8], "raw_restart_s": 138.1,
                             "note": "warm start hit the 30-epoch cap with validation loss still falling; the cap is raised to 60 so the pre-balance baseline is converged, as raw fine-tuning is (cap 60)"},
            "deviations_from_prd_section_6": [
                "one mask-conditioned warm start shared by all splits instead of one per split (compute)",
                "trunk frozen after warm start; balance learning updates the per-split head only (compute)",
                "continuous 16-number code instead of the 256-way hard code (the hard code served the literal-theorem arm)",
                "augmented critic reads frozen trunk features of the held-out pixels instead of an independent CNN (compute)",
                "balance: 100 full-batch head updates at lr 1e-3, critics refreshed every 5 (review of 2026-09-17)"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "freeze", "dev"], required=True)
    args = ap.parse_args()
    if args.stage == "freeze":
        p = protocol_for("dev")
        p["frozen_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        path = OUT / "dev_protocol_v1.json"
        if path.exists():
            raise SystemExit(f"{path} exists; a frozen protocol is never overwritten")
        write_json(path, p)
        print("frozen", path)
        return 0
    if args.stage == "dev":
        p = json.loads((OUT / "dev_protocol_v1.json").read_text())
        for s, h in p["source_sha256"].items():
            if sha(s) != h:
                raise SystemExit(f"source changed since the protocol was frozen: {s}")
        if (OUT / "dev_v1.json").exists():
            raise SystemExit("dev_v1.json exists; the development run is not repeated")
        out = run("dev", CONFIGS["dev"], p)
    else:
        out = run("smoke", CONFIGS["smoke"], None)
    for k, v in out["methods"].items():
        print(f"{k:12s} estimate {v['estimate']}  abs_error {v.get('abs_error')}  "
              + (f"R {v['R']} return {v['return_kind']}" if "R" in v else ""))
    print("timing", {k: round(v, 1) for k, v in out["timing_s"].items()})
    print("ledger", out["ledger"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
