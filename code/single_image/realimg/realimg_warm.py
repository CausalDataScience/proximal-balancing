"""PROBE with a warm-started CNN representation on PneumoniaMNIST (benchmark v1 scenarios, user request
"PROBE를 하되 첫 편향을 없애봐. CNN을 warm start 하던지", 2026-10-10).

For each held-out set S (|S| in {1, 2}; 15 splits):
  V_S = MAE-trunk features (128) of the image with the blocks in S masked (cross-block context kept, CNN_all minus S),
        reduced to Q principal components fitted on fold D and standardised;
  T_S = the sealed Prep features (pixel PCA) of the held-out blocks;
  Z0  = the top-K principal components of V_S (the warm start, no balance);
  Z   = B V_S, rank K, learned by the sealed learn_representation (its first start IS the warm start Z0) to minimise
        the Brier gap; then the sealed honest screen on S and the sealed AIPW on N -> E, 4 rotations.
Reported per split: theta for V_S-full (no balance), Z0 (no balance), Z (PROBE); screen gaps; relaxed-rule estimate.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import realimg_benchmark as rb  # noqa: E402
from realimg_benchmark import pr, level_a, nets  # noqa: E402

Q, K, DELTA2 = 32, 8, 0.005
SPLITS = [s for r in (1, 2) for s in itertools.combinations(range(5), r)]


def name(s):
    return "S" + "".join(str(j + 1) for j in s)


def pca_fit(f):
    mu = f.mean(0)
    _, _, vt = np.linalg.svd(f - mu, full_matrices=False)
    proj = vt[:Q].T
    sd = ((f - mu) @ proj).std(0) + 1e-12
    return mu, proj, sd


def run(scenarios, reps, tag):
    x, u_star = rb.load()
    m = rb.masks()
    trunk, stats, _ = rb.get_mae(x)
    out = rb.OUT / f"warm_{tag}.jsonl"
    for c in scenarios:
        for r in range(reps):
            t0 = time.time()
            d = rb.draw(c, r, x, u_star, m)
            img, a, y = d["img"], d["A"], d["Y"]
            n = len(a)
            seed = 7000 + 100 * c + r                       # same folds as the v1 benchmark replicate
            pix = [img[:, m[k]].astype(np.float64) for k in range(5)]
            view = level_a.make_view(np.zeros((n, 0)), pix, a, y)
            feats = {}
            for s in SPLITS:
                vis = np.ones((64, 64), bool)
                for j in s:
                    vis &= ~m[j]
                feats[s] = rb.trunk_features(trunk, stats, img, vis)
            fold_rows = pr.folds(n, seed)
            scores = {s: {"full": np.zeros(n), "z0": np.zeros(n), "probe": np.zeros(n)} for s in SPLITS}
            screens = {s: [] for s in SPLITS}
            for rot in range(4):
                rl = pr.roles(rot)
                rows = {k: fold_rows[v] for k, v in rl.items()}
                prep = pr.Prep(view, rows["D"])
                x0 = {k: np.zeros((len(v), 0)) for k, v in rows.items()}
                for s_idx, s in enumerate(SPLITS):
                    mu, proj, sd = pca_fit(feats[s][rows["D"]])
                    vf = {k: ((feats[s][v] - mu) @ proj) / sd for k, v in rows.items()}
                    tf = {k: np.column_stack([prep.block(view, v, j) for j in s]) for k, v in rows.items()}
                    learned = pr.learn_representation(x0["D"], vf["D"], tf["D"], a[rows["D"]], K, seed + 1000 * rot + s_idx)
                    b = learned["b"]
                    scr = pr.screen(x0["S"], vf["S"], tf["S"], a[rows["S"]], b, DELTA2)
                    screens[s].append({k: scr[k] for k in ("gap", "se", "upper", "overlap")} | {"gap_fit": learned["gap_fit"], "start": learned["start"]})
                    for key, zf in (("full", lambda k: vf[k]), ("z0", lambda k: vf[k][:, :K]), ("probe", lambda k: vf[k] @ b.T)):
                        fn = np.column_stack([np.ones(len(rows["N"])), zf("N")])
                        fe = np.column_stack([np.ones(len(rows["E"])), zf("E")])
                        scores[s][key][rows["E"]] = pr.aipw(fn, a[rows["N"]], y[rows["N"]], fe, a[rows["E"]], y[rows["E"]])
            rec = {"scenario": c, "rep": r, "seed": seed, "Q": Q, "K": K, "splits": {}}
            for s in SPLITS:
                up = [z["upper"] for z in screens[s]]
                rec["splits"][name(s)] = {"theta_full": float(scores[s]["full"].mean()), "theta_z0": float(scores[s]["z0"].mean()),
                                          "theta_probe": float(scores[s]["probe"].mean()),
                                          "pooled_gap": float(np.mean([z["gap"] for z in screens[s]])),
                                          "max_upper": float(max(up)), "overlap_all": bool(all(z["overlap"] for z in screens[s])),
                                          "starts": [z["start"] for z in screens[s]]}
            kept = [k for k, v in rec["splits"].items() if v["max_upper"] <= DELTA2 and v["overlap_all"]]
            rec["relaxed_retained"] = kept
            rec["relaxed_estimate"] = float(np.median([rec["splits"][k]["theta_probe"] for k in kept])) if kept else None
            rec["seconds"] = time.time() - t0
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            sp = rec["splits"]
            rb.log(f"C{c} r{r}: S1-4 full {[round(sp[f'S{j}']['theta_full'], 3) for j in range(1, 5)]} "
                   f"z0 {[round(sp[f'S{j}']['theta_z0'], 3) for j in range(1, 5)]} probe {[round(sp[f'S{j}']['theta_probe'], 3) for j in range(1, 5)]} "
                   f"| gaps S1-4 {[round(sp[f'S{j}']['pooled_gap'], 4) for j in range(1, 5)]} S5 {sp['S5']['pooled_gap']:.4f} "
                   f"| relaxed {rec['relaxed_estimate']} kept {kept} | {rec['seconds']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=lambda s: [int(v) for v in s.split(",")], default=[0, 1, 2])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--tag", default="all")
    a = ap.parse_args()
    nets.setup_torch(2)
    run(a.scenarios, a.reps, a.tag)
