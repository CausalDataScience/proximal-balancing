"""PROBE with the full CNN representation and block selection (no unsupervised compression).

Diagnostic on 3 replicates showed the warm-start bias came from PCA compression (CNN_all 128 -> PCA32 adds
about +0.22) and only little from masking one block (<= +0.10).  So Z is kept at the full 128 MAE features.
Balance is used to choose which blocks the representation may see:

  candidates (S, R): S = one held-out block (T = its sealed pixel-Prep features),
                     R = all other blocks, or all other blocks minus one dropped block (5 x 5 = 25 candidates);
  Z_{S,R} = MAE-trunk features (128) of the image with only the blocks in R (and the unassigned margins) visible;
  per rotation: sealed honest screen of A _||_ T | Z on fold S (b = identity), sealed AIPW on N -> E;
  rule (pre-registered): keep candidates whose max upper over 4 rotations <= 0.005 with overlap; estimate = median
  theta over kept candidates.  Also reported: CNN_all and every candidate theta, paired with the same E rows.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import realimg_benchmark as rb  # noqa: E402
from realimg_benchmark import pr, level_a, nets  # noqa: E402

DELTA2 = 0.005


def candidates():
    out = []
    for s in range(5):
        rest = [j for j in range(5) if j != s]
        out.append((s, tuple(rest), None))
        for drop in rest:
            out.append((s, tuple(j for j in rest if j != drop), drop))
    return out


def cname(s, r, drop):
    return f"S{s + 1}" + ("" if drop is None else f"-drop{drop + 1}")


def run(scenarios, reps, tag):
    x, u_star = rb.load()
    m = rb.masks()
    trunk, stats, _ = rb.get_mae(x)
    out = rb.OUT / f"select_{tag}.jsonl"
    cands = candidates()
    for c in scenarios:
        for r in range(reps):
            t0 = time.time()
            d = rb.draw(c, r, x, u_star, m)
            img, a, y = d["img"], d["A"], d["Y"]
            n = len(a)
            seed = 7000 + 100 * c + r
            pix = [img[:, m[k]].astype(np.float64) for k in range(5)]
            view = level_a.make_view(np.zeros((n, 0)), pix, a, y)
            fold_rows = pr.folds(n, seed)
            feats_all = rb.trunk_features(trunk, stats, img)
            cache = {}
            res = {}
            score_all = np.zeros(n)
            preps = {rot: pr.Prep(view, fold_rows[pr.roles(rot)["D"]]) for rot in range(4)}
            for rot in range(4):
                rl = pr.roles(rot)
                rn, re = fold_rows[rl["N"]], fold_rows[rl["E"]]
                mu, sd = feats_all[rn].mean(0), feats_all[rn].std(0) + 1e-12
                fn = np.column_stack([np.ones(len(rn)), (feats_all[rn] - mu) / sd])
                fe = np.column_stack([np.ones(len(re)), (feats_all[re] - mu) / sd])
                score_all[re] = pr.aipw(fn, a[rn], y[rn], fe, a[re], y[re])
            for s, rset, drop in cands:
                key = cname(s, rset, drop)
                if rset not in cache:
                    vis = np.zeros((64, 64), bool)
                    vis |= ~m.any(0)                      # unassigned margins always visible
                    for j in rset:
                        vis |= m[j]
                    cache[rset] = rb.trunk_features(trunk, stats, img, vis)
                f = cache[rset]
                score = np.zeros(n)
                scr = []
                for rot in range(4):
                    rl = pr.roles(rot)
                    rs, rn, re = fold_rows[rl["S"]], fold_rows[rl["N"]], fold_rows[rl["E"]]
                    mu, sd = f[rn].mean(0), f[rn].std(0) + 1e-12
                    z = lambda rows: (f[rows] - mu) / sd
                    t_s = preps[rot].block(view, rs, s)
                    sc = pr.screen(np.zeros((len(rs), 0)), z(rs), t_s, a[rs], np.eye(f.shape[1]), DELTA2)
                    scr.append(sc)
                    fn = np.column_stack([np.ones(len(rn)), z(rn)])
                    fe = np.column_stack([np.ones(len(re)), z(re)])
                    score[re] = pr.aipw(fn, a[rn], y[rn], fe, a[re], y[re])
                d_all = score - score_all
                res[key] = {"theta": float(score.mean()), "paired_vs_CNN_all": float(d_all.mean()),
                            "paired_se": float(d_all.std(ddof=1) / np.sqrt(n)),
                            "pooled_gap": float(np.mean([q["gap"] for q in scr])),
                            "max_upper": float(max(q["upper"] for q in scr)),
                            "overlap_all": bool(all(q["overlap"] for q in scr))}
                res[key]["pooled_se"] = float(np.sqrt(np.mean([q["se"] ** 2 for q in scr]) / 4))
            # CNN_all as the starting point: audit each block with the held-out test S{j} (Z = CNN_all with only j
            # hidden), drop flagged blocks, and re-estimate with CNN_all on the image with the flagged blocks hidden.
            gaps = {j: res[f"S{j + 1}"]["pooled_gap"] for j in range(5)}
            med = float(np.median(list(gaps.values())))
            flagged = [j for j in range(5) if gaps[j] > med + 2 * res[f"S{j + 1}"]["pooled_se"]]
            flagged_abs = [j for j in range(5) if res[f"S{j + 1}"]["max_upper"] > DELTA2]
            def cnn_all_without(drop):
                if not drop:
                    return float(score_all.mean()), 0.0, 0.0
                vis = np.ones((64, 64), bool)
                for j in drop:
                    vis &= ~m[j]
                f = rb.trunk_features(trunk, stats, img, vis)
                sc = np.zeros(n)
                for rot in range(4):
                    rl = pr.roles(rot)
                    rn, re = fold_rows[rl["N"]], fold_rows[rl["E"]]
                    mu, sd = f[rn].mean(0), f[rn].std(0) + 1e-12
                    fn = np.column_stack([np.ones(len(rn)), (f[rn] - mu) / sd])
                    fe = np.column_stack([np.ones(len(re)), (f[re] - mu) / sd])
                    sc[re] = pr.aipw(fn, a[rn], y[rn], fe, a[re], y[re])
                dd = sc - score_all
                return float(sc.mean()), float(dd.mean()), float(dd.std(ddof=1) / np.sqrt(n))
            audit = {"flagged_relative": [j + 1 for j in flagged], "flagged_absolute": [j + 1 for j in flagged_abs]}
            audit["final_relative"], audit["final_relative_paired"], audit["final_relative_se"] = cnn_all_without(flagged)
            audit["final_absolute"], audit["final_absolute_paired"], audit["final_absolute_se"] = cnn_all_without(flagged_abs)
            kept = [k for k, v in res.items() if v["max_upper"] <= DELTA2 and v["overlap_all"]]
            best = min(res, key=lambda k: res[k]["pooled_gap"])
            rec = {"scenario": c, "rep": r, "seed": seed, "CNN_all": float(score_all.mean()), "candidates": res,
                   "kept": kept, "select_estimate": float(np.median([res[k]["theta"] for k in kept])) if kept else None,
                   "min_gap_candidate": best, "min_gap_theta": res[best]["theta"], "audit": audit, "seconds": time.time() - t0}
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            g5 = np.mean([res[cname(s, None, None)]["pooled_gap"] for s in range(4)]) if False else None
            rb.log(f"C{c} r{r}: CNN_all {rec['CNN_all']:.3f} | audit drop {audit['flagged_relative']} -> {audit['final_relative']:.3f} "
                   f"(abs drop {audit['flagged_absolute']} -> {audit['final_absolute']:.3f}) | singleton gaps {[round(gaps[j], 4) for j in range(5)]}")
            rb.log(f"C{c} r{r}: select {rec['select_estimate']} (kept {len(kept)}: {kept[:8]}) | "
                   f"min-gap {best} {res[best]['theta']:.3f} | S1 {res['S1']['theta']:.3f} S1-drop5 {res['S1-drop5']['theta']:.3f} "
                   f"gaps S1 {res['S1']['pooled_gap']:.4f} S1-drop5 {res['S1-drop5']['pooled_gap']:.4f} S5 {res['S5']['pooled_gap']:.4f} | {rec['seconds']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=lambda s: [int(v) for v in s.split(",")], default=[0, 1, 2])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--tag", default="all")
    a = ap.parse_args()
    nets.setup_torch(2)
    run(a.scenarios, a.reps, a.tag)
