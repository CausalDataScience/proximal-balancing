"""Benchmark v2 on PneumoniaMNIST: U recoverable from the lungs + treatment footprints in block 5.

U definitions (both image-defined, hidden from every learner, oracle only):
  lin   : cross-fitted ridge prediction of the pneumonia label from lung-only MAE features (U is linear in what the
          lungs show; ORC holds by construction; floor ~ 0 for CNN_all and for lungs-only)
  logit : cross-fitted CNN classifier logit (whole clean image), from realimg_u_recoverable.py (floor ~ 0.14 with
          the sealed linear nuisances)
Scenarios (Y = A + U + eps, tau = 1; P(A=1|U) = 0.1 + 0.8 Phi(1.5 U)):
  D0 clean
  D1 treatment footprint : a tube drawn in block 5 with P(tube|A=1) = 0.9, P(tube|A=0) = 0.1  (device caused by treatment)
  D2 treatment response  : blob in block 5 with amplitude 0.25 * standardised (U - 1.5 A + 0.5 eps)  (post-treatment collider)
Methods on the sealed folds / AIPW: naive, oracle(U), CNN_all, all pixel blocks (sealed Prep, no balance),
  PROBE audit from CNN_all (relative rule: block gap > median + 2 SE -> hide it; final = CNN_all on the image with the
  flagged blocks hidden), reference CNN_all minus W5 (oracle knowledge of the harmful block).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import realimg_benchmark as rb  # noqa: E402
import realimg_select as rs  # noqa: E402
from realimg_benchmark import pr, level_a, nets  # noqa: E402


def u_lin(label, feats_lung, seed=5):
    n = len(label)
    fold = np.random.default_rng(seed).permutation(n) % 2
    pred = np.zeros(n)
    for k in (0, 1):
        tr, te = fold != k, fold == k
        mu, sd = feats_lung[tr].mean(0), feats_lung[tr].std(0) + 1e-9
        F = np.column_stack([np.ones(n), (feats_lung - mu) / sd])
        b = np.linalg.solve(F[tr].T @ F[tr] + 1e-2 * np.eye(F.shape[1]), F[tr].T @ label[tr])
        pred[te] = F[te] @ b
    return (pred - pred.mean()) / pred.std()


def draw(c, r, x, u, m, udef):
    rng = np.random.default_rng(20262000 + 1000 * (udef == "logit") + 100 * c + r)
    n = len(x)
    a = (rng.random(n) < 0.1 + 0.8 * norm.cdf(1.5 * u)).astype(float)
    y = a + u + rng.standard_normal(n)
    img = x.copy()
    if c == 1:
        tube = (rng.random(n) < np.where(a == 1, 0.9, 0.1))
        line = np.zeros((64, 64), np.float32); line[4:45, 31:33] = 0.35
        img[tube] = np.clip(img[tube] + line[None] * m[4][None], 0, 1)
    if c == 2:
        rr = u - 1.5 * a + 0.5 * rng.standard_normal(n)
        rt = np.clip((rr - rr.mean()) / rr.std(), -2.5, 2.5)
        yy, xx = np.mgrid[0:64, 0:64]
        blob = (np.exp(-((yy - 20) ** 2 + (xx - 31.5) ** 2) / (2 * 2.0 ** 2)) * m[4]).astype(np.float32)
        img = np.clip(img + 0.25 * rt[:, None, None].astype(np.float32) * blob[None], 0, 1)
    return a, y, img


def aipw_with_diag(view, feats, seed):
    """Sealed AIPW on arbitrary features plus the fraction of E rows whose propensity was clipped."""
    n = len(view["A"]); fold_rows = pr.folds(n, seed); score = np.zeros(n); clipped = 0.0
    for rot in range(4):
        rl = pr.roles(rot); rn, re = fold_rows[rl["N"]], fold_rows[rl["E"]]
        mu, sd = feats[rn].mean(0), feats[rn].std(0) + 1e-12
        fn = np.column_stack([np.ones(len(rn)), (feats[rn] - mu) / sd]); fe = np.column_stack([np.ones(len(re)), (feats[re] - mu) / sd])
        c, _ = pr.fit_critic(fn, view["A"][rn], None, pr.RIDGE)
        e = pr.link(fe @ c); clipped += float(np.mean((e < pr.CLIP[0]) | (e > pr.CLIP[1]))) / 4
        score[re] = pr.aipw(fn, view["A"][rn], view["Y"][rn], fe, view["A"][re], view["Y"][re])
    return score, clipped


def run(udef, scenarios, reps, tag):
    x, label = rb.load(); m = rb.masks(); trunk, stats, _ = rb.get_mae(x); n = len(x)
    feats_lung_clean = rb.trunk_features(trunk, stats, x, ~m[4])
    u = u_lin(label, feats_lung_clean) if udef == "lin" else np.load(rb.OUT / "u_logit_crossfit.npy")
    u = (u - u.mean()) / u.std()
    out = rb.OUT / f"v2_{udef}_{tag}.jsonl"
    for c in scenarios:
        for r in range(reps):
            t0 = time.time()
            a, y, img = draw(c, r, x, u, m, udef)
            seed = 9000 + 100 * c + r
            pix = [img[:, m[k]].astype(np.float64) for k in range(5)]
            view = level_a.make_view(np.zeros((n, 0)), pix, a, y)
            fold_rows = pr.folds(n, seed)
            feats_all = rb.trunk_features(trunk, stats, img)
            score_all, clip_all = aipw_with_diag(view, feats_all, seed)
            score_orc, _ = aipw_with_diag(view, u[:, None], seed)
            preps = {rot: pr.Prep(view, fold_rows[pr.roles(rot)["D"]]) for rot in range(4)}
            cache = {}
            def feats_hiding(drop):
                key = tuple(sorted(drop))
                if key not in cache:
                    vis = np.ones((64, 64), bool)
                    for j in key:
                        vis &= ~m[j]
                    cache[key] = rb.trunk_features(trunk, stats, img, vis) if key else feats_all
                return cache[key]
            blocks = {}
            for j in range(5):
                f = feats_hiding([j]); scr = []
                for rot in range(4):
                    rl = pr.roles(rot); rs_ = fold_rows[rl["S"]]; rn = fold_rows[rl["N"]]
                    mu, sd = f[rn].mean(0), f[rn].std(0) + 1e-12
                    scr.append(pr.screen(np.zeros((len(rs_), 0)), (f[rs_] - mu) / sd, preps[rot].block(view, rs_, j), a[rs_], np.eye(f.shape[1]), rs.DELTA2))
                blocks[j] = {"gap": float(np.mean([q["gap"] for q in scr])), "se": float(np.sqrt(np.mean([q["se"] ** 2 for q in scr]) / 4)),
                             "max_upper": float(max(q["upper"] for q in scr))}
            med = float(np.median([b["gap"] for b in blocks.values()]))
            flagged = [j for j in range(5) if blocks[j]["gap"] > med + 2 * blocks[j]["se"]]
            score_probe, clip_probe = aipw_with_diag(view, feats_hiding(flagged), seed)
            score_ref, clip_ref = aipw_with_diag(view, feats_hiding([4]), seed)
            score_blocks, _ = (lambda: (None, None))()
            # all pixel blocks after the sealed per-block PCA, no balance
            sb = np.zeros(n)
            for rot in range(4):
                rl = pr.roles(rot); rn, re = fold_rows[rl["N"]], fold_rows[rl["E"]]
                F = lambda rows: np.column_stack([np.ones(len(rows))] + [preps[rot].block(view, rows, j) for j in range(5)])
                sb[re] = pr.aipw(F(rn), a[rn], y[rn], F(re), a[re], y[re])
            pd = lambda s: {"theta": float(s.mean()), "vs_CNN_all": float((s - score_all).mean()), "se_paired": float((s - score_all).std(ddof=1) / np.sqrt(n))}
            rec = {"udef": udef, "scenario": c, "rep": r, "seed": seed, "naive": float(y[a == 1].mean() - y[a == 0].mean()),
                   "oracle": float(score_orc.mean()), "CNN_all": float(score_all.mean()), "CNN_all_clipped_frac": clip_all,
                   "all_blocks_no_balance": float(sb.mean()), "blocks": {f"W{j + 1}": blocks[j] for j in range(5)},
                   "flagged": [j + 1 for j in flagged], "PROBE": pd(score_probe) | {"clipped_frac": clip_probe},
                   "ref_CNN_all_minus_W5": pd(score_ref) | {"clipped_frac": clip_ref}, "seconds": time.time() - t0}
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            rb.log(f"{udef} D{c} r{r}: naive {rec['naive']:.3f} oracle {rec['oracle']:.3f} CNN_all {rec['CNN_all']:.3f} (clip {clip_all:.2f}) "
                   f"blocks {rec['all_blocks_no_balance']:.3f} | PROBE {rec['PROBE']['theta']:.3f} flagged {rec['flagged']} | ref -W5 {rec['ref_CNN_all_minus_W5']['theta']:.3f} "
                   f"| gaps {[round(blocks[j]['gap'], 4) for j in range(5)]} | {rec['seconds']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--udef", choices=["lin", "logit"], required=True)
    ap.add_argument("--scenarios", type=lambda s: [int(v) for v in s.split(",")], default=[0, 1, 2])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="r5")
    a = ap.parse_args()
    nets.setup_torch(2)
    run(a.udef, a.scenarios, a.reps, a.tag)
