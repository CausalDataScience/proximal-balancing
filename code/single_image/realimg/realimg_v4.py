"""Benchmark v4: block-additive representation, so that hiding a block removes exactly that block's information.

Diagnosis (memo section 17): with whole-image MAE pooled features, which blocks are visible changes the embedding
globally, so U-recoverability depended on the visible set and on which set U was defined from (R^2 0.98 vs 0.90).
v4 removes that artifact:
  block features  f_k = MAE trunk pooled features of the image with only block k visible (128), reduced to Q=16 PCs
                        fitted on all rows (unsupervised, role-blind), standardised           -> 5 x 16 = 80 features
  U (lin)         = cross-fitted ridge prediction of the pneumonia label from the lung blocks' features (W1-W4, 64 dims),
                    standardised.  U is exactly linear in the lung features; W5 carries no U.
  "all blocks"    = all 80 features (the CNN_all analogue);  hide H = drop those blocks' 16 features each
  audit           = singleton held-out gaps (Z = other blocks' features, T = sealed-style pixel PCA of block j),
                    relative rule (gap > median + 2 SE) -> hide;  PROBE = AIPW on the remaining blocks' features
Scenarios: V0 clean | V1 tube in W5 (treatment footprint 90/10) | V2 tube inside W2 (lung block) | V3 response blob in W5
(amplitude 0.25).  tau = 1, reps 5.  Oracle = AIPW on U.  Reference = hide the true footprint blocks.
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
import realimg_v3 as v3  # noqa: E402
from realimg_benchmark import pr, nets  # noqa: E402

Q = 16
DELTA2 = 0.005


def pca16(f):
    mu = f.mean(0)
    _, _, vt = np.linalg.svd(f - mu, full_matrices=False)
    z = (f - mu) @ vt[:Q].T
    return (z - z.mean(0)) / (z.std(0) + 1e-9)


def block_feats(trunk, stats, img, masks):
    return [pca16(rb.trunk_features(trunk, stats, img, masks[k])) for k in range(len(masks))]


def u_from(label, feats, seed=5):
    F = np.column_stack(feats); n = len(label)
    fold = np.random.default_rng(seed).permutation(n) % 2
    pred = np.zeros(n)
    X = np.column_stack([np.ones(n), F])
    for k in (0, 1):
        tr, te = fold != k, fold == k
        b = np.linalg.solve(X[tr].T @ X[tr] + 1e-2 * np.eye(X.shape[1]), X[tr].T @ label[tr])
        pred[te] = X[te] @ b
    return (pred - pred.mean()) / pred.std()


def run(scenarios, reps, tag, udef="lin"):
    x, label = rb.load(); m = rb.masks(); trunk, stats, _ = rb.get_mae(x); n = len(x)
    clean = block_feats(trunk, stats, x, m)
    u = u_from(label, clean[:4]) if udef == "lin" else np.load(rb.OUT / "u_logit_crossfit.npy")
    u = (u - u.mean()) / u.std()
    out = rb.OUT / f"v4_{tag}.jsonl"
    for c in scenarios:
        for r in range(reps):
            t0 = time.time()
            rng = np.random.default_rng(20264000 + 100 * c + r)
            a = (rng.random(n) < 0.1 + 0.8 * norm.cdf(1.5 * u)).astype(float)
            y = a + u + rng.standard_normal(n)
            img = x.copy(); true_blocks = []
            tube = rng.random(n) < np.where(a == 1, 0.9, 0.1)
            if c == 1:
                line = np.zeros((64, 64), np.float32); line[4:45, 31:33] = 0.35
                img[tube] = np.clip(img[tube] + line[None] * m[4][None], 0, 1); true_blocks = [4]
            if c == 2:
                line = np.zeros((64, 64), np.float32); line[36:53, 14:16] = 0.35
                img[tube] = np.clip(img[tube] + line[None] * m[1][None], 0, 1); true_blocks = [1]
            if c == 3:
                rr = u - 1.5 * a + 0.5 * rng.standard_normal(n); rt = np.clip((rr - rr.mean()) / rr.std(), -2.5, 2.5)
                yy, xx = np.mgrid[0:64, 0:64]
                blob = (np.exp(-((yy - 20) ** 2 + (xx - 31.5) ** 2) / (2 * 2.0 ** 2)) * m[4]).astype(np.float32)
                img = np.clip(img + 0.25 * rt[:, None, None].astype(np.float32) * blob[None], 0, 1); true_blocks = [4]
            feats = list(clean)
            for j in true_blocks:
                feats[j] = pca16(rb.trunk_features(trunk, stats, img, m[j]))
            seed = 13000 + 100 * c + r
            fold_rows = pr.folds(n, seed); roles = [pr.roles(rot) for rot in range(4)]
            pix = [img[:, m[j]].astype(np.float64) for j in range(5)]
            tfeat = [[v3.t_features(pix[j], fold_rows[roles[rot]["D"]]) for rot in range(4)] for j in range(5)]

            def Z(hidden):
                return np.column_stack([feats[j] for j in range(5) if j not in hidden])

            def aipw(F):
                sc = np.zeros(n)
                for rot in range(4):
                    rn, re = fold_rows[roles[rot]["N"]], fold_rows[roles[rot]["E"]]
                    fn = np.column_stack([np.ones(len(rn)), F[rn]]); fe = np.column_stack([np.ones(len(re)), F[re]])
                    sc[re] = pr.aipw(fn, a[rn], y[rn], fe, a[re], y[re])
                return sc

            gaps = {}
            for j in range(5):
                F = Z([j]); gs, ses = [], []
                for rot in range(4):
                    rs_ = fold_rows[roles[rot]["S"]]
                    sc = pr.screen(np.zeros((len(rs_), 0)), F[rs_], tfeat[j][rot](rs_), a[rs_], np.eye(F.shape[1]), DELTA2)
                    gs.append(sc["gap"]); ses.append(sc["se"])
                gaps[j] = (float(np.mean(gs)), float(np.sqrt(np.mean(np.square(ses)) / 4)))
            med = float(np.median([g for g, _ in gaps.values()]))
            H = [j for j, (g, s) in gaps.items() if g > med + 2 * s]
            s_all, s_probe, s_ref, s_orc = aipw(Z([])), aipw(Z(H)), aipw(Z(true_blocks)), aipw(u[:, None])
            s_lung = aipw(Z([4]))
            rec = {"scenario": c, "rep": r, "seed": seed, "true_blocks": [j + 1 for j in true_blocks],
                   "naive": float(y[a == 1].mean() - y[a == 0].mean()), "oracle": float(s_orc.mean()),
                   "all_blocks": float(s_all.mean()), "lung_blocks_only": float(s_lung.mean()),
                   "PROBE": float(s_probe.mean()), "PROBE_vs_all": float((s_probe - s_all).mean()), "H": [j + 1 for j in H],
                   "ref_true_hidden": float(s_ref.mean()), "gaps": {f"W{j + 1}": gaps[j] for j in range(5)}, "seconds": time.time() - t0}
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            rb.log(f"v4[{udef}] V{c} r{r}: naive {rec['naive']:.3f} oracle {rec['oracle']:.3f} all {rec['all_blocks']:.3f} lungs {rec['lung_blocks_only']:.3f} "
                   f"| PROBE {rec['PROBE']:.3f} H{rec['H']} | ref {rec['ref_true_hidden']:.3f} | gaps {[round(gaps[j][0], 4) for j in range(5)]} | {rec['seconds']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=lambda s: [int(v) for v in s.split(",")], default=[0, 1, 2, 3])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="r5")
    ap.add_argument("--udef", default="lin", choices=["lin", "logit"])
    a = ap.parse_args()
    nets.setup_torch(2)
    run(a.scenarios, a.reps, a.tag, a.udef)
