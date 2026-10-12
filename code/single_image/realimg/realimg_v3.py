"""Benchmark v3 on PneumoniaMNIST (approved 2026-10-10: (가) footprints inside a lung block / across two blocks,
(나) a 3x3 grid layout instead of anatomical blocks).  U = lin (lung-linear, floor ~ 0), tau = 1.

Layouts
  anat  : the five v1 blocks (W1-W4 lungs, W5 central strip)
  grid3 : 3x3 cells of ~21 px, no margins, no anatomy knowledge
Scenarios (A, Y as in v2; P(A=1|U) = 0.1 + 0.8 Phi(1.5 U); Y = A + U + eps)
  anat : A0 clean | A1 tube inside W2 (lung block) | A2 tube across W4 and W5 (two blocks, no margin pixels)
         | A3 tube in W5 only (v2 D1 repeated under this code)
  grid3: G0 clean | G1 central tube of v2 (spans cells (0,1) and (1,1)) | G2 response blob of v2 (cell (0,1), amp 0.25)
Audit from CNN_all over singletons and pairs of blocks:
  g_j  : Z = CNN_all with block j hidden,  T = sealed-style PCA (MP edge, <= 4) features of block j
  g_jk : Z = CNN_all with blocks j, k hidden, T = features of both blocks
  flag j      if g_j  > median(singletons) + 2 s_j
  flag {j,k}  if g_jk > median(pairs) + 2 s_jk  and  g_jk > g_j + g_k + 2 s_jk      (the pair reveals more than its parts)
  H_single = flagged singletons;  H_pair = H_single U blocks of flagged pairs
  PROBE_single = CNN_all with H_single hidden;  PROBE_pair = CNN_all with H_pair hidden
Also: naive, oracle, CNN_all, reference CNN_all with the true footprint blocks hidden.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import realimg_benchmark as rb  # noqa: E402
import realimg_v2 as v2  # noqa: E402
from realimg_benchmark import pr, nets  # noqa: E402

DELTA2 = 0.005


def grid_masks():
    edges = [0, 21, 42, 64]
    m = np.zeros((9, 64, 64), bool)
    k = 0
    for r in range(3):
        for c in range(3):
            m[k, edges[r]:edges[r + 1], edges[c]:edges[c + 1]] = True
            k += 1
    return m


def overlays(layout, c, a, rng, m_anat):
    """Return (image overlay (n,64,64) float32, true footprint block ids for the layout's masks)."""
    n = len(a)
    ov = np.zeros((n, 64, 64), np.float32)
    tube = rng.random(n) < np.where(a == 1, 0.9, 0.1)
    if layout == "anat":
        if c == 1:    # inside W2 (rows 33-55, cols 4-27)
            line = np.zeros((64, 64), np.float32); line[36:53, 14:16] = 0.35
            ov[tube] = line * m_anat[1]
            return ov, [1]
        if c == 2:    # across W4 (rows 33-55, cols 36-59) and W5 (cols 29-34): horizontal line rows 44-45, cols 30-50, block pixels only
            line = np.zeros((64, 64), np.float32); line[44:46, 30:51] = 0.35
            ov[tube] = line * (m_anat[3] | m_anat[4])
            return ov, [3, 4]
        if c == 3:    # W5 only (v2 D1)
            line = np.zeros((64, 64), np.float32); line[4:45, 31:33] = 0.35
            ov[tube] = line * m_anat[4]
            return ov, [4]
        return ov, []
    # grid3: v2 footprints placed by the same pixels; true blocks = cells they touch
    if c == 1:
        line = np.zeros((64, 64), np.float32); line[4:45, 31:33] = 0.35
        ov[tube] = line
        return ov, [1, 4]                       # cells (0,1) and (1,1)
    if c == 2:
        return None, [1]                        # handled by caller (needs U)
    return ov, []


def t_features(pixels, rows_d):
    """Sealed-Prep-style block features: PCA above the MP edge (<= 4 components), fitted on D rows, standardised."""
    w = pixels[rows_d]
    mu = w.mean(0)
    _, sv, vt = np.linalg.svd(w - mu, full_matrices=False)
    eig = sv ** 2 / len(rows_d)
    proj = vt[:pr.spike_count(eig, len(rows_d), w.shape[1])].T
    sd = ((w - mu) @ proj).std(0) + 1e-12
    return lambda rows: ((pixels[rows] - mu) @ proj) / sd


def run(layout, scenarios, reps, tag, adjacent_only=False):
    x, label = rb.load()
    m_anat = rb.masks()
    masks = m_anat if layout == "anat" else grid_masks()
    nb = len(masks)
    trunk, stats, _ = rb.get_mae(x)
    n = len(x)
    u = v2.u_lin(label, rb.trunk_features(trunk, stats, x, ~m_anat[4]))
    u = (u - u.mean()) / u.std()
    singles = [(j,) for j in range(nb)]
    pairs = list(itertools.combinations(range(nb), 2))
    if layout == "grid3" and adjacent_only:          # v3.1: only edge-adjacent cell pairs (12 of 36), cost amendment
        pairs = [(j, k) for j, k in pairs if abs(j // 3 - k // 3) + abs(j % 3 - k % 3) == 1]
    out = rb.OUT / f"v3_{layout}_{tag}.jsonl"
    for c in scenarios:
        for r in range(reps):
            t0 = time.time()
            rng = np.random.default_rng(20263000 + 1000 * (layout == "grid3") + 100 * c + r)
            a = (rng.random(n) < 0.1 + 0.8 * norm.cdf(1.5 * u)).astype(float)
            y = a + u + rng.standard_normal(n)
            ov, true_blocks = overlays(layout, c, a, rng, m_anat)
            if ov is None:                      # grid3 response blob (v2 D2 overlay)
                rr = u - 1.5 * a + 0.5 * rng.standard_normal(n)
                rt = np.clip((rr - rr.mean()) / rr.std(), -2.5, 2.5)
                yy, xx = np.mgrid[0:64, 0:64]
                blob = np.exp(-((yy - 20) ** 2 + (xx - 31.5) ** 2) / (2 * 2.0 ** 2)).astype(np.float32)
                ov = 0.25 * rt[:, None, None].astype(np.float32) * blob[None]
            img = np.clip(x + ov, 0, 1)
            seed = 11000 + 100 * c + r
            fold_rows = pr.folds(n, seed)
            roles = [pr.roles(rot) for rot in range(4)]
            pix = [img[:, masks[j]].astype(np.float64) for j in range(nb)]
            tfeat = [[t_features(pix[j], fold_rows[roles[rot]["D"]]) for rot in range(4)] for j in range(nb)]
            cache = {}

            def feats_hiding(hidden):
                key = tuple(sorted(hidden))
                if key not in cache:
                    vis = np.ones((64, 64), bool)
                    for j in key:
                        vis &= ~masks[j]
                    cache[key] = rb.trunk_features(trunk, stats, img, vis)
                return cache[key]

            def aipw(feats):
                sc = np.zeros(n)
                for rot in range(4):
                    rn, re = fold_rows[roles[rot]["N"]], fold_rows[roles[rot]["E"]]
                    mu, sd = feats[rn].mean(0), feats[rn].std(0) + 1e-12
                    fn = np.column_stack([np.ones(len(rn)), (feats[rn] - mu) / sd])
                    fe = np.column_stack([np.ones(len(re)), (feats[re] - mu) / sd])
                    sc[re] = pr.aipw(fn, a[rn], y[rn], fe, a[re], y[re])
                return sc

            def gap_of(subset):
                f = feats_hiding(subset)
                gs, ses = [], []
                for rot in range(4):
                    rs_, rn = fold_rows[roles[rot]["S"]], fold_rows[roles[rot]["N"]]
                    mu, sd = f[rn].mean(0), f[rn].std(0) + 1e-12
                    t = np.column_stack([tfeat[j][rot](rs_) for j in subset])
                    sc = pr.screen(np.zeros((len(rs_), 0)), (f[rs_] - mu) / sd, t, a[rs_], np.eye(f.shape[1]), DELTA2)
                    gs.append(sc["gap"]); ses.append(sc["se"])
                return float(np.mean(gs)), float(np.sqrt(np.mean(np.square(ses)) / 4))

            g1 = {s[0]: gap_of(s) for s in singles}
            med1 = float(np.median([g for g, _ in g1.values()]))
            h_single = [j for j, (g, s) in g1.items() if g > med1 + 2 * s]
            g2 = {p: gap_of(p) for p in pairs}
            med2 = float(np.median([g for g, _ in g2.values()]))
            flagged_pairs = [p for p, (g, s) in g2.items() if g > med2 + 2 * s and g > g1[p[0]][0] + g1[p[1]][0] + 2 * s]
            h_pair = sorted(set(h_single) | {j for p in flagged_pairs for j in p})
            score_all = aipw(feats_hiding([]))
            score_single = aipw(feats_hiding(h_single))
            score_pair = aipw(feats_hiding(h_pair))
            score_ref = aipw(feats_hiding(true_blocks))
            score_orc = aipw(u[:, None])
            pd = lambda s: {"theta": float(s.mean()), "vs_CNN_all": float((s - score_all).mean())}
            rec = {"layout": layout, "scenario": c, "rep": r, "seed": seed, "true_blocks": [j + 1 for j in true_blocks],
                   "naive": float(y[a == 1].mean() - y[a == 0].mean()), "oracle": float(score_orc.mean()),
                   "CNN_all": float(score_all.mean()),
                   "single_gaps": {f"B{j + 1}": g1[j] for j in range(nb)},
                   "top_pair_gaps": {"+".join(f"B{j + 1}" for j in p): g2[p] for p in sorted(g2, key=lambda p: -g2[p][0])[:4]},
                   "H_single": [j + 1 for j in h_single], "H_pair": [j + 1 for j in h_pair],
                   "flagged_pairs": [[j + 1 for j in p] for p in flagged_pairs],
                   "PROBE_single": pd(score_single), "PROBE_pair": pd(score_pair), "ref_true_hidden": pd(score_ref),
                   "seconds": time.time() - t0}
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            rb.log(f"{layout} S{c} r{r}: naive {rec['naive']:.3f} oracle {rec['oracle']:.3f} CNN_all {rec['CNN_all']:.3f} | "
                   f"single H{rec['H_single']} {rec['PROBE_single']['theta']:.3f} | pair H{rec['H_pair']} {rec['PROBE_pair']['theta']:.3f} "
                   f"| ref(true {rec['true_blocks']}) {rec['ref_true_hidden']['theta']:.3f} | top single gaps "
                   f"{sorted(((round(g, 4), f'B{j + 1}') for j, (g, _) in g1.items()), reverse=True)[:3]} | {rec['seconds']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["anat", "grid3"], required=True)
    ap.add_argument("--scenarios", type=lambda s: [int(v) for v in s.split(",")], required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--tag", default="r10")
    ap.add_argument("--adjacent-pairs", action="store_true")
    a = ap.parse_args()
    nets.setup_torch(2)
    run(a.layout, a.scenarios, a.reps, a.tag, a.adjacent_pairs)
