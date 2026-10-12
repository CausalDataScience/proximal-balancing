"""v6: automatic block layouts with the block-additive MAE representation (user approval 2026-10-11).

Layouts
  anat  : the five hand-drawn blocks (reference)
  grid3 : 3x3 cells (21/21/22 px), no anatomy knowledge
  grid4 : 4x4 cells (16 px), no anatomy knowledge
  auto  : blocks derived from the images themselves: per-image dark-region (lung field) detection by Otsu
          thresholding inside the thorax, a consensus mask over all images, each lung split at its median row,
          and the central strip between the lungs; blocks eroded so that they are at least two pixels apart
Representation: block k = MAE trunk pooled features of the image with only block k visible, PCA to Q components
  (Q = 16 for <= 9 blocks, 8 for 16 blocks), concatenated.  U = hidden classifier logit.  Scenarios V0 clean,
  V2 footprint inside the lung region (same pixels as before), V3 treatment response in the central strip (same pixels).
Audit: one-pass relative rule (pre-registered) and, recorded for sensitivity, sequential elimination with and without
  the absolute floor 0.005.  Estimate: sealed AIPW on the remaining blocks.  Reference: hide the cells the overlay touches.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import norm

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import realimg_benchmark as rb  # noqa: E402
import realimg_v3 as v3  # noqa: E402
from realimg_benchmark import pr, nets  # noqa: E402

DELTA2 = 0.005


def pca_q(f, q):
    mu = f.mean(0); _, _, vt = np.linalg.svd(f - mu, full_matrices=False); z = (f - mu) @ vt[:q].T
    return (z - z.mean(0)) / (z.std(0) + 1e-9)


def grid_masks(g):
    edges = np.linspace(0, 64, g + 1).round().astype(int)
    m = np.zeros((g * g, 64, 64), bool); k = 0
    for r in range(g):
        for c in range(g):
            m[k, edges[r]:edges[r + 1], edges[c]:edges[c + 1]] = True; k += 1
    return m


def otsu(v):
    hist, edges = np.histogram(v, bins=64); mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0; m0 = np.cumsum(hist * mids) / np.maximum(w0, 1); m1 = (np.sum(hist * mids) - np.cumsum(hist * mids)) / np.maximum(w1, 1)
    between = w0[:-1] * w1[:-1] * (m0[:-1] - m1[:-1]) ** 2
    return mids[int(np.argmax(between))]


def auto_masks(x, log):
    """Consensus lung fields from the images alone; returns (5, 64, 64) masks and a report."""
    n = len(x)
    thorax = np.zeros((64, 64), bool); thorax[6:58, 6:58] = True
    votes = np.zeros((64, 64)); kept = 0
    for i in range(n):
        img = ndimage.gaussian_filter(x[i], 1.0)
        t = otsu(img[thorax]); dark = (img < t) & thorax
        lab, nl = ndimage.label(dark)
        if nl < 2: continue
        sizes = ndimage.sum(dark, lab, range(1, nl + 1)); cents = ndimage.center_of_mass(dark, lab, range(1, nl + 1))
        left = [(s, j + 1) for j, (s, c) in enumerate(zip(sizes, cents)) if c[1] < 32 and s > 40]
        right = [(s, j + 1) for j, (s, c) in enumerate(zip(sizes, cents)) if c[1] >= 32 and s > 40]
        if not left or not right: continue
        votes += (lab == max(left)[1]) | (lab == max(right)[1]); kept += 1
    cons = votes / max(kept, 1) >= 0.5
    lab, nl = ndimage.label(cons)
    sizes = ndimage.sum(cons, lab, range(1, nl + 1)); order = np.argsort(sizes)[::-1][:2]
    lungs = [lab == (j + 1) for j in order]
    lungs.sort(key=lambda m: ndimage.center_of_mass(m)[1])          # image-left lung first
    m = np.zeros((5, 64, 64), bool)
    for q, lung in enumerate(lungs):
        rows = np.where(lung.any(1))[0]; mid = int(np.median(rows))
        upper = lung.copy(); upper[mid:] = False; lower = lung.copy(); lower[:mid] = False
        m[2 * q] = ndimage.binary_erosion(upper, iterations=1); m[2 * q + 1] = ndimage.binary_erosion(lower, iterations=1)
    cols_l = np.where(lungs[0].any(0))[0].max(); cols_r = np.where(lungs[1].any(0))[0].min()
    strip = np.zeros((64, 64), bool); strip[:, cols_l + 2:cols_r - 1] = True
    m[4] = strip & ~ndimage.binary_dilation(lungs[0] | lungs[1], iterations=1)
    rep = {"images_used": kept, "block_sizes": [int(b.sum()) for b in m], "strip_cols": [int(cols_l + 2), int(cols_r - 2)]}
    log(f"auto layout: {rep}")
    return m, rep


def run(layout, scenarios, reps, tag):
    x, label = rb.load(); m_anat = rb.masks(); trunk, stats, _ = rb.get_mae(x); n = len(x)
    report = {}
    masks = {"anat": m_anat, "grid3": grid_masks(3), "grid4": grid_masks(4)}.get(layout)
    if layout == "auto":
        masks, report = auto_masks(x, rb.log)
    nb = len(masks); Q = 16 if nb <= 9 else 8
    u = np.load(rb.OUT / "u_logit_crossfit.npy"); u = (u - u.mean()) / u.std()
    clean = [pca_q(rb.trunk_features(trunk, stats, x, masks[k]), Q) for k in range(nb)]
    F = np.column_stack(clean); fold = np.random.default_rng(1).permutation(n) % 4; pred = np.zeros(n); X = np.column_stack([np.ones(n), F])
    for j in range(4):
        tr, te = fold != j, fold == j; b = np.linalg.solve(X[tr].T @ X[tr] + 1e-2 * np.eye(X.shape[1]), X[tr].T @ u[tr]); pred[te] = X[te] @ b
    r2 = float(1 - np.mean((u - pred) ** 2) / np.var(u)); rb.log(f"{layout}: {nb} blocks, Q={Q}, CV R2(U|all) = {r2:.3f}")
    out = rb.OUT / f"v6_{layout}_{tag}.jsonl"
    with (rb.OUT / f"v6_{layout}_layout.json").open("w") as fh:
        json.dump({"layout": layout, "n_blocks": nb, "Q": Q, "R2_U_all": r2, "block_sizes": [int(b.sum()) for b in masks], "report": report,
                   "masks_rle": [np.packbits(b.ravel()).tolist() for b in masks]}, fh)
    for c in scenarios:
        for r in range(reps):
            t0 = time.time(); rng = np.random.default_rng(20266000 + 100 * c + r)
            a = (rng.random(n) < 0.1 + 0.8 * norm.cdf(1.5 * u)).astype(float); y = a + u + rng.standard_normal(n)
            img = x.copy(); overlay = np.zeros((64, 64), bool)
            if c == 2:
                tube = rng.random(n) < np.where(a == 1, 0.9, 0.1); line = np.zeros((64, 64), np.float32); line[36:53, 14:16] = 0.35
                img[tube] = np.clip(img[tube] + line[None] * m_anat[1][None], 0, 1); overlay = (line > 0) & m_anat[1]
            if c == 3:
                rr = u - 1.5 * a + 0.5 * rng.standard_normal(n); rt = np.clip((rr - rr.mean()) / rr.std(), -2.5, 2.5)
                yy, xx = np.mgrid[0:64, 0:64]; blob = (np.exp(-((yy - 20) ** 2 + (xx - 31.5) ** 2) / (2 * 2.0 ** 2)) * m_anat[4]).astype(np.float32)
                img = np.clip(img + 0.25 * rt[:, None, None].astype(np.float32) * blob[None], 0, 1); overlay = blob > 0.05
            true_blocks = [k for k in range(nb) if (masks[k] & overlay).sum() > 0]
            feats = list(clean)
            for k in true_blocks:
                feats[k] = pca_q(rb.trunk_features(trunk, stats, img, masks[k]), Q)
            seed = 17000 + 100 * c + r; fold_rows = pr.folds(n, seed); roles = [pr.roles(rot) for rot in range(4)]
            pix = [img[:, masks[k]].astype(np.float64) for k in range(nb)]
            tfeat = [[v3.t_features(pix[k], fold_rows[roles[rot]["D"]]) for rot in range(4)] for k in range(nb)]
            Z = lambda hidden: np.column_stack([feats[k] for k in range(nb) if k not in hidden])

            def aipw(Fm):
                sc = np.zeros(n)
                for rot in range(4):
                    rn, re = fold_rows[roles[rot]["N"]], fold_rows[roles[rot]["E"]]
                    sc[re] = pr.aipw(np.column_stack([np.ones(len(rn)), Fm[rn]]), a[rn], y[rn], np.column_stack([np.ones(len(re)), Fm[re]]), a[re], y[re])
                return sc

            def audit(hidden):
                out_ = {}
                for k in range(nb):
                    if k in hidden: continue
                    Fk = Z(list(hidden) + [k]); gs, ses = [], []
                    for rot in range(4):
                        rs_ = fold_rows[roles[rot]["S"]]
                        sc = pr.screen(np.zeros((len(rs_), 0)), Fk[rs_], tfeat[k][rot](rs_), a[rs_], np.eye(Fk.shape[1]), DELTA2); gs.append(sc["gap"]); ses.append(sc["se"])
                    out_[k] = (float(np.mean(gs)), float(np.sqrt(np.mean(np.square(ses)) / 4)))
                return out_

            gaps = audit([]); med = float(np.median([g for g, _ in gaps.values()]))
            H1 = [k for k, (g, s) in gaps.items() if g > med + 2 * s]
            Hf = [k for k in H1 if gaps[k][0] > DELTA2]
            Hs, cur = [], gaps
            while len(Hs) < 3:
                m_ = float(np.median([g for g, _ in cur.values()])); cand = [(g, k) for k, (g, s) in cur.items() if g > m_ + 2 * s and g > DELTA2]
                if not cand: break
                Hs.append(max(cand)[1]); cur = audit(Hs)
            s_all, s_1, s_f, s_s, s_ref, s_orc = aipw(Z([])), aipw(Z(H1)), aipw(Z(Hf)), aipw(Z(Hs)), aipw(Z(true_blocks)), aipw(u[:, None])
            rec = {"layout": layout, "scenario": c, "rep": r, "seed": seed, "n_blocks": nb, "true_blocks": [k + 1 for k in true_blocks],
                   "naive": float(y[a == 1].mean() - y[a == 0].mean()), "oracle": float(s_orc.mean()), "all_blocks": float(s_all.mean()),
                   "PROBE": float(s_1.mean()), "H": [k + 1 for k in H1], "PROBE_floor": float(s_f.mean()), "H_floor": [k + 1 for k in Hf],
                   "PROBE_seq_floor": float(s_s.mean()), "H_seq_floor": [k + 1 for k in Hs], "ref_true_hidden": float(s_ref.mean()),
                   "gaps": {f"B{k + 1}": gaps[k] for k in range(nb)}, "seconds": time.time() - t0}
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            rb.log(f"{layout} V{c} r{r}: oracle {rec['oracle']:.3f} all {rec['all_blocks']:.3f} | PROBE {rec['PROBE']:.3f} H{rec['H']} | floor {rec['PROBE_floor']:.3f} H{rec['H_floor']} "
                   f"| seq+floor {rec['PROBE_seq_floor']:.3f} H{rec['H_seq_floor']} | ref(true {rec['true_blocks']}) {rec['ref_true_hidden']:.3f} | {rec['seconds']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["anat", "grid3", "grid4", "auto"], required=True)
    ap.add_argument("--scenarios", type=lambda s: [int(v) for v in s.split(",")], default=[0, 3, 2])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--tag", default="r10")
    a = ap.parse_args()
    nets.setup_torch(2)
    run(a.layout, a.scenarios, a.reps, a.tag)
