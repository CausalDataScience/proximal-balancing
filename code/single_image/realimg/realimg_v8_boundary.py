"""v8: where PROBE succeeds and where it fails, at 64 px with the v5 supervised block encoders (memo section 33).

Three pre-registered scenarios added to V0-V3 of realimg_v5_sup.py (same images, masks, encoders, audit and sealed AIPW):
  V4 mediator in W5.      M = A + 0.5 nu (nu ~ N(0,1), independent of U); the mark in W5 is -0.25 * std(M) (treatment darkens
                          it); Y = 0.5 A + 0.5 M + U + eps, so the total effect is still tau = 1.  Adjusting for the mark
                          estimates the direct effect 0.5.  The correct analysis must hide W5.
  V5 collider in W2+W5.   One latent response r = U - 1.5 A + 0.5 nu drawn as two blobs, one in W5 (as V3) and one in W2.
                          Both blocks are colliders; hiding only one leaves bias.  Correct: hide {W2, W5}.
  V6 collider everywhere. The same r drawn as five blobs, one per block.  No retained subset is clean; the relative audit sees
                          five similar gaps and cannot single any block out.  PROBE is predicted to equal whole-image adjustment.
Recorded per replicate: naive, oracle, all blocks, PROBE (one-pass relative rule), PROBE_seq_floor (sequential, floor 0.005,
at most 2), PROBE_top1 (hide the single largest-gap block), ref_true_hidden (hide the true harmful set; NaN for V6),
gaps, R2(U | all features).  Resumable: results/single_image/pneumoniamnist_v1/v8ext_<tag>.jsonl, one line per (scenario, rep).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import norm

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import realimg_benchmark as rb  # noqa: E402
import realimg_v3 as v3  # noqa: E402
import realimg_v4 as v4  # noqa: E402
import realimg_v5_sup as v5  # noqa: E402
from realimg_benchmark import pr, nets  # noqa: E402

CENTERS = {0: (19.0, 15.5), 1: (44.0, 15.5), 2: (19.0, 47.5), 3: (44.0, 47.5), 4: (20.0, 31.5)}   # (row, col) per block, 64 px


def blob(block, m):
    yy, xx = np.mgrid[0:64, 0:64]; cy, cx = CENTERS[block]
    return (np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2)) * m[block]).astype(np.float32)


def std_clip(v):
    return np.clip((v - v.mean()) / v.std(), -2.5, 2.5).astype(np.float32)


def scenario(c, x, a, u, m, rng):
    """Return (img, y, true_blocks) for scenario c in {4, 5, 6}."""
    n = len(x); eps = rng.standard_normal(n); nu = rng.standard_normal(n)
    if c == 4:
        M = a + 0.5 * nu; y = 0.5 * a + 0.5 * M + u + eps
        img = np.clip(x - 0.25 * std_clip(M)[:, None, None] * blob(4, m)[None], 0, 1); return img, y, [4]
    y = a + u + eps
    rt = std_clip(u - 1.5 * a + 0.5 * nu)
    blocks = [1, 4] if c == 5 else [0, 1, 2, 3, 4]
    img = x.copy()
    for b in blocks:
        img = img + 0.25 * rt[:, None, None] * blob(b, m)[None]
    return np.clip(img, 0, 1), y, blocks


def done_pairs(out):
    if not out.exists():
        return set()
    return {(json.loads(l)["scenario"], json.loads(l)["rep"]) for l in out.read_text().splitlines() if l.strip()}


def run(scenarios, rep_start, reps, tag, smoke=False):
    out = rb.OUT / f"v8ext_{tag}.jsonl"; done = done_pairs(out)
    todo = [(c, r) for c in scenarios for r in range(rep_start, reps) if (c, r) not in done]
    if not todo:
        rb.log(f"v8ext[{tag}]: nothing left"); return
    x, label = rb.load(); m = rb.masks(); _, stats, _ = rb.get_mae(x)
    u = np.load(rb.OUT / "u_logit_crossfit.npy"); u = (u - u.mean()) / u.std()
    if smoke:
        keep = np.random.default_rng(0).permutation(len(x))[:800]; x, u = x[keep], u[keep]; v5.EPOCHS = 1
    n = len(x)
    for c, r in todo:
        t0 = time.time(); rng = np.random.default_rng(20268000 + 100 * c + r)
        a = (rng.random(n) < 0.1 + 0.8 * norm.cdf(1.5 * u)).astype(float)
        img, y, true_blocks = scenario(c, x, a, u, m, rng)
        seed = 16000 + 100 * c + r
        feats = [v4.pca16(v5.supervised_block_features(img, stats, m[k], a, y, seed + 10 * k)) for k in range(5)]
        t_feat = time.time() - t0
        fold_rows = pr.folds(n, seed); roles = [pr.roles(rot) for rot in range(4)]
        pix = [img[:, m[j]].astype(np.float64) for j in range(5)]
        tfeat = [[v3.t_features(pix[j], fold_rows[roles[rot]["D"]]) for rot in range(4)] for j in range(5)]
        Z = lambda hidden: np.column_stack([feats[j] for j in range(5) if j not in hidden])

        def aipw(F):
            sc = np.zeros(n)
            for rot in range(4):
                rn, re = fold_rows[roles[rot]["N"]], fold_rows[roles[rot]["E"]]
                sc[re] = pr.aipw(np.column_stack([np.ones(len(rn)), F[rn]]), a[rn], y[rn], np.column_stack([np.ones(len(re)), F[re]]), a[re], y[re])
            return sc

        def audit(hidden):
            o = {}
            for j in range(5):
                if j in hidden: continue
                F = Z(list(hidden) + [j]); gs, ses = [], []
                for rot in range(4):
                    rs_ = fold_rows[roles[rot]["S"]]
                    sc = pr.screen(np.zeros((len(rs_), 0)), F[rs_], tfeat[j][rot](rs_), a[rs_], np.eye(F.shape[1]), v4.DELTA2); gs.append(sc["gap"]); ses.append(sc["se"])
                o[j] = (float(np.mean(gs)), float(np.sqrt(np.mean(np.square(ses)) / 4)))
            return o

        gaps = audit([]); med = float(np.median([g for g, _ in gaps.values()]))
        H = [j for j, (g, s) in gaps.items() if g > med + 2 * s]
        top1 = [max(gaps, key=lambda j: gaps[j][0])]
        Hs, cur = [], gaps
        while len(Hs) < 2:
            m_ = float(np.median([g for g, _ in cur.values()])); cand = [(g, j) for j, (g, s) in cur.items() if g > m_ + 2 * s and g > v4.DELTA2]
            if not cand: break
            Hs.append(max(cand)[1]); cur = audit(Hs)
        s_all, s_probe, s_seq, s_top1, s_orc = aipw(Z([])), aipw(Z(H)), aipw(Z(Hs)), aipw(Z(top1)), aipw(u[:, None])
        s_ref = aipw(Z(true_blocks)) if len(true_blocks) < 5 else None
        F = Z([]); fold = np.random.default_rng(1).permutation(n) % 4; pred = np.zeros(n); X = np.column_stack([np.ones(n), F])
        for j in range(4):
            tr, te = fold != j, fold == j; b = np.linalg.solve(X[tr].T @ X[tr] + 1e-2 * np.eye(X.shape[1]), X[tr].T @ u[tr]); pred[te] = X[te] @ b
        r2 = float(1 - np.mean((u - pred) ** 2) / np.var(u))
        rec = {"scenario": c, "rep": r, "seed": seed, "n": n, "smoke": smoke, "true_blocks": [j + 1 for j in true_blocks], "R2_U_from_all": r2,
               "naive": float(y[a == 1].mean() - y[a == 0].mean()), "oracle": float(s_orc.mean()), "all_blocks": float(s_all.mean()),
               "PROBE": float(s_probe.mean()), "H": [j + 1 for j in H], "PROBE_seq_floor": float(s_seq.mean()), "H_seq_floor": [j + 1 for j in Hs],
               "PROBE_top1": float(s_top1.mean()), "H_top1": [j + 1 for j in top1],
               "ref_true_hidden": (float(s_ref.mean()) if s_ref is not None else None),
               "gaps": {f"W{j + 1}": gaps[j] for j in range(5)}, "seconds_features": t_feat, "seconds": time.time() - t0,
               "finished_at": datetime.now().isoformat(timespec="seconds")}
        with out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        ref = f"{rec['ref_true_hidden']:.3f}" if rec["ref_true_hidden"] is not None else "n/a"
        rb.log(f"v8ext[{tag}] V{c} r{r}: R2 {r2:.3f} oracle {rec['oracle']:.3f} all {rec['all_blocks']:.3f} | PROBE {rec['PROBE']:.3f} H{rec['H']} "
               f"| seq+floor {rec['PROBE_seq_floor']:.3f} H{rec['H_seq_floor']} | top1 {rec['PROBE_top1']:.3f} H{rec['H_top1']} | ref {ref} "
               f"| gaps {[round(gaps[j][0], 4) for j in range(5)]} | {rec['seconds']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=lambda s: [int(v) for v in s.split(",")], default=[4, 5, 6])
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--tag", default="r10")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--group", default="")
    a = ap.parse_args()
    nets.setup_torch(2)
    if a.smoke:
        run([4, 5, 6], 0, 1, "smoke", smoke=True)
    else:
        run(a.scenarios, a.rep_start, a.reps, a.tag)
