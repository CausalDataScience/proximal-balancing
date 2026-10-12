"""v5b: lower the representation floor with supervision the analyst is allowed to use (A and Y, never U).

Per block k and per cross-fit half: a small CNN (ConvTrunk + 2-output head: logit of A, prediction of Y) is trained
on the OTHER half's images with only block k visible; its pooled trunk features (128) are then computed for THIS
half.  So every row's features come from a model that never saw that row.  Features -> PCA 16 per block (unsupervised,
all rows) -> the v4 block-additive pipeline (audit from all blocks, relative singleton rule, sealed AIPW).
U = cross-fitted classifier logit (hidden; oracle only).  Scenarios V0 clean, V2 footprint in W2, V3 response in W5.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.stats import norm

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import realimg_benchmark as rb  # noqa: E402
import realimg_v3 as v3  # noqa: E402
import realimg_v4 as v4  # noqa: E402
from realimg_benchmark import pr, nets  # noqa: E402

EPOCHS, BATCH, LR = 6, 128, 1e-3


class SupNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nets.ConvTrunk(64)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2))

    def pooled(self, x):
        f = self.trunk(x)
        vis = nn.functional.avg_pool2d(x[:, 1:2], 8)
        mean = (f * vis).sum((2, 3)) / (vis.sum((2, 3)) + 1e-6)
        return torch.cat([mean, f.amax((2, 3))], 1)

    def forward(self, x):
        return self.head(self.pooled(x))


def block_input(img, stats, mask):
    z = nets.standardise(img, stats).astype(np.float32)
    vis = mask.astype(np.float32)
    return np.stack([z * vis, np.broadcast_to(vis, z.shape)], 1).astype(np.float32)


def supervised_block_features(img, stats, mask, a, y, seed):
    """Cross-fitted (2 halves) pooled features of a CNN trained to predict (A, Y) from this block only."""
    n = len(img)
    rng = np.random.default_rng(seed)
    half = rng.permutation(n) % 2
    xin = block_input(img, stats, mask)
    ys = (y - y.mean()) / y.std()
    feats = np.zeros((n, 128))
    for h in (0, 1):
        tr, te = np.where(half != h)[0], np.where(half == h)[0]
        nets.seed_all(seed + h)
        net = SupNet()
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
        for ep in range(EPOCHS):
            net.train()
            order = rng.permutation(tr)
            for s in range(0, len(order), BATCH):
                idx = order[s:s + BATCH]
                out = net(torch.from_numpy(xin[idx]))
                loss = nn.functional.binary_cross_entropy_with_logits(out[:, 0], torch.from_numpy(a[idx].astype(np.float32))) \
                    + nn.functional.mse_loss(out[:, 1], torch.from_numpy(ys[idx].astype(np.float32)))
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for s in range(0, len(te), 512):
                feats[te[s:s + 512]] = net.pooled(torch.from_numpy(xin[te[s:s + 512]])).numpy()
    return feats


def run(scenarios, reps, tag):
    x, label = rb.load(); m = rb.masks(); _, stats, _ = rb.get_mae(x); n = len(x)
    u = np.load(rb.OUT / "u_logit_crossfit.npy"); u = (u - u.mean()) / u.std()
    out = rb.OUT / f"v5sup_{tag}.jsonl"
    for c in scenarios:
        for r in range(reps):
            t0 = time.time()
            rng = np.random.default_rng(20265000 + 100 * c + r)
            a = (rng.random(n) < 0.1 + 0.8 * norm.cdf(1.5 * u)).astype(float)
            y = a + u + rng.standard_normal(n)
            img = x.copy(); true_blocks = []
            if c == 1:                                  # V1: treatment footprint in W5 (as in v4)
                tube = rng.random(n) < np.where(a == 1, 0.9, 0.1)
                line = np.zeros((64, 64), np.float32); line[4:45, 31:33] = 0.35
                img[tube] = np.clip(img[tube] + line[None] * m[4][None], 0, 1); true_blocks = [4]
            if c == 2:
                tube = rng.random(n) < np.where(a == 1, 0.9, 0.1)
                line = np.zeros((64, 64), np.float32); line[36:53, 14:16] = 0.35
                img[tube] = np.clip(img[tube] + line[None] * m[1][None], 0, 1); true_blocks = [1]
            if c == 3:
                rr = u - 1.5 * a + 0.5 * rng.standard_normal(n); rt = np.clip((rr - rr.mean()) / rr.std(), -2.5, 2.5)
                yy, xx = np.mgrid[0:64, 0:64]
                blob = (np.exp(-((yy - 20) ** 2 + (xx - 31.5) ** 2) / (2 * 2.0 ** 2)) * m[4]).astype(np.float32)
                img = np.clip(img + 0.25 * rt[:, None, None].astype(np.float32) * blob[None], 0, 1); true_blocks = [4]
            seed = 15000 + 100 * c + r
            feats = [v4.pca16(supervised_block_features(img, stats, m[k], a, y, seed + 10 * k)) for k in range(5)]
            t_feat = time.time() - t0
            fold_rows = pr.folds(n, seed); roles = [pr.roles(rot) for rot in range(4)]
            pix = [img[:, m[j]].astype(np.float64) for j in range(5)]
            tfeat = [[v3.t_features(pix[j], fold_rows[roles[rot]["D"]]) for rot in range(4)] for j in range(5)]
            Z = lambda hidden: np.column_stack([feats[j] for j in range(5) if j not in hidden])

            def aipw(F):
                sc = np.zeros(n)
                for rot in range(4):
                    rn, re = fold_rows[roles[rot]["N"]], fold_rows[roles[rot]["E"]]
                    fn = np.column_stack([np.ones(len(rn)), F[rn]]); fe = np.column_stack([np.ones(len(re)), F[re]])
                    sc[re] = pr.aipw(fn, a[rn], y[rn], fe, a[re], y[re])
                return sc

            def audit(hidden):
                out = {}
                for j in range(5):
                    if j in hidden:
                        continue
                    F = Z(list(hidden) + [j]); gs, ses = [], []
                    for rot in range(4):
                        rs_ = fold_rows[roles[rot]["S"]]
                        sc = pr.screen(np.zeros((len(rs_), 0)), F[rs_], tfeat[j][rot](rs_), a[rs_], np.eye(F.shape[1]), v4.DELTA2)
                        gs.append(sc["gap"]); ses.append(sc["se"])
                    out[j] = (float(np.mean(gs)), float(np.sqrt(np.mean(np.square(ses)) / 4)))
                return out
            gaps = audit([])
            med = float(np.median([g for g, _ in gaps.values()]))
            H = [j for j, (g, s) in gaps.items() if g > med + 2 * s]            # one-pass relative rule (as before)
            # sequential elimination (v5.1, pre-registered): hide the largest flagged block, re-audit the rest, repeat
            H_seq, rounds = [], []
            cur = gaps
            while len(H_seq) < 2:
                m_ = float(np.median([g for g, _ in cur.values()]))
                cand = [(g, j) for j, (g, s) in cur.items() if g > m_ + 2 * s]
                rounds.append({f"W{j + 1}": cur[j] for j in cur})
                if not cand:
                    break
                H_seq.append(max(cand)[1])
                cur = audit(H_seq)
            s_all, s_probe, s_ref, s_orc, s_lung = aipw(Z([])), aipw(Z(H)), aipw(Z(true_blocks)), aipw(u[:, None]), aipw(Z([4]))
            s_seq = aipw(Z(H_seq))
            # floor diagnostic: CV R2 of U from the supervised all-block features (cross-fitted ridge)
            F = Z([]); fold = np.random.default_rng(1).permutation(n) % 4; pred = np.zeros(n)
            X = np.column_stack([np.ones(n), F])
            for j in range(4):
                tr, te = fold != j, fold == j
                b = np.linalg.solve(X[tr].T @ X[tr] + 1e-2 * np.eye(X.shape[1]), X[tr].T @ u[tr]); pred[te] = X[te] @ b
            r2 = float(1 - np.mean((u - pred) ** 2) / np.var(u))
            rec = {"scenario": c, "rep": r, "seed": seed, "true_blocks": [j + 1 for j in true_blocks], "R2_U_from_all": r2,
                   "naive": float(y[a == 1].mean() - y[a == 0].mean()), "oracle": float(s_orc.mean()),
                   "all_blocks": float(s_all.mean()), "lung_blocks_only": float(s_lung.mean()), "PROBE": float(s_probe.mean()),
                   "H": [j + 1 for j in H], "ref_true_hidden": float(s_ref.mean()), "gaps": {f"W{j + 1}": gaps[j] for j in range(5)},
                   "H_seq": [j + 1 for j in H_seq], "PROBE_seq": float(s_seq.mean()), "seq_rounds": rounds,
                   "seconds_features": t_feat, "seconds": time.time() - t0}
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            rb.log(f"v5sup V{c} r{r}: R2(U|all) {r2:.3f} naive {rec['naive']:.3f} oracle {rec['oracle']:.3f} all {rec['all_blocks']:.3f} "
                   f"lungs {rec['lung_blocks_only']:.3f} | PROBE {rec['PROBE']:.3f} H{rec['H']} | seq {rec['PROBE_seq']:.3f} H{rec['H_seq']} | ref {rec['ref_true_hidden']:.3f} "
                   f"| gaps {[round(gaps[j][0], 4) for j in range(5)]} | {rec['seconds']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=lambda s: [int(v) for v in s.split(",")], default=[0, 3, 2])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="r5")
    a = ap.parse_args()
    nets.setup_torch(2)
    run(a.scenarios, a.reps, a.tag)
