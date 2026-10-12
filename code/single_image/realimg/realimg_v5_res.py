"""v5a: lower the representation floor with resolution: PneumoniaMNIST 128 px, self-supervised MAE at 128
(16 x 16 patches of 8 px, 75% hidden), then the v4 block-additive pipeline with masks and overlays scaled x2.
U stays the hidden 64-px classifier logit.  Scenarios V0 clean, V2 footprint in W2, V3 response in W5.
Also prints the floor diagnostic: CV R2 of U from the 128-px all-block features.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import realimg_benchmark as rb  # noqa: E402
import realimg_v3 as v3  # noqa: E402
import realimg_v4 as v4  # noqa: E402
from realimg_benchmark import pr, nets  # noqa: E402
from single_image import mae as mae_mod  # noqa: E402

G = 128
DATA128 = SP.parents[2] / "data/medmnist/pneumoniamnist_128.npz"
MAE_EPOCHS, MAE_SEED = 20, 20261011


def load128():
    z = np.load(DATA128, allow_pickle=False)
    x = np.concatenate([z["train_images"], z["val_images"], z["test_images"]]).astype(np.float32) / 255.0
    lab = np.concatenate([z["train_labels"], z["val_labels"], z["test_labels"]]).ravel()
    return x, lab


def patch_vis(n, rng, grid=G, patch=8, ratio=0.75):
    side = grid // patch; npatch = side * side; hidden = int(round(ratio * npatch))
    keys = rng.random((n, npatch)); hid = np.argsort(keys, axis=1)[:, :hidden]
    vis = np.ones((n, npatch), np.float32); np.put_along_axis(vis, hid, 0.0, axis=1)
    return np.kron(vis.reshape(n, side, side), np.ones((1, patch, patch), np.float32))


def train_mae128(x, seed, epochs, log):
    """train_mae's loop with a 128-px patch grid (the model is fully convolutional)."""
    nets.seed_all(seed); rng = np.random.default_rng([seed, 6]); n = len(x)
    perm = rng.permutation(n); nv = int(round(0.2 * n)); val, tr = np.sort(perm[:nv]), np.sort(perm[nv:])
    stats = (float(x[tr].mean()), float(x[tr].std())); z = nets.standardise(x, stats).astype(np.float32)
    model = mae_mod.MaskedAutoencoder(64); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    steps = int(np.ceil(len(tr) / 64)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * steps, pct_start=0.1)
    vis_val = patch_vis(len(val), np.random.default_rng([seed, 7]))
    for ep in range(epochs):
        order = rng.permutation(len(tr)); vis_ep = patch_vis(len(tr), rng); run = 0.0
        for b in range(steps):
            idx = order[b * 64:(b + 1) * 64]
            xb = torch.from_numpy(np.stack([z[tr][idx] * vis_ep[idx], vis_ep[idx]], 1)); yb = torch.from_numpy(z[tr][idx][:, None])
            pred = model(xb); hid = 1.0 - xb[:, 1:2]
            loss = (((pred - yb) ** 2) * hid).sum() / hid.sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); run += float(loss.detach())
        if (ep + 1) % 5 == 0 or ep == 0:
            model.eval(); tot = cnt = 0.0
            with torch.no_grad():
                for c0 in range(0, len(val), 256):
                    xb = torch.from_numpy(np.stack([z[val][c0:c0 + 256] * vis_val[c0:c0 + 256], vis_val[c0:c0 + 256]], 1))
                    pred = model(xb); hid = 1.0 - xb[:, 1:2]
                    tot += float((((pred - torch.from_numpy(z[val][c0:c0 + 256][:, None])) ** 2) * hid).sum()); cnt += float(hid.sum())
            model.train(); log(f"    MAE128 epoch {ep + 1}/{epochs}: train {run / steps:.4f} val {tot / cnt:.4f}")
    return model.trunk.eval(), stats


def trunk_features(trunk, stats, imgs, vis_mask):
    vis = vis_mask.astype(np.float32); out = []
    with torch.no_grad():
        for s in range(0, len(imgs), 256):
            z = nets.standardise(imgs[s:s + 256], stats).astype(np.float32)
            x = torch.from_numpy(np.stack([z * vis, np.broadcast_to(vis, z.shape)], 1).astype(np.float32))
            f = trunk(x); v = torch.nn.functional.avg_pool2d(x[:, 1:2], 8)
            mean = (f * v).sum((2, 3)) / (v.sum((2, 3)) + 1e-6)
            out.append(torch.cat([mean, f.amax((2, 3))], 1).numpy())
    return np.concatenate(out).astype(np.float64)


def run(scenarios, reps, tag):
    x64, lab64 = rb.load(); x, lab = load128(); assert np.array_equal(lab64, lab), "128-px labels differ from 64-px"
    m = np.kron(rb.masks(), np.ones((1, 2, 2), bool)); n = len(x)
    ck = rb.OUT / f"mae128_e{MAE_EPOCHS}_s{MAE_SEED}.pt"
    if ck.exists():
        d = torch.load(ck, map_location="cpu", weights_only=False); trunk = nets.ConvTrunk(64); trunk.load_state_dict(d["trunk"]); trunk.eval(); stats = tuple(d["stats"])
    else:
        t0 = time.time(); trunk, stats = train_mae128(x, MAE_SEED, MAE_EPOCHS, rb.log)
        torch.save({"trunk": trunk.state_dict(), "stats": stats}, ck); rb.log(f"MAE128 trained in {time.time() - t0:.0f}s")
    u = np.load(rb.OUT / "u_logit_crossfit.npy"); u = (u - u.mean()) / u.std()
    clean = [v4.pca16(trunk_features(trunk, stats, x, m[k])) for k in range(5)]
    F = np.column_stack(clean); fold = np.random.default_rng(1).permutation(n) % 4; pred = np.zeros(n); X = np.column_stack([np.ones(n), F])
    for j in range(4):
        tr, te = fold != j, fold == j; b = np.linalg.solve(X[tr].T @ X[tr] + 1e-2 * np.eye(X.shape[1]), X[tr].T @ u[tr]); pred[te] = X[te] @ b
    rb.log(f"floor diagnostic: CV R2 of U(logit) from 128-px all-block MAE features = {1 - np.mean((u - pred) ** 2) / np.var(u):.3f}")
    out = rb.OUT / f"v5res_{tag}.jsonl"
    for c in scenarios:
        for r in range(reps):
            t0 = time.time(); rng = np.random.default_rng(20265000 + 100 * c + r)      # same draws as v5sup
            a = (rng.random(n) < 0.1 + 0.8 * norm.cdf(1.5 * u)).astype(float); y = a + u + rng.standard_normal(n)
            img = x.copy(); true_blocks = []
            if c == 2:
                tube = rng.random(n) < np.where(a == 1, 0.9, 0.1); line = np.zeros((G, G), np.float32); line[72:106, 28:32] = 0.35
                img[tube] = np.clip(img[tube] + line[None] * m[1][None], 0, 1); true_blocks = [1]
            if c == 3:
                rr = u - 1.5 * a + 0.5 * rng.standard_normal(n); rt = np.clip((rr - rr.mean()) / rr.std(), -2.5, 2.5)
                yy, xx = np.mgrid[0:G, 0:G]; blob = (np.exp(-((yy - 40) ** 2 + (xx - 63) ** 2) / (2 * 4.0 ** 2)) * m[4]).astype(np.float32)
                img = np.clip(img + 0.25 * rt[:, None, None].astype(np.float32) * blob[None], 0, 1); true_blocks = [4]
            feats = list(clean)
            for j in true_blocks:
                feats[j] = v4.pca16(trunk_features(trunk, stats, img, m[j]))
            seed = 16000 + 100 * c + r; fold_rows = pr.folds(n, seed); roles = [pr.roles(rot) for rot in range(4)]
            pix = [img[:, m[j]].astype(np.float64) for j in range(5)]
            tfeat = [[v3.t_features(pix[j], fold_rows[roles[rot]["D"]]) for rot in range(4)] for j in range(5)]
            Z = lambda hidden: np.column_stack([feats[j] for j in range(5) if j not in hidden])

            def aipw(Fm):
                sc = np.zeros(n)
                for rot in range(4):
                    rn, re = fold_rows[roles[rot]["N"]], fold_rows[roles[rot]["E"]]
                    sc[re] = pr.aipw(np.column_stack([np.ones(len(rn)), Fm[rn]]), a[rn], y[rn], np.column_stack([np.ones(len(re)), Fm[re]]), a[re], y[re])
                return sc

            gaps = {}
            for j in range(5):
                Fj = Z([j]); gs, ses = [], []
                for rot in range(4):
                    rs_ = fold_rows[roles[rot]["S"]]
                    sc = pr.screen(np.zeros((len(rs_), 0)), Fj[rs_], tfeat[j][rot](rs_), a[rs_], np.eye(Fj.shape[1]), v4.DELTA2); gs.append(sc["gap"]); ses.append(sc["se"])
                gaps[j] = (float(np.mean(gs)), float(np.sqrt(np.mean(np.square(ses)) / 4)))
            med = float(np.median([g for g, _ in gaps.values()])); H = [j for j, (g, s) in gaps.items() if g > med + 2 * s]
            s_all, s_probe, s_ref, s_orc, s_lung = aipw(Z([])), aipw(Z(H)), aipw(Z(true_blocks)), aipw(u[:, None]), aipw(Z([4]))
            rec = {"scenario": c, "rep": r, "seed": seed, "true_blocks": [j + 1 for j in true_blocks], "naive": float(y[a == 1].mean() - y[a == 0].mean()),
                   "oracle": float(s_orc.mean()), "all_blocks": float(s_all.mean()), "lung_blocks_only": float(s_lung.mean()), "PROBE": float(s_probe.mean()),
                   "H": [j + 1 for j in H], "ref_true_hidden": float(s_ref.mean()), "gaps": {f"W{j + 1}": gaps[j] for j in range(5)}, "seconds": time.time() - t0}
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            rb.log(f"v5res V{c} r{r}: naive {rec['naive']:.3f} oracle {rec['oracle']:.3f} all {rec['all_blocks']:.3f} lungs {rec['lung_blocks_only']:.3f} "
                   f"| PROBE {rec['PROBE']:.3f} H{rec['H']} | ref {rec['ref_true_hidden']:.3f} | gaps {[round(gaps[j][0], 4) for j in range(5)]} | {rec['seconds']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=lambda s: [int(v) for v in s.split(",")], default=[0, 3, 2])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="r5")
    a = ap.parse_args()
    nets.setup_torch(2)
    run(a.scenarios, a.reps, a.tag)
