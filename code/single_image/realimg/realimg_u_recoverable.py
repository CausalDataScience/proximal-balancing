"""Diagnostic for the next benchmark design: is the hidden U recoverable from the image?

(1) U_label  = standardised binary pneumonia label (benchmark v1).  Floor = how well CNN_all's 128 MAE features
    explain it linearly.
(2) U_logit  = standardised cross-fitted logit of a small supervised CNN (2 folds, trained on clean images with the
    real label).  A frozen "reader" like the paper's SCM-4 reader; defines a severity that lives in the image.
For both: linear R^2 from CNN_all features, from lung-only features (block 5 hidden), per-block features; AUC of
the classifier; and one replicate of the clean scenario (A, Y simulated from U) with oracle, CNN_all and
CNN_all-without-block-5 AIPW on the sealed folds.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import realimg_benchmark as rb  # noqa: E402
from realimg_benchmark import pr, level_a, nets  # noqa: E402


def r2(y, X, train, test, lam=1e-3):
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-9
    Xs = (X - mu) / sd
    F = np.column_stack([np.ones(len(y)), Xs])
    b = np.linalg.solve(F[train].T @ F[train] + lam * np.eye(F.shape[1]), F[train].T @ y[train])
    return 1 - np.mean((y[test] - F[test] @ b) ** 2) / np.var(y[test])


class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nets.ConvTrunk(64)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        f = self.trunk(x)
        return self.head(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)).squeeze(1)


def crossfit_logits(x, label, stats, epochs=6, seed=11):
    n = len(x)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(n) % 2
    z = nets.standardise(x, stats).astype(np.float32)
    xin = np.stack([z, np.ones_like(z)], 1)
    logits = np.zeros(n)
    for k in (0, 1):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        nets.seed_all(seed + k)
        m = Clf()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
        pos = label[tr].mean()
        w = torch.tensor([(1 - pos) / pos], dtype=torch.float32)
        for ep in range(epochs):
            order = rng.permutation(tr)
            m.train()
            for s in range(0, len(order), 128):
                idx = order[s:s + 128]
                xb = torch.from_numpy(xin[idx]); yb = torch.from_numpy(label[idx].astype(np.float32))
                loss = nn.functional.binary_cross_entropy_with_logits(m(xb), yb, pos_weight=w)
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            for s in range(0, len(te), 512):
                logits[te[s:s + 512]] = m(torch.from_numpy(xin[te[s:s + 512]])).numpy()
    return logits


def main():
    t0 = time.time()
    nets.setup_torch(2)
    x, label = rb.load()
    m = rb.masks()
    trunk, stats, _ = rb.get_mae(x)
    n = len(x)
    rng = np.random.default_rng(3)
    perm = rng.permutation(n); train, test = perm[: n // 2], perm[n // 2:]
    feats_all = rb.trunk_features(trunk, stats, x)
    feats_lung = rb.trunk_features(trunk, stats, x, ~m[4])
    feats_block = [rb.trunk_features(trunk, stats, x, m[k]) for k in range(5)]
    logit = crossfit_logits(x, label, stats)
    rb.log(f"classifier cross-fitted AUC {roc_auc_score(label, logit):.3f}  ({time.time() - t0:.0f}s)")
    for name, u in (("U_label", (label - label.mean()) / label.std()), ("U_logit", (logit - logit.mean()) / logit.std())):
        rb.log(f"{name}: R2 from CNN_all 128 {r2(u, feats_all, train, test):.3f} | lungs only {r2(u, feats_lung, train, test):.3f} | "
               f"per block {[round(r2(u, fb, train, test), 3) for fb in feats_block]}")
    # one clean replicate per U definition: A, Y from U; AIPW oracle / CNN_all / CNN_all minus block 5
    for name, u in (("U_label", (label - label.mean()) / label.std()), ("U_logit", (logit - logit.mean()) / logit.std())):
        g = np.random.default_rng(99)
        a = (g.random(n) < 0.1 + 0.8 * norm.cdf(1.5 * u)).astype(float)
        y = a + u + g.standard_normal(n)
        view = level_a.make_view(np.zeros((n, 0)), [x[:, m[k]].astype(float) for k in range(5)], a, y)
        seed = 4242
        res = {"oracle": rb.aipw_features(view, u[:, None], seed), "CNN_all": rb.aipw_features(view, feats_all, seed),
               "CNN_all_minus_W5": rb.aipw_features(view, feats_lung, seed),
               "naive": float(y[a == 1].mean() - y[a == 0].mean())}
        rb.log(f"{name} clean replicate: " + " ".join(f"{k} {v:.3f}" for k, v in res.items()))
    np.save(rb.OUT / "u_logit_crossfit.npy", logit)
    rb.log(f"saved logits; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
