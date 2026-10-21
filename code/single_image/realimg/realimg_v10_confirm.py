"""v10: confirmatory run with new seeds, two block layouts, eight scenarios, GPU-capable (memo section 38).

Layouts: anat (the five hand-drawn blocks, 16 principal components per block) and grid4 (sixteen 16-pixel cells,
8 components per cell, as in v6).  Scenarios (64 px, U = cross-fitted classifier logit, hidden, tau = 1).  The overlays
are drawn on the anatomical block pixels, so the images are identical in both layouts and the replicate draws are
shared, which pairs the two layouts replicate by replicate:
  0 clean;  1 footprint line in W5 (drawn with prob .9 if treated, .1 if not);  2 the same line in W2;
  3 treatment-response mark in W5 (U - 1.5A + .5nu);  4 mediator mark in W5 (M = A + .5nu, Y = .5A + .5M + U + eps);
  5 response mark in W2 and W5;  6 response mark in all five blocks;  7 device line in W5, prob expit(-2 + 2A + 1.5U).
Added for memo section 40 (second DeltaAI job):
  8 footprint line in W5 (.9/.1) together with a response mark in W2 (two different contents in two blocks);
  9-13 response mark in W5 with imprint kappa = 0, 0.5, 1, 2, 3 in U - kappa*A + .5nu (3 is kappa = 1.5);
  14 footprint line in W5 drawn with prob .98 if treated and .02 if not (near-deterministic device).
Per block (cell) and cross-fit half, the supervised encoder of realimg_v5_sup.py (ConvTrunk + head predicting A and the
standardised Y from that block alone, six epochs) gives 128 pooled features, reduced by PCA.  Methods: naive, oracle,
whole image (all blocks), PROBE (main-text rule: hide a block whose held-out gap exceeds the median gap by more than two
standard errors, then AIPW on the rest), the sequential rule with the floor 0.005 (at most 2 blocks, 3 cells; a
sensitivity), and hiding the true blocks (reference; none when every block is touched).
Resumable: results/single_image/pneumoniamnist_v1/v10_<layout>_<tag>.jsonl, one line per (scenario, rep).
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import norm
from torch import nn

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import realimg_benchmark as rb  # noqa: E402
import realimg_v3 as v3  # noqa: E402
from realimg_benchmark import pr, nets  # noqa: E402

EPOCHS, BATCH, LR = 6, 128, 1e-3
DELTA2 = 0.005
DEV = torch.device("cpu")
CENTERS = {0: (19.0, 15.5), 1: (44.0, 15.5), 2: (19.0, 47.5), 3: (44.0, 47.5), 4: (20.0, 31.5)}
LAYOUT_ID = {"anat": 0, "grid4": 1}


def grid_masks(g):
    edges = np.linspace(0, 64, g + 1).round().astype(int)
    m = np.zeros((g * g, 64, 64), bool); k = 0
    for r in range(g):
        for c in range(g):
            m[k, edges[r]:edges[r + 1], edges[c]:edges[c + 1]] = True; k += 1
    return m


def pca_q(f, q):
    mu = f.mean(0); _, _, vt = np.linalg.svd(f - mu, full_matrices=False); z = (f - mu) @ vt[:q].T
    return (z - z.mean(0)) / (z.std(0) + 1e-9)


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


def supervised_block_features(img, stats, mask, a, y, seed, epochs):
    """As realimg_v5_sup.supervised_block_features, with the tensors on DEV."""
    n = len(img); rng = np.random.default_rng(seed); half = rng.permutation(n) % 2
    z = nets.standardise(img, stats); vis = mask.astype(np.float32)
    xin = torch.from_numpy(np.stack([z * vis, np.broadcast_to(vis, z.shape)], 1).astype(np.float32)).to(DEV)
    ys = (y - y.mean()) / y.std(); feats = np.zeros((n, 128))
    a_t = torch.as_tensor(a.astype(np.float32), device=DEV); y_t = torch.as_tensor(ys.astype(np.float32), device=DEV)
    for h in (0, 1):
        tr, te = np.where(half != h)[0], np.where(half == h)[0]
        nets.seed_all(seed + h); net = SupNet().to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
        for ep in range(epochs):
            net.train(); order = rng.permutation(tr)
            for s in range(0, len(order), BATCH):
                idx = torch.as_tensor(order[s:s + BATCH], device=DEV); out = net(xin[idx])
                loss = nn.functional.binary_cross_entropy_with_logits(out[:, 0], a_t[idx]) + nn.functional.mse_loss(out[:, 1], y_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for s in range(0, len(te), 512):
                idx = torch.as_tensor(te[s:s + 512], device=DEV)
                feats[te[s:s + 512]] = net.pooled(xin[idx]).cpu().numpy()
    del xin
    return feats


def draw(c, r, x, u, m):
    """Replicate r of scenario c: treatment, outcome and overlays, shared by both layouts."""
    n = len(x); rng = np.random.default_rng(20271000 + 100 * c + r)
    a = (rng.random(n) < 0.1 + 0.8 * norm.cdf(1.5 * u)).astype(float)
    eps = rng.standard_normal(n); nu = rng.standard_normal(n)
    y = a + u + eps
    img = x.copy(); ov = np.zeros((64, 64), bool); anat_true = []; extra = {}
    yy, xx = np.mgrid[0:64, 0:64]
    blob = lambda k: (np.exp(-((yy - CENTERS[k][0]) ** 2 + (xx - CENTERS[k][1]) ** 2) / (2 * 2.0 ** 2)) * m[k]).astype(np.float32)
    zc = lambda v: np.clip((v - v.mean()) / v.std(), -2.5, 2.5).astype(np.float32)
    KAPPA = {3: 1.5, 9: 0.0, 10: 0.5, 11: 1.0, 12: 2.0, 13: 3.0}
    if c in (1, 2, 7, 8, 14):
        k = 1 if c == 2 else 4
        line = np.zeros((64, 64), np.float32)
        if c == 2:
            line[36:53, 14:16] = 0.35
        else:
            line[4:45, 31:33] = 0.35
        line = line * m[k]
        if c == 7:
            carry = rng.random(n) < expit(-2.0 + 2.0 * a + 1.5 * u)
        elif c == 14:
            carry = rng.random(n) < np.where(a == 1, 0.98, 0.02)
        else:
            carry = rng.random(n) < np.where(a == 1, 0.9, 0.1)
        if c in (7, 14):
            extra = {"device_rate": float(carry.mean()), "device_rate_A0": float(carry[a == 0].mean()), "device_rate_A1": float(carry[a == 1].mean())}
        img[carry] = np.clip(img[carry] + line[None], 0, 1); ov = line > 0; anat_true = [k]
    if c in (3, 5, 6, 8, 9, 10, 11, 12, 13):
        kappa = KAPPA.get(c, 1.5)
        rt = zc(u - kappa * a + 0.5 * nu); ks = {5: [1, 4], 6: [0, 1, 2, 3, 4], 8: [1]}.get(c, [4])
        mark = sum(blob(k) for k in ks)
        img = np.clip(img + 0.25 * rt[:, None, None] * mark[None], 0, 1); ov = ov | (mark > 0.05); anat_true = sorted(set(anat_true) | set(ks))
        extra = {**extra, "kappa": kappa}
    if c == 4:
        M = a + 0.5 * nu; y = 0.5 * a + 0.5 * M + u + eps
        img = np.clip(img - 0.25 * zc(M)[:, None, None] * blob(4)[None], 0, 1); ov = blob(4) > 0.05; anat_true = [4]
    return a, y, img, ov, anat_true, extra


def done_pairs(out):
    if not out.exists():
        return set()
    return {(json.loads(l)["scenario"], json.loads(l)["rep"]) for l in out.read_text().splitlines() if l.strip()}


def run(layout, scenarios, rep_start, reps, tag, deadline=None, smoke=False, rep_mod=1, rep_res=0):
    out = rb.OUT / f"v10_{layout}_{tag}.jsonl"; done = done_pairs(out)
    # replicate-major order, so that a run stopped early leaves every scenario with replicates; rep_mod/rep_res split the
    # replicates across workers (worker w takes r = w mod W) so that all workers carry the same load
    todo = [(c, r) for r in range(rep_start, reps) if r % rep_mod == rep_res for c in scenarios if (c, r) not in done]
    if not todo:
        rb.log(f"v10[{layout},{tag}]: nothing left"); return
    x, label = rb.load(); m = rb.masks()
    ck = rb.OUT / f"mae_pneu_e{rb.MAE_EPOCHS}_s{rb.MAE_SEED}.pt"
    if not ck.exists():
        sys.exit(f"missing {ck}: the standardisation statistics come from this checkpoint; refusing to train a new one")
    _, stats, _ = rb.get_mae(x)
    u = np.load(rb.OUT / "u_logit_crossfit.npy"); u = (u - u.mean()) / u.std()
    masks = m if layout == "anat" else grid_masks(4); nb = len(masks); Q = 16 if nb <= 9 else 8; cap = 2 if layout == "anat" else 3
    epochs = EPOCHS
    if smoke:
        keep = np.random.default_rng(0).permutation(len(x))[:800]; x, u = x[keep], u[keep]; epochs = 1
    n = len(x); last = None
    rb.log(f"v10[{layout},{tag}]: {len(todo)} replicates, {nb} blocks, Q={Q}, device {DEV}, torch {torch.__version__}, host {socket.gethostname()}")
    for c, r in todo:
        if deadline is not None:
            est = 1.25 * last if last else 600.0
            if time.time() + est > deadline:
                rb.log(f"v10[{layout},{tag}]: deadline rule, stopping before V{c} r{r}"); return
        t0 = time.time()
        a, y, img, ov, anat_true, extra = draw(c, r, x, u, m)
        true_blocks = anat_true if layout == "anat" else [k for k in range(nb) if (masks[k] & ov).any()]
        base = 3_000_000 + 100_000 * LAYOUT_ID[layout] + 4_000 * c + 40 * r
        feats = [pca_q(supervised_block_features(img, stats, masks[k], a, y, base + 2 * k, epochs), Q) for k in range(nb)]
        t_feat = time.time() - t0
        fold_rows = pr.folds(n, base); roles = [pr.roles(rot) for rot in range(4)]
        pix = [img[:, masks[k]].astype(np.float64) for k in range(nb)]
        tfeat = [[v3.t_features(pix[k], fold_rows[roles[rot]["D"]]) for rot in range(4)] for k in range(nb)]
        Z = lambda hidden: np.column_stack([feats[k] for k in range(nb) if k not in hidden])

        def aipw(F):
            sc = np.zeros(n)
            for rot in range(4):
                rn, re = fold_rows[roles[rot]["N"]], fold_rows[roles[rot]["E"]]
                sc[re] = pr.aipw(np.column_stack([np.ones(len(rn)), F[rn]]), a[rn], y[rn], np.column_stack([np.ones(len(re)), F[re]]), a[re], y[re])
            return sc

        def audit(hidden):
            o = {}
            for k in range(nb):
                if k in hidden: continue
                F = Z(list(hidden) + [k]); gs, ses = [], []
                for rot in range(4):
                    rs_ = fold_rows[roles[rot]["S"]]
                    sc = pr.screen(np.zeros((len(rs_), 0)), F[rs_], tfeat[k][rot](rs_), a[rs_], np.eye(F.shape[1]), DELTA2); gs.append(sc["gap"]); ses.append(sc["se"])
                o[k] = (float(np.mean(gs)), float(np.sqrt(np.mean(np.square(ses)) / 4)))
            return o

        gaps = audit([]); med = float(np.median([g for g, _ in gaps.values()]))
        H = [k for k, (g, s) in gaps.items() if g > med + 2 * s]
        Hs, cur = [], gaps
        while len(Hs) < cap:
            m_ = float(np.median([g for g, _ in cur.values()])); cand = [(g, k) for k, (g, s) in cur.items() if g > m_ + 2 * s and g > DELTA2]
            if not cand: break
            Hs.append(max(cand)[1]); cur = audit(Hs)
        s_all, s_probe, s_seq, s_orc = aipw(Z([])), aipw(Z(H)), aipw(Z(Hs)), aipw(u[:, None])
        s_ref = aipw(Z(true_blocks)) if 0 < len(true_blocks) < nb else None
        F = Z([]); fold = np.random.default_rng(1).permutation(n) % 4; pred = np.zeros(n); X = np.column_stack([np.ones(n), F])
        for j in range(4):
            tr, te = fold != j, fold == j; b = np.linalg.solve(X[tr].T @ X[tr] + 1e-2 * np.eye(X.shape[1]), X[tr].T @ u[tr]); pred[te] = X[te] @ b
        r2 = float(1 - np.mean((u - pred) ** 2) / np.var(u))
        rec = {"layout": layout, "scenario": c, "rep": r, "seed": base, "n": n, "smoke": smoke, "n_blocks": nb, "Q": Q,
               "true_blocks": [k + 1 for k in true_blocks], **extra, "R2_U_from_all": r2,
               "naive": float(y[a == 1].mean() - y[a == 0].mean()), "oracle": float(s_orc.mean()), "all_blocks": float(s_all.mean()),
               "PROBE": float(s_probe.mean()), "H": [k + 1 for k in H], "PROBE_seq_floor": float(s_seq.mean()), "H_seq_floor": [k + 1 for k in Hs],
               "ref_true_hidden": (float(s_ref.mean()) if s_ref is not None else None),
               "gaps": {f"B{k + 1}": gaps[k] for k in range(nb)}, "seconds_features": t_feat, "seconds": time.time() - t0,
               "device": str(DEV), "host": socket.gethostname(), "torch": torch.__version__, "finished_at": datetime.now().isoformat(timespec="seconds")}
        with out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        last = rec["seconds"]
        ref = "n/a" if rec["ref_true_hidden"] is None else f"{rec['ref_true_hidden']:.3f}"
        rb.log(f"v10[{layout},{tag}] V{c} r{r}: R2 {r2:.3f} oracle {rec['oracle']:.3f} all {rec['all_blocks']:.3f} | PROBE {rec['PROBE']:.3f} H{rec['H']} "
               f"| seq {rec['PROBE_seq_floor']:.3f} H{rec['H_seq_floor']} | ref {ref} true{rec['true_blocks']} | feat {t_feat:.0f}s total {rec['seconds']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["anat", "grid4"], required=True)
    ap.add_argument("--scenarios", type=lambda s: [int(v) for v in s.split(",")], default=list(range(8)))
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--tag", default="r50")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rep-mod", type=int, default=1)
    ap.add_argument("--rep-res", type=int, default=0)
    a = ap.parse_args()
    nets.setup_torch(a.threads)
    DEV = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device))
    if DEV.type == "cuda":
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if a.smoke:
        run(a.layout, a.scenarios, 0, 1, f"smoke_{DEV.type}", smoke=True)
    else:
        run(a.layout, a.scenarios, a.rep_start, a.reps, a.tag, deadline=a.deadline, rep_mod=a.rep_mod, rep_res=a.rep_res)
