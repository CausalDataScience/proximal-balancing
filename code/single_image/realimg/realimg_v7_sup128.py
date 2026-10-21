"""v7: supervised block encoders at 128 px (floor-lowering extension (a)), resumable, GPU-capable.

Same design as realimg_v5_sup.py (64 px): per block and per cross-fit half, a ConvTrunk + head trained to predict
(A, standardised Y) from the image with only that block visible; pooled trunk features (128) -> PCA 16 per block ->
block-additive representation (80) -> one-pass relative audit (+ sequential-with-floor variant recorded) -> sealed AIPW.
Differences from v5: 128-px images (pneumoniamnist_128.npz), masks and overlays scaled x2, standardisation stats from
the 128-px images.  U stays the hidden 64-px classifier logit (results/single_image/pneumoniamnist_v1/u_logit_crossfit.npy).
Scenarios V0 clean, V1 footprint (W5), V2 footprint (W2), V3 treatment response (W5).  Seeds, draws, overlays, audit
and AIPW are identical on CPU and GPU; only the encoder training runs on the device given by --device.

Resumable: one JSON line per (scenario, rep) in results/single_image/pneumoniamnist_v1/v7sup128_<tag>.jsonl; lines
that exist are skipped.  Options:
  --device auto|cpu|cuda   encoder device (auto = cuda when available, else cpu)
  --threads N              torch/OMP threads for the CPU parts (default 2)
  --deadline EPOCH         do not start a replicate that is not expected to finish before this unix time
  --ignore-window          ignore the 22:00-05:30 local start window (the Mac night run uses the window)
  --remaining              print the number of (scenario, rep) pairs still missing and exit
  --smoke                  one quick replicate on 800 images with one epoch (window ignored), tag 'smoke'
Memo: memo/2026-10-10-pneumoniamnist-benchmark-v1.md, sections 32 and 33.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from torch import nn

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import realimg_benchmark as rb  # noqa: E402
import realimg_v3 as v3  # noqa: E402
import realimg_v4 as v4  # noqa: E402
from realimg_benchmark import pr, nets  # noqa: E402

G = 128
DATA128 = SP.parents[2] / "data/medmnist/pneumoniamnist_128.npz"
EPOCHS, BATCH, LR = 6, 128, 1e-3
WINDOW = ((22, 0), (5, 30))          # start replicates only from 22:00 to 05:30 unless --ignore-window
DEV = torch.device("cpu")


def in_window(now=None):
    now = now or datetime.now()
    hm = (now.hour, now.minute)
    return hm >= WINDOW[0] or hm < WINDOW[1]


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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


def load128():
    z = np.load(DATA128, allow_pickle=False)
    x = np.concatenate([z["train_images"], z["val_images"], z["test_images"]]).astype(np.float32) / 255.0
    lab = np.concatenate([z["train_labels"], z["val_labels"], z["test_labels"]]).ravel()
    return x, lab


def block_input(img, stats, mask):
    z = ((img - stats[0]) / stats[1]).astype(np.float32)
    vis = mask.astype(np.float32)
    return np.stack([z * vis, np.broadcast_to(vis, z.shape)], 1).astype(np.float32)


def supervised_block_features(img, stats, mask, a, y, seed, epochs):
    n = len(img); rng = np.random.default_rng(seed); half = rng.permutation(n) % 2
    xin = torch.from_numpy(block_input(img, stats, mask)).to(DEV)
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
            for s in range(0, len(te), 256):
                idx = torch.as_tensor(te[s:s + 256], device=DEV)
                feats[te[s:s + 256]] = net.pooled(xin[idx]).cpu().numpy()
    del xin
    return feats


def overlay(c, img, a, u, m, rng):
    img = img.copy(); true_blocks = []
    if c in (1, 2):
        tube = rng.random(len(img)) < np.where(a == 1, 0.9, 0.1); line = np.zeros((G, G), np.float32)
        if c == 1:
            line[8:90, 62:66] = 0.35; blk = 4
        else:
            line[72:106, 28:32] = 0.35; blk = 1
        img[tube] = np.clip(img[tube] + line[None] * m[blk][None], 0, 1); true_blocks = [blk]
    if c == 3:
        rr = u - 1.5 * a + 0.5 * rng.standard_normal(len(img)); rt = np.clip((rr - rr.mean()) / rr.std(), -2.5, 2.5)
        yy, xx = np.mgrid[0:G, 0:G]; blob = (np.exp(-((yy - 40) ** 2 + (xx - 63) ** 2) / (2 * 4.0 ** 2)) * m[4]).astype(np.float32)
        img = np.clip(img + 0.25 * rt[:, None, None].astype(np.float32) * blob[None], 0, 1); true_blocks = [4]
    return img, true_blocks


def done_pairs(out):
    if not out.exists():
        return set()
    return {(json.loads(l)["scenario"], json.loads(l)["rep"]) for l in out.read_text().splitlines() if l.strip()}


def run(scenarios, reps, tag, ignore_window=False, smoke=False, deadline=None):
    out = rb.OUT / f"v7sup128_{tag}.jsonl"
    done = done_pairs(out)
    todo = [(c, r) for c in scenarios for r in range(reps) if (c, r) not in done]
    if not todo:
        rb.log(f"v7sup128[{tag}]: nothing left"); return
    x64, lab64 = rb.load(); x, lab = load128(); assert np.array_equal(lab64, lab)
    u = np.load(rb.OUT / "u_logit_crossfit.npy"); u = (u - u.mean()) / u.std()
    m = np.kron(rb.masks(), np.ones((1, 2, 2), bool))
    if smoke:
        keep = np.random.default_rng(0).permutation(len(x))[:800]; x, u = x[keep], u[keep]
    n = len(x); stats = (float(x.mean()), float(x.std())); epochs = 1 if smoke else EPOCHS
    rb.log(f"v7sup128[{tag}]: device {DEV}, {len(todo)} replicates to do, torch {torch.__version__}, host {socket.gethostname()}")
    last = None
    for c, r in todo:
        if not (ignore_window or in_window()):
            rb.log(f"v7sup128[{tag}]: outside the 22:00-05:30 window, stopping before V{c} r{r}"); return
        if deadline is not None:
            est = 1.25 * last if last else 900.0
            if time.time() + est > deadline:
                rb.log(f"v7sup128[{tag}]: deadline rule, stopping before V{c} r{r} (need about {est:.0f}s, have {deadline - time.time():.0f}s)"); return
        t0 = time.time(); rng = np.random.default_rng(20267000 + 100 * c + r)
        a = (rng.random(n) < 0.1 + 0.8 * norm.cdf(1.5 * u)).astype(float); y = a + u + rng.standard_normal(n)
        img, true_blocks = overlay(c, x, a, u, m, rng)
        seed = 18000 + 100 * c + r
        feats = [v4.pca16(supervised_block_features(img, stats, m[k], a, y, seed + 10 * k, epochs)) for k in range(5)]
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
        Hs, cur = [], gaps
        while len(Hs) < 2:
            m_ = float(np.median([g for g, _ in cur.values()])); cand = [(g, j) for j, (g, s) in cur.items() if g > m_ + 2 * s and g > v4.DELTA2]
            if not cand: break
            Hs.append(max(cand)[1]); cur = audit(Hs)
        s_all, s_probe, s_seq, s_ref, s_orc, s_lung = aipw(Z([])), aipw(Z(H)), aipw(Z(Hs)), aipw(Z(true_blocks)), aipw(u[:, None]), aipw(Z([4]))
        F = Z([]); fold = np.random.default_rng(1).permutation(n) % 4; pred = np.zeros(n); X = np.column_stack([np.ones(n), F])
        for j in range(4):
            tr, te = fold != j, fold == j; b = np.linalg.solve(X[tr].T @ X[tr] + 1e-2 * np.eye(X.shape[1]), X[tr].T @ u[tr]); pred[te] = X[te] @ b
        r2 = float(1 - np.mean((u - pred) ** 2) / np.var(u))
        rec = {"scenario": c, "rep": r, "seed": seed, "px": G, "n": n, "smoke": smoke, "true_blocks": [j + 1 for j in true_blocks], "R2_U_from_all": r2,
               "naive": float(y[a == 1].mean() - y[a == 0].mean()), "oracle": float(s_orc.mean()), "all_blocks": float(s_all.mean()),
               "lung_blocks_only": float(s_lung.mean()), "PROBE": float(s_probe.mean()), "H": [j + 1 for j in H],
               "PROBE_seq_floor": float(s_seq.mean()), "H_seq_floor": [j + 1 for j in Hs], "ref_true_hidden": float(s_ref.mean()),
               "gaps": {f"W{j + 1}": gaps[j] for j in range(5)}, "seconds_features": t_feat, "seconds": time.time() - t0,
               "device": str(DEV), "host": socket.gethostname(), "torch": torch.__version__,
               "finished_at": datetime.now().isoformat(timespec="seconds")}
        with out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        last = rec["seconds"]
        rb.log(f"v7sup128[{tag}] V{c} r{r}: R2 {r2:.3f} oracle {rec['oracle']:.3f} all {rec['all_blocks']:.3f} | PROBE {rec['PROBE']:.3f} H{rec['H']} "
               f"| seq+floor {rec['PROBE_seq_floor']:.3f} H{rec['H_seq_floor']} | ref {rec['ref_true_hidden']:.3f} | feat {t_feat:.0f}s total {rec['seconds']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=lambda s: [int(v) for v in s.split(",")], default=[0, 1, 2, 3])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--tag", default="r10")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--ignore-window", action="store_true")
    ap.add_argument("--remaining", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--group", default="")          # label only, used by the Mac supervisor to find the process
    a = ap.parse_args()
    if a.remaining:
        done = done_pairs(rb.OUT / f"v7sup128_{a.tag}.jsonl")
        print(sum(1 for c in a.scenarios for r in range(a.reps) if (c, r) not in done)); sys.exit(0)
    nets.setup_torch(a.threads)
    DEV = pick_device(a.device)
    if DEV.type == "cuda":
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if a.smoke:
        run([3], 1, "smoke", ignore_window=True, smoke=True)
    else:
        run(a.scenarios, a.reps, a.tag, ignore_window=a.ignore_window, deadline=a.deadline)
