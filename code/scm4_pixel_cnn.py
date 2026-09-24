"""SCM-4 comparator: adjust for every pixel of the six images with an end-to-end network.

The question it answers is the one a reader of the paper would ask first: if the proxies are images, why not give
all of them to a strong network and adjust for whatever it extracts?  To make that comparison fair the network
starts from the same pretrained reader PROBE's numbers come from:

    image tower  = the frozen reader's layers up to its 256 features, initialised with the reader's weights and
                   then fine-tuned; shared by the six images
    projection   = Linear(256, 16), shared; its first output row is initialised to the reader's own output layer,
                   so at step zero the network already knows every image's orientation
    heads        = (6 x 16 features, standardised X) -> 128 -> 64 -> propensity, m0, m1

It is fitted on the N fold of each of the four role rotations PROBE uses (a fifth of N decides when to stop),
and the AIPW score is averaged on the E fold with the propensity clipped to the same [0.05, 0.95].
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

import family_v2_images as im
import family_v2_probe as pr

CONFIG = {"epochs": 5, "batch_rows": 128, "lr_tower": 1e-4, "lr_rest": 1e-3, "val_fraction": 0.2, "patience": 2,
          "feature_dim": 16, "hidden": [128, 64], "max_seconds": 2400, "eval_rows": 500}


class PixelNet(nn.Module):
    def __init__(self, reader_state: dict, x_dim: int, n_images: int) -> None:
        super().__init__()
        reader = im.build()
        reader.load_state_dict(reader_state)
        layers = list(reader.children())
        self.tower = nn.Sequential(*layers[:-1])                       # ... -> Linear(2048, 256) -> ReLU
        self.project = nn.Linear(256, CONFIG["feature_dim"])
        with torch.no_grad():                                           # row 0 starts as the reader's output layer
            self.project.weight[0].copy_(layers[-1].weight[0])
            self.project.bias[0].copy_(layers[-1].bias[0])
        h1, h2 = CONFIG["hidden"]
        self.body = nn.Sequential(nn.Linear(n_images * CONFIG["feature_dim"] + x_dim, h1), nn.ReLU(),
                                  nn.Linear(h1, h2), nn.ReLU())
        self.e, self.m0, self.m1 = nn.Linear(h2, 1), nn.Linear(h2, 1), nn.Linear(h2, 1)
        self.n_images = n_images

    def forward(self, pixels: torch.Tensor, x: torch.Tensor):
        """pixels: (rows * n_images, 3, 64, 64) in row-major order; x: (rows, x_dim)."""
        feats = self.project(self.tower(pixels)).reshape(len(x), -1)
        h = self.body(torch.cat([feats, x], dim=1))
        return self.e(h).squeeze(1), self.m0(h).squeeze(1), self.m1(h).squeeze(1)


def _pixels(images, src_rows: np.ndarray, dev) -> torch.Tensor:
    """The original images of the given (rows, n_images) bank indices, as one batch in row-major order."""
    flat = src_rows.ravel()
    uniq, inverse = np.unique(flat, return_inverse=True)
    block = torch.as_tensor(np.asarray(images[uniq]), device=dev)
    return im.to_input(block[torch.as_tensor(inverse, device=dev)])


def _loss(net, images, src, x, a, y_std, rows, dev) -> torch.Tensor:
    logit, m0, m1 = net(_pixels(images, src[rows], dev), x[rows])
    at = a[rows]
    fitted = torch.where(at > 0.5, m1, m0)
    return ((at - torch.sigmoid(logit)) ** 2).mean() + ((y_std[rows] - fitted) ** 2).mean()


@torch.no_grad()
def _evaluate(net, images, src, x, rows, dev):
    net.eval()
    e, m0, m1 = [], [], []
    for s in range(0, len(rows), CONFIG["eval_rows"]):
        r = rows[s:s + CONFIG["eval_rows"]]
        logit, b0, b1 = net(_pixels(images, src[r], dev), x[r])
        e.append(torch.sigmoid(logit)); m0.append(b0); m1.append(b1)
    return [torch.cat(v).cpu().numpy().astype(np.float64) for v in (e, m0, m1)]


def fit_rotation(images, src, x_np, a_np, y_np, n_rows, e_rows, reader_state, seed: int, dev, deadline: float) -> dict:
    """One rotation: fit on the N fold, score on the E fold.  Raises TimeoutError at the batch that passes the cap."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_val = int(round(CONFIG["val_fraction"] * len(perm)))
    val, train = np.sort(perm[:n_val]), perm[n_val:]
    mu_x, sd_x = x_np[train].mean(0), x_np[train].std(0) + 1e-12
    mu_y, sd_y = float(y_np[train].mean()), float(y_np[train].std() + 1e-12)
    x = torch.as_tensor((x_np - mu_x) / sd_x, dtype=torch.float32, device=dev)
    a = torch.as_tensor(a_np, dtype=torch.float32, device=dev)
    y_std = torch.as_tensor((y_np - mu_y) / sd_y, dtype=torch.float32, device=dev)
    net = PixelNet(reader_state, x_np.shape[1], src.shape[1]).to(dev)
    tower = list(net.tower.parameters())
    rest = [p for name, p in net.named_parameters() if not name.startswith("tower.")]
    opt = torch.optim.Adam([{"params": tower, "lr": CONFIG["lr_tower"]}, {"params": rest, "lr": CONFIG["lr_rest"]}])
    best = {"loss": np.inf, "epoch": 0, "state": None}
    history, bad = [], 0
    for epoch in range(1, CONFIG["epochs"] + 1):
        net.train()
        order = rng.permutation(train)
        for s in range(0, len(order), CONFIG["batch_rows"]):
            if time.time() > deadline:
                raise TimeoutError(f"pixel CNN passed its time cap in epoch {epoch}")
            rows = np.sort(order[s:s + CONFIG["batch_rows"]])
            opt.zero_grad(set_to_none=True)
            _loss(net, images, src, x, a, y_std, rows, dev).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            parts = [float(_loss(net, images, src, x, a, y_std, val[s:s + CONFIG["eval_rows"]], dev))
                     * len(val[s:s + CONFIG["eval_rows"]]) for s in range(0, len(val), CONFIG["eval_rows"])]
        v = sum(parts) / len(val)
        history.append(v)
        if v < best["loss"] - 1e-5:
            best, bad = {"loss": v, "epoch": epoch, "state": {k: t.detach().clone() for k, t in net.state_dict().items()}}, 0
        else:
            bad += 1
            if bad >= CONFIG["patience"]:
                break
    net.load_state_dict(best["state"])
    e_raw, m0, m1 = _evaluate(net, images, src, x, np.sort(e_rows), dev)
    e_rows = np.sort(e_rows)
    m0, m1 = m0 * sd_y + mu_y, m1 * sd_y + mu_y
    if not (np.isfinite(e_raw).all() and np.isfinite(m0).all() and np.isfinite(m1).all()):
        return {"finite": False, "val_history": history}
    e = np.clip(e_raw, *pr.CLIP)
    a_e, y_e = a_np[e_rows], y_np[e_rows]
    psi = m1 - m0 + a_e * (y_e - m1) / e - (1 - a_e) * (y_e - m0) / (1 - e)
    return {"finite": True, "theta": float(psi.mean()), "epoch": best["epoch"], "val_history": history,
            "outside_clip_before": float(((e_raw < pr.CLIP[0]) | (e_raw > pr.CLIP[1])).mean())}


def estimate(images, src: np.ndarray, x: np.ndarray, a: np.ndarray, y: np.ndarray, reader_state: dict,
             fold_seed: int, dev) -> dict:
    """The comparator's estimate: the mean over the four rotations of the E-fold AIPW means."""
    t0 = time.time()
    deadline = t0 + CONFIG["max_seconds"]
    folds = pr.folds(len(a), fold_seed)
    rotations = []
    try:
        for rot in range(4):
            rl = pr.roles(rot)
            rotations.append(fit_rotation(images, src, x, a, y, folds[rl["N"]], folds[rl["E"]], reader_state,
                                          fold_seed + 1_000 + rot, dev, deadline))
    except TimeoutError as err:
        return {"finite": False, "ate": None, "reason": str(err), "rotations": rotations,
                "seconds": round(time.time() - t0, 1)}
    finite = all(r["finite"] for r in rotations)
    return {"finite": finite, "ate": float(np.mean([r["theta"] for r in rotations])) if finite else None,
            "rotations": rotations, "seconds": round(time.time() - t0, 1), "config": CONFIG}
