"""Masked autoencoder pre-training on fold-R images only (plan section 6.2).

Convolutional masked image modelling: 8 x 8 patches, mask ratio 0.75 (48 of 64 patches hidden per image),
hidden pixels set to 0 in the standardised image, a visibility channel marks what is shown.  The trunk is
nets.ConvTrunk; a light transposed-convolution decoder reconstructs the full image and the loss is the mean
squared error on hidden pixels only.  The trunk weights initialise the stage-2 encoder phi_S later
(load_trunk).  Reads images only.

Reported: hidden-pixel MSE on a fixed-mask validation part of R, next to two references, predicting the
fold-R mean image and predicting 0 (the global mean), both in the same standardised units.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import nets

PATCH = 8
MASK_RATIO = 0.75
N_PATCH = (64 // PATCH) ** 2
N_HIDDEN = int(round(MASK_RATIO * N_PATCH))


class MaskedAutoencoder(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.trunk = nets.ConvTrunk(width)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(width, 32, 4, 2, 1), nn.GELU(),     # 16
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.GELU(),        # 32
            nn.ConvTranspose2d(16, 8, 4, 2, 1), nn.GELU(),         # 64
            nn.Conv2d(8, 1, 3, 1, 1),
        )

    def forward(self, x):
        return self.decoder(self.trunk(x))


def patch_visibility(n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 64, 64) float32, 1 on the 16 visible patches of each image, 0 on the 48 hidden ones."""
    keys = rng.random((n, N_PATCH))
    hidden = np.argsort(keys, axis=1)[:, :N_HIDDEN]
    vis = np.ones((n, N_PATCH), dtype=np.float32)
    np.put_along_axis(vis, hidden, 0.0, axis=1)
    side = 64 // PATCH
    return np.kron(vis.reshape(n, side, side), np.ones((1, PATCH, PATCH), dtype=np.float32))


def _batch(z: np.ndarray, vis: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.stack([z * vis, vis], axis=1)
    return torch.from_numpy(x).to(device), torch.from_numpy(z[:, None]).to(device)


def _masked_mse(pred, target, vis) -> torch.Tensor:
    hid = 1.0 - vis
    return (((pred - target) ** 2) * hid).sum() / hid.sum()


def train_mae(images_r: np.ndarray, seed: int, device: torch.device, epochs: int = 40, batch: int = 128,
              lr: float = 2e-3, val_frac: float = 0.2, width: int = 64, log=print) -> dict:
    if images_r.ndim != 3:
        raise TypeError("train_mae takes a (n, 64, 64) image array and nothing else")
    t0 = time.time()
    nets.seed_all(seed)
    rng = np.random.default_rng([seed, 6])
    n = images_r.shape[0]
    perm = rng.permutation(n)
    n_val = int(round(val_frac * n))
    val_rows, tr_rows = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    stats = (float(images_r[tr_rows].mean()), float(images_r[tr_rows].std()))
    z_all = nets.standardise(images_r, stats)
    z_tr, z_val = z_all[tr_rows], z_all[val_rows]
    vis_val = patch_visibility(len(val_rows), np.random.default_rng([seed, 7]))
    model = MaskedAutoencoder(width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps_per_epoch = int(np.ceil(len(tr_rows) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch,
                                                pct_start=0.1)
    hist = []

    def evaluate() -> float:
        model.eval()
        tot, cnt = 0.0, 0.0
        with torch.no_grad():
            for c0 in range(0, len(val_rows), 512):
                x, y = _batch(z_val[c0:c0 + 512], vis_val[c0:c0 + 512], device)
                pred = model(x)
                hid = 1.0 - x[:, 1:2]
                tot += float((((pred - y) ** 2) * hid).sum())
                cnt += float(hid.sum())
        model.train()
        return tot / cnt

    for ep in range(epochs):
        order = rng.permutation(len(tr_rows))
        vis_ep = patch_visibility(len(tr_rows), rng)
        run = 0.0
        for b in range(steps_per_epoch):
            idx = order[b * batch:(b + 1) * batch]
            x, y = _batch(z_tr[idx], vis_ep[idx], device)
            loss = _masked_mse(model(x), y, x[:, 1:2])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss.detach())
        val = evaluate()
        hist.append({"epoch": ep + 1, "train_masked_mse": run / steps_per_epoch, "val_masked_mse": val})
        if (ep + 1) % 10 == 0 or ep == 0:
            log(f"    MAE epoch {ep + 1}/{epochs}: train {run / steps_per_epoch:.4f}  val {val:.4f}")
    hid_val = 1.0 - vis_val
    mean_img = z_tr.mean(0)
    ref_mean_image = float((((z_val - mean_img[None]) ** 2) * hid_val).sum() / hid_val.sum())
    ref_zero = float(((z_val ** 2) * hid_val).sum() / hid_val.sum())
    return {"model": model.cpu(), "stats": stats, "history": hist, "val_masked_mse": hist[-1]["val_masked_mse"],
            "reference_mean_image_mse": ref_mean_image, "reference_zero_mse": ref_zero,
            "n_train": int(len(tr_rows)), "n_val": int(len(val_rows)), "epochs": epochs, "batch": batch, "lr": lr,
            "parameters": nets.count_parameters(model), "seconds": round(time.time() - t0, 1),
            "device": str(device)}


def save(path: Path, result: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": result["model"].state_dict(), "stats": result["stats"],
                "width": result["model"].trunk.width, "meta": meta}, path)


def load_trunk(path: Path) -> tuple[nets.ConvTrunk, tuple[float, float]]:
    """Stage-2 interface: the pre-trained trunk and the image standardisation it expects."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = MaskedAutoencoder(ck["width"])
    model.load_state_dict(ck["state_dict"])
    return model.trunk, tuple(ck["stats"])
