"""Shared torch pieces: device and seeding, the convolutional trunk the masked autoencoder pre-trains, and
the stage-2 encoder phi_S that will start from it (plan sections 6.2 and 7.1).

Input format everywhere: (B, 2, 64, 64) float32 with channel 0 the standardised image with hidden pixels
set to 0 and channel 1 the visibility indicator.  The masked autoencoder hides random 8 x 8 patches; the
stage-2 encoder hides the held-out regions (regions.holdout_view).  Same format, so the trunk transfers.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch import nn

THREADS = 2


def setup_torch(threads: int = THREADS) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(requested)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def standardise(images: np.ndarray, stats: tuple[float, float]) -> np.ndarray:
    mu, sd = stats
    return ((images - mu) / sd).astype(np.float32)


class ConvTrunk(nn.Module):
    """(B, 2, 64, 64) -> (B, width, 8, 8): one cell per 8 x 8 patch.  Receptive field 33 px."""

    def __init__(self, width: int = 64):
        super().__init__()
        w2 = width // 2
        self.net = nn.Sequential(
            nn.Conv2d(2, w2 // 2, 3, 1, 1), nn.GELU(),          # 64
            nn.Conv2d(w2 // 2, w2, 3, 2, 1), nn.GELU(),         # 32
            nn.Conv2d(w2, width, 3, 2, 1), nn.GELU(),           # 16
            nn.Conv2d(width, width, 3, 2, 1), nn.GELU(),        # 8
            nn.Conv2d(width, width, 3, 1, 1), nn.GELU(),        # 8
        )
        self.width = width

    def forward(self, x):
        return self.net(x)


class RegionEncoder(nn.Module):
    """Stage-2 encoder phi_S: trunk, visibility-weighted average pool, MLP to d_Z.  Built here so that
    stage 2 only has to load the masked-autoencoder trunk (mae.load_trunk) and train."""

    def __init__(self, trunk: ConvTrunk, d_out: int = 2, hidden: int = 64):
        super().__init__()
        self.trunk = trunk
        self.head = nn.Sequential(nn.Linear(2 * trunk.width, hidden), nn.GELU(), nn.Linear(hidden, d_out))

    def forward(self, x):
        f = self.trunk(x)
        vis = nn.functional.avg_pool2d(x[:, 1:2], 8)                         # (B, 1, 8, 8)
        mean = (f * vis).sum((2, 3)) / (vis.sum((2, 3)) + 1e-6)
        mx = f.amax((2, 3))
        return self.head(torch.cat([mean, mx], 1))


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
