"""SCM-4e v2 comparator: adjust for every pixel of the six images with an end-to-end network
(PRD memo/2026-09-21-scm4e-v2-labelfree-association-prd-v1.md, section 3.7).

The training procedure is the sealed one in `scm4_pixel_cnn.py`, called unchanged.  Only the starting point
differs, and it differs so that the comparison stays fair: the image tower starts from the frozen label-free
encoder, and the projection starts from that encoder's own last layer, so at step zero this network holds
exactly the 32 numbers per image that PROBE's front end works on.  Whatever it does after that, it does with a
head start PROBE never gets, since PROBE never fine-tunes the encoder.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

import scm4_pixel_cnn as px
import scm4e_encoder as enc

CONFIG = dict(px.CONFIG, feature_dim=enc.LATENT,
              initialised_from="the frozen label-free autoencoder encoder (tower and projection)")


class EncoderPixelNet(nn.Module):
    def __init__(self, encoder_state: dict, x_dim: int, n_images: int) -> None:
        super().__init__()
        encoder, _ = enc.build()
        encoder.load_state_dict(encoder_state)
        layers = list(encoder.children())
        self.tower = nn.Sequential(*layers[:-1])                       # ... -> Linear(2048, 256) -> ReLU
        self.project = nn.Linear(256, CONFIG["feature_dim"])
        with torch.no_grad():                                           # the encoder's own last layer
            self.project.weight.copy_(layers[-1].weight)
            self.project.bias.copy_(layers[-1].bias)
        h1, h2 = CONFIG["hidden"]
        self.body = nn.Sequential(nn.Linear(n_images * CONFIG["feature_dim"] + x_dim, h1), nn.ReLU(),
                                  nn.Linear(h1, h2), nn.ReLU())
        self.e, self.m0, self.m1 = nn.Linear(h2, 1), nn.Linear(h2, 1), nn.Linear(h2, 1)
        self.n_images = n_images

    def forward(self, pixels: torch.Tensor, x: torch.Tensor):
        feats = self.project(self.tower(pixels)).reshape(len(x), -1)
        h = self.body(torch.cat([feats, x], dim=1))
        return self.e(h).squeeze(1), self.m0(h).squeeze(1), self.m1(h).squeeze(1)


def estimate(images, src: np.ndarray, x: np.ndarray, a: np.ndarray, y: np.ndarray, encoder_state: dict,
             fold_seed: int, dev) -> dict:
    """The sealed fit and the sealed AIPW, started from the encoder instead of the reader."""
    sealed = px.PixelNet
    px.PixelNet = EncoderPixelNet
    try:
        out = px.estimate(images, src, x, a, y, encoder_state, fold_seed, dev)
    finally:
        px.PixelNet = sealed
    if "config" in out:
        out["config"] = CONFIG
    return out
