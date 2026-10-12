"""Fixed per-region summaries chi_l (plan section 6.3).

Sixteen numbers per region, computed from the region's own pixels only:

    0        mean intensity
    1        variance
    2..9     8 quantiles at linspace(0.1, 0.9, 8) = 0.1, 0.214, ..., 0.9
    10..13   fraction of pixels above 4 thresholds; the thresholds are the region's pooled pixel quantiles
             at 0.50, 0.75, 0.90, 0.97 on fold R
    14..15   2 fixed random projections of the region's pixels (Gaussian directions, fixed seed, unit norm)

`fit` reads fold-R images only, sets the thresholds and the per-coordinate centring and scale, and freezes
the object: a second `fit` raises, and `transform` never changes a parameter.  Nothing here is learned
from A or Y, and nothing is learned after `fit`.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from . import regions as rg

D_CHI = 16
QUANTILES = np.linspace(0.1, 0.9, 8)
THRESHOLD_LEVELS = (0.50, 0.75, 0.90, 0.97)
PROJECTION_SEED = 20_261_008
CHUNK = 4000


def _projections() -> list[np.ndarray]:
    rng = np.random.default_rng(PROJECTION_SEED)
    out = []
    for k in range(1, rg.J + 1):
        v = rng.standard_normal((2, len(rg.region_pixels(k))))
        out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    return out


class FixedSummaries:
    def __init__(self):
        self._proj = _projections()
        self._idx = [rg.label_map().ravel() == k for k in range(1, rg.J + 1)]
        self.thresholds: np.ndarray | None = None       # (5, 4)
        self.center: np.ndarray | None = None           # (5, 16)
        self.scale: np.ndarray | None = None            # (5, 16)
        self.n_fit: int | None = None

    @property
    def frozen(self) -> bool:
        return self.thresholds is not None

    def _region_pixels(self, images: np.ndarray, k: int) -> np.ndarray:
        return images.reshape(images.shape[0], -1)[:, self._idx[k - 1]].astype(np.float64)

    def _raw_region(self, px: np.ndarray, k: int, thresholds: np.ndarray) -> np.ndarray:
        q = np.quantile(px, QUANTILES, axis=1).T
        above = (px[:, None, :] > thresholds[k - 1][None, :, None]).mean(axis=2)
        proj = px @ self._proj[k - 1].T
        return np.column_stack([px.mean(1), px.var(1), q, above, proj])

    def _raw(self, images: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        out = np.empty((images.shape[0], rg.J, D_CHI))
        for c0 in range(0, images.shape[0], CHUNK):
            block = images[c0:c0 + CHUNK]
            for k in range(1, rg.J + 1):
                out[c0:c0 + CHUNK, k - 1] = self._raw_region(self._region_pixels(block, k), k, thresholds)
        return out

    def fit(self, images_r: np.ndarray) -> "FixedSummaries":
        if self.frozen:
            raise RuntimeError("FixedSummaries is frozen; it is fitted once on fold R and never refitted")
        if images_r.ndim != 3:
            raise TypeError("fit takes a (n, 64, 64) image array and nothing else")
        thr = np.stack([np.quantile(self._region_pixels(images_r, k), THRESHOLD_LEVELS)
                        for k in range(1, rg.J + 1)])
        raw = self._raw(images_r, thr)
        center, scale = raw.mean(0), raw.std(0) + 1e-8
        for name, arr in (("thr", thr), ("center", center), ("scale", scale)):
            arr.setflags(write=False)
        self.thresholds, self.center, self.scale, self.n_fit = thr, center, scale, int(images_r.shape[0])
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        """(n, 5, 16) standardised summaries with the frozen fold-R parameters."""
        if not self.frozen:
            raise RuntimeError("fit on fold R first")
        return ((self._raw(images, self.thresholds) - self.center) / self.scale).astype(np.float32)

    def state(self) -> dict:
        return {"thresholds": self.thresholds.tolist(), "center": self.center.tolist(),
                "scale": self.scale.tolist(), "n_fit": self.n_fit, "quantiles": QUANTILES.tolist(),
                "threshold_levels": list(THRESHOLD_LEVELS), "projection_seed": PROJECTION_SEED}

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.state(), sort_keys=True).encode()).hexdigest()[:16]
