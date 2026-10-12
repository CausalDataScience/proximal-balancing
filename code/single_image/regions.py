"""Fixed lung-lobe regions on the 64 x 64 grid (plan section 6.1).

The regions and their adjacency are fixed before any data are drawn and are never changed afterwards.

    label 0      background (body and outside the body)
    label 1      right upper lobe   (RUL)    right lung = image left, radiological convention
    label 2      right middle lobe  (RML)
    label 3      right lower lobe   (RLL)
    label 4      left upper lobe    (LUL)
    label 5      left lower lobe    (LLL)    the only region that may carry the device mark

Two lobes are adjacent when they share a pixel edge (4-neighbourhood).  With the fissures below this gives
exactly {1-2, 2-3, 4-5}: the middle lobe separates the right upper and lower lobes, and the mediastinal gap
separates the two lungs.  `ADJACENCY` records the relation and a test checks it against the masks.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

GRID = 64
J = 5
REGION_NAMES = ("RUL", "RML", "RLL", "LUL", "LLL")
DEVICE_REGION = 5                                   # 1-based label of the device-carrying lobe
ADJACENCY = frozenset({(1, 2), (2, 3), (4, 5)})     # unordered pairs, 1-based labels

# geometry, in pixel units with pixel centres at (col + 0.5, row + 0.5)
BODY = dict(cx=32.0, cy=33.0, ax=30.0, ay=28.0)
RIGHT_LUNG = dict(cx=19.5, cy=32.0, ax=11.0, ay=23.0)
LEFT_LUNG = dict(cx=44.5, cy=33.0, ax=10.0, ay=22.0)


def _ellipse(xx, yy, cx, cy, ax, ay):
    return ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1.0


@lru_cache(maxsize=1)
def _label_map() -> np.ndarray:
    yy, xx = np.mgrid[0:GRID, 0:GRID].astype(float) + 0.5
    lab = np.zeros((GRID, GRID), dtype=np.int8)
    right = _ellipse(xx, yy, **RIGHT_LUNG)
    left = _ellipse(xx, yy, **LEFT_LUNG)
    horizontal = 24.0 + 0.15 * (xx - RIGHT_LUNG["cx"])          # RUL above
    oblique_r = 38.0 - 0.35 * (xx - RIGHT_LUNG["cx"])           # RLL below
    oblique_l = 34.0 - 0.40 * (xx - LEFT_LUNG["cx"])            # LLL below
    lab[right & (yy < horizontal)] = 1
    lab[right & (yy >= horizontal) & (yy < oblique_r)] = 2
    lab[right & (yy >= oblique_r)] = 3
    lab[left & (yy < oblique_l)] = 4
    lab[left & (yy >= oblique_l)] = 5
    lab.setflags(write=False)
    return lab


def label_map() -> np.ndarray:
    """(64, 64) int8 map with 0 = background and 1..5 = lobes.  Read-only."""
    return _label_map()


@lru_cache(maxsize=1)
def _masks() -> np.ndarray:
    lab = _label_map()
    m = np.stack([lab == k for k in range(1, J + 1)]).astype(np.float32)
    m.setflags(write=False)
    return m


def masks() -> np.ndarray:
    """(5, 64, 64) float32 indicator masks, region k at index k - 1.  Read-only."""
    return _masks()


@lru_cache(maxsize=1)
def body_mask() -> np.ndarray:
    yy, xx = np.mgrid[0:GRID, 0:GRID].astype(float) + 0.5
    m = _ellipse(xx, yy, **BODY)
    m.setflags(write=False)
    return m


def lung_mask() -> np.ndarray:
    return _label_map() > 0


def region_pixels(k: int) -> np.ndarray:
    """(n_k, 2) integer (row, col) coordinates of region k (1-based)."""
    return np.argwhere(_label_map() == k)


def is_adjacent(k: int, l: int) -> bool:
    return (min(k, l), max(k, l)) in ADJACENCY


def non_adjacent(k: int) -> tuple[int, ...]:
    """N(k): the regions other than k that do not share a boundary with k (1-based)."""
    return tuple(l for l in range(1, J + 1) if l != k and not is_adjacent(k, l))


def computed_adjacency() -> frozenset:
    """Adjacency recomputed from the masks (shared 4-neighbour pixel edge); used only by the tests."""
    lab = _label_map()
    pairs = set()
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        diff = (a != b) & (a > 0) & (b > 0)
        for u, v in zip(a[diff].tolist(), b[diff].tolist()):
            pairs.add((min(u, v), max(u, v)))
    return frozenset(pairs)


@lru_cache(maxsize=None)
def bounding_box(k: int) -> tuple[int, int, int, int]:
    """(row0, row1, col0, col1), half-open, of region k."""
    pix = region_pixels(k)
    return int(pix[:, 0].min()), int(pix[:, 0].max()) + 1, int(pix[:, 1].min()), int(pix[:, 1].max()) + 1


def crop_size() -> tuple[int, int]:
    """A common crop size (multiple of 4) that contains every region's bounding box."""
    h = max(bounding_box(k)[1] - bounding_box(k)[0] for k in range(1, J + 1))
    w = max(bounding_box(k)[3] - bounding_box(k)[2] for k in range(1, J + 1))
    return int(4 * np.ceil(h / 4)), int(4 * np.ceil(w / 4))


def region_crops(images: np.ndarray, k: int) -> np.ndarray:
    """(n, 2, H, W) float32 crops for region k: channel 0 the pixels of region k (other pixels set to 0),
    channel 1 the region-k indicator.  The crop is the region's bounding box, centred in a common H x W
    canvas, so the reader for region k never sees a pixel of any other region."""
    hh, ww = crop_size()
    r0, r1, c0, c1 = bounding_box(k)
    h, w = r1 - r0, c1 - c0
    oy, ox = (hh - h) // 2, (ww - w) // 2
    m = masks()[k - 1, r0:r1, c0:c1]
    out = np.zeros((images.shape[0], 2, hh, ww), dtype=np.float32)
    out[:, 0, oy:oy + h, ox:ox + w] = images[:, r0:r1, c0:c1] * m
    out[:, 1, oy:oy + h, ox:ox + w] = m
    return out


def holdout_view(images: np.ndarray, held_out: tuple[int, ...]) -> np.ndarray:
    """Stage-2 interface: V_S as a (n, 2, 64, 64) float32 array.  Channel 0 is the image with the pixels
    of the held-out regions S set to 0; channel 1 is the visibility indicator (1 = pixel is shown).  The
    same two-channel format is what the masked autoencoder is trained on, so its trunk transfers."""
    vis = np.ones((GRID, GRID), dtype=np.float32)
    for k in held_out:
        vis = vis * (1.0 - masks()[k - 1])
    out = np.empty((images.shape[0], 2, GRID, GRID), dtype=np.float32)
    out[:, 0] = images * vis
    out[:, 1] = vis
    return out
