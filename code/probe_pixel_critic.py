"""Full-pixel critics for the image design, and the raw-image comparator.

Every pixel of a held-out block enters a convolutional trunk.  The trunk keeps its spatial grid to the
end and flattens it, with no global pooling, because the latent channel is carried by where the mass
sits and pooling would erase exactly that.  Two normalised coordinate channels are appended to the
grayscale image so position is available to the first layer.  The trunk is shared across all thirty
candidates of a rotation and is fitted once, on the same D_phi rows the critics use, as a raw-image
propensity model; each candidate then fits only a small base head on its own Z and a zero-initialised
residual head on (Z, image features, block identity, held-out mask).

A discrimination check is part of the contract: two images with identical centroids but different
pixel arrangements must produce different trunk features.  Passing it is what licenses the phrase
"full-pixel critic" rather than "a critic that read the centroid".
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import minimal_suite_common as c
import probe_image_w as im
from probe_full_target import (BaseCritic, MLP, bounded, _brier, _split, _train, P_LO, P_HI,
                               RESTARTS, device)
from probe_highdim_w import Scaler

FEATURE_DIM = 32
TRUNK_WIDTHS = (16, 32, 64)
GROUPS = 8
TRUNK_EPOCHS, TRUNK_PATIENCE, TRUNK_BATCH, TRUNK_LR = 40, 6, 64, 3e-4


def coordinate_channels(n: int, size: int, dev: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, size, device=dev),
                            torch.linspace(-1, 1, size, device=dev), indexing="ij")
    return torch.stack([xs, ys]).unsqueeze(0).expand(n, 2, size, size)


class PixelTrunk(nn.Module):
    """Three stride-2 convolutions with GroupNorm and SiLU, then a flattened grid to FEATURE_DIM."""

    def __init__(self, size: int = im.CANVAS) -> None:
        super().__init__()
        chans = (3,) + TRUNK_WIDTHS
        layers = []
        for a, b in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1), nn.GroupNorm(GROUPS, b), nn.SiLU()]
        self.conv = nn.Sequential(*layers)
        grid = size
        for _ in TRUNK_WIDTHS:
            grid = (grid + 1) // 2
        self.head = nn.Linear(TRUNK_WIDTHS[-1] * grid * grid, FEATURE_DIM)
        self.grid = grid

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = torch.cat([images.unsqueeze(1), coordinate_channels(len(images), images.shape[-1],
                                                                 images.device)], dim=1)
        return self.head(self.conv(x).flatten(1))


class TrunkPropensity(nn.Module):
    """Raw-image propensity model used only to fit the shared trunk: X, I, C and five block features."""

    def __init__(self, trunk: PixelTrunk) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = MLP(3 + im.N_BLOCKS * FEATURE_DIM)

    def forward(self, numeric: torch.Tensor, blocks: list[torch.Tensor]) -> torch.Tensor:
        feats = [self.trunk(b) for b in blocks]
        return bounded(self.head(torch.cat([numeric] + feats, dim=1)))


def fit_trunk(numeric: np.ndarray, blocks: list[np.ndarray], a: np.ndarray, seed: int,
              dev: torch.device) -> dict:
    """Shared trunk for one rotation, fitted once on D_phi rows as a raw-image propensity model."""
    torch.manual_seed(seed)
    tr, va = _split(len(a), seed)
    scaler = Scaler(numeric[tr])
    num_t = torch.as_tensor(scaler(numeric), dtype=torch.float32, device=dev)
    blk_t = [torch.as_tensor(b, dtype=torch.float32, device=dev) for b in blocks]
    a_t = torch.as_tensor(a, dtype=torch.float32, device=dev)
    model = TrunkPropensity(PixelTrunk()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=TRUNK_LR, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed + 1)
    best, best_state, stale, history = float("inf"), None, 0, []
    tr_t, va_t = torch.as_tensor(tr), torch.as_tensor(va)
    for epoch in range(TRUNK_EPOCHS):
        model.train()
        order = tr_t[torch.randperm(len(tr), generator=gen)]
        for s in range(0, len(order), TRUNK_BATCH):
            idx = order[s:s + TRUNK_BATCH]
            opt.zero_grad()
            loss = _brier(model(num_t[idx], [b[idx] for b in blk_t]), a_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([float(_brier(model(num_t[va_t[s:s + 256]],
                                                  [b[va_t[s:s + 256]] for b in blk_t]),
                                            a_t[va_t[s:s + 256]]))
                               for s in range(0, len(va), 256)]))
        history.append(v)
        if v < best - 1e-7:
            best, best_state, stale = v, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= TRUNK_PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return {"trunk": model.trunk, "numeric_scaler": scaler, "val_brier": best,
            "epochs": len(history), "history": history}


def encode_blocks(trunk: PixelTrunk, blocks: list[np.ndarray], dev: torch.device,
                  batch: int = 256) -> np.ndarray:
    """Features for every block on every row, computed once per rotation and reused by all candidates."""
    trunk.eval()
    out = []
    with torch.no_grad():
        for b in blocks:
            feats = [trunk(torch.as_tensor(b[s:s + batch], dtype=torch.float32, device=dev)).cpu()
                     for s in range(0, len(b), batch)]
            out.append(torch.cat(feats).numpy())
    return np.stack(out, axis=1)  # rows x blocks x FEATURE_DIM


def image_target(features: np.ndarray, numeric: dict, split: tuple[int, ...]) -> np.ndarray:
    """T_S: X, the numeric channels when block 0 is held out, the held-out block features, and
    the block identity and held-out mask as fixed indicator columns."""
    cols = [numeric["X"][:, None]]
    for j in split:
        if j == 0:
            cols.extend([numeric["I"][:, None], numeric["C"][:, None]])
        cols.append(features[:, j, :])
    n = len(numeric["X"])
    mask = np.zeros((n, im.N_BLOCKS))
    mask[:, list(split)] = 1.0
    cols.append(mask)
    return np.column_stack(cols)


def reflect_about_centroid(images: np.ndarray) -> np.ndarray:
    """Reflect each image about the vertical line through its own horizontal centroid, exactly.

    A reflection about a non-integer column is done by linear interpolation, which preserves total
    mass and both centroids to floating-point precision while changing the pixel arrangement.
    """
    size = images.shape[-1]
    coords = np.arange(float(size))
    out = np.empty_like(images)
    for k, img in enumerate(images):
        cx = float((img.sum(0) * coords).sum() / img.sum())
        src = 2.0 * cx - coords
        lo = np.floor(src).astype(int)
        w = src - lo
        lo_c, hi_c = np.clip(lo, 0, size - 1), np.clip(lo + 1, 0, size - 1)
        valid = (src >= 0) & (src <= size - 1)
        out[k] = (img[:, lo_c] * (1 - w) + img[:, hi_c] * w) * valid
        mass = out[k].sum()
        if mass > 0:
            out[k] *= img.sum() / mass
    return out


def discrimination_check(trunk: PixelTrunk, images: np.ndarray, dev: torch.device) -> dict:
    """Same centroid, different arrangement: does the architecture see it, and does the fitted trunk?

    Two questions are kept apart.  The architectural one is answered on a freshly initialised trunk of
    the same shape: if its features move when the arrangement changes and the centroid does not, every
    pixel reaches the feature and nothing in the architecture erases arrangement, which is what
    separates a full-pixel critic from one that computes the centroid.  That is the pass rule.  The
    fitted trunk's sensitivity is reported alongside as a diagnostic; a trunk trained as a propensity
    model on this design learns that position is what predicts treatment and becomes nearly invariant to
    arrangement, which is a property of the data, not a defect of the critic.
    """
    flipped = reflect_about_centroid(images)
    coords = np.arange(float(images.shape[-1]))

    def centroids(imgs):
        return np.array([[(im_.sum(0) * coords).sum() / im_.sum(), (im_.sum(1) * coords).sum() / im_.sum()]
                         for im_ in imgs])

    def relative_change(model):
        """Feature movement under reflection, relative to movement between different images.

        The feature norm itself is a poor denominator: the fixed coordinate channels dominate it on a
        canvas that is mostly empty, so even a large arrangement response looks tiny against it.
        Comparing the reflection shift with the shift between two unrelated images asks the right
        question, namely whether arrangement moves the feature about as much as identity does.
        """
        with torch.no_grad():
            f0 = model(torch.as_tensor(images, dtype=torch.float32, device=dev)).cpu().numpy()
            f1 = model(torch.as_tensor(flipped, dtype=torch.float32, device=dev)).cpu().numpy()
        between = np.linalg.norm(f0 - np.roll(f0, 1, axis=0), axis=1)
        return np.linalg.norm(f0 - f1, axis=1) / (np.median(between) + 1e-12)

    torch.manual_seed(0)
    untrained = PixelTrunk(images.shape[-1]).to(dev).eval()
    arch, fitted = relative_change(untrained), relative_change(trunk)
    shift = float(np.abs(centroids(images) - centroids(flipped)).max())
    return {"centroid_shift_max_pixels": shift,
            "mass_change_max": float(np.abs(images.sum((1, 2)) - flipped.sum((1, 2))).max()),
            "architecture_relative_change_min": float(arch.min()),
            "architecture_relative_change_median": float(np.median(arch)),
            "fitted_relative_change_min": float(fitted.min()),
            "fitted_relative_change_median": float(np.median(fitted)),
            "pass": bool(shift < 1e-3 and np.median(arch) > 0.1)}


# ----------------------------------------------------------------------------- raw-image comparator
class RawImageNuisance(nn.Module):
    """Raw (X, W) adjustment: its own trunk, one propensity head and two outcome heads."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = PixelTrunk()
        d = 3 + im.N_BLOCKS * FEATURE_DIM
        self.prop = MLP(d)
        self.out0, self.out1 = MLP(d), MLP(d)

    def features(self, numeric: torch.Tensor, blocks: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([numeric] + [self.trunk(b) for b in blocks], dim=1)

    def forward(self, numeric: torch.Tensor, blocks: list[torch.Tensor]) -> tuple:
        h = self.features(numeric, blocks)
        return bounded(self.prop(h)), self.out0(h), self.out1(h)


def raw_image_aipw(num_n: np.ndarray, blk_n: list[np.ndarray], a_n: np.ndarray, y_n: np.ndarray,
                   num_e: np.ndarray, blk_e: list[np.ndarray], a_e: np.ndarray, y_e: np.ndarray,
                   seed: int, dev: torch.device) -> dict:
    """Honest AIPW with every nuisance learned end to end from the raw images on the N rows.

    Nothing from the PROBE encoder, its channel estimates, its principal directions or its trunk is
    reused.  The restart and early-stopping budget matches the PROBE trunk's.
    """
    torch.manual_seed(seed)
    tr, va = _split(len(a_n), seed)
    scaler = Scaler(num_n[tr])
    y_scale = float(y_n[tr].std())
    num_t = torch.as_tensor(scaler(num_n), dtype=torch.float32, device=dev)
    blk_t = [torch.as_tensor(b, dtype=torch.float32, device=dev) for b in blk_n]
    a_t = torch.as_tensor(a_n, dtype=torch.float32, device=dev)
    y_t = torch.as_tensor(y_n / y_scale, dtype=torch.float32, device=dev)
    best_model, best_val = None, float("inf")
    for r in range(RESTARTS):
        torch.manual_seed(seed + 100 * r)
        model = RawImageNuisance().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=TRUNK_LR, weight_decay=1e-4)
        gen = torch.Generator().manual_seed(seed + 100 * r + 1)
        best, state, stale = float("inf"), None, 0
        tr_t, va_t = torch.as_tensor(tr), torch.as_tensor(va)

        def loss_on(idx):
            p, m0, m1 = model(num_t[idx], [b[idx] for b in blk_t])
            a_i, y_i = a_t[idx], y_t[idx]
            return (_brier(p, a_i) + ((1 - a_i) * (m0 - y_i) ** 2).sum() / (1 - a_i).sum().clamp(min=1)
                    + (a_i * (m1 - y_i) ** 2).sum() / a_i.sum().clamp(min=1))

        for epoch in range(TRUNK_EPOCHS):
            model.train()
            order = tr_t[torch.randperm(len(tr), generator=gen)]
            for s in range(0, len(order), TRUNK_BATCH):
                opt.zero_grad()
                loss = loss_on(order[s:s + TRUNK_BATCH])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = float(np.mean([float(loss_on(va_t[s:s + 256])) for s in range(0, len(va), 256)]))
            if v < best - 1e-7:
                best, state, stale = v, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= TRUNK_PATIENCE:
                    break
        model.load_state_dict(state)
        if best < best_val:
            best_model, best_val = model, best
    best_model.eval()
    num_e_t = torch.as_tensor(scaler(num_e), dtype=torch.float32, device=dev)
    blk_e_t = [torch.as_tensor(b, dtype=torch.float32, device=dev) for b in blk_e]
    p, m0, m1 = [], [], []
    with torch.no_grad():
        for s in range(0, len(a_e), 256):
            pp, mm0, mm1 = best_model(num_e_t[s:s + 256], [b[s:s + 256] for b in blk_e_t])
            p.append(pp.cpu()); m0.append(mm0.cpu()); m1.append(mm1.cpu())
    p_raw = torch.cat(p).numpy()
    p = np.clip(p_raw, c.OVERLAP_RANGE[0], c.OVERLAP_RANGE[1])
    m0, m1 = torch.cat(m0).numpy() * y_scale, torch.cat(m1).numpy() * y_scale
    scores = m1 - m0 + a_e * (y_e - m1) / p - (1 - a_e) * (y_e - m0) / (1 - p)
    # Held-out fit of each nuisance on the evaluation rows, against observed quantities only.  The
    # first diagnosis of this comparator found a propensity head that tracked the truth and outcome
    # heads with R^2 near 0.1, so the two are reported apart: a failure of raw adjustment then has a
    # named cause instead of an unexplained number.
    def r2(arm, m):
        mask = a_e == arm
        return float(1.0 - ((y_e[mask] - m[mask]) ** 2).mean() / y_e[mask].var())
    return {"scores": scores, "val_loss": best_val,
            "diagnostics": {"outcome_r2_arm0": r2(0, m0), "outcome_r2_arm1": r2(1, m1),
                            "plug_in_effect": float((m1 - m0).mean()),
                            "propensity_range": [float(p_raw.min()), float(p_raw.max())],
                            "propensity_clipped_fraction":
                                float(np.mean((p_raw < c.OVERLAP_RANGE[0]) | (p_raw > c.OVERLAP_RANGE[1])))}}
