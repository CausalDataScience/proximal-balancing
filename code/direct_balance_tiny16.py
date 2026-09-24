"""Tiny16 direct image balancing (PRD "Tiny16 Direct Image Balancing Feasibility", 2026-09-21 night).

The observed proxy is redefined as W^16 = R_16(W^64): every Shapes3D image is shrunk from 64 x 64 to 16 x 16 by
area averaging, once, deterministically, with no crop and no augmentation.  The learner then sees only
(X, retained W^16, held-out W^16, A).  The CNN starts at random and every weight moves to lower the residual
Brier discrepancy and nothing else: no label, no reconstruction, no outcome.

Two things differ from the earlier runs in `direct_balance_net`.
  1. The gap that drives the CNN is CROSS-FITTED inside the representation fold: three folds, critics fitted on
     two, the signed gap and the gradient taken on the third, rotated, averaged, clipped once.  A critic's memory
     of its own rows never reaches the gradient.  This is an operational surrogate for the population D_res, not
     the manuscript's same-sample empirical objective.
  2. The tower for a 16 x 16 input is two stride-two convolutions (3->16->32) and a linear map to the feature
     width; `direct_balance_net.ImageTower` builds exactly that from the spec below.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

import direct_balance_data as db
import direct_balance_net as dn

SIDE, FACTOR = 16, 4
SPEC16 = {"channels": 3, "size": SIDE}
FOLDS = 3
CHECKPOINTS = (0, 5, 10, 20, 40)


# ----------------------------------------------------------------------------- the fixed resize

def resize16(pixels_u8: torch.Tensor) -> torch.Tensor:
    """uint8 (n, 64, 64, 3) -> float32 (n, 3, 16, 16) in [0, 1]: area averaging over 4 x 4 blocks.

    The same input gives the same output, byte for byte, because the average of sixteen fixed numbers is a fixed
    number; no random crop, no interpolation choice, nothing that depends on the batch it sits in.
    """
    x = dn.to_input(pixels_u8)                                   # (n, 3, 64, 64), contiguous
    return nn.functional.avg_pool2d(x, FACTOR).contiguous()


class TinyBatches(dn.Batches):
    """Every image column served at 16 x 16, normalised per channel with statistics from the rows named.

    The resized images of every bank row the data set uses are computed once and kept on the device; a batch is
    a gather.  For the original-resolution audit, `held64` names the columns to serve at 64 x 64 instead, from
    the parent class's own cache, untouched by the normalisation.
    """

    def __init__(self, images, src: np.ndarray, x: np.ndarray, a: np.ndarray, dev, standardise_rows: np.ndarray,
                 held64: np.ndarray | None = None) -> None:
        super().__init__(images, src, x, a, dev, cache_rows=np.arange(len(src)), standardise_rows=standardise_rows)
        wanted = self.cache_index                                   # every bank row this data set draws
        small = []
        with torch.no_grad():
            for s in range(0, len(wanted), 4096):
                small.append(resize16(self.cache[s:s + 4096]))
        self.small = torch.cat(small)                              # (n_unique, 3, 16, 16) in [0, 1]
        fit_rows = np.asarray(standardise_rows)
        at = torch.as_tensor(np.searchsorted(self.cache_index, self.src[fit_rows].ravel()), device=dev)
        ref = self.small[at]
        self.channel_mean = ref.mean(dim=(0, 2, 3))
        self.channel_sd = ref.std(dim=(0, 2, 3)) + 1e-6
        self.small = ((self.small - self.channel_mean[None, :, None, None]) / self.channel_sd[None, :, None, None]).contiguous()
        self.held64 = set() if held64 is None else set(int(c) for c in held64)

    def pixels(self, rows: np.ndarray, columns: np.ndarray) -> list[torch.Tensor]:
        out = []
        for c in columns:
            if int(c) in self.held64:
                out.append(dn.to_input(self._gather(self.src[rows, c])))
            else:
                at = torch.as_tensor(np.searchsorted(self.cache_index, self.src[rows, c]), device=self.dev)
                out.append(self.small[at])
        return out

    def report(self) -> dict:
        return {"side": SIDE, "factor": FACTOR, "mode": "area (avg_pool2d 4x4 on [0,1] pixels)",
                "unique_images": int(self.cache_index.size), "channel_mean": self.channel_mean.tolist(),
                "channel_sd": self.channel_sd.tolist(), "held_columns_at_64": sorted(self.held64)}


# ----------------------------------------------------------------------------- the cross-fitted training

def train_split_kfold(data: TinyBatches, rows: dict, split, seed: int, x_dim: int, folds: int = FOLDS,
                      checkpoints=CHECKPOINTS, max_seconds: float | None = None, log=None,
                      validated: bool = False) -> dict:
    """PRD section 7: critics fitted on all folds but one, the signed gap and gradient on the one left out,
    rotated over the folds, averaged, clipped once, one CNN step when the average is positive.

    Every checkpoint's weights are returned so that a fresh device can choose among them (PRD section 8)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    held, kept = db.image_columns(split)
    dev = data.dev
    rep = dn.Representation(kept, x_dim, spec=SPEC16).to(dev)
    init_state = {k: v.detach().cpu().clone() for k, v in rep.state_dict().items()}
    opt = torch.optim.Adam(rep.parameters(), lr=dn.TRAIN["lr_cnn"])
    fold = np.asarray(rows["represent"])
    order = np.random.default_rng(seed + 17).permutation(len(fold))
    parts = [np.sort(fold[order[i::folds]]) for i in range(folds)]
    fit_sets = [np.sort(np.concatenate([p for j, p in enumerate(parts) if j != i])) for i in range(folds)]
    score_sets = parts
    states = [None] * folds
    t0 = time.time()

    fitter = dn.fit_critics_validated if validated else dn.fit_critics
    budget = (dn.CRITIC_V["initial_steps"], dn.CRITIC_V["steps_per_iter"]) if validated \
        else (dn.TRAIN["critic_initial"], dn.TRAIN["critic_per_iter"])

    def refit(steps: int) -> list:
        pairs = []
        for i, fit_rows in enumerate(fit_sets):
            base, aug, states[i] = fitter(rep, data, fit_rows, kept, held, steps, rng, states[i], spec=SPEC16)
            pairs.append((base, aug))
        return pairs

    pairs = refit(budget[0])
    history, saved = [], {}

    def take(iteration: int) -> None:
        if iteration in checkpoints:
            on_select = [dn.signed_gap(rep, base, aug, data, rows["select"], kept, held) for base, aug in pairs]
            pooled = {k: float(np.mean([g_[k] for g_ in on_select])) for k in ("risk_base", "risk_augmented", "signed_gap")}
            saved[iteration] = {"state": {k: v.detach().cpu().clone() for k, v in rep.state_dict().items()},
                                "select_training_critics": pooled, "fit": [st["info"] for st in states]}

    take(0)
    n_iters = max(checkpoints)
    for it in range(1, n_iters + 1):
        if max_seconds is not None and time.time() - t0 > max_seconds:
            raise TimeoutError(f"training passed its {max_seconds:.0f} s cap at iteration {it}")
        groups = [(score, base, aug) for score, (base, aug) in zip(score_sets, pairs)]
        step = dn.cnn_step(rep, groups, data, kept, held, opt)      # pooled over the three left-out folds
        pairs = refit(budget[1])
        history.append(dict(step, iteration=it, seconds=round(time.time() - t0, 1),
                            critics=[st["info"] for st in states]))
        take(it)
        if log:
            log(f"    iter {it:2d}: cross-fitted gap {step['signed_gap']:+.5f} stepped {step['stepped']} "
                f"grad {step['grad_norm']:.2e} ({time.time() - t0:.0f} s)")
    return {"split": db.split_name(split), "kept_images": kept.tolist(), "held_images": held.tolist(),
            "folds": folds, "fold_sizes": [len(p) for p in parts], "initial_state": init_state,
            "critics": "validated (early-stopped, lifted base decided on the validation fifth)" if validated
                       else "plain minibatch (the diagnosed failure mode)", "critic_budget": list(budget),
            "checkpoint_states": {k: v["state"] for k, v in saved.items()},
            "checkpoint_select_training_critics": {k: v["select_training_critics"] for k, v in saved.items()},
            "checkpoint_fit_risks": {k: v["fit"] for k, v in saved.items()},
            "history": history, "seconds": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- A1: does 16 x 16 keep the angle?

class Reader16(nn.Module):
    """Diagnosis only: a small regressor of the orientation level from a 16 x 16 image.  It never touches the
    main learner; it exists to show the resize did not destroy the information."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(64 * 16, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def information_check(images, train_rows: np.ndarray, test_rows: np.ndarray, levels_train: np.ndarray,
                      levels_test: np.ndarray, dev, seed: int, epochs: int = 6, batch: int = 512,
                      max_seconds: float = 300.0) -> dict:
    """Fit the reader on `train_rows`, score on `test_rows` (disjoint bank splits, both disjoint from the causal
    bank).  R^2 of the level, the share within one level, the share two or more levels off."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    def small(rows):
        out = []
        with torch.no_grad():
            for s in range(0, len(rows), 4096):
                out.append(resize16(torch.as_tensor(np.asarray(images[np.sort(rows[s:s + 4096])]), device=dev)))
        return torch.cat(out)

    tr_rows, te_rows = np.sort(train_rows), np.sort(test_rows)
    xtr, xte = small(tr_rows), small(te_rows)
    mean, sd = xtr.mean(dim=(0, 2, 3)), xtr.std(dim=(0, 2, 3)) + 1e-6
    xtr = (xtr - mean[None, :, None, None]) / sd[None, :, None, None]
    xte = (xte - mean[None, :, None, None]) / sd[None, :, None, None]
    ytr = torch.as_tensor(levels_train, dtype=torch.float32, device=dev)
    net = Reader16().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    t0, hist = time.time(), []
    for ep in range(epochs):
        perm = rng.permutation(len(tr_rows))
        net.train()
        tot = 0.0
        for s in range(0, len(perm), batch):
            if time.time() - t0 > max_seconds:
                raise TimeoutError("the information check passed its time cap")
            idx = torch.as_tensor(perm[s:s + batch], device=dev)
            opt.zero_grad(set_to_none=True)
            loss = ((net(xtr[idx]) - ytr[idx]) ** 2).mean()
            loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        hist.append(round(tot / len(perm), 4))
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(xte[s:s + 4096]) for s in range(0, len(te_rows), 4096)]).cpu().numpy().astype(np.float64)
    truth = levels_test.astype(np.float64)
    err = pred - truth
    hard = np.clip(np.rint(pred), 0, 14)
    return {"train_images": int(len(tr_rows)), "test_images": int(len(te_rows)), "epochs": epochs,
            "train_mse_by_epoch": hist, "r2": float(1 - err.var() / truth.var()),
            "within_one_level": float(np.mean(np.abs(hard - truth) <= 1)),
            "two_or_more_off": float(np.mean(np.abs(hard - truth) >= 2)),
            "rms_levels": float(np.sqrt(np.mean(err ** 2))), "seconds": round(time.time() - t0, 1)}
