"""Evaluator-only: the causal target a learned representation actually aims at.

    theta_S(phi_hat) = E[ E(Y | A=1, Z_hat) - E(Y | A=0, Z_hat) ],   Z_hat = phi_hat_S(X, W_-S).

This is the quantity the manuscript's identification argument is about, and it is what the returned
effect converges to if nuisance estimation is consistent.  It is computed on a fresh sample far larger
than any training fold, so it isolates the representation's error from estimation noise.  The learned
map itself, its scalers and its coefficient are frozen inputs; nothing here is fitted on rows the
pipeline saw.

Two estimates are returned.  A two-fold cross-fitted MLP regression of Y on (A, Z_hat) is the primary
one, because E(Y | A, Z_hat) need not be linear when A enters through a probit link.  A per-arm linear
regression is the cross-check; the two agree closely on the Gaussian designs here.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from probe_highdim_w import Scaler

HIDDEN, MAX_EPOCHS, PATIENCE, BATCH, LR = 64, 60, 8, 1024, 2e-3


class Regressor(nn.Module):
    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, HIDDEN), nn.SiLU(),
                                 nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _fit(x: torch.Tensor, y: torch.Tensor, seed: int) -> Regressor:
    torch.manual_seed(seed)
    n = len(y)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 1))
    va, tr = perm[: n // 5], perm[n // 5:]
    model = Regressor(x.shape[1]).to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, stale = float("inf"), None, 0
    gen = torch.Generator().manual_seed(seed + 2)
    for _ in range(MAX_EPOCHS):
        model.train()
        order = tr[torch.randperm(len(tr), generator=gen)]
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            opt.zero_grad()
            loss = ((model(x[idx]) - y[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(((model(x[va]) - y[va]) ** 2).mean())
        if v < best - 1e-7:
            best, best_state, stale = v, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def learned_target(z: np.ndarray, a: np.ndarray, y: np.ndarray, seed: int,
                   dev: torch.device) -> dict:
    """theta for a frozen representation, from a large evaluator sample (z, a, y)."""
    n = len(a)
    scaler = Scaler(z)
    zs = torch.as_tensor(scaler(z), dtype=torch.float32, device=dev)
    at = torch.as_tensor(a, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    halves = np.array_split(np.random.default_rng(seed).permutation(n), 2)
    plug = np.zeros(n)
    for k in range(2):
        fit_idx, eval_idx = halves[1 - k], halves[k]
        x_fit = torch.cat([at[fit_idx, None], zs[fit_idx]], dim=1)
        model = _fit(x_fit, yt[fit_idx], seed + 10 * k)
        with torch.no_grad():
            z_e = zs[eval_idx]
            ones, zeros = torch.ones(len(eval_idx), 1, device=dev), torch.zeros(len(eval_idx), 1, device=dev)
            mu1 = model(torch.cat([ones, z_e], dim=1)).cpu().numpy()
            mu0 = model(torch.cat([zeros, z_e], dim=1)).cpu().numpy()
        plug[eval_idx] = mu1 - mu0
    # linear cross-check: separate regressions per arm, then averaged over the whole sample
    design = np.column_stack([np.ones(n), scaler(z)])
    coef = {arm: np.linalg.lstsq(design[a == arm], y[a == arm], rcond=None)[0] for arm in (0, 1)}
    linear = float((design @ coef[1] - design @ coef[0]).mean())
    return {"theta": float(plug.mean()), "theta_linear": linear, "n": n,
            "theta_se": float(plug.std(ddof=1) / np.sqrt(n))}
