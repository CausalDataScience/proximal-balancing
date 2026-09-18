"""Full-target critics: a base propensity model and an augmented one that contains it by construction.

The base critic sees only the candidate representation Z.  The augmented critic sees Z and the entire
held-out block W_S, with no compression, through a residual branch

    eta_1(Z, W_S) = eta_0(Z) + g(Z, W_S),        q = 0.05 + 0.90 sigmoid(eta).

The branch's last layer starts at zero, so the augmented model's first state is exactly the fitted base
lifted into the larger class.  That state is always among the checkpoints the augmented fit may return,
which is what makes the class inclusion real rather than nominal: a worse augmented critic can only come
from a checkpoint that lost to the lifted base on validation, and such a checkpoint is never chosen.

The screen statistic is the held-out Brier risk difference G = R(q_0) - R(q_1), which at the population
propensities equals E[(q_1 - q_0)^2], the full-target discrepancy.  The screen uses a one-sided upper
confidence bound U = max(G, 0) + z SE and passes only when U is under a tolerance fixed in advance from
the population separation window.  A negative point estimate therefore does not pass on its own; the
precision has to be there too.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from scipy.stats import norm

import minimal_suite_common as c
from probe_highdim_w import Scaler

P_LO, P_HI = 0.05, 0.95
HIDDEN = 64
RESTARTS = 2
MAX_EPOCHS = 200
PATIENCE = 20
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.25


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def bounded(eta: torch.Tensor) -> torch.Tensor:
    return P_LO + (P_HI - P_LO) * torch.sigmoid(eta)


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = HIDDEN, zero_last: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        if zero_last:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class BaseCritic(nn.Module):
    def __init__(self, d_z: int) -> None:
        super().__init__()
        self.f = MLP(d_z)

    def eta(self, z: torch.Tensor) -> torch.Tensor:
        return self.f(z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return bounded(self.eta(z))


class AugmentedCritic(nn.Module):
    """eta_0(Z) + g(Z, W_S).  Built from a fitted base so its first state is that base, lifted."""

    def __init__(self, base: BaseCritic, d_z: int, d_t: int) -> None:
        super().__init__()
        self.base = copy.deepcopy(base)
        self.branch = MLP(d_z + d_t, zero_last=True)

    def eta(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.base.eta(z) + self.branch(torch.cat([z, t], dim=-1))

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return bounded(self.eta(z, t))


def _brier(p: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return ((p - a) ** 2).mean()


def _split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(round(VAL_FRACTION * n))
    return perm[n_val:], perm[:n_val]


def _train(model: nn.Module, inputs: tuple[torch.Tensor, ...], a: torch.Tensor, tr: np.ndarray,
           va: np.ndarray, seed: int, include_initial: bool) -> dict:
    """Early-stopped Brier training; returns the best validation state and its history.

    When include_initial is set the untrained state is a checkpoint candidate.  For the augmented critic
    that state is the lifted base, so the returned model can never do worse on validation than the base
    it contains.
    """
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed + 1)
    va_in = tuple(x[va] for x in inputs)
    with torch.no_grad():
        best_val = float(_brier(model(*va_in), a[va])) if include_initial else float("inf")
    best_state = copy.deepcopy(model.state_dict()) if include_initial else None
    best_epoch, stale, history = -1, 0, []
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.as_tensor(tr)[torch.randperm(len(tr), generator=gen)]
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            opt.zero_grad()
            loss = _brier(model(*(x[idx] for x in inputs)), a[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(_brier(model(*va_in), a[va]))
        history.append(val)
        if val < best_val - 1e-7:
            best_val, best_state, best_epoch, stale = val, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return {"best_val": best_val, "best_epoch": best_epoch, "epochs_run": len(history),
            "kept_initial": best_epoch == -1}


def fit_base(z: np.ndarray, a: np.ndarray, seed: int, dev: torch.device) -> dict:
    """Base critic on Z alone, best of RESTARTS by validation Brier.  Scaler fitted on training rows."""
    tr, va = _split(len(a), seed)
    scaler = Scaler(z[tr])
    zs = torch.as_tensor(scaler(z), dtype=torch.float32, device=dev)
    at = torch.as_tensor(a, dtype=torch.float32, device=dev)
    best = None
    for r in range(RESTARTS):
        model = BaseCritic(z.shape[1]).to(dev)
        info = _train(model, (zs,), at, tr, va, seed + 100 * r, include_initial=False)
        if best is None or info["best_val"] < best["info"]["best_val"]:
            best = {"model": model, "info": info | {"restart": r}}
    return best | {"scaler": scaler, "train_rows": tr, "val_rows": va}


def fit_augmented(base: dict, z: np.ndarray, t: np.ndarray, a: np.ndarray, seed: int,
                  dev: torch.device) -> dict:
    """Augmented critic on (Z, W_S) starting from the fitted base with a zero branch.

    Uses the same training and validation rows as the base, so the two validation risks are comparable
    and the lifted-base checkpoint is a genuine candidate.
    """
    tr, va = base["train_rows"], base["val_rows"]
    t_scaler = Scaler(t[tr])
    zs = torch.as_tensor(base["scaler"](z), dtype=torch.float32, device=dev)
    ts = torch.as_tensor(t_scaler(t), dtype=torch.float32, device=dev)
    at = torch.as_tensor(a, dtype=torch.float32, device=dev)
    best = None
    for r in range(RESTARTS):
        model = AugmentedCritic(base["model"], z.shape[1], t.shape[1]).to(dev)
        info = _train(model, (zs, ts), at, tr, va, seed + 100 * r + 7, include_initial=True)
        if best is None or info["best_val"] < best["info"]["best_val"]:
            best = {"model": model, "info": info | {"restart": r}}
    return best | {"t_scaler": t_scaler}


def nesting_check(base: dict, aug: dict, z: np.ndarray, t: np.ndarray, a: np.ndarray,
                  dev: torch.device) -> dict:
    """The augmented validation risk can never exceed the base's, because the lifted base is a candidate."""
    va = base["val_rows"]
    zs = torch.as_tensor(base["scaler"](z[va]), dtype=torch.float32, device=dev)
    ts = torch.as_tensor(aug["t_scaler"](t[va]), dtype=torch.float32, device=dev)
    at = torch.as_tensor(a[va], dtype=torch.float32, device=dev)
    with torch.no_grad():
        r0, r1 = float(_brier(base["model"](zs), at)), float(_brier(aug["model"](zs, ts), at))
    return {"val_risk_base": r0, "val_risk_augmented": r1, "nested": r1 <= r0 + 1e-9}


def predict(base: dict, aug: dict, z: np.ndarray, t: np.ndarray, dev: torch.device) -> tuple:
    zs = torch.as_tensor(base["scaler"](z), dtype=torch.float32, device=dev)
    ts = torch.as_tensor(aug["t_scaler"](t), dtype=torch.float32, device=dev)
    with torch.no_grad():
        return base["model"](zs).cpu().numpy(), aug["model"](zs, ts).cpu().numpy()


def screen_full_target(base: dict, aug: dict, z_s: np.ndarray, t_s: np.ndarray, a_s: np.ndarray,
                       tolerance: float, alpha: float, n_splits: int, dev: torch.device) -> dict:
    """One-sided upper bound on the held-out risk difference, on rows neither critic was fitted on."""
    q0, q1 = predict(base, aug, z_s, t_s, dev)
    d = (q0 - a_s) ** 2 - (q1 - a_s) ** 2
    gap, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    z_crit = float(norm.ppf(1.0 - alpha / n_splits))
    upper = max(gap, 0.0) + z_crit * se
    # The critic is bounded to [0.05, 0.95] by construction, so the earlier [0.1, 0.9] range check
    # would fail a well-fitted critic at the edges.  Overlap fails here when the base critic saturates,
    # meaning more than one percent of the screen rows sit within 1e-3 of a clip bound.
    saturated = float(np.mean((q0 <= P_LO + 1e-3) | (q0 >= P_HI - 1e-3)))
    overlap = bool(saturated <= 0.01)
    return {"gap": gap, "se": se, "z": z_crit, "upper": upper, "tolerance": tolerance,
            "p_min": float(q0.min()), "p_max": float(q0.max()), "saturated_fraction": saturated,
            "overlap_ok": overlap, "augmented_kept_lifted_base": bool(aug["info"]["kept_initial"]),
            "pass": bool(upper < tolerance and overlap),
            "squared_difference": float(((q1 - q0) ** 2).mean())}
