"""Networks and fitting routines for the Sim5 CNN run (PRD section 6 and 9, operational arm).

Engineering choices fixed before any dev result, recorded in the dev protocol:
- One mask-conditioned warm start is shared by all sampled splits.  It is trained on D with the sampled
  retained masks and their complements, so the frozen trunk reads either side of any sampled split.
- The trunk is frozen after the warm start.  Balance learning updates the per-split head that maps trunk
  features of the retained pixels, together with X, to a continuous code z.  The representation is Z = (X, z).
- The code is continuous (16 numbers).  The PRD's 256-way hard code served the literal-theorem arm, which
  the user moved out of scope on 2026-09-18.
- Critics read Z; the augmented critic also reads the frozen trunk's features of the held-out pixels.
A unit's Z never reads its held-out pixels: the retained mask zeroes them before the trunk.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_DIM = 16
X_DIM = 4
FEAT = 1024
ETA = 0.05
BATCH = 256


def bounded(logit: torch.Tensor) -> torch.Tensor:
    return ETA + (1 - 2 * ETA) * torch.sigmoid(logit)


def mlp(sizes: list[int]) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class Trunk(nn.Module):
    """PRD 6.1: input (masked grayscale, retained mask), output 1024 features."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())

    def forward(self, w: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        k = keep.to(w.dtype)
        if k.dim() == 2:
            k = k.expand_as(w)
        return self.net(torch.stack([w * k, k], dim=1))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp([FEAT + X_DIM, 128, CODE_DIM])

    def forward(self, f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([f, x], dim=1))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(CODE_DIM, 64 * 7 * 7)
        self.up = nn.Sequential(nn.SiLU(), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
                                nn.ConvTranspose2d(32, 1, 4, 2, 1))

    def forward(self, z):
        return torch.sigmoid(self.up(self.fc(z).view(-1, 64, 7, 7))).squeeze(1)


class WarmModel(nn.Module):
    """PRD 6.2 label-free warm start: masked reconstruction + A (BCE) + standardized Y (MSE), weights 1."""

    def __init__(self):
        super().__init__()
        self.trunk, self.head, self.dec = Trunk(), Head(), Decoder()
        self.a_head = mlp([X_DIM + CODE_DIM, 64, 1])
        self.y_head = mlp([X_DIM + CODE_DIM + 1, 64, 1])

    def loss(self, w, keep, x, a, y_std):
        z = self.head(self.trunk(w, keep), x)
        zx = torch.cat([x, z], 1)
        recon = F.mse_loss(self.dec(z), w)
        la = F.binary_cross_entropy_with_logits(self.a_head(zx).squeeze(1), a)
        ly = F.mse_loss(self.y_head(torch.cat([zx, a[:, None]], 1)).squeeze(1), y_std)
        return recon + la + ly


def _seed(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def warm_start(w, x, a, y, masks, fit, val, seed, max_epochs=30, patience=5, lr=1e-3, wd=1e-4, ledger=None):
    """Train the shared warm model on D.  Each fit row draws a random mask from `masks` every epoch;
    validation rows keep one fixed mask each so the stopping criterion is not noisy."""
    gen = _seed(seed)
    model = WarmModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    mu, sd = y[fit].mean(), y[fit].std()
    y_std = (y - mu) / sd
    val_mask = masks[torch.randint(len(masks), (len(val),), generator=gen)]
    best, best_state, bad, hist = float("inf"), None, 0, []
    for epoch in range(max_epochs):
        model.train()
        order = fit[torch.randperm(len(fit), generator=gen)]
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            mk = masks[torch.randint(len(masks), (len(idx),), generator=gen)]
            opt.zero_grad()
            model.loss(w[idx], mk, x[idx], a[idx], y_std[idx]).backward()
            opt.step()
        if ledger is not None:
            ledger["cnn_backprop_images"] += len(fit)
        model.eval()
        with torch.no_grad():
            v = float(sum(model.loss(w[val[b:b + BATCH]], val_mask[b:b + BATCH], x[val[b:b + BATCH]],
                                     a[val[b:b + BATCH]], y_std[val[b:b + BATCH]]) * len(val[b:b + BATCH])
                          for b in range(0, len(val), BATCH)) / len(val))
        if ledger is not None:
            ledger["cnn_forward_images"] += len(val)
        hist.append(v)
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs": len(hist), "best_val_loss": best, "val_history": hist}


@torch.no_grad()
def trunk_features(trunk: Trunk, w: torch.Tensor, keep: torch.Tensor, ledger=None) -> torch.Tensor:
    out = torch.cat([trunk(w[b:b + 1024], keep) for b in range(0, len(w), 1024)])
    if ledger is not None:
        ledger["cnn_forward_images"] += len(w)
    return out


class Standardize(nn.Module):
    def __init__(self, data: torch.Tensor):
        super().__init__()
        self.register_buffer("mu", data.mean(0).detach())
        self.register_buffer("sd", data.std(0).detach().clamp_min(1e-6))

    def forward(self, v):
        return (v - self.mu) / self.sd


class BaseCritic(nn.Module):
    def __init__(self, z_fit: torch.Tensor):
        super().__init__()
        self.std = Standardize(z_fit)
        self.net = mlp([z_fit.shape[1], 128, 64, 1])

    def forward(self, z, h=None):
        return bounded(self.net(self.std(z)).squeeze(1))


class AugCritic(nn.Module):
    """Base branch on Z plus a residual branch on (Z, held-out features); the residual output layer starts
    at zero, so a lifted start reproduces the base critic exactly."""

    def __init__(self, z_fit: torch.Tensor, h_fit: torch.Tensor):
        super().__init__()
        self.std_z, self.std_h = Standardize(z_fit), Standardize(h_fit)
        self.base = mlp([z_fit.shape[1], 128, 64, 1])
        self.res = mlp([z_fit.shape[1] + h_fit.shape[1], 128, 64, 1])
        nn.init.zeros_(self.res[-1].weight); nn.init.zeros_(self.res[-1].bias)

    def lift(self, base: BaseCritic):
        self.std_z.load_state_dict(base.std.state_dict())
        self.base.load_state_dict(base.net.state_dict())
        nn.init.zeros_(self.res[-1].weight); nn.init.zeros_(self.res[-1].bias)
        return self

    def forward(self, z, h):
        zs = self.std_z(z)
        return bounded((self.base(zs) + self.res(torch.cat([zs, self.std_h(h)], 1))).squeeze(1))


def brier(q, a):
    return ((q - a) ** 2).mean()


def fit_critic(critic, z, h, a, fit, val, seed, max_epochs=40, patience=5, lr=1e-3, wd=1e-4):
    gen = _seed(seed)
    opt = torch.optim.Adam(critic.parameters(), lr=lr, weight_decay=wd)

    def risk(rows):
        with torch.no_grad():
            return float(brier(critic(z[rows], None if h is None else h[rows]), a[rows]))

    best, best_state, bad, epochs = risk(val), copy.deepcopy(critic.state_dict()), 0, 0
    for epoch in range(max_epochs):
        order = fit[torch.randperm(len(fit), generator=gen)]
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            opt.zero_grad()
            brier(critic(z[idx], None if h is None else h[idx]), a[idx]).backward()
            opt.step()
        epochs += 1
        v = risk(val)
        if v < best - 1e-6:
            best, best_state, bad = v, copy.deepcopy(critic.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    critic.load_state_dict(best_state)
    return critic, {"epochs": epochs, "val_brier": best, "fit_brier": risk(fit)}


def critic_gap(z, h, a, fit, val, seed) -> dict:
    """Fresh nested critics on D_fit, early stopped on D_val; risks and the paired gap on D_val.

    The augmented critic is the better, by D_fit Brier, of a trained lifted start, a trained fresh start and the
    unoptimized lifted base.  Both risks are computed on the same D_val rows."""
    z, h = z.detach(), h.detach()
    q0, info0 = fit_critic(BaseCritic(z[fit]), z, None, a, fit, val, seed)
    starts = []
    lifted, i1 = fit_critic(AugCritic(z[fit], h[fit]).lift(q0), z, h, a, fit, val, seed + 1)
    starts.append(("lifted", lifted, i1["fit_brier"]))
    fresh, i2 = fit_critic(AugCritic(z[fit], h[fit]), z, h, a, fit, val, seed + 2)
    starts.append(("fresh", fresh, i2["fit_brier"]))
    fallback = AugCritic(z[fit], h[fit]).lift(q0)
    with torch.no_grad():
        starts.append(("unoptimized_lift", fallback, float(brier(fallback(z[fit], h[fit]), a[fit]))))
    name, q1, _ = min(starts, key=lambda s: s[2])
    with torch.no_grad():
        p0, p1 = q0(z[val]), q1(z[val], h[val])
        d = (a[val] - p0) ** 2 - (a[val] - p1) ** 2
    gap = float(d.mean())
    return {"gap": gap, "clipped": max(gap, 0.0), "se": float(d.std() / np.sqrt(len(val))),
            "risk_base": float(((a[val] - p0) ** 2).mean()), "risk_aug": float(((a[val] - p1) ** 2).mean()),
            "aug_start": name, "base_epochs": info0["epochs"],
            "p_base_min": float(p0.min()), "p_base_max": float(p0.max()), "critics": (q0, q1)}


def balance_head(head0: Head, f_fit_keep, x_fit_rows, h, a, x, fit, val, seed, updates=100, refresh=5,
                 refresh_epochs=3, lr=1e-3, checkpoints=(0, 25, 50, 75, 100)):
    """Full-batch head updates on D_fit against critics refreshed every `refresh` updates.
    Each checkpoint is scored with fresh critics (critic_gap); the smallest D_val gap wins, ties to the earlier.

    f_fit_keep: trunk features of the retained pixels for all D rows (indexed by fit/val), x: X for all D rows."""
    head = copy.deepcopy(head0)
    for p in head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    def z_of(hd):
        return torch.cat([x, hd(f_fit_keep, x)], 1)

    with torch.no_grad():
        z0 = z_of(head)
    q0 = fit_critic(BaseCritic(z0[fit]), z0, None, a, fit, val, seed + 10)[0]
    q1 = fit_critic(AugCritic(z0[fit], h[fit]).lift(q0), z0, h, a, fit, val, seed + 11)[0]
    saved, trace, zero_grad = {}, [], 0
    w0 = torch.cat([p.detach().flatten() for p in head0.parameters()])
    for t in range(updates + 1):
        if t in checkpoints:
            saved[t] = copy.deepcopy(head.state_dict())
        if t == updates:
            break
        if t > 0 and t % refresh == 0:
            with torch.no_grad():
                zc = z_of(head)
            q0 = fit_critic(q0, zc, None, a, fit, val, seed + 100 + t, max_epochs=refresh_epochs, patience=refresh_epochs)[0]
            q1 = fit_critic(q1, zc, h, a, fit, val, seed + 200 + t, max_epochs=refresh_epochs, patience=refresh_epochs)[0]
        for p in list(q0.parameters()) + list(q1.parameters()):
            p.requires_grad_(False)
        z = z_of(head)
        gap = brier(q0(z[fit]), a[fit]) - brier(q1(z[fit], h[fit]), a[fit])
        loss = torch.clamp(gap, min=0.0)
        opt.zero_grad()
        if loss.item() > 0:
            loss.backward()
            opt.step()
        else:
            zero_grad += 1
        for p in list(q0.parameters()) + list(q1.parameters()):
            p.requires_grad_(True)
        trace.append(gap.item())
    scored = []
    for t in checkpoints:
        hd = copy.deepcopy(head0); hd.load_state_dict(saved[t])
        with torch.no_grad():
            zc = z_of(hd)
        g = critic_gap(zc, h, a, fit, val, seed + 1000 + t)
        wt = torch.cat([p.detach().flatten() for p in hd.parameters()])
        scored.append({"update": t, "gap": g["gap"], "clipped": g["clipped"], "se": g["se"],
                       "aug_start": g["aug_start"], "p_base_min": g["p_base_min"], "p_base_max": g["p_base_max"],
                       "weight_change": float((wt - w0).norm() / w0.norm())})
    best = min(scored, key=lambda s: (round(s["clipped"], 6), s["update"]))
    final = copy.deepcopy(head0); final.load_state_dict(saved[best["update"]])
    return final, {"checkpoints": scored, "selected_update": best["update"], "fit_gap_trace": trace,
                   "zero_gradient_updates": zero_grad}


class Nuisance(nn.Module):
    def __init__(self, z_fit, out):
        super().__init__()
        self.std = Standardize(z_fit)
        self.net = mlp([z_fit.shape[1], 128, 64, out])

    def forward(self, z):
        return self.net(self.std(z))


def fit_nuisance(z, a, y, fit, val, seed, max_epochs=60, patience=8, lr=1e-3, wd=1e-4):
    """Propensity (BCE, output mapped to [ETA, 1-ETA]) and a two-arm outcome net on the observed arm (MSE)."""
    gen = _seed(seed)
    ymu, ysd = y[fit].mean(), y[fit].std()
    ys = (y - ymu) / ysd
    prop, out = Nuisance(z[fit], 1), Nuisance(z[fit], 2)

    def prop_loss(rows):
        return F.binary_cross_entropy(bounded(prop(z[rows]).squeeze(1)), a[rows])

    def out_loss(rows):
        o = out(z[rows])
        pred = torch.where(a[rows] > 0.5, o[:, 1], o[:, 0])
        return F.mse_loss(pred, ys[rows])

    for net, lossf in ((prop, prop_loss), (out, out_loss)):
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        with torch.no_grad():
            best = float(lossf(val))
        best_state, bad = copy.deepcopy(net.state_dict()), 0
        for epoch in range(max_epochs):
            order = fit[torch.randperm(len(fit), generator=gen)]
            for b in range(0, len(order), BATCH):
                opt.zero_grad(); lossf(order[b:b + BATCH]).backward(); opt.step()
            with torch.no_grad():
                v = float(lossf(val))
            if v < best - 1e-6:
                best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(best_state)

    def predict(zz):
        with torch.no_grad():
            e = bounded(prop(zz).squeeze(1)).numpy()
            raw = torch.sigmoid(prop(zz).squeeze(1)).numpy()
            o = out(zz).numpy() * float(ysd) + float(ymu)
        return e, o[:, 0], o[:, 1], raw
    return predict


class RawNet(nn.Module):
    """Matched raw-image network: the PRD 6.1 trunk on the full image (mask all ones) plus X."""

    def __init__(self, out):
        super().__init__()
        self.trunk = Trunk()
        self.head = mlp([FEAT + X_DIM, 128, out])

    def forward(self, w, x):
        return self.head(torch.cat([self.trunk(w, torch.ones(w.shape[1:], dtype=torch.bool)), x], 1))


def fit_raw(w, x, a, y, d_fit, d_val, n_fit, n_val, seed, pre_epochs=30, fine_epochs=60,
            pre_patience=5, fine_patience=8, lr=1e-3, wd=1e-4, ledger=None):
    """Pretrain on D (A by BCE, observed-arm Y by MSE), then fine-tune every parameter on N; early stopping on
    the matching validation rows.  Returns predict(w, x) -> (e, m0, m1)."""
    gen = _seed(seed)
    ymu, ysd = y[n_fit].mean(), y[n_fit].std()
    ys = (y - ymu) / ysd
    prop, out = RawNet(1), RawNet(2)

    def prop_loss(net, rows):
        return F.binary_cross_entropy(bounded(net(w[rows], x[rows]).squeeze(1)), a[rows])

    def out_loss(net, rows):
        o = net(w[rows], x[rows])
        return F.mse_loss(torch.where(a[rows] > 0.5, o[:, 1], o[:, 0]), ys[rows])

    def run(net, lossf, fit, val, epochs, patience):
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

        def vloss():
            with torch.no_grad():
                return float(sum(lossf(net, val[b:b + 512]) * len(val[b:b + 512])
                                 for b in range(0, len(val), 512)) / len(val))

        best, best_state, bad = vloss(), copy.deepcopy(net.state_dict()), 0
        for epoch in range(epochs):
            order = fit[torch.randperm(len(fit), generator=gen)]
            for b in range(0, len(order), BATCH):
                opt.zero_grad(); lossf(net, order[b:b + BATCH]).backward(); opt.step()
            if ledger is not None:
                ledger["cnn_backprop_images"] += len(fit); ledger["cnn_forward_images"] += len(val)
            v = vloss()
            if v < best - 1e-6:
                best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(best_state)

    for net, lossf in ((prop, prop_loss), (out, out_loss)):
        run(net, lossf, d_fit, d_val, pre_epochs, pre_patience)
        run(net, lossf, n_fit, n_val, fine_epochs, fine_patience)

    def predict(ww, xx):
        with torch.no_grad():
            e = torch.cat([bounded(prop(ww[b:b + 1024], xx[b:b + 1024]).squeeze(1)) for b in range(0, len(ww), 1024)])
            o = torch.cat([out(ww[b:b + 1024], xx[b:b + 1024]) for b in range(0, len(ww), 1024)])
        if ledger is not None:
            ledger["cnn_forward_images"] += 2 * len(ww)
        o = o.numpy() * float(ysd) + float(ymu)
        return e.numpy(), o[:, 0], o[:, 1]
    return predict
