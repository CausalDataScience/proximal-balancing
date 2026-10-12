"""Objective comparison v1: three balance-only training objectives for the pixel encoder phi.

Protocol: memo/2026-10-09-single-image-objective-comparison-v1.md.  The bounded generator, patient roles,
held-out T, encoder family (pipeline_v2.NeuralImageEncoder), predictor ladder (pipeline_v2.CriticBundle) and
the independent P-fit / S-audit / E-effect evaluation are imported from decisive_v2 / pipeline_v2 unchanged.
Only the encoder training objective changes, and the summary-reconstruction term L_keep is removed:

    O1  Brier gap R0(q0) - R1(q1); both predictors refitted on D after every 20-step chunk (decisive_v2 loop)
    O2  log-loss gap CE0(q0) - CE1(q1); same loop, predictors refitted with the log loss
    O3  kernel conditional independence tr(R K_T R R K_A R) / m^2, R = eps (K_Z + eps I)^-1; no inner refits

Every arm adds mu * mean[s softplus((q0 - .95)/s) + s softplus((.05 - q0)/s)] with q0 a logistic regression
on Z(D) refitted every 20 encoder steps.  Truth variables (U, L, I, G) enter only the fresh generator and
the explicitly labelled evaluation diagnostics, never an encoder or an ordinary predictor.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
import time

import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from torch import nn

from . import decisive_v2 as dv, folds, level_a, nets, phantom as ph, pipeline_v2 as pv, reader as rd

ARMS = {"O1": ("mae", "warm"), "O2": ("warm",), "O3": ("mae", "warm")}
ARM_OFFSET = {"O1": 70000, "O2": 80000, "O3": 90000}
INIT_OFFSET = {"mae": 0, "warm": 500}
OBJECTIVE = {"O1": "brier", "O2": "logloss", "O3": "kci"}
LR = 2e-4                  # decisive_v2 encoder learning rate
CLIP = 5.                  # decisive_v2 gradient clip
# Overlap weight. decisive_v2 used 10 against L_keep ~ 1 and a squared hinge. Without L_keep every objective
# here lives on a 1e-4..1e-2 scale; in the n=2400 development probe mu=10 made the penalty dominate and the
# O3 C-KCI rose from 2.6e-4 to 7.0e-4, while mu in {0, .1} lowered it. mu=.1 is common to all arms.
MU = .1
OVERLAP_TEMPERATURE = .01  # smooth-hinge scale of the softplus penalty (see overlap_softplus)
KCI_EPS = 1e-2
KCI_BATCH = 512
CHUNK_STEPS = 20
GAP_BATCH = 128            # decisive_v2 minibatch for the Brier/log-loss arms
STOP_PATIENCE = 3
STOP_RELATIVE = .01
POOL = (16, 16)
CE_CLAMP = 1e-6


def sha_bytes(value) -> str:
    return hashlib.sha256(value).hexdigest()


def state_sha(state: dict) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


# ----------------------------------------------------------------------------- overlap penalty

def overlap_softplus(q: torch.Tensor, temperature: float = OVERLAP_TEMPERATURE) -> torch.Tensor:
    """s softplus((q - .95)/s) + s softplus((.05 - q)/s): a smooth hinge on [.05, .95].

    With s = 1 the softplus has gradient about .47 (q - .5) inside the interval, which would penalise the
    information in Z rather than overlap.  s = .01 keeps the penalty near zero inside and linear outside.
    """
    s = temperature
    return s * (nn.functional.softplus((q - .95) / s) + nn.functional.softplus((.05 - q) / s))


class LogisticQ0:
    """Penalty-only logistic q0 on standardised Z. It is not a witness and never audits anything."""

    def __init__(self, z: np.ndarray, a: np.ndarray):
        z = np.asarray(z, float)
        self.mu, self.sd = z.mean(0), z.std(0) + 1e-6
        fit = LogisticRegression(C=1.0, max_iter=5000).fit((z - self.mu) / self.sd, a)
        self.coef, self.intercept = fit.coef_[0].astype(float), float(fit.intercept_[0])
        q = self.predict(z)
        self.report = {"n": len(a), "fraction_outside_05_95": float(np.mean((q < .05) | (q > .95))),
                       "brier": float(np.mean((q - a) ** 2))}
        self._t = [torch.as_tensor(x, dtype=torch.float32) for x in (self.mu, self.sd, self.coef)]

    def predict(self, z: np.ndarray) -> np.ndarray:
        return expit(((np.asarray(z, float) - self.mu) / self.sd) @ self.coef + self.intercept)

    def torch_prob(self, z: torch.Tensor) -> torch.Tensor:
        mu, sd, coef = self._t
        return torch.sigmoid(((z - mu) / sd) @ coef + self.intercept)


# ----------------------------------------------------------------------------- losses

def brier_terms(q0, q1, a):
    return (q0 - a) ** 2 - (q1 - a) ** 2


def ce_np(q, a):
    q = np.clip(q, CE_CLAMP, 1 - CE_CLAMP)
    return -(a * np.log(q) + (1 - a) * np.log1p(-q))


def ce_torch(q, a):
    q = q.clamp(CE_CLAMP, 1 - CE_CLAMP)
    return -(a * torch.log(q) + (1 - a) * torch.log1p(-q))


def center(k: torch.Tensor) -> torch.Tensor:
    return k - k.mean(0, keepdim=True) - k.mean(1, keepdim=True) + k.mean()


def sq_dists(x: torch.Tensor) -> torch.Tensor:
    sq = (x * x).sum(1)
    return (sq[:, None] + sq[None, :] - 2 * x @ x.T).clamp_min(0.)


def median_sq_distance(x: torch.Tensor) -> torch.Tensor:
    d2 = sq_dists(x.detach())
    iu = torch.triu_indices(len(x), len(x), 1)
    return d2[iu[0], iu[1]].median()


def gaussian_kernel(x: torch.Tensor, h2) -> torch.Tensor:
    """exp(-|x-y|^2 / (2 h^2)); h is the median pairwise distance, so h2 is the median squared distance."""
    h2 = torch.as_tensor(h2, dtype=x.dtype).clamp_min(1e-12)
    return torch.exp(-sq_dists(x) / (2 * h2))


def kci_loss(z: torch.Tensor, a, kt_centered: torch.Tensor, eps: float = KCI_EPS):
    """(1/m^2) tr(R K_T R R K_A R), R = eps (K_Z + eps I)^-1, centred kernels as in Zhang et al. (2011) KCI.

    K_A = (Ha)(Ha)^T, so the trace is w^T K_T w with w = R R H a. The Z bandwidth is the per-minibatch
    median (detached); the loss is differentiable in Z through K_Z only.
    """
    z = z.double()
    m = z.shape[0]
    h2 = median_sq_distance(z)
    kz = center(gaussian_kernel(z, h2))
    a = torch.as_tensor(a, dtype=torch.float64)
    v = (a - a.mean())[:, None]
    chol = torch.linalg.cholesky(kz + eps * torch.eye(m, dtype=torch.float64))
    w = eps * eps * torch.cholesky_solve(torch.cholesky_solve(v, chol), chol)
    return (w.T @ kt_centered @ w).squeeze() / m ** 2, float(h2)


# ----------------------------------------------------------------------------- generalised ladder fit

def fit_probability(model, features, base_logit, a, train, val, crop, epochs, seed, *, patience=6,
                    loss="brier", batch=128, lr=.003) -> dict:
    """level_a._fit_probability with a selectable proper loss and batch size.

    loss="brier", batch=128 reproduces level_a._fit_probability operation for operation (tested).
    Internal validation selects the epoch with the same loss; scoring rows are never supplied.
    """
    if loss not in ("brier", "logloss"):
        raise ValueError("loss must be brier or logloss")
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    y = torch.as_tensor(a, dtype=torch.float32)

    def risk(pred, target):
        return float(np.mean((pred - target) ** 2)) if loss == "brier" else float(np.mean(ce_np(pred, target)))

    initial = level_a._predict_probability(model, features[val], base_logit[val], None if crop is None else crop[val])
    best = risk(initial, a[val])
    best_state = copy.deepcopy(model.state_dict())
    initial_risk, best_epoch, stale, ep = best, 0, 0, -1
    for ep in range(epochs):
        model.train()
        order = rng.permutation(train)
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            q = model(features[idx], base_logit[idx], None if crop is None else crop[idx])
            value = ((q - y[idx]) ** 2).mean() if loss == "brier" else ce_torch(q, y[idx]).mean()
            opt.zero_grad(set_to_none=True); value.backward(); opt.step()
        pred = level_a._predict_probability(model, features[val], base_logit[val], None if crop is None else crop[val])
        current = risk(pred, a[val])
        if not np.isfinite(current):
            raise RuntimeError("nonfinite internal validation risk")
        if current < best - 1e-6:
            best, best_epoch, stale = current, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return {"epochs_run": ep + 1, "selected_epoch": best_epoch, "initial_val_risk": initial_risk,
            "selected_val_risk": best, "zero_residual_selected": best_epoch == 0, "loss": loss, "batch": batch}


class LadderBundle(pv.CriticBundle):
    """The decisive_v2 ladder (q0 MLP, reader q1, raw-pixel q1) with a selectable loss and batch size.

    The architecture, feature scaling, nesting and seeds are those of pipeline_v2.CriticBundle; prediction
    and freezing are inherited unchanged.
    """

    def __init__(self, z, t, crop, a, seed, epochs, raw_epochs, *, loss="brier", batch=128, patience=6,
                 raw_patience=12, texture=None):
        n = len(a)
        order = np.random.default_rng(seed).permutation(n)
        nv = max(4, n // 5); self.val, self.train = order[:nv], order[nv:]
        texture = rd.raw_texture_features(crop) if texture is None else texture
        self.scales = []
        arrays = (z, np.column_stack([z, t]), np.column_stack([z, t, texture]))
        tensors = []
        for arr in arrays:
            mu, sd = arr[self.train].mean(0), arr[self.train].std(0) + 1e-6
            self.scales.append((mu.astype(np.float32), sd.astype(np.float32)))
            tensors.append(torch.as_tensor((arr - mu) / sd, dtype=torch.float32))
        f0, fr, fraw = tensors
        zeros = torch.zeros(n)
        kw = {"loss": loss, "batch": batch}
        nets.seed_all(seed)
        self.q0 = level_a._ProbabilityResidual(f0.shape[1])
        with torch.no_grad():
            rate = np.clip(a[self.train].mean(), .001, .999)
            self.q0.head[-1].bias.fill_(math.log(rate / (1 - rate)))
        report0 = fit_probability(self.q0, f0, zeros, a, self.train, self.val, None, epochs, seed,
                                  patience=patience, **kw)
        q0 = level_a._predict_probability(self.q0, f0, zeros)
        nets.seed_all(seed + 1)
        self.reader = level_a._ProbabilityResidual(fr.shape[1])
        reportr = fit_probability(self.reader, fr, level_a._logit(q0), a, self.train, self.val, None, epochs,
                                  seed + 1, patience=patience, **kw)
        qr = level_a._predict_probability(self.reader, fr, level_a._logit(q0))
        nets.seed_all(seed + 2)
        self.raw = level_a._ProbabilityResidual(fraw.shape[1], raw=True)
        reportraw = fit_probability(self.raw, fraw, level_a._logit(qr), a, self.train, self.val,
                                    torch.as_tensor(crop), raw_epochs, seed + 2, patience=raw_patience, **kw)
        self.report = {"q0": report0, "reader": reportr, "raw": reportraw, "loss": loss, "batch": batch,
                       "n_train": len(self.train), "n_internal_val": len(self.val),
                       "train_rows_sha256": level_a.array_sha(self.train),
                       "internal_val_rows_sha256": level_a.array_sha(self.val), "scoring_rows_used": False}
        self.freeze()

    def predict_with_texture(self, z, t, texture, crop, chunk=2048):
        out = [[], [], []]
        with torch.no_grad():
            for start in range(0, len(z), chunk):
                sl = slice(start, start + chunk)
                values = self.tensor_predict(torch.as_tensor(z[sl], dtype=torch.float32),
                    torch.as_tensor(t[sl], dtype=torch.float32), torch.as_tensor(texture[sl]),
                    torch.as_tensor(crop[sl]))
                for dst, value in zip(out, values):
                    dst.append(value.numpy())
        return tuple(np.concatenate(v).astype(float) for v in out)


# ----------------------------------------------------------------------------- fixed T features (O3)

class TFeatures:
    """f(T): held-out raw region (standardised crop) average-pooled to 16 x 16, plus the six fixed
    high-pass features; standardised on R; Gaussian bandwidth by the R median heuristic. Images only."""

    def __init__(self, mode, frame, rows_r, split, seed, n_bandwidth=2000):
        self.split = split
        f = self.raw(mode, frame, rows_r)
        mu, sd = f.mean(0), f.std(0)
        self.keep = sd > 1e-6
        self.mu, self.sd = mu[self.keep], sd[self.keep]
        fs = torch.as_tensor((f[:, self.keep] - self.mu) / self.sd)
        sub = np.sort(np.random.default_rng([seed, 17]).choice(len(rows_r), min(n_bandwidth, len(rows_r)),
                                                                 replace=False))
        self.h2 = float(median_sq_distance(fs[sub]))
        self.report = {"pooled_cells": int(POOL[0] * POOL[1]), "texture_features": len(rd.RAW_TEXTURE_FEATURES),
                       "dropped_constant_on_R": int((~self.keep).sum()), "dimension": int(self.keep.sum()),
                       "bandwidth_median_distance_R": math.sqrt(self.h2), "n_bandwidth_rows": len(sub),
                       "standardised_on": "R", "uses_labels": False}

    def raw(self, mode, frame, rows, chunk=4000):
        out = []
        for start in range(0, len(rows), chunk):
            crop = mode.crop(frame, rows[start:start + chunk], self.split)
            pooled = nn.functional.adaptive_avg_pool2d(torch.from_numpy(crop[:, :1]), POOL).reshape(len(crop), -1)
            out.append(np.column_stack([pooled.numpy().astype(float), rd.raw_texture_features(crop).astype(float)]))
        return np.concatenate(out)

    def transform(self, mode, frame, rows) -> torch.Tensor:
        return torch.as_tensor((self.raw(mode, frame, rows)[:, self.keep] - self.mu) / self.sd)

    def kernel(self, f: torch.Tensor) -> torch.Tensor:
        return center(gaussian_kernel(f, self.h2))


# ----------------------------------------------------------------------------- stop rule / checkpoints

class StopRule:
    """Stop after `patience` consecutive evaluations that fail to beat the running minimum by more than
    `relative` of its magnitude; the selected checkpoint is the lowest evaluated value."""

    def __init__(self, patience=STOP_PATIENCE, relative=STOP_RELATIVE):
        self.patience, self.relative = patience, relative
        self.best_value, self.best_index, self.best_state = None, None, None
        self.stale = 0

    def update(self, value: float, index: int, state) -> bool:
        if not np.isfinite(value):
            raise RuntimeError("nonfinite C criterion")
        if self.best_value is None:
            self.best_value, self.best_index, self.best_state = value, index, state
            return False
        if value < self.best_value - self.relative * abs(self.best_value):
            self.stale = 0
        else:
            self.stale += 1
        if value < self.best_value:
            self.best_value, self.best_index, self.best_state = value, index, state
        return self.stale >= self.patience


# ----------------------------------------------------------------------------- encoders

def base_encoder(mode, split, cfg):
    """decisive_v2 MAE initialisation for this split (same seed, so the same head draw)."""
    return pv.NeuralImageEncoder(mode, split, cfg.seed + 1000 + split, "mae")


def clone_encoder(mode, split, cfg, state):
    enc = base_encoder(mode, split, cfg)
    enc.model.load_state_dict(copy.deepcopy(state))
    return enc


def warm_start(mode, cfg, split, target, rows_r, seed, epochs, lr, patience, log, batch=128):
    """Regress phi (from the MAE initialisation) onto the learned coordinates of Z_lin, R images only."""
    t0 = time.time()
    enc = base_encoder(mode, split, cfg)
    frame = mode.frame
    y_all = target.encode(frame, rows_r)[:, 1:]
    tmu, tsd = y_all.mean(0), y_all.std(0) + 1e-12
    y = torch.as_tensor((y_all - tmu) / tsd, dtype=torch.float32)
    rng = np.random.default_rng([seed, 23])
    perm = rng.permutation(len(rows_r)); nv = max(4, len(rows_r) // 5)
    val, train = np.sort(perm[:nv]), np.sort(perm[nv:])
    opt = torch.optim.Adam(enc.model.parameters(), lr=lr)

    def val_pred():
        enc.model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(val), 256):
                out.append(enc.model(torch.from_numpy(enc.view(frame, rows_r[val[start:start + 256]]))))
        return torch.cat(out)

    def mse(pred):
        return float((pred - y[val]).square().mean())

    best = mse(val_pred()); best_state = copy.deepcopy(enc.model.state_dict())
    history, best_epoch, stale, initial = [], 0, 0, best
    for ep in range(epochs):
        enc.model.train()
        order = rng.permutation(train)
        run = []
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            loss = (enc.model(torch.from_numpy(enc.view(frame, rows_r[idx]))) - y[idx]).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); run.append(float(loss.detach()))
        current = mse(val_pred())
        history.append({"epoch": ep + 1, "train_mse": float(np.mean(run)), "R_val_mse": current})
        log(f"warm start S{split + 1} epoch {ep + 1}/{epochs}: train {np.mean(run):.4f} R-val {current:.4f}")
        if current < best - 1e-6:
            best, best_epoch, stale = current, ep + 1, 0
            best_state = copy.deepcopy(enc.model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    enc.model.load_state_dict(best_state); enc.model.eval()
    pred = val_pred().numpy()
    r2 = 1 - ((pred - y[val].numpy()) ** 2).mean(0) / y[val].numpy().var(0)
    report = {"target": "learned coordinates of level-A Z_lin (fixed reader, rank 2), standardised on R",
              "rows": "R images only", "n_train": len(train), "n_R_val": len(val), "lr": lr, "batch": batch,
              "epochs_run": len(history), "selected_epoch": best_epoch, "initial_R_val_mse": initial,
              "selected_R_val_mse": best, "R_val_R2_per_coordinate": r2.tolist(), "history": history,
              "target_mean": tmu.tolist(), "target_sd": tsd.tolist(), "start": "decisive_v2 MAE initialisation",
              "state_sha256": state_sha(enc.model.state_dict()), "seconds": time.time() - t0}
    return enc, report


# ----------------------------------------------------------------------------- O1 / O2

def make_bundle(objective, z, t, crop, a, seed, epochs, raw_epochs, texture=None):
    if objective == "brier":
        return pv.CriticBundle(z, t, crop, a, seed, epochs, raw_epochs)   # decisive_v2 refit, unchanged
    return LadderBundle(z, t, crop, a, seed, epochs, raw_epochs, loss="logloss", texture=texture)


def train_gap_arm(mode, data, roles, encoder, split, objective, seed, args, log, name):
    t0 = time.time()
    frame, d, c = mode.frame, roles["D"], roles["C"]
    a_d, a_c = data.A[d], data.A[c]
    psi_d, psi_c = frame.psi[split][d], frame.psi[split][c]
    crops_d, crops_c = mode.crop(frame, d, split), mode.crop(frame, c, split)
    texture_d = rd.raw_texture_features(crops_d)
    tex_t, crop_t, psi_t = torch.as_tensor(texture_d), torch.as_tensor(crops_d), torch.as_tensor(psi_d, dtype=torch.float32)
    a_t = torch.as_tensor(a_d, dtype=torch.float32)
    opt = torch.optim.Adam(encoder.model.parameters(), lr=args.lr)
    rng = np.random.default_rng(seed)

    def refit(fit_seed):
        z_d = encoder.encode(frame, d)
        bundle = make_bundle(objective, z_d, psi_d, crops_d, a_d, fit_seed, args.critic_epochs, args.raw_epochs,
                             texture=texture_d)
        logistic = LogisticQ0(z_d, a_d)
        q0, q1, qr = bundle.predict(encoder.encode(frame, c), psi_c, crops_c)
        diag = {"C_brier_gap": float(brier_terms(q0, q1, a_c).mean()),
                "C_brier_gap_reader": float(brier_terms(q0, qr, a_c).mean()),
                "C_logloss_gap": float((ce_np(q0, a_c) - ce_np(q1, a_c)).mean()),
                "C_q0_outside_05_95": float(np.mean((q0 < .05) | (q0 > .95))),
                "D_logistic_q0": logistic.report,
                "selected_epochs": {k: bundle.report[k]["selected_epoch"] for k in ("q0", "reader", "raw")}}
        crit = diag["C_brier_gap"] if objective == "brier" else diag["C_logloss_gap"]
        return bundle, logistic, crit, diag

    bundle, logistic, crit, diag = refit(seed + 10)
    rule = StopRule()
    rule.update(crit, 0, copy.deepcopy(encoder.model.state_dict()))
    history = [{"round": 0, "steps": 0, "C_criterion": crit, **diag, "seconds": time.time() - t0}]
    log(f"loop {name} round 0: C {objective} gap={crit:.6f}")
    stopped = False
    for rnd in range(1, args.rounds + 1):
        encoder.model.train()
        mains, overs, losses = [], [], []
        for _ in range(args.encoder_steps):
            local = rng.choice(len(d), size=min(GAP_BATCH, len(d)), replace=False)
            z = encoder.forward(frame, d[local])
            q0, q1, _ = bundle.tensor_predict(z, psi_t[local], tex_t[local], crop_t[local])
            a = a_t[local]
            main = brier_terms(q0, q1, a).mean() if objective == "brier" else (ce_torch(q0, a) - ce_torch(q1, a)).mean()
            over = overlap_softplus(logistic.torch_prob(z)).mean()
            loss = main + args.mu * over
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), CLIP)
            opt.step()
            mains.append(float(main.detach())); overs.append(float(over.detach())); losses.append(float(loss.detach()))
        bundle, logistic, crit, diag = refit(seed + 1000 + rnd)   # mandatory new-phi refit before C
        stop = rule.update(crit, rnd, copy.deepcopy(encoder.model.state_dict()))
        history.append({"round": rnd, "steps": rnd * args.encoder_steps, "loss_mean": float(np.mean(losses)),
                        "main_mean": float(np.mean(mains)), "overlap_mean": float(np.mean(overs)),
                        "C_criterion": crit, **diag, "seconds": time.time() - t0})
        log(f"loop {name} round {rnd}/{args.rounds}: C {objective} gap={crit:.6f}, C Brier gap="
            f"{diag['C_brier_gap']:.6f}, D q0log out={diag['D_logistic_q0']['fraction_outside_05_95']:.3%}, "
            f"stale={rule.stale}")
        if stop:
            stopped = True
            break
    encoder.model.load_state_dict(rule.best_state); encoder.model.eval()
    return {"objective": objective, "history": history, "rounds_completed": len(history) - 1,
            "encoder_steps_total": (len(history) - 1) * args.encoder_steps, "stopped_by_C_rule": stopped,
            "capped": not stopped, "selected_round": rule.best_index, "selected_C_criterion": rule.best_value,
            "selected_encoder_sha256": state_sha(encoder.model.state_dict()), "S_or_E_used_for_selection": False,
            "inner_predictor_refit_every_chunk": True, "L_keep": False, "minibatch": GAP_BATCH,
            "lr": args.lr, "mu": args.mu, "overlap_temperature": OVERLAP_TEMPERATURE,
            "encoder_training_seconds": time.time() - t0}


# ----------------------------------------------------------------------------- O3

def train_kci_arm(mode, data, roles, encoder, split, tfeat, seed, args, log, name):
    t0 = time.time()
    frame, d, c = mode.frame, roles["D"], roles["C"]
    a_d, a_c = data.A[d], data.A[c]
    ft_d = tfeat.transform(mode, frame, d)
    ft_c = tfeat.transform(mode, frame, c)
    m, mc = min(args.kci_batch, len(d)), min(args.kci_batch, len(c))
    c_perm = np.random.default_rng([seed, 1]).permutation(len(c))
    c_batches = [np.sort(c_perm[i:i + mc]) for i in range(0, len(c) - mc + 1, mc)]
    kt_c = [tfeat.kernel(ft_c[b]) for b in c_batches]
    opt = torch.optim.Adam(encoder.model.parameters(), lr=args.lr)
    rng = np.random.default_rng(seed)

    def evaluate():
        z_d = encoder.encode(frame, d)
        logistic = LogisticQ0(z_d, a_d)
        z_c = torch.as_tensor(encoder.encode(frame, c))
        with torch.no_grad():
            values = [float(kci_loss(z_c[b], a_c[b], kt, args.kci_eps)[0]) for b, kt in zip(c_batches, kt_c)]
        return logistic, float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(len(values))) \
            if len(values) > 1 else None

    logistic, crit, crit_se = evaluate()
    rule = StopRule()
    rule.update(crit, 0, copy.deepcopy(encoder.model.state_dict()))
    history = [{"block": 0, "step": 0, "C_kci": crit, "C_kci_batch_SE": crit_se,
                "D_logistic_q0": logistic.report, "seconds": time.time() - t0}]
    log(f"loop {name} step 0: C KCI={crit:.6e}")
    stopped, kcis, overs, losses, bands = False, [], [], [], []
    for step in range(1, args.o3_steps + 1):
        encoder.model.train()
        local = np.sort(rng.choice(len(d), size=m, replace=False))
        z = encoder.forward(frame, d[local])
        lk, h2 = kci_loss(z, a_d[local], tfeat.kernel(ft_d[local]), args.kci_eps)
        over = overlap_softplus(logistic.torch_prob(z)).mean()
        loss = lk.float() + args.mu * over
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), CLIP)
        opt.step()
        kcis.append(float(lk.detach())); overs.append(float(over.detach())); losses.append(float(loss.detach()))
        bands.append(math.sqrt(h2))
        if step % CHUNK_STEPS == 0 or step == args.o3_steps:
            logistic, crit, crit_se = evaluate()
            stop = rule.update(crit, step, copy.deepcopy(encoder.model.state_dict()))
            history.append({"block": len(history), "step": step, "loss_mean": float(np.mean(losses)),
                            "kci_mean": float(np.mean(kcis)), "overlap_mean": float(np.mean(overs)),
                            "Z_bandwidth_median_mean": float(np.mean(bands)), "C_kci": crit,
                            "C_kci_batch_SE": crit_se, "D_logistic_q0": logistic.report,
                            "seconds": time.time() - t0})
            log(f"loop {name} step {step}/{args.o3_steps}: C KCI={crit:.6e}, train KCI={np.mean(kcis):.6e}, "
                f"D q0log out={logistic.report['fraction_outside_05_95']:.3%}, stale={rule.stale}")
            kcis, overs, losses, bands = [], [], [], []
            if stop:
                stopped = True
                break
    encoder.model.load_state_dict(rule.best_state); encoder.model.eval()
    return {"objective": "kci", "history": history, "encoder_steps_total": history[-1]["step"],
            "stopped_by_C_rule": stopped, "capped": not stopped, "selected_step": rule.best_index,
            "selected_C_criterion": rule.best_value, "selected_encoder_sha256": state_sha(encoder.model.state_dict()),
            "S_or_E_used_for_selection": False, "inner_predictor_refit_every_chunk": False, "L_keep": False,
            "minibatch": m, "C_batches": len(c_batches), "C_batch_size": mc, "eps": args.kci_eps,
            "lr": args.lr, "mu": args.mu, "overlap_temperature": OVERLAP_TEMPERATURE, "T_features": tfeat.report,
            "kernel_centering": "K_T, K_Z and K_A centred (KCI form); K_A linear in A",
            "encoder_training_seconds": time.time() - t0}


# ----------------------------------------------------------------------------- fresh-sample D2

class LazyCorrected:
    def __init__(self, images, offset):
        self.images, self.offset = images, offset

    def __getitem__(self, rows):
        return pv.corrected_images(self.images[rows], self.offset[rows])

    def __len__(self):
        return len(self.images)


class FreshSample:
    """New bounded-generator draws in seeded chunks (chunk c uses seed + start_c); phi and T stay fixed.

    Images are regenerated per chunk on demand (bitwise identical), so only reader features, labels and the
    current split's crops stay in memory. The first and second halves give two independent replicates.
    """

    def __init__(self, mode, cfg, n, seed, log, chunk=10000):
        t0 = time.time()
        self.mode, self.cfg, self.n, self.seed = mode, cfg, n, seed
        self.chunks = [(start, min(start + chunk, n)) for start in range(0, n, chunk)]
        a, p_li, offsets, blocks, digest = [], [], [], [[] for _ in range(5)], hashlib.sha256()
        for start, stop in self.chunks:
            data = self._generate(start, stop)
            a.append(data.A); p_li.append(dv.p_li(data))      # p_LI: evaluation-only reference diagnostic
            offset = pv.body_offset(data.images, mode.body_base)
            offsets.append(offset)
            for j, block in enumerate(mode.fit["bank"].transform(pv.corrected_images(data.images, offset))):
                blocks[j].append(block)
            digest.update(np.ascontiguousarray(data.images).tobytes())
            del data
        self.A, self.p_LI, self.offset = np.concatenate(a), np.concatenate(p_li), np.concatenate(offsets)
        self.reference = (self.offset - mode.ref_stats[0]) / mode.ref_stats[1]
        self.psi = [np.concatenate(b) for b in blocks]
        half = n // 2
        self.halves = (np.arange(half), np.arange(half, n))
        self._split, self._inputs = None, None
        self.report = {"n": n, "seed_rule": f"{seed}+chunk_start", "chunk": chunk,
                       "generator": "decisive_v2.bounded_generate", "halves": [half, n - half],
                       "A_sha256": level_a.array_sha(self.A), "images_sha256": digest.hexdigest(),
                       "psi_sha256": [level_a.array_sha(p) for p in self.psi], "seconds": time.time() - t0}
        log(f"fresh sample n={n} seed={seed}+start ready in {time.time() - t0:.1f}s")

    def _generate(self, start, stop):
        return dv.bounded_generate(ph.with_changes(self.cfg, n=stop - start, seed=self.seed + start))

    def frame(self, start, stop, images=None):
        sl = slice(start, stop)
        processed = None if images is None else LazyCorrected(images, self.offset[sl])
        return pv.ImageFrame(images, processed, self.reference[sl], [p[sl] for p in self.psi], None)

    def encode(self, encoder) -> np.ndarray:
        if isinstance(encoder, pv.LinearImageEncoder):            # reader features only; no pixels needed
            return encoder.encode(self.frame(0, self.n), np.arange(self.n))
        out = []
        for start, stop in self.chunks:
            out.append(encoder.encode(self.frame(start, stop, self._generate(start, stop).images),
                                      np.arange(stop - start)))
        return np.concatenate(out)

    def inputs(self, mode, split):
        if self._split != split:
            self._inputs = None
            crops = np.concatenate([mode.crop(self.frame(start, stop, self._generate(start, stop).images),
                                              np.arange(stop - start), split) for start, stop in self.chunks])
            self._inputs = (crops, rd.raw_texture_features(crops), self.psi[split])
            self._split = split
        return self._inputs


def fresh_d2(mode, encoder, split, fresh, args, log, name):
    t0 = time.time()
    z = fresh.encode(encoder)
    crops, texture, psi = fresh.inputs(mode, split)
    t_inputs = time.time() - t0
    a = fresh.A
    reps = []
    for r, (fit_rows, other) in enumerate((fresh.halves, fresh.halves[::-1])):
        t1 = time.time()
        sl = slice(fit_rows[0], fit_rows[-1] + 1)
        bundle = LadderBundle(z[sl], psi[sl], crops[sl], a[sl], args.fresh_fit_seed + 10 * r, args.fresh_epochs,
                              args.fresh_raw_epochs, loss="brier", batch=args.fresh_batch,
                              patience=args.fresh_patience, raw_patience=args.fresh_raw_patience, texture=texture[sl])
        q0, q1, qr = bundle.predict_with_texture(z, psi, texture, crops)
        reps.append({"q0": q0, "q1": q1, "qr": qr, "fit_rows": fit_rows, "other": other, "report": bundle.report})
        log(f"fresh D2[{name}] replicate {r + 1}: epochs run q0/reader/raw="
            f"{[bundle.report[k]['epochs_run'] for k in ('q0', 'reader', 'raw')]}, selected="
            f"{[bundle.report[k]['selected_epoch'] for k in ('q0', 'reader', 'raw')]}, {time.time() - t1:.0f}s")
    q0o, q1o, qro = (np.empty(fresh.n) for _ in range(3))
    for rep in reps:
        q0o[rep["other"]], q1o[rep["other"]], qro[rep["other"]] = rep["q0"][rep["other"]], \
            rep["q1"][rep["other"]], rep["qr"][rep["other"]]
    diff2 = (q1o - q0o) ** 2
    rep_d2 = [float(np.mean((rep["q1"][rep["other"]] - rep["q0"][rep["other"]]) ** 2)) for rep in reps]
    delta = [rep["q1"] - rep["q0"] for rep in reps]
    gap_terms = brier_terms(q0o, q1o, a)
    out = {"D2_fresh": float(diff2.mean()), "D2_fresh_MC_SE": float(diff2.std(ddof=1) / math.sqrt(fresh.n)),
           "D2_fresh_reader_stage": float(np.mean((qro - q0o) ** 2)),
           "replicate_D2_out_of_fold": rep_d2, "second_replicate_difference": rep_d2[0] - rep_d2[1],
           "replicate_disagreement_mean_sq_delta": float(np.mean((delta[0] - delta[1]) ** 2)),
           "replicate_disagreement_mean_sq_q1": float(np.mean((reps[0]["q1"] - reps[1]["q1"]) ** 2)),
           "replicate_disagreement_mean_sq_q0": float(np.mean((reps[0]["q0"] - reps[1]["q0"]) ** 2)),
           "validation_brier": [{k: rep["report"][k]["selected_val_risk"] for k in ("q0", "reader", "raw")}
                                for rep in reps],
           "selected_epochs": [{k: rep["report"][k]["selected_epoch"] for k in ("q0", "reader", "raw")}
                               for rep in reps],
           "epochs_run": [{k: rep["report"][k]["epochs_run"] for k in ("q0", "reader", "raw")} for rep in reps],
           "out_of_fold_brier": {"q0": float(np.mean((q0o - a) ** 2)), "q1": float(np.mean((q1o - a) ** 2)),
                                 "gap": float(gap_terms.mean()),
                                 "gap_SE": float(gap_terms.std(ddof=1) / math.sqrt(fresh.n))},
           "evaluation_only_vs_exact_p_LI": {"q1_mean_sq_error": float(np.mean((q1o - fresh.p_LI) ** 2)),
                                             "q0_mean_sq_error": float(np.mean((q0o - fresh.p_LI) ** 2)),
                                             "scope": "p(A|L_1:5,I) truth reference; diagnostic only"},
           "Z_sha256": level_a.array_sha(z), "fit": "two cross-fitted replicates on contiguous halves of the "
           "fresh sample; each predicts the other half; D2_fresh = mean over all draws of out-of-fold (q1*-q0*)^2",
           "architecture": "decisive_v2 ladder (q0 MLP, reader q1, raw-pixel q1 = ladder top)",
           "batch": args.fresh_batch, "max_epochs": [args.fresh_epochs, args.fresh_epochs, args.fresh_raw_epochs],
           "patience": [args.fresh_patience, args.fresh_patience, args.fresh_raw_patience],
           "labels_from_fresh_draws_only": True, "seconds_encode_and_inputs": t_inputs, "seconds": time.time() - t0}
    log(f"fresh D2[{name}] = {out['D2_fresh']:.6f} (reps {rep_d2[0]:.6f}/{rep_d2[1]:.6f}; "
        f"val Brier raw {out['validation_brier'][0]['raw']:.5f}) in {out['seconds']:.0f}s")
    return out


# ----------------------------------------------------------------------------- evaluation per model

def audit_summary(audit: dict) -> dict:
    return {"gap": audit["gap"], "se": audit["se"], "R0_q0_brier": audit["q0"]["brier"],
            "R1_raw_brier": audit["q1"]["brier"], "R1_reader_brier": audit["reader_only"]["q1"]["brier"],
            "gap_reader_stage": audit["reader_only"]["gap"], "B2": audit["B2"],
            "q0_outside_05_95": audit["q0"]["fraction_outside_05_95"], "q0_auc": audit["q0"]["auc"],
            "epoch_selection": audit["epoch_selection"], "Z_sha256": audit["Z_sha256"],
            "q1_prediction_sha256": audit["q1_prediction_sha256"]}


def final_evaluation(mode, data, roles, encoder, bundle_seed, args, name):
    t0 = time.time()
    bundle = pv.fit_for_rows(mode, encoder, mode.frame, roles["P"], data.A, bundle_seed,
                             args.critic_epochs, args.raw_epochs)
    audit = pv.evaluate_rows(mode, encoder, bundle, mode.frame, roles["S"], data.A)
    effect, score = dv.effect_candidate(mode, data, roles, encoder, bundle, name)
    return audit, effect, score, time.time() - t0


def level_a_split(mode, data, roles, cfg, split, args):
    """decisive_v2 fixed-reader rank-2 PROBE encoder, P predictors, S audit and E candidate, same seeds."""
    t0 = time.time()
    enc = pv.LinearImageEncoder(mode, roles["D"], data.A, split, cfg.seed + 100 + split)
    t_enc = time.time() - t0
    audit, effect, score, t_eval = final_evaluation(mode, data, roles, enc, cfg.seed + 200 + split, args,
                                                    f"fixed_S{split + 1}")
    return enc, audit, effect, score, t_enc, t_eval


def compare_with_reference(reference: dict | None, split: int, enc, audit, effect) -> dict:
    if reference is None:
        return {"reference_used": False}
    old_a = reference["audits"][f"fixed_S{split + 1}"]
    old_e = reference["effects"][f"fixed_S{split + 1}"]
    checks = {"B_sha256": old_a["encoder"]["B_sha256"] == enc.report["B_sha256"],
              "S_Z_sha256": old_a["Z_sha256"] == audit["Z_sha256"],
              "S_q1_prediction_sha256": old_a["q1_prediction_sha256"] == audit["q1_prediction_sha256"],
              "S_gap": old_a["gap"] == audit["gap"],
              "E_score_sha256": old_e["score_sha256"] == effect["score_sha256"],
              "E_candidate": old_e["candidate_estimate"] == effect["candidate_estimate"]}
    return {"reference_used": True, "checks": checks, "all_identical": all(checks.values()),
            "reference_gap": old_a["gap"], "reference_candidate": old_e["candidate_estimate"]}


# ----------------------------------------------------------------------------- stage 1 with cache

def rebuild_mode(cached: dict, data):
    images = data.images
    offset = pv.body_offset(images, cached["body_base"])
    processed = pv.corrected_images(images, offset)
    ref = cached["ref_stats"]
    psi = cached["fit"]["bank"].transform(processed)
    frame = pv.ImageFrame(images, processed, (offset - ref[0]) / ref[1], psi,
                          cached["fit"]["summaries"].transform(processed))
    return pv.Mode("corrected", cached["fit"], frame, tuple(cached["raw_stats"]), tuple(ref),
                   cached["body_base"], cached["report"])


def stage1(data, roles, cfg, args, log, cache_dir=None):
    key = f"stage1_n{cfg.n}_sigma{cfg.sigma}_seed{cfg.seed}_r{args.reader_epochs}_m{args.mae_epochs}.pt"
    path = None if cache_dir is None else cache_dir / key
    if path is not None and path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        mode = rebuild_mode(cached, data)
        psi_sha = [level_a.array_sha(p) for p in mode.frame.psi]
        if psi_sha != cached["psi_sha256"] or level_a.array_sha(data.images) != cached["images_sha256"]:
            raise RuntimeError("stage-1 cache does not reproduce the reader features; refuse to use it")
        log(f"stage1 loaded from cache {path.name}; reader features identical")
        return mode, {"cache": str(path), "loaded": True, "psi_sha256": psi_sha}
    mode = pv.prepare_mode(data, roles, "corrected", cfg.seed + 2, args.reader_epochs, args.mae_epochs, log)
    psi_sha = [level_a.array_sha(p) for p in mode.frame.psi]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        torch.save({"fit": mode.fit, "raw_stats": mode.raw_stats, "ref_stats": mode.ref_stats,
                    "body_base": mode.body_base, "report": mode.report, "psi_sha256": psi_sha,
                    "images_sha256": level_a.array_sha(data.images)}, tmp)
        tmp.replace(path)
    return mode, {"cache": None if path is None else str(path), "loaded": False, "psi_sha256": psi_sha}


def warm_cached(mode, cfg, split, target, roles, args, log, cache_dir):
    key = (f"warm_n{cfg.n}_sigma{cfg.sigma}_seed{cfg.seed}_S{split + 1}_e{args.warm_epochs}_lr{args.warm_lr}"
           f"_p{args.warm_patience}_B{target.report['B_sha256'][:16]}.pt")
    path = None if cache_dir is None else cache_dir / key
    if path is not None and path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if state_sha(cached["state"]) != cached["report"]["state_sha256"]:
            raise RuntimeError("warm-start cache corrupted")
        log(f"warm start S{split + 1} loaded from cache")
        return cached["state"], {**cached["report"], "loaded_from_cache": True}
    enc, report = warm_start(mode, cfg, split, target, roles["R"], cfg.seed + 60000 + split, args.warm_epochs,
                             args.warm_lr, args.warm_patience, log)
    state = copy.deepcopy(enc.model.state_dict())
    if path is not None:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        torch.save({"state": state, "report": report}, tmp)
        tmp.replace(path)
    return state, {**report, "loaded_from_cache": False}


# ----------------------------------------------------------------------------- one part of the run

def model_record(name, arm, init, split, seed, training, audit, effect, score, levela_score, t_eval, fresh):
    paired = dv.paired_sampling(score, levela_score)
    return {"name": name, "arm": arm, "init": init, "split": split + 1, "seed": seed,
            "training": training, "audit_S": audit_summary(audit), "effect_E": effect,
            "paired_vs_levelA_same_split": paired,
            "absolute_error_difference_vs_levelA": abs(score.mean() - 1.) - abs(levela_score.mean() - 1.),
            "clipped_fraction_E": effect["propensity_tail_fraction"],
            "D2_fresh": fresh, "E_scores": score.tolist(),
            "seconds": {"encoder_training": training.get("encoder_training_seconds"), "final_P_S_E": t_eval,
                        "fresh_D2": None if fresh is None else fresh["seconds"]}}


def run_part(args, log, guard, checkpoint, reference=None, cache_dir=None):
    nets.setup_torch(2)
    cfg = ph.PhantomConfig(n=args.n, sigma=args.sigma, seed=args.seed, image_only=True)
    data = dv.bounded_generate(cfg)
    grouped = folds.six_groups(args.n, args.seed)
    roles = {"R": grouped["R"], **folds.roles(grouped["groups"], 0)}
    if len(np.unique(np.concatenate(list(roles.values())))) != args.n or data.X.shape[1] != 0:
        raise AssertionError("patient roles or image-only contract failed")
    arms = [arm for arm in ARMS if arm in args.arms]
    plan = [(arm, init) for arm in arms for init in ARMS[arm] if init in args.inits]
    result = {"artifact": "single_image_objective_v1_part", "computation_complete": False, "tag": args.tag,
              "feature_mode": "image_only", "config": {**ph.config_dict(cfg), "link": "bounded"},
              "role_sizes": {k: len(v) for k, v in roles.items()},
              "role_rows_sha256": {k: level_a.array_sha(v) for k, v in roles.items()},
              "plan": {"splits": [s + 1 for s in args.splits], "arms_inits": plan,
                       "level_A_fresh_D2": args.level_a_fresh},
              "settings": {"rounds": args.rounds, "encoder_steps": args.encoder_steps, "o3_steps": args.o3_steps,
                           "kci_batch": args.kci_batch, "kci_eps": args.kci_eps, "lr": args.lr, "mu": args.mu,
                           "overlap_temperature": OVERLAP_TEMPERATURE, "gap_minibatch": GAP_BATCH,
                           "stop_rule": {"patience": STOP_PATIENCE, "relative": STOP_RELATIVE,
                                         "checkpoint": "lowest C criterion including the initial point"},
                           "reader_epochs": args.reader_epochs, "mae_epochs": args.mae_epochs,
                           "critic_epochs": args.critic_epochs, "raw_epochs": args.raw_epochs,
                           "warm": {"epochs": args.warm_epochs, "lr": args.warm_lr, "patience": args.warm_patience},
                           "fresh": {"n": args.mc, "seed": args.seed + 30000001, "fit_seed": args.fresh_fit_seed,
                                     "batch": args.fresh_batch, "epochs": args.fresh_epochs,
                                     "raw_epochs": args.fresh_raw_epochs, "patience": args.fresh_patience,
                                     "raw_patience": args.fresh_raw_patience}},
              "models": {}, "level_A": {}, "warm_start": {},
              "protocol": {"L_keep": False, "S_E_used_for_selection": False, "rotation": 0,
                           "truth_inputs_to_encoder_or_ordinary_predictors": False,
                           "returned_causal_estimate": None},
              "completion": []}
    checkpoint(result)
    log(f"objective_v1 part {args.tag}: splits={[s + 1 for s in args.splits]} plan={plan}")
    mode, cache_info = stage1(data, roles, cfg, args, log, cache_dir)
    result["stage1"] = {**{k: v for k, v in mode.report.items() if k not in ("reader",)}, "cache": cache_info,
                        "mae_val_masked_mse": mode.fit["mae"]["val_masked_mse"]}
    checkpoint(result)
    fresh = None

    def get_fresh():
        nonlocal fresh
        if fresh is None:
            fresh = FreshSample(mode, cfg, args.mc, args.seed + 30000001, log)
            result["fresh_sample"] = fresh.report
        return fresh

    for split in args.splits:
        guard()
        sname = f"S{split + 1}"
        enc_a, audit_a, effect_a, score_a, t_enc, t_eval = level_a_split(mode, data, roles, cfg, split, args)
        rec = {"name": f"levelA_{sname}", "arm": "levelA", "init": "fixed_reader", "split": split + 1,
               "seed": args.seed, "encoder": enc_a.report, "audit_S": audit_summary(audit_a), "effect_E": effect_a,
               "clipped_fraction_E": effect_a["propensity_tail_fraction"], "E_scores": score_a.tolist(),
               "reproduces_decisive_v2": compare_with_reference(reference, split, enc_a, audit_a, effect_a),
               "seconds": {"encoder_training": t_enc, "final_P_S_E": t_eval, "fresh_D2": None}, "D2_fresh": None}
        log(f"levelA {sname}: S gap={audit_a['gap']:.6f}, theta={effect_a['candidate_estimate']:.5f}, "
            f"reproduces v2={rec['reproduces_decisive_v2'].get('all_identical')}")
        if args.level_a_fresh:
            rec["D2_fresh"] = fresh_d2(mode, enc_a, split, get_fresh(), args, log, rec["name"])
            rec["seconds"]["fresh_D2"] = rec["D2_fresh"]["seconds"]
        result["level_A"][sname] = rec
        checkpoint(result)
        base_state = copy.deepcopy(base_encoder(mode, split, cfg).model.state_dict())
        warm_state = None
        if any(init == "warm" for _, init in plan):
            guard()
            warm_state, report = warm_cached(mode, cfg, split, enc_a, roles, args, log, cache_dir)
            result["warm_start"][sname] = report
            checkpoint(result)
        tfeat = None
        for arm, init in plan:
            guard()
            name = f"{arm}_{init}_{sname}"
            seed = args.seed + ARM_OFFSET[arm] + INIT_OFFSET[init] + split
            encoder = clone_encoder(mode, split, cfg, base_state if init == "mae" else warm_state)
            init_sha = state_sha(encoder.model.state_dict())
            if OBJECTIVE[arm] == "kci":
                if tfeat is None:
                    tfeat = TFeatures(mode, mode.frame, roles["R"], split, args.seed + 95000 + split)
                training = train_kci_arm(mode, data, roles, encoder, split, tfeat, seed, args, log, name)
            else:
                training = train_gap_arm(mode, data, roles, encoder, split, OBJECTIVE[arm], seed, args, log, name)
            training["initial_encoder_sha256"] = init_sha
            audit, effect, score, t_eval = final_evaluation(mode, data, roles, encoder, seed + 9000, args, name)
            guard()
            fresh_rec = fresh_d2(mode, encoder, split, get_fresh(), args, log, name) if args.mc > 0 else None
            result["models"][name] = model_record(name, arm, init, split, args.seed, training, audit, effect,
                                                  score, score_a, t_eval, fresh_rec)
            result["completion"].append(name)
            checkpoint(result)
            log(f"{name}: train {training['encoder_training_seconds']:.0f}s, S gap={audit['gap']:.6f}, "
                f"theta={effect['candidate_estimate']:.5f} (levelA {effect_a['candidate_estimate']:.5f}), "
                f"D2_fresh={'NA' if fresh_rec is None else format(fresh_rec['D2_fresh'], '.6f')}")
    result["computation_complete"] = True
    checkpoint(result)
    return result


# ----------------------------------------------------------------------------- merge and decision table

def merge_parts(parts: list[dict]) -> dict:
    if not parts:
        raise ValueError("no parts to merge")
    base = parts[0]
    for part in parts[1:]:
        for key in ("config", "role_rows_sha256"):
            if part[key] != base[key]:
                raise ValueError(f"parts disagree on {key}")
        if part["stage1"]["cache"]["psi_sha256"] != base["stage1"]["cache"]["psi_sha256"]:
            raise ValueError("parts disagree on stage-1 reader features")
    level, models, warm = {}, {}, {}
    for part in parts:
        if not part.get("computation_complete"):
            raise ValueError(f"part {part.get('tag')} is incomplete")
        for split, rec in part["level_A"].items():
            if split in level:
                old = level[split]
                if (old["audit_S"]["Z_sha256"], old["effect_E"]["score_sha256"]) != \
                        (rec["audit_S"]["Z_sha256"], rec["effect_E"]["score_sha256"]):
                    raise ValueError(f"level A {split} differs between parts")
                if old["D2_fresh"] is None and rec["D2_fresh"] is not None:
                    level[split] = rec
            else:
                level[split] = rec
        for name, rec in part["models"].items():
            if name in models and models[name]["audit_S"]["Z_sha256"] != rec["audit_S"]["Z_sha256"]:
                raise ValueError(f"model {name} differs between parts")
            models[name] = rec
        for split, rec in part["warm_start"].items():
            if split in warm and warm[split]["state_sha256"] != rec["state_sha256"]:
                raise ValueError(f"warm start {split} differs between parts")
            warm[split] = rec
    fresh = {p["fresh_sample"]["images_sha256"] for p in parts if "fresh_sample" in p}
    if len(fresh) > 1:
        raise ValueError("parts used different fresh samples")
    return {"artifact": "single_image_objective_v1", "config": base["config"], "role_sizes": base["role_sizes"],
            "role_rows_sha256": base["role_rows_sha256"], "settings": base["settings"], "stage1": base["stage1"],
            "fresh_sample": next((p["fresh_sample"] for p in parts if "fresh_sample" in p), None),
            "level_A": dict(sorted(level.items())), "warm_start": dict(sorted(warm.items())),
            "models": dict(sorted(models.items())), "protocol": base["protocol"],
            "parts": [{"tag": p["tag"], "plan": p["plan"], "provenance": p.get("provenance")} for p in parts]}


VALID = ("S1", "S2", "S3", "S4")


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def decision_table(merged_by_seed: dict, tau: float = 1.) -> dict:
    """Protocol section 4 at sigma = .3: valid splits S1-S4, averaged over the available seeds.

    Arm values pool the arm's initialisations (and seeds); per-initialisation values are also reported.
    Paired SE: per seed, per E patient, average the (arm - level A) score differences over S1-S4 (and inits);
    SE of that mean, combined over independent seeds.
    """
    seeds = sorted(merged_by_seed)
    groups = {}
    for arm, inits in ARMS.items():
        groups[arm] = inits
        for init in inits:
            groups[f"{arm}/{init}"] = (init,)
    out = {"seeds": seeds, "valid_splits": list(VALID), "rows": {}, "level_A": {}}
    la = {s: merged_by_seed[s]["level_A"] for s in seeds}
    la_d2 = _mean([la[s][v]["D2_fresh"]["D2_fresh"] for s in seeds for v in VALID
                   if v in la[s] and la[s][v].get("D2_fresh")])
    la_err = _mean([abs(la[s][v]["effect_E"]["candidate_estimate"] - tau) for s in seeds for v in VALID if v in la[s]])
    la_range = None
    if len(seeds) >= 2:
        la_range = _mean([np.ptp([la[s][v]["effect_E"]["candidate_estimate"] for s in seeds]) for v in VALID
                          if all(v in la[s] for s in seeds)])
    la_gap = {v: _mean([la[s][v]["audit_S"]["gap"] for s in seeds if v in la[s]]) for v in VALID + ("S5",)}
    out["level_A"] = {"D2_fresh_mean_S1_S4": la_d2, "abs_error_mean_S1_S4": la_err,
                      "theta_range_over_seeds_mean_S1_S4": la_range, "audit_gap_by_split": la_gap,
                      "R4_S5_gap_exceeds_max_S1_S4": None if la_gap["S5"] is None or None in [la_gap[v] for v in VALID]
                      else la_gap["S5"] > max(la_gap[v] for v in VALID)}
    for label, inits in groups.items():
        arm = label.split("/")[0]
        recs = [(s, m) for s in seeds for m in merged_by_seed[s]["models"].values()
                if m["arm"] == arm and m["init"] in inits]
        valid = [(s, m) for s, m in recs if f"S{m['split']}" in VALID]
        if not valid:
            continue
        d2 = _mean([m["D2_fresh"]["D2_fresh"] for _, m in valid if m.get("D2_fresh")])
        err = _mean([abs(m["effect_E"]["candidate_estimate"] - tau) for _, m in valid])
        se_terms = []
        for s in seeds:
            diffs = [np.asarray(m["E_scores"]) - np.asarray(la[s][f"S{m['split']}"]["E_scores"])
                     for ss, m in valid if ss == s]
            if diffs:
                per_patient = np.mean(diffs, axis=0)
                se_terms.append(per_patient.std(ddof=1) / math.sqrt(len(per_patient)))
        paired_se = float(math.sqrt(sum(x ** 2 for x in se_terms)) / len(se_terms)) if se_terms else None
        ranges = []
        for v in VALID:
            thetas = [m["effect_E"]["candidate_estimate"] for _, m in valid if f"S{m['split']}" == v]
            if thetas:
                ranges.append(float(np.ptp(thetas)))
        arm_range = _mean(ranges)
        n_var = len({(s, m["init"]) for s, m in valid})
        gaps = {v: _mean([m["audit_S"]["gap"] for _, m in recs if f"S{m['split']}" == v]) for v in VALID + ("S5",)}
        r4 = None if gaps["S5"] is None or None in [gaps[v] for v in VALID] else gaps["S5"] > max(gaps[v] for v in VALID)
        complete = all(any(f"S{m['split']}" == v for _, m in valid) for v in VALID)
        row = {"inits": list(inits), "n_models_S1_S4": len(valid), "complete_S1_S4": complete,
               "D2_fresh_mean_S1_S4": d2, "D2_ratio_to_level_A": None if d2 is None or not la_d2 else d2 / la_d2,
               "R1_balance_pass": None if d2 is None or la_d2 is None else d2 <= 1.5 * la_d2,
               "abs_error_mean_S1_S4": err, "paired_SE_vs_level_A": paired_se,
               "R2_accuracy_pass": None if paired_se is None or la_err is None else err <= la_err + paired_se,
               "theta_range_mean_S1_S4": arm_range, "range_variants": n_var,
               "R3_stability_pass": None if la_range is None or n_var < 2 else arm_range <= 2 * la_range,
               "R3_status": "evaluable" if la_range is not None and n_var >= 2 else
               "pending: level-A seed range needs two seeds" if la_range is None else "single variant",
               "audit_gap_by_split": gaps, "R4_device_pass": r4,
               "audit_gap_mean_S1_S4": _mean([gaps[v] for v in VALID])}
        out["rows"][label] = row
    out["branches"] = decision_branches(out)
    return out


def decision_branches(table: dict) -> dict:
    rows, la = table["rows"], table["level_A"]

    def get(label, key):
        return rows.get(label, {}).get(key)

    fired, notes = [], []
    o3 = rows.get("O3", {})
    r1, r2, r3 = (get("O3", k) for k in ("R1_balance_pass", "R2_accuracy_pass", "R3_stability_pass"))
    if r1 is True and r2 is True and r3 is True:
        fired.append("ADOPT_O3: O3 passes R1-R3; drop the reader, balance pixels only; next experiment (SCM-1 mirror) uses O3")
    elif r1 is True and r2 is True and r3 is None:
        notes.append("O3 passes R1 and R2; R3 pending (needs the second seed), so ADOPT_O3 is provisional")
    if any(get(a, "R3_stability_pass") is None for a in ("O1", "O2", "O3")):
        notes.append("R3 not evaluable for some arms with the available seeds; DROP_ALTERNATING cannot fire yet")
    o3_gap, la_gap_mean = o3.get("audit_gap_mean_S1_S4"), _mean([la["audit_gap_by_split"][v] for v in VALID])
    if get("O3", "R1_balance_pass") is False:
        large = o3_gap is not None and la_gap_mean is not None and o3_gap > 1.5 * la_gap_mean
        notes.append(f"O3 fails R1; ladder-3 S audit gap mean {o3_gap} vs level A {la_gap_mean} "
                     f"({'large' if large else 'not large'} at 1.5x)")
        if large:
            fired.append("KERNEL_BLIND_SPOT: widen f(T) (32x32, more features) and rerun O3")
    if get("O1", "R3_stability_pass") is False and get("O2", "R3_stability_pass") is False and \
            get("O3", "R3_stability_pass") is True:
        fired.append("DROP_ALTERNATING: inner loop causes instability")
    if all(get(a, "R2_accuracy_pass") is False for a in ("O1", "O2", "O3")):
        fired.append("KEEP_READER: balance alone from pixels does not follow the linear solution; record as first "
                     "negative result for the universal representer search")
    r4 = [rows[label]["R4_device_pass"] for label in rows if "/" in label]
    if any(v is None for v in r4):
        notes.append("R4 missing for some arm/init (S5 or S1-S4 not run)")
    if any(v is False for v in r4):
        fired.append("R4_FAIL: audit predictor problem; raise the device coefficient to 1.5 and recheck")
    if not fired:
        fired.append("NO_PRESPECIFIED_BRANCH: none of the protocol section-4 branches applies with the available evidence")
    return {"fired": fired, "notes": notes}
