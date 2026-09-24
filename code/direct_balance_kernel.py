"""Direct image balancing, last attempt: kernel critics that are differentiated through
(PRD in the review of 2026-09-21 evening, "직접 이미지 균형학습의 커널 critic 최종 진단").

What changes and what does not.  The CNN, its objective and its update rule are the same as before: from a random
start, all weights move to lower the clipped Brier gap on the whole representation fold.  Only the two critics
that MEASURE that gap during training are replaced.  They were small neural networks fitted by a few dozen
Adam steps, which memorised the fold and handed the CNN a gradient about their own memory.  Here each critic is a
ridge regression on fixed random Fourier features, solved exactly at every update, and the CNN's gradient is
taken through the solve itself: the loss is a function of the representation Z alone, so d(loss)/dZ includes
how the fitted coefficients would move if Z moved.

    base critic       q0 = clip(F0(Z) c0),         F0(Z) = [1, RFF_D(Z)]
    augmented critic  q1 = clip(F1(Z, W_S) c1),    F1 = [F0(Z), RFF_D^joint(Z, W_S)]
    fit               c_b = argmin mean (A - F_b c)^2 + lambda ||c_{-intercept}||^2, by one linear solve
    nesting           c1 = [c0, 0] reproduces q0 exactly; if the augmented fit's plain Brier risk on the fit rows
                      is worse than the base's, the lifted base is used and the gap is zero
    loss              L = max(R0 - R1, 0) with R_b the plain Brier risk on the whole fold

The joint feature is a Gaussian kernel on Z together with the held-out image's raw pixels, which is what makes
q1 able to use the interaction of the two rather than an additive image term.  The pixel part of the joint
projection is fixed by the pixels, so it is computed once per row set and reused at every update.

Everything numeric that touches the regression is float64 on the CPU.  The CNN runs on the device.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import direct_balance_data as db
import direct_balance_net as dn

D_FEATURES = 128
LAMBDA, EPS = 1e-3, 1e-8
SIGMA_Z = 2.0
SD_FLOOR = 1e-3
CLIP = (0.1, 0.9)
PAIRS_FOR_WIDTH = 256
ALPHA, N_SPLITS = 0.05, 30
Z_CRIT = None


def z_crit() -> float:
    global Z_CRIT
    if Z_CRIT is None:
        from scipy.stats import norm
        Z_CRIT = float(norm.ppf(1.0 - ALPHA / N_SPLITS))
    return Z_CRIT


# ----------------------------------------------------------------------------- the fixed random features

class Features:
    """Frequencies and phases drawn once from their own generator and never redrawn.

    z_dim   width of Z = (X, h)
    n_pix   number of raw pixel values in the held-out images of one row
    sigma_w kernel width of the pixel part, on the scale sqrt(||w - w'||^2 / n_pix)
    """

    def __init__(self, z_dim: int, n_pix: int, sigma_w: float, seed: int, dev, d: int = D_FEATURES) -> None:
        if not (np.isfinite(sigma_w) and sigma_w > 0):
            raise RuntimeError(f"pixel kernel width {sigma_w!r} is not a positive finite number; stopping")
        rng = np.random.default_rng(seed)
        self.d, self.z_dim, self.n_pix, self.sigma_w, self.sigma_z = d, z_dim, n_pix, float(sigma_w), SIGMA_Z
        self.omega_z = torch.as_tensor(rng.standard_normal((d, z_dim)), dtype=torch.float64)
        self.phase_z = torch.as_tensor(rng.uniform(0, 2 * math.pi, d), dtype=torch.float64)
        self.omega_joint_z = torch.as_tensor(rng.standard_normal((d, z_dim)), dtype=torch.float64)
        self.omega_joint_w = torch.as_tensor(rng.standard_normal((d, n_pix)), dtype=torch.float32, device=dev)
        self.phase_joint = torch.as_tensor(rng.uniform(0, 2 * math.pi, d), dtype=torch.float64)
        self.seed = seed

    def pixel_projection(self, data: dn.Batches, rows: np.ndarray, held: np.ndarray, batch: int = 512) -> torch.Tensor:
        """The pixel half of the joint feature's argument, (n, d) float64 on the CPU.  Pixels are fixed, so this
        is computed once per row set."""
        rows = np.asarray(rows)
        scale = 1.0 / (math.sqrt(self.n_pix) * self.sigma_w)
        out = []
        with torch.no_grad():
            for s in range(0, len(rows), batch):
                r = rows[s:s + batch]
                w = torch.cat([p.reshape(len(r), -1) for p in data.pixels(r, held)], dim=1)
                out.append((w * scale) @ self.omega_joint_w.T)
        return torch.cat(out).to("cpu", torch.float64)

    def f0(self, z_norm: torch.Tensor) -> torch.Tensor:
        n = z_norm.shape[0]
        rff = math.sqrt(2.0 / self.d) * torch.cos((z_norm / self.sigma_z) @ self.omega_z.T + self.phase_z)
        return torch.cat([torch.ones(n, 1, dtype=torch.float64), rff], dim=1)

    def f1(self, z_norm: torch.Tensor, pw: torch.Tensor) -> torch.Tensor:
        joint = math.sqrt(2.0 / self.d) * torch.cos((z_norm / self.sigma_z) @ self.omega_joint_z.T + pw
                                                    + self.phase_joint)
        return torch.cat([self.f0(z_norm), joint], dim=1)

    def record(self) -> dict:
        return {"d": self.d, "z_dim": self.z_dim, "n_pix": self.n_pix, "sigma_z": self.sigma_z,
                "sigma_w": self.sigma_w, "seed": self.seed,
                "omega_z_sha256": _sha(self.omega_z), "omega_joint_z_sha256": _sha(self.omega_joint_z),
                "omega_joint_w_sha256": _sha(self.omega_joint_w.cpu()), "phase_z_sha256": _sha(self.phase_z),
                "phase_joint_sha256": _sha(self.phase_joint)}


def _sha(t: torch.Tensor) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()


def pixel_width(data: dn.Batches, rows: np.ndarray, held: np.ndarray, seed: int) -> dict:
    """The median over PAIRS_FOR_WIDTH random pairs of rows of sqrt(||w - w'||^2 / n_pix), chosen once."""
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows)
    i, j = rng.integers(0, len(rows), PAIRS_FOR_WIDTH), rng.integers(0, len(rows), PAIRS_FOR_WIDTH)
    with torch.no_grad():
        wi = torch.cat([p.reshape(PAIRS_FOR_WIDTH, -1) for p in data.pixels(rows[i], held)], dim=1)
        wj = torch.cat([p.reshape(PAIRS_FOR_WIDTH, -1) for p in data.pixels(rows[j], held)], dim=1)
        n_pix = wi.shape[1]
        dist = torch.sqrt(((wi - wj) ** 2).sum(1) / n_pix).cpu().numpy().astype(np.float64)
    sigma = float(np.median(dist))
    return {"sigma_w": sigma, "n_pix": int(n_pix), "pairs": PAIRS_FOR_WIDTH, "seed": seed,
            "distance_quantiles": {q: float(np.quantile(dist, q)) for q in (0.05, 0.5, 0.95)},
            "finite": bool(np.isfinite(dist).all())}


# ----------------------------------------------------------------------------- normalisation of h

def h_stats(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and floored sd of the h columns (everything after X), kept in the graph when z requires grad."""
    h = z[:, 2:]
    return h.mean(0), torch.clamp(h.std(0, unbiased=False), min=SD_FLOOR)


def normalise(z: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return torch.cat([z[:, :2], (z[:, 2:] - mu) / sd], dim=1)


# ----------------------------------------------------------------------------- the two critics, by one solve each

def ridge_solve(f: torch.Tensor, a: torch.Tensor, lam: float = LAMBDA, eps: float = EPS):
    n, p = f.shape
    penalty = torch.ones(p, dtype=torch.float64)
    penalty[0] = 0.0                                     # the intercept is not penalised
    m = f.T @ f / n + torch.diag(lam * penalty) + eps * torch.eye(p, dtype=torch.float64)
    rhs = f.T @ a / n
    c = torch.linalg.solve(m, rhs)
    with torch.no_grad():
        residual = float((m @ c - rhs).norm() / max(float(rhs.norm()), 1e-300))
        cond = float(torch.linalg.cond(m))
    return c, {"cond": cond, "solve_residual": residual, "finite": bool(torch.isfinite(c).all())}


def critics(z_norm: torch.Tensor, pw: torch.Tensor, a: torch.Tensor, feats: Features, nest: bool = True):
    """Fit q0 and q1 on the rows given (in the graph), with the nesting rule of the sealed learner.

    Returns the two coefficient vectors and the plain Brier risks on these rows, plus diagnostics.  `q_b` is the
    clipped prediction; the fraction outside the clip is recorded and is NOT taken as an overlap check.
    """
    f0, f1 = feats.f0(z_norm), feats.f1(z_norm, pw)
    c0, d0 = ridge_solve(f0, a)
    c1, d1 = ridge_solve(f1, a)
    if not (d0["finite"] and d1["finite"]):
        raise RuntimeError("a critic's linear solve was not finite; stopping without changing the regularisation")
    raw0, raw1 = f0 @ c0, f1 @ c1
    q0, q1 = torch.clamp(raw0, *CLIP), torch.clamp(raw1, *CLIP)
    r0, r1 = ((a - q0) ** 2).mean(), ((a - q1) ** 2).mean()
    lifted = False
    if nest and r1.detach().item() > r0.detach().item():  # the augmented fit could not beat the base: use the base
        c1 = torch.cat([c0, torch.zeros(feats.d, dtype=torch.float64)])
        raw1 = f1 @ c1
        q1 = torch.clamp(raw1, *CLIP)
        r1 = ((a - q1) ** 2).mean()
        lifted = True
    info = {"risk_base": float(r0), "risk_augmented": float(r1), "lifted_base_used": lifted,
            "outside_clip_base": float(((raw0 < CLIP[0]) | (raw0 > CLIP[1])).double().mean()),
            "outside_clip_augmented": float(((raw1 < CLIP[0]) | (raw1 > CLIP[1])).double().mean()),
            "cond_base": d0["cond"], "cond_augmented": d1["cond"],
            "solve_residual_base": d0["solve_residual"], "solve_residual_augmented": d1["solve_residual"]}
    return {"c0": c0, "c1": c1, "q0": q0, "q1": q1, "r0": r0, "r1": r1, "f0": f0, "f1": f1, "info": info}


def training_loss(z: torch.Tensor, pw: torch.Tensor, a: torch.Tensor, feats: Features):
    """L = max(R0 - R1, 0) on the whole fold, with the critics refitted inside the graph.

    `z` is the raw representation (X standardised, h as the CNN made it).  The h normalisation uses this fold's
    own statistics and stays in the graph.  Nothing is clipped per batch: the whole-fold gap is clipped once.
    """
    mu, sd = h_stats(z)
    fit = critics(normalise(z, mu, sd), pw, a, feats)
    signed = fit["r0"] - fit["r1"]
    loss = torch.clamp(signed, min=0.0)
    return loss, dict(fit["info"], signed_gap=float(signed), clipped_gap=float(loss),
                      h_mean=mu.detach().tolist(), h_sd=sd.detach().tolist())


# ----------------------------------------------------------------------------- the update, in two stages

def representation_all(rep: dn.Representation, data: dn.Batches, rows: np.ndarray, kept: np.ndarray,
                       batch: int = 512) -> torch.Tensor:
    return dn.representation_values(rep, data, rows, kept, grad=False, batch=batch).to("cpu", torch.float64)


def two_stage_step(rep: dn.Representation, data: dn.Batches, rows: np.ndarray, kept: np.ndarray,
                   pw: torch.Tensor, a: torch.Tensor, feats: Features, opt, batch: int = 512) -> dict:
    """One update of the CNN.

    1. Z for the whole fold, without a graph (the images never sit in memory together).
    2. The loss as a function of Z alone, differentiated to dL/dZ, regression solves included.
    3. The CNN again, batch by batch, each batch's output pushed backwards with its slice of dL/dZ.
    The step is taken only when the fold's signed gap is positive.
    """
    rows = np.asarray(rows)
    z_leaf = representation_all(rep, data, rows, kept, batch).clone().requires_grad_(True)
    loss, info = training_loss(z_leaf, pw, a, feats)
    stepped, grad_norm, first_conv_grad = False, 0.0, 0.0
    if info["signed_gap"] > 0:
        loss.backward()
        upstream = z_leaf.grad.to(data.dev, torch.float32)
        rep.train()
        opt.zero_grad(set_to_none=True)
        for s in range(0, len(rows), batch):
            r = rows[s:s + batch]
            idx = torch.as_tensor(r, device=data.dev)
            z_b = rep(data.pixels(r, kept), data.x[idx])
            z_b.backward(upstream[s:s + len(r)])
        grads = [(n, p.grad) for n, p in rep.named_parameters() if p.grad is not None]
        grad_norm = float(torch.sqrt(sum((g_ ** 2).sum() for _, g_ in grads)))
        first_conv_grad = float(sum(g_.norm() for n, g_ in grads if n.endswith("towers.0.body.0.weight")))
        opt.step()
        stepped = True
        rep.eval()
    return dict(info, stepped=stepped, grad_norm=grad_norm, first_conv_grad_norm=first_conv_grad,
                dLdZ_norm=float(z_leaf.grad.norm()) if z_leaf.grad is not None else 0.0)


def full_autograd_step_gradients(rep: dn.Representation, data: dn.Batches, rows: np.ndarray, kept: np.ndarray,
                                 pw: torch.Tensor, a: torch.Tensor, feats: Features) -> dict:
    """For the test only: the same gradient with the whole graph kept, so the two-stage version can be checked."""
    rows = np.asarray(rows)
    rep.train()
    rep.zero_grad(set_to_none=True)
    idx = torch.as_tensor(rows, device=data.dev)
    z = rep(data.pixels(rows, kept), data.x[idx]).to("cpu", torch.float64)
    loss, _ = training_loss(z, pw, a, feats)
    loss.backward()
    return {n: p.grad.detach().clone() for n, p in rep.named_parameters() if p.grad is not None}


# ----------------------------------------------------------------------------- the kernel device as a check

def kernel_check(z: torch.Tensor, pw: torch.Tensor, a: torch.Tensor, feats: Features, mu: torch.Tensor,
                 sd: torch.Tensor, threshold: float) -> dict:
    """Split-half, both directions: critics fitted on one half, per-row d = (q0 - A)^2 - (q1 - A)^2 on the other.
    gap = mean d, se = sd(d)/sqrt(n), upper = max(gap, 0) + z se, lower = gap - z se.  `mu`, `sd` are the h
    statistics of the rows the candidate was fitted on, as the PRD prescribes."""
    with torch.no_grad():
        zn = normalise(z, mu, sd)
        n = zn.shape[0]
        half = torch.arange(n) % 2 == 0
        d = torch.empty(n, dtype=torch.float64)
        q0_all = torch.empty(n, dtype=torch.float64)
        infos = []
        for fit_mask, eval_mask in ((half, ~half), (~half, half)):
            fit = critics(zn[fit_mask], pw[fit_mask], a[fit_mask], feats)
            infos.append(fit["info"])
            f0e, f1e = feats.f0(zn[eval_mask]), feats.f1(zn[eval_mask], pw[eval_mask])
            q0 = torch.clamp(f0e @ fit["c0"], *CLIP)
            q1 = torch.clamp(f1e @ fit["c1"], *CLIP)
            ae = a[eval_mask]
            d[eval_mask] = (q0 - ae) ** 2 - (q1 - ae) ** 2
            q0_all[eval_mask] = q0
        gap, se = float(d.mean()), float(d.std(unbiased=True) / math.sqrt(n))
        z_ = z_crit()
        return {"gap": gap, "se": se, "upper": max(gap, 0.0) + z_ * se, "lower": gap - z_ * se,
                "passes_upper": bool(max(gap, 0.0) + z_ * se <= threshold),
                "clearly_imbalanced": bool(gap - z_ * se > threshold),
                "risk_base": float(((q0_all - a) ** 2).mean()), "risk_augmented": float(((q0_all - a) ** 2).mean() - gap),
                "fits": infos, "n": int(n), "finite": bool(torch.isfinite(d).all()),
                "rows_d": d.numpy()}
