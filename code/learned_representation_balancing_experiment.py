#!/usr/bin/env python3
"""Learned-representation single-proxy balancing experiment.

This experiment implements the approximate theory in
`Single-Proximal-Balancing-approx-theory-v2026-08-17.md`.

Design contract (section 8 of the theory document):
  1. h(X) is LEARNED by minimizing the stratified conditional kernel
     discrepancy \\hat{delta}^2 between treatment arms of V=(X,W) given
     H=h(X).  W enters only the objective, never the adjustment set.
  2. Sample splitting: h is trained on D1; \\hat{delta}, \\hat{tau} and all
     diagnostics are computed on an independent D2.
  3. Chain validation with oracle access to U:
       (a) observable certificate  \\hat{delta}  vs  |tau_hat - tau|,
       (b) per-stratum MMD_V (observed)  vs  d_XU (latent, oracle),
           whose envelope slope is the empirical inverse-stability
           modulus omega_K; it steepens as channel noise sigma_W grows,
       (c) per-stratum L_Y * d_XU  vs  oracle stratum bias |B_m|,
           which verifies Theorem 3 (all points below the y=x line).
  4. Baselines: naive, X-adjust (g-formula), (X,W)-adjust (the old
     "balance W directly" surrogate), random-h stratification (the old
     hard-coded-representation surrogate), oracle (X,U)-adjust.
  5. Held-out diagnostic: within-stratum permutation p-value for the
     residual dependence A vs (X,W) given H.  It is a diagnostic, not a
     bias measure.
  6. The training curve plateaus above zero: the empirical trace of the
     exact-balance impossibility (Proposition 0).

DGP (genuine unmeasured confounding):
  X ~ N(0, I_4)
  U = rho_xu * g(X) + sqrt(1-rho_xu^2) * eps_U          (eps_U unmeasured)
  A ~ Bernoulli(sigmoid(0.6*t(X) + gamma_U * U))
  W = U + 0.5*(X2 + 0.5*X4) + sigma_W * eps_W           (A-invariant channel)
  Y = tau*A + X1 + 0.5*X2*X3 + beta_U*U + 0.5*eps_Y     (tau = 1.0)

Outputs:
  results/learned_representation_balancing_summary.json
  results/learned_representation_balancing_runs.csv
  figures/learned_representation_chain_validation.(png|pdf)
  figures/learned_representation_method_comparison.(png|pdf)
  figures/learned_representation_training_curves.(png|pdf)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

TRUE_TAU = 1.0
BETA_U = 1.5
RHO_XU = 0.5
W_SCALE = 2.0  # weight of the W coordinate inside the observation kernel


# --------------------------------------------------------------------------
# DGP
# --------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def f_outcome_x(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + 0.5 * x[:, 1] * x[:, 2]


def simulate(rng: np.random.Generator, n: int, sigma_w: float, gamma_u: float) -> dict[str, np.ndarray]:
    x = rng.standard_normal((n, 4))
    g_x = np.tanh(x[:, 0]) + 0.5 * x[:, 1]
    g_x = g_x / math.sqrt(1.0 + 0.25)  # roughly unit scale
    u = RHO_XU * g_x + math.sqrt(1.0 - RHO_XU**2) * rng.standard_normal(n)
    t_x = x[:, 0] - 0.5 * x[:, 2] + 0.5 * (x[:, 1] ** 2 - 1.0)
    a = rng.binomial(1, sigmoid(0.6 * t_x + gamma_u * u)).astype(float)
    w = u + 0.5 * (x[:, 1] + 0.5 * x[:, 3]) + sigma_w * rng.standard_normal(n)
    y = TRUE_TAU * a + f_outcome_x(x) + BETA_U * u + 0.5 * rng.standard_normal(n)
    return {"X": x, "U": u, "A": a, "W": w, "Y": y}


# --------------------------------------------------------------------------
# Kernel utilities
# --------------------------------------------------------------------------

def build_v_features(x: np.ndarray, w: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    v = np.column_stack([x, w])
    v = (v - mean) / std
    v[:, -1] *= W_SCALE
    return v


def median_bandwidth(v: np.ndarray, rng: np.random.Generator, cap: int = 1200) -> float:
    idx = rng.choice(len(v), size=min(cap, len(v)), replace=False)
    sub = v[idx]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[np.triu_indices_from(d2, k=1)])
    return float(np.sqrt(max(med, 1e-12) / 2.0))


def rbf_gram(v1: np.ndarray, v2: np.ndarray, bw: float) -> np.ndarray:
    d2 = ((v1[:, None, :] - v2[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2.0 * bw**2))


# --------------------------------------------------------------------------
# Representation model and training objective
# --------------------------------------------------------------------------

class StratumNet(torch.nn.Module):
    def __init__(self, d_in: int, m_strata: int, hidden: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, m_strata),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def soft_delta_sq(soft: torch.Tensor, a: torch.Tensor, gram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft stratified conditional kernel discrepancy (theory eq. 3.3).

    soft: (n, M) soft assignments; a: (n,) treatment; gram: (n, n) kernel.
    Returns (delta_sq, p_hat, pi_hat).
    """
    n = soft.shape[0]
    p_hat = soft.mean(0).clamp_min(1e-8)                      # (M,)
    s_hat = (soft * a[:, None]).mean(0)                       # (M,)
    pi_hat = (s_hat / p_hat).clamp(1e-4, 1.0 - 1e-4)          # (M,)
    c = soft * (a[:, None] - pi_hat[None, :])                 # (n, M)
    kc = gram @ c                                             # (n, M)
    quad = (c * kc).sum(0)                                    # (M,)
    delta_sq = (quad / (n**2 * p_hat)).sum()
    return delta_sq, p_hat, pi_hat


def hard_delta_sq(strata: np.ndarray, a: np.ndarray, gram: np.ndarray, m_strata: int) -> float:
    total = 0.0
    n = len(a)
    for m in range(m_strata):
        mask = strata == m
        nm = int(mask.sum())
        if nm == 0:
            continue
        am = a[mask]
        if am.sum() in (0, nm):
            continue  # degenerate stratum: covariance identically zero
        pi_m = am.mean()
        c = np.zeros(n)
        c[mask] = am - pi_m
        total += float(c @ gram @ c) / (n**2 * (nm / n))
    return total


def train_representation(
    data: dict[str, np.ndarray],
    gram: np.ndarray,
    m_strata: int,
    steps: int,
    seed: int,
    lr: float = 3e-3,
    overlap_floor: float = 0.09,
    lambda_overlap: float = 1.0,
    lambda_mass: float = 1.0,
) -> tuple[StratumNet, list[float]]:
    torch.manual_seed(seed)
    net = StratumNet(d_in=4, m_strata=m_strata)
    x_t = torch.tensor(data["Xs"], dtype=torch.float32)
    a_t = torch.tensor(data["A"], dtype=torch.float32)
    k_t = torch.tensor(gram, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mass_floor = 0.5 / m_strata
    curve: list[float] = []
    for step in range(steps):
        temp = 1.0 * (0.25 / 1.0) ** (step / max(steps - 1, 1))
        soft = torch.softmax(net(x_t) / temp, dim=1)
        dsq, p_hat, pi_hat = soft_delta_sq(soft, a_t, k_t)
        pen_overlap = (p_hat * torch.relu(overlap_floor - pi_hat * (1.0 - pi_hat))).sum()
        pen_mass = torch.relu(mass_floor - p_hat).sum()
        loss = dsq + lambda_overlap * pen_overlap + lambda_mass * pen_mass
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(float(dsq.detach()))
    return net, curve


def assign_strata(net: StratumNet, xs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = net(torch.tensor(xs, dtype=torch.float32))
    return logits.argmax(1).numpy()


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def stratified_tau(strata: np.ndarray, a: np.ndarray, y: np.ndarray, m_strata: int) -> tuple[float, float]:
    """Within-stratum difference of means; returns (tau_hat, dropped_mass)."""
    n = len(a)
    est, used = 0.0, 0.0
    for m in range(m_strata):
        mask = strata == m
        nm = int(mask.sum())
        if nm == 0:
            continue
        am, ym = a[mask], y[mask]
        n1 = int(am.sum())
        if n1 < 2 or nm - n1 < 2:
            continue
        est += (nm / n) * (ym[am == 1].mean() - ym[am == 0].mean())
        used += nm / n
    if used == 0.0:
        return float("nan"), 1.0
    return est / used, 1.0 - used


def g_formula_tau(
    features_tr: np.ndarray, a_tr: np.ndarray, y_tr: np.ndarray, features_ev: np.ndarray, seed: int
) -> float:
    """Arm-wise gradient boosting outcome regression, averaged over eval covariates."""
    preds = []
    for arm in (1, 0):
        mask = a_tr == arm
        model = GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.06, random_state=seed
        )
        model.fit(features_tr[mask], y_tr[mask])
        preds.append(model.predict(features_ev))
    return float(np.mean(preds[0] - preds[1]))


# --------------------------------------------------------------------------
# Oracle chain diagnostics (latent side; simulation-only)
# --------------------------------------------------------------------------

def latent_features(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    cols = [x[:, i] for i in range(4)]
    cols += [x[:, i] * x[:, j] for i in range(4) for j in range(i, 4)]
    cols.append(u)
    return np.column_stack(cols)


def latent_feature_coeffs(std: np.ndarray) -> np.ndarray:
    """Coefficients of b(x,u) = X1 + 0.5*X2*X3 + beta_U*u in standardized features."""
    names = [f"x{i}" for i in range(4)] + [f"x{i}x{j}" for i in range(4) for j in range(i, 4)] + ["u"]
    raw = dict.fromkeys(names, 0.0)
    raw["x0"] = 1.0
    raw["x1x2"] = 0.5
    raw["u"] = BETA_U
    return np.array([raw[k] for k in names]) * std


def stratum_chain_stats(
    strata: np.ndarray,
    a: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    v_feat: np.ndarray,
    bw: float,
    m_strata: int,
) -> tuple[list[dict[str, float]], float]:
    """Per-stratum: observed MMD_V, latent d_XU, oracle stratum bias B_m, L_Y."""
    psi_raw = latent_features(x, u)
    psi_mean, psi_std = psi_raw.mean(0), psi_raw.std(0) + 1e-12
    psi = (psi_raw - psi_mean) / psi_std
    coeff = latent_feature_coeffs(psi_std)
    l_y = float(np.linalg.norm(coeff))
    b_val = psi_raw @ (coeff / psi_std)  # equals b(x,u) up to an additive constant
    rows = []
    n = len(a)
    for m in range(m_strata):
        mask = strata == m
        nm = int(mask.sum())
        n1 = int(a[mask].sum())
        if nm < 8 or n1 < 4 or nm - n1 < 4:
            continue
        t_mask = mask & (a == 1)
        c_mask = mask & (a == 0)
        # observed per-stratum MMD_V (biased V-statistic)
        k11 = rbf_gram(v_feat[t_mask], v_feat[t_mask], bw).mean()
        k00 = rbf_gram(v_feat[c_mask], v_feat[c_mask], bw).mean()
        k10 = rbf_gram(v_feat[t_mask], v_feat[c_mask], bw).mean()
        mmd_v = math.sqrt(max(k11 + k00 - 2.0 * k10, 0.0))
        # latent d_XU as explicit-feature IPM
        d_xu = float(np.linalg.norm(psi[t_mask].mean(0) - psi[c_mask].mean(0)))
        # oracle stratum bias  B_m = int b d(Q1m - Q0m)
        b_m = float(b_val[t_mask].mean() - b_val[c_mask].mean())
        rows.append(
            {
                "p_m": nm / n,
                "pi_m": n1 / nm,
                "mmd_v": mmd_v,
                "d_xu": d_xu,
                "abs_bias_m": abs(b_m),
                "signed_bias_m": b_m,
            }
        )
    return rows, l_y


def permutation_pvalue(
    strata: np.ndarray, a: np.ndarray, gram: np.ndarray, m_strata: int, rng: np.random.Generator, n_perm: int = 200
) -> float:
    obs = hard_delta_sq(strata, a, gram, m_strata)
    count = 0
    for _ in range(n_perm):
        a_perm = a.copy()
        for m in range(m_strata):
            mask = np.where(strata == m)[0]
            a_perm[mask] = rng.permutation(a_perm[mask])
        if hard_delta_sq(strata, a_perm, gram, m_strata) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


# --------------------------------------------------------------------------
# One replication
# --------------------------------------------------------------------------

@dataclass
class RunRecord:
    sigma_w: float
    gamma_u: float
    rep: int
    estimates: dict[str, float] = field(default_factory=dict)
    delta_hat_eval: float = 0.0
    delta_hat_random: float = 0.0
    delta_sq_train_final: float = 0.0
    weighted_mmd_v: float = 0.0
    weighted_d_xu: float = 0.0
    l_y: float = 0.0
    perm_pvalue: float = 0.0
    dropped_mass: float = 0.0
    min_overlap: float = 0.0
    theorem3_lhs: float = 0.0
    theorem3_rhs: float = 0.0
    stratum_rows: list[dict[str, float]] = field(default_factory=list)
    train_curve: list[float] = field(default_factory=list)


def prepare_run(sigma_w: float, gamma_u: float, rep: int, n: int) -> tuple[int, np.random.Generator, dict[str, np.ndarray]]:
    """Deterministic data preparation shared by the full run and figure-only reruns."""
    seed = 20260817 + 7919 * rep + int(1000 * sigma_w) + int(10 * gamma_u)
    rng = np.random.default_rng(seed)
    data = simulate(rng, n, sigma_w, gamma_u)
    half = n // 2
    idx = rng.permutation(n)
    tr, ev = idx[:half], idx[half:]

    x_tr, x_ev = data["X"][tr], data["X"][ev]
    w_tr, w_ev = data["W"][tr], data["W"][ev]

    x_mean, x_std = x_tr.mean(0), x_tr.std(0) + 1e-12
    v_raw_tr = np.column_stack([x_tr, w_tr])
    v_mean, v_std = v_raw_tr.mean(0), v_raw_tr.std(0) + 1e-12
    v_tr = build_v_features(x_tr, w_tr, v_mean, v_std)
    v_ev = build_v_features(x_ev, w_ev, v_mean, v_std)
    bw = median_bandwidth(v_tr, rng)

    prep = {
        "x_tr": x_tr, "x_ev": x_ev, "w_tr": w_tr, "w_ev": w_ev,
        "a_tr": data["A"][tr], "a_ev": data["A"][ev],
        "y_tr": data["Y"][tr], "y_ev": data["Y"][ev],
        "u_tr": data["U"][tr], "u_ev": data["U"][ev],
        "xs_tr": (x_tr - x_mean) / x_std, "xs_ev": (x_ev - x_mean) / x_std,
        "v_tr": v_tr, "v_ev": v_ev, "bw": np.array([bw]),
        "gram_tr": rbf_gram(v_tr, v_tr, bw), "gram_ev": rbf_gram(v_ev, v_ev, bw),
    }
    return seed, rng, prep


def train_only(sigma_w: float, gamma_u: float, rep: int, n: int, m_strata: int, steps: int) -> RunRecord:
    seed, _, prep = prepare_run(sigma_w, gamma_u, rep, n)
    rec = RunRecord(sigma_w=sigma_w, gamma_u=gamma_u, rep=rep)
    _, curve = train_representation({"Xs": prep["xs_tr"], "A": prep["a_tr"]}, prep["gram_tr"], m_strata, steps, seed)
    rec.train_curve = curve[:: max(1, len(curve) // 60)]
    rec.delta_sq_train_final = float(np.mean(curve[-10:]))
    return rec


def run_once(sigma_w: float, gamma_u: float, rep: int, n: int, m_strata: int, steps: int) -> RunRecord:
    seed, rng, prep = prepare_run(sigma_w, gamma_u, rep, n)
    x_tr, x_ev = prep["x_tr"], prep["x_ev"]
    w_tr, w_ev = prep["w_tr"], prep["w_ev"]
    a_tr, a_ev = prep["a_tr"], prep["a_ev"]
    y_tr, y_ev = prep["y_tr"], prep["y_ev"]
    u_ev = prep["u_ev"]
    xs_tr, xs_ev = prep["xs_tr"], prep["xs_ev"]
    v_ev = prep["v_ev"]
    bw = float(prep["bw"][0])
    gram_tr, gram_ev = prep["gram_tr"], prep["gram_ev"]

    rec = RunRecord(sigma_w=sigma_w, gamma_u=gamma_u, rep=rep)

    # --- learned representation ---
    net, curve = train_representation({"Xs": xs_tr, "A": a_tr}, gram_tr, m_strata, steps, seed)
    rec.train_curve = curve[:: max(1, len(curve) // 60)]
    rec.delta_sq_train_final = float(np.mean(curve[-10:]))
    strata_ev = assign_strata(net, xs_ev)
    rec.delta_hat_eval = math.sqrt(hard_delta_sq(strata_ev, a_ev, gram_ev, m_strata))
    tau_h, dropped = stratified_tau(strata_ev, a_ev, y_ev, m_strata)
    rec.estimates["learned_h"] = tau_h
    rec.dropped_mass = dropped

    pis = [a_ev[strata_ev == m].mean() for m in range(m_strata) if (strata_ev == m).sum() >= 8]
    rec.min_overlap = float(min(p * (1 - p) for p in pis)) if pis else 0.0

    # --- random (untrained) representation: hard-coded surrogate ---
    torch.manual_seed(seed + 555)
    net_rand = StratumNet(d_in=4, m_strata=m_strata)
    strata_rand = assign_strata(net_rand, xs_ev)
    rec.delta_hat_random = math.sqrt(hard_delta_sq(strata_rand, a_ev, gram_ev, m_strata))
    rec.estimates["random_h"], _ = stratified_tau(strata_rand, a_ev, y_ev, m_strata)

    # --- baselines ---
    rec.estimates["naive"] = float(y_ev[a_ev == 1].mean() - y_ev[a_ev == 0].mean())
    rec.estimates["x_adjust"] = g_formula_tau(x_tr, a_tr, y_tr, x_ev, seed)
    rec.estimates["xw_adjust"] = g_formula_tau(
        np.column_stack([x_tr, w_tr]), a_tr, y_tr, np.column_stack([x_ev, w_ev]), seed
    )
    rec.estimates["oracle_xu"] = g_formula_tau(
        np.column_stack([x_tr, prep["u_tr"]]), a_tr, y_tr, np.column_stack([x_ev, u_ev]), seed
    )

    # --- chain diagnostics on eval split ---
    rows, l_y = stratum_chain_stats(strata_ev, a_ev, x_ev, u_ev, v_ev, bw, m_strata)
    rec.stratum_rows = rows
    rec.l_y = l_y
    rec.weighted_mmd_v = float(sum(r["p_m"] * r["mmd_v"] for r in rows))
    rec.weighted_d_xu = float(sum(r["p_m"] * r["d_xu"] for r in rows))
    rec.theorem3_lhs = float(abs(sum(r["p_m"] * r["signed_bias_m"] for r in rows)))
    rec.theorem3_rhs = float(l_y * sum(r["p_m"] * r["d_xu"] for r in rows))
    rec.perm_pvalue = permutation_pvalue(strata_ev, a_ev, gram_ev, m_strata, rng)
    return rec


# --------------------------------------------------------------------------
# Sweep, summaries, figures
# --------------------------------------------------------------------------

METHOD_LABELS = {
    "naive": "Naive",
    "x_adjust": "Adjust $X$ (g-formula)",
    "xw_adjust": "Adjust $(X,W)$",
    "random_h": "Random $h$ strata",
    "learned_h": "Learned $h$ strata (ours)",
    "oracle_xu": "Oracle adjust $(X,U)$",
}

SIGMA_COLORS = {0.3: "#2563eb", 1.0: "#d97706", 2.0: "#dc2626"}


def sweep(n: int, m_strata: int, steps: int, reps: int, sigmas: list[float], gammas: list[float]) -> list[RunRecord]:
    records = []
    t0 = time.time()
    total = len(sigmas) * len(gammas) * reps
    done = 0
    for sw in sigmas:
        for gu in gammas:
            for rep in range(reps):
                records.append(run_once(sw, gu, rep, n, m_strata, steps))
                done += 1
                if done % 5 == 0 or done == total:
                    print(f"[{done}/{total}] elapsed {time.time() - t0:.0f}s", flush=True)
    return records


def summarize(records: list[RunRecord]) -> dict:
    summary: dict = {"config": {}, "methods": {}}
    keys = sorted({(r.sigma_w, r.gamma_u) for r in records})
    for sw, gu in keys:
        rs = [r for r in records if r.sigma_w == sw and r.gamma_u == gu]
        cfg = f"sigma_w={sw}|gamma_u={gu}"
        method_stats = {}
        for meth in METHOD_LABELS:
            errs = np.array([r.estimates[meth] - TRUE_TAU for r in rs if np.isfinite(r.estimates[meth])])
            method_stats[meth] = {
                "mean_bias": float(errs.mean()),
                "mean_abs_error": float(np.abs(errs).mean()),
                "rmse": float(np.sqrt((errs**2).mean())),
                "se_abs_error": float(np.abs(errs).std(ddof=1) / math.sqrt(len(errs))),
            }
        summary["methods"][cfg] = method_stats
        summary["config"][cfg] = {
            "delta_hat_eval_mean": float(np.mean([r.delta_hat_eval for r in rs])),
            "delta_hat_random_mean": float(np.mean([r.delta_hat_random for r in rs])),
            "delta_sq_train_plateau_mean": float(np.mean([r.delta_sq_train_final for r in rs])),
            "weighted_d_xu_mean": float(np.mean([r.weighted_d_xu for r in rs])),
            "perm_pvalue_median": float(np.median([r.perm_pvalue for r in rs])),
            "dropped_mass_mean": float(np.mean([r.dropped_mass for r in rs])),
            "min_overlap_mean": float(np.mean([r.min_overlap for r in rs])),
            "theorem3_holds_frac": float(
                np.mean([1.0 if r.theorem3_lhs <= r.theorem3_rhs + 1e-9 else 0.0 for r in rs])
            ),
            "l_y_mean": float(np.mean([r.l_y for r in rs])),
        }
    return summary


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float((rx * ry).mean())


def figure_chain(records: list[RunRecord], gammas: list[float]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    for sw, color in SIGMA_COLORS.items():
        rs = [r for r in records if r.sigma_w == sw]
        if not rs:
            continue
        xs_l = np.array([r.delta_hat_eval for r in rs])
        ys_l = np.array([abs(r.estimates["learned_h"] - TRUE_TAU) for r in rs])
        xs_r = np.array([r.delta_hat_random for r in rs])
        ys_r = np.array([abs(r.estimates["random_h"] - TRUE_TAU) for r in rs if np.isfinite(r.estimates["random_h"])])
        rho = spearman(np.concatenate([xs_l, xs_r]), np.concatenate([ys_l, ys_r]))
        ax.scatter(xs_l, ys_l, s=24, alpha=0.8, color=color, label=f"$\\sigma_W={sw}$ ($\\rho_s$={rho:.2f})")
        ax.scatter(xs_r, ys_r, s=24, alpha=0.5, facecolors="none", edgecolors=color, linewidths=1.0)
    ax.set_xlabel(r"observable certificate $\hat\delta(h)$ (eval split)")
    ax.set_ylabel(r"$|\hat\tau_H-\tau|$")
    ax.set_title("(a) certificate vs realized bias\n(filled: learned $\\hat h$; hollow: random $h$)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for sw, color in SIGMA_COLORS.items():
        pts = [(row["mmd_v"], row["d_xu"]) for r in records if r.sigma_w == sw for row in r.stratum_rows]
        if not pts:
            continue
        xs, ys = np.array(pts).T
        ax.scatter(xs, ys, s=14, alpha=0.5, color=color)
        slope = float(np.linalg.lstsq(xs[:, None], ys, rcond=None)[0][0])
        xg = np.linspace(0, xs.max() * 1.02, 20)
        ax.plot(xg, slope * xg, color=color, lw=1.8, label=f"$\\sigma_W={sw}$: slope {slope:.2f}")
    ax.set_xlabel(r"observed per-stratum $\mathrm{MMD}_{V,m}$")
    ax.set_ylabel(r"latent per-stratum $d_{XU,m}$ (oracle)")
    ax.set_title(r"(b) channel inversion: slope $\approx$ empirical $\omega_K$")
    ax.legend(fontsize=8)

    ax = axes[2]
    for sw, color in SIGMA_COLORS.items():
        pts = [
            (r.l_y * row["d_xu"], row["abs_bias_m"])
            for r in records
            if r.sigma_w == sw
            for row in r.stratum_rows
        ]
        if not pts:
            continue
        xs, ys = np.array(pts).T
        ax.scatter(xs, ys, s=14, alpha=0.5, color=color, label=f"$\\sigma_W={sw}$")
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="$y=x$ (Theorem 3 bound)")
    ax.set_xlabel(r"$L_Y\, d_{XU,m}$ (oracle bound)")
    ax.set_ylabel(r"oracle stratum bias $|B_m|$")
    ax.set_title("(c) latent imbalance bounds bias")
    ax.legend(fontsize=8)

    gamma_list = ", ".join(str(g) for g in gammas)
    fig.suptitle(
        "Chain validation: small observable discrepancy $\\Rightarrow$ small latent imbalance $\\Rightarrow$ small bias "
        f"(pooled over $\\gamma_U \\in \\{{{gamma_list}\\}}$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"learned_representation_chain_validation.{ext}", dpi=200)
    plt.close(fig)


def figure_methods(records: list[RunRecord], sigmas: list[float], gammas: list[float]) -> None:
    fig, axes = plt.subplots(1, len(gammas), figsize=(6.4 * len(gammas), 4.6), sharey=True)
    if len(gammas) == 1:
        axes = [axes]
    palette = ["#94a3b8", "#7c3aed", "#0ea5e9", "#f59e0b", "#059669", "#111827"]
    for ax, gu in zip(axes, gammas):
        width = 0.13
        xs = np.arange(len(sigmas))
        for k, meth in enumerate(METHOD_LABELS):
            means, ses = [], []
            for sw in sigmas:
                rs = [r for r in records if r.sigma_w == sw and r.gamma_u == gu]
                errs = np.array([abs(r.estimates[meth] - TRUE_TAU) for r in rs if np.isfinite(r.estimates[meth])])
                means.append(errs.mean())
                ses.append(errs.std(ddof=1) / math.sqrt(len(errs)))
            ax.bar(
                xs + (k - 2.5) * width,
                means,
                width,
                yerr=ses,
                capsize=2,
                color=palette[k],
                label=METHOD_LABELS[meth],
            )
        ax.set_xticks(xs)
        ax.set_xticklabels([f"$\\sigma_W={s}$" for s in sigmas])
        ax.set_title(f"$\\gamma_U={gu}$")
        ax.set_ylabel(r"mean $|\hat\tau-\tau|$")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Method comparison: absolute error by channel noise and confounding strength", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"learned_representation_method_comparison.{ext}", dpi=200)
    plt.close(fig)


def figure_training(records: list[RunRecord], sigmas: list[float], gammas: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for sw in sigmas:
        rs = [r for r in records if r.sigma_w == sw and r.gamma_u == max(gammas)]
        if not rs:
            continue
        min_len = min(len(r.train_curve) for r in rs)
        curves = np.array([r.train_curve[:min_len] for r in rs])
        mean_curve = curves.mean(0)
        steps_axis = np.linspace(0, 1, min_len)
        ax.plot(steps_axis, mean_curve, color=SIGMA_COLORS[sw], lw=2, label=f"$\\sigma_W={sw}$")
    ax.set_yscale("log")
    ax.set_xlabel("training progress")
    ax.set_ylabel(r"soft $\hat\delta^2$ (train split)")
    ax.set_title(
        "Training curves plateau above zero\n"
        f"(the empirical trace of the exact-balance impossibility, $\\gamma_U={max(gammas)}$)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"learned_representation_training_curves.{ext}", dpi=200)
    plt.close(fig)


def write_outputs(records: list[RunRecord], summary: dict, args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    payload = {
        "provenance": {
            "script": "code/learned_representation_balancing_experiment.py",
            "theory_doc": "Single-Proximal-Balancing-approx-theory-v2026-08-17.md",
            "date": "2026-08-17",
            "n": args.n,
            "m_strata": args.strata,
            "steps": args.steps,
            "reps": args.reps,
            "sigmas": args.sigmas,
            "gammas": args.gammas,
            "true_tau": TRUE_TAU,
        },
        "summary": summary,
    }
    with open(RESULTS_DIR / "learned_representation_balancing_summary.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(RESULTS_DIR / "learned_representation_balancing_runs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["sigma_w", "gamma_u", "rep", "method", "estimate", "abs_error", "delta_hat_eval",
             "delta_hat_random", "weighted_d_xu", "perm_pvalue", "theorem3_lhs", "theorem3_rhs", "l_y"]
        )
        for r in records:
            for meth, est in r.estimates.items():
                writer.writerow(
                    [r.sigma_w, r.gamma_u, r.rep, meth, f"{est:.6f}", f"{abs(est - TRUE_TAU):.6f}",
                     f"{r.delta_hat_eval:.6f}", f"{r.delta_hat_random:.6f}", f"{r.weighted_d_xu:.6f}",
                     f"{r.perm_pvalue:.4f}", f"{r.theorem3_lhs:.6f}", f"{r.theorem3_rhs:.6f}", f"{r.l_y:.4f}"]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--strata", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.3, 1.0, 2.0])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--smoke", action="store_true", help="tiny run for a quick correctness check")
    parser.add_argument(
        "--only-training-figure",
        action="store_true",
        help="deterministically re-run training alone (same seeds) and regenerate the training-curve figure",
    )
    args = parser.parse_args()
    if args.smoke:
        args.n, args.steps, args.reps = 1200, 60, 2
        args.sigmas, args.gammas = [0.3, 2.0], [2.0]

    torch.set_num_threads(max(1, torch.get_num_threads()))
    if args.only_training_figure:
        gu = max(args.gammas)
        records = [
            train_only(sw, gu, rep, args.n, args.strata, args.steps)
            for sw in args.sigmas
            for rep in range(args.reps)
        ]
        figure_training(records, args.sigmas, args.gammas)
        print("training-curve figure regenerated")
        return
    records = sweep(args.n, args.strata, args.steps, args.reps, args.sigmas, args.gammas)
    summary = summarize(records)
    write_outputs(records, summary, args)
    figure_chain(records, args.gammas)
    figure_methods(records, args.sigmas, args.gammas)
    figure_training(records, args.sigmas, args.gammas)
    print(json.dumps(summary["config"], indent=2))
    for cfg, stats in summary["methods"].items():
        print(cfg)
        for meth, st in stats.items():
            print(f"  {meth:12s} bias={st['mean_bias']:+.3f}  |err|={st['mean_abs_error']:.3f}  rmse={st['rmse']:.3f}")


if __name__ == "__main__":
    main()
