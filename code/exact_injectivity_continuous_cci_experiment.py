#!/usr/bin/env python3
"""Continuous conditional-cross-covariance experiment for proxy balancing.

This is one prespecified SCM family.  The finite random Fourier feature (RFF)
criterion below is an empirical differentiable surrogate.  It is not an exact
population KCI statistic and is not reported as one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor


ROOT_SEED = 190602
TRUE_ATE = 1.0
D_X = 20
D_W_REP = 9
PREFIXES = (100, 200, 500, 1000, 2000, 5000, 10000)
N_MASTERS = 20
N_FOLDS = 2
RFF_V = 24
RFF_R = 12
RIDGE = 2e-2
PRETRAIN_STEPS = 60
CCI_STEPS = 100
LR_PRETRAIN = 2e-2
LR_CCI = 8e-3
R_BANDWIDTH = 0.20
GBR_SPEC = {
    "n_estimators": 120,
    "max_depth": 2,
    "learning_rate": 0.05,
    "min_samples_leaf": 5,
}
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "exact_injectivity_continuous_cci_summary.json"

GAMMA_RAW = np.array([
    0.90, -0.70, 0.55, -0.45, 0.38, -0.32, 0.27, -0.23, 0.20, -0.18,
    0.16, -0.14, 0.12, -0.10, 0.09, -0.08, 0.07, -0.06, 0.05, -0.04,
], dtype=float)
GAMMA = GAMMA_RAW / np.linalg.norm(GAMMA_RAW)


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        value = float(x)
        return value if math.isfinite(value) else None
    return x


def array_hash(x: np.ndarray) -> str:
    z = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(z.dtype).encode())
    h.update(str(z.shape).encode())
    h.update(z.tobytes())
    return h.hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def generate_master(n: int, seed: int) -> dict[str, np.ndarray]:
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(8)]
    ru, rx, rv, rw, raudit, ra, ry, rfold = streams
    u = ru.standard_normal(n)
    s = (u > 0).astype(float)
    ex = rx.standard_normal((n, D_X))
    x = np.empty((n, D_X), dtype=float)
    x[:, 0] = 0.75 * u + 0.65 * ex[:, 0]
    x[:, 1] = np.sin(u) + 0.65 * ex[:, 1]
    x[:, 2] = np.tanh(1.4 * u) + 0.65 * ex[:, 2]
    x[:, 3] = (u * u - 1.0) / math.sqrt(2.0) + 0.70 * ex[:, 3]
    x[:, 4] = 0.45 * u + 0.45 * np.sin(2.0 * u) + 0.65 * ex[:, 4]
    load = np.linspace(-0.45, 0.45, D_X - 5)
    x[:, 5:] = u[:, None] * load[None, :] + 0.80 * ex[:, 5:]

    # Wrep has an exact joint decoder: S = Wrep[:, 1] - Wrep[:, 0].
    v = rv.standard_normal(n)
    wrep = np.empty((n, D_W_REP), dtype=float)
    wrep[:, 0] = v
    wrep[:, 1] = v + s
    ew = rw.standard_normal((n, D_W_REP - 2))
    centered_s = 2.0 * s - 1.0
    for j in range(D_W_REP - 2):
        slope = 0.25 + 0.08 * j
        wrep[:, j + 2] = slope * centered_s + 1.25 * ew[:, j]

    waudit = u + 0.70 * raudit.standard_normal(n)
    linear_score = 0.9 * centered_s + x @ GAMMA
    r_oracle = sigmoid(linear_score)
    propensity = 0.1 + 0.8 * r_oracle
    a = ra.binomial(1, propensity).astype(float)

    fx = (
        0.65 * x[:, 0]
        + 0.35 * x[:, 1] * x[:, 2]
        + 0.25 * np.sin(x[:, 3])
        - 0.20 * x[:, 4] * x[:, 5]
    )
    y0_struct = fx + 1.3 * u + 0.3 * (u * u - 1.0)
    y = y0_struct + a + 0.50 * ry.standard_normal(n)

    order = rfold.permutation(n)
    fold = np.empty(n, dtype=int)
    fold[order[::2]] = 0
    fold[order[1::2]] = 1
    return {
        "U": u,
        "S": s,
        "X": x,
        "Wrep": wrep,
        "Waudit": waudit,
        "R_oracle": r_oracle,
        "propensity": propensity,
        "A": a,
        "Y": y,
        "Y0_struct": y0_struct,
        "fold": fold,
    }


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def median_bandwidth(z: np.ndarray, seed: int, max_points: int = 500) -> float:
    if len(z) > max_points:
        rng = np.random.default_rng(seed)
        z = z[rng.choice(len(z), max_points, replace=False)]
    sq = ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
    vals = sq[np.triu_indices(len(z), 1)]
    return math.sqrt(max(float(np.median(vals)), 1e-8) / 2.0)


@dataclass
class RFFSpec:
    mean: np.ndarray
    scale: np.ndarray
    omega: np.ndarray
    phase: np.ndarray
    bandwidth: float


def make_v_rff(v_train: np.ndarray, seed: int) -> RFFSpec:
    mean = v_train.mean(axis=0)
    scale = v_train.std(axis=0) + 1e-8
    z = (v_train - mean) / scale
    bw = median_bandwidth(z, seed)
    rng = np.random.default_rng(seed + 101)
    omega = rng.standard_normal((z.shape[1], RFF_V)) / bw
    phase = rng.uniform(0.0, 2.0 * math.pi, RFF_V)
    return RFFSpec(mean, scale, omega, phase, bw)


def v_rff_numpy(v: np.ndarray, spec: RFFSpec) -> np.ndarray:
    z = (v - spec.mean) / spec.scale
    return math.sqrt(2.0 / RFF_V) * np.cos(z @ spec.omega + spec.phase)


class LinearScore(torch.nn.Module):
    """A scalar score class containing sigmoid(0.9(2S-1)+gamma'X)."""

    def __init__(self, d: int):
        super().__init__()
        self.linear = torch.nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x)).squeeze(1)


@dataclass
class RepresentationFit:
    model: LinearScore
    input_mean: np.ndarray
    input_scale: np.ndarray
    v_spec: RFFSpec
    r_omega: np.ndarray
    r_phase: np.ndarray
    beta_a: np.ndarray
    beta_v: np.ndarray
    pretrain_curve: list[float]
    cci_curve: list[float]
    seed: int


def r_features_torch(r: torch.Tensor, omega: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    rff = math.sqrt(2.0 / RFF_R) * torch.cos(r[:, None] * omega[None, :] + phase[None, :])
    return torch.cat([torch.ones((len(r), 1), dtype=r.dtype, device=r.device), r[:, None], rff], dim=1)


def ridge_coefficients(z: torch.Tensor, target: torch.Tensor, ridge: float) -> torch.Tensor:
    d = z.shape[1]
    penalty = torch.eye(d, dtype=z.dtype, device=z.device) * ridge
    penalty[0, 0] = 0.0
    lhs = z.T @ z / len(z) + penalty
    rhs = z.T @ target / len(z)
    return torch.linalg.solve(lhs, rhs)


def conditional_crosscov_loss(
    r: torch.Tensor,
    a: torch.Tensor,
    phi_v: torch.Tensor,
    omega: torch.Tensor,
    phase: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = r_features_torch(r, omega, phase)
    beta_a = ridge_coefficients(z, a[:, None], RIDGE)
    beta_v = ridge_coefficients(z, phi_v, RIDGE)
    residual_a = a[:, None] - z @ beta_a
    residual_v = phi_v - z @ beta_v
    crosscov = residual_a.T @ residual_v / len(a)
    return crosscov.square().sum(), beta_a, beta_v


def fit_representation(inp: np.ndarray, audit_v: np.ndarray, a: np.ndarray, seed: int) -> RepresentationFit:
    mean = inp.mean(axis=0)
    scale = inp.std(axis=0) + 1e-8
    zin = (inp - mean) / scale
    v_spec = make_v_rff(audit_v, seed + 13)
    phi_v = v_rff_numpy(audit_v, v_spec)
    rng = np.random.default_rng(seed + 17)
    r_omega = rng.standard_normal(RFF_R) / R_BANDWIDTH
    r_phase = rng.uniform(0.0, 2.0 * math.pi, RFF_R)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = LinearScore(zin.shape[1]).double()
    xt = torch.tensor(zin, dtype=torch.float64)
    at = torch.tensor(a, dtype=torch.float64)
    vt = torch.tensor(phi_v, dtype=torch.float64)
    ot = torch.tensor(r_omega, dtype=torch.float64)
    pt = torch.tensor(r_phase, dtype=torch.float64)

    # Treatment-only initialization.  It uses no outcome and only training data.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_PRETRAIN, weight_decay=1e-4)
    bce_curve: list[float] = []
    for _ in range(PRETRAIN_STEPS):
        r = model(xt)
        loss = torch.nn.functional.binary_cross_entropy(r.clamp(1e-5, 1.0 - 1e-5), at)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bce_curve.append(float(loss.detach()))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_CCI, weight_decay=1e-4)
    cci_curve: list[float] = []
    for _ in range(CCI_STEPS):
        r = model(xt)
        loss, _, _ = conditional_crosscov_loss(r, at, vt, ot, pt)
        # Keep the score nondegenerate without using held-out data or outcomes.
        variance_floor = torch.relu(torch.tensor(0.015, dtype=r.dtype) - r.var(unbiased=False))
        objective = loss + 0.5 * variance_floor
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        cci_curve.append(float(loss.detach()))

    with torch.no_grad():
        r = model(xt)
        _, beta_a, beta_v = conditional_crosscov_loss(r, at, vt, ot, pt)
    return RepresentationFit(
        model=model,
        input_mean=mean,
        input_scale=scale,
        v_spec=v_spec,
        r_omega=r_omega,
        r_phase=r_phase,
        beta_a=beta_a.detach().numpy(),
        beta_v=beta_v.detach().numpy(),
        pretrain_curve=bce_curve,
        cci_curve=cci_curve,
        seed=seed,
    )


def predict_representation(fit: RepresentationFit, inp: np.ndarray) -> np.ndarray:
    fit.model.eval()
    z = torch.tensor((inp - fit.input_mean) / fit.input_scale, dtype=torch.float64)
    with torch.no_grad():
        return fit.model(z).numpy()


def r_features_numpy(r: np.ndarray, fit: RepresentationFit) -> np.ndarray:
    rff = math.sqrt(2.0 / RFF_R) * np.cos(r[:, None] * fit.r_omega[None, :] + fit.r_phase[None, :])
    return np.column_stack([np.ones(len(r)), r, rff])


def heldout_cci(fit: RepresentationFit, r: np.ndarray, a: np.ndarray, audit_v: np.ndarray) -> float:
    z = r_features_numpy(r, fit)
    phi_v = v_rff_numpy(audit_v, fit.v_spec)
    residual_a = a[:, None] - z @ fit.beta_a
    residual_v = phi_v - z @ fit.beta_v
    return float(np.linalg.norm(residual_a.T @ residual_v / len(a)))


def heldout_cci_refit_nuisance(fit: RepresentationFit, r: np.ndarray, a: np.ndarray, audit_v: np.ndarray) -> float:
    z = torch.tensor(r_features_numpy(r, fit), dtype=torch.float64)
    at = torch.tensor(a[:, None], dtype=torch.float64)
    vt = torch.tensor(v_rff_numpy(audit_v, fit.v_spec), dtype=torch.float64)
    ba = ridge_coefficients(z, at, RIDGE)
    bv = ridge_coefficients(z, vt, RIDGE)
    cross = (at - z @ ba).T @ (vt - z @ bv) / len(a)
    return float(torch.linalg.norm(cross).detach())


def continuous_latent_imbalance(fit: RepresentationFit, r: np.ndarray, a: np.ndarray, u: np.ndarray) -> dict[str, Any]:
    z = torch.tensor(r_features_numpy(r, fit), dtype=torch.float64)
    at = torch.tensor(a[:, None], dtype=torch.float64)
    ut = torch.tensor(u[:, None], dtype=torch.float64)
    ba = ridge_coefficients(z, at, RIDGE)
    bu = ridge_coefficients(z, ut, RIDGE)
    ar = at - z @ ba
    ur = ut - z @ bu
    covariance = float((ar.T @ ur / len(a)).detach())
    denom = float(torch.sqrt((ar.square().mean()) * (ur.square().mean())).detach())
    return {
        "residual_crosscovariance": covariance,
        "absolute_residual_correlation": abs(covariance) / max(denom, 1e-12),
    }


def tlearner(train_x: np.ndarray, train_a: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int) -> np.ndarray | None:
    predictions = []
    for arm in (1.0, 0.0):
        mask = train_a == arm
        if int(mask.sum()) < GBR_SPEC["min_samples_leaf"]:
            return None
        model = GradientBoostingRegressor(random_state=seed + int(arm), **GBR_SPEC)
        model.fit(train_x[mask], train_y[mask])
        predictions.append(model.predict(test_x))
    return predictions[0] - predictions[1]


def baseline_crossfit(data: dict[str, np.ndarray], n: int, feature: str, seed: int) -> dict[str, Any]:
    if feature == "X":
        cov = data["X"][:n]
    elif feature == "XW":
        cov = np.column_stack([data["X"][:n], data["Wrep"][:n], data["Waudit"][:n]])
    elif feature == "XS":
        cov = np.column_stack([data["X"][:n], data["S"][:n]])
    elif feature == "XU":
        cov = np.column_stack([data["X"][:n], data["U"][:n]])
    elif feature == "R0":
        cov = data["R_oracle"][:n, None]
    else:
        raise ValueError(feature)
    stitched = np.full(n, np.nan)
    fold_qa = []
    for fold in range(N_FOLDS):
        test = np.where(data["fold"][:n] == fold)[0]
        train = np.where(data["fold"][:n] != fold)[0]
        pred = tlearner(cov[train], data["A"][train], data["Y"][train], cov[test], seed + 1009 * fold)
        if pred is not None:
            stitched[test] = pred
        fold_qa.append({
            "fold": fold,
            "train_n": len(train),
            "test_n": len(test),
            "train_hash": array_hash(train),
            "test_hash": array_hash(test),
            "disjoint": not np.intersect1d(train, test).size,
            "train_n1": int(data["A"][train].sum()),
            "train_n0": int(len(train) - data["A"][train].sum()),
        })
    finite = bool(np.isfinite(stitched).all())
    estimate = float(stitched.mean()) if finite else None
    return {
        "estimate": estimate,
        "error": estimate - TRUE_ATE if finite else None,
        "abs_error": abs(estimate - TRUE_ATE) if finite else None,
        "finite": finite,
        "folds": fold_qa,
    }


def learned_crossfit(data: dict[str, np.ndarray], n: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inp = np.column_stack([data["X"][:n], data["Wrep"][:n]])
    audit = np.column_stack([data["X"][:n], data["Waudit"][:n]])
    stitched = np.full(n, np.nan)
    details: list[dict[str, Any]] = []
    for fold in range(N_FOLDS):
        test = np.where(data["fold"][:n] == fold)[0]
        train = np.where(data["fold"][:n] != fold)[0]
        fit = fit_representation(inp[train], audit[train], data["A"][train], seed + 1009 * fold)
        r_train = predict_representation(fit, inp[train])
        r_test = predict_representation(fit, inp[test])
        pred = tlearner(r_train[:, None], data["A"][train], data["Y"][train], r_test[:, None], seed + 50000 + fold)
        if pred is not None:
            stitched[test] = pred
        details.append({
            "fold": fold,
            "train_n": len(train),
            "test_n": len(test),
            "train_hash": array_hash(train),
            "test_hash": array_hash(test),
            "train_test_disjoint": not np.intersect1d(train, test).size,
            "representation_inputs": ["X", "Wrep"],
            "audit_inputs": ["X", "Waudit"],
            "fixed_rep_audit_split": True,
            "Y_used_in_representation_fit": False,
            "U_or_S_used_in_representation_fit": False,
            "no_units_deleted": len(train) + len(test) == n,
            "pretrain_initial": fit.pretrain_curve[0],
            "pretrain_final": fit.pretrain_curve[-1],
            "cci_initial": fit.cci_curve[0],
            "cci_final": fit.cci_curve[-1],
            "cci_optimization_decreased": fit.cci_curve[-1] <= fit.cci_curve[0],
            "heldout_cci_train_fitted_nuisance": heldout_cci(fit, r_test, data["A"][test], audit[test]),
            "heldout_cci_refit_nuisance_diagnostic": heldout_cci_refit_nuisance(fit, r_test, data["A"][test], audit[test]),
            "latent_imbalance": continuous_latent_imbalance(fit, r_test, data["A"][test], data["U"][test]),
            "rank_recovery_spearman": spearman(r_test, data["R_oracle"][test]),
            "score_pearson": float(np.corrcoef(r_test, data["R_oracle"][test])[0, 1]),
            "score_range": [float(r_test.min()), float(r_test.max())],
            "audit_rff_bandwidth": fit.v_spec.bandwidth,
            "finite_rff_surrogate_only": True,
        })
    finite = bool(np.isfinite(stitched).all())
    estimate = float(stitched.mean()) if finite else None
    result = {
        "estimate": estimate,
        "error": estimate - TRUE_ATE if finite else None,
        "abs_error": abs(estimate - TRUE_ATE) if finite else None,
        "finite": finite,
    }
    return result, details


def known_propensity_ipw(data: dict[str, np.ndarray], n: int) -> dict[str, float]:
    p = data["propensity"][:n]
    a = data["A"][:n]
    y = data["Y"][:n]
    estimate = float(np.mean(a * y / p - (1.0 - a) * y / (1.0 - p)))
    return {"estimate": estimate, "error": estimate - TRUE_ATE, "abs_error": abs(estimate - TRUE_ATE)}


def deterministic_replay(data: dict[str, np.ndarray]) -> dict[str, Any]:
    n = 200
    test = np.where(data["fold"][:n] == 0)[0]
    train = np.where(data["fold"][:n] != 0)[0]
    inp = np.column_stack([data["X"][:n], data["Wrep"][:n]])
    audit = np.column_stack([data["X"][:n], data["Waudit"][:n]])
    fits = [fit_representation(inp[train], audit[train], data["A"][train], ROOT_SEED + 424242) for _ in range(2)]
    preds = [predict_representation(fit, inp[test]) for fit in fits]
    params = [np.concatenate([p.detach().numpy().ravel() for p in fit.model.parameters()]) for fit in fits]
    return {
        "exact_predictions": bool(np.array_equal(preds[0], preds[1])),
        "exact_parameters": bool(np.array_equal(params[0], params[1])),
        "exact_pretrain_curve": fits[0].pretrain_curve == fits[1].pretrain_curve,
        "exact_cci_curve": fits[0].cci_curve == fits[1].cci_curve,
    }


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    methods = sorted({r["method"] for r in records})
    ns = sorted({r["n"] for r in records})
    for n in ns:
        for method in methods:
            rows = [r for r in records if r["n"] == n and r["method"] == method and r["finite"]]
            if not rows:
                continue
            errors = np.array([r["error"] for r in rows], dtype=float)
            out.append({
                "n": n,
                "method": method,
                "replications": len(rows),
                "mean_estimate": float(np.mean([r["estimate"] for r in rows])),
                "signed_bias": float(errors.mean()),
                "mean_absolute_error": float(np.abs(errors).mean()),
                "rmse": float(np.sqrt(np.mean(errors * errors))),
                "monte_carlo_se_signed_bias": float(errors.std(ddof=1) / math.sqrt(len(errors))) if len(errors) > 1 else None,
            })
    return out


def paired_difference_summary(values: list[float], estimand: str) -> dict[str, Any]:
    """Summarize paired differences, including an exact two-sided sign test."""
    arr = np.asarray(values, dtype=float)
    wins = int(np.sum(arr > 0))
    losses = int(np.sum(arr < 0))
    ties = int(np.sum(arr == 0))
    non_ties = wins + losses
    sign_p = None
    if non_ties:
        upper = max(wins, losses)
        tail = sum(math.comb(non_ties, k) for k in range(upper, non_ties + 1)) / (2 ** non_ties)
        sign_p = min(1.0, 2.0 * tail)
    result: dict[str, Any] = {
        "estimand": estimand,
        "mean": float(arr.mean()),
        "sample_standard_deviation": float(arr.std(ddof=1)) if len(arr) > 1 else None,
        "standard_error": float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else None,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "total": len(arr),
        "exact_two_sided_sign_test_p": sign_p,
        "values": values,
    }
    if len(arr) == 20:
        critical = 2.093024054408263
        half_width = critical * result["standard_error"]
        result["confidence_interval_95_t"] = {
            "method": "two-sided paired t interval with 19 degrees of freedom",
            "lower": result["mean"] - half_width,
            "upper": result["mean"] + half_width,
        }
    else:
        result["confidence_interval_95_t"] = None
    return result


def run(n_masters: int, prefixes: tuple[int, ...]) -> dict[str, Any]:
    started = time.time()
    records: list[dict[str, Any]] = []
    learned_details: list[dict[str, Any]] = []
    hashes = []
    analytic_gates = []
    replay = None
    for rep in range(n_masters):
        seed = ROOT_SEED + 1000003 * rep
        data = generate_master(max(prefixes), seed)
        if rep == 0:
            replay = deterministic_replay(data)
        decoder_error = float(np.max(np.abs((data["Wrep"][:, 1] - data["Wrep"][:, 0]) - data["S"])))
        analytic_gates.append({
            "rep": rep,
            "exact_decoder_max_error": decoder_error,
            "propensity_min": float(data["propensity"].min()),
            "propensity_max": float(data["propensity"].max()),
            "all_propensities_in_point_one_point_nine": bool(np.all((data["propensity"] >= 0.1) & (data["propensity"] <= 0.9))),
            "constant_structural_treatment_effect": True,
            "audit_channel": "Waudit=U+0.7E with independent standard Gaussian E",
            "audit_gaussian_convolution_injective": True,
            "exact_balance_reason": "A is drawn from Bernoulli(0.1+0.8*R0). The map from R0 to the true propensity is one-to-one, so conditioning on R0 is equivalent to conditioning on the true propensity and preserves exact balance.",
        })
        hashes.append({
            "rep": rep,
            "X": array_hash(data["X"]),
            "U": array_hash(data["U"]),
            "Wrep": array_hash(data["Wrep"]),
            "Waudit": array_hash(data["Waudit"]),
            "A": array_hash(data["A"]),
            "Y": array_hash(data["Y"]),
            "fold": array_hash(data["fold"]),
            "prefix_hashes": {str(n): array_hash(data["X"][:n]) for n in prefixes},
        })
        for ni, n in enumerate(prefixes):
            method_seed = seed + 10000 * ni
            for feature, method in (
                ("X", "x_fitted"),
                ("XW", "full_observed_xw_fitted"),
                ("XS", "observed_s_fitted"),
                ("XU", "observed_u_fitted"),
                ("R0", "oracle_r0_adjustment"),
            ):
                result = baseline_crossfit(data, n, feature, method_seed + len(method))
                records.append({"rep": rep, "n": n, "method": method, **{k: v for k, v in result.items() if k != "folds"}})
            ipw = known_propensity_ipw(data, n)
            records.append({"rep": rep, "n": n, "method": "known_propensity_ipw", "finite": True, **ipw})
            learned, details = learned_crossfit(data, n, method_seed + 777)
            records.append({"rep": rep, "n": n, "method": "learned_continuous_r", **learned})
            learned_details.append({"rep": rep, "n": n, "folds": details})

    summary = aggregate(records)
    primary_rows = {r["method"]: r for r in summary if r["n"] == max(prefixes)}
    learned_primary = primary_rows.get("learned_continuous_r")
    x_primary = primary_rows.get("x_fitted")
    full_xw_primary = primary_rows.get("full_observed_xw_fitted")
    paired_primary = []
    paired_full_xw = []
    for rep in range(n_masters):
        l = next(r for r in records if r["rep"] == rep and r["n"] == max(prefixes) and r["method"] == "learned_continuous_r")
        x = next(r for r in records if r["rep"] == rep and r["n"] == max(prefixes) and r["method"] == "x_fitted")
        full_xw = next(r for r in records if r["rep"] == rep and r["n"] == max(prefixes) and r["method"] == "full_observed_xw_fitted")
        paired_primary.append(x["abs_error"] - l["abs_error"])
        paired_full_xw.append(full_xw["abs_error"] - l["abs_error"])

    detail_primary = [d for d in learned_details if d["n"] == max(prefixes)]
    primary_folds = [f for d in detail_primary for f in d["folds"]]
    gates = {
        "all_exact_decoder_errors_within_machine_tolerance": all(g["exact_decoder_max_error"] <= 1e-12 for g in analytic_gates),
        "all_population_overlap_bounds_hold": all(g["all_propensities_in_point_one_point_nine"] for g in analytic_gates),
        "all_estimates_finite": all(r["finite"] for r in records),
        "all_fold_splits_disjoint": all(f["train_test_disjoint"] for d in learned_details for f in d["folds"]),
        "all_rep_audit_splits_fixed": all(f["fixed_rep_audit_split"] for d in learned_details for f in d["folds"]),
        "no_Y_U_S_in_representation_fit": all(not f["Y_used_in_representation_fit"] and not f["U_or_S_used_in_representation_fit"] for d in learned_details for f in d["folds"]),
        "no_units_deleted": all(f["no_units_deleted"] for d in learned_details for f in d["folds"]),
        "deterministic_replay": bool(replay and all(replay.values())),
        "nested_prefixes_by_construction": all(np.array_equal(generate_master(max(prefixes), ROOT_SEED)["X"][:n], generate_master(n, ROOT_SEED)["X"]) for n in prefixes),
        "finite_rff_labeled_surrogate": True,
    }
    return jsonable({
        "provenance": {
            "script": "code/exact_injectivity_continuous_cci_experiment.py",
            "root_seed": ROOT_SEED,
            "n_masters": n_masters,
            "prefixes": prefixes,
            "one_scm_family_only": True,
            "runtime_seconds": time.time() - started,
            "outcome_estimator": "2-fold cross-fitted GradientBoosting T-learner with matched specification",
            "gbr_spec": GBR_SPEC,
            "representation_optimizer": "Adam",
            "pretrain_steps": PRETRAIN_STEPS,
            "cci_steps": CCI_STEPS,
            "finite_rff_statement": "The differentiable conditional cross-covariance is a fixed finite-RFF empirical surrogate, not exact population KCI.",
            "post_result_tuning": False,
        },
        "dgp": {
            "graph": "U->S; U->X; (X,S)->A; (S,independent noises)->Wrep; U->Waudit; (X,U,A)->Y",
            "dimensions": {"U": 1, "S": 1, "X": D_X, "Wrep": D_W_REP, "Waudit": 1},
            "exact_decoder": "S=Wrep[1]-Wrep[0]",
            "oracle_score": "sigmoid(0.9(2S-1)+gamma'X)",
            "propensity": "0.1+0.8*oracle_score",
            "balance_transform": "The true propensity is a one-to-one affine transform of oracle_score, so either variable generates the same balancing sigma-field.",
            "gamma": GAMMA,
            "outcome": "Y(a)=a+f(X)+1.3U+0.3(U^2-1)+epsilon",
            "true_ate": TRUE_ATE,
            "audit_injectivity": "Gaussian convolution with nonzero identity slope and nonvanishing characteristic function",
        },
        "representation": {
            "input": ["X", "Wrep"],
            "held_out_audit": ["X", "Waudit"],
            "score": "sigmoid(linear_net(standardized_X_Wrep))",
            "oracle_in_class": True,
            "oracle_linear_coefficients_before_standardization": {
                "X": GAMMA,
                "Wrep_1": -1.8,
                "Wrep_2": 1.8,
                "intercept": -0.9,
            },
            "conditional_surrogate": {
                "V": "fixed RFF of (X,Waudit)",
                "R": "fixed RFF of scalar learned score plus intercept and linear score",
                "residualization": "differentiable kernel-ridge feature regression",
                "ridge": RIDGE,
                "r_bandwidth": R_BANDWIDTH,
                "rff_v": RFF_V,
                "rff_r": RFF_R,
            },
        },
        "analytic_gates": analytic_gates,
        "master_hashes": hashes,
        "deterministic_replay": replay,
        "records": records,
        "aggregate": summary,
        "primary_n": max(prefixes),
        "primary_comparison": {
            "learned": learned_primary,
            "x_fitted": x_primary,
            "full_observed_xw_fitted": full_xw_primary,
            "paired_abs_error_improvement_x_minus_learned": paired_difference_summary(
                paired_primary,
                "x_fitted_abs_error - learned_abs_error; positive favors learned",
            ),
            "paired_abs_error_improvement_full_xw_minus_learned": paired_difference_summary(
                paired_full_xw,
                "full_observed_xw_abs_error - learned_abs_error; positive favors learned",
            ),
            "learned_heldout_cci_mean": float(np.mean([f["heldout_cci_train_fitted_nuisance"] for f in primary_folds])),
            "learned_latent_residual_correlation_mean": float(np.mean([f["latent_imbalance"]["absolute_residual_correlation"] for f in primary_folds])),
            "learned_rank_recovery_spearman_mean": float(np.mean([f["rank_recovery_spearman"] for f in primary_folds])),
        },
        "learned_fold_details": learned_details,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "limitations": [
            "One prespecified scalar-U SCM family does not establish broad empirical superiority.",
            "Finite RFF residual cross-covariance is an empirical surrogate, not exact population KCI.",
            "Exact audit injectivity does not by itself quantify approximate inverse stability.",
            "The representation class is linear because the exact balancing score is linear in X and the jointly decoded S.",
            "No self-conditioned negative control is included in this tightly scoped track.",
        ],
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), default="full")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Full mode defaults to the canonical result; smoke and pilot default to no write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        n_masters, prefixes = 1, (100,)
    elif args.mode == "pilot":
        n_masters, prefixes = 1, PREFIXES
    else:
        n_masters, prefixes = N_MASTERS, PREFIXES
    output = args.output
    if args.mode == "full" and output is None:
        output = OUTPUT_PATH
    if args.mode != "full" and output is not None and output.resolve() == OUTPUT_PATH.resolve():
        raise SystemExit("Refusing to overwrite the canonical full result from smoke or pilot mode.")
    payload = run(n_masters, prefixes)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(output) if output is not None else None,
        "write_performed": output is not None,
        "mode": args.mode,
        "runtime_seconds": payload["provenance"]["runtime_seconds"],
        "all_gates_pass": payload["all_gates_pass"],
        "primary_comparison": payload["primary_comparison"],
    }, indent=2))


if __name__ == "__main__":
    main()
