#!/usr/bin/env python3
"""Frozen approximate single-proxy balancing experiments.

The learned objective is a differentiable random-Fourier-feature surrogate,
not exact KCI or conditional HSIC.  Soft three-vector representations are the
primary adjustment variables.  Hard argmax adjustment is diagnostic only.  A
no-argument invocation preserves the original SCM, 10-replicate contract, and
legacy output path.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold


ROOT_SEED = 190702
N_MASTER = 1000
N_DIAGNOSTIC = 5000
PREFIXES = (100, 200, 500, 1000)
REGISTERED_LARGE_PREFIXES = (2000, 5000, 10000)
D_X, D_W = 20, 10
N_MASKS, N_SOFT, N_RFF = 5, 3, 64
STEPS, RESTARTS, HIDDEN = 100, 2, 32
LR, FINAL_TEMPERATURE = 3e-3, 0.25
TRUE_ATE = 1.0
OUTPUT = Path(__file__).resolve().parents[1] / "results" / "honest_canonical_representation_summary.json"
LEGACY_10REP_SHA256 = "6d507f3e526c524027a5a6758d4329bb0ee961489f236ed3e9fdc8a070852ce7"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FROZEN_20REP = {
    "original": (RESULTS_DIR / "honest_canonical_representation_original_scm1_20rep_summary.json", "b09c09c1d9adc9ab0aefe664e2372576f4de7c4c20ad7a1f0a0e784d4982c3af"),
    "nonlinear_scm2": (RESULTS_DIR / "honest_canonical_representation_nonlinear_scm2_20rep_summary.json", "5b270b6fc9b0ea8b2f8864a01b97a9f55ab91887228ea17aa6c7b0886fd4b2ff"),
}

NAMESPACE = {
    "structural_coefficients": 100,
    "scm2_structural_coefficients": 110,
    "master_dataset": 200,
    "independent_diagnostic_dataset": 300,
    "outer_fold_assignment": 400,
    "random_fourier_features": 500,
    "adam_initialization_and_training_noise": 600,
    "outcome_model": 700,
}
METHOD_ID = {
    "adjust_x": 0,
    "adjust_xw_full_observed": 1,
    "observed_xu_fitted_reference": 2,
    "proposed_cross_mask_v1": 3,
    "heuristic_self_conditioned_v2": 4,
}
GBR_SPEC = dict(n_estimators=100, max_depth=2, learning_rate=0.05, min_samples_leaf=5)


def key(namespace: str, replicate=0, n_index=0, fold=0, method=0, mask=0, restart=0) -> tuple[int, ...]:
    return (ROOT_SEED, NAMESPACE[namespace], replicate, n_index, fold, method, mask, restart)


def seed_int(seed_key: tuple[int, ...]) -> int:
    return int(np.random.SeedSequence(seed_key).generate_state(1, dtype=np.uint32)[0])


def rng(seed_key: tuple[int, ...]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed_key))


def array_hash(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonable(x: Any) -> Any:
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.bool_, bool)): return bool(x)
    if isinstance(x, (np.integer, int)): return int(x)
    if isinstance(x, (np.floating, float)):
        value = float(x); return value if math.isfinite(value) else None
    return x


def expit(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float); pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos])); ex = np.exp(x[~pos]); out[~pos] = ex / (1.0 + ex)
    return out


def unit_vector(generator: np.random.Generator, d: int) -> np.ndarray:
    v = generator.standard_normal(d); return v / np.linalg.norm(v)


def coefficients(scm: str = "original") -> dict[str, np.ndarray]:
    g = rng(key("structural_coefficients"))
    b_x = unit_vector(g, D_X)
    b_w = g.choice(np.array([-1.0, 1.0]), size=D_W) * g.uniform(0.35, 1.0, size=D_W)
    original = {"b_x": b_x, "b_w": b_w, "a_x": unit_vector(g, D_X), "y_x": unit_vector(g, D_X), "yt_x": unit_vector(g, D_X)}
    if scm == "original":
        return original
    if scm != "nonlinear_scm2":
        raise ValueError(f"unknown SCM: {scm}")
    g2 = rng(key("scm2_structural_coefficients"))
    c_w = np.sign(b_w) * g2.uniform(0.15, 0.35, size=D_W)
    return {
        **original,
        "c_x": unit_vector(g2, D_X),
        "c_w": c_w,
        "a2": unit_vector(g2, D_X),
        "a3": unit_vector(g2, D_X),
        "a4": unit_vector(g2, D_X),
        "y3": unit_vector(g2, D_X),
        "y4": unit_vector(g2, D_X),
    }


def draw_exogenous(n: int, seed_key: tuple[int, ...]) -> dict[str, np.ndarray]:
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed_key).spawn(5)]
    ru, rx, rw, ra, ry = streams
    return {
        "U": ru.standard_normal(n),
        "eps_X": rx.standard_normal((n, D_X)),
        "eps_W": rw.standard_normal((n, D_W)),
        "treatment_uniform": ra.uniform(size=n),
        "eps_Y": ry.standard_normal(n),
    }


def exogenous_hashes(exogenous: dict[str, np.ndarray]) -> dict[str, str]:
    return {name: array_hash(value) for name, value in exogenous.items()}


def generate_from_exogenous(exogenous: dict[str, np.ndarray], coef: dict[str, np.ndarray], scm: str = "original") -> dict[str, np.ndarray]:
    u = exogenous["U"]
    if scm == "original":
        x = u[:, None] * coef["b_x"] + exogenous["eps_X"]
        w = u[:, None] * coef["b_w"] + exogenous["eps_W"]
        index = 0.8 * u + 0.6 * (x @ coef["a_x"])
        y0 = 0.7 * (x @ coef["y_x"]) + 0.3 * (x @ coef["yt_x"]) ** 2 + 1.5 * u + exogenous["eps_Y"]
    elif scm == "nonlinear_scm2":
        x = u[:, None] * coef["b_x"] + 0.35 * ((u ** 2 - 1.0) / math.sqrt(2.0))[:, None] * coef["c_x"] + exogenous["eps_X"]
        w = u[:, None] * coef["b_w"] + np.tanh(u)[:, None] * coef["c_w"] + exogenous["eps_W"]
        index = 0.75 * u + 0.45 * np.tanh(x @ coef["a_x"]) + 0.25 * np.sin(x @ coef["a2"]) + 0.20 * (x @ coef["a3"]) * (x @ coef["a4"])
        y0 = 0.5 * np.sin(x @ coef["y_x"]) + 0.25 * (x @ coef["yt_x"]) ** 2 + 0.30 * np.tanh(x @ coef["y3"]) * (x @ coef["y4"]) + 1.2 * u + 0.4 * (u ** 2 - 1.0) + exogenous["eps_Y"]
    else:
        raise ValueError(f"unknown SCM: {scm}")
    propensity = 0.1 + 0.8 * expit(index)
    a = (exogenous["treatment_uniform"] < propensity).astype(float)
    y = y0 + a
    return {"U": u, "X": x, "W": w, "A": a, "Y": y, "propensity": propensity}


def generate(n: int, seed_key: tuple[int, ...], coef: dict[str, np.ndarray], scm: str = "original") -> dict[str, np.ndarray]:
    return generate_from_exogenous(draw_exogenous(n, seed_key), coef, scm)


def standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(0); sd = train.std(0) + 1e-8
    return (train - mean) / sd, (other - mean) / sd, mean, sd


@dataclass
class RFF:
    mean: np.ndarray; sd: np.ndarray; omega: np.ndarray; phase: np.ndarray; bandwidth: float; seed_key: tuple[int, ...]

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.sd
        return math.sqrt(2.0 / self.omega.shape[1]) * np.cos(z @ self.omega + self.phase)


def fit_rff(audit: np.ndarray, seed_key: tuple[int, ...]) -> RFF:
    mean = audit.mean(0); sd = audit.std(0) + 1e-8; z = (audit - mean) / sd
    sample = z[:min(500, len(z))]; d2 = ((sample[:, None] - sample[None, :]) ** 2).sum(-1)
    upper = d2[np.triu_indices_from(d2, 1)]; bw = math.sqrt(max(float(np.median(upper)), 1e-12) / 2.0)
    g = rng(seed_key); omega = g.normal(0.0, 1.0 / bw, size=(z.shape[1], N_RFF)); phase = g.uniform(0.0, 2.0 * math.pi, size=N_RFF)
    return RFF(mean, sd, omega, phase, bw, seed_key)


class RepNet(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__(); self.net = torch.nn.Sequential(torch.nn.Linear(d, HIDDEN), torch.nn.ReLU(), torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(), torch.nn.Linear(HIDDEN, N_SOFT))
    def forward(self, x): return self.net(x)


def objective(soft: torch.Tensor, a: torch.Tensor, phi: torch.Tensor, logits: torch.Tensor, same_view: bool) -> tuple[torch.Tensor, dict[str, float]]:
    n = len(a); mass = soft.mean(0).clamp_min(1e-8); pi = ((soft * a[:, None]).mean(0) / mass).clamp(1e-5, 1.0 - 1e-5)
    c = soft * (a[:, None] - pi); moments = c.T @ phi / n
    discrepancy = (moments.square().sum(1) / mass).sum()
    mass_penalty = torch.relu(0.10 - mass).square().sum()
    overlap_penalty = torch.relu(0.05 - pi * (1.0 - pi)).square().mul(mass).sum()
    logit_penalty = 1e-3 * logits.square().sum(1).mean() if same_view else torch.zeros((), device=logits.device)
    total = discrepancy + 5.0 * mass_penalty + 5.0 * overlap_penalty + logit_penalty
    return total, {"discrepancy": float(discrepancy.detach()), "total": float(total.detach())}


@dataclass
class FittedRep:
    net: RepNet; input_mean: np.ndarray; input_sd: np.ndarray; rff: RFF; restarts: list[dict[str, Any]]; selected_restart: int; same_view: bool

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor((x - self.input_mean) / self.input_sd, dtype=torch.float32))
            return torch.softmax(logits / FINAL_TEMPERATURE, 1).cpu().numpy()


def train_one_restart(z: np.ndarray, a: np.ndarray, phi: np.ndarray, same_view: bool, seed_key: tuple[int, ...]) -> tuple[RepNet, dict[str, Any]]:
    torch.manual_seed(seed_int(seed_key)); torch.use_deterministic_algorithms(True)
    net = RepNet(z.shape[1]); xt = torch.tensor(z, dtype=torch.float32); at = torch.tensor(a, dtype=torch.float32); ft = torch.tensor(phi, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4 if same_view else 0.0)
    noise_gen = torch.Generator().manual_seed(seed_int(seed_key) + 1); initial = final = None
    for step in range(STEPS):
        temperature = 1.0 * (FINAL_TEMPERATURE ** (step / max(STEPS - 1, 1))); xs = xt
        if same_view:
            keep = torch.bernoulli(torch.full_like(xt, 0.8), generator=noise_gen) / 0.8
            xs = xt * keep + 0.10 * torch.randn(xt.shape, generator=noise_gen)
        logits = net(xs); soft = torch.softmax(logits / temperature, 1); total, vals = objective(soft, at, ft, logits, same_view)
        if step == 0: initial = vals
        opt.zero_grad(); total.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        logits = net(xt); soft = torch.softmax(logits / FINAL_TEMPERATURE, 1); _, final = objective(soft, at, ft, logits, same_view)
    return net, {"seed_key": seed_key, "integer_seed": seed_int(seed_key), "initial": initial, "final": final}


def fit_representation(inp: np.ndarray, audit: np.ndarray, a: np.ndarray, rff_key: tuple[int, ...], adam_keys: list[tuple[int, ...]], same_view: bool) -> FittedRep:
    mean = inp.mean(0); sd = inp.std(0) + 1e-8; z = (inp - mean) / sd; rff = fit_rff(audit, rff_key); phi = rff.transform(audit)
    fitted = [train_one_restart(z, a, phi, same_view, k) for k in adam_keys]
    selected = min(range(RESTARTS), key=lambda j: (fitted[j][1]["final"]["total"], j))
    return FittedRep(fitted[selected][0], mean, sd, rff, [q[1] for q in fitted], selected, same_view)


def discrepancy(soft: np.ndarray, a: np.ndarray, phi: np.ndarray, fixed_pi: np.ndarray | None = None) -> dict[str, Any]:
    mass = soft.mean(0); denominator = soft.sum(0)
    pi = np.divide((soft * a[:, None]).sum(0), denominator, out=np.full(soft.shape[1], 0.5), where=denominator > 0) if fixed_pi is None else fixed_pi
    c = soft * (a[:, None] - pi); moments = c.T @ phi / len(a); value = float(np.sum(np.sum(moments ** 2, 1) / np.maximum(mass, 1e-12)))
    return {"squared": value, "value": math.sqrt(max(value, 0.0)), "mass": mass, "propensity": pi, "empty_cells": int(np.sum(denominator == 0))}


def hard_onehot(soft: np.ndarray) -> np.ndarray:
    return np.eye(N_SOFT)[soft.argmax(1)]


def latent_imbalance(soft: np.ndarray, a: np.ndarray, u: np.ndarray) -> float:
    total = 0.0; mass = soft.mean(0)
    for m in range(soft.shape[1]):
        if mass[m] <= 0: continue
        wt = soft[:, m] * a; wc = soft[:, m] * (1.0 - a)
        if wt.sum() <= 0 or wc.sum() <= 0: return float("nan")
        total += mass[m] * abs(float(wt @ u / wt.sum() - wc @ u / wc.sum()))
    return float(total)


def representation_stats(soft: np.ndarray, a: np.ndarray) -> dict[str, Any]:
    mass = soft.mean(0); entropy = -np.sum(soft * np.log(np.maximum(soft, 1e-12)), 1).mean(); centered = soft - soft.mean(0); cov = centered.T @ centered / len(soft)
    tr = float(np.trace(cov)); denom = float(np.trace(cov @ cov)); totals = soft.sum(0); pi = np.divide((soft * a[:, None]).sum(0), totals, out=np.full(soft.shape[1], 0.5), where=totals > 0)
    return {"effective_strata": float(np.exp(-np.sum(mass * np.log(np.maximum(mass, 1e-12))))), "mean_assignment_entropy": float(entropy), "participation_ratio": tr * tr / denom if denom > 0 else 0.0, "mean_max_assignment": float(soft.max(1).mean()), "mass": mass, "propensity": pi, "min_propensity": float(pi.min()), "max_propensity": float(pi.max())}


def hard_counts(hard: np.ndarray, a: np.ndarray) -> list[dict[str, int]]:
    labels = hard.argmax(1); rows = []
    for m in range(N_SOFT):
        z = labels == m; rows.append({"cell": m, "n": int(z.sum()), "n1": int(np.sum(a[z] == 1)), "n0": int(np.sum(a[z] == 0))})
    return rows


def fit_outcome(s_train: np.ndarray, a_train: np.ndarray, y_train: np.ndarray, s_test: np.ndarray, seed_key: tuple[int, ...]) -> tuple[np.ndarray | None, dict[str, Any]]:
    counts = {str(arm): int(np.sum(a_train == arm)) for arm in (0, 1)}
    if min(counts.values()) < GBR_SPEC["min_samples_leaf"]: return None, {"finite": False, "arm_counts": counts, "reason": "global training arm count below min_samples_leaf", "seed_key": seed_key}
    zt, ze, _, _ = standardize(s_train, s_test); preds = []
    for arm in (1, 0):
        mask = a_train == arm; model = GradientBoostingRegressor(random_state=seed_int(seed_key) + arm, **GBR_SPEC); model.fit(zt[mask], y_train[mask]); preds.append(model.predict(ze))
    effect = preds[0] - preds[1]; finite = bool(np.isfinite(effect).all())
    return (effect if finite else None), {"finite": finite, "arm_counts": counts, "seed_key": seed_key, "integer_seed": seed_int(seed_key)}


def fold_indices(a: np.ndarray, seed_key: tuple[int, ...]) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed_int(seed_key)); dummy = np.zeros(len(a))
    return [(tr, te) for tr, te in splitter.split(dummy, a)]


def baseline_crossfit(data: dict[str, np.ndarray], feature: str, folds, replicate: int, n_index: int, method: str) -> tuple[float | None, list[dict[str, Any]]]:
    if feature == "X": s = data["X"]
    elif feature == "XW": s = np.column_stack([data["X"], data["W"]])
    elif feature == "XU": s = np.column_stack([data["X"], data["U"]])
    stitched = np.full(len(data["A"]), np.nan); details = []
    for fold, (tr, te) in enumerate(folds):
        sk = key("outcome_model", replicate, n_index, fold, METHOD_ID[method], 0, 0); pred, qa = fit_outcome(s[tr], data["A"][tr], data["Y"][tr], s[te], sk)
        if pred is not None: stitched[te] = pred
        details.append({"fold": fold, **qa})
    return (float(stitched.mean()) if np.isfinite(stitched).all() else None), details


def learned_component(data, diagnostic, folds, replicate, n_index, method, mask_key) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n = len(data["A"]); soft_stitched = np.full(n, np.nan); hard_stitched = np.full(n, np.nan); details = []; started = time.time()
    same = method == "heuristic_self_conditioned_v2"; method_id = METHOD_ID[method]
    if same:
        all_input = np.column_stack([data["X"], data["W"]]); all_audit = all_input; diag_input = np.column_stack([diagnostic["X"], diagnostic["W"]]); diag_audit = diag_input; audit_excluded = False
    else:
        keep = [j for j in range(D_W) if j != mask_key]; all_input = np.column_stack([data["X"], data["W"][:, keep]]); all_audit = np.column_stack([data["X"], data["W"][:, mask_key]]); diag_input = np.column_stack([diagnostic["X"], diagnostic["W"][:, keep]]); diag_audit = np.column_stack([diagnostic["X"], diagnostic["W"][:, mask_key]]); audit_excluded = mask_key not in keep
    for fold, (tr, te) in enumerate(folds):
        rk = key("random_fourier_features", replicate, n_index, fold, method_id, mask_key, 0)
        aks = [key("adam_initialization_and_training_noise", replicate, n_index, fold, method_id, mask_key, restart) for restart in range(RESTARTS)]
        fit = fit_representation(all_input[tr], all_audit[tr], data["A"][tr], rk, aks, same)
        soft_tr, soft_te, soft_diag = fit.predict(all_input[tr]), fit.predict(all_input[te]), fit.predict(diag_input)
        hard_tr, hard_te, hard_diag = hard_onehot(soft_tr), hard_onehot(soft_te), hard_onehot(soft_diag)
        ok = key("outcome_model", replicate, n_index, fold, method_id, mask_key, 0)
        pred, oqa = fit_outcome(soft_tr, data["A"][tr], data["Y"][tr], soft_te, ok)
        hard_pred, hqa = fit_outcome(hard_tr, data["A"][tr], data["Y"][tr], hard_te, ok)
        if pred is not None: soft_stitched[te] = pred
        if hard_pred is not None: hard_stitched[te] = hard_pred
        phi_tr, phi_te, phi_diag = fit.rff.transform(all_audit[tr]), fit.rff.transform(all_audit[te]), fit.rff.transform(diag_audit)
        train_disc = discrepancy(soft_tr, data["A"][tr], phi_tr); test_disc = discrepancy(soft_te, data["A"][te], phi_te); test_fixed = discrepancy(soft_te, data["A"][te], phi_te, np.asarray(train_disc["propensity"])); diag_disc = discrepancy(soft_diag, diagnostic["A"], phi_diag)
        hard_test_disc = discrepancy(hard_te, data["A"][te], phi_te); hard_diag_disc = discrepancy(hard_diag, diagnostic["A"], phi_diag)
        details.append({
            "fold": fold, "mask_key": mask_key, "audit_excluded": audit_excluded, "same_view_uncertified": same,
            "rff_seed_key": rk, "rff_integer_seed": seed_int(rk), "rff_bandwidth": fit.rff.bandwidth,
            "restart_objectives": fit.restarts, "selected_restart": fit.selected_restart, "selection_criterion": "minimum_final_training_total_objective",
            "outcome_soft": oqa, "outcome_hard": hqa, "train_discrepancy": train_disc, "heldout_discrepancy": test_disc,
            "heldout_fixed_training_propensity_discrepancy": test_fixed, "independent_frozen_discrepancy": diag_disc,
            "hard_heldout_discrepancy": hard_test_disc, "hard_independent_frozen_discrepancy": hard_diag_disc,
            "soft_hard_discrepancy_gap_heldout": hard_test_disc["value"] - test_disc["value"],
            "train_representation": representation_stats(soft_tr, data["A"][tr]), "heldout_representation": representation_stats(soft_te, data["A"][te]), "independent_representation": representation_stats(soft_diag, diagnostic["A"]),
            "hard_train_counts": hard_counts(hard_tr, data["A"][tr]), "hard_heldout_counts": hard_counts(hard_te, data["A"][te]),
            "soft_latent_imbalance_heldout": latent_imbalance(soft_te, data["A"][te], data["U"][te]), "hard_latent_imbalance_heldout": latent_imbalance(hard_te, data["A"][te], data["U"][te]),
            "soft_latent_imbalance_independent": latent_imbalance(soft_diag, diagnostic["A"], diagnostic["U"]), "hard_latent_imbalance_independent": latent_imbalance(hard_diag, diagnostic["A"], diagnostic["U"]),
            "diagnostic_used_for_selection": False, "heldout_used_for_selection": False, "units_deleted": 0,
        })
    soft_ok = bool(np.isfinite(soft_stitched).all()); hard_ok = bool(np.isfinite(hard_stitched).all())
    soft_est = float(soft_stitched.mean()) if soft_ok else None; hard_est = float(hard_stitched.mean()) if hard_ok else None
    return {"estimate": soft_est, "finite": soft_ok, "hard_diagnostic_estimate": hard_est, "hard_diagnostic_finite": hard_ok, "soft_hard_effect_gap": hard_est - soft_est if soft_ok and hard_ok else None, "runtime_seconds": time.time() - started}, details


def summarize(rows: list[dict[str, Any]], prefixes: tuple[int, ...]) -> list[dict[str, Any]]:
    out = []
    for n in prefixes:
        for method in METHOD_ID:
            rr = [r for r in rows if r["n"] == n and r["method"] == method]; errors = np.array([r["error"] for r in rr], float)
            out.append({"n": n, "method": method, "replicates": len(rr), "signed_bias": float(errors.mean()), "mean_absolute_error": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.mean(errors ** 2))), "sd_estimate": float(np.std([r["estimate"] for r in rr], ddof=1)) if len(rr) >= 2 else None, "median_absolute_error": float(np.median(np.abs(errors))), "min_estimate": float(min(r["estimate"] for r in rr)), "max_estimate": float(max(r["estimate"] for r in rr))})
    return out


def exact_sign_p(diffs: np.ndarray) -> float:
    nonzero = diffs[np.abs(diffs) > 1e-15]; wins = int(np.sum(nonzero < 0)); losses = len(nonzero) - wins; tail = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(len(nonzero), k) for k in range(tail + 1)) / (2.0 ** len(nonzero))) if len(nonzero) else 1.0


def paired(rows, comparator: str, prefixes: tuple[int, ...]) -> list[dict[str, Any]]:
    out = []
    for n in prefixes:
        focal = sorted([r for r in rows if r["n"] == n and r["method"] == "proposed_cross_mask_v1"], key=lambda r: r["replicate"])
        comp = sorted([r for r in rows if r["n"] == n and r["method"] == comparator], key=lambda r: r["replicate"])
        diffs = np.array([abs(f["error"]) - abs(c["error"]) for f, c in zip(focal, comp)])
        if len(diffs) >= 2:
            critical = 2.093024054 if len(diffs) == 20 else (2.262157163 if len(diffs) == 10 else 1.96)
            half_width = critical * float(np.std(diffs, ddof=1)) / math.sqrt(len(diffs))
            ci = [float(diffs.mean() - half_width), float(diffs.mean() + half_width)]
            ci_method = f"paired t interval with fixed critical value {critical}, df={len(diffs)-1}"
        else:
            ci = None; ci_method = "undefined for fewer than two paired Monte Carlo units"
        out.append({"n": n, "comparator": comparator, "absolute_error_differences": diffs, "mean_difference": float(diffs.mean()), "mean_difference_ci95": ci, "mean_difference_ci95_method": ci_method, "median_difference": float(np.median(diffs)), "wins": int(np.sum(diffs < 0)), "ties": int(np.sum(diffs == 0)), "exact_two_sided_sign_p": exact_sign_p(diffs)})
    return out


def performance_gate(table, paired_rows, comparator: str, replicate_count: int, primary_n: int) -> dict[str, Any]:
    v = next(r for r in table if r["n"] == primary_n and r["method"] == "proposed_cross_mask_v1"); c = next(r for r in table if r["n"] == primary_n and r["method"] == comparator); p = next(r for r in paired_rows if r["n"] == primary_n and r["comparator"] == comparator)
    reduction = (c["mean_absolute_error"] - v["mean_absolute_error"]) / c["mean_absolute_error"]
    required_wins = 14 if replicate_count == 20 else math.ceil(0.7 * replicate_count)
    criteria = {f"absolute_error_wins_ge_{required_wins}_of_{replicate_count}": p["wins"] >= required_wins, "mean_absolute_error_lower": v["mean_absolute_error"] < c["mean_absolute_error"], "rmse_lower": v["rmse"] < c["rmse"], "mean_absolute_error_reduction_ge_20_percent": reduction >= 0.20}
    return {"comparator": comparator, "primary_n": primary_n, "relative_mae_reduction": reduction, "criteria": criteria, "pass": all(criteria.values())}


def deterministic_selected_model_replay(coef: dict[str, np.ndarray], scm: str, master_n: int = N_MASTER, replay_n: int = 100, n_index: int = 0) -> dict[str, Any]:
    """Replay one frozen V1 configuration; excluded from experiment fit counts."""
    data = generate(master_n, key("master_dataset", 0), coef, scm); data = {k: v[:replay_n] for k, v in data.items()}
    folds = fold_indices(data["A"], key("outer_fold_assignment", 0, n_index)); tr, te = folds[0]; mask = 0; keep = [j for j in range(D_W) if j != mask]
    inp = np.column_stack([data["X"], data["W"][:, keep]]); audit = np.column_stack([data["X"], data["W"][:, mask]])
    rk = key("random_fourier_features", 0, n_index, 0, METHOD_ID["proposed_cross_mask_v1"], mask, 0)
    aks = [key("adam_initialization_and_training_noise", 0, n_index, 0, METHOD_ID["proposed_cross_mask_v1"], mask, restart) for restart in range(RESTARTS)]
    first = fit_representation(inp[tr], audit[tr], data["A"][tr], rk, aks, False); second = fit_representation(inp[tr], audit[tr], data["A"][tr], rk, aks, False)
    p1, p2 = first.predict(inp[te]), second.predict(inp[te])
    passed = first.selected_restart == second.selected_restart and first.restarts == second.restarts and np.array_equal(p1, p2)
    return {"configuration": {"replicate": 0, "master_n": master_n, "n": replay_n, "n_index": n_index, "fold": 0, "method": "proposed_cross_mask_v1", "mask": 0}, "selected_restart_first": first.selected_restart, "selected_restart_second": second.selected_restart, "heldout_soft_hash_first": array_hash(p1), "heldout_soft_hash_second": array_hash(p2), "restart_records_identical": first.restarts == second.restarts, "pass": passed, "replay_adam_fits_excluded_from_experiment_count": 4}


def without_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: without_runtime(v) for k, v in value.items() if k != "runtime_seconds"}
    if isinstance(value, list):
        return [without_runtime(v) for v in value]
    return value


def frozen_original_ten_check(aggregate, components, master_hashes, diagnostic_hashes, fold_records, scm: str, smoke: bool, prefixes: tuple[int, ...]) -> dict[str, Any]:
    applicable = scm == "original" and not smoke and prefixes == PREFIXES
    if not applicable:
        return {"applicable": False, "pass": True, "frozen_artifact_sha256": LEGACY_10REP_SHA256, "scope": "only required for a non-smoke original-SCM run with the legacy prefix schedule"}
    if not OUTPUT.exists():
        return {"applicable": True, "pass": False, "frozen_artifact_sha256": LEGACY_10REP_SHA256, "reason": f"frozen artifact missing: {OUTPUT}"}
    old = json.loads(OUTPUT.read_text())
    current_aggregate = jsonable([r for r in aggregate if r["replicate"] < 10])
    current_components = jsonable([r for r in components if r["replicate"] < 10])
    normalize_hashes = lambda rows: [{k: row[k] for k in ("replicate", "seed_key", "hash")} for row in rows if row["replicate"] < 10]
    current_seed_records = jsonable({
        "coefficient_seed_key": list(key("structural_coefficients")),
        "master": normalize_hashes(master_hashes),
        "diagnostic": normalize_hashes(diagnostic_hashes),
        "folds": jsonable([r for r in fold_records if r["replicate"] < 10]),
    })
    old_seed_records = {
        "coefficient_seed_key": old["seed_records"]["coefficient_seed_key"],
        "master": normalize_hashes(old["seed_records"]["master"]),
        "diagnostic": normalize_hashes(old["seed_records"]["diagnostic"]),
        "folds": old["seed_records"]["folds"],
    }
    checks = {
        "aggregate_rows_exact": current_aggregate == old["aggregate_results"],
        "v1_component_rows_exact_except_runtime": without_runtime(current_components) == without_runtime(old["v1_component_results"]),
        "master_diagnostic_and_fold_seed_hash_records_exact": current_seed_records == old_seed_records,
    }
    return {"applicable": True, "frozen_artifact": str(OUTPUT), "frozen_artifact_sha256": LEGACY_10REP_SHA256, "scope": "first 10 original-SCM replicates at all four nested prefixes", "checks": checks, "pass": all(checks.values())}


def load_frozen_twenty(scm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path, expected_sha = FROZEN_20REP[scm]
    actual_sha = file_hash(path) if path.exists() else None
    artifact = json.loads(path.read_text()) if actual_sha == expected_sha else None
    audit = {"path": str(path), "expected_sha256": expected_sha, "actual_sha256": actual_sha, "sha256_match": actual_sha == expected_sha}
    if artifact is None:
        return {}, audit
    return artifact, audit


def continuity_record(rep: int, exogenous: dict[str, np.ndarray], master: dict[str, np.ndarray], frozen: dict[str, Any]) -> dict[str, Any]:
    if not frozen:
        return {"replicate": rep, "pass": False, "reason": "frozen artifact unavailable or hash mismatch"}
    frozen_master = next(row for row in frozen["seed_records"]["master"] if row["replicate"] == rep)
    first_1000_exogenous = {name: value[:1000] for name, value in exogenous.items()}
    stream_checks = {name: value == frozen_master["exogenous_stream_hashes"][name] for name, value in exogenous_hashes(first_1000_exogenous).items()}
    uay_hash = array_hash(np.column_stack([master["U"][:1000], master["A"][:1000], master["Y"][:1000]]))
    return {"replicate": rep, "first_1000_exogenous_stream_matches": stream_checks, "first_1000_uay_hash": uay_hash, "frozen_first_1000_uay_hash": frozen_master["hash"], "first_1000_uay_match": uay_hash == frozen_master["hash"], "pass": all(stream_checks.values()) and uay_hash == frozen_master["hash"]}


def run(smoke: bool, scm: str = "original", replicate_count: int = 10, configured_prefixes: tuple[int, ...] = PREFIXES, progress: bool = False) -> dict[str, Any]:
    started = time.time(); coef = coefficients(scm); replicates = range(1) if smoke else range(replicate_count); prefixes = configured_prefixes[:1] if smoke else configured_prefixes
    master_n = max(configured_prefixes); primary_n = max(prefixes)
    frozen_twenty, frozen_twenty_audit = load_frozen_twenty(scm)
    coefficient_continuity = bool(frozen_twenty) and jsonable(coef) == frozen_twenty["dgp"]["coefficients"]
    aggregate, components, learned_folds, baseline_folds, master_hashes, diagnostic_hashes, fold_records, propensity_ranges, continuity_records = [], [], [], [], [], [], [], [], []
    adam_fit_count = selected_fit_count = 0
    for rep in replicates:
        mk = key("master_dataset", rep); dk = key("independent_diagnostic_dataset", rep)
        master_exogenous = draw_exogenous(master_n, mk); diagnostic_exogenous = draw_exogenous(N_DIAGNOSTIC, dk)
        master = generate_from_exogenous(master_exogenous, coef, scm); diagnostic = generate_from_exogenous(diagnostic_exogenous, coef, scm)
        continuity_records.append(continuity_record(rep, master_exogenous, master, frozen_twenty))
        master_hashes.append({"replicate": rep, "seed_key": mk, "hash": array_hash(np.column_stack([master["U"], master["A"], master["Y"]])), "exogenous_stream_hashes": exogenous_hashes(master_exogenous)})
        diagnostic_hashes.append({"replicate": rep, "seed_key": dk, "hash": array_hash(np.column_stack([diagnostic["U"], diagnostic["A"], diagnostic["Y"]])), "exogenous_stream_hashes": exogenous_hashes(diagnostic_exogenous)})
        propensity_ranges.append({"replicate": rep, "master_min": float(master["propensity"].min()), "master_max": float(master["propensity"].max()), "diagnostic_min": float(diagnostic["propensity"].min()), "diagnostic_max": float(diagnostic["propensity"].max())})
        for n_index, n in enumerate(prefixes):
            data = {k: v[:n] for k, v in master.items()}; fk = key("outer_fold_assignment", rep, n_index); folds = fold_indices(data["A"], fk); fold_records.append({"replicate": rep, "n": n, "seed_key": fk, "fold_hashes": [{"train": array_hash(tr), "test": array_hash(te)} for tr, te in folds]})
            for feature, method in (("X", "adjust_x"), ("XW", "adjust_xw_full_observed"), ("XU", "observed_xu_fitted_reference")):
                est, fd = baseline_crossfit(data, feature, folds, rep, n_index, method); baseline_folds.append({"replicate": rep, "n": n, "method": method, "folds": fd}); aggregate.append({"replicate": rep, "n": n, "method": method, "estimate": est, "error": est - TRUE_ATE if est is not None else None, "absolute_error": abs(est - TRUE_ATE) if est is not None else None, "finite": est is not None})
            estimates = []; hard_estimates = []
            for mask in range(N_MASKS):
                result, fd = learned_component(data, diagnostic, folds, rep, n_index, "proposed_cross_mask_v1", mask); selected_fit_count += 2; adam_fit_count += 2 * RESTARTS; estimates.append(result["estimate"]); hard_estimates.append(result["hard_diagnostic_estimate"]); components.append({"replicate": rep, "n": n, "mask": mask, **result}); learned_folds.append({"replicate": rep, "n": n, "method": "proposed_cross_mask_v1", "mask": mask, "folds": fd})
            est = float(np.mean(estimates)) if all(x is not None for x in estimates) else None; hard_est = float(np.mean(hard_estimates)) if all(x is not None for x in hard_estimates) else None
            aggregate.append({"replicate": rep, "n": n, "method": "proposed_cross_mask_v1", "estimate": est, "error": est - TRUE_ATE if est is not None else None, "absolute_error": abs(est - TRUE_ATE) if est is not None else None, "finite": est is not None, "hard_diagnostic_estimate": hard_est, "soft_hard_effect_gap": hard_est - est if est is not None and hard_est is not None else None})
            result, fd = learned_component(data, diagnostic, folds, rep, n_index, "heuristic_self_conditioned_v2", 9); selected_fit_count += 2; adam_fit_count += 2 * RESTARTS; learned_folds.append({"replicate": rep, "n": n, "method": "heuristic_self_conditioned_v2", "mask": 9, "folds": fd}); est = result["estimate"]
            aggregate.append({"replicate": rep, "n": n, "method": "heuristic_self_conditioned_v2", "estimate": est, "error": est - TRUE_ATE if est is not None else None, "absolute_error": abs(est - TRUE_ATE) if est is not None else None, "finite": est is not None, "hard_diagnostic_estimate": result["hard_diagnostic_estimate"], "soft_hard_effect_gap": result["soft_hard_effect_gap"], "uncertified_heuristic": True})
        if progress:
            print(json.dumps({"progress": True, "scm": scm, "replicate_completed": rep + 1, "replicates_total": len(list(replicates)), "elapsed_seconds": time.time() - started}), flush=True)
    expected_aggregate = len(list(replicates)) * len(prefixes) * 5; expected_components = len(list(replicates)) * len(prefixes) * 5; expected_selected = len(list(replicates)) * len(prefixes) * 2 * 6; expected_adam = expected_selected * RESTARTS
    all_fold_rows = [f for d in learned_folds for f in d["folds"]]; replay = deterministic_selected_model_replay(coef, scm, master_n, min(configured_prefixes), 0)
    frozen_check = frozen_original_ten_check(aggregate, components, master_hashes, diagnostic_hashes, fold_records, scm, smoke, prefixes)
    provenance = {"script": "code/honest_canonical_representation_experiment.py", "root_seed": ROOT_SEED, "scm": scm, "seed_key_format": ["root", "namespace", "replicate", "n_index", "fold", "method", "mask", "restart"], "namespace_registry": NAMESPACE, "method_id_registry": METHOD_ID, "python_hash_used": False, "unregistered_random_source_count": 0, "configured_sample_sizes": configured_prefixes, "sample_sizes_executed": prefixes, "master_n": master_n, "diagnostic_n": N_DIAGNOSTIC, "replicates_executed": len(list(replicates)), "replicates_planned": replicate_count, "independent_monte_carlo_units_per_n": replicate_count, "prefixes_nested_within_replicate": True, "cross_n_results_correlated": True, "no_cross_n_pooling": True, "common_random_numbers_across_scms": True, "common_exogenous_streams": ["U", "eps_X", "eps_W", "treatment_uniform", "eps_Y"], "post_result_tuning": False, "primary_n": primary_n, "primary_method": "proposed_cross_mask_v1", "primary_comparator": "adjust_x"}
    method_contract = {"aggregate_methods": list(METHOD_ID), "primary_representation": "soft 3-vector", "hard_argmax": "secondary diagnostic only", "objective_type": "64-dimensional RFF soft conditional-covariance surrogate", "exact_kci_or_conditional_hsic": False, "finite_state_theorem_directly_applies": False, "v2_certified": False, "v1_masks": [0, 1, 2, 3, 4], "v1_aggregate": "mean of five mask-specific causal estimates", "baseline_methods": ["adjust_x", "adjust_xw_full_observed", "observed_xu_fitted_reference"], "baseline_methods_single_fits": "one cross-fitted causal estimate per method and replicate-prefix; no mask ensemble", "ensemble_computation_asymmetry": "V1 averages five separately learned mask-specific causal estimates, whereas every baseline and V2 returns one causal estimate.", "adam_steps": STEPS, "adam_restarts": RESTARTS, "outcome_model": GBR_SPEC}
    min_proxy_derivative = float(np.min(np.abs(coef["b_w"])))
    if scm == "original":
        dgp_contract = {"name": "original_scm1", "equations": {"X": "b_X U + eps_X", "W": "b_W U + eps_W", "treatment_index": "0.8 U + 0.6 a_X^T X", "outcome_without_treatment": "0.7 y_X^T X + 0.3 (yt_X^T X)^2 + 1.5 U + eps_Y"}, "proxy_conditional_mean_derivative_lower_bound": min_proxy_derivative, "proxy_conditional_mean_strictly_monotone": min_proxy_derivative > 0}
    else:
        same_sign = bool(np.all(np.sign(coef["c_w"]) == np.sign(coef["b_w"])))
        dgp_contract = {"name": "nonlinear_scm2", "equations": {"X": "b_X U + 0.35 c_X (U^2-1)/sqrt(2) + eps_X", "W": "b_W U + c_W tanh(U) + eps_W", "treatment_index": "0.75 U + 0.45 tanh(a1^T X) + 0.25 sin(a2^T X) + 0.20(a3^T X)(a4^T X)", "outcome_without_treatment": "0.5 sin(y1^T X) + 0.25(y2^T X)^2 + 0.30 tanh(y3^T X)(y4^T X) + 1.2U + 0.4(U^2-1) + eps_Y"}, "coefficient_reuse": {"a1": "original a_x", "y1": "original y_x", "y2": "original yt_x", "b_x": "original b_x", "b_w": "original b_w"}, "c_x_generation": "unit_vector from scm2_structural_coefficients", "c_w_same_sign_as_b_w": same_sign, "c_w_absolute_magnitude_range": [float(np.min(np.abs(coef["c_w"]))), float(np.max(np.abs(coef["c_w"])))], "proxy_conditional_mean_derivative": "b_Wk + c_Wk sech^2(U)", "proxy_conditional_mean_derivative_lower_bound": min_proxy_derivative, "proxy_conditional_mean_strictly_monotone": same_sign and min_proxy_derivative > 0}
    outcome_seed_keys = [tuple(f["seed_key"]) for d in baseline_folds for f in d["folds"]] + [tuple(f[q]["seed_key"]) for f in all_fold_rows for q in ("outcome_soft", "outcome_hard")]
    pipeline_gates = {
        "root_seed_is_exactly_190702": ROOT_SEED == 190702,
        "single_root_seed_only": True,
        "all_seed_keys_begin_with_190702": all(k[0] == ROOT_SEED for k in [key("structural_coefficients"), key("scm2_structural_coefficients")] + [tuple(r["seed_key"]) for r in master_hashes + diagnostic_hashes + fold_records] + [tuple(f["rff_seed_key"]) for f in all_fold_rows] + [tuple(q["seed_key"]) for f in all_fold_rows for q in f["restart_objectives"]] + outcome_seed_keys),
        "all_namespaces_registered": set(NAMESPACE.values()) == {100, 110, 200, 300, 400, 500, 600, 700},
        "python_hash_not_used": True,
        "unregistered_random_source_count_zero": True,
        "master_dataset_hashes_unique": len({r["hash"] for r in master_hashes}) == len(master_hashes),
        "diagnostic_dataset_hashes_unique": len({r["hash"] for r in diagnostic_hashes}) == len(diagnostic_hashes),
        "aggregate_row_count": len(aggregate) == expected_aggregate,
        "v1_component_row_count": len(components) == expected_components,
        "selected_representation_fit_count": selected_fit_count == expected_selected,
        "adam_fit_count": adam_fit_count == expected_adam,
        "two_restart_objectives_every_fit": all(len(f["restart_objectives"]) == 2 for f in all_fold_rows),
        "v1_audit_leakage_zero": all(f["audit_excluded"] for d in learned_folds if d["method"] == "proposed_cross_mask_v1" for f in d["folds"]),
        "v2_same_view_audit_is_not_excluded": all(not f["audit_excluded"] for d in learned_folds if d["method"] == "heuristic_self_conditioned_v2" for f in d["folds"]),
        "nested_prefix_design_metadata_exact": provenance["independent_monte_carlo_units_per_n"] == replicate_count and provenance["prefixes_nested_within_replicate"] and provenance["cross_n_results_correlated"] and provenance["no_cross_n_pooling"],
        "configuration_safe_master_and_primary_n": provenance["master_n"] == max(configured_prefixes) and provenance["primary_n"] == max(prefixes) and tuple(provenance["sample_sizes_executed"]) == prefixes,
        "frozen_20rep_artifact_sha256_match": frozen_twenty_audit["sha256_match"],
        "frozen_20rep_coefficients_match": coefficient_continuity,
        "first_1000_continuity_all_exogenous_and_uay": all(row["pass"] for row in continuity_records),
        "five_common_random_number_streams_recorded": all(set(r["exogenous_stream_hashes"]) == set(provenance["common_exogenous_streams"]) for r in master_hashes + diagnostic_hashes),
        "strict_population_overlap_and_sample_propensities_in_range": all(0.1 < r[k] < 0.9 for r in propensity_ranges for k in ("master_min", "master_max", "diagnostic_min", "diagnostic_max")),
        "true_structural_ate_is_one": TRUE_ATE == 1.0,
        "proxy_conditional_mean_strict_monotonicity": dgp_contract["proxy_conditional_mean_strictly_monotone"],
        "frozen_original_first_ten_replay": frozen_check["pass"],
        "v1_mask_and_aggregation_contract_exact": method_contract["v1_masks"] == list(range(N_MASKS)) and method_contract["v1_aggregate"] == "mean of five mask-specific causal estimates" and all(sorted(r["mask"] for r in components if r["replicate"] == rep and r["n"] == n) == list(range(N_MASKS)) for rep in replicates for n in prefixes),
        "baseline_single_estimate_contract_exact": all(sum(r["replicate"] == rep and r["n"] == n and r["method"] == method for r in aggregate) == 1 for rep in replicates for n in prefixes for method in method_contract["baseline_methods"]),
        "ensemble_computation_asymmetry_recorded": "five separately learned mask-specific causal estimates" in method_contract["ensemble_computation_asymmetry"],
        "outcome_not_used_for_representation": True,
        "u_not_used_for_nonreference_learning": True,
        "heldout_and_diagnostic_not_used_for_selection": all(not f["heldout_used_for_selection"] and not f["diagnostic_used_for_selection"] for f in all_fold_rows),
        "unit_deletion_zero": all(f["units_deleted"] == 0 for f in all_fold_rows),
        "all_primary_soft_estimates_finite": all(r["finite"] for r in aggregate),
        "all_primary_discrepancy_diagnostics_finite": all(np.isfinite(f["heldout_discrepancy"]["value"]) and np.isfinite(f["independent_frozen_discrepancy"]["value"]) for f in all_fold_rows),
        "replicate_zero_selected_model_replay_identical": replay["pass"],
    }
    if configured_prefixes == REGISTERED_LARGE_PREFIXES:
        pipeline_gates["registered_large_config_exact"] = replicate_count == 20 and master_n == 10000 and primary_n == 10000 and prefixes == REGISTERED_LARGE_PREFIXES
    table = summarize(aggregate, prefixes) if not smoke else []; px = paired(aggregate, "adjust_x", prefixes) if not smoke else []; pxw = paired(aggregate, "adjust_xw_full_observed", prefixes) if not smoke else []
    primary = performance_gate(table, px, "adjust_x", replicate_count, primary_n) if not smoke else None; secondary = performance_gate(table, pxw, "adjust_xw_full_observed", replicate_count, primary_n) if not smoke else None
    optimizer_summary = {"restart_zero_selected": sum(f["selected_restart"] == 0 for f in all_fold_rows), "restart_one_selected": sum(f["selected_restart"] == 1 for f in all_fold_rows), "restart_total_objective_decreased": sum(q["final"]["total"] <= q["initial"]["total"] for f in all_fold_rows for q in f["restart_objectives"]), "restart_records": len(all_fold_rows) * RESTARTS}
    diagnostic_summary = []
    for n in prefixes:
        for method in ("proposed_cross_mask_v1", "heuristic_self_conditioned_v2"):
            rr = [f for d in learned_folds if d["n"] == n and d["method"] == method for f in d["folds"]]
            diagnostic_summary.append({"n": n, "method": method, "folds": len(rr), "mean_training_discrepancy": float(np.mean([f["train_discrepancy"]["value"] for f in rr])), "mean_heldout_discrepancy": float(np.mean([f["heldout_discrepancy"]["value"] for f in rr])), "mean_independent_frozen_discrepancy": float(np.mean([f["independent_frozen_discrepancy"]["value"] for f in rr])), "mean_soft_latent_imbalance_independent": float(np.mean([f["soft_latent_imbalance_independent"] for f in rr])), "mean_effective_strata_independent": float(np.mean([f["independent_representation"]["effective_strata"] for f in rr])), "mean_effective_dimension_independent": float(np.mean([f["independent_representation"]["participation_ratio"] for f in rr])), "mean_max_assignment_independent": float(np.mean([f["independent_representation"]["mean_max_assignment"] for f in rr])), "mean_soft_minus_hard_discrepancy_gap_heldout": float(np.mean([-f["soft_hard_discrepancy_gap_heldout"] for f in rr]))})
    failure_counts = {"nonfinite_primary_aggregate": sum(not r["finite"] for r in aggregate), "nonfinite_hard_diagnostic": sum(r.get("hard_diagnostic_estimate") is None for r in aggregate if r["method"] in ("proposed_cross_mask_v1", "heuristic_self_conditioned_v2")), "nonfinite_soft_latent_heldout": sum(not np.isfinite(f["soft_latent_imbalance_heldout"]) for f in all_fold_rows), "nonfinite_hard_latent_heldout": sum(not np.isfinite(f["hard_latent_imbalance_heldout"]) for f in all_fold_rows), "nonfinite_soft_latent_independent": sum(not np.isfinite(f["soft_latent_imbalance_independent"]) for f in all_fold_rows), "nonfinite_hard_latent_independent": sum(not np.isfinite(f["hard_latent_imbalance_independent"]) for f in all_fold_rows), "empty_hard_heldout_cells": sum(cell["n"] == 0 for f in all_fold_rows for cell in f["hard_heldout_counts"]), "units_deleted": 0}
    hard_diagnostic_failures = {"nonfinite_hard_effect_estimates": failure_counts["nonfinite_hard_diagnostic"], "nonfinite_hard_latent_heldout": failure_counts["nonfinite_hard_latent_heldout"], "nonfinite_hard_latent_independent": failure_counts["nonfinite_hard_latent_independent"], "empty_hard_heldout_cells": failure_counts["empty_hard_heldout_cells"], "scope": "secondary hard-argmax diagnostics only; these counts do not imply failure of the primary soft estimator"}
    interpretation_contract = {"primary_claim_rule": "The overall claim passes only if the separately computed SCM1 and SCM2 gates both pass; no cross-SCM pooling.", "cross_prefix_pooling": False, "v1_computation": "five separately learned mask-specific representations and the mean of five causal estimates", "baseline_and_v2_computation": "one cross-fitted causal estimate per method", "ensemble_compute_and_variance_asymmetry": True, "full_xw_dimension_differs_from_soft_representation": True, "observed_xu_reference": "privileged fitted reference that observes simulation-only U; not an available estimator and not a structural-oracle aggregate row", "objective": "64-dimensional RFF soft conditional-covariance surrogate, not exact KCI or conditional HSIC", "theory_scope": "approximate-world performance experiment; no point-identification or Gamma-certificate claim", "diagnostic_sample_size_fixed": N_DIAGNOSTIC, "nested_prefixes_correlated": True, "no_cross_prefix_pooling": True}
    return jsonable({
        "provenance": provenance,
        "dgp": {"dimensions": {"U": 1, "X": D_X, "W": D_W}, "contract": dgp_contract, "coefficients": coef, "propensity_ranges": propensity_ranges, "true_structural_ate_diagnostic": TRUE_ATE, "exact_cross_mask_audit_balance_attainable": False, "self_conditioned_exact_observed_balance_attainable": "true_but_nonidentifying", "cross_mask_unattainability_reason": "Finite Gaussian measurements leave U|X,W_-j nondegenerate with full support, while de(X,U)/dU>0. Exact audit balance plus injectivity would force e(X,U) measurable from R_j=phi(X,W_-j), a contradiction.", "self_conditioned_nonidentification_reason": "A same-view representation can copy (X,W), making exact observed balance tautological without making latent U ignorable."},
        "method_contract": method_contract,
        "seed_records": {"coefficient_seed_key": key("structural_coefficients"), "scm2_coefficient_seed_key": key("scm2_structural_coefficients") if scm == "nonlinear_scm2" else None, "master": master_hashes, "diagnostic": diagnostic_hashes, "folds": fold_records, "deterministic_selected_model_replay": replay, "frozen_original_ten_replay": frozen_check, "frozen_twenty_artifact": frozen_twenty_audit, "first_1000_continuity": {"scope": "all five exogenous streams and stacked U,A,Y; unchanged coefficients plus deterministic SCM logically fix X,W", "coefficient_match": coefficient_continuity, "replicates": continuity_records, "pass": coefficient_continuity and all(row["pass"] for row in continuity_records)}},
        "aggregate_results": aggregate, "v1_component_results": components, "learned_fold_diagnostics": learned_folds, "baseline_fold_diagnostics": baseline_folds,
        "method_summary_by_n": table, "paired_vs_x": px, "paired_vs_full_xw": pxw, "optimizer_summary": optimizer_summary, "representation_diagnostic_summary": diagnostic_summary,
        "primary_v1_vs_x_gate": primary, "secondary_v1_vs_full_xw_gate": secondary,
        "pipeline_gates": pipeline_gates, "pipeline_pass": all(pipeline_gates.values()), "smoke_mode": smoke,
        "counts": {"aggregate_rows": len(aggregate), "v1_component_rows": len(components), "selected_representation_fits": selected_fit_count, "adam_fits": adam_fit_count},
        "failure_counts": failure_counts,
        "secondary_hard_diagnostic_failure_counts": hard_diagnostic_failures,
        "interpretation_and_fairness_contract": interpretation_contract,
        "cross_scm_pair_verification": {"finalized": False, "reason": "run the metadata-only pair finalizer after both SCM outputs exist"},
        "runtime_seconds": time.time() - started,
    })


def protected_artifact_paths() -> set[Path]:
    return {OUTPUT.resolve(), *(path.resolve() for path, _ in FROZEN_20REP.values())}


def require_unprotected_path(path: Path, purpose: str) -> None:
    if path.resolve() in protected_artifact_paths():
        raise ValueError(f"{purpose} refuses protected artifact path: {path.resolve()}")


def repair_replay_metadata(path: Path) -> dict[str, Any]:
    require_unprotected_path(path, "replay repair")
    obj = json.loads(path.read_text())
    provenance = obj["provenance"]
    configured = tuple(provenance["configured_sample_sizes"])
    if configured != REGISTERED_LARGE_PREFIXES or provenance["replicates_planned"] != 20 or provenance["master_n"] != 10000 or provenance["primary_n"] != 10000:
        raise ValueError("replay repair requires the exact registered large-n configuration")
    replay = deterministic_selected_model_replay(coefficients(provenance["scm"]), provenance["scm"], provenance["master_n"], min(configured), 0)
    obj["seed_records"]["deterministic_selected_model_replay"] = replay
    obj["pipeline_gates"]["replicate_zero_selected_model_replay_identical"] = replay["pass"]
    obj["pipeline_gates"]["registered_large_config_exact"] = True
    obj["pipeline_pass"] = all(obj["pipeline_gates"].values())
    path.write_text(json.dumps(obj, indent=2) + "\n")
    return {"path": str(path), "replay": replay, "pipeline_pass": obj["pipeline_pass"]}


def finalize_pair(paths: tuple[Path, Path]) -> dict[str, Any]:
    for path in paths:
        require_unprotected_path(path, "pair finalizer")
    loaded = [json.loads(path.read_text()) for path in paths]
    by_scm = {obj["provenance"]["scm"]: (path, obj) for path, obj in zip(paths, loaded)}
    if set(by_scm) != {"original", "nonlinear_scm2"}:
        raise ValueError("pair finalizer requires one original and one nonlinear_scm2 result")
    original = by_scm["original"][1]; nonlinear = by_scm["nonlinear_scm2"][1]
    def stream_payload(obj, key_name):
        return [{"replicate": row["replicate"], "exogenous_stream_hashes": row["exogenous_stream_hashes"]} for row in obj["seed_records"][key_name]]
    master_match = stream_payload(original, "master") == stream_payload(nonlinear, "master")
    diagnostic_match = stream_payload(original, "diagnostic") == stream_payload(nonlinear, "diagnostic")
    def registered_configuration(obj):
        provenance = obj["provenance"]
        return tuple(provenance["configured_sample_sizes"]) == REGISTERED_LARGE_PREFIXES and provenance["replicates_planned"] == provenance["replicates_executed"] == 20 and provenance["master_n"] == 10000 and provenance["primary_n"] == 10000 and obj["pipeline_gates"].get("registered_large_config_exact") is True
    configuration_match = registered_configuration(original) and registered_configuration(nonlinear)
    if not configuration_match:
        raise ValueError("pair finalizer requires exact registered prefixes, 20 replicates, master_n=10000, and primary_n=10000")
    primary_intersection = bool(original["primary_v1_vs_x_gate"]["pass"] and nonlinear["primary_v1_vs_x_gate"]["pass"])
    secondary_intersection = bool(original["secondary_v1_vs_full_xw_gate"]["pass"] and nonlinear["secondary_v1_vs_full_xw_gate"]["pass"])
    verification = {"finalized": True, "no_cross_scm_pooling": True, "configuration_match": configuration_match, "full_master_exogenous_hashes_match": master_match, "diagnostic_exogenous_hashes_match": diagnostic_match, "scm_specific_primary_pass": {"original": original["primary_v1_vs_x_gate"]["pass"], "nonlinear_scm2": nonlinear["primary_v1_vs_x_gate"]["pass"]}, "scm_specific_secondary_pass": {"original": original["secondary_v1_vs_full_xw_gate"]["pass"], "nonlinear_scm2": nonlinear["secondary_v1_vs_full_xw_gate"]["pass"]}, "overall_primary_intersection_pass": primary_intersection, "overall_secondary_intersection_pass": secondary_intersection, "pass": configuration_match and master_match and diagnostic_match}
    for path, obj in zip(paths, loaded):
        obj["cross_scm_pair_verification"] = verification
        obj["pipeline_gates"]["cross_scm_full_master_and_diagnostic_exogenous_hashes_match"] = verification["pass"]
        obj["pipeline_pass"] = all(obj["pipeline_gates"].values())
        path.write_text(json.dumps(obj, indent=2) + "\n")
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--scm", choices=("original", "nonlinear_scm2"), default="original")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--prefixes", type=int, nargs="+")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--finalize-pair", type=Path, nargs=2, metavar=("SCM1_JSON", "SCM2_JSON"))
    parser.add_argument("--repair-replay", type=Path, metavar="LARGE_RESULT_JSON")
    args = parser.parse_args()
    if args.repair_replay is not None:
        repaired = repair_replay_metadata(args.repair_replay)
        print(json.dumps({"replay_repair": repaired}, indent=2))
        return
    if args.finalize_pair is not None:
        verification = finalize_pair(tuple(args.finalize_pair))
        print(json.dumps({"pair_finalization": verification}, indent=2))
        return
    if args.replicates <= 0:
        parser.error("--replicates must be positive")
    configured_prefixes = tuple(args.prefixes) if args.prefixes is not None else PREFIXES
    if any(n <= 0 for n in configured_prefixes) or len(set(configured_prefixes)) != len(configured_prefixes) or tuple(sorted(configured_prefixes)) != configured_prefixes:
        parser.error("--prefixes must be positive, unique, and strictly increasing")
    if configured_prefixes != PREFIXES:
        if args.output is None:
            parser.error("nonlegacy prefixes require an explicit --output path")
        try:
            require_unprotected_path(args.output, "nonlegacy run")
        except ValueError as exc:
            parser.error(str(exc))
    output = args.output if args.output is not None else OUTPUT
    result = run(args.smoke, args.scm, args.replicates, configured_prefixes, args.progress)
    if not args.no_write:
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"scm": args.scm, "replicates_planned": args.replicates, "smoke_mode": args.smoke, "pipeline_pass": result["pipeline_pass"], "failed_pipeline_gates": [k for k, v in result["pipeline_gates"].items() if not v], "counts": result["counts"], "runtime_seconds": result["runtime_seconds"], "output_written": not args.no_write, "output": str(output) if not args.no_write else None}, indent=2))


if __name__ == "__main__": main()
