#!/usr/bin/env python3
"""Honest canonical SCM generator and assumption sanity audit.

This script contains no representation learning.  It freezes structural
coefficients before drawing data, generates two scalar-U regimes, and writes
only machine-readable diagnostics to results/honest_canonical_scm_sanity.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


ROOT_SEED = 1950702
DATA_SEED = 1950703
N = 200_000
D_X = 20
D_W = 10
OUTPUT = Path(__file__).resolve().parents[1] / "results" / "honest_canonical_scm_sanity.json"


def expit(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    ex = np.exp(x[~positive])
    out[~positive] = ex / (1.0 + ex)
    return out


def array_hash(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(str(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


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


def unit_vector(rng: np.random.Generator, d: int) -> np.ndarray:
    vector = rng.standard_normal(d)
    return vector / np.linalg.norm(vector)


def frozen_coefficients() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(ROOT_SEED)
    b_x = unit_vector(rng, D_X)
    signs = rng.choice(np.array([-1.0, 1.0]), size=D_W)
    b_w = signs * rng.uniform(0.35, 1.0, size=D_W)
    return {
        "b_x": b_x,
        "b_w": b_w,
        "a_x": unit_vector(rng, D_X),
        "y_x": unit_vector(rng, D_X),
        "yt_x": unit_vector(rng, D_X),
    }


def generate(regime: str, seed: int, coef: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(7)]
    rng_u, rng_c, rng_z, rng_x, rng_w, rng_a, rng_y = streams
    if regime == "gaussian":
        u = rng_u.standard_normal(N)
    elif regime == "standardized_mixture":
        c = rng_c.binomial(1, 0.5, size=N)
        z = rng_z.standard_normal(N)
        u = ((2.0 * c - 1.0) + 0.5 * z) / math.sqrt(1.25)
    else:
        raise ValueError(regime)

    eps_x = rng_x.standard_normal((N, D_X))
    eps_w = rng_w.standard_normal((N, D_W))
    eps_a = rng_a.uniform(0.0, 1.0, size=N)
    eps_y = rng_y.standard_normal(N)
    x = u[:, None] * coef["b_x"][None, :] + eps_x
    w = u[:, None] * coef["b_w"][None, :] + eps_w
    ax = x @ coef["a_x"]
    index = 0.8 * u + 0.6 * ax
    propensity = 0.1 + 0.8 * expit(index)
    a = (eps_a < propensity).astype(float)
    baseline = 0.7 * (x @ coef["y_x"]) + 0.3 * (x @ coef["yt_x"]) ** 2 + 1.5 * u + eps_y
    y0 = baseline
    y1 = baseline + 1.0
    y = np.where(a == 1.0, y1, y0)
    return {
        "U": u,
        "X": x,
        "W": w,
        "A": a,
        "Y": y,
        "Y0": y0,
        "Y1": y1,
        "propensity": propensity,
        "index": index,
        "a_x_score": ax,
    }


def linear_r2_and_residual(u: np.ndarray, features: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(u)), features])
    beta, *_ = np.linalg.lstsq(design, u, rcond=None)
    residual = u - design @ beta
    r2 = 1.0 - float(residual @ residual) / float(((u - u.mean()) ** 2).sum())
    return r2, float(np.var(residual))


def treatment_logloss_diagnostic(data: dict[str, np.ndarray]) -> dict[str, Any]:
    split = int(0.7 * N)
    x_train, x_test = data["X"][:split], data["X"][split:]
    xu_train = np.column_stack([x_train, data["U"][:split]])
    xu_test = np.column_stack([x_test, data["U"][split:]])
    a_train, a_test = data["A"][:split], data["A"][split:]
    spec = dict(C=1e6, max_iter=1000, solver="lbfgs", random_state=ROOT_SEED)
    model_x = LogisticRegression(**spec).fit(x_train, a_train)
    model_xu = LogisticRegression(**spec).fit(xu_train, a_train)
    loss_x = float(log_loss(a_test, model_x.predict_proba(x_test)[:, 1]))
    loss_xu = float(log_loss(a_test, model_xu.predict_proba(xu_test)[:, 1]))
    return {
        "heldout_fraction": 0.3,
        "log_loss_x": loss_x,
        "log_loss_x_plus_u": loss_xu,
        "improvement": loss_x - loss_xu,
        "fitted_u_coefficient": float(model_xu.coef_[0, -1]),
        "is_diagnostic_not_proof": True,
    }


def diagnose(regime: str, data: dict[str, np.ndarray], coef: dict[str, np.ndarray]) -> dict[str, Any]:
    u, x, w = data["U"], data["X"], data["W"]
    corr_ux = np.array([np.corrcoef(u, x[:, j])[0, 1] for j in range(D_X)])
    corr_uw = np.array([np.corrcoef(u, w[:, j])[0, 1] for j in range(D_W)])
    cov_ux = ((u - u.mean())[:, None] * (x - x.mean(0))).mean(0)
    cov_uw = ((u - u.mean())[:, None] * (w - w.mean(0))).mean(0)
    r2_x, residual_x = linear_r2_and_residual(u, x)
    r2_w, residual_w = linear_r2_and_residual(u, w)
    p_u_zero = 0.1 + 0.8 * expit(0.6 * data["a_x_score"])
    p_x_zero = 0.1 + 0.8 * expit(0.8 * u)
    derivative = 0.64 * expit(data["index"]) * (1.0 - expit(data["index"]))
    ite = data["Y1"] - data["Y0"]
    all_arrays_finite = all(np.isfinite(value).all() for value in data.values())
    metrics = {
        "u_mean": float(u.mean()),
        "u_variance": float(u.var()),
        "corr_u_x": corr_ux,
        "corr_u_w": corr_uw,
        "cov_u_x_l2": float(np.linalg.norm(cov_ux)),
        "cov_u_w_l2": float(np.linalg.norm(cov_uw)),
        "linear_r2_u_on_x": r2_x,
        "linear_r2_u_on_w": r2_w,
        "linear_residual_variance_u_after_x": residual_x,
        "linear_residual_variance_u_after_w": residual_w,
        "mean_abs_direct_u_propensity_contribution": float(np.mean(np.abs(data["propensity"] - p_u_zero))),
        "mean_abs_x_propensity_contribution": float(np.mean(np.abs(data["propensity"] - p_x_zero))),
        "propensity_min": float(data["propensity"].min()),
        "propensity_max": float(data["propensity"].max()),
        "propensity_violation_count": int(np.sum((data["propensity"] <= 0.1) | (data["propensity"] >= 0.9))),
        "fixed_x_derivative_min": float(derivative.min()),
        "fixed_x_derivative_max": float(derivative.max()),
        "treatment_rate": float(data["A"].mean()),
        "structural_ate": float(ite.mean()),
        "max_abs_ite_minus_one": float(np.max(np.abs(ite - 1.0))),
        "all_arrays_finite": all_arrays_finite,
        "treatment_logloss": treatment_logloss_diagnostic(data),
    }
    gates = {
        "u_mean_abs_lt_0_02": abs(metrics["u_mean"]) < 0.02,
        "u_variance_within_0_03_of_one": abs(metrics["u_variance"] - 1.0) < 0.03,
        "min_abs_b_w_positive": float(np.min(np.abs(coef["b_w"]))) > 0.0,
        "cov_u_x_l2_gt_0_1": metrics["cov_u_x_l2"] > 0.1,
        "cov_u_w_l2_gt_0_1": metrics["cov_u_w_l2"] > 0.1,
        "direct_u_propensity_contribution_gt_0_01": metrics["mean_abs_direct_u_propensity_contribution"] > 0.01,
        "x_propensity_contribution_gt_0_01": metrics["mean_abs_x_propensity_contribution"] > 0.01,
        "residual_variance_u_after_x_gt_0_01": metrics["linear_residual_variance_u_after_x"] > 0.01,
        "x_plus_u_logloss_better_than_x": metrics["treatment_logloss"]["improvement"] > 0.0,
        "empirical_propensity_strictly_inside_0_1_0_9": metrics["propensity_min"] > 0.1 and metrics["propensity_max"] < 0.9,
        "propensity_violation_count_zero": metrics["propensity_violation_count"] == 0,
        "fixed_x_derivative_empirically_positive": metrics["fixed_x_derivative_min"] > 0.0,
        "max_abs_ite_minus_one_lt_1e_12": metrics["max_abs_ite_minus_one"] < 1e-12,
        "structural_ate_error_lt_1e_12": abs(metrics["structural_ate"] - 1.0) < 1e-12,
        "all_arrays_finite": metrics["all_arrays_finite"],
    }
    hashes = {key: array_hash(data[key]) for key in ("U", "X", "W", "A", "Y", "propensity")}
    return {"regime": regime, "metrics": metrics, "acceptance_gates": gates, "hashes": hashes}


def run() -> dict[str, Any]:
    started = time.time()
    coef = frozen_coefficients()
    regime_seeds = {"gaussian": DATA_SEED, "standardized_mixture": DATA_SEED + 1}
    rows = []
    replay_rows = []
    for regime, seed in regime_seeds.items():
        data = generate(regime, seed, coef)
        row = diagnose(regime, data, coef)
        replay = generate(regime, seed, coef)
        replay_hashes = {key: array_hash(replay[key]) for key in ("U", "X", "W", "A", "Y", "propensity")}
        identical = replay_hashes == row["hashes"]
        replay_rows.append({
            "regime": regime,
            "seed": seed,
            "hashes_first": row["hashes"],
            "hashes_replay": replay_hashes,
            "all_hashes_identical": identical,
        })
        row["acceptance_gates"]["deterministic_replay_hashes_identical"] = identical
        rows.append(row)
        del data, replay

    analytic = {
        "graph": "U->X; U->W; (X,U)->A; (X,U,A)->Y",
        "disturbances": {
            "epsilon_x": "N(0,I_20)",
            "epsilon_w": "N(0,I_10)",
            "epsilon_a": "Uniform(0,1)",
            "epsilon_y": "N(0,1)",
            "mutually_independent_rng_streams": True,
        },
        "standardized_mixture": "C~Bernoulli(0.5), Z~N(0,1), U=((2C-1)+0.5Z)/sqrt(1.25)",
        "proxy_channel_injectivity": {
            "projection": "T=b_W^T W/||b_W||^2=U+Gaussian(0,1/||b_W||^2)",
            "b_w_nonzero": bool(np.linalg.norm(coef["b_w"]) > 0.0),
            "b_w_norm_sq": float(coef["b_w"] @ coef["b_w"]),
            "projected_noise_variance": float(1.0 / (coef["b_w"] @ coef["b_w"])),
            "reason": "The Gaussian characteristic function is nonzero at every frequency, so its convolution operator is injective.",
        },
        "residual_confounding_after_x": {
            "fixed_x_derivative": "de/dU=0.64 expit(s){1-expit(s)}>0",
            "u_given_x_nondegenerate": "X=b_X U+epsilon_X with nondegenerate Gaussian epsilon_X, so U|X is not a point mass in either regime.",
            "conclusion": "A is not independent of U given X; numerical log-loss is diagnostic only.",
        },
        "latent_exchangeability_by_scm": True,
        "proxy_pretreatment_and_treatment_invariant_by_scm": True,
        "theoretical_propensity_range": "strictly between 0.1 and 0.9",
        "structural_individual_effect": 1.0,
        "structural_ate": 1.0,
        "representation_learning_call_count": 0,
    }
    global_gates = {
        "all_regime_gates_pass": all(all(row["acceptance_gates"].values()) for row in rows),
        "all_replays_identical": all(row["all_hashes_identical"] for row in replay_rows),
        "regime_u_hashes_distinct": replay_rows[0]["hashes_first"]["U"] != replay_rows[1]["hashes_first"]["U"],
        "representation_learning_call_count_zero": analytic["representation_learning_call_count"] == 0,
    }
    return jsonable({
        "provenance": {
            "script": "code/honest_canonical_scm_sanity.py",
            "coefficient_seed": ROOT_SEED,
            "data_seed_base": DATA_SEED,
            "sample_size_per_regime": N,
            "post_result_tuning": False,
        },
        "dimensions": {"U": 1, "X": D_X, "W": D_W},
        "coefficients": coef,
        "coefficient_audit": {
            "min_abs_b_w": float(np.min(np.abs(coef["b_w"]))),
            "all_b_w_coordinates_nonzero": bool(np.all(coef["b_w"] != 0.0)),
        },
        "analytic_assumption_audit": analytic,
        "regimes": rows,
        "deterministic_replay": replay_rows,
        "global_acceptance_gates": global_gates,
        "sanity_pass": all(global_gates.values()),
        "runtime_seconds": time.time() - started,
    })


def main() -> None:
    payload = run()
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(OUTPUT),
        "runtime_seconds": payload["runtime_seconds"],
        "sanity_pass": payload["sanity_pass"],
        "failed_global_gates": [key for key, value in payload["global_acceptance_gates"].items() if not value],
        "failed_regime_gates": {
            row["regime"]: [key for key, value in row["acceptance_gates"].items() if not value]
            for row in payload["regimes"]
        },
    }, indent=2))


if __name__ == "__main__":
    main()
