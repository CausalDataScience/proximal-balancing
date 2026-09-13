#!/usr/bin/env python3
"""Honest Zika comparison for cross-view proxy balancing.

The representation learner minimizes a finite-random-Fourier-feature (RFF)
conditional-cross-covariance surrogate.  The statistic is not exact KCI and
zero empirical loss is not reported as conditional independence.

The causal target is the effect of treatment on the treated (ETT), matching
Park, Richardson, and Tchetgen Tchetgen's Zika application.  The real dataset
has no known causal ground truth, so the output reports estimates, held-out
audit diagnostics, overlap, and algorithmic split sensitivity.  It does not
rank estimators by causal accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "zika_crossview_comparison.json"

SOURCE_COMMIT = "37270e8ef8782d41d87c60f94ace6384e908615d"
SOURCE_REPOSITORY = "https://github.com/qkrcks0218/SingleProxyControl.git"
SOURCE_URL = f"{SOURCE_REPOSITORY}@{SOURCE_COMMIT}:Zika_Brazil_2013.csv"
SOURCE_SHA256 = "a1a86446a6038ab79f7f29f767f54ae4e113288d9950c5ce75eeea59d7328664"

ROOT_SEED = 190602
N_REPEATS = 20
N_FOLDS = 5
N_EVAL_FOLDS = 2
RFF_V = 24
RFF_R = 12
RIDGE = 2e-2
PRETRAIN_STEPS = 60
CCI_STEPS = 100
LR_PRETRAIN = 2e-2
LR_CCI = 8e-3
R_BANDWIDTH = 0.20
OVERLAP_CENTRAL_LOWER = 0.05
OVERLAP_CENTRAL_UPPER = 0.95
OVERLAP_MIN_CENTRAL_FRACTION = 0.80
OVERLAP_MIN_ESS_FRACTION = 0.10
OVERLAP_MIN_ESS_ABSOLUTE = 20.0
OVERLAP_LOCAL_QUANTILE_CELLS = 5
OVERLAP_MIN_ARM_COUNT_PER_CELL = 5
GBR_SPEC = {
    "n_estimators": 120,
    "max_depth": 2,
    "learning_rate": 0.05,
    "min_samples_leaf": 5,
}

OFFICIAL = {
    "article": "Single proxy control",
    "doi": "10.1093/biomtc/ujae027",
    "article_url": "https://academic.oup.com/biometrics/article/80/2/ujae027/7655733",
    "replication_repository": SOURCE_REPOSITORY,
    "replication_commit": SOURCE_COMMIT,
    "n": 673,
    "n_treated": 185,
    "crude": 3.38429585129043,
    "did_w2013": -1.1561458630441406,
    "did_w2014": -1.0411656487529235,
    "semiparametric_coca": {
        "w2013": {"estimate": -2.410, "se": 0.356},
        "w2014": {"estimate": -2.182, "se": 0.503},
        "both": {"estimate": -2.180, "se": 0.342},
    },
    "doubly_robust_parametric_coca": {
        "w2013": {"estimate": -2.235, "se": 0.502},
        "w2014": {"estimate": -1.833, "se": 0.519},
        "both": {"estimate": -2.182, "se": 0.415},
    },
}


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    return value


def download_source() -> tuple[Path, tempfile.TemporaryDirectory[str], dict[str, Any]]:
    holder: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(prefix="spb-zika-")
    checkout = Path(holder.name) / "SingleProxyControl"
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", SOURCE_REPOSITORY],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "fetch", "--quiet", "--depth", "1", "origin", SOURCE_COMMIT],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        check=True,
    )
    resolved_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved_commit != SOURCE_COMMIT:
        raise RuntimeError(f"official source commit mismatch: {resolved_commit}")
    path = checkout / "Zika_Brazil_2013.csv"
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"official source hash mismatch: {digest}")
    return path, holder, {
        "url": SOURCE_URL,
        "repository": SOURCE_REPOSITORY,
        "git_commit": SOURCE_COMMIT,
        "sha256": digest,
        "bytes": len(payload),
        "temporary_checkout_parent": "/tmp",
    }


def load_wide(path: Path) -> dict[str, np.ndarray]:
    by_id: dict[str, dict[int, float]] = {}
    treatment: dict[str, int] = {}
    covariates: dict[str, tuple[float, float, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            unit = row["ID"]
            time_point = int(row["Time"])
            by_id.setdefault(unit, {})[time_point] = float(row["Y"])
            treatment[unit] = int(row["Trt"])
            if time_point == 0:
                covariates[unit] = (float(row["X_1"]), float(row["X_2"]), float(row["X_3"]))
    units = sorted(by_id, key=int)
    if any(set(by_id[unit]) != {-1, 0, 1} for unit in units):
        raise RuntimeError("each municipality must have 2013, 2014, and 2016 rows")
    return {
        "unit_id": np.asarray(units),
        "A": np.asarray([treatment[unit] for unit in units], dtype=float),
        "X": np.asarray([covariates[unit] for unit in units], dtype=float),
        "W2013": np.asarray([by_id[unit][-1] for unit in units], dtype=float),
        "W2014": np.asarray([by_id[unit][0] for unit in units], dtype=float),
        "Y2016": np.asarray([by_id[unit][1] for unit in units], dtype=float),
    }


def crude_effect(data: dict[str, np.ndarray]) -> float:
    a, y = data["A"], data["Y2016"]
    return float(y[a == 1].mean() - y[a == 0].mean())


def regression_did(data: dict[str, np.ndarray], baseline_key: str) -> float:
    """Two-period outcome-regression DiD used by did::att_gt(est_method='reg')."""
    a, x, y = data["A"], data["X"], data["Y2016"]
    delta = y - data[baseline_key]
    controls = a == 0
    treated = a == 1
    z0 = np.column_stack([np.ones(controls.sum()), x[controls]])
    beta = np.linalg.lstsq(z0, delta[controls], rcond=None)[0]
    z1 = np.column_stack([np.ones(treated.sum()), x[treated]])
    return float(np.mean(delta[treated] - z1 @ beta))


def stratified_folds(a: np.ndarray, seed: int, n_folds: int = N_FOLDS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold = np.empty(len(a), dtype=int)
    for arm in (0.0, 1.0):
        index = np.flatnonzero(a == arm)
        index = index[rng.permutation(len(index))]
        fold[index] = np.arange(len(index)) % n_folds
    return fold


def median_bandwidth(z: np.ndarray, seed: int, max_points: int = 500) -> float:
    if len(z) > max_points:
        rng = np.random.default_rng(seed)
        z = z[rng.choice(len(z), max_points, replace=False)]
    sq = ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
    values = sq[np.triu_indices(len(z), 1)]
    return math.sqrt(max(float(np.median(values)), 1e-8) / 2.0)


@dataclass
class RFFSpec:
    mean: np.ndarray
    scale: np.ndarray
    omega: np.ndarray
    phase: np.ndarray
    bandwidth: float


@dataclass
class EvaluationSpec:
    """Fixed held-out feature draw and internal evaluation-fold assignment."""

    v_spec: RFFSpec
    r_omega: np.ndarray
    r_phase: np.ndarray
    inner_folds: np.ndarray
    signature: str


def make_v_rff(v_train: np.ndarray, seed: int) -> RFFSpec:
    mean = v_train.mean(axis=0)
    scale = v_train.std(axis=0) + 1e-8
    z = (v_train - mean) / scale
    bandwidth = median_bandwidth(z, seed)
    rng = np.random.default_rng(seed + 101)
    omega = rng.standard_normal((z.shape[1], RFF_V)) / bandwidth
    phase = rng.uniform(0.0, 2.0 * math.pi, RFF_V)
    return RFFSpec(mean, scale, omega, phase, bandwidth)


def v_rff_numpy(v: np.ndarray, spec: RFFSpec) -> np.ndarray:
    z = (v - spec.mean) / spec.scale
    return math.sqrt(2.0 / RFF_V) * np.cos(z @ spec.omega + spec.phase)


def r_features_torch(r: torch.Tensor, omega: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    features = math.sqrt(2.0 / RFF_R) * torch.cos(r[:, None] * omega[None, :] + phase[None, :])
    return torch.cat([torch.ones((len(r), 1), dtype=r.dtype), r[:, None], features], dim=1)


def ridge_coefficients(z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dimension = z.shape[1]
    penalty = torch.eye(dimension, dtype=z.dtype) * RIDGE
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
) -> torch.Tensor:
    z = r_features_torch(r, omega, phase)
    residual_a = a[:, None] - z @ ridge_coefficients(z, a[:, None])
    residual_v = phi_v - z @ ridge_coefficients(z, phi_v)
    crosscov = residual_a.T @ residual_v / len(a)
    return crosscov.square().sum()


class LinearScore(torch.nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.linear = torch.nn.Linear(dimension, 1)

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
    pretrain_curve: list[float]
    cci_curve: list[float]


def fit_representation(inp: np.ndarray, audit: np.ndarray, a: np.ndarray, seed: int) -> RepresentationFit:
    input_mean = inp.mean(axis=0)
    input_scale = inp.std(axis=0) + 1e-8
    standardized = (inp - input_mean) / input_scale
    v_spec = make_v_rff(audit, seed + 13)
    phi_v = v_rff_numpy(audit, v_spec)
    rng = np.random.default_rng(seed + 17)
    r_omega = rng.standard_normal(RFF_R) / R_BANDWIDTH
    r_phase = rng.uniform(0.0, 2.0 * math.pi, RFF_R)

    torch.manual_seed(seed)
    model = LinearScore(standardized.shape[1]).double()
    xt = torch.tensor(standardized, dtype=torch.float64)
    at = torch.tensor(a, dtype=torch.float64)
    vt = torch.tensor(phi_v, dtype=torch.float64)
    omega = torch.tensor(r_omega, dtype=torch.float64)
    phase = torch.tensor(r_phase, dtype=torch.float64)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_PRETRAIN, weight_decay=1e-4)
    pretrain_curve: list[float] = []
    for _ in range(PRETRAIN_STEPS):
        r = model(xt)
        loss = torch.nn.functional.binary_cross_entropy(r.clamp(1e-5, 1.0 - 1e-5), at)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pretrain_curve.append(float(loss.detach()))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_CCI, weight_decay=1e-4)
    cci_curve: list[float] = []
    for _ in range(CCI_STEPS):
        r = model(xt)
        loss = conditional_crosscov_loss(r, at, vt, omega, phase)
        variance_floor = torch.relu(torch.tensor(0.015, dtype=r.dtype) - r.var(unbiased=False))
        objective = loss + 0.5 * variance_floor
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        cci_curve.append(float(loss.detach()))
    return RepresentationFit(model, input_mean, input_scale, v_spec, r_omega, r_phase, pretrain_curve, cci_curve)


def predict_representation(fit: RepresentationFit, inp: np.ndarray) -> np.ndarray:
    standardized = (inp - fit.input_mean) / fit.input_scale
    with torch.no_grad():
        return fit.model(torch.tensor(standardized, dtype=torch.float64)).numpy()


def r_features_numpy(r: np.ndarray, omega: np.ndarray, phase: np.ndarray) -> np.ndarray:
    features = math.sqrt(2.0 / RFF_R) * np.cos(r[:, None] * omega[None, :] + phase[None, :])
    return np.column_stack([np.ones(len(r)), r, features])


def make_evaluation_spec(audit_train: np.ndarray, a_test: np.ndarray, seed: int) -> EvaluationSpec:
    """Create one method-independent held-out evaluation contract."""
    v_spec = make_v_rff(audit_train, seed + 11)
    rng = np.random.default_rng(seed + 13)
    r_omega = rng.standard_normal(RFF_R) / R_BANDWIDTH
    r_phase = rng.uniform(0.0, 2.0 * math.pi, RFF_R)
    inner_folds = stratified_folds(a_test, seed + 17, N_EVAL_FOLDS)
    digest = hashlib.sha256()
    for array in (v_spec.mean, v_spec.scale, v_spec.omega, v_spec.phase, r_omega, r_phase, inner_folds):
        digest.update(np.ascontiguousarray(array).tobytes())
    return EvaluationSpec(v_spec, r_omega, r_phase, inner_folds, digest.hexdigest())


def heldout_crossfit_surrogate(
    spec: EvaluationSpec,
    r: np.ndarray,
    a: np.ndarray,
    audit: np.ndarray,
) -> float:
    """Evaluate residual cross-covariance with internal sample splitting.

    The RFF draw is fixed before method comparison.  Within each internal
    split, residual regressions are fit on the other split and evaluated only
    on observations not used to fit those regressions.
    """
    z = torch.tensor(r_features_numpy(r, spec.r_omega, spec.r_phase), dtype=torch.float64)
    at = torch.tensor(a, dtype=torch.float64)
    vt = torch.tensor(v_rff_numpy(audit, spec.v_spec), dtype=torch.float64)
    crosscov_sum = torch.zeros((1, vt.shape[1]), dtype=torch.float64)
    evaluated = 0
    for inner_fold in range(N_EVAL_FOLDS):
        fit_index = torch.tensor(spec.inner_folds != inner_fold)
        eval_index = torch.tensor(spec.inner_folds == inner_fold)
        z_fit = z[fit_index]
        beta_a = ridge_coefficients(z_fit, at[fit_index, None])
        beta_v = ridge_coefficients(z_fit, vt[fit_index])
        residual_a = at[eval_index, None] - z[eval_index] @ beta_a
        residual_v = vt[eval_index] - z[eval_index] @ beta_v
        crosscov_sum += residual_a.T @ residual_v
        evaluated += int(eval_index.sum())
    return float(torch.linalg.norm(crosscov_sum / evaluated))


def standardize_train(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0) + 1e-8
    return (train - mean) / scale, (test - mean) / scale


def fit_propensity(train_feature: np.ndarray, a_train: np.ndarray, test_feature: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_z, test_z = standardize_train(train_feature, test_feature)
    model = LogisticRegression(C=10.0, max_iter=2000, random_state=seed, solver="lbfgs")
    model.fit(train_z, a_train.astype(int))
    return model.predict_proba(train_z)[:, 1], model.predict_proba(test_z)[:, 1]


def outcome_predictions(
    feature_train: np.ndarray,
    a_train: np.ndarray,
    y_train: np.ndarray,
    feature_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    controls = a_train == 0
    model = GradientBoostingRegressor(random_state=seed, **GBR_SPEC)
    model.fit(feature_train[controls], y_train[controls])
    return model.predict(feature_test)


def weighted_audit_smd(a: np.ndarray, propensity: np.ndarray, audit: np.ndarray) -> float:
    propensity = np.clip(propensity, 0.01, 0.99)
    treated = a == 1
    control = ~treated
    weights = propensity[control] / (1.0 - propensity[control])
    weights = weights / weights.sum()
    mean_t = audit[treated].mean(axis=0)
    mean_c = np.sum(audit[control] * weights[:, None], axis=0)
    scale = audit.std(axis=0) + 1e-8
    return float(np.max(np.abs((mean_t - mean_c) / scale)))


def effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.sum(weights**2))


def overlap_summary(a: np.ndarray, propensity: np.ndarray) -> dict[str, Any]:
    raw_propensity = propensity.copy()
    propensity = np.clip(propensity, 0.01, 0.99)
    treated = a == 1
    control = ~treated
    treated_weights = np.ones(int(treated.sum()))
    control_weights = propensity[control] / (1.0 - propensity[control])
    treated_ess = effective_sample_size(treated_weights)
    control_ess = effective_sample_size(control_weights)
    treated_ess_minimum = max(OVERLAP_MIN_ESS_ABSOLUTE, OVERLAP_MIN_ESS_FRACTION * treated.sum())
    control_ess_minimum = max(OVERLAP_MIN_ESS_ABSOLUTE, OVERLAP_MIN_ESS_FRACTION * control.sum())
    central_fraction = float(np.mean(
        (raw_propensity >= OVERLAP_CENTRAL_LOWER)
        & (raw_propensity <= OVERLAP_CENTRAL_UPPER)
    ))

    quantiles = np.quantile(raw_propensity, np.linspace(0.0, 1.0, OVERLAP_LOCAL_QUANTILE_CELLS + 1))
    cell_id = np.searchsorted(quantiles[1:-1], raw_propensity, side="right")
    cells: list[dict[str, Any]] = []
    supported = np.zeros(len(a), dtype=bool)
    for cell in range(OVERLAP_LOCAL_QUANTILE_CELLS):
        in_cell = cell_id == cell
        n_treated = int(np.sum(in_cell & treated))
        n_control = int(np.sum(in_cell & control))
        passes = (
            n_treated >= OVERLAP_MIN_ARM_COUNT_PER_CELL
            and n_control >= OVERLAP_MIN_ARM_COUNT_PER_CELL
        )
        if passes:
            supported[in_cell] = True
        cells.append({
            "cell": cell,
            "n": int(in_cell.sum()),
            "n_treated": n_treated,
            "n_control": n_control,
            "passes_minimum_arm_count": passes,
        })

    components = {
        "central_propensity_fraction": central_fraction >= OVERLAP_MIN_CENTRAL_FRACTION,
        "treated_ess": treated_ess >= treated_ess_minimum,
        "control_ess": control_ess >= control_ess_minimum,
        "all_quantile_cells_have_two_arm_support": all(
            cell["passes_minimum_arm_count"] for cell in cells
        ),
    }
    return {
        "propensity_min": float(raw_propensity.min()),
        "propensity_max": float(raw_propensity.max()),
        "propensity_fraction_within_0p05_0p95": central_fraction,
        "treated_ess": treated_ess,
        "control_ess": control_ess,
        "treated_ess_fraction": treated_ess / float(treated.sum()),
        "control_ess_fraction": control_ess / float(control.sum()),
        "local_quantile_cells": cells,
        "fraction_in_two_arm_supported_cells": float(supported.mean()),
        "gate_components": components,
        "gate_pass": all(components.values()),
    }


def direct_adjustment(
    data: dict[str, np.ndarray],
    features: np.ndarray,
    folds: np.ndarray,
    seed: int,
    evaluations: dict[str, tuple[np.ndarray, list[EvaluationSpec]]] | None = None,
) -> dict[str, Any]:
    a, y = data["A"], data["Y2016"]
    counterfactual = np.full(len(a), np.nan)
    propensity = np.full(len(a), np.nan)
    heldout = {name: [] for name in (evaluations or {})}
    for fold_index in range(N_FOLDS):
        train = folds != fold_index
        test = ~train
        counterfactual[test] = outcome_predictions(features[train], a[train], y[train], features[test], seed + 31 * fold_index)
        _, r_test = fit_propensity(features[train], a[train], features[test], seed + 37 * fold_index)
        propensity[test] = r_test
        for name, (audit, specs) in (evaluations or {}).items():
            heldout[name].append(
                heldout_crossfit_surrogate(specs[fold_index], r_test, a[test], audit[test])
            )
    treated = a == 1
    estimate = float(np.mean(y[treated] - counterfactual[treated]))
    result: dict[str, Any] = {
        "estimate": estimate,
        "overlap": overlap_summary(a, propensity),
        "eligible_for_certificate_ranking": bool(evaluations),
    }
    if evaluations:
        result["heldout_evaluations"] = {
            name: {
                "finite_rff_crossfit_surrogate_mean": float(np.mean(values)),
                "finite_rff_crossfit_surrogate_by_fold": values,
                "audit_max_abs_att_weighted_smd": weighted_audit_smd(a, propensity, evaluations[name][0]),
                "evaluation_signatures": [spec.signature for spec in evaluations[name][1]],
            }
            for name, values in heldout.items()
        }
    return result


def build_evaluation_specs(
    audit: np.ndarray,
    a: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> list[EvaluationSpec]:
    specs: list[EvaluationSpec] = []
    for fold_index in range(N_FOLDS):
        train = folds != fold_index
        test = ~train
        specs.append(make_evaluation_spec(audit[train], a[test], seed + 59 * fold_index))
    return specs


def representation_adjustment(
    data: dict[str, np.ndarray],
    inp: np.ndarray,
    audit: np.ndarray,
    folds: np.ndarray,
    seed: int,
    evaluation_specs: list[EvaluationSpec],
    replay: bool = False,
    certificate_eligible: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    a, y = data["A"], data["Y2016"]
    counterfactual = np.full(len(a), np.nan)
    propensity = np.full(len(a), np.nan)
    heldout: list[float] = []
    terminal_losses: list[float] = []
    replay_result: dict[str, Any] | None = None
    for fold_index in range(N_FOLDS):
        train = folds != fold_index
        test = ~train
        fit_seed = seed + 101 * fold_index
        fit = fit_representation(inp[train], audit[train], a[train], fit_seed)
        r_train = predict_representation(fit, inp[train])
        r_test = predict_representation(fit, inp[test])
        counterfactual[test] = outcome_predictions(r_train[:, None], a[train], y[train], r_test[:, None], seed + 103 * fold_index)
        _, propensity[test] = fit_propensity(r_train[:, None], a[train], r_test[:, None], seed + 107 * fold_index)
        heldout.append(
            heldout_crossfit_surrogate(evaluation_specs[fold_index], r_test, a[test], audit[test])
        )
        terminal_losses.append(fit.cci_curve[-1])
        if replay and fold_index == 0:
            fit_again = fit_representation(inp[train], audit[train], a[train], fit_seed)
            prediction_again = predict_representation(fit_again, inp[test])
            replay_result = {
                "max_abs_prediction_difference": float(np.max(np.abs(r_test - prediction_again))),
                "training_curve_exact_match": fit.cci_curve == fit_again.cci_curve,
                "pass": bool(np.array_equal(r_test, prediction_again) and fit.cci_curve == fit_again.cci_curve),
            }
    treated = a == 1
    estimate = float(np.mean(y[treated] - counterfactual[treated]))
    return {
        "estimate": estimate,
        "heldout_finite_rff_crossfit_surrogate_mean": float(np.mean(heldout)),
        "heldout_finite_rff_crossfit_surrogate_by_fold": heldout,
        "evaluation_signatures": [spec.signature for spec in evaluation_specs],
        "training_surrogate_terminal_mean": float(np.mean(terminal_losses)),
        "audit_max_abs_att_weighted_smd": weighted_audit_smd(a, propensity, audit),
        "overlap": overlap_summary(a, propensity),
        "eligible_for_certificate_ranking": certificate_eligible,
    }, replay_result


def summarize_repeats(records: list[dict[str, Any]]) -> dict[str, Any]:
    methods = ["x_only", "full_observed", "cross_2013_rep_2014_audit", "cross_2014_rep_2013_audit", "cross_view_average", "self_conditioned_negative_control"]
    output: dict[str, Any] = {}
    for method in methods:
        estimates = np.asarray([row[method]["estimate"] for row in records])
        block: dict[str, Any] = {
            "estimate_mean": float(estimates.mean()),
            "estimate_median": float(np.median(estimates)),
            "estimate_sd_across_algorithmic_splits": float(
                estimates.std(ddof=1 if len(estimates) > 1 else 0)
            ),
            "estimate_min": float(estimates.min()),
            "estimate_max": float(estimates.max()),
            "estimates": estimates,
        }
        if method != "cross_view_average":
            overlaps = [row[method]["overlap"] for row in records]
            block["eligible_for_certificate_ranking"] = bool(
                records[0][method]["eligible_for_certificate_ranking"]
            )
            block["overlap"] = {
                "propensity_fraction_within_0p05_0p95_mean": float(np.mean([
                    item["propensity_fraction_within_0p05_0p95"] for item in overlaps
                ])),
                "treated_ess_mean": float(np.mean([item["treated_ess"] for item in overlaps])),
                "control_ess_mean": float(np.mean([item["control_ess"] for item in overlaps])),
                "treated_ess_fraction_mean": float(np.mean([
                    item["treated_ess_fraction"] for item in overlaps
                ])),
                "control_ess_fraction_mean": float(np.mean([
                    item["control_ess_fraction"] for item in overlaps
                ])),
                "fraction_in_two_arm_supported_cells_mean": float(np.mean([
                    item["fraction_in_two_arm_supported_cells"] for item in overlaps
                ])),
                "gate_pass_rate": float(np.mean([item["gate_pass"] for item in overlaps])),
                "component_pass_rates": {
                    key: float(np.mean([item["gate_components"][key] for item in overlaps]))
                    for key in overlaps[0]["gate_components"]
                },
            }
        if method == "x_only":
            block["heldout_evaluations"] = {}
            for audit_name in records[0][method]["heldout_evaluations"]:
                evaluations = [row[method]["heldout_evaluations"][audit_name] for row in records]
                block["heldout_evaluations"][audit_name] = {
                    "finite_rff_crossfit_surrogate_mean": float(np.mean([
                        item["finite_rff_crossfit_surrogate_mean"] for item in evaluations
                    ])),
                    "audit_max_abs_att_weighted_smd_mean": float(np.mean([
                        item["audit_max_abs_att_weighted_smd"] for item in evaluations
                    ])),
                }
        elif method in {
            "cross_2013_rep_2014_audit",
            "cross_2014_rep_2013_audit",
            "self_conditioned_negative_control",
        }:
            block["heldout_finite_rff_crossfit_surrogate_mean"] = float(np.mean([
                row[method]["heldout_finite_rff_crossfit_surrogate_mean"] for row in records
            ]))
            block["audit_max_abs_att_weighted_smd_mean"] = float(np.mean([
                row[method]["audit_max_abs_att_weighted_smd"] for row in records
            ]))
        output[method] = block
    disagreements = np.asarray([
        abs(row["cross_2013_rep_2014_audit"]["estimate"] - row["cross_2014_rep_2013_audit"]["estimate"])
        for row in records
    ])
    output["cross_view_orientation_disagreement"] = {
        "mean_absolute_difference": float(disagreements.mean()),
        "max_absolute_difference": float(disagreements.max()),
        "values": disagreements,
    }
    return output


def run(n_repeats: int) -> dict[str, Any]:
    started = time.time()
    source_path, holder, source = download_source()
    try:
        data = load_wide(source_path)
    finally:
        holder.cleanup()

    smoke_values = {
        "n": len(data["A"]),
        "n_treated": int(data["A"].sum()),
        "crude": crude_effect(data),
        "did_w2013": regression_did(data, "W2013"),
        "did_w2014": regression_did(data, "W2014"),
    }
    smoke_gates = {
        "n_matches_673": smoke_values["n"] == OFFICIAL["n"],
        "treated_matches_185": smoke_values["n_treated"] == OFFICIAL["n_treated"],
        "crude_matches_official": abs(smoke_values["crude"] - OFFICIAL["crude"]) < 1e-10,
        "did_w2013_matches_official": abs(smoke_values["did_w2013"] - OFFICIAL["did_w2013"]) < 1e-10,
        "did_w2014_matches_official": abs(smoke_values["did_w2014"] - OFFICIAL["did_w2014"]) < 1e-10,
    }
    if not all(smoke_gates.values()):
        raise RuntimeError(f"official preprocessing gates failed: {smoke_gates}")

    x = data["X"]
    w2013 = data["W2013"][:, None]
    w2014 = data["W2014"][:, None]
    full = np.column_stack([x, w2013, w2014])
    records: list[dict[str, Any]] = []
    replay_result: dict[str, Any] | None = None
    for repeat in range(n_repeats):
        seed = ROOT_SEED + 1009 * repeat
        folds = stratified_folds(data["A"], seed)
        audit_2014 = np.column_stack([x, w2014])
        audit_2013 = np.column_stack([x, w2013])
        eval_2014 = build_evaluation_specs(audit_2014, data["A"], folds, seed + 701)
        eval_2013 = build_evaluation_specs(audit_2013, data["A"], folds, seed + 709)
        eval_self = build_evaluation_specs(full, data["A"], folds, seed + 719)
        x_only = direct_adjustment(
            data,
            x,
            folds,
            seed + 11,
            evaluations={
                "audit_x_w2014": (audit_2014, eval_2014),
                "audit_x_w2013": (audit_2013, eval_2013),
            },
        )
        full_observed = direct_adjustment(data, full, folds, seed + 13)
        cross_13, replay_candidate = representation_adjustment(
            data,
            np.column_stack([x, w2013]),
            audit_2014,
            folds,
            seed + 17,
            eval_2014,
            replay=repeat == 0,
        )
        cross_14, _ = representation_adjustment(
            data,
            np.column_stack([x, w2014]),
            audit_2013,
            folds,
            seed + 19,
            eval_2013,
        )
        self_conditioned, _ = representation_adjustment(
            data,
            full,
            full,
            folds,
            seed + 23,
            eval_self,
            certificate_eligible=False,
        )
        paired_feature_gates = {
            "2013_rep_2014_audit": (
                x_only["heldout_evaluations"]["audit_x_w2014"]["evaluation_signatures"]
                == cross_13["evaluation_signatures"]
            ),
            "2014_rep_2013_audit": (
                x_only["heldout_evaluations"]["audit_x_w2013"]["evaluation_signatures"]
                == cross_14["evaluation_signatures"]
            ),
        }
        if not all(paired_feature_gates.values()):
            raise RuntimeError(f"paired evaluation feature contract failed: {paired_feature_gates}")
        if replay_candidate is not None:
            replay_result = replay_candidate
        records.append({
            "repeat": repeat,
            "seed": seed,
            "x_only": x_only,
            "full_observed": full_observed,
            "cross_2013_rep_2014_audit": cross_13,
            "cross_2014_rep_2013_audit": cross_14,
            "cross_view_average": {"estimate": 0.5 * (cross_13["estimate"] + cross_14["estimate"])},
            "self_conditioned_negative_control": self_conditioned,
            "paired_evaluation_feature_contract": paired_feature_gates,
        })
    smoke_gates["deterministic_replay"] = bool(replay_result and replay_result["pass"])
    if not smoke_gates["deterministic_replay"]:
        raise RuntimeError(f"deterministic replay failed: {replay_result}")

    summary = summarize_repeats(records)
    comparison_pairs = {
        "2013_rep_2014_audit": ("cross_2013_rep_2014_audit", "audit_x_w2014"),
        "2014_rep_2013_audit": ("cross_2014_rep_2013_audit", "audit_x_w2013"),
    }
    paired_comparison: dict[str, Any] = {}
    for comparison_name, (cross_method, x_audit_name) in comparison_pairs.items():
        cross_values = np.asarray([
            row[cross_method]["heldout_finite_rff_crossfit_surrogate_mean"] for row in records
        ])
        x_values = np.asarray([
            row["x_only"]["heldout_evaluations"][x_audit_name]["finite_rff_crossfit_surrogate_mean"]
            for row in records
        ])
        paired_comparison[comparison_name] = {
            "audit_set": "(X,W2014)" if "2013_rep" in comparison_name else "(X,W2013)",
            "fixed_evaluation_features_exact_match_all_repeats": all(
                row["paired_evaluation_feature_contract"][comparison_name] for row in records
            ),
            "cross_mean": float(cross_values.mean()),
            "x_only_mean": float(x_values.mean()),
            "cross_minus_x_only_mean": float(np.mean(cross_values - x_values)),
            "cross_below_x_only_win_rate": float(np.mean(cross_values < x_values)),
            "cross_overlap_gate_pass_rate": summary[cross_method]["overlap"]["gate_pass_rate"],
            "x_only_overlap_gate_pass_rate": summary["x_only"]["overlap"]["gate_pass_rate"],
        }
    observed_gate = {
        "paired_surrogate_lower_in_both_orientations_all_repeats": all(
            item["cross_below_x_only_win_rate"] == 1.0 for item in paired_comparison.values()
        ),
        "both_compared_methods_pass_overlap_gate_all_repeats": all(
            item["cross_overlap_gate_pass_rate"] == 1.0
            and item["x_only_overlap_gate_pass_rate"] == 1.0
            for item in paired_comparison.values()
        ),
    }
    observed_gate["certificate_diagnostic_pass"] = all(observed_gate.values())
    return jsonable({
        "status": "experiment_only_real_data_no_causal_ground_truth",
        "source": source,
        "target": {
            "name": "effect_on_the_treated",
            "formula": "E[Y(1)-Y(0)|A=1]",
            "units": "births per 1,000 persons",
            "unit_of_observation": "municipality",
            "treatment_interpretation": "Pernambuco municipality exposed to the severe 2015 Zika outbreak versus Rio Grande do Sul municipality minimally affected",
        },
        "variables": {
            "A": "Pernambuco indicator; Rio Grande do Sul is control",
            "X": ["2014 log population", "2014 log population density", "2014 female proportion"],
            "W2013": "2013 pre-outbreak birth rate",
            "W2014": "2014 pre-outbreak birth rate",
            "Y2016": "2016 post-outbreak birth rate",
        },
        "method_contract": {
            "representation_objective": "finite-RFF conditional-cross-covariance surrogate",
            "not_exact_kci": True,
            "not_population_conditional_independence_test": True,
            "representation_uses_outcome": False,
            "cross_2013_rep_2014_audit_input": ["X", "W2013"],
            "cross_2013_rep_2014_audit_audit": ["X", "W2014"],
            "cross_2014_rep_2013_audit_input": ["X", "W2014"],
            "cross_2014_rep_2013_audit_audit": ["X", "W2013"],
            "paired_audit_contract": {
                "2013_rep_orientation": "cross-view and X-only are both evaluated on (X,W2014)",
                "2014_rep_orientation": "cross-view and X-only are both evaluated on (X,W2013)",
                "same_outer_fold_fixed_rff_draw_across_compared_methods": True,
            },
            "outer_cross_folds": N_FOLDS,
            "evaluation_internal_cross_folds": N_EVAL_FOLDS,
            "evaluation_residual_regressions_fit_outside_scored_internal_fold": True,
            "outcome_estimator": "cross-fitted GradientBoosting control-outcome regression",
            "same_outcome_model_specification_across_adjustment_methods": True,
            "full_observed_role": "ETT comparator only; excluded from certificate ranking because its input contains the audit variables",
            "self_conditioned_role": "negative-control diagnostic only; not identification evidence",
            "overlap_gate": {
                "central_propensity_interval": [OVERLAP_CENTRAL_LOWER, OVERLAP_CENTRAL_UPPER],
                "minimum_fraction_in_central_interval": OVERLAP_MIN_CENTRAL_FRACTION,
                "minimum_ess_per_arm": f"max({OVERLAP_MIN_ESS_ABSOLUTE}, {OVERLAP_MIN_ESS_FRACTION} * observed arm size)",
                "local_cells": f"{OVERLAP_LOCAL_QUANTILE_CELLS} pooled propensity quantile cells",
                "minimum_observations_per_arm_per_cell": OVERLAP_MIN_ARM_COUNT_PER_CELL,
                "rationale": "A descriptive fail-closed gate requiring most propensities away from deterministic assignment, at least a minimally usable effective sample in each ATT arm, and observable two-arm comparisons throughout the score range. These are declared diagnostics, not universal identification thresholds.",
            },
        },
        "smoke": {
            "values": smoke_values,
            "gates": smoke_gates,
            "deterministic_replay": replay_result,
            "pass": all(smoke_gates.values()),
        },
        "official_reference": OFFICIAL,
        "execution": {
            "root_seed": ROOT_SEED,
            "algorithmic_split_repeats": n_repeats,
            "outer_folds": N_FOLDS,
            "runtime_seconds": time.time() - started,
            "repeats_are_not_independent_datasets": True,
            "repeat_dispersion_is_not_a_sampling_standard_error": True,
        },
        "summary": summary,
        "paired_diagnostic_comparison": paired_comparison,
        "observed_certificate_diagnostic_gate": observed_gate,
        "records": records,
        "interpretation": {
            "accuracy_ranking_allowed": False,
            "reason": "The Zika dataset has no known causal ground truth.",
            "positive_diagnostic": "Cross-view learning is encouraging only if held-out audit dependence decreases without overlap or ESS collapse and estimates are stable across proxy orientations and splits.",
            "negative_diagnostic": "A lower paired surrogate is not accepted as certificate evidence when the declared overlap gate fails; failure also includes large orientation instability.",
            "observed_result": "The paired surrogate is lower for cross-view in both orientations, but the declared overlap gate fails; therefore this experiment does not pass the certificate diagnostic.",
            "full_observed_excluded_from_certificate_ranking": True,
            "coca_agreement_is_validation": False,
            "past_plus_0p03_reused": False,
        },
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run one deterministic split repetition.")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = run(1 if args.smoke else N_REPEATS)
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "smoke_mode": args.smoke,
        "smoke_pass": result["smoke"]["pass"],
        "repeats": result["execution"]["algorithmic_split_repeats"],
        "runtime_seconds": result["execution"]["runtime_seconds"],
        "summary": {
            key: value["estimate_median"]
            for key, value in result["summary"].items()
            if isinstance(value, dict) and "estimate_median" in value
        },
        "output_written": not args.no_write,
        "output": str(args.output) if not args.no_write else None,
    }, indent=2))


if __name__ == "__main__":
    main()
