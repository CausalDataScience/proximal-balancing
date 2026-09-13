#!/usr/bin/env python3
"""Two-view proxy-balancing experiment on the KMU 2022 simulation.

This script preserves the published two-proxy structure rather than hiding Z.
It maps the treatment-inducing proxy Z to the representation view and the
outcome-inducing proxy W to an independent audit view:

    R = phi(X, Z),    audit V = (X, W).

The learned scalar R minimizes a finite random-Fourier-feature conditional
cross-covariance surrogate between A and V given R.  This is not exact KCI or
conditional HSIC.  W, Y, U, and the evaluation fold never enter the proposed
representation.  A same-view representation phi(X,Z,W) is retained only as an
uncertified negative control.

The original ablation and its outputs are deliberately left untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

import kmu_2022_single_proxy_ablation_experiment as kmu


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results" / "kmu_2022_two_view_representation_summary.json"
RUNS_PATH = ROOT / "results" / "kmu_2022_two_view_representation_runs.csv"

TRUE_ATE = 1.0
DIMENSION = kmu.DIMENSION
TRUE_J1 = 1.0 + 0.7 * DIMENSION + DIMENSION * (
    0.2 + 0.5 * DIMENSION / (DIMENSION + 1.0)
)
BASE_SEED = 190602
N_FOLDS = 2
RFF_AUDIT = 32
RFF_SCORE = 12
RIDGE_NUISANCE = 2e-2
PROPENSITY_LOWER = 0.05
PROPENSITY_UPPER = 0.95
MIN_PROPENSITY_INTERIOR_FRACTION = 0.90
MIN_ARM_ESS_FRACTION = 0.25
N_OVERLAP_CELLS = 5
MIN_CELL_ARM_COUNT = 2


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    return value


def array_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def stable_seed(transform: str, n: int, rep: int, fold: int, method: str) -> int:
    transform_id = {"sin": 1, "cube": 2}[transform]
    method_id = {"x_only": 1, "cross_view": 2, "self_conditioned": 3}[method]
    return BASE_SEED + 1_000_003 * rep + 1009 * n + 101 * transform_id + 17 * fold + method_id


def sample_seed(n: int, rep: int) -> int:
    # The raw Gaussian draw is identical for sin and cube at fixed (n, rep).
    return BASE_SEED + 1_000_003 * rep + 1009 * n


def diagnostic_seed(transform: str, n: int, rep: int, fold: int) -> int:
    """A method-invariant seed for paired held-out discrepancy evaluation."""
    transform_id = {"sin": 1, "cube": 2}[transform]
    return BASE_SEED + 2_000_003 * rep + 1009 * n + 101 * transform_id + 17 * fold


def stratified_folds(a: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold = np.empty(len(a), dtype=int)
    for arm in (0.0, 1.0):
        idx = np.flatnonzero(a == arm)
        idx = idx[rng.permutation(len(idx))]
        fold[idx] = np.arange(len(idx)) % N_FOLDS
    return fold


def standardize(train: np.ndarray, evaluate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0) + 1e-8
    return (train - mean) / scale, (evaluate - mean) / scale


def median_bandwidth(z: np.ndarray, seed: int, max_points: int = 400) -> float:
    if len(z) > max_points:
        rng = np.random.default_rng(seed)
        z = z[rng.choice(len(z), max_points, replace=False)]
    squared = ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
    upper = squared[np.triu_indices(len(z), 1)]
    return math.sqrt(max(float(np.median(upper)), 1e-8) / 2.0)


@dataclass
class RFFSpec:
    mean: np.ndarray
    scale: np.ndarray
    omega: np.ndarray
    phase: np.ndarray
    bandwidth: float


def fit_rff(v: np.ndarray, seed: int) -> RFFSpec:
    mean = v.mean(axis=0)
    scale = v.std(axis=0) + 1e-8
    standardized = (v - mean) / scale
    bandwidth = median_bandwidth(standardized, seed)
    rng = np.random.default_rng(seed + 13)
    omega = rng.standard_normal((v.shape[1], RFF_AUDIT)) / bandwidth
    phase = rng.uniform(0.0, 2.0 * math.pi, RFF_AUDIT)
    return RFFSpec(mean, scale, omega, phase, bandwidth)


def transform_rff(v: np.ndarray, spec: RFFSpec) -> np.ndarray:
    standardized = (v - spec.mean) / spec.scale
    return math.sqrt(2.0 / RFF_AUDIT) * np.cos(standardized @ spec.omega + spec.phase)


class ScoreNet(torch.nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        hidden = min(64, max(24, dimension // 2))
        self.network = torch.nn.Sequential(
            torch.nn.Linear(dimension, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 24),
            torch.nn.Tanh(),
            torch.nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x)).squeeze(1)


@dataclass
class RepresentationFit:
    model: ScoreNet
    input_mean: np.ndarray
    input_scale: np.ndarray
    audit_spec: RFFSpec
    score_omega: np.ndarray
    score_phase: np.ndarray
    initial_objective: float
    final_objective: float


def score_features(score: torch.Tensor, omega: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    rff = math.sqrt(2.0 / RFF_SCORE) * torch.cos(
        score[:, None] * omega[None, :] + phase[None, :]
    )
    return torch.cat(
        [torch.ones((len(score), 1), dtype=score.dtype), score[:, None], rff], dim=1
    )


def ridge_coefficients(features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dimension = features.shape[1]
    penalty = torch.eye(dimension, dtype=features.dtype) * RIDGE_NUISANCE
    penalty[0, 0] = 0.0
    lhs = features.T @ features / len(features) + penalty
    rhs = features.T @ target / len(features)
    return torch.linalg.solve(lhs, rhs)


def cci_loss(
    score: torch.Tensor,
    a: torch.Tensor,
    phi_audit: torch.Tensor,
    omega: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    features = score_features(score, omega, phase)
    beta_a = ridge_coefficients(features, a[:, None])
    beta_v = ridge_coefficients(features, phi_audit)
    residual_a = a[:, None] - features @ beta_a
    residual_v = phi_audit - features @ beta_v
    return (residual_a.T @ residual_v / len(a)).square().sum()


def fit_representation(
    representation_input: np.ndarray,
    audit_input: np.ndarray,
    a: np.ndarray,
    seed: int,
    steps: int,
) -> RepresentationFit:
    input_mean = representation_input.mean(axis=0)
    input_scale = representation_input.std(axis=0) + 1e-8
    x = (representation_input - input_mean) / input_scale
    audit_spec = fit_rff(audit_input, seed + 101)
    phi_audit = transform_rff(audit_input, audit_spec)
    rng = np.random.default_rng(seed + 202)
    score_omega = rng.standard_normal(RFF_SCORE) / 0.20
    score_phase = rng.uniform(0.0, 2.0 * math.pi, RFF_SCORE)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = ScoreNet(x.shape[1]).double()
    xt = torch.tensor(x, dtype=torch.float64)
    at = torch.tensor(a, dtype=torch.float64)
    vt = torch.tensor(phi_audit, dtype=torch.float64)
    ot = torch.tensor(score_omega, dtype=torch.float64)
    pt = torch.tensor(score_phase, dtype=torch.float64)

    pretrain_steps = max(8, steps // 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=2e-4)
    for _ in range(pretrain_steps):
        score = model(xt).clamp(1e-5, 1.0 - 1e-5)
        loss = torch.nn.functional.binary_cross_entropy(score, at)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    optimizer = torch.optim.Adam(model.parameters(), lr=4e-3, weight_decay=2e-4)
    initial = None
    final = None
    for _ in range(steps):
        score = model(xt)
        discrepancy = cci_loss(score, at, vt, ot, pt)
        variance_penalty = torch.relu(torch.tensor(0.01, dtype=score.dtype) - score.var(unbiased=False))
        entropy_penalty = 0.01 * torch.nn.functional.binary_cross_entropy(
            score.clamp(1e-5, 1.0 - 1e-5), at
        )
        objective = discrepancy + 0.5 * variance_penalty + entropy_penalty
        if initial is None:
            initial = float(discrepancy.detach())
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        final = float(discrepancy.detach())
    assert initial is not None and final is not None
    return RepresentationFit(
        model, input_mean, input_scale, audit_spec, score_omega, score_phase, initial, final
    )


def predict_representation(fit: RepresentationFit, x: np.ndarray) -> np.ndarray:
    standardized = (x - fit.input_mean) / fit.input_scale
    with torch.no_grad():
        return fit.model(torch.tensor(standardized, dtype=torch.float64)).numpy()


def crossfitted_evaluation_diagnostics(
    score: np.ndarray,
    a: np.ndarray,
    audit: np.ndarray,
    spec: RFFSpec,
    inner_folds: np.ndarray,
) -> dict[str, Any]:
    """Evaluate CCI and overlap without fitting nuisances on evaluated units."""
    phi = transform_rff(audit, spec)
    omega = torch.tensor(np.linspace(-12.0, 12.0, RFF_SCORE), dtype=torch.float64)
    phase = torch.tensor(np.linspace(0.0, 2.0 * math.pi, RFF_SCORE, endpoint=False), dtype=torch.float64)
    st = torch.tensor(score, dtype=torch.float64)
    features = score_features(st, omega, phase)
    at = torch.tensor(a[:, None], dtype=torch.float64)
    vt = torch.tensor(phi, dtype=torch.float64)
    residual_a = torch.full_like(at, float("nan"))
    residual_v = torch.full_like(vt, float("nan"))
    propensity = np.full(len(a), np.nan)
    inner_qa = []
    for inner_fold in range(N_FOLDS):
        evaluate = np.flatnonzero(inner_folds == inner_fold)
        train = np.flatnonzero(inner_folds != inner_fold)
        beta_a = ridge_coefficients(features[train], at[train])
        beta_v = ridge_coefficients(features[train], vt[train])
        residual_a[evaluate] = at[evaluate] - features[evaluate] @ beta_a
        residual_v[evaluate] = vt[evaluate] - features[evaluate] @ beta_v
        propensity_model = LogisticRegression(
            C=1.0,
            fit_intercept=False,
            solver="lbfgs",
            max_iter=1000,
        )
        propensity_model.fit(features[train].numpy(), a[train])
        propensity[evaluate] = propensity_model.predict_proba(features[evaluate].numpy())[:, 1]
        inner_qa.append(
            {
                "inner_fold": inner_fold,
                "train_n": len(train),
                "evaluate_n": len(evaluate),
                "train_evaluate_disjoint": not np.intersect1d(train, evaluate).size,
                "train_has_both_arms": len(np.unique(a[train])) == 2,
                "evaluate_has_both_arms": len(np.unique(a[evaluate])) == 2,
            }
        )
    if not torch.isfinite(residual_a).all() or not torch.isfinite(residual_v).all():
        raise AssertionError("non-finite internally cross-fitted residuals")
    if not np.isfinite(propensity).all():
        raise AssertionError("non-finite internally cross-fitted propensities")
    crosscov = residual_a.T @ residual_v / len(a)

    clipped = np.clip(propensity, 1e-3, 1.0 - 1e-3)
    treated_weights = 1.0 / clipped[a == 1]
    control_weights = 1.0 / (1.0 - clipped[a == 0])

    def effective_sample_size(weights: np.ndarray) -> float:
        return float(weights.sum() ** 2 / max(float(np.square(weights).sum()), 1e-12))

    treated_ess = effective_sample_size(treated_weights)
    control_ess = effective_sample_size(control_weights)
    treated_ess_fraction = treated_ess / max(int(np.sum(a == 1)), 1)
    control_ess_fraction = control_ess / max(int(np.sum(a == 0)), 1)

    treated = score[a == 1]
    control = score[a == 0]
    common_low = max(float(treated.min()), float(control.min()))
    common_high = min(float(treated.max()), float(control.max()))
    cuts = np.unique(np.quantile(score, np.linspace(0.0, 1.0, N_OVERLAP_CELLS + 1)))
    cell_counts = []
    if len(cuts) > 1:
        labels = np.clip(np.digitize(score, cuts[1:-1]), 0, len(cuts) - 2)
        for cell in np.unique(labels):
            aa = a[labels == cell]
            cell_counts.append(
                {"cell": int(cell), "n1": int(aa.sum()), "n0": int(len(aa) - aa.sum())}
            )
    supported = [
        item["n1"] >= MIN_CELL_ARM_COUNT and item["n0"] >= MIN_CELL_ARM_COUNT
        for item in cell_counts
    ]
    minimum_cell_arm_count = min(
        [min(item["n1"], item["n0"]) for item in cell_counts], default=0
    )
    propensity_interior_fraction = float(
        np.mean((propensity >= PROPENSITY_LOWER) & (propensity <= PROPENSITY_UPPER))
    )
    overlap_components = {
        "propensity_interior": propensity_interior_fraction >= MIN_PROPENSITY_INTERIOR_FRACTION,
        "treated_ess": treated_ess_fraction >= MIN_ARM_ESS_FRACTION,
        "control_ess": control_ess_fraction >= MIN_ARM_ESS_FRACTION,
        "local_cell_support": bool(supported) and all(supported),
    }
    return {
        "heldout_joint_cci_crossfitted": float(torch.linalg.norm(crosscov)),
        "inner_nuisance_crossfit": inner_qa,
        "common_score_support": common_low < common_high,
        "common_support_interval": [common_low, common_high],
        "propensity_interior_fraction": propensity_interior_fraction,
        "treated_ess": treated_ess,
        "control_ess": control_ess,
        "treated_ess_fraction": treated_ess_fraction,
        "control_ess_fraction": control_ess_fraction,
        "quantile_cell_counts": cell_counts,
        "supported_quantile_cell_fraction": float(np.mean(supported)) if supported else 0.0,
        "minimum_quantile_cell_arm_count": minimum_cell_arm_count,
        "overlap_components": overlap_components,
        "overlap_gate_pass": all(overlap_components.values()),
        "score_range": [float(score.min()), float(score.max())],
    }


def fit_outcome(
    train_x: np.ndarray,
    train_a: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    linear: bool,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    for arm in (1.0, 0.0):
        mask = train_a == arm
        if int(mask.sum()) < 8:
            raise RuntimeError(f"too few training observations in arm {int(arm)}")
        if linear:
            model = Ridge(alpha=1e-3)
        else:
            model = GradientBoostingRegressor(
                n_estimators=80,
                max_depth=2,
                learning_rate=0.05,
                min_samples_leaf=5,
                random_state=seed + int(arm),
            )
        model.fit(train_x[mask], train_y[mask])
        predictions.append(model.predict(test_x))
    return predictions[0], predictions[1]


@dataclass
class RunRecord:
    transform: str
    n: int
    rep: int
    method: str
    ate_estimate: float
    ate_error: float
    j1_estimate: float
    j1_error: float
    heldout_joint_cci: float | None
    common_score_support: bool | None
    propensity_interior_fraction: float | None
    treated_ess: float | None
    control_ess: float | None
    treated_ess_fraction: float | None
    control_ess_fraction: float | None
    supported_quantile_cell_fraction: float | None
    minimum_quantile_cell_arm_count: int | None
    overlap_gate_pass: bool | None
    score_min: float | None
    score_max: float | None
    optimization_decreased_fraction: float | None


def method_arrays(data: dict[str, np.ndarray], method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == "x_only":
        return data["X"], data["X"]
    if method == "cross_view":
        return np.column_stack([data["X"], data["Z"]]), np.column_stack([data["X"], data["W"]])
    if method == "self_conditioned":
        all_observed = np.column_stack([data["X"], data["Z"], data["W"]])
        return all_observed, np.column_stack([data["X"], data["W"]])
    raise ValueError(method)


def run_representation_method(
    data: dict[str, np.ndarray],
    folds: np.ndarray,
    transform: str,
    n: int,
    rep: int,
    method: str,
    steps: int,
    evaluation_contract: dict[int, dict[str, Any]],
    replay: bool = False,
) -> tuple[RunRecord, dict[str, Any]]:
    rep_input, training_audit = method_arrays(data, method)
    joint_audit = np.column_stack([data["X"], data["W"]])
    m1 = np.full(n, np.nan)
    m0 = np.full(n, np.nan)
    diagnostics = []
    replay_passes = []
    for fold_id in range(N_FOLDS):
        test = np.flatnonzero(folds == fold_id)
        train = np.flatnonzero(folds != fold_id)
        seed = stable_seed(transform, n, rep, fold_id, method)
        fit = fit_representation(rep_input[train], training_audit[train], data["A"][train], seed, steps)
        score_train = predict_representation(fit, rep_input[train])
        score_test = predict_representation(fit, rep_input[test])
        pred1, pred0 = fit_outcome(
            score_train[:, None], data["A"][train], data["Y"][train], score_test[:, None], seed + 5000, False
        )
        m1[test], m0[test] = pred1, pred0
        evaluation = evaluation_contract[fold_id]
        joint_spec = evaluation["audit_rff_spec"]
        diagnostic = crossfitted_evaluation_diagnostics(
            score_test,
            data["A"][test],
            joint_audit[test],
            joint_spec,
            evaluation["inner_folds"],
        )
        diagnostics.append(
            {
                "fold": fold_id,
                "train_test_disjoint": not np.intersect1d(train, test).size,
                "audit_w_excluded_from_representation": method != "self_conditioned",
                "outcome_or_u_used_in_representation": False,
                "evaluation_audit_variables": ["X", "W"],
                "evaluation_audit_rff_hash": evaluation["audit_rff_hash"],
                "evaluation_inner_fold_hash": evaluation["inner_fold_hash"],
                "optimization_decreased": fit.final_objective <= fit.initial_objective,
                **diagnostic,
            }
        )
        if replay and fold_id == 0:
            duplicate = fit_representation(
                rep_input[train], training_audit[train], data["A"][train], seed, steps
            )
            replay_passes.append(
                np.array_equal(score_test, predict_representation(duplicate, rep_input[test]))
            )
    if not np.isfinite(m1).all() or not np.isfinite(m0).all():
        raise AssertionError(f"non-finite cross-fitted outcome prediction for {method}")
    ate = float(np.mean(m1 - m0))
    j1 = float(np.mean(m1))
    record = RunRecord(
        transform,
        n,
        rep,
        method,
        ate,
        ate - TRUE_ATE,
        j1,
        j1 - TRUE_J1,
        float(np.mean([item["heldout_joint_cci_crossfitted"] for item in diagnostics])),
        all(item["common_score_support"] for item in diagnostics),
        float(np.mean([item["propensity_interior_fraction"] for item in diagnostics])),
        float(np.mean([item["treated_ess"] for item in diagnostics])),
        float(np.mean([item["control_ess"] for item in diagnostics])),
        float(np.mean([item["treated_ess_fraction"] for item in diagnostics])),
        float(np.mean([item["control_ess_fraction"] for item in diagnostics])),
        float(np.mean([item["supported_quantile_cell_fraction"] for item in diagnostics])),
        int(min(item["minimum_quantile_cell_arm_count"] for item in diagnostics)),
        all(item["overlap_gate_pass"] for item in diagnostics),
        float(min(item["score_range"][0] for item in diagnostics)),
        float(max(item["score_range"][1] for item in diagnostics)),
        float(np.mean([item["optimization_decreased"] for item in diagnostics])),
    )
    return record, {"folds": diagnostics, "deterministic_replay": all(replay_passes) if replay_passes else None}


def run_direct_method(
    data: dict[str, np.ndarray], folds: np.ndarray, transform: str, n: int, rep: int, method: str
) -> RunRecord:
    if method == "full_observed":
        features = np.column_stack([data["X"], data["Z"], data["W"]])
        linear = False
    elif method == "privileged_xprime_u":
        features = np.column_stack([data["Xp"], data["U"]])
        linear = True
    else:
        raise ValueError(method)
    m1 = np.full(n, np.nan)
    m0 = np.full(n, np.nan)
    for fold_id in range(N_FOLDS):
        test = np.flatnonzero(folds == fold_id)
        train = np.flatnonzero(folds != fold_id)
        pred1, pred0 = fit_outcome(
            features[train], data["A"][train], data["Y"][train], features[test],
            BASE_SEED + 3001 * rep + 101 * fold_id, linear,
        )
        m1[test], m0[test] = pred1, pred0
    ate = float(np.mean(m1 - m0))
    j1 = float(np.mean(m1))
    return RunRecord(
        transform, n, rep, method, ate, ate - TRUE_ATE, j1, j1 - TRUE_J1,
        None, None, None, None, None, None, None, None, None, None, None, None, None,
    )


def run_once(transform: str, n: int, rep: int, steps: int, replay: bool) -> tuple[list[RunRecord], dict[str, Any]]:
    rng = np.random.default_rng(sample_seed(n, rep))
    data = kmu.simulate_kmu(rng, n, transform)
    folds = stratified_folds(data["A"], BASE_SEED + 7907 * rep + n)
    joint_audit = np.column_stack([data["X"], data["W"]])
    evaluation_contract: dict[int, dict[str, Any]] = {}
    for fold_id in range(N_FOLDS):
        test = np.flatnonzero(folds == fold_id)
        train = np.flatnonzero(folds != fold_id)
        seed = diagnostic_seed(transform, n, rep, fold_id)
        audit_spec = fit_rff(joint_audit[train], seed)
        inner_folds = stratified_folds(data["A"][test], seed + 100_003)
        rff_array = np.concatenate(
            [
                audit_spec.mean.ravel(),
                audit_spec.scale.ravel(),
                audit_spec.omega.ravel(),
                audit_spec.phase.ravel(),
                np.array([audit_spec.bandwidth]),
            ]
        )
        evaluation_contract[fold_id] = {
            "audit_rff_spec": audit_spec,
            "audit_rff_hash": array_hash(rff_array),
            "inner_folds": inner_folds,
            "inner_fold_hash": array_hash(inner_folds),
        }
    records = []
    details: dict[str, Any] = {}
    for method in ("x_only", "cross_view", "self_conditioned"):
        record, detail = run_representation_method(
            data, folds, transform, n, rep, method, steps, evaluation_contract, replay
        )
        records.append(record)
        details[method] = detail
    for method in ("full_observed", "privileged_xprime_u"):
        records.append(run_direct_method(data, folds, transform, n, rep, method))
    return records, {
        "fold_counts": [int(np.sum(folds == fold)) for fold in range(N_FOLDS)],
        "fold_arm_counts": [
            {"n1": int(data["A"][folds == fold].sum()), "n0": int(np.sum(folds == fold) - data["A"][folds == fold].sum())}
            for fold in range(N_FOLDS)
        ],
        "methods": details,
    }


def metric(errors: np.ndarray) -> dict[str, float]:
    return {
        "signed_bias": float(errors.mean()),
        "mean_absolute_error": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "monte_carlo_se_bias": float(errors.std(ddof=1) / math.sqrt(len(errors))) if len(errors) > 1 else 0.0,
    }


def sign_test(differences: np.ndarray) -> dict[str, Any]:
    nonzero = differences[np.abs(differences) > 1e-15]
    wins = int(np.sum(nonzero < 0))
    losses = int(np.sum(nonzero > 0))
    if len(nonzero):
        tail = min(wins, losses)
        pvalue = min(1.0, 2.0 * sum(math.comb(len(nonzero), k) for k in range(tail + 1)) / 2.0 ** len(nonzero))
    else:
        pvalue = 1.0
    return {"wins": wins, "losses": losses, "ties": int(len(differences) - len(nonzero)), "two_sided_exact_p": pvalue}


def summarize(records: list[RunRecord], args: argparse.Namespace, qa: dict[str, Any], runtime: float) -> dict[str, Any]:
    configurations: dict[str, Any] = {}
    primary_passes = []
    for transform in args.transforms:
        for n in args.sample_sizes:
            key = f"transform={transform}|n={n}"
            subset = [row for row in records if row.transform == transform and row.n == n]
            methods = {}
            for method in ("x_only", "cross_view", "full_observed", "privileged_xprime_u", "self_conditioned"):
                rows = [row for row in subset if row.method == method]
                ate_errors = np.array([row.ate_error for row in rows])
                j1_errors = np.array([row.j1_error for row in rows])
                methods[method] = {"ate": metric(ate_errors), "j1": metric(j1_errors)}
                if method in ("x_only", "cross_view", "self_conditioned"):
                    methods[method].update(
                        {
                            "certificate_competitor": method != "self_conditioned",
                            "mean_heldout_joint_cci_crossfitted": float(np.mean([row.heldout_joint_cci for row in rows])),
                            "common_score_support_rate": float(np.mean([row.common_score_support for row in rows])),
                            "mean_propensity_interior_fraction": float(np.mean([row.propensity_interior_fraction for row in rows])),
                            "mean_treated_ess": float(np.mean([row.treated_ess for row in rows])),
                            "mean_control_ess": float(np.mean([row.control_ess for row in rows])),
                            "mean_treated_ess_fraction": float(np.mean([row.treated_ess_fraction for row in rows])),
                            "mean_control_ess_fraction": float(np.mean([row.control_ess_fraction for row in rows])),
                            "mean_supported_quantile_cell_fraction": float(np.mean([row.supported_quantile_cell_fraction for row in rows])),
                            "minimum_quantile_cell_arm_count": int(min(row.minimum_quantile_cell_arm_count for row in rows)),
                            "overlap_gate_pass_rate": float(np.mean([row.overlap_gate_pass for row in rows])),
                            "all_runs_overlap_gate_pass": all(row.overlap_gate_pass for row in rows),
                            "optimization_decreased_rate": float(np.mean([row.optimization_decreased_fraction for row in rows])),
                        }
                    )
                else:
                    methods[method]["certificate_competitor"] = False
                    methods[method]["certificate_exclusion_reason"] = (
                        "Consumes the audit variables or simulation-only latent variables; estimation reference only."
                    )
            xrows = sorted([row for row in subset if row.method == "x_only"], key=lambda row: row.rep)
            crows = sorted([row for row in subset if row.method == "cross_view"], key=lambda row: row.rep)
            ate_diff = np.array([abs(c.ate_error) - abs(x.ate_error) for c, x in zip(crows, xrows)])
            cci_diff = np.array([c.heldout_joint_cci - x.heldout_joint_cci for c, x in zip(crows, xrows)])
            relative_improvement_pass = (
                methods["cross_view"]["ate"]["mean_absolute_error"] < methods["x_only"]["ate"]["mean_absolute_error"]
                and methods["cross_view"]["ate"]["rmse"] < methods["x_only"]["ate"]["rmse"]
                and methods["cross_view"]["mean_heldout_joint_cci_crossfitted"]
                < methods["x_only"]["mean_heldout_joint_cci_crossfitted"]
            )
            overlap_gate_pass = methods["cross_view"]["all_runs_overlap_gate_pass"]
            absolute_accuracy_pass = (
                methods["cross_view"]["ate"]["mean_absolute_error"] <= abs(TRUE_ATE)
            )
            cell_primary_success = (
                relative_improvement_pass and overlap_gate_pass and absolute_accuracy_pass
            )
            if transform == "cube":
                primary_passes.append(cell_primary_success)
            configurations[key] = {
                "role": "primary_assumption_aligned" if transform == "cube" else "published_noninvertible_stress_test",
                "methods": methods,
                "paired_cross_minus_x_absolute_ate_error": {
                    "mean": float(ate_diff.mean()),
                    **sign_test(ate_diff),
                },
                "paired_cross_minus_x_heldout_cci": {
                    "mean": float(cci_diff.mean()),
                    **sign_test(cci_diff),
                },
                "relative_improvement_pass": relative_improvement_pass,
                "overlap_gate_pass": overlap_gate_pass,
                "absolute_accuracy_pass": absolute_accuracy_pass,
                "cell_primary_success": cell_primary_success,
            }
    analytic = 1.0 + 0.7 * DIMENSION + DIMENSION * (0.2 + 0.5 * DIMENSION / (DIMENSION + 1.0))
    pipeline = {
        "analytic_j1_assertion": abs(analytic - TRUE_J1) < 1e-12,
        "all_outputs_finite": all(math.isfinite(row.ate_estimate) and math.isfinite(row.j1_estimate) for row in records),
        "all_fold_splits_disjoint": all(
            fold["train_test_disjoint"]
            for item in qa.values() for method in item.get("methods", {}).values() for fold in method["folds"]
        ),
        "proposed_audit_excluded": all(
            fold["audit_w_excluded_from_representation"]
            for item in qa.values() for fold in item.get("methods", {}).get("cross_view", {}).get("folds", [])
        ),
        "no_outcome_or_u_leakage": all(
            not fold["outcome_or_u_used_in_representation"]
            for item in qa.values() for method in item.get("methods", {}).values() for fold in method["folds"]
        ),
        "proposed_common_score_support": all(
            row.common_score_support for row in records if row.method == "cross_view"
        ),
        "common_evaluation_audit_map_across_methods": all(
            len(
                {
                    method["folds"][fold_id]["evaluation_audit_rff_hash"]
                    for method in item.get("methods", {}).values()
                }
            )
            == 1
            for item in qa.values()
            for fold_id in range(N_FOLDS)
        ),
        "common_evaluation_inner_split_across_methods": all(
            len(
                {
                    method["folds"][fold_id]["evaluation_inner_fold_hash"]
                    for method in item.get("methods", {}).values()
                }
            )
            == 1
            for item in qa.values()
            for fold_id in range(N_FOLDS)
        ),
        "all_inner_nuisance_splits_disjoint": all(
            inner["train_evaluate_disjoint"]
            for item in qa.values()
            for method in item.get("methods", {}).values()
            for fold in method["folds"]
            for inner in fold["inner_nuisance_crossfit"]
        ),
    }
    replay_values = [
        method["deterministic_replay"]
        for item in qa.values() for method in item.get("methods", {}).values()
        if method["deterministic_replay"] is not None
    ]
    pipeline["deterministic_replay"] = all(replay_values) if replay_values else None
    return jsonable(
        {
            "provenance": {
                "script": "code/kmu_2022_two_view_representation_experiment.py",
                "source": kmu.SOURCE_URL,
                "dgp_generator": "code/kmu_2022_single_proxy_ablation_experiment.py::simulate_kmu",
                "dgp_completions_frozen_from_existing_ablation": True,
                "base_seed": BASE_SEED,
                "sample_sizes": args.sample_sizes,
                "transforms": args.transforms,
                "repetitions": args.reps,
                "steps": args.steps,
                "dimension_each_of_x_z_w_u": DIMENSION,
                "representation_input": "(X,Z) where KMU Z is Wrep",
                "audit_input": "(X,W) where KMU W is Waudit",
                "evaluation_audit_contract": "For each transform,n,rep,outer fold, all representation methods share the same (X,W), fitted RFF map, and inner split.",
                "evaluation_residualization": "Two-fold internal cross-fitting inside each held-out outer fold.",
                "objective": "finite-RFF conditional-cross-covariance surrogate, not exact KCI or conditional HSIC",
                "final_adjustment": "two-fold cross-fitted T-learner using scalar R only",
                "cube_role": "primary because cube is invertible",
                "sin_role": "stress test because sin is not globally invertible despite the paper's invertibility wording",
                "native_proximal_comparator": "not implemented; published Tables 2-5 are external reference only",
                "full_observed_label": "direct regression baseline, not a proximal estimator",
                "privileged_label": "simulation-only Xprime,U linear reference",
                "no_sample_deletion": True,
                "runtime_seconds": runtime,
            },
            "targets": {
                "primary_ate": TRUE_ATE,
                "secondary_published_j1": TRUE_J1,
                "j1_derivation": "1 + 60*0.7 + 60*(0.2 + 0.5*60/61)",
            },
            "overlap_gate": {
                "status": "fixed before the repair rerun but after inspection of the initial run",
                "propensity_interval": [PROPENSITY_LOWER, PROPENSITY_UPPER],
                "minimum_propensity_interior_fraction": MIN_PROPENSITY_INTERIOR_FRACTION,
                "minimum_treated_ess_fraction": MIN_ARM_ESS_FRACTION,
                "minimum_control_ess_fraction": MIN_ARM_ESS_FRACTION,
                "quantile_cells": N_OVERLAP_CELLS,
                "minimum_units_per_arm_per_cell": MIN_CELL_ARM_COUNT,
                "run_pass_rule": "Every outer fold passes every overlap component.",
            },
            "published_external_reference": {
                "scope": "KMU Tables 2-5; normalized errors over 200 replications, not reproduced here and not directly pooled with our completed DGP",
                "neural_dr_without_stabilizer": {
                    "transform=sin|n=400": {"normalized_mse": 0.0025, "normalized_squared_bias": 0.0004},
                    "transform=sin|n=1200": {"normalized_mse": 0.0020, "normalized_squared_bias": 0.0004},
                    "transform=cube|n=400": {"normalized_mse": 0.0068, "normalized_squared_bias": 0.0021},
                    "transform=cube|n=1200": {"normalized_mse": 0.0055, "normalized_squared_bias": 0.0029},
                },
                "comparison_warning": "The paper leaves G and Gaussian completion values unspecified; these published numbers are context, not a same-seed baseline.",
            },
            "pipeline_gates": pipeline,
            "pipeline_pass": all(value is True for value in pipeline.values() if value is not None),
            "primary_success_rule": {
                "relative": "Cross-view lowers internally cross-fitted held-out joint CCI, ATE MAE, and ATE RMSE versus matched X-only.",
                "overlap": "Every cross-view run passes the fixed propensity, ESS, and local-cell overlap gate.",
                "absolute": "Cross-view ATE MAE is no greater than |ATE|=1.",
                "cell": "relative AND overlap AND absolute",
                "overall": "Both cube cells pass the cell rule.",
                "timing_disclosure": "The absolute and strengthened overlap gates were fixed after the initial run and before this repair rerun.",
            },
            "primary_success": all(primary_passes) if primary_passes else None,
            "configurations": configurations,
            "qa": qa,
        }
    )


def write_outputs(records: list[RunRecord], summary: dict[str, Any]) -> None:
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    with RUNS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[400, 1200])
    parser.add_argument("--transforms", nargs="+", choices=["sin", "cube"], default=["sin", "cube"])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    requested = (tuple(args.sample_sizes), tuple(args.transforms), args.reps, args.steps)
    if args.smoke:
        if not args.no_write:
            parser.error("smoke runs require --no-write so they cannot overwrite registered outputs")
        args.sample_sizes = [200]
        args.transforms = ["sin", "cube"]
        args.reps = 1
        args.steps = 12
    registered = requested == ((400, 1200), ("sin", "cube"), 20, 45)
    if not args.smoke and not args.no_write and not registered:
        parser.error("nonregistered configurations require --no-write")

    started = time.time()
    records: list[RunRecord] = []
    qa: dict[str, Any] = {}
    for transform in args.transforms:
        for n in args.sample_sizes:
            for rep in range(args.reps):
                replay = args.smoke or (
                    transform == args.transforms[0] and n == args.sample_sizes[0] and rep == 0
                )
                rows, detail = run_once(transform, n, rep, args.steps, replay=replay)
                records.extend(rows)
                qa[f"transform={transform}|n={n}|rep={rep}"] = detail
                print(f"transform={transform} n={n} rep={rep + 1}/{args.reps}", flush=True)
    summary = summarize(records, args, qa, time.time() - started)
    if not args.no_write:
        write_outputs(records, summary)
    print(
        json.dumps(
            {
                "smoke": args.smoke,
                "pipeline_pass": summary["pipeline_pass"],
                "failed_gates": [key for key, value in summary["pipeline_gates"].items() if value is False],
                "primary_success": summary["primary_success"],
                "runtime_seconds": summary["provenance"]["runtime_seconds"],
                "output_written": not args.no_write,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
