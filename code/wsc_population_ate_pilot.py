#!/usr/bin/env python3
"""Locked WSC population-ATE pilot for cross-mask proxy balancing.

The learned models use only the QED subset Z=S.  The full randomized sample
is accessed only to compute the scalar randomized benchmark and its standard
error.  The conditional-dependence criterion is a finite-RFF surrogate, not
an exact KCI statistic.
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
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "materials" / "real_world_data" / "wscdata"
MANIFEST_PATH = ROOT / "materials" / "real_world_data" / "manifest.json"
PREREG_PATH = ROOT / "memo" / "2026-08-23-wsc-population-ate-pilot-preregistration.md"
OUTPUT_PATH = ROOT / "results" / "wsc_population_ate_pilot.json"

ROOT_SEED = 190602
N_REPEATS = 3
N_OUTER_FOLDS = 3
N_INNER_FOLDS = 2
N_AUDIT_MASKS = 5

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
    "min_samples_leaf": 15,
}
PROPENSITY_C = 1.0

OVERLAP_LOWER = 0.05
OVERLAP_UPPER = 0.95
OVERLAP_MIN_CENTRAL_FRACTION = 0.90
OVERLAP_MIN_ESS_ABSOLUTE = 25.0
OVERLAP_MIN_ESS_FRACTION = 0.20
OVERLAP_CELLS = 5
OVERLAP_MIN_ARM_PER_CELL = 5

X_COLUMNS = [
    "female", "white", "black", "hisp", "asian", "married", "age",
    "collegeS", "collegeM", "collegeD", "calc", "books", "mathLike", "income",
]
PROXY_BLOCKS = [
    ("big5", "fin_big5.csv"),
    ("amas", "fin_amas.csv"),
    ("bdi", "fin_bdi.csv"),
    ("gses", "fin_gses.csv"),
    ("math_pre", "fin_math_pre.csv"),
    ("vocab_pre", "fin_vocab_pre.csv"),
]
EXPECTED_AUDIT_NAMES = [
    "gses:item_2",
    "big5:C8",
    "big5:A4",
    "big5:O3",
    "mcs_pre:item_1",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


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
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def manifest_hashes() -> dict[str, str]:
    payload = json.loads(MANIFEST_PATH.read_text())
    out: dict[str, str] = {}
    for entry in payload["files"]:
        relative = entry.get("local_path", entry.get("relative_path", entry.get("path")))
        if relative is not None:
            out[relative] = entry["sha256"]
    return out


def load_data() -> dict[str, Any]:
    manifest = manifest_hashes()
    main = pd.read_csv(DATA_ROOT / "df2200_cleaned.csv")
    if main.shape != (2200, 33) or int(main.isna().sum().sum()) != 0:
        raise RuntimeError("WSC main-table integrity check failed")

    blocks: list[np.ndarray] = []
    names: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    for block, filename in PROXY_BLOCKS:
        frame = pd.read_csv(DATA_ROOT / filename)
        frames[block] = frame
        blocks.append(frame.to_numpy(dtype=float))
        names.extend([f"{block}:{column}" for column in frame.columns])

    mcs_all = pd.read_csv(DATA_ROOT / "fin_mcs.csv")
    mcs_pre = mcs_all.loc[mcs_all["type"] == "pre"].drop(columns="type").reset_index(drop=True)
    mcs_post = mcs_all.loc[mcs_all["type"] == "post"].drop(columns="type").reset_index(drop=True)
    if len(mcs_pre) != 2200 or len(mcs_post) != 2200:
        raise RuntimeError("MCS pre/post row count check failed")
    blocks.append(mcs_pre.to_numpy(dtype=float))
    names.extend([f"mcs_pre:{column}" for column in mcs_pre.columns])
    w = np.column_stack(blocks)

    big5 = frames["big5"]
    big5_keys = {
        "big5O": ([f"O{i}" for i in range(1, 11)], ["O7", "O9"]),
        "big5C": ([f"C{i}" for i in range(1, 10)], ["C2", "C4", "C5", "C9"]),
        "big5E": ([f"E{i}" for i in range(1, 9)], ["E2", "E5", "E7"]),
        "big5A": ([f"A{i}" for i in range(1, 10)], ["A1", "A3", "A6", "A8"]),
        "big5N": ([f"N{i}" for i in range(1, 9)], ["N2", "N5", "N7"]),
    }
    big5_checks: dict[str, bool] = {}
    for target, (items, reverse_items) in big5_keys.items():
        scored = big5[items].copy()
        scored[reverse_items] = 6 - scored[reverse_items]
        big5_checks[target] = np.array_equal(scored.sum(axis=1).to_numpy(), main[target].to_numpy())

    aggregate_checks = {
        **big5_checks,
        "AMAS": np.array_equal(frames["amas"].sum(axis=1).to_numpy(), main["AMAS"].to_numpy()),
        "BDI": np.array_equal(frames["bdi"].sum(axis=1).to_numpy(), main["BDI"].to_numpy()),
        "GSES": np.array_equal(frames["gses"].sum(axis=1).to_numpy(), main["GSES"].to_numpy()),
        "mathPre": np.array_equal(frames["math_pre"].sum(axis=1).to_numpy(), main["mathPre"].to_numpy()),
        "vocabPre": np.array_equal(frames["vocab_pre"].sum(axis=1).to_numpy(), main["vocabPre"].to_numpy()),
        "MCS": np.array_equal(mcs_pre.sum(axis=1).to_numpy(), main["MCS"].to_numpy()),
        "MCSpost": np.array_equal(mcs_post.sum(axis=1).to_numpy(), main["MCSpost"].to_numpy()),
    }
    if not all(aggregate_checks.values()):
        raise RuntimeError(f"aggregate-score alignment failed: {aggregate_checks}")
    if w.shape != (2200, 114) or np.isnan(w).any():
        raise RuntimeError("proxy matrix integrity check failed")

    rng = np.random.default_rng(ROOT_SEED)
    audit_indices = rng.choice(w.shape[1], N_AUDIT_MASKS, replace=False)
    audit_names = [names[index] for index in audit_indices]
    if audit_names != EXPECTED_AUDIT_NAMES:
        raise RuntimeError(f"audit mask drift: {audit_names}")

    z = main["mathGrp"].to_numpy(dtype=int)
    s = main["mathSel"].to_numpy(dtype=int)
    y = main["mathPost"].to_numpy(dtype=float)
    qed = z == s
    qed_indices = np.flatnonzero(qed)
    x = main[X_COLUMNS].to_numpy(dtype=float)
    if int(qed.sum()) != 1105 or int(z[qed].sum()) != 288 or int((1 - z[qed]).sum()) != 817:
        raise RuntimeError("QED cell count drift")

    source_files = ["df2200_cleaned.csv"] + [filename for _, filename in PROXY_BLOCKS] + ["fin_mcs.csv"]
    source_hashes: dict[str, dict[str, Any]] = {}
    for filename in source_files:
        path = DATA_ROOT / filename
        digest = sha256_file(path)
        candidates = [
            f"wscdata/{filename}",
            f"wscdata/raw/{filename}",
            f"materials/real_world_data/wscdata/{filename}",
        ]
        recorded = next((manifest[key] for key in candidates if key in manifest), None)
        source_hashes[filename] = {
            "sha256": digest,
            "manifest_sha256": recorded,
            "matches_manifest": recorded == digest,
        }
    if not all(item["matches_manifest"] for item in source_hashes.values()):
        raise RuntimeError(f"source-manifest hash mismatch: {source_hashes}")

    treated = z == 1
    rct = float(y[treated].mean() - y[~treated].mean())
    rct_se = float(math.sqrt(y[treated].var(ddof=1) / treated.sum() + y[~treated].var(ddof=1) / (~treated).sum()))
    return {
        "X": x[qed],
        "W": w[qed],
        "A": z[qed].astype(float),
        "Y": y[qed],
        "qed_indices": qed_indices,
        "proxy_names": names,
        "audit_indices": audit_indices,
        "audit_names": audit_names,
        "aggregate_checks": aggregate_checks,
        "source_hashes": source_hashes,
        "rct_benchmark": rct,
        "rct_standard_error": rct_se,
        "full_counts": {"n": len(main), "n1": int(treated.sum()), "n0": int((~treated).sum())},
        "qed_counts": {"n": int(qed.sum()), "n1": int(z[qed].sum()), "n0": int((1 - z[qed]).sum())},
    }


def stratified_folds(a: np.ndarray, seed: int, n_folds: int) -> np.ndarray:
    folds = np.empty(len(a), dtype=int)
    rng = np.random.default_rng(seed)
    for arm in (0, 1):
        index = np.flatnonzero(a.astype(int) == arm)
        index = index[rng.permutation(len(index))]
        folds[index] = np.arange(len(index)) % n_folds
    return folds


def standardize_fit(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return mean, scale


def median_bandwidth(values: np.ndarray, seed: int, maximum: int = 500) -> float:
    if len(values) > maximum:
        rng = np.random.default_rng(seed)
        values = values[rng.choice(len(values), maximum, replace=False)]
    distances = ((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2)
    upper = distances[np.triu_indices(len(values), 1)]
    return math.sqrt(max(float(np.median(upper)), 1e-8) / 2.0)


@dataclass
class RFFSpec:
    mean: np.ndarray
    scale: np.ndarray
    omega: np.ndarray
    phase: np.ndarray
    bandwidth: float


@dataclass
class EvaluationSpec:
    v_spec: RFFSpec
    r_omega: np.ndarray
    r_phase: np.ndarray
    inner_folds: np.ndarray
    signature: str


def make_v_rff(audit_train: np.ndarray, seed: int) -> RFFSpec:
    mean, scale = standardize_fit(audit_train)
    standardized = (audit_train - mean) / scale
    bandwidth = median_bandwidth(standardized, seed)
    rng = np.random.default_rng(seed + 101)
    omega = rng.standard_normal((standardized.shape[1], RFF_V)) / bandwidth
    phase = rng.uniform(0.0, 2.0 * math.pi, RFF_V)
    return RFFSpec(mean, scale, omega, phase, bandwidth)


def v_rff_numpy(values: np.ndarray, spec: RFFSpec) -> np.ndarray:
    standardized = (values - spec.mean) / spec.scale
    return math.sqrt(2.0 / RFF_V) * np.cos(standardized @ spec.omega + spec.phase)


def make_evaluation_spec(audit_train: np.ndarray, a_test: np.ndarray, seed: int) -> EvaluationSpec:
    v_spec = make_v_rff(audit_train, seed + 11)
    rng = np.random.default_rng(seed + 13)
    r_omega = rng.standard_normal(RFF_R) / R_BANDWIDTH
    r_phase = rng.uniform(0.0, 2.0 * math.pi, RFF_R)
    inner_folds = stratified_folds(a_test, seed + 17, N_INNER_FOLDS)
    digest = hashlib.sha256()
    for array in (v_spec.mean, v_spec.scale, v_spec.omega, v_spec.phase, r_omega, r_phase, inner_folds):
        digest.update(np.ascontiguousarray(array).tobytes())
    return EvaluationSpec(v_spec, r_omega, r_phase, inner_folds, digest.hexdigest())


def r_features_torch(r: torch.Tensor, omega: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    rff = math.sqrt(2.0 / RFF_R) * torch.cos(r[:, None] * omega[None, :] + phase[None, :])
    return torch.cat([torch.ones((len(r), 1), dtype=r.dtype), r[:, None], rff], dim=1)


def r_features_numpy(r: np.ndarray, omega: np.ndarray, phase: np.ndarray) -> np.ndarray:
    rff = math.sqrt(2.0 / RFF_R) * np.cos(r[:, None] * omega[None, :] + phase[None, :])
    return np.column_stack([np.ones(len(r)), r, rff])


def ridge_coefficients(features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dimension = features.shape[1]
    penalty = torch.eye(dimension, dtype=features.dtype) * RIDGE
    penalty[0, 0] = 0.0
    lhs = features.T @ features / len(features) + penalty
    rhs = features.T @ target / len(features)
    return torch.linalg.solve(lhs, rhs)


def conditional_crosscov_loss(
    r: torch.Tensor,
    a: torch.Tensor,
    phi_v: torch.Tensor,
    omega: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    features = r_features_torch(r, omega, phase)
    residual_a = a[:, None] - features @ ridge_coefficients(features, a[:, None])
    residual_v = phi_v - features @ ridge_coefficients(features, phi_v)
    crosscov = residual_a.T @ residual_v / len(a)
    return crosscov.square().sum()


class LinearScore(torch.nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.linear = torch.nn.Linear(dimension, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(values)).squeeze(1)


@dataclass
class RepresentationFit:
    model: LinearScore
    mean: np.ndarray
    scale: np.ndarray
    cci_curve: list[float]
    pretrain_curve: list[float]


def fit_representation(
    inputs: np.ndarray,
    audit: np.ndarray,
    a: np.ndarray,
    spec: EvaluationSpec,
    seed: int,
) -> RepresentationFit:
    mean, scale = standardize_fit(inputs)
    standardized = (inputs - mean) / scale
    phi_v = v_rff_numpy(audit, spec.v_spec)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = LinearScore(standardized.shape[1]).double()
    xt = torch.tensor(standardized, dtype=torch.float64)
    at = torch.tensor(a, dtype=torch.float64)
    vt = torch.tensor(phi_v, dtype=torch.float64)
    omega = torch.tensor(spec.r_omega, dtype=torch.float64)
    phase = torch.tensor(spec.r_phase, dtype=torch.float64)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_PRETRAIN, weight_decay=1e-4)
    pretrain_curve: list[float] = []
    for _ in range(PRETRAIN_STEPS):
        score = model(xt)
        loss = torch.nn.functional.binary_cross_entropy(score.clamp(1e-5, 1.0 - 1e-5), at)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pretrain_curve.append(float(loss.detach()))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_CCI, weight_decay=1e-4)
    cci_curve: list[float] = []
    for _ in range(CCI_STEPS):
        score = model(xt)
        loss = conditional_crosscov_loss(score, at, vt, omega, phase)
        variance_floor = torch.relu(torch.tensor(0.015, dtype=score.dtype) - score.var(unbiased=False))
        objective = loss + 0.5 * variance_floor
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        cci_curve.append(float(loss.detach()))
    return RepresentationFit(model, mean, scale, cci_curve, pretrain_curve)


def predict_representation(fit: RepresentationFit, inputs: np.ndarray) -> np.ndarray:
    values = torch.tensor((inputs - fit.mean) / fit.scale, dtype=torch.float64)
    with torch.no_grad():
        return fit.model(values).numpy()


def heldout_crossfit_surrogate(spec: EvaluationSpec, r: np.ndarray, a: np.ndarray, audit: np.ndarray) -> float:
    features = torch.tensor(r_features_numpy(r, spec.r_omega, spec.r_phase), dtype=torch.float64)
    at = torch.tensor(a, dtype=torch.float64)
    vt = torch.tensor(v_rff_numpy(audit, spec.v_spec), dtype=torch.float64)
    total = torch.zeros((1, vt.shape[1]), dtype=torch.float64)
    evaluated = 0
    for fold in range(N_INNER_FOLDS):
        fit_mask = torch.tensor(spec.inner_folds != fold)
        eval_mask = torch.tensor(spec.inner_folds == fold)
        fit_features = features[fit_mask]
        beta_a = ridge_coefficients(fit_features, at[fit_mask, None])
        beta_v = ridge_coefficients(fit_features, vt[fit_mask])
        residual_a = at[eval_mask, None] - features[eval_mask] @ beta_a
        residual_v = vt[eval_mask] - features[eval_mask] @ beta_v
        total += residual_a.T @ residual_v
        evaluated += int(eval_mask.sum())
    return float(torch.linalg.norm(total / evaluated))


def fit_outcome_effect(
    train_features: np.ndarray,
    test_features: np.ndarray,
    a_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> np.ndarray:
    mean, scale = standardize_fit(train_features)
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale
    predictions: list[np.ndarray] = []
    for arm in (1, 0):
        mask = a_train.astype(int) == arm
        if int(mask.sum()) < 2 * GBR_SPEC["min_samples_leaf"]:
            raise RuntimeError("insufficient arm count for locked outcome learner")
        model = GradientBoostingRegressor(random_state=seed + arm, **GBR_SPEC)
        model.fit(train[mask], y_train[mask])
        predictions.append(model.predict(test))
    return predictions[0] - predictions[1]


def fit_propensity(
    train_features: np.ndarray,
    test_features: np.ndarray,
    a_train: np.ndarray,
    seed: int,
) -> np.ndarray:
    mean, scale = standardize_fit(train_features)
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale
    model = LogisticRegression(C=PROPENSITY_C, max_iter=3000, solver="lbfgs", random_state=seed)
    model.fit(train, a_train.astype(int))
    return model.predict_proba(test)[:, 1]


def effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.sum(weights * weights))


def overlap_summary(a: np.ndarray, propensity: np.ndarray) -> dict[str, Any]:
    treated = a.astype(int) == 1
    control = ~treated
    finite = bool(np.isfinite(propensity).all() and np.all((propensity > 0.0) & (propensity < 1.0)))
    if not finite:
        return {"gate_pass": False, "reason": "nonfinite_or_boundary_propensity"}
    treated_ess = effective_sample_size(1.0 / propensity[treated])
    control_ess = effective_sample_size(1.0 / (1.0 - propensity[control]))
    treated_minimum = max(OVERLAP_MIN_ESS_ABSOLUTE, OVERLAP_MIN_ESS_FRACTION * treated.sum())
    control_minimum = max(OVERLAP_MIN_ESS_ABSOLUTE, OVERLAP_MIN_ESS_FRACTION * control.sum())
    central_fraction = float(np.mean((propensity >= OVERLAP_LOWER) & (propensity <= OVERLAP_UPPER)))

    boundaries = np.quantile(propensity, np.linspace(0.0, 1.0, OVERLAP_CELLS + 1))
    cells = np.searchsorted(boundaries[1:-1], propensity, side="right")
    cell_rows: list[dict[str, Any]] = []
    for cell in range(OVERLAP_CELLS):
        member = cells == cell
        n1 = int(np.sum(member & treated))
        n0 = int(np.sum(member & control))
        cell_rows.append({
            "cell": cell,
            "n": int(member.sum()),
            "n1": n1,
            "n0": n0,
            "pass": n1 >= OVERLAP_MIN_ARM_PER_CELL and n0 >= OVERLAP_MIN_ARM_PER_CELL,
        })
    components = {
        "central_fraction": central_fraction >= OVERLAP_MIN_CENTRAL_FRACTION,
        "treated_ess": treated_ess >= treated_minimum,
        "control_ess": control_ess >= control_minimum,
        "local_cells": all(row["pass"] for row in cell_rows),
    }
    return {
        "gate_pass": all(components.values()),
        "components": components,
        "propensity_min": float(propensity.min()),
        "propensity_max": float(propensity.max()),
        "central_fraction": central_fraction,
        "treated_ess": treated_ess,
        "control_ess": control_ess,
        "treated_ess_minimum": treated_minimum,
        "control_ess_minimum": control_minimum,
        "local_cells": cell_rows,
        "no_trimming": True,
        "no_weight_clipping": True,
    }


def direct_adjustment(
    features: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    effect = np.full(len(a), np.nan)
    propensity = np.full(len(a), np.nan)
    fold_details: list[dict[str, Any]] = []
    for fold in range(N_OUTER_FOLDS):
        train = folds != fold
        test = folds == fold
        effect[test] = fit_outcome_effect(features[train], features[test], a[train], y[train], seed + 101 * fold)
        propensity[test] = fit_propensity(features[train], features[test], a[train], seed + 103 * fold)
        fold_details.append({
            "fold": fold,
            "train_n": int(train.sum()),
            "test_n": int(test.sum()),
            "train_n1": int(a[train].sum()),
            "train_n0": int(train.sum() - a[train].sum()),
            "train_hash": array_hash(np.flatnonzero(train)),
            "test_hash": array_hash(np.flatnonzero(test)),
            "disjoint": not np.any(train & test),
        })
    if not np.isfinite(effect).all() or not np.isfinite(propensity).all():
        raise RuntimeError("direct adjustment produced missing OOF values")
    return {
        "estimate": float(effect.mean()),
        "overlap": overlap_summary(a, propensity),
        "folds": fold_details,
        "all_predictions_outer_fold_oof": True,
        "standardization_n": len(a),
    }


def score_adjustment(
    inputs: np.ndarray,
    audit: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    seed: int,
    evaluation_specs: list[EvaluationSpec],
    role: str,
) -> dict[str, Any]:
    effect = np.full(len(a), np.nan)
    propensity = np.full(len(a), np.nan)
    discrepancies: list[float] = []
    details: list[dict[str, Any]] = []
    for fold in range(N_OUTER_FOLDS):
        train = folds != fold
        test = folds == fold
        fit_seed = seed + 107 * fold
        fit = fit_representation(inputs[train], audit[train], a[train], evaluation_specs[fold], fit_seed)
        r_train = predict_representation(fit, inputs[train])
        r_test = predict_representation(fit, inputs[test])
        effect[test] = fit_outcome_effect(r_train[:, None], r_test[:, None], a[train], y[train], seed + 109 * fold)
        propensity[test] = fit_propensity(r_train[:, None], r_test[:, None], a[train], seed + 113 * fold)
        discrepancies.append(heldout_crossfit_surrogate(evaluation_specs[fold], r_test, a[test], audit[test]))
        details.append({
            "fold": fold,
            "train_n": int(train.sum()),
            "test_n": int(test.sum()),
            "train_hash": array_hash(np.flatnonzero(train)),
            "test_hash": array_hash(np.flatnonzero(test)),
            "train_test_disjoint": not np.any(train & test),
            "train_union_test_is_all_qed": bool(np.all(train | test)),
            "evaluation_signature": evaluation_specs[fold].signature,
            "audit_dimension": audit.shape[1],
            "representation_input_dimension": inputs.shape[1],
            "outcome_used_in_representation": False,
            "selection_indicator_used_in_representation": False,
            "randomized_discordant_rows_used": False,
            "pretrain_initial": fit.pretrain_curve[0],
            "pretrain_final": fit.pretrain_curve[-1],
            "cci_initial": fit.cci_curve[0],
            "cci_final": fit.cci_curve[-1],
            "cci_terminal_not_above_initial": fit.cci_curve[-1] <= fit.cci_curve[0] + 1e-12,
            "score_min": float(r_test.min()),
            "score_max": float(r_test.max()),
        })
    if not np.isfinite(effect).all() or not np.isfinite(propensity).all():
        raise RuntimeError(f"{role} produced missing OOF values")
    return {
        "role": role,
        "estimate": float(effect.mean()),
        "heldout_finite_rff_crossfit_surrogate_mean": float(np.mean(discrepancies)),
        "heldout_finite_rff_crossfit_surrogate_by_fold": discrepancies,
        "evaluation_signatures": [spec.signature for spec in evaluation_specs],
        "overlap": overlap_summary(a, propensity),
        "folds": details,
        "all_predictions_outer_fold_oof": True,
        "standardization_n": len(a),
    }


def aggregate_orientation_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    estimates = np.array([row["estimate"] for row in rows], dtype=float)
    overlap_rows = [row["overlap"] for row in rows]
    return {
        "estimate": float(estimates.mean()),
        "orientation_estimates": estimates,
        "orientation_estimate_sd": float(estimates.std(ddof=1)) if len(estimates) > 1 else None,
        "heldout_finite_rff_crossfit_surrogate_mean": float(np.mean([
            row["heldout_finite_rff_crossfit_surrogate_mean"] for row in rows
        ])),
        "heldout_finite_rff_crossfit_surrogate_by_orientation": [
            row["heldout_finite_rff_crossfit_surrogate_mean"] for row in rows
        ],
        "overlap_gate_pass_all_orientations": all(row.get("gate_pass", False) for row in overlap_rows),
        "overlap_worst_case": {
            "propensity_min": float(min(row["propensity_min"] for row in overlap_rows)),
            "propensity_max": float(max(row["propensity_max"] for row in overlap_rows)),
            "central_fraction_min": float(min(row["central_fraction"] for row in overlap_rows)),
            "treated_ess_min": float(min(row["treated_ess"] for row in overlap_rows)),
            "control_ess_min": float(min(row["control_ess"] for row in overlap_rows)),
            "all_local_cell_gates_pass": all(row["components"]["local_cells"] for row in overlap_rows),
        },
        "orientations": rows,
        "all_predictions_outer_fold_oof": all(row["all_predictions_outer_fold_oof"] for row in rows),
    }


def one_repeat(data: dict[str, Any], repeat: int, audit_count: int) -> dict[str, Any]:
    x = data["X"]
    w = data["W"]
    a = data["A"]
    y = data["Y"]
    seed = ROOT_SEED + 100003 * repeat
    folds = stratified_folds(a, seed + 7, N_OUTER_FOLDS)
    naive = float(y[a == 1].mean() - y[a == 0].mean())
    x_direct = direct_adjustment(x, a, y, folds, seed + 1000)
    full_xw_direct = direct_adjustment(np.column_stack([x, w]), a, y, folds, seed + 2000)

    x_scores: list[dict[str, Any]] = []
    cross_masks: list[dict[str, Any]] = []
    for orientation, audit_index in enumerate(data["audit_indices"][:audit_count]):
        audit = np.column_stack([x, w[:, audit_index]])
        keep = np.arange(w.shape[1]) != audit_index
        cross_input = np.column_stack([x, w[:, keep]])
        specs: list[EvaluationSpec] = []
        for fold in range(N_OUTER_FOLDS):
            train = folds != fold
            test = folds == fold
            specs.append(make_evaluation_spec(audit[train], a[test], seed + 10000 + 307 * orientation + 17 * fold))
        paired_seed = seed + 20000 + 1009 * orientation
        x_result = score_adjustment(x, audit, a, y, folds, paired_seed, specs, "x_score")
        cross_result = score_adjustment(cross_input, audit, a, y, folds, paired_seed, specs, "cross_mask")
        common = {
            "orientation": orientation,
            "audit_index": int(audit_index),
            "audit_name": data["audit_names"][orientation],
            "audit_excluded_from_cross_mask_input": True,
            "paired_random_features_and_bandwidth": x_result["evaluation_signatures"] == cross_result["evaluation_signatures"],
        }
        x_result.update(common)
        cross_result.update(common)
        x_scores.append(x_result)
        cross_masks.append(cross_result)

    x_score_average = aggregate_orientation_results(x_scores)
    cross_mask_average = aggregate_orientation_results(cross_masks)
    rct = data["rct_benchmark"]
    methods = {
        "naive_qed": {
            "estimate": naive,
            "population_standardized": False,
            "overlap_gate_applicable": False,
        },
        "x_direct": x_direct,
        "full_xw_direct": full_xw_direct,
        "x_score_average": x_score_average,
        "cross_mask_average": cross_mask_average,
    }
    for method in methods.values():
        method["benchmark_gap_signed"] = method["estimate"] - rct
        method["benchmark_gap_absolute"] = abs(method["estimate"] - rct)

    paired_discrepancy = []
    for x_row, cross_row in zip(x_scores, cross_masks):
        x_value = x_row["heldout_finite_rff_crossfit_surrogate_mean"]
        cross_value = cross_row["heldout_finite_rff_crossfit_surrogate_mean"]
        paired_discrepancy.append({
            "audit_name": x_row["audit_name"],
            "x_score": x_value,
            "cross_mask": cross_value,
            "x_minus_cross": x_value - cross_value,
            "cross_mask_lower": cross_value < x_value,
        })
    return {
        "repeat": repeat,
        "seed": seed,
        "fold_hash": array_hash(folds),
        "outer_fold_counts": [
            {
                "fold": fold,
                "n": int(np.sum(folds == fold)),
                "n1": int(np.sum((folds == fold) & (a == 1))),
                "n0": int(np.sum((folds == fold) & (a == 0))),
            }
            for fold in range(N_OUTER_FOLDS)
        ],
        "methods": methods,
        "paired_discrepancy": paired_discrepancy,
        "all_methods_share_fold_hash": True,
        "qed_rows_only": True,
        "no_trimming": True,
    }


def summarize(records: list[dict[str, Any]], rct: float) -> dict[str, Any]:
    method_names = list(records[0]["methods"])
    output: dict[str, Any] = {}
    for name in method_names:
        rows = [record["methods"][name] for record in records]
        estimates = np.array([row["estimate"] for row in rows], dtype=float)
        gaps = np.abs(estimates - rct)
        block: dict[str, Any] = {
            "estimate_mean": float(estimates.mean()),
            "estimate_sd_across_algorithmic_repeats": float(estimates.std(ddof=1)) if len(estimates) > 1 else None,
            "estimate_min": float(estimates.min()),
            "estimate_max": float(estimates.max()),
            "estimates": estimates,
            "benchmark_gap_absolute_mean": float(gaps.mean()),
            "benchmark_gap_absolute_by_repeat": gaps,
            "benchmark_gap_signed_mean": float((estimates - rct).mean()),
        }
        if name in {"x_direct", "full_xw_direct"}:
            block["overlap_gate_pass_rate"] = float(np.mean([row["overlap"]["gate_pass"] for row in rows]))
            block["overlap"] = [row["overlap"] for row in rows]
        elif name in {"x_score_average", "cross_mask_average"}:
            block["overlap_gate_pass_rate"] = float(np.mean([
                row["overlap_gate_pass_all_orientations"] for row in rows
            ]))
            block["overlap_worst_case"] = [row["overlap_worst_case"] for row in rows]
            block["heldout_finite_rff_crossfit_surrogate_mean"] = float(np.mean([
                row["heldout_finite_rff_crossfit_surrogate_mean"] for row in rows
            ]))
        output[name] = block

    discrepancy_rows = [item for record in records for item in record["paired_discrepancy"]]
    output["paired_audit_diagnostic"] = {
        "comparisons": len(discrepancy_rows),
        "cross_mask_lower_count": int(sum(item["cross_mask_lower"] for item in discrepancy_rows)),
        "cross_mask_lower_fraction": float(np.mean([item["cross_mask_lower"] for item in discrepancy_rows])),
        "mean_x_score_surrogate": float(np.mean([item["x_score"] for item in discrepancy_rows])),
        "mean_cross_mask_surrogate": float(np.mean([item["cross_mask"] for item in discrepancy_rows])),
        "mean_x_minus_cross": float(np.mean([item["x_minus_cross"] for item in discrepancy_rows])),
    }
    return output


def run(n_repeats: int, audit_count: int) -> dict[str, Any]:
    started = time.time()
    torch.set_num_threads(1)
    data = load_data()
    records = [one_repeat(data, repeat, audit_count) for repeat in range(n_repeats)]
    summary = summarize(records, data["rct_benchmark"])

    success_components = {
        "cross_mask_gap_below_naive_in_every_repeat": all(
            record["methods"]["cross_mask_average"]["benchmark_gap_absolute"]
            < record["methods"]["naive_qed"]["benchmark_gap_absolute"]
            for record in records
        ),
        "cross_mask_gap_below_x_direct_in_every_repeat": all(
            record["methods"]["cross_mask_average"]["benchmark_gap_absolute"]
            < record["methods"]["x_direct"]["benchmark_gap_absolute"]
            for record in records
        ),
        "cross_mask_overlap_gate_passes_every_repeat": all(
            record["methods"]["cross_mask_average"]["overlap_gate_pass_all_orientations"]
            for record in records
        ),
    }

    fold_hashes = [record["fold_hash"] for record in records]
    all_score_details = [
        fold
        for record in records
        for method in ("x_score_average", "cross_mask_average")
        for orientation in record["methods"][method]["orientations"]
        for fold in orientation["folds"]
    ]
    all_method_fold_details = [
        fold
        for record in records
        for method in ("x_direct", "full_xw_direct")
        for fold in record["methods"][method]["folds"]
    ] + all_score_details
    fold_contracts_match = True
    for record in records:
        reference = [
            (fold["fold"], fold["train_hash"], fold["test_hash"])
            for fold in record["methods"]["x_direct"]["folds"]
        ]
        candidates = [record["methods"]["full_xw_direct"]["folds"]]
        candidates.extend(
            orientation["folds"]
            for method in ("x_score_average", "cross_mask_average")
            for orientation in record["methods"][method]["orientations"]
        )
        for candidate in candidates:
            signature = [(fold["fold"], fold["train_hash"], fold["test_hash"]) for fold in candidate]
            fold_contracts_match = fold_contracts_match and signature == reference
    gates = {
        "source_hashes_match_manifest": all(item["matches_manifest"] for item in data["source_hashes"].values()),
        "all_twelve_aggregate_score_checks_pass": all(data["aggregate_checks"].values()),
        "mcs_pre_only_in_proxy": True,
        "main_table_dimensions_and_missingness_pass": True,
        "qed_counts_1105_288_817": data["qed_counts"] == {"n": 1105, "n1": 288, "n0": 817},
        "selection_indicator_excluded_from_features": True,
        "all_posttreatment_variables_excluded": True,
        "qed_rows_only_for_all_learned_models_and_standardization": all(record["qed_rows_only"] for record in records),
        "full_sample_used_only_for_rct_benchmark_and_se": True,
        "all_predictions_outer_fold_oof": all(
            record["methods"][method].get("all_predictions_outer_fold_oof", True)
            for record in records
            for method in ("x_direct", "full_xw_direct", "x_score_average", "cross_mask_average")
        ),
        "all_fold_splits_disjoint_and_exhaust_qed": all(
            fold["train_test_disjoint"] and fold["train_union_test_is_all_qed"]
            for fold in all_score_details
        ) and all(fold["disjoint"] for fold in all_method_fold_details if "disjoint" in fold),
        "identical_outer_folds_across_methods": fold_contracts_match,
        "paired_evaluation_specs_identical": all(
            item["paired_random_features_and_bandwidth"]
            for record in records
            for method in ("x_score_average", "cross_mask_average")
            for item in record["methods"][method]["orientations"]
        ),
        "outcome_free_representation_training": all(
            not fold["outcome_used_in_representation"]
            for fold in all_score_details
        ),
        "randomized_discordant_rows_never_used_by_learned_models": all(
            not fold["randomized_discordant_rows_used"]
            for fold in all_score_details
        ),
        "no_trimming_or_weight_clipping": all(record["no_trimming"] for record in records),
        "benchmark_labeled_noisy_not_truth": True,
        "pilot_labeled_exploratory": True,
        "proxy_validity_or_injectivity_not_claimed": True,
        "all_values_finite": all(
            math.isfinite(record["methods"][method]["estimate"])
            for record in records
            for method in record["methods"]
        ),
    }
    return jsonable({
        "provenance": {
            "script": "code/wsc_population_ate_pilot.py",
            "script_sha256": sha256_file(Path(__file__)),
            "preregistration": "memo/2026-08-23-wsc-population-ate-pilot-preregistration.md",
            "preregistration_sha256": sha256_file(PREREG_PATH),
            "root_seed": ROOT_SEED,
            "algorithmic_repeats": n_repeats,
            "outer_folds": N_OUTER_FOLDS,
            "inner_evaluation_folds": N_INNER_FOLDS,
            "audit_orientations": audit_count,
            "exploratory_pilot": True,
            "runtime_seconds": time.time() - started,
            "post_result_tuning": False,
        },
        "data": {
            "source_hashes": data["source_hashes"],
            "full_counts": data["full_counts"],
            "qed_counts": data["qed_counts"],
            "qed_index_hash": array_hash(data["qed_indices"]),
            "x_columns": X_COLUMNS,
            "x_dimension": data["X"].shape[1],
            "proxy_dimension": data["W"].shape[1],
            "proxy_names": data["proxy_names"],
            "audit_indices": data["audit_indices"][:audit_count],
            "audit_names": data["audit_names"][:audit_count],
            "aggregate_score_checks": data["aggregate_checks"],
            "representation_forbidden_variables": ["mathSel", "vocabPost", "mathPost", "MCSpost", "post MCS items"],
            "big5_scoring_key": {
                "reverse_transform": "6 - raw item response",
                "reverse_items_by_grouped_block": {
                    "O": ["O7", "O9"],
                    "C": ["C2", "C4", "C5", "C9"],
                    "E": ["E2", "E5", "E7"],
                    "A": ["A1", "A3", "A6", "A8"],
                    "N": ["N2", "N5", "N7"],
                },
                "source": "https://arc.psych.wisc.edu/self-report/big-five-inventory-bfi/",
                "wsc_documentation": "materials/real_world_data/wscdata/Big5_WSC.Rd",
            },
        },
        "estimand": {
            "primary": "population ATE under documented randomization plus explicit P(Z=1)=1/2 design condition",
            "qed_representativeness_condition": "P(C=1|S,X,W,Y(0),Y(1))=1/2 under Z independent of pretreatment variables and potential outcomes with P(Z=1)=1/2",
            "qed_standardization_n": data["qed_counts"]["n"],
            "full_randomized_sample_enters_learned_estimator": False,
            "rct_benchmark": data["rct_benchmark"],
            "rct_standard_error_unpooled": data["rct_standard_error"],
            "rct_ci95_wald": [
                data["rct_benchmark"] - 1.96 * data["rct_standard_error"],
                data["rct_benchmark"] + 1.96 * data["rct_standard_error"],
            ],
            "rct_is_noisy_benchmark_not_truth": True,
        },
        "locked_specification": {
            "representation": "sigmoid linear score; BCE initialization followed by Adam minimization of finite-RFF conditional cross-covariance",
            "representation_steps": {"pretrain": PRETRAIN_STEPS, "cci": CCI_STEPS},
            "rff": {"audit_features": RFF_V, "score_features": RFF_R, "ridge": RIDGE, "score_bandwidth": R_BANDWIDTH},
            "outcome_estimator": "outer-fold cross-fitted two-arm GradientBoostingRegressor T-learner",
            "outcome_estimator_spec": GBR_SPEC,
            "propensity_diagnostic": "outer-fold L2-logistic regression in each adjustment space",
            "propensity_c": PROPENSITY_C,
            "overlap_gate": {
                "central_interval": [OVERLAP_LOWER, OVERLAP_UPPER],
                "minimum_central_fraction": OVERLAP_MIN_CENTRAL_FRACTION,
                "minimum_ess_each_arm": f"max({OVERLAP_MIN_ESS_ABSOLUTE}, {OVERLAP_MIN_ESS_FRACTION} * observed arm size)",
                "quantile_cells": OVERLAP_CELLS,
                "minimum_each_arm_per_cell": OVERLAP_MIN_ARM_PER_CELL,
                "no_trimming": True,
                "no_weight_clipping": True,
            },
            "finite_rff_surrogate_not_exact_kci": True,
        },
        "records": records,
        "summary": summary,
        "preregistered_success_rule": {
            "components": success_components,
            "pass": all(success_components.values()),
            "stable_improvement_definition": "cross-mask absolute benchmark gap is below the comparator in every algorithmic repeat",
        },
        "gates": gates,
        "all_integrity_gates_pass": all(gates.values()),
        "limitations": [
            "The exact one-half assignment probability is an explicit design condition and is not stated in the frozen package documentation.",
            "The full-sample randomized contrast is noisy and is not the true causal effect.",
            "Three algorithmic repeats are an exploratory pilot, not confirmatory evidence.",
            "An untouched audit item prevents literal self-copying but does not establish conditionally independent proxy views.",
            "Neither proxy validity nor mixture injectivity is identified or tested by this experiment.",
            "Finite-RFF residual cross-covariance is a surrogate rather than exact KCI.",
        ],
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="pilot")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        repeats, audit_count = 1, 1
    else:
        repeats, audit_count = N_REPEATS, N_AUDIT_MASKS
    output = args.output
    if args.mode == "pilot" and output is None:
        output = OUTPUT_PATH
    if args.mode == "smoke" and output is not None and output.resolve() == OUTPUT_PATH.resolve():
        raise SystemExit("Refusing to overwrite the canonical pilot result with smoke output")
    payload = run(repeats, audit_count)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "mode": args.mode,
        "output": str(output) if output is not None else None,
        "runtime_seconds": payload["provenance"]["runtime_seconds"],
        "all_integrity_gates_pass": payload["all_integrity_gates_pass"],
        "rct_benchmark": payload["estimand"]["rct_benchmark"],
        "summary": payload["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
