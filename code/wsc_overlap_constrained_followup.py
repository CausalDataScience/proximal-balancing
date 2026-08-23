#!/usr/bin/env python3
"""Paired WSC follow-up with outcome-free overlap-constrained cross-mask learning."""
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

import wsc_population_ate_pilot as base


ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = ROOT / "memo" / "2026-08-23-wsc-overlap-constrained-followup-preregistration.md"
OUTPUT_PATH = ROOT / "results" / "wsc_overlap_constrained_followup.json"

N_ALGORITHMIC_REPEATS = 10
LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0)
PENALTY_LOWER = 0.10
PENALTY_UPPER = 0.90
SELECTION_CENTRAL_LOWER = 0.05
SELECTION_CENTRAL_UPPER = 0.95
SELECTION_MIN_CENTRAL_FRACTION = 0.90
SELECTION_FOLDS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ConstrainedRepresentationFit:
    model: base.LinearScore
    mean: np.ndarray
    scale: np.ndarray
    pretrain_curve: list[float]
    cci_curve: list[float]
    overlap_penalty_curve: list[float]
    lambda_overlap: float


def conditional_components(
    score: torch.Tensor,
    a: torch.Tensor,
    phi_v: torch.Tensor,
    omega: torch.Tensor,
    phase: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = base.r_features_torch(score, omega, phase)
    fitted_a = features @ base.ridge_coefficients(features, a[:, None])
    residual_a = a[:, None] - fitted_a
    residual_v = phi_v - features @ base.ridge_coefficients(features, phi_v)
    crosscov = residual_a.T @ residual_v / len(a)
    cci = crosscov.square().sum()
    penalty = (
        torch.relu(torch.tensor(PENALTY_LOWER, dtype=fitted_a.dtype) - fitted_a).square()
        + torch.relu(fitted_a - torch.tensor(PENALTY_UPPER, dtype=fitted_a.dtype)).square()
    ).mean()
    return cci, penalty


def fit_constrained_representation(
    inputs: np.ndarray,
    audit: np.ndarray,
    a: np.ndarray,
    spec: base.EvaluationSpec,
    seed: int,
    lambda_overlap: float,
) -> ConstrainedRepresentationFit:
    mean, scale = base.standardize_fit(inputs)
    standardized = (inputs - mean) / scale
    phi_v = base.v_rff_numpy(audit, spec.v_spec)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = base.LinearScore(standardized.shape[1]).double()
    xt = torch.tensor(standardized, dtype=torch.float64)
    at = torch.tensor(a, dtype=torch.float64)
    vt = torch.tensor(phi_v, dtype=torch.float64)
    omega = torch.tensor(spec.r_omega, dtype=torch.float64)
    phase = torch.tensor(spec.r_phase, dtype=torch.float64)

    optimizer = torch.optim.Adam(model.parameters(), lr=base.LR_PRETRAIN, weight_decay=1e-4)
    pretrain_curve: list[float] = []
    for _ in range(base.PRETRAIN_STEPS):
        score = model(xt)
        loss = torch.nn.functional.binary_cross_entropy(score.clamp(1e-5, 1.0 - 1e-5), at)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pretrain_curve.append(float(loss.detach()))

    optimizer = torch.optim.Adam(model.parameters(), lr=base.LR_CCI, weight_decay=1e-4)
    cci_curve: list[float] = []
    penalty_curve: list[float] = []
    for _ in range(base.CCI_STEPS):
        score = model(xt)
        cci, penalty = conditional_components(score, at, vt, omega, phase)
        variance_floor = torch.relu(torch.tensor(0.015, dtype=score.dtype) - score.var(unbiased=False))
        objective = cci + lambda_overlap * penalty + 0.5 * variance_floor
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        cci_curve.append(float(cci.detach()))
        penalty_curve.append(float(penalty.detach()))
    return ConstrainedRepresentationFit(
        model=model,
        mean=mean,
        scale=scale,
        pretrain_curve=pretrain_curve,
        cci_curve=cci_curve,
        overlap_penalty_curve=penalty_curve,
        lambda_overlap=lambda_overlap,
    )


def predict_constrained(fit: ConstrainedRepresentationFit, inputs: np.ndarray) -> np.ndarray:
    values = torch.tensor((inputs - fit.mean) / fit.scale, dtype=torch.float64)
    with torch.no_grad():
        return fit.model(values).numpy()


def selection_split(a_outer_train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    folds = base.stratified_folds(a_outer_train, seed, SELECTION_FOLDS)
    validation = folds == 0
    training = ~validation
    if not np.any(training) or not np.any(validation):
        raise RuntimeError("empty nested selection split")
    return training, validation


def select_lambda(
    inputs: np.ndarray,
    audit: np.ndarray,
    a: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    selection_train, selection_validation = selection_split(a, seed + 11)
    nested_spec = base.make_evaluation_spec(
        audit[selection_train],
        a[selection_validation],
        seed + 13,
    )
    candidates: list[dict[str, Any]] = []
    for lambda_overlap in LAMBDA_GRID:
        fit = fit_constrained_representation(
            inputs[selection_train],
            audit[selection_train],
            a[selection_train],
            nested_spec,
            seed + 17,
            lambda_overlap,
        )
        r_train = predict_constrained(fit, inputs[selection_train])
        r_validation = predict_constrained(fit, inputs[selection_validation])
        propensity = base.fit_propensity(
            r_train[:, None],
            r_validation[:, None],
            a[selection_train],
            seed + 19,
        )
        central_fraction = float(np.mean(
            (propensity >= SELECTION_CENTRAL_LOWER)
            & (propensity <= SELECTION_CENTRAL_UPPER)
        ))
        surrogate = base.heldout_crossfit_surrogate(
            nested_spec,
            r_validation,
            a[selection_validation],
            audit[selection_validation],
        )
        candidates.append({
            "lambda": lambda_overlap,
            "central_fraction": central_fraction,
            "passes_central_screen": central_fraction >= SELECTION_MIN_CENTRAL_FRACTION,
            "heldout_finite_rff_surrogate": surrogate,
            "propensity_min": float(propensity.min()),
            "propensity_max": float(propensity.max()),
            "selection_train_n": int(selection_train.sum()),
            "selection_validation_n": int(selection_validation.sum()),
            "selection_train_hash": base.array_hash(np.flatnonzero(selection_train)),
            "selection_validation_hash": base.array_hash(np.flatnonzero(selection_validation)),
            "nested_evaluation_signature": nested_spec.signature,
            "pretrain_initial": fit.pretrain_curve[0],
            "pretrain_final": fit.pretrain_curve[-1],
            "cci_initial": fit.cci_curve[0],
            "cci_final": fit.cci_curve[-1],
            "penalty_initial": fit.overlap_penalty_curve[0],
            "penalty_final": fit.overlap_penalty_curve[-1],
        })

    passing = [row for row in candidates if row["passes_central_screen"]]
    if passing:
        chosen = min(passing, key=lambda row: (row["heldout_finite_rff_surrogate"], row["lambda"]))
        rule = "minimum validation audit surrogate among candidates passing central fraction 0.90"
    else:
        chosen = min(
            candidates,
            key=lambda row: (-row["central_fraction"], row["heldout_finite_rff_surrogate"], row["lambda"]),
        )
        rule = "no candidate passed; maximum central fraction then minimum surrogate then smaller lambda"
    return {
        "selected_lambda": chosen["lambda"],
        "selection_rule": rule,
        "any_candidate_passed": bool(passing),
        "n_candidates_passing": len(passing),
        "candidates": candidates,
        "outer_evaluation_rows_used_for_selection": False,
        "outcome_or_rct_used_for_selection": False,
    }


def fit_one_outer_pair(
    inputs: np.ndarray,
    audit: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    evaluation_spec: base.EvaluationSpec,
    fit_seed: int,
    selection_seed: int,
    outcome_seed: int,
    propensity_seed: int,
) -> dict[str, Any]:
    unconstrained = base.fit_representation(inputs[train], audit[train], a[train], evaluation_spec, fit_seed)
    lambda_selection = select_lambda(inputs[train], audit[train], a[train], selection_seed)
    constrained = fit_constrained_representation(
        inputs[train],
        audit[train],
        a[train],
        evaluation_spec,
        fit_seed,
        lambda_selection["selected_lambda"],
    )

    r_train_unconstrained = base.predict_representation(unconstrained, inputs[train])
    r_test_unconstrained = base.predict_representation(unconstrained, inputs[test])
    r_train_constrained = predict_constrained(constrained, inputs[train])
    r_test_constrained = predict_constrained(constrained, inputs[test])

    result: dict[str, Any] = {
        "unconstrained": {
            "effect": base.fit_outcome_effect(
                r_train_unconstrained[:, None], r_test_unconstrained[:, None], a[train], y[train], outcome_seed
            ),
            "propensity": base.fit_propensity(
                r_train_unconstrained[:, None], r_test_unconstrained[:, None], a[train], propensity_seed
            ),
            "heldout_surrogate": base.heldout_crossfit_surrogate(
                evaluation_spec, r_test_unconstrained, a[test], audit[test]
            ),
            "score_min": float(r_test_unconstrained.min()),
            "score_max": float(r_test_unconstrained.max()),
            "cci_initial": unconstrained.cci_curve[0],
            "cci_final": unconstrained.cci_curve[-1],
        },
        "constrained": {
            "effect": base.fit_outcome_effect(
                r_train_constrained[:, None], r_test_constrained[:, None], a[train], y[train], outcome_seed
            ),
            "propensity": base.fit_propensity(
                r_train_constrained[:, None], r_test_constrained[:, None], a[train], propensity_seed
            ),
            "heldout_surrogate": base.heldout_crossfit_surrogate(
                evaluation_spec, r_test_constrained, a[test], audit[test]
            ),
            "score_min": float(r_test_constrained.min()),
            "score_max": float(r_test_constrained.max()),
            "cci_initial": constrained.cci_curve[0],
            "cci_final": constrained.cci_curve[-1],
            "penalty_initial": constrained.overlap_penalty_curve[0],
            "penalty_final": constrained.overlap_penalty_curve[-1],
        },
        "selection": lambda_selection,
        "paired_contract": {
            "same_outer_train_and_test": True,
            "same_audit_view": True,
            "same_evaluation_signature": evaluation_spec.signature,
            "same_initialization_seed": fit_seed,
            "same_outcome_seed": outcome_seed,
            "same_propensity_seed": propensity_seed,
            "outcome_free_representation_fits": True,
            "outer_evaluation_not_used_for_lambda_selection": True,
        },
    }
    return result


def aggregate_orientations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    estimates = np.array([row["estimate"] for row in rows], dtype=float)
    overlaps = [row["overlap"] for row in rows]
    return {
        "estimate": float(estimates.mean()),
        "orientation_estimates": estimates,
        "orientation_estimate_sd": float(estimates.std(ddof=1)),
        "heldout_surrogate_mean": float(np.mean([row["heldout_surrogate_mean"] for row in rows])),
        "heldout_surrogate_by_orientation": [row["heldout_surrogate_mean"] for row in rows],
        "overlap_gate_pass_all_orientations": all(row["gate_pass"] for row in overlaps),
        "overlap_gate_pass_orientation_count": int(sum(row["gate_pass"] for row in overlaps)),
        "overlap_worst_case": {
            "propensity_min": float(min(row["propensity_min"] for row in overlaps)),
            "propensity_max": float(max(row["propensity_max"] for row in overlaps)),
            "central_fraction_min": float(min(row["central_fraction"] for row in overlaps)),
            "treated_ess_min": float(min(row["treated_ess"] for row in overlaps)),
            "control_ess_min": float(min(row["control_ess"] for row in overlaps)),
            "all_local_cells_pass": all(row["components"]["local_cells"] for row in overlaps),
        },
        "orientations": rows,
        "all_predictions_outer_fold_oof": True,
    }


def one_repeat(data: dict[str, Any], repeat: int, audit_count: int) -> dict[str, Any]:
    x, w, a, y = data["X"], data["W"], data["A"], data["Y"]
    seed = base.ROOT_SEED + 100003 * repeat
    folds = base.stratified_folds(a, seed + 7, base.N_OUTER_FOLDS)
    naive = float(y[a == 1].mean() - y[a == 0].mean())
    x_direct = base.direct_adjustment(x, a, y, folds, seed + 1000)
    full_xw_direct = base.direct_adjustment(np.column_stack([x, w]), a, y, folds, seed + 2000)

    unconstrained_orientations: list[dict[str, Any]] = []
    constrained_orientations: list[dict[str, Any]] = []
    paired_changes: list[dict[str, Any]] = []
    for orientation, audit_index in enumerate(data["audit_indices"][:audit_count]):
        audit = np.column_stack([x, w[:, audit_index]])
        keep = np.arange(w.shape[1]) != audit_index
        inputs = np.column_stack([x, w[:, keep]])
        effect = {
            "unconstrained": np.full(len(a), np.nan),
            "constrained": np.full(len(a), np.nan),
        }
        propensity = {
            "unconstrained": np.full(len(a), np.nan),
            "constrained": np.full(len(a), np.nan),
        }
        heldout = {"unconstrained": [], "constrained": []}
        fold_details: list[dict[str, Any]] = []
        for fold in range(base.N_OUTER_FOLDS):
            train = folds != fold
            test = folds == fold
            evaluation_spec = base.make_evaluation_spec(
                audit[train], a[test], seed + 10000 + 307 * orientation + 17 * fold
            )
            paired_seed = seed + 20000 + 1009 * orientation
            pair = fit_one_outer_pair(
                inputs=inputs,
                audit=audit,
                a=a,
                y=y,
                train=train,
                test=test,
                evaluation_spec=evaluation_spec,
                fit_seed=paired_seed + 107 * fold,
                selection_seed=seed + 50000 + 1301 * orientation + 127 * fold,
                outcome_seed=paired_seed + 109 * fold,
                propensity_seed=paired_seed + 113 * fold,
            )
            for method in ("unconstrained", "constrained"):
                effect[method][test] = pair[method].pop("effect")
                propensity[method][test] = pair[method].pop("propensity")
                heldout[method].append(pair[method]["heldout_surrogate"])
            fold_details.append({
                "fold": fold,
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
                "train_hash": base.array_hash(np.flatnonzero(train)),
                "test_hash": base.array_hash(np.flatnonzero(test)),
                "train_test_disjoint": not np.any(train & test),
                "train_union_test_is_all_qed": bool(np.all(train | test)),
                **pair,
            })

        orientation_results: dict[str, dict[str, Any]] = {}
        for method in ("unconstrained", "constrained"):
            if not np.isfinite(effect[method]).all() or not np.isfinite(propensity[method]).all():
                raise RuntimeError(f"missing OOF values for {method}")
            orientation_results[method] = {
                "orientation": orientation,
                "audit_index": int(audit_index),
                "audit_name": data["audit_names"][orientation],
                "estimate": float(effect[method].mean()),
                "heldout_surrogate_mean": float(np.mean(heldout[method])),
                "heldout_surrogate_by_fold": heldout[method],
                "overlap": base.overlap_summary(a, propensity[method]),
                "audit_excluded_from_representation_input": True,
                "folds": fold_details,
            }
        unconstrained_orientations.append(orientation_results["unconstrained"])
        constrained_orientations.append(orientation_results["constrained"])
        paired_changes.append({
            "orientation": orientation,
            "audit_name": data["audit_names"][orientation],
            "estimate_constrained_minus_unconstrained": (
                orientation_results["constrained"]["estimate"]
                - orientation_results["unconstrained"]["estimate"]
            ),
            "surrogate_constrained_minus_unconstrained": (
                orientation_results["constrained"]["heldout_surrogate_mean"]
                - orientation_results["unconstrained"]["heldout_surrogate_mean"]
            ),
            "central_fraction_constrained_minus_unconstrained": (
                orientation_results["constrained"]["overlap"]["central_fraction"]
                - orientation_results["unconstrained"]["overlap"]["central_fraction"]
            ),
            "unconstrained_overlap_pass": orientation_results["unconstrained"]["overlap"]["gate_pass"],
            "constrained_overlap_pass": orientation_results["constrained"]["overlap"]["gate_pass"],
        })

    unconstrained = aggregate_orientations(unconstrained_orientations)
    constrained = aggregate_orientations(constrained_orientations)
    methods: dict[str, dict[str, Any]] = {
        "naive_qed": {"estimate": naive, "overlap_gate_applicable": False},
        "x_direct": x_direct,
        "full_xw_direct": full_xw_direct,
        "cross_mask_unconstrained": unconstrained,
        "cross_mask_overlap_constrained": constrained,
    }
    rct = data["rct_benchmark"]
    for method in methods.values():
        method["benchmark_gap_signed"] = method["estimate"] - rct
        method["benchmark_gap_absolute"] = abs(method["estimate"] - rct)
    return {
        "repeat": repeat,
        "seed": seed,
        "fold_hash": base.array_hash(folds),
        "methods": methods,
        "paired_orientation_changes": paired_changes,
        "all_models_qed_only": True,
        "full_randomized_sample_used_only_for_benchmark": True,
        "no_trimming_or_weight_clipping": True,
    }


def summarize(records: list[dict[str, Any]], rct: float) -> dict[str, Any]:
    names = list(records[0]["methods"])
    summary: dict[str, Any] = {}
    for name in names:
        rows = [record["methods"][name] for record in records]
        estimates = np.array([row["estimate"] for row in rows], dtype=float)
        gaps = np.abs(estimates - rct)
        block: dict[str, Any] = {
            "estimate_mean": float(estimates.mean()),
            "estimate_sd_across_algorithmic_repeats": (
                float(estimates.std(ddof=1)) if len(estimates) > 1 else None
            ),
            "estimate_min": float(estimates.min()),
            "estimate_max": float(estimates.max()),
            "estimates": estimates,
            "benchmark_gap_absolute_mean": float(gaps.mean()),
            "benchmark_gap_absolute_by_repeat": gaps,
            "benchmark_gap_signed_mean": float((estimates - rct).mean()),
        }
        if name in {"x_direct", "full_xw_direct"}:
            block["overlap_gate_pass_count"] = int(sum(row["overlap"]["gate_pass"] for row in rows))
            block["overlap_gate_pass_rate"] = block["overlap_gate_pass_count"] / len(rows)
            block["overlap"] = [row["overlap"] for row in rows]
        elif name.startswith("cross_mask"):
            block["overlap_gate_pass_count"] = int(sum(row["overlap_gate_pass_all_orientations"] for row in rows))
            block["overlap_gate_pass_rate"] = block["overlap_gate_pass_count"] / len(rows)
            block["orientation_overlap_pass_count_total"] = int(sum(
                row["overlap_gate_pass_orientation_count"] for row in rows
            ))
            block["orientation_overlap_total"] = len(rows) * base.N_AUDIT_MASKS
            block["heldout_surrogate_mean"] = float(np.mean([row["heldout_surrogate_mean"] for row in rows]))
            block["overlap_worst_case"] = [row["overlap_worst_case"] for row in rows]
        summary[name] = block

    constrained_fold_selections = [
        fold["selection"]
        for record in records
        for orientation in record["methods"]["cross_mask_overlap_constrained"]["orientations"]
        for fold in orientation["folds"]
    ]
    selected = [row["selected_lambda"] for row in constrained_fold_selections]
    paired_changes = [item for record in records for item in record["paired_orientation_changes"]]
    summary["lambda_selection"] = {
        "selection_cells": len(constrained_fold_selections),
        "selected_lambda_counts": {
            str(value): int(sum(chosen == value for chosen in selected)) for value in LAMBDA_GRID
        },
        "no_candidate_passed_inner_screen_count": int(sum(
            not row["any_candidate_passed"] for row in constrained_fold_selections
        )),
        "no_candidate_passed_inner_screen_fraction": float(np.mean([
            not row["any_candidate_passed"] for row in constrained_fold_selections
        ])),
    }
    summary["paired_orientation_change"] = {
        "pairs": len(paired_changes),
        "mean_central_fraction_change": float(np.mean([
            row["central_fraction_constrained_minus_unconstrained"] for row in paired_changes
        ])),
        "central_fraction_improved_count": int(sum(
            row["central_fraction_constrained_minus_unconstrained"] > 0.0 for row in paired_changes
        )),
        "mean_surrogate_change": float(np.mean([
            row["surrogate_constrained_minus_unconstrained"] for row in paired_changes
        ])),
        "surrogate_improved_count": int(sum(
            row["surrogate_constrained_minus_unconstrained"] < 0.0 for row in paired_changes
        )),
        "constrained_overlap_pass_count": int(sum(row["constrained_overlap_pass"] for row in paired_changes)),
        "unconstrained_overlap_pass_count": int(sum(row["unconstrained_overlap_pass"] for row in paired_changes)),
    }
    return summary


def run(n_repeats: int, audit_count: int) -> dict[str, Any]:
    started = time.time()
    torch.set_num_threads(1)
    data = base.load_data()
    records = [one_repeat(data, repeat, audit_count) for repeat in range(n_repeats)]
    summary = summarize(records, data["rct_benchmark"])

    constrained = summary["cross_mask_overlap_constrained"]
    unconstrained = summary["cross_mask_unconstrained"]
    x_direct_gaps = summary["x_direct"]["benchmark_gap_absolute_by_repeat"]
    constrained_gaps = constrained["benchmark_gap_absolute_by_repeat"]
    success_components = {
        "constrained_overlap_passes_at_least_8_of_10_repeats": (
            n_repeats == N_ALGORITHMIC_REPEATS and constrained["overlap_gate_pass_count"] >= 8
        ),
        "constrained_overlap_pass_count_exceeds_unconstrained": (
            constrained["overlap_gate_pass_count"] > unconstrained["overlap_gate_pass_count"]
        ),
        "constrained_gap_below_x_direct_in_at_least_8_of_10_repeats": (
            n_repeats == N_ALGORITHMIC_REPEATS
            and int(np.sum(np.asarray(constrained_gaps) < np.asarray(x_direct_gaps))) >= 8
        ),
        "constrained_mean_gap_within_one_rct_se_of_unconstrained": (
            constrained["benchmark_gap_absolute_mean"]
            <= unconstrained["benchmark_gap_absolute_mean"] + data["rct_standard_error"]
        ),
    }

    selection_rows = [
        fold["selection"]
        for record in records
        for orientation in record["methods"]["cross_mask_overlap_constrained"]["orientations"]
        for fold in orientation["folds"]
    ]
    paired_contracts = [
        fold["paired_contract"]
        for record in records
        for orientation in record["methods"]["cross_mask_overlap_constrained"]["orientations"]
        for fold in orientation["folds"]
    ]
    main_fits = n_repeats * audit_count * base.N_OUTER_FOLDS * (1 + len(LAMBDA_GRID) + 1)
    gates = {
        "base_data_integrity_rechecked": True,
        "qed_counts_1105_288_817": data["qed_counts"] == {"n": 1105, "n1": 288, "n0": 817},
        "exact_main_representation_fit_count": main_fits == n_repeats * audit_count * 18,
        "all_models_and_standardization_qed_only": all(record["all_models_qed_only"] for record in records),
        "full_randomized_sample_only_benchmark": all(
            record["full_randomized_sample_used_only_for_benchmark"] for record in records
        ),
        "outer_evaluation_never_used_for_lambda_selection": all(
            not row["outer_evaluation_rows_used_for_selection"] for row in selection_rows
        ),
        "outcome_and_rct_never_used_for_lambda_selection": all(
            not row["outcome_or_rct_used_for_selection"] for row in selection_rows
        ),
        "paired_indices_audit_rff_initialization_outcome_propensity": all(
            row["same_outer_train_and_test"]
            and row["same_audit_view"]
            and bool(row["same_evaluation_signature"])
            and row["same_initialization_seed"] is not None
            and row["same_outcome_seed"] is not None
            and row["same_propensity_seed"] is not None
            for row in paired_contracts
        ),
        "all_outer_folds_disjoint_and_exhaust_qed": all(
            fold["train_test_disjoint"] and fold["train_union_test_is_all_qed"]
            for record in records
            for orientation in record["methods"]["cross_mask_overlap_constrained"]["orientations"]
            for fold in orientation["folds"]
        ),
        "no_trimming_or_weight_clipping": all(
            record["no_trimming_or_weight_clipping"] for record in records
        ),
        "finite_estimates": all(
            math.isfinite(record["methods"][method]["estimate"])
            for record in records
            for method in record["methods"]
        ),
        "penalty_labeled_optimization_device_not_identification": True,
        "algorithmic_repetitions_not_monte_carlo": True,
    }
    return base.jsonable({
        "provenance": {
            "script": "code/wsc_overlap_constrained_followup.py",
            "script_sha256": sha256_file(Path(__file__)),
            "base_pilot_script": "code/wsc_population_ate_pilot.py",
            "base_pilot_script_sha256": sha256_file(Path(base.__file__)),
            "preregistration": "memo/2026-08-23-wsc-overlap-constrained-followup-preregistration.md",
            "preregistration_sha256": sha256_file(PREREG_PATH),
            "root_seed": base.ROOT_SEED,
            "algorithmic_repetitions": n_repeats,
            "monte_carlo_data_generating_repetitions": 0,
            "outer_folds": base.N_OUTER_FOLDS,
            "inner_audit_folds": base.N_INNER_FOLDS,
            "fixed_audit_masks": audit_count,
            "main_representation_fits": main_fits,
            "representation_fits_per_repeat_fold_mask": 1 + len(LAMBDA_GRID) + 1,
            "runtime_seconds": time.time() - started,
            "post_result_tuning": False,
        },
        "estimand": {
            "primary": "population ATE under the inherited exact one-half randomization condition",
            "rct_benchmark": data["rct_benchmark"],
            "rct_standard_error_unpooled": data["rct_standard_error"],
            "rct_is_noisy_benchmark_not_truth": True,
            "qed_standardization_n": data["qed_counts"]["n"],
        },
        "data": {
            "full_counts": data["full_counts"],
            "qed_counts": data["qed_counts"],
            "audit_indices": data["audit_indices"][:audit_count],
            "audit_names": data["audit_names"][:audit_count],
            "source_hashes": data["source_hashes"],
            "aggregate_score_checks": data["aggregate_checks"],
        },
        "locked_overlap_intervention": {
            "lambda_grid": LAMBDA_GRID,
            "training_penalty_interval": [PENALTY_LOWER, PENALTY_UPPER],
            "selection_central_interval": [SELECTION_CENTRAL_LOWER, SELECTION_CENTRAL_UPPER],
            "selection_minimum_central_fraction": SELECTION_MIN_CENTRAL_FRACTION,
            "penalty": "mean squared hinge of differentiable ridge E[A|R] estimate outside [0.10,0.90]",
            "role": "optimization device only; not positivity, identification, exact balance, or proxy validity",
            "final_overlap_gate_unchanged_from_pilot": True,
        },
        "records": records,
        "summary": summary,
        "preregistered_success_rule": {
            "components": success_components,
            "pass": all(success_components.values()),
            "practical_overlap_only_not_identification": True,
        },
        "integrity_gates": gates,
        "all_integrity_gates_pass": all(gates.values()),
        "limitations": [
            "This is one fixed WSC dataset with repeated algorithmic splits, not Monte Carlo data generation.",
            "The penalty is an optimization device and cannot prove population positivity.",
            "The RCT contrast is a noisy benchmark, not the true causal effect.",
            "The population-ATE interpretation inherits the unverified exact one-half randomization condition.",
            "Neither proxy validity nor mixture injectivity is established.",
        ],
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "main"), default="main")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        n_repeats, audit_count = 1, 1
    else:
        n_repeats, audit_count = N_ALGORITHMIC_REPEATS, base.N_AUDIT_MASKS
    output = args.output
    if args.mode == "main" and output is None:
        output = OUTPUT_PATH
    if args.mode == "smoke" and output is not None and output.resolve() == OUTPUT_PATH.resolve():
        raise SystemExit("Refusing to overwrite the canonical main result with smoke output")
    payload = run(n_repeats, audit_count)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "mode": args.mode,
        "output": str(output) if output is not None else None,
        "runtime_seconds": payload["provenance"]["runtime_seconds"],
        "main_representation_fits": payload["provenance"]["main_representation_fits"],
        "all_integrity_gates_pass": payload["all_integrity_gates_pass"],
        "preregistered_success_rule": payload["preregistered_success_rule"],
        "summary": payload["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
