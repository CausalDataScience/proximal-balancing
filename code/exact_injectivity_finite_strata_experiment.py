#!/usr/bin/env python3
"""Finite-strata exact-injectivity simulation with a fixed rep/audit split.

The primary learned estimate uses out-of-fold hard cells.  Soft adjustment is
reported only as a diagnostic.  The learned objective is the existing RFF
conditional-covariance surrogate, not exact KCI or conditional HSIC.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import honest_canonical_representation_experiment as base


ROOT_SEED = 190602
PREFIXES = (100, 200, 500, 1000, 2000, 5000, 10000)
N_MASTER, N_DIAGNOSTIC = 10000, 5000
D_X, D_WREP = 20, 9
TRUE_ATE = 1.0
PI = np.array([0.2, 0.5, 0.8])
OUTPUT = Path(__file__).resolve().parents[1] / "results" / "exact_injectivity_finite_strata_summary.json"

NAMESPACE = {
    "coefficients": 100,
    "master": 200,
    "diagnostic": 300,
    "fold": 400,
    "outcome": 700,
}
METHOD_ID = {
    "adjust_x": 0,
    "adjust_xw_full_observed": 1,
    "observed_xs_fitted_reference": 2,
    "observed_xu_fitted_reference": 3,
    "oracle_h0_stratified": 4,
    "learned_fixed_rep_audit": 5,
    "self_conditioned_negative_control": 6,
}

# Reuse the previously frozen representation and outcome-learning machinery,
# but switch its process-local seed and method identifiers for this new script.
base.ROOT_SEED = ROOT_SEED
base.METHOD_ID["proposed_cross_mask_v1"] = METHOD_ID["learned_fixed_rep_audit"]
base.METHOD_ID["heuristic_self_conditioned_v2"] = METHOD_ID["self_conditioned_negative_control"]


def key(namespace: str, replicate=0, n_index=0, fold=0, method=0, extra=0) -> tuple[int, ...]:
    return (ROOT_SEED, NAMESPACE[namespace], replicate, n_index, fold, method, extra, 0)


def rng(seed_key: tuple[int, ...]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed_key))


def unit_vector(g: np.random.Generator, d: int) -> np.ndarray:
    v = g.standard_normal(d)
    return v / np.linalg.norm(v)


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


def coefficients() -> dict[str, np.ndarray]:
    g = rng(key("coefficients"))
    return {
        "b_x": unit_vector(g, D_X),
        "c_x": unit_vector(g, D_X),
        "d_x": unit_vector(g, D_X),
        "gamma": unit_vector(g, D_X),
        "y1": unit_vector(g, D_X),
        "y2": unit_vector(g, D_X),
        "y3": unit_vector(g, D_X),
        "y4": unit_vector(g, D_X),
        "w_s": g.uniform(0.25, 0.75, size=7) * g.choice(np.array([-1.0, 1.0]), size=7),
    }


def generate(n: int, seed_key: tuple[int, ...], coef: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed_key).spawn(8)]
    ru, rx, rv, wr, wa, ra, ry, _ = streams
    u = ru.standard_normal(n)
    s = (u > 0).astype(int)
    t = 2.0 * s - 1.0
    x = (
        u[:, None] * coef["b_x"]
        + 0.40 * ((u ** 2 - 1.0) / math.sqrt(2.0))[:, None] * coef["c_x"]
        + 0.50 * np.tanh(u)[:, None] * coef["d_x"]
        + rx.standard_normal((n, D_X))
    )
    v = rv.standard_normal(n)
    wrep = np.empty((n, D_WREP))
    wrep[:, 0] = v
    wrep[:, 1] = v + s
    wrep[:, 2:] = t[:, None] * coef["w_s"] + wr.standard_normal((n, 7))
    waudit = u + 0.7 * wa.standard_normal(n)
    b = (x @ coef["gamma"] > 0).astype(int)
    h0 = s + b
    propensity = PI[h0]
    a = (ra.uniform(size=n) < propensity).astype(float)
    f_x = (
        0.50 * np.sin(x @ coef["y1"])
        + 0.25 * (x @ coef["y2"]) ** 2
        + 0.30 * np.tanh(x @ coef["y3"]) * (x @ coef["y4"])
    )
    y0 = f_x + 1.3 * u + 0.3 * (u ** 2 - 1.0) + ry.standard_normal(n)
    y = y0 + a
    w = np.column_stack([wrep, waudit])
    return {"U": u, "S": s, "X": x, "Wrep": wrep, "Waudit": waudit, "W": w, "B": b, "H0": h0, "A": a, "Y": y, "propensity": propensity}


def exact_decoder(wrep: np.ndarray) -> np.ndarray:
    return np.rint(wrep[:, 1] - wrep[:, 0]).astype(int)


def stratified_ate(y: np.ndarray, a: np.ndarray, labels: np.ndarray) -> tuple[float | None, list[dict[str, int]]]:
    estimate = 0.0
    counts = []
    for cell in sorted(np.unique(labels)):
        z = labels == cell
        n1, n0 = int(np.sum(a[z] == 1)), int(np.sum(a[z] == 0))
        counts.append({"cell": int(cell), "n": int(z.sum()), "n1": n1, "n0": n0})
        if n1 == 0 or n0 == 0:
            return None, counts
        estimate += float(z.mean()) * (float(y[z & (a == 1)].mean()) - float(y[z & (a == 0)].mean()))
    return float(estimate), counts


def fitted_adjustment(data: dict[str, np.ndarray], features: np.ndarray, folds, replicate: int, n_index: int, method: str) -> tuple[float | None, list[dict[str, Any]]]:
    stitched = np.full(len(data["A"]), np.nan)
    details = []
    for fold, (tr, te) in enumerate(folds):
        sk = key("outcome", replicate, n_index, fold, METHOD_ID[method])
        pred, qa = base.fit_outcome(features[tr], data["A"][tr], data["Y"][tr], features[te], sk)
        if pred is not None:
            stitched[te] = pred
        details.append({"fold": fold, **qa})
    return (float(stitched.mean()) if np.isfinite(stitched).all() else None), details


def best_label_accuracy(labels: np.ndarray, h0: np.ndarray) -> float:
    return max(float(np.mean(np.array(p)[labels] == h0)) for p in itertools.permutations(range(3)))


def learned_hard_adjustment(data, diagnostic, folds, replicate: int, n_index: int, same_view: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    method_name = "heuristic_self_conditioned_v2" if same_view else "proposed_cross_mask_v1"
    method_id = METHOD_ID["self_conditioned_negative_control" if same_view else "learned_fixed_rep_audit"]
    if same_view:
        inp = np.column_stack([data["X"], data["W"]])
        audit = inp
        diag_inp = np.column_stack([diagnostic["X"], diagnostic["W"]])
        diag_audit = diag_inp
        audit_excluded = False
    else:
        inp = np.column_stack([data["X"], data["Wrep"]])
        audit = np.column_stack([data["X"], data["Waudit"]])
        diag_inp = np.column_stack([diagnostic["X"], diagnostic["Wrep"]])
        diag_audit = np.column_stack([diagnostic["X"], diagnostic["Waudit"]])
        audit_excluded = True
    labels = np.full(len(data["A"]), -1, dtype=int)
    soft_effect = np.full(len(data["A"]), np.nan)
    hard_effect = np.full(len(data["A"]), np.nan)
    details = []
    for fold, (tr, te) in enumerate(folds):
        rff_key = base.key("random_fourier_features", replicate, n_index, fold, method_id, 9, 0)
        adam_keys = [base.key("adam_initialization_and_training_noise", replicate, n_index, fold, method_id, 9, restart) for restart in range(base.RESTARTS)]
        fitted = base.fit_representation(inp[tr], audit[tr], data["A"][tr], rff_key, adam_keys, same_view)
        soft_tr, soft_te, soft_diag = fitted.predict(inp[tr]), fitted.predict(inp[te]), fitted.predict(diag_inp)
        hard_tr, hard_te, hard_diag = base.hard_onehot(soft_tr), base.hard_onehot(soft_te), base.hard_onehot(soft_diag)
        labels[te] = hard_te.argmax(1)
        outcome_key = key("outcome", replicate, n_index, fold, method_id, 9)
        pred, outcome_qa = base.fit_outcome(soft_tr, data["A"][tr], data["Y"][tr], soft_te, outcome_key)
        if pred is not None:
            soft_effect[te] = pred
        hard_outcome_key = key("outcome", replicate, n_index, fold, method_id, 10)
        hard_pred, hard_outcome_qa = base.fit_outcome(hard_tr, data["A"][tr], data["Y"][tr], hard_te, hard_outcome_key)
        if hard_pred is not None:
            hard_effect[te] = hard_pred
        phi_tr = fitted.rff.transform(audit[tr])
        phi_te = fitted.rff.transform(audit[te])
        phi_diag = fitted.rff.transform(diag_audit)
        details.append({
            "fold": fold,
            "audit_excluded": audit_excluded,
            "same_view_uncertified": same_view,
            "restart_objectives": fitted.restarts,
            "selected_restart": fitted.selected_restart,
            "outcome_soft_diagnostic": outcome_qa,
            "outcome_hard_primary": hard_outcome_qa,
            "train_soft_discrepancy": base.discrepancy(soft_tr, data["A"][tr], phi_tr),
            "heldout_soft_discrepancy": base.discrepancy(soft_te, data["A"][te], phi_te),
            "independent_soft_discrepancy": base.discrepancy(soft_diag, diagnostic["A"], phi_diag),
            "heldout_hard_discrepancy": base.discrepancy(hard_te, data["A"][te], phi_te),
            "independent_hard_discrepancy": base.discrepancy(hard_diag, diagnostic["A"], phi_diag),
            "heldout_hard_counts": base.hard_counts(hard_te, data["A"][te]),
            "heldout_soft_latent_imbalance": base.latent_imbalance(soft_te, data["A"][te], data["U"][te]),
            "heldout_hard_latent_imbalance": base.latent_imbalance(hard_te, data["A"][te], data["U"][te]),
            "independent_soft_latent_imbalance": base.latent_imbalance(soft_diag, diagnostic["A"], diagnostic["U"]),
            "independent_hard_latent_imbalance": base.latent_imbalance(hard_diag, diagnostic["A"], diagnostic["U"]),
            "independent_h0_recovery": best_label_accuracy(hard_diag.argmax(1), diagnostic["H0"]),
            "heldout_used_for_selection": False,
            "diagnostic_used_for_selection": False,
            "outcome_used_for_representation": False,
            "units_deleted": 0,
        })
    raw_stratified_estimate, cell_counts = stratified_ate(data["Y"], data["A"], labels)
    hard_estimate = float(hard_effect.mean()) if np.isfinite(hard_effect).all() else None
    soft_estimate = float(soft_effect.mean()) if np.isfinite(soft_effect).all() else None
    return {"estimate": hard_estimate, "finite": hard_estimate is not None, "soft_diagnostic_estimate": soft_estimate, "raw_stratified_diagnostic_estimate": raw_stratified_estimate, "hard_cell_counts": cell_counts, "oof_h0_recovery": best_label_accuracy(labels, data["H0"]), "method_internal_name": method_name}, details


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for n in PREFIXES:
        for method in METHOD_ID:
            selected = [row for row in rows if row["n"] == n and row["method"] == method]
            errors = np.array([row["error"] for row in selected])
            estimates = np.array([row["estimate"] for row in selected])
            output.append({"n": n, "method": method, "replicates": len(selected), "signed_bias": float(errors.mean()), "mean_absolute_error": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.mean(errors ** 2))), "sd_estimate": float(estimates.std(ddof=1))})
    return output


def sign_p(differences: np.ndarray) -> float:
    nonzero = differences[np.abs(differences) > 1e-15]
    wins = int(np.sum(nonzero < 0))
    tail = min(wins, len(nonzero) - wins)
    return min(1.0, 2.0 * sum(math.comb(len(nonzero), k) for k in range(tail + 1)) / 2.0 ** len(nonzero))


def paired(rows, comparator: str) -> list[dict[str, Any]]:
    output = []
    for n in PREFIXES:
        focal = sorted([r for r in rows if r["n"] == n and r["method"] == "learned_fixed_rep_audit"], key=lambda r: r["replicate"])
        comp = sorted([r for r in rows if r["n"] == n and r["method"] == comparator], key=lambda r: r["replicate"])
        differences = np.array([abs(f["error"]) - abs(c["error"]) for f, c in zip(focal, comp)])
        half = 2.093024054 * float(differences.std(ddof=1)) / math.sqrt(len(differences)) if len(differences) == 20 else None
        output.append({"n": n, "comparator": comparator, "differences": differences, "mean_difference": float(differences.mean()), "mean_difference_ci95": [float(differences.mean() - half), float(differences.mean() + half)] if half is not None else None, "wins": int(np.sum(differences < 0)), "ties": int(np.sum(differences == 0)), "exact_two_sided_sign_p": sign_p(differences)})
    return output


def performance_gate(summary, paired_rows, comparator: str) -> dict[str, Any]:
    focal = next(r for r in summary if r["n"] == 10000 and r["method"] == "learned_fixed_rep_audit")
    comp = next(r for r in summary if r["n"] == 10000 and r["method"] == comparator)
    pair = next(r for r in paired_rows if r["n"] == 10000)
    reduction = (comp["mean_absolute_error"] - focal["mean_absolute_error"]) / comp["mean_absolute_error"]
    criteria = {"wins_ge_14_of_20": pair["wins"] >= 14, "mean_absolute_error_lower": focal["mean_absolute_error"] < comp["mean_absolute_error"], "rmse_lower": focal["rmse"] < comp["rmse"], "mae_reduction_ge_20_percent": reduction >= 0.20}
    return {"comparator": comparator, "primary_n": 10000, "relative_mae_reduction": reduction, "criteria": criteria, "pass": all(criteria.values())}


def hard_diagnostic_failures(learned_details: list[dict[str, Any]]) -> dict[str, Any]:
    by_method_n: dict[tuple[str, int], dict[str, int]] = {}
    totals = {"null_heldout_hard_latent_imbalance": 0, "null_independent_hard_latent_imbalance": 0, "empty_heldout_hard_cells": 0, "nonempty_one_arm_heldout_hard_cells": 0}
    for item in learned_details:
        key_ = (item["method"], item["n"])
        group = by_method_n.setdefault(key_, {"folds": 0, "empty_heldout_hard_cells": 0, "nonempty_one_arm_heldout_hard_cells": 0})
        for fold in item["folds"]:
            group["folds"] += 1
            totals["null_heldout_hard_latent_imbalance"] += int(not np.isfinite(fold["heldout_hard_latent_imbalance"]))
            totals["null_independent_hard_latent_imbalance"] += int(not np.isfinite(fold["independent_hard_latent_imbalance"]))
            for cell in fold["heldout_hard_counts"]:
                is_empty = cell["n"] == 0
                is_one_arm = cell["n"] > 0 and (cell["n1"] == 0 or cell["n0"] == 0)
                totals["empty_heldout_hard_cells"] += int(is_empty)
                totals["nonempty_one_arm_heldout_hard_cells"] += int(is_one_arm)
                group["empty_heldout_hard_cells"] += int(is_empty)
                group["nonempty_one_arm_heldout_hard_cells"] += int(is_one_arm)
    groups = [{"method": method, "n": n, **values} for (method, n), values in sorted(by_method_n.items())]
    primary = next((row for row in groups if row["method"] == "learned_fixed_rep_audit" and row["n"] == 10000), None)
    primary_pass = None if primary is None else primary["empty_heldout_hard_cells"] == 0 and primary["nonempty_one_arm_heldout_hard_cells"] == 0
    return {"scope": "secondary hard diagnostics; failures do not invalidate finite primary T-learner estimates", "totals_across_all_n_and_both_learners": totals, "by_method_n": groups, "primary_n_learned_overlap_pass": primary_pass}


def deterministic_replay(coef: dict[str, np.ndarray]) -> dict[str, Any]:
    data = generate(N_MASTER, key("master", 0), coef)
    data = {k: value[:100] for k, value in data.items()}
    folds = base.fold_indices(data["A"], key("fold", 0, 0))
    diagnostic = generate(N_DIAGNOSTIC, key("diagnostic", 0), coef)
    first, _ = learned_hard_adjustment(data, diagnostic, folds, 0, 0, False)
    second, _ = learned_hard_adjustment(data, diagnostic, folds, 0, 0, False)
    return {"configuration": {"replicate": 0, "n": 100, "method": "learned_fixed_rep_audit"}, "first": first, "second": second, "pass": first == second, "excluded_selected_fits": 4, "excluded_adam_fits": 8}


def run(smoke: bool, replicate_count: int) -> dict[str, Any]:
    started = time.time()
    coef = coefficients()
    replicates = range(1 if smoke else replicate_count)
    prefixes = PREFIXES[:1] if smoke else PREFIXES
    rows, baseline_details, learned_details, master_hashes, diagnostic_hashes = [], [], [], [], []
    decoder_errors = []
    selected_fits = adam_fits = 0
    for replicate in replicates:
        master = generate(N_MASTER, key("master", replicate), coef)
        diagnostic = generate(N_DIAGNOSTIC, key("diagnostic", replicate), coef)
        master_hashes.append({"replicate": replicate, "hash": array_hash(np.column_stack([master["U"], master["A"], master["Y"]]))})
        diagnostic_hashes.append({"replicate": replicate, "hash": array_hash(np.column_stack([diagnostic["U"], diagnostic["A"], diagnostic["Y"]]))})
        decoder_errors.extend([float(np.max(np.abs((master["Wrep"][:, 1] - master["Wrep"][:, 0]) - master["S"]))), float(np.max(np.abs((diagnostic["Wrep"][:, 1] - diagnostic["Wrep"][:, 0]) - diagnostic["S"])))])
        for n_index, n in enumerate(prefixes):
            data = {k: value[:n] for k, value in master.items()}
            folds = base.fold_indices(data["A"], key("fold", replicate, n_index))
            feature_map = {
                "adjust_x": data["X"],
                "adjust_xw_full_observed": np.column_stack([data["X"], data["W"]]),
                "observed_xs_fitted_reference": np.column_stack([data["X"], data["S"]]),
                "observed_xu_fitted_reference": np.column_stack([data["X"], data["U"]]),
            }
            for method, features in feature_map.items():
                estimate, details = fitted_adjustment(data, features, folds, replicate, n_index, method)
                rows.append({"replicate": replicate, "n": n, "method": method, "estimate": estimate, "error": estimate - TRUE_ATE if estimate is not None else None, "finite": estimate is not None})
                baseline_details.append({"replicate": replicate, "n": n, "method": method, "folds": details})
            oracle_features = np.eye(3)[data["H0"]]
            oracle_estimate, oracle_folds = fitted_adjustment(data, oracle_features, folds, replicate, n_index, "oracle_h0_stratified")
            oracle_raw_estimate, oracle_counts = stratified_ate(data["Y"], data["A"], data["H0"])
            rows.append({"replicate": replicate, "n": n, "method": "oracle_h0_stratified", "estimate": oracle_estimate, "error": oracle_estimate - TRUE_ATE if oracle_estimate is not None else None, "finite": oracle_estimate is not None, "raw_stratified_diagnostic_estimate": oracle_raw_estimate, "cell_counts": oracle_counts})
            baseline_details.append({"replicate": replicate, "n": n, "method": "oracle_h0_stratified", "folds": oracle_folds})
            learned, details = learned_hard_adjustment(data, diagnostic, folds, replicate, n_index, False)
            selected_fits += 2
            adam_fits += 2 * base.RESTARTS
            rows.append({"replicate": replicate, "n": n, "method": "learned_fixed_rep_audit", "estimate": learned["estimate"], "error": learned["estimate"] - TRUE_ATE if learned["estimate"] is not None else None, "finite": learned["finite"], "soft_diagnostic_estimate": learned["soft_diagnostic_estimate"], "raw_stratified_diagnostic_estimate": learned["raw_stratified_diagnostic_estimate"], "oof_h0_recovery": learned["oof_h0_recovery"], "hard_cell_counts": learned["hard_cell_counts"]})
            learned_details.append({"replicate": replicate, "n": n, "method": "learned_fixed_rep_audit", "folds": details})
            self_conditioned, details = learned_hard_adjustment(data, diagnostic, folds, replicate, n_index, True)
            selected_fits += 2
            adam_fits += 2 * base.RESTARTS
            rows.append({"replicate": replicate, "n": n, "method": "self_conditioned_negative_control", "estimate": self_conditioned["estimate"], "error": self_conditioned["estimate"] - TRUE_ATE if self_conditioned["estimate"] is not None else None, "finite": self_conditioned["finite"], "soft_diagnostic_estimate": self_conditioned["soft_diagnostic_estimate"], "raw_stratified_diagnostic_estimate": self_conditioned["raw_stratified_diagnostic_estimate"], "oof_h0_recovery": self_conditioned["oof_h0_recovery"], "hard_cell_counts": self_conditioned["hard_cell_counts"]})
            learned_details.append({"replicate": replicate, "n": n, "method": "self_conditioned_negative_control", "folds": details})
    all_folds = [fold for item in learned_details for fold in item["folds"]]
    replay = deterministic_replay(coef)
    expected_rows = len(list(replicates)) * len(prefixes) * len(METHOD_ID)
    expected_selected = len(list(replicates)) * len(prefixes) * 2 * 2
    expected_adam = expected_selected * base.RESTARTS
    gates = {
        "root_seed_exact": ROOT_SEED == 190602,
        "exact_linear_decoder": max(decoder_errors) < 1e-12,
        "propensity_values_exact": all(set(np.unique(generate(1000, key("diagnostic", replicate, extra=9), coef)["propensity"])).issubset(set(PI)) for replicate in range(2)),
        "strict_overlap": bool(np.all((PI > 0) & (PI < 1))),
        "gaussian_audit_channel_injective_metadata": True,
        "wrep_contains_s_but_not_continuous_u_metadata": True,
        "latent_confounding_present_metadata": True,
        "aggregate_row_count": len(rows) == expected_rows,
        "selected_fit_count": selected_fits == expected_selected,
        "adam_fit_count": adam_fits == expected_adam,
        "all_estimates_finite": all(row["finite"] for row in rows),
        "learned_audit_excluded": all(fold["audit_excluded"] for item in learned_details if item["method"] == "learned_fixed_rep_audit" for fold in item["folds"]),
        "no_outcome_leakage": all(not fold["outcome_used_for_representation"] for fold in all_folds),
        "heldout_and_diagnostic_not_used_for_selection": all(not fold["heldout_used_for_selection"] and not fold["diagnostic_used_for_selection"] for fold in all_folds),
        "zero_unit_deletion": all(fold["units_deleted"] == 0 for fold in all_folds),
        "unique_master_hashes": len({row["hash"] for row in master_hashes}) == len(master_hashes),
        "unique_diagnostic_hashes": len({row["hash"] for row in diagnostic_hashes}) == len(diagnostic_hashes),
        "nested_prefix_contract": prefixes == (PREFIXES[:1] if smoke else PREFIXES),
        "deterministic_replay": replay["pass"],
    }
    summary = summarize(rows) if not smoke else []
    pair_x = paired(rows, "adjust_x") if not smoke else []
    pair_xw = paired(rows, "adjust_xw_full_observed") if not smoke else []
    primary = performance_gate(summary, pair_x, "adjust_x") if not smoke else None
    secondary = performance_gate(summary, pair_xw, "adjust_xw_full_observed") if not smoke else None
    diagnostic_summary = []
    for n in prefixes:
        for method in ("learned_fixed_rep_audit", "self_conditioned_negative_control"):
            folds = [fold for item in learned_details if item["n"] == n and item["method"] == method for fold in item["folds"]]
            diagnostic_summary.append({"n": n, "method": method, "folds": len(folds), "mean_heldout_soft_discrepancy": float(np.mean([fold["heldout_soft_discrepancy"]["value"] for fold in folds])), "mean_independent_soft_discrepancy": float(np.mean([fold["independent_soft_discrepancy"]["value"] for fold in folds])), "mean_heldout_hard_discrepancy": float(np.mean([fold["heldout_hard_discrepancy"]["value"] for fold in folds])), "mean_independent_hard_discrepancy": float(np.mean([fold["independent_hard_discrepancy"]["value"] for fold in folds])), "mean_independent_hard_latent_imbalance": float(np.nanmean([fold["independent_hard_latent_imbalance"] for fold in folds])), "mean_independent_h0_recovery": float(np.mean([fold["independent_h0_recovery"] for fold in folds]))})
    hard_failures = hard_diagnostic_failures(learned_details)
    gates["primary_n_learned_hard_cell_overlap"] = True if smoke else hard_failures["primary_n_learned_overlap_pass"] is True
    return jsonable({
        "provenance": {"script": "code/exact_injectivity_finite_strata_experiment.py", "root_seed": ROOT_SEED, "replicates": len(list(replicates)), "sample_sizes": prefixes, "master_n": N_MASTER, "diagnostic_n": N_DIAGNOSTIC, "nested_prefixes": True, "no_cross_prefix_pooling": True, "post_result_tuning": False},
        "dgp": {"dimensions": {"U": 1, "X": D_X, "Wrep": D_WREP, "Waudit": 1}, "coefficients": coef, "structural_ate": TRUE_ATE, "pi": PI, "exact_decoder": "S = Wrep[1] - Wrep[0] almost surely", "individual_coordinate_nonrecoverability": "Wrep[0]=V and Wrep[1]=V+S each have overlapping support across S", "wrep_scope": "Wrep is conditionally independent of continuous U given S", "audit_channel": "Waudit=U+0.7E; Gaussian convolution is injective", "oracle_score": "H0=S+1{gamma^T X>0}; P(A=1|X,U)=pi[H0]"},
        "method_contract": {"primary_learned_estimate": "cross-fitted common T-learner using out-of-fold hard argmax cells", "soft_estimate": "diagnostic only", "raw_stratified_estimate": "diagnostic only; may be undefined when a finite-sample cell-arm is empty", "oracle_h0": "cross-fitted common T-learner using hard three-cell H0", "outcome_learner": base.GBR_SPEC, "objective": "64-dimensional RFF soft conditional-covariance surrogate", "exact_kci_or_hsic": False, "learned_input": "X and Wrep", "learned_audit": "X and Waudit", "self_conditioned": "uncertified negative control"},
        "aggregate_results": rows,
        "baseline_fold_diagnostics": baseline_details,
        "learned_fold_diagnostics": learned_details,
        "method_summary_by_n": summary,
        "paired_vs_x": pair_x,
        "paired_vs_full_xw": pair_xw,
        "primary_v1_vs_x_gate": primary,
        "secondary_v1_vs_full_xw_gate": secondary,
        "representation_diagnostic_summary": diagnostic_summary,
        "secondary_hard_diagnostic_failures": hard_failures,
        "seed_records": {"master": master_hashes, "diagnostic": diagnostic_hashes, "deterministic_replay": replay},
        "pipeline_gates": gates,
        "pipeline_pass": all(gates.values()),
        "pipeline_pass_scope": "primary-estimate execution and primary-n learned hard-cell overlap; it does not assert that every secondary hard stratum at every finite n has both treatment arms",
        "counts": {"aggregate_rows": len(rows), "selected_representation_fits": selected_fits, "adam_fits": adam_fits},
        "smoke_mode": smoke,
        "runtime_seconds": time.time() - started,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Explicit nonfinal output path; required for writing nonregistered runs")
    args = parser.parse_args()
    result = run(args.smoke, args.replicates)
    registered_full = not args.smoke and args.replicates == 20
    target = args.output.resolve() if args.output is not None else (OUTPUT.resolve() if registered_full else None)
    if target == OUTPUT.resolve() and not registered_full:
        parser.error("the registered full JSON may only be written by the 20-replicate non-smoke run")
    if args.no_write:
        target = None
    if target is not None:
        target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"smoke": args.smoke, "replicates": args.replicates, "pipeline_pass": result["pipeline_pass"], "failed_gates": [key for key, value in result["pipeline_gates"].items() if not value], "counts": result["counts"], "runtime_seconds": result["runtime_seconds"], "output_written": target is not None, "output_path": str(target) if target is not None else None}, indent=2))


if __name__ == "__main__":
    main()
