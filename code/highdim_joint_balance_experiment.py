#!/usr/bin/env python3
"""Preregistered high-dimensional joint-balance experiment.

The representation always has the form H=phi(X).  The matched learners have
identical data, architecture, initialization, optimizer, penalties, capacity,
and training length.  Their only difference is the balancing kernel:

* joint: k_XW((x,w),(x',w')) = k_X(x,x') k_W(w,w'),
* x_only: k_X(x,x').

W is used only in the joint training loss and held-out diagnostics.  It never
enters H assignment or the final H-stratified effect estimator.

Outputs:
  results/highdim_joint_balance_summary.json
  results/highdim_joint_balance_runs.csv
  figures/highdim_joint_balance_comparison.(png|pdf)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import learned_representation_balancing_experiment as base


TRUE_TAU = 1.0
BASE_SEED = 2026081801
PRIMARY_REGIMES = ("primary",)


@dataclass
class RepresentationRecord:
    regime: str
    dimension: int
    rep: int
    method: str
    estimate: float
    tau_error: float
    oracle_effect: float
    oracle_bias: float
    d_joint: float
    d_x: float
    d_w: float
    dropped_mass: float


@dataclass
class BaselineRecord:
    regime: str
    dimension: int
    rep: int
    method: str
    estimate: float
    tau_error: float


@lru_cache(maxsize=None)
def directions(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed dense orthonormal severity and treatment-only directions."""
    rng = np.random.default_rng(BASE_SEED + 97 * dimension)
    raw = rng.standard_normal((dimension, 2))
    q, _ = np.linalg.qr(raw)
    b_c = q[:, 0]
    b_t = q[:, 1]
    if abs(float(b_c @ b_t)) > 1e-12:
        raise AssertionError("generated directions are not orthogonal")
    return b_c, b_t


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


def simulate(
    rng: np.random.Generator,
    n: int,
    dimension: int,
    regime: str,
) -> dict[str, np.ndarray]:
    x = rng.standard_normal((n, dimension))
    b_c, b_t = directions(dimension)
    r = x @ b_c
    q = x @ b_t
    eps_u = rng.standard_normal(n)
    if regime == "x_signal_null":
        u = eps_u
    elif regime in PRIMARY_REGIMES or regime == "proxy_null":
        u = r + 0.5 * np.sin(r) + 0.5 * eps_u
    else:
        raise ValueError(f"unknown regime: {regime}")

    if regime == "proxy_null":
        w = rng.standard_normal(n)
    else:
        w = u + 0.5 * rng.standard_normal(n)

    propensity = 0.05 + 0.90 * _sigmoid(0.8 * u + 1.5 * q)
    a = rng.binomial(1, propensity).astype(float)
    structural_y = TRUE_TAU * a + 2.0 * u
    y = structural_y + rng.standard_normal(n)
    arrays = (x, r, q, u, w, propensity, a, structural_y, y)
    if not all(np.isfinite(value).all() for value in arrays):
        raise FloatingPointError("non-finite simulated value")
    return {
        "X": x,
        "U": u,
        "W": w,
        "A": a,
        "Y_struct": structural_y,
        "Y": y,
        "propensity": propensity,
    }


def _standardize(
    train: np.ndarray,
    evaluate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-8
    return (train - mean) / std, (evaluate - mean) / std


def _gram_pair(
    train: np.ndarray,
    evaluate: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    bandwidth = base.median_bandwidth(train, rng)
    return (
        base.rbf_gram(train, train, bandwidth),
        base.rbf_gram(evaluate, evaluate, bandwidth),
        bandwidth,
    )


def train_representation(
    xs: np.ndarray,
    a: np.ndarray,
    gram: np.ndarray,
    m_strata: int,
    steps: int,
    seed: int,
) -> base.StratumNet:
    """Matched existing learner with input dimension supplied by X."""
    torch.manual_seed(seed)
    net = base.StratumNet(d_in=xs.shape[1], m_strata=m_strata)
    x_t = torch.tensor(xs, dtype=torch.float32)
    a_t = torch.tensor(a, dtype=torch.float32)
    k_t = torch.tensor(gram, dtype=torch.float32)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    overlap_floor = 0.09
    mass_floor = 0.5 / m_strata
    for step in range(steps):
        temperature = 1.0 * (0.25 / 1.0) ** (step / max(steps - 1, 1))
        soft = torch.softmax(net(x_t) / temperature, dim=1)
        discrepancy, p_hat, pi_hat = base.soft_delta_sq(soft, a_t, k_t)
        overlap_penalty = (
            p_hat * torch.relu(overlap_floor - pi_hat * (1.0 - pi_hat))
        ).sum()
        mass_penalty = torch.relu(mass_floor - p_hat).sum()
        loss = discrepancy + overlap_penalty + mass_penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return net


def _seed(regime: str, dimension: int, rep: int) -> int:
    regime_code = {"primary": 0, "proxy_null": 1, "x_signal_null": 2}[regime]
    return BASE_SEED + 7919 * rep + 101 * dimension + 100003 * regime_code


def run_once(
    regime: str,
    dimension: int,
    rep: int,
    n: int,
    m_strata: int,
    steps: int,
    include_baselines: bool,
) -> tuple[list[RepresentationRecord], list[BaselineRecord]]:
    seed = _seed(regime, dimension, rep)
    rng = np.random.default_rng(seed)
    data = simulate(rng, n, dimension, regime)
    permutation = rng.permutation(n)
    half = n // 2
    train_idx, eval_idx = permutation[:half], permutation[half:]
    if np.intersect1d(train_idx, eval_idx).size:
        raise AssertionError("training and evaluation folds overlap")

    x_tr, x_ev = data["X"][train_idx], data["X"][eval_idx]
    w_tr, w_ev = data["W"][train_idx], data["W"][eval_idx]
    a_tr, a_ev = data["A"][train_idx], data["A"][eval_idx]
    y_tr, y_ev = data["Y"][train_idx], data["Y"][eval_idx]
    u_tr, u_ev = data["U"][train_idx], data["U"][eval_idx]
    y_struct_ev = data["Y_struct"][eval_idx]

    xs_tr, xs_ev = _standardize(x_tr, x_ev)
    ws_tr, ws_ev = _standardize(w_tr[:, None], w_ev[:, None])
    x_train, x_eval, _ = _gram_pair(xs_tr, xs_ev, seed + 101)
    w_train, w_eval, _ = _gram_pair(ws_tr, ws_ev, seed + 202)
    joint_train = x_train * w_train
    joint_eval = x_eval * w_eval

    nets = {
        "joint": train_representation(
            xs_tr, a_tr, joint_train, m_strata, steps, seed
        ),
        "x_only": train_representation(
            xs_tr, a_tr, x_train, m_strata, steps, seed
        ),
    }
    representation_rows: list[RepresentationRecord] = []
    for method, net in nets.items():
        strata = base.assign_strata(net, xs_ev)
        estimate, dropped = base.stratified_tau(strata, a_ev, y_ev, m_strata)
        oracle_effect, oracle_dropped = base.stratified_tau(
            strata, a_ev, y_struct_ev, m_strata
        )
        if abs(dropped - oracle_dropped) > 1e-12:
            raise AssertionError("noisy and structural estimators use different strata")
        representation_rows.append(
            RepresentationRecord(
                regime=regime,
                dimension=dimension,
                rep=rep,
                method=method,
                estimate=estimate,
                tau_error=estimate - TRUE_TAU,
                oracle_effect=oracle_effect,
                oracle_bias=oracle_effect - TRUE_TAU,
                d_joint=math.sqrt(
                    max(base.hard_delta_sq(strata, a_ev, joint_eval, m_strata), 0.0)
                ),
                d_x=math.sqrt(
                    max(base.hard_delta_sq(strata, a_ev, x_eval, m_strata), 0.0)
                ),
                d_w=math.sqrt(
                    max(base.hard_delta_sq(strata, a_ev, w_eval, m_strata), 0.0)
                ),
                dropped_mass=dropped,
            )
        )

    baseline_rows: list[BaselineRecord] = []
    if include_baselines:
        specifications = {
            "naive": float(y_ev[a_ev == 1].mean() - y_ev[a_ev == 0].mean()),
            "full_x": base.g_formula_tau(x_tr, a_tr, y_tr, x_ev, seed),
            "oracle_xu": base.g_formula_tau(
                np.column_stack([x_tr, u_tr]),
                a_tr,
                y_tr,
                np.column_stack([x_ev, u_ev]),
                seed,
            ),
        }
        baseline_rows = [
            BaselineRecord(
                regime=regime,
                dimension=dimension,
                rep=rep,
                method=method,
                estimate=estimate,
                tau_error=estimate - TRUE_TAU,
            )
            for method, estimate in specifications.items()
        ]
    return representation_rows, baseline_rows


def _aggregate(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "mean_abs": float(np.abs(values).mean()),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "se": float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
    }


def summarize(
    representation_rows: list[RepresentationRecord],
    baseline_rows: list[BaselineRecord],
    args: argparse.Namespace,
) -> dict:
    payload: dict = {
        "provenance": {
            "script": "code/highdim_joint_balance_experiment.py",
            "base_seed": BASE_SEED,
            "seed_formula": (
                "2026081801 + 7919*rep + 101*d + 100003*regime_code"
            ),
            "n": args.n,
            "reps": args.reps,
            "steps": args.steps,
            "strata": args.strata,
            "train_evaluation_split": "50/50",
            "dimensions": args.dimensions,
            "controls_at_dimension": 30,
            "true_tau": TRUE_TAU,
            "joint_kernel": "k_XW=k_X*k_W; separate train-median RBF bandwidths",
            "matched_learner_contract": "only the balancing kernel differs",
            "h_input": "X only",
            "w_used_by_effect_estimator": False,
            "post_result_tuning": False,
        },
        "configs": {},
        "predeclared_success": {
            "primary_criterion": (
                "both primary d configs have lower mean held-out D_XW, lower "
                "mean oracle absolute bias, lower noisy RMSE, and paired all-three "
                "win rate >=0.625"
            ),
            "negative_control_criterion": (
                "neither control has paired all-three win rate >=0.625"
            ),
        },
    }
    config_order = [("primary", d) for d in args.dimensions]
    config_order += [("proxy_null", 30), ("x_signal_null", 30)]
    primary_passes: list[bool] = []
    control_passes: list[bool] = []
    for regime, dimension in config_order:
        key = f"regime={regime}|d={dimension}"
        cfg_rows = [
            row
            for row in representation_rows
            if row.regime == regime and row.dimension == dimension
        ]
        methods = {}
        by_rep = {}
        for method in ("joint", "x_only"):
            rows = [row for row in cfg_rows if row.method == method]
            by_rep[method] = {row.rep: row for row in rows}
            methods[method] = {
                "tau_error": _aggregate(np.array([row.tau_error for row in rows])),
                "oracle_bias": _aggregate(np.array([row.oracle_bias for row in rows])),
                "mean_d_joint": float(np.mean([row.d_joint for row in rows])),
                "mean_d_x": float(np.mean([row.d_x for row in rows])),
                "mean_d_w": float(np.mean([row.d_w for row in rows])),
                "mean_dropped_mass": float(np.mean([row.dropped_mass for row in rows])),
            }
        common = sorted(set(by_rep["joint"]) & set(by_rep["x_only"]))
        paired_wins = [
            by_rep["joint"][rep].d_joint < by_rep["x_only"][rep].d_joint
            and abs(by_rep["joint"][rep].oracle_bias)
            < abs(by_rep["x_only"][rep].oracle_bias)
            and abs(by_rep["joint"][rep].tau_error)
            < abs(by_rep["x_only"][rep].tau_error)
            for rep in common
        ]
        paired_rate = float(np.mean(paired_wins))
        mean_order = (
            methods["joint"]["mean_d_joint"] < methods["x_only"]["mean_d_joint"]
            and methods["joint"]["oracle_bias"]["mean_abs"]
            < methods["x_only"]["oracle_bias"]["mean_abs"]
            and methods["joint"]["tau_error"]["rmse"]
            < methods["x_only"]["tau_error"]["rmse"]
        )
        config_pass = bool(mean_order and paired_rate >= 0.625)
        if regime == "primary":
            primary_passes.append(config_pass)
        else:
            control_passes.append(paired_rate < 0.625)
        cfg_baselines = [
            row
            for row in baseline_rows
            if row.regime == regime and row.dimension == dimension
        ]
        baselines = {
            method: _aggregate(
                np.array(
                    [row.tau_error for row in cfg_baselines if row.method == method]
                )
            )
            for method in ("naive", "full_x", "oracle_xu")
        } if cfg_baselines else {}
        payload["configs"][key] = {
            "representations": methods,
            "baselines": baselines,
            "paired_all_three_win_rate": paired_rate,
            "config_pass": config_pass if regime == "primary" else None,
            "negative_control_sanity_pass": (
                paired_rate < 0.625 if regime != "primary" else None
            ),
        }
    payload["predeclared_success"]["primary_configs_passed"] = int(sum(primary_passes))
    payload["predeclared_success"]["primary_configs_total"] = len(primary_passes)
    payload["predeclared_success"]["negative_controls_passed"] = int(sum(control_passes))
    payload["predeclared_success"]["negative_controls_total"] = len(control_passes)
    payload["predeclared_success"]["passed"] = bool(
        all(primary_passes) and all(control_passes)
    )
    return payload


def write_outputs(
    representation_rows: list[RepresentationRecord],
    summary: dict,
) -> None:
    base.RESULTS_DIR.mkdir(exist_ok=True)
    base.FIGURES_DIR.mkdir(exist_ok=True)
    (base.RESULTS_DIR / "highdim_joint_balance_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    with (base.RESULTS_DIR / "highdim_joint_balance_runs.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(asdict(representation_rows[0]).keys())
        )
        writer.writeheader()
        for row in representation_rows:
            writer.writerow(asdict(row))

    labels = [
        key.replace("regime=", "").replace("|d=", "\n$d=") + "$"
        for key in summary["configs"]
    ]
    metrics = (
        ("mean_d_joint", "Held-out $D_{XW}$"),
        ("oracle_bias", "Oracle absolute bias"),
        ("tau_error", "Finite-sample RMSE"),
    )
    colors = {"joint": "#0f766e", "x_only": "#d97706"}
    positions = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    for ax, (metric, title) in zip(axes, metrics):
        for offset, method in ((-width / 2, "joint"), (width / 2, "x_only")):
            values = []
            for cfg in summary["configs"].values():
                method_summary = cfg["representations"][method]
                if metric == "mean_d_joint":
                    values.append(method_summary[metric])
                elif metric == "oracle_bias":
                    values.append(method_summary[metric]["mean_abs"])
                else:
                    values.append(method_summary[metric]["rmse"])
            ax.bar(positions + offset, values, width, color=colors[method], label=method)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(title)
    axes[0].legend()
    fig.suptitle("High-dimensional matched joint-balance experiment")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for extension in ("png", "pdf"):
        fig.savefig(
            base.FIGURES_DIR / f"highdim_joint_balance_comparison.{extension}",
            dpi=200,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[10, 30])
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--strata", type=int, default=5)
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.n = 400
        args.dimensions = [10, 30]
        args.reps = 1
        args.steps = 30

    configurations = [("primary", d) for d in args.dimensions]
    configurations += [("proxy_null", 30), ("x_signal_null", 30)]
    representation_rows: list[RepresentationRecord] = []
    baseline_rows: list[BaselineRecord] = []
    for regime, dimension in configurations:
        for rep in range(args.reps):
            reps, bases = run_once(
                regime,
                dimension,
                rep,
                args.n,
                args.strata,
                args.steps,
                not args.no_baselines,
            )
            representation_rows.extend(reps)
            baseline_rows.extend(bases)
            print(
                f"regime={regime} d={dimension} rep={rep + 1}/{args.reps}",
                flush=True,
            )
    summary = summarize(representation_rows, baseline_rows, args)
    if not args.no_write:
        write_outputs(representation_rows, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
