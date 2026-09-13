#!/usr/bin/env python3
"""Matched joint-(X,W) versus X-only representation experiment.

This script keeps the mechanism-first DGP and estimators from
``learned_representation_balancing_experiment.py`` unchanged.  It learns two
representations H=phi(X) with the same data split, network architecture,
initialization, optimizer, penalties, number of strata, and training steps.
The only difference is the kernel in the balancing loss:

* joint: characteristic RBF kernel on standardized (X,W),
* x_only: characteristic RBF kernel on standardized X.

W is used only by the joint balancing loss and held-out diagnostics.  The
effect estimator receives only H, A, and Y.  The simulation-only oracle bias
uses the noiseless structural outcome to separate representation bias from
outcome noise.

Outputs for a main run:
  results/joint_vs_xonly_representation_summary.json
  results/joint_vs_xonly_representation_runs.csv
  figures/joint_vs_xonly_representation_comparison.(png|pdf)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import learned_representation_balancing_experiment as base


@dataclass
class RepresentationRecord:
    sigma_w: float
    gamma_u: float
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
    sigma_w: float
    gamma_u: float
    rep: int
    method: str
    estimate: float
    tau_error: float


def _bandwidth_and_grams(
    train: np.ndarray,
    evaluate: np.ndarray,
    seed: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    bandwidth = base.median_bandwidth(train, rng)
    return (
        bandwidth,
        base.rbf_gram(train, train, bandwidth),
        base.rbf_gram(evaluate, evaluate, bandwidth),
    )


def _oracle_structural_outcome(
    x: np.ndarray,
    u: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """Observed noise-free structural outcome under the realized treatment."""
    return base.TRUE_TAU * a + base.f_outcome_x(x) + base.BETA_U * u


def run_once(
    sigma_w: float,
    gamma_u: float,
    rep: int,
    n: int,
    m_strata: int,
    steps: int,
) -> tuple[list[RepresentationRecord], list[BaselineRecord]]:
    seed, _, prep = base.prepare_run(sigma_w, gamma_u, rep, n)

    xs_tr, xs_ev = prep["xs_tr"], prep["xs_ev"]
    a_tr, a_ev = prep["a_tr"], prep["a_ev"]
    y_tr, y_ev = prep["y_tr"], prep["y_ev"]
    x_tr, x_ev = prep["x_tr"], prep["x_ev"]
    u_tr, u_ev = prep["u_tr"], prep["u_ev"]
    w_tr, w_ev = prep["w_tr"], prep["w_ev"]

    joint_train, joint_eval = prep["gram_tr"], prep["gram_ev"]
    _, x_train, x_eval = _bandwidth_and_grams(xs_tr, xs_ev, seed + 101)

    w_mean = float(w_tr.mean())
    w_std = float(w_tr.std() + 1e-12)
    ws_tr = ((w_tr - w_mean) / w_std)[:, None]
    ws_ev = ((w_ev - w_mean) / w_std)[:, None]
    _, _, w_eval = _bandwidth_and_grams(ws_tr, ws_ev, seed + 202)

    train_data = {"Xs": xs_tr, "A": a_tr}
    nets = {}
    for method, gram in (("joint", joint_train), ("x_only", x_train)):
        # train_representation resets torch's seed internally.  Using the same
        # seed gives identical initial weights and training settings.
        nets[method], _ = base.train_representation(
            train_data,
            gram,
            m_strata,
            steps,
            seed,
        )

    structural_y = _oracle_structural_outcome(x_ev, u_ev, a_ev)
    representation_rows: list[RepresentationRecord] = []
    for method, net in nets.items():
        strata = base.assign_strata(net, xs_ev)
        estimate, dropped = base.stratified_tau(
            strata,
            a_ev,
            y_ev,
            m_strata,
        )
        oracle_effect, dropped_oracle = base.stratified_tau(
            strata,
            a_ev,
            structural_y,
            m_strata,
        )
        if abs(dropped - dropped_oracle) > 1e-12:
            raise AssertionError("observed and oracle estimators retained different strata")
        representation_rows.append(
            RepresentationRecord(
                sigma_w=sigma_w,
                gamma_u=gamma_u,
                rep=rep,
                method=method,
                estimate=estimate,
                tau_error=estimate - base.TRUE_TAU,
                oracle_effect=oracle_effect,
                oracle_bias=oracle_effect - base.TRUE_TAU,
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

    baseline_rows = [
        BaselineRecord(
            sigma_w,
            gamma_u,
            rep,
            "naive",
            float(y_ev[a_ev == 1].mean() - y_ev[a_ev == 0].mean()),
            0.0,
        ),
        BaselineRecord(
            sigma_w,
            gamma_u,
            rep,
            "full_x",
            base.g_formula_tau(x_tr, a_tr, y_tr, x_ev, seed),
            0.0,
        ),
        BaselineRecord(
            sigma_w,
            gamma_u,
            rep,
            "oracle_xu",
            base.g_formula_tau(
                np.column_stack([x_tr, u_tr]),
                a_tr,
                y_tr,
                np.column_stack([x_ev, u_ev]),
                seed,
            ),
            0.0,
        ),
    ]
    for row in baseline_rows:
        row.tau_error = row.estimate - base.TRUE_TAU
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
            "script": "code/joint_vs_xonly_representation_experiment.py",
            "source_dgp": "code/learned_representation_balancing_experiment.py",
            "base_seed": 20260817,
            "seed_formula": (
                "20260817 + 7919*rep + int(1000*sigma_w) + int(10*gamma_u)"
            ),
            "n": args.n,
            "strata": args.strata,
            "steps": args.steps,
            "reps": args.reps,
            "sigmas": args.sigmas,
            "gammas": args.gammas,
            "true_tau": base.TRUE_TAU,
            "matched_learner_contract": "only the balancing kernel differs",
        },
        "configs": {},
        "predeclared_success": {
            "criterion": (
                "joint has lower held-out D_XW, lower absolute oracle bias, "
                "and lower RMSE than x_only in a majority of informative configurations"
            )
        },
    }
    for sigma_w in args.sigmas:
        for gamma_u in args.gammas:
            key = f"sigma_w={sigma_w}|gamma_u={gamma_u}"
            cfg_rep = [
                row
                for row in representation_rows
                if row.sigma_w == sigma_w and row.gamma_u == gamma_u
            ]
            methods = {}
            for method in ("joint", "x_only"):
                rows = [row for row in cfg_rep if row.method == method]
                methods[method] = {
                    "tau_error": _aggregate(np.array([row.tau_error for row in rows])),
                    "oracle_bias": _aggregate(np.array([row.oracle_bias for row in rows])),
                    "mean_d_joint": float(np.mean([row.d_joint for row in rows])),
                    "mean_d_x": float(np.mean([row.d_x for row in rows])),
                    "mean_d_w": float(np.mean([row.d_w for row in rows])),
                    "mean_dropped_mass": float(np.mean([row.dropped_mass for row in rows])),
                }
            baselines = {}
            cfg_base = [
                row
                for row in baseline_rows
                if row.sigma_w == sigma_w and row.gamma_u == gamma_u
            ]
            for method in ("naive", "full_x", "oracle_xu"):
                errors = np.array([row.tau_error for row in cfg_base if row.method == method])
                baselines[method] = _aggregate(errors)
            methods_by_rep = {
                method: {row.rep: row for row in cfg_rep if row.method == method}
                for method in ("joint", "x_only")
            }
            common_reps = sorted(set(methods_by_rep["joint"]) & set(methods_by_rep["x_only"]))
            wins = [
                (
                    methods_by_rep["joint"][rep].d_joint
                    < methods_by_rep["x_only"][rep].d_joint
                    and abs(methods_by_rep["joint"][rep].oracle_bias)
                    < abs(methods_by_rep["x_only"][rep].oracle_bias)
                    and abs(methods_by_rep["joint"][rep].tau_error)
                    < abs(methods_by_rep["x_only"][rep].tau_error)
                )
                for rep in common_reps
            ]
            payload["configs"][key] = {
                "representations": methods,
                "baselines": baselines,
                "paired_all_three_win_rate": float(np.mean(wins)) if wins else float("nan"),
            }
    informative = [
        cfg
        for key, cfg in payload["configs"].items()
        if float(key.split("|")[0].split("=")[1]) <= 1.0
        and float(key.split("|")[1].split("=")[1]) >= 1.0
    ]
    config_wins = [
        cfg["representations"]["joint"]["mean_d_joint"]
        < cfg["representations"]["x_only"]["mean_d_joint"]
        and cfg["representations"]["joint"]["oracle_bias"]["mean_abs"]
        < cfg["representations"]["x_only"]["oracle_bias"]["mean_abs"]
        and cfg["representations"]["joint"]["tau_error"]["rmse"]
        < cfg["representations"]["x_only"]["tau_error"]["rmse"]
        for cfg in informative
    ]
    payload["predeclared_success"]["informative_configs"] = len(informative)
    payload["predeclared_success"]["config_win_rate"] = (
        float(np.mean(config_wins)) if config_wins else float("nan")
    )
    payload["predeclared_success"]["passed"] = bool(
        config_wins and np.mean(config_wins) > 0.5
    )
    return payload


def write_outputs(
    representation_rows: list[RepresentationRecord],
    baseline_rows: list[BaselineRecord],
    summary: dict,
) -> None:
    base.RESULTS_DIR.mkdir(exist_ok=True)
    base.FIGURES_DIR.mkdir(exist_ok=True)
    (base.RESULTS_DIR / "joint_vs_xonly_representation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    with (base.RESULTS_DIR / "joint_vs_xonly_representation_runs.csv").open(
        "w", newline=""
    ) as handle:
        fieldnames = list(asdict(representation_rows[0]).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in representation_rows:
            writer.writerow(asdict(row))

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    colors = {"joint": "#0f766e", "x_only": "#d97706"}
    for ax, metric, title in (
        (axes[0], "d_joint", "Held-out joint discrepancy"),
        (axes[1], "oracle_bias", "Oracle absolute representation bias"),
        (axes[2], "tau_error", "Finite-sample absolute error"),
    ):
        for method in ("joint", "x_only"):
            rows = [row for row in representation_rows if row.method == method]
            x = np.array([row.d_joint for row in rows])
            if metric == "d_joint":
                y = x
            elif metric == "oracle_bias":
                y = np.abs([row.oracle_bias for row in rows])
            else:
                y = np.abs([row.tau_error for row in rows])
            ax.scatter(x, y, s=22, alpha=0.65, color=colors[method], label=method)
        ax.set_xlabel(r"held-out $D_{XW}$")
        ax.set_ylabel(title)
        ax.set_title(title)
    axes[0].legend()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            base.FIGURES_DIR / f"joint_vs_xonly_representation_comparison.{extension}",
            dpi=200,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--strata", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.3, 1.0, 2.0])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.n = 1200
        args.steps = 60
        args.reps = 2
        args.sigmas = [0.3]
        args.gammas = [2.0]

    representation_rows: list[RepresentationRecord] = []
    baseline_rows: list[BaselineRecord] = []
    for sigma_w in args.sigmas:
        for gamma_u in args.gammas:
            for rep in range(args.reps):
                reps, bases = run_once(
                    sigma_w,
                    gamma_u,
                    rep,
                    args.n,
                    args.strata,
                    args.steps,
                )
                representation_rows.extend(reps)
                baseline_rows.extend(bases)
                print(
                    f"sigma_w={sigma_w} gamma_u={gamma_u} rep={rep + 1}/{args.reps}",
                    flush=True,
                )
    summary = summarize(representation_rows, baseline_rows, args)
    if not args.no_write:
        write_outputs(representation_rows, baseline_rows, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
