#!/usr/bin/env python3
"""Kallus--Mao--Uehara 2022 single-proxy ablation experiment.

IMPORTANT LABEL: this is an external published proximal DGP with a
single-proxy ablation, not a native single-proxy benchmark.  The original
experiment uses both negative controls W and Z.  This ablation generates the
full published joint DGP but hides Z completely; both representation learners
and every available estimator receive only X, W, A, and Y.  W is used in the
joint balancing loss and held-out diagnostics, never in the final adjustment.

Primary source:
  Kallus, Mao, and Uehara (2022), "Causal Inference Under Unmeasured
  Confounding With Negative Controls: A Minimax Learning Approach,"
  Section 9.1 and Appendix I, arXiv:2103.14029v4.
  https://arxiv.org/abs/2103.14029

Source completions and deliberate deviations:
  1. Appendix I dimensionally writes alpha_a=kappa_a=I_d although A is
     scalar.  We use the dimensionally conformable all-one vector e_d.
  2. The paper leaves the invertible matrix G unspecified.  We use one fixed,
     seed-generated orthogonal matrix for every configuration and replication.
  3. The paper says to choose Sigma_wz and mu_a so Eq. (47) is independent of
     A and Z but does not print their values.  We use the exact Gaussian
     completion Sigma_wz=Sigma_wu Sigma_u^{-1} Sigma_uz and
     mu_a=Sigma_wu Sigma_u^{-1} kappa_a.
  4. The paper targets J=E[Y(1)] and observes both W and Z.  At the user's
     direction, this ablation hides Z and targets the ATE, which equals 1 from
     the additive coefficient on A.

Outputs:
  results/kmu_2022_single_proxy_ablation_summary.json
  results/kmu_2022_single_proxy_ablation_runs.csv
  figures/kmu_2022_single_proxy_ablation_comparison.(png|pdf)
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


DIMENSION = 60
TRUE_ATE = 1.0
BASE_SEED = 210314029
SOURCE_URL = "https://arxiv.org/abs/2103.14029"
SOURCE_LABEL = (
    "external published proximal DGP; single-proxy ablation, "
    "not a native single-proxy benchmark"
)


@dataclass
class RepresentationRecord:
    transform: str
    n: int
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
    transform: str
    n: int
    rep: int
    method: str
    estimate: float
    tau_error: float


@lru_cache(maxsize=1)
def orthogonal_g() -> np.ndarray:
    rng = np.random.default_rng(BASE_SEED + 11)
    raw = rng.standard_normal((DIMENSION, DIMENSION))
    q, r = np.linalg.qr(raw)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


@lru_cache(maxsize=1)
def gaussian_parameters() -> dict[str, np.ndarray | float]:
    d = DIMENSION
    eye = np.eye(d)
    ones = np.ones(d)
    ee = np.outer(ones, ones)
    sigma_w = 0.1 * (eye + ee)
    sigma_z = 0.1 * (eye + ee)
    sigma_u = 0.1 * (eye + ee)
    sigma_wu = 0.1 * ee
    sigma_zu = 0.1 * ee
    sigma_wz = sigma_wu @ np.linalg.solve(sigma_u, sigma_zu.T)
    mu_a = sigma_wu @ np.linalg.solve(sigma_u, ones)
    covariance = np.block(
        [
            [sigma_w, sigma_wz, sigma_wu],
            [sigma_wz.T, sigma_z, sigma_zu],
            [sigma_wu.T, sigma_zu.T, sigma_u],
        ]
    )
    min_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
    if min_eigenvalue <= 0:
        raise AssertionError(f"published Gaussian completion is not positive definite: {min_eigenvalue}")
    chol = np.linalg.cholesky(covariance)
    conditional_z_residual = float(
        np.linalg.norm(sigma_wz - sigma_wu @ np.linalg.solve(sigma_u, sigma_zu.T))
    )
    conditional_a_residual = float(
        np.linalg.norm(mu_a - sigma_wu @ np.linalg.solve(sigma_u, ones))
    )
    return {
        "chol": chol,
        "mu_a": mu_a,
        "min_eigenvalue": min_eigenvalue,
        "conditional_z_residual": conditional_z_residual,
        "conditional_a_residual": conditional_a_residual,
    }


def _transform(values: np.ndarray, name: str) -> np.ndarray:
    mixed = values @ orthogonal_g().T
    if name == "sin":
        return np.sin(mixed)
    if name == "cube":
        return mixed**3
    raise ValueError(f"unknown transformation: {name}")


def simulate_kmu(
    rng: np.random.Generator,
    n: int,
    transform: str,
) -> dict[str, np.ndarray]:
    """Generate the complete Appendix-I DGP, including the hidden Z."""
    d = DIMENSION
    ones = np.ones(d)
    xp = math.sqrt(0.5) * rng.standard_normal((n, d))
    propensity = 1.0 / (1.0 + np.exp(xp @ (0.5 * ones - 0.05 * ones)))
    a = rng.binomial(1, propensity).astype(float)

    pars = gaussian_parameters()
    common = 0.2 * ones + xp + a[:, None] * ones
    mean_w = 0.2 * ones + xp + a[:, None] * np.asarray(pars["mu_a"])
    mean = np.concatenate([mean_w, common, common], axis=1)
    latent_draw = mean + rng.standard_normal((n, 3 * d)) @ np.asarray(pars["chol"]).T
    wp = latent_draw[:, :d]
    zp = latent_draw[:, d : 2 * d]
    u = latent_draw[:, 2 * d :]

    structural_y = a + xp.sum(axis=1) + u.sum(axis=1) + wp.sum(axis=1)
    y = structural_y + rng.standard_normal(n)
    x = _transform(xp, transform)
    z = _transform(zp, transform)
    w = _transform(wp, transform)
    arrays = (xp, wp, zp, u, x, z, w, propensity, a, structural_y, y)
    if not all(np.isfinite(value).all() for value in arrays):
        raise FloatingPointError(f"non-finite draw under g={transform}")
    return {
        "Xp": xp,
        "Wp": wp,
        "Zp": zp,
        "U": u,
        "X": x,
        "Z": z,
        "W": w,
        "A": a,
        "Y_struct": structural_y,
        "Y": y,
    }


def _standardize(train: np.ndarray, evaluate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def train_representation_d(
    xs: np.ndarray,
    a: np.ndarray,
    gram: np.ndarray,
    m_strata: int,
    steps: int,
    seed: int,
) -> base.StratumNet:
    """Existing training objective with d_in changed from 4 to 60."""
    torch.manual_seed(seed)
    net = base.StratumNet(d_in=DIMENSION, m_strata=m_strata)
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


def _seed(n: int, rep: int) -> int:
    # Identical raw sample for sin and cube at a given (n, rep).
    return BASE_SEED + 7919 * rep + 13 * n


def run_once(
    transform: str,
    n: int,
    rep: int,
    m_strata: int,
    steps: int,
) -> tuple[list[RepresentationRecord], list[BaselineRecord]]:
    seed = _seed(n, rep)
    rng = np.random.default_rng(seed)
    data = simulate_kmu(rng, n, transform)
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
    ws_tr, ws_ev = _standardize(w_tr, w_ev)
    vs_tr, vs_ev = np.column_stack([xs_tr, ws_tr]), np.column_stack([xs_ev, ws_ev])
    joint_train, joint_eval, _ = _gram_pair(vs_tr, vs_ev, seed + 101)
    x_train, x_eval, _ = _gram_pair(xs_tr, xs_ev, seed + 202)
    _, w_eval, _ = _gram_pair(ws_tr, ws_ev, seed + 303)

    nets = {
        "joint": train_representation_d(xs_tr, a_tr, joint_train, m_strata, steps, seed),
        "x_only": train_representation_d(xs_tr, a_tr, x_train, m_strata, steps, seed),
    }
    representation_rows: list[RepresentationRecord] = []
    for method, net in nets.items():
        strata = base.assign_strata(net, xs_ev)
        estimate, dropped = base.stratified_tau(strata, a_ev, y_ev, m_strata)
        oracle_effect, oracle_dropped = base.stratified_tau(
            strata, a_ev, y_struct_ev, m_strata
        )
        if abs(dropped - oracle_dropped) > 1e-12:
            raise AssertionError("oracle and noisy estimates use different strata")
        representation_rows.append(
            RepresentationRecord(
                transform=transform,
                n=n,
                rep=rep,
                method=method,
                estimate=estimate,
                tau_error=estimate - TRUE_ATE,
                oracle_effect=oracle_effect,
                oracle_bias=oracle_effect - TRUE_ATE,
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

    baseline_specs = {
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
        BaselineRecord(transform, n, rep, method, estimate, estimate - TRUE_ATE)
        for method, estimate in baseline_specs.items()
    ]
    return representation_rows, baseline_rows


def _aggregate(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(errors.mean()),
        "mean_abs": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "se": float(errors.std(ddof=1) / math.sqrt(len(errors))) if len(errors) > 1 else 0.0,
    }


def summarize(
    representation_rows: list[RepresentationRecord],
    baseline_rows: list[BaselineRecord],
    args: argparse.Namespace,
) -> dict:
    pars = gaussian_parameters()
    source_deviations = [
        "alpha_a and kappa_a repaired from dimensionally incompatible I_d to e_d",
        "unspecified invertible G fixed to one seed-generated orthogonal matrix",
        "Sigma_wz and mu_a completed exactly from Appendix-I Eq. (47)",
        "Z hidden and estimand changed from E[Y(1)] to ATE=1",
    ]
    payload: dict = {
        "provenance": {
            "label": SOURCE_LABEL,
            "source_url": SOURCE_URL,
            "source_location": "Section 9.1 and Appendix I, arXiv v4",
            "script": "code/kmu_2022_single_proxy_ablation_experiment.py",
            "dimension": DIMENSION,
            "base_seed": BASE_SEED,
            "seed_formula": "210314029 + 7919*rep + 13*n; shared across transformations",
            "transforms": args.transforms,
            "sample_sizes": args.sample_sizes,
            "reps": args.reps,
            "steps": args.steps,
            "strata": args.strata,
            "target": "ATE=1",
            "source_completions_and_deviations": source_deviations,
            "z_generated_but_hidden": True,
            "w_used_by_final_estimator": False,
            "gaussian_min_eigenvalue": pars["min_eigenvalue"],
            "conditional_z_residual": pars["conditional_z_residual"],
            "conditional_a_residual": pars["conditional_a_residual"],
        },
        "configs": {},
        "predeclared_success": {
            "criterion": (
                "joint has lower mean held-out D_XW, lower oracle absolute bias, "
                "and lower finite-sample RMSE than x_only in a majority of four configs"
            )
        },
    }
    config_wins = []
    for transform in args.transforms:
        for n in args.sample_sizes:
            key = f"transform={transform}|n={n}"
            cfg_rep = [
                row for row in representation_rows if row.transform == transform and row.n == n
            ]
            methods = {}
            by_rep = {}
            for method in ("joint", "x_only"):
                rows = [row for row in cfg_rep if row.method == method]
                by_rep[method] = {row.rep: row for row in rows}
                methods[method] = {
                    "tau_error": _aggregate(np.array([row.tau_error for row in rows])),
                    "oracle_bias": _aggregate(np.array([row.oracle_bias for row in rows])),
                    "mean_d_joint": float(np.mean([row.d_joint for row in rows])),
                    "mean_d_x": float(np.mean([row.d_x for row in rows])),
                    "mean_d_w": float(np.mean([row.d_w for row in rows])),
                    "mean_dropped_mass": float(np.mean([row.dropped_mass for row in rows])),
                }
            cfg_base = [
                row for row in baseline_rows if row.transform == transform and row.n == n
            ]
            baselines = {
                method: _aggregate(
                    np.array([row.tau_error for row in cfg_base if row.method == method])
                )
                for method in ("naive", "full_x", "oracle_xu")
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
            config_win = (
                methods["joint"]["mean_d_joint"] < methods["x_only"]["mean_d_joint"]
                and methods["joint"]["oracle_bias"]["mean_abs"]
                < methods["x_only"]["oracle_bias"]["mean_abs"]
                and methods["joint"]["tau_error"]["rmse"]
                < methods["x_only"]["tau_error"]["rmse"]
            )
            config_wins.append(config_win)
            payload["configs"][key] = {
                "representations": methods,
                "baselines": baselines,
                "paired_all_three_win_rate": float(np.mean(paired_wins)),
                "config_pass": bool(config_win),
            }
    payload["predeclared_success"]["configs_passed"] = int(sum(config_wins))
    payload["predeclared_success"]["total_configs"] = len(config_wins)
    payload["predeclared_success"]["passed"] = bool(sum(config_wins) > len(config_wins) / 2)
    return payload


def write_outputs(
    representation_rows: list[RepresentationRecord],
    summary: dict,
) -> None:
    base.RESULTS_DIR.mkdir(exist_ok=True)
    base.FIGURES_DIR.mkdir(exist_ok=True)
    (base.RESULTS_DIR / "kmu_2022_single_proxy_ablation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    with (base.RESULTS_DIR / "kmu_2022_single_proxy_ablation_runs.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(representation_rows[0]).keys()))
        writer.writeheader()
        for row in representation_rows:
            writer.writerow(asdict(row))

    labels = [key.replace("transform=", "").replace("|n=", "\n$n=") + "$" for key in summary["configs"]]
    metrics = (
        ("mean_d_joint", "Held-out $D_{XW}$"),
        ("oracle_bias", "Oracle absolute bias"),
        ("tau_error", "Finite-sample RMSE"),
    )
    colors = {"joint": "#0f766e", "x_only": "#d97706"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    positions = np.arange(len(labels))
    width = 0.36
    for ax, (metric, title) in zip(axes, metrics):
        for offset, method in ((-width / 2, "joint"), (width / 2, "x_only")):
            vals = []
            for cfg in summary["configs"].values():
                method_summary = cfg["representations"][method]
                if metric == "mean_d_joint":
                    vals.append(method_summary[metric])
                elif metric == "oracle_bias":
                    vals.append(method_summary[metric]["mean_abs"])
                else:
                    vals.append(method_summary[metric]["rmse"])
            ax.bar(positions + offset, vals, width, color=colors[method], label=method)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(title)
    axes[0].legend()
    fig.suptitle("External proximal DGP, single-proxy ablation")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for extension in ("png", "pdf"):
        fig.savefig(
            base.FIGURES_DIR / f"kmu_2022_single_proxy_ablation_comparison.{extension}",
            dpi=200,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[400, 1200])
    parser.add_argument("--transforms", nargs="+", choices=["sin", "cube"], default=["sin", "cube"])
    parser.add_argument("--reps", type=int, default=6)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--strata", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.sample_sizes = [200]
        args.transforms = ["sin", "cube"]
        args.reps = 1
        args.steps = 40

    representation_rows: list[RepresentationRecord] = []
    baseline_rows: list[BaselineRecord] = []
    for transform in args.transforms:
        for n in args.sample_sizes:
            for rep in range(args.reps):
                reps, bases = run_once(transform, n, rep, args.strata, args.steps)
                representation_rows.extend(reps)
                baseline_rows.extend(bases)
                print(f"transform={transform} n={n} rep={rep + 1}/{args.reps}", flush=True)
    summary = summarize(representation_rows, baseline_rows, args)
    if not args.no_write:
        write_outputs(representation_rows, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
