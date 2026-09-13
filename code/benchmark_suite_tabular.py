#!/usr/bin/env python3
"""Preregistered tabular benchmark suite (Track 1).

Contract: memo/2026-08-18-benchmark-suite-preregistration.md (frozen).
Sources: real ACIC 2016 covariates (semi-synthetic augmentation: U, W, A, Y
generated) and two self-contained CausalPy-style SCM families. Arms: joint
product-kernel replication (A1), constrained D_W-minimization with the
X-imbalance anchor c = train D_X of the x_only learner (A2), and the x_only
baseline. Estimation is the identical H-stratified plug-in on the held-out
fold in every arm. Mandatory logging: held-out D_XW, D_X, D_W, per-stratum
oracle shifts delta_z, T = sum p_z delta_z, S2 = sum p_z delta_z^2.

Decision rule (frozen): per source, each arm passes against x_only if
(1) paired bias-and-RMSE win rate >= 0.70, (2) one-sided Wilcoxon on paired
|oracle bias| reductions p < 0.05, (3) both negative controls show no gain.

Outputs: results/benchmark_suite_tabular_summary.json, _runs.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import learned_representation_balancing_experiment as base  # noqa: E402

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None

TRUE_TAU = 1.0
BASE_SEED = 2026081901
PEN_X = 25.0
REPO = pathlib.Path(__file__).resolve().parents[4]
ACIC_CSV = (REPO / "research/papers/DataFusionPFN/code/"
            "acic2016_ir_benchmark_2026-08-14/x.csv")
DGP_VARIANTS = ("v1_linear", "v2_sine", "v3_interact", "v4_hetero")
ARMS = ("joint", "constrained", "x_only")


@dataclass
class Row:
    source: str
    variant: str
    regime: str
    rep: int
    method: str
    estimate: float
    tau_error: float
    oracle_bias: float
    d_joint: float
    d_x: float
    d_w: float
    t_sum: float
    s2_sum: float
    n_neg_shift: int
    dropped_mass: float


_ACIC_CACHE: np.ndarray | None = None


def acic_matrix() -> np.ndarray:
    global _ACIC_CACHE
    if _ACIC_CACHE is None:
        import pandas as pd
        frame = pd.read_csv(ACIC_CSV)
        frame = pd.get_dummies(frame, drop_first=True)
        _ACIC_CACHE = frame.to_numpy(dtype=float)
    return _ACIC_CACHE


def directions(d: int, tag: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(BASE_SEED + 97 * d + 11 * tag)
    q, _ = np.linalg.qr(rng.standard_normal((d, 2)))
    return q[:, 0], q[:, 1]


def _sig(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(v, -35.0, 35.0)))


def draw_covariates(source: str, rng: np.random.Generator, n: int) -> np.ndarray:
    if source == "acic":
        mat = acic_matrix()
        idx = rng.choice(mat.shape[0], size=n, replace=False)
        x = mat[idx]
        return (x - x.mean(0)) / (x.std(0) + 1e-8)
    if source == "scm_dense":
        return rng.standard_normal((n, 25))
    if source == "scm_mixed":
        cont = rng.standard_normal((n, 15))
        lat = rng.standard_normal((n, 3))
        bins = (lat @ rng.standard_normal((3, 10)) > 0).astype(float)
        return np.column_stack([cont, bins - bins.mean(0)])
    raise ValueError(source)


def simulate(source: str, variant: str, regime: str,
             rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    x = draw_covariates(source, rng, n)
    d = x.shape[1]
    b_c, b_t = directions(d, tag={"acic": 1, "scm_dense": 2, "scm_mixed": 3}[source])
    r, q = x @ b_c, x @ b_t
    r = (r - r.mean()) / (r.std() + 1e-8)
    q = (q - q.mean()) / (q.std() + 1e-8)
    eps_u = rng.standard_normal(n)
    if regime == "x_signal_null":
        u = eps_u
    elif variant == "v1_linear":
        u = r + 0.5 * eps_u
    elif variant == "v2_sine":
        u = r + 0.5 * np.sin(r) + 0.5 * eps_u
    elif variant == "v3_interact":
        u = r + 0.3 * r * np.tanh(q) + 0.5 * eps_u
    elif variant == "v4_hetero":
        u = r + 0.5 * eps_u
    else:
        raise ValueError(variant)
    if regime == "proxy_null":
        w = rng.standard_normal(n)
    elif variant == "v4_hetero":
        w = u + (0.3 + 0.4 * _sig(r)) * rng.standard_normal(n)
    else:
        w = u + 0.5 * rng.standard_normal(n)
    lin = 0.8 * u + 1.5 * q
    if variant == "v3_interact":
        lin = 0.8 * u + 1.0 * q + 0.5 * u * np.tanh(q)
    a = rng.binomial(1, 0.05 + 0.90 * _sig(lin)).astype(float)
    y_struct = TRUE_TAU * a + 2.0 * u
    y = y_struct + rng.standard_normal(n)
    return {"X": x, "U": u, "W": w, "A": a, "Y_struct": y_struct, "Y": y}


def _standardize(tr: np.ndarray, ev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = tr.mean(0), tr.std(0) + 1e-8
    return (tr - mean) / std, (ev - mean) / std


def _grams(tr: np.ndarray, ev: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    bw = base.median_bandwidth(tr, rng)
    return base.rbf_gram(tr, tr, bw), base.rbf_gram(ev, ev, bw)


def train_net(xs: np.ndarray, a: np.ndarray, terms: list[tuple[np.ndarray, str]],
              m: int, steps: int, seed: int, x_cap: float | None = None):
    """terms: list of (gram, role) with role in {objective, capped_x}."""
    torch.manual_seed(seed)
    net = base.StratumNet(d_in=xs.shape[1], m_strata=m)
    x_t = torch.tensor(xs, dtype=torch.float32)
    a_t = torch.tensor(a, dtype=torch.float32)
    grams = [(torch.tensor(g, dtype=torch.float32), role) for g, role in terms]
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    overlap_floor, mass_floor = 0.09, 0.5 / m
    for step in range(steps):
        temp = 1.0 * 0.25 ** (step / max(steps - 1, 1))
        soft = torch.softmax(net(x_t) / temp, dim=1)
        loss = torch.zeros(())
        p_hat = pi_hat = None
        for k_t, role in grams:
            disc, p_hat, pi_hat = base.soft_delta_sq(soft, a_t, k_t)
            if role == "objective":
                loss = loss + disc
            elif role == "capped_x":
                loss = loss + PEN_X * torch.relu(disc - x_cap)
        loss = loss + (p_hat * torch.relu(
            overlap_floor - pi_hat * (1.0 - pi_hat))).sum()
        loss = loss + torch.relu(mass_floor - p_hat).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def final_train_disc(net, xs, a, gram) -> float:
    with torch.no_grad():
        soft = torch.softmax(net(torch.tensor(xs, dtype=torch.float32)) / 0.25, 1)
        disc, _, _ = base.soft_delta_sq(
            soft, torch.tensor(a, dtype=torch.float32),
            torch.tensor(gram, dtype=torch.float32))
    return float(disc)


def shift_stats(strata, a, u, m) -> tuple[float, float, int]:
    t_sum = s2 = 0.0
    n_neg = 0
    n = len(a)
    for z in range(m):
        in_z = strata == z
        if in_z.sum() == 0:
            continue
        u1, u0 = u[in_z & (a == 1)], u[in_z & (a == 0)]
        if u1.size == 0 or u0.size == 0:
            continue
        dz = float(u1.mean() - u0.mean())
        p_z = in_z.mean()
        t_sum += p_z * dz
        s2 += p_z * dz * dz
        n_neg += int(dz < 0)
    return t_sum, s2, n_neg


def run_once(source: str, variant: str, regime: str, rep: int,
             n: int, m: int, steps: int) -> list[Row]:
    seed = (BASE_SEED + 7919 * rep + 1009 * DGP_VARIANTS.index(variant)
            + 104729 * {"primary": 0, "proxy_null": 1, "x_signal_null": 2}[regime]
            + 13 * {"acic": 1, "scm_dense": 2, "scm_mixed": 3}[source])
    rng = np.random.default_rng(seed)
    data = simulate(source, variant, regime, rng, n)
    perm = rng.permutation(n)
    tr, ev = perm[: n // 2], perm[n // 2:]
    xs_tr, xs_ev = _standardize(data["X"][tr], data["X"][ev])
    ws_tr, ws_ev = _standardize(data["W"][tr][:, None], data["W"][ev][:, None])
    a_tr, a_ev = data["A"][tr], data["A"][ev]
    y_ev, ystr_ev, u_ev = data["Y"][ev], data["Y_struct"][ev], data["U"][ev]
    kx_tr, kx_ev = _grams(xs_tr, xs_ev, seed + 101)
    kw_tr, kw_ev = _grams(ws_tr, ws_ev, seed + 202)
    kj_tr, kj_ev = kx_tr * kw_tr, kx_ev * kw_ev

    net_x = train_net(xs_tr, a_tr, [(kx_tr, "objective")], m, steps, seed)
    x_cap = final_train_disc(net_x, xs_tr, a_tr, kx_tr)
    nets = {
        "x_only": net_x,
        "joint": train_net(xs_tr, a_tr, [(kj_tr, "objective")], m, steps, seed),
        "constrained": train_net(
            xs_tr, a_tr, [(kw_tr, "objective"), (kx_tr, "capped_x")],
            m, steps, seed, x_cap=x_cap),
    }
    rows: list[Row] = []
    for method, net in nets.items():
        strata = base.assign_strata(net, xs_ev)
        est, dropped = base.stratified_tau(strata, a_ev, y_ev, m)
        oracle, _ = base.stratified_tau(strata, a_ev, ystr_ev, m)
        t_sum, s2, n_neg = shift_stats(strata, a_ev, u_ev, m)
        rows.append(Row(
            source=source, variant=variant, regime=regime, rep=rep,
            method=method, estimate=est, tau_error=est - TRUE_TAU,
            oracle_bias=oracle - TRUE_TAU,
            d_joint=math.sqrt(max(base.hard_delta_sq(strata, a_ev, kj_ev, m), 0.0)),
            d_x=math.sqrt(max(base.hard_delta_sq(strata, a_ev, kx_ev, m), 0.0)),
            d_w=math.sqrt(max(base.hard_delta_sq(strata, a_ev, kw_ev, m), 0.0)),
            t_sum=t_sum, s2_sum=s2, n_neg_shift=n_neg, dropped_mass=dropped))
    return rows


def decide(rows: list[Row], arm: str) -> dict:
    pairs = {}
    for row in rows:
        pairs.setdefault((row.source, row.variant, row.rep), {})[row.method] = row
    bias_red, wins = [], []
    for cell in pairs.values():
        if arm not in cell or "x_only" not in cell:
            continue
        ours, ref = cell[arm], cell["x_only"]
        bias_red.append(abs(ref.oracle_bias) - abs(ours.oracle_bias))
        wins.append(abs(ours.oracle_bias) < abs(ref.oracle_bias)
                    and abs(ours.tau_error) < abs(ref.tau_error))
    win_rate = float(np.mean(wins)) if wins else float("nan")
    p_val = None
    if wilcoxon is not None and len(bias_red) >= 8 and np.any(np.array(bias_red) != 0):
        p_val = float(wilcoxon(bias_red, alternative="greater").pvalue)
    return dict(n_pairs=len(wins), win_rate_bias_and_rmse=win_rate,
                wilcoxon_p_bias_reduction=p_val,
                mean_bias_reduction=float(np.mean(bias_red)) if bias_red else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--strata", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.reps, args.steps = 500, 1, 25

    sources = ("acic", "scm_dense", "scm_mixed")
    rows: list[Row] = []
    for source in sources:
        for variant in DGP_VARIANTS:
            for rep in range(args.reps):
                rows.extend(run_once(source, variant, "primary", rep,
                                     args.n, args.strata, args.steps))
                print(f"{source}/{variant} rep={rep + 1}/{args.reps}", flush=True)
    controls: list[Row] = []
    for source in ("acic", "scm_dense"):
        for regime in ("proxy_null", "x_signal_null"):
            for rep in range(max(args.reps // 2, 1)):
                controls.extend(run_once(source, "v2_sine", regime, rep,
                                         args.n, args.strata, args.steps))
                print(f"control {source}/{regime} rep={rep + 1}", flush=True)

    summary = dict(provenance=dict(
        script="code/benchmark_suite_tabular.py",
        preregistration="memo/2026-08-18-benchmark-suite-preregistration.md",
        base_seed=BASE_SEED, n=args.n, reps=args.reps, steps=args.steps,
        strata=args.strata, arms=list(ARMS), pen_x=PEN_X,
        acic_csv=str(ACIC_CSV), true_tau=TRUE_TAU,
        anchor="constrained arm caps train soft D_X^2 at x_only's achieved value",
        matched_learner_contract="identical net, seed, optimizer, schedule; "
                                 "only the balance objective differs"))
    per_source = {}
    for source in sources:
        src_rows = [r for r in rows if r.source == source]
        agg = {}
        for method in ARMS:
            sel = [r for r in src_rows if r.method == method]
            agg[method] = dict(
                mean_abs_bias=float(np.mean([abs(r.oracle_bias) for r in sel])),
                rmse=float(np.sqrt(np.mean([r.tau_error ** 2 for r in sel]))),
                mean_d_w=float(np.mean([r.d_w for r in sel])),
                mean_d_x=float(np.mean([r.d_x for r in sel])),
                mean_d_joint=float(np.mean([r.d_joint for r in sel])),
                mean_T=float(np.mean([r.t_sum for r in sel])),
                mean_S2=float(np.mean([r.s2_sum for r in sel])),
                neg_shift_frac=float(np.mean([r.n_neg_shift for r in sel]))
                / args.strata)
        per_source[source] = dict(
            aggregates=agg,
            decision_joint_vs_xonly=decide(src_rows, "joint"),
            decision_constrained_vs_xonly=decide(src_rows, "constrained"))
    control_summary = {}
    for regime in ("proxy_null", "x_signal_null"):
        creg = [r for r in controls if r.regime == regime]
        control_summary[regime] = {
            arm: decide(creg, arm)["win_rate_bias_and_rmse"]
            for arm in ("joint", "constrained")}
    summary["per_source"] = per_source
    summary["negative_controls_win_rates"] = control_summary

    base.RESULTS_DIR.mkdir(exist_ok=True)
    out = base.RESULTS_DIR / "benchmark_suite_tabular_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    with (base.RESULTS_DIR / "benchmark_suite_tabular_runs.csv").open(
            "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows + controls:
            writer.writerow(asdict(row))
    print(json.dumps(summary["per_source"], indent=1)[:2000])
    print("controls:", json.dumps(control_summary))


if __name__ == "__main__":
    main()
