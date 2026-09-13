#!/usr/bin/env python3
"""Dimension sweep d in {5,10,20,30,40,50,100} (registration v4, frozen).

Corrected direct-X template (registration v3 form, variant v2_sine) on
Gaussian covariates with controllable dimension. Three matched arms.
Output: results/benchmark_suite_dimsweep.json
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import learned_representation_balancing_experiment as base  # noqa: E402
from benchmark_suite_tabular import (  # noqa: E402
    _standardize, _grams, train_net, final_train_disc)

from scipy.stats import wilcoxon  # noqa: E402

REG_SHA = "ea19d60a59d33fea3b5fc45b4c7918dde1cac5ba11bb51cc41e4a3a8eac32240"
SEED_BASE = 2026081907
N, REPS, STEPS, M = 1200, 6, 120, 5
DIMS = (5, 10, 20, 30, 40, 50, 100)
LAMBDAS = (0.0, 0.5, 1.0)
ARMS = ("joint", "constrained", "x_only")


def _sig(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -35.0, 35.0)))


def directions3(d: int):
    rng = np.random.default_rng(SEED_BASE + 97 * d)
    q, _ = np.linalg.qr(rng.standard_normal((d, 3)))
    return q[:, 0], q[:, 1], q[:, 2]


def simulate(d, lam_f, rng, n):
    x = rng.standard_normal((n, d))
    b_c, b_t, b_y = directions3(d)
    def std(v):
        return (v - v.mean()) / (v.std() + 1e-8)
    r, q, by = std(x @ b_c), std(x @ b_t), std(x @ b_y)
    u = r + 0.5 * np.sin(r) + 0.5 * rng.standard_normal(n)
    w = u + 0.5 * rng.standard_normal(n)
    a = rng.binomial(1, 0.05 + 0.90 * _sig(0.8 * u + 1.5 * q)).astype(float)
    y_struct = 1.0 * a + 2.0 * u + lam_f * (q + 0.5 * by)
    y = y_struct + rng.standard_normal(n)
    return dict(X=x, U=u, W=w, A=a, Y_struct=y_struct, Y=y)


def delta_vec(strata, a, u, m):
    out = []
    for z in range(m):
        in_z = strata == z
        u1, u0 = u[in_z & (a == 1)], u[in_z & (a == 0)]
        if u1.size and u0.size:
            out.append(dict(p_z=float(in_z.mean()),
                            delta_z=float(u1.mean() - u0.mean())))
    return out


def run_once(d, lam_f, rep):
    seed = SEED_BASE + 7919 * rep + 101 * d + 401 * int(lam_f * 2)
    rng = np.random.default_rng(seed)
    data = simulate(d, lam_f, rng, N)
    perm = rng.permutation(N)
    tr, ev = perm[: N // 2], perm[N // 2:]
    xs_tr, xs_ev = _standardize(data["X"][tr], data["X"][ev])
    ws_tr, ws_ev = _standardize(data["W"][tr][:, None], data["W"][ev][:, None])
    a_tr, a_ev = data["A"][tr], data["A"][ev]
    y_ev, ystr_ev, u_ev = data["Y"][ev], data["Y_struct"][ev], data["U"][ev]
    kx_tr, kx_ev = _grams(xs_tr, xs_ev, seed + 101)
    kw_tr, kw_ev = _grams(ws_tr, ws_ev, seed + 202)
    kj_tr, kj_ev = kx_tr * kw_tr, kx_ev * kw_ev
    net_x = train_net(xs_tr, a_tr, [(kx_tr, "objective")], M, STEPS, seed)
    x_cap = final_train_disc(net_x, xs_tr, a_tr, kx_tr)
    nets = dict(
        x_only=net_x,
        joint=train_net(xs_tr, a_tr, [(kj_tr, "objective")], M, STEPS, seed),
        constrained=train_net(xs_tr, a_tr,
                              [(kw_tr, "objective"), (kx_tr, "capped_x")],
                              M, STEPS, seed, x_cap=x_cap))
    row = dict(d=d, lam_f=lam_f, rep=rep)
    for name, net in nets.items():
        strata = base.assign_strata(net, xs_ev)
        est, _ = base.stratified_tau(strata, a_ev, y_ev, M)
        oracle, _ = base.stratified_tau(strata, a_ev, ystr_ev, M)
        row[name] = dict(
            tau_error=float(est - 1.0), oracle_bias=float(oracle - 1.0),
            d_x=float(math.sqrt(max(base.hard_delta_sq(strata, a_ev, kx_ev, M), 0))),
            d_w=float(math.sqrt(max(base.hard_delta_sq(strata, a_ev, kw_ev, M), 0))),
            strata=delta_vec(strata, a_ev, u_ev, M))
    return row


def decide(rows, arm):
    red = [abs(r["x_only"]["oracle_bias"]) - abs(r[arm]["oracle_bias"])
           for r in rows]
    wins = [abs(r[arm]["oracle_bias"]) < abs(r["x_only"]["oracle_bias"])
            and abs(r[arm]["tau_error"]) < abs(r["x_only"]["tau_error"])
            for r in rows]
    return dict(n_pairs=len(wins), win_rate=float(np.mean(wins)),
                wilcoxon_p=float(wilcoxon(red, alternative="greater").pvalue),
                mean_bias_reduction=float(np.mean(red)))


def main() -> None:
    rows = []
    for lam_f in LAMBDAS:
        for d in DIMS:
            for rep in range(REPS):
                rows.append(run_once(d, lam_f, rep))
                print(f"lam={lam_f} d={d} rep={rep + 1}", flush=True)
    per_lambda = {}
    for lam_f in LAMBDAS:
        sel = [r for r in rows if r["lam_f"] == lam_f]
        curves = {}
        for d in DIMS:
            cell = [r for r in sel if r["d"] == d]
            curves[str(d)] = {m: dict(
                mean_abs_bias=float(np.mean(
                    [abs(r[m]["oracle_bias"]) for r in cell])),
                rmse=float(np.sqrt(np.mean(
                    [r[m]["tau_error"] ** 2 for r in cell]))),
                mean_d_w=float(np.mean([r[m]["d_w"] for r in cell])))
                for m in ARMS}
        per_lambda[str(lam_f)] = dict(
            pooled_joint=decide(sel, "joint"),
            pooled_constrained=decide(sel, "constrained"),
            curves=curves)
    summary = dict(provenance=dict(
        script="code/benchmark_suite_dimsweep.py",
        registration="memo/2026-08-19-rerun-registration-v4-dimsweep.md",
        registration_sha256=REG_SHA, base_seed=SEED_BASE, n=N, reps=REPS,
        steps=STEPS, strata=M, dims=list(DIMS), lambdas=list(LAMBDAS)),
        per_lambda=per_lambda, runs=rows)
    out = pathlib.Path(base.RESULTS_DIR) / "benchmark_suite_dimsweep.json"
    out.write_text(json.dumps(summary, indent=1))
    for lam, v in per_lambda.items():
        print(f"lam={lam} pooled joint={v['pooled_joint']} "
              f"constrained={v['pooled_constrained']}")


if __name__ == "__main__":
    main()
