#!/usr/bin/env python3
"""Tabular suite v2 with direct X-to-Y effects (registration v3, frozen).

Y = tau A + 2U + lambda_f {q(X) + 0.5 b_y' X} + eps, lambda_f in {0, .5, 1}.
q drives treatment, so for lambda_f > 0 it is a genuine observed confounder;
b_y is a third orthogonal direction. Everything else matches the first
suite. Output: results/benchmark_suite_tabular_v2_directx.json
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
    draw_covariates, _standardize, _grams, train_net, final_train_disc)

from scipy.stats import wilcoxon  # noqa: E402

REG_SHA = "7110b10c3f077d37aa3a0881243a7257109832cfebfc014fe445dd4d5071cb8b"
ACIC_SHA = "0d6387ad45d23e54b11cbf967248fb68ad5e578db80d193208b8b4c98b1ed0c1"
SEED_BASE = 2026081905
N, REPS, STEPS, M = 1200, 6, 120, 5
LAMBDAS = (0.0, 0.5, 1.0)
SOURCES = ("acic", "scm_dense", "scm_mixed")
VARIANTS = ("v1_linear", "v2_sine")
ARMS = ("joint", "constrained", "x_only")


def _sig(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -35.0, 35.0)))


def directions3(d: int, tag: int):
    rng = np.random.default_rng(SEED_BASE + 97 * d + 11 * tag)
    q, _ = np.linalg.qr(rng.standard_normal((d, 3)))
    return q[:, 0], q[:, 1], q[:, 2]


def simulate_v2(source, variant, lam_f, rng, n):
    x = draw_covariates(source, rng, n)
    b_c, b_t, b_y = directions3(
        x.shape[1], {"acic": 1, "scm_dense": 2, "scm_mixed": 3}[source])
    def std(v):
        return (v - v.mean()) / (v.std() + 1e-8)
    r, q, by = std(x @ b_c), std(x @ b_t), std(x @ b_y)
    eps_u = rng.standard_normal(n)
    u = r + 0.5 * eps_u if variant == "v1_linear" \
        else r + 0.5 * np.sin(r) + 0.5 * eps_u
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


def run_once(source, variant, lam_f, rep):
    seed = (SEED_BASE + 7919 * rep + 1009 * VARIANTS.index(variant)
            + 401 * int(lam_f * 2)
            + 13 * {"acic": 1, "scm_dense": 2, "scm_mixed": 3}[source])
    rng = np.random.default_rng(seed)
    data = simulate_v2(source, variant, lam_f, rng, N)
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
    row = dict(source=source, variant=variant, lam_f=lam_f, rep=rep)
    for name, net in nets.items():
        strata = base.assign_strata(net, xs_ev)
        est, _ = base.stratified_tau(strata, a_ev, y_ev, M)
        oracle, _ = base.stratified_tau(strata, a_ev, ystr_ev, M)
        row[name] = dict(
            tau_error=float(est - 1.0), oracle_bias=float(oracle - 1.0),
            d_x=float(math.sqrt(max(base.hard_delta_sq(strata, a_ev, kx_ev, M), 0))),
            d_w=float(math.sqrt(max(base.hard_delta_sq(strata, a_ev, kw_ev, M), 0))),
            d_joint=float(math.sqrt(max(base.hard_delta_sq(strata, a_ev, kj_ev, M), 0))),
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
        for source in SOURCES:
            for variant in VARIANTS:
                for rep in range(REPS):
                    rows.append(run_once(source, variant, lam_f, rep))
                    print(f"lam={lam_f} {source}/{variant} rep={rep + 1}",
                          flush=True)
    per_lambda = {}
    for lam_f in LAMBDAS:
        sel = [r for r in rows if r["lam_f"] == lam_f]
        agg = {m: dict(
            mean_abs_bias=float(np.mean([abs(r[m]["oracle_bias"]) for r in sel])),
            rmse=float(np.sqrt(np.mean([r[m]["tau_error"] ** 2 for r in sel]))),
            mean_d_w=float(np.mean([r[m]["d_w"] for r in sel])),
            mean_d_x=float(np.mean([r[m]["d_x"] for r in sel])))
            for m in ARMS}
        per_lambda[str(lam_f)] = dict(
            aggregates=agg,
            decision_joint=decide(sel, "joint"),
            decision_constrained=decide(sel, "constrained"),
            per_source={s: {m: float(np.mean(
                [abs(r[m]["oracle_bias"]) for r in sel if r["source"] == s]))
                for m in ARMS} for s in SOURCES})
    summary = dict(provenance=dict(
        script="code/benchmark_suite_tabular_v2_directx.py",
        registration="memo/2026-08-19-rerun-registration-v3-directx.md",
        registration_sha256=REG_SHA, acic_sha256=ACIC_SHA,
        base_seed=SEED_BASE, n=N, reps=REPS, steps=STEPS, strata=M,
        lambdas=list(LAMBDAS), sources=list(SOURCES),
        variants=list(VARIANTS)),
        per_lambda=per_lambda, runs=rows)
    out = pathlib.Path(base.RESULTS_DIR) / "benchmark_suite_tabular_v2_directx.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: dict(decision_joint=v["decision_joint"],
                              decision_constrained=v["decision_constrained"],
                              aggregates=v["aggregates"])
                      for k, v in per_lambda.items()}, indent=1))


if __name__ == "__main__":
    main()
