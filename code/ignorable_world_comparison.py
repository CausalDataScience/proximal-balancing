#!/usr/bin/env python3
"""Ignorable-X world comparison (registration v6, frozen).

X satisfies ignorability by construction; compares naive, full-X g-formula,
our three arms, Park's covariate-adjusted linear bridge, and a parametric
double-proxy proximal 2SLS. Output: results/ignorable_world_comparison.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import learned_representation_balancing_experiment as base  # noqa: E402
from benchmark_suite_tabular import (  # noqa: E402
    draw_covariates, _standardize, _grams, train_net, final_train_disc)

REG_SHA = "bf3837460f7705828cc3d6938aa56fdd8a05fe3ca8d04d726d0bfef75067123e"
ACIC_SHA = "0d6387ad45d23e54b11cbf967248fb68ad5e578db80d193208b8b4c98b1ed0c1"
SEED_BASE = 2026081910
N, REPS, STEPS, M = 2000, 20, 120, 5
SOURCES = ("acic", "scm_dense")
ARMS = ("joint", "constrained", "x_only")


def _sig(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -35.0, 35.0)))


def std(v):
    return (v - v.mean()) / (v.std() + 1e-8)


def directions3(d: int, tag: int):
    rng = np.random.default_rng(SEED_BASE + 97 * d + 11 * tag)
    q, _ = np.linalg.qr(rng.standard_normal((d, 3)))
    return q[:, 0], q[:, 1], q[:, 2]


def simulate(source, rng, n):
    x = draw_covariates(source, rng, n)
    b_c, b_t, b_y = directions3(x.shape[1],
                                {"acic": 1, "scm_dense": 2}[source])
    r, q, by = std(x @ b_c), std(x @ b_t), std(x @ b_y)
    u = 0.7 * rng.standard_normal(n) + 0.3 * r
    a = rng.binomial(1, 0.05 + 0.90 * _sig(1.5 * q)).astype(float)  # X only
    w = u + 0.5 * rng.standard_normal(n)
    z = u + 0.5 * rng.standard_normal(n)
    f_x = q + 0.5 * by
    y_struct = 1.0 * a + f_x + 2.0 * u
    y = y_struct + rng.standard_normal(n)
    return dict(X=x, U=u, W=w, Z=z, A=a, Y_struct=y_struct, Y=y)


def park_x_ett(w, x, a, y):
    ctrl = a == 0
    mw = np.column_stack([np.ones(ctrl.sum()), w[ctrl], x[ctrl]])
    r = np.column_stack([np.ones(ctrl.sum()), y[ctrl], x[ctrl]])
    eta, *_ = np.linalg.lstsq(r.T @ mw, r.T @ y[ctrl], rcond=None)
    b1 = np.column_stack([np.ones((a == 1).sum()), w[a == 1], x[a == 1]]) @ eta
    return float(y[a == 1].mean() - b1.mean())


def p2sls_tau(w, z, x, a, y):
    design = np.column_stack([np.ones(len(a)), w, x, a])
    instr = np.column_stack([np.ones(len(a)), z, x, a])
    coef, *_ = np.linalg.lstsq(instr.T @ design, instr.T @ y, rcond=None)
    return float(coef[-1])


def run_once(source, rep):
    seed = SEED_BASE + 7919 * rep + 13 * {"acic": 1, "scm_dense": 2}[source]
    rng = np.random.default_rng(seed)
    data = simulate(source, rng, N)
    perm = rng.permutation(N)
    tr, ev = perm[: N // 2], perm[N // 2:]
    xs_tr, xs_ev = _standardize(data["X"][tr], data["X"][ev])
    ws_tr, ws_ev = _standardize(data["W"][tr][:, None], data["W"][ev][:, None])
    a_tr, a_ev = data["A"][tr], data["A"][ev]
    y_ev, ystr_ev = data["Y"][ev], data["Y_struct"][ev]
    x_ev_raw = xs_ev
    kx_tr, kx_ev = _grams(xs_tr, xs_ev, seed + 101)
    kw_tr, kw_ev = _grams(ws_tr, ws_ev, seed + 202)
    kj_tr = kx_tr * kw_tr
    net_x = train_net(xs_tr, a_tr, [(kx_tr, "objective")], M, STEPS, seed)
    x_cap = final_train_disc(net_x, xs_tr, a_tr, kx_tr)
    nets = dict(
        x_only=net_x,
        joint=train_net(xs_tr, a_tr, [(kj_tr, "objective")], M, STEPS, seed),
        constrained=train_net(xs_tr, a_tr,
                              [(kw_tr, "objective"), (kx_tr, "capped_x")],
                              M, STEPS, seed, x_cap=x_cap))
    row = dict(source=source, rep=rep)
    for name, net in nets.items():
        strata = base.assign_strata(net, xs_ev)
        est, _ = base.stratified_tau(strata, a_ev, y_ev, M)
        oracle, _ = base.stratified_tau(strata, a_ev, ystr_ev, M)
        import math as _m
        row[name] = dict(err=float(est - 1.0), oerr=float(oracle - 1.0),
                         d_w=float(_m.sqrt(max(base.hard_delta_sq(
                             strata, a_ev, kw_ev, M), 0.0))))
    row["naive"] = dict(
        err=float(y_ev[a_ev == 1].mean() - y_ev[a_ev == 0].mean() - 1.0),
        oerr=float(ystr_ev[a_ev == 1].mean() - ystr_ev[a_ev == 0].mean() - 1.0))
    row["full_x"] = dict(
        err=float(base.g_formula_tau(xs_tr, a_tr, data["Y"][tr],
                                     xs_ev, seed) - 1.0),
        oerr=float(base.g_formula_tau(xs_tr, a_tr, data["Y_struct"][tr],
                                      xs_ev, seed) - 1.0))
    row["park_x"] = dict(
        err=park_x_ett(data["W"][ev], x_ev_raw, a_ev, y_ev) - 1.0,
        oerr=park_x_ett(data["W"][ev], x_ev_raw, a_ev, ystr_ev) - 1.0)
    row["p2sls"] = dict(
        err=p2sls_tau(data["W"][ev], data["Z"][ev], x_ev_raw, a_ev, y_ev) - 1.0,
        oerr=p2sls_tau(data["W"][ev], data["Z"][ev], x_ev_raw, a_ev,
                       ystr_ev) - 1.0)
    return row


def main() -> None:
    ests = ("naive", "full_x", "x_only", "joint", "constrained",
            "park_x", "p2sls")
    rows = []
    for source in SOURCES:
        for rep in range(REPS):
            rows.append(run_once(source, rep))
            print(source, rep + 1, flush=True)
    summary = dict(provenance=dict(
        script="code/ignorable_world_comparison.py",
        registration="memo/2026-08-19-rerun-registration-v6-ignorable.md",
        registration_sha256=REG_SHA, acic_sha256=ACIC_SHA,
        seed_base=SEED_BASE, n=N, reps=REPS, steps=STEPS, strata=M))
    for source in SOURCES:
        sel = [r for r in rows if r["source"] == source]
        agg = {}
        for est in ests:
            o = np.array([r[est]["oerr"] for r in sel])
            e = np.array([r[est]["err"] for r in sel])
            agg[est] = dict(mean_abs_bias=float(np.abs(o).mean()),
                            median_abs_bias=float(np.median(np.abs(o))),
                            rmse=float(np.sqrt((e ** 2).mean())))
        agg["certificate_d_w"] = dict(
            joint=float(np.mean([r["joint"]["d_w"] for r in sel])),
            constrained=float(np.mean(
                [r["constrained"]["d_w"] for r in sel])),
            x_only=float(np.mean([r["x_only"]["d_w"] for r in sel])))
        summary[source] = agg
    summary["runs"] = rows
    out = pathlib.Path(base.RESULTS_DIR) / "ignorable_world_comparison.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({s: summary[s] for s in SOURCES}, indent=1)[:1800])


if __name__ == "__main__":
    main()
