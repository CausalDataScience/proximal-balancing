#!/usr/bin/env python3
"""T1: Park single-proxy-control cross-world comparison (registration v5).

Two worlds (Park-conforming DGP-P, our template DGP-O) x estimators
(Park linear-bridge estimator per arXiv:2302.06054 Result 3; our three
matched representation arms with treated-weighted stratified estimation).
Estimand: ETT = 1 in both worlds.
Output: results/external_park_crossworld.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import learned_representation_balancing_experiment as base  # noqa: E402
from benchmark_suite_tabular import (  # noqa: E402
    _standardize, _grams, train_net, final_train_disc)

from scipy.stats import wilcoxon  # noqa: E402

REG_SHA = "3f7fa8eb550c3e4df076ed4e1eda85cb826bd8db0b2c97b9028c6792867ec6eb"
SEED_BASE = 2026081908
AMEND_SHA = "07afaa9863bcdb5158037b8d8fa2c326a6bbf0c8dd6289fe2faa7fefb355bd60"
D, N, REPS, STEPS, M = 20, 2000, 20, 120, 5     # B=20 per registration v5.1
ARMS = ("joint", "constrained", "x_only")


def _sig(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -35.0, 35.0)))


def directions3(d: int):
    rng = np.random.default_rng(SEED_BASE + 97 * d)
    q, _ = np.linalg.qr(rng.standard_normal((d, 3)))
    return q[:, 0], q[:, 1], q[:, 2]


def std(v):
    return (v - v.mean()) / (v.std() + 1e-8)


def simulate(world: str, rng: np.random.Generator, n: int):
    x = rng.standard_normal((n, D))
    b_c, b_t, b_y = directions3(D)
    r, q, by = std(x @ b_c), std(x @ b_t), std(x @ b_y)
    if world == "park":
        y0 = r + 0.5 * np.sin(r) + 0.5 * rng.standard_normal(n)
        a = rng.binomial(1, 0.05 + 0.90 * _sig(
            0.8 * std(y0) + 1.5 * q)).astype(float)
        w = y0 + 0.5 * rng.standard_normal(n)
        y = y0 + 1.0 * a
        u = y0                                  # latent for our arms
        y_struct = y                            # noise-free outcome itself
    elif world == "ours":
        u = r + 0.5 * np.sin(r) + 0.5 * rng.standard_normal(n)
        w = u + 0.5 * rng.standard_normal(n)
        a = rng.binomial(1, 0.05 + 0.90 * _sig(0.8 * u + 1.5 * q)).astype(float)
        y_struct = 1.0 * a + 2.0 * u + 0.5 * (q + 0.5 * by)
        y = y_struct + rng.standard_normal(n)
    else:
        raise ValueError(world)
    return dict(X=x, U=u, W=w, A=a, Y=y, Y_struct=y_struct)


def park_bridge_ett(w, a, y):
    """Linear-bridge GMM of Result 3: solve control-arm moments, then
    ETT = mean_treated(Y) - mean_treated(b(W))."""
    ctrl = a == 0
    g = np.column_stack([np.ones(ctrl.sum()), y[ctrl]])       # r_W(Y)=(1,Y)
    mw = np.column_stack([np.ones(ctrl.sum()), w[ctrl]])      # m_w(W)=(1,W)
    lhs = g.T @ mw
    rhs = g.T @ y[ctrl]
    eta = np.linalg.solve(lhs, rhs)
    b_treated = eta[0] + eta[1] * w[a == 1]
    return float(y[a == 1].mean() - b_treated.mean())


def ett_stratified(strata, a, y, m):
    n1 = (a == 1).sum()
    total = 0.0
    for z in range(m):
        in_z = strata == z
        y1, y0 = y[in_z & (a == 1)], y[in_z & (a == 0)]
        if y1.size and y0.size:
            total += (y1.size / n1) * (y1.mean() - y0.mean())
    return float(total)


def run_once(world, rep):
    seed = SEED_BASE + 7919 * rep + 13 * {"park": 1, "ours": 2}[world]
    rng = np.random.default_rng(seed)
    data = simulate(world, rng, N)
    perm = rng.permutation(N)
    tr, ev = perm[: N // 2], perm[N // 2:]
    xs_tr, xs_ev = _standardize(data["X"][tr], data["X"][ev])
    ws_tr, ws_ev = _standardize(data["W"][tr][:, None], data["W"][ev][:, None])
    a_tr, a_ev = data["A"][tr], data["A"][ev]
    y_ev, ystr_ev = data["Y"][ev], data["Y_struct"][ev]
    kx_tr, _ = _grams(xs_tr, xs_ev, seed + 101)
    kw_tr, _ = _grams(ws_tr, ws_ev, seed + 202)
    kj_tr = kx_tr * kw_tr
    net_x = train_net(xs_tr, a_tr, [(kx_tr, "objective")], M, STEPS, seed)
    x_cap = final_train_disc(net_x, xs_tr, a_tr, kx_tr)
    nets = dict(
        x_only=net_x,
        joint=train_net(xs_tr, a_tr, [(kj_tr, "objective")], M, STEPS, seed),
        constrained=train_net(xs_tr, a_tr,
                              [(kw_tr, "objective"), (kx_tr, "capped_x")],
                              M, STEPS, seed, x_cap=x_cap))
    row = dict(world=world, rep=rep,
               park=dict(ett_error=park_bridge_ett(
                   data["W"][ev], a_ev, y_ev) - 1.0,
                   oracle_error=park_bridge_ett(
                   data["W"][ev], a_ev, ystr_ev) - 1.0))
    for name, net in nets.items():
        strata = base.assign_strata(net, xs_ev)
        row[name] = dict(
            ett_error=ett_stratified(strata, a_ev, y_ev, M) - 1.0,
            oracle_error=ett_stratified(strata, a_ev, ystr_ev, M) - 1.0)
    return row


def main() -> None:
    rows = [run_once(world, rep) for world in ("park", "ours")
            for rep in range(REPS)]
    for r in rows:
        print(r["world"], r["rep"],
              {k: round(r[k]["oracle_error"], 3)
               for k in ("park",) + ARMS}, flush=True)
    summary = dict(provenance=dict(
        script="code/external_park_crossworld.py",
        registration="memo/2026-08-19-rerun-registration-v5-external.md",
        registration_sha256=REG_SHA, seed_base=SEED_BASE, d=D, n=N,
        reps=REPS, steps=STEPS, strata=M,
        park_estimator="linear bridge GMM, Result 3 of arXiv:2302.06054"))
    for world in ("park", "ours"):
        sel = [r for r in rows if r["world"] == world]
        agg = {}
        for est in ("park",) + ARMS:
            errs = np.array([r[est]["oracle_error"] for r in sel])
            noisy = np.array([r[est]["ett_error"] for r in sel])
            agg[est] = dict(mean_abs_bias=float(np.abs(errs).mean()),
                            rmse=float(np.sqrt((noisy ** 2).mean())))
        diff = [abs(r["park"]["oracle_error"])
                - abs(r["constrained"]["oracle_error"]) for r in sel]
        summary[world] = dict(
            aggregates=agg,
            wilcoxon_constrained_beats_park_p=float(
                wilcoxon(diff, alternative="greater").pvalue),
            wilcoxon_park_beats_constrained_p=float(
                wilcoxon(diff, alternative="less").pvalue))
    summary["runs"] = rows
    summary["provenance"]["amendment_sha256"] = AMEND_SHA
    out = pathlib.Path(base.RESULTS_DIR) / "external_park_crossworld_B20.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({w: summary[w] for w in ("park", "ours")}, indent=1,
                     default=str)[:1500])


if __name__ == "__main__":
    main()
