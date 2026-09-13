#!/usr/bin/env python3
"""E3: x_signal_null resolution (registration v2, frozen).

E3a instruments the original x_signal_null control (U independent of X,
logistic propensity with q-saturation) and tests the saturation channel:
|delta_z| should rise with the interior-propensity measure pi_z(1-pi_z),
and the joint arm should achieve smaller T than x_only. E3b closes the
channel with a flat-slope propensity clip(0.5 + 0.12U + 0.12q, .1, .9) and
requires the joint arm's gain to vanish.
Output: results/rerun_e3_control_resolution.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import learned_representation_balancing_experiment as base  # noqa: E402
from benchmark_suite_tabular import (  # noqa: E402
    draw_covariates, directions, _standardize, _grams, train_net)

from scipy.stats import spearmanr, wilcoxon  # noqa: E402

REG_SHA = "8e6bf77079422cac55d06db2d671f0f72e4970e3334e877e749511f22a12400a"
BASE_SEED = 2026081904
N, REPS, STEPS, M = 1200, 9, 120, 5
SOURCES = ("acic", "scm_dense")


def _sig(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -35.0, 35.0)))


def simulate_null(source: str, regime: str, rng: np.random.Generator, n: int):
    x = draw_covariates(source, rng, n)
    d = x.shape[1]
    _, b_t = directions(d, tag={"acic": 1, "scm_dense": 2}[source])
    q = x @ b_t
    q = (q - q.mean()) / (q.std() + 1e-8)
    u = rng.standard_normal(n)                   # U independent of X
    w = u + 0.5 * rng.standard_normal(n)
    if regime == "x_signal_null":
        prop = 0.05 + 0.90 * _sig(0.8 * u + 1.5 * q)
    elif regime == "x_signal_null_flat":
        prop = np.clip(0.5 + 0.12 * u + 0.12 * q, 0.10, 0.90)
    else:
        raise ValueError(regime)
    a = rng.binomial(1, prop).astype(float)
    y_struct = 1.0 * a + 2.0 * u
    y = y_struct + rng.standard_normal(n)
    return dict(X=x, U=u, W=w, A=a, Y_struct=y_struct, Y=y)


def stratum_records(strata, a, u, m):
    out = []
    for z in range(m):
        in_z = strata == z
        if in_z.sum() == 0:
            continue
        u1, u0 = u[in_z & (a == 1)], u[in_z & (a == 0)]
        if u1.size == 0 or u0.size == 0:
            continue
        pi_z = a[in_z].mean()
        out.append(dict(p_z=float(in_z.mean()),
                        delta_z=float(u1.mean() - u0.mean()),
                        sat_z=float(pi_z * (1 - pi_z))))
    return out


def run_cell(source, regime, rep):
    seed = (BASE_SEED + 7919 * rep
            + 104729 * {"x_signal_null": 0, "x_signal_null_flat": 1}[regime]
            + 13 * {"acic": 1, "scm_dense": 2}[source])
    rng = np.random.default_rng(seed)
    data = simulate_null(source, regime, rng, N)
    perm = rng.permutation(N)
    tr, ev = perm[: N // 2], perm[N // 2:]
    xs_tr, xs_ev = _standardize(data["X"][tr], data["X"][ev])
    ws_tr, _ = _standardize(data["W"][tr][:, None], data["W"][ev][:, None])
    a_tr, a_ev = data["A"][tr], data["A"][ev]
    y_ev, ystr_ev, u_ev = data["Y"][ev], data["Y_struct"][ev], data["U"][ev]
    kx_tr, _ = _grams(xs_tr, xs_ev, seed + 101)
    kw_tr, _ = _grams(ws_tr, ws_tr, seed + 202)
    kj_tr = kx_tr * kw_tr
    nets = dict(
        x_only=train_net(xs_tr, a_tr, [(kx_tr, "objective")], M, STEPS, seed),
        joint=train_net(xs_tr, a_tr, [(kj_tr, "objective")], M, STEPS, seed))
    row = dict(source=source, regime=regime, rep=rep)
    for name, net in nets.items():
        strata = base.assign_strata(net, xs_ev)
        est, _ = base.stratified_tau(strata, a_ev, y_ev, M)
        oracle, _ = base.stratified_tau(strata, a_ev, ystr_ev, M)
        recs = stratum_records(strata, a_ev, u_ev, M)
        row[name] = dict(tau_error=float(est - 1.0),
                         oracle_bias=float(oracle - 1.0),
                         T=float(sum(r["p_z"] * r["delta_z"] for r in recs)),
                         strata=recs)
    return row


def main() -> None:
    rows = []
    for regime in ("x_signal_null", "x_signal_null_flat"):
        for source in SOURCES:
            for rep in range(REPS):
                rows.append(run_cell(source, regime, rep))
                print(regime, source, rep + 1, flush=True)

    e3a = [r for r in rows if r["regime"] == "x_signal_null"]
    deltas = [abs(s["delta_z"]) for r in e3a for s in r["joint"]["strata"]]
    sats = [s["sat_z"] for r in e3a for s in r["joint"]["strata"]]
    rho, rho_p = spearmanr(sats, deltas)
    t_diff = [r["x_only"]["T"] - r["joint"]["T"] for r in e3a]
    t_p = float(wilcoxon(t_diff, alternative="greater").pvalue)
    e3a_pass = bool(rho > 0 and rho_p < 0.05 and t_p < 0.05)

    e3b = [r for r in rows if r["regime"] == "x_signal_null_flat"]
    wins = [abs(r["joint"]["oracle_bias"]) < abs(r["x_only"]["oracle_bias"])
            and abs(r["joint"]["tau_error"]) < abs(r["x_only"]["tau_error"])
            for r in e3b]
    red = [abs(r["x_only"]["oracle_bias"]) - abs(r["joint"]["oracle_bias"])
           for r in e3b]
    win_rate = float(np.mean(wins))
    wil_p = float(wilcoxon(red, alternative="greater").pvalue)
    e3b_pass = bool(win_rate < 0.70 and wil_p >= 0.05)

    summary = dict(provenance=dict(
        script="code/rerun_e3_control_resolution.py",
        registration="memo/2026-08-19-rerun-registration-v2.md",
        registration_sha256=REG_SHA, base_seed=BASE_SEED,
        n=N, reps=REPS, steps=STEPS, strata=M, sources=list(SOURCES)),
        e3a=dict(n_strata=len(deltas), spearman_rho=float(rho),
                 spearman_p=float(rho_p),
                 mean_T_joint=float(np.mean([r["joint"]["T"] for r in e3a])),
                 mean_T_xonly=float(np.mean([r["x_only"]["T"] for r in e3a])),
                 wilcoxon_T_p=t_p, saturation_channel_confirmed=e3a_pass),
        e3b=dict(n_pairs=len(wins), joint_win_rate=win_rate,
                 wilcoxon_p=wil_p,
                 mean_bias_joint=float(np.mean(
                     [abs(r["joint"]["oracle_bias"]) for r in e3b])),
                 mean_bias_xonly=float(np.mean(
                     [abs(r["x_only"]["oracle_bias"]) for r in e3b])),
                 gain_vanishes=e3b_pass),
        decision_pass=bool(e3a_pass and e3b_pass),
        runs=rows)
    out = pathlib.Path(base.RESULTS_DIR) / "rerun_e3_control_resolution.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(e3a=summary["e3a"], e3b=summary["e3b"],
                          decision_pass=summary["decision_pass"]), indent=1))


if __name__ == "__main__":
    main()
