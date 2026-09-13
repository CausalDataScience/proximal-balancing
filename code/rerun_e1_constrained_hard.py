#!/usr/bin/env python3
"""E1: hard-constrained arm rerun (registration v2, frozen).

Dual-ascent enforcement of train-side soft D_X^2 <= c with the x_only
anchor, paired seed-for-seed with the first suite. See
memo/2026-08-19-rerun-registration-v2.md for every fixed parameter.
Output: results/rerun_e1_constrained_hard.json
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import learned_representation_balancing_experiment as base  # noqa: E402
from benchmark_suite_tabular import (  # noqa: E402
    BASE_SEED, DGP_VARIANTS, simulate, _standardize, _grams,
    train_net, final_train_disc)

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

REG_SHA = "8e6bf77079422cac55d06db2d671f0f72e4970e3334e877e749511f22a12400a"
ACIC_SHA = "0d6387ad45d23e54b11cbf967248fb68ad5e578db80d193208b8b4c98b1ed0c1"
N, REPS, STEPS, M = 1200, 6, 120, 5
SOURCES = ("acic", "scm_dense", "scm_mixed")


def train_hard(xs, a, kw, kx, m, steps, seed, c):
    torch.manual_seed(seed)
    net = base.StratumNet(d_in=xs.shape[1], m_strata=m)
    x_t = torch.tensor(xs, dtype=torch.float32)
    a_t = torch.tensor(a, dtype=torch.float32)
    kw_t = torch.tensor(kw, dtype=torch.float32)
    kx_t = torch.tensor(kx, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    overlap_floor, mass_floor = 0.09, 0.5 / m
    lam = 25.0

    def one_step(temp):
        nonlocal lam
        soft = torch.softmax(net(x_t) / temp, dim=1)
        dw, _, _ = base.soft_delta_sq(soft, a_t, kw_t)
        dx, p_hat, pi_hat = base.soft_delta_sq(soft, a_t, kx_t)
        loss = dw + lam * torch.relu(dx - c)
        loss = loss + (p_hat * torch.relu(
            overlap_floor - pi_hat * (1.0 - pi_hat))).sum()
        loss = loss + torch.relu(mass_floor - p_hat).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        lam += 500.0 * max(float(dx) - c, 0.0)

    for step in range(steps):
        one_step(1.0 * 0.25 ** (step / max(steps - 1, 1)))
    rounds = 0
    while rounds < 3 and final_train_disc(net, xs, a, kx) > c + 1e-6:
        rounds += 1
        lam *= 2.0
        for _ in range(40):
            one_step(0.25)
    feasible = final_train_disc(net, xs, a, kx) <= c + 1e-6
    return net, feasible, rounds, lam


def stratum_vectors(strata, a, u, pi_from, m):
    out = []
    for z in range(m):
        in_z = strata == z
        if in_z.sum() == 0:
            continue
        u1, u0 = u[in_z & (a == 1)], u[in_z & (a == 0)]
        if u1.size == 0 or u0.size == 0:
            continue
        pi_z = a[in_z].mean()
        out.append(dict(z=int(z), p_z=float(in_z.mean()),
                        delta_z=float(u1.mean() - u0.mean()),
                        sat_z=float(pi_z * (1 - pi_z))))
    return out


def main() -> None:
    records = []
    for source in SOURCES:
        for variant in DGP_VARIANTS:
            for rep in range(REPS):
                seed = (BASE_SEED + 7919 * rep + 1009 * DGP_VARIANTS.index(variant)
                        + 13 * {"acic": 1, "scm_dense": 2, "scm_mixed": 3}[source])
                rng = np.random.default_rng(seed)
                data = simulate(source, variant, "primary", rng, N)
                perm = rng.permutation(N)
                tr, ev = perm[: N // 2], perm[N // 2:]
                xs_tr, xs_ev = _standardize(data["X"][tr], data["X"][ev])
                ws_tr, ws_ev = _standardize(data["W"][tr][:, None],
                                            data["W"][ev][:, None])
                a_tr, a_ev = data["A"][tr], data["A"][ev]
                y_ev, ystr_ev = data["Y"][ev], data["Y_struct"][ev]
                u_ev = data["U"][ev]
                kx_tr, kx_ev = _grams(xs_tr, xs_ev, seed + 101)
                kw_tr, kw_ev = _grams(ws_tr, ws_ev, seed + 202)

                net_x = train_net(xs_tr, a_tr, [(kx_tr, "objective")],
                                  M, STEPS, seed)
                c = final_train_disc(net_x, xs_tr, a_tr, kx_tr)
                net_h, feasible, rounds, lam = train_hard(
                    xs_tr, a_tr, kw_tr, kx_tr, M, STEPS, seed, c)

                row = dict(source=source, variant=variant, rep=rep,
                           anchor_c=float(c), feasible=bool(feasible),
                           feasibility_rounds=int(rounds),
                           final_lambda=float(lam))
                for name, net in (("x_only", net_x), ("constrained_hard", net_h)):
                    strata = base.assign_strata(net, xs_ev)
                    est, _ = base.stratified_tau(strata, a_ev, y_ev, M)
                    oracle, _ = base.stratified_tau(strata, a_ev, ystr_ev, M)
                    row[name] = dict(
                        tau_error=float(est - 1.0),
                        oracle_bias=float(oracle - 1.0),
                        d_x=float(math.sqrt(max(
                            base.hard_delta_sq(strata, a_ev, kx_ev, M), 0.0))),
                        d_w=float(math.sqrt(max(
                            base.hard_delta_sq(strata, a_ev, kw_ev, M), 0.0))),
                        strata=stratum_vectors(strata, a_ev, u_ev, None, M))
                records.append(row)
                print(f"{source}/{variant} rep={rep + 1}/{REPS} "
                      f"feasible={feasible}", flush=True)

    per_source = {}
    for source in SOURCES:
        rows = [r for r in records if r["source"] == source]
        red = [abs(r["x_only"]["oracle_bias"])
               - abs(r["constrained_hard"]["oracle_bias"]) for r in rows]
        wins = [abs(r["constrained_hard"]["oracle_bias"])
                < abs(r["x_only"]["oracle_bias"])
                and abs(r["constrained_hard"]["tau_error"])
                < abs(r["x_only"]["tau_error"]) for r in rows]
        exceed = sum(r["constrained_hard"]["d_x"] > r["x_only"]["d_x"]
                     for r in rows)
        per_source[source] = dict(
            n_runs=len(rows),
            feasibility_rate=float(np.mean([r["feasible"] for r in rows])),
            mean_abs_bias_hard=float(np.mean(
                [abs(r["constrained_hard"]["oracle_bias"]) for r in rows])),
            mean_abs_bias_xonly=float(np.mean(
                [abs(r["x_only"]["oracle_bias"]) for r in rows])),
            rmse_hard=float(np.sqrt(np.mean(
                [r["constrained_hard"]["tau_error"] ** 2 for r in rows]))),
            rmse_xonly=float(np.sqrt(np.mean(
                [r["x_only"]["tau_error"] ** 2 for r in rows]))),
            win_rate=float(np.mean(wins)),
            wilcoxon_p=(float(wilcoxon(red, alternative="greater").pvalue)
                        if wilcoxon is not None else None),
            heldout_dx_exceedances=int(exceed),
            mean_d_w_hard=float(np.mean(
                [r["constrained_hard"]["d_w"] for r in rows])),
            mean_d_w_xonly=float(np.mean(
                [r["x_only"]["d_w"] for r in rows])))
    overall_pass = all(
        v["feasibility_rate"] == 1.0 and v["win_rate"] >= 0.70
        and (v["wilcoxon_p"] is not None and v["wilcoxon_p"] < 0.05)
        for v in per_source.values())
    summary = dict(provenance=dict(
        script="code/rerun_e1_constrained_hard.py",
        registration="memo/2026-08-19-rerun-registration-v2.md",
        registration_sha256=REG_SHA, acic_sha256=ACIC_SHA,
        base_seed=BASE_SEED, n=N, reps=REPS, steps=STEPS, strata=M),
        per_source=per_source, decision_pass=bool(overall_pass),
        runs=records)
    out = pathlib.Path(base.RESULTS_DIR) / "rerun_e1_constrained_hard.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(per_source=per_source, decision_pass=overall_pass),
                     indent=1))


if __name__ == "__main__":
    main()
