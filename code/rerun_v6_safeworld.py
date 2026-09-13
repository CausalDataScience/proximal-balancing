#!/usr/bin/env python3
"""Registration v6: KMU rescue cell + ignorability-world comparison.

T1: linear P2SLS double-proxy estimator on the KMU DGP with the hidden Z
revealed, on untransformed and transformed observables.
T2: SAFE-X world where X satisfies ignorability; compare our arms, Park
linear bridge, linear P2SLS, and naive.
Output: results/rerun_v6_safeworld.json
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
from external_park_crossworld import park_bridge_ett  # noqa: E402
from kmu_2022_single_proxy_ablation_experiment import (  # noqa: E402
    simulate_kmu, _seed as kmu_seed)

REG_SHA = "c6dcb38e051afb616322fc6ece79463b7826964201a461bc71e43c3267a5ebd4"
SEED_BASE = 2026081910
D, N, REPS, STEPS, M = 20, 2000, 10, 120, 5


def _sig(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -35.0, 35.0)))


def std(v):
    return (v - v.mean()) / (v.std() + 1e-8)


def p2sls_ate(x, z, w, a, y) -> float:
    """Linear proximal 2SLS: W ~ (Z,A,X); Y ~ (W_hat,A,X); return A coef."""
    ones = np.ones((len(a), 1))
    s1 = np.column_stack([ones, z, a[:, None], x])
    coef1, *_ = np.linalg.lstsq(s1, w, rcond=None)
    w_hat = s1 @ coef1
    s2 = np.column_stack([ones, w_hat, a[:, None], x])
    coef2, *_ = np.linalg.lstsq(s2, y, rcond=None)
    return float(coef2[1 + (w.shape[1] if w.ndim > 1 else 1)])


def t1_kmu_rescue() -> dict:
    out = {}
    for transform in ("sin", "cube"):
        errs = {"p2sls_untransformed": [], "p2sls_transformed": []}
        for rep in range(6):
            rng = np.random.default_rng(kmu_seed(1200, rep))
            d = simulate_kmu(rng, 1200, transform)
            a, ys = d["A"], d["Y_struct"]
            errs["p2sls_untransformed"].append(
                p2sls_ate(d["Xp"], d["Zp"], d["Wp"], a, ys) - 1.0)
            errs["p2sls_transformed"].append(
                p2sls_ate(d["X"], d["Z"], d["W"], a, ys) - 1.0)
        out[transform] = {
            k: dict(mean_abs_bias=float(np.abs(np.array(v)).mean()),
                    sd=float(np.array(v).std()))
            for k, v in errs.items()}
    return out


def simulate_safe(rng, n):
    x = rng.standard_normal((n, D))
    qmat, _ = np.linalg.qr(
        np.random.default_rng(SEED_BASE + 97 * D).standard_normal((D, 3)))
    b_c, b_t, b_y = qmat[:, 0], qmat[:, 1], qmat[:, 2]
    r, q, by = std(x @ b_c), std(x @ b_t), std(x @ b_y)
    u = r + 0.5 * np.sin(r) + 0.5 * rng.standard_normal(n)
    w = u + 0.5 * rng.standard_normal(n)
    z = u + 0.5 * rng.standard_normal(n)
    a = rng.binomial(1, 0.05 + 0.90 * _sig(0.8 * r + 1.5 * q)).astype(float)
    y_struct = 1.0 * a + 2.0 * u + 0.5 * (q + 0.5 * by)
    y = y_struct + rng.standard_normal(n)
    return dict(X=x, U=u, W=w, Z=z, A=a, Y_struct=y_struct, Y=y)


def ate_stratified(strata, a, y, m, n):
    total = 0.0
    for zi in range(m):
        in_z = strata == zi
        y1, y0 = y[in_z & (a == 1)], y[in_z & (a == 0)]
        if y1.size and y0.size:
            total += (in_z.sum() / n) * (y1.mean() - y0.mean())
    return float(total)


def t2_safe_world() -> dict:
    rows = []
    for rep in range(REPS):
        seed = SEED_BASE + 7919 * rep
        rng = np.random.default_rng(seed)
        data = simulate_safe(rng, N)
        perm = rng.permutation(N)
        tr, ev = perm[: N // 2], perm[N // 2:]
        xs_tr, xs_ev = _standardize(data["X"][tr], data["X"][ev])
        ws_tr, ws_ev = _standardize(data["W"][tr][:, None],
                                    data["W"][ev][:, None])
        a_tr, a_ev = data["A"][tr], data["A"][ev]
        y_ev, ystr_ev = data["Y"][ev], data["Y_struct"][ev]
        kx_tr, kx_ev = _grams(xs_tr, xs_ev, seed + 101)
        kw_tr, kw_ev = _grams(ws_tr, ws_ev, seed + 202)
        kj_tr = kx_tr * kw_tr
        net_x = train_net(xs_tr, a_tr, [(kx_tr, "objective")], M, STEPS, seed)
        x_cap = final_train_disc(net_x, xs_tr, a_tr, kx_tr)
        nets = dict(
            x_only=net_x,
            joint=train_net(xs_tr, a_tr, [(kj_tr, "objective")],
                            M, STEPS, seed),
            constrained=train_net(
                xs_tr, a_tr, [(kw_tr, "objective"), (kx_tr, "capped_x")],
                M, STEPS, seed, x_cap=x_cap))
        row = dict(rep=rep)
        for name, net in nets.items():
            strata = base.assign_strata(net, xs_ev)
            row[name] = dict(
                oracle_error=ate_stratified(strata, a_ev, ystr_ev, M,
                                            len(ev)) - 1.0,
                noisy_error=ate_stratified(strata, a_ev, y_ev, M,
                                           len(ev)) - 1.0,
                d_w=float(math.sqrt(max(
                    base.hard_delta_sq(strata, a_ev, kw_ev, M), 0.0))))
        row["park"] = dict(
            oracle_error=park_bridge_ett(data["W"][ev], a_ev, ystr_ev) - 1.0,
            noisy_error=park_bridge_ett(data["W"][ev], a_ev, y_ev) - 1.0)
        row["p2sls"] = dict(
            oracle_error=p2sls_ate(data["X"][ev], data["Z"][ev][:, None],
                                   data["W"][ev][:, None], a_ev,
                                   ystr_ev) - 1.0,
            noisy_error=p2sls_ate(data["X"][ev], data["Z"][ev][:, None],
                                  data["W"][ev][:, None], a_ev, y_ev) - 1.0)
        row["naive"] = dict(
            oracle_error=float(ystr_ev[a_ev == 1].mean()
                               - ystr_ev[a_ev == 0].mean() - 1.0),
            noisy_error=float(y_ev[a_ev == 1].mean()
                              - y_ev[a_ev == 0].mean() - 1.0))
        rows.append(row)
        print(f"safe rep={rep + 1}/{REPS}", flush=True)
    ests = ("joint", "constrained", "x_only", "park", "p2sls", "naive")
    agg = {}
    for est in ests:
        oe = np.array([r[est]["oracle_error"] for r in rows])
        ne = np.array([r[est]["noisy_error"] for r in rows])
        agg[est] = dict(mean_abs_bias=float(np.abs(oe).mean()),
                        sd=float(oe.std()),
                        rmse=float(np.sqrt((ne ** 2).mean())))
    agg["mean_d_w_ours"] = float(np.mean(
        [r[a]["d_w"] for r in rows for a in ("joint", "constrained")]))
    return dict(aggregates=agg, runs=rows)


def main() -> None:
    summary = dict(provenance=dict(
        script="code/rerun_v6_safeworld.py",
        registration="memo/2026-08-19-rerun-registration-v6-safeworld.md",
        registration_sha256=REG_SHA, seed_base=SEED_BASE,
        d=D, n=N, reps=REPS, steps=STEPS, strata=M),
        t1_kmu_rescue=t1_kmu_rescue(),
        t2_safe_world=t2_safe_world())
    out = pathlib.Path(base.RESULTS_DIR) / "rerun_v6_safeworld.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(t1=summary["t1_kmu_rescue"],
                          t2=summary["t2_safe_world"]["aggregates"]),
                     indent=1))


if __name__ == "__main__":
    main()
