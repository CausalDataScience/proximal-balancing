#!/usr/bin/env python3
"""E2: T4b-exact world validation (registration v2, frozen).

Population-exact finite-state world in which Q1 to Q5 hold by construction:
U | X=x, A=a ~ Normal(m(x) + a d(x), sigma_U^2), e(x) = 0.5, W = U + eps,
Y = A + 2U + eps. D_W is computed in closed form (Gaussian-mixture MMD),
bias = beta T exactly, and the T4b sandwich, ordering criterion, and the
per-stratum Taylor bound are checked with zero-violation criteria.
Output: results/rerun_e2_t4b_exact.json
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

REG_SHA = "8e6bf77079422cac55d06db2d671f0f72e4970e3334e877e749511f22a12400a"
S, M = 12, 3
BETA, SIG_U, SIG_W, BW = 2.0, 0.6, 0.5, 1.0
SEED = 2026081903
M_X = np.linspace(-1.0, 1.0, S)
D_X = np.where(np.arange(S) < 6, 0.8, 0.3)
P_X = np.full(S, 1.0 / S)
S2W = SIG_U ** 2 + SIG_W ** 2          # proxy component variance
RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def pair_integral(mu_a: np.ndarray, mu_b: np.ndarray) -> np.ndarray:
    """E k(Wa, Wb) for Gaussians with variance S2W and kernel bandwidth BW."""
    denom = BW ** 2 + 2 * S2W
    return (BW / math.sqrt(denom)) * np.exp(
        -0.5 * (mu_a[:, None] - mu_b[None, :]) ** 2 / denom)


def stratum_quantities(idx: np.ndarray) -> dict:
    mu0 = M_X[idx]
    mu1 = M_X[idx] + D_X[idx]
    w = 1.0 / idx.size
    g11 = pair_integral(mu1, mu1).sum() * w * w
    g00 = pair_integral(mu0, mu0).sum() * w * w
    g10 = pair_integral(mu1, mu0).sum() * w * w
    mmd2 = max(g11 + g00 - 2 * g10, 0.0)
    return dict(delta=float(D_X[idx].mean()),
                d_w=float(math.sqrt(mmd2)),
                pure=bool((idx < 6).all() or (idx >= 6).all()))


GRID = np.linspace(-6.0, 6.0, 400)
DW_GRID = GRID[1] - GRID[0]
K_GRID = np.exp(-0.5 * (GRID[:, None] - GRID[None, :]) ** 2 / BW ** 2)


def density_derivs(idx: np.ndarray):
    """Control-arm proxy density derivative and second derivative on GRID."""
    h1 = np.zeros_like(GRID)
    h2 = np.zeros_like(GRID)
    w = 1.0 / idx.size
    for mu in M_X[idx]:
        z = (GRID - mu) / math.sqrt(S2W)
        phi = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi * S2W)
        h1 += w * (-z / math.sqrt(S2W)) * phi
        h2 += w * ((z ** 2 - 1) / S2W) * phi
    s = math.sqrt(max(h1 @ K_GRID @ h1, 0.0)) * DW_GRID
    c = float(np.abs(h2).sum() * DW_GRID)       # kappa_W = 1
    return s, c


def evaluate(lab: np.ndarray) -> dict | None:
    strata = [np.flatnonzero(lab == z) for z in range(M)]
    if any(s.size == 0 for s in strata):
        return None
    qs = [stratum_quantities(s) for s in strata]
    p = [s.size / S for s in strata]
    t_sum = sum(pi * q["delta"] for pi, q in zip(p, qs))
    s2 = sum(pi * q["delta"] ** 2 for pi, q in zip(p, qs))
    d_w = sum(pi * q["d_w"] for pi, q in zip(p, qs))
    return dict(T=t_sum, S2=s2, D_W=d_w, bias=BETA * t_sum,
                pure=all(q["pure"] for q in qs),
                strata=strata, qs=qs, p=p)


def main() -> None:
    rng = np.random.default_rng(SEED)
    maps = []
    for _ in range(5000):                        # random maps
        lab = rng.integers(0, M, S)
        ev = evaluate(lab)
        if ev is not None:
            maps.append(ev)
    for _ in range(2000):                        # constructed group-pure maps
        split = rng.integers(1, M)               # labels for group A
        la = rng.integers(0, split, 6)
        lb = rng.integers(split, M, 6)
        ev = evaluate(np.concatenate([la, lb]))
        if ev is not None and ev["pure"]:
            maps.append(ev)
    n_pure = sum(m["pure"] for m in maps)

    # band constants over all evaluated strata
    s_vals, c_vals, taylor_viol, taylor_checked = [], [], 0, 0
    for m in maps:
        for st, q in zip(m["strata"], m["qs"]):
            s_z, c_z = density_derivs(st)
            s_vals.append(s_z)
            c_vals.append(c_z)
            if q["pure"]:
                taylor_checked += 1
                if abs(q["d_w"] - s_z * q["delta"]) > 0.5 * c_z * q["delta"] ** 2:
                    taylor_viol += 1
    s_min, s_max, c_bar = min(s_vals), max(s_vals), max(c_vals)

    sandwich_viol = 0
    for m in maps:
        lo = s_min * m["T"] - 0.5 * c_bar * m["S2"]
        hi = s_max * m["T"] + 0.5 * c_bar * m["S2"]
        if not (lo - 1e-9 <= m["D_W"] <= hi + 1e-9):
            sandwich_viol += 1

    order_viol = order_checked = 0
    for _ in range(50000):
        i, j = rng.integers(0, len(maps), 2)
        a, b = maps[i], maps[j]
        lhs = (a["D_W"] + 0.5 * c_bar * a["S2"]) / s_min
        rhs = (b["D_W"] - 0.5 * c_bar * b["S2"]) / s_max
        if lhs < rhs:                            # criterion strict inequality
            order_checked += 1
            if not abs(a["bias"]) < abs(b["bias"]):
                order_viol += 1

    ratio = [m["bias"] / m["D_W"] for m in maps if m["D_W"] > 1e-12]
    summary = dict(provenance=dict(
        script="code/rerun_e2_t4b_exact.py",
        registration="memo/2026-08-19-rerun-registration-v2.md",
        registration_sha256=REG_SHA, seed=SEED, S=S, M=M, beta=BETA,
        sigma_u=SIG_U, sigma_w=SIG_W, kernel_bw=BW,
        d_groups=[0.8, 0.3], e_const=0.5),
        n_maps=len(maps), n_pure_maps=int(n_pure),
        s_min=float(s_min), s_max=float(s_max), c_bar=float(c_bar),
        sandwich_violations=int(sandwich_viol),
        taylor_bound_checked=int(taylor_checked),
        taylor_bound_violations=int(taylor_viol),
        ordering_pairs_with_criterion=int(order_checked),
        ordering_violations=int(order_viol),
        bias_over_DW_min=float(min(ratio)), bias_over_DW_max=float(max(ratio)),
        decision_pass=bool(sandwich_viol == 0 and taylor_viol == 0
                           and order_viol == 0))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "rerun_e2_t4b_exact.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
