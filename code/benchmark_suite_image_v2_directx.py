#!/usr/bin/env python3
"""Image suite v2 with direct X-to-Y effects (registration v3, frozen).

Y = tau A + 2U + lambda_f q(X) + eps with q the treatment-driving image
position factor, lambda_f in {0, .5, 1}. CNN learns H = h(X) from pixels;
arms and protocol match the first image suite (variant v1_std).
Output: results/benchmark_suite_image_v2_directx.json
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import learned_representation_balancing_experiment as base  # noqa: E402
from benchmark_suite_image import (  # noqa: E402
    digit_images, lesion_images, train_cnn, cnn_disc, cnn_strata, _grams)

from scipy.stats import wilcoxon  # noqa: E402

REG_SHA = "7110b10c3f077d37aa3a0881243a7257109832cfebfc014fe445dd4d5071cb8b"
SEED_BASE = 2026081906
N, REPS, STEPS, M = 1200, 4, 100, 5
LAMBDAS = (0.0, 0.5, 1.0)
DOMAINS = ("digits", "lesions")
ARMS = ("joint", "constrained", "x_only")


def _sig(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -35.0, 35.0)))


def make_data_v2(domain, lam_f, rng, n):
    if domain == "digits":
        pool = digit_images()
        idx = rng.choice(pool.shape[0], size=n, replace=False)
        imgs = pool[idx]
        r_raw = imgs.mean(axis=(1, 2))
        col = np.arange(imgs.shape[2])
        q_raw = (imgs * col[None, None, :]).sum((1, 2)) / (
            imgs.sum((1, 2)) + 1e-8)
    else:
        imgs, r_raw, q_raw = lesion_images(rng, n)
    def std(v):
        return (v - v.mean()) / (v.std() + 1e-8)
    r, q = std(r_raw), std(q_raw)
    u = r + 0.5 * rng.standard_normal(n)
    w = u + 0.5 * rng.standard_normal(n)
    a = rng.binomial(1, 0.05 + 0.90 * _sig(0.8 * u + 1.5 * q)).astype(float)
    y_struct = 1.0 * a + 2.0 * u + lam_f * q
    y = y_struct + rng.standard_normal(n)
    return dict(IMG=imgs, U=u, W=w, A=a, Y_struct=y_struct, Y=y)


def tau_plugin(x_strata, a, y, m, n):
    total = 0.0
    for z in range(m):
        in_z = x_strata == z
        y1, y0 = y[in_z & (a == 1)], y[in_z & (a == 0)]
        if y1.size and y0.size:
            total += (in_z.sum() / n) * (y1.mean() - y0.mean())
    return total


def run_once(domain, lam_f, rep):
    seed = (SEED_BASE + 7919 * rep + 401 * int(lam_f * 2)
            + 13 * {"digits": 1, "lesions": 2}[domain])
    rng = np.random.default_rng(seed)
    data = make_data_v2(domain, lam_f, rng, N)
    perm = rng.permutation(N)
    tr, ev = perm[: N // 2], perm[N // 2:]
    img_tr, img_ev = data["IMG"][tr], data["IMG"][ev]
    mean, std_ = img_tr.mean(), img_tr.std() + 1e-8
    img_tr, img_ev = (img_tr - mean) / std_, (img_ev - mean) / std_
    flat_tr = img_tr.reshape(len(tr), -1)
    flat_ev = img_ev.reshape(len(ev), -1)
    fm, fs = flat_tr.mean(0), flat_tr.std(0) + 1e-8
    w_tr = data["W"][tr][:, None]
    w_ev = data["W"][ev][:, None]
    wm, ws = w_tr.mean(0), w_tr.std(0) + 1e-8
    kx_tr, kx_ev = _grams((flat_tr - fm) / fs, (flat_ev - fm) / fs, seed + 101)
    kw_tr, kw_ev = _grams((w_tr - wm) / ws, (w_ev - wm) / ws, seed + 202)
    kj_tr, kj_ev = kx_tr * kw_tr, kx_ev * kw_ev
    a_tr, a_ev = data["A"][tr], data["A"][ev]
    y_ev, ystr_ev = data["Y"][ev], data["Y_struct"][ev]

    net_x = train_cnn(img_tr, a_tr, [(kx_tr, "objective")], M, STEPS, seed)
    x_cap = cnn_disc(net_x, img_tr, a_tr, kx_tr)
    nets = dict(
        x_only=net_x,
        joint=train_cnn(img_tr, a_tr, [(kj_tr, "objective")], M, STEPS, seed),
        constrained=train_cnn(img_tr, a_tr,
                              [(kw_tr, "objective"), (kx_tr, "capped_x")],
                              M, STEPS, seed, x_cap=x_cap))
    row = dict(domain=domain, lam_f=lam_f, rep=rep)
    for name, net in nets.items():
        strata = cnn_strata(net, img_ev)
        est = tau_plugin(strata, a_ev, y_ev, M, len(ev))
        oracle = tau_plugin(strata, a_ev, ystr_ev, M, len(ev))
        row[name] = dict(
            tau_error=float(est - 1.0), oracle_bias=float(oracle - 1.0),
            d_x=float(math.sqrt(max(base.hard_delta_sq(strata, a_ev, kx_ev, M), 0))),
            d_w=float(math.sqrt(max(base.hard_delta_sq(strata, a_ev, kw_ev, M), 0))))
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
        for domain in DOMAINS:
            for rep in range(REPS):
                rows.append(run_once(domain, lam_f, rep))
                print(f"lam={lam_f} {domain} rep={rep + 1}", flush=True)
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
            decision_constrained=decide(sel, "constrained"))
    summary = dict(provenance=dict(
        script="code/benchmark_suite_image_v2_directx.py",
        registration="memo/2026-08-19-rerun-registration-v3-directx.md",
        registration_sha256=REG_SHA, base_seed=SEED_BASE, n=N, reps=REPS,
        steps=STEPS, strata=M, lambdas=list(LAMBDAS)),
        per_lambda=per_lambda, runs=rows)
    out = pathlib.Path(base.RESULTS_DIR) / "benchmark_suite_image_v2_directx.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: dict(decision_joint=v["decision_joint"],
                              decision_constrained=v["decision_constrained"],
                              aggregates=v["aggregates"])
                      for k, v in per_lambda.items()}, indent=1))


if __name__ == "__main__":
    main()
