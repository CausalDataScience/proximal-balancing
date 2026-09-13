#!/usr/bin/env python3
"""Preregistered image benchmark suite (Track 2).

Contract: memo/2026-08-18-benchmark-suite-preregistration.md (frozen).
Domains: (a) local sklearn handwritten digits (8x8), severity driver r(X) =
mean ink intensity; (b) fully synthetic lesion images (32x32), severity
driver r(X) = lesion radius. In both, q(X) = horizontal position factor
drives treatment only. U = r + 0.5 eps, W = U + sigma_w eps, A ~
0.05 + 0.90 expit(0.8 U + 1.5 q), Y = tau A + 2 U + eps. A CNN learns
H = h(X) from pixels; arms joint / constrained / x_only differ only in the
balance objective; estimation is the identical H-stratified plug-in on the
held-out fold. Same logging and decision rule as the tabular track.

Outputs: results/benchmark_suite_image_summary.json, _runs.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import learned_representation_balancing_experiment as base  # noqa: E402
from benchmark_suite_tabular import Row, decide, TRUE_TAU  # noqa: E402

BASE_SEED = 2026081902
PEN_X = 25.0
VARIANTS = ("v1_std", "v2_weakproxy", "v3_strongsel")
ARMS = ("joint", "constrained", "x_only")


def _sig(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(v, -35.0, 35.0)))


_DIGITS: np.ndarray | None = None


def digit_images() -> np.ndarray:
    global _DIGITS
    if _DIGITS is None:
        from sklearn.datasets import load_digits
        _DIGITS = load_digits().images / 16.0            # (1797, 8, 8)
    return _DIGITS


def lesion_images(rng: np.random.Generator, n: int):
    """32x32 synthetic lesions; returns images, radius, x-position."""
    size = 32
    yy, xx = np.mgrid[0:size, 0:size]
    radius = rng.uniform(3.0, 9.0, n)
    cx = rng.uniform(10.0, 22.0, n)
    cy = rng.uniform(10.0, 22.0, n)
    imgs = np.empty((n, size, size), dtype=float)
    for i in range(n):
        blob = np.exp(-(((xx - cx[i]) ** 2 + (yy - cy[i]) ** 2)
                        / (2.0 * radius[i] ** 2)))
        imgs[i] = 0.8 * blob + 0.15 * rng.standard_normal((size, size))
    return imgs, radius, cx


def make_data(domain: str, variant: str, regime: str,
              rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    if domain == "digits":
        pool = digit_images()
        idx = rng.choice(pool.shape[0], size=n, replace=False)
        imgs = pool[idx]
        r_raw = imgs.mean(axis=(1, 2))                    # ink intensity
        col = np.arange(imgs.shape[2])
        q_raw = (imgs * col[None, None, :]).sum((1, 2)) / (
            imgs.sum((1, 2)) + 1e-8)                      # horizontal mass
    elif domain == "lesions":
        imgs, r_raw, q_raw = lesion_images(rng, n)
    else:
        raise ValueError(domain)
    r = (r_raw - r_raw.mean()) / (r_raw.std() + 1e-8)
    q = (q_raw - q_raw.mean()) / (q_raw.std() + 1e-8)
    eps_u = rng.standard_normal(n)
    u = eps_u if regime == "image_signal_null" else r + 0.5 * eps_u
    sigma_w = 1.0 if variant == "v2_weakproxy" else 0.5
    if regime == "proxy_null":
        w = rng.standard_normal(n)
    else:
        w = u + sigma_w * rng.standard_normal(n)
    coef_u = 1.2 if variant == "v3_strongsel" else 0.8
    a = rng.binomial(1, 0.05 + 0.90 * _sig(coef_u * u + 1.5 * q)).astype(float)
    y_struct = TRUE_TAU * a + 2.0 * u
    y = y_struct + rng.standard_normal(n)
    return {"IMG": imgs, "U": u, "W": w, "A": a,
            "Y_struct": y_struct, "Y": y}


class ConvStratumNet(torch.nn.Module):
    def __init__(self, m_strata: int):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(2))
        self.head = torch.nn.Linear(64, m_strata)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


def _grams(tr: np.ndarray, ev: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    bw = base.median_bandwidth(tr, rng)
    return base.rbf_gram(tr, tr, bw), base.rbf_gram(ev, ev, bw)


def train_cnn(imgs: np.ndarray, a: np.ndarray,
              terms: list[tuple[np.ndarray, str]], m: int, steps: int,
              seed: int, x_cap: float | None = None) -> ConvStratumNet:
    torch.manual_seed(seed)
    net = ConvStratumNet(m)
    x_t = torch.tensor(imgs[:, None, :, :], dtype=torch.float32)
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


def cnn_disc(net, imgs, a, gram) -> float:
    with torch.no_grad():
        logits = net(torch.tensor(imgs[:, None], dtype=torch.float32))
        soft = torch.softmax(logits / 0.25, 1)
        disc, _, _ = base.soft_delta_sq(
            soft, torch.tensor(a, dtype=torch.float32),
            torch.tensor(gram, dtype=torch.float32))
    return float(disc)


def cnn_strata(net, imgs) -> np.ndarray:
    with torch.no_grad():
        logits = net(torch.tensor(imgs[:, None], dtype=torch.float32))
    return logits.argmax(dim=1).numpy()


def shift_stats(strata, a, u, m):
    t_sum = s2 = 0.0
    n_neg = 0
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


def run_once(domain: str, variant: str, regime: str, rep: int,
             n: int, m: int, steps: int) -> list[Row]:
    seed = (BASE_SEED + 7919 * rep + 1009 * VARIANTS.index(variant)
            + 104729 * {"primary": 0, "proxy_null": 1,
                        "image_signal_null": 2}[regime]
            + 13 * {"digits": 1, "lesions": 2}[domain])
    rng = np.random.default_rng(seed)
    data = make_data(domain, variant, regime, rng, n)
    perm = rng.permutation(n)
    tr, ev = perm[: n // 2], perm[n // 2:]
    img_tr, img_ev = data["IMG"][tr], data["IMG"][ev]
    mean, std = img_tr.mean(), img_tr.std() + 1e-8
    img_tr, img_ev = (img_tr - mean) / std, (img_ev - mean) / std
    flat_tr = img_tr.reshape(len(tr), -1)
    flat_ev = img_ev.reshape(len(ev), -1)
    w_tr = data["W"][tr][:, None]
    w_ev = data["W"][ev][:, None]
    w_mean, w_std = w_tr.mean(0), w_tr.std(0) + 1e-8
    kx_tr, kx_ev = _grams((flat_tr - flat_tr.mean(0)) / (flat_tr.std(0) + 1e-8),
                          (flat_ev - flat_tr.mean(0)) / (flat_tr.std(0) + 1e-8),
                          seed + 101)
    kw_tr, kw_ev = _grams((w_tr - w_mean) / w_std, (w_ev - w_mean) / w_std,
                          seed + 202)
    kj_tr, kj_ev = kx_tr * kw_tr, kx_ev * kw_ev
    a_tr, a_ev = data["A"][tr], data["A"][ev]
    y_ev, ystr_ev, u_ev = data["Y"][ev], data["Y_struct"][ev], data["U"][ev]

    net_x = train_cnn(img_tr, a_tr, [(kx_tr, "objective")], m, steps, seed)
    x_cap = cnn_disc(net_x, img_tr, a_tr, kx_tr)
    nets = {
        "x_only": net_x,
        "joint": train_cnn(img_tr, a_tr, [(kj_tr, "objective")], m, steps, seed),
        "constrained": train_cnn(
            img_tr, a_tr, [(kw_tr, "objective"), (kx_tr, "capped_x")],
            m, steps, seed, x_cap=x_cap),
    }
    rows: list[Row] = []
    for method, net in nets.items():
        strata = cnn_strata(net, img_ev)
        est, dropped = base.stratified_tau(strata, a_ev, y_ev, m)
        oracle, _ = base.stratified_tau(strata, a_ev, ystr_ev, m)
        t_sum, s2, n_neg = shift_stats(strata, a_ev, u_ev, m)
        rows.append(Row(
            source=domain, variant=variant, regime=regime, rep=rep,
            method=method, estimate=est, tau_error=est - TRUE_TAU,
            oracle_bias=oracle - TRUE_TAU,
            d_joint=math.sqrt(max(base.hard_delta_sq(strata, a_ev, kj_ev, m), 0.0)),
            d_x=math.sqrt(max(base.hard_delta_sq(strata, a_ev, kx_ev, m), 0.0)),
            d_w=math.sqrt(max(base.hard_delta_sq(strata, a_ev, kw_ev, m), 0.0)),
            t_sum=t_sum, s2_sum=s2, n_neg_shift=n_neg, dropped_mass=dropped))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--strata", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.reps, args.steps = 400, 1, 20

    rows: list[Row] = []
    for domain in ("digits", "lesions"):
        for variant in VARIANTS:
            for rep in range(args.reps):
                rows.extend(run_once(domain, variant, "primary", rep,
                                     args.n, args.strata, args.steps))
                print(f"{domain}/{variant} rep={rep + 1}/{args.reps}",
                      flush=True)
    controls: list[Row] = []
    for domain in ("digits", "lesions"):
        for regime in ("proxy_null", "image_signal_null"):
            for rep in range(max(args.reps // 2, 1)):
                controls.extend(run_once(domain, "v1_std", regime, rep,
                                         args.n, args.strata, args.steps))
                print(f"control {domain}/{regime} rep={rep + 1}", flush=True)

    summary = dict(provenance=dict(
        script="code/benchmark_suite_image.py",
        preregistration="memo/2026-08-18-benchmark-suite-preregistration.md",
        base_seed=BASE_SEED, n=args.n, reps=args.reps, steps=args.steps,
        strata=args.strata, arms=list(ARMS), pen_x=PEN_X, true_tau=TRUE_TAU,
        domains=dict(digits="sklearn load_digits, severity=ink intensity",
                     lesions="synthetic 32x32 blobs, severity=radius"),
        matched_learner_contract="identical CNN, seed, optimizer, schedule; "
                                 "only the balance objective differs"))
    per_domain = {}
    for domain in ("digits", "lesions"):
        d_rows = [r for r in rows if r.source == domain]
        agg = {}
        for method in ARMS:
            sel = [r for r in d_rows if r.method == method]
            agg[method] = dict(
                mean_abs_bias=float(np.mean([abs(r.oracle_bias) for r in sel])),
                rmse=float(np.sqrt(np.mean([r.tau_error ** 2 for r in sel]))),
                mean_d_w=float(np.mean([r.d_w for r in sel])),
                mean_d_x=float(np.mean([r.d_x for r in sel])),
                mean_T=float(np.mean([r.t_sum for r in sel])),
                mean_S2=float(np.mean([r.s2_sum for r in sel])),
                neg_shift_frac=float(np.mean([r.n_neg_shift for r in sel]))
                / args.strata)
        per_domain[domain] = dict(
            aggregates=agg,
            decision_joint_vs_xonly=decide(d_rows, "joint"),
            decision_constrained_vs_xonly=decide(d_rows, "constrained"))
    control_summary = {}
    for regime in ("proxy_null", "image_signal_null"):
        creg = [r for r in controls if r.regime == regime]
        control_summary[regime] = {
            arm: decide(creg, arm)["win_rate_bias_and_rmse"]
            for arm in ("joint", "constrained")}
    summary["per_domain"] = per_domain
    summary["negative_controls_win_rates"] = control_summary

    base.RESULTS_DIR.mkdir(exist_ok=True)
    (base.RESULTS_DIR / "benchmark_suite_image_summary.json").write_text(
        json.dumps(summary, indent=2))
    with (base.RESULTS_DIR / "benchmark_suite_image_runs.csv").open(
            "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows + controls:
            writer.writerow(asdict(row))
    print(json.dumps(per_domain, indent=1)[:1600])
    print("controls:", json.dumps(control_summary))


if __name__ == "__main__":
    main()
