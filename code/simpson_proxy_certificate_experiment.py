"""Simpson-paradox experiment for the single-proxy certificate.

Two worlds share the same observable-X analysis surface and differ only in
latent selection:
  World CONF: every hospital treats high-severity patients more,
    e(x,u) = clip(sigmoid(a_x + b_sel (2u-1))), b_sel = 1.5.
  World SAFE: b_sel = 0, so U is not selected on at all.

Outcome fixed once: Y = tau_true A + beta U + eps with tau_true = -1, beta = 3.
In CONF the latent floor beta * sum_x p_x delta(x) is large and positive, so
the naive contrast, full-X adjustment, and every phi(X)-stratification report
a POSITIVE effect near +0.8 while the true effect is -1: a Simpson reversal
that X-based analysis cannot detect. The proxy discrepancy D-hat separates
the worlds (large in CONF, near zero in SAFE), the certificate
Gamma_true * D-hat covers the true effect in CONF where naive and full-X
confidence intervals miss it, and the breakdown value |tau-hat|/D-hat shows
how little stability is needed to overturn the sign.

Learners are mechanism-blind greedy stratifiers as before (W-supervised,
X-only, random at M=3), with full-X and naive as references.

numpy only. Output: results/simpson_proxy_certificate_summary.json
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

S, M = 20, 3
ETA, P_MIN = 0.10, 0.05
SIGMA_X, W_SCALE = 4.0, 2.0
SEED = 20260821
TAU_TRUE, BETA, NOISE_SD, RHO = -1.0, 3.0, 0.5, 0.6

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAT = [(x, u) for x in range(S) for u in range(2)]
OBS = [(x, w) for x in range(S) for w in range(2)]


def gram(full: bool) -> np.ndarray:
    k = np.empty((2 * S, 2 * S))
    for i, (x, w) in enumerate(OBS):
        for j, (x2, w2) in enumerate(OBS):
            val = -0.5 * ((x - x2) / SIGMA_X) ** 2
            if full:
                val -= 0.5 * (W_SCALE * (w - w2)) ** 2
            k[i, j] = math.exp(val)
    return k


K_FULL, K_XONLY = gram(True), gram(False)


def bridge_norm(g_vals: np.ndarray) -> float:
    lam, vec = np.linalg.eigh(K_FULL)
    keep = lam > lam.max() * 1e-10
    coef = vec[:, keep].T @ g_vals
    return float(math.sqrt(np.sum(coef ** 2 / lam[keep])))


def build_dgp(b_sel: float) -> dict:
    p_u1 = 0.3 + 0.4 * np.arange(S) / (S - 1)
    p_lat = np.array([(p_u1[x] if u == 1 else 1 - p_u1[x]) / S
                      for x, u in LAT])
    a_x = np.linspace(-1.0, 1.0, S)
    zval = np.array([a_x[x] + b_sel * (2 * u - 1) for x, u in LAT])
    e = np.clip(1.0 / (1.0 + np.exp(-zval)), ETA, 1 - ETA)
    q0, q1 = 0.5 - RHO / 2, 0.5 + RHO / 2
    cmat = np.zeros((2 * S, 2 * S))
    for j, (x, u) in enumerate(LAT):
        pw1 = q1 if u == 1 else q0
        cmat[2 * x + 1, j] = pw1
        cmat[2 * x + 0, j] = 1.0 - pw1
    mu = np.array([[TAU_TRUE * a + BETA * u for _, u in LAT]
                   for a in range(2)], dtype=float)
    g1 = np.array([TAU_TRUE + BETA * (w - q0) / RHO for _, w in OBS])
    g0 = np.array([BETA * (w - q0) / RHO for _, w in OBS])
    gam = max(bridge_norm(g1), bridge_norm(g0))
    p1_lat, p0_lat = p_lat * e, p_lat * (1 - e)
    naive = float((mu[1] @ p1_lat) / p1_lat.sum()
                  - (mu[0] @ p0_lat) / p0_lat.sum())
    return dict(p_lat=p_lat, p1_lat=p1_lat, p0_lat=p0_lat, e=e,
                G=cmat.T @ K_FULL @ cmat, mu=mu, tau=TAU_TRUE,
                gamma_true=gam, naive_pop=naive)


def pop_stats(dgp: dict, lab: np.ndarray, m: int) -> dict | None:
    d_total, tau_phi = 0.0, 0.0
    for z in range(m):
        idx = np.array([j for j, (x, _) in enumerate(LAT) if lab[x] == z])
        if idx.size == 0:
            return None
        m1, m0 = dgp["p1_lat"][idx].sum(), dgp["p0_lat"][idx].sum()
        p_z = m1 + m0
        if m == M and p_z < P_MIN - 1e-12:
            return None
        q1 = np.zeros(2 * S); q1[idx] = dgp["p1_lat"][idx] / m1
        q0 = np.zeros(2 * S); q0[idx] = dgp["p0_lat"][idx] / m0
        d = q1 - q0
        d_total += p_z * math.sqrt(max(0.0, d @ dgp["G"] @ d))
        tau_phi += p_z * ((dgp["mu"][1] @ q1) - (dgp["mu"][0] @ q0))
    return dict(D=d_total, bias=tau_phi - dgp["tau"])


def emp_D(counts, kmat, lab, n) -> float | None:
    total = 0.0
    for z in range(M):
        idx = np.array([2 * xx + w for xx in range(S) if lab[xx] == z
                        for w in (0, 1)], dtype=int)
        if idx.size == 0:
            return None
        c1z, c0z = counts[1, idx].sum(), counts[0, idx].sum()
        if min(c1z, c0z) < 2 or (c1z + c0z) / n < P_MIN:
            return None
        r1 = np.zeros(2 * S); r1[idx] = counts[1, idx] / c1z
        r0 = np.zeros(2 * S); r0[idx] = counts[0, idx] / c0z
        dd = r1 - r0
        total += ((c1z + c0z) / n) * math.sqrt(max(0.0, dd @ kmat @ dd))
    return total


def greedy(counts, kmat, n, rng, restarts: int = 5):
    best_lab, best_val = None, math.inf
    for _ in range(restarts):
        lab = None
        for _ in range(50):
            cand = rng.integers(0, M, S)
            v = emp_D(counts, kmat, cand, n)
            if v is not None:
                lab, val = cand, v
                break
        if lab is None:
            continue
        improved = True
        while improved:
            improved = False
            for x in range(S):
                for z in range(M):
                    if z == lab[x]:
                        continue
                    trial = lab.copy(); trial[x] = z
                    tv = emp_D(counts, kmat, trial, n)
                    if tv is not None and tv < val - 1e-12:
                        lab, val, improved = trial, tv, True
        if val < best_val:
            best_lab, best_val = lab, val
    return best_lab


def tau_plugin(x, a, y, lab, m, n) -> float:
    total = 0.0
    for z in range(m):
        in_z = np.isin(x, np.flatnonzero(lab == z))
        y1, y0 = y[(a == 1) & in_z], y[(a == 0) & in_z]
        if y1.size and y0.size:
            total += (in_z.sum() / n) * (y1.mean() - y0.mean())
    return total


def run_world(name: str, b_sel: float, n: int, reps: int,
              rng: np.random.Generator) -> dict:
    dgp = build_dgp(b_sel)
    gam = dgp["gamma_true"]
    full_x = pop_stats(dgp, np.arange(S), S)
    acc = {k: dict(absbias=[], D=[], tau_hat=[])
           for k in ("w_supervised", "x_only", "random")}
    naive_cov = fullx_cov = cert_cov = 0
    d_hat_sel_list, breakdown = [], []
    for _ in range(reps):
        draw = rng.choice(2 * S, size=n, p=dgp["p_lat"])
        x = np.array([LAT[j][0] for j in draw])
        u = np.array([LAT[j][1] for j in draw])
        a = (rng.random(n) < dgp["e"][draw]).astype(int)
        w = (rng.random(n) < (0.5 + RHO * (u - 0.5))).astype(int)
        y = TAU_TRUE * a + BETA * u + rng.normal(0, NOISE_SD, n)
        counts = np.zeros((2, 2 * S))
        for ai, xi, wi in zip(a, x, w):
            counts[ai, 2 * xi + wi] += 1
        y1, y0 = y[a == 1], y[a == 0]
        naive_hat = y1.mean() - y0.mean()
        naive_se = math.sqrt(y1.var() / y1.size + y0.var() / y0.size)
        naive_cov += int(abs(naive_hat - TAU_TRUE) <= 1.96 * naive_se)
        fullx_hat = tau_plugin(x, a, y, np.arange(S), S, n)
        fullx_se = 2.0 * y.std() / math.sqrt(n * (1 / S) * ETA)
        fullx_cov += int(abs(fullx_hat - TAU_TRUE) <= 1.96 * fullx_se)
        labs = dict(w_supervised=greedy(counts, K_FULL, n, rng),
                    x_only=greedy(counts, K_XONLY, n, rng))
        for _ in range(200):
            cand = rng.integers(0, M, S)
            if emp_D(counts, K_FULL, cand, n) is not None:
                labs["random"] = cand
                break
        for mname, lab in labs.items():
            if lab is None:
                continue
            st = pop_stats(dgp, lab, M)
            if st is None:
                continue
            acc[mname]["absbias"].append(abs(st["bias"]))
            acc[mname]["D"].append(st["D"])
            acc[mname]["tau_hat"].append(tau_plugin(x, a, y, lab, M, n))
        lab_sel = labs["w_supervised"]
        if lab_sel is not None:
            d_hat = emp_D(counts, K_FULL, lab_sel, n)
            tau_hat = tau_plugin(x, a, y, lab_sel, M, n)
            se = 2.0 * y.std() / math.sqrt(n * P_MIN * ETA)
            half = 1.96 * se + gam * d_hat
            cert_cov += int(abs(tau_hat - TAU_TRUE) <= half)
            d_hat_sel_list.append(d_hat)
            breakdown.append(abs(tau_hat) / d_hat if d_hat > 1e-12
                             else math.inf)
    def agg(v):
        return dict(mean_abs_pop_bias=float(np.mean(v["absbias"])),
                    mean_pop_D=float(np.mean(v["D"])),
                    mean_certificate=float(gam * np.mean(v["D"])),
                    mean_tau_hat=float(np.mean(v["tau_hat"])))
    return dict(world=name, b_sel=b_sel, gamma_true=gam,
                tau_true=TAU_TRUE, naive_pop=dgp["naive_pop"],
                full_x=dict(pop_bias=full_x["bias"], pop_D=full_x["D"],
                            certificate=gam * full_x["D"]),
                methods={k: agg(v) for k, v in acc.items() if v["absbias"]},
                mean_D_hat_selected=float(np.mean(d_hat_sel_list)),
                mean_breakdown_gamma=float(np.mean(breakdown)),
                coverage=dict(naive_ci=naive_cov / reps,
                              full_x_ci=fullx_cov / reps,
                              certificate_interval=cert_cov / reps))


def main(n: int = 20000, reps: int = 10) -> None:
    rng = np.random.default_rng(SEED)
    worlds = [run_world("CONF", 1.5, n, reps, rng),
              run_world("SAFE", 0.0, n, reps, rng)]
    summary = dict(provenance=dict(
        script="code/simpson_proxy_certificate_experiment.py",
        date="2026-08-18", seed=SEED, S=S, M=M, n=n, reps=reps,
        tau_true=TAU_TRUE, beta=BETA, rho=RHO,
        design="Simpson reversal world vs safe world; Y = tau A + beta U"),
        worlds=worlds)
    out = ROOT / "results" / "simpson_proxy_certificate_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    for wd in worlds:
        print(f"== {wd['world']} (b_sel={wd['b_sel']}) gamma="
              f"{wd['gamma_true']:.2f} naive_pop_bias="
              f"{wd['naive_pop'] - wd['tau_true']:.3f} fullX_bias="
              f"{wd['full_x']['pop_bias']:.3f}")
        for mname, v in wd["methods"].items():
            print(f"   {mname}: |bias|={v['mean_abs_pop_bias']:.3f} "
                  f"D={v['mean_pop_D']:.3f} cert={v['mean_certificate']:.3f} "
                  f"tau_hat={v['mean_tau_hat']:.3f}")
        print(f"   D_hat_sel={wd['mean_D_hat_selected']:.3f} "
              f"breakdown_gamma={wd['mean_breakdown_gamma']:.2f} "
              f"coverage naive/fullX/cert = "
              f"{wd['coverage']['naive_ci']:.2f}/"
              f"{wd['coverage']['full_x_ci']:.2f}/"
              f"{wd['coverage']['certificate_interval']:.2f}")


if __name__ == "__main__":
    main()
