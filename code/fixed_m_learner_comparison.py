"""Fixed-M learner comparison: W-supervised vs X-only vs random stratification.

Follows the main-comparison discipline in memo/2026-08-18-finite-state-
experiment-plan.md: same M, same partition class, same greedy learner; only
the objective differs. Full-X adjustment is reported as a reference, and the
oracle population quantities (D, bias, certificate) are exact because the
state space is finite. S=20 makes exhaustive enumeration infeasible (3^20),
which is the regime where a learner is actually needed.

DGP cancellation structure: propensity u-coefficient alternates sign across
x (b_x = +b for even x, -b for odd x) while pairs (2j, 2j+1) share the same
x-level a_j. Grouping opposite-sign x's cancels the latent tilt inside a
stratum; the X-only objective cannot see the difference, the W-supervised
objective can.

numpy only. Output: results/fixed_m_learner_comparison_summary.json
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

S, M = 20, 3
ETA, P_MIN = 0.10, 0.05
SIGMA_X, W_SCALE = 4.0, 2.0
SEED = 20260819
B_TILT = 1.5

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


def build_dgp() -> dict:
    rng = np.random.default_rng(SEED)
    p_lat = np.full(2 * S, 1.0 / (2 * S))              # P(U=1|x)=0.5, uniform x
    a_lev = np.linspace(-1.0, 1.0, S // 2)
    zvals = np.array([a_lev[x // 2] + (B_TILT if x % 4 == 0 else -0.6)
                      * (2 * u - 1) for x, u in LAT])
    e = np.clip(1.0 / (1.0 + np.exp(-zvals)), ETA, 1 - ETA)
    cmat = np.zeros((2 * S, 2 * S))
    for j, (x, u) in enumerate(LAT):
        pw1 = 0.8 if u == 1 else 0.2
        cmat[2 * x + 1, j] = pw1
        cmat[2 * x + 0, j] = 1.0 - pw1
    kfull = gram(full=True)
    base = np.array([0.15 if w == 1 else -0.15 for _, w in OBS])
    alpha = np.vstack([base, base]) + rng.normal(0.0, 0.05, size=(2, 2 * S))
    g = alpha @ kfull
    gam_true = max(math.sqrt(alpha[a] @ kfull @ alpha[a]) for a in range(2))
    mu = g @ cmat
    tau = float((mu[1] - mu[0]) @ p_lat)
    assert abs(tau - float((g[1] - g[0]) @ (cmat @ p_lat))) < 1e-10
    return dict(p_lat=p_lat, p1_lat=p_lat * e, p0_lat=p_lat * (1 - e), e=e,
                K=kfull, K_xonly=gram(full=False), G=cmat.T @ kfull @ cmat,
                mu=mu, tau=tau, gamma_true=gam_true)


def pop_stats(dgp: dict, lab: np.ndarray, m: int) -> dict | None:
    """Exact population p_z, D (full kernel), bias for a stratification."""
    d_total, tau_phi = 0.0, 0.0
    for z in range(m):
        idx = np.array([j for j, (x, _) in enumerate(LAT) if lab[x] == z])
        if idx.size == 0:
            return None
        m1, m0 = dgp["p1_lat"][idx].sum(), dgp["p0_lat"][idx].sum()
        p_z = m1 + m0
        if p_z < P_MIN - 1e-12:
            return None
        q1 = np.zeros(2 * S); q1[idx] = dgp["p1_lat"][idx] / m1
        q0 = np.zeros(2 * S); q0[idx] = dgp["p0_lat"][idx] / m0
        d = q1 - q0
        d_total += p_z * math.sqrt(max(0.0, d @ dgp["G"] @ d))
        tau_phi += p_z * ((dgp["mu"][1] @ q1) - (dgp["mu"][0] @ q0))
    return dict(D=d_total, bias=tau_phi - dgp["tau"])


def emp_D(counts: np.ndarray, kmat: np.ndarray, lab: np.ndarray,
          n: int, m: int) -> float | None:
    total = 0.0
    for z in range(m):
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


def greedy(counts: np.ndarray, kmat: np.ndarray, n: int,
           rng: np.random.Generator, restarts: int = 5) -> np.ndarray:
    best_lab, best_val = None, math.inf
    for _ in range(restarts):
        lab, val = None, None
        for _ in range(50):                      # admissible random init
            cand = rng.integers(0, M, S)
            v = emp_D(counts, kmat, cand, n, M)
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
                    tv = emp_D(counts, kmat, trial, n, M)
                    if tv is not None and tv < val - 1e-12:
                        lab, val, improved = trial, tv, True
        if val < best_val:
            best_lab, best_val = lab, val
    return best_lab


def tau_plugin(x: np.ndarray, a: np.ndarray, y: np.ndarray,
               lab: np.ndarray, m: int, n: int) -> float:
    total = 0.0
    for z in range(m):
        in_z = np.isin(x, np.flatnonzero(lab == z))
        if in_z.sum() == 0:
            continue
        y1 = y[(a == 1) & in_z]
        y0 = y[(a == 0) & in_z]
        if y1.size == 0 or y0.size == 0:
            continue
        total += (in_z.sum() / n) * (y1.mean() - y0.mean())
    return total


def main(n: int = 20000, reps: int = 10) -> None:
    dgp = build_dgp()
    gam = dgp["gamma_true"]
    rng = np.random.default_rng(SEED + 7)
    full_x = pop_stats(dgp, np.arange(S), S)
    acc = {k: dict(absbias=[], D=[], tau_err=[])
           for k in ("w_supervised", "x_only", "random")}
    for _ in range(reps):
        draw = rng.choice(2 * S, size=n, p=dgp["p_lat"])
        x = np.array([LAT[j][0] for j in draw])
        a = (rng.random(n) < dgp["e"][draw]).astype(int)
        pw1 = np.where(np.array([LAT[j][1] for j in draw]) == 1, 0.8, 0.2)
        w = (rng.random(n) < pw1).astype(int)
        y = dgp["mu"][1][draw] * a + dgp["mu"][0][draw] * (1 - a) \
            + rng.normal(0, 0.3, n)
        counts = np.zeros((2, 2 * S))
        for ai, xi, wi in zip(a, x, w):
            counts[ai, 2 * xi + wi] += 1
        labs = dict(
            w_supervised=greedy(counts, dgp["K"], n, rng),
            x_only=greedy(counts, dgp["K_xonly"], n, rng))
        for _ in range(200):                      # admissible random baseline
            cand = rng.integers(0, M, S)
            if emp_D(counts, dgp["K"], cand, n, M) is not None:
                labs["random"] = cand
                break
        for name, lab in labs.items():
            st = pop_stats(dgp, lab, M)
            if st is None:
                continue
            acc[name]["absbias"].append(abs(st["bias"]))
            acc[name]["D"].append(st["D"])
            acc[name]["tau_err"].append(
                tau_plugin(x, a, y, lab, M, n) - dgp["tau"])
    def agg(v: dict) -> dict:
        return dict(mean_abs_pop_bias=float(np.mean(v["absbias"])),
                    mean_pop_D=float(np.mean(v["D"])),
                    mean_certificate=float(gam * np.mean(v["D"])),
                    tau_rmse=float(np.sqrt(np.mean(np.square(v["tau_err"])))),
                    n_reps_used=len(v["absbias"]))
    summary = dict(
        provenance=dict(script="code/fixed_m_learner_comparison.py",
                        plan="memo/2026-08-18-finite-state-experiment-plan.md",
                        date="2026-08-18", seed=SEED, S=S, M=M, n=n,
                        reps=reps, gamma_true=gam, tau=dgp["tau"]),
        methods={k: agg(v) for k, v in acc.items()},
        full_x_reference=dict(pop_bias=full_x["bias"], pop_D=full_x["D"],
                              certificate=gam * full_x["D"]))
    out = ROOT / "results" / "fixed_m_learner_comparison_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
