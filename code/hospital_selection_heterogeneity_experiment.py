"""Hospital selection-heterogeneity experiment for Single Proxy Balancing.

Mechanism-first design (no outcome reverse-engineering):
  - X = hospital id x in {0..S-1} carrying an observable case-mix score s=x/S.
  - Practice type t(x): hospitals x < S/2 treat high-severity patients MORE
    (t=+1), hospitals x >= S/2 treat them LESS (t=-1). Types are invisible
    to every learner; hospitals x and x+S/2 share the same score-level
    propensity a-level, so marginal e(x) does not reveal the type.
  - U in {0,1} severity, P(U=1|x)=0.5.
  - Treatment e(x,u) = clip(sigmoid(a_x + t(x) * b_sel * (2u-1)), eta, 1-eta).
  - Proxy W in {0,1}: pre-treatment lab, P(W=1|U=u) = 0.5 + rho*(u - 0.5),
    treatment-free. rho = proxy informativeness.
  - Outcome fixed once and for all: Y = tau_true * A + beta * U + eps.
    mu_a(x,u) = tau_true * a + beta * u has no x-dependence, so all
    stratification bias comes from within-stratum latent (U) imbalance:
    bias(phi) = beta * sum_z p_z {E[U|A=1,z] - E[U|A=0,z]}.

Learners (never see the mechanism): greedy local search over phi: X -> [M]
minimizing the empirical conditional discrepancy, with the full (x,w) kernel
(W-supervised) or the x-only kernel (X-only); plus a random admissible map.
Full-X adjustment is the reference. Sweep over rho x b_sel x n, reporting
mean |population bias|, mean D, certificate Gamma_true(rho) * mean D,
tau-hat RMSE, and a type-mixing diagnostic per learned stratification.

Gamma_true(rho): the exact outcome bridge is g_a(x,w) = tau_true * a
+ beta * (w - q0) / rho (then T g_a = mu_a exactly); its RKHS norm is
computed with a spectral pseudoinverse of the kernel Gram matrix and the
certificate validity is additionally checked empirically per map.

numpy only. Output: results/hospital_selection_heterogeneity_summary.json
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

S, M = 20, 3
ETA, P_MIN = 0.10, 0.05
SIGMA_X, W_SCALE = 4.0, 2.0
SEED = 20260820
TAU_TRUE, BETA, NOISE_SD = 1.0, 2.0, 0.5

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


def bridge_norm(g_vals: np.ndarray) -> tuple[float, float]:
    """RKHS norm of a function given by values on the support, via pinv."""
    lam, vec = np.linalg.eigh(K_FULL)
    keep = lam > lam.max() * 1e-10
    coef = vec[:, keep].T @ g_vals
    resid = float(np.linalg.norm(g_vals - vec[:, keep] @ coef)
                  / max(1.0, np.linalg.norm(g_vals)))
    return float(math.sqrt(np.sum(coef ** 2 / lam[keep]))), resid


def build_dgp(rho: float, b_sel: float) -> dict:
    p_lat = np.full(2 * S, 1.0 / (2 * S))
    a_lev = np.linspace(-1.0, 1.0, S // 2)
    t_x = np.array([1.0 if x < S // 2 else -1.0 for x in range(S)])
    zval = np.array([a_lev[x % (S // 2)] + t_x[x] * b_sel * (2 * u - 1)
                     for x, u in LAT])
    e = np.clip(1.0 / (1.0 + np.exp(-zval)), ETA, 1 - ETA)
    q0, q1 = 0.5 - rho / 2, 0.5 + rho / 2
    cmat = np.zeros((2 * S, 2 * S))
    for j, (x, u) in enumerate(LAT):
        pw1 = q1 if u == 1 else q0
        cmat[2 * x + 1, j] = pw1
        cmat[2 * x + 0, j] = 1.0 - pw1
    mu = np.array([[TAU_TRUE * a + BETA * u for _, u in LAT]
                   for a in range(2)], dtype=float)
    g1 = np.array([TAU_TRUE + BETA * (w - q0) / rho for _, w in OBS])
    g0 = np.array([BETA * (w - q0) / rho for _, w in OBS])
    n1, r1 = bridge_norm(g1)
    n0, r0 = bridge_norm(g0)
    return dict(p_lat=p_lat, p1_lat=p_lat * e, p0_lat=p_lat * (1 - e), e=e,
                G=cmat.T @ K_FULL @ cmat, mu=mu, tau=TAU_TRUE, t_x=t_x,
                gamma_true=max(n1, n0), bridge_resid=max(r1, r0), rho=rho)


def pop_stats(dgp: dict, lab: np.ndarray, m: int) -> dict | None:
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
          n: int) -> float | None:
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


def greedy(counts: np.ndarray, kmat: np.ndarray, n: int,
           rng: np.random.Generator, restarts: int = 5) -> np.ndarray | None:
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


def type_mixing(lab: np.ndarray, t_x: np.ndarray) -> float:
    """Mean absolute net practice-type imbalance across strata (0 = mixed)."""
    vals = []
    for z in range(M):
        members = np.flatnonzero(lab == z)
        if members.size:
            vals.append(abs(t_x[members].sum()) / members.size)
    return float(np.mean(vals))


def tau_plugin(x, a, y, lab, n) -> float:
    total = 0.0
    for z in range(M):
        in_z = np.isin(x, np.flatnonzero(lab == z))
        y1, y0 = y[(a == 1) & in_z], y[(a == 0) & in_z]
        if y1.size and y0.size:
            total += (in_z.sum() / n) * (y1.mean() - y0.mean())
    return total


def run_config(rho: float, b_sel: float, n: int, reps: int,
               rng: np.random.Generator) -> dict:
    dgp = build_dgp(rho, b_sel)
    gam = dgp["gamma_true"]
    acc = {k: dict(absbias=[], D=[], tau_err=[], mix=[])
           for k in ("w_supervised", "x_only", "random")}
    cert_viol = 0
    for _ in range(reps):
        draw = rng.choice(2 * S, size=n, p=dgp["p_lat"])
        x = np.array([LAT[j][0] for j in draw])
        a = (rng.random(n) < dgp["e"][draw]).astype(int)
        u = np.array([LAT[j][1] for j in draw])
        w = (rng.random(n) < (0.5 + rho * (u - 0.5))).astype(int)
        y = TAU_TRUE * a + BETA * u + rng.normal(0, NOISE_SD, n)
        counts = np.zeros((2, 2 * S))
        for ai, xi, wi in zip(a, x, w):
            counts[ai, 2 * xi + wi] += 1
        labs = dict(w_supervised=greedy(counts, K_FULL, n, rng),
                    x_only=greedy(counts, K_XONLY, n, rng))
        for _ in range(200):
            cand = rng.integers(0, M, S)
            if emp_D(counts, K_FULL, cand, n) is not None:
                labs["random"] = cand
                break
        for name, lab in labs.items():
            if lab is None:
                continue
            st = pop_stats(dgp, lab, M)
            if st is None:
                continue
            if abs(st["bias"]) > gam * st["D"] + 1e-9:
                cert_viol += 1
            acc[name]["absbias"].append(abs(st["bias"]))
            acc[name]["D"].append(st["D"])
            acc[name]["tau_err"].append(tau_plugin(x, a, y, lab, n)
                                        - dgp["tau"])
            acc[name]["mix"].append(type_mixing(lab, dgp["t_x"]))
    full_x = pop_stats(dgp, np.arange(S), S)
    def agg(v):
        return dict(mean_abs_pop_bias=float(np.mean(v["absbias"])),
                    mean_pop_D=float(np.mean(v["D"])),
                    mean_certificate=float(gam * np.mean(v["D"])),
                    tau_rmse=float(np.sqrt(np.mean(np.square(v["tau_err"])))),
                    mean_type_mixing=float(np.mean(v["mix"])),
                    n_reps_used=len(v["absbias"]))
    return dict(rho=rho, b_sel=b_sel, n=n, gamma_true=gam,
                bridge_residual=dgp["bridge_resid"],
                certificate_violations=cert_viol,
                methods={k: agg(v) for k, v in acc.items() if v["absbias"]},
                full_x_reference=dict(pop_bias=full_x["bias"],
                                      pop_D=full_x["D"],
                                      certificate=gam * full_x["D"]))


def main() -> None:
    rng = np.random.default_rng(SEED)
    configs = [(rho, b, n) for rho in (0.6, 0.3, 0.1)
               for b in (1.5, 0.75) for n in (20000, 5000)]
    rows = [run_config(rho, b, n, reps=10, rng=rng) for rho, b, n in configs]
    summary = dict(
        provenance=dict(
            script="code/hospital_selection_heterogeneity_experiment.py",
            date="2026-08-18", seed=SEED, S=S, M=M,
            tau_true=TAU_TRUE, beta=BETA, reps_per_config=10,
            design="mechanism-first hospital DGP; Y = tau A + beta U + eps"),
        configs=rows)
    out = ROOT / "results" / "hospital_selection_heterogeneity_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    key = ["rho", "b_sel", "n"]
    for r in rows:
        head = " ".join(f"{k}={r[k]}" for k in key)
        line = " | ".join(
            f"{m}: bias={v['mean_abs_pop_bias']:.3f} D={v['mean_pop_D']:.3f}"
            f" cert={v['mean_certificate']:.3f} mix={v['mean_type_mixing']:.2f}"
            for m, v in r["methods"].items())
        print(f"[{head}] gamma={r['gamma_true']:.2f} viol="
              f"{r['certificate_violations']} fullX_bias="
              f"{r['full_x_reference']['pop_bias']:.4f} :: {line}")


if __name__ == "__main__":
    main()
