"""Finite-state population verification for Single Proxy Balancing.

Implements memo/2026-08-18-finite-state-experiment-plan.md on top of THEORY.md.
Every assumption holds by construction (bridge-first DGP), all representations
phi: {0..S-1} -> {0..M-1} are enumerated exhaustively (eps_opt = 0), and all
oracle quantities (D, bias, Gamma*, certificate) are exact finite matrices.

Regimes:
  R1 exact          propensity depends on x only with M levels -> attainability.
  R2 approximate    propensity depends on (x,u) -> certificate verification.
  R4 sweep          channel weakens toward non-injectivity -> Gamma* divergence.
  finite-sample     r_n event and combined sensitivity-interval coverage.

numpy only. Output: results/finite_state_bridge_summary.json
"""

from __future__ import annotations

import itertools
import json
import math
import pathlib

import numpy as np

S, M = 10, 3
ETA, P_MIN = 0.10, 0.05
SIGMA_X, W_SCALE = 2.0, 2.0
SEED = 20260818
TOL = 1e-9

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAT = [(x, u) for x in range(S) for u in range(2)]   # (x, u) support
OBS = [(x, w) for x in range(S) for w in range(2)]   # (x, w) support


def kernel_matrix() -> np.ndarray:
    k = np.empty((2 * S, 2 * S))
    for i, (x, w) in enumerate(OBS):
        for j, (x2, w2) in enumerate(OBS):
            k[i, j] = math.exp(-0.5 * ((x - x2) / SIGMA_X) ** 2
                               - 0.5 * (W_SCALE * (w - w2)) ** 2)
    return k


def channel_matrix(c1: float, c0: float) -> np.ndarray:
    """C[(x,w),(x,u)] = P(W=w | U=u), zero across different x."""
    c = np.zeros((2 * S, 2 * S))
    for j, (x, u) in enumerate(LAT):
        pw1 = c1 if u == 1 else c0
        c[OBS.index((x, 1)), j] = pw1
        c[OBS.index((x, 0)), j] = 1.0 - pw1
    return c


def build_dgp(regime: str, c1: float = 0.8, c0: float = 0.2) -> dict:
    rng = np.random.default_rng(SEED)
    p_x = np.full(S, 1.0 / S)
    p_u1 = np.linspace(0.15, 0.85, S)
    p_lat = np.array([p_x[x] * (p_u1[x] if u == 1 else 1 - p_u1[x])
                      for x, u in LAT])
    if regime == "R1":
        levels = [0.25, 0.50, 0.75]
        e = np.array([levels[0 if x < 3 else (1 if x < 7 else 2)]
                      for x, _ in LAT])
    else:
        z = np.array([0.8 * (x - 4.5) / 2.5 + 1.2 * (2 * u - 1)
                      for x, u in LAT])
        e = np.clip(1.0 / (1.0 + np.exp(-z)), ETA, 1 - ETA)
    kmat = kernel_matrix()
    cmat = channel_matrix(c1, c0)
    alpha = rng.normal(0.0, 0.3, size=(2, 2 * S))
    g = alpha @ kmat                                  # g_a on OBS support
    gam_true = max(math.sqrt(alpha[a] @ kmat @ alpha[a]) for a in range(2))
    mu = g @ cmat                                     # mu_a = C^T g_a on LAT
    p1_lat, p0_lat = p_lat * e, p_lat * (1 - e)
    tau = float((mu[1] - mu[0]) @ p_lat)
    tau_obs = float((g[1] - g[0]) @ (cmat @ p_lat))
    assert abs(tau - tau_obs) < 1e-10
    return dict(p_lat=p_lat, p1_lat=p1_lat, p0_lat=p0_lat, e=e, K=kmat,
                C=cmat, G=cmat.T @ kmat @ cmat, g=g, mu=mu, tau=tau,
                gamma_true=gam_true)


def stratum_members(labels: np.ndarray) -> list[np.ndarray]:
    return [np.array([j for j, (x, _) in enumerate(LAT) if labels[x] == z])
            for z in range(M)]


def phi_stats(dgp: dict, labels: np.ndarray) -> dict | None:
    """Population p_z, pi_z, Delta_z, D, tau_phi for one representation."""
    out = dict(p_z=[], pi_z=[], delta=[], q1=[], q0=[])
    for idx in stratum_members(labels):
        if idx.size == 0:
            return None
        m1, m0 = dgp["p1_lat"][idx].sum(), dgp["p0_lat"][idx].sum()
        p_z = m1 + m0
        if p_z < P_MIN:
            return None
        q1 = np.zeros(2 * S); q1[idx] = dgp["p1_lat"][idx] / m1
        q0 = np.zeros(2 * S); q0[idx] = dgp["p0_lat"][idx] / m0
        d = q1 - q0
        delta = math.sqrt(max(0.0, d @ dgp["G"] @ d))
        out["p_z"].append(p_z); out["pi_z"].append(m1 / p_z)
        out["delta"].append(delta); out["q1"].append(q1); out["q0"].append(q0)
    out["D"] = float(sum(p * d for p, d in zip(out["p_z"], out["delta"])))
    tau_phi = sum(p * ((dgp["mu"][1] @ q1) - (dgp["mu"][0] @ q0))
                  for p, q1, q0 in zip(out["p_z"], out["q1"], out["q0"]))
    out["bias"] = float(tau_phi - dgp["tau"])
    return out


def gamma_star(dgp: dict, idx: np.ndarray, mu_a: np.ndarray) -> float:
    """Minimal stability constant on one stratum's mass-zero contrast space."""
    d = idx.size
    if d < 2:
        return 0.0
    h = np.linalg.qr(np.eye(d) - np.full((d, d), 1.0 / d))[0][:, :d - 1]
    basis = np.zeros((2 * S, d - 1)); basis[idx, :] = h
    g_r = basis.T @ dgp["G"] @ basis
    m_r = basis.T @ mu_a
    lam, vec = np.linalg.eigh(g_r)
    keep = lam > max(1e-12, lam.max() * 1e-10)
    resid = m_r - vec[:, keep] @ (vec[:, keep].T @ m_r)
    if np.linalg.norm(resid) > 1e-7 * max(1.0, np.linalg.norm(m_r)):
        return math.inf
    coef = vec[:, keep].T @ m_r
    return float(math.sqrt(np.sum(coef ** 2 / lam[keep])))


def enumerate_phi():
    for labels in itertools.product(range(M), repeat=S):
        yield np.array(labels)


def run_r1() -> dict:
    dgp = build_dgp("R1")
    exact, n_adm = [], 0
    for labels in enumerate_phi():
        st = phi_stats(dgp, labels)
        if st is None:
            continue
        n_adm += 1
        if st["D"] < 1e-10:
            exact.append((labels.tolist(), st["bias"]))
    max_bias_exact = max((abs(b) for _, b in exact), default=0.0)
    return dict(n_admissible=n_adm, n_exact=len(exact),
                expected_n_exact=math.factorial(M),
                max_abs_bias_at_exact=max_bias_exact,
                attainability_ok=(len(exact) == math.factorial(M)
                                  and max_bias_exact < 1e-9))


def run_r2() -> dict:
    dgp = build_dgp("R2")
    gam = dgp["gamma_true"]
    n_adm = viol_common = viol_strat = viol_wit = 0
    best = (math.inf, None, None)
    max_ratio, rng = 0.0, np.random.default_rng(SEED + 1)
    t1_checked = 0
    for labels in enumerate_phi():
        st = phi_stats(dgp, labels)
        if st is None:
            continue
        n_adm += 1
        b, d_val = abs(st["bias"]), st["D"]
        if b > gam * d_val + TOL:
            viol_common += 1
        if d_val > TOL:
            max_ratio = max(max_ratio, b / d_val)
        if d_val < best[0]:
            best = (d_val, labels.copy(), st)
        if rng.random() < 0.0015 and t1_checked < 60:      # spot checks
            t1_checked += 1
            dec = sum(p * ((1 - pi) * (dgp["mu"][1] @ (q1 - q0))
                           + pi * (dgp["mu"][0] @ (q1 - q0)))
                      for p, pi, q1, q0 in zip(st["p_z"], st["pi_z"],
                                               st["q1"], st["q0"]))
            assert abs(dec - st["bias"]) < 1e-9            # T1 identity
            bound2 = 0.0
            for p, pi, dl, idx in zip(st["p_z"], st["pi_z"], st["delta"],
                                      stratum_members(labels)):
                gs = [gamma_star(dgp, idx, dgp["mu"][a]) for a in (0, 1)]
                if any(g > gam + 1e-6 for g in gs):
                    viol_wit += 1                          # witness must cap
                bound2 += p * ((1 - pi) * gs[1] + pi * gs[0]) * dl
            if abs(st["bias"]) > bound2 + 1e-7:
                viol_strat += 1
    d_min, lab, st = best
    return dict(n_admissible=n_adm, violations_common_full=viol_common,
                violations_stratum_spotcheck=viol_strat,
                violations_witness_spotcheck=viol_wit,
                spot_checked_maps=t1_checked, inf_D=d_min,
                argmin_labels=lab.tolist(), bias_at_argmin=st["bias"],
                certificate_at_argmin=gam * d_min, gamma_true=gam,
                max_bias_over_D=max_ratio)


def run_r4() -> dict:
    rows = []
    mu_fixed = np.array([[0.3 * x / 9 + 1.0 * u for x, u in LAT],
                         [0.5 + 0.3 * x / 9 + 1.0 * u for x, u in LAT]])
    for c1 in (0.8, 0.65, 0.55, 0.51, 0.5):
        dgp = build_dgp("R2", c1=c1, c0=1 - c1)
        dgp["mu"] = mu_fixed
        dgp["tau"] = float((mu_fixed[1] - mu_fixed[0]) @ dgp["p_lat"])
        best = (math.inf, None, None)
        for labels in enumerate_phi():
            st = phi_stats(dgp, labels)
            if st is not None and st["D"] < best[0]:
                best = (st["D"], labels.copy(), st)
        d_min, lab, st = best
        gs = max(gamma_star(dgp, idx, mu_fixed[a])
                 for idx in stratum_members(lab) for a in (0, 1))
        rows.append(dict(c1=c1, min_D=d_min, bias_at_argmin=st["bias"],
                         max_gamma_star_at_argmin=(None if math.isinf(gs)
                                                   else gs),
                         gamma_star_infinite_at_argmin=math.isinf(gs)))
    return dict(sweep=rows)


def r_n_theoretical(n: int, delta: float, n_phi: int) -> float:
    e_n = ((2 + math.sqrt(2 * math.log(4 * M * n_phi / delta)))
           * math.sqrt(2 / (n * P_MIN * ETA)))
    tail = 2 * math.sqrt(2 * (M * math.log(2)
                              + math.log(4 * n_phi / delta)) / n)
    return 2 * e_n + tail


def emp_D(counts: np.ndarray, kmat: np.ndarray, lab: np.ndarray,
          n: int) -> float | None:
    """Empirical D-hat from per-arm cell counts; None if a cell is too thin."""
    total = 0.0
    for z in range(M):
        idx = np.array([2 * xx + w for xx in range(S) if lab[xx] == z
                        for w in (0, 1)], dtype=int)
        if idx.size == 0:
            return None
        c1z, c0z = counts[1, idx].sum(), counts[0, idx].sum()
        if min(c1z, c0z) < 2:
            return None
        r1 = np.zeros(2 * S); r1[idx] = counts[1, idx] / c1z
        r0 = np.zeros(2 * S); r0[idx] = counts[0, idx] / c0z
        dd = r1 - r0
        total += ((c1z + c0z) / n) * math.sqrt(max(0.0, dd @ kmat @ dd))
    return total


def run_finite_sample(n: int = 20000, reps: int = 30) -> dict:
    dgp = build_dgp("R2")
    gam, rng = dgp["gamma_true"], np.random.default_rng(SEED + 2)
    subset = [np.array(rng.integers(0, M, S)) for _ in range(400)]
    pop = {tuple(l): phi_stats(dgp, l) for l in subset}
    subset = [l for l in subset if pop[tuple(l)] is not None]
    r_thy = r_n_theoretical(n, 0.05, M ** S)
    sup_subset, covered, cert_at_sel = [], 0, []
    sup_full_rep0, skipped_rep0 = None, None
    for rep in range(reps):
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
        best, sup_dev = (math.inf, None), 0.0
        for lab in subset:
            d_hat = emp_D(counts, dgp["K"], lab, n)
            if d_hat is None:
                continue
            sup_dev = max(sup_dev, abs(d_hat - pop[tuple(lab)]["D"]))
            if d_hat < best[0]:
                best = (d_hat, lab)
        sup_subset.append(sup_dev)
        if rep == 0:            # exhaustive event check for one repetition
            sup_f, skip = 0.0, 0
            for labels in enumerate_phi():
                st = phi_stats(dgp, labels)
                if st is None:
                    continue
                d_full = emp_D(counts, dgp["K"], labels, n)
                if d_full is None:
                    skip += 1
                    continue
                sup_f = max(sup_f, abs(d_full - st["D"]))
            sup_full_rep0, skipped_rep0 = sup_f, skip
        d_hat_sel, lab_sel = best
        tau_hat = 0.0
        for z in range(M):
            in_z = np.isin(x, np.flatnonzero(lab_sel == z))
            y1 = y[(a == 1) & in_z].mean()
            y0 = y[(a == 0) & in_z].mean()
            tau_hat += (in_z.sum() / n) * (y1 - y0)
        se = 2.0 / math.sqrt(n * P_MIN * ETA)          # crude conservative SE
        half = 1.96 * se + gam * (d_hat_sel + r_thy)
        covered += int(abs(tau_hat - dgp["tau"]) <= half)
        cert_at_sel.append(gam * (d_hat_sel + r_thy))
    return dict(n=n, reps=reps, n_subset=len(subset),
                r_n_theoretical=r_thy,
                sup_dev_subset_max=float(max(sup_subset)),
                sup_dev_full_enum_rep0=float(sup_full_rep0),
                full_enum_rep0_skipped_lowcount=int(skipped_rep0),
                r_n_event_holds_subset_all_reps=bool(
                    all(s <= r_thy for s in sup_subset)),
                r_n_event_holds_full_rep0=bool(sup_full_rep0 <= r_thy),
                combined_interval_coverage=covered / reps,
                mean_certificate_at_selection=float(np.mean(cert_at_sel)))


def main() -> None:
    summary = dict(
        provenance=dict(script="code/finite_state_bridge_experiment.py",
                        plan="memo/2026-08-18-finite-state-experiment-plan.md",
                        theory="THEORY.md", date="2026-08-18", seed=SEED,
                        S=S, M=M, eta=ETA, p_min=P_MIN, n_phi=M ** S),
        R1_exact=run_r1(), R2_approximate=run_r2(),
        R4_illposedness=run_r4(), finite_sample=run_finite_sample())
    out = ROOT / "results" / "finite_state_bridge_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
