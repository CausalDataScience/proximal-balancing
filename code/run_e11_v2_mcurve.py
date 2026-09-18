"""Success probability of literal Algorithm 1 against the theorem's lower bound, as the number of sampled
splits m grows.  A descriptive extension of the confirmed E11-v2 control, for a figure; not a new gate.

One outer state (the five confirmed states were identical in N, Delta, L and rho).  Each repetition draws a
fresh evaluation sample, computes the candidate estimate of every screened split once, draws one uniform
permutation of the fourteen splits, and scores Algorithm 1 on every prefix length m.  A prefix of a uniform
permutation is a uniform m-subset, so each m has the correct marginal law; the m values share repetitions.
"""
from __future__ import annotations

import time
from fractions import Fraction as F

import numpy as np
from scipy.stats import beta as beta_dist

import e11_v2_exact_theorem as ex
import minimal_suite_common as c
import run_e11_v2 as r11

SEED = 92000000
REPS = 2000
M_VALUES = list(range(1, ex.T_V2 + 1))


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a, k + 1, n - k))
    return lo, hi


def main() -> int:
    t0 = time.time()
    eta, delta = F(1, 10), F(5, 100)
    fz = r11.FROZEN
    splits = ex.splits_v2()
    theta = {sp: ex.exact_theta(sp) for sp in splits}
    d_rows, n_rows = r11.draw(fz["n_D"], SEED), r11.draw(fz["n_N"], SEED + 1)
    fitted = {sp: ex.fit_saturated(n_rows, sp, eta) for sp in splits}
    screened = [sp for sp in splits if ex.screen_split(d_rows, sp, fitted[sp], fz["t"], eta)["retained"]]
    N = len(screened)
    n_tau = sum(1 for sp in screened if theta[sp] == ex.TAU)
    others = {}
    for sp in screened:
        if theta[sp] != ex.TAU:
            others[theta[sp]] = others.get(theta[sp], 0) + 1
    L = len(others)
    Delta = F(n_tau, N) - (F(max(others.values()), N) if others else F(0))
    rad = ex.exact_rho({sp: ex.exact_radius_terms(sp, fitted[sp], eta) for sp in screened},
                       fz["n_E"], delta, N, eta)
    rho = float(rad["rho"])
    screened_set = set(screened)
    succ = {m: 0 for m in M_VALUES}
    perm_rng, data_rng = np.random.default_rng(SEED + 70), np.random.default_rng(SEED + 90)
    for k in range(REPS):
        e_rows = r11.draw(fz["n_E"], int(data_rng.integers(1, 2 ** 31 - 1)))
        est = {sp: ex.aipw_estimate(e_rows, sp, fitted[sp]) for sp in screened}
        order = [splits[j] for j in perm_rng.permutation(len(splits))]
        for m in M_VALUES:
            kept = [sp for sp in order[:m] if sp in screened_set]
            agg = ex.link_and_return([c.split_id(s) for s in kept], np.array([est[s] for s in kept]), rho)
            succ[m] += int(agg["return_kind"] == "unique_largest" and agg["output"] is not None
                           and abs(agg["output"] - 1.0) <= rho)
        if (k + 1) % 250 == 0:
            print(f"  {k + 1}/{REPS} reps ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for m in M_VALUES:
        b = ex.theorem_bound(ex.T_V2, N, m, delta, Delta, L)
        lo, hi = clopper_pearson(succ[m], REPS)
        rows.append({"m": m, "bound": b["B"], "P_R_zero": b["P_R_zero"], "E_exp": b["E_exp"],
                     "success": succ[m] / REPS, "ci95": [lo, hi], "successes": succ[m]})
        print(f"m={m:2d}  bound {b['B']:+.4f}  empirical {succ[m] / REPS:.4f}  [{lo:.4f}, {hi:.4f}]")
    out = {"experiment": "E11-v2 m-curve (descriptive)", "seed": SEED, "reps": REPS, "T": ex.T_V2, "N": N,
           "n_tau": n_tau, "L": L, "Delta": float(Delta), "rho": rho, "rows": rows,
           "source_snapshot": c.source_snapshot(["e11_v2_exact_theorem.py", "run_e11_v2.py",
                                                  "run_e11_v2_mcurve.py", "minimal_suite_common.py"]),
           "seconds": time.time() - t0}
    c.write_json_new("../results/e11_v2_mcurve_v1.json", out)
    print("written to ../results/e11_v2_mcurve_v1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
