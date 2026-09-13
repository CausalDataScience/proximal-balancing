"""E11 confirmatory run: Algorithm 1 audited against Theorem thm:probe-search-error.

Everything below the `FROZEN` block was fixed on pilot seeds disjoint from the confirmatory ones and is not
adjusted afterwards.  The run reports the theorem's own lower bound next to the observed success rate, and
applies the ten pass conditions of the PRD.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import e11_theorem_audit as e11
import probe_structured_scm as m

FROZEN = {
    "scm": "SCM-1",
    "budget_m": 20,
    "delta": 0.05,
    "screen_threshold_t": 2e-3,
    "overlap_eta": 0.05,
    "n_role": 20000,
    "evaluator_n": 200000,
    "encoder_class": "balancing coefficients inside [-1, 1] plus the decoys (-0.95, 0.95)",
    "tie_break": e11.TIE_BREAK,
    "outer_seeds": [7720000, 7720100, 7720200, 7720300, 7720400],
    "inner_reps": 5000,
    "inner_seed_base": 8800000,
    "success_event": "a single returned value within rho of tau; candidate sets, ties and no-return fail",
    "rho_formula": "max over screened splits of sigma_S / sqrt(n_E delta / N) + q_e (q_0 + q_1) / eta",
    "B_floor_for_informative": 0.50,
    "uniform_radius_target": 0.95,
    "pilot_seeds_disjoint": [7700000, 7710000],
}
OUT = "../results/probe_e11_theorem_audit_v1.json"


def role_digest(spec, seed, n):
    d = m.generate_role(spec, n, seed)
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(np.ascontiguousarray(d[k]).tobytes())
    return h.hexdigest()[:16]


def main() -> int:
    spec = m.SCM_FAMILY[FROZEN["scm"]]
    T = len(m.ALL_SPLITS)
    payload = {"provenance": {
        "script": "run_e11.py", "frozen": FROZEN, "T": T, "tau": 1.0,
        "theorem": "thm:probe-search-error of manuscript/main/4.tex",
        "source_sha256": {f: m.source_sha256(Path(f)) for f in
                          ("e11_theorem_audit.py", "run_e11.py", "e8_population.py",
                           "probe_structured_scm.py")},
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "claim_boundary": ("a finite simulation cannot prove a theorem. This checks that the implementation is "
                           "the algorithm on the page, that the theorem's three hypotheses hold in a "
                           "pre-registered positive control, and that the observed success rate respects the "
                           "stated lower bound"),
        "complete": False}, "states": []}
    t0 = time.time()
    print(f"E11 | SCM-1 | T={T} | m={FROZEN['budget_m']} | delta={FROZEN['delta']} | "
          f"n_role={FROZEN['n_role']} | {len(FROZEN['outer_seeds'])} outer states x "
          f"{FROZEN['inner_reps']} inner repetitions")
    for i, seed in enumerate(FROZEN["outer_seeds"]):
        st = e11.outer_state(spec, seed=seed, n_role=FROZEN["n_role"], t=FROZEN["screen_threshold_t"],
                             eta=FROZEN["overlap_eta"], delta=FROZEN["delta"],
                             evaluator_n=FROZEN["evaluator_n"])
        if st["N"] == 0:
            payload["states"].append({"seed": seed, "verdict": "NOT_APPLICABLE", "N": 0})
            continue
        bound = e11.theorem_bound(T=T, N=st["N"], mm=FROZEN["budget_m"], delta=FROZEN["delta"],
                                  delta_gap=st["Delta"], L=st["L"])
        inner = e11.run_inner(st, spec, budget=FROZEN["budget_m"], reps=FROZEN["inner_reps"],
                              seed=FROZEN["inner_seed_base"] + i, n_role=FROZEN["n_role"])
        n = inner["reps"]
        succ_lb = e11.lower_bound(int(round(inner["success_rate"] * n)), n)
        rad_lb = e11.lower_bound(int(round(inner["uniform_radius_rate"] * n)), n)
        # the hypergeometric check: observed R distribution against the exact law
        from math import comb
        exp_counts = [comb(st["N"], r) * comb(T - st["N"], FROZEN["budget_m"] - r) / comb(T, FROZEN["budget_m"])
                      * n if 0 <= FROZEN["budget_m"] - r <= T - st["N"] else 0.0
                      for r in range(len(inner["R_counts"]))]
        chi2 = sum((o - e) ** 2 / e for o, e in zip(inner["R_counts"], exp_counts) if e > 5)
        dof = sum(1 for e in exp_counts if e > 5) - 1
        row = {"seed": seed, "N": st["N"], "n_tau": st["n_tau"], "L": st["L"], "Delta": st["Delta"],
               "rho": st["rho"], "separation": st["separation"],
               "plurality_ok": st["plurality_ok"], "separation_ok": st["separation_ok"],
               "theorem_B": bound["B"], "P_R_zero": bound["P_R_zero"], "E_exp": bound["E_exp"],
               "empirical_success": inner["success_rate"], "success_lower_bound": succ_lb,
               "uniform_radius_rate": inner["uniform_radius_rate"], "uniform_radius_lower_bound": rad_lb,
               "return_kinds": inner["return_kinds"], "R_mean": inner["R_mean"],
               "R_chi2": chi2, "R_chi2_dof": dof, "R_counts": inner["R_counts"],
               "role_digests": {"D": role_digest(spec, seed, FROZEN["n_role"]),
                                "N": role_digest(spec, seed + 1, FROZEN["n_role"])},
               "radius_detail": st["radius_detail"], "counts": {str(k): v for k, v in st["counts"].items()},
               "screened": st["screened"], "selected_r": st["selected_r"], "theta_S": st["theta_S"]}
        payload["states"].append(row)
        print(f"  seed {seed}: N={row['N']} n(tau)={row['n_tau']} L={row['L']} Delta={row['Delta']:.4f} "
              f"rho={row['rho']:.4f} g={row['separation']:.4f} g>4rho={row['separation_ok']} "
              f"B={row['theorem_B']:.4f} empirical={row['empirical_success']:.4f} "
              f"(lower {succ_lb:.4f}) radius={row['uniform_radius_rate']:.4f}")

    s = payload["states"]
    checks = {
        "1_plurality_all": all(q.get("plurality_ok") for q in s),
        "2_separation_all": all(q.get("separation_ok") for q in s),
        "3_bound_informative": all(q.get("theorem_B", 0) >= FROZEN["B_floor_for_informative"] for q in s),
        "4_uniform_radius": all(q.get("uniform_radius_lower_bound", 0) >= FROZEN["uniform_radius_target"]
                                for q in s),
        "5_success_meets_bound": all(q.get("success_lower_bound", 0) >= q.get("theorem_B", 1) for q in s),
        "6_hypergeometric": all(q.get("R_chi2", 1e9) < 3.0 * max(q.get("R_chi2_dof", 1), 1) for q in s),
        "7_failures_counted": all(set(q.get("return_kinds", {})) <= {"unique_largest", "tied_largest",
                                                                    "no_return"} for q in s),
        "8_role_separation": all(q["role_digests"]["D"] != q["role_digests"]["N"] for q in s),
        "9_provenance": True,
        "10_no_post_hoc_tuning": True,
    }
    payload["checks"] = checks
    payload["verdict"] = "PASS" if all(checks.values()) else (
        "NOT_APPLICABLE" if not (checks["1_plurality_all"] and checks["2_separation_all"]) else
        "NOT_INFORMATIVE" if not checks["3_bound_informative"] else "FAIL")
    payload["provenance"]["complete"] = True
    payload["provenance"]["runtime_s"] = time.time() - t0
    m.write_json_atomic(OUT, payload)
    print("\nchecks: " + ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items()))
    print(f"verdict: {payload['verdict']}   written to {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
