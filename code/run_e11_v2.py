"""E11-v2 confirmatory run: Algorithm 1 audited literally on the exact finite-state control."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

import e11_v2_exact_theorem as ex
import minimal_suite_common as c

SOURCES = ["minimal_suite_common.py", "e11_v2_exact_theorem.py", "run_e11_v2.py"]
PROTOCOL_PATH = "../results/e11_v2_exact_protocol_v2.json"
OUT_PATH = "../results/e11_v2_exact_confirm_v2.json"
FROZEN = {
    "protocol_id": "e11-v2-exact-finite-state-audit-v2",
    "experiment_id": "E11-v2", "mode": "literal_one_shot", "phase": "confirmation",
    "n_D": 20000, "n_N": 20000, "n_E": 20000,
    "t": 0.02, "eta": "1/10", "delta": "5/100", "m": 8, "T": ex.T_V2,
    "outer_seeds": [92000000 + 1000 * r for r in range(5)],
    "inner_reps": 5000,
    "success_event": "a unique output with |tau_hat - 1| <= rho; candidate sets, ties and no-return fail",
    "equality_rule": "candidate values are equal when their encoder map ids are equal, never by tolerance",
    "radius_rule": "exact rational sigma and nuisance errors over the eight states; the variance term is "
                   "one upward-rounded root of sigma^2 N / (n_E delta) and each root is verified to be an "
                   "upper bound before it is used",
    "supersedes": "e11-v2-exact-finite-state-audit-v1",
    "supersede_reason": "v1 rounded the numerator and the denominator of the variance term upward "
                        "separately, which can shrink the quotient, and _sqrt_up floored the scaled value "
                        "before taking the integer root, which fails when that floor is a perfect square. "
                        "Neither is one sided, so v1 could not certify the radius. Both are fixed here; "
                        "the radius is unchanged to twelve decimals.",
    "simultaneous_level": "one-sided Clopper-Pearson at 1 - 0.05/5 = 0.99 across the five states",
    "gates": {"Delta_positive": True, "separation_4rho_lt_g": True, "B_min": 0.80,
              "uniform_radius_lower": 0.95, "success_lower_at_least_B": True},
}


def draw(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 2, n)
    i = rng.integers(0, 2, n)
    p = 0.25 + 0.5 * u
    a = (rng.random(n) < p).astype(int)
    return {"U": u, "I": i, "A": a, "Y": a - 2.0 * u}


def lower(successes: int, n: int, level: float) -> float:
    if successes == 0:
        return 0.0
    if successes == n:
        return float((1.0 - level) ** (1.0 / n))
    return float(beta_dist.ppf(1.0 - level, successes, n - successes + 1))


def freeze_protocol() -> dict:
    p = Path(PROTOCOL_PATH)
    if p.exists():
        return json.loads(p.read_text())
    proto = dict(FROZEN)
    proto["design_table"] = [{**r, "theta_S": str(r["theta_S"]), "population_gap": str(r["population_gap"])}
                             for r in ex.design_table()]
    proto["source_snapshot"] = c.source_snapshot(SOURCES)
    proto["frozen_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    proto["runs"] = [{"experiment_id": "E11-v2", "mode": "literal_one_shot", "phase": "confirmation",
                      "outer_seeds": FROZEN["outer_seeds"], "inner_reps": FROZEN["inner_reps"]}]
    return c.seal_protocol(p, proto)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner-reps", type=int, default=FROZEN["inner_reps"])
    args = ap.parse_args()
    proto = freeze_protocol()
    c.check_protocol(proto, {"experiment_id": "E11-v2", "mode": "literal_one_shot",
                             "phase": "confirmation", "outer_seeds": FROZEN["outer_seeds"],
                             "inner_reps": args.inner_reps}, PROTOCOL_PATH, SOURCES)
    eta, delta = F(1, 10), F(5, 100)
    root = c.result_root("E11-v2", "literal_one_shot", "confirmation", PROTOCOL_PATH, SOURCES)
    splits = ex.splits_v2()
    theta = {sp: ex.exact_theta(sp) for sp in splits}
    maps = {sp: ex.encoder_map(sp) for sp in splits}
    print(f"E11-v2 | T={ex.T_V2} m={FROZEN['m']} delta=0.05 n={FROZEN['n_D']} | "
          f"{len(FROZEN['outer_seeds'])} outer states x {args.inner_reps} inner repetitions")

    for seed in FROZEN["outer_seeds"]:
        d_rows, n_rows = draw(FROZEN["n_D"], seed), draw(FROZEN["n_N"], seed + 1)
        fitted = {sp: ex.fit_saturated(n_rows, sp, eta) for sp in splits}
        # the theorem-audit branch fits and screens all fourteen; the algorithm below never sees this
        audit_screen = {sp: ex.screen_split(d_rows, sp, fitted[sp], FROZEN["t"], eta) for sp in splits}
        screened = [sp for sp in splits if audit_screen[sp]["retained"]]
        N = len(screened)
        by_map: dict[str, int] = {}
        for sp in screened:
            by_map[maps[sp]] = by_map.get(maps[sp], 0) + 1
        n_tau = sum(1 for sp in screened if theta[sp] == ex.TAU)
        other_counts: dict[str, int] = {}
        for sp in screened:
            if theta[sp] != ex.TAU:
                other_counts[str(theta[sp])] = other_counts.get(str(theta[sp]), 0) + 1
        L = len(other_counts)
        Delta = F(n_tau, N) - (F(max(other_counts.values()), N) if other_counts else F(0))
        terms = {sp: ex.exact_radius_terms(sp, fitted[sp], eta) for sp in screened}
        rad = ex.exact_rho(terms, FROZEN["n_E"], delta, N, eta)
        rho = float(rad["rho"])
        distinct = sorted({theta[sp] for sp in screened})
        g = float(min((abs(a - b) for i, a in enumerate(distinct) for b in distinct[i + 1:]), default=F(0)))
        bound = ex.theorem_bound(ex.T_V2, N, FROZEN["m"], delta, Delta, L)

        draw_rng = np.random.default_rng(seed + 7)
        data_rng = np.random.default_rng(seed + 9)
        ok_radius = np.zeros(args.inner_reps, dtype=bool)
        success = np.zeros(args.inner_reps, dtype=bool)
        kinds: dict[str, int] = {}
        r_counts = np.zeros(args.inner_reps, dtype=int)
        for k in range(args.inner_reps):
            drawn = [splits[j] for j in draw_rng.choice(len(splits), size=FROZEN["m"], replace=False)]
            e_rows = draw(FROZEN["n_E"], int(data_rng.integers(1, 2 ** 31 - 1)))
            est_all = {sp: ex.aipw_estimate(e_rows, sp, fitted[sp]) for sp in screened}
            ok_radius[k] = all(abs(est_all[sp] - float(theta[sp])) <= rho for sp in screened)
            retained = [sp for sp in drawn if sp in set(screened)]
            r_counts[k] = len(retained)
            agg = ex.link_and_return([c.split_id(s) for s in retained],
                                     np.array([est_all[s] for s in retained]), rho)
            kinds[agg["return_kind"]] = kinds.get(agg["return_kind"], 0) + 1
            success[k] = bool(agg["return_kind"] == "unique_largest" and agg["output"] is not None
                              and abs(agg["output"] - 1.0) <= rho)
        n = args.inner_reps
        cell = {"data_seed": seed, "N": N, "n_tau": n_tau, "L": L, "Delta": float(Delta),
                "Delta_exact": str(Delta), "rho": rho, "rho_exact": str(rad["rho"]),
                "radius_argmax": c.split_id(rad["argmax"]), "separation_g": g,
                "plurality_ok": bool(Delta > 0), "separation_ok": bool(4.0 * rho < g),
                "theorem_B": bound["B"], "P_R_zero": bound["P_R_zero"], "E_exp": bound["E_exp"],
                "screened": [c.split_id(s) for s in screened],
                "screened_by_map": by_map, "other_value_counts": other_counts,
                "empirical_success": float(success.mean()),
                "success_lower_99": lower(int(success.sum()), n, 0.99),
                "uniform_radius_rate": float(ok_radius.mean()),
                "uniform_radius_lower_99": lower(int(ok_radius.sum()), n, 0.99),
                "return_kinds": kinds, "R_mean": float(r_counts.mean()),
                "R_counts": np.bincount(r_counts, minlength=FROZEN["m"] + 1).tolist(),
                "fold_hashes": {"D": c.array_sha256(d_rows["A"]), "N": c.array_sha256(n_rows["A"])},
                "complete": True}
        root["cells"].append(cell)
        print(f"  seed {seed}: N={N} n(tau)={n_tau} L={L} Delta={float(Delta):.4f} rho={rho:.6f} "
              f"g={g:.4f} 4rho<g={cell['separation_ok']} B={bound['B']:.6f} "
              f"success={cell['empirical_success']:.4f} (lower99 {cell['success_lower_99']:.4f}) "
              f"radius={cell['uniform_radius_rate']:.4f}")

    cells = root["cells"]
    checks = {
        "delta_positive": all(q["plurality_ok"] for q in cells),
        "separation": all(q["separation_ok"] for q in cells),
        "bound_at_least_0.80": all(q["theorem_B"] >= 0.80 for q in cells),
        "uniform_radius_lower_at_least_0.95": all(q["uniform_radius_lower_99"] >= 0.95 for q in cells),
        "success_lower_at_least_B": all(q["success_lower_99"] >= q["theorem_B"] for q in cells),
        "failures_counted": all(set(q["return_kinds"]) <= {"unique_largest", "tied_largest", "no_return"}
                                for q in cells),
        "all_states_reported": len(cells) == len(FROZEN["outer_seeds"]),
    }
    root["summary"] = {"checks": checks, "states": len(cells)}
    root["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    root["complete"] = True
    c.write_json_new(OUT_PATH, root)
    print("\nchecks: " + ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items()))
    print(f"verdict: {root['verdict']}  written to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
