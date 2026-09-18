"""Population preflight for the PROBE positive-control audit (PRD v4, section 3).

Solves the two free coefficients so that a designated library element is an exact
balance point for the target family and for the wrong family, enumerates all 2^J - 2
splits against the whole library, and checks the three facts the PRD asks for:

    1. the target class is the largest class among population candidates,
    2. at least one non-target class exists,
    3. an evaluation sample size exists at which 4 rho < g_min.

Balance is decided algebraically by B_S gamma(beta) = 0, not by numerical integration.
No seed is drawn for the confirmation here; the only randomness is a labelled Monte
Carlo estimate of the AIPW score standard deviations.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import positive_control_scm as pc
from positive_control_scm import Design

RESULTS = Path(__file__).resolve().parents[1] / "results" / "positive_control"
OUTPUT = RESULTS / "preflight_v1.json"

J = 7
B_C = (1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
B_T_FIXED = -0.6                         # block 3, the target family's balance channel
B_T_PURE = 1.0                           # blocks 5 and 6
SIGMA = (0.85,) * J
KAPPA, ALPHA_C, OUTCOME_U, OUTCOME_NOISE = 2.0, 1.0, 2.5, 0.3
TAU = 1.0
DELTA = 0.05
SIGMA_MC_N = 1_000_000
SIGMA_MC_SEED = 8_100_000                # labelled, never reused by a confirmation
PERMUTATION_SEED = 4242


def beta_on(weights: dict[int, float], split: tuple[int, ...]) -> np.ndarray:
    """The library element with the given signed weights on the named blocks."""
    kept = pc.kept_blocks(J, split)
    v = np.array([weights.get(j, 0.0) for j in kept], dtype=float)
    return v / np.linalg.norm(v)


def build(alpha_t: float, b_4t: float) -> Design:
    """The design with its two solved coefficients in place.

    Block 3 carries the target family's balance channel and block 4 the wrong family's,
    so the two solves are independent: alpha_t moves only the first, b_4t only the second.
    """
    return Design(b_c=B_C, b_t=(0.0, 0.0, 0.0, B_T_FIXED, b_4t, B_T_PURE, B_T_PURE),
                  sigma=SIGMA, kappa=KAPPA, alpha_c=ALPHA_C, alpha_t=alpha_t,
                  outcome_u=OUTCOME_U, outcome_noise=OUTCOME_NOISE)


def solve_alpha_t() -> float:
    """gamma_c is affine in alpha_t, so the target family solves in closed form."""
    split, beta = (0,), beta_on({3: 1.0}, (0,))
    g0 = pc.gamma(build(0.0, -1.0), split, beta)[0]
    g1 = pc.gamma(build(1.0, -1.0), split, beta)[0]
    return float(-g0 / (g1 - g0))


def solve_b_4t(alpha_t: float) -> tuple[float, tuple[float, float]]:
    """gamma_t at the designated wrong-family element, by bracketed root finding."""
    split = (5,)
    beta = beta_on({4: 1.0}, split)
    f = lambda b: pc.gamma(build(alpha_t, b), split, beta)[1]
    lo, hi = -12.0, -0.05
    if f(lo) * f(hi) >= 0:
        raise SystemExit("no sign change for the wrong-family root; adjust the design")
    return float(brentq(f, lo, hi, xtol=1e-14, rtol=8.9e-16)), (lo, hi)


def census(design: Design) -> list[dict]:
    rows = []
    for split in pc.all_splits(J):
        lib = pc.library(J - len(split))
        residuals = np.array([pc.balance_residual(design, split, b) for b in lib])
        balanced = [i for i, r in enumerate(residuals) if r <= pc.BALANCE_TOL]
        thetas, theta_res = [], []
        for i in balanced:
            value, residual = pc.theta_with_residual(design, split, lib[i])
            thetas.append(value)
            theta_res.append(residual)
        rows.append({
            "split": list(split),
            "size": len(split),
            "orc_valid": pc.orc_valid(design, split),
            "min_balance_residual": float(residuals.min()),
            "balanced_count": len(balanced),
            "balanced_beta": [lib[i].tolist() for i in balanced],
            "theta": thetas,
            "theta_residual": theta_res,
            "theta_spread": float(max(thetas) - min(thetas)) if len(thetas) > 1 else 0.0,
        })
    return rows


def score_sd(design: Design, split: tuple[int, ...], beta: np.ndarray) -> tuple[float, float, float]:
    data = pc.generate(design, SIGMA_MC_N, SIGMA_MC_SEED)
    scores = pc.aipw_scores(design, split, beta, data)
    sd = float(scores.std(ddof=1))
    mean = float(scores.mean())
    return sd, mean, float(sd / math.sqrt(SIGMA_MC_N))


def main() -> int:
    alpha_t = solve_alpha_t()
    b_4t, bracket = solve_b_4t(alpha_t)
    design = build(alpha_t, b_4t)
    print(f"solved alpha_t = {alpha_t:.10f}   b_4t = {b_4t:.10f}  (bracket {bracket})")

    rows = census(design)
    candidates = [r for r in rows if r["balanced_count"] > 0]
    ambiguous = [r for r in candidates if r["theta_spread"] > 1e-8]
    target = [r for r in candidates if r["orc_valid"]]
    wrong = [r for r in candidates if not r["orc_valid"]]

    # classes of wrong values, exact agreement up to the quadrature residual
    classes: list[dict] = []
    for r in wrong:
        value = r["theta"][0]
        hit = next((c for c in classes if abs(c["value"] - value) <= 1e-8), None)
        if hit is None:
            classes.append({"value": value, "splits": [r["split"]]})
        else:
            hit["splits"].append(r["split"])
    n_cand = len(candidates)
    p_tau = len(target) / n_cand if n_cand else 0.0
    p_star = max((len(c["splits"]) / n_cand for c in classes), default=0.0)
    values = [TAU] + [c["value"] for c in classes]
    g_min = min(abs(a - b) for i, a in enumerate(values) for b in values[i + 1:])

    sds = {}
    for r in candidates:
        split, beta = tuple(r["split"]), np.array(r["balanced_beta"][0])
        sd, mean, mc = score_sd(design, split, beta)
        sds[str(split)] = {"sd": sd, "score_mean": mean, "mc_error": mc, "theta": r["theta"][0]}
    sigma_max = max(v["sd"] for v in sds.values())
    rho = lambda n_e: sigma_max * math.sqrt(n_cand / (n_e * DELTA))
    table = [{"n_E": n, "rho": rho(n), "four_rho": 4 * rho(n)} for n in
             (10_000, 25_000, 50_000, 100_000, 200_000, 400_000, 800_000)]
    need = math.ceil((4 * sigma_max) ** 2 * n_cand / (DELTA * g_min ** 2))

    # permutation test: relabelling the blocks permutes the census and nothing else
    order = tuple(np.random.default_rng(PERMUTATION_SEED).permutation(J).tolist())
    permuted_rows = census(design.permuted(order))
    index = {tuple(r["split"]): r for r in permuted_rows}
    inverse = {order[i]: i for i in range(J)}       # block i of the permuted design is original order[i]
    diffs = []
    for r in rows:
        image = tuple(sorted(inverse[j] for j in r["split"]))
        other = index[image]
        diffs.append(abs(r["min_balance_residual"] - other["min_balance_residual"]))
        diffs.append(abs(r["balanced_count"] - other["balanced_count"]))
        if r["theta"] and other["theta"]:
            diffs.append(abs(sorted(r["theta"])[0] - sorted(other["theta"])[0]))
    permutation_ok = max(diffs) <= 1e-9

    facts = {
        "plurality": p_tau - p_star > 0,
        "non_target_exists": len(classes) >= 1,
        "separation_possible": g_min > 0 and need > 0,
        "no_ambiguous_tie": not ambiguous,
        "permutation": bool(permutation_ok),
    }
    payload = {
        "complete": len(rows) == 2 ** J - 2,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prd": "memo/2026-09-16-population-census-prd-v4.md",
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("positive_control_scm.py", "run_positive_control_preflight.py",
                                       "test_positive_control.py")},
        "design": {"J": J, "b_c": list(B_C), "b_t": list(design.b_t), "sigma": list(SIGMA),
                   "kappa": KAPPA, "alpha_c": ALPHA_C, "alpha_t": alpha_t,
                   "outcome_u": OUTCOME_U, "outcome_noise": OUTCOME_NOISE, "tau": TAU},
        "solved": {"alpha_t": alpha_t, "b_4t": b_4t, "bracket": list(bracket),
                   "designated_target_beta": "block 3 alone",
                   "designated_wrong_beta": "block 4 alone"},
        "library": {"alphabet": [-1, 0, 1], "max_support": pc.MAX_SUPPORT, "pairs": sum(len(pc.library(J - len(s))) for s in pc.all_splits(J))},
        "balance_tol": pc.BALANCE_TOL, "quadrature_orders": list(pc.GH_ORDERS),
        "rows": rows,
        "candidates": {"count": n_cand,
                       "target_splits": [r["split"] for r in target],
                       "wrong_classes": classes,
                       "ambiguous_tie_splits": [r["split"] for r in ambiguous]},
        "plurality": {"p_tau": p_tau, "p_star": p_star, "delta": p_tau - p_star,
                      "L_wrong_values": len(classes)},
        "separation": {"g_min": g_min, "values": values},
        "score_sd": {"monte_carlo_n": SIGMA_MC_N, "seed": SIGMA_MC_SEED, "per_candidate": sds,
                     "sigma_max": sigma_max},
        "rho_table": table, "min_n_E_for_4rho_below_gmin": need, "delta_level": DELTA,
        "permutation_test": {"order": list(order), "max_abs_diff": float(max(diffs)), "pass": permutation_ok},
        "facts": facts,
        "verdict": "PASS" if all(facts.values()) else "FAIL",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"\ncandidates {n_cand}  |  target {len(target)}  |  wrong {len(wrong)} in {len(classes)} class(es)")
    print(f"{'split':>16s} {'ORC':>5s} {'balanced':>9s} {'theta':>10s}")
    for r in candidates:
        print(f"{str(tuple(r['split'])):>16s} {str(r['orc_valid']):>5s} {r['balanced_count']:9d} {r['theta'][0]:10.5f}")
    print(f"\np_tau {p_tau:.4f}  p_star {p_star:.4f}  Delta {p_tau - p_star:.4f}  L {len(classes)}")
    print(f"g_min {g_min:.5f}   sigma_max {sigma_max:.4f}")
    print(f"{'n_E':>10s} {'rho':>10s} {'4 rho':>10s}")
    for e in table:
        print(f"{e['n_E']:10d} {e['rho']:10.5f} {e['four_rho']:10.5f}")
    print(f"smallest n_E with 4 rho < g_min: {need:,}")
    print(f"permutation test max diff {max(diffs):.2e} -> {'pass' if permutation_ok else 'FAIL'}")
    print(f"\nverdict {payload['verdict']}   facts {facts}")
    print(f"written to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
