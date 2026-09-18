"""Literal Algorithm 1 confirmation for the positive-control audit (PRD v4, section 4).

The manuscript theorem is conditional on the learning sample, so the run is organised as
independent learning states.  Inside one state the Brier screen runs once, the retained
set and the selected representations are fixed, and only the evaluation sample is redrawn.

Every quantity of the theorem is recomputed on the realised retained set: N, Delta, L,
g_min and rho.  Nothing is taken from the population census except the labels used to
report what each retained split is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import positive_control_scm as pc
from positive_control_scm import Design

RESULTS = Path(__file__).resolve().parents[1] / "results" / "positive_control"
PROTOCOL = RESULTS / "confirmation_protocol_v1.json"
OUTPUT = RESULTS / "confirmation_v1.json"
TAU = 1.0
VALUE_TOL = 1e-8
SIGMA_MC_N = 1_000_000
SIGMA_MC_SEED = 8_300_000


def load_protocol() -> tuple[Design, dict]:
    protocol = json.loads(PROTOCOL.read_text())
    here = Path(__file__).parent
    for name, expected in protocol["source_sha256"].items():
        actual = hashlib.sha256((here / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"source hash differs from the sealed protocol: {name}")
    d = protocol["design"]
    design = Design(b_c=tuple(d["b_c"]), b_t=tuple(d["b_t"]), sigma=tuple(d["sigma"]),
                    kappa=d["kappa"], alpha_c=d["alpha_c"], alpha_t=d["alpha_t"],
                    outcome_u=d["outcome_u"], outcome_noise=d["outcome_noise"])
    if pc.MAX_SUPPORT != protocol["representation_class"]["max_support"]:
        raise SystemExit("representation class differs from the sealed protocol")
    return design, protocol


def score_sd(design: Design, split: tuple[int, ...], beta: np.ndarray, index: int) -> float:
    data = pc.generate(design, SIGMA_MC_N, SIGMA_MC_SEED + index)
    return float(pc.aipw_scores(design, split, beta, data).std(ddof=1))


def classify(values: list[float]) -> tuple[float, float, int, float]:
    """Plurality share of tau, the largest wrong share, L, and the minimum gap."""
    n = len(values)
    target = sum(1 for v in values if abs(v - TAU) <= VALUE_TOL)
    wrong: dict[float, int] = {}
    for v in values:
        if abs(v - TAU) > VALUE_TOL:
            key = next((k for k in wrong if abs(k - v) <= VALUE_TOL), None)
            wrong[v if key is None else key] = wrong.get(key, 0) + 1 if key is not None else 1
    distinct = ([TAU] if target else []) + list(wrong)
    gaps = [abs(a - b) for i, a in enumerate(distinct) for b in distinct[i + 1:]]
    return (target / n, max((c / n for c in wrong.values()), default=0.0), len(wrong),
            min(gaps) if gaps else float("inf"))


def run_state(design: Design, protocol: dict, state: int, draws: int, reference: dict) -> dict:
    n_d = protocol["screen"]["n_D"]
    threshold = protocol["screen"]["threshold_t"]
    n_e = protocol["evaluation"]["n_E"]
    delta = protocol["evaluation"]["delta"]
    d_seed = 5_300_000 + 10_000 * state

    learning = pc.generate(design, n_d, d_seed)
    retained, fits = [], []
    for split in pc.all_splits(design.J):
        out = pc.select_and_screen(design, split, learning, threshold)
        fits.append({"split": list(split), "objective": out["objective"],
                     "retained": out["retained"], "beta": out["beta"],
                     "selection_margin": out["selection_margin"]})
        if out["retained"]:
            retained.append((split, np.array(out["beta"])))

    values = [pc.theta(design, split, beta) for split, beta in retained]
    p_tau, p_star, n_wrong, g_min = classify(values)
    n_retained = len(retained)
    sds = [score_sd(design, split, beta, i) for i, (split, beta) in enumerate(retained)]
    rho = max(sds) * math.sqrt(n_retained / (n_e * delta)) if n_retained else float("inf")

    unexpected = [list(s) for s, _ in retained if list(s) not in reference["splits"]]
    wrong_beta = [list(s) for s, b in retained
                  if list(s) in reference["splits"]
                  and not np.allclose(b, reference["beta"][str(list(s))], atol=1e-9)]
    conditions = {
        "plurality": p_tau - p_star > 0,
        "separation": g_min > 4.0 * rho,
        "retained_set_matches_population": not unexpected and not wrong_beta,
    }

    successes, radius_ok, kinds = 0, 0, {"unique_largest": 0, "tied_largest": 0, "no_return": 0}
    errors = []
    for draw in range(draws):
        e_seed = 5_400_000 + 10_000 * state + draw
        evaluation = pc.generate(design, n_e, e_seed)
        estimates = np.array([pc.aipw_scores(design, split, beta, evaluation).mean()
                              for split, beta in retained])
        radius_ok += int(np.all(np.abs(estimates - np.array(values)) <= rho))
        result = pc.aggregate([s for s, _ in retained], estimates, rho)
        kinds[result["return_kind"]] += 1
        if result["output"] is not None:
            errors.append(float(result["output"] - TAU))
            successes += int(abs(result["output"] - TAU) <= rho)
        else:
            errors.append(float("nan"))

    bound = 1.0 - delta - n_wrong * math.exp(-n_retained * (p_tau - p_star) ** 2 / 2.0)
    finite = [e for e in errors if not math.isnan(e)]
    return {
        "state": state, "learning_seed": d_seed,
        "retained": [{"split": list(s), "beta": b.tolist(), "theta": v, "score_sd": sd}
                     for (s, b), v, sd in zip(retained, values, sds)],
        "N": n_retained, "p_tau": p_tau, "p_star": p_star, "Delta": p_tau - p_star,
        "L": n_wrong, "g_min": g_min, "rho": rho, "four_rho": 4.0 * rho,
        "unexpected_splits": unexpected, "wrong_beta_splits": wrong_beta,
        "conditions": conditions,
        "theorem_condition_verdict": "PASS" if all(conditions.values()) else "FAIL",
        "draws": draws, "return_kinds": kinds,
        "uniform_radius_rate": radius_ok / draws,
        "success_rate": successes / draws,
        "manuscript_bound": bound,
        "bound_respected": successes / draws >= bound,
        "mean_error": float(np.mean(finite)) if finite else None,
        "max_abs_error": float(np.max(np.abs(finite))) if finite else None,
        "screen": {"objective_retained_max": max((f["objective"] for f in fits if f["retained"]), default=None),
                   "objective_rejected_min": min((f["objective"] for f in fits if not f["retained"]), default=None)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=5)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    design, protocol = load_protocol()
    preflight = json.loads((RESULTS / "preflight_v1.json").read_text())
    reference = {"splits": [r["split"] for r in preflight["rows"] if r["balanced_count"]],
                 "beta": {str(r["split"]): r["balanced_beta"][0] for r in preflight["rows"]
                          if r["balanced_count"]}}

    states = [run_state(design, protocol, s, args.draws, reference) for s in range(args.states)]
    for s in states:
        print(f"\nstate {s['state']}  seed {s['learning_seed']}  "
              f"N {s['N']}  Delta {s['Delta']:.4f}  L {s['L']}  g_min {s['g_min']:.5f}  "
              f"rho {s['rho']:.5f}  4rho {s['four_rho']:.5f}")
        print(f"   conditions {s['conditions']} -> {s['theorem_condition_verdict']}")
        print(f"   returns {s['return_kinds']}  uniform radius {s['uniform_radius_rate']:.3f}  "
              f"success {s['success_rate']:.3f}  bound {s['manuscript_bound']:.4f}  "
              f"respected {s['bound_respected']}")
        if s["unexpected_splits"] or s["wrong_beta_splits"]:
            print(f"   unexpected splits {s['unexpected_splits']}  wrong beta {s['wrong_beta_splits']}")

    payload = {"complete": True,
               "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
               "protocol": {"path": str(PROTOCOL), "protocol_id": protocol["protocol_id"],
                            "sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()},
               "source_sha256": protocol["source_sha256"],
               "states": states,
               "summary": {
                   "states_passing_conditions": sum(1 for s in states if s["theorem_condition_verdict"] == "PASS"),
                   "states": len(states),
                   "success_rate_min": min(s["success_rate"] for s in states),
                   "bound_min": min(s["manuscript_bound"] for s in states),
                   "all_bounds_respected": all(s["bound_respected"] for s in states)}}
    Path(args.output).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
