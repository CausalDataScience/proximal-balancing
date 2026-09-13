"""Reproduce the frozen G1--G5 Gaussian population benchmarks.

This checker is deliberately independent of the finite-sample experiment runner.
It writes only numerical population functionals; it does not certify a learned
representation, Algorithm 1, or the manuscript's uniform finite-sample bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, roots_hermitenorm


SQ2PI = np.sqrt(2.0 * np.pi)
ORDERS = (40, 80, 160)
ABS_TOL_CANONICAL = 5e-7


SPECS = {
    "g1_low_noise": {"sigma": [0.85] * 5, "c": 2.5, "h": 0.0, "population_like": None},
    "g2_heteroscedastic_noise": {"sigma": [0.85] * 5, "c": 2.5, "h": 0.0, "population_like": "g1_low_noise"},
    "g3_heterogeneous_low_noise": {"sigma": [0.85] * 5, "c": 2.7, "h": 0.3, "population_like": None},
    "g4_proxy_heteroscedastic_low_noise": {"sigma": [0.8, 0.9, 1.0, 0.85, 0.95], "c": 2.5, "h": 0.0, "population_like": None},
    "g5_sinh_low_noise": {"sigma": [0.85] * 5, "c": 2.5, "h": 0.0, "population_like": "g1_low_noise"},
}

EXPECTED = {
    "g1_low_noise": {"naive": -0.174151070977, "x": 0.048012505814, "raw": 0.825562138072, "bad_r": -0.650170889312, "bad_theta": -0.443197874524},
    "g2_heteroscedastic_noise": {"naive": -0.174151070977, "x": 0.048012505814, "raw": 0.825562138072, "bad_r": -0.650170889312, "bad_theta": -0.443197874524},
    "g3_heterogeneous_low_noise": {"naive": -0.199768912525, "x": 0.030301110573, "raw": 0.822073380833, "bad_r": -0.650170889312, "bad_theta": -0.472061832015},
    "g4_proxy_heteroscedastic_low_noise": {"naive": -0.174151070977, "x": 0.048012505814, "raw": 0.811007939916, "bad_r": -0.654990994912, "bad_theta": -0.442279026777},
    "g5_sinh_low_noise": {"naive": -0.174151070977, "x": 0.048012505814, "raw": 0.825562138072, "bad_r": -0.650170889312, "bad_theta": -0.443197874524},
}


def normal_expectation_gh(order: int, function) -> float:
    nodes, weights = roots_hermitenorm(order)
    return float(np.sum(weights * function(nodes)) / SQ2PI)


def tau_integrand(z: np.ndarray | float, c_ul: float, c_ql: float, v_l: float, v_m: float, h: float) -> np.ndarray | float:
    m = np.sqrt(v_m) * z
    q = m / np.sqrt(1.0 + v_l)
    p = 0.1 + 0.8 * ndtr(q)
    derivative = 0.8 * np.exp(-q * q / 2.0) / (SQ2PI * np.sqrt(1.0 + v_l))
    return derivative * (h * c_ul / p + c_ql / (p * (1.0 - p)))


def tau_gh(order: int, c_ul: float, c_ql: float, v_l: float, v_m: float, h: float) -> float:
    return 1.0 + normal_expectation_gh(order, lambda z: tau_integrand(z, c_ul, c_ql, v_l, v_m, h))


def tau_quad(c_ul: float, c_ql: float, v_l: float, v_m: float, h: float) -> float:
    value, _ = quad(
        lambda z: float(tau_integrand(z, c_ul, c_ql, v_l, v_m, h)) * np.exp(-z * z / 2.0) / SQ2PI,
        -np.inf,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-12,
        limit=300,
    )
    return 1.0 + float(value)


def conditional_tuples(sigma: np.ndarray, c: float, b: float = 2.5) -> dict[str, tuple[float, float, float, float]]:
    var_l = 1.69 + b * b
    v_r = 1.0 / (1.0 / 0.8 + np.sum(1.0 / sigma**2))
    return {
        "naive": (1.2, -1.2 * c + 0.25, var_l, 0.0),
        "x": (0.8, -0.8 * c - 0.15, 0.89 + b * b, 0.8),
        "raw": (v_r, -c * v_r, v_r, var_l - v_r),
    }


def clean_roots(sigma: np.ndarray, b: float = 2.5, bad_i: float = -0.6) -> dict[str, float]:
    roots = {}
    for mask in ((1,), (2,), (1, 2), (3,), (1, 3), (2, 3), (1, 2, 3)):
        remaining = sorted(set(range(5)) - set(mask))
        k = len(remaining)
        s = float(np.sum(sigma[remaining] ** 2) / k**2)
        t = (b - np.sqrt(b * b - 4.0 * s)) / 2.0
        roots[",".join(map(str, mask))] = float(t - bad_i / k)
    return roots


def bad_root_and_tuple(sigma: np.ndarray, c: float, b: float = 2.5, bad_i: float = -0.6):
    s = float(np.sum(sigma[:4] ** 2) / 16.0)
    coefficients = (1.0, -(b + bad_i), s + b * bad_i * (1.0 + s / 0.8))
    roots = np.roots(coefficients)
    r = float(sorted((float(x.real) for x in roots if abs(x.imag) < 1e-12), key=lambda x: (abs(x), x))[0])
    denominator = 0.8 + s + r * r
    c_ul = 0.8 - 0.8 * (0.8 + b * r) / denominator
    v_l = 0.8 + b * b - (0.8 + b * r) ** 2 / denominator
    return r, (c_ul, -c * c_ul, v_l, (1.69 + b * b) - v_l)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    settings = {}
    global_40_80 = 0.0
    global_160_quad = 0.0
    for name, spec in SPECS.items():
        sigma = np.asarray(spec["sigma"], dtype=float)
        tuples = conditional_tuples(sigma, spec["c"])
        bad_r, bad_tuple = bad_root_and_tuple(sigma, spec["c"])
        tuples["bad"] = bad_tuple
        values = {}
        for label, parameters in tuples.items():
            gh_values = {str(order): tau_gh(order, *parameters, spec["h"]) for order in ORDERS}
            quad_value = tau_quad(*parameters, spec["h"])
            global_40_80 = max(global_40_80, abs(gh_values["40"] - gh_values["80"]))
            global_160_quad = max(global_160_quad, abs(gh_values["160"] - quad_value))
            values[label] = {"gh": gh_values, "adaptive_quad": quad_value}
        observed = {
            "naive": values["naive"]["adaptive_quad"],
            "x": values["x"]["adaptive_quad"],
            "raw": values["raw"]["adaptive_quad"],
            "bad_r": bad_r,
            "bad_theta": values["bad"]["adaptive_quad"],
        }
        deviations = {key: abs(observed[key] - EXPECTED[name][key]) for key in observed}
        settings[name] = {
            "spec": spec,
            "population_functionals": values,
            "clean_roots_by_split": clean_roots(sigma),
            "bad_selected_root": bad_r,
            "oracle_and_exact_clean": 1.0,
            "canonical_expected": EXPECTED[name],
            "absolute_deviation_from_canonical": deviations,
            "canonical_check_pass": max(deviations.values()) < ABS_TOL_CANONICAL,
        }
    payload = {
        "complete": True,
        "method": "probabilists' Gauss-Hermite orders 40/80/160 plus scipy adaptive quadrature",
        "formula": "tau_Z=1+E[d(m){h*c_UL/p(m)+c_QL/(p(m)(1-p(m)))}]",
        "settings": settings,
        "convergence": {
            "max_abs_gh40_minus_gh80": global_40_80,
            "max_abs_gh160_minus_adaptive_quad": global_160_quad,
            "gh40_80_tolerance": 1e-3,
            "gh160_quad_tolerance": 2e-7,
            "pass": global_40_80 < 1e-3 and global_160_quad < 2e-7,
        },
        "canonical_tolerance": ABS_TOL_CANONICAL,
        "all_checks_pass": all(row["canonical_check_pass"] for row in settings.values()) and global_40_80 < 1e-3 and global_160_quad < 2e-7,
        "claim_boundary": "Numerical population benchmarks only; not a learned-representation, aggregation, or finite-sample theorem certificate.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
