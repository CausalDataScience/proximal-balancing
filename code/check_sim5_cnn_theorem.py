"""P0 feasibility ledger for the Sim5 CNN PRD (memo/2026-09-18-sim5-cnn-implementation-prd-v1.md, section 4).

Everything here is a population or bank-level computation.  No learner is fitted and no data seed is used,
except the split-draw Monte Carlo, whose seed is a diagnostic seed recorded in the ledger.

P0-A  Is the latent state recoverable from the full image?  If so, ideal raw adjustment targets tau.
P0-B  Can N, Delta, L over all T = 2^64 - 2 candidates be certified?  Block heterogeneity decides whether
      a symmetry reduction exists; an ideal-decoder Monte Carlo is reported as a diagnostic only.
P0-C  Exact or certified approximate candidate values for a learned CNN.
P0-D  Uniform radius and separation.  The manuscript's sufficient construction (Chebyshev with a union bound,
      delta_E = delta / N) is evaluated; the theorem itself only needs some valid common radius.

Verdicts say what the current code and the manuscript's sufficient constructions certify.  NOT VERIFIED means
not certified here, not shown impossible.

Writes results/sim5_cnn/p0_ledger_v1.json.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import norm

import sim5_cnn_dgp as g

OUT = Path(__file__).resolve().parents[1] / "results" / "sim5_cnn" / "p0_ledger_v1.json"
DIAG_SEED = 70_918_001              # split-draw Monte Carlo only; not a data seed
N_SPLITS_MC = 20_000
N_E, DELTA, M = 20_000, 0.05, 128   # PRD provisional values
T = 2 ** 64 - 2


def pair_distinguishability() -> tuple[np.ndarray, np.ndarray]:
    """dist[k, j] = block j separates state pair k by more than twice the noise half width."""
    tpl = g.bank()["templates"].reshape(g.N_STATES, -1)
    masks = g.block_masks().reshape(64, -1)
    iu, ju = np.triu_indices(g.N_STATES, 1)
    gap = np.abs(tpl[iu] - tpl[ju])                                   # (pairs, 784)
    dist = np.stack([gap[:, m].max(axis=1) for m in masks], axis=1) > 2 * g.NOISE_HALF_WIDTH
    full = gap.max(axis=1)
    return dist, full


def oracle_score_sd(order: int = 40) -> float:
    """sd of the AIPW score with Z = (X, U) and true nuisances: Var = Var(tau_U) + E[s2/e + s2/(1-e)]."""
    st = g.states()
    t, w = np.polynomial.hermite_e.hermegauss(order); w = w / w.sum()
    s2 = (0.3 + 0.2 * np.abs(st["s"])) ** 2
    inv = np.zeros(g.N_STATES)
    for x3 in (-1.0, 1.0):
        e = g.propensity(st["G"][:, None], 0.6 * st["r"][:, None] + t[None, :], x3)
        inv += 0.5 * ((1 / e + 1 / (1 - e)) * w).sum(1)
    return float(math.sqrt(st["tau_u"].var() + (s2 * inv).mean()))


def main() -> int:
    b = g.bank()
    pop = g.population_summary(order=24)
    dist, full = pair_distinguishability()
    n_pairs = dist.shape[0]

    # P0-A: full-image recovery
    full_min = float(full.min())
    recoverable = full_min > 2 * g.NOISE_HALF_WIDTH

    # P0-B: block heterogeneity and the ideal-decoder diagnostic
    per_block = dist.sum(axis=0)
    profiles = {bytes(np.packbits(dist[:, j])) for j in range(64)}
    tpl = b["templates"].reshape(g.N_STATES, -1)
    common_zero = (tpl == 0).all(axis=0)
    masks = g.block_masks().reshape(64, -1)
    background_blocks = [j for j in range(64) if common_zero[masks[j]].all()]
    b_k = dist.sum(axis=1)
    union_one_side = float(np.sum(2.0 ** (-b_k.astype(float))))
    rng = np.random.default_rng(DIAG_SEED)
    packed = np.packbits(dist.T, axis=1)                              # (64, bytes)
    full_bytes = np.packbits(np.ones(n_pairs, dtype=bool))
    held_ok = kept_ok = both_ok = held_background = 0
    for _ in range(N_SPLITS_MC):
        while True:
            sel = rng.random(64) < 0.5
            if 0 < sel.sum() < 64:
                break
        or_held = np.bitwise_or.reduce(packed[sel], axis=0)
        or_kept = np.bitwise_or.reduce(packed[~sel], axis=0)
        h_ok = np.array_equal(or_held, full_bytes); k_ok = np.array_equal(or_kept, full_bytes)
        held_ok += h_ok; kept_ok += k_ok; both_ok += h_ok and k_ok
        held_background += set(np.flatnonzero(sel)) <= set(background_blocks)
    p_background_only = (2 ** len(background_blocks) - 1) / T        # exact under uniform draws

    # P0-D: radius and separation
    sigma = oracle_score_sd()
    gap_tau_x = 1.0 - pop["theta_X"]
    rho_sampled = sigma * math.sqrt(M / (N_E * DELTA))                # delta_E = delta / R, R <= m
    rho_universe = {f"p={p}": sigma * math.sqrt(p * T / (N_E * DELTA)) for p in (0.01, 0.5, 1.0)}

    ledger = {
        "prd": "memo/2026-09-18-sim5-cnn-implementation-prd-v1.md",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_sha256": {n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest()
                          for n in ("sim5_cnn_dgp.py", "check_sim5_cnn_theorem.py")},
        "mnist_sha256": b["mnist_sha256"], "bank_source_index": b["source_index"].tolist(),
        "S0_population": {**pop, "prd_c": g.C_OUTCOME, "prd_naive": -0.5, "prd_theta_X": -0.53656},
        "P0_A": {
            "min_pairwise_sup_gap_template_units": full_min, "noise_threshold": 2 * g.NOISE_HALF_WIDTH,
            "U_exactly_recoverable_from_full_W": bool(recoverable),
            "ideal_raw_adjustment_target": 1.0 if recoverable else None,
            "verdict": "STRUCTURAL FAIL / FINITE-SAMPLE FALLBACK AVAILABLE" if recoverable else "RECHECK",
            "reason": "nearest-template decoding on (X, W) is exact, so the population raw target equals tau",
        },
        "P0_B": {
            "T": str(T), "m": M, "pairs": n_pairs,
            "pairs_separated_per_block": {"min": int(per_block.min()), "median": float(np.median(per_block)),
                                          "max": int(per_block.max())},
            "distinct_block_profiles": len(profiles), "background_only_blocks": background_blocks,
            "hardest_pair_block_count": int(b_k.min()),
            "union_bound_one_side_fails": union_one_side,
            "ideal_decoder_mc": {"seed": DIAG_SEED, "splits": N_SPLITS_MC,
                                 "held_out_identifies_U": held_ok / N_SPLITS_MC,
                                 "kept_identifies_U": kept_ok / N_SPLITS_MC,
                                 "both_identify_U": both_ok / N_SPLITS_MC,
                                 "held_out_all_background": held_background / N_SPLITS_MC,
                                 "reading": "information exists on each side; says nothing about what a learned CNN encodes"},
            "exact_prob_held_out_background_only": p_background_only,
            "verdict": "NOT VERIFIED",
            "reason": ("blocks carry different information (56 distinct profiles), so the block-permutation "
                       "symmetry reduction is unavailable; no other certificate of N, Delta, L for a learned CNN "
                       "is implemented. Not shown impossible."),
        },
        "P0_C": {
            "exact_path": "NOT VERIFIED: equality theta_S = tau needs the learned hard code to make (X, W_S) "
                          "independent of A given Z exactly; finite test accuracy does not establish it",
            "approximate_path": "NOT VERIFIED: b_S needs Gamma_S(phi_hat) and a population D_res bound; "
                                "no such bound is computed here for a learned CNN",
            "verdict": "NOT VERIFIED",
        },
        "P0_D": {
            "oracle_score_sd": sigma, "n_E": N_E, "delta": DELTA,
            "rho_with_delta_over_R_max": rho_sampled,
            "rho_with_delta_over_N": rho_universe,
            "gap_tau_minus_theta_X": gap_tau_x, "four_rho_sampled": 4 * rho_sampled,
            "verdict": "NOT VERIFIED",
            "reason": ("the manuscript's sufficient construction sigma/sqrt(n_E delta/N) with N up to 2^64-2 is "
                       "vacuous even for the oracle score. Condition (b) only needs a valid common radius; a "
                       "sharper concentration inequality could supply one but is not derived here. The nuisance "
                       "term q_e(q_0+q_1)/eta is not bounded here for fitted networks."),
        },
        "go_no_go": {
            "literal_theorem_arm": "NOT STARTED: P0-B, P0-C, P0-D unresolved (PRD 4.5); not shown impossible",
            "operational_empirical_arm": "requires user approval and a protocol revision (PRD section 17)",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=1, allow_nan=False) + "\n")
    tmp.replace(OUT)
    print(json.dumps({k: ledger[k] for k in ("S0_population",)}, indent=1))
    for k in ("P0_A", "P0_B", "P0_D"):
        print(k, json.dumps({kk: vv for kk, vv in ledger[k].items() if kk not in ("reason",)}, default=str))
    print("written", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
