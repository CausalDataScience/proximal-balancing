"""Freeze the v3 prospective-confirmation protocol before any designated replicate is opened
(memo/2026-09-22-v3-confirmation-prd-v1.md, section 9).

Writes results/v3_confirmation_protocol.json, write-once.  It records the pre-registration, every source
file the run and the scoring depend on, the benchmark inputs, the reference file and its stored gate
values, and the replicate and seed manifest for each data set.  It refuses to write if any designated
replicate already has a shard, so "fresh" is verified rather than assumed.

    python3 -B freeze_v3_protocol.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import ihdp_acic_data as ia
import rhc_planted_data as rp
import run_family_v2 as rf
import run_ihdp_acic as ri
import run_rhc_planted as rrp
import run_twins as rtw
import twins_data as td
import v3_fresh_scorer as sc

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
MEMO = HERE.parent / "memo"
PRD = MEMO / "2026-09-22-v3-confirmation-prd-v1.md"
OUT = RESULTS / "v3_confirmation_protocol.json"

# Section 3 of the pre-registration.  Replicate ids and the seed rule of each data set, unchanged.
PLAN = {
    "twins": {"replicates": list(range(11, 21)), "seed_base": td.REPLICATE_SEED,
              "fold_offset": rtw.FOLD_SEED_OFFSET, "x_coef_seed": td.X_COEF_SEED,
              "unit": "the real X, hidden gestational age and both potential outcomes are fixed; the proxy "
                      "noise and the treatment assignment are redrawn"},
    "rhc_planted": {"replicates": list(range(11, 21)), "seed_base": rp.REPLICATE_SEED,
                    "fold_offset": rtw.FOLD_SEED_OFFSET, "x_coef_seed": rp.X_COEF_SEED,
                    "unit": "the real X, hidden APACHE score, real proxies and the outcome baseline are all "
                            "fixed; only the treatment assignment is redrawn"},
    "ihdp": {"replicates": list(range(12, 22)), "seed_base": ia.IHDP_SEED,
             "fold_offset": rtw.FOLD_SEED_OFFSET, "x_coef_seed": ia.IHDP_X_COEF_SEED,
             "unit": "the replicate index selects a different benchmark realization of the potential "
                     "outcomes as well; the proxies and treatment are then generated on it"},
}

# The file list each runner actually stamps into its own shards, per data set.  Using another runner's
# list makes the integrity check compare the wrong names, which is what happened when this protocol was
# first frozen (see the correction recorded in v3_confirmation_protocol_v2.json).
RUNNER_SOURCES = {"twins": rtw.SOURCES, "rhc_planted": rrp.SOURCES, "ihdp": ri.SOURCES}

# Every file whose content can change a number in this experiment.
SOURCES = sorted(set(rtw.SOURCES) | set(rrp.SOURCES) | set(ri.SOURCES) |
                 {"noise_scaled.py", "test_noise_scaled.py", "v3_fresh_scorer.py", "freeze_v3_protocol.py",
                  "rhc_pooled.py", "hidden_proxy_recipe.py", "twins_data.py", "rhc_planted_data.py",
                  "rhc_data.py", "ihdp_acic_data.py", "probe_structured_scm.py", "family_v2_probe.py",
                  "family_v2_dgp.py", "family_v2_competitors.py", "run_family_v2.py"})

BENCHMARK_INPUTS = {
    "twins": ["twins/twin_pairs_X_3years_samesex.csv", "twins/twin_pairs_T_3years_samesex.csv",
              "twins/twin_pairs_Y_3years_samesex.csv"],
    "rhc_planted": ["rhc/rhc.csv"],
    "ihdp": ["ihdp/ihdp_npci_1-100.train.npz", "ihdp/ihdp_npci_1-100.test.npz"],
}


def existing_shards(dataset: str, replicates: list[int]) -> list[int]:
    return [r for r in replicates if (RESULTS / dataset / f"replicate_{r}.json").exists()]


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; the protocol is write-once")
    clashes = {d: existing_shards(d, p["replicates"]) for d, p in PLAN.items()}
    if any(clashes.values()):
        raise SystemExit(f"designated replicates already have shards, so they are not fresh: {clashes}")

    source_hashes = {name: rf.sha256(HERE / name) for name in SOURCES if (HERE / name).exists()}
    missing = [name for name in SOURCES if not (HERE / name).exists()]
    if missing:
        raise SystemExit(f"these source files are named in the protocol but absent: {missing}")

    reference = json.loads(sc.REFERENCE.read_text())
    manifests = {}
    for dataset, plan in PLAN.items():
        expected = {}
        for r in plan["replicates"]:
            expected[str(r)] = {"dataset": dataset,
                                "seed": plan["seed_base"] + 1000 * r + plan["fold_offset"],
                                "source_sha256": {k: source_hashes[k] for k in RUNNER_SOURCES[dataset]
                                                  if k in source_hashes}}
        ref = reference["real_data"][dataset]
        manifests[dataset] = {
            "replicates": plan["replicates"], "seed_base": plan["seed_base"],
            "fold_seed_offset": plan["fold_offset"], "x_coefficient_seed": plan["x_coef_seed"],
            "replicate_unit": plan["unit"], "expected": expected,
            "reference_conditional_mae": ref["mae"],
            "reference_replicates": [x["replicate"] for x in ref["replicates"]],
            "gate_limit": sc.GATE_MAE_MULTIPLE * ref["mae"],
            "gate_return_minimum": sc.GATE_RETURN_MIN,
            "tpr_fpr_available": sc.HAS_VALIDITY_TRUTH[dataset],
            "benchmark_input_sha256": {p: rf.sha256(HERE.parent / "materials" / "real_world_data" / p)
                                       for p in BENCHMARK_INPUTS[dataset]},
        }

    payload = {
        "artifact": "v3_confirmation_protocol",
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "prd": str(PRD.relative_to(HERE.parent)),
        "protocol_sha256_of_prd": rf.sha256(PRD),
        "rule": "v3: threshold = median over the thirty candidates of z * SE; see noise_scaled.py",
        "z": ns_z(),
        "gates": {"return_rate_minimum": sc.GATE_RETURN_MIN,
                  "conditional_mae_multiple": sc.GATE_MAE_MULTIPLE,
                  "note": "gate (i) checks a tie in the aggregation, not the power of the screen; the "
                          "informative gate is (ii)"},
        "reference_file": {"path": str(sc.REFERENCE.relative_to(HERE.parent)),
                           "sha256": rf.sha256(sc.REFERENCE)},
        "acic": {"status": "excluded from the prospective confirmation",
                 "reason": "the ten distributed realizations were all used; realization 1 was the design "
                           "check and 2-10 are the data on which the v3 rule was chosen"},
        "source_sha256": source_hashes,
        "manifests": manifests,
        "verified_before_freezing": {"designated_replicates_have_no_prior_shards": True,
                                     "checked": {d: p["replicates"] for d, p in PLAN.items()}},
    }
    with open(OUT, "x") as handle:
        handle.write(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(f"frozen {OUT} at {payload['frozen_utc']}")
    for dataset, m in manifests.items():
        print(f"  {dataset:12s} replicates {m['replicates'][0]}-{m['replicates'][-1]} | "
              f"reference MAE {m['reference_conditional_mae']:.6f} limit {m['gate_limit']:.6f} | "
              f"TPR/FPR {'yes' if m['tpr_fpr_available'] else 'N/A'}")
    print(f"  {len(source_hashes)} source files hashed")
    return 0


def ns_z() -> float:
    import noise_scaled as ns
    return ns.Z


if __name__ == "__main__":
    raise SystemExit(main())
