"""Fail-closed verdicts for the fixed Design-F P0 protocol.

Usage:
  python3 verdicts.py --probe-manuscript PM.json --probe-variant PV.json \
      --stage4-manuscript S4.json [--out VERDICTS.json]

The three required artifacts are one evidence bundle. A missing, malformed, incomplete, wrong-mode or
unverifiable artifact makes every verdict NOT_EVALUATED. Performance is evaluated only from Stage 4, with
optimizer seeds aggregated inside each independent data seed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import provenance as prov

PROTOCOL_ID = "dgp-f-r10-p0-v1"
CRITERIA_PATH = Path(__file__).with_name("frozen_criteria.json")
CRITERIA = json.loads(CRITERIA_PATH.read_text())
CRITERIA_SHA256 = prov.file_sha256(CRITERIA_PATH)
REQUIRED_CONTROLS = ("identical_initial_weights", "val_rows_fixed", "audit_rows_fixed",
                     "estimation_rows_fixed")
PROBE_DATA_SEEDS = tuple(range(20260912, 20260917))
PROBE_OPT_SEEDS = tuple(range(5))
STAGE4_DATA_SEEDS = (20261211, 20261212, 20261213)
STAGE4_OPT_SEEDS = (0, 1, 2)
STAGE4_SIZES = (3000, 12000, 48000)
PROBE_SOURCE_ROLES = {"coefficient_probe.py", "staged_dgp_f.py", "probe_brier_neural.py",
                      "probe_dgp_ae.py", "provenance.py", "frozen_criteria.json"}
STAGE4_SOURCE_ROLES = {"staged_dgp_f.py", "probe_brier_neural.py", "probe_dgp_ae.py",
                       "provenance.py", "frozen_criteria.json"}
HYPOTHESIS_KEYS = (
    "population_grid_selects_causally_successful_candidate",
    "true_probability_split_sample_grid_selects_causally_successful_candidate",
    "optimizer_misses_own_finite_candidate_minimum",
    "critic_approximation_vs_sampling_attribution",
)


def hierarchical_stage4(cells: list[dict], n_rep: int) -> dict:
    """Aggregate optimizer variation within data seed, then summarize independent data seeds."""
    chosen = [c for c in cells if int(c["n_rep"]) == int(n_rep)]
    if not chosen:
        raise ValueError(f"no stage4 cells at n_rep={n_rep}")
    grouped: dict[int, list[float]] = {}
    for c in chosen:
        grouped.setdefault(int(c["data_seed"]), []).append(float(c["err_representation"]))
    per = [float(np.median(grouped[k])) for k in sorted(grouped)]
    return {"n_rep": int(n_rep), "per_data_seed": per, "data_seeds": len(per),
            "optimizer_cells": len(chosen), "median": float(np.median(per)),
            "p90": float(np.quantile(per, 0.9))}


def _probe_cells(payload: dict) -> list[dict]:
    return [pick for run in payload["runs"] for pick in run["picks"]]


def _probe_expected(payload: dict) -> set[tuple[int, int]]:
    return {(d, o) for d in PROBE_DATA_SEEDS for o in PROBE_OPT_SEEDS}


def _stage4_cells(payload: dict) -> list[dict]:
    return payload["stage4"]["cells"]


def _stage4_expected(payload: dict) -> set[tuple[int, int, int]]:
    return {(d, n, o) for d in STAGE4_DATA_SEEDS for n in STAGE4_SIZES for o in STAGE4_OPT_SEEDS}


def _read(path: str | None, role: str) -> tuple[dict | None, list[str]]:
    if not path:
        return None, [f"required artifact {role} was not supplied"]
    try:
        return json.loads(Path(path).read_text()), []
    except (OSError, ValueError, TypeError) as e:
        return None, [f"{role} at {path} cannot be read: {e}"]


def _input_records(paths: dict[str, str | None]) -> dict:
    out = {}
    for role in ("probe_manuscript", "probe_variant", "stage4_manuscript"):
        path = paths.get(role)
        record = {"path": str(Path(path).resolve()) if path else None, "sha256": None}
        if path:
            try:
                record["sha256"] = prov.file_sha256(path)
            except OSError:
                pass
        out[role] = record
    return out


def _validate_probe(payload: dict, path: str, mode: str) -> list[str]:
    problems = prov.verify(
        payload, path, expect_mode=mode, expect_cells=_probe_expected(payload),
        cell_key=lambda c: (int(c["data_seed"]), int(c["opt_seed"])), cells_at=_probe_cells,
        require_manifest=True, require_checkpoints=True, expected_criteria_sha256=CRITERIA_SHA256,
        expected_source_roles=PROBE_SOURCE_ROLES, checkpoint_protocol_id=PROTOCOL_ID)
    protocol = payload.get("provenance", {}).get("protocol_id")
    if protocol != PROTOCOL_ID:
        problems.append(f"unexpected protocol_id {protocol!r}")
    p = payload["provenance"]
    if tuple(map(int, p["seeds"]["data"])) != PROBE_DATA_SEEDS:
        problems.append("coefficient data-seed manifest differs from the frozen protocol")
    if tuple(map(int, p["seeds"]["optimizer"])) != PROBE_OPT_SEEDS:
        problems.append("coefficient optimizer-seed manifest differs from the frozen protocol")
    expected_args = {"n": 24000, "data_seeds": 5, "opt_seeds": 5, "training_mode": mode}
    wrong = {k: (p["args"].get(k), v) for k, v in expected_args.items() if p["args"].get(k) != v}
    if wrong:
        problems.append(f"coefficient arguments differ from the frozen protocol: {wrong}")
    for cell in _probe_cells(payload):
        common = cell.get("common_evaluation")
        needed = ("select_objective", "audit_objective", "select_regret", "audit_regret")
        if not isinstance(common, dict) or any(k not in common for k in needed):
            problems.append("a coefficient cell lacks common-objective select/audit regret")
            break
    if payload.get("diagnosis_complete") is not True:
        problems.append("coefficient diagnosis_complete is not true")
    return problems


def _validate_stage4(payload: dict, path: str) -> list[str]:
    problems = prov.verify(
        payload, path, expect_mode="manuscript", expect_cells=_stage4_expected(payload),
        cell_key=lambda c: (int(c["data_seed"]), int(c["n_rep"]), int(c["opt_seed"])),
        cells_at=_stage4_cells, require_manifest=True, require_checkpoints=True,
        expected_criteria_sha256=CRITERIA_SHA256, expected_source_roles=STAGE4_SOURCE_ROLES,
        checkpoint_protocol_id=PROTOCOL_ID)
    protocol = payload.get("provenance", {}).get("protocol_id")
    if protocol != PROTOCOL_ID:
        problems.append(f"unexpected protocol_id {protocol!r}")
    p, s4 = payload["provenance"], payload["stage4"]
    if tuple(map(int, p["seeds"]["data"])) != STAGE4_DATA_SEEDS:
        problems.append("stage4 data-seed manifest differs from the frozen protocol")
    if tuple(map(int, p["seeds"]["optimizer"])) != STAGE4_OPT_SEEDS:
        problems.append("stage4 optimizer-seed manifest differs from the frozen protocol")
    expected_args = {"stage4_rep": list(STAGE4_SIZES), "stage4_est": 32000, "data_seeds": 3,
                     "opt_seeds": 3, "fixed_preprocessing": True, "stages": "4",
                     "training_mode": "manuscript"}
    wrong = {k: (p["args"].get(k), v) for k, v in expected_args.items() if p["args"].get(k) != v}
    if wrong:
        problems.append(f"stage4 arguments differ from the frozen protocol: {wrong}")
    if tuple(map(int, s4["sizes"])) != STAGE4_SIZES:
        problems.append("stage4 saved sizes differ from the frozen protocol")
    if int(s4["data_seeds"]) != 3 or int(s4["opt_seeds"]) != 3:
        problems.append("stage4 saved replicate counts differ from the frozen protocol")
    controls = payload.get("stage4", {}).get("controls", {})
    missing = [k for k in REQUIRED_CONTROLS if controls.get(k) is not True]
    if missing:
        problems.append(f"stage4 controls are not all true: {missing}")
    # Recompute split invariants from every cell for every data seed.  A producer's summary boolean is not
    # evidence: round 10 found that an earlier producer had checked only the first data seed.
    recomputed = {}
    for data_seed in STAGE4_DATA_SEEDS:
        selected = [c for c in _stage4_cells(payload) if int(c["data_seed"]) == data_seed]
        recomputed[str(data_seed)] = {
            "val_rows_fixed": len({c["split_ids"]["val"] for c in selected}) == 1,
            "audit_rows_fixed": len({c["split_ids"]["audit"] for c in selected}) == 1,
            "estimation_rows_fixed": len({c["estimation_split_id"] for c in selected}) == 1,
        }
    failed = {ds: [k for k, ok in values.items() if not ok]
              for ds, values in recomputed.items() if not all(values.values())}
    if failed:
        problems.append(f"stage4 split invariants fail when recomputed by data seed: {failed}")
    recorded_by_seed = s4.get("controls_by_data_seed")
    if not isinstance(recorded_by_seed, dict):
        problems.append("stage4 has no controls_by_data_seed record")
    else:
        for ds, values in recomputed.items():
            recorded = recorded_by_seed.get(ds)
            if not isinstance(recorded, dict):
                problems.append(f"stage4 controls_by_data_seed lacks {ds}")
                continue
            for name, ok in values.items():
                if recorded.get(name) is not ok:
                    problems.append(f"stage4 recorded {name} for data seed {ds} differs from recomputation")
    return problems


def _diagnostic_hypotheses(pm: dict, pv: dict) -> dict:
    """Protocol-specific scientific interpretations, not additional frozen pass gates."""
    pop = [r["grid_minimisers"]["population_D2"]["selected_success"] for r in pm["runs"]]
    truth = [r["grid_minimisers"]["finite_sample_oracle_gap_select"]["selected_success"]
             for r in pm["runs"]]
    fitted = [r["grid_minimisers"]["common_gap_select"]["selected_success"] for r in pm["runs"]]
    regrets = [float(p["common_evaluation"]["select_regret"])
               for payload in (pm, pv) for p in _probe_cells(payload)]
    return {
        "population_grid_selects_causally_successful_candidate": {
            "status": "SUPPORTED" if all(pop) else "NOT_SUPPORTED", "successes": int(sum(pop)), "total": len(pop)},
        "true_probability_split_sample_grid_selects_causally_successful_candidate": {
            "status": "SUPPORTED" if all(truth) else "NOT_SUPPORTED",
            "successes": int(sum(truth)), "total": len(truth)},
        "optimizer_misses_own_finite_candidate_minimum": {
            "status": "NOT_SUPPORTED", "median_select_regret": float(np.median(regrets)),
            "p90_select_regret": float(np.quantile(regrets, 0.9)),
            "note": "observed finite-candidate regrets are small; this is not a global optimisation bound"},
        "critic_approximation_vs_sampling_attribution": {
            "status": "UNRESOLVED", "fitted_grid_successes": int(sum(fitted)), "total": len(fitted),
            "note": "the saved run separates true-probability and fitted-critic rankings but does not isolate critic approximation from sampling variation"},
    }


def evaluate(paths: dict[str, str | None]) -> tuple[dict, int]:
    payloads, problems = {}, []
    evaluation_provenance = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluator": {"path": str(Path(__file__).resolve()), "sha256": prov.file_sha256(__file__)},
        "criteria": {"path": str(CRITERIA_PATH.resolve()), "sha256": CRITERIA_SHA256},
        "inputs": _input_records(paths),
    }
    for role in ("probe_manuscript", "probe_variant", "stage4_manuscript"):
        payloads[role], errs = _read(paths.get(role), role)
        problems.extend(errs)
    if not problems:
        for role, mode in (("probe_manuscript", "manuscript"), ("probe_variant", "variant")):
            try:
                problems.extend(f"{role}: {x}" for x in
                                _validate_probe(payloads[role], paths[role], mode))
            except (KeyError, TypeError, ValueError) as e:
                problems.append(f"{role}: required schema cannot be read: {e}")
        try:
            problems.extend(f"stage4_manuscript: {x}" for x in
                            _validate_stage4(payloads["stage4_manuscript"], paths["stage4_manuscript"]))
        except (KeyError, TypeError, ValueError) as e:
            problems.append(f"stage4_manuscript: required schema cannot be read: {e}")
    if problems:
        verdicts = {k: "NOT_EVALUATED" for k in
                    ("implementation", "diagnostic_computation", "performance")}
        current = {role: (prov.current_source_status(payload) if payload is not None
                          else {"manifest": "UNAVAILABLE"})
                   for role, payload in payloads.items()}
        result = {"protocol_id": PROTOCOL_ID, "criteria_sha256": CRITERIA_SHA256,
                  "verdicts": verdicts, "problems": problems, "authoritative": False,
                  "evaluation_provenance": evaluation_provenance,
                  "current_source_status": current,
                  "hypotheses": {k: {"status": "NOT_EVALUATED"} for k in HYPOTHESIS_KEYS}}
        try:
            result["non_authoritative_descriptive_hypotheses"] = _diagnostic_hypotheses(
                payloads["probe_manuscript"], payloads["probe_variant"])
            result["non_authoritative_note"] = (
                "These labels are a descriptive rescore of readable fields in rejected artifacts; they are "
                "not verdicts and cannot turn NOT_EVALUATED into PASS or FAIL.")
        except (KeyError, TypeError, ValueError):
            pass
        return result, 2

    pm, pv, s4 = payloads["probe_manuscript"], payloads["probe_variant"], payloads["stage4_manuscript"]
    diagnosis_ok = all(p.get("diagnosis_complete") is True for p in (pm, pv))
    largest = max(int(x) for x in s4["stage4"]["sizes"])
    perf = hierarchical_stage4(_stage4_cells(s4), largest)
    gate = CRITERIA["performance"]["development_gate"]
    performance_ok = perf["median"] <= gate["median_at_most"] and perf["p90"] <= gate["p90_at_most"]
    status = {"implementation": "PASS",
              "diagnostic_computation": "PASS" if diagnosis_ok else "FAIL",
              "performance": "PASS" if performance_ok else "FAIL"}
    current = {role: prov.current_source_status(payload) for role, payload in payloads.items()}
    return {"protocol_id": PROTOCOL_ID, "criteria_sha256": CRITERIA_SHA256, "verdicts": status,
            "stage4_hierarchical": perf, "current_source_status": current,
            "authoritative": True, "evaluation_provenance": evaluation_provenance,
            "hypotheses": _diagnostic_hypotheses(pm, pv),
            "note": "diagnostic computation PASS means the four diagnostics were computed, not that a tied "
                    "or selected coefficient is causally successful"}, 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-manuscript")
    ap.add_argument("--probe-variant")
    ap.add_argument("--stage4-manuscript")
    ap.add_argument("--out")
    args = ap.parse_args()
    result, code = evaluate(vars(args))
    print(f"protocol {PROTOCOL_ID} | criteria {CRITERIA_SHA256}")
    print("\nverdicts")
    for name in ("implementation", "diagnostic_computation", "performance"):
        print(f"  {name:15s} {result['verdicts'][name]}")
    for problem in result.get("problems", []):
        print(f"  - {problem}")
    if "stage4_hierarchical" in result:
        h = result["stage4_hierarchical"]
        print(f"  stage4 n={h['n_rep']}: data-seed median {h['median']:.6f}, p90 {h['p90']:.6f}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
