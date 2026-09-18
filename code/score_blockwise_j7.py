"""Score the seven-block confirmation against its frozen gates.

The primary result is the budget m the protocol fixed before the run.  The budget
curve is reported for every m in the frozen sweep, ungated, whatever it shows.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

import probe_structured_scm as p
from probe_structured_scm import source_sha256, write_json_atomic

RESULTS = Path("../results")
PROTOCOL = RESULTS / "probe_structured_scm_blockwise_j7_confirm_protocol_v1.json"
OUTPUT = RESULTS / "probe_structured_scm_blockwise_j7_confirm_summary_v1.json"
TAU = 1.0
FILES = {
    "g1_low_noise": "probe_structured_scm_blockwise_j7_confirm_v1_g1.json",
    "g2_heteroscedastic_noise": "probe_structured_scm_blockwise_j7_confirm_v1_g2.json",
    "g3_heterogeneous_low_noise": "probe_structured_scm_blockwise_j7_confirm_v1_g3.json",
    "g4_proxy_heteroscedastic_low_noise": "probe_structured_scm_blockwise_j7_confirm_v1_g4.json",
    "g5_sinh_low_noise": "probe_structured_scm_blockwise_j7_confirm_v1_g5.json",
}


def upper_t95(values: np.ndarray) -> float:
    n = len(values)
    se = values.std(ddof=1) / np.sqrt(n)
    return float(values.mean() + student_t.ppf(0.95, n - 1) * se)


def score_setting(rows: list[dict], gates: dict) -> dict:
    estimates = np.array([r["aggregation"]["output"] if r["aggregation"]["output"] is not None else np.nan
                          for r in rows])
    returned = ~np.isnan(estimates)
    err = np.abs(estimates[returned] - TAU)
    base = {name: np.array([r["baseline_estimates"][name] for r in rows])[returned]
            for name in ("X", "raw_XW", "oracle", "naive")}
    base_err = {name: np.abs(values - TAU) for name, values in base.items()}
    paired = {
        "vs_X": {"mean": float((err - base_err["X"]).mean()), "upper_t95": upper_t95(err - base_err["X"])},
        "vs_raw_XW": {"mean": float((err - base_err["raw_XW"]).mean()),
                      "upper_t95": upper_t95(err - base_err["raw_XW"])},
        "oracle_excess": {"mean": float((err - base_err["oracle"]).mean()),
                          "upper_t95": upper_t95(err - base_err["oracle"])},
    }
    # category accounting restricted to the splits the primary budget actually drew
    retained_primary, component_primary = Counter(), Counter()
    for row in rows:
        order = p.budget_permutation(row["split_sample_seed"])
        drawn = set(order[: row["budget_m"]])
        for fit in row["split_fits"]:
            split = tuple(fit["split"])
            if split in drawn and fit["pass_all_four"]:
                retained_primary[p.split_category(split)] += 1
        for component in row["aggregation"].get("members", []):
            for split in component:
                component_primary[p.split_category(tuple(split))] += 1
    curve = {}
    for row in rows:
        for entry in row["budget_curve"]:
            curve.setdefault(entry["m"], []).append(entry)
    budget_curve = []
    for m in sorted(curve):
        outs = np.array([e["output"] if e["output"] is not None else np.nan for e in curve[m]])
        got = ~np.isnan(outs)
        budget_curve.append({
            "m": m,
            "return_rate": float(got.mean()),
            "mae": float(np.abs(outs[got] - TAU).mean()) if got.any() else None,
            "p90_abs_error": float(np.quantile(np.abs(outs[got] - TAU), 0.9)) if got.any() else None,
            "mean_retained": float(np.mean([e["retained"] for e in curve[m]])),
        })
    mae, p90 = float(err.mean()), float(np.quantile(err, 0.9))
    return_rate = float(returned.mean())
    checks = {
        "return_rate": return_rate >= gates["return_rate_min"],
        "mae": mae <= gates["aggregate_mae_max"],
        "p90": p90 <= gates["aggregate_p90_abs_error_max"],
        "beats_X": paired["vs_X"]["upper_t95"] < gates["paired_abs_error_ci_upper_vs_X"],
        "beats_raw_XW": paired["vs_raw_XW"]["upper_t95"] < gates["paired_abs_error_ci_upper_vs_raw_XW"],
        "oracle_excess": paired["oracle_excess"]["upper_t95"] <= gates["paired_oracle_excess_ci_upper"],
    }
    return {
        "mean_estimate": float(np.nanmean(estimates)),
        "naive_mean": float(np.mean([r["baseline_estimates"]["naive"] for r in rows])),
        "naive_negative_count": int(sum(r["baseline_estimates"]["naive"] < 0 for r in rows)),
        "mae": mae,
        "p90_abs_error": p90,
        "return_rate": return_rate,
        "baseline_mae": {name: float(values.mean()) for name, values in base_err.items()},
        "paired_abs_error_difference": paired,
        "retained_totals_primary_budget": dict(retained_primary),
        "selected_component_totals_primary_budget": dict(component_primary),
        "budget_curve": budget_curve,
        "checks": checks,
        "all_gates_pass": all(checks.values()),
    }


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text())
    gates = protocol["gates"]
    summary = {
        "complete": True,
        "protocol": {"path": str(PROTOCOL), "protocol_id": protocol["protocol_id"],
                     "sha256": source_sha256(PROTOCOL)},
        "frozen_gates": gates,
        "block_partition": protocol["block_partition"],
        "primary_budget_m": protocol["budget"]["primary_m"],
        "settings": {},
    }
    for setting, filename in FILES.items():
        path = RESULTS / filename
        payload = json.loads(path.read_text())
        assert payload["complete"] and payload["settings"] == [setting]
        entry = score_setting(payload["rows"], gates)
        entry["raw_result"] = str(path)
        entry["raw_sha256"] = source_sha256(path)
        summary["settings"][setting] = entry
    passed = sum(v["all_gates_pass"] for v in summary["settings"].values())
    summary["settings_passing_all_gates"] = passed
    summary["verdict"] = "PASS" if passed == len(FILES) else "FAIL"
    write_json_atomic(OUTPUT, summary)

    print(f"verdict {summary['verdict']}  ({passed}/{len(FILES)} settings pass every gate)")
    header = f"{'setting':34s} {'mean':>7s} {'MAE':>7s} {'p90':>7s} {'ret':>5s} {'vsX':>8s} {'vsXW':>8s} {'orc':>7s}  gates"
    print(header)
    for setting, e in summary["settings"].items():
        pd = e["paired_abs_error_difference"]
        print(f"{setting:34s} {e['mean_estimate']:7.4f} {e['mae']:7.4f} {e['p90_abs_error']:7.4f} "
              f"{e['return_rate']:5.2f} {pd['vs_X']['upper_t95']:8.4f} {pd['vs_raw_XW']['upper_t95']:8.4f} "
              f"{pd['oracle_excess']['upper_t95']:7.4f}  {'PASS' if e['all_gates_pass'] else 'FAIL'}")
    print("\nbudget curve, pooled over the five settings")
    pooled = {}
    for e in summary["settings"].values():
        for entry in e["budget_curve"]:
            pooled.setdefault(entry["m"], []).append(entry)
    print(f"{'m':>5s} {'return':>7s} {'MAE':>7s} {'p90':>7s} {'retained':>9s}")
    for m in sorted(pooled):
        rows = pooled[m]
        maes = [r["mae"] for r in rows if r["mae"] is not None]
        p90s = [r["p90_abs_error"] for r in rows if r["p90_abs_error"] is not None]
        print(f"{m:5d} {np.mean([r['return_rate'] for r in rows]):7.2f} {np.mean(maes):7.4f} "
              f"{np.mean(p90s):7.4f} {np.mean([r['mean_retained'] for r in rows]):9.1f}")
    print("\nretained splits by category, primary budget, summed over settings and seeds")
    totals, chosen = Counter(), Counter()
    for e in summary["settings"].values():
        totals.update(e["retained_totals_primary_budget"])
        chosen.update(e["selected_component_totals_primary_budget"])
    print(f"{'category':20s} {'retained':>9s} {'in the returned component':>26s}")
    for category in p.SPLIT_CATEGORIES:
        print(f"{category:20s} {totals.get(category,0):9d} {chosen.get(category,0):26d}")
    print(f"\nwritten to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
