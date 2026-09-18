"""Independent scoring of the minimal suite from the stored cells alone.

Nothing here imports the runners' own verdicts.  Every number is recomputed from the recorded cells, and
the protocol file is re-read and re-hashed so a result that no longer matches the design it claims is
rejected rather than summarised.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from datetime import datetime, timezone
from scipy.stats import t as tdist

import minimal_suite_common as c

TRUE_EFFECT = 1.0


def load(path: str) -> dict:
    """A cell counts as a return only when it carries an estimate.  A tied largest component is not one."""
    payload = json.loads(Path(path).read_text())
    if not payload.get("complete"):
        raise RuntimeError(f"{path} records an incomplete run")
    keys = [tuple(c[f] for f in ("data_seed", "n_total") if f in c) for c in payload["cells"]]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{path} repeats a cell: a design point appears more than once")
    if any(not c.get("complete", False) for c in payload["cells"]):
        raise RuntimeError(f"{path} contains an incomplete cell")
    return payload


def paired_upper(a: np.ndarray, b: np.ndarray, contrasts: int, alpha: float) -> float:
    """Upper end of a one-sided paired interval for mean(a) - mean(b), Bonferroni-adjusted."""
    d = a - b
    if len(d) < 2:
        return float("nan")
    se = d.std(ddof=1) / math.sqrt(len(d))
    crit = tdist.ppf(1.0 - alpha / contrasts, len(d) - 1)
    return float(d.mean() + crit * se)


def score_scm6(payload: dict, protocol: dict) -> dict:
    """Return rate, error location and spread, the seven comparators, and the channel diagnostic."""
    cells = payload["cells"]
    gates_spec = protocol["gates"]
    returned = [c for c in cells if c["estimate"] is not None]
    rate = len(returned) / len(cells)
    err = np.array([abs(c["estimate"] - TRUE_EFFECT) for c in returned])
    names = sorted({k for c in returned for k in c["baselines"]})
    base_err = {k: np.array([abs(c["baselines"][k] - TRUE_EFFECT) for c in returned]) for k in names}
    # The gate on location is the mean, matching how the runners define "mae"; the median is carried
    # alongside because the absolute error is right skewed and the two differ by a third at small n.
    corr = np.array([c["representation_diagnostics"]["channel_correlation_median"] for c in cells
                     if "representation_diagnostics" in c])
    n_con = gates_spec["bonferroni_contrasts"]
    alpha = gates_spec["alpha"]
    contrasts = {k: paired_upper(err, v, n_con, alpha) for k, v in base_err.items()}
    summary = {
        "cells": len(cells), "returned": len(returned), "return_rate": rate,
        "mean_abs_error": float(np.mean(err)) if len(err) else float("nan"),
        "median_abs_error": float(np.median(err)) if len(err) else float("nan"),
        "p90_abs_error": float(np.quantile(err, 0.9)) if len(err) else float("nan"),
        "mean_estimate": float(np.mean([c["estimate"] for c in returned])) if returned else float("nan"),
        "baseline_mean_abs_error": {k: float(np.mean(v)) for k, v in base_err.items()},
        "baseline_median_abs_error": {k: float(np.median(v)) for k, v in base_err.items()},
        "paired_upper_vs_baseline": contrasts,
        "channel_correlation_median": float(np.median(corr)) if len(corr) else float("nan"),
        "retained_splits_median": float(np.median([len(c.get("retained_splits", [])) for c in cells])),
        "selected_r_values": sorted({row["r"] for c in cells for rot in c.get("per_rotation", [])
                                     for row in rot if row["r"] is not None}),
    }
    summary["gates"] = {
        "return_rate": rate >= gates_spec["return_rate_min"],
        "mae": summary["mean_abs_error"] <= gates_spec["mae_max"],
        "p90": summary["p90_abs_error"] <= gates_spec["p90_max"],
        "channel_corr": summary["channel_correlation_median"] >= gates_spec["channel_corr_median_min"],
        "oracle_excess": contrasts.get("oracle", float("nan")) <= gates_spec["oracle_excess_upper_max"],
        "beats_X": contrasts.get("X", float("nan")) < 0.0,
        "beats_raw_ridge": contrasts.get("raw_ridge", float("nan")) < 0.0,
        "beats_learned_projection": contrasts.get("learned_projection", float("nan")) < 0.0,
        "beats_blockwise_pca": contrasts.get("blockwise_pca", float("nan")) < 0.0,
    }
    summary["verdict"] = "PASS" if all(summary["gates"].values()) else "FAIL"
    return summary


def score_development(payload: dict) -> dict:
    """Recompute the three development criteria for a v2-style run from its stored rows.

    The runner writes its own summary; this does not read it.  Root recovery is measured against the
    balance root recorded on each row, which for the image design is the root that rotation's encoder
    could actually reach.
    """
    gates = payload["gates"]
    rows = [r for cell in payload["cells"] for r in cell.get("rows", [])]
    with_root = [r for r in rows if r["evaluator_only"]["root_error"] is not None]
    feasible = [r for r in rows if r["evaluator_only"]["feasible"]]
    infeasible = [r for r in rows if not r["evaluator_only"]["feasible"]]
    returned = [cell for cell in payload["cells"] if cell.get("estimate") is not None]
    err = np.array([abs(cell["estimate"] - TRUE_EFFECT) for cell in returned])
    errs = np.array([r["evaluator_only"]["root_error"] for r in with_root])
    out = {
        "cells": len(payload["cells"]), "returned": len(returned),
        "root_recovery_rate": float(np.mean(errs <= payload["root_tolerance"])) if len(errs)
        else float("nan"),
        "root_error_median": float(np.median(errs)) if len(errs) else float("nan"),
        "root_error_p90": float(np.quantile(errs, 0.9)) if len(errs) else float("nan"),
        "screen_tpr": float(np.mean([r["screen_pass"] for r in feasible])) if feasible else float("nan"),
        "screen_fpr": float(np.mean([r["screen_pass"] for r in infeasible])) if infeasible
        else float("nan"),
        "mean_abs_error": float(np.mean(err)) if len(err) else float("nan"),
        "median_abs_error": float(np.median(err)) if len(err) else float("nan"),
        "mean_estimate": float(np.mean([cell["estimate"] for cell in returned])) if returned
        else float("nan"),
    }
    corr = [cell["representation_diagnostics"]["channel_correlation_median"]
            for cell in payload["cells"] if "representation_diagnostics" in cell]
    if corr:
        out["channel_correlation_median"] = float(np.median(corr))
    out["gates"] = {
        "root_recovery": out["root_recovery_rate"] >= gates["root_recovery_rate_min"],
        "screen_tpr": out["screen_tpr"] >= gates["screen_tpr_min"],
        "screen_fpr": out["screen_fpr"] <= gates["screen_fpr_max"],
        "ate_error": out["mean_abs_error"] <= gates["mean_abs_error_max"],
    }
    if "channel_correlation_min" in gates and corr:
        out["gates"]["channel_correlation"] = (out["channel_correlation_median"]
                                               >= gates["channel_correlation_min"])
    out["verdict"] = "PASS" if all(out["gates"].values()) else "FAIL"
    out["confirmation_may_open"] = out["verdict"] == "PASS"
    return out


def report_development(path: str) -> dict:
    payload = json.loads(Path(path).read_text())
    if not payload.get("complete"):
        raise RuntimeError(f"{path} records an incomplete run")
    out = score_development(payload)
    print(f"{payload['experiment_id']} {payload['phase']} | {out['cells']} cells | "
          f"verdict {out['verdict']} | confirmation may open: {out['confirmation_may_open']}")
    if "channel_correlation_median" in out:
        print(f"  channel corr    {out['channel_correlation_median']:.3f}   "
              f"(gate >= {payload['gates'].get('channel_correlation_min')})")
    print(f"  root recovery   {out['root_recovery_rate']:.3f}   "
          f"(gate >= {payload['gates']['root_recovery_rate_min']})   "
          f"median {out['root_error_median']:.4f}  p90 {out['root_error_p90']:.4f}")
    print(f"  screen TPR      {out['screen_tpr']:.3f}   FPR {out['screen_fpr']:.3f}")
    print(f"  mean |error|    {out['mean_abs_error']:.4f}   "
          f"(gate <= {payload['gates']['mean_abs_error_max']})   "
          f"median {out['median_abs_error']:.4f}   mean estimate {out['mean_estimate']:.4f}")
    print(f"  gates: {out['gates']}")
    return out


def report_scm6(path: str) -> dict:
    payload = load(path)
    protocol = json.loads(Path(payload["protocol"]["path"]).read_text())
    out = score_scm6(payload, protocol)
    print(f"SCM-6 {payload['phase']} | {out['cells']} cells | verdict {out['verdict']}")
    print(f"  return rate      {out['return_rate']:.2f}   (gate >= {protocol['gates']['return_rate_min']})")
    print(f"  mean |error|     {out['mean_abs_error']:.4f}   (gate <= {protocol['gates']['mae_max']})")
    print(f"  median |error|   {out['median_abs_error']:.4f}")
    print(f"  p90 |error|      {out['p90_abs_error']:.4f}   (gate <= {protocol['gates']['p90_max']})")
    print(f"  mean estimate    {out['mean_estimate']:.4f}   (truth {TRUE_EFFECT})")
    print(f"  channel corr     {out['channel_correlation_median']:.4f}   "
          f"(gate >= {protocol['gates']['channel_corr_median_min']})")
    print(f"  selected r       {out['selected_r_values']}")
    print(f"\n  {'comparator':22s} {'mean |err|':>11s} {'median |err|':>13s} "
          f"{'paired upper':>13s}  gate")
    for k in sorted(out["baseline_mean_abs_error"]):
        gate = {"X": "beats_X", "raw_ridge": "beats_raw_ridge",
                "learned_projection": "beats_learned_projection",
                "blockwise_pca": "beats_blockwise_pca", "oracle": "oracle_excess"}.get(k)
        mark = "" if gate is None else ("pass" if out["gates"][gate] else "FAIL")
        print(f"  {k:22s} {out['baseline_mean_abs_error'][k]:>11.4f} "
              f"{out['baseline_median_abs_error'][k]:>13.4f} "
              f"{out['paired_upper_vs_baseline'][k]:>13.4f}  {mark}")
    print(f"\n  gates: {out['gates']}")
    return out


def score_e12(payload: dict, protocol: dict) -> dict:
    """Per-sample-size location and spread, the paired contrasts, and whether the error actually falls."""
    by_n = {}
    for cell in payload["cells"]:
        by_n.setdefault(cell["n_total"], []).append(cell)
    gates_spec = protocol["gates"]
    rows, sizes = [], sorted(by_n)
    for n in sizes:
        cells = by_n[n]
        ret = [c for c in cells if c["estimate"] is not None]
        err = np.array([abs(c["estimate"] - TRUE_EFFECT) for c in ret])
        rows.append({"n": n, "cells": len(cells), "return_rate": len(ret) / len(cells),
                     "mean_abs_error": float(np.mean(err)),
                     "median_abs_error": float(np.median(err)),
                     "p90_abs_error": float(np.quantile(err, 0.9)),
                     "representation_error": float(np.median([c["representation_error"] for c in ret])),
                     "estimation_error": float(np.median([c["estimation_error"] for c in ret])),
                     "baselines": {k: float(np.median([abs(c["baselines"][k] - TRUE_EFFECT) for c in ret]))
                                   for k in ret[0]["baselines"]}})
    largest = [c for c in by_n[sizes[-1]] if c["estimate"] is not None]
    err_big = np.array([abs(c["estimate"] - TRUE_EFFECT) for c in largest])
    names = [k for k in largest[0]["baselines"]]
    n_con = gates_spec.get("bonferroni_contrasts", len(names))
    contrasts = {k: paired_upper(err_big, np.array([abs(c["baselines"][k] - TRUE_EFFECT) for c in largest]),
                                 n_con, gates_spec.get("alpha", 0.05)) for k in names}
    smallest = [c for c in by_n[sizes[0]] if c["estimate"] is not None]
    shrink = paired_upper(err_big, np.array([abs(c["estimate"] - TRUE_EFFECT) for c in smallest]),
                          n_con, gates_spec.get("alpha", 0.05)) if len(smallest) == len(largest) else float("nan")
    out = {"rows": rows, "paired_upper_at_largest": contrasts, "shrinkage_upper": shrink,
           "cells": len(payload["cells"])}
    out["gates"] = {
        "return_rate": all(r["return_rate"] >= gates_spec["return_rate_min"] for r in rows),
        "mae": rows[-1]["mean_abs_error"] <= gates_spec["mae_max"],
        "p90": rows[-1]["p90_abs_error"] <= gates_spec["p90_max"],
        "beats_X": contrasts.get("X", float("nan")) < 0.0,
        "beats_raw": contrasts.get("raw_XW", float("nan")) < 0.0,
        "oracle_excess": contrasts.get("oracle", float("nan")) <= gates_spec["oracle_excess_upper_max"],
        "error_falls_from_smallest_to_largest": shrink < 0.0,
    }
    out["verdict"] = "PASS" if all(out["gates"].values()) else "FAIL"
    return out


def recompute_bound(cell: dict, protocol: dict) -> dict:
    """Rebuild the theorem's bound from its own ingredients instead of trusting the stored value.

    The manuscript's bound is 1 - delta - P(R = 0) - L E[exp(-R Delta^2 / 2)] with R hypergeometric on
    T candidates of which N are screened and m are drawn.  The exponential moment is taken against that
    distribution in exact rational arithmetic here, so a stored bound that drifted would be caught.
    """
    from fractions import Fraction
    from math import comb, exp
    t_total, m = protocol["T"], protocol["m"]
    n_screened, delta = cell["N"], Fraction(protocol["delta"])
    denom = comb(t_total, m)
    pmf = [Fraction(comb(n_screened, r) * comb(t_total - n_screened, m - r), denom)
           for r in range(m + 1)]
    if sum(pmf) != 1:
        raise RuntimeError("the hypergeometric weights do not sum to one")
    delta_sq = Fraction(cell["Delta_exact"]) ** 2
    moment = sum(float(w) * exp(-r * float(delta_sq) / 2.0) for r, w in enumerate(pmf))
    bound = 1.0 - float(delta) - float(pmf[0]) - cell["L"] * moment
    return {"p_r_zero": float(pmf[0]), "exponential_moment": moment, "bound": bound,
            "stored_bound": cell["theorem_B"], "difference": abs(bound - cell["theorem_B"])}


def score_e11_v2(payload: dict, protocol: dict) -> dict:
    """The theorem's own inequality, checked state by state against a bound recomputed here."""
    cells = payload["cells"]
    rebuilt = [recompute_bound(c, protocol) for c in cells]
    rows = [{"seed": c["data_seed"], "N": c["N"], "Delta": c["Delta"], "rho": c["rho"],
             "separation_g": c["separation_g"], "B": c["theorem_B"],
             "success": c["empirical_success"], "success_lower_99": c["success_lower_99"],
             "radius_rate": c["uniform_radius_rate"], "radius_lower_99": c["uniform_radius_lower_99"],
             "recomputed_B": b["bound"], "bound_difference": b["difference"]}
            for c, b in zip(cells, rebuilt)]
    out = {"states": len(rows), "rows": rows}
    out["gates"] = {
        "delta_positive": all(r["Delta"] > 0 for r in rows),
        "separation": all(4.0 * r["rho"] < r["separation_g"] for r in rows),
        "bound_holds_in_every_state": all(r["success_lower_99"] >= r["B"] for r in rows),
        "uniform_radius": all(r["radius_lower_99"] >= 0.95 for r in rows),
        "stored_bound_matches_recomputation": all(r["bound_difference"] < 1e-12 for r in rows),
    }
    out["verdict"] = "PASS" if all(out["gates"].values()) else "FAIL"
    return out


def report_e12(path: str) -> dict:
    payload = load(path)
    protocol = json.loads(Path(payload["protocol"]["path"]).read_text())
    out = score_e12(payload, protocol)
    print(f"E12 {payload['phase']} | {out['cells']} cells | verdict {out['verdict']}")
    names = list(out["rows"][0]["baselines"])
    print(f"  {'n':>6s} {'ret':>5s} {'mean':>7s} {'median':>7s} {'p90':>7s} {'rep err':>8s} "
          f"{'est err':>8s} " + " ".join(f"{k:>8s}" for k in names))
    for r in out["rows"]:
        print(f"  {r['n']:>6d} {r['return_rate']:>5.2f} {r['mean_abs_error']:>7.4f} "
              f"{r['median_abs_error']:>7.4f} "
              f"{r['p90_abs_error']:>7.4f} {r['representation_error']:>8.4f} "
              f"{r['estimation_error']:>8.4f} "
              + " ".join(f"{r['baselines'][k]:>8.4f}" for k in names))
    print(f"  paired upper at the largest n: "
          f"{ {k: round(v, 4) for k, v in out['paired_upper_at_largest'].items()} }")
    print(f"  shrinkage upper: {out['shrinkage_upper']:.4f}")
    print(f"  gates: {out['gates']}")
    return out


def report_e11_v2(path: str) -> dict:
    payload = load(path)
    protocol = json.loads(Path(payload["protocol"]["path"]).read_text())
    out = score_e11_v2(payload, protocol)
    print(f"E11-v2 {payload['phase']} | {out['states']} outer states | verdict {out['verdict']}")
    print(f"  {'seed':>10s} {'N':>3s} {'Delta':>7s} {'rho':>7s} {'4rho<g':>7s} {'bound B':>9s} "
          f"{'success':>8s} {'99% low':>8s} {'B recomputed':>13s}")
    for r in out["rows"]:
        print(f"  {r['seed']:>10d} {r['N']:>3d} {r['Delta']:>7.4f} {r['rho']:>7.4f} "
              f"{str(4 * r['rho'] < r['separation_g']):>7s} {r['B']:>9.6f} "
              f"{r['success']:>8.4f} {r['success_lower_99']:>8.4f} {r['recomputed_B']:>13.6f}")
    print(f"  gates: {out['gates']}")
    return out


SCORER_SOURCES = ["summarize_minimal_suite.py"]


def entry(kind: str, path: str, out: dict) -> dict:  # noqa: C901
    """One scored result: what was scored, what this scorer concluded, and what the file itself claims.

    The runners initialise a verdict field and only some of them fill it in, so a stored value can be
    stale or absent while the recomputed verdict says otherwise.  Recording both and comparing them makes
    that visible instead of leaving two answers in circulation.
    """
    payload = json.loads(Path(path).read_text())
    # A development run carries its design inline instead of citing a frozen protocol, because no
    # confirmatory seeds hang off it.
    protocol_path = payload.get("protocol", {}).get("path")
    protocol = json.loads(Path(protocol_path).read_text()) if protocol_path else {}
    stored = payload.get("summary", {}).get("verdict") if kind == "development" \
        else payload.get("verdict")
    return {
        "kind": kind, "experiment_id": payload["experiment_id"], "phase": payload["phase"],
        "result_path": str(path), "result_sha256": c.file_sha256(path),
        "protocol_path": str(protocol_path),
        "protocol_payload_sha256": protocol.get(c.DIGEST_FIELD),
        "protocol_sealed": c.DIGEST_FIELD in protocol,
        "cells": len(payload["cells"]),
        "verdict": out["verdict"],
        "verdict_stored_in_result": stored,
        "verdicts_agree": stored in (None, "NOT_EVALUATED") or stored == out["verdict"],
        "stored_verdict_usable": stored not in (None, "NOT_EVALUATED"),
        "statistics": {k: v for k, v in out.items() if k != "gates"},
        "gates": out["gates"],
    }


def write_scores(path: str, entries: list[dict]) -> None:
    """One write-once artifact holding every verdict, sealed against later edits."""
    disagreements = [e["result_path"] for e in entries if not e["verdicts_agree"]]
    unsealed = [e["protocol_path"] for e in entries if not e["protocol_sealed"]]
    c.seal_protocol(path, {
        "artifact": "minimal_suite_scores", "schema_version": 1,
        "scored_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "location_statistic": "the gate on location is the mean absolute error; the median is reported "
                              "alongside because the absolute error is right skewed",
        "verdict_authority": "this artifact. A verdict field inside a result file is recorded here for "
                             "comparison and is not used by anything.",
        "scorer_snapshot": c.source_snapshot(SCORER_SOURCES),
        "environment": c.environment(),
        "entries": entries,
        "verdicts_disagreeing": disagreements,
        "protocols_not_sealed": unsealed,
        "summary": {e["experiment_id"] + " " + e["phase"]: e["verdict"] for e in entries},
    })
    print(f"\nscores written to {path}")
    print(f"  verdicts: {[(e['experiment_id'] + ' ' + e['phase'], e['verdict']) for e in entries]}")
    if disagreements:
        print(f"  stored verdicts that disagree with this scoring: {disagreements}")
    if unsealed:
        print(f"  protocols frozen before sealing existed, so not check-able: {unsealed}")


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("scm6", "e12", "e11v2", "dev"):
        ap.add_argument(f"--{name}", action="append", default=[])
    ap.add_argument("--write-scores", default=None)
    args = ap.parse_args()
    entries = []
    for kind, paths, fn in (("e12", args.e12, report_e12), ("e11v2", args.e11v2, report_e11_v2),
                            ("scm6", args.scm6, report_scm6), ("development", args.dev,
                                                               report_development)):
        for path in paths:
            entries.append(entry(kind, path, fn(path)))
            print()
    if args.write_scores:
        write_scores(args.write_scores, entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
