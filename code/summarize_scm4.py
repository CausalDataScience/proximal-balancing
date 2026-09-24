"""Summaries for SCM-4 with the sealed family-v2 rules (threshold rule, replayed aggregation, section 10 gates).

    python3 -B summarize_scm4.py dev        choose t by the sealed rule on the development shards and replay
    python3 -B summarize_scm4.py confirm    score the confirmation shards at the frozen threshold

The threshold rule, the replay, the mechanism rates and the paired one-sided bound are the functions of
summarize_family_v2.py, imported and not copied.  Two things are added for the image level: the pixel CNN joins
the comparators and the gates, and a data set whose PROBE record is incomplete, or whose comparator is not
finite, fails completeness instead of being dropped.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone

import numpy as np

import run_scm4 as rs
import summarize_family_v2 as sm


def load(stage: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(rs.SHARDS[stage].glob("*.json"))]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def comparators(shards: list[dict]) -> dict:
    table = sm.comparator_table(shards)
    values = [sh["pixel_cnn"].get("ate") for sh in shards]
    finite = [v for v in values if v is not None and np.isfinite(v)]
    table["pixel_cnn"] = {"values": values, "mean": float(np.mean(finite)) if finite else None,
                          "mae": float(np.mean(np.abs(np.array(finite) - 1.0))) if finite else None,
                          "finite": len(finite)}
    return table


def reading_totals(shards: list[dict]) -> dict:
    keys = ("n", "nonfinite", "adjacent", "far")
    return {k: int(sum(v[k] for sh in shards for v in sh["reading"].values())) for k in keys}


def dev() -> int:
    out_path = rs.RESULTS / "dev_summary_v1.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; summaries are write-once")
    proto = json.loads(rs.PROTOCOLS["dev"].read_text())
    shards = sorted(load("dev"), key=lambda s: s["task"]["seed"])
    if len(shards) != len(proto["tasks"]):
        raise SystemExit(f"{len(shards)} development shards, {len(proto['tasks'])} expected; nothing is summarised early")
    problems = {sh["task"]["seed"]: sh["probe_problems"] for sh in shards if sh["probe_problems"]}
    thr = sm.choose_threshold(shards)
    reps = []
    for sh in shards:
        rp = sm.replay(sh, thr["t"])
        rp.update(sm.mechanism(rp))
        rp |= {"seed": sh["task"]["seed"], "timing_s": sh["timing_s"],
               "error": None if rp["estimate"] is None else rp["estimate"] - 1.0}
        reps.append(rp)
    table = comparators(shards)
    returned = [r for r in reps if r["return_kind"] == "unique_largest"]
    errs = [abs(r["error"]) for r in returned]
    entry = {"complete": not problems, "probe_problems": problems, "returned": len(returned),
             "dev_mae": float(np.mean(errs)) if errs else None,
             "beats_raw": bool(errs) and float(np.mean(errs)) < table["raw"]["mae"],
             "beats_X": bool(errs) and float(np.mean(errs)) < table["X"]["mae"],
             "beats_pixel_cnn": bool(errs) and table["pixel_cnn"]["mae"] is not None
                                and float(np.mean(errs)) < table["pixel_cnn"]["mae"],
             "threshold_separates": not thr["overlap_fail"]}
    summary = {"artifact": "family_v2_scm4_dev_summary", "generated_utc": utc(), "rule": "family v2 PRD section 5",
               "threshold": thr, "replay": reps, "comparators": table, "reading": reading_totals(shards),
               "entry_check": entry}
    print(f"t = {thr['t']:.2e} (a {thr['a_p95_balanceable']:.2e}, b {thr['b_p05_other']:.2e}, overlap fail "
          f"{thr['overlap_fail']})")
    for rp in reps:
        print(f"  seed {rp['seed']}: {rp['return_kind']} estimate {rp['estimate']} | sizes {rp['component_sizes']} | "
              f"TPR {rp['tpr']:.2f} FPR {rp['fpr']:.3f} purity {rp['purity']} | {rp['timing_s'].get('total')} s")
    for k, v in table.items():
        print(f"  {k:28s} mean {v['mean']}  MAE {v['mae']}")
    print("reading:", summary["reading"], "| entry check:", entry)
    out_path.write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    print(f"written {out_path}")
    return 0


def confirm() -> int:
    out_path = rs.RESULTS / "confirm_summary_v1.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; summaries are write-once")
    proto = json.loads(rs.PROTOCOLS["confirm"].read_text())
    t = proto["screen_thresholds"][rs.LEVEL_NAME]
    expected = len(proto["tasks"])
    shards = sorted(load("confirm"), key=lambda s: s["task"]["seed"])
    reps = []
    for sh in shards:
        rp = sm.replay(sh, t)
        rp.update(sm.mechanism(rp))
        rp["seed"] = sh["task"]["seed"]
        reps.append(rp)
    returned = [r for r in reps if r["return_kind"] == "unique_largest"]
    err = np.array([abs(r["estimate"] - 1.0) for r in returned]) if returned else np.array([np.nan])
    pixel = [sh["pixel_cnn"].get("ate") for sh in shards]
    pixel_ok = all(v is not None and np.isfinite(v) for v in pixel)
    records_ok = all(not sh["probe_problems"] or all(p.startswith(("return kind", "no finite estimate"))
                                                     for p in sh["probe_problems"]) for sh in shards)
    paired = {}
    if len(returned) == len(reps) == expected:
        probe_abs = np.array([abs(r["estimate"] - 1.0) for r in reps])
        paired["oracle_excess_upper"] = sm._t_upper(probe_abs - np.array([abs(sh["baselines"]["oracle"] - 1.0) for sh in shards]))
        for key in ("X", "raw", "summary"):
            paired[f"minus_{key}_upper"] = sm._t_upper(probe_abs - np.array([abs(sh["baselines"][key] - 1.0) for sh in shards]))
        if pixel_ok:
            paired["minus_pixel_cnn_upper"] = sm._t_upper(probe_abs - np.abs(np.array(pixel, dtype=float) - 1.0))
    tpr = float(np.mean([r["tpr"] for r in reps])) if reps else None
    fpr = float(np.mean([r["fpr"] for r in reps])) if reps else None
    purity = [r["purity"] for r in reps if r["purity"] is not None]
    gates = {"complete": len(reps) == expected and records_ok and pixel_ok,
             "return_all": len(returned) == expected,
             "mae": bool(np.nanmean(err) <= 0.05),
             "p90": bool(np.nanpercentile(err, 90) <= 0.10),
             "oracle_excess": paired.get("oracle_excess_upper", math.inf) <= 0.05,
             "beats_X": paired.get("minus_X_upper", math.inf) < 0,
             "beats_raw": paired.get("minus_raw_upper", math.inf) < 0,
             "beats_summary": paired.get("minus_summary_upper", math.inf) < 0,
             "beats_pixel_cnn": paired.get("minus_pixel_cnn_upper", math.inf) < 0,
             "screen_tpr": tpr is not None and tpr >= 0.8,
             "screen_fpr": fpr is not None and fpr <= 0.2,
             "unique_largest": len(returned) / max(expected, 1) >= 0.9,
             "purity": bool(purity) and float(np.mean(purity)) >= 0.8}
    summary = {"artifact": "family_v2_scm4_confirm_summary", "generated_utc": utc(), "threshold": t,
               "expected": expected, "replay": reps, "mae": float(np.nanmean(err)),
               "median_abs_error": float(np.nanmedian(err)), "p90": float(np.nanpercentile(err, 90)),
               "paired_upper": paired, "tpr": tpr, "fpr": fpr, "purity_mean": float(np.mean(purity)) if purity else None,
               "comparators": comparators(shards), "reading": reading_totals(shards), "gates": gates,
               "verdict": "PASS" if all(gates.values()) else "FAIL"}
    print(f"t {t:.2e} | returned {len(returned)}/{expected} | MAE {summary['mae']:.4f} p90 {summary['p90']:.4f} | "
          f"TPR {tpr} FPR {fpr} | paired {paired} | verdict {summary['verdict']} "
          f"{[k for k, v in gates.items() if not v]}")
    out_path.write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    print(f"written {out_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("dev", "confirm"):
        raise SystemExit("usage: summarize_scm4.py dev|confirm")
    raise SystemExit(dev() if sys.argv[1] == "dev" else confirm())
