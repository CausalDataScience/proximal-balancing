"""Summaries for SCM family v2 (PRD section 5 threshold rule, replayed aggregation, comparator tables).

    python3 summarize_family_v2.py dev        choose t per level by the PRD rule on development shards,
                                              replay Algorithm 1 at that t, write dev_summary_v1.json
    python3 summarize_family_v2.py confirm    score the confirmation shards at the frozen thresholds and the
                                              PRD section 10 gates, write confirm_summary_v1.json

The replay uses only what the shards stored: every split's rotation-level screen bounds and estimates and the
cross-product of the E-fold AIPW scores, so the radius is the sealed common_radius rule applied to the
retained columns.  Split labels (target, W4, other) come from the population census and are evaluator-only.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import probe_structured_scm as ps

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "family_v2"
TARGETS = ("S1", "S2", "S3", "S12", "S13", "S23", "S123")
WRONG = ("S4",)
BALANCEABLE = TARGETS + WRONG
T_CAP = 2.0e-3
MAIN = ("SCM-1", "SCM-2", "SCM-3")


def round2(x: float) -> float:
    if x <= 0:
        return x
    e = math.floor(math.log10(x))
    return round(x / 10 ** e, 1) * 10 ** e


def load(stage: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((RESULTS / stage).glob("*.json"))]


def split_max_upper(rec: dict) -> tuple[float, bool]:
    rots = rec["rotations"]
    return max(r["screen"]["upper"] for r in rots), all(r["screen"]["overlap"] for r in rots)


def choose_threshold(shards: list[dict]) -> dict:
    bal, oth = [], []
    for sh in shards:
        for name, rec in sh["probe"]["records"].items():
            m, _ = split_max_upper(rec)
            (bal if name in BALANCEABLE else oth).append(m)
    a = float(np.percentile(bal, 95))
    b = float(np.percentile(oth, 5))
    t_raw = math.sqrt(a * b)
    t = min(round2(t_raw), T_CAP)
    return {"a_p95_balanceable": a, "b_p05_other": b, "t_raw": t_raw, "t": t, "overlap_fail": a >= b,
            "n_balanceable": len(bal), "n_other": len(oth),
            "max_balanceable": float(max(bal)), "min_other": float(min(oth))}


def replay(shard: dict, t: float) -> dict:
    pr_ = shard["probe"]
    names = pr_["split_names"]
    recs = pr_["records"]
    retained = []
    for i, name in enumerate(names):
        m, ov = split_max_upper(recs[name])
        if m < t and ov:
            retained.append(i)
    n = pr_["n"]
    fold_seed = shard["task"]["seed"] + 7
    if not retained:
        return {"return_kind": "no_return", "estimate": None, "retained": [], "rho": None, "component_sizes": [],
                "members": []}
    cross = np.array(pr_["score_crossproduct"])
    cov = cross[np.ix_(retained, retained)] / (n * n)
    draws = np.random.default_rng(fold_seed + 5).multivariate_normal(np.zeros(len(retained)), cov,
                                                                     size=ps.BOOTSTRAP_DRAWS)
    rho = float(np.quantile(np.max(np.abs(draws), axis=1), 0.95))
    splits = [tuple(int(c) - 1 for c in names[i][1:]) for i in retained]
    est = np.array([recs[names[i]]["estimate"] for i in retained])
    agg = ps.aggregate(splits, est, rho)
    members = ["S" + "".join(str(j + 1) for j in s) for s in (agg.get("members") or [[]])[0]] if agg["return_kind"] == "unique_largest" else []
    return {"return_kind": agg["return_kind"], "estimate": agg["output"], "retained": [names[i] for i in retained],
            "rho": rho, "component_sizes": agg.get("component_sizes", []), "members": members}


def mechanism(rep: dict) -> dict:
    ret = set(rep["retained"])
    tpr = len(ret & set(TARGETS)) / len(TARGETS)
    fpr = len(ret - set(TARGETS)) / (30 - len(TARGETS))
    purity = (len(set(rep["members"]) & set(TARGETS)) / len(rep["members"])) if rep["members"] else None
    return {"tpr": tpr, "fpr": fpr, "purity": purity}


def learned_diagnostics(shards: list[dict]) -> dict:
    dev_t, dev_all = [], []
    for sh in shards:
        for name, rec in sh["probe"]["records"].items():
            for r in rec["rotations"]:
                if name in TARGETS:
                    dev_t.append(abs(r["learned_tau"] - 1.0))
    return {"target_rotation_learned_tau_abs_error_median": float(np.median(dev_t)) if dev_t else None,
            "target_rotation_learned_tau_abs_error_p90": float(np.percentile(dev_t, 90)) if dev_t else None,
            "n": len(dev_t)}


COMPARATORS = ("naive", "X", "raw", "summary", "oracle")
PROXIMAL = ("p2sls_given_roles", "p2sls_swapped_roles", "park_spc_clean_proxy", "park_spc_contaminated_proxy")


def comparator_table(shards: list[dict]) -> dict:
    rows = {}
    for key in COMPARATORS:
        vals = [sh["baselines"][key] for sh in shards]
        rows[key] = vals
    for key in PROXIMAL:
        vals = [sh["proximal"][key] for sh in shards if "proximal" in sh]
        if vals:
            rows[key] = vals
    rows["cevae"] = [sh["cevae"]["ate"] for sh in shards]
    return {k: {"values": v, "mean": float(np.mean(v)), "mae": float(np.mean(np.abs(np.array(v) - 1.0)))}
            for k, v in rows.items()}


def dev() -> int:
    out_path = RESULTS / "dev_summary_v1.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; summaries are write-once")
    shards = load("dev")
    by_level: dict[str, list[dict]] = {}
    for sh in shards:
        by_level.setdefault(sh["task"]["level"], []).append(sh)
    summary = {"artifact": "family_v2_dev_summary", "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
               "rule": "PRD section 5: per data set and split, m = max over rotations of the screen upper bound; "
                       "a = 95th percentile of m over the 8 balanceable splits, b = 5th percentile over the other 22; "
                       "t = min(round to two significant figures of sqrt(ab), 2.0e-3); development failure when a >= b",
               "levels": {}}
    for level in MAIN:
        shs = by_level.get(level, [])
        if not shs:
            continue
        thr = choose_threshold(shs)
        reps = []
        for sh in shs:
            rp = replay(sh, thr["t"])
            rp.update(mechanism(rp))
            rp["seed"] = sh["task"]["seed"]
            rp["error"] = None if rp["estimate"] is None else rp["estimate"] - 1.0
            rp["timing_s"] = sh["timing_s"]
            reps.append(rp)
        summary["levels"][level] = {"threshold": thr, "replay": reps, "learned": learned_diagnostics(shs),
                                    "comparators": comparator_table(shs)}
        print(f"\n{level}: t = {thr['t']:.2e} (a {thr['a_p95_balanceable']:.2e}, b {thr['b_p05_other']:.2e}, "
              f"max balanceable {thr['max_balanceable']:.2e}, min other {thr['min_other']:.2e}, overlap fail {thr['overlap_fail']})")
        for rp in reps:
            print(f"  seed {rp['seed']}: {rp['return_kind']} estimate {rp['estimate']} | retained {rp['retained']} | "
                  f"sizes {rp['component_sizes']} | TPR {rp['tpr']:.2f} FPR {rp['fpr']:.3f} purity {rp['purity']} | "
                  f"time {rp['timing_s'].get('total')} s (probe {rp['timing_s'].get('probe')}, cevae {rp['timing_s'].get('cevae')})")
        ld = summary["levels"][level]["learned"]
        print(f"  learned tau on target splits: median |error| {ld['target_rotation_learned_tau_abs_error_median']:.3f}, "
              f"p90 {ld['target_rotation_learned_tau_abs_error_p90']:.3f}")
        for k, v in summary["levels"][level]["comparators"].items():
            print(f"  {k:28s} mean {v['mean']:+.3f}  MAE {v['mae']:.3f}  values {[round(x, 3) for x in v['values']]}")
    if "SCM-1c" in by_level:
        shs = by_level["SCM-1c"]
        summary["scm1c"] = {"comparators": comparator_table(shs), "note": "PROBE not run: P0 found no balanceable split"}
        print("\nSCM-1c (comparators only):")
        for k, v in summary["scm1c"]["comparators"].items():
            print(f"  {k:28s} mean {v['mean']:+.3f}  MAE {v['mae']:.3f}  values {[round(x, 3) for x in v['values']]}")
    checks = [sh["cevae_exact_proxy_check"] for sh in shards if "cevae_exact_proxy_check" in sh]
    summary["cevae_exact_proxy_check"] = checks
    print(f"\nCEVAE exact-proxy check (W1 := U): {checks}")
    out_path.write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    print(f"\nwritten {out_path}")
    return 0


def _t_upper(d: np.ndarray, level: float = 0.95) -> float:
    from scipy.stats import t as student_t
    k = len(d)
    return float(d.mean() + student_t.ppf(level, k - 1) * d.std(ddof=1) / math.sqrt(k))


def confirm() -> int:
    """PRD section 10 gates on the sealed confirmation shards, at the frozen thresholds."""
    out_path = RESULTS / "confirm_summary_v1.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; summaries are write-once")
    proto = json.loads((RESULTS / "confirm_protocol_v1.json").read_text())
    thresholds = proto["screen_thresholds"]
    shards = load("confirm")
    by_level: dict[str, list[dict]] = {}
    for sh in shards:
        by_level.setdefault(sh["task"]["level"], []).append(sh)
    summary = {"artifact": "family_v2_confirm_summary",
               "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
               "protocol_sha256": __import__("hashlib").sha256((RESULTS / "confirm_protocol_v1.json").read_bytes()).hexdigest(),
               "levels": {}}
    for level in MAIN:
        shs = sorted(by_level.get(level, []), key=lambda s: s["task"]["seed"])
        expected = sum(1 for t in proto["tasks"] if t["level"] == level)
        t = thresholds[level]
        reps = []
        for sh in shs:
            rp = replay(sh, t)
            rp.update(mechanism(rp))
            rp["seed"] = sh["task"]["seed"]
            reps.append(rp)
        returned = [r for r in reps if r["return_kind"] == "unique_largest"]
        err = np.array([abs(r["estimate"] - 1.0) for r in returned]) if returned else np.array([np.nan])
        paired = {}
        if len(returned) == len(reps) == expected:
            probe_abs = np.array([abs(r["estimate"] - 1.0) for r in reps])
            oracle_abs = np.array([abs(sh["baselines"]["oracle"] - 1.0) for sh in shs])
            paired["oracle_excess_upper"] = _t_upper(probe_abs - oracle_abs)
            for key in ("X", "raw", "summary"):
                paired[f"minus_{key}_upper"] = _t_upper(probe_abs - np.array([abs(sh["baselines"][key] - 1.0) for sh in shs]))
        tpr = float(np.mean([r["tpr"] for r in reps])) if reps else None
        fpr = float(np.mean([r["fpr"] for r in reps])) if reps else None
        purity = [r["purity"] for r in reps if r["purity"] is not None]
        gates = {
            "complete": len(reps) == expected,
            "return_all": len(returned) == expected,
            "mae": bool(np.nanmean(err) <= 0.05),
            "p90": bool(np.nanpercentile(err, 90) <= 0.10),
            "oracle_excess": paired.get("oracle_excess_upper", math.inf) <= 0.05,
            "beats_X": paired.get("minus_X_upper", math.inf) < 0,
            "beats_raw": paired.get("minus_raw_upper", math.inf) < 0,
            "beats_summary": paired.get("minus_summary_upper", math.inf) < 0,
            "screen_tpr": tpr is not None and tpr >= 0.8,
            "screen_fpr": fpr is not None and fpr <= 0.2,
            "unique_largest": len(returned) / max(expected, 1) >= 0.9,
            "purity": bool(purity) and float(np.mean(purity)) >= 0.8,
        }
        summary["levels"][level] = {"threshold": t, "replay": reps, "mae": float(np.nanmean(err)),
                                    "median_abs_error": float(np.nanmedian(err)), "p90": float(np.nanpercentile(err, 90)),
                                    "paired_upper": paired, "tpr": tpr, "fpr": fpr,
                                    "purity_mean": float(np.mean(purity)) if purity else None,
                                    "learned": learned_diagnostics(shs), "comparators": comparator_table(shs),
                                    "gates": gates, "verdict": "PASS" if all(gates.values()) else "FAIL"}
        print(f"{level}: t {t:.2e} | returned {len(returned)}/{expected} | MAE {np.nanmean(err):.4f} p90 "
              f"{np.nanpercentile(err, 90):.4f} | TPR {tpr} FPR {fpr} | paired {paired} | verdict "
              f"{summary['levels'][level]['verdict']} {[k for k, v in gates.items() if not v]}")
    out_path.write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    print(f"written {out_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("dev", "confirm"):
        raise SystemExit("usage: summarize_family_v2.py dev|confirm")
    raise SystemExit(dev() if sys.argv[1] == "dev" else confirm())
