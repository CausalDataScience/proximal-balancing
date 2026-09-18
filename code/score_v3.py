"""Independent scorer for the v3 confirmation shards.  Reads shards, recomputes everything, decides.

Nothing here reads a verdict a runner wrote.  The gates are the ones frozen in the protocol the shards
cite, and every number is recomputed from the per-seed rows: the effect estimates, the paired
comparator contrasts with a familywise bound, the screen's behaviour against the exactly known feasible
set, the plurality mechanism, and the learned target error.  Root recovery is reported as a diagnostic
and decides nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

import minimal_suite_common as c

TRUE_EFFECT = 1.0


def load_shards(pattern: str) -> tuple[list[dict], dict]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no shards match {pattern}")
    cells = [json.loads(Path(p).read_text()) for p in paths]
    keys = [(q["data_seed"], q["n_total"]) for q in cells]
    if len(keys) != len(set(keys)):
        raise RuntimeError("a (seed, n) design point appears in more than one shard")
    if any(not q.get("complete") for q in cells):
        raise RuntimeError("an incomplete shard is present")
    digests = {q["protocol"][c.DIGEST_FIELD] for q in cells}
    if len(digests) != 1:
        raise RuntimeError(f"shards cite {len(digests)} different protocol digests")
    protocol = json.loads(Path(cells[0]["protocol"]["path"]).read_text())
    if c.canonical_digest(protocol) != digests.pop():
        raise RuntimeError("the protocol on disk does not match the digest the shards cite")
    return cells, protocol


def paired_upper(d: np.ndarray, contrasts: int, alpha: float) -> float:
    """Familywise one-sided upper bound on a mean paired difference, Bonferroni over the contrasts."""
    n = len(d)
    if n < 2:
        return float("nan")
    return float(d.mean() + tdist.ppf(1 - alpha / contrasts, n - 1) * d.std(ddof=1) / np.sqrt(n))


def score(cells: list[dict], protocol: dict, arm: str) -> dict:
    gates = protocol["gates"]
    perf, mech = gates["performance"], gates["mechanism"]
    expected = set(protocol["arms"][arm]["seeds"])
    seen = {q["data_seed"] for q in cells}
    if seen != expected:
        raise RuntimeError(f"{arm}: seeds present {sorted(seen)} differ from the frozen {sorted(expected)}")
    returned = [q for q in cells if q["return_kind"] == "unique_largest" and q["estimate"] is not None]
    err = np.array([abs(q["estimate"] - TRUE_EFFECT) for q in returned])
    out = {"arm": arm, "n": protocol["arms"][arm]["n"], "cells": len(cells), "returned": len(returned),
           "return_rate": len(returned) / len(cells),
           "mean_abs_error": float(err.mean()) if len(err) else float("nan"),
           "median_abs_error": float(np.median(err)) if len(err) else float("nan"),
           "p90_abs_error": float(np.quantile(err, 0.9)) if len(err) else float("nan"),
           "estimates": [q["estimate"] for q in returned]}
    names = sorted({k for q in returned for k in q["baselines"]})
    comp_err = {k: np.array([abs(q["baselines"][k] - TRUE_EFFECT) for q in returned if k in q["baselines"]])
                for k in names}
    out["comparators"] = {k: {"mean_abs_error": float(v.mean()), "median_abs_error": float(np.median(v)),
                              "n": int(len(v))} for k, v in comp_err.items()}
    no_balance = [k for k in names if k not in ("naive", "X", "oracle", "raw_ridge", "raw_cnn")]
    best_nb = min(no_balance, key=lambda k: comp_err[k].mean()) if no_balance else None
    out["best_no_balance"] = best_nb
    raw_name = "raw_ridge" if "raw_ridge" in names else ("raw_cnn" if "raw_cnn" in names else None)
    contrasts = {}
    for label, k in (("X", "X"), ("raw", raw_name), ("best_no_balance", best_nb), ("oracle", "oracle")):
        if k is None or k not in comp_err or len(comp_err[k]) != len(err):
            contrasts[label] = float("nan")
            continue
        contrasts[label] = paired_upper(err - comp_err[k], perf["bonferroni_contrasts"], perf["alpha"])
    out["paired_upper"] = contrasts
    rows = [r for q in cells for r in q.get("rows", [])]
    feas = [r for r in rows if r["evaluator_only"]["feasible"]]
    infe = [r for r in rows if not r["evaluator_only"]["feasible"]]
    unique = [q["return_kind"] == "unique_largest" for q in cells]
    purity = [q["largest_component_purity"] for q in cells if q.get("largest_component_purity") is not None]
    # Each learned target is compared with the population target of its own split.  A split whose
    # exact-root target is another value, the designed adversary S4 here, would otherwise carry an
    # error near 1.4 by construction and make the gate unpassable; its error against its own target
    # is reported separately.
    facts = {r["split_id"]: r["evaluator_only"] for q in cells for r in q.get("rows", [])}
    lt_err, lt_err_other = [], []
    for q in cells:
        for sid, theta in q.get("learned_target_by_split", {}).items():
            f = facts[sid]
            if f["targets_truth"]:
                lt_err.append(abs(theta - TRUE_EFFECT))
            elif f.get("tau_at_root") is not None:
                lt_err_other.append(abs(theta - f["tau_at_root"]))
    lt_err, lt_err_other = np.array(lt_err), np.array(lt_err_other)
    with_root = [r for r in rows if r["evaluator_only"]["root_error"] is not None]
    root_err = np.array([r["evaluator_only"]["root_error"] for r in with_root])
    out["mechanism"] = {
        "screen_tpr": float(np.mean([r["screen"]["pass"] for r in feas])) if feas else float("nan"),
        "screen_fpr": float(np.mean([r["screen"]["pass"] for r in infe])) if infe else float("nan"),
        "screen_pass_via_lifted_base_fraction":
            float(np.mean([r["screen"]["augmented_kept_lifted_base"] for r in rows if r["screen"]["pass"]]))
            if any(r["screen"]["pass"] for r in rows) else float("nan"),
        "nesting_violations": int(sum(not r["nesting"]["nested"] for r in rows)),
        "unique_largest_rate": float(np.mean(unique)),
        "component_purity_mean": float(np.mean(purity)) if purity else float("nan"),
        "component_purity_min": float(np.min(purity)) if purity else float("nan"),
        "learned_target_error_median": float(np.median(lt_err)) if len(lt_err) else float("nan"),
        "learned_target_error_p90": float(np.quantile(lt_err, 0.9)) if len(lt_err) else float("nan"),
        "learned_target_values": int(len(lt_err)),
        "learned_target_error_other_target_median": float(np.median(lt_err_other)) if len(lt_err_other)
        else float("nan"),
    }
    out["secondary"] = {
        "root_recovery_rate": float(np.mean(root_err <= 0.10)) if len(root_err) else float("nan"),
        "median_root_error": float(np.median(root_err)) if len(root_err) else float("nan"),
        "channel_correlation_median": float(np.median(
            [q["representation_diagnostics"]["channel_correlation_median"] for q in cells
             if "representation_diagnostics" in q])),
        "seconds_per_seed_median": float(np.median([q["timing_seconds"] for q in cells])),
        "peak_gpu_gb_max": max((q.get("peak_gpu_gb") or 0.0) for q in cells),
    }
    m = out["mechanism"]
    out["gates"] = {
        "return_rate": out["return_rate"] >= perf["return_rate_min"],
        "mae": out["mean_abs_error"] <= perf["mae_max"],
        "p90": out["p90_abs_error"] <= perf["p90_max"],
        "oracle_excess": contrasts["oracle"] <= perf["oracle_excess_upper_max"],
        "beats_X": contrasts["X"] < 0.0,
        "beats_raw": contrasts["raw"] < 0.0,
        "beats_best_no_balance": contrasts["best_no_balance"] < 0.0,
        "screen_tpr": m["screen_tpr"] >= mech["screen_tpr_min"],
        "screen_fpr": m["screen_fpr"] <= mech["screen_fpr_max"],
        "unique_largest": m["unique_largest_rate"] >= mech["unique_largest_rate_min"],
        "component_purity": m["component_purity_mean"] >= mech["component_purity_min"],
        "learned_target_median": m["learned_target_error_median"] <= mech["learned_target_error_median_max"],
        "learned_target_p90": m["learned_target_error_p90"] <= mech["learned_target_error_p90_max"],
    }
    out["verdict"] = "PASS" if all(out["gates"].values()) else "FAIL"
    return out


def report(out: dict) -> None:
    print(f"{out['arm']} | n={out['n']} | {out['cells']} cells | verdict {out['verdict']}")
    print(f"  return {out['return_rate']:.3f}  MAE {out['mean_abs_error']:.4f}  median {out['median_abs_error']:.4f}"
          f"  p90 {out['p90_abs_error']:.4f}")
    print(f"  {'comparator':22s} {'mean':>8s} {'median':>8s}")
    for k, v in sorted(out["comparators"].items(), key=lambda kv: kv[1]["mean_abs_error"]):
        print(f"  {k:22s} {v['mean_abs_error']:>8.4f} {v['median_abs_error']:>8.4f}")
    print(f"  paired upper bounds: { {k: round(v, 4) for k, v in out['paired_upper'].items()} }  "
          f"(best no-balance = {out['best_no_balance']})")
    m = out["mechanism"]
    print(f"  screen TPR {m['screen_tpr']:.3f} FPR {m['screen_fpr']:.3f}  "
          f"via lifted base {m['screen_pass_via_lifted_base_fraction']:.2f}  nesting violations {m['nesting_violations']}")
    print(f"  unique largest {m['unique_largest_rate']:.3f}  purity mean {m['component_purity_mean']:.3f} "
          f"min {m['component_purity_min']:.3f}")
    print(f"  learned target error median {m['learned_target_error_median']:.4f} "
          f"p90 {m['learned_target_error_p90']:.4f} over {m['learned_target_values']} values")
    s = out["secondary"]
    print(f"  secondary: root recovery {s['root_recovery_rate']:.3f}  median root err {s['median_root_error']:.4f}  "
          f"corr {s['channel_correlation_median']:.3f}  {s['seconds_per_seed_median']:.0f}s/seed")
    print(f"  gates: {out['gates']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="glob, e.g. ../results/scm6_v3/shards/confirmation_*.json")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--write", default=None, help="sealed score artifact path")
    args = ap.parse_args()
    cells, protocol = load_shards(args.shards)
    out = score(cells, protocol, args.arm)
    report(out)
    if args.write:
        c.seal_protocol(args.write, {
            "artifact": "v3_scores", "scored_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "protocol_path": cells[0]["protocol"]["path"],
            "protocol_payload_sha256": cells[0]["protocol"][c.DIGEST_FIELD],
            "shard_sha256": {p: c.file_sha256(p) for p in sorted(glob.glob(args.shards))},
            "scorer_snapshot": c.source_snapshot(["score_v3.py"]), "result": out})
        print(f"  sealed score artifact written to {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
