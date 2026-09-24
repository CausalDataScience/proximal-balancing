"""The prospective scorer for the v3 rule
(memo/2026-09-22-v3-confirmation-prd-v1.md, sections 4, 5 and 9).

It reads only the replicate identifiers named in the manifest, never a glob and never a range from an
existing summary, checks every shard against the manifest and against a schema before any number is
computed, and separates an execution failure (`incomplete`) from a statistical non-return (`no_return`).

    python3 -B v3_fresh_scorer.py <dataset>        score the frozen manifest for one data set

Two gates are declared in the pre-registration and are the only ones applied:
    (i)  at least 8 of the 10 designated replicates return
    (ii) the mean absolute error conditional on returning is within 1.5x the reference value stored in
         results/noise_scaled_v3.json for that data set

Gate (i) is a tie check, not a power check: with the threshold set at the median of the candidate noise
terms, about half the candidates sit below it by construction, so an empty retained set is rare.  The
informative gate is (ii).  The pre-registration says so and so does this file.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

import noise_scaled as ns

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
PROTOCOL_V1 = RESULTS / "v3_confirmation_protocol.json"
PROTOCOL_V2 = RESULTS / "v3_confirmation_protocol_v2.json"
REFERENCE = RESULTS / "noise_scaled_v3.json"
GATE_RETURN_MIN = 8
GATE_MAE_MULTIPLE = 1.5
EXPECTED_SPLITS = 30
EXPECTED_ROTATIONS = 4
COMPARATORS = ("naive", "X", "raw", "oracle", "cevae", "p2sls_given_roles", "p2sls_swapped_roles")
PAIRED_AGAINST = ("oracle", "X", "raw", "cevae")
# TPR and FPR need a validity ground truth for each candidate.  The synthetic recipe supplies one; the real
# clinical proxies of RHC and the planted RHC design do not, so those two report N/A (pre-registration 5).
TARGETS = ("S1", "S2", "S3", "S12", "S13", "S23", "S123")
HAS_VALIDITY_TRUTH = {"twins": True, "ihdp": True, "acic": True, "rhc_planted": False, "rhc": False}


class Rejected(Exception):
    """A shard that fails an integrity check.  Never scored, never silently skipped."""


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def check_shard(shard: dict, dataset: str, replicate: int, expected: dict) -> None:
    """Every structural and numerical precondition, before any statistic is formed."""
    if shard.get("replicate") != replicate:
        raise Rejected(f"shard says replicate {shard.get('replicate')}, manifest says {replicate}")
    if expected.get("dataset") not in (None, shard.get("dataset", expected.get("dataset"))):
        raise Rejected("dataset field disagrees with the manifest")
    if shard.get("seed") != expected["seed"]:
        raise Rejected(f"seed {shard.get('seed')} does not match the manifest seed {expected['seed']}")
    for key, value in expected.get("source_sha256", {}).items():
        got = shard.get("source_sha256", {}).get(key)
        if got != value:
            raise Rejected(f"source hash for {key} differs from the frozen protocol")
    if not _finite(shard.get("tau")):
        raise Rejected("tau is missing or not finite")
    probe = shard.get("probe") or {}
    names = probe.get("split_names")
    if not isinstance(names, list) or len(names) != EXPECTED_SPLITS or len(set(names)) != EXPECTED_SPLITS:
        raise Rejected(f"expected {EXPECTED_SPLITS} distinct split names, got {names if names is None else len(names)}")
    per_split = probe.get("per_split")
    if set(per_split or {}) != set(names):
        raise Rejected("per_split keys do not match split_names")
    for nm in names:
        rec = per_split[nm]
        if not _finite(rec.get("estimate")):
            raise Rejected(f"{nm}: estimate missing or not finite")
        screens = ns.screens_of(rec)
        if len(screens) != EXPECTED_ROTATIONS:
            raise Rejected(f"{nm}: {len(screens)} rotations, expected {EXPECTED_ROTATIONS}")
        for r, s in enumerate(screens):
            for field in ("gap", "se"):
                if not _finite(s.get(field)):
                    raise Rejected(f"{nm} rotation {r}: {field} missing or not finite")
            if float(s["se"]) < 0.0:
                raise Rejected(f"{nm} rotation {r}: negative standard error")
            if s.get("overlap") is not True:
                raise Rejected(f"{nm} rotation {r}: the overlap check did not pass")
    cross = np.array(probe.get("score_crossproduct", []), dtype=float)
    if cross.shape != (EXPECTED_SPLITS, EXPECTED_SPLITS) or not np.all(np.isfinite(cross)):
        raise Rejected(f"score cross-product has shape {cross.shape} or is not finite")
    if np.any(np.diag(cross) < 0.0):
        raise Rejected("score cross-product has a negative diagonal entry")
    for key in COMPARATORS:
        value = shard.get("naive") if key == "naive" else (
            shard.get("cevae", {}).get("ate") if key == "cevae" else
            shard.get("proximal", {}).get(key) if key.startswith(("p2sls", "park")) else
            shard.get("baselines", {}).get(key))
        if not _finite(value):
            raise Rejected(f"comparator {key} missing or not finite")


def comparator(shard: dict, key: str) -> float:
    if key == "naive":
        return float(shard["naive"])
    if key == "cevae":
        return float(shard["cevae"]["ate"])
    if key.startswith(("p2sls", "park")):
        return float(shard["proximal"][key])
    return float(shard["baselines"][key])


def score(dataset: str, manifest: dict, shard_paths: dict[int, Path]) -> dict:
    """Score exactly the manifest's replicates.  All must be complete before any gate is read."""
    ids = list(manifest["replicates"])
    rows, rejected = [], {}
    for r in ids:
        path = shard_paths[r]
        if not path.exists():
            rejected[r] = "shard missing"
            continue
        shard = json.loads(path.read_text())
        try:
            check_shard(shard, dataset, r, manifest["expected"][str(r)])
        except Rejected as exc:
            rejected[r] = str(exc)
            continue
        decision = ns.decide(shard["probe"], shard["n"], shard["seed"])
        rows.append({"replicate": r, "tau": shard["tau"], "n_retained": decision["n_retained"],
                     "retained": decision["retained"], "floor": decision["floor"],
                     "return_kind": decision["return_kind"], "output": decision["output"],
                     "error": None if decision["output"] is None else abs(decision["output"] - shard["tau"]),
                     "comparators": {k: comparator(shard, k) for k in COMPARATORS},
                     "tpr": decision["tpr"] if HAS_VALIDITY_TRUTH[dataset] else None,
                     "fpr": decision["fpr"] if HAS_VALIDITY_TRUTH[dataset] else None})
    complete = len(rows) == len(ids)
    if not complete:
        return {"dataset": dataset, "status": "incomplete", "expected": ids, "scored": [r["replicate"] for r in rows],
                "rejected": rejected, "note": "an execution failure or a manifest mismatch; no gate is read"}

    returned = [r for r in rows if r["output"] is not None]
    errs = [r["error"] for r in returned]
    reference = json.loads(REFERENCE.read_text())["real_data"][dataset]["mae"]
    limit = GATE_MAE_MULTIPLE * reference
    mae = float(np.mean(errs)) if errs else None

    paired = {}
    for key in PAIRED_AGAINST:
        both = [r for r in returned]          # comparators are finite on every scored shard by check_shard
        diffs = np.array([abs(r["output"] - r["tau"]) - abs(r["comparators"][key] - r["tau"]) for r in both])
        k = len(diffs)
        paired[key] = {"k": k, "ids": [r["replicate"] for r in both],
                       "upper": (None if k < 2 else
                                 float(diffs.mean() + tdist.ppf(0.95, k - 1) * diffs.std(ddof=1) / np.sqrt(k))),
                       "note": "one-sided upper bound on the paired difference of absolute errors, computed on "
                               "the replicates where both methods returned; not a statement about all "
                               f"{len(ids)} replicates"}

    gates = {"return_rate": {"returned": len(returned), "of": len(ids), "minimum": GATE_RETURN_MIN,
                             "pass": len(returned) >= GATE_RETURN_MIN,
                             "what_it_checks": "a tie in the aggregation, not the power of the screen"},
             "conditional_mae": {"mae": mae, "reference": reference, "limit": limit,
                                 "pass": mae is not None and mae <= limit,
                                 "what_it_checks": "whether the error conditional on returning reproduces the "
                                                   "value already observed on the earlier replicates"}}
    comparator_full = {k: {"values": [r["comparators"][k] for r in rows],
                           "mae": float(np.mean([abs(r["comparators"][k] - r["tau"]) for r in rows]))}
                       for k in COMPARATORS}
    return {"dataset": dataset, "status": "complete", "replicates": rows,
            "return_denominator": len(ids), "returned": len(returned),
            "no_return_ids": [r["replicate"] for r in rows if r["output"] is None],
            "conditional_mae": mae,
            "conditional_p90": float(np.quantile(errs, 0.9)) if errs else None,
            "comparators_on_all_replicates": comparator_full,
            "paired_upper_on_both_returned": paired,
            "tpr_fpr_available": HAS_VALIDITY_TRUTH[dataset],
            "tpr": float(np.mean([r["tpr"] for r in rows])) if HAS_VALIDITY_TRUTH[dataset] else None,
            "fpr": float(np.mean([r["fpr"] for r in rows])) if HAS_VALIDITY_TRUTH[dataset] else None,
            "gates": gates, "pass": all(g["pass"] for g in gates.values())}


def shard_paths_for(dataset: str, ids) -> dict[int, Path]:
    return {r: RESULTS / dataset / f"replicate_{r}.json" for r in ids}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: v3_fresh_scorer.py <dataset>")
    dataset = argv[1]
    # the correction supersedes the first freeze when it exists; it is provably neutral and says so
    protocol_path = PROTOCOL_V2 if PROTOCOL_V2.exists() else PROTOCOL_V1
    if not protocol_path.exists():
        raise SystemExit(f"{PROTOCOL_V1} does not exist; freeze the protocol before scoring")
    protocol = json.loads(protocol_path.read_text())
    manifest = protocol["manifests"][dataset]
    result = score(dataset, manifest, shard_paths_for(dataset, manifest["replicates"]))
    out = RESULTS / dataset / ("summary_v3_fresh.json" if not (RESULTS / dataset / "summary_v3_fresh.json").exists()
                               else "summary_v3_fresh_after_correction.json")
    if out.exists():
        raise SystemExit(f"{out} exists; results are write-once")
    result["protocol_file"] = protocol_path.name
    result["protocol_sha256"] = protocol["protocol_sha256_of_prd"]
    if "correction" in protocol:
        result["protocol_correction"] = protocol["correction"]
    with open(out, "x") as handle:
        handle.write(json.dumps(result, indent=1, allow_nan=False) + "\n")
    if result["status"] != "complete":
        print(f"{dataset}: INCOMPLETE. rejected {result['rejected']}")
        return 1
    g = result["gates"]
    print(f"{dataset}: returned {g['return_rate']['returned']}/{g['return_rate']['of']} "
          f"(gate {g['return_rate']['pass']}), conditional MAE {result['conditional_mae']:.4f} "
          f"vs limit {g['conditional_mae']['limit']:.4f} (gate {g['conditional_mae']['pass']}) "
          f"-> {'PASS' if result['pass'] else 'FAIL'}")
    print(f"  written {out}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
