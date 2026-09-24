"""Scoring for the direct image balancing runs: the threshold rule, the replayed aggregation, the entry check and
the report summary (execution PRD memo/2026-09-21-direct-image-balancing-figure-run-prd-v1.md, sections 4 and 7).

    python3 -B summarize_direct_balance.py report BENCH     summarise the report data sets at the frozen threshold

Every split shard stores the independent check's upper bound and the evaluation-fold AIPW scores, for the network
before balancing and after it.  So the screen and Algorithm 1's aggregation are REPLAYED here at a threshold, and
the threshold can be chosen on development data after those data sets were run, exactly as for SCM-1 to SCM-4.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import direct_balance_data as db
import direct_balance_eval as de
import summarize_family_v2 as sm

ARMS = ("after", "before")


def load_dataset(folder: Path) -> dict:
    """All split shards of one data set, the score vectors, and the comparators; missing pieces are reported."""
    splits = db.all_splits()
    names = [db.split_name(s) for s in splits]
    shards, scores, missing = {}, {arm: {} for arm in ARMS}, []
    for name in names:
        path = folder / f"{name}.json"
        if not path.exists():
            missing.append(name)
            continue
        shards[name] = json.loads(path.read_text())
        npz = folder / f"{name}.npz"
        if npz.exists():
            with np.load(npz) as f:
                for arm in ARMS:
                    if arm in f.files:
                        scores[arm][name] = f[arm]
    comp = folder / "comparators.json"
    return {"folder": folder, "names": names, "splits": splits, "shards": shards, "scores": scores,
            "missing": missing, "comparators": json.loads(comp.read_text()) if comp.exists() else None}


def problems(ds: dict) -> list[str]:
    out = [f"missing split {n}" for n in ds["missing"]]
    for name, sh in ds["shards"].items():
        if sh.get("status") != "ok":
            out.append(f"{name}: {sh.get('status')} ({sh.get('reason')})")
            continue
        for arm in ARMS:
            rec = sh[arm]
            if not (rec["check"]["finite"] and rec["estimate"]["finite"]):
                out.append(f"{name}/{arm}: a non-finite value")
            if name not in ds["scores"][arm]:
                out.append(f"{name}/{arm}: no score vector")
    if ds["comparators"] is None:
        out.append("no comparators")
    elif not ds["comparators"]["raw_cnn"].get("finite"):
        out.append("raw CNN not finite")
    return out


def upper(ds: dict, name: str, arm: str) -> float:
    sh = ds["shards"].get(name)
    if sh is None or sh.get("status") != "ok":
        return math.inf
    return float(sh[arm]["check"]["upper"])


def replay(ds: dict, threshold: float, arm: str, seed: int) -> dict:
    """The screen at `threshold` and then the sealed aggregation, for one arm of one data set."""
    names, splits = ds["names"], ds["splits"]
    usable = [n for n in names if n in ds["scores"][arm] and ds["shards"].get(n, {}).get("status") == "ok"]
    passed = [upper(ds, n, arm) < threshold and ds["shards"][n][arm]["check"]["overlap"] for n in usable]
    estimates = [ds["shards"][n][arm]["estimate"]["theta"] for n in usable]
    if not usable:
        return {"return_kind": "no_return", "estimate": None, "retained": [], "rho": None, "component_sizes": [],
                "members": [], "tpr": 0.0, "fpr": 0.0, "purity": None}
    matrix = np.column_stack([ds["scores"][arm][n] for n in usable])
    out = de.combine(usable, [splits[names.index(n)] for n in usable], estimates, matrix, passed, seed + 5)
    out.update(sm.mechanism(out))
    return out


def choose_threshold(datasets: list[dict]) -> dict:
    """The sealed rule's shape on the 'after' arm: a = p95 of the upper bounds of the eight splits that are
    balanceable in SCM-1, b = p05 of the other twenty-two, t = sqrt(a b) to two significant digits.  No cap: the
    numeric cap of 2.0e-3 belonged to probit critics scored on 6,000 rows."""
    bal, oth = [], []
    for ds in datasets:
        for name in ds["names"]:
            (bal if name in sm.BALANCEABLE else oth).append(upper(ds, name, "after"))
    a, b = float(np.percentile(bal, 95)), float(np.percentile(oth, 5))
    t_raw = math.sqrt(a * b) if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0 else math.nan
    return {"a_p95_balanceable": a, "b_p05_other": b, "t_raw": t_raw,
            "t": sm.round2(t_raw) if np.isfinite(t_raw) else None, "overlap_fail": not (a < b),
            "n_balanceable": len(bal), "n_other": len(oth)}


def comparator_errors(ds: dict) -> dict:
    c = ds["comparators"] or {}
    pick = {"X": c.get("X", {}).get("theta"), "oracle": c.get("oracle", {}).get("theta"),
            "naive": c.get("naive", {}).get("theta"), "raw_cnn": c.get("raw_cnn", {}).get("theta")}
    return {k: (None if v is None else abs(v - 1.0)) for k, v in pick.items()}, pick


def entry_check(datasets: list[dict], seeds: list[int]) -> dict:
    """Execution PRD section 4: the five conditions that open the report seeds."""
    thr = choose_threshold(datasets)
    reps, errs, raw_errs, x_errs, issues = [], [], [], [], {}
    for ds, seed in zip(datasets, seeds):
        found = problems(ds)
        if found:
            issues[seed] = found
        rp = replay(ds, thr["t"], "after", seed) if thr["t"] else {"return_kind": "no_return", "estimate": None}
        rp["seed"] = seed
        reps.append(rp)
        comp_err, _ = comparator_errors(ds)
        raw_errs.append(comp_err["raw_cnn"]); x_errs.append(comp_err["X"])
        if rp["estimate"] is not None:
            errs.append(abs(rp["estimate"] - 1.0))
    complete = not issues
    returned = all(r["return_kind"] == "unique_largest" for r in reps)
    gate = {"complete": complete, "unique_return_on_every_data_set": returned,
            "each_abs_error_at_most_0.10": returned and all(e <= 0.10 for e in errs),
            "beats_raw_cnn_and_X": (returned and None not in raw_errs and None not in x_errs
                                    and float(np.mean(errs)) < float(np.mean(raw_errs))
                                    and float(np.mean(errs)) < float(np.mean(x_errs))),
            "threshold_separates": not thr["overlap_fail"]}
    return {"threshold": thr, "replay": reps, "abs_errors": errs, "problems": issues, "gate": gate,
            "pass": all(gate.values())}


def _t_upper(d: np.ndarray) -> float | None:
    return sm._t_upper(d) if len(d) > 1 else None


def report(bench: str) -> int:
    import direct_balance_run as dr
    import run_family_v2 as rf
    proto_path = dr.protocol_path(bench, "report")
    proto = json.loads(proto_path.read_text())
    out_path = dr.RESULTS / bench / "report_summary_v1.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; summaries are write-once")
    t = proto["threshold"]["t"]
    done = [task for task in proto["tasks"] if (dr.dataset_dir(bench, "report", task["seed"]) / "done.json").exists()]
    rows = []
    for task in done:
        ds = load_dataset(dr.dataset_dir(bench, "report", task["seed"]))
        after, before = replay(ds, t, "after", task["seed"]), replay(ds, t, "before", task["seed"])
        comp_err, comp = comparator_errors(ds)
        rows.append({"seed": task["seed"], "problems": problems(ds), "after": after, "before": before,
                     "comparators": comp,
                     "per_split": {n: {arm: {"upper": upper(ds, n, arm),
                                             "theta": ds["shards"][n][arm]["estimate"]["theta"]
                                             if ds["shards"].get(n, {}).get("status") == "ok" else None}
                                       for arm in ARMS} for n in ds["names"]}})
    est = [r["after"]["estimate"] for r in rows]
    returned = [e for e in est if e is not None]
    err = np.array([abs(e - 1.0) for e in returned])
    paired = {}
    if len(returned) == len(rows) and rows:
        mine = np.array([abs(r["after"]["estimate"] - 1.0) for r in rows])
        for key in ("oracle", "X", "raw_cnn"):
            other = [r["comparators"][key] for r in rows]
            if None not in other:
                paired[f"abs_error_minus_{key}_upper95"] = _t_upper(mine - np.abs(np.array(other) - 1.0))
    summary = {"artifact": "direct_balance_report_summary", "benchmark": bench, "generated_utc": rf.now(),
               "protocol_sha256": rf.sha256(proto_path), "threshold": t, "planned": len(proto["tasks"]),
               "run": len(rows), "returned": len(returned),
               "estimates": est, "mae": float(err.mean()) if len(err) else None,
               "p90": float(np.percentile(err, 90)) if len(err) else None,
               "paired_reference_only": paired,
               "comparator_means": {k: (float(np.mean([r["comparators"][k] for r in rows]))
                                        if rows and None not in [r["comparators"][k] for r in rows] else None)
                                    for k in ("naive", "X", "raw_cnn", "oracle")},
               "before_balancing": {"returned": sum(r["before"]["estimate"] is not None for r in rows),
                                    "estimates": [r["before"]["estimate"] for r in rows]},
               "note": "independent data sets fixed in advance, all reported; the original PRD's gates assume "
                       "twenty data sets and are not judged here",
               "datasets": rows}
    out_path.write_text(json.dumps(rf._jsonable(summary), indent=1, allow_nan=False) + "\n")
    print(f"{bench}: run {len(rows)}/{len(proto['tasks'])} | returned {len(returned)} | estimates "
          f"{[None if e is None else round(e, 3) for e in est]} | MAE {summary['mae']} p90 {summary['p90']}")
    print(f"comparators {summary['comparator_means']} | paired (reference) {paired}\nwritten {out_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "report":
        raise SystemExit("usage: summarize_direct_balance.py report BENCH")
    raise SystemExit(report(sys.argv[2]))
