"""Image-specific screen tolerance for SCM-7-v4, set on development data only.

Provenance of the development data.  The SCM-7-v3 confirmation (protocol 2ae2898a60ee, 20 data sets)
failed five of its thirteen frozen gates.  Its shards are reclassified here as development data for a new
protocol.  The v4 confirmation will use fresh seeds that no step of this calibration has seen.

The rule is fixed in this docstring before any number below is computed.

  1. Calibration half: data sets 95200000..95209000.  Check half: 95210000..95219000.
  2. Clean rows: per-rotation screen records of choices that are balance-feasible and target tau, with the
     overlap condition met.
  3. t0 = 95th percentile of the one-sided upper bound U over clean rows of the calibration half.  A
     per-rotation pass rate of 0.95 makes the four-rotation retention of one clean choice about 0.95^4 = 0.81.
  4. t must lie inside the population separation window (2.983e-5, 7.711e-3) of the image design.
  5. The infeasible-row pass rate at t on the calibration half must be at most 0.10.  If it is not, t is
     lowered to the largest value that meets it.
  6. t is rounded down to two significant figures.
Everything is then reported on both halves, together with a replay of which choices four-rotation retention
would keep at t.  Candidate estimates exist only for choices that were retained under the old tolerance,
so the replay counts retained choices by class; it cannot recompute returned estimates.
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np

import minimal_suite_common as c

SHARDS = "../results/scm7_v3/shards/confirmation_*.json"
WINDOW = (2.983e-5, 7.711e-3)
OLD_T = 2e-3


def floor_sig(x: float, sig: int = 2) -> float:
    e = math.floor(math.log10(x))
    return math.floor(x / 10 ** (e - sig + 1)) * 10 ** (e - sig + 1)


def rows_of(cells):
    out = []
    for q in cells:
        for r in q["rows"]:
            ev = r["evaluator_only"]
            kind = ("clean" if ev["feasible"] and ev["targets_truth"] else
                    "contaminated" if ev["feasible"] else "infeasible")
            out.append({"seed": q["data_seed"], "rot": r["rotation"], "split": r["split_id"], "kind": kind,
                        "U": r["screen"]["upper"], "overlap": r["screen"]["overlap_ok"]})
    return out


def rates(rows, t):
    passes = lambda k: [r["U"] < t and r["overlap"] for r in rows if r["kind"] == k]
    feas = [r["U"] < t and r["overlap"] for r in rows if r["kind"] in ("clean", "contaminated")]
    return {"clean_pass": float(np.mean(passes("clean"))), "contaminated_pass": float(np.mean(passes("contaminated"))),
            "infeasible_pass": float(np.mean(passes("infeasible"))), "scorer_tpr": float(np.mean(feas)),
            "scorer_fpr": float(np.mean(passes("infeasible")))}


def replay(rows, t):
    by = {}
    for r in rows:
        by.setdefault((r["seed"], r["split"], r["kind"]), []).append(r["U"] < t and r["overlap"])
    seeds = sorted({r["seed"] for r in rows})
    per = []
    for s in seeds:
        kept = [(sp, k) for (ss, sp, k), v in by.items() if ss == s and len(v) == 4 and all(v)]
        per.append({"seed": s, "clean": sum(k == "clean" for _, k in kept),
                    "contaminated": sum(k == "contaminated" for _, k in kept),
                    "infeasible": sum(k == "infeasible" for _, k in kept)})
    return {"per_seed": per,
            "mean_clean_retained": float(np.mean([p["clean"] for p in per])),
            "seeds_with_at_least_2_clean": sum(p["clean"] >= 2 for p in per),
            "seeds_with_clean_plurality": sum(p["clean"] > p["contaminated"] + p["infeasible"] for p in per),
            "seeds_with_nothing_retained": sum(p["clean"] + p["contaminated"] + p["infeasible"] == 0 for p in per),
            "seeds_retaining_an_infeasible_choice": sum(p["infeasible"] > 0 for p in per)}


def main() -> int:
    cells = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(SHARDS))]
    assert len(cells) == 20, len(cells)
    calib = [q for q in cells if q["data_seed"] < 95210000]
    check = [q for q in cells if q["data_seed"] >= 95210000]
    rc, rk = rows_of(calib), rows_of(check)
    clean_u = np.array([r["U"] for r in rc if r["kind"] == "clean" and r["overlap"]])
    inf_u = np.array([r["U"] for r in rc if r["kind"] == "infeasible"])
    print("calibration half: U quantiles by class")
    for name, arr in (("clean", clean_u), ("contaminated", np.array([r["U"] for r in rc if r["kind"] == "contaminated"])),
                      ("infeasible", inf_u)):
        q = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"  {name:13s} n={len(arr):4d}  p05 {q[0]:.2e}  p25 {q[1]:.2e}  p50 {q[2]:.2e}  p75 {q[3]:.2e}  p95 {q[4]:.2e}")
    t0 = float(np.quantile(clean_u, 0.95))
    t = min(t0, WINDOW[1] * (1 - 1e-9))
    steps = []
    if rates(rc, t)["infeasible_pass"] > 0.10:
        grid = np.sort(np.unique(inf_u))
        ok = [u for u in grid if u <= t and rates(rc, u)["infeasible_pass"] <= 0.10]
        t = float(max(ok)); steps.append("lowered to meet the infeasible pass-rate cap")
    t = floor_sig(t)
    assert WINDOW[0] < t < WINDOW[1]
    print(f"\nrule: t0 = {t0:.3e} (95th percentile of clean U) -> t = {t:.2e} {steps}")
    report = {}
    for label, tt in (("old_tolerance", OLD_T), ("new_tolerance", t)):
        report[label] = {"t": tt, "calibration": rates(rc, tt), "check": rates(rk, tt),
                         "replay_calibration": replay(rc, tt), "replay_check": replay(rk, tt)}
        a, b = report[label]["calibration"], report[label]["check"]
        ra, rb = report[label]["replay_calibration"], report[label]["replay_check"]
        print(f"\n{label} t={tt:.2e}")
        print(f"  per-rotation pass  clean {a['clean_pass']:.3f} / {b['clean_pass']:.3f}   contaminated {a['contaminated_pass']:.3f} / {b['contaminated_pass']:.3f}"
              f"   infeasible {a['infeasible_pass']:.3f} / {b['infeasible_pass']:.3f}   (calibration / check)")
        print(f"  scorer TPR {a['scorer_tpr']:.3f} / {b['scorer_tpr']:.3f}   scorer FPR {a['scorer_fpr']:.3f} / {b['scorer_fpr']:.3f}")
        for nm, rp in (("calibration", ra), ("check", rb)):
            print(f"  replay {nm:11s}: mean clean retained {rp['mean_clean_retained']:.1f}; >=2 clean {rp['seeds_with_at_least_2_clean']}/10; "
                  f"clean plurality {rp['seeds_with_clean_plurality']}/10; nothing retained {rp['seeds_with_nothing_retained']}/10; "
                  f"an infeasible retained {rp['seeds_retaining_an_infeasible_choice']}/10")
    out = {"artifact": "scm7_v4_tolerance_calibration", "development_source": "SCM-7-v3 confirmation shards, protocol 2ae2898a60ee",
           "rule": __doc__, "population_window": list(WINDOW), "t0": t0, "tolerance": t, "adjustments": steps,
           "report": report, "source_snapshot": c.source_snapshot(["calibrate_scm7_tolerance.py"])}
    c.write_json_new("../results/scm7_v4/tolerance_calibration.json", out)
    print("\nwritten to ../results/scm7_v4/tolerance_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
