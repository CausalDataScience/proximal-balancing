"""v3 screen rule: the threshold is the data set's own noise floor.

The sealed thresholds of SCM-1 to SCM-4 (1.5, 1.8, 1.6, 1.7 x 10^-3) were calibrated on development data
at n = 24,000.  Read against the screen statistics they produced, each equals the median over the thirty
held-out choices of the noise term z * SE to within ten percent (ratios 0.93, 1.11, 0.94, 1.02).  So the
calibration was, in effect, "retain a choice whose upper bound sits at or below the typical noise", and
carrying the number to a data set with fewer rows or more covariates carried the wrong noise level.

v3 makes the rule explicit and data-adaptive, with no free parameter:

    floor = median_S  z * SE_S            over the thirty pooled screens of the data set
    retain S  iff  max(mean_r gap_{S,r}, 0) + z * SE_S  <=  floor

Everything else, the pooled screen of v2, the common radius and the largest-component median, is unchanged.
Nothing is refitted: the gaps and standard errors are the ones the sealed run stored.

    python3 -B noise_scaled.py        validates on the 80 sealed data sets, then scores every real-data run
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

import rhc_pooled as rp

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
Z = rp.Z
TARGETS = ("S1", "S2", "S3", "S12", "S13", "S23", "S123")
SEALED_T = {"SCM-1": 0.0015, "SCM-2": 0.0018, "SCM-3": 0.0016, "SCM-4": 0.0017}


def screens_of(record: dict) -> list[dict]:
    """The four per-rotation screens of one held-out choice, in either shard layout."""
    return record["screen"] if "screen" in record else [r["screen"] for r in record["rotations"]]


def pooled(record: dict) -> tuple[float, float]:
    g = np.array([s["gap"] for s in screens_of(record)])
    se = np.array([s["se"] for s in screens_of(record)])
    return max(float(g.mean()), 0.0), float(np.sqrt((se ** 2).sum()) / len(se))


def decide(probe: dict, n: int, seed: int) -> dict:
    """The v3 decision on one stored run: floor, retained set, and the sealed aggregation of it."""
    records = probe["records"] if "records" in probe else probe["per_split"]
    names = probe["split_names"]
    stats = {nm: pooled(records[nm]) for nm in names}
    floor = float(np.median([Z * se for _, se in stats.values()]))
    keep = [i for i, nm in enumerate(names) if stats[nm][0] + Z * stats[nm][1] <= floor]
    like = {"split_names": names, "score_crossproduct": probe["score_crossproduct"],
            "records": {nm: {"estimate": records[nm]["estimate"]} for nm in names}}
    agg = rp.aggregate_kept(like, keep, n, seed)
    kept = {names[i] for i in keep}
    return {"floor": floor, "n_retained": len(keep), "retained": sorted(kept), **agg,
            "tpr": len(kept & set(TARGETS)) / len(TARGETS),
            "fpr": len(kept - set(TARGETS)) / (len(names) - len(TARGETS)),
            "kept_w4": "S4" in kept, "kept_any_w5": any("5" in nm for nm in kept)}


def validate_on_simulations() -> dict:
    """v3 against the sealed rule on the 80 sealed confirmations, where the truth is one."""
    out = {}
    for level, t in SEALED_T.items():
        folder = "family_v2/scm4/confirm" if level == "SCM-4" else "family_v2/confirm"
        v3, sealed, floors = [], [], []
        for f in sorted(glob.glob(str(RESULTS / folder / f"{level}_seed*.json"))):
            d = json.load(open(f))
            dec = decide(d["probe"], d["n"], d["task"]["seed"])
            v3.append(dec); sealed.append(d["probe"]["estimate"]); floors.append(dec["floor"])
        err = [abs(x["output"] - 1.0) for x in v3 if x["output"] is not None]
        out[level] = {"sealed_t": t, "floor_median": float(np.median(floors)),
                      "returned": sum(x["output"] is not None for x in v3),
                      "mae_v3": float(np.mean(err)), "mae_sealed": float(np.mean([abs(s - 1.0) for s in sealed])),
                      "max_abs_change_in_estimate": float(max(abs(x["output"] - s) for x, s in zip(v3, sealed))),
                      "tpr": float(np.mean([x["tpr"] for x in v3])), "fpr": float(np.mean([x["fpr"] for x in v3])),
                      "kept_any_w5": int(sum(x["kept_any_w5"] for x in v3))}
    return out


def score_real(name: str) -> dict:
    files = sorted(glob.glob(str(RESULTS / name / "replicate_*.json")))
    rows = []
    for f in files:
        d = json.load(open(f))
        dec = decide(d["probe"], d["n"], d["seed"])
        rows.append({"replicate": d["replicate"], "tau": d["tau"], **dec,
                     "error": None if dec["output"] is None else abs(dec["output"] - d["tau"])})
    errs = [r["error"] for r in rows if r["error"] is not None]
    return {"replicates": rows, "returned": len(errs), "of": len(rows),
            "mae": float(np.mean(errs)) if errs else None,
            "p90": float(np.quantile(errs, 0.9)) if errs else None,
            "mean_estimate": float(np.mean([r["output"] for r in rows if r["output"] is not None])) if errs else None,
            "tpr": float(np.mean([r["tpr"] for r in rows])), "fpr": float(np.mean([r["fpr"] for r in rows])),
            "kept_any_w5": int(sum(r["kept_any_w5"] for r in rows)),
            "mean_retained": float(np.mean([r["n_retained"] for r in rows])),
            "floor_median": float(np.median([r["floor"] for r in rows]))}


def score_rhc() -> dict:
    """The real RHC run, no ground truth: what v3 retains and returns."""
    d = json.load(open(RESULTS / "rhc" / "diagnostic_v1_full_records.json"))
    import rhc_data as rd
    dec = decide(d["probe"], rd.N, rd.SEED)
    return dec


def main() -> int:
    out_path = RESULTS / "noise_scaled_v3.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; write-once")
    import run_family_v2 as rf
    sims = validate_on_simulations()
    print("=== v3 on the 80 sealed simulation data sets")
    for level, v in sims.items():
        print(f"  {level}: floor {v['floor_median']:.5f} vs sealed t {v['sealed_t']:.4f} | returned {v['returned']}/20 | "
              f"MAE v3 {v['mae_v3']:.4f} sealed {v['mae_sealed']:.4f} | max |change| {v['max_abs_change_in_estimate']:.4f} | "
              f"TPR {v['tpr']:.3f} FPR {v['fpr']:.3f} W5 kept {v['kept_any_w5']}")
    real = {name: score_real(name) for name in ("twins", "rhc_planted", "acic", "ihdp")}
    print("\n=== v3 on the real-data runs")
    for name, v in real.items():
        orc = json.load(open(RESULTS / name / "summary_v1.json"))["methods"]["oracle"]["mae"]
        print(f"  {name:12s} returned {v['returned']}/{v['of']} | MAE {v['mae']:.4f} (oracle {orc:.4f}) p90 {v['p90']:.4f} | "
              f"mean kept {v['mean_retained']:.1f} TPR {v['tpr']:.2f} FPR {v['fpr']:.2f} W5 kept {v['kept_any_w5']} | floor {v['floor_median']:.4f}")
    rhc = score_rhc()
    print(f"\n=== v3 on RHC (no ground truth): floor {rhc['floor']:.4f}, retained {rhc['n_retained']} {rhc['retained']}, "
          f"{rhc['return_kind']} {rhc['output']}")
    payload = {"artifact": "noise_scaled_v3", "generated_utc": rf.now(), "rule": __doc__.strip().splitlines()[0],
               "z": Z, "why": "the sealed thresholds equal the noise floor of their own data sets to within ten percent; "
                              "carried to smaller or wider data sets the fixed number sat below the noise and the screen "
                              "retained nothing (RHC, RHC-planted, ACIC, IHDP)",
               "adopted_after_seeing": "the no-return results of RHC, RHC-planted, ACIC and IHDP",
               "validated_on": "the 80 sealed simulation data sets, above",
               "simulations": sims, "real_data": real, "rhc": rhc,
               "source_sha256": {f: rf.sha256(HERE / f) for f in ("noise_scaled.py", "rhc_pooled.py")}}
    out_path.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(f"\nwritten {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
