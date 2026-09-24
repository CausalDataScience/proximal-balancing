"""PROBE on the RHC data, once, exactly as memo/2026-09-22-rhc-real-data-prereg-v1.md fixes it.

    python3 -B run_rhc.py

Writes results/rhc/result_v1.json, write-once.  The pre-registration fixes the blocks, the covariates, the
seed, the threshold grid and the reproduction gate; this file only executes them and reports every item of
section 8 whatever the numbers turn out to be.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "4")

import json
import sys
import time
from pathlib import Path

import numpy as np

import family_v2_probe as pr
import probe_structured_scm as ps
import rhc_data as rd
import run_family_v2 as rf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "rhc" / "result_v1.json"
PREREG = HERE.parent / "memo" / "2026-09-22-rhc-real-data-prereg-v1.md"
SOURCES = ["family_v2_probe.py", "family_v2_dgp.py", "probe_structured_scm.py", "rhc_data.py", "run_rhc.py"]
SEALED = ["family_v2_probe.py", "family_v2_dgp.py", "probe_structured_scm.py"]
GRID = [0.5e-4, 1e-4, 2e-4, 4e-4, 8e-4, 16e-4, 32e-4]      # section 6, fixed before the run
BOOTSTRAP_SEED_OFFSET = 5                                   # the sealed aggregation's own offset


def x_only_aipw(view: dict, seed: int) -> dict:
    """AIPW adjusting for X alone, on the same four folds the pipeline uses, for the reproduction gate."""
    fold_rows = pr.folds(len(view["A"]), seed)
    scores = np.zeros(len(view["A"]))
    for rot in range(4):
        roles = pr.roles(rot)
        rows = {k: fold_rows[v] for k, v in roles.items()}
        prep = pr.Prep(view, rows["D"])
        f = {role: np.column_stack([np.ones(len(rows[role])), prep.x(view, rows[role])]) for role in ("N", "E")}
        psi = pr.aipw(f["N"], view["A"][rows["N"]], view["Y"][rows["N"]],
                      f["E"], view["A"][rows["E"]], view["Y"][rows["E"]])
        scores[rows["E"]] = psi
    n = len(scores)
    return {"estimate": float(scores.mean()), "se": float(scores.std(ddof=1) / np.sqrt(n))}


def path_over_grid(probe: dict, n: int, seed: int) -> list[dict]:
    """The retained set and the returned value at every threshold of the fixed grid.

    The common radius is recomputed for each retained set from the stored score cross-product, which is what
    the sealed aggregation does with the score matrix itself.
    """
    names = probe["split_names"]
    cross = np.array(probe["score_crossproduct"])
    splits = pr.all_splits()
    out = []
    for t in GRID:
        keep = [i for i, name in enumerate(names)
                if all(r["screen"]["upper"] <= t for r in probe["records"][name]["rotations"])]
        row = {"t": t, "n_retained": len(keep), "retained": [names[i] for i in keep]}
        if keep:
            covariance = cross[np.ix_(keep, keep)] / (n * n)
            draws = np.random.default_rng(seed + BOOTSTRAP_SEED_OFFSET).multivariate_normal(
                np.zeros(len(keep)), covariance, size=ps.BOOTSTRAP_DRAWS)
            rho = float(np.quantile(np.max(np.abs(draws), axis=1), 0.95))
            values = np.array([probe["records"][names[i]]["estimate"] for i in keep])
            agg = ps.aggregate([splits[i] for i in keep], values, rho)
            se = float(np.sqrt(np.median(np.diag(covariance))))
            row.update({"rho": rho, "return_kind": agg["return_kind"], "estimate": agg["output"],
                        "component_sizes": agg.get("component_sizes"), "se_of_median_member": se})
        else:
            row.update({"rho": None, "return_kind": "no_return", "estimate": None,
                        "component_sizes": [], "se_of_median_member": None})
        out.append(row)
    return out


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; the pre-registration allows one run and results are write-once")
    t0 = time.time()
    data = rd.load()
    view, n = data["view"], data["n"]
    print(f"RHC: n = {n}, treated {data['treated']}, X has {data['d_x']} columns, "
          f"proxy {data['proxy_names']}", flush=True)

    gate_value = x_only_aipw(view, rd.SEED)
    lo = rd.GATE["target"] - rd.GATE["multiple"] * rd.GATE["se"]
    hi = rd.GATE["target"] + rd.GATE["multiple"] * rd.GATE["se"]
    passed = lo <= gate_value["estimate"] <= hi
    print(f"reproduction gate: X-only AIPW {gate_value['estimate']:.4f} (se {gate_value['se']:.4f}); "
          f"published standard DR {rd.GATE['target']} within [{lo:.2f}, {hi:.2f}] -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)

    probe = pr.run_dataset(view, rd.level(data["d_x"]), GRID[2], rd.SEED)
    print(f"PROBE finished in {probe['seconds']} s", flush=True)

    records = {name: {"estimate": rec["estimate"],
                      "screen_upper_by_rotation": [r["screen"]["upper"] for r in rec["rotations"]],
                      "max_screen_upper": max(r["screen"]["upper"] for r in rec["rotations"]),
                      "theta_by_rotation": [r["theta"] for r in rec["rotations"]]}
               for name, rec in probe["records"].items()}
    payload = {
        "artifact": "rhc_probe_result",
        "generated_utc": rf.now(),
        "prereg_sha256": rf.sha256(PREREG),
        "source_sha256": {f: rf.sha256(HERE / f) for f in SOURCES},
        "data": {"path": str(rd.CSV.relative_to(HERE.parent)), "sha256": rf.sha256(rd.CSV), "n": n,
                 "treated": data["treated"], "d_x": data["d_x"], "x_names": data["x_names"],
                 "proxy_names": data["proxy_names"],
                 "blocks": {f"W{j + 1}": list(b) for j, b in enumerate(rd.BLOCKS)}},
        "seed": rd.SEED,
        "reproduction_gate": {**gate_value, "published_standard_dr": rd.GATE["target"],
                              "published_se": rd.GATE["se"], "interval": [lo, hi], "pass": bool(passed)},
        "per_split": records,
        "path_over_threshold_grid": path_over_grid(probe, n, rd.SEED),
        "sealed_run_at_t": {"t": GRID[2], "retained": probe["retained"], "rho": probe["rho"],
                            "return_kind": probe["return_kind"], "estimate": probe["estimate"],
                            "component_sizes": probe["component_sizes"]},
        "published_comparison": rd.PUBLISHED,
        "notes": {
            "claim": "a demonstration on real data; there is no ground truth here and no validation is claimed",
            "threshold": "the pre-registration designates no single t; the whole grid is reported",
            "roles": "the learner received no label identifying a block; W1 is the pair Cui et al. call Z and "
                     "W2 the pair they call W, which is used only for the comparison in the report",
        },
        "seconds": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "x") as handle:
        handle.write(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(f"written {OUT}")
    for row in payload["path_over_threshold_grid"]:
        print(f"  t = {row['t']:.1e}: retained {row['n_retained']:2d} {row['retained']} -> "
              f"{row['return_kind']} {row['estimate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
