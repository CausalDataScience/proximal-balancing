"""E8 population sweep: D_res^2(r) and tau_{Z_r} on the frozen delta grid, for every SCM and split."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import e8_population as e8
import probe_structured_scm as m

DELTAS = (0.0, 0.01, -0.01, 0.02, -0.02, 0.05, -0.05, 0.10, -0.10, 0.20, -0.20)   # frozen in section 2.8
OUT = "../results/probe_e8_population_v2.json"


def main() -> int:
    t0 = time.time()
    splits = [s for s in m.ALL_SPLITS if 0 not in s]      # block 0 must stay in the representation
    payload = {"provenance": {
        "script": "run_e8_population.py",
        "experiment": "E8 population half, section 2.8 of memo/2026-09-12-scm-family-and-experiment-plan.md",
        "deltas": list(DELTAS), "quadrature_nodes": e8.GH_NODES,
        "root_rule": ("the balance condition is quadratic in r and has two roots; the implementation searches "
                      "np.linspace(-1, 1, 101), so the reported r* is the root inside that interval and both "
                      "roots are stored with the effect each one targets"),
        "supersedes": "probe_e8_population_v1.json, which used a unimodal search and reported a mix of the two roots",
        "source_sha256": {f: m.source_sha256(Path(f)) for f in
                          ("e8_population.py", "run_e8_population.py", "probe_structured_scm.py")},
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": ("exact for the probit-linear arms; SCM-5 applies a strictly increasing sinh to the "
                 "measurements, so the rank-Gaussian channel recovers the same K and the same population "
                 "values as SCM-1, which is asserted by a check rather than assumed"),
        "complete": False}, "settings": {}}

    for scm, spec in m.SCM_FAMILY.items():
        rows = []
        for split in splits:
            category = m.split_category(split)
            found = e8.operative_root(spec, split)
            r_star = found["r_star"]
            if r_star is None:
                rows.append({"split": list(split), "category": category, "roots": [],
                             "reachable_on_implementation_grid": False, "r_star": None,
                             "note": "no r balances this held-out block; the candidate cannot be made exact"})
                continue
            d2_star = e8.residual_discrepancy(spec, split, r_star)
            tau_star = e8.adjusted_effect(spec, split, r_star)
            grid = []
            for delta in DELTAS:
                r = r_star + delta
                d2 = e8.residual_discrepancy(spec, split, r)
                tau = e8.adjusted_effect(spec, split, r)
                grid.append({"delta": delta, "r": r, "D2": d2, "D": float(np.sqrt(max(d2, 0.0))),
                             "tau_Z": tau, "abs_bias": abs(tau - 1.0),
                             "amplification": (abs(tau - 1.0) / np.sqrt(d2)) if d2 > 1e-18 else None})
            rows.append({"split": list(split), "category": category, "roots": found["roots"],
                         "reachable_on_implementation_grid": found["reachable"],
                         "tau_Z_at_each_root": [e8.adjusted_effect(spec, split, r) for r in found["roots"]],
                         "r_star": r_star, "D2_at_root": d2_star, "tau_Z_at_root": tau_star,
                         "abs_bias_at_root": abs(tau_star - 1.0), "grid": grid})
        payload["settings"][scm] = {"legacy_key": m.SCM_FAMILY_NOTES[scm]["legacy_key"], "splits": rows}
        exact = [r for r in rows if r.get("D2_at_root") is not None and r["D2_at_root"] < 1e-12]
        blind = [r for r in exact if r["abs_bias_at_root"] > 1e-6]
        unreachable = [r for r in rows if r["r_star"] is None]
        print(f"{scm}: {len(rows)} splits | {len(exact)} balance exactly at the reachable root | "
              f"{len(blind)} of those still target the wrong value "
              f"(max bias {max([r['abs_bias_at_root'] for r in blind], default=0.0):.4f}) | "
              f"{len(unreachable)} admit no balancing r")
    payload["provenance"]["complete"] = True
    payload["provenance"]["runtime_s"] = time.time() - t0
    m.write_json_atomic(OUT, payload)
    print(f"\nwritten to {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
