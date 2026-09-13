#!/usr/bin/env python3
"""Staged validation of design F (review of 2026-09-11, section 6).

Row 1 of the staged table is analytic: with the verification representation the balance covariance is exactly
zero, so the population adjusted contrast equals tau.  This script runs the rows that need data:

  row 2  verification representation + learned nuisances     tests the estimation implementation
  row 3  learned representation       + learned nuisances     tests representation learning (optional, slow)

and reports the comparators at the same sample size.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import probe_brier_neural as pbn
import probe_dgp_ae as dgp


def adjust(feat: np.ndarray, data: dict, idx: dict, seed: int) -> dict:
    nuis = pbn.fit_nuisances(feat[idx["N"]], data["A"][idx["N"]], data["Y"][idx["N"]], seed)
    est = pbn.aipw_scores(nuis, feat[idx["E"]], data["A"][idx["E"]], data["Y"][idx["E"]])
    ov = pbn.overlap_pass(nuis, feat[idx["N"]])
    return {"theta": est["theta"], "se": est["sigma"] / math.sqrt(len(idx["E"])),
            "clip_rate": est["clip_rate"], "overlap_share": ov["overlap_share"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[24000])
    ap.add_argument("--replicates", type=int, nargs="+", default=[10000, 10001, 10002])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    rows = []
    for n in args.n:
        for rep in args.replicates:
            d = dgp.generate("F", n, rep, "pilot")
            g = np.random.default_rng(np.random.SeedSequence(dgp.run_key("unit_split", "pilot", "F", n, rep)))
            perm = g.permutation(n)
            third = n // 3
            idx = {"D": perm[:third], "N": perm[third:2 * third], "E": perm[2 * third:]}
            ye, ae = d["Y"][idx["E"]], d["A"][idx["E"]]
            out = {"n": n, "replicate": rep,
                   "naive": float(ye[ae == 1].mean() - ye[ae == 0].mean()),
                   "X": adjust(d["X"], d, idx, 11),
                   "XC": adjust(np.column_stack([d["X"], d["C"]]), d, idx, 12),
                   "raw": adjust(dgp.raw_matrix(d), d, idx, 13),
                   "oracle": adjust(np.column_stack([d["X"], d["U"], d["C"]]), d, idx, 14)}
            required = [(1,), (2, 3), (1, 2, 3, 4)]
            for S in required:
                if not dgp.balance_feasible("F", S):
                    raise AssertionError(f"design F must admit an exact representation for S={S}; "
                                         "a required verification case cannot be skipped")
                z = dgp.f_exact_representation(d, S)
                out[f"exact_S{''.join(map(str, S))}"] = adjust(z, d, idx, 15 + sum(S))
            rows.append(out)
            fmt = lambda k: f"{out[k]['theta']:+.3f}" if isinstance(out[k], dict) else f"{out[k]:+.3f}"
            keys = [k for k in out if k not in ("n", "replicate")]
            print(f"F n={n} rep={rep}: " + " | ".join(f"{k} {fmt(k)}" for k in keys), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"design": "F", "note": "staged validation, learned nuisances",
                                              "rows": rows}, indent=1, default=float))
    print("\nmeans over replicates")
    for k in rows[0]:
        if k in ("n", "replicate"):
            continue
        vals = [r[k]["theta"] if isinstance(r[k], dict) else r[k] for r in rows]
        print(f"  {k:20s} {np.mean(vals):+.3f}  (abs error {np.mean(np.abs(np.array(vals) - 1)):.3f})")


if __name__ == "__main__":
    main()
