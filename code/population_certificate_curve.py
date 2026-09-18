"""Exact population curves for the bias certificate figure: discrepancy and bias along r, every split.

For each proxy split S and coefficient r, the representation Z_S(r) = (X, C, mean of kept channels + r I)
has a residual discrepancy D_res(S, r) and an adjustment target tau_Z(S, r), both computed from closed-form
conditional propensities and regressions (scm6_population), with only an outer expectation simulated.
Nothing is fitted.

Each split is labelled by which assumption of the approximate bias bound it meets, read off the SCM:
  valid          S within {1,2,3}: pure measurements of U, kept side holds X, I and C;
  completeness   4 in S, 0 not in S: block 4 loads on I, which also sits in block 0 on the kept side, so the
                 held-out block is not conditionally independent of the kept blocks given (X, U).  S4 still
                 balances exactly with consistency, exchangeability, separation and overlap all holding,
                 so by Theorem exact identification the failing assumption is outcome-relevant completeness;
  exchangeability  0 in S: the kept side loses C, which drives both A and Y.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import minimal_suite_common as c
import scm6_population as sp

GRID = np.round(np.arange(-1.0, 1.0001, 0.05), 4)
N_MC = 100000


def label(split: tuple[int, ...]) -> str:
    if 0 in split:
        return "exchangeability_fails"
    if 4 in split:
        return "completeness_fails"
    return "valid"


def main() -> int:
    out = {"grid": GRID.tolist(), "n_mc": N_MC, "tau": 1.0, "splits": {}}
    for split in c.oriented_splits():
        sid = c.split_id(split)
        root = sp.operative_root(split)
        rs = [None] if 0 in split else sorted(set(GRID.tolist()) | ({round(root, 6)} if root is not None else set()))
        pts = []
        for r in rs:
            d2 = sp.residual_discrepancy(split, r if r is not None else 0.0, n=N_MC)["d2_res"]
            tau_z = sp.adjusted_effect(split, r, n=N_MC)["tau_Z"]
            pts.append({"r": r, "d_res": float(np.sqrt(max(d2, 0.0))), "tau_z": float(tau_z)})
        out["splits"][sid] = {"label": label(split), "root": root, "points": pts}
        at_root = [q for q in pts if root is not None and q["r"] is not None and abs(q["r"] - root) < 1e-5]
        note = "" if not at_root else f"D={at_root[0]['d_res']:.2e} tau={at_root[0]['tau_z']:+.4f}"
        print(f"{sid:6s} {label(split):22s} root={root} {note}", flush=True)
    c.write_json_new("../results/population_certificate_curve_v1.json", out)
    print("written to ../results/population_certificate_curve_v1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
