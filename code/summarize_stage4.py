"""Re-derive the stage 4 tables from the saved cells, so a reviewer can recompute every number in the report.

Usage: python3 summarize_stage4.py ../results/dgp_f_staged_r6_s4_fixed_data.json [more files]
Each file's cells are summarised separately; files are never pooled, since the fixed-data run varies only the
optimizer seed and the data-seed run varies both.
"""
from __future__ import annotations

import json
import math
import argparse

import numpy as np

import provenance as prov
import staged_dgp_f as sf
import verdicts

STAGE_SEED_BASE = sf.STAGE_SEED + 300


def table(rows, cols, head):
    print(f"\n{head}")
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  " + "  ".join(c.rjust(w[c]) for c in cols))
    for r in rows:
        print("  " + "  ".join(str(r[c]).rjust(w[c]) for c in cols))


def summarise(path: str, expect_mode: str | None = None) -> None:
    d = json.load(open(path))
    meta, s4 = d["provenance"], d["stage4"]
    cells = s4["cells"]
    print(f"\n=== {path} | code {meta["code_hash"]} | complete={meta.get("complete")} | "
          f"{s4['data_seeds']} data seeds x {s4['opt_seeds']} optimizer seeds | n_est={s4['n_est']} | {len(cells)} cells")
    # Refuse to summarise a run that is not exactly what it claims to be.  The checks live in
    # `provenance.verify` so that every consumer applies the same ones, including the checkpoint digests
    # the round 6 summariser never looked at (round 7 review, PRD-5).
    expected = {(int(ds), int(n), int(o)) for ds in meta["seeds"]["data"]
                for n in s4["sizes"] for o in meta["seeds"]["optimizer"]}
    prov.require(d, path, expect_mode=expect_mode,
                 expect_cells=expected, cell_key=lambda c: (c["data_seed"], c["n_rep"], c["opt_seed"]),
                 cells_at=lambda payload: payload["stage4"]["cells"], require_manifest=True,
                 require_checkpoints=True, expected_criteria_sha256=verdicts.CRITERIA_SHA256)
    status = prov.current_source_status(d)
    print("current source status: " + ", ".join(f"{k}={v}" for k, v in status.items()))
    ctrl = s4.get("controls")
    if ctrl:
        print("controls: " + ", ".join(f"{k}={v}" for k, v in ctrl.items())
              + f", fixed_preprocessing={s4.get('fixed_preprocessing')}")

    rows = []
    for n in s4["sizes"]:
        c = [x for x in cells if x["n_rep"] == n]
        tot = np.array([x["err_total"] for x in c]); rep = np.array([x["err_representation"] for x in c])
        est = np.array([x["err_estimation"] for x in c]); d2 = np.array([x["pop_D2"] for x in c])
        gap = np.array([x["audit_gap"] for x in c]); raw = np.array([x["err_raw"] for x in c])
        ex = np.array([x["err_exact"] for x in c])
        by_ds = {}
        for x in c:
            by_ds.setdefault(x["data_seed"], []).append(x["err_total"])
        within = math.sqrt(float(np.mean([np.var(v, ddof=1) if len(v) > 1 else 0.0 for v in by_ds.values()])))
        between = (math.sqrt(float(np.var([np.mean(v) for v in by_ds.values()], ddof=1)))
                   if len(by_ds) > 1 else float("nan"))
        # paired against raw, per cell; and the share of cells beating raw
        beats = int((tot < raw).sum())
        hierarchical = verdicts.hierarchical_stage4(cells, n)
        rows.append({"n_rep": n, "cells": len(c),
                     "median |theta-1|": f"{np.median(tot):.4f}", "p90": f"{np.quantile(tot, 0.9):.4f}",
                     "max": f"{tot.max():.4f}",
                     "data-seed median |tau_Z-1|": f"{hierarchical['median']:.4f}",
                     "data-seed p90 |tau_Z-1|": f"{hierarchical['p90']:.4f}",
                     "pooled-cell p90 (diagnostic)": f"{np.quantile(rep, 0.9):.4f}",
                     "median |theta-tau_Z|": f"{np.median(est):.4f}", "max |theta-tau_Z|": f"{est.max():.4f}",
                     "median pop D2": f"{np.median(d2):.1e}", "median audit gap": f"{np.median(gap):+.5f}",
                     "sd within data seed": f"{within:.4f}",
                     "sd between data seeds": "n/a" if math.isnan(between) else f"{between:.4f}",
                     "beats raw": f"{beats}/{len(c)}", "raw median": f"{np.median(raw):.4f}",
                     "exact median": f"{np.median(ex):.4f}"})
    table(rows, list(rows[0].keys()), "per representation sample size")

    # data-seed level replicate view (the statistical unit): per data seed, the median over optimizer seeds
    if s4["data_seeds"] > 1:
        rows2 = []
        for n in s4["sizes"]:
            for ds in sorted({x["data_seed"] for x in cells}):
                c = [x for x in cells if x["n_rep"] == n and x["data_seed"] == ds]
                rows2.append({"n_rep": n, "data_seed": ds, "opt seeds": len(c),
                              "median |theta-1|": f"{np.median([x['err_total'] for x in c]):.4f}",
                              "median |tau_Z-1|": f"{np.median([x['err_representation'] for x in c]):.4f}",
                              "raw": f"{c[0]['err_raw']:.4f}", "exact": f"{c[0]['err_exact']:.4f}",
                              "oracle": f"{c[0]['err_oracle']:.4f}"})
        table(rows2, list(rows2[0].keys()), "per data seed (median over optimizer seeds)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--expect-mode", choices=["manuscript", "variant"], default="manuscript")
    args = ap.parse_args()
    for p in args.paths:
        summarise(p, expect_mode=args.expect_mode)
