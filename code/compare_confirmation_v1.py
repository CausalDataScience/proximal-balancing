"""Summary of the confirmation run (run_confirmation_v1.py, 2026-10-20, memo section 43).

Twins replicates 111-210: MAE of PROBE with the merged discovered blocks, PROBE with the designed blocks, and the
comparison methods computed on the same replicates; paired differences of absolute errors (merged minus each other
method) with their standard errors.  SCM-1 to SCM-3 (new seeds): MAE of the designed and the merged runs.

Usage: compare_confirmation_v1.py RESULTS_DIR
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np


def paired(a: np.ndarray, b: np.ndarray) -> str:
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean():+.4f} (se {se:.4f}, t {d.mean() / se:+.2f})"


def main(argv: list[str]) -> int:
    folder = Path(argv[1])
    rows = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(folder / "confirm_twins_replicate_*.json")))]
    if rows:
        tau = np.array([r["tau"] for r in rows])
        est = {"PROBE merged": [r["merged"]["v3"]["output"] for r in rows],
               "PROBE designed": [r["designed"]["v3"]["output"] for r in rows],
               "KSPC given W1": [r["kspc"]["W1"] for r in rows], "KSPC given W4": [r["kspc"]["W4"] for r in rows],
               "2SLS given roles": [r["proximal"]["p2sls_given_roles"] for r in rows],
               "2SLS swapped roles": [r["proximal"]["p2sls_swapped_roles"] for r in rows],
               "adjust (X, W)": [r["baselines"]["raw"] for r in rows], "adjust X": [r["baselines"]["X"] for r in rows],
               "oracle": [r["baselines"]["oracle"] for r in rows]}
        none = {k: sum(v is None for v in vals) for k, vals in est.items()}
        err = {k: np.abs(np.array([np.nan if v is None else v for v in vals]) - tau) for k, vals in est.items()}
        print(f"Twins, {len(rows)} new replicates; J of the merged blocks: {sorted({len(r['blocks']) for r in rows})}; "
              f"no return: {none}")
        for k, e in sorted(err.items(), key=lambda kv: np.nanmean(kv[1])):
            line = f"  {k:20s} MAE {np.nanmean(e):.4f}"
            if k != "PROBE merged" and not np.isnan(e).any() and not np.isnan(err["PROBE merged"]).any():
                line += f"   merged minus this: {paired(err['PROBE merged'], e)}"
            print(line)
    for lvn in ("SCM-1", "SCM-2", "SCM-3"):
        rs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(folder / f"confirm_{lvn}_seed*.json")))]
        if not rs:
            continue
        both = [r for r in rs if r["designed"]["estimate"] is not None and r["merged"]["estimate"] is not None]
        ed = np.array([abs(r["designed"]["estimate"] - 1.0) for r in both])
        em = np.array([abs(r["merged"]["estimate"] - 1.0) for r in both])
        print(f"{lvn}: {len(rs)} new seeds ({len(both)} returned by both), J {sorted({len(r['blocks']) for r in rs})}; MAE merged {em.mean():.4f} "
              f"designed {ed.mean():.4f}; merged minus designed {paired(em, ed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
