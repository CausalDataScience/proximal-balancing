"""The two curves that explain the SCM-6 result: what the objective should be, and what it is.

The population curve is computed exactly by scm6_population and involves no fitting.  The estimated
curve is the nested Brier gap Algorithm 1 actually minimises, measured at several critic sample sizes
and several target widths so the bias can be attributed to each.
"""
from __future__ import annotations

import argparse
import numpy as np

import minimal_suite_common as c
import probe_highdim_w as hd
import scm6_population as sp

SPLIT = (1,)
GRID = tuple(np.round(np.arange(-1.0, 1.001, 0.05), 4))
PENALTIES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SAMPLES = (1500, 6000, 24000)
WIDTHS = (1, 2, 4, 8, 32, 128, 256)
DATA_SEED = 620261


def critic_rows(n: int, seed: int) -> tuple[dict, np.ndarray]:
    """Fit the embedding on its own rows, then hand back n independent rows for the critics."""
    data = hd.generate(n + 6000, seed)
    view = hd.observed_view(data)
    sl = lambda i: {k: ([b[i] for b in v] if k == "H" else v[i]) for k, v in view.items()}
    emb = hd.fit_embedding(sl(np.arange(4800)), sl(np.arange(4800, 6000)))
    if emb["fail_closed"]:
        raise RuntimeError(emb["fail_closed"])
    rows = sl(np.arange(6000, 6000 + n))
    return rows, hd.apply_embedding(emb, rows)


def gap_curve(rows: dict, k_hat: np.ndarray, projection: np.ndarray | None) -> list[float]:
    """The estimated gap over the r grid, optionally against a projected rather than a full target."""
    original = hd.heldout
    if projection is not None:
        hd.heldout = lambda r, s, _p=projection: np.column_stack([r["X"], r["H"][SPLIT[0]] @ _p])
    try:
        pen = hd.split_penalties(rows, k_hat, SPLIT, PENALTIES, 7)
        out = []
        for r in GRID:
            try:
                out.append(hd.select_r(rows, k_hat, SPLIT, pen, grid=(float(r),))["signed_gap"])
            except RuntimeError as exc:
                out.append(float(str(exc).rsplit(": ", 1)[1]))
        return out
    finally:
        hd.heldout = original


def compute() -> dict:
    root = sp.operative_root(SPLIT)
    out = {"split_id": c.split_id(SPLIT), "r_grid": list(GRID), "balance_root": root,
           "population_d2": [sp.residual_discrepancy(SPLIT, float(r), n=200000)["d2_res"]
                             for r in GRID],
           "population_tau_z": [sp.adjusted_effect(SPLIT, float(r), n=200000)["tau_Z"] for r in GRID],
           "by_sample": {}, "by_width": {}}
    for n in SAMPLES:
        rows, k_hat = critic_rows(n, DATA_SEED)
        out["by_sample"][str(n)] = gap_curve(rows, k_hat, None)
        print(f"  n_phi={n}: argmin r={GRID[int(np.argmin(out['by_sample'][str(n)]))]:+.2f}")
    rows, k_hat = critic_rows(SAMPLES[0], DATA_SEED)
    block = rows["H"][SPLIT[0]]
    _, _, vt = np.linalg.svd(block - block.mean(0), full_matrices=False)
    for d in WIDTHS:
        out["by_width"][str(d + 1)] = gap_curve(rows, k_hat, vt[:d].T)
        print(f"  width={d + 1}: argmin r={GRID[int(np.argmin(out['by_width'][str(d + 1)]))]:+.2f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/scm6_objective_curves_v1.json")
    args = ap.parse_args()
    print(f"SCM-6 objective curves | split {c.split_id(SPLIT)} | "
          f"balance root {sp.operative_root(SPLIT):+.4f}")
    c.write_json_new(args.out, compute() | {"complete": True,
                                            "source_snapshot": c.source_snapshot(
                                                ["scm6_population.py", "probe_highdim_w.py",
                                                 "scm6_objective_curves.py"]),
                                            "environment": c.environment()})
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
