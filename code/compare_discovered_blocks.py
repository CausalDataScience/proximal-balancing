"""Discovered against designed blocks, data set by data set (2026-10-19, memo section 43).

SCM-1 to SCM-3: the refined-discovery runs (run_discovered_blocks.py --refined) against the E1 estimate with the
designed blocks, both under the sealed tolerance.  Twins, IHDP, ACIC: the refined-discovery runs
(run_benchmarks_discovered.py) against the stored runs behind the figure, both under the v3 noise-scaled decision.
Only data sets present in both are compared, and the fold seeds must agree.

Usage: compare_discovered_blocks.py SCM_DIR BENCH_DIR [TAG]   (TAG: refined, the default, or merged)
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import noise_scaled as ns  # noqa: E402

RES = CODE.parent / "results"
STORED = {"twins": [("twins", range(11, 21)), ("extension_v1/E5", range(21, 111))],
          "ihdp": [("ihdp", range(12, 22)), ("extension_v1/E6", range(22, 100))],
          "acic": [("acic", range(2, 11))]}


def stored_path(bench: str, r: int) -> Path:
    for folder, reps in STORED[bench]:
        if r in reps:
            return RES / folder / f"replicate_{r}.json"
    raise KeyError((bench, r))


def summary(rows: list[tuple[float | None, float | None]], truth: list[float]) -> str:
    both = [(a, b, t) for (a, b), t in zip(rows, truth) if a is not None and b is not None]
    da = np.array([abs(a - t) for a, _, t in both]); db = np.array([abs(b - t) for _, b, t in both])
    none = sum(a is None for a, _ in rows), sum(b is None for _, b in rows)
    return (f"{len(rows)} sets, returned discovered {len(rows) - none[0]} designed {len(rows) - none[1]}; "
            f"MAE discovered {da.mean():.4f} designed {db.mean():.4f}; "
            f"discovered better in {(da < db).sum()}, same in {(da == db).sum()}, worse in {(da > db).sum()}")


def main(argv: list[str]) -> int:
    scm_dir, bench_dir = Path(argv[1]), Path(argv[2])
    tag = argv[3] if len(argv) > 3 else "refined"
    for lvn in ("SCM-1", "SCM-2", "SCM-3"):
        rows = []
        for f in sorted(glob.glob(str(scm_dir / f"{tag}_{lvn}_seed*.json"))):
            d = json.loads(Path(f).read_text())
            rows.append((d["estimate"], d["designed_estimate"]))
        if rows:
            print(f"{lvn}: {summary(rows, [1.0] * len(rows))}")
    for bench in ("twins", "ihdp", "acic"):
        rows, truth, same_blocks = [], [], 0
        for f in sorted(glob.glob(str(bench_dir / f"{tag}_{bench}_replicate_*.json"))):
            d = json.loads(Path(f).read_text())
            s = json.loads(stored_path(bench, d["replicate"]).read_text())
            assert s["seed"] == d["seed"] and s["n"] == d["n"] and abs(s["tau"] - d["tau"]) < 1e-12, f
            rows.append((d["v3"]["output"], ns.decide(s["probe"], s["n"], s["seed"])["output"]))
            truth.append(d["tau"])
            same_blocks += all(sum(v > 0 for v in c["from"].values()) == 1 for c in d["blocks"])
        if rows:
            print(f"{bench}: {summary(rows, truth)}; discovered blocks equal the designed ones in {same_blocks}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
