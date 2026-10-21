"""PROBE's errors with the blocks found from the proxy, in the order in which the Section 5 figures hold the data sets
(2026-10-20, memo section 43).

The figures (make_section5_v4_extension, make_section5_v5_figure, make_section5_v6_figure) hold, per panel, one error
per data set: SCM-1 to SCM-3 the 20 sealed confirmations (results/family_v2/confirm) then the 80 E1 data sets
(results/extension_v1/E1), each in file order, with the sealed-tolerance estimate; Twins and IHDP the sealed replicates
then the extension replicates in numeric order, and ACIC the replicates of its summary, each with the v3 decision.
This module returns the same data sets, in the same order, with PROBE run on the merged discovered blocks
(discover_blocks.discover_blocks_merged, rule fixed in job 3419470) under the same rule, and fails if any is missing.
RHC: the merged discovered blocks' v3 output and its resampled 95% interval (run_rhc_discovered.py merged).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
RES = CODE.parent / "results"
DISC = RES / "discovered_blocks_v1"
SCM_DIRS = (DISC / "deltaai_merged", DISC / "deltaai_figure_scm")      # jobs 3419470 (20 E1 per level) and 3422555
BENCH_DIR = DISC / "deltaai_bench_merged"                               # job 3419470
SEALED_IDS = {"twins": range(11, 21), "ihdp": range(12, 22)}
EXT = {"twins": "E5", "ihdp": "E6"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def synth_errors(level: str) -> np.ndarray:
    seeds = [_load(p)["task"]["seed"] for folder in (RES / "family_v2" / "confirm", RES / "extension_v1" / "E1")
             for p in sorted(folder.glob(f"{level}_seed*.json"))]
    out = []
    for s in seeds:
        hits = [d / f"merged_{level}_seed{s}.json" for d in SCM_DIRS if (d / f"merged_{level}_seed{s}.json").exists()]
        if not hits:
            raise FileNotFoundError(f"{level} seed {s}: no discovered-block run")
        est = _load(hits[0])["estimate"]
        out.append(np.nan if est is None else est - 1.0)
    return np.array(out, dtype=float)


def replicate_order(name: str) -> list[int]:
    if name in SEALED_IDS:
        ext = sorted(int(p.stem.split("_")[1]) for p in (RES / "extension_v1" / EXT[name]).glob("replicate_*.json"))
        return list(SEALED_IDS[name]) + ext
    import make_section5_v4_figures as v4
    return [r["replicate"] for r in v4.pf.summary(name)["replicates"]]


def real_errors(name: str) -> np.ndarray:
    out = []
    for r in replicate_order(name):
        d = _load(BENCH_DIR / f"merged_{name}_replicate_{r}.json")
        out.append(np.nan if d["v3"]["output"] is None else d["v3"]["output"] - d["tau"])
    return np.array(out, dtype=float)


def rhc() -> tuple[float, tuple[float, float]]:
    v = _load(RES / "rhc" / "discovered_blocks_v2_merged.json")["variants"]["merged"]
    return v["v3"]["output"], (v["spread"]["q025"], v["spread"]["q975"])
