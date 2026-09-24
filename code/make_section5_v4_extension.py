"""The paper's Section 5 figures (version 4) drawn with the sealed and the added data sets together
(EXPERIMENT_INDEX_EXTENSION_v1.md; memo/2026-09-23-replicate-extension-prd-v2.md).

make_section5_v4_figures draws and checks, unchanged; this script only hands it other data and other reference
values:
    E1, E2        summarise_extension_v1.scm_pairs, sealed and added, 100 data sets per level
    E5, E6        summarise_extension_v1.real_pairs (100 Twins, 88 IHDP); ACIC as before, 9 realizations
    E3            v4's own RHC values, without Cui et al.'s standard DR (amendment 2)
    MAE checks    against results/extension_v1/summary_v1.json (pooled) for E1, E2, E5, E6; the frozen index for ACIC
Two axes widen so that every point stays inside, as v4's own check requires: the synthetic axis to +50 (proximal
2SLS reaches +34 on SCM-2) and IHDP's to -1000 to +400 (the single-proxy control reaches -811).

Writes figures/section5/section5-v4-{synthetic,realdata}-ext.{pdf,png} and copies
ICLR/paper/figures/probe-{synthetic,realdata}-v4-ext.pdf.  The v4 files are not touched.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import shutil
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import make_section5_v4_figures as v4
import matplotlib.pyplot as plt
import summarise_extension_v1 as se

SUMMARY = se.EXT / "summary_v1.json"
STEMS = {"synthetic": ("section5-v4-synthetic-ext", "probe-synthetic-v4-ext.pdf"),
         "realdata": ("section5-v4-realdata-ext", "probe-realdata-v4-ext.pdf")}
REAL_EXP = {"twins": "E5", "ihdp": "E6"}
REAL_KEY = {"probe": "probe", "cevae": "cevae", "raw": "raw", "X": "X", "p2sls_given_roles": "p2sls_given_roles",
            "p2sls_swapped_roles": "p2sls_swapped_roles", "coca": "coca_ate", "kspc_W1": "kspc_clean",
            "kspc_W4": "kspc_contaminated", "oracle": "oracle"}

v4.SYNTH_AXIS = dict(v4.SYNTH_AXIS, xlim=(-15.0, 50.0), ticks=v4.SYNTH_AXIS["ticks"] + [10.0],
                     labels=v4.SYNTH_AXIS["labels"] + ["10"])
v4.REAL_AXIS["ihdp"] = v4.symlog(0.2, (-1000.0, 400.0), [-100, 0, 1, 100], [-10, -1, 10])
v4.RHC_ROWS = [row for row in v4.RHC_ROWS if row[1] != "standard_dr"]


def load_synthetic():
    errors, available = {}, {}
    for level in v4.v3.LEVELS:
        pairs = {src: se.scm_pairs(level, src) for src in ("sealed", "extension")}
        available[level] = len(pairs["sealed"]["probe"]) + len(pairs["extension"]["probe"])
        errors[level] = {}
        for _, key, _, _ in v4.v3.ROWS:
            if key == "pixel" and level != "SCM-4":
                errors[level][key] = None
                continue
            e = np.array([v for src in pairs for v, _ in pairs[src][key]], dtype=float) - 1.0
            if len(e) != available[level] or not np.all(np.isfinite(e)):
                raise ValueError(f"{level} {key}: {np.isfinite(e).sum()} finite of {available[level]}")
            errors[level][key] = e
    return errors, available


def load_real():
    """v4's own loader for ACIC (and, as a check, the sealed Twins and IHDP); pooled data for Twins and IHDP."""
    sealed, available, _ = v4.load_real()
    errors = {"acic": sealed["acic"]}
    for name, exp in REAL_EXP.items():
        pairs = {src: se.real_pairs(exp, src) for src in ("sealed", "extension")}
        errors[name] = {}
        for key, skey in REAL_KEY.items():
            first = np.array([v - t for v, t in pairs["sealed"][skey]], dtype=float)
            if not np.allclose(np.sort(first), np.sort(sealed[name][key])):
                raise ValueError(f"{name} {key}: the sealed values differ from v4's own loader")
            e = np.array([v - t for src in pairs for v, t in pairs[src][skey]], dtype=float)
            if not np.all(np.isfinite(e)):
                raise ValueError(f"{name} {key}: a value is not finite")
            errors[name][key] = e
        available[name] = len(errors[name]["probe"])
    return errors, available


def pooled_mae_cells(decimals: int, keymap: dict | None, panels: dict) -> dict:
    s = json.loads(SUMMARY.read_text())["experiments"]
    cells = {}
    for panel, exp in panels.items():
        for key, block in s[exp].items():
            mine = (keymap or {}).get(key, key) if keymap is None else next((k for k, v in keymap.items() if v == key), None)
            if mine is None or block["pooled"]["mae"] is None:
                continue
            cells[(panel, mine)] = f"{block['pooled']['mae']:.{decimals}f}"
    return cells


def main() -> int:
    problems = []
    synth, synth_available = load_synthetic()
    real, real_available = load_real()
    rhc, _ = v4.load_rhc()
    with plt.rc_context(v4.STYLE):
        print("== Figure A, E1 and E2, sealed and added")
        fig, drawn, g = v4.draw_synthetic(synth)
        problems += v4.check_points(synth, synth_available, drawn, v4.v3.LEVELS, lambda _: v4.SYNTH_AXIS, v4.SYNTH_GROUPS)
        cells = pooled_mae_cells(3, None, {lv: lv for lv in v4.v3.LEVELS})
        cells = {k: v for k, v in cells.items() if not (k[1] == "pixel" and k[0] != "SCM-4")}
        problems += v4.check_mae(synth, v4.v3.LEVELS, cells, 3, v4.SYNTH_GROUPS)
        pdf_a, png_a, problems = v4.finish(fig, STEMS["synthetic"][0], v4.m.FIGDIR, g["height"], problems)
        print(f"   {v4.WIDTH} x {g['height']:.3f} in")

        names = [name for name, _, _ in v4.REAL_PANELS]
        print("== Figure B, Twins and IHDP sealed and added, ACIC as before, RHC without the standard DR")
        fig, drawn, g = v4.draw_realdata(real, rhc)
        problems += v4.check_points(real, real_available, drawn, names, lambda name: v4.REAL_AXIS[name], v4.REAL_GROUPS)
        cells = pooled_mae_cells(4, REAL_KEY, REAL_EXP)
        cells.update({k: v for k, v in v4.index_mae_real().items() if k[0] == "acic"})
        problems += v4.check_mae(real, names, cells, 4, v4.REAL_GROUPS)
        problems += v4.check_rhc(rhc, drawn["rhc"])
        pdf_b, png_b, problems = v4.finish(fig, STEMS["realdata"][0], v4.m.FIGDIR, g["height"], problems)
        print(f"   {v4.WIDTH} x {g['height']:.3f} in")
    for (key, (_, target)), pdf in zip(STEMS.items(), (pdf_a, pdf_b)):
        shutil.copyfile(pdf, v4.PAPER_FIGURES / target)
        print(f"copied {pdf.name} to {v4.PAPER_FIGURES / target}")
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
