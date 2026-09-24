"""The RHC data as a learner view for the sealed PROBE pipeline
(pre-registration memo/2026-09-22-rhc-real-data-prereg-v1.md, sections 1 to 4).

The right heart catheterization study of Connors et al. (1996), as re-analysed under the proximal framework
by Tchetgen Tchetgen et al. (2020) and Cui et al. (2022).  Those papers name ten physiological measures from a
blood test in the first 24 hours as the proxy candidates, and then allocate four of them to two named roles by
expert judgement.  This module keeps all ten and gives them to the learner in five fixed blocks with no role
label, which is the whole point of the comparison.

Nothing here reads the outcome to decide anything.  Every choice below is written in the pre-registration.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "materials" / "real_world_data" / "rhc" / "rhc.csv"
SHA256 = "9ef4ab578be4b40ad5d97d3a7e08ffdc1f9f76aeeefee51b4996e4221556f8e8"
N = 5_735
SEED = 98_100_000

TREATMENT, OUTCOME = "swang1", "t3d30"
TREATED_VALUE = "RHC"

# The five proxy blocks, grouped by clinical system before any relationship in the data was looked at.
# W1 is the pair Cui et al. call Z and W2 is the pair they call W; the learner is told neither.
BLOCKS: tuple[tuple[str, ...], ...] = (
    ("pafi1", "paco21"),        # blood gas: oxygenation and ventilation
    ("ph1", "hema1"),           # acid-base balance and haematocrit
    ("wblc1", "alb1"),          # white cell count and albumin
    ("bili1", "crea1"),         # liver and kidney function
    ("sod1", "pot1"),           # electrolytes
)
PROXY = tuple(c for block in BLOCKS for c in block)

CONTINUOUS_X = ("age", "edu", "surv2md1", "das2d3pc", "aps1", "scoma1", "meanbp1", "hrt1", "resp1", "temp1",
                "wtkilo1")
CATEGORICAL_X = ("sex", "race", "income", "ninsclas", "cat1", "ca", "dnr1",
                 "cardiohx", "chfhx", "dementhx", "psychhx", "chrpulhx", "renalhx", "liverhx", "gibledhx",
                 "malighx", "immunhx", "transhx", "amihx",
                 "resp", "card", "neuro", "gastr", "renal", "meta", "hema", "seps", "trauma", "ortho")
# Dropped, with the reason recorded in the pre-registration: more than half missing (cat2, adld3p, urin1),
# a date (sadmdte, dschdte, dthdte, lstctdte), an identifier (ptid), the treatment, or an outcome.
DROPPED = ("cat2", "adld3p", "urin1", "sadmdte", "dschdte", "dthdte", "lstctdte", "ptid",
           "death", "dth30", "swang1", "t3d30")

PUBLISHED = {"standard_dr": (-1.17, 0.32), "proximal_or": (-1.80, 0.44),
             "proximal_ipw": (-1.72, 0.30), "proximal_dr": (-1.66, 0.43)}
GATE = {"target": -1.17, "se": 0.32, "multiple": 3.0}   # the reproduction gate of section 7


def _rows() -> list[dict[str, str]]:
    with CSV.open(newline="") as handle:
        rows = [{k: v for k, v in row.items() if k} for row in csv.DictReader(handle)]
    if len(rows) != N:
        raise ValueError(f"{CSV} has {len(rows)} rows, the pre-registration fixes {N}")
    return rows


def _one_hot(values: list[str]) -> tuple[np.ndarray, list[str]]:
    """Indicator columns for every level but the first, so the encoding has no constant direction."""
    levels = sorted(set(values))
    if len(levels) < 2:
        raise ValueError("a categorical covariate with one level carries nothing")
    columns = [np.array([1.0 if v == level else 0.0 for v in values]) for level in levels[1:]]
    return np.column_stack(columns), [f"{level}" for level in levels[1:]]


def load() -> dict:
    """The learner view plus the column names, for the record."""
    rows = _rows()
    missing = {c: sum(1 for r in rows if r[c] in ("NA", "")) for c in PROXY}
    if any(missing.values()):
        raise ValueError(f"the proxy has missing values, which the pre-registration did not allow: {missing}")
    a = np.array([1.0 if r[TREATMENT] == TREATED_VALUE else 0.0 for r in rows])
    y = np.array([float(r[OUTCOME]) for r in rows])
    view = {"A": a, "Y": y}
    for j, block in enumerate(BLOCKS):
        view[f"W{j + 1}"] = np.column_stack([np.array([float(r[c]) for r in rows]) for c in block])
    columns, names = [], []
    for c in CONTINUOUS_X:
        columns.append(np.array([float(r[c]) for r in rows]))
        names.append(c)
    for c in CATEGORICAL_X:
        block, levels = _one_hot([r[c] for r in rows])
        columns.extend(block.T)
        names.extend(f"{c}={level}" for level in levels)
    view["X"] = np.column_stack(columns)
    return {"view": view, "x_names": names, "proxy_names": list(PROXY),
            "n": len(rows), "treated": int(a.sum()), "d_x": view["X"].shape[1]}


def level(d_x: int):
    """A Level for the sealed runner.  It reads only `k` and `name`; the dimensions are for the record."""
    import family_v2_dgp as g
    return g.Level(name="RHC", index=0, d_x=d_x, d_blocks=tuple(len(b) for b in BLOCKS), k=2, variant="real")
