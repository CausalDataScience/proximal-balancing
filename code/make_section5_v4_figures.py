"""Section 5 figures, version 4: the v3 design at 5.5 in wide and at most 2.4 in tall.

    section5-v4-synthetic   E1 and E2.  Four panels (SCM-1 to SCM-4), one row per method, one dot per data set; x is
                            the error, estimate minus the true effect 1, on v3's symmetric-log axis (linear within
                            +-0.1).  A role-requiring method takes one row: filled dots just above the row centre are
                            the correct roles (roles given, or the clean block), hollow dots just below are the wrong
                            roles (roles swapped, or the contaminated block), and its two MAE numbers are stacked in
                            the same order.  A black tick marks each set's mean; the numbers at the right of each
                            panel are the MAE, the mean absolute error over data sets.
    section5-v4-realdata    E5 (Twins), E6 (IHDP) and E7 (ACIC, exploratory): the same rows without the pixel CNN,
                            one dot per replicate, x the estimate minus that replicate's exact effect on a
                            symmetric-log axis scaled for each data set.  E3 (RHC): a narrow panel on its own linear
                            axis, the point estimates, PROBE's resampled 95% interval and KSPC's five blocks.

Data, read-only, through the existing loaders, imported rather than copied:
    E1, E2   make_section5_v3_figures.load, which reads through make_section5_v2_figures (collect, shards,
             pixel_cnn_amended, kspc_rows) and checks that every KSPC file pairs with its sealed seed
    E5-E7    make_prospective_figure.summary, .errors and .coca: results/{twins,ihdp}/summary_v3_fresh.json,
             results/acic/summary_v1.json with results/noise_scaled_v3.json (acic), results/coca_ate_all_v1.json,
             results/kspc_comparator/E{5,6,7}/
    E3       the files make_rhc_figure.py reads: results/noise_scaled_v3.json (rhc), results/rhc/
             aggregate_uncertainty_v1.json, and materials/real_world_data/rhc/rhc.csv through its
             observational_contrast; the two values Cui et al. published, from the E3 table of EXPERIMENT_INDEX.md
             (checked against make_rhc_figure.PUBLISHED); KSPC from results/kspc_comparator/E3/rhc_replicate_0.json
Every KSPC file read is checked against its sha256 in results/kspc_comparator/FROZEN_v1.json.  Nothing is refitted
or rerun and nothing under results/ is written.

Writes figures/section5/section5-v4-{synthetic,realdata}.{pdf,png} (png at 200 dpi) and the copies
ICLR/paper/figures/probe-{synthetic,realdata}-v4.pdf.  Checks printed on every run: points drawn against data sets
available for every panel and row, every point inside its axis, text collisions, text over data marks, content
inside the page, the height limit, and every MAE shown against EXPERIMENT_INDEX.md (E1 and E2 to 3 decimals, E5 to
E7 to 4 decimals) and every RHC value against its E3 table.
"""
from __future__ import annotations

import sys

# Importing the older scripts must not write into code/__pycache__: its .pyc files are older than their sources, so
# Python would otherwise rewrite them.
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import re
import shutil
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory as blend

with warnings.catch_warnings():
    # v2's coverage docstring holds an invalid escape sequence; the warning concerns that text only (as in v3)
    warnings.simplefilter("ignore", SyntaxWarning)
    import make_section5_v3_figures as v3
    import make_prospective_figure as pf
    import make_rhc_figure as rf

m = v3.m  # make_section5_figures: the results path and text_collisions
ROOT = v3.ROOT
RESULTS = m.RESULTS
PAPER_FIGURES = ROOT / "ICLR" / "paper" / "figures"
STEMS = {"synthetic": ("section5-v4-synthetic", "probe-synthetic-v4.pdf"),
         "realdata": ("section5-v4-realdata", "probe-realdata-v4.pdf")}

# ----------------------------------------------------------------------------- style, from v3

BLUE, GREEN, GRAY, ORANGE, DARK = v3.BLUE, v3.GREEN, v3.GRAY, v3.ORANGE, v3.DARK
INK, INK2, HAIR, AXIS = v3.INK, v3.INK2, v3.HAIR, v3.AXIS
FILLED, HOLLOW = v3.FILLED, v3.HOLLOW
LABEL_PT, TICK_PT, MAE_PT, NOTE_PT = 7.0, 6.5, 6.0, 6.0  # names, titles, axis labels / ticks / MAE / legend, notes
STYLE = {**v3.STYLE, "font.size": LABEL_PT, "axes.titlesize": LABEL_PT, "axes.labelsize": LABEL_PT,
         "xtick.labelsize": TICK_PT, "ytick.labelsize": LABEL_PT, "figure.dpi": 200}
# Text is measured, and every collision checked, at two resolutions: 200 dpi, the png, whose hinted glyphs run up to
# 9% wider than the outlines at these sizes, and 800 dpi, close to the unhinted outlines of the pdf.  The layout
# uses the larger width, so it holds in both files.
CHECK_DPI = (200, 800)
FOOTNOTE = v3.FOOTNOTE                        # "† run after the protocol was frozen"
PUBLISHED_NOTE = "* published by Cui et al. (2024)"
LEGEND = [(FILLED, "filled: correct roles"), (HOLLOW, "hollow: wrong roles")]

# Geometry, inches unless a name says pt.  5.5 in is the paper's text width.
WIDTH, MAX_HEIGHT = 5.5, 2.4
MARGIN, GAP, MAE_PAD = v3.MARGIN, v3.GAP, v3.MAE_PAD
LINE_GAP = 0.03       # between the tick labels, the x label and the note line
RHC_GAP = 0.15        # from the ACIC MAE column to the RHC panel
RHC_MIN_W = 0.9       # the RHC panel is as wide as its title or widest name needs, and at least this
YPAD_PT, TITLE_PAD_PT, TICK_LEN_PT, TICK_PAD_PT = v3.YPAD_PT, v3.TITLE_PAD_PT, v3.TICK_LEN_PT, v3.TICK_PAD_PT
DOT_S, RHC_DOT_S = v3.DOT_S, 14.0

# Rows, in points: the y data unit of every panel is one point, so the offsets below are exact on paper.
ONE_PT = 10.5         # pitch of a row with one version
TWO_PT = 15.5         # pitch of a row with a correct and a wrong version
WRAP_PT = 17.0        # pitch of a one-version row whose name takes two lines
SPLIT_PT = 3.3        # filled dots this far above the row centre, hollow dots this far below; also the two MAE numbers
SEP_PT = 2.0          # extra space at a group separator
HEADER_PT = 8.5       # the MAE header, above the first row centre
TOP_PT, FLOOR_PT = 4.0, 1.0  # above the header; below the last row
JITTER_PT = {False: 1.5, True: 0.9}          # vertical jitter half-width: one-version row, one band of a split row
TICK_HALF_PT = {False: 4.2, True: 2.5}       # the mean tick, drawn under the dots
RHC_LABEL_PT = 6.0    # the RHC panel's own names
RHC_LIFT_PT = 2.5     # from an RHC mark up to its name

# ----------------------------------------------------------------------------- rows and panels

# (name as drawn, ((row key, marker style), ...) correct version first, colour).  Groups are split by a thin line;
# the three group headers of v3 are gone.
SYNTH_GROUPS = [
    [("PROBE (ours)", (("probe", FILLED),), BLUE),
     ("CEVAE", (("cevae", FILLED),), GREEN),
     ("Adjust for $X$ and all of $W$", (("raw", FILLED),), GRAY),
     ("Adjust for $X$", (("x", FILLED),), GRAY),
     ("Pixel CNN", (("pixel", FILLED),), GRAY)],
    [("Proximal 2SLS", (("p2sls_given", FILLED), ("p2sls_swapped", HOLLOW)), ORANGE),
     ("Single-proxy control", (("park_clean", FILLED), ("park_contaminated", HOLLOW)), ORANGE),
     ("KSPC", (("kspc_clean", FILLED), ("kspc_contaminated", HOLLOW)), ORANGE)],
    [("Oracle (adjusts for $(X,U)$)", (("oracle", FILLED),), DARK)],
]
# make_prospective_figure's keys.  The single-proxy control is its one version, COCA turned into an ATE on block
# W1, the clean block (coca_ate_all_v1.json), so it is drawn filled.
REAL_GROUPS = [
    [("PROBE (ours)", (("probe", FILLED),), BLUE),
     ("CEVAE", (("cevae", FILLED),), GREEN),
     ("Adjust for $X$ and all of $W$", (("raw", FILLED),), GRAY),
     ("Adjust for $X$", (("X", FILLED),), GRAY)],
    [("Proximal 2SLS", (("p2sls_given_roles", FILLED), ("p2sls_swapped_roles", HOLLOW)), ORANGE),
     ("Single-proxy control, as an ATE", (("coca", FILLED),), ORANGE),
     ("KSPC", (("kspc_W1", FILLED), ("kspc_W4", HOLLOW)), ORANGE)],
    [("Oracle (adjusts for $(X,U)$)", (("oracle", FILLED),), DARK)],
]
# Names broken once, after the comma, the words unchanged: on one line the first would widen the label column by
# 0.3 in and the second the RHC panel by 0.15 in.
WRAP = {"Single-proxy control, as an ATE": ", ", "proximal DR, roles assigned*": ", "}
VERSION = {"p2sls_given": "roles given", "p2sls_swapped": "roles swapped", "park_clean": "clean block",
           "park_contaminated": "contaminated block", "kspc_clean": "clean block",
           "kspc_contaminated": "contaminated block", "p2sls_given_roles": "roles given",
           "p2sls_swapped_roles": "roles swapped", "kspc_W1": "clean block", "kspc_W4": "contaminated block"}


def minus(v: float) -> str:
    return f"{v:g}".replace("-", "−")


def symlog(linthresh: float, xlim: tuple, ticks: list, minor: list = ()) -> dict:
    return {"linthresh": linthresh, "linscale": 1.0, "xlim": xlim, "ticks": ticks,
            "labels": [minus(t) for t in ticks], "minor": list(minor)}


# E1 and E2: v3's axis, unchanged.
SYNTH_AXIS = {"linthresh": v3.LINTHRESH, "linscale": v3.LINSCALE, "xlim": v3.XLIM, "ticks": v3.TICKS,
              "labels": v3.TICK_LABELS, "minor": v3.MINOR_TICKS}
SYNTH_PANELS = [(level, v3.TITLES[level]) for level in v3.LEVELS]
# E5 to E7: linear within +-linthresh, about the size of the typical error (Twins 0.01, IHDP and ACIC 0.2), so the
# accurate methods are resolved and the far errors still fit.  The limits hold every point with room for the dot,
# checked below; the farthest are the single-proxy control on Twins (+0.87), IHDP (-23.9) and ACIC (-58.9 and
# +129.8, its MAE being 34.6).  Labelled ticks are as dense as 6.5 pt labels allow without touching.
REAL_PANELS = [("twins", "Twins", "E5"), ("ihdp", "IHDP", "E6"), ("acic", "ACIC (exploratory)", "E7")]
REAL_AXIS = {"twins": symlog(0.01, (-0.5, 2.0), [-0.1, 0, 0.1, 1], [-0.01, 0.01]),
             "ihdp": symlog(0.2, (-50.0, 20.0), [-10, 0, 1, 10], [-1]),
             "acic": symlog(0.2, (-150.0, 500.0), [-10, 0, 1, 100], [-100, -1, 10])}
EXPECTED_REPLICATES = {"twins": 10, "ihdp": 10, "acic": 9}
REAL_XLABEL = "estimate minus the exact effect"

# E3: (name, key, colour); * marks the two values Cui et al. published.
RHC_ROWS = [("PROBE (no roles)", "probe", BLUE),
            ("standard DR*", "standard_dr", GRAY),
            ("proximal DR, roles assigned*", "proximal_dr", ORANGE),
            ("difference in means", "crude", GRAY),
            ("KSPC, five blocks", "kspc", ORANGE)]
RHC_TITLE, RHC_XLABEL = "RHC (no ground truth)", "effect on days survived"
RHC_XLIM, RHC_TICKS = (-2.4, -0.5), [-2.0, -1.5, -1.0]


def display(label: str) -> str:
    """The name as drawn: broken once at WRAP[label] if it has an entry, the words unchanged."""
    if label not in WRAP:
        return label
    cut = WRAP[label]
    shown = label.replace(cut, cut.replace(" ", "\n", 1), 1)
    if shown.replace("\n", " ") != label:
        raise ValueError(f"line break changed the name {label!r}")
    return shown


def pitch(row) -> float:
    label, versions, _ = row
    if len(versions) == 2:
        return TWO_PT
    return WRAP_PT if label in WRAP else ONE_PT


# ----------------------------------------------------------------------------- data

def sha_check(paths: list[Path]) -> list[str]:
    """Every KSPC file read, against its sha256 in the KSPC freeze record."""
    frozen = json.loads((RESULTS / "kspc_comparator" / "FROZEN_v1.json").read_text())["sha256"]
    problems = []
    for p in paths:
        rel = str(p.relative_to(ROOT))
        if rel not in frozen:
            problems.append(f"{rel}: not in FROZEN_v1.json")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != frozen[rel]:
            problems.append(f"{rel}: sha256 differs from FROZEN_v1.json")
    return problems


def load_synthetic() -> tuple[dict, dict, list[Path]]:
    """v3's errors (estimate minus 1) per level and row key, the data sets available, and the KSPC files read."""
    errors, available = v3.load()
    paths = []
    for level in v3.LEVELS:
        folder = RESULTS / "kspc_comparator" / ("E2" if level == "SCM-4" else "E1")
        paths += [folder / f"{level}_replicate_{s['task']['rep']}.json" for s in v3.v2.shards(level)]
    return errors, available, paths


def load_real() -> tuple[dict, dict, list[Path]]:
    """Errors, estimate minus the exact effect of that replicate, per data set and row key, through
    make_prospective_figure; all or nothing for every row."""
    keys = [key for group in REAL_GROUPS for _, versions, _ in group for key, _ in versions]
    errors, available, paths = {}, {}, []
    for name, _, exp in REAL_PANELS:
        s = pf.summary(name)
        reps = [r["replicate"] for r in s["replicates"]]
        available[name] = len(reps)
        if len(reps) != EXPECTED_REPLICATES[name] or s["returned"] != len(reps):
            raise ValueError(f"{name}: {len(reps)} replicates, {s['returned']} returned; "
                             f"the index lists {EXPECTED_REPLICATES[name]}")
        # the exact effect each comparator file carries is the one the summary carries
        coca = {r["replicate"]: r for r in json.loads((RESULTS / "coca_ate_all_v1.json").read_text())
                ["datasets"][name]["replicates"]}
        for r in s["replicates"]:
            path = RESULTS / "kspc_comparator" / exp / f"{name}_replicate_{r['replicate']}.json"
            paths.append(path)
            k_tau = json.loads(path.read_text())["tau"]
            if abs(k_tau - r["tau"]) > 1e-12 or abs(coca[r["replicate"]]["tau"] - r["tau"]) > 1e-12:
                raise ValueError(f"{name} {r['replicate']}: exact effect differs between files")
        errors[name] = {}
        for key in keys:
            e = np.asarray(pf.errors(s, key, name), dtype=float)
            if len(e) != len(reps) or not np.all(np.isfinite(e)):
                raise ValueError(f"{name} {key}: {np.isfinite(e).sum()} finite of {len(reps)} replicates")
            errors[name][key] = e
    return errors, available, paths


def load_rhc() -> tuple[dict, list[Path]]:
    """E3 from the files make_rhc_figure.py reads, the published values from the index, KSPC's five blocks."""
    retained = json.loads((RESULTS / "noise_scaled_v3.json").read_text())["rhc"]
    if any("1" in nm for nm in retained["retained"]) or len(retained["retained"]) != 15:  # make_rhc_figure's check
        raise ValueError(f"unexpected retained set {retained['retained']}")
    spread = json.loads((RESULTS / "rhc" / "aggregate_uncertainty_v1.json").read_text())["resampled"]
    crude, crude_se = rf.observational_contrast()
    index = index_rhc()
    for (label, value, _, _), key in zip(rf.PUBLISHED, ("standard_dr", "proximal_dr")):
        if f"{value:.2f}" != f"{index[key][0]:.2f}":
            raise ValueError(f"{key}: make_rhc_figure.PUBLISHED {value} against the index {index[key][0]}")
    path = RESULTS / "kspc_comparator" / "E3" / "rhc_replicate_0.json"
    kspc = json.loads(path.read_text())["kspc_skpv_with_x"]
    blocks = [kspc[f"W{b}"]["ate"] for b in range(1, 6)]
    return {"probe": (retained["output"], (spread["q025"], spread["q975"])),
            "standard_dr": (index["standard_dr"][0], None),
            "proximal_dr": (index["proximal_dr"][0], None),
            "crude": (crude, None), "crude_se": crude_se,
            "kspc": blocks}, [path]


# ----------------------------------------------------------------------------- the index

def index_block(exp: str) -> list[list[str]]:
    """The table rows of one experiment's section of EXPERIMENT_INDEX.md, bold marks removed."""
    text = (ROOT / "EXPERIMENT_INDEX.md").read_text()
    block = text.split(f"\n## {exp}.", 1)[1].split("\n## ", 1)[0]
    lines = [ln for ln in block.splitlines() if ln.startswith("|")]
    return [[c.strip().replace("**", "") for c in ln.strip().strip("|").split("|")] for ln in lines[2:]]


REAL_INDEX_ROWS = {"PROBE": "probe", "CEVAE": "cevae", "2SLS, roles given": "p2sls_given_roles",
                   "2SLS, roles given (role-requiring)": "p2sls_given_roles", "adjust for X and all of W": "raw",
                   "adjust for X": "X", "2SLS, roles swapped": "p2sls_swapped_roles", "COCA as an ATE": "coca",
                   "KSPC with X, clean block†": "kspc_W1", "KSPC with X, contaminated block†": "kspc_W4",
                   "oracle": "oracle"}


def index_mae_real() -> dict[tuple[str, str], str]:
    """The MAE cells of the E5, E6 and E7 tables, keyed by (data set, row key)."""
    cells = {}
    for name, _, exp in REAL_PANELS:
        for row_name, value in index_block(exp):
            if row_name not in REAL_INDEX_ROWS:
                raise KeyError(f"{exp}: index row {row_name!r} has no figure row")
            cells[(name, REAL_INDEX_ROWS[row_name])] = value
    return cells


def numbers(cell: str) -> list[float]:
    return [float(x.replace("−", "-")) for x in re.findall(r"[−-]?\d+\.\d+", cell)]


def index_rhc() -> dict[str, list[float]]:
    """The E3 table: PROBE with its interval, the two published values, the difference in means, KSPC's range."""
    starts = {"PROBE": "probe", "standard doubly robust": "standard_dr", "proximal doubly robust": "proximal_dr",
              "difference in means": "crude", "KSPC": "kspc"}
    out = {}
    for row_name, value in index_block("E3"):
        key = next((k for s, k in starts.items() if row_name.startswith(s)), None)
        if key is None:
            raise KeyError(f"E3: index row {row_name!r} has no figure row")
        out[key] = numbers(value)
    return out


# ----------------------------------------------------------------------------- layout

def positions(groups) -> tuple[list, list]:
    """Row centres in points, the first at 0 and the rest below, and the separators between groups."""
    rows, seps, y, prev = [], [], 0.0, None
    for members in groups:
        for i, row in enumerate(members):
            half = pitch(row) / 2
            if prev is not None:
                if i == 0:
                    seps.append(y - prev - SEP_PT / 2)
                y -= prev + (SEP_PT if i == 0 else 0.0) + half
            rows.append((row, y))
            prev = half
    return rows, seps


def ylimits(rows) -> tuple[float, float]:
    last_row, last_y = rows[-1]
    return last_y - pitch(last_row) / 2 - FLOOR_PT, HEADER_PT + TOP_PT


class Ruler:
    """Text extents in inches through v3.measure, the larger of the CHECK_DPI renderings."""

    def __init__(self) -> None:
        self.figs = [plt.figure(figsize=(WIDTH, 1.0), dpi=d) for d in CHECK_DPI]
        for f in self.figs:
            f.canvas.draw()

    def __call__(self, text: str, **kw) -> tuple[float, float]:
        sizes = [v3.measure(f, text, **kw) for f in self.figs]
        return max(s[0] for s in sizes), max(s[1] for s in sizes)

    def close(self) -> None:
        for f in self.figs:
            plt.close(f)


def mae_text(e: np.ndarray, decimals: int) -> str:
    return f"{np.mean(np.abs(e)):.{decimals}f}"


def vertical(ruler: Ruler, rows, titles: list[str], note_texts: list[str]) -> dict:
    """Heights from measured text, bottom to top: note line, x label, tick labels, panels, title."""
    lo, hi = ylimits(rows)
    title_h = max(ruler(t, fontsize=LABEL_PT, linespacing=1.15)[1] for t in titles)
    tick_h = ruler("0.1", fontsize=TICK_PT)[1]
    xlabel_h = ruler("estimate", fontsize=LABEL_PT)[1]
    note_h = max(ruler(t, fontsize=NOTE_PT)[1] for t in note_texts)
    bottom = MARGIN + note_h + LINE_GAP + xlabel_h + LINE_GAP + tick_h + (TICK_LEN_PT + TICK_PAD_PT) / 72
    axes_h = (hi - lo) / 72
    height = bottom + axes_h + TITLE_PAD_PT / 72 + title_h + MARGIN
    return {"ylim": (lo, hi), "bottom": bottom, "axes_h": axes_h, "height": height, "note_h": note_h,
            "xlabel_y": MARGIN + note_h + LINE_GAP}


# ----------------------------------------------------------------------------- drawing

def style_of(colour: str, style: str) -> tuple[str, str, float]:
    # hollow rings are white inside so a stack of them still reads as rings (v3)
    return (colour, "white", 0.35) if style == FILLED else ("white", colour, 0.6)


def set_symlog(ax, spec: dict) -> None:
    ax.set_xscale("symlog", linthresh=spec["linthresh"], linscale=spec["linscale"])
    ax.set_xlim(*spec["xlim"])
    ax.xaxis.set_major_locator(FixedLocator(spec["ticks"]))
    ax.xaxis.set_major_formatter(FixedFormatter(spec["labels"]))
    ax.xaxis.set_minor_locator(FixedLocator(spec["minor"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=TICK_LEN_PT * 0.6, width=0.6, color=AXIS)


def strip_panel(ax, p: int, data: dict, rows, spec: dict, title: str, mae_x: float, decimals: int) -> dict:
    """One panel of dot strips; returns what was drawn per row key, for the checks."""
    set_symlog(ax, spec)
    ax.tick_params(axis="y", length=0, pad=YPAD_PT, labelleft=(p == 0))
    ax.set_title(title, fontsize=LABEL_PT, linespacing=1.15, pad=TITLE_PAD_PT)
    ax.axvline(0.0, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=2)
    at_row = blend(ax.transAxes, ax.transData)
    ax.text(mae_x, HEADER_PT, "MAE", transform=at_row, ha="right", va="center", fontsize=MAE_PT, color=INK2)
    drawn = {}
    for r, ((label, versions, colour), y) in enumerate(rows):
        present = [(key, style) for key, style in versions if data.get(key) is not None]
        if not present:  # the pixel CNN exists only for the image setting: row left blank, no guide, no MAE
            drawn.update({key: None for key, _ in versions})
            continue
        if len(present) != len(versions):
            raise ValueError(f"{title}: {label} has only some of its versions")
        ax.axhline(y, color=HAIR, lw=0.4, zorder=0)
        split = len(versions) == 2
        for j, (key, style) in enumerate(versions):
            e = data[key]
            yc = y + (SPLIT_PT if style == FILLED else -SPLIT_PT) if split else y
            jitter = np.random.default_rng(1000 * p + 10 * r + j).uniform(-JITTER_PT[split], JITTER_PT[split], len(e))
            face, edge, width = style_of(colour, style)
            dots = ax.scatter(e, yc + jitter, s=DOT_S, facecolors=face, edgecolors=edge, linewidths=width,
                              zorder=3, clip_on=False)
            half = TICK_HALF_PT[split]
            ax.plot([e.mean()] * 2, [yc - half, yc + half], color=INK, lw=1.0, solid_capstyle="butt", zorder=2.5,
                    clip_on=False, gid="mark")
            ax.text(mae_x, yc, mae_text(e, decimals), transform=at_row, ha="right", va="center", fontsize=MAE_PT,
                    color=INK2)
            drawn[key] = dots
    return drawn


def row_labels(ax, rows) -> None:
    ax.set_yticks([y for _, y in rows])
    ax.set_yticklabels([display(label) for (label, _, _), _ in rows], linespacing=1.0)


def separators(fig, ax, seps: list, x_end: float) -> None:
    across = blend(fig.transFigure, ax.transData)
    for y in seps:
        fig.add_artist(Line2D([MARGIN / WIDTH, x_end / WIDTH], [y, y], transform=across, color=AXIS, lw=0.5))


def note_line(fig, ruler: Ruler, height: float, note_h: float, right_notes: list[str]) -> None:
    """The legend at the left, the notes right-aligned at the right, on one line at the bottom.  The legend's
    markers are the data's own orange dots, a little larger."""
    x = MARGIN
    for style, text in LEGEND:
        face, edge, width = style_of(ORANGE, style)
        size = np.sqrt(DOT_S) * 1.15
        fig.add_artist(Line2D([(x + size / 144) / WIDTH], [(MARGIN + note_h / 2) / height], marker="o",
                              markersize=size, markerfacecolor=face, markeredgecolor=edge, markeredgewidth=width,
                              linestyle="none", transform=fig.transFigure, gid="legend"))
        x += size / 72 + 2.0 / 72
        fig.text(x / WIDTH, MARGIN / height, text, ha="left", va="bottom", fontsize=NOTE_PT, color=INK2)
        x += ruler(text, fontsize=NOTE_PT)[0] + 8.0 / 72
    x = WIDTH - MARGIN
    for text in reversed(right_notes):
        fig.text(x / WIDTH, MARGIN / height, text, ha="right", va="bottom", fontsize=NOTE_PT, color=INK2)
        x -= ruler(text, fontsize=NOTE_PT)[0] + 10.0 / 72


def label_width(ruler: Ruler, groups) -> float:
    return max(ruler(display(label), fontsize=LABEL_PT, linespacing=1.0)[0]
               for group in groups for label, _, _ in group)


# ----------------------------------------------------------------------------- figure A

def draw_synthetic(errors: dict) -> tuple[plt.Figure, dict, dict]:
    rows, seps = positions(SYNTH_GROUPS)
    maes = [mae_text(e, 3) for lv in errors.values() for e in lv.values() if e is not None]
    right = []  # the post-freeze footnote was removed at the author's request (2026-09-23)
    titles = [t for _, t in SYNTH_PANELS]
    ruler = Ruler()
    g = vertical(ruler, rows, titles, [t for _, t in LEGEND] + right)
    g["label_w"] = label_width(ruler, SYNTH_GROUPS)
    g["mae_col"] = MAE_PAD + max(ruler(t, fontsize=MAE_PT)[0] for t in maes + ["MAE"])
    height = g["height"]
    fig = plt.figure(figsize=(WIDTH, height))
    left = MARGIN + g["label_w"] + YPAD_PT / 72
    n = len(SYNTH_PANELS)
    g["left"] = left
    g["data_w"] = (WIDTH - MARGIN - left - n * g["mae_col"] - (n - 1) * GAP) / n
    axes, drawn = [], {}
    for p, (level, title) in enumerate(SYNTH_PANELS):
        x0 = left + p * (g["data_w"] + g["mae_col"] + GAP)
        ax = fig.add_axes([x0 / WIDTH, g["bottom"] / height, g["data_w"] / WIDTH, g["axes_h"] / height],
                          sharey=axes[0] if axes else None)
        axes.append(ax)
        mae_x = 1 + g["mae_col"] / g["data_w"]
        drawn[level] = strip_panel(ax, p, errors[level], rows, SYNTH_AXIS, title, mae_x, 3)
    axes[0].set_ylim(*g["ylim"])
    row_labels(axes[0], rows)
    separators(fig, axes[0], seps, WIDTH - MARGIN)
    x_mid = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    fig.text(x_mid, g["xlabel_y"] / height, v3.XLABEL, ha="center", va="bottom", fontsize=LABEL_PT)
    note_line(fig, ruler, height, g["note_h"], right)
    ruler.close()
    g["axes"] = axes
    return fig, drawn, g


# ----------------------------------------------------------------------------- figure B

def rhc_positions(rows) -> list[float]:
    """The RHC marks, evenly spaced from just below the header to the last strip row."""
    top = HEADER_PT + TOP_PT - RHC_LIFT_PT - RHC_LABEL_PT * 1.25 - 1.0
    return list(np.linspace(top, rows[-1][1], len(RHC_ROWS)))


def draw_rhc(ax, rhc: dict, ys: list[float]) -> dict:
    ax.set_xlim(*RHC_XLIM)
    ax.xaxis.set_major_locator(FixedLocator(RHC_TICKS))
    ax.xaxis.set_major_formatter(FixedFormatter([minus(t) for t in RHC_TICKS]))
    ax.set_yticks([])
    ax.tick_params(axis="y", length=0, labelleft=False)
    ax.set_title(RHC_TITLE, fontsize=LABEL_PT, pad=TITLE_PAD_PT)
    at_row = blend(ax.transAxes, ax.transData)
    drawn = {}
    for (label, key, colour), y in zip(RHC_ROWS, ys):
        ax.axhline(y, color=HAIR, lw=0.4, zorder=0)
        ax.text(0.0, y + RHC_LIFT_PT, display(label), transform=at_row, ha="left", va="bottom",
                fontsize=RHC_LABEL_PT, linespacing=1.0, color=INK)
        if key == "kspc":
            values = rhc["kspc"]
            ax.plot([min(values), max(values)], [y, y], color=colour, lw=0.7, solid_capstyle="butt", zorder=2,
                    gid="mark")
            for v in values:
                ax.plot([v, v], [y - 2.4, y + 2.4], color=colour, lw=0.9, solid_capstyle="butt", zorder=3,
                        gid="mark")
            drawn[key] = list(values)
            continue
        value, interval = rhc[key]
        if interval is not None:
            ax.plot(list(interval), [y, y], color=colour, lw=1.0, solid_capstyle="butt", zorder=2, gid="mark")
        ax.scatter([value], [y], s=RHC_DOT_S, facecolors=colour, edgecolors="white", linewidths=0.4, zorder=3)
        drawn[key] = [value] + ([] if interval is None else list(interval))
    return drawn


def draw_realdata(errors: dict, rhc: dict) -> tuple[plt.Figure, dict, dict]:
    rows, seps = positions(REAL_GROUPS)
    right = [PUBLISHED_NOTE]  # post-freeze footnote removed at the author's request
    titles = [t for _, t, _ in REAL_PANELS] + [RHC_TITLE]
    ruler = Ruler()
    g = vertical(ruler, rows, titles, [t for _, t in LEGEND] + right)
    g["label_w"] = label_width(ruler, REAL_GROUPS)
    g["mae_col"] = [MAE_PAD + max(ruler(t, fontsize=MAE_PT)[0] for t in
                                  [mae_text(e, 4) for e in errors[name].values()] + ["MAE"])
                    for name, _, _ in REAL_PANELS]
    g["rhc_w"] = max(RHC_MIN_W, ruler(RHC_TITLE, fontsize=LABEL_PT)[0],
                     max(ruler(display(label), fontsize=RHC_LABEL_PT, linespacing=1.0)[0]
                         for label, _, _ in RHC_ROWS) + 0.03)
    height = g["height"]
    fig = plt.figure(figsize=(WIDTH, height))
    left = MARGIN + g["label_w"] + YPAD_PT / 72
    n = len(REAL_PANELS)
    g["left"] = left
    g["data_w"] = (WIDTH - MARGIN - left - sum(g["mae_col"]) - (n - 1) * GAP - RHC_GAP - g["rhc_w"]) / n
    axes, drawn, x0 = [], {}, left
    for p, (name, title, _) in enumerate(REAL_PANELS):
        ax = fig.add_axes([x0 / WIDTH, g["bottom"] / height, g["data_w"] / WIDTH, g["axes_h"] / height],
                          sharey=axes[0] if axes else None)
        axes.append(ax)
        mae_x = 1 + g["mae_col"][p] / g["data_w"]
        drawn[name] = strip_panel(ax, p, errors[name], rows, REAL_AXIS[name], title, mae_x, 4)
        x0 += g["data_w"] + g["mae_col"][p] + (GAP if p < n - 1 else 0.0)
    axes[0].set_ylim(*g["ylim"])
    row_labels(axes[0], rows)
    separators(fig, axes[0], seps, x0)
    x_rhc = WIDTH - MARGIN - g["rhc_w"]
    ax_r = fig.add_axes([x_rhc / WIDTH, g["bottom"] / height, g["rhc_w"] / WIDTH, g["axes_h"] / height])
    ax_r.set_ylim(*g["ylim"])
    drawn["rhc"] = draw_rhc(ax_r, rhc, rhc_positions(rows))
    x_mid = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    fig.text(x_mid, g["xlabel_y"] / height, REAL_XLABEL, ha="center", va="bottom", fontsize=LABEL_PT)
    r_mid = (ax_r.get_position().x0 + ax_r.get_position().x1) / 2
    fig.text(r_mid, g["xlabel_y"] / height, RHC_XLABEL, ha="center", va="bottom", fontsize=LABEL_PT)
    note_line(fig, ruler, height, g["note_h"], right)
    ruler.close()
    g["axes"], g["rhc_ax"], g["rhc_gap_actual"] = axes, ax_r, x_rhc - x0
    return fig, drawn, g


# ----------------------------------------------------------------------------- checks

def version_names(groups):
    for group in groups:
        for label, versions, _ in group:
            for key, _ in versions:
                yield (label.replace("$", "") + (f", {VERSION[key]}" if key in VERSION else "")), key


def check_points(errors: dict, available: dict, drawn: dict, panels: list[str], axis_of, groups) -> list[str]:
    """Points drawn against data sets available, every drawn point inside the x axis, the values unchanged."""
    problems = []
    print(f"  {'row':50s}" + "".join(f"{p:>10s}" for p in panels))
    for name, key in version_names(groups):
        cells = []
        for panel in panels:
            e, dots = errors[panel][key], drawn[panel][key]
            if e is None:
                cells.append("n/a")
                if dots is not None:
                    problems.append(f"{panel} {key}: drew points where the row does not apply")
                continue
            xy = np.asarray(dots.get_offsets())
            lo, hi = axis_of(panel)["xlim"]
            inside = int(np.sum((xy[:, 0] >= lo) & (xy[:, 0] <= hi)))
            cells.append(f"{len(xy)}/{available[panel]}")
            if len(xy) != available[panel] or inside != len(xy) or not np.allclose(np.sort(xy[:, 0]), np.sort(e)):
                problems.append(f"{panel} {key}: drew {len(xy)}, inside {inside}, available {available[panel]}")
        print(f"  {name:50s}" + "".join(f"{c:>10s}" for c in cells))
    for panel in panels:
        values = np.concatenate([e for e in errors[panel].values() if e is not None])
        lo, hi = axis_of(panel)["xlim"]
        print(f"  {panel}: data from {values.min():+.4f} to {values.max():+.4f}, axis {lo:g} to {hi:g}, "
              f"linear within +-{axis_of(panel)['linthresh']:g}")
    return problems


def check_mae(errors: dict, panels: list[str], index: dict, decimals: int, groups) -> list[str]:
    """Every MAE the figure prints beside the index, cell by cell."""
    problems, shown = [], set()
    print(f"  {'row':50s}" + "".join(f"{p:>20s}" for p in panels))
    for name, key in version_names(groups):
        cells = []
        for panel in panels:
            e, ref = errors[panel][key], index.get((panel, key))
            if e is None:
                cells.append("blank" + ("" if ref is None else f" ({ref})"))
                if ref is not None:
                    problems.append(f"{panel} {name}: blank in the figure, {ref} in the index")
                continue
            shown.add((panel, key))
            mine = mae_text(e, decimals)
            cells.append(f"{mine} ({ref if ref is not None else 'none'})")
            if ref is None:
                problems.append(f"{panel} {name}: {mine} in the figure, no cell in the index")
            elif ref != mine:
                problems.append(f"{panel} {name}: MISMATCH figure {mine}, index {ref}")
        print(f"  {name:50s}" + "".join(f"{c:>20s}" for c in cells))
    for cell in sorted(set(index) - shown):
        problems.append(f"index cell {cell} = {index[cell]} is not in the figure")
    return problems


def check_rhc(rhc: dict, drawn: dict) -> list[str]:
    """Every RHC value drawn beside the E3 table of the index, at its 2 decimals, and inside the axis."""
    index, problems = index_rhc(), []
    probe, (lo, hi) = rhc["probe"]
    mine = {"probe": [probe, lo, hi], "standard_dr": [rhc["standard_dr"][0]], "proximal_dr": [rhc["proximal_dr"][0]],
            "crude": [rhc["crude"][0]], "kspc": [max(rhc["kspc"]), min(rhc["kspc"])]}  # the index's order
    print(f"  {'row':32s}{'points':>8s}   figure (EXPERIMENT_INDEX.md)")
    for label, key, _ in RHC_ROWS:
        a = [f"{v:.2f}" for v in mine[key]]
        b = [f"{v:.2f}" for v in index[key]]
        n = len(drawn[key]) if key == "kspc" else 1
        print(f"  {label:32s}{n:>8d}   {', '.join(a)} ({', '.join(b)})")
        if a != b:
            problems.append(f"RHC {label}: MISMATCH figure {a}, index {b}")
        xs = drawn[key]
        if not all(RHC_XLIM[0] <= v <= RHC_XLIM[1] for v in xs):
            problems.append(f"RHC {label}: a value lies outside the axis {RHC_XLIM}")
    if len(rhc["kspc"]) != 5:
        problems.append(f"RHC KSPC: {len(rhc['kspc'])} blocks, not 5")
    print(f"  KSPC blocks W1 to W5: {', '.join(f'{v:.3f}' for v in rhc['kspc'])}; difference in means "
          f"{rhc['crude'][0]:.4f} (its standard error {rhc['crude_se']:.3f} is not drawn)")
    return problems


def mark_collisions(fig) -> list[str]:
    """Text boxes that cover a drawn data mark: a dot, a mean tick, an interval, a range, a legend marker."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        for coll in ax.collections:
            xy = ax.transData.transform(np.asarray(coll.get_offsets()))
            r = np.sqrt(coll.get_sizes()) / 2 * fig.dpi / 72
            r = np.broadcast_to(r if r.size else np.zeros(1), (len(xy),))
            boxes.append(np.column_stack([xy[:, 0] - r, xy[:, 1] - r, xy[:, 0] + r, xy[:, 1] + r]))
        for line in ax.lines:
            if line.get_gid() == "mark":
                bb = line.get_window_extent(renderer)
                boxes.append(np.array([[bb.x0, bb.y0, bb.x1, bb.y1]]))
    for artist in fig.artists + fig.lines:
        if artist.get_gid() == "legend":
            bb = artist.get_window_extent(renderer)
            boxes.append(np.array([[bb.x0, bb.y0, bb.x1, bb.y1]]))
    boxes = np.vstack(boxes)
    hits = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(renderer)
        if bb.width < 1 or bb.height < 1:
            continue
        over = (boxes[:, 0] < bb.x1 - 0.5) & (boxes[:, 2] > bb.x0 + 0.5) & (boxes[:, 1] < bb.y1 - 0.5) & \
               (boxes[:, 3] > bb.y0 + 0.5)
        if over.any():
            hits.append(f"text {t.get_text()!r} covers {int(over.sum())} data mark(s)")
    return hits


def check_canvas(fig, height: float) -> list[str]:
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    tol, problems = 0.005, []
    if bb.x0 < -tol or bb.y0 < -tol or bb.x1 > WIDTH + tol or bb.y1 > height + tol:
        problems.append(f"content leaves the {WIDTH} x {height:.3f} in page: {bb}")
    if height > MAX_HEIGHT + 1e-9:
        problems.append(f"height {height:.3f} in is above {MAX_HEIGHT} in")
    return problems


def save(fig, stem: str, outdir: Path) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    pdf, png = outdir / f"{stem}.pdf", outdir / f"{stem}.png"
    fig.savefig(pdf, metadata={"CreationDate": None})
    fig.savefig(png, dpi=200)
    return pdf, png


def finish(fig, stem: str, outdir: Path, height: float, problems: list[str]) -> tuple[Path, Path, list[str]]:
    """The layout checks at each of CHECK_DPI, then the pdf and the 200 dpi png."""
    for dpi in CHECK_DPI:
        fig.set_dpi(dpi)
        found = [f"text collision: {a!r} and {b!r}" for a, b in m.text_collisions(fig)]
        found += mark_collisions(fig) + check_canvas(fig, height)
        problems += [f"{p} (at {dpi} dpi)" for p in found]
    fig.set_dpi(CHECK_DPI[0])
    pdf, png = save(fig, stem, outdir)
    plt.close(fig)
    return pdf, png, problems


# ----------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outdir", type=Path, default=m.FIGDIR, help="where the pdf and png go")
    parser.add_argument("--no-copy", action="store_true", help="skip the copies to ICLR/paper/figures")
    args = parser.parse_args(argv)
    problems, written = [], []

    synth, synth_available, synth_kspc = load_synthetic()
    real, real_available, real_kspc = load_real()
    rhc, rhc_kspc = load_rhc()
    kspc_files = synth_kspc + real_kspc + rhc_kspc
    problems += sha_check(kspc_files)
    print(f"KSPC files read: {len(kspc_files)}, each checked against results/kspc_comparator/FROZEN_v1.json")

    with plt.rc_context(STYLE):
        print("\n== Figure A, E1 and E2: points drawn / data sets available (every point inside the x axis)")
        fig, drawn, g = draw_synthetic(synth)
        problems += check_points(synth, synth_available, drawn, v3.LEVELS, lambda _: SYNTH_AXIS, SYNTH_GROUPS)
        print("\n   MAE, 3 decimals: figure (EXPERIMENT_INDEX.md)")
        problems += check_mae(synth, v3.LEVELS, v3.index_mae(), 3, SYNTH_GROUPS)
        pdf, png, problems = finish(fig, STEMS["synthetic"][0], args.outdir, g["height"], problems)
        written.append(("synthetic", pdf, png))
        print(f"\n   {WIDTH} x {g['height']:.3f} in; label column {g['label_w']:.3f} in; data area "
              f"{g['data_w']:.3f} in and MAE column {g['mae_col']:.3f} in per panel")

        names = [name for name, _, _ in REAL_PANELS]
        print("\n== Figure B, E5 to E7: points drawn / replicates available (every point inside the x axis)")
        fig, drawn, g = draw_realdata(real, rhc)
        problems += check_points(real, real_available, drawn, names, lambda name: REAL_AXIS[name], REAL_GROUPS)
        print("\n   MAE, 4 decimals: figure (EXPERIMENT_INDEX.md)")
        problems += check_mae(real, names, index_mae_real(), 4, REAL_GROUPS)
        print("\n   E3, RHC: values drawn, 2 decimals: figure (EXPERIMENT_INDEX.md)")
        problems += check_rhc(rhc, drawn["rhc"])
        pdf, png, problems = finish(fig, STEMS["realdata"][0], args.outdir, g["height"], problems)
        written.append(("realdata", pdf, png))
        print(f"\n   {WIDTH} x {g['height']:.3f} in; label column {g['label_w']:.3f} in; data area "
              f"{g['data_w']:.3f} in per panel, MAE columns {', '.join(f'{w:.3f}' for w in g['mae_col'])} in; "
              f"RHC panel {g['rhc_w']:.3f} in after a {g['rhc_gap_actual']:.3f} in gap")

    print()
    for key, pdf, png in written:
        print(f"wrote {pdf}\nwrote {png} (200 dpi)")
        if not args.no_copy:
            target = PAPER_FIGURES / STEMS[key][1]
            if target.name not in ("probe-synthetic-v4.pdf", "probe-realdata-v4.pdf"):
                raise RuntimeError(f"refusing to write {target}")
            shutil.copyfile(pdf, target)
            print(f"copied to {target}")
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print("\nAll checks passed: every point drawn and inside its axis, every MAE and RHC value equal to the index, "
          "no text collisions, no text over a data mark, content inside the page, height within the limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
