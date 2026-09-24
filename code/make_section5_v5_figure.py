"""Figure 3, version 5: the synthetic and the semi-synthetic results in one row of dot strips (design A of the
version 5 prototypes, with less text and larger text).

    Seven panels, SCM-1 to SCM-4 | Twins, IHDP, ACIC, and one shared column of method names at the left.  Each dot is
    one data set's error, the estimate minus the known effect, on a symmetric-log axis; a black tick marks the mean.  A
    method that needs proxy roles takes one row: filled dots above its centre line use the correct roles, hollow dots
    below use wrong roles.  One legend line, one x label, three labelled ticks per panel.  PROBE's row stands apart,
    under a line of its own, and its MAE (2 significant digits) is written in gray under each panel title; no other
    MAE is printed.  "NA" in gray marks a panel where a method was not run (the pixel CNN outside SCM-4).

Data: the data behind the version 4 Figure 3, through make_section5_v4_extension's own loaders: pooled SCM-1 to
SCM-4, 100 data sets each; pooled Twins, 100, and IHDP, 88; ACIC, 9 realizations.  RHC is left out.  Nothing is
refitted or rerun, and nothing under results/ is written.

Checks printed on every run:
    points   every panel and row draws all of its data sets, the values unchanged, every point inside its x axis
    MAE      PROBE's drawn, the others not, all checked against the data: every row's pooled MAE against
             results/extension_v1/summary_v1.json (3 decimals for SCM, 4 for Twins and IHDP, and at full precision),
             the frozen index for ACIC (4 decimals), and the numbers the version 4 figure prints (read from its pdfs
             with pdftotext); then make_section5_v4_figures.check_mae, as make_section5_v4_extension runs it.  The
             2-digit numbers drawn for PROBE are checked against a decimal half-up rounding of the full value, of
             summary_v1.json's value, and of the version 4 figure's number
    layout   at 200 and 800 dpi: text collisions, text over a data mark, content inside the page, the height limit,
             the font sizes, at most three tick labels per panel, and the clear space between neighbouring texts

Writes figures/section5/section5-v5-experiments.{pdf,png} (png at 200 dpi) and, when every check passes, copies the
pdf to ICLR/paper/figures/probe-experiments-v5.pdf.
    python3 -B make_section5_v5_figure.py [--outdir DIR] [--no-copy]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # code/__pycache__ is stale: importing the other scripts must not rewrite it

import argparse
import json
import re
import shutil
import subprocess
import warnings
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory as blend

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)  # an old docstring's escape sequence in v2, as in v3 and v4
    import make_section5_v4_extension as ext  # imports make_section5_v4_figures (v4) and summarise_extension_v1

v4 = ext.v4
ROOT = v4.ROOT
FIGDIR = v4.m.FIGDIR                                     # figures/section5
STEM = "section5-v5-experiments"
PAPER_COPY = v4.PAPER_FIGURES / "probe-experiments-v5.pdf"
LEVELS = list(v4.v3.LEVELS)
REAL = ["twins", "ihdp", "acic"]
CURRENT_PDF = {"synthetic": v4.PAPER_FIGURES / "probe-synthetic-v4-ext.pdf",   # the version 4 Figure 3
               "realdata": v4.PAPER_FIGURES / "probe-realdata-v4-ext.pdf"}

# ----------------------------------------------------------------------------- style

BLUE, GREEN, GRAY, ORANGE, DARK = v4.BLUE, v4.GREEN, v4.GRAY, v4.ORANGE, v4.DARK
INK, INK2, HAIR, AXIS = v4.INK, v4.INK2, v4.HAIR, v4.AXIS
FILLED, HOLLOW = v4.FILLED, v4.HOLLOW

# Text, in points: one size for every word, one for the tick numbers, the largest that fit the 5.5 in width without
# crowding.  Larger names widen the name column and so narrow the seven panels: at 11 pt the panel titles fill 96% of
# their panels and the tick numbers must drop to 8 pt; at 10.5 pt the titles fill 88%, and 8.5 pt tick numbers keep
# at least 1.7 pt between "-10" and "0" on IHDP, the tightest pair (9 pt leaves 1.2 pt).
TEXT_PT = 10.5     # method names, panel titles, group titles, legend, x label
TICK_PT = 8.5      # tick labels
MIN_TEXT_PT, MIN_TICK_PT = 9.0, 8.0  # the floors asked for, checked on every text drawn
MIN_GAP_PT = 1.5   # the least clear space between the boxes of two texts, checked (the ink gap is wider)

# Geometry, inches unless a name says pt.  5.5 in is the paper's text width; the figure is placed at 100%.
WIDTH, MAX_HEIGHT = 5.5, 3.0              # 2.9 before PROBE's MAE line and the line under PROBE were added
MARGIN = 0.02
PANEL_GAP, GROUP_GAP = 0.10, 0.22          # between panels, and between the synthetic and the semi-synthetic group
YPAD_PT = 6.0                              # from a method name to the first panel (v4: 4), clear of the clusters at
                                           # -5, which sit 1 pt inside SCM-1's left edge
TICK_LEN_PT, TICK_PAD_PT = v4.TICK_LEN_PT, v4.TICK_PAD_PT
LINE_GAP_PT = 2.0                          # from the legend and x label line up to the tick labels
TITLE_PAD_PT, RULE_PAD_PT = 2.0, 1.0       # panels to PROBE's MAE; titles to the group rule, rule to group title
MAE_GAP_PT = 2.0                           # PROBE's MAE to the panel titles
NOTE = GRAY                                # PROBE's MAE and "NA"
MAE_LABEL, NA = "PROBE’s MAE", "NA"

# Rows, in points: the y data unit of every panel is one point, so these are exact on paper.  Set for 10 pt names
# and scaled with TEXT_PT, so that changing TEXT_PT alone keeps the names clear of each other.
ROW_SCALE = TEXT_PT / 10.0
ONE_PT, TWO_PT = 13.0 * ROW_SCALE, 17.5 * ROW_SCALE  # pitch of a row with one version, and with a correct and a wrong
SEP_PT, PAD_PT = 3.0 * ROW_SCALE, 1.5 * ROW_SCALE    # extra space at a separator; above the first row, below the last
SPLIT_PT = 3.6 * ROW_SCALE                 # filled dots this far above the row centre, hollow dots this far below
JITTER_PT = {False: 1.8, True: 1.0}        # vertical jitter half-width: one-version row, one band of a split row
TICK_HALF_PT = {False: 5.0, True: 3.0}     # the mean tick's half-height, drawn under the dots
DOT_S = 9.0                                # dot area, pt^2 (3 pt across)
LEGEND_DOT_PT = 4.5                        # legend marker diameter, about the x-height of the legend text

# (panel key, title, kind); a gap and a group title split the two kinds
PANELS = [(lv, lv, "synth") for lv in LEVELS] + [("twins", "Twins", "real"), ("ihdp", "IHDP", "real"),
                                                 ("acic", "ACIC", "real")]
COLUMN_GROUPS = [("Synthetic", 0, 4), ("Semi-synthetic", 4, 7)]

# Methods, in groups split by a thin line: (name, colour, versions); a version is (style, synthetic key, real key),
# the correct roles first.  None: not run on that kind of data.  PROBE is a group of its own.
METHODS = [
    [("PROBE (ours)", BLUE, [(FILLED, "probe", "probe")])],
    [("CEVAE", GREEN, [(FILLED, "cevae", "cevae")]),
     ("Adjust for $(X,W)$", GRAY, [(FILLED, "raw", "raw")]),
     ("Adjust for $X$", GRAY, [(FILLED, "x", "X")]),
     ("Pixel CNN", GRAY, [(FILLED, "pixel", None)])],
    [("Proximal 2SLS", ORANGE, [(FILLED, "p2sls_given", "p2sls_given_roles"),
                                (HOLLOW, "p2sls_swapped", "p2sls_swapped_roles")]),
     ("Single-proxy control", ORANGE, [(FILLED, "park_clean", "coca"), (HOLLOW, "park_contaminated", None)]),
     ("KSPC", ORANGE, [(FILLED, "kspc_clean", "kspc_W1"), (HOLLOW, "kspc_contaminated", "kspc_W4")])],
    [("Oracle", DARK, [(FILLED, "oracle", "oracle")])],
]
ROLE = {FILLED: "correct roles", HOLLOW: "wrong roles"}
XLABEL = "estimate minus true effect"

# v4's symmetric-log axes (linear threshold and limits, widened as in make_section5_v4_extension).  Three labels each,
# -a, 0, a, and unlabelled minor ticks one decade inside and outside, so the log scale shows without more text.
AXES = {"synth": {"linthresh": 0.1, "xlim": v4.SYNTH_AXIS["xlim"], "ticks": [-1.0, 0.0, 1.0],
                  "minor": [-10.0, -0.1, 0.1, 10.0]},
        "twins": {"linthresh": 0.01, "xlim": v4.REAL_AXIS["twins"]["xlim"], "ticks": [-0.1, 0.0, 0.1],
                  "minor": [-0.01, 0.01, 1.0]},
        "ihdp": {"linthresh": 0.2, "xlim": v4.REAL_AXIS["ihdp"]["xlim"], "ticks": [-10.0, 0.0, 10.0],
                 "minor": [-100.0, -1.0, 1.0, 100.0]},
        "acic": {"linthresh": 0.2, "xlim": v4.REAL_AXIS["acic"]["xlim"], "ticks": [-10.0, 0.0, 10.0],
                 "minor": [-100.0, -1.0, 1.0, 100.0]}}
MAX_TICK_LABELS = 3


def style() -> dict:
    """v4's rc style (Times, STIX math, TrueType fonts in the pdf, v3's colours) at this figure's sizes."""
    return {**v4.STYLE, "font.size": TEXT_PT, "axes.titlesize": TEXT_PT, "axes.labelsize": TEXT_PT,
            "xtick.labelsize": TICK_PT, "ytick.labelsize": TEXT_PT}


def axis_of(panel: str, kind: str) -> dict:
    return AXES["synth" if kind == "synth" else panel]


def key_of(version, kind: str):
    return version[1] if kind == "synth" else version[2]


def errors_of(data: dict, panel: str, kind: str, version) -> np.ndarray | None:
    key = key_of(version, kind)
    return None if key is None else data[panel].get(key)


def mae(e: np.ndarray) -> float:
    return float(np.mean(np.abs(e)))


def sig2(v: float | str) -> str:
    """v to 2 significant digits, decimal half-up on its exact value (a float's binary value, a string's digits),
    never in exponent form (0.0073, 0.019, 0.41, 1.4, 25)."""
    d = Decimal(v)
    places = 1 - d.copy_abs().adjusted()          # adjusted(): the exponent of the leading digit
    r = d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    if r.copy_abs().adjusted() > d.copy_abs().adjusted():  # 0.0995 -> 0.10: one place fewer keeps 2 digits
        r = r.quantize(Decimal(1).scaleb(-(places - 1)), rounding=ROUND_HALF_UP)
    return f"{r:f}"


# ----------------------------------------------------------------------------- data and MAE checks

def load() -> tuple[dict, dict]:
    synth, synth_n = ext.load_synthetic()
    real, real_n = ext.load_real()
    return {**synth, **real}, {**synth_n, **real_n}


def reference_cells() -> dict:
    """What make_section5_v4_extension checks its MAE against: summary_v1.json pooled (SCM 3 decimals, Twins and
    IHDP 4 decimals) and the frozen index for ACIC (4 decimals)."""
    cells = ext.pooled_mae_cells(3, None, {lv: lv for lv in LEVELS})
    cells = {k: v for k, v in cells.items() if not (k[1] == "pixel" and k[0] != "SCM-4")}
    real = ext.pooled_mae_cells(4, ext.REAL_KEY, ext.REAL_EXP)
    real.update({k: v for k, v in v4.index_mae_real().items() if k[0] == "acic"})
    return {**cells, **real}


def reference_full() -> dict:
    """summary_v1.json's pooled MAE at full precision (SCM-1 to SCM-4, Twins, IHDP); the index gives ACIC to 4
    decimals only."""
    s = json.loads(ext.SUMMARY.read_text())["experiments"]
    out = {(lv, key): b["pooled"]["mae"] for lv in LEVELS for key, b in s[lv].items() if b["pooled"]["mae"] is not None}
    out.update({(name, mine): s[exp][skey]["pooled"]["mae"] for name, exp in ext.REAL_EXP.items()
                for mine, skey in ext.REAL_KEY.items()})
    return out


def current_figure_cells(data: dict) -> tuple[dict, str | None]:
    """The MAE numbers the version 4 Figure 3 prints, read from its two pdfs: numbers in each MAE column, top to
    bottom, in v4's row order (correct roles above wrong roles).  Returns the cells and, if they could not be read,
    why not."""
    if shutil.which("pdftotext") is None:  # poppler's command-line tool, only for this read-only cross-check
        return {}, "pdftotext not found"
    missing = [str(p) for p in CURRENT_PDF.values() if not p.exists()]
    if missing:
        return {}, f"not found: {', '.join(missing)}"
    orders = {"synthetic": (LEVELS, 3, [k for _, versions, _ in sum(v4.SYNTH_GROUPS, []) for k, _ in versions]),
              "realdata": (REAL, 4, [k for _, versions, _ in sum(v4.REAL_GROUPS, []) for k, _ in versions])}
    cells = {}
    for which, (panels, dp, order) in orders.items():
        txt = subprocess.run(["pdftotext", "-bbox", str(CURRENT_PDF[which]), "-"], capture_output=True, text=True,
                             check=True).stdout
        words = re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="[\d.]+">([^<]*)</word>', txt)
        nums = [(float(x1), float(y0), w) for _, y0, x1, w in words if re.fullmatch(rf"\d+\.\d{{{dp}}}", w)]
        columns = []  # right edges of the MAE columns, left to right (the numbers are right-aligned)
        for x1 in sorted(x for x, _, _ in nums):
            if not columns or x1 - columns[-1] > 3.0:
                columns.append(x1)
        if len(columns) != len(panels):
            raise ValueError(f"{which}: {len(columns)} MAE columns in the pdf, expected {len(panels)}")
        for panel, col in zip(panels, columns):
            values = [w for x1, _, w in sorted(nums, key=lambda t: t[1]) if abs(x1 - col) <= 3.0]
            keys = [k for k in order if data[panel].get(k) is not None]
            if len(values) != len(keys):
                raise ValueError(f"{which} {panel}: {len(values)} numbers in the pdf, {len(keys)} rows")
            cells.update({(panel, k): v for k, v in zip(keys, values)})
    return cells, None


def mae_table(data: dict) -> tuple[list[str], str | None]:
    """Every row's MAE beside the reference, the version 4 figure and summary_v1.json; returns the problems and, if
    the version 4 figure could not be read, why not."""
    ref, full_ref = reference_cells(), reference_full()
    cur, skipped = current_figure_cells(data)
    problems, n_full = [], 0
    print("MAE of every row (only PROBE's drawn; all checked): full value from the data; 'check' at the version 4 figure's decimals; "
          "'reference' what make_section5_v4_extension\nchecks it against; 'v4 figure' the number that figure prints; "
          "'summary' = equal to summary_v1.json's full-precision value (idx: ACIC, index to 4 decimals only)")
    if skipped:
        print(f"  NOTE: the version 4 figure was not read ({skipped}); its column is 'none' and not compared")
    print(f"\n  {'panel':7s} {'method':36s} {'MAE (full)':>11s} {'check':>8s} {'reference':>10s} {'v4 figure':>10s}"
          f" {'summary':>8s}")
    for panel, _, kind in PANELS:
        dp = 3 if kind == "synth" else 4
        for name, _, versions in sum(METHODS, []):
            for version in versions:
                e = errors_of(data, panel, kind, version)
                if e is None:
                    continue
                key = key_of(version, kind)
                full = mae(e)
                check = f"{full:.{dp}f}"
                r, c = ref.get((panel, key), "none"), cur.get((panel, key), "none")
                f_ref = full_ref.get((panel, key))
                same_full = "idx" if f_ref is None else ("yes" if abs(full - f_ref) <= 1e-9 * max(1.0, f_ref) else "NO")
                n_full += same_full == "yes"
                label = name.replace("$", "") + (f", {ROLE[version[0]]}" if len(versions) == 2 else "")
                bad = (check != r or (bool(cur) and c != check) or same_full == "NO"
                       or (same_full == "idx" and panel != "acic"))
                if bad:
                    problems.append(f"{panel} {label}: MAE {check}, reference {r}, v4 figure {c}, summary {same_full}")
                print(f"  {panel:7s} {label:36s} {full:11.6f} {check:>8s} {r:>10s} {c:>10s} {same_full:>8s}"
                      + ("  <-- MISMATCH" if bad else ""))
    matched = sum(1 for k in cur if k in ref)
    print(f"\n  {len(cur)} numbers read from the version 4 figure, {matched} with a reference cell, {n_full} equal to "
          f"summary_v1.json at full precision; {len(problems)} mismatches")
    drawn_keys = {(p, key_of(v, k)) for p, _, k in PANELS for _, _, vs in sum(METHODS, []) for v in vs}
    problems += [f"reference cell {m} = {ref[m]} has no drawn row" for m in sorted(set(ref) - drawn_keys)]
    return problems, skipped


def probe_mae_shown(data: dict) -> dict:
    """The number drawn under each panel title: PROBE's pooled MAE to 2 significant digits."""
    return {panel: sig2(mae(errors_of(data, panel, kind, METHODS[0][0][2][0]))) for panel, _, kind in PANELS}


def probe_mae_check(data: dict) -> list[str]:
    """PROBE's drawn MAE against the same rounding of summary_v1.json's full value (must match), and of the
    3- or 4-decimal reference and version 4 figure numbers (a difference there is double rounding: reported, not
    failed)."""
    shown, full_ref, ref = probe_mae_shown(data), reference_full(), reference_cells()
    cur, _ = current_figure_cells(data)
    problems = []
    print(f"\n  {'panel':7s} {'MAE (full)':>11s} {'drawn':>7s} {'summary':>8s} {'reference':>10s} {'v4 figure':>10s}")
    for panel, _, kind in PANELS:
        full = mae(errors_of(data, panel, kind, METHODS[0][0][2][0]))
        f_ref = full_ref.get((panel, "probe"))
        from_summary = "idx" if f_ref is None else sig2(f_ref)
        from_ref = sig2(ref[(panel, "probe")]) if (panel, "probe") in ref else "none"
        from_cur = sig2(cur[(panel, "probe")]) if (panel, "probe") in cur else "none"
        bad = shown[panel] != sig2(full) or (from_summary != "idx" and from_summary != shown[panel]) \
            or (from_summary == "idx" and panel != "acic")
        notes = [n for n, v in (("reference", from_ref), ("v4 figure", from_cur)) if v not in ("none", shown[panel])]
        if bad:
            problems.append(f"{panel} PROBE: drawn MAE {shown[panel]}, from summary_v1.json {from_summary}")
        print(f"  {panel:7s} {full:11.6f} {shown[panel]:>7s} {from_summary:>8s} {from_ref:>10s} {from_cur:>10s}"
              + ("  <-- MISMATCH" if bad else "") + (f"  (double rounding differs: {', '.join(notes)})" if notes else ""))
    return problems


# ----------------------------------------------------------------------------- layout helpers

def T(fig, x: float, y: float, s: str, **kw):
    """Text placed in inches from the lower-left corner."""
    return fig.text(x, y, s, transform=fig.dpi_scale_trans, **kw)


def hline(fig, x0: float, x1: float, y: float, **kw) -> None:
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.dpi_scale_trans, **kw))


def descent(fontsize: float) -> float:
    """Inches from the bottom of a one-line text box to its baseline (matplotlib's box keeps room for 'p')."""
    fig = plt.figure(figsize=(1.0, 1.0), dpi=800)
    t = fig.text(0.0, 0.5, "correct roles", fontsize=fontsize, va="baseline")
    fig.canvas.draw()
    d = (0.5 * fig.dpi - t.get_window_extent(fig.canvas.get_renderer()).y0) / fig.dpi
    plt.close(fig)
    return d


def row_layout() -> tuple[list, list, float, float]:
    """Row centres in points, 0 at the first and negative below; separators; the y limits."""
    rows, seps, y, prev = [], [], 0.0, None
    for members in METHODS:
        for i, method in enumerate(members):
            half = (TWO_PT if len(method[2]) == 2 else ONE_PT) / 2
            if prev is not None:
                if i == 0:
                    seps.append(y - prev - SEP_PT / 2)
                y -= prev + half + (SEP_PT if i == 0 else 0.0)
            rows.append((method, y))
            prev = half
    return rows, seps, rows[-1][1] - prev - PAD_PT, ONE_PT / 2 + PAD_PT


def set_symlog(ax, spec: dict) -> None:
    ax.set_xscale("symlog", linthresh=spec["linthresh"], linscale=1.0)
    ax.set_xlim(*spec["xlim"])
    ax.xaxis.set_major_locator(FixedLocator(spec["ticks"]))
    ax.xaxis.set_major_formatter(FixedFormatter([v4.minus(t) for t in spec["ticks"]]))
    lo, hi = spec["xlim"]
    ax.xaxis.set_minor_locator(FixedLocator([t for t in spec["minor"] if lo < t < hi]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelsize=TICK_PT, length=TICK_LEN_PT, pad=TICK_PAD_PT)
    ax.tick_params(axis="x", which="minor", length=TICK_LEN_PT * 0.6, width=0.5, color=AXIS)


# ----------------------------------------------------------------------------- the figure

def draw(data: dict) -> tuple[plt.Figure, dict, dict]:
    ruler = v4.Ruler()  # text extents, the larger of the 200 and the 800 dpi rendering
    rows, seps, lo, hi = row_layout()
    names = [m[0] for m, _ in rows]
    shown = probe_mae_shown(data)
    label_col = max(ruler(n, fontsize=TEXT_PT)[0] for n in names + [MAE_LABEL])
    h_title = max(ruler(t, fontsize=TEXT_PT)[1] for _, t, _ in PANELS)
    h_mae = max(ruler(s, fontsize=TEXT_PT)[1] for s in [MAE_LABEL, *shown.values()])
    h_group = max(ruler(t, fontsize=TEXT_PT, style="italic")[1] for t, _, _ in COLUMN_GROUPS)
    h_tick = max(ruler(v4.minus(t), fontsize=TICK_PT)[1] for spec in AXES.values() for t in spec["ticks"])
    h_line = max(ruler(s, fontsize=TEXT_PT)[1] for s in [XLABEL, *ROLE.values()])

    # vertical, bottom to top: legend and x label, tick labels, panels, PROBE's MAE, panel titles, group rule, group
    # titles
    bottom = MARGIN + h_line + LINE_GAP_PT / 72 + h_tick + (TICK_LEN_PT + TICK_PAD_PT) / 72
    axes_h = (hi - lo) / 72
    y_mae = bottom + axes_h + TITLE_PAD_PT / 72
    y_title = y_mae + h_mae + MAE_GAP_PT / 72
    y_rule = y_title + h_title + RULE_PAD_PT / 72
    y_group = y_rule + RULE_PAD_PT / 72
    height = y_group + h_group + MARGIN

    # horizontal: the name column, then seven panels of one width, a wider gap between the two groups
    left = MARGIN + label_col + YPAD_PT / 72
    panel_w = (WIDTH - MARGIN - left - 5 * PANEL_GAP - GROUP_GAP) / 7
    x0 = [left + i * (panel_w + PANEL_GAP) + (GROUP_GAP - PANEL_GAP if i >= 4 else 0.0) for i in range(7)]

    fig = plt.figure(figsize=(WIDTH, height))
    axes, drawn = [], {}
    for p, (panel, title, kind) in enumerate(PANELS):
        ax = fig.add_axes([x0[p] / WIDTH, bottom / height, panel_w / WIDTH, axes_h / height],
                          sharey=axes[0] if axes else None)
        axes.append(ax)
        set_symlog(ax, axis_of(panel, kind))
        ax.tick_params(axis="y", length=0, pad=YPAD_PT, labelleft=(p == 0))
        ax.axvline(0.0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=2)
        drawn[panel] = {}
        for r, ((name, colour, versions), y) in enumerate(rows):
            present = [v for v in versions if errors_of(data, panel, kind, v) is not None]
            for v in versions:
                if key_of(v, kind) is not None and v not in present:
                    drawn[panel][key_of(v, kind)] = None
            if not present:  # the pixel CNN exists only for SCM-4: elsewhere its row says NA, on the zero line
                fig.text(0.0, y, NA, transform=ax.transData, ha="center", va="center_baseline", fontsize=TEXT_PT,
                         color=NOTE, gid="na", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
                continue
            ax.axhline(y, color=HAIR, lw=0.4, zorder=0)
            split = len(versions) == 2
            for j, v in enumerate(present):
                e = errors_of(data, panel, kind, v)
                yc = y + (SPLIT_PT if v[0] == FILLED else -SPLIT_PT) if split else y
                jitter = np.random.default_rng(1000 * p + 10 * r + j).uniform(-JITTER_PT[split], JITTER_PT[split],
                                                                              len(e))
                face, edge, lw = v4.style_of(colour, v[0])
                drawn[panel][key_of(v, kind)] = ax.scatter(e, yc + jitter, s=DOT_S, facecolors=face,
                                                           edgecolors=edge, linewidths=lw, zorder=3, clip_on=False)
                half = TICK_HALF_PT[split]
                ax.plot([e.mean()] * 2, [yc - half, yc + half], color=INK, lw=1.0, solid_capstyle="butt",
                        zorder=2.5, clip_on=False, gid="mark")
        T(fig, x0[p] + panel_w / 2, y_title, title, ha="center", va="bottom", fontsize=TEXT_PT, gid="title")
        T(fig, x0[p] + panel_w / 2, y_mae, shown[panel], ha="center", va="bottom", fontsize=TEXT_PT, color=NOTE,
          gid="mae")
    T(fig, left - YPAD_PT / 72, y_mae, MAE_LABEL, ha="right", va="bottom", fontsize=TEXT_PT, color=NOTE, gid="mae")
    axes[0].set_ylim(lo, hi)
    axes[0].set_yticks([y for _, y in rows])
    axes[0].set_yticklabels(names, fontsize=TEXT_PT)
    for title, a, b in COLUMN_GROUPS:
        xa, xb = x0[a], x0[b - 1] + panel_w
        T(fig, (xa + xb) / 2, y_group, title, ha="center", va="bottom", fontsize=TEXT_PT, style="italic",
          gid="group")
        hline(fig, xa, xb, y_rule, color=AXIS, lw=0.6)
    across = blend(fig.transFigure, axes[0].transData)
    for y in seps:
        fig.add_artist(Line2D([MARGIN / WIDTH, 1 - MARGIN / WIDTH], [y, y], transform=across, color=AXIS, lw=0.5))

    # bottom line: the legend at the left, the x label centred under the panels
    T(fig, (x0[0] + x0[-1] + panel_w) / 2, MARGIN, XLABEL, ha="center", va="bottom", fontsize=TEXT_PT, gid="xlabel")
    y_mark = MARGIN + descent(TEXT_PT) + 0.23 * TEXT_PT / 72  # half the x-height above the baseline
    x = MARGIN
    for style_, text in ROLE.items():
        face, edge, lw = v4.style_of(ORANGE, style_)
        fig.add_artist(Line2D([x + LEGEND_DOT_PT / 144], [y_mark], marker="o", markersize=LEGEND_DOT_PT,
                              markerfacecolor=face, markeredgecolor=edge, markeredgewidth=lw, linestyle="none",
                              transform=fig.dpi_scale_trans, gid="legend"))
        x += LEGEND_DOT_PT / 72 + 2.5 / 72
        T(fig, x, MARGIN, text, ha="left", va="bottom", fontsize=TEXT_PT, color=INK2, gid="legend")
        x += ruler(text, fontsize=TEXT_PT)[0] + 9.0 / 72
    ruler.close()
    geo = {"height": height, "panel_w": panel_w, "label_col": label_col, "axes_h": axes_h, "left": left,
           "legend_end": x - 9.0 / 72, "rows": rows, "axes": axes}
    return fig, drawn, geo


# ----------------------------------------------------------------------------- checks

def check_points(data: dict, available: dict, drawn: dict) -> list[str]:
    """v4's own check: points drawn against data sets available, every point inside its axis, values unchanged."""
    problems = v4.check_points({p: data[p] for p in LEVELS}, available, drawn, LEVELS,
                               lambda _: axis_of("SCM-1", "synth"), v4.SYNTH_GROUPS)
    problems += v4.check_points({p: data[p] for p in REAL}, available, drawn, REAL,
                                lambda name: axis_of(name, "real"), v4.REAL_GROUPS)
    return problems


def texts(fig) -> list[tuple[str, matplotlib.text.Text]]:
    """Every text drawn, with its role: 'tick', 'name', or the gid given to a figure text."""
    fig.canvas.draw()
    out = []
    for ax in fig.axes:
        lo, hi = sorted(ax.get_xlim())
        for tick in ax.xaxis.get_major_ticks() + ax.xaxis.get_minor_ticks():
            if lo <= tick.get_loc() <= hi:
                out.append(("tick", tick.label1))
        for tick in ax.yaxis.get_major_ticks():
            out.append(("name", tick.label1))
    out += [(t.get_gid() or "text", t) for t in fig.texts]
    return [(role, t) for role, t in out if t.get_visible() and t.get_text().strip()]


def font_problems(fig) -> tuple[list[str], dict]:
    """Every text at or above its floor; returns the problems and the sizes used per role."""
    problems, sizes = [], {}
    for role, t in texts(fig):
        size = t.get_fontsize()
        sizes.setdefault(role, set()).add(size)
        floor = MIN_TICK_PT if role == "tick" else MIN_TEXT_PT
        if size < floor - 1e-9:
            problems.append(f"{role} {t.get_text()!r} is {size:g} pt, below {floor:g} pt")
    return problems, sizes


def tick_problems(axes) -> list[str]:
    problems = []
    for ax in axes:
        major = [t.label1.get_text() for t in ax.xaxis.get_major_ticks() if t.label1.get_text().strip()]
        minor = [t.label1.get_text() for t in ax.xaxis.get_minor_ticks() if t.label1.get_text().strip()]
        if len(major) > MAX_TICK_LABELS or minor:
            problems.append(f"a panel has tick labels {major} and minor labels {minor}")
    return problems


def gaps(fig) -> tuple[list[str], dict]:
    """The clear space between every two text boxes, in points: the larger of the horizontal and the vertical gap,
    negative when they overlap.  Returns the problems (below MIN_GAP_PT) and the smallest gap per pair of roles."""
    renderer = fig.canvas.get_renderer()
    boxes = [(role, t.get_text(), t.get_window_extent(renderer)) for role, t in texts(fig)]
    smallest, problems = {}, []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (ra, ta, a), (rb, tb, b) = boxes[i], boxes[j]
            gap = max(b.x0 - a.x1, a.x0 - b.x1, b.y0 - a.y1, a.y0 - b.y1) * 72 / fig.dpi
            pair = tuple(sorted((ra, rb)))
            if pair not in smallest or gap < smallest[pair][0]:
                smallest[pair] = (gap, ta, tb)
            if gap < MIN_GAP_PT:
                problems.append(f"{ta!r} and {tb!r} are {gap:.2f} pt apart, below {MIN_GAP_PT} pt")
    return problems, smallest


def canvas_problems(fig, height: float) -> list[str]:
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    tol, out = 0.005, []
    if bb.x0 < -tol or bb.y0 < -tol or bb.x1 > WIDTH + tol or bb.y1 > height + tol:
        out.append(f"content leaves the {WIDTH} x {height:.3f} in page: {bb}")
    if height > MAX_HEIGHT + 1e-9:
        out.append(f"height {height:.3f} in is above {MAX_HEIGHT} in")
    return out


def layout_checks(fig, geo: dict) -> tuple[list[str], dict]:
    """The layout checks at 200 dpi (the png, whose hinted glyphs run wider) and 800 dpi (close to the pdf)."""
    problems, report = [], {}
    for dpi in v4.CHECK_DPI:
        fig.set_dpi(dpi)
        found = [f"text collision: {a!r} and {b!r}" for a, b in v4.m.text_collisions(fig)]
        found += v4.mark_collisions(fig) + canvas_problems(fig, geo["height"]) + tick_problems(geo["axes"])
        f_problems, sizes = font_problems(fig)
        g_problems, smallest = gaps(fig)
        found += f_problems + g_problems
        problems += [f"{p} (at {dpi} dpi)" for p in found]
        report[dpi] = {"sizes": sizes, "smallest": smallest}
    fig.set_dpi(v4.CHECK_DPI[0])
    return problems, report


def print_layout(geo: dict, report: dict) -> None:
    sizes = report[v4.CHECK_DPI[0]]["sizes"]
    print(f"  {WIDTH} x {geo['height']:.3f} in; name column {geo['label_col']:.3f} in, seven panels "
          f"{geo['panel_w']:.3f} in wide, {geo['axes_h']:.3f} in tall; legend ends at {geo['legend_end']:.3f} in, "
          f"the first panel starts at {geo['left']:.3f} in")
    print("  font sizes: " + ", ".join(f"{role} {'/'.join(f'{s:g}' for s in sorted(v))} pt"
                                       for role, v in sorted(sizes.items())))
    print("  smallest clear space between text boxes, pt, at 200 dpi | 800 dpi:")
    lo, hi = (report[d]["smallest"] for d in v4.CHECK_DPI)
    for pair in sorted(lo, key=lambda k: lo[k][0])[:8]:
        g, a, b = lo[pair]
        print(f"    {' / '.join(pair):16s} {g:6.2f} | {hi[pair][0]:6.2f}   ({a!r} and {b!r})")


def save(fig, outdir: Path) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    pdf, png = outdir / f"{STEM}.pdf", outdir / f"{STEM}.png"
    fig.savefig(pdf, metadata={"CreationDate": None})
    fig.savefig(png, dpi=200)
    return pdf, png


# ----------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outdir", type=Path, default=FIGDIR, help="where the pdf and png go")
    parser.add_argument("--no-copy", action="store_true", help=f"skip the copy to {PAPER_COPY.relative_to(ROOT)}")
    args = parser.parse_args(argv)

    data, available = load()
    print("data sets: " + ", ".join(f"{p} {available[p]}" for p, _, _ in PANELS) + "\n")
    problems, skipped = mae_table(data)
    print("\n== PROBE's MAE as drawn, 2 significant digits, decimal half-up")
    problems += probe_mae_check(data)
    with plt.rc_context(style()):
        print("\n== the version 4 figure's own MAE checks, repeated: MAE from the data (reference)")
        ref = reference_cells()
        problems += v4.check_mae({p: data[p] for p in LEVELS}, LEVELS,
                                 {k: v for k, v in ref.items() if k[0] in LEVELS}, 3, v4.SYNTH_GROUPS)
        problems += v4.check_mae({p: data[p] for p in REAL}, REAL,
                                 {k: v for k, v in ref.items() if k[0] in REAL}, 4, v4.REAL_GROUPS)

        print("\n== points drawn / data sets available (every point inside its axis)")
        fig, drawn, geo = draw(data)
        problems += check_points(data, available, drawn)

        print("\n== layout")
        found, report = layout_checks(fig, geo)
        problems += found
        print_layout(geo, report)
        pdf, png = save(fig, args.outdir)
        plt.close(fig)
    print(f"\nwrote {pdf}\nwrote {png} (200 dpi)")
    if not args.no_copy:
        if problems:
            print(f"not copied to {PAPER_COPY}: the checks below failed")
        else:
            shutil.copyfile(pdf, PAPER_COPY)
            print(f"copied to {PAPER_COPY}")
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print("\nAll checks passed: every point drawn and inside its axis; every MAE equal to its reference"
          + ("" if skipped else " and to the version 4 figure") + "; no text collisions; no text over a data mark; "
          f"every text at its size floor and at least {MIN_GAP_PT} pt from its neighbours; at most "
          f"{MAX_TICK_LABELS} tick labels per panel; content inside the page; height within {MAX_HEIGHT} in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
