"""Section 5 main experiment figure, version 3: what every method returns on the sealed E1 and E2 data sets.

    section5-v3-estimates   Four panels (SCM-1 to SCM-4), one row per method, one dot per data set.  x is the
                            error, estimate minus the true effect 1, on a symmetric-log axis that is linear within
                            +-0.1, so the cluster at 0 and the errors near -5 are both drawn and no point leaves the
                            axis.  A short black tick marks each row's mean error; the number at the right edge of
                            each panel is the row's MAE, the mean absolute error over data sets.

Data, read-only, through the loading functions of make_section5_v2_figures.py, imported rather than copied:
    v2.collect            (its lines 51-72)  every row of one level, in data-set order
    v2.shards             (its lines 38-41)  results/family_v2/confirm/SCM-{1,2,3}_seed*.json (E1)
                                             results/family_v2/scm4/confirm/SCM-4_seed*.json (E2)
    v2.pixel_cnn_amended  (its lines 44-48)  results/family_v2/scm4/confirm_summary_v1_amended.json
    v2.kspc_rows          (its lines 75-87)  results/kspc_comparator/E1 and E2, run after the protocol was frozen
Nothing is refitted or rerun, and nothing under results/ is written.

Writes figures/section5/section5-v3-estimates.{pdf,png} and the copy ICLR/paper/figures/probe-estimates-v3.pdf;
ICLR/paper/figures/probe-estimates.pdf is left alone.  Checks printed on every run: points drawn against data sets
available for every panel and row, every point inside the axis (v2 dropped partly off-axis rows silently at its
line 161), text collisions, and the MAE table against the E1 and E2 tables of EXPERIMENT_INDEX.md.
"""
from __future__ import annotations

import sys

# Importing v2 must not write into code/__pycache__: its cached .pyc is older than its source, so Python would
# otherwise rewrite that existing file.
sys.dont_write_bytecode = True

import argparse
import json
import shutil
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory as blend

with warnings.catch_warnings():
    # v2's coverage docstring (its line 301) holds an invalid escape sequence; the warning concerns that text only
    warnings.simplefilter("ignore", SyntaxWarning)
    import make_section5_v2_figures as v2

m = v2.m  # make_section5_figures: the results path and text_collisions
ROOT = m.HERE.parent
PAPER_COPY = ROOT / "ICLR" / "paper" / "figures" / "probe-estimates-v3.pdf"
STEM = "section5-v3-estimates"
TAU = v2.TAU  # the true effect, 1 in every data set
LEVELS = v2.LEVELS
# The counts are the dimension of (X, W), what the learner sees (Appendix D.1: SCM-1 has 2 + 6 = 8).
TITLES = {"SCM-1": "SCM-1\n$\\mathrm{dim}(X,W)$ = 8", "SCM-2": "SCM-2\n$\\mathrm{dim}(X,W)$ = 88",
          "SCM-3": "SCM-3\n$\\mathrm{dim}(X,W)$ = 1,312", "SCM-4": "SCM-4\nsix images"}
XLABEL = "estimate minus true effect"
FOOTNOTE = "† run after the protocol was frozen"

BLUE, GREEN, GRAY, ORANGE, DARK = "#2a78d6", "#166534", "#898781", "#eb6834", "#52514e"
INK, INK2, HAIR, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

FILLED, HOLLOW = "filled", "hollow"
GROUPS = [
    ("No roles needed", [("PROBE (ours)", "probe", BLUE, FILLED),
                         ("CEVAE", "cevae", GREEN, FILLED),
                         ("Adjust for $X$ and all of $W$", "raw", GRAY, FILLED),
                         ("Adjust for $X$", "x", GRAY, FILLED),
                         ("Pixel CNN", "pixel", GRAY, FILLED)]),
    ("Correct roles given", [("Proximal 2SLS", "p2sls_given", ORANGE, FILLED),
                             ("Single-proxy control, clean block", "park_clean", ORANGE, FILLED),
                             ("KSPC†, clean block", "kspc_clean", ORANGE, FILLED)]),
    ("Wrong roles given", [("Proximal 2SLS, roles swapped", "p2sls_swapped", ORANGE, HOLLOW),
                           ("Single-proxy control, contaminated block", "park_contaminated", ORANGE, HOLLOW),
                           ("KSPC†, contaminated block", "kspc_contaminated", ORANGE, HOLLOW)]),
    (None, [("Oracle (adjusts for $(X,U)$)", "oracle", DARK, FILLED)]),  # the reference row, below a separator
]
ROWS = [row for _, members in GROUPS for row in members]
# Names wider than "Oracle (adjusts for (X,U))" break onto two lines at the space named here; the words are unchanged,
# which keeps the label column narrow enough for four readable panels and titles that do not touch.
BREAKS = {"raw": " and ", "park_clean": ", ", "p2sls_swapped": ", ", "park_contaminated": ", ",
          "kspc_contaminated": ", "}

# Symmetric-log x axis: linear on [-0.1, 0.1], logarithmic beyond.  The left limit leaves room for the farthest
# point (-9.33, 2SLS with swapped roles in SCM-2); the checks below fail loudly if any point lies outside.
LINTHRESH, LINSCALE = 0.1, 1.0
XLIM = (-15.0, 1.0)
TICKS = [-5.0, -0.1, 0.0, 0.1, 1.0]
TICK_LABELS = ["−5", "−0.1", "0", "0.1", "1"]
MINOR_TICKS = [-1.0]  # marked but not labelled: at 7 pt a "-1" label would touch "-5" and "-0.1" in a 0.78 in panel

# Geometry, inches unless a name says pt.  5.5 in is the paper's text width.
WIDTH, HEIGHT = 5.5, 3.8
MARGIN = 0.02
GAP = 0.09        # from one panel's MAE column to the next panel's data area
MAE_PAD = 0.03    # from a panel's data area to its MAE column
LINE_GAP = 0.035  # between the tick labels, the x label and the footnote
YPAD_PT, TITLE_PAD_PT, TICK_LEN_PT, TICK_PAD_PT = 4.0, 3.0, 2.5, 1.5
MIXED_STEP, DOUBLE_STEP = 1.1, 1.25         # row step next to one or between two two-line names, in row units
DOT_S, JITTER, TICK_HALF = 9.0, 0.12, 0.42  # the mean tick is drawn under the dots and reaches past them

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.grid": False,
    "xtick.color": AXIS, "xtick.labelcolor": INK2, "xtick.major.width": 0.6,
    "xtick.major.size": TICK_LEN_PT, "xtick.major.pad": TICK_PAD_PT, "ytick.labelcolor": INK,
    "axes.unicode_minus": True, "savefig.bbox": "standard", "savefig.dpi": 200,
}


def display(label: str, key: str) -> str:
    """The row name as drawn: the exact name, broken once at BREAKS[key] if it has an entry."""
    if key not in BREAKS:
        return label
    cut = BREAKS[key]
    shown = label.replace(cut, cut.replace(" ", "\n", 1), 1)  # the first space of the cut becomes the break
    if "\n" not in shown or shown.replace("\n", " ") != label:
        raise ValueError(f"line break changed the name {label!r}")
    return shown


def lines(label: str, key: str) -> int:
    return display(label, key).count("\n") + 1


# ----------------------------------------------------------------------------- data

def load() -> tuple[dict, dict]:
    """Errors per level and row key, read through v2, with the counts of data sets available."""
    errors, available = {}, {}
    for level in LEVELS:
        sealed = v2.shards(level)
        data = v2.collect(level)
        available[level] = len(sealed)
        kspc_folder = m.RESULTS / "kspc_comparator" / ("E2" if level == "SCM-4" else "E1")
        for s in sealed:  # v2 pairs KSPC files by replicate number; confirm each pairs with the same seed
            k = json.loads((kspc_folder / f"{level}_replicate_{s['task']['rep']}.json").read_text())["task"]
            if (k["seed"], k["replicate"]) != (s["task"]["seed"], s["task"]["rep"]):
                raise ValueError(f"{level}: KSPC replicate {k['replicate']} does not match sealed seed {s['task']['seed']}")
        errors[level] = {}
        for _, key, _, _ in ROWS:
            values = data[key]
            if values is None:
                errors[level][key] = None
                continue
            e = np.asarray(values, dtype=float) - TAU
            if len(e) != len(sealed) or not np.all(np.isfinite(e)):
                raise ValueError(f"{level} {key}: {np.isfinite(e).sum()} finite of {len(sealed)} data sets")
            errors[level][key] = e
    return errors, available


# ----------------------------------------------------------------------------- layout

def positions() -> tuple[list, list, list]:
    """y of each row, each group label and each separator, top to bottom, in row units.  A two-line name gets a
    wider step to its neighbours so the text never crowds."""
    rows, headers, seps, y = [], [], [], 0.0
    for g, (header, members) in enumerate(GROUPS):
        if g:
            y -= 0.65
            seps.append(y)
            y -= 0.55 if header else 0.65
        if header:
            headers.append((header, y))
            y -= 0.9
        for i, row in enumerate(members):
            if i:
                wrapped = (lines(*members[i - 1][:2]) > 1) + (lines(*row[:2]) > 1)
                y -= (1.0, MIXED_STEP, DOUBLE_STEP)[wrapped]
            rows.append((row, y))
    return rows, headers, seps


def measure(fig, text: str, **kw) -> tuple[float, float]:
    t = fig.text(0, 0, text, **kw)
    bb = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def geometry(fig) -> dict:
    """Margins from measured text, so nothing is guessed from font size."""
    label_w = max(measure(fig, display(label, key), fontsize=8, linespacing=1.0)[0] for label, key, _, _ in ROWS)
    label_w = max(label_w, max(measure(fig, h, fontsize=7, style="italic")[0] for h, _ in GROUPS if h) - YPAD_PT / 72)
    mae_w = measure(fig, "0.000", fontsize=6.5)[0]
    title_h = max(measure(fig, t, fontsize=8, linespacing=1.15)[1] for t in TITLES.values())
    tick_h = measure(fig, "0.1", fontsize=7)[1]
    xlabel_h = measure(fig, XLABEL, fontsize=8)[1]
    foot_h = measure(fig, FOOTNOTE, fontsize=7)[1]
    left = MARGIN + label_w + YPAD_PT / 72
    mae_col = MAE_PAD + mae_w
    data_w = (WIDTH - MARGIN - left - 4 * mae_col - 3 * GAP) / 4
    top = HEIGHT - MARGIN - title_h - TITLE_PAD_PT / 72
    bottom = MARGIN + foot_h + LINE_GAP + xlabel_h + LINE_GAP + tick_h + (TICK_LEN_PT + TICK_PAD_PT) / 72
    return {"left": left, "data_w": data_w, "mae_col": mae_col, "top": top, "bottom": bottom,
            "xlabel_y": MARGIN + foot_h + LINE_GAP, "label_w": label_w}


# ----------------------------------------------------------------------------- figure

def draw(errors: dict) -> tuple[plt.Figure, dict, dict]:
    rows, headers, seps = positions()
    ylim = (rows[-1][1] - 0.62, headers[0][1] + 0.5)
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    fig.canvas.draw()
    g = geometry(fig)
    axes, drawn = [], {}
    for p, level in enumerate(LEVELS):
        x0 = g["left"] + p * (g["data_w"] + g["mae_col"] + GAP)
        ax = fig.add_axes([x0 / WIDTH, g["bottom"] / HEIGHT, g["data_w"] / WIDTH, (g["top"] - g["bottom"]) / HEIGHT],
                          sharey=axes[0] if axes else None)
        axes.append(ax)
        ax.set_xscale("symlog", linthresh=LINTHRESH, linscale=LINSCALE)
        ax.set_xlim(*XLIM)
        ax.xaxis.set_major_locator(FixedLocator(TICKS))
        ax.xaxis.set_major_formatter(FixedFormatter(TICK_LABELS))
        ax.xaxis.set_minor_locator(FixedLocator(MINOR_TICKS))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=TICK_LEN_PT * 0.6, width=0.6, color=AXIS)
        ax.tick_params(axis="y", length=0, pad=YPAD_PT, labelleft=(p == 0))
        ax.set_title(TITLES[level], fontsize=8, linespacing=1.15, pad=TITLE_PAD_PT)
        ax.axvline(0.0, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=2)
        at_row = blend(ax.transAxes, ax.transData)
        mae_x = 1 + g["mae_col"] / g["data_w"]
        ax.text(mae_x, headers[0][1], "MAE", transform=at_row, ha="right", va="center", fontsize=6.5, color=INK2)
        for (label, key, colour, style), y in rows:
            e = errors[level][key]
            if e is None:  # the pixel CNN exists only for the image setting: row left blank, no guide, no MAE
                drawn[(level, key)] = 0
                continue
            ax.axhline(y, color=HAIR, lw=0.4, zorder=0)
            jitter = np.random.default_rng(1000 * p + ROWS.index((label, key, colour, style))).uniform(
                -JITTER, JITTER, len(e))
            # hollow rings are white inside so a stack of them still reads as rings
            face, edge, width = (colour, "white", 0.35) if style == FILLED else ("white", colour, 0.6)
            dots = ax.scatter(e, y + jitter, s=DOT_S, facecolors=face, edgecolors=edge, linewidths=width,
                              zorder=3, clip_on=False)
            ax.plot([e.mean()] * 2, [y - TICK_HALF, y + TICK_HALF], color=INK, lw=1.0, solid_capstyle="butt",
                    zorder=2.5, clip_on=False)
            ax.text(mae_x, y, f"{np.mean(np.abs(e)):.3f}", transform=at_row, ha="right", va="center",
                    fontsize=6.5, color=INK2)
            drawn[(level, key)] = dots
    axes[0].set_ylim(*ylim)
    axes[0].set_yticks([y for _, y in rows])
    axes[0].set_yticklabels([display(label, key) for (label, key, _, _), y in rows], linespacing=1.0)

    across = blend(fig.transFigure, axes[0].transData)
    for header, y in headers:
        fig.text(MARGIN / WIDTH, y, header, transform=across, ha="left", va="center", fontsize=7, style="italic",
                 color=INK2)
    for y in seps:
        fig.add_artist(Line2D([MARGIN / WIDTH, 1 - MARGIN / WIDTH], [y, y], transform=across, color=AXIS, lw=0.5))
    x_mid = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    fig.text(x_mid, g["xlabel_y"] / HEIGHT, XLABEL, ha="center", va="bottom", fontsize=8)
    fig.text(MARGIN / WIDTH, MARGIN / HEIGHT, FOOTNOTE, ha="left", va="bottom", fontsize=7, color=INK2)
    g["row_pitch_pt"] = (g["top"] - g["bottom"]) * 72 / (ylim[1] - ylim[0])
    return fig, drawn, g


# ----------------------------------------------------------------------------- checks

def check_points(errors: dict, available: dict, drawn: dict) -> list[str]:
    """Points drawn against data sets available, and every drawn point inside the axis."""
    problems = []
    print("\nPoints drawn / data sets available (every drawn point inside the x axis):")
    print(f"  {'row':42s}" + "".join(f"{level:>10s}" for level in LEVELS))
    for label, key, _, _ in ROWS:
        cells = []
        for level in LEVELS:
            e, dots = errors[level][key], drawn[(level, key)]
            if e is None:
                cells.append("n/a")
                if dots != 0:
                    problems.append(f"{level} {key}: drew points where the row does not apply")
                continue
            xy = np.asarray(dots.get_offsets())
            inside = int(np.sum((xy[:, 0] >= XLIM[0]) & (xy[:, 0] <= XLIM[1])))
            cells.append(f"{len(xy)}/{available[level]}")
            if len(xy) != available[level] or inside != len(xy) or not np.allclose(np.sort(xy[:, 0]), np.sort(e)):
                problems.append(f"{level} {key}: drew {len(xy)}, inside {inside}, available {available[level]}")
        print(f"  {label:42s}" + "".join(f"{c:>10s}" for c in cells))
    return problems


INDEX_ROWS = {  # EXPERIMENT_INDEX.md row name -> figure rows; the index calls the single-proxy control COCA
    "PROBE": ["probe"], "CEVAE": ["cevae"],
    "2SLS, roles given (role-requiring)": ["p2sls_given"], "2SLS, roles given": ["p2sls_given"],
    "COCA, clean block (role-requiring)": ["park_clean"], "COCA, clean block": ["park_clean"],
    "adjust for X and all of W": ["raw"], "adjust for X": ["x"], "pixel CNN": ["pixel"],
    "2SLS, roles swapped": ["p2sls_swapped"], "COCA, contaminated block": ["park_contaminated"],
    "2SLS, roles swapped / COCA, contaminated": ["p2sls_swapped", "park_contaminated"],
    "KSPC with X, clean block†": ["kspc_clean"], "KSPC with X, contaminated block†": ["kspc_contaminated"],
    "KSPC with X, clean / contaminated block†": ["kspc_clean", "kspc_contaminated"],
    "oracle (adjusts for U)": ["oracle"], "oracle": ["oracle"],
}


def index_mae() -> dict[tuple[str, str], str]:
    """The MAE cells of the E1 and E2 tables in EXPERIMENT_INDEX.md, keyed by (level, row key)."""
    text = (ROOT / "EXPERIMENT_INDEX.md").read_text()
    cells = {}
    for exp in ("E1", "E2"):
        block = text.split(f"\n## {exp}.", 1)[1].split("\n## ", 1)[0]
        table = [line for line in block.splitlines() if line.startswith("|")]
        levels = [c.strip() for c in table[0].strip().strip("|").split("|")][1:]
        for line in table[2:]:
            name, *values = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
            if name not in INDEX_ROWS:
                raise KeyError(f"{exp}: index row {name!r} has no figure row")
            keys = INDEX_ROWS[name]
            for level, value in zip(levels, values):
                parts = [v.strip() for v in value.split("/")]
                if len(parts) != len(keys):
                    raise ValueError(f"{exp} {name!r} {level}: {value!r} does not split into {len(keys)} values")
                for key, part in zip(keys, parts):
                    cells[(level, key)] = part
    return cells


def check_mae(errors: dict) -> list[str]:
    """The figure's MAE table beside the index, cell by cell, at 3 decimals."""
    index, problems = index_mae(), []
    print("\nMAE over data sets, 3 decimals: figure (EXPERIMENT_INDEX.md)")
    print(f"  {'row':42s}" + "".join(f"{level:>16s}" for level in LEVELS))
    for label, key, _, _ in ROWS:
        cells = []
        for level in LEVELS:
            e = errors[level][key]
            ref = index.get((level, key))
            if e is None:
                cells.append("blank" + ("" if ref is None else f" ({ref})"))
                if ref is not None:
                    problems.append(f"{level} {label}: blank in the figure, {ref} in the index")
                continue
            mine = f"{np.mean(np.abs(e)):.3f}"
            cells.append(f"{mine} ({ref if ref is not None else 'none'})")
            if ref is None:
                problems.append(f"{level} {label}: {mine} in the figure, no cell in the index")
            elif ref != mine:
                problems.append(f"{level} {label}: MISMATCH figure {mine}, index {ref}")
        print(f"  {label:42s}" + "".join(f"{c:>16s}" for c in cells))
    unused = sorted(k for k in index if errors.get(k[0], {}).get(k[1]) is None)
    for level, key in unused:
        problems.append(f"index cell {level} {key} = {index[(level, key)]} has no drawn row")
    return problems


def check_canvas(fig) -> list[str]:
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    tol = 0.005
    if bb.x0 < -tol or bb.y0 < -tol or bb.x1 > WIDTH + tol or bb.y1 > HEIGHT + tol:
        return [f"content leaves the {WIDTH} x {HEIGHT} in page: {bb}"]
    return []


# ----------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outdir", type=Path, default=m.FIGDIR, help="where the pdf and png go")
    parser.add_argument("--no-copy", action="store_true", help=f"skip the copy to {PAPER_COPY.relative_to(ROOT)}")
    args = parser.parse_args(argv)

    errors, available = load()
    with plt.rc_context(STYLE):
        fig, drawn, g = draw(errors)
        problems = check_points(errors, available, drawn)
        problems += check_mae(errors)
        collisions = m.text_collisions(fig)
        problems += [f"text collision: {a!r} and {b!r}" for a, b in collisions]
        problems += check_canvas(fig)
        args.outdir.mkdir(parents=True, exist_ok=True)
        pdf, png = args.outdir / f"{STEM}.pdf", args.outdir / f"{STEM}.png"
        fig.savefig(pdf, metadata={"CreationDate": None})
        fig.savefig(png, dpi=200)
        plt.close(fig)
    print(f"\nFigure {WIDTH} x {HEIGHT} in; label column {g['label_w']:.2f} in; data area {g['data_w']:.2f} in and "
          f"MAE column {g['mae_col']:.2f} in per panel; row pitch {g['row_pitch_pt']:.1f} pt")
    print(f"wrote {pdf}\nwrote {png} (200 dpi)")
    if not args.no_copy:
        if PAPER_COPY.name == "probe-estimates.pdf":
            raise RuntimeError("refusing to overwrite the v2 figure")
        shutil.copyfile(pdf, PAPER_COPY)
        print(f"copied to {PAPER_COPY}")
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print("\nAll checks passed: every point drawn and inside the axis, MAE table equal to the index, no text collisions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
