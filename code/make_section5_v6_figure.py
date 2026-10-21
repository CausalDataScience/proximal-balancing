"""Figure 3, version 6: version 5 (make_section5_v5_figure) with a chest X-ray group of two panels after the synthetic
and the semi-synthetic groups.

The X-ray panels show the treatment-response and the mediator scenarios of the single-image experiment (50 replicates,
new seeds, hand-drawn blocks, supervised encoder; results/single_image/pneumoniamnist_v1/v10_anat_r50_w*.jsonl):
PROBE is the block screen of the chest X-ray analysis, "Pixel CNN" is adjustment for the convolutional representation
of the whole image (all five block encoders, no block held out), "Adjust for X" is the difference in means (the X-ray data have no covariates), and the oracle adjusts
for U.  The methods not run on the X-rays are marked NA, as the pixel CNN is outside SCM-4.  The X-ray panels use the
synthetic panels' symmetric-log axis.  Every check of version 5 is kept for the seven original panels; the layout checks
(text collisions, text over marks, font floors, clear space, page) run on the whole figure.  The text size is the
largest of 10.5, 10, 9.5 and 9 pt that passes the layout checks with nine panels; to make room, each axis
labels only its outer ticks (the dashed zero line marks 0) and the two X-ray panels are a group gap apart.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory as blend

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import make_section5_v5_figure as f5  # noqa: E402

XRAY = CODE.parent / "results/single_image/pneumoniamnist_v1"
XRAY_PANELS = [("xray_resp", "Response", "synth", 3), ("xray_med", "Mediator", "synth", 4)]
ORIG_PANELS, ORIG_GROUPS = list(f5.PANELS), list(f5.COLUMN_GROUPS)
# Groups by kind of proxy: Synthetic (SCM-1 to SCM-3), Semi-synthetic, and Image, which holds Shapes3D (SCM-4: SCM-1
# with each proxy coordinate shown as a Shapes3D image) and the two chest X-ray scenarios, named by scenario.
XRAY_TITLE = {"xray_resp": "X-ray\nresponse", "xray_med": "X-ray\nmediator"}
NEW_PANELS = (ORIG_PANELS[:3] + ORIG_PANELS[4:7] + [(ORIG_PANELS[3][0], "Shapes3D", ORIG_PANELS[3][2])]
              + [(p, XRAY_TITLE[p], k) for p, t, k, _ in XRAY_PANELS])
NEW_GROUPS = [("Synthetic", 0, 3), ("Real-world", 3, 6), ("Image", 6, 9)]   # "Real-world" as in Section 5
STEM = "section5-v6-experiments"
PAPER_COPY = f5.v4.PAPER_FIGURES / "probe-experiments-v6.pdf"


_orig_set_symlog = f5.set_symlog


def set_symlog_outer(ax, spec: dict) -> None:
    """Version 5's axis with the centre label left out: nine panels leave too little room between '0' and its
    neighbours, and the dashed zero line already marks 0.  The tick mark at 0 stays."""
    _orig_set_symlog(ax, spec)
    from matplotlib.ticker import FixedFormatter
    ax.xaxis.set_major_formatter(FixedFormatter(["" if t == 0.0 else f5.v4.minus(t) for t in spec["ticks"]]))


WIDE_AFTER = {7, 8}
TITLE_LS = 0.9      # line spacing of the two-line X-ray titles, so that the figure stays within its 3.0 in height   # a group-sized gap also between the two X-ray panels, so that their titles clear each other


def load_xray() -> tuple[dict, dict]:
    rows = [json.loads(l) for f in sorted(glob.glob(str(XRAY / "v10_anat_r50_w*.jsonl"))) for l in Path(f).read_text().splitlines() if l.strip()]
    data, n = {}, {}
    for panel, _, _, c in XRAY_PANELS:
        rr = sorted([r for r in rows if r["scenario"] == c], key=lambda r: r["rep"])
        data[panel] = {"probe": np.array([r["PROBE_seq_floor"] for r in rr]) - 1.0, "pixel": np.array([r["all_blocks"] for r in rr]) - 1.0,
                       "x": np.array([r["naive"] for r in rr]) - 1.0, "oracle": np.array([r["oracle"] for r in rr]) - 1.0}
        n[panel] = len(rr)
    return data, n


def draw(data: dict):
    """f5.draw generalised to any number of panels in groups (GROUP_GAP between groups, PANEL_GAP inside)."""
    P, G = f5.PANELS, f5.COLUMN_GROUPS
    ruler = f5.v4.Ruler()
    rows, seps, lo, hi = f5.row_layout()
    names = [m[0] for m, _ in rows]
    shown = f5.probe_mae_shown(data)
    TEXT_PT = f5.TEXT_PT
    label_col = max(ruler(n, fontsize=TEXT_PT)[0] for n in names + [f5.MAE_LABEL])
    h_title = max(ruler(t, fontsize=TEXT_PT, linespacing=TITLE_LS)[1] for _, t, _ in P)
    h_mae = max(ruler(s, fontsize=TEXT_PT)[1] for s in [f5.MAE_LABEL, *shown.values()])
    h_group = max(ruler(t, fontsize=TEXT_PT, style="italic")[1] for t, _, _ in G if t)
    h_tick = max(ruler(f5.v4.minus(t), fontsize=f5.TICK_PT)[1] for spec in f5.AXES.values() for t in spec["ticks"])
    h_line = max(ruler(s, fontsize=TEXT_PT)[1] for s in [f5.XLABEL, *f5.ROLE.values()])
    bottom = f5.MARGIN + h_line + f5.LINE_GAP_PT / 72 + h_tick + (f5.TICK_LEN_PT + f5.TICK_PAD_PT) / 72
    axes_h = (hi - lo) / 72
    y_mae = bottom + axes_h + f5.TITLE_PAD_PT / 72
    y_title = y_mae + h_mae + f5.MAE_GAP_PT / 72
    y_rule = y_title + h_title + f5.RULE_PAD_PT / 72
    y_group = y_rule + f5.RULE_PAD_PT / 72
    height = y_group + h_group + f5.MARGIN
    left = f5.MARGIN + label_col + f5.YPAD_PT / 72
    starts = {a for _, a, _ in G if a > 0} | WIDE_AFTER
    gaps = [f5.GROUP_GAP if i in starts else f5.PANEL_GAP for i in range(1, len(P))]
    avail = f5.WIDTH - f5.MARGIN - left - sum(gaps)
    panel_w = avail / len(P)
    # the last panel's title is wider than the panel; narrow the panels so that it ends at the right margin
    last_w = ruler(P[-1][1], fontsize=TEXT_PT)[0]
    panel_w = min(panel_w, (avail - last_w / 2) / (len(P) - 0.5))
    x0 = [left]
    for g in gaps:
        x0.append(x0[-1] + panel_w + g)
    fig = plt.figure(figsize=(f5.WIDTH, height))
    axes, drawn = [], {}
    for p, (panel, title, kind) in enumerate(P):
        ax = fig.add_axes([x0[p] / f5.WIDTH, bottom / height, panel_w / f5.WIDTH, axes_h / height], sharey=axes[0] if axes else None)
        axes.append(ax)
        set_symlog_outer(ax, f5.axis_of(panel, kind))
        ax.tick_params(axis="y", length=0, pad=f5.YPAD_PT, labelleft=(p == 0))
        ax.axvline(0.0, color=f5.INK, lw=0.6, ls=(0, (3, 2)), zorder=2)
        drawn[panel] = {}
        for r, ((name, colour, versions), y) in enumerate(rows):
            present = [v for v in versions if f5.errors_of(data, panel, kind, v) is not None]
            for v in versions:
                if f5.key_of(v, kind) is not None and v not in present:
                    drawn[panel][f5.key_of(v, kind)] = None
            if not present:
                fig.text(0.0, y, f5.NA, transform=ax.transData, ha="center", va="center_baseline", fontsize=TEXT_PT,
                         color=f5.NOTE, gid="na", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
                continue
            ax.axhline(y, color=f5.HAIR, lw=0.4, zorder=0)
            split = len(versions) == 2
            for j, v in enumerate(present):
                e = f5.errors_of(data, panel, kind, v)
                yc = y + (f5.SPLIT_PT if v[0] == f5.FILLED else -f5.SPLIT_PT) if split else y
                jitter = np.random.default_rng(1000 * p + 10 * r + j).uniform(-f5.JITTER_PT[split], f5.JITTER_PT[split], len(e))
                face, edge, lw = f5.v4.style_of(colour, v[0])
                drawn[panel][f5.key_of(v, kind)] = ax.scatter(e, yc + jitter, s=f5.DOT_S, facecolors=face, edgecolors=edge,
                                                              linewidths=lw, zorder=3, clip_on=False)
                half = f5.TICK_HALF_PT[split]
                ax.plot([e.mean()] * 2, [yc - half, yc + half], color=f5.INK, lw=1.0, solid_capstyle="butt", zorder=2.5,
                        clip_on=False, gid="mark")
        f5.T(fig, x0[p] + panel_w / 2, y_title, title, ha="center", va="bottom", fontsize=TEXT_PT, linespacing=TITLE_LS, gid="title")
        f5.T(fig, x0[p] + panel_w / 2, y_mae, shown[panel], ha="center", va="bottom", fontsize=TEXT_PT, color=f5.NOTE, gid="mae")
    f5.T(fig, left - f5.YPAD_PT / 72, y_mae, f5.MAE_LABEL, ha="right", va="bottom", fontsize=TEXT_PT, color=f5.NOTE, gid="mae")
    axes[0].set_ylim(lo, hi)
    axes[0].set_yticks([y for _, y in rows])
    axes[0].set_yticklabels(names, fontsize=TEXT_PT)
    for title, a, b in G:
        if not title:      # a one-column group shows only its column title, with group gaps on both sides
            continue
        xa, xb = x0[a], x0[b - 1] + panel_w
        f5.T(fig, (xa + xb) / 2, y_group, title, ha="center", va="bottom", fontsize=TEXT_PT, style="italic", gid="group")
        f5.hline(fig, xa, xb, y_rule, color=f5.AXIS, lw=0.6)
    across = blend(fig.transFigure, axes[0].transData)
    for y in seps:
        fig.add_artist(Line2D([f5.MARGIN / f5.WIDTH, 1 - f5.MARGIN / f5.WIDTH], [y, y], transform=across, color=f5.AXIS, lw=0.5))
    f5.T(fig, (x0[0] + x0[-1] + panel_w) / 2, f5.MARGIN, f5.XLABEL, ha="center", va="bottom", fontsize=TEXT_PT, gid="xlabel")
    y_mark = f5.MARGIN + f5.descent(TEXT_PT) + 0.23 * TEXT_PT / 72
    x = f5.MARGIN
    for style_, text in f5.ROLE.items():
        face, edge, lw = f5.v4.style_of(f5.ORANGE, style_)
        fig.add_artist(Line2D([x + f5.LEGEND_DOT_PT / 144], [y_mark], marker="o", markersize=f5.LEGEND_DOT_PT, markerfacecolor=face,
                              markeredgecolor=edge, markeredgewidth=lw, linestyle="none", transform=fig.dpi_scale_trans, gid="legend"))
        x += f5.LEGEND_DOT_PT / 72 + 2.5 / 72
        f5.T(fig, x, f5.MARGIN, text, ha="left", va="bottom", fontsize=TEXT_PT, color=f5.INK2, gid="legend")
        x += ruler(text, fontsize=TEXT_PT)[0] + 9.0 / 72
    ruler.close()
    geo = {"height": height, "panel_w": panel_w, "label_col": label_col, "axes_h": axes_h, "left": left,
           "legend_end": x - 9.0 / 72, "rows": rows, "axes": axes}
    return fig, drawn, geo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outdir", type=Path, default=f5.FIGDIR)
    ap.add_argument("--copy", action="store_true", help=f"copy the pdf to {PAPER_COPY}")
    a = ap.parse_args(argv)
    data, available = f5.load()
    problems = []
    # version 5's MAE checks on the seven original panels
    f5.PANELS, f5.COLUMN_GROUPS = ORIG_PANELS, ORIG_GROUPS
    p, skipped = f5.mae_table(data); problems += p
    problems += f5.probe_mae_check(data)
    xdata, xn = load_xray(); data.update(xdata); available.update(xn)
    f5.PANELS, f5.COLUMN_GROUPS = NEW_PANELS, NEW_GROUPS
    print("\n== chest X-ray panels: MAE of the drawn rows")
    for panel, title, _, c in XRAY_PANELS:
        print(f"  {title:9s} n={xn[panel]}  " + "  ".join(f"{k} {np.mean(np.abs(v)):.3f}" for k, v in xdata[panel].items()))
    chosen = None
    for text_pt, tick_pt in ((10.5, 8.5), (10.0, 8.5), (9.5, 8.0), (9.0, 8.0)):
        f5.TEXT_PT, f5.TICK_PT = text_pt, tick_pt
        with plt.rc_context(f5.style()):
            fig, drawn, geo = draw(data)
            found, report = f5.layout_checks(fig, geo)
        print(f"\n== layout at {text_pt} pt text, {tick_pt} pt ticks: {len(found)} problems, panel width {geo['panel_w']:.3f} in, height {geo['height']:.3f} in")
        for q in found[:6]:
            print("   ", q)
        if not found:
            chosen = (fig, drawn, geo, text_pt, tick_pt); break
        plt.close(fig)
    if chosen is None:
        problems.append("no text size passes the layout checks with nine panels")
        print("\nPROBLEMS:\n  " + "\n  ".join(problems)); return 1
    fig, drawn, geo, text_pt, tick_pt = chosen
    problems += f5.check_points(data, available, drawn)
    missing = [f"{panel} {k}: drawn {0 if drawn[panel].get(k) is None else len(drawn[panel][k].get_offsets())}, available {len(v)}"
               for panel, *_ in XRAY_PANELS for k, v in data[panel].items()
               if drawn[panel].get(k) is None or len(drawn[panel][k].get_offsets()) != len(v)]
    problems += missing
    a.outdir.mkdir(parents=True, exist_ok=True)
    pdf, png = a.outdir / f"{STEM}.pdf", a.outdir / f"{STEM}.png"
    with plt.rc_context(f5.style()):
        fig.savefig(pdf, metadata={"CreationDate": None}); fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"\nwrote {pdf} ({text_pt} pt text, {tick_pt} pt ticks, {geo['panel_w']:.3f} in panels, {geo['height']:.3f} in tall)")
    if a.copy and not problems:
        shutil.copyfile(pdf, PAPER_COPY); print(f"copied to {PAPER_COPY}")
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems)); return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
