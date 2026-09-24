"""The two data sets that are real all the way through: Zika and LaLonde, each with two real proxies.

Every row is the same estimator in the two panels.  Each row carries two marks: an open one for the sample
as published, a filled one for the units inside the declared propensity band.  The band is not cosmetic.
On the published samples the treated arm has an effective size of 1.0 of 185 on Zika and 16.1 of 185 on
LaLonde, so every reweighting estimator is reading a handful of units, and the open marks show what that
does.

LaLonde carries an answer from the randomised arm of the same experiment, drawn as a line with its standard
error.  Zika carries no answer.  The line on the Zika panel is another paper's estimate, it is labelled as
one, and it is not a truth.

Source: results/two_proxy/{zika,lalonde}_v1.json and uncertainty_v1.json.  Nothing is recomputed.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import make_section5_figures as m
import make_section5_v2_figures as v2

TWO = m.RESULTS / "two_proxy"
NO_AUDIT, NEEDS_ROLES, PROBE = v2.NO_AUDIT, v2.NEEDS_ROLES, m.BLUE

ROWS = [("Naive difference", "naive", NO_AUDIT),
        ("Adjust for $X$", "X", NO_AUDIT),
        ("Adjust for $X$\nand both proxies", "raw", NO_AUDIT),
        ("Proximal 2SLS,\n$W_1$ treats", "p2sls_W1_treats_W2_outcome", NEEDS_ROLES),
        ("Proximal 2SLS,\n$W_2$ treats", "p2sls_W2_treats_W1_outcome", NEEDS_ROLES),
        ("COCA as ATE,\nfrom $W_1$", "coca_W1_ate", NEEDS_ROLES),
        ("COCA as ATE,\nfrom $W_2$", "coca_W2_ate", NEEDS_ROLES),
        ("PROBE (ours)", "probe", PROBE)]

PANELS = [("lalonde", "LaLonde job training\n185 trained, 15,992 CPS controls", (-9500, 11000),
           [-8000, -4000, 0, 4000, 8000], "1978 earnings, dollars"),
          ("zika", "Zika in Brazil\n673 municipalities, 185 exposed", (-5.4, 4.4),
           [-4, -2, 0, 2, 4], "2016 birth rate")]
REFERENCE = {"lalonde": (1794.34, 670.99, "randomised arm of the\nsame experiment, $\\pm$1 s.e."),
             "zika": (-2.180, 0.342, "Park et al. (2024),\nanother estimate, on the treated")}


def load(name: str) -> dict:
    return json.loads((TWO / f"{name}_v1.json").read_text())


def value(reading: dict, key: str, unc: dict, which: str) -> tuple[float | None, float | None]:
    """The estimate and, for PROBE alone, the resampled spread of the aggregate."""
    if key != "probe":
        return reading["comparators"][key], None
    s = unc[which]["significance"]
    return s["output"], (None if s["output"] is None else s["sd"])


def main() -> int:
    W, H = 5.5, 3.1
    unc_all = json.loads((TWO / "uncertainty_v1.json").read_text())["datasets"]
    with plt.rc_context({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "savefig.bbox": "standard"}):
        fig, axes = plt.subplots(1, 2, figsize=(W, H), sharey=True)
        for ax, (name, title, xlim, ticks, xlabel) in zip(axes, PANELS):
            d, unc = load(name), unc_all[name]
            span = xlim[1] - xlim[0]
            ref, se, ref_label = REFERENCE[name]
            if name == "lalonde":
                ax.axvspan(ref - se, ref + se, color=m.HAIR, zorder=0)
                ax.axvline(ref, color=m.INK, lw=1.0, zorder=2)
            else:
                ax.axvline(ref, color=m.GRAY, lw=0.8, ls=(0, (1, 1.6)), zorder=2)
            frac = (ref - xlim[0]) / span
            ha = "center" if 0.22 < frac < 0.78 else ("left" if frac <= 0.22 else "right")
            nudge = 0.0 if ha == "center" else (0.02 if ha == "left" else -0.02) * span
            ax.text(ref + nudge, len(ROWS) - 0.42, ref_label, ha=ha, va="bottom", fontsize=5.8,
                    color=m.INK if name == "lalonde" else m.INK2, linespacing=1.15, zorder=5,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
            for i, (_, keyname, colour) in enumerate(ROWS):
                y = len(ROWS) - 1 - i
                ax.axhline(y, color=m.HAIR, lw=0.4, zorder=0)
                for which, dy, filled in (("whole_sample", 0.20, False), ("common_support", -0.20, True)):
                    v, sd = value(d["readings"][which], keyname, unc, which)
                    if v is None:
                        ax.text(xlim[0] + 0.03 * span, y + dy, "no return", ha="left", va="center",
                                fontsize=5.4, color=m.INK2, style="italic")
                        continue
                    face = colour if filled else "white"
                    if v < xlim[0] or v > xlim[1]:
                        left = v < xlim[0]
                        xe = xlim[0] + 0.035 * span if left else xlim[1] - 0.035 * span
                        ax.scatter([xe], [y + dy], marker="<" if left else ">", s=16, color=colour,
                                   zorder=3)
                        ax.text(xe + (0.045 if left else -0.045) * span, y + dy,
                                f"{v:,.0f}" if abs(v) >= 100 else f"{v:+.2f}",
                                ha="left" if left else "right", va="center", fontsize=5.4, color=m.INK2)
                        continue
                    if sd is not None:
                        ax.plot([v - sd, v + sd], [y + dy] * 2, color=colour, lw=1.0, zorder=3,
                                solid_capstyle="butt")
                    ax.scatter([v], [y + dy], s=20, facecolors=face, edgecolors=colour,
                               linewidths=1.0, zorder=4)
            ax.set_xlim(*xlim)
            ax.set_xticks(ticks)
            if name == "lalonde":
                ax.set_xticklabels([f"{t:,}" for t in ticks])
            ax.set_ylim(-0.6, len(ROWS) + 0.55)
            ax.set_title(title, fontsize=7.2, linespacing=1.25)
            ess = {w: d["readings"][w]["overlap"] for w in ("whole_sample", "common_support")}
            ax.text(0.5, -0.235,
                    "treated effective sample size\n"
                    f"{ess['whole_sample']['treated_ess']:.0f} of {ess['whole_sample']['treated']}"
                    f"  →  {ess['common_support']['treated_ess']:.0f} of "
                    f"{ess['common_support']['treated']}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=6.0, color=m.INK2,
                    linespacing=1.2)
            ax.set_xlabel(xlabel, fontsize=7)
            ax.tick_params(axis="y", length=0)
            ax.grid(False)
        axes[0].set_yticks(range(len(ROWS)))
        axes[0].set_yticklabels([label for label, _, _ in reversed(ROWS)], fontsize=6.6, linespacing=1.05)
        for tick, (_, _, colour) in zip(axes[0].get_yticklabels(), reversed(ROWS)):
            tick.set_color(colour)
        marks = [Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor=m.INK2,
                        markersize=4.5, label="the sample as published"),
                 Line2D([], [], marker="o", ls="none", markerfacecolor=m.INK2, markeredgecolor=m.INK2,
                        markersize=4.5, label="propensity in $[0.05, 0.95]$, declared in advance")]
        m.key(fig, marks, y=1.005, ncol=2)
        v2.fit_labels(fig, axes, W, H)
        fig.subplots_adjust(top=0.805, bottom=0.265)
        m.save(fig, "two-proxy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
