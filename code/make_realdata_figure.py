"""The real-data figure for Section 5: Twins beside RHC, three panels at the ICLR text width.

  (a) Twins.  What each method returns on ten treatment draws over the real covariates, the real hidden
      gestational age and the real outcomes; the dashed line is the exact effect -0.0252.  Rows and colours
      follow the simulation figure, so the reader compares like with like.  Values outside the axis are
      drawn as one triangle at the edge carrying their mean.
  (b) RHC.  The pooled balance statistic of all thirty held-out choices, sorted; the shaded band is the
      pre-registered threshold grid and the one choice inside it is labelled.
  (c) RHC.  That choice's estimate beside the published analyses of the same data, with 95% intervals.

Sources, read and never recomputed:
  results/twins/summary_v1.json, results/twins/park_ate_v1.json
  results/rhc/result_v2_pooled.json, materials/real_world_data/rhc/rhc.csv (the unadjusted contrast)
  Cui et al. (2022) Table 3, transcribed in make_rhc_figure.PUBLISHED
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import make_rhc_figure as rhc
import make_section5_figures as m
import make_section5_v2_figures as v2
import run_rhc as rr

LIGHT = "#b9b8b2"
NO_AUDIT, NEEDS_ROLES, LATENT, PROBE = v2.NO_AUDIT, v2.NEEDS_ROLES, v2.LATENT, m.BLUE
TAU_TWINS = -0.0252


def twins_rows() -> list[tuple[str, np.ndarray, str]]:
    s = json.loads((m.RESULTS / "twins" / "summary_v1.json").read_text())["methods"]
    park = json.loads((m.RESULTS / "twins" / "park_ate_v1.json").read_text())["twins"]
    val = lambda k: np.array(s[k]["values"], dtype=float)
    return [("Adjust for $X$", val("X"), NO_AUDIT),
            ("Adjust for $X$\nand all of $W$", val("raw"), NO_AUDIT),
            ("Proximal 2SLS,\nroles given", val("p2sls_given_roles"), NEEDS_ROLES),
            ("Proximal 2SLS,\nroles swapped", val("p2sls_swapped_roles"), NEEDS_ROLES),
            ("Park SPC as ATE,\nclean proxy", np.array([p["park_clean_w1"]["ate"] for p in park]), NEEDS_ROLES),
            ("CEVAE", val("cevae"), LATENT),
            ("PROBE (ours)", val("probe_pooled"), PROBE),
            ("Oracle\n(uses hidden $U$)", val("oracle"), LIGHT)]


def strip(ax, rows, xlim):
    """Rows of dots with a mean bar; a whole row outside the axis becomes one triangle at that edge."""
    for i, (label, values, colour) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.axhline(y, color=m.HAIR, lw=0.4, zorder=0)
        if np.all(values < xlim[0]) or np.all(values > xlim[1]):
            left = np.all(values < xlim[0])
            x_edge = xlim[0] + 0.03 * (xlim[1] - xlim[0]) if left else xlim[1] - 0.03 * (xlim[1] - xlim[0])
            ax.scatter([x_edge], [y], marker="<" if left else ">", s=24, color=colour, edgecolors="white",
                       linewidths=0.4, zorder=3)
            ax.text(x_edge + (0.05 if left else -0.05) * (xlim[1] - xlim[0]), y, f"{values.mean():+.2f}",
                    ha="left" if left else "right", va="center", fontsize=6.3, color=m.INK2)
        else:
            m.dots(ax, values, y, colour, size=10)


def main() -> int:
    rows_a = twins_rows()
    d = rhc.load() if hasattr(rhc, "load") else None
    v2r = json.loads((m.RESULTS / "rhc" / "result_v2_pooled.json").read_text())
    per_split = v2r["per_split"]
    order = sorted(per_split, key=lambda n: per_split[n]["pooled_upper"])
    upper = np.array([per_split[n]["pooled_upper"] for n in order])
    keep = np.array([n == rhc.RETAINED for n in order])
    y_b = np.arange(len(order))[::-1]
    retained = next(r for r in v2r["path_over_threshold_grid"] if r["n_retained"] == 1)
    crude, crude_se = rhc.observational_contrast()
    rows_c = [("PROBE\nno roles given", retained["output"], retained["se_member"], PROBE)]
    rows_c += list(rhc.PUBLISHED)
    rows_c += [("$E[Y\\mid A{=}1]$\n$-\\,E[Y\\mid A{=}0]$", crude, crude_se, NO_AUDIT)]

    W, H = 5.5, 2.75
    xlim_a = (-0.148, 0.022)
    with plt.rc_context({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "savefig.bbox": "standard"}):
        fig = plt.figure(figsize=(W, H))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 0.85, 1.10], wspace=0.24,
                              left=0.175, right=0.845, bottom=0.185, top=0.83)
        a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))

        # (a) Twins
        strip(a, rows_a, xlim_a)
        a.axvline(TAU_TWINS, color=m.INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
        a.set_xlim(*xlim_a)
        a.set_xticks([-0.1, -0.05, 0.0], ["$-0.10$", "$-0.05$", "$0$"])
        a.set_ylim(-0.6, len(rows_a) - 0.4)
        a.set_yticks(range(len(rows_a)))
        a.set_yticklabels([label for label, _, _ in reversed(rows_a)], fontsize=6.8, linespacing=1.1)
        for tick, (_, _, colour) in zip(a.get_yticklabels(), reversed(rows_a)):
            tick.set_color(m.INK if colour is LIGHT else colour)
        a.tick_params(axis="y", length=0, pad=2)
        a.set_title("(a) Twins: real outcomes,\nexact effect $\\tau=-0.025$", fontsize=7.5, linespacing=1.25)
        a.set_xlabel("effect on first-year mortality", fontsize=7)
        a.grid(False)

        # (b) RHC audit
        b.axvspan(min(rr.GRID), max(rr.GRID), color=m.HAIR, zorder=0)
        b.scatter(upper[~keep], y_b[~keep], s=9, color=m.GRAY, edgecolors="white", linewidths=0.3, zorder=3)
        b.scatter(upper[keep], y_b[keep], s=24, color=PROBE, edgecolors="white", linewidths=0.4, zorder=4)
        b.annotate("retained:\nhold out\n$(\\mathtt{ph1},\\mathtt{hema1})$", (upper[keep][0], y_b[keep][0]),
                   xytext=(0.50, 0.80), textcoords="axes fraction", fontsize=6.0, color=PROBE, ha="left",
                   va="center", linespacing=1.15,
                   arrowprops=dict(arrowstyle="-", color=PROBE, lw=0.6, shrinkA=0, shrinkB=3))
        b.set_xscale("log")
        b.set_xlim(8e-4, 6e-2)
        b.set_xticks([1e-3, 1e-2], ["$10^{-3}$", "$10^{-2}$"])
        b.set_ylim(-1.5, len(order) + 1.2)
        b.set_yticks([])
        b.set_title("(b) RHC: balance\naudit, 30 choices", fontsize=7.5, linespacing=1.25)
        b.set_xlabel("screen statistic", fontsize=7)
        b.grid(False)

        # (c) RHC estimates against the published ones
        for i, (label, value, se, colour) in enumerate(rows_c):
            row = len(rows_c) - 1 - i
            c.axhline(row, color=m.HAIR, lw=0.4, zorder=0)
            c.plot([value - 1.96 * se, value + 1.96 * se], [row, row], color=colour, lw=1.4,
                   solid_capstyle="round", zorder=3)
            c.scatter([value], [row], s=20, color=colour, edgecolors="white", linewidths=0.5, zorder=4)
            c.text(value, row + 0.27, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2, color=colour)
        c.axvline(0.0, color=m.INK, lw=0.8, ls=(0, (3, 2)), zorder=2)
        c.set_xlim(-2.75, 0.45)
        c.set_xticks([-2, -1, 0])
        c.set_ylim(-0.6, len(rows_c) - 0.25)
        c.yaxis.tick_right()
        c.set_yticks(range(len(rows_c)))
        c.set_yticklabels([label for label, _, _, _ in reversed(rows_c)], fontsize=6.5, linespacing=1.12)
        for tick, (_, _, _, colour) in zip(c.get_yticklabels(), reversed(rows_c)):
            tick.set_color(colour)
        c.tick_params(axis="y", length=0, pad=1)
        c.set_title("(c) RHC: real data,\nno ground truth", fontsize=7.5, linespacing=1.25)
        c.set_xlabel("effect on days survived in 30", fontsize=7)
        c.grid(False)
        m.save(fig, "realdata")

    for label, values, _ in rows_a:
        print(f"{label.replace(chr(10), ' '):32s} mean {values.mean():+.4f}  MAE {np.mean(np.abs(values - TAU_TWINS)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
