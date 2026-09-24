"""The RHC figure: the estimate PROBE returns beside the published analyses of the same data.

One panel, sized for a wrapfigure.  Four rows, chosen so that every row is a doubly robust estimator of the
same contrast or the raw contrast itself, which keeps the comparison like for like:

  observational contrast  E[Y | A = 1] - E[Y | A = 0], computed here from the data, no adjustment at all
  standard DR             Cui et al. (2022) Table 3, adjusts for the recorded covariates and nothing else
  proximal DR             Cui et al. (2022) Table 3, uses the four proxies after a role is assigned to each
  PROBE                   this work, receives ten proxies in five blocks and no role label

None of the four is a known value.  The data set has no ground truth and no oracle, so the panel carries no
reference line; it shows four estimates of the same unknown quantity and how far apart they sit.

Sources, all read and never recomputed here:
  results/noise_scaled_v3.json          the v3 screen: fifteen retained choices, all keeping W1, and their median
  results/rhc/aggregate_uncertainty_v1.json  the spread of that median, from resampling the aggregation
  materials/real_world_data/rhc/rhc.csv the observational contrast
  Cui et al. (2022) Table 3, transcribed in PUBLISHED
"""
from __future__ import annotations

import csv
import json

import matplotlib.pyplot as plt
import numpy as np

import make_section5_figures as m
import make_section5_v2_figures as v2
import rhc_data as rd

NO_PROXY, NEEDS_ROLES, PROBE = m.GRAY, m.ORANGE, m.BLUE

# Cui et al. (2022) Table 3, printed as estimate (standard deviation).
PUBLISHED = [("standard DR$^\\dagger$", -1.17, 0.32, NO_PROXY),
             ("proximal DR$^\\dagger$\nroles assigned", -1.66, 0.43, NEEDS_ROLES)]


def observational_contrast() -> tuple[float, float]:
    """E[Y | A = 1] - E[Y | A = 0] with its unpooled standard error, straight from the file."""
    with rd.CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    a = np.array([1.0 if r[rd.TREATMENT] == rd.TREATED_VALUE else 0.0 for r in rows])
    y = np.array([float(r[rd.OUTCOME]) for r in rows])
    treated, control = y[a == 1], y[a == 0]
    se = np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
    return float(treated.mean() - control.mean()), float(se)


def main() -> int:
    # the v3 rule (noise_scaled.py): the threshold is the data set's own noise floor
    retained = json.loads((v2.m.RESULTS / "noise_scaled_v3.json").read_text())["rhc"]
    if any("1" in nm for nm in retained["retained"]) or len(retained["retained"]) != 15:
        raise ValueError(f"unexpected retained set {retained['retained']}")
    crude, crude_se = observational_contrast()

    # The reported quantity is the median of the largest agreeing group, so its interval comes from
    # resampling that aggregation, not from one member's standard error (rhc_aggregate_uncertainty.py).
    spread = json.loads((v2.m.RESULTS / "rhc" / "aggregate_uncertainty_v1.json").read_text())["resampled"]
    rows = [("PROBE\nno roles given", retained["output"], (spread["q025"], spread["q975"]), PROBE)]
    rows += [(lab, v, (v - 1.96 * se, v + 1.96 * se), c) for lab, v, se, c in PUBLISHED]
    rows += [("$E[Y|A{=}1]-E[Y|A{=}0]$\nno adjustment", crude,
              (crude - 1.96 * crude_se, crude + 1.96 * crude_se), NO_PROXY)]

    W, H = 2.6, 1.95
    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.8,
                         "ytick.labelsize": 6.5, "savefig.bbox": "tight"}):
        fig, ax = plt.subplots(figsize=(W, H))
        for i, (label, value, interval, colour) in enumerate(rows):
            row = len(rows) - 1 - i
            ax.axhline(row, color=m.HAIR, lw=0.4, zorder=0)
            ax.plot(list(interval), [row, row], color=colour, lw=1.5, solid_capstyle="round", zorder=3)
            ax.scatter([value], [row], s=22, color=colour, edgecolors="white", linewidths=0.5, zorder=4)
            ax.text(value, row + 0.26, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2, color=colour)
        # No reference line.  RHC has no known effect and no oracle, because the hidden confounder it is
        # used to study is unidentified by construction; an unlabelled dashed line would read as an answer.
        ax.set_xlim(-2.75, 0.45)
        ax.set_xticks([-2, -1, 0])
        ax.set_ylim(-0.6, len(rows) - 0.25)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([label for label, _, _, _ in reversed(rows)], fontsize=6.5, linespacing=1.15)
        for tick, (_, _, _, colour) in zip(ax.get_yticklabels(), reversed(rows)):
            tick.set_color(colour)
        ax.tick_params(axis="y", length=0, pad=1)
        ax.set_title("RHC as observed:\nno ground truth, no oracle", fontsize=7.2, linespacing=1.2)
        ax.set_xlabel("effect on days survived in 30", fontsize=7)
        ax.grid(False)
        m.save(fig, "rhc-comparison")

    lo, hi = spread["q025"], spread["q975"]
    print(f"observational contrast {crude:.3f} (se {crude_se:.3f})")
    print(f"PROBE v3 {retained['output']:.3f}, resampled aggregate sd {spread['sd']:.3f}, "
          f"95% interval ({lo:.2f}, {hi:.2f}); retained {retained['n_retained']} choices, all of which keep W1")
    print(f"published values inside that interval: "
          f"{[label.split(chr(10))[0] for label, v, _, _ in PUBLISHED if lo <= v <= hi]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
