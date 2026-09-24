"""The ground-truth real-data figure: four benchmarks with an exactly known effect, one panel each, the
rows and colours of the simulation figure.

  Twins        real covariates, real hidden gestational age, real outcomes; proxies generated
  ACIC 2016    real covariates, hidden x_44; potential outcomes of the benchmark; proxies generated
  IHDP         real covariates, hidden column 5; potential outcomes of the benchmark; proxies generated
  RHC planted  real covariates, real hidden APACHE score, REAL proxies, real outcome plus a planted -1.0

The x axis of each panel is the estimate minus the exact effect of that replicate, so the dashed line at 0
means "right" in every panel although the effects differ (about -0.025, +3.8, +4.7 and -1.0).
PROBE is the v3 rule (noise_scaled.py); everything else comes from each benchmark's summary_v1.json.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import make_section5_figures as m
import make_section5_v2_figures as v2

LIGHT = "#b9b8b2"
NO_AUDIT, NEEDS_ROLES, LATENT, PROBE = v2.NO_AUDIT, v2.NEEDS_ROLES, v2.LATENT, m.BLUE
PANELS = [("twins", "Twins\nreal outcomes, $\\tau=-0.025$"), ("acic", "ACIC 2016\n$\\tau\\approx 3.8$"),
          ("ihdp", "IHDP\n$\\tau\\approx 4.7$"), ("rhc_planted", "RHC, planted\nreal proxies, $\\tau=-1$")]
XLIM = {"twins": (-0.125, 0.045), "acic": (-2.4, 1.1), "ihdp": (-1.7, 2.9), "rhc_planted": (-2.9, 0.9)}
TICKS = {"twins": [-0.1, -0.05, 0], "acic": [-2, -1, 0, 1], "ihdp": [-1, 0, 1, 2], "rhc_planted": [-2, -1, 0]}


def rows_for(name: str) -> list[tuple[str, np.ndarray, str]]:
    s = json.loads((m.RESULTS / name / "summary_v1.json").read_text())
    v3 = json.loads((m.RESULTS / "noise_scaled_v3.json").read_text())["real_data"][name]["replicates"]
    # align everything by replicate number: the summary lists replicates in run order, the v3 file in
    # file-name order, and ACIC and IHDP have a different exact effect in every replicate
    order = s["replicates"]
    by_rep = {r["replicate"]: r for r in v3}
    taus = np.array([by_rep[r]["tau"] for r in order])
    methods = s["methods"]
    err = lambda k: np.array(methods[k]["values"], dtype=float) - taus
    probe = np.array([by_rep[r]["output"] for r in order], dtype=float) - taus
    return [("Adjust for $X$", err("X"), NO_AUDIT),
            ("Adjust for $X$\nand all of $W$", err("raw"), NO_AUDIT),
            ("Proximal 2SLS,\nroles given", err("p2sls_given_roles"), NEEDS_ROLES),
            ("Proximal 2SLS,\nroles swapped", err("p2sls_swapped_roles"), NEEDS_ROLES),
            ("CEVAE", err("cevae"), LATENT),
            ("PROBE (ours)", probe, PROBE),
            ("Oracle\n(uses hidden $U$)", err("oracle"), LIGHT)]


def main() -> int:
    W, H = 5.5, 2.75
    with plt.rc_context({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "savefig.bbox": "standard"}):
        fig, axes = plt.subplots(1, 4, figsize=(W, H), sharey=True)
        n_rows = None
        for ax, (name, title) in zip(axes, PANELS):
            rows = rows_for(name); n_rows = len(rows)
            xlim = XLIM[name]
            for i, (label, values, colour) in enumerate(rows):
                y = len(rows) - 1 - i
                ax.axhline(y, color=m.HAIR, lw=0.4, zorder=0)
                if np.all(values < xlim[0]) or np.all(values > xlim[1]):
                    left = np.all(values < xlim[0]); span = xlim[1] - xlim[0]
                    xe = xlim[0] + 0.04 * span if left else xlim[1] - 0.04 * span
                    ax.scatter([xe], [y], marker="<" if left else ">", s=22, color=colour, edgecolors="white",
                               linewidths=0.4, zorder=3)
                    ax.text(xe + (0.06 if left else -0.06) * span, y, f"{values.mean():+.2f}",
                            ha="left" if left else "right", va="center", fontsize=6.2, color=m.INK2)
                else:
                    inside = values[(values >= xlim[0]) & (values <= xlim[1])]
                    m.dots(ax, values, y, colour, size=9)
                    if len(inside) < len(values):
                        ax.text(xlim[1] - 0.02 * (xlim[1] - xlim[0]), y + 0.28, f"{len(values) - len(inside)} off",
                                ha="right", va="bottom", fontsize=5.5, color=m.INK2)
            ax.axvline(0.0, color=m.INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
            ax.set_xlim(*xlim); ax.set_xticks(TICKS[name])
            ax.set_ylim(-0.6, len(rows) - 0.4)
            ax.set_title(title, fontsize=7.5, linespacing=1.25)
            ax.tick_params(axis="y", length=0); ax.grid(False)
        rows = rows_for("twins")
        axes[0].set_yticks(range(n_rows))
        axes[0].set_yticklabels([label for label, _, _ in reversed(rows)], fontsize=7, linespacing=1.1)
        for tick, (_, _, colour) in zip(axes[0].get_yticklabels(), reversed(rows)):
            tick.set_color(m.INK if colour is LIGHT else colour)
        fig.text(0.5, 0.03, "estimate minus the exact effect  (dashed line: no error)", ha="center", va="bottom", fontsize=8)
        v2.fit_labels(fig, axes, W, H)
        m.save(fig, "realdata-groundtruth")
    for name, _ in PANELS:
        for label, values, _ in rows_for(name):
            if label.startswith("PROBE") or label.startswith("Oracle"):
                print(f"{name:12s} {label.split(chr(10))[0]:14s} MAE {np.mean(np.abs(values)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
