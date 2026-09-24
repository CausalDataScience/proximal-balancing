"""The KSPC benchmark adapted to a binary treatment: every method, 20 replicates, error against the exact
effect of 1.  Single panel, sized as a wrapfigure.  Source: results/kspc_design/replicate_*.json."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import kspc_design as kd
import make_section5_figures as m
import make_section5_v2_figures as v2

ROWS = [("Naive difference", lambda s: s["comparators"]["naive"], v2.NO_AUDIT),
        ("Adjust for all of $W$", lambda s: s["comparators"]["raw"], v2.NO_AUDIT),
        ("2SLS, $W_1$ treats", lambda s: s["comparators"]["p2sls_W1_treats_W2_outcome"], v2.NEEDS_ROLES),
        ("2SLS, $W_2$ treats", lambda s: s["comparators"]["p2sls_W2_treats_W1_outcome"], v2.NEEDS_ROLES),
        ("KSPC (SKPV), given $W_1$", lambda s: s["kspc"]["skpv_W1"]["ate"], v2.NEEDS_ROLES),
        ("KSPC (SKPV), given image", lambda s: s["kspc"]["skpv_W2"]["ate"], v2.NEEDS_ROLES),
        ("KSPC (SPMMR), given $W_1$", lambda s: s["kspc"]["spmmr_W1"]["ate"], v2.NEEDS_ROLES),
        ("KSPC (SPMMR), given image", lambda s: s["kspc"]["spmmr_W2"]["ate"], v2.NEEDS_ROLES),
        ("PROBE (ours)", lambda s: s["probe"]["readings"]["significance"]["output"], m.BLUE),
        ("Oracle (uses $U$)", lambda s: s["comparators"]["oracle"], "#b9b8b2")]


def main() -> int:
    shards = [json.loads((m.RESULTS / "kspc_design" / f"replicate_{r}.json").read_text()) for r in kd.REPLICATES]
    W, H = 3.2, 3.3
    xlim = (-0.45, 0.8)
    with plt.rc_context({"font.size": 7.0, "xtick.labelsize": 6.6, "ytick.labelsize": 6.4,
                         "savefig.bbox": "standard"}):
        fig, ax = plt.subplots(figsize=(W, H))
        for i, (label, get, colour) in enumerate(ROWS):
            y = len(ROWS) - 1 - i
            ax.axhline(y, color=m.HAIR, lw=0.4, zorder=0)
            vals = np.array([get(s) - kd.TAU for s in shards])
            m.dots(ax, vals, y, colour, size=7)
            out = int(np.sum((vals < xlim[0]) | (vals > xlim[1])))
            if out:
                ax.text(xlim[1] - 0.02, y + 0.28, f"{out} off", ha="right", va="bottom", fontsize=5.2, color=m.INK2)
            ax.text(xlim[1] + 0.03, y, f"{np.mean(np.abs(vals)):.3f}", ha="left", va="center", fontsize=5.8,
                    color=m.INK2, clip_on=False)
        ax.text(xlim[1] + 0.03, len(ROWS) - 0.55, "MAE", ha="left", va="bottom", fontsize=5.8, color=m.INK2,
                clip_on=False)
        ax.axvline(0.0, color=m.INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([r[0] for r in reversed(ROWS)])
        for tick, (_, _, colour) in zip(ax.get_yticklabels(), reversed(ROWS)):
            tick.set_color(m.INK if colour == "#b9b8b2" else colour)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("estimate minus the exact effect", fontsize=6.8)
        ax.set_title("KSPC benchmark, binary treatment", fontsize=7.2, pad=8)
        ax.grid(False)
        fig.subplots_adjust(left=0.47, right=0.87, top=0.92, bottom=0.13)
        m.save(fig, "kspc-benchmark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
