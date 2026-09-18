"""Diagnostic figure for the seven-block confirmation.

Three panels, all from the sealed result files; no simulation is run here.
(a) what the balance screen let through, (b) why the retained wrong candidates
linked to the clean ones, (c) how much they moved the returned estimate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import probe_structured_scm as p

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGDIR = Path(__file__).resolve().parents[1] / "figures" / "diagnostics"
TAGS = {"SCM-1": "g1", "SCM-2": "g2", "SCM-3": "g3", "SCM-4": "g4", "SCM-5": "g5"}
SCM_COLOR = {"SCM-1": "#2a78d6", "SCM-2": "#eb6834", "SCM-3": "#898781",
             "SCM-4": "#7a4fb5", "SCM-5": "#1b8a5a"}
SCM_MARKER = {"SCM-1": "o", "SCM-2": "s", "SCM-3": "^", "SCM-4": "D", "SCM-5": "v"}
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#898781", "#2b2b2b"
LABEL = {"clean": "clean", "bad_singleton": "$W_4$ out", "mixed": "mixed",
         "drops_cause": "$W_6$ out", "drops_confounder": "$W_7$ out"}
CAT_COLOR = {"clean": BLUE, "bad_singleton": GRAY, "mixed": GRAY,
             "drops_cause": GRAY, "drops_confounder": ORANGE}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.bbox": "standard", "pdf.fonttype": 42,
})


def collect() -> tuple[list, list, list]:
    screen_rows, link_rows, replicate_rows = [], [], []
    for scm, tag in TAGS.items():
        path = RESULTS / f"probe_structured_scm_blockwise_j7_confirm_v1_{tag}.json"
        for row in json.loads(path.read_text())["rows"]:
            drawn = set(p.budget_permutation(row["split_sample_seed"])[: row["budget_m"]])
            rho, output = row["rho"], row["aggregation"]["output"]
            retained: dict[tuple[int, ...], float] = {}
            for fit in row["split_fits"]:
                split = tuple(fit["split"])
                if split not in drawn:
                    continue
                category = p.split_category(split)
                worst = max(r["screen"]["signed_gap"] / (2.0 * r["screen"]["se"])
                            for r in fit["rotations"])
                screen_rows.append({"scm": scm, "cat": category, "t": worst,
                                    "passed": fit["pass_all_four"]})
                if fit["pass_all_four"]:
                    retained[split] = float(np.mean([r["estimate"] for r in fit["rotations"]]))
            clean = [v for s, v in retained.items() if p.split_category(s) == "clean"]
            if clean and rho:
                for split, estimate in retained.items():
                    category = p.split_category(split)
                    if category == "clean":
                        continue
                    gap = min(abs(estimate - c) for c in clean)
                    link_rows.append({"scm": scm, "cat": category, "ratio": gap / (2.0 * rho)})
            members = row["aggregation"]["members"]
            if output is not None and len(members) == 1:
                component = [tuple(s) for s in members[0]]
                share = float(np.mean([p.split_category(s) == "drops_confounder" for s in component]))
                replicate_rows.append({"scm": scm, "share": share, "error": output - 1.0})
    return screen_rows, link_rows, replicate_rows


def box(ax, x, values, color):
    """One compact box: p10 to p90 whisker, interquartile body, median tick."""
    q10, q25, q50, q75, q90 = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    ax.plot([x, x], [q10, q90], color=color, lw=0.9, zorder=2, solid_capstyle="butt")
    ax.add_patch(plt.Rectangle((x - 0.17, q25), 0.34, max(q75 - q25, 1e-9),
                               facecolor=color, alpha=0.30, edgecolor=color, lw=0.7, zorder=3))
    ax.plot([x - 0.22, x + 0.22], [q50, q50], color=color, lw=1.6, zorder=4, solid_capstyle="butt")


def panel_screen(ax, rows):
    order = ["clean", "bad_singleton", "mixed", "drops_cause", "drops_confounder"]
    for i, category in enumerate(order):
        values = np.array([r["t"] for r in rows if r["cat"] == category])
        box(ax, i, values, CAT_COLOR[category])
        for scm in TAGS:
            v = [r["t"] for r in rows if r["cat"] == category and r["scm"] == scm]
            if v:
                ax.plot(i + 0.30, np.median(v), marker=SCM_MARKER[scm], ms=2.4,
                        color=INK, mfc="white", mew=0.5, zorder=5)
        rate = np.mean([r["passed"] for r in rows if r["cat"] == category]) * 100
        ax.text(i, 12.5, f"{rate:.0f}%", ha="center", va="bottom", fontsize=7,
                color=CAT_COLOR[category])
    ax.axhline(1.0, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(4.45, 1.05, "boundary", ha="right", va="bottom", fontsize=6.5, color=INK)
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.9)
    ax.set_yticks([0, 1, 2, 5, 10])
    ax.set_yticklabels(["0", "1", "2", "5", "10"])
    ax.set_ylim(-0.4, 22)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABEL[c] for c in order], rotation=30, ha="right")
    ax.set_ylabel("screen statistic / boundary")
    ax.set_title("(a) what the screen let through", loc="left")


def panel_link(ax, rows):
    order = ["bad_singleton", "drops_confounder"]
    for i, category in enumerate(order):
        values = np.array([r["ratio"] for r in rows if r["cat"] == category])
        if not len(values):
            continue
        rng = np.random.default_rng(7)
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(values)), values, s=4,
                   color=CAT_COLOR[category], alpha=0.35, linewidths=0, zorder=2)
        box(ax, i, values, CAT_COLOR[category])
        share = np.mean(values <= 1) * 100
        ax.text(i, 24, f"{share:.0f}% linked", ha="center", va="bottom", fontsize=7,
                color=CAT_COLOR[category])
    ax.axhline(1.0, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(1.45, 1.12, "link boundary", ha="right", va="bottom", fontsize=6.5, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.008, 60)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABEL[c] for c in order])
    ax.set_ylabel(r"distance to clean cluster / $2\rho$")
    ax.set_title("(b) which retained ones linked", loc="left")


def panel_pull(ax, rows):
    for scm in TAGS:
        xs = [r["share"] for r in rows if r["scm"] == scm]
        ys = [r["error"] for r in rows if r["scm"] == scm]
        ax.scatter(xs, ys, s=9, color=SCM_COLOR[scm], marker=SCM_MARKER[scm],
                   alpha=0.85, linewidths=0, label=scm, zorder=3)
    share = np.array([r["share"] for r in rows])
    error = np.array([r["error"] for r in rows])
    edges = [-0.01, 0.001, 0.34, 0.67, 1.01]
    centres, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (share >= lo) & (share < hi)
        if mask.sum():
            centres.append(share[mask].mean())
            means.append(error[mask].mean())
    ax.plot(centres, means, color=INK, lw=1.2, marker="_", ms=9, zorder=4)
    ax.axhline(0.0, color=INK, lw=0.7, zorder=1)
    ax.text(0.30, 0.004, "no error", fontsize=6.5, color=INK, va="bottom")
    ax.text(0.97, 0.052, f"correlation {np.corrcoef(share, error)[0, 1]:+.2f}",
            ha="right", va="top", fontsize=7, color=INK)
    ax.set_xlabel("share of $W_7$-out in the group")
    ax.set_ylabel(r"$\widehat\tau-1$")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.145, 0.065)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, -0.02), ncol=2,
              handletextpad=0.2, borderpad=0.1, labelspacing=0.15, columnspacing=0.7)
    ax.set_title("(c) pull on the estimate", loc="left")


def main() -> int:
    screen_rows, link_rows, replicate_rows = collect()
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.05))
    panel_screen(axes[0], screen_rows)
    panel_link(axes[1], link_rows)
    panel_pull(axes[2], replicate_rows)
    fig.subplots_adjust(left=0.088, right=0.985, top=0.855, bottom=0.235, wspace=0.50)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    stem = FIGDIR / "j7-screen-and-linking"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    print("wrote", stem.with_suffix(".pdf"))
    print(f"panel a rows {len(screen_rows)} | panel b rows {len(link_rows)} | panel c rows {len(replicate_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
