#!/usr/bin/env python3
"""Polish single-proxy-plus-mediator figures for CAREER use."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


SRC = Path("/tmp/paios_single_proxy_mediator_dgps")
OUT = Path("/tmp/paios_single_proxy_mediator_dgps/career")
OUT.mkdir(parents=True, exist_ok=True)

RED = "#d62728"
BLUE = "#1f77b4"
GRAY = "#7f7f7f"
DARK = "#202124"
LIGHT_GRAY = "#f4f6f8"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def savefig(fig, base: Path, dpi: int = 300) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=dpi)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def dgp_short(name: str) -> str:
    return (
        name.replace("DGP1_", "")
        .replace("DGP2_", "")
        .replace("DGP3_", "")
        .replace("_response", "")
    )


def draw_node(ax, xy, label, color=DARK, face="white", size=0.13, lw=1.2):
    circ = patches.Circle(xy, radius=size, facecolor=face, edgecolor=color, linewidth=lw)
    ax.add_patch(circ)
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=9, color=color, weight="bold")


def draw_arrow(ax, xy0, xy1, color=DARK, lw=1.2, dashed=False, rad=0.0):
    style = "->"
    arrow = patches.FancyArrowPatch(
        xy0,
        xy1,
        arrowstyle=style,
        mutation_scale=9,
        linewidth=lw,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def draw_mechanism(ax, compact: bool = False) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    bg = patches.FancyBboxPatch(
        (0.05, 0.46),
        0.38,
        0.42,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=LIGHT_GRAY,
        edgecolor="#c9ced6",
        linewidth=0.8,
    )
    ax.add_patch(bg)
    ax.text(0.24, 0.84, "hidden severity", ha="center", va="center", fontsize=8, color=DARK)
    draw_node(ax, (0.16, 0.64), r"$B$", color=GRAY, face="white", size=0.095)
    draw_node(ax, (0.32, 0.64), r"$C$", color=GRAY, face="white", size=0.095)

    draw_node(ax, (0.16, 0.23), r"$W$", color=GREEN, face="#eef7ee", size=0.10)
    draw_node(ax, (0.54, 0.23), r"$A$", color=RED, face="#fff1f1", size=0.10)
    draw_node(ax, (0.76, 0.23), r"$M$", color=DARK, face="white", size=0.10)
    draw_node(ax, (0.91, 0.23), r"$Y$", color=BLUE, face="#eef5ff", size=0.10)

    draw_arrow(ax, (0.16, 0.54), (0.16, 0.34), color=GREEN, lw=1.3)
    draw_arrow(ax, (0.32, 0.54), (0.53, 0.32), color=GRAY, lw=1.0, dashed=True, rad=-0.10)
    draw_arrow(ax, (0.54, 0.23), (0.66, 0.23), color=RED, lw=1.1)
    draw_arrow(ax, (0.67, 0.23), (0.76, 0.23), color=DARK, lw=1.1)
    draw_arrow(ax, (0.76, 0.23), (0.83, 0.23), color=DARK, lw=1.1)
    draw_arrow(ax, (0.32, 0.54), (0.76, 0.34), color=GRAY, lw=1.0, dashed=True, rad=-0.18)
    draw_arrow(ax, (0.32, 0.54), (0.91, 0.34), color=GRAY, lw=1.0, dashed=True, rad=-0.22)

    ax.text(0.12, 0.42, r"$W=B$", fontsize=8, color=GREEN, ha="center")
    ax.text(0.56, 0.57, r"$C=A\oplus M$", fontsize=9, color=BLUE, ha="center", weight="bold")
    ax.text(0.56, 0.49, "mediator reveals\nmissing bit", fontsize=7.5, color=BLUE, ha="center")
    if not compact:
        ax.text(0.18, 0.05, "single proxy sees only one latent bit", fontsize=8, color=GREEN, ha="center")
        ax.text(0.72, 0.05, "treatment-response pattern recovers the other", fontsize=8, color=BLUE, ha="center")


def plot_full() -> None:
    summary = read_csv(SRC / "simulation_summary.csv")

    fig, (ax0, ax1, ax2) = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.45),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.86, 0.86, 1.34]},
    )

    w_only = np.array([[0.11, 0.06], [0.77, 0.65]])
    decoded = np.array([[0.0, 1.0], [0.0, 1.0]])
    ax0.imshow(w_only, cmap="Reds", vmin=0, vmax=1)
    ax1.imshow(decoded, cmap="Blues", vmin=0, vmax=1)
    for ax, mat, xlabel in [(ax0, w_only, "$W$"), (ax1, decoded, r"decoded $C$")]:
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel(xlabel, labelpad=0)
        ax.set_ylabel("$A$", labelpad=0)
        ax.tick_params(labelsize=7)
        for i in [0, 1]:
            for j in [0, 1]:
                txt = f"{mat[i, j]:.2f}" if mat[i, j] not in [0, 1] else f"{int(mat[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    ax0.set_title("proxy alone:\nsubtype remains mixed", fontsize=8.2)
    ax1.set_title("with mediator:\n$C=A\\oplus M$ decoded", fontsize=8.2)
    ax0.text(-0.24, 1.08, "a", transform=ax0.transAxes, fontsize=10, weight="bold")

    for method, color, label in [
        ("W-only proxy balancing", RED, "W-only"),
        ("single proxy + mediator", BLUE, "W+mediator"),
    ]:
        vals = [r for r in summary if r["method"] == method]
        ns = np.array([int(r["n"]) for r in vals])
        ys = np.array([float(r["mean_abs_error"]) for r in vals])
        ses = np.array([float(r["se_abs_error"]) for r in vals])
        order = np.argsort(ns)
        ns, ys, ses = ns[order], ys[order], ses[order]
        xs = np.arange(len(ns))
        ax2.plot(xs, ys, marker="o", markersize=3.7, linewidth=1.7, color=color, label=label)
        ax2.fill_between(xs, ys - 1.96 * ses, ys + 1.96 * ses, color=color, alpha=0.13)
    ax2.set_title("Finite-sample error", fontsize=9)
    ax2.set_xlabel("sample size", labelpad=1)
    ax2.set_ylabel("mean absolute error", labelpad=1)
    ax2.set_xticks(np.arange(5))
    ax2.set_xticklabels(["200", "500", "1k", "2k", "5k"], fontsize=7)
    ax2.grid(True, alpha=0.22, linewidth=0.6)
    ax2.legend(frameon=False, fontsize=7, loc="upper right")
    ax2.text(-0.18, 1.08, "b", transform=ax2.transAxes, fontsize=10, weight="bold")

    savefig(fig, OUT / "career_single_proxy_mediator_full")


def plot_wrap() -> None:
    exact = read_csv(SRC / "exact_dgp_summary.csv")

    fig, ax1 = plt.subplots(figsize=(3.05, 2.1), constrained_layout=True)

    # Show exact absolute error by DGP. This is more stable at wrapfigure size than
    # a three-method ATE bar plot.
    labels = ["Subtype", "Adherence", "Resistance"]
    x = np.arange(len(labels))
    w_err = np.array([float(r["w_only_abs_error"]) for r in exact])
    med_err = np.array([max(float(r["mediator_abs_error"]), 0.0075) for r in exact])
    width = 0.34
    ax1.bar(x - width / 2, w_err, width, color=RED, label="proxy only")
    ax1.bar(x + width / 2, med_err, width, color=BLUE, label="proxy + mediator")
    ax1.set_ylabel(r"$|\widehat{\tau}-\tau|$", labelpad=1, fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=14, ha="right", fontsize=6.2)
    ax1.set_ylim(0, 0.78)
    ax1.set_yticks([0.0, 0.25, 0.5, 0.75])
    ax1.tick_params(axis="y", labelsize=6.5)
    ax1.grid(True, axis="y", alpha=0.22, linewidth=0.5)
    ax1.legend(frameon=False, fontsize=6.0, loc="upper right", handlelength=1.25)
    ax1.text(1.33, 0.34, r"$W=B$ misses $C$", ha="left", va="center", fontsize=6.1, color=GRAY)
    ax1.text(1.33, 0.26, r"$C=A\oplus M$ recovered", ha="left", va="center", fontsize=6.1, color=BLUE)
    ax1.text(2.16, 0.035, "near zero", color=BLUE, fontsize=5.9, ha="center")
    ax1.set_title("Mediator unlocks a single proxy", fontsize=8.2, pad=2)
    savefig(fig, OUT / "career_single_proxy_mediator_wrapfigure")


def plot_clean_heatmap() -> None:
    # Recast the mechanism heatmap as a compact, grant-friendly panel.
    fig, axes = plt.subplots(1, 2, figsize=(4.9, 1.95), constrained_layout=True)
    w_only = np.array([[0.11, 0.06], [0.77, 0.65]])
    decoded = np.array([[0.0, 1.0], [0.0, 1.0]])
    im0 = axes[0].imshow(w_only, cmap="Reds", vmin=0, vmax=1)
    axes[0].set_title(r"Proxy alone leaves $C$ mixed", fontsize=8)
    axes[0].set_xlabel("$W$", labelpad=0)
    axes[0].set_ylabel("$A$", labelpad=0)
    im1 = axes[1].imshow(decoded, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title(r"Mediator decodes $C=A\oplus M$", fontsize=8)
    axes[1].set_xlabel(r"decoded $C$", labelpad=0)
    axes[1].set_ylabel("$A$", labelpad=0)
    for ax, mat in [(axes[0], w_only), (axes[1], decoded)]:
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.tick_params(labelsize=7)
        for i in [0, 1]:
            for j in [0, 1]:
                txt = f"{mat[i, j]:.2f}" if mat[i, j] not in [0, 1] else f"{int(mat[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.047, pad=0.02)
    fig.colorbar(im1, ax=axes[1], fraction=0.047, pad=0.02)
    savefig(fig, OUT / "career_single_proxy_mediator_heatmap")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    plot_full()
    plot_wrap()
    plot_clean_heatmap()
    print(f"outdir={OUT}")
    for path in sorted(OUT.glob("*")):
        print(path)


if __name__ == "__main__":
    main()
