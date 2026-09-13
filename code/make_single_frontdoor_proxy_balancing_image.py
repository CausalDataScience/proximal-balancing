#!/usr/bin/env python3
"""Create a square CAREER-style front-door proxy balancing illustration."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


OUT = Path("/tmp")
PNG = OUT / "single_frontdoor_proxy_balancing_wrap_square_imagegen.png"
PDF = OUT / "single_frontdoor_proxy_balancing_wrap_square_imagegen.pdf"

BLUE = "#0b58d0"
GREEN = "#0b7d18"
PURPLE = "#5b249b"
ORANGE = "#d68000"
RED = "#e21a1a"
CONTROL_BLUE = "#064cff"
GRAY = "#606060"
LIGHT_BLUE = "#eef5ff"
LIGHT_GREEN = "#eef8ef"
LIGHT_PURPLE = "#f5efff"
LIGHT_ORANGE = "#fff4e5"


def rounded_box(ax, xy, w, h, edge, face, lw=2.2, radius=0.018):
    box = patches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    return box


def arrow(ax, start, end, color, lw=2.8):
    arr = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="Simple,tail_width=0.85,head_width=8,head_length=12",
        color=color,
        linewidth=0,
        mutation_scale=1,
    )
    ax.add_patch(arr)


def density_panel(ax, xy, w, h, title, before=True):
    rounded_box(ax, xy, w, h, edge="#666666", face="white", lw=1.15, radius=0.012)
    x0, y0 = xy
    ax.text(x0 + w / 2, y0 + h - 0.033, title, ha="center", va="center", fontsize=18, weight="bold")

    xs = np.linspace(-3.2, 3.2, 380)
    if before:
        y_t = np.exp(-0.5 * ((xs + 1.2) / 0.55) ** 2) + 0.07 * np.exp(-0.5 * ((xs + 2.35) / 0.55) ** 2)
        y_c = np.exp(-0.5 * ((xs - 1.25) / 0.55) ** 2) + 0.07 * np.exp(-0.5 * ((xs - 2.35) / 0.55) ** 2)
        x_t = x0 + 0.05 + (xs + 3.2) / 6.4 * (w - 0.10)
        x_c = x_t
    else:
        y_t = np.exp(-0.5 * ((xs + 0.10) / 0.70) ** 2) + 0.035 * np.exp(-0.5 * ((xs + 1.95) / 0.85) ** 2)
        y_c = np.exp(-0.5 * ((xs - 0.08) / 0.70) ** 2) + 0.035 * np.exp(-0.5 * ((xs - 1.95) / 0.85) ** 2)
        x_t = x0 + 0.05 + (xs + 3.2) / 6.4 * (w - 0.10)
        x_c = x_t

    scale = h * 0.20
    base = y0 + 0.060
    ax.plot(x_t, base + y_t / y_t.max() * scale, color=RED, lw=2.3)
    ax.plot(x_c, base + y_c / y_c.max() * scale, color=CONTROL_BLUE, lw=2.3)
    ax.plot([x0 + 0.05, x0 + w - 0.05], [base - 0.012, base - 0.012], color="black", lw=1.5)
    ax.text(x0 + w / 2, y0 + 0.017, "score", ha="center", va="center", fontsize=17, family="DejaVu Serif")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0,
        }
    )

    fig = plt.figure(figsize=(8, 8), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Outer print boundary echoes the reference image.
    ax.add_patch(patches.Rectangle((0.002, 0.002), 0.996, 0.996, fill=False, edgecolor="black", linewidth=1.8))

    left_x = 0.13
    right_x = 0.59
    box_w = 0.29
    box_h = 0.105
    rows = [0.78, 0.63, 0.48]

    rounded_box(ax, (left_x, rows[0]), box_w, box_h, BLUE, LIGHT_BLUE)
    ax.text(left_x + box_w / 2, rows[0] + 0.066, r"$X$", ha="center", va="center", fontsize=25)
    ax.text(left_x + box_w / 2, rows[0] + 0.030, "(covariates)", ha="center", va="center", fontsize=17)
    rounded_box(ax, (right_x, rows[0]), box_w, box_h, PURPLE, LIGHT_PURPLE)
    ax.text(right_x + box_w / 2, rows[0] + 0.066, r"$R_X$", ha="center", va="center", fontsize=25)
    ax.text(right_x + box_w / 2, rows[0] + 0.030, r"encoding of $X$", ha="center", va="center", fontsize=16)
    arrow(ax, (left_x + box_w + 0.018, rows[0] + box_h / 2), (right_x - 0.018, rows[0] + box_h / 2), BLUE)

    rounded_box(ax, (left_x, rows[1]), box_w, box_h, GREEN, LIGHT_GREEN)
    ax.text(left_x + box_w / 2, rows[1] + 0.066, r"$W$", ha="center", va="center", fontsize=25)
    ax.text(left_x + box_w / 2, rows[1] + 0.030, "single proxy", ha="center", va="center", fontsize=17)
    rounded_box(ax, (right_x, rows[1]), box_w, box_h, GREEN, LIGHT_GREEN)
    ax.text(right_x + box_w / 2, rows[1] + 0.066, r"$R_W$", ha="center", va="center", fontsize=25)
    ax.text(right_x + box_w / 2, rows[1] + 0.030, r"encoding of $W$", ha="center", va="center", fontsize=16)
    arrow(ax, (left_x + box_w + 0.018, rows[1] + box_h / 2), (right_x - 0.018, rows[1] + box_h / 2), GREEN)

    rounded_box(ax, (left_x, rows[2]), box_w, box_h, ORANGE, LIGHT_ORANGE)
    ax.text(left_x + box_w / 2, rows[2] + 0.067, r"$(A,M)$", ha="center", va="center", fontsize=23)
    ax.text(left_x + box_w / 2, rows[2] + 0.031, "mediator response", ha="center", va="center", fontsize=15.2)
    rounded_box(ax, (right_x, rows[2]), box_w, box_h, ORANGE, LIGHT_ORANGE)
    ax.text(right_x + box_w / 2, rows[2] + 0.067, r"$R_C$", ha="center", va="center", fontsize=25)
    ax.text(right_x + box_w / 2, rows[2] + 0.031, r"decodes missing $C$", ha="center", va="center", fontsize=15.2)
    arrow(ax, (left_x + box_w + 0.018, rows[2] + box_h / 2), (right_x - 0.018, rows[2] + box_h / 2), ORANGE)

    rounded_box(ax, (0.21, 0.345), 0.58, 0.088, GRAY, "white", lw=1.65, radius=0.013)
    ax.text(0.50, 0.398, "balance treated and controls on", ha="center", va="center", fontsize=15.5, weight="bold")
    ax.text(0.50, 0.360, r"$R=(R_X,\;R_W,\;R_C)$", ha="center", va="center", fontsize=20)

    density_panel(ax, (0.12, 0.135), 0.36, 0.155, "before", before=True)
    density_panel(ax, (0.52, 0.135), 0.36, 0.155, "after", before=False)

    y_leg = 0.066
    ax.plot([0.265, 0.345], [y_leg, y_leg], color=RED, lw=2.8)
    ax.text(0.378, y_leg - 0.005, "treated", ha="left", va="center", fontsize=19)
    ax.plot([0.545, 0.625], [y_leg, y_leg], color=CONTROL_BLUE, lw=2.8)
    ax.text(0.660, y_leg - 0.005, "controls", ha="left", va="center", fontsize=19)

    fig.savefig(PNG, dpi=220)
    fig.savefig(PDF)
    plt.close(fig)
    print(PNG)
    print(PDF)


if __name__ == "__main__":
    main()
