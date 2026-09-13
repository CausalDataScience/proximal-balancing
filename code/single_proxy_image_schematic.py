#!/usr/bin/env python3
"""Schematic figure for Single Proxy Balancing with image covariates/proxies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("/tmp")
SIDE = 24


def severity_template() -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, SIDE)
    xx, yy = np.meshgrid(grid, grid)
    blob_pos = np.exp(-((xx + 0.35) ** 2 + (yy - 0.25) ** 2) / 0.08)
    blob_neg = np.exp(-((xx - 0.35) ** 2 + (yy + 0.25) ** 2) / 0.10)
    ridge = 0.45 * np.exp(-(yy**2) / 0.035)
    img = blob_pos - 0.80 * blob_neg + ridge
    img = img - img.mean()
    return img / np.max(np.abs(img))


def add_box(ax, x, y, w, h, text, fc, ec, fontsize=7.6, lw=1.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def add_circle(ax, x, y, r, text, fc, ec, fontsize=7.6, lw=1.0, linestyle="solid"):
    patch = Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=linestyle)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)
    return patch


def add_arrow(ax, start, end, color="#374151", lw=1.2, style="solid", rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)
    return arrow


def main() -> None:
    img = severity_template()
    fig, ax = plt.subplots(figsize=(6.9, 2.45), dpi=260)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel backgrounds.
    panels = [
        (0.020, 0.080, 0.305, 0.840, "Observed system"),
        (0.355, 0.080, 0.285, 0.840, "Image representation"),
        (0.670, 0.080, 0.310, 0.840, "Proxy balancing"),
    ]
    for x, y, w, h, title in panels:
        add_box(ax, x, y, w, h, "", "#f8fafc", "#cbd5e1", lw=0.8)
        ax.text(x + 0.015, y + h - 0.058, title, ha="left", va="center", fontsize=7.8, weight="bold")

    # Panel 1: causal system.
    add_circle(ax, 0.105, 0.690, 0.052, "hidden\n$U$", "#e5e7eb", "#6b7280", linestyle="dashed")
    add_box(ax, 0.053, 0.315, 0.095, 0.105, "measured\n$X$", "#dbeafe", "#2563eb")
    add_box(ax, 0.183, 0.325, 0.067, 0.090, "treat\n$A$", "#fee2e2", "#dc2626", fontsize=7.2)
    add_box(ax, 0.271, 0.325, 0.044, 0.090, "$Y$", "#ffedd5", "#ea580c", fontsize=8.0)
    add_box(ax, 0.213, 0.635, 0.085, 0.142, "", "#ccfbf1", "#0f766e")
    ax.text(0.256, 0.753, "proxy $W$", ha="center", va="center", fontsize=7.3)
    ax.imshow(img, cmap="RdBu_r", extent=(0.222, 0.289, 0.655, 0.720), zorder=4, aspect="auto")

    add_arrow(ax, (0.153, 0.682), (0.213, 0.705), "#6b7280", style="dashed")
    add_arrow(ax, (0.122, 0.640), (0.198, 0.415), "#6b7280", style="dashed", rad=-0.10)
    add_arrow(ax, (0.141, 0.650), (0.285, 0.418), "#6b7280", style="dashed", rad=-0.10)
    add_arrow(ax, (0.148, 0.365), (0.183, 0.370), "#2563eb")
    add_arrow(ax, (0.143, 0.340), (0.271, 0.350), "#2563eb", rad=-0.08)
    add_arrow(ax, (0.250, 0.370), (0.271, 0.370), "#dc2626")
    ax.text(0.047, 0.185, "Observed data: $(X,W,A,Y)$\nTarget: effect of $A$ on $Y$", fontsize=7.0, ha="left")

    # Panel 2: image/covariate encoders.
    add_box(ax, 0.382, 0.635, 0.088, 0.105, "measured\n$X$", "#dbeafe", "#2563eb", fontsize=7.0)
    add_box(ax, 0.382, 0.405, 0.088, 0.135, "", "#ccfbf1", "#0f766e", fontsize=7.0)
    ax.text(0.426, 0.515, "proxy $W$", ha="center", va="center", fontsize=7.0)
    ax.imshow(img, cmap="RdBu_r", extent=(0.394, 0.458, 0.425, 0.487), zorder=4, aspect="auto")
    add_box(ax, 0.520, 0.630, 0.092, 0.112, "$R_X=H(X)$", "#eef2ff", "#4f46e5", fontsize=7.2)
    add_box(ax, 0.520, 0.420, 0.092, 0.112, "$R_W=G(W)$", "#dcfce7", "#16a34a", fontsize=7.2)
    add_arrow(ax, (0.470, 0.690), (0.520, 0.690), "#4f46e5")
    add_arrow(ax, (0.470, 0.480), (0.520, 0.480), "#16a34a")
    add_box(
        ax,
        0.388,
        0.185,
        0.214,
        0.110,
        "Key condition:\n$R_W$ preserves the $U$ signal",
        "#ffffff",
        "#94a3b8",
        fontsize=7.1,
        lw=0.8,
    )

    # Panel 3: balancing and target.
    rng = np.random.default_rng(20260706)
    tx = 0.710 + 0.070 * rng.random(14)
    ty = 0.515 + 0.130 * rng.random(14)
    cx = 0.705 + 0.105 * rng.random(24)
    cy = 0.260 + 0.145 * rng.random(24)
    ax.scatter(cx, cy, s=12, color="#93c5fd", edgecolors="#1d4ed8", linewidths=0.35, zorder=3)
    ax.scatter(tx, ty, s=14, color="#fca5a5", edgecolors="#b91c1c", linewidths=0.35, zorder=3)
    ax.text(0.700, 0.675, "treated", fontsize=6.8, color="#b91c1c")
    ax.text(0.700, 0.218, "controls", fontsize=6.8, color="#1d4ed8")
    add_arrow(ax, (0.805, 0.420), (0.872, 0.500), "#374151", lw=1.3)
    add_box(ax, 0.875, 0.445, 0.088, 0.140, "weighted\ncontrols\nmatch", "#fef9c3", "#ca8a04", fontsize=6.9)
    add_box(
        ax,
        0.706,
        0.755,
        0.225,
        0.058,
        "Balance on $(R_X,R_W)$",
        "#ffffff",
        "#94a3b8",
        fontsize=7.2,
        lw=0.8,
    )
    add_box(
        ax,
        0.705,
        0.105,
        0.240,
        0.078,
        "Then latent imbalance in $U$\nshrinks without observing $U$",
        "#ffffff",
        "#94a3b8",
        fontsize=6.8,
        lw=0.8,
    )
    ax.text(0.835, 0.315, "$\\widehat{\\tau}$ for $A\\to Y$", fontsize=8.3, color="#111827", ha="center")

    # Global caption line inside the figure.
    ax.text(
        0.500,
        0.035,
        "Single Proxy Balancing: use image-derived proxy representations to balance hidden severity, not just observed covariates.",
        fontsize=7.2,
        ha="center",
        color="#111827",
    )

    fig.tight_layout(pad=0.08)
    fig.savefig(OUT_DIR / "single_proxy_image_schematic.pdf")
    fig.savefig(OUT_DIR / "single_proxy_image_schematic.png", dpi=320)
    print(OUT_DIR / "single_proxy_image_schematic.pdf")
    print(OUT_DIR / "single_proxy_image_schematic.png")


if __name__ == "__main__":
    main()
