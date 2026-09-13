#!/usr/bin/env python3
"""Square wrapfigure for Single Proxy Balancing: representation plus balancing."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("/tmp")


def severity_image(side: int = 24) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, side)
    xx, yy = np.meshgrid(grid, grid)
    blob_pos = np.exp(-((xx + 0.35) ** 2 + (yy - 0.25) ** 2) / 0.08)
    blob_neg = np.exp(-((xx - 0.35) ** 2 + (yy + 0.25) ** 2) / 0.10)
    ridge = 0.45 * np.exp(-(yy**2) / 0.035)
    img = blob_pos - 0.80 * blob_neg + ridge
    img -= img.mean()
    return img / np.max(np.abs(img))


def box(ax, x, y, w, h, text="", fc="white", ec="#94a3b8", fs=7.0, lw=0.9, weight=None):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)
    return patch


def arrow(ax, xy1, xy2, color="#334155", lw=1.1, rad=0.0):
    patch = FancyArrowPatch(
        xy1,
        xy2,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def distribution_pair(ax, x, y, w, h):
    xx = np.linspace(-3.0, 3.0, 200)
    treated = np.exp(-0.5 * ((xx - 0.70) / 0.66) ** 2)
    control = np.exp(-0.5 * ((xx + 0.70) / 0.76) ** 2)
    weighted = np.exp(-0.5 * ((xx - 0.60) / 0.72) ** 2)
    treated /= treated.max()
    control /= control.max()
    weighted /= weighted.max()

    gap = 0.055
    panel_w = (w - gap) / 2
    base = y + 0.18 * h
    amp = 0.54 * h

    def plot_panel(px: float, blue_curve: np.ndarray, title: str) -> None:
        tx = px + panel_w * (xx - xx.min()) / (xx.max() - xx.min())
        ax.fill_between(tx, base, base + amp * treated, color="#fecaca", alpha=0.42)
        ax.fill_between(tx, base, base + amp * blue_curve, color="#bfdbfe", alpha=0.42)
        ax.plot(tx, base + amp * treated, color="#dc2626", lw=1.35)
        ax.plot(tx, base + amp * blue_curve, color="#2563eb", lw=1.35)
        ax.plot([px, px + panel_w], [base, base], color="#475569", lw=0.55)
        ax.text(px + panel_w / 2, y + h - 0.010, title, ha="center", va="top", fontsize=6.9, weight="bold")
        ax.text(px + panel_w / 2, base - 0.050, "score", ha="center", va="top", fontsize=5.5)

    plot_panel(x, control, "before")
    plot_panel(x + panel_w + gap, weighted, "after")


def main() -> None:
    img = severity_image()
    fig, ax = plt.subplots(figsize=(2.10, 2.10), dpi=380)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Representation block.
    box(ax, 0.060, 0.775, 0.225, 0.095, "$X$\n(covariate)", fc="#dbeafe", ec="#2563eb", fs=4.85)
    box(ax, 0.060, 0.625, 0.225, 0.120, "$W$\nhigh-dim proxy\n(e.g., image)", fc="#ccfbf1", ec="#0f766e", fs=3.95)
    box(ax, 0.405, 0.775, 0.255, 0.095, "$R_X$\nencoding of $X$", fc="#eef2ff", ec="#4f46e5", fs=4.85)
    box(ax, 0.405, 0.625, 0.255, 0.120, "$R_W$\nencoding of $W$", fc="#dcfce7", ec="#16a34a", fs=4.85)
    arrow(ax, (0.285, 0.823), (0.405, 0.823), color="#4f46e5")
    arrow(ax, (0.285, 0.685), (0.405, 0.685), color="#16a34a")

    # Balancing block.
    box(
        ax,
        0.065,
        0.402,
        0.870,
        0.082,
        "match controls to treated on\n$R=(R_X,R_W)$",
        fc="#ffffff",
        ec="#94a3b8",
        fs=5.25,
    )
    distribution_pair(ax, 0.085, 0.120, 0.830, 0.230)
    ax.plot([0.105, 0.150], [0.065, 0.065], color="#dc2626", lw=1.3)
    ax.text(0.158, 0.065, "treated", color="#dc2626", fontsize=5.2, va="center")
    ax.plot([0.345, 0.390], [0.065, 0.065], color="#2563eb", lw=1.3)
    ax.text(0.398, 0.065, "controls", color="#2563eb", fontsize=5.2, va="center")

    fig.tight_layout(pad=0.03)
    fig.savefig(OUT_DIR / "single_proxy_balancing_wrap_square.pdf")
    fig.savefig(OUT_DIR / "single_proxy_balancing_wrap_square.png", dpi=420)
    print(OUT_DIR / "single_proxy_balancing_wrap_square.pdf")
    print(OUT_DIR / "single_proxy_balancing_wrap_square.png")


if __name__ == "__main__":
    main()
