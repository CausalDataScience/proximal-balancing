#!/usr/bin/env python3
"""Compact NSF-style schematic for Single Proxy Balancing."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("/tmp")


def severity_image(side: int = 28) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, side)
    xx, yy = np.meshgrid(grid, grid)
    blob_pos = np.exp(-((xx + 0.35) ** 2 + (yy - 0.25) ** 2) / 0.08)
    blob_neg = np.exp(-((xx - 0.35) ** 2 + (yy + 0.25) ** 2) / 0.10)
    ridge = 0.45 * np.exp(-(yy**2) / 0.035)
    img = blob_pos - 0.80 * blob_neg + ridge
    img -= img.mean()
    return img / np.max(np.abs(img))


def arrow(ax, xy1, xy2, color="#334155", lw=1.0, dashed=False, rad=0.0):
    patch = FancyArrowPatch(
        xy1,
        xy2,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def box(ax, x, y, w, h, text="", fc="white", ec="#94a3b8", lw=0.9, fs=7.2, weight=None):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)
    return patch


def node(ax, x, y, text, fc="white", ec="#334155", color="#111827", dashed=False):
    circ = Circle((x, y), 0.033, facecolor=fc, edgecolor=ec, linewidth=0.9, linestyle="--" if dashed else "-")
    ax.add_patch(circ)
    ax.text(x, y, text, ha="center", va="center", fontsize=8.0, color=color)
    return circ


def mini_distribution(ax, x, y, w, h, title, after=False):
    xx = np.linspace(-3.2, 3.2, 200)
    treated = np.exp(-0.5 * ((xx - 0.75) / 0.65) ** 2)
    control = np.exp(-0.5 * ((xx + 0.70) / 0.75) ** 2)
    weighted = np.exp(-0.5 * ((xx - 0.62) / 0.72) ** 2)
    treated /= treated.max()
    control /= control.max()
    weighted /= weighted.max()
    to_x = x + w * (xx - xx.min()) / (xx.max() - xx.min())
    base = y + 0.18 * h
    amp = 0.55 * h
    ax.plot(to_x, base + amp * treated, color="#dc2626", lw=1.35)
    ax.plot(to_x, base + amp * (weighted if after else control), color="#2563eb", lw=1.35)
    ax.fill_between(to_x, base, base + amp * treated, color="#fecaca", alpha=0.42)
    ax.fill_between(to_x, base, base + amp * (weighted if after else control), color="#bfdbfe", alpha=0.42)
    ax.plot([x, x + w], [base, base], color="#475569", lw=0.55)
    ax.text(x, y + h - 0.018, title, ha="left", va="top", fontsize=6.9, weight="bold")
    ax.text(x + w - 0.004, base - 0.025, "$R$ score", ha="right", va="top", fontsize=5.6)


def main() -> None:
    img = severity_image()
    fig, ax = plt.subplots(figsize=(6.95, 2.08), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Three compact regions, with Figure-7-like DAG as the first region.
    ax.text(0.030, 0.925, "1. Single proxy structure", fontsize=8.0, weight="bold", ha="left")
    node(ax, 0.120, 0.585, "$U$", fc="#f1f5f9", ec="#64748b", color="#64748b", dashed=True)
    node(ax, 0.055, 0.340, "$A$", fc="#fff1f2", ec="#ef4444", color="#dc2626")
    node(ax, 0.185, 0.340, "$Y$", fc="#eff6ff", ec="#3b82f6", color="#2563eb")
    node(ax, 0.232, 0.585, "$W$", fc="#ecfdf5", ec="#059669", color="#047857")
    arrow(ax, (0.088, 0.340), (0.152, 0.340), color="#111827", lw=1.0)
    arrow(ax, (0.110, 0.555), (0.070, 0.370), color="#64748b", lw=0.9, dashed=True)
    arrow(ax, (0.132, 0.555), (0.175, 0.370), color="#64748b", lw=0.9, dashed=True)
    arrow(ax, (0.153, 0.585), (0.199, 0.585), color="#64748b", lw=0.9, dashed=True)
    ax.text(0.042, 0.150, "$W$: pre-treatment proxy\n$A$ does not cause $W$", fontsize=6.7, ha="left")

    arrow(ax, (0.284, 0.475), (0.340, 0.475), color="#475569", lw=1.0)

    # Representation region.
    ax.text(0.355, 0.925, "2. Representation", fontsize=8.0, weight="bold", ha="left")
    box(ax, 0.350, 0.610, 0.088, 0.135, "$X$", fc="#dbeafe", ec="#2563eb", fs=8.0)
    box(ax, 0.350, 0.325, 0.088, 0.170, "", fc="#ccfbf1", ec="#0f766e")
    ax.text(0.394, 0.470, "$W$", fontsize=8.0, ha="center", va="center")
    ax.imshow(img, cmap="RdBu_r", extent=(0.365, 0.423, 0.345, 0.405), zorder=4, aspect="auto")
    box(ax, 0.495, 0.610, 0.115, 0.135, "$R_X=H(X)$", fc="#eef2ff", ec="#4f46e5", fs=7.5)
    box(ax, 0.495, 0.325, 0.115, 0.170, "$R_W=G(W)$", fc="#dcfce7", ec="#16a34a", fs=7.5)
    arrow(ax, (0.438, 0.678), (0.495, 0.678), color="#4f46e5", lw=1.2)
    arrow(ax, (0.438, 0.410), (0.495, 0.410), color="#16a34a", lw=1.2)
    ax.text(
        0.350,
        0.165,
        "Required: $R_W$ retains the\nhidden-severity signal in $W$.",
        fontsize=6.6,
        ha="left",
    )

    arrow(ax, (0.635, 0.475), (0.682, 0.475), color="#475569", lw=1.0)

    # Balancing region: before/after distribution match.
    ax.text(0.700, 0.925, "3. Proxy balancing", fontsize=8.0, weight="bold", ha="left")
    box(
        ax,
        0.700,
        0.748,
        0.252,
        0.100,
        "Weight controls to match treated\non $R=(R_X,R_W)$",
        fc="#ffffff",
        ec="#94a3b8",
        fs=6.4,
    )
    mini_distribution(ax, 0.700, 0.430, 0.105, 0.245, "before", after=False)
    arrow(ax, (0.815, 0.530), (0.852, 0.530), color="#475569", lw=1.0)
    mini_distribution(ax, 0.860, 0.430, 0.105, 0.245, "after", after=True)
    ax.plot([0.708, 0.730], [0.314, 0.314], color="#dc2626", lw=1.4)
    ax.text(0.734, 0.314, "treated", color="#dc2626", fontsize=5.8, va="center")
    ax.plot([0.795, 0.817], [0.314, 0.314], color="#2563eb", lw=1.4)
    ax.text(0.821, 0.314, "controls / weighted controls", color="#2563eb", fontsize=5.8, va="center")
    box(
        ax,
        0.708,
        0.145,
        0.245,
        0.120,
        "If $R_W$ is informative,\nbalancing $R$ reduces imbalance in $U$.",
        fc="#f8fafc",
        ec="#94a3b8",
        fs=6.4,
    )

    ax.text(
        0.500,
        0.035,
        "Single Proxy Balancing: a pre-treatment proxy $W$ supplies a representation that makes hidden-severity imbalance visible enough to balance.",
        fontsize=6.8,
        ha="center",
        color="#111827",
    )

    fig.tight_layout(pad=0.08)
    fig.savefig(OUT_DIR / "single_proxy_balancing_schematic_v2.pdf")
    fig.savefig(OUT_DIR / "single_proxy_balancing_schematic_v2.png", dpi=340)
    print(OUT_DIR / "single_proxy_balancing_schematic_v2.pdf")
    print(OUT_DIR / "single_proxy_balancing_schematic_v2.png")


if __name__ == "__main__":
    main()
