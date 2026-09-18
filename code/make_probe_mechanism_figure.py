"""PROBE as a mechanism, 1x4 at text width, drawn against the run that implements Algorithm 1 literally.

Panels (a) and (b) are schematic.  The blocks are drawn as images to make the point that a block may be
an image; the thumbnails are one illustrative unit from probe_image_w.latents(1, 7) and belong to no data
set used here.

Panels (c) and (d) carry real numbers from the positive-control audit, which is the only run whose
representation class refers to no block role and whose screen, linking and plurality are the manuscript's:

    Z = (X_1, X_2, W_j) for one kept block j        positive_control_scm.representation
    keep S if max{R_0 - R_1, 0} <= t                positive_control_scm.screen_statistic
    link within 2 rho, return the largest median    positive_control_scm.aggregate

(c) is the screen statistic of all 126 splits on one sealed learning sample; (d) is one sealed evaluation
draw.  Both come from results/positive_control/screen_display_v1.json, a replay of the sealed audit
results/positive_control/confirmation_v1.json.  Writes figures/section5/probe-mechanism.{pdf,png}.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle

import make_section5_figures as m
import probe_image_w as im

DISPLAY = m.RESULTS / "positive_control" / "screen_display_v1.json"
J = 7
W, H = 6.5, 1.55                         # the float is \linewidth wide; keep it shallow
pw, ph, gx, margin = 1.44, 1.43, 0.20, 0.05
FONT, SMALL = 7.2, 6.2

d = json.loads(DISPLAY.read_text())
threshold = d["threshold_t"]
census = d["census"]
objective = np.array([r["objective"] for r in census])
retained = np.array([r["retained"] for r in census])
index = {tuple(r["split"]): k for k, r in enumerate(census)}
example = {tuple(r["split"]): r["objective"] for r in d["screen"]}
draw = d["evaluation_draw"]
estimates = np.array(draw["estimates"])
theta = np.array(draw["theta"])
rho, tau_hat = draw["rho"], draw["output"]
target = np.abs(theta - 1.0) < 1e-8
wrong_x, right_x = float(estimates[~target][0]), float(estimates[target][0])
print(f"screen: {retained.sum()} of {len(census)} splits kept at t = {threshold}")
print(f"draw: {target.sum()} choices at {right_x:.5f}, {(~target).sum()} at {wrong_x:.5f}, "
      f"returned {tau_hat:.5f}, 2rho = {2 * rho:.4f}")

lat = im.latents(1, 7)
thumbs = [im.block(lat, j % im.N_BLOCKS)[0][24:72, 24:72] for j in range(J)]

TITLES = ["(a) Split proxy $W$ into\nblocks; hold a group out",
          "(b) Learn $Z$ from $X$ and\nthe kept blocks",
          "(c) Given $Z$, does the held-out\ngroup still predict $A$?",
          "(d) Repeat; the largest\nagreeing group gives $\\widehat\\tau$"]


def frame(inset, colour=None, dashed=False):
    inset.set_xticks([]); inset.set_yticks([])
    for sp in inset.spines.values():
        sp.set_visible(True); sp.set_color(colour or m.AXIS); sp.set_linewidth(1.2 if colour else 0.5)
        if dashed:
            sp.set_linestyle((0, (2, 1.2)))


def bare(ax):
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(m.AXIS)
    ax.set_facecolor("none")
    ax.set_yticks([])


with plt.rc_context({"font.size": FONT, "savefig.bbox": "standard"}):
    fig = plt.figure(figsize=(W, H))
    y0 = (H - margin - ph) / H
    axs = [fig.add_axes([(margin + k * (pw + gx)) / W, y0, pw / W, ph / H]) for k in range(4)]
    for ax, t in zip(axs, TITLES):
        ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc="#f7f7f5", ec=m.AXIS, lw=0.6, zorder=0))
        ax.text(0.5, 0.97, t, ha="center", va="top", fontsize=FONT)
    for k in range(3):
        p, q = axs[k].get_position(), axs[k + 1].get_position()
        y = p.y0 + 0.45 * p.height
        fig.add_artist(FancyArrowPatch((p.x1 + 0.006, y), (q.x0 - 0.006, y), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=8, color=m.INK2, lw=0.9))
    ratio = pw / ph

    # (a) the proxy in J blocks, one group held out; every group is a candidate
    a = axs[0]
    s, row = 0.122, 0.40
    for j in range(J):
        x0 = 0.055 + j * 0.133
        inset = a.inset_axes([x0, row, s, s * ratio])
        inset.imshow(thumbs[j], cmap="gray_r", interpolation="nearest")
        frame(inset, m.BLUE if j == 0 else None, dashed=(j == 0))
        a.text(x0 + s / 2, row - 0.02, "$W_%d$" % (j + 1), ha="center", va="top",
               fontsize=SMALL, color=m.BLUE if j == 0 else m.INK2)
    a.text(0.055 + s / 2, row + s * ratio + 0.02, "held out", ha="center", va="bottom",
           fontsize=SMALL, color=m.BLUE)
    a.text(0.5, 0.05, "$2^7-2 = 126$ groups in all", ha="center", va="bottom",
           fontsize=SMALL, color=m.INK2)

    # (b) the kept blocks and X give Z; phi is chosen to make the held-out group look balanced
    b = axs[1]
    sb = 0.115
    for k in range(J - 1):
        inset = b.inset_axes([0.085 + k * 0.135, 0.64, sb, sb * ratio])
        inset.imshow(thumbs[k + 1], cmap="gray_r", interpolation="nearest"); frame(inset)
    b.text(0.5, 0.625, "$W_2,\\ldots,W_7$ and $X$", ha="center", va="top", fontsize=SMALL, color=m.INK2)
    b.add_patch(FancyArrowPatch((0.5, 0.560), (0.5, 0.525), arrowstyle="-|>", mutation_scale=6,
                                color=m.INK2, lw=0.8))
    b.add_patch(Polygon([(0.06, 0.515), (0.94, 0.515), (0.74, 0.285), (0.26, 0.285)], closed=True,
                        fc="#dde9f8", ec=m.BLUE, lw=0.9))
    b.text(0.5, 0.450, "pick $\\phi$ to minimise", ha="center", va="center", fontsize=SMALL)
    b.text(0.5, 0.340, "$\\widetilde D_S^2$", ha="center", va="center", fontsize=FONT + 1)
    b.add_patch(FancyArrowPatch((0.5, 0.275), (0.5, 0.240), arrowstyle="-|>", mutation_scale=6,
                                color=m.INK2, lw=0.8))
    for k in range(3):
        b.add_patch(Rectangle((0.38 + k * 0.07, 0.140), 0.06, 0.09, fc=m.BLUE,
                              alpha=0.3 + 0.2 * k, ec="white", lw=0.5))
    b.text(0.62, 0.185, "$Z$", ha="left", va="center", fontsize=FONT)
    b.text(0.5, 0.04, "never sees the held-out group", ha="center", va="bottom",
           fontsize=SMALL, color=m.INK2)

    # (c) the screen statistic of every group against the threshold
    cx = axs[2].inset_axes([0.13, 0.31, 0.82, 0.38])
    rng = np.random.default_rng(11)
    jit = rng.uniform(-0.62, 0.62, size=len(objective))
    for split in ((0,), (3,)):                   # the two named examples sit on the centre line
        jit[index[split]] = 0.0
    cx.scatter(objective[~retained], jit[~retained], s=6, color=m.GRAY, alpha=0.55,
               linewidths=0, zorder=2)
    cx.scatter(objective[retained], jit[retained], s=8, color=m.BLUE, linewidths=0, zorder=3)
    for split, colour in (((0,), m.BLUE), ((3,), m.ORANGE)):
        cx.scatter([example[split]], [0.0], s=30, facecolors="none", edgecolors=colour,
                   linewidths=0.9, zorder=5)
    cx.axvline(threshold, color=m.INK, lw=0.9, ls=(0, (3, 2)), zorder=4)
    cx.set_xscale("log"); cx.set_xlim(8e-8, 6e-1); cx.set_ylim(-0.95, 2.5)
    cx.set_xticks([1e-6, 1e-3], ["$10^{-6}$", "$10^{-3}$"])
    cx.tick_params(axis="x", labelsize=SMALL, length=2, pad=1.5); bare(cx)
    cx.text(1.3e-7, 2.42, "keep 10", ha="left", va="top", fontsize=SMALL, color=m.BLUE)
    cx.text(4.5e-1, 2.42, "drop 116", ha="right", va="top", fontsize=SMALL, color=m.GRAY)
    cx.text(example[(0,)], 0.85, "$W_1$ out", ha="center", va="bottom", fontsize=SMALL, color=m.BLUE)
    cx.text(example[(3,)], 0.85, "$W_4$ out", ha="center", va="bottom", fontsize=SMALL, color=m.ORANGE)
    axs[2].text(0.5, 0.05, "$\\widetilde D_S^2$ per group; keep below $t$", ha="center", va="bottom",
                fontsize=SMALL, color=m.INK2)

    # (d) one estimate per kept group; the largest agreeing group gives the answer
    dx = axs[3].inset_axes([0.11, 0.31, 0.83, 0.38])
    ry = np.linspace(-0.42, 0.42, int(target.sum()))
    wy = np.linspace(-0.18, 0.18, int((~target).sum()))
    dx.scatter(estimates[target], ry, s=15, color=m.BLUE, edgecolors="white", linewidths=0.4, zorder=3)
    dx.scatter(estimates[~target], wy, s=15, color=m.ORANGE, edgecolors="white", linewidths=0.4, zorder=3)
    dx.scatter([tau_hat], [2.00], marker="*", s=60, color=m.INK, zorder=4)
    dx.text(tau_hat - 0.06, 2.00, "$\\widehat\\tau$ = %.2f" % tau_hat, ha="right", va="center",
            fontsize=SMALL + 0.5)
    dx.axvline(1.0, color=m.INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
    dx.annotate("", xy=(wrong_x, 0.80), xytext=(wrong_x + 2 * rho, 0.80),
                arrowprops=dict(arrowstyle="<->", color=m.INK2, lw=0.7, shrinkA=0, shrinkB=0))
    dx.text(wrong_x + rho, 0.90, "$2\\rho$", ha="center", va="bottom", fontsize=SMALL, color=m.INK2)
    dx.text(right_x + 0.07, 0.0, "%d" % target.sum(), ha="left", va="center",
            fontsize=SMALL, color=m.BLUE)
    dx.text(wrong_x - 0.07, 0.0, "%d" % (~target).sum(), ha="right", va="center",
            fontsize=SMALL, color=m.ORANGE)
    dx.set_xlim(0.14, 1.30); dx.set_ylim(-0.60, 2.40)
    dx.set_xticks([0.5, 1.0], ["0.5", "1"])
    dx.tick_params(labelsize=SMALL, length=2, pad=1.5); bare(dx)
    axs[3].text(0.5, 0.05, "one dot per group; dashed $\\tau$", ha="center", va="bottom",
                fontsize=SMALL, color=m.INK2)
    m.save(fig, "probe-mechanism")
