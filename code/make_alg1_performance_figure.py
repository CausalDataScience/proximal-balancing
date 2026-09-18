"""Section 5 figure: Algorithm 1 performance in the three renderings, 1x3.

Top row: one sample of W in each rendering (generators probe_structured_scm, probe_highdim_w, probe_image_w, seed 20260915).
Below: one dot per sealed data set at n = 12,000 (E12 SCM-1: 20, SCM-6-v3 confirmation: 30, SCM-7-v4 confirmation: 15).
Writes figures/section5/section5-alg1-performance.{pdf,png}.
"""
import glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import make_section5_figures as m
import probe_structured_scm as ps
import probe_highdim_w as hd
import probe_image_w as im

R, LIGHT, SEED = m.RESULTS, "#b9b8b2", 20260915

def shards(p):
    return [json.loads(Path(x).read_text()) for x in sorted(glob.glob(str(R / p)))]

e12 = json.loads((R / "e12_scm1_ncurve_confirm_v2.json").read_text())
syn = [q for q in e12["cells"] if q["n_total"] == 12000]
vec, img = shards("scm6_v3/shards/confirmation_*.json"), shards("scm7_v4/shards/confirmation_*.json")

role = ps.generate_role(ps.SCM_FAMILY["SCM-1"], 1, SEED)
seven = [role["M"][0, 1], role["M"][0, 2], role["M"][0, 3], role["M"][0, 4], role["I"][0], role["C"][0], role["M"][0, 0]]
hdd = hd.generate(1, SEED)
vector = np.concatenate([hdd["H"][1][0], hdd["H"][2][0], hdd["H"][3][0], hdd["H"][4][0], [hdd["I"][0], hdd["C"][0]], hdd["H"][0][0]])
lat = im.latents(1, SEED)
thumbs = [im.block(lat, j)[0][24:72, 24:72] for j in (1, 2, 3, 4, 0)]
assert len(vector) == 1282

rows = [("Adjust for $X$", m.GRAY), ("Adjust for $X$\nand all of $W$", m.GRAY), ("Learned summary (all $W$),\nno balance check", m.GRAY),
        ("PROBE (ours)", m.BLUE), ("Oracle\n(uses hidden $U$)", LIGHT)]
est = lambda q: q["estimate"]
cols = [(syn, [lambda q: q["baselines"]["X"], lambda q: q["baselines"]["raw_XW"], None, est, lambda q: q["baselines"]["oracle"]]),
        (vec, [lambda q: q["baselines"]["X"], lambda q: q["baselines"]["raw_ridge"], lambda q: q["baselines"]["blockwise_pca"], est, lambda q: q["baselines"]["oracle"]]),
        (img, [lambda q: q["baselines"]["X"], lambda q: q["baselines"]["raw_cnn"], lambda q: q["baselines"]["cnn_channels"], est, lambda q: q["baselines"]["oracle"]])]
titles = ["Synthetic\n$W$ = 7 numbers", "High-dimensional\n$W$ = 1,282 numbers", "Image\n$W$ = five 96$\\times$96 images"]

W, H = 6.5, 4.15
with plt.rc_context({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "savefig.bbox": "standard"}):
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 2.9], hspace=0.03, wspace=0.12)
    heads = [fig.add_subplot(gs[0, c]) for c in range(3)]
    axes = [fig.add_subplot(gs[1, c]) for c in range(3)]
    for k, (ax, (cells, getters)) in enumerate(zip(axes, cols)):
        for i, (label, colour) in enumerate(rows):
            y = len(rows) - 1 - i
            if getters[i] is None:
                ax.text(0.45, y, "not applicable", ha="center", va="center", fontsize=8, color=LIGHT)
                continue
            xs = np.array([getters[i](q) for q in cells], dtype=float)
            jitter = np.random.default_rng(100 * k + i).uniform(-0.16, 0.16, len(xs))
            ax.scatter(xs, y + jitter, s=16, color=colour, edgecolors="white", linewidths=0.35, zorder=3)
        ax.axvline(1.0, color=m.INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.set_xlim(-0.35, 1.3); ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=m.HAIR, lw=0.5, zorder=0); ax.set_axisbelow(True)
        if k:
            ax.set_yticks([])
    axes[0].set_yticks(range(len(rows)), [label for label, _ in rows][::-1])
    for h in heads:
        h.set_axis_off(); h.set_xlim(0, 1); h.set_ylim(0, 1)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_w = max(t.get_window_extent(renderer).width for t in axes[0].get_yticklabels()) / fig.dpi
    fig.subplots_adjust(left=(label_w + 0.1) / W, right=0.99, bottom=0.5 / H, top=0.99)
    fig.canvas.draw()
    for h, t in zip(heads, titles):
        h.text(0.5, 1.0, t, ha="center", va="top", fontsize=8.5)
    # synthetic: the seven numbers of one unit as small bars around zero
    h = heads[0]; x0, span = 0.14, 0.72; bw = span / 7; base, scale = 0.36, 0.06
    base = 0.43
    h.plot([x0 - 0.02, x0 + span + 0.02], [base, base], color=m.AXIS, lw=0.6)
    for c, v in enumerate(seven):
        h.add_patch(Rectangle((x0 + c * bw + 0.15 * bw, base), 0.7 * bw, float(np.clip(v, -2.5, 2.5)) * scale, fc=m.INK2, ec="none"))
    for c in range(4):
        h.text(x0 + (c + 0.5) * bw, 0.17, f"$W_{c + 1}$", ha="center", va="center", fontsize=7, color=m.INK2)
    h.plot([x0 + 4 * bw + 0.1 * bw, x0 + 7 * bw - 0.1 * bw], [0.225, 0.225], color=m.AXIS, lw=0.6)
    h.text(x0 + 5.5 * bw, 0.17, "$W_5$", ha="center", va="center", fontsize=7, color=m.INK2)
    h.text(0.5, 0.04, "each bar is one number", ha="center", va="center", fontsize=6.5, color=m.INK2)
    # high-dimensional: 1,282 numbers as one strip
    h = heads[1]
    h.imshow(np.clip(vector, -2.5, 2.5)[None, :], aspect="auto", cmap="gray_r", extent=(0.04, 0.96, 0.28, 0.55),
             vmin=-2.5, vmax=2.5, interpolation="nearest")
    edges = np.cumsum([0, 256, 256, 256, 256, 258]) / 1282
    for e in edges[1:-1]:
        h.plot([0.04 + 0.92 * e] * 2, [0.26, 0.57], color="white", lw=1.4)
    for c in range(5):
        h.text(0.04 + 0.92 * (edges[c] + edges[c + 1]) / 2, 0.17, f"$W_{c + 1}$", ha="center", va="center", fontsize=7, color=m.INK2)
    h.text(0.5, 0.04, "each shade is one number", ha="center", va="center", fontsize=6.5, color=m.INK2)
    h.set_xlim(0, 1); h.set_ylim(0, 1)
    # image: five square crops
    h = heads[2]
    bb = h.get_window_extent(renderer)
    tw = 0.16; th = tw * bb.width / bb.height
    for c, t in enumerate(thumbs):
        xa = 0.04 + c * (tw + 0.03)
        inset = h.inset_axes([xa, 0.25, tw, th])
        inset.imshow(t, cmap="gray_r", interpolation="nearest"); inset.set_xticks([]); inset.set_yticks([])
        for sp in inset.spines.values():
            sp.set_visible(True); sp.set_color(m.AXIS); sp.set_linewidth(0.5)
        h.text(xa + tw / 2, 0.17, f"$W_{c + 1}$", ha="center", va="center", fontsize=7, color=m.INK2)
    h.text(0.5, 0.04, "each image is one block", ha="center", va="center", fontsize=6.5, color=m.INK2)
    hp = heads[0].get_position()
    fig.text(hp.x0 - 0.1 / W, hp.y0 + 0.42 * hp.height, "One sample\nof $W$", ha="right", va="center", fontsize=9, color=m.INK2)
    x0f, x1f = axes[0].get_position().x0, axes[-1].get_position().x1
    fig.text((x0f + x1f) / 2, 0.04 / H, "Estimated treatment effect  (dashed line: true effect)", ha="center", va="bottom", fontsize=9)
    m.save(fig, "section5-alg1-performance")
