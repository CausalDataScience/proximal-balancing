"""Figure 4, version 3: one row, in the style of Figure 3.  The five-scenario dot panels of version 2 moved into Figure 3
(make_section5_v6_figure, chest X-ray group); this figure keeps the example images and the imprint sweep.

Version 2 docstring follows.


Top left: one chest X-ray with the five blocks, the same image with the treatment-response mark, and with a device line
and a response mark in two blocks.  Top right: the imprint sweep, estimate minus true effect against kappa in the mark
U - kappa A + 0.5 nu, lines joining the means of 50 replicates per value.  Bottom: five scenario panels; each dot is one
replicate's estimate minus the true effect (50 replicates, new seeds, sequential rule), a black tick marks the mean;
rows: adjust for the whole image, PROBE, oracle.  Data: results/single_image/pneumoniamnist_v1/v10_anat_r50{,x}_w*.jsonl.
Writes probe-single-image-v3.{pdf,png} to --outdir (default: the results folder).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import realimg_benchmark as rb  # noqa: E402

# Figure 3's colours, fonts and text sizes (make_section5_v3_figures.STYLE, make_section5_v5_figure TEXT_PT / TICK_PT)
BLUE, GREEN, GRAY, ORANGE, DARK = "#2a78d6", "#166534", "#898781", "#eb6834", "#52514e"
INK, INK2, HAIR, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
TEXT_PT, TICK_PT, DOT_S = 10.5, 8.5, 9.0
STYLE = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
         "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
         "font.size": TEXT_PT, "axes.titlesize": TEXT_PT, "axes.labelsize": TEXT_PT, "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
         "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
         "xtick.color": AXIS, "xtick.labelcolor": INK2, "xtick.major.width": 0.6, "xtick.major.size": 2.5, "xtick.major.pad": 1.5,
         "ytick.color": AXIS, "ytick.labelcolor": INK2, "ytick.major.width": 0.6, "ytick.major.size": 2.5, "ytick.major.pad": 1.5,
         "axes.unicode_minus": True, "savefig.dpi": 200}
MAIN = [(0, "Clean image"), (3, "Treatment\nresponse"), (4, "Mediator"), (5, "Response in\ntwo blocks"), (8, "Device and\nresponse")]
ROWS = [("all_blocks", "Pixel CNN", GRAY), ("PROBE_seq_floor", "PROBE (ours)", BLUE), ("oracle", "Oracle", DARK)]
KAPPA = [(0.0, "x", 9), (0.5, "x", 10), (1.0, "x", 11), (1.5, "", 3), (2.0, "x", 12), (3.0, "x", 13)]   # (kappa, file suffix, scenario)
XLIM, XTICKS = (-0.56, 0.86), [-0.5, 0.0, 0.5]


def minus(v):
    return f"{v:g}".replace("-", "−")


def load(suffix):
    rows = []
    for f in sorted(glob.glob(str(rb.OUT / f"v10_anat_r50{suffix}_w*.jsonl"))):
        rows += [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
    return rows


def example_images():
    x, label = rb.load(); m = rb.masks(); base = x[int(np.where(label == 1)[0][3])]
    yy, xx = np.mgrid[0:64, 0:64]
    blob = lambda cy, cx, k: (np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2)) * m[k]).astype(np.float32)
    resp = np.clip(base + 0.25 * 2.0 * blob(20, 31.5, 4), 0, 1)
    # mediator scenario (realimg_v10_confirm, c == 4): the mark at the same place darkens by 0.25 times the
    # standardised M = A + 0.5 nu; shown for a treated unit with standardised M = 2
    med = np.clip(base - 0.25 * 2.0 * blob(20, 31.5, 4), 0, 1)
    return [(base, "Blocks"), (resp, "Response"), (med, "Mediator")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", type=Path, default=rb.OUT); a = ap.parse_args()
    plt.rcParams.update(STYLE)
    main_rows, extra_rows = load(""), load("x")
    rows = main_rows + extra_rows
    fig = plt.figure(figsize=(5.5, 1.72))
    top = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.22, left=0.015, right=0.985, top=0.86, bottom=0.17)
    # ---- left: example images
    gl = top[0].subgridspec(1, 3, wspace=0.12)
    for k, (img, title) in enumerate(example_images()):
        ax = fig.add_subplot(gl[k]); ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); ax.set_axis_off()
        ax.set_title(title, pad=2, fontsize=TICK_PT, linespacing=0.95)
        if k == 0:
            for j, ((r0, r1), (c0, c1)) in enumerate(rb.BLOCKS):
                ax.add_patch(Rectangle((c0 - .5, r0 - .5), c1 - c0 + 1, r1 - r0 + 1, fill=False, lw=0.7, ec=ORANGE if j == 4 else BLUE))
                ax.text((c0 + c1) / 2, (r0 + r1) / 2, f"$W_{j + 1}$", color="white", ha="center", va="center", fontsize=6.5)
    # ---- right: kappa sweep
    ax = fig.add_subplot(top[1]); rng = np.random.default_rng(0)
    for key, name, col in ROWS:
        ks, ms = [], []
        for kappa, suf, c in KAPPA:
            rr = [r for r in (extra_rows if suf else main_rows) if r["scenario"] == c]
            v = np.array([r[key] for r in rr]) - 1.0
            if key != "oracle":
                ax.scatter(kappa + rng.uniform(-.045, .045, len(v)), v, s=DOT_S * 0.45, color=col, alpha=.35, linewidths=0, zorder=2)
            ks.append(kappa); ms.append(v.mean())
        ax.plot(ks, ms, "-o", color=col, lw=1.1, ms=2.6, zorder=4)
        ax.text(3.1, ms[-1], {"Pixel CNN": "Pixel\nCNN", "PROBE (ours)": "PROBE", "Oracle": "Oracle"}[name], color=col, ha="left", va="center", fontsize=TICK_PT, linespacing=0.9)
    ax.plot([-0.15, 3.03], [0, 0], color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(-0.15, 3.95); ax.set_xticks([0, 1, 2, 3]); ax.set_ylim(-0.08, 0.82); ax.set_yticks([0, 0.4, 0.8]); ax.set_yticklabels(["0", "0.4", "0.8"])
    ax.spines["left"].set_visible(True); ax.tick_params(axis="y", length=2.5)
    ax.set_xlabel("treatment imprint $\\kappa$", fontsize=TICK_PT, labelpad=1.5); ax.set_ylabel("estimate minus\ntrue effect", fontsize=TICK_PT, labelpad=2, linespacing=0.9)
    ax.set_title("Mark $U-\\kappa A+\\nu$ in $W_5$", pad=2, fontsize=TICK_PT)
    a.outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(a.outdir / f"probe-single-image-v3.{ext}")
    print("saved", a.outdir / "probe-single-image-v3.pdf")


if __name__ == "__main__":
    main()
