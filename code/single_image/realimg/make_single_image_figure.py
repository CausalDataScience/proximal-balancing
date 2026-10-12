"""Figure for the single-image proxy paragraph (Section 5) in the style of Figure 3.

Left: one chest X-ray with the five blocks, then the same image with the treatment footprint and with the
treatment-response mark.  Right: four scenario panels; each dot is one replicate's estimate minus the true effect
(supervised block encoder, 10 replicates), a black tick marks the mean; rows: adjust for the whole image, PROBE, oracle.
Writes probe-single-image-v1.{pdf,png} to the directory given by --outdir (default: scratchpad/realimg).
"""
from __future__ import annotations

import argparse
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

BLUE, GREEN, GRAY, ORANGE, DARK = "#2a78d6", "#166534", "#898781", "#eb6834", "#52514e"
INK, INK2, HAIR, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
STYLE = {"font.family": "serif", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
         "xtick.labelsize": 7, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
         "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6}
SCEN = [(0, "Clean image"), (1, "Footprint,\ncentral strip"), (2, "Footprint,\nlung block"), (3, "Treatment\nresponse")]
ROWS = [("all_blocks", "Whole image", DARK), ("PROBE", "PROBE (ours)", BLUE), ("oracle", "Oracle", GREEN)]


def load_rows():
    rows = []
    for f in ("v5sup_r10a.jsonl", "v5sup_r10b.jsonl"):
        rows += [json.loads(l) for l in (rb.OUT / f).read_text().splitlines() if l.strip()]
    return rows


def example_images():
    x, label = rb.load(); m = rb.masks()
    i = int(np.where(label == 1)[0][3])
    base = x[i]
    tube = np.zeros((64, 64), np.float32); tube[4:45, 31:33] = 0.35
    foot = np.clip(base + tube * m[4], 0, 1)
    yy, xx = np.mgrid[0:64, 0:64]
    blob = (np.exp(-((yy - 20) ** 2 + (xx - 31.5) ** 2) / (2 * 2.0 ** 2)) * m[4]).astype(np.float32)
    resp = np.clip(base + 0.25 * 2.0 * blob, 0, 1)
    return [(base, "Blocks"), (foot, "Footprint"), (resp, "Response")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", type=Path, default=rb.OUT); a = ap.parse_args()
    plt.rcParams.update(STYLE)
    rows = load_rows()
    fig = plt.figure(figsize=(5.5, 2.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 2.55], wspace=0.02, left=0.01, right=0.995, top=0.86, bottom=0.17)
    gl = gs[0].subgridspec(1, 3, wspace=0.18)
    for k, (img, title) in enumerate(example_images()):
        ax = fig.add_subplot(gl[k]); ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); ax.set_axis_off()
        ax.set_title(title, pad=2, fontsize=7)
        if k == 0:
            for j, ((r0, r1), (c0, c1)) in enumerate(rb.BLOCKS):
                ax.add_patch(Rectangle((c0 - .5, r0 - .5), c1 - c0 + 1, r1 - r0 + 1, fill=False, lw=0.7,
                                       ec=ORANGE if j == 4 else BLUE))
                ax.text((c0 + c1) / 2, (r0 + r1) / 2, f"$W_{j + 1}$", color="white", ha="center", va="center", fontsize=6)
    gr = gs[1].subgridspec(1, 5, wspace=0.08, width_ratios=[1.0, 1, 1, 1, 1])
    rng = np.random.default_rng(0)
    lab = fig.add_subplot(gr[0]); lab.set_axis_off(); lab.set_ylim(-.6, 2.6)
    for i, (key, name, col) in enumerate(ROWS):
        lab.text(1.0, 2 - i, name, color=col if key != "all_blocks" else INK, ha="right", va="center", fontsize=8, transform=lab.get_yaxis_transform())
    for k, (c, title) in enumerate(SCEN):
        ax = fig.add_subplot(gr[k + 1]); rr = [r for r in rows if r["scenario"] == c]
        for i, (key, name, col) in enumerate(ROWS):
            v = np.array([r[key] for r in rr]) - 1.0
            y = (2 - i) + rng.uniform(-.16, .16, len(v))
            ax.scatter(v, y, s=9, color=col, alpha=.85, linewidths=0, zorder=3)
            ax.plot([v.mean()] * 2, [(2 - i) - .3, (2 - i) + .3], color=INK, lw=1.0, zorder=4)
        ax.axvline(0, color=HAIR, lw=0.8, zorder=1)
        ax.set_xlim(-0.12, 0.42); ax.set_ylim(-.6, 2.6); ax.set_yticks([])
        ax.set_xticks([0, .2, .4]); ax.set_xticklabels(["0", "0.2", "0.4"])
        ax.set_title(title, pad=2, linespacing=1.0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fig.text(0.66, 0.02, "Estimate minus true effect (10 replicates; black tick = mean)", ha="center", va="bottom", fontsize=7.5, color=INK2)
    a.outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(a.outdir / f"probe-single-image-v1.{ext}", dpi=200 if ext == "png" else None)
    # numbers for the caption / table
    for c, title in SCEN:
        rr = [r for r in rows if r["scenario"] == c]
        line = " | ".join(f"{name} {np.mean([r[key] for r in rr]):.3f} (RMSE {np.sqrt(np.mean([(r[key]-1)**2 for r in rr])):.3f})" for key, name, _ in ROWS)
        print(f"{title.replace(chr(10), ' ')}: naive {np.mean([r['naive'] for r in rr]):.3f} | {line} | n={len(rr)}")
    print("saved", a.outdir / "probe-single-image-v1.pdf")


if __name__ == "__main__":
    main()
