"""Diagram of the Shapes3D setting (SCM-4) for the wrapped figure beside the Shapes3D paragraph of Section 5.

One unit of SCM-1 has six proxy values: W1 to W4 one each, W5 two.  Each value is recorded to one of 15 levels
(family_v2_images.record: 15 equally spaced levels on [-3 sd, 3 sd]) and shown as one Shapes3D image whose
orientation factor equals that level; every other factor of the image is drawn at random (appendix, Image
rendering).  PROBE receives the level that a frozen reader recovers from each image; the pixel CNN learns from all
pixels of the six images.  The six values below are illustrative, in units of each coordinate's sd; the images are
real Shapes3D images (materials/benchmarks/shapes3d/3dshapes.h5, md5 checked against manifest.json) with the
recorded orientation and the other factors drawn with a fixed seed.

Output: manuscript/figures/probe-shapes3d-diagram.{pdf,png}, designed at its printed width.
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import make_section5_v5_figure as f5  # noqa: E402  (manuscript fonts and colours)

BANK = CODE.parent / "materials" / "benchmarks" / "shapes3d"
H5 = BANK / "3dshapes.h5"
OUT = CODE.parent / "manuscript" / "figures" / "probe-shapes3d-diagram"
W, H = 2.70, 1.60                     # inches, printed size
TEXT_PT, SMALL_PT = 8.5, 7.5
VALUES = [0.8, -1.2, 0.3, 1.6, -0.4, 2.1]
LABELS = ["$W_1$", "$W_2$", "$W_3$", "$W_4$", "$W_5$"]
MARGIN, TILE = 0.06, 0.375
GAP = (W - 2 * MARGIN - 6 * TILE) / 5
SIZES = (10, 10, 10, 8, 4, 15)        # floor hue, wall hue, object hue, scale, shape, orientation (fastest)
LEVELS = SIZES[-1]
SEED = 20261014


def tile_x(k: int) -> float:
    return MARGIN + k * (TILE + GAP)


def record(v: float) -> int:
    """family_v2_images.record for one value in sd units."""
    return int(np.clip(np.rint((v + 3.0) / 6.0 * (LEVELS - 1)), 0, LEVELS - 1))


def check_file() -> None:
    manifest = json.loads((BANK / "manifest.json").read_text())["h5"]
    if H5.stat().st_size != manifest["bytes"]:
        raise SystemExit(f"STOP: {H5} has {H5.stat().st_size} bytes, manifest says {manifest['bytes']}")
    md5 = hashlib.md5()
    with open(H5, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            md5.update(chunk)
    if md5.hexdigest() != manifest["md5"]:
        raise SystemExit(f"STOP: md5 of {H5} is {md5.hexdigest()}, manifest says {manifest['md5']}")


def example_images() -> list[tuple[np.ndarray, int]]:
    """For each value: one image with the recorded orientation and the other five factors drawn at random."""
    rng = np.random.default_rng(SEED)
    out = []
    with h5py.File(H5, "r") as f:
        orient_values = np.linspace(-30.0, 30.0, LEVELS)
        for v in VALUES:
            k = record(v)
            other = [int(rng.integers(s)) for s in SIZES[:-1]]
            idx = 0
            for factor, size in zip(other + [k], SIZES):
                idx = idx * size + factor
            label = f["labels"][idx]
            if abs(label[5] - orient_values[k]) > 1e-4 or int(round(label[4])) != other[4]:
                raise SystemExit(f"STOP: image {idx} has labels {label}, expected orientation level {k}, shape {other[4]}")
            out.append((f["images"][idx], k))
    return out


def arrow(ax, p, q, **kw) -> None:
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.7, color=f5.INK2,
                                 shrinkA=0, shrinkB=0, **kw))


def main() -> int:
    check_file()
    images = example_images()
    plt.rcParams.update(f5.style())
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    y_box, h_box = 0.04, 0.36                      # bottom: what each method receives
    y_tile = y_box + h_box + 0.24
    y_num = y_tile + TILE + 0.20
    h_num = 0.19
    y_lab = y_num + h_num + 0.03
    # proxy values of one unit, with block labels (W5 spans its two values), and the image that shows each value
    for k, (v, (img, level)) in enumerate(zip(VALUES, images)):
        x = tile_x(k)
        ax.add_patch(Rectangle((x, y_num), TILE, h_num, fill=False, ec=f5.AXIS, lw=0.6))
        ax.text(x + TILE / 2, y_num + h_num / 2, f"{v:g}".replace("-", "−"), ha="center", va="center",
                fontsize=TEXT_PT, color=f5.INK)
        arrow(ax, (x + TILE / 2, y_num - 0.02), (x + TILE / 2, y_tile + TILE + 0.02))
        ax.imshow(img, extent=(x, x + TILE, y_tile, y_tile + TILE), interpolation="nearest", zorder=1)
        ax.add_patch(Rectangle((x, y_tile), TILE, TILE, fill=False, ec=f5.AXIS, lw=0.5, zorder=2))
        print(f"  value {v:+.1f} -> orientation level {level} ({np.linspace(-30, 30, LEVELS)[level]:+.1f} degrees)")
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    for k, lab in enumerate(LABELS):
        xc = tile_x(k) + TILE / 2 if k < 4 else (tile_x(4) + tile_x(5) + TILE) / 2
        ax.text(xc, y_lab, lab, ha="center", va="bottom", fontsize=TEXT_PT, color=f5.INK)
    ax.plot([tile_x(4), tile_x(5) + TILE], [y_lab - 0.005] * 2, color=f5.AXIS, lw=0.6)
    # the two ways the images are used
    half = (W - 2 * MARGIN - 0.10) / 2
    boxes = [(MARGIN, "frozen reader: level\nof each image → PROBE", f5.BLUE),
             (MARGIN + half + 0.10, "all pixels of the six\nimages → pixel CNN", f5.GRAY)]
    for x, text, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y_box), half, h_box, boxstyle="round,pad=0,rounding_size=0.04",
                                    fill=False, ec=col, lw=0.8))
        ax.text(x + half / 2, y_box + h_box / 2, text, ha="center", va="center", fontsize=SMALL_PT,
                color=f5.INK, linespacing=1.1)
        arrow(ax, (x + half / 2, y_tile - 0.03), (x + half / 2, y_box + h_box + 0.02))
    ax.plot([MARGIN + half / 2, MARGIN + 1.5 * half + 0.10], [y_tile - 0.03] * 2, color=f5.INK2, lw=0.7)
    # checks: text inside the figure and inside its box
    fig.canvas.draw(); r = fig.canvas.get_renderer(); bad = []
    for t in ax.texts:
        bb = t.get_window_extent(r)
        if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > W * fig.dpi or bb.y1 > H * fig.dpi:
            bad.append(f"text leaves the figure: {t.get_text()!r}")
    for (x, text, _), t in zip(boxes, [t for t in ax.texts if "→" in t.get_text()]):
        bb = t.get_window_extent(r).transformed(ax.transData.inverted())
        if bb.x0 < x + 0.03 or bb.x1 > x + half - 0.03 or bb.height > h_box - 0.04:
            bad.append(f"text does not fit its box: {text!r} ({bb.width:.2f} x {bb.height:.2f} in)")
    fig.savefig(OUT.with_suffix(".pdf"), metadata={"CreationDate": None}, dpi=600)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    print(f"wrote {OUT}.pdf ({W} x {H} in)")
    if bad:
        print("PROBLEMS:\n  " + "\n  ".join(bad)); return 1
    print("All checks passed."); return 0


if __name__ == "__main__":
    raise SystemExit(main())
