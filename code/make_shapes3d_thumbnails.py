"""The six Shapes3D images of Figure 4 (make_shapes3d_diagram.example_images: same values, levels and seed) as small
PNG files, for the Shapes3D panel of the TikZ scenario diagrams (figures/2026-10-18-scenario-causal-diagrams.tex).

Output: figures/shapes3d-thumbnails/w1.png ... w6.png (W1, W2, W3, W4, and the two images of W5), 64 x 64 pixels.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import make_shapes3d_diagram as s3d  # noqa: E402

OUT = CODE.parent / "figures" / "shapes3d-thumbnails"


def main() -> int:
    s3d.check_file()
    OUT.mkdir(parents=True, exist_ok=True)
    for k, (img, level) in enumerate(s3d.example_images(), start=1):
        plt.imsave(OUT / f"w{k}.png", img)
        print(f"  w{k}.png: orientation level {level}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
