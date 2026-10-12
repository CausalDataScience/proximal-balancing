"""Shared-noise check: is a held-out block predictable from the retained blocks beyond U?
Compares region blocks against interleaved (every-other-pixel) blocks on the current phantom."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from single_image import phantom as ph, decisive_v2 as dv, regions as rg

cfg = ph.PhantomConfig(n=4000, sigma=.3, seed=777, image_only=True)
d = dv.bounded_generate(cfg)
img = d.images.astype(np.float64)
n = len(img)
U = d.U.reshape(n, 1)
G = getattr(d, "G", None)
masks = [m.astype(bool) for m in rg.masks()]
body = np.zeros_like(masks[0]); [body.__ior__(m) for m in masks]

def r2(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return 1 - np.mean((y - X1 @ b) ** 2) / np.var(y)

covars = U if G is None else np.column_stack([U, G])
print("covariates for the U-only predictor:", "U" if G is None else "U,G")

# (a) region blocks: mean lit intensity per lobe
f = np.column_stack([img[:, m].mean(1) for m in masks])
for k in range(4):
    others = np.delete(f, k, axis=1)
    print(f"region {k+1}: R2 from other regions {r2(f[:,k], others):.3f} | from U(,G) {r2(f[:,k], covars):.3f} | from both {r2(f[:,k], np.column_stack([others, covars])):.3f}")

# (b) interleaved blocks with spacing s: pixel (i,j) -> block (i%s, j%s); hold out block (0,0)
for s in (2, 4, 8):
    ii, jj = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    held = (ii % s == 0) & (jj % s == 0) & body
    rest = body & ~held
    fT = img[:, held].mean(1)
    # retained-block features: mean per retained interleave class (s*s - 1 numbers)
    feats = [img[:, (ii % s == a) & (jj % s == b) & body].mean(1) for a in range(s) for b in range(s) if (a, b) != (0, 0)]
    fV = np.column_stack(feats)
    print(f"interleave s={s}: held-out pixels {held.sum()}, R2 of T-mean from retained-block means {r2(fT, fV):.3f} | from U(,G) {r2(fT, covars):.3f}")
    # per-pixel reconstruction of held-out pixels from 4 nearest retained neighbours (s=2 only: direct neighbours)
    if s == 2:
        nb = np.zeros_like(img)
        nb[:, :, :] = (np.roll(img, 1, 1) + np.roll(img, -1, 1) + np.roll(img, 1, 2) + np.roll(img, -1, 2)) / 4
        inner = held.copy(); inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
        err = ((img - nb)[:, inner]) ** 2
        print(f"  per-pixel: R2 of held-out pixels from 4 neighbours = {1 - err.mean() / img[:, inner].var():.3f}")
