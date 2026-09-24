"""SCM-4: the observed coordinates of SCM-1 delivered as original Shapes3D images, and the frozen reader that
turns an image back into a number (spec memo/2026-09-20-scm4-shapes3d-reader-spec-v1.md).

The chain is
    SCM-1 observed coordinate  ->  level 0..14  ->  one ORIGINAL image with that orientation  ->  reader  ->  number
and the sealed SCM-1 pipeline receives those numbers with the block structure (1, 1, 1, 1, 2) unchanged.

Every reading that reaches a learner passes through `reading`, which applies the one fixed post-processing
(clip to the recording scale) to the reader's raw output.  The verification tables score exactly those numbers,
so what is checked and what is used are the same thing.  A non-finite reading is never clipped away: it is
counted and fails the gate.

The reader is trained on `reader_train`, every choice is made on `reader_selection`, and `causal_bank` is read
once after the freeze.  The reader sees an orientation label and nothing else: no X, A, Y, latent state or block
role ever reaches it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import family_v2_dgp as g

ROOT = Path(__file__).resolve().parent.parent
BANK_DIR = ROOT / "materials" / "benchmarks" / "shapes3d"
H5_NAME, CACHE_NAME = "3dshapes.h5", "3dshapes_images_uint8.npy"
SOURCE_URL = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
H5_BYTES, H5_MD5 = 267_573_662, "099a2078d58cec4daad0702c55d06868"
PYTHON_PREFIX = "/Library/Frameworks/Python.framework/Versions/3.14"
FACTORS = ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation")
SIZES = (10, 10, 10, 8, 4, 15)
N_IMAGES = 480_000
LEVELS = SIZES[-1]                                   # the recording scale is the orientation factor, 15 levels
ORIENTATION = 5                                      # its position in FACTORS
SPLIT_SEED = 97_350_001
SPLIT = {"reader_train": 200_000, "reader_selection": 40_000, "causal_bank": 240_000}
READER_SEED = 97_350_002
READER = {"epochs": 6, "batch": 256, "lr": 1e-3, "max_seconds": 3600}
MIN_FREE_GIB = {"before_download": 8.0, "before_cache": 12.0}
GATE = {"far_overall": 2e-4, "far_cell": 1e-3, "adjacent_overall": 0.10, "bias_level": 0.25}
LEVEL_NAME = "SCM-4"


# ----------------------------------------------------------------------------- environment and storage

def check_interpreter() -> None:
    if not os.path.realpath(sys.executable).startswith(PYTHON_PREFIX):
        raise SystemExit(f"STOP: run with the project interpreter under {PYTHON_PREFIX}; this is {sys.executable}")


def free_gib(path: Path) -> float:
    target = path if path.exists() else path.parent
    return shutil.disk_usage(target).free / 2 ** 30


def require_free(path: Path, stage: str) -> float:
    """Swap shares this volume, so the room is measured again immediately before each large write."""
    free = free_gib(path)
    if free < MIN_FREE_GIB[stage]:
        raise SystemExit(f"STOP {stage}: {free:.1f} GiB free, {MIN_FREE_GIB[stage]} GiB required")
    return free


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=1, allow_nan=False))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- the bank

def factor_index(rows: np.ndarray) -> np.ndarray:
    """(n, 6) factor indices of bank rows under the C-order layout, orientation fastest."""
    return np.column_stack(np.unravel_index(np.asarray(rows), SIZES))


def split_rows() -> dict[str, np.ndarray]:
    """Three disjoint row sets with different jobs; see the module docstring."""
    perm = np.random.default_rng(SPLIT_SEED).permutation(N_IMAGES)
    out, start = {}, 0
    for name, size in SPLIT.items():
        out[name] = np.sort(perm[start:start + size])
        start += size
    return out


def verify_source(bank_dir: Path = BANK_DIR) -> dict:
    path = bank_dir / H5_NAME
    size = path.stat().st_size
    digest = file_md5(path)
    if size != H5_BYTES or digest != H5_MD5:
        raise SystemExit(f"STOP: {path} is {size} bytes with MD5 {digest}; approved {H5_BYTES} / {H5_MD5}")
    return {"bytes": size, "md5": digest}


def build_cache(bank_dir: Path = BANK_DIR) -> dict:
    """One uint8 cache of the images, plus the checks that the C-order layout this module assumes is real."""
    import h5py

    cache = bank_dir / CACHE_NAME
    if cache.exists():
        raise SystemExit(f"{cache} exists; it is write-once")
    with h5py.File(bank_dir / H5_NAME, "r") as f:
        images, labels = f["images"], f["labels"][:]
        if images.shape != (N_IMAGES, 64, 64, 3) or images.dtype != np.uint8 or labels.shape != (N_IMAGES, 6):
            raise SystemExit(f"STOP: unexpected arrays {images.shape} {images.dtype} {labels.shape}")
        values = [np.unique(labels[:, k]) for k in range(6)]
        if tuple(len(v) for v in values) != SIZES:
            raise SystemExit(f"STOP: factor sizes {[len(v) for v in values]} are not {SIZES}")
        implied = factor_index(np.arange(N_IMAGES))
        for k in range(6):
            if not np.array_equal(np.searchsorted(values[k], labels[:, k]), implied[:, k]):
                raise SystemExit(f"STOP: factor {FACTORS[k]} does not follow the C-order layout")
        require_free(bank_dir, "before_cache")
        tmp = cache.with_suffix(".partial.npy")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=images.shape)
        for s in range(0, N_IMAGES, 10_000):
            out[s:s + 10_000] = images[s:s + 10_000]
        out.flush()
        spot = np.sort(np.random.default_rng(SPLIT_SEED + 1).choice(N_IMAGES, 2_000, replace=False))
        if not np.array_equal(np.asarray(out[spot]), images[spot]):
            os.unlink(tmp)
            raise SystemExit("STOP: the cache differs from the HDF5 on the spot check")
        del out
    os.replace(tmp, cache)
    return {"factor_values": {FACTORS[k]: values[k].tolist() for k in range(6)}}


def prepare(bank_dir: Path = BANK_DIR) -> dict:
    check_interpreter()
    source = verify_source(bank_dir)
    t0 = time.time()
    facts = build_cache(bank_dir)
    rows = split_rows()
    np.savez(bank_dir / "split_rows.npz", **rows)
    manifest = {"source_url": SOURCE_URL, "h5": source, "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "cache": CACHE_NAME, "cache_bytes": (bank_dir / CACHE_NAME).stat().st_size,
                "images": [N_IMAGES, 64, 64, 3], "dtype": "uint8", "factors": dict(zip(FACTORS, SIZES)), **facts,
                "split_seed": SPLIT_SEED,
                "split": {k: {"rows": len(v), "sha256": hashlib.sha256(v.tobytes()).hexdigest()} for k, v in rows.items()},
                "split_file_sha256": file_sha256(bank_dir / "split_rows.npz"),
                "python": sys.version.split()[0], "seconds": round(time.time() - t0),
                "free_gib_after": round(free_gib(bank_dir), 1)}
    write_json_atomic(bank_dir / "manifest.json", manifest)
    return manifest


def load_images(bank_dir: Path = BANK_DIR):
    return np.load(bank_dir / CACHE_NAME, mmap_mode="r")


# ----------------------------------------------------------------------------- the recording rule

def population_sd(level: g.Level | None = None) -> dict[str, np.ndarray]:
    """Exact sd of every OBSERVED (rotated) coordinate of W, from the fixed rotations and the latent variances.

    What becomes an image is the coordinate the learner already received, never a latent or a pre-rotation value.
    """
    level = level or g.LEVELS["SCM-1"]
    q = g.rotations(level)
    m_var = 1.0 + g.PROXY_SD ** 2
    bad_var = 1.0 + g.bad_i(level) ** 2 + g.PROXY_SD ** 2
    info = {"W1": [m_var], "W2": [m_var], "W3": [m_var], "W4": [bad_var], "W5": [m_var, 1.0]}
    return {k: np.sqrt(np.diag(q[k] @ np.diag(v) @ q[k].T)) for k, v in info.items()}


def record(w: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Spec section 2: 15 equally spaced levels on [-3 sd, 3 sd]; anything outside is clipped to an end level."""
    return np.clip(np.rint((w / sd[None, :] + 3.0) / 6.0 * (LEVELS - 1)), 0, LEVELS - 1).astype(np.int64)


def bank_by_level(bank_rows: np.ndarray, excluded: dict[str, list[int]] | None = None) -> list[np.ndarray]:
    """The causal bank's rows grouped by orientation level, after any frozen exclusion of a nuisance value."""
    rows = np.asarray(bank_rows)
    fac = factor_index(rows)
    for name, values in (excluded or {}).items():
        keep = ~np.isin(fac[:, FACTORS.index(name)], values)
        rows, fac = rows[keep], fac[keep]
    groups = [rows[fac[:, ORIENTATION] == k] for k in range(LEVELS)]
    if any(len(v) == 0 for v in groups):
        raise SystemExit("STOP: a recording level has no image left in the causal bank")
    return groups


def draw_images(levels: np.ndarray, groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """One bank row per recorded level, uniform with replacement.  Reads the level and its own stream only."""
    sizes = np.array([len(v) for v in groups])
    pick = (rng.random(levels.shape) * sizes[levels]).astype(np.int64)
    flat = np.array([groups[k][p] for k, p in zip(levels.ravel(), pick.ravel())], dtype=np.int64)
    return flat.reshape(levels.shape)


# ----------------------------------------------------------------------------- the reader

def build():
    from torch import nn
    return nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                         nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU(),
                         nn.Flatten(), nn.Linear(128 * 16, 256), nn.ReLU(), nn.Linear(256, 1))


def device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def to_input(pixels_u8):
    """uint8 (n, 64, 64, 3) -> float32 (n, 3, 64, 64) in [0, 1]; the only transformation an image ever gets."""
    return pixels_u8.permute(0, 3, 1, 2).to(dtype=__import__("torch").float32) / 255.0


def raw_read(net, images, rows: np.ndarray, dev, chunk: int = 2000) -> np.ndarray:
    """The network's own output on the 0..14 scale, in the order of `rows` (which may repeat)."""
    import torch

    rows = np.asarray(rows).ravel()
    order = np.argsort(rows, kind="stable")
    out = np.empty(len(rows), dtype=np.float64)
    net.eval()
    half = (LEVELS - 1) / 2
    with torch.no_grad():
        for s in range(0, len(rows), chunk):
            idx = order[s:s + chunk]
            uniq, inverse = np.unique(rows[idx], return_inverse=True)
            x = to_input(torch.as_tensor(np.asarray(images[uniq]), device=dev))
            y = net(x).squeeze(1).cpu().numpy().astype(np.float64)
            out[idx] = (y * half + half)[inverse]
    return out


def reading(net, images, rows: np.ndarray, dev, chunk: int = 2000) -> np.ndarray:
    """THE number a learner receives: the raw output clipped to the recording scale.  Non-finite values survive
    the clip on purpose, so that the verification table counts them and the gate fails."""
    raw = raw_read(net, images, rows, dev, chunk)
    return np.clip(raw, 0.0, LEVELS - 1.0)


def fit(images, levels, train_rows, val_rows, dev, seed: int = READER_SEED, epochs: int = READER["epochs"],
        batch: int = READER["batch"], lr: float = READER["lr"], max_seconds: float = READER["max_seconds"],
        log=print):
    """Squared error on the orientation label.  The epoch is chosen on `val_rows`.  The time cap is checked at
    every batch boundary, so an overrun stops the run instead of being reported after the fact."""
    import torch

    torch.manual_seed(seed)
    net = build().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    half = (LEVELS - 1) / 2
    target = (np.asarray(levels, dtype=np.float32) - half) / half
    best = {"mse": np.inf, "state": None, "epoch": None}
    start = time.time()
    for epoch in range(epochs):
        t0 = time.time()
        order = rng.permutation(np.asarray(train_rows))
        net.train()
        for s in range(0, len(order), batch):
            if time.time() - start > max_seconds:
                raise TimeoutError(f"reader training passed {max_seconds:.0f} s inside epoch {epoch + 1}")
            rows = np.sort(order[s:s + batch])
            x = to_input(torch.as_tensor(np.asarray(images[rows]), device=dev))
            y = torch.as_tensor(target[rows], device=dev)
            opt.zero_grad(set_to_none=True)
            ((net(x).squeeze(1) - y) ** 2).mean().backward()
            opt.step()
        pred = reading(net, images, val_rows, dev)
        mse = float(np.mean((pred - np.asarray(levels)[np.asarray(val_rows)]) ** 2))
        log(f"epoch {epoch + 1}: validation rms {np.sqrt(mse):.4f} levels, {time.time() - t0:.0f} s")
        if mse < best["mse"]:
            best = {"mse": mse, "epoch": epoch + 1,
                    "state": {k: v.detach().clone() for k, v in net.state_dict().items()}}
    net.load_state_dict(best["state"])
    return net.eval(), {"epoch": best["epoch"], "val_mse": best["mse"], "seconds": round(time.time() - start)}


# ----------------------------------------------------------------------------- verification table and gate

def report(pred: np.ndarray, truth: np.ndarray, groups: dict[str, np.ndarray]) -> dict:
    """Counts, never bare rates: every cell carries its image count, its one-level errors and its larger errors.

    A non-finite reading is counted in `nonfinite` and excluded from the error statistics, so it can never be
    rounded into a plausible-looking level.
    """
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth)
    finite = np.isfinite(pred)
    hard = np.zeros(len(pred), dtype=np.int64)
    hard[finite] = np.clip(np.rint(pred[finite]), 0, LEVELS - 1).astype(np.int64)
    err = np.where(finite, hard - truth, 0)
    far, adj = finite & (np.abs(err) >= 2), finite & (np.abs(err) == 1)
    conf = np.zeros((LEVELS, LEVELS), dtype=np.int64)
    np.add.at(conf, (truth[finite], hard[finite]), 1)

    def cell(rows: np.ndarray) -> dict:
        n, ok = int(rows.sum()), rows & finite
        d = (pred - truth)[ok]
        return {"n": n, "nonfinite": int((rows & ~finite).sum()), "far": int(far[rows].sum()),
                "adjacent": int(adj[rows].sum()),
                "bias": float(d.mean()) if len(d) else None,
                "rms": float(np.sqrt(np.mean(d ** 2))) if len(d) else None}

    out = {"overall": cell(np.ones(len(truth), dtype=bool)), "confusion": conf.tolist(),
           "by_level": {int(k): cell(truth == k) for k in range(LEVELS)}, "by_factor": {}}
    for name, values in groups.items():
        out["by_factor"][name] = {int(v): cell(np.asarray(values) == v) for v in np.unique(values)}
    return out


def gate(rep: dict) -> dict:
    """PASS or FAIL of a verification table against spec section 5, listing every failing cell."""
    o, fails = rep["overall"], []
    if o["nonfinite"]:
        fails.append((f"overall nonfinite", o["nonfinite"], o["n"]))
    if o["far"] > GATE["far_overall"] * o["n"]:
        fails.append(("overall far", o["far"], o["n"]))
    if o["adjacent"] > GATE["adjacent_overall"] * o["n"]:
        fails.append(("overall adjacent", o["adjacent"], o["n"]))
    cells = [(f"level {k}", c) for k, c in rep["by_level"].items()]
    cells += [(f"{name}={v}", c) for name, d in rep["by_factor"].items() for v, c in d.items()]
    for label, c in cells:
        if c["n"] == 0:
            fails.append((label + " empty", 0, 0))
        elif c["far"] > GATE["far_cell"] * c["n"]:
            fails.append((label + " far", c["far"], c["n"]))
    for k, c in rep["by_level"].items():
        if c["bias"] is None or abs(c["bias"]) > GATE["bias_level"]:
            fails.append((f"level {k} bias {c['bias']}", None, c["n"]))
    return {"pass": not fails, "failing": fails, "thresholds": GATE}


def verification_table(net, images, rows: np.ndarray, dev, excluded: dict | None = None) -> dict:
    rows = np.asarray(rows)
    fac = factor_index(rows)
    for name, values in (excluded or {}).items():
        keep = ~np.isin(fac[:, FACTORS.index(name)], values)
        rows, fac = rows[keep], fac[keep]
    groups = {name: fac[:, k] for k, name in enumerate(FACTORS) if k != ORIENTATION}
    return report(reading(net, images, rows, dev), fac[:, ORIENTATION], groups)


MAX_EXCLUDED = 2
EXCLUSION_COVERAGE = 0.9


def propose_exclusion(rep: dict) -> dict | None:
    """Spec section 5, on counts rather than labels: an exclusion is allowed only when the failing readings sit
    inside at most two values of ONE nuisance factor and those values hold almost all of the large errors.

    A bad shape shows up in every orientation cell as well, so a rule that refused whenever a level cell failed
    would never fire.  What separates the two cases is where the errors are concentrated: a badly read shape puts
    them in one shape, a badly read angle spreads them over every shape.  The caller still has to recompute the
    table without the proposed values and see the gate pass; this function only proposes.
    """
    if rep["overall"]["nonfinite"]:
        return None
    total = rep["overall"]["far"]
    if total == 0:
        return None
    best = None
    for name, cells in rep["by_factor"].items():
        bad = sorted(v for v, c in cells.items() if c["n"] and c["far"] > GATE["far_cell"] * c["n"])
        if not bad or len(bad) > MAX_EXCLUDED:
            continue
        covered = sum(cells[v]["far"] for v in bad)
        if covered >= EXCLUSION_COVERAGE * total and (best is None or covered > best[1]):
            best = ({name: bad}, covered)
    return best[0] if best else None
