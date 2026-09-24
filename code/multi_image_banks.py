"""Original-image banks and the selection kernel of the multi-image experiments (PRD sections 3.3, 4 and 5).

An image W_j is drawn with replacement from a fixed bank given the sign M_j alone.  The kernel reads the sign
and its own generator stream and nothing else: not X, U, A, Y, and not the row drawn for another block.  The
bank's bytes are never changed; a learner receives float32 pixels / 255 and nothing about labels or factors.

    MNIST      M = -1: a digit uniform on 0-4;  M = +1: uniform on 5-9;  then an image uniform within the digit.
    Shapes3D   M = -1: object hue index uniform on 0-4;  M = +1: on 5-9;  floor hue, wall hue, scale, shape and
               orientation uniform on all their values.  The row is the C-order index of the six factors.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import multi_image_scm as scm

ROOT = Path(__file__).resolve().parent.parent
MNIST_PATH = ROOT / "materials" / "benchmarks" / "mnist" / "mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
MNIST_BYTES = 11_490_434

SHAPES3D_FACTORS = ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation")
SHAPES3D_SIZES = (10, 10, 10, 8, 4, 15)
SHAPES3D_ROWS = 480_000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MnistBank:
    """The 60,000 training images.  Labels stay inside the bank: the kernel reads them, a learner never does."""

    name, channels, side = "mnist", 1, 28

    def __init__(self, path: Path = MNIST_PATH) -> None:
        self.path, self.bytes, self.sha256 = Path(path), Path(path).stat().st_size, file_sha256(path)
        if self.sha256 != MNIST_SHA256:
            raise RuntimeError(f"MNIST bank digest {self.sha256} is not the frozen {MNIST_SHA256}")
        raw = np.load(path)
        self.pixels = raw["x_train"]
        self._labels = raw["y_train"].astype(np.int64)
        if self.pixels.shape != (60_000, 28, 28) or self.pixels.dtype != np.uint8:
            raise RuntimeError(f"unexpected x_train {self.pixels.shape} {self.pixels.dtype}")
        self.pixels.setflags(write=False)
        self._by_digit = [np.flatnonzero(self._labels == d) for d in range(10)]
        self._sizes = np.array([len(v) for v in self._by_digit])

    def draw(self, sign: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Source rows for one block: digit uniform within the sign's group, then an image uniform within it."""
        digit = rng.integers(0, 5, size=len(sign)) + 5 * (sign == 1)
        within = (rng.random(len(sign)) * self._sizes[digit]).astype(np.int64)
        return np.array([self._by_digit[d][w] for d, w in zip(digit, within)], dtype=np.int64)

    def group(self, rows: np.ndarray) -> np.ndarray:
        """Evaluator-only: the sign a source row stands for."""
        return np.where(self._labels[rows] >= 5, 1, -1)

    def images(self, rows: np.ndarray) -> np.ndarray:
        """float32 (n, 1, 28, 28) in [0, 1]; the only transformation a learner's input ever gets."""
        return self.pixels[rows].astype(np.float32)[:, None] / 255.0

    def manifest(self) -> dict:
        return {"bank": self.name, "path": str(self.path.relative_to(ROOT)), "bytes": self.bytes,
                "sha256": self.sha256, "array": "x_train", "shape": list(self.pixels.shape),
                "dtype": str(self.pixels.dtype), "label_array": "y_train (kernel only)",
                "group_rows": {"minus (digits 0-4)": int(self._sizes[:5].sum()),
                               "plus (digits 5-9)": int(self._sizes[5:].sum())},
                "rows_by_digit": self._sizes.tolist()}

    def collision_audit(self) -> dict:
        """PRD section 5 items 3 and 4: is any image byte-identical to an image of the opposite group?"""
        return _collisions(self.pixels.reshape(len(self.pixels), -1), self.group(np.arange(len(self.pixels))))


def _collisions(flat: np.ndarray, group: np.ndarray) -> dict:
    seen: dict[bytes, list[int]] = {}
    for row in range(len(flat)):
        seen.setdefault(hashlib.blake2b(flat[row].tobytes(), digest_size=16).digest(), []).append(row)
    crossing, within = [], 0
    for rows in seen.values():
        if len(rows) < 2:
            continue
        # equal hashes are confirmed on the bytes themselves before anything is called a duplicate
        same = [r for r in rows[1:] if np.array_equal(flat[r], flat[rows[0]])]
        if not same:
            continue
        if len({int(group[r]) for r in [rows[0]] + same}) > 1:
            crossing.append([rows[0]] + same)
        else:
            within += len(same)
    return {"images": len(flat), "distinct_hashes": len(seen), "duplicates_within_group": within,
            "duplicates_across_groups": crossing, "exact_decoding_certificate": not crossing}


def shapes3d_row(factors: np.ndarray) -> np.ndarray:
    """C-order row of factor indices (n, 6) in the order SHAPES3D_FACTORS, orientation fastest."""
    return np.ravel_multi_index(np.asarray(factors).T, SHAPES3D_SIZES)


def shapes3d_draw(sign: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Source rows for one block.  Needs no file: the bank is the full factor grid."""
    n = len(sign)
    factors = np.column_stack([rng.integers(0, size, size=n) for size in SHAPES3D_SIZES])
    factors[:, 2] = rng.integers(0, 5, size=n) + 5 * (sign == 1)
    return shapes3d_row(factors)


def shapes3d_group(rows: np.ndarray) -> np.ndarray:
    hue = np.unravel_index(np.asarray(rows), SHAPES3D_SIZES)[2]
    return np.where(hue >= 5, 1, -1)


def draw_blocks(draw, signs: np.ndarray, seed: int) -> np.ndarray:
    """Source rows (n, J), one generator stream per block so that no block's draw can lean on another's."""
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(scm.J)]
    return np.column_stack([draw(signs[:, j], streams[j]) for j in range(scm.J)])
