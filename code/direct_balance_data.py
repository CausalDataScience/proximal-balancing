"""Direct image balancing: the data, the recording rule and the row split
(PRD memo/2026-09-21-direct-image-balancing-prd-v1.md, sections 2 and 5).

The causal structure is the sealed SCM-1, unchanged: the same coefficients, the same rotations, the same true
effect tau = 1.  What changes is how a learner sees the proxy.  Each of the six observed coordinates of W is
recorded to L levels and delivered as one original benchmark image at that level, and nothing else about the
image reaches the learner: not the orientation, not the digit, not the block's role.

L is 15 for Shapes3D (its orientation factor has 15 values) and 10 for MNIST (one digit per level).

The row split replaces the sealed four-fold rotation with one independent split, because every candidate split
of the blocks now trains its own CNN and four rotations would cost four times as much (PRD section 10).  The five
row sets are disjoint and each has exactly one job.
"""
from __future__ import annotations

import numpy as np

import family_v2_dgp as g
import family_v2_images as im

BASE_LEVEL = "SCM-1"
N = 30_000
ROLES = {"represent": 4_800, "select": 1_200, "check": 6_000, "nuisance": 6_000, "evaluate": 12_000}
SPLIT_SEED_OFFSET = 11                      # the row split's seed is the data seed plus this
BENCHMARKS = {"shapes3d": 15, "mnist": 10}  # levels the observed coordinate is recorded to
BLOCK_IMAGES = (1, 1, 1, 1, 2)              # W1..W4 are one image, W5 is two; the five block are never split apart
N_IMAGES = sum(BLOCK_IMAGES)
BLOCK_KEYS = [f"W{j + 1}" for j in range(g.N_BLOCKS)]


def population_sd() -> dict[str, np.ndarray]:
    """Exact sd of each OBSERVED coordinate of W, from the sealed rotations and latent variances."""
    return im.population_sd(g.LEVELS[BASE_LEVEL])


def record(w: np.ndarray, sd: np.ndarray, levels: int) -> np.ndarray:
    """L equally spaced levels on [-3 sd, 3 sd]; anything outside is clipped to an end level.

    The sealed SCM-4 rule with L a parameter.  At L = 15 it is that rule exactly.
    """
    return np.clip(np.rint((w / sd[None, :] + 3.0) / 6.0 * (levels - 1)), 0, levels - 1).astype(np.int64)


def recorded(data: dict[str, np.ndarray], levels: int) -> dict[str, np.ndarray]:
    """The recorded level of every observed coordinate, one array per block."""
    sds = population_sd()
    return {key: record(data[key], sds[key], levels) for key in BLOCK_KEYS}


def as_numbers(data: dict[str, np.ndarray], levels: int) -> dict[str, np.ndarray]:
    """The data set with every block replaced by its recorded levels as plain numbers.

    This is the numeric contrast of PRD section 8 (P0): what a learner would see if it read every image
    perfectly.  The evaluator-only keys are carried through untouched, for the oracle comparator.
    """
    out = dict(data)
    out.update({k: v.astype(np.float64) for k, v in recorded(data, levels).items()})
    return out


IMAGE_STREAM_OFFSET = 91                    # a stream no earlier experiment used, so these images are its own


MNIST_FILE = im.BANK_DIR.parent / "mnist" / "mnist.npz"


def load_bank(benchmark: str):
    """The images a benchmark draws from, and their rows grouped by recorded level.

    shapes3d  the causal bank of the 480,000 originals, grouped by the orientation factor (15 levels)
    mnist     the 60,000 training digits, grouped by digit (10 levels); level l is shown as a digit l

    The grouping label is the benchmark's own and is read by the generator alone, to pick an image.  No learner
    receives it.  Nothing is pretrained on either bank, so there is no separate initialisation set to keep apart.
    """
    if benchmark == "shapes3d":
        rows = np.load(im.BANK_DIR / "split_rows.npz")["causal_bank"]
        return im.load_images(), im.bank_by_level(rows)
    if benchmark == "mnist":
        with np.load(MNIST_FILE) as f:
            images, digits = np.ascontiguousarray(f["x_train"]), f["y_train"]
        return images, [np.flatnonzero(digits == k) for k in range(BENCHMARKS["mnist"])]
    raise ValueError(f"unknown benchmark {benchmark!r}")


def image_bank(bank_dir=None) -> tuple:
    """The Shapes3D causal bank grouped by level (kept for the P1 script)."""
    rows = np.load((bank_dir or im.BANK_DIR) / "split_rows.npz")["causal_bank"]
    return im.bank_by_level(rows)


def draw_images(data: dict, groups: list, levels: int, seed: int) -> np.ndarray:
    """One bank row per observed coordinate: (n, 6) image indices, in block order W1..W5.

    The level decides which images are eligible; everything else about the image is drawn uniformly from that
    level's group.  Each of the six coordinates gets its own random stream, so two coordinates at the same level
    do not get the same picture.
    """
    if len(groups) != levels:
        raise ValueError(f"the bank has {len(groups)} level groups, the recording has {levels} levels")
    recorded_levels = recorded(data, levels)
    streams = np.random.SeedSequence(seed + IMAGE_STREAM_OFFSET).spawn(N_IMAGES)
    out, used = [], 0
    for key in BLOCK_KEYS:
        lv = recorded_levels[key]
        for c in range(lv.shape[1]):
            out.append(im.draw_images(lv[:, c], groups, np.random.default_rng(streams[used])))
            used += 1
    return np.column_stack(out)


def split_rows(n: int, seed: int) -> dict[str, np.ndarray]:
    """One independent split into the five row sets, disjoint by construction and sorted within each set."""
    if sum(ROLES.values()) != n:
        raise ValueError(f"the row split is defined for n = {sum(ROLES.values())}, not {n}")
    perm = np.random.default_rng(seed + SPLIT_SEED_OFFSET).permutation(n)
    out, start = {}, 0
    for role, size in ROLES.items():
        out[role] = np.sort(perm[start:start + size])
        start += size
    return out


def all_splits() -> list[tuple[int, ...]]:
    """The thirty candidate splits: every subset of the five blocks except the empty set and the whole set.

    The sealed enumeration, so a split carries the same name and the same meaning as in SCM-1 to SCM-4.
    """
    import family_v2_probe as pr
    return pr.all_splits()


def split_name(split) -> str:
    return "S" + "".join(str(j + 1) for j in split)


def image_columns(split) -> tuple[np.ndarray, np.ndarray]:
    """Which of the six image columns are held out by `split` and which are kept.

    The five blocks own (1, 1, 1, 1, 2) images in order, so the two images of W5 always move together.
    """
    starts = np.cumsum([0] + list(BLOCK_IMAGES))
    held = [c for j in split for c in range(starts[j], starts[j + 1])]
    kept = [c for c in range(N_IMAGES) if c not in held]
    return np.array(held, dtype=np.int64), np.array(kept, dtype=np.int64)
