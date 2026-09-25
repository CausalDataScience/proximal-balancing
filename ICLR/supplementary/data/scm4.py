"""SCM-4: SCM-1 with its six observed proxy coordinates delivered as Shapes3D images and read back by a frozen reader.

For every observed proxy coordinate of an SCM-1 data set the chain is

    coordinate  ->  level 0..14 on [-3 sd, 3 sd]  ->  one original Shapes3D image with that orientation
                ->  frozen reader  ->  the number the learner receives

Blocks W1 to W4 contribute one image each and W5 two, so every unit carries six images.  X, A and Y are SCM-1's.

    prepare(bank_dir)              verify 3dshapes.h5 and write the uint8 image cache and the fixed bank split
    load_reader(weights)           the reader network with frozen weights, in evaluation mode
    generate(n, seed, ...)         one SCM-4 data set: SCM-1 at `seed`, rendered as images and read back
    image_view(...)                the rendering and reading step itself
    train_reader(bank_dir, out)    train a reader by the procedure of the paper; the frozen weights used for the
                                   paper are data/reader_frozen.pt, so this is needed only to check how they were made
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import scm_family as g
from . import shapes3d as im

BASE_LEVEL = "SCM-1"                  # the data generating process and the learner's block layout
IMAGE_STREAM_OFFSET = 78              # the image draws of a data set use its seed plus this offset
READER_WEIGHTS = Path(__file__).resolve().parent / "reader_frozen.pt"
FROZEN_READER_EXCLUDED: dict = {}     # the frozen reader passed its checks with no bank value excluded


def prepare(bank_dir: Path = im.BANK_DIR) -> dict:
    """Verify the downloaded 3dshapes.h5, write the uint8 cache and the three disjoint bank row sets."""
    return im.prepare(Path(bank_dir))


def load_reader(weights: Path = READER_WEIGHTS, dev=None):
    """The reader with frozen weights, in evaluation mode, on `dev` (default: Apple MPS if available, else CPU)."""
    import torch

    dev = dev or im.device()
    net = im.build().to(dev)
    net.load_state_dict(torch.load(weights, map_location=dev))
    return net.eval(), dev


def excluded_for(weights: Path) -> dict:
    """Bank values excluded by the reader's checks: none for the frozen reader; for a reader trained with
    `train_reader`, whatever its record next to the weights says."""
    record = Path(weights).with_suffix(".json")
    if record.exists():
        import json
        return json.loads(record.read_text()).get("excluded", {})
    return FROZEN_READER_EXCLUDED


def image_view(data: dict, net, images, dev, groups: list, seed: int) -> tuple[dict, dict, np.ndarray, dict]:
    """What the learner of SCM-4 receives, the matching evaluator record, the bank rows of the six images, and the
    reading errors.  The image draws use one random stream per image column, spawned from seed + IMAGE_STREAM_OFFSET."""
    data4 = dict(data)
    streams, used, src_cols, stats = np.random.SeedSequence(seed + IMAGE_STREAM_OFFSET).spawn(6), 0, [], {}
    for key, sd in im.population_sd(g.LEVELS[BASE_LEVEL]).items():
        levels = im.record(data[key], sd)
        src = np.column_stack([im.draw_images(levels[:, c], groups, np.random.default_rng(streams[used + c]))
                               for c in range(levels.shape[1])])
        used += levels.shape[1]
        read = im.reading(net, images, src.ravel(), dev).reshape(src.shape)
        finite = np.isfinite(read)
        hard = np.where(finite, np.clip(np.rint(np.nan_to_num(read)), 0, im.LEVELS - 1), -99).astype(np.int64)
        err = hard - levels
        stats[key] = {"n": int(read.size), "nonfinite": int((~finite).sum()),
                      "adjacent": int((finite & (np.abs(err) == 1)).sum()),
                      "far": int((finite & (np.abs(err) >= 2)).sum())}
        data4[key] = read
        src_cols.append(src)
    return g.learner_view(data4), data4, np.column_stack(src_cols), stats


def generate(n: int, seed: int, bank_dir: Path = im.BANK_DIR, weights: Path = READER_WEIGHTS) -> dict:
    """One SCM-4 data set.  `view` is what a learner receives; `data` adds the evaluator-only fields; `reading`
    counts the reader's level errors per block.  Stops if any reading is not finite."""
    net, dev = load_reader(weights)
    images = im.load_images(Path(bank_dir))
    groups = im.bank_by_level(np.load(Path(bank_dir) / "split_rows.npz")["causal_bank"], excluded_for(weights))
    data = g.generate(g.LEVELS[BASE_LEVEL], n, seed)
    view, data4, src, reading = image_view(data, net, images, dev, groups, seed)
    if any(s["nonfinite"] for s in reading.values()):
        raise RuntimeError("a reading is not finite; the data set is not used")
    return {"view": view, "data": data4, "image_rows": src, "reading": reading, "device": str(dev)}


def train_reader(bank_dir: Path = im.BANK_DIR, out: Path = Path("reader.pt")) -> dict:
    """Train, choose, check and freeze a reader: the procedure that produced data/reader_frozen.pt.

    Squared error on the orientation label of the reader_train rows; the epoch with the lowest error on the
    reader_selection rows is kept; the verification gate is applied on reader_selection (with at most one allowed
    exclusion of nuisance values); the weights are saved to `out`; then the causal bank is read once for the final
    check.  Writes `out` and a record with the suffix .json next to it, and refuses to overwrite either."""
    import torch

    bank_dir, out = Path(bank_dir), Path(out)
    record_path = out.with_suffix(".json")
    for path in (out, record_path):
        if path.exists():
            raise SystemExit(f"{path} exists; a trained reader is written once")
    images = im.load_images(bank_dir)
    rows = np.load(bank_dir / "split_rows.npz")
    levels = im.factor_index(np.arange(im.N_IMAGES))[:, im.ORIENTATION]
    dev = im.device()
    net, fit_info = im.fit(images, levels, rows["reader_train"], rows["reader_selection"], dev)

    excluded: dict = {}
    selection = im.verification_table(net, images, rows["reader_selection"], dev, excluded)
    if not im.gate(selection)["pass"]:
        excluded = im.propose_exclusion(selection) or {}
        repaired = im.verification_table(net, images, rows["reader_selection"], dev, excluded) if excluded else None
        if not excluded or not im.gate(repaired)["pass"]:
            raise SystemExit("STOP: the selection gate fails and no allowed exclusion repairs it")
        selection = repaired

    torch.save(net.state_dict(), out)                  # frozen before the causal bank is touched
    final = im.verification_table(net, images, rows["causal_bank"], dev, excluded)   # the one read of the bank
    record = {"seed": im.READER_SEED, "fit": fit_info, "excluded": excluded,
              "selection_gate": im.gate(selection), "final_overall": final["overall"], "final_gate": im.gate(final)}
    im.write_json_atomic(record_path, record)
    return record
