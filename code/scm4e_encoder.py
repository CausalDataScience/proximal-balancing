"""SCM-4e v2: the label-free encoder that turns a Shapes3D image into 32 numbers
(PRD memo/2026-09-21-scm4e-v2-labelfree-association-prd-v1.md, sections 3.2 and 3.3).

    python3 -B scm4e_encoder.py freeze     train, freeze, and write the encoder and its check

It is an autoencoder: the encoder compresses an image to 32 numbers and the decoder rebuilds the image from
them, and the only loss is how well the image comes back.  No orientation label, no X, A or Y, and no image of
the causal bank ever reaches it.  The check that follows reads the orientation label, but only to report how much
of the angle the 32 numbers carry; nothing about the encoder is chosen by it.

The images are a random 100,000 of `reader_train`.  A random subset, not the first rows: the split is stored in
row order and the file is laid out with floor hue changing slowest, so the first 100,000 rows hold six of the ten
floor hues while the causal bank uses all ten.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import family_v2_images as im

LATENT = 32
ENCODER = {"images": 100_000, "epochs": 6, "batch": 256, "lr": 1e-3, "max_seconds": 3600}
INIT_SEED = 97_350_003
ORDER_SEED = 97_350_004
CHECK_SEED = 97_350_005
SUBSET_SEED = 97_350_006
CHECK = {"level_r2_min": 0.90}
WEIGHTS = im.BANK_DIR / "encoder_frozen.pt"
RECORD = im.BANK_DIR / "encoder_frozen.json"
CHECK_FILE = im.BANK_DIR / "encoder_check.json"


def build():
    """Encoder and decoder.  The encoder's convolutions are the reader's, and only its last layer differs."""
    from torch import nn
    enc = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                        nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU(),
                        nn.Flatten(), nn.Linear(128 * 16, 256), nn.ReLU(), nn.Linear(256, LATENT))
    dec = nn.Sequential(nn.Linear(LATENT, 256), nn.ReLU(), nn.Linear(256, 128 * 16), nn.ReLU(),
                        nn.Unflatten(1, (128, 4, 4)),
                        nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1),
                        nn.ReLU(), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                        nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())
    return enc, dec


def training_rows(bank_dir: Path = im.BANK_DIR) -> np.ndarray:
    pool = np.load(bank_dir / "split_rows.npz")["reader_train"]
    pick = np.random.default_rng(SUBSET_SEED).choice(len(pool), ENCODER["images"], replace=False)
    return np.sort(pool[pick])


def embed(enc, images, rows: np.ndarray, dev, chunk: int = 2000) -> np.ndarray:
    """The 32 numbers of each bank row, in the order of `rows` (which may repeat)."""
    import torch

    rows = np.asarray(rows).ravel()
    order = np.argsort(rows, kind="stable")
    out = np.empty((len(rows), LATENT), dtype=np.float64)
    enc.eval()
    with torch.no_grad():
        for s in range(0, len(rows), chunk):
            idx = order[s:s + chunk]
            uniq, inverse = np.unique(rows[idx], return_inverse=True)
            f = enc(im.to_input(torch.as_tensor(np.asarray(images[uniq]), device=dev)))
            out[idx] = f.cpu().numpy().astype(np.float64)[inverse]
    return out


def fit(images, rows: np.ndarray, dev, log=print):
    """Reconstruction only.  The time cap is checked at every batch boundary."""
    import torch

    torch.manual_seed(INIT_SEED)
    enc, dec = build()
    enc, dec = enc.to(dev), dec.to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=ENCODER["lr"])
    rng = np.random.default_rng(ORDER_SEED)
    start, history = time.time(), []
    for epoch in range(ENCODER["epochs"]):
        order, total, seen = rng.permutation(rows), 0.0, 0
        enc.train(); dec.train()
        for s in range(0, len(order), ENCODER["batch"]):
            if time.time() - start > ENCODER["max_seconds"]:
                raise TimeoutError(f"encoder training passed {ENCODER['max_seconds']:.0f} s in epoch {epoch + 1}")
            r = np.sort(order[s:s + ENCODER["batch"]])
            x = im.to_input(torch.as_tensor(np.asarray(images[r]), device=dev))
            loss = ((dec(enc(x)) - x) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(r)
            seen += len(r)
        history.append(round(total / seen, 6))
        log(f"epoch {epoch + 1}: reconstruction mse {history[-1]:.5f}, {time.time() - start:.0f} s", flush=True)
    return enc.eval(), {"reconstruction_mse_by_epoch": history, "seconds": round(time.time() - start)}


def check(enc, images, dev, bank_dir: Path = im.BANK_DIR) -> dict:
    """Evaluator-only: how much of the orientation level the 32 numbers carry, on images the encoder never saw.

    Half of `reader_selection` fits one linear read-out and the other half scores it, so the number is honest.
    Nothing here changes the encoder; it has already been frozen when this runs.
    """
    rows = np.load(bank_dir / "split_rows.npz")["reader_selection"]
    z = embed(enc, images, rows, dev)
    facts = im.factor_index(rows)
    level = facts[:, im.ORIENTATION].astype(np.float64)
    finite = int(np.isfinite(z).all(axis=1).sum())
    if finite != len(rows):
        return {"rows": len(rows), "finite_rows": finite, "pass": False, "reason": "an embedding is not finite"}
    perm = np.random.default_rng(CHECK_SEED).permutation(len(rows))
    half = len(rows) // 2
    fit_rows, score_rows = perm[:half], perm[half:]
    design = lambda r: np.column_stack([np.ones(len(r)), z[r]])
    coef = np.linalg.lstsq(design(fit_rows), level[fit_rows], rcond=None)[0]
    err = level[score_rows] - design(score_rows) @ coef
    r2 = float(1 - np.mean(err ** 2) / np.var(level[score_rows]))
    by = {name: {int(v): round(float(np.sqrt(np.mean(err[facts[score_rows, k] == v] ** 2))), 3)
                 for v in np.unique(facts[:, k])}
          for k, name in enumerate(im.FACTORS) if k != im.ORIENTATION}
    return {"rows": len(rows), "finite_rows": finite, "fit_rows": half, "scored_rows": len(score_rows),
            "level_r2": round(r2, 4), "rms_levels": round(float(np.sqrt(np.mean(err ** 2))), 3),
            "abs_error_quantiles": {q: round(float(np.quantile(np.abs(err), q)), 2)
                                    for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
            "share_two_levels_or_more": round(float(np.mean(np.abs(err) >= 2)), 4),
            "rms_by_factor": by, "requirement": CHECK, "pass": r2 >= CHECK["level_r2_min"]}


def freeze() -> int:
    import torch

    im.check_interpreter()
    for path in (WEIGHTS, RECORD, CHECK_FILE):
        if path.exists():
            raise SystemExit(f"{path} exists; the encoder and its check are write-once")
    images = im.load_images()
    dev = im.device()
    rows = training_rows()
    facts = im.factor_index(rows)
    print(f"training on {len(rows)} images, factor values covered: "
          f"{ {n: int(len(np.unique(facts[:, k]))) for k, n in enumerate(im.FACTORS)} }", flush=True)
    enc, fitted = fit(images, rows, dev)
    torch.save({k: v.detach().cpu() for k, v in enc.state_dict().items()}, WEIGHTS)
    record = {"artifact": "scm4e_v2_label_free_encoder", "latent": LATENT, "config": ENCODER,
              "seeds": {"init": INIT_SEED, "batch_order": ORDER_SEED, "subset": SUBSET_SEED, "check": CHECK_SEED},
              "training_rows": {"n": len(rows), "source": "reader_train", "sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
                                "factor_values_covered": {n: int(len(np.unique(facts[:, k]))) for k, n in enumerate(im.FACTORS)}},
              "sees": "pixels only: no orientation label, no X, A or Y, no causal-bank image",
              "trained_by": {"file": Path(__file__).name, "sha256": im.file_sha256(Path(__file__).resolve())},
              "fit": fitted, "device": str(dev), "weights_sha256": im.file_sha256(WEIGHTS),
              "split_file_sha256": im.file_sha256(im.BANK_DIR / "split_rows.npz"),
              "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    im.write_json_atomic(RECORD, record)
    print("frozen; running the evaluator-only check", flush=True)
    result = dict(check(enc, images, dev), encoder_sha256=record["weights_sha256"],
                  checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    im.write_json_atomic(CHECK_FILE, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rms_by_factor"}, indent=1))
    return 0 if result["pass"] else 1


def load(dev=None, bank_dir: Path = im.BANK_DIR):
    """The frozen encoder, refused unless its check exists, passed, and matches these weights."""
    import torch

    for path in (WEIGHTS, RECORD, CHECK_FILE):
        if not path.exists():
            raise SystemExit(f"no {path.name}; freeze the encoder first")
    record, result = json.loads(RECORD.read_text()), json.loads(CHECK_FILE.read_text())
    digest = im.file_sha256(WEIGHTS)
    if im.file_sha256(Path(__file__).resolve()) != record["trained_by"]["sha256"]:
        raise SystemExit("scm4e_encoder.py is not the file that trained these weights; it is sealed once it has trained")
    if not result.get("pass"):
        raise SystemExit("the encoder check did not pass; nothing may use this encoder")
    if digest != record["weights_sha256"] or digest != result["encoder_sha256"]:
        raise SystemExit("the encoder weights differ from the ones its record and check describe")
    dev = dev or im.device()
    enc, _ = build()
    enc.load_state_dict(torch.load(WEIGHTS, map_location=dev))
    return enc.to(dev).eval(), dev, {"weights_sha256": digest, "record_sha256": im.file_sha256(RECORD),
                                     "check_sha256": im.file_sha256(CHECK_FILE)}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "freeze":
        raise SystemExit("usage: scm4e_encoder.py freeze")
    raise SystemExit(freeze())
