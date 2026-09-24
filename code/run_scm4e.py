"""Stage runner for SCM-4e: the image level in which PROBE receives each image's embedding, not a decoded number
(spec memo/2026-09-20-scm4-shapes3d-reader-spec-v1.md, section 14).

    python3 -B run_scm4e.py freeze-dev                 write results/family_v2/scm4e/dev_protocol_v1.json
    python3 -B run_scm4e.py dev [--workers 3]          PROBE and the number-based comparators on the dev data sets
    python3 -B run_scm4e.py pixel-dev                  the pixel CNN on the same data sets, one at a time
    python3 -B run_scm4e.py freeze-confirm T           freeze the threshold chosen on development data
    python3 -B run_scm4e.py confirm [--workers 3]      the confirmation data sets
    python3 -B run_scm4e.py pixel-confirm              the pixel CNN on the confirmation data sets, one at a time

The data generating process, the recording rule, the image bank, the frozen network and the image draws are those
of SCM-4.  What changes is what the learner is handed.  In SCM-4 every image became one number, the network's own
output.  Here every image becomes the 256 numbers of the layer before that output, so a block is a 256- or
512-dimensional vector, exactly the shape of an SCM-3 block, and the sealed PROBE code treats it as one: its own
front end (principal components above the Marchenko-Pastur edge, at most four) reduces the block and the balance
objective learns the combination.  Nobody tells the learner which direction of the embedding is the angle.

The pixel CNN uses the GPU for minutes at a time, so it runs as its own sequential pass and writes its own shards.
Sharing the GPU between tasks is what made it hit its time cap in the SCM-4 confirmation.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import family_v2_dgp as g
import family_v2_images as im

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "family_v2" / "scm4e"
SPEC = HERE.parent / "memo" / "2026-09-20-scm4-shapes3d-reader-spec-v1.md"
SOURCES = ["family_v2_dgp.py", "family_v2_probe.py", "family_v2_competitors.py", "family_v2_images.py",
           "probe_structured_scm.py", "positive_control_scm.py", "scm4_pixel_cnn.py", "run_scm4_images.py",
           "run_scm4.py", "summarize_family_v2.py", "summarize_scm4.py", "summarize_scm4e.py", "run_scm4e.py"]
LEVEL_NAME = "SCM-4e"
BASE_LEVEL = "SCM-1"
EMBED_DIM = 256
LEVEL = g.Level(LEVEL_NAME, 5, 2, (EMBED_DIM,) * 4 + (2 * EMBED_DIM,), 2)     # the learner's block layout; k as in SCM-1 to 3
N = 24_000
DEV_REPS, CONFIRM_REPS = 3, 20
DEV_SEED, CONFIRM_SEED = 97_500_000, 97_600_000       # a range no earlier run has used
FOLD_SEED_OFFSET, IMAGE_STREAM_OFFSET = 7, 78
PROTOCOLS = {"dev": RESULTS / "dev_protocol_v1.json", "confirm": RESULTS / "confirm_protocol_v1.json"}
SHARDS = {"dev": RESULTS / "dev", "confirm": RESULTS / "confirm"}
PIXEL_SHARDS = {"dev": RESULTS / "dev_pixel", "confirm": RESULTS / "confirm_pixel"}


# ----------------------------------------------------------------------------- what the learner receives

def embed(net, images, rows: np.ndarray, dev, chunk: int = 2000) -> np.ndarray:
    """The 256 numbers of the frozen network's last hidden layer for each bank row, in the order of `rows`."""
    import torch
    from torch import nn

    tower = nn.Sequential(*list(net.children())[:-1]).eval()
    rows = np.asarray(rows).ravel()
    order = np.argsort(rows, kind="stable")
    out = np.empty((len(rows), EMBED_DIM), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(rows), chunk):
            idx = order[s:s + chunk]
            uniq, inverse = np.unique(rows[idx], return_inverse=True)
            f = tower(im.to_input(torch.as_tensor(np.asarray(images[uniq]), device=dev)))
            out[idx] = f.cpu().numpy().astype(np.float64)[inverse]
    return out


def draw_sources(data: dict, groups: list, seed: int) -> tuple[dict, dict]:
    """Recorded levels and bank rows per block, with the image streams SCM-4 uses (so the same seed would give the
    same six images in both experiments)."""
    streams, used = np.random.SeedSequence(seed + IMAGE_STREAM_OFFSET).spawn(6), 0
    levels, src = {}, {}
    for key, sd in im.population_sd(g.LEVELS[BASE_LEVEL]).items():
        lv = im.record(data[key], sd)
        src[key] = np.column_stack([im.draw_images(lv[:, c], groups, np.random.default_rng(streams[used + c]))
                                    for c in range(lv.shape[1])])
        levels[key] = lv
        used += lv.shape[1]
    return levels, src


def embedding_view(data: dict, net, images, dev, groups: list, seed: int) -> tuple[dict, dict, np.ndarray, dict]:
    """The learner's view (each block an embedding), the evaluator record with the same W, the bank rows of the six
    images, and an evaluator-only check of how much of each recorded level survives the sealed front end."""
    import family_v2_probe as pr

    levels, src = draw_sources(data, groups, seed)
    data_e = dict(data)
    for key, rows in src.items():
        data_e[key] = np.column_stack([embed(net, images, rows[:, c], dev) for c in range(rows.shape[1])])
    if not all(np.isfinite(data_e[k]).all() for k in src):
        raise RuntimeError("an embedding is not finite; the task stops before any estimator runs")
    view = g.learner_view(data_e)
    fold_rows = pr.folds(len(view["A"]), seed + FOLD_SEED_OFFSET)
    fit, rest = fold_rows[0], np.concatenate(fold_rows[1:])
    prep, check = pr.Prep(view, fit), {}
    for j, key in enumerate(src):
        f_fit, f_rest = prep.block(view, fit, j), prep.block(view, rest, j)
        r2 = []
        for c in range(levels[key].shape[1]):
            t = levels[key][:, c].astype(float)
            coef = np.linalg.lstsq(np.column_stack([np.ones(len(fit)), f_fit]), t[fit], rcond=None)[0]
            pred = np.column_stack([np.ones(len(rest)), f_rest]) @ coef
            r2.append(float(1 - np.mean((t[rest] - pred) ** 2) / np.var(t[rest])))
        check[key] = {"components_kept": int(f_fit.shape[1]), "level_r2_from_kept_components": r2}
    return view, data_e, np.column_stack([src[k] for k in src]), check


# ----------------------------------------------------------------------------- protocol

def design_snapshot() -> dict:
    import run_family_v2 as rf
    import run_scm4_images as ri
    import scm4_pixel_cnn as px
    base = rf.design_snapshot()
    sds = {k: v.tolist() for k, v in im.population_sd(g.LEVELS[BASE_LEVEL]).items()}
    return {"family_v2": {"coefficients": base["coefficients"], "base_level": base["levels"][BASE_LEVEL],
                          "probe": base["probe"], "competitors": base["competitors"]},
            "learner_blocks": {"dims": list(LEVEL.d_blocks), "k": LEVEL.k, "embedding": "the frozen network's last "
                               "hidden layer (256 numbers per image); W5 is its two images side by side"},
            "recording": {"levels": im.LEVELS, "population_sd": sds, "range_sd": 3.0,
                          "image_stream_offset": IMAGE_STREAM_OFFSET},
            "bank": {"split": im.SPLIT, "split_seed": im.SPLIT_SEED, "source_md5": im.H5_MD5},
            "network": {"seed": im.READER_SEED, "training": im.READER, "gate": im.GATE, "digests": ri.current_digests()},
            "pixel_cnn": px.CONFIG}


def tasks_for(stage: str) -> list[dict]:
    if stage == "dev":
        return [{"level": LEVEL_NAME, "rep": rep, "seed": DEV_SEED + rep} for rep in range(DEV_REPS)]
    return [{"level": LEVEL_NAME, "rep": rep, "seed": CONFIRM_SEED + 1_000 * rep} for rep in range(CONFIRM_REPS)]


def freeze(stage: str, threshold: float | None = None) -> int:
    import run_family_v2 as rf
    path = PROTOCOLS[stage]
    if stage == "confirm" and not (threshold and 0 < threshold <= 2.0e-3):
        raise SystemExit("give the development threshold, a positive number no larger than the 2.0e-3 cap")
    if path.exists():
        raise SystemExit(f"{path} exists; protocols are write-once")
    if stage == "confirm" and not PROTOCOLS["dev"].exists():
        raise SystemExit("the development protocol must exist first")
    payload = {"protocol_id": f"family-v2-scm4e-{stage}-v1", "stage": stage, "frozen_at_utc": rf.now(),
               "spec": {"path": str(SPEC.relative_to(HERE.parent)), "sha256": rf.sha256(SPEC)},
               "n": N, "design": design_snapshot(), "tasks": tasks_for(stage),
               "source_sha256": {f: rf.sha256(HERE / f) for f in SOURCES},
               "screen_thresholds": {LEVEL_NAME: threshold} if threshold else None,
               "gates": "the thirteen gates of SCM-4 (family v2 PRD section 10 plus the pixel CNN)",
               "notes": {"dev": "the threshold is chosen after development by the sealed family-v2 rule",
                         "confirm": "threshold, sources, network and design are frozen; nothing is chosen on these data"}[stage]}
    if stage == "confirm":
        payload["dev_protocol_sha256"] = rf.sha256(PROTOCOLS["dev"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    tmp.rename(path)
    print(f"frozen {path} ({len(payload['tasks'])} tasks), sha256 {rf.sha256(path)}")
    return 0


def check_protocol(stage: str) -> dict:
    import run_family_v2 as rf
    path = PROTOCOLS[stage]
    if not path.exists():
        raise SystemExit(f"no protocol at {path}")
    proto = json.loads(path.read_text())
    bad = [f for f, h in proto["source_sha256"].items() if rf.sha256(HERE / f) != h]
    if bad:
        raise SystemExit(f"source files differ from the frozen protocol: {bad}")
    if json.dumps(design_snapshot(), sort_keys=True) != json.dumps(proto["design"], sort_keys=True):
        raise SystemExit("the design, the bank or the network differs from the frozen protocol")
    return proto


def shard_path(stage: str, task: dict, pixel: bool = False) -> Path:
    return (PIXEL_SHARDS if pixel else SHARDS)[stage] / f"{task['level']}_seed{task['seed']}.json"


def _write_once(path: Path, payload: dict) -> None:
    import run_family_v2 as rf
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rf._jsonable(payload), allow_nan=False) + "\n")
    tmp.rename(path)


def _load_bank():
    import run_scm4_images as ri
    net, dev, excluded, _ = ri.load_frozen_reader()
    images = im.load_images()
    groups = im.bank_by_level(np.load(im.BANK_DIR / "split_rows.npz")["causal_bank"], excluded)
    return net, dev, images, groups


# ----------------------------------------------------------------------------- the two passes

def run_task(stage: str, task: dict, proto_sha: str, tolerance: float | None) -> dict:
    import torch
    torch.set_num_threads(1)
    import family_v2_competitors as fc
    import family_v2_probe as pr
    import run_family_v2 as rf
    import run_scm4_images as ri

    out_path = shard_path(stage, task)
    if out_path.exists():
        return {"task": task, "status": "exists"}
    t_all = time.time()
    net, dev, images, groups = _load_bank()
    seed = task["seed"]
    fold_seed = seed + FOLD_SEED_OFFSET
    shard = {"protocol_sha256": proto_sha, "stage": stage, "task": task, "n": N, "generated_utc": rf.now(),
             "timing_s": {}, "device": str(dev)}
    t0 = time.time()
    data = g.generate(g.LEVELS[BASE_LEVEL], N, seed)
    view, data_e, _, shard["embedding_check"] = embedding_view(data, net, images, dev, groups, seed)
    shard["timing_s"]["images_and_embedding"] = round(time.time() - t0, 1)
    del net, images

    t0 = time.time()
    res = pr.run_dataset(view, LEVEL, 1.0 if tolerance is None else tolerance, fold_seed, keep_b=True)
    problems = ri.validate(res)
    if tolerance is None:
        problems = [p for p in problems if not p.startswith(("return kind", "no finite estimate"))]
    shard["probe"], shard["probe_problems"] = res, problems
    shard["timing_s"]["probe"] = round(time.time() - t0, 1)
    t0 = time.time()
    shard["baselines"] = fc.aipw_baselines(data_e, fold_seed)
    shard["timing_s"]["baselines"] = round(time.time() - t0, 1)
    t0 = time.time()
    shard["proximal"] = fc.proximal_rows(view)
    shard["timing_s"]["proximal"] = round(time.time() - t0, 2)
    t0 = time.time()
    cev = fc.cevae_ate(view, seed + 11)
    shard["cevae"] = {"ate": cev["ate"], "final_loss": cev["final_loss"]}
    shard["timing_s"]["cevae"] = round(time.time() - t0, 1)
    shard["timing_s"]["total"] = round(time.time() - t_all, 1)
    _write_once(out_path, shard)
    return {"task": task, "status": "written", "seconds": shard["timing_s"]["total"], "estimate": res.get("estimate"),
            "problems": problems, "cevae": shard["cevae"]["ate"], "raw": shard["baselines"]["raw"],
            "kept": {k: v["components_kept"] for k, v in shard["embedding_check"].items()}}


def run(stage: str, workers: int) -> int:
    import multiprocessing
    import run_family_v2 as rf
    proto = check_protocol(stage)
    proto_sha = rf.sha256(PROTOCOLS[stage])
    thresholds = proto.get("screen_thresholds") or {}
    if stage == "confirm" and not thresholds.get(LEVEL_NAME):
        raise SystemExit("the confirmation protocol has no screen threshold")
    todo = [t for t in proto["tasks"] if not shard_path(stage, t).exists()]
    print(f"{stage}: {len(todo)} tasks to run with {workers} workers (protocol {proto_sha[:12]})", flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_task, stage, t, proto_sha, thresholds.get(LEVEL_NAME)): t for t in todo}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                r = fut.result()
                print(f"  seed {t['seed']}: {r['status']} {r.get('seconds')} s | PROBE {r.get('estimate')} raw "
                      f"{r.get('raw')} CEVAE {r.get('cevae')} | kept {r.get('kept')} | problems {r.get('problems')}",
                      flush=True)
            except Exception:
                failures += 1
                print(f"  seed {t['seed']}: FAILED\n{traceback.format_exc()}", flush=True)
    print(f"{stage} finished with {failures} failures", flush=True)
    return 1 if failures else 0


def run_pixel(stage: str) -> int:
    """The pixel CNN on every data set of the stage, strictly one at a time, each written to its own shard."""
    import run_family_v2 as rf
    import scm4_pixel_cnn as px

    proto = check_protocol(stage)
    proto_sha = rf.sha256(PROTOCOLS[stage])
    net, dev, images, groups = _load_bank()
    reader_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    failures = 0
    for task in proto["tasks"]:
        out_path = shard_path(stage, task, pixel=True)
        if out_path.exists():
            continue
        t0 = time.time()
        seed = task["seed"]
        data = g.generate(g.LEVELS[BASE_LEVEL], N, seed)
        _, src = draw_sources(data, groups, seed)
        src_all = np.column_stack([src[k] for k in src])
        view = g.learner_view(data)
        result = px.estimate(images, src_all, view["X"], view["A"], view["Y"], reader_state,
                             seed + FOLD_SEED_OFFSET, dev)
        _write_once(out_path, {"protocol_sha256": proto_sha, "stage": stage, "task": task, "pixel_cnn": result,
                               "seconds": round(time.time() - t0, 1), "generated_utc": rf.now()})
        failures += not result.get("finite")
        print(f"  seed {seed}: pixel CNN {result.get('ate')} in {round(time.time() - t0)} s "
              f"{'' if result.get('finite') else '| ' + str(result.get('reason'))}", flush=True)
    print(f"pixel pass of {stage} finished with {failures} missing values", flush=True)
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["freeze-dev", "dev", "pixel-dev", "freeze-confirm", "confirm", "pixel-confirm"])
    ap.add_argument("threshold", nargs="?", type=float)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv[1:])
    if args.command == "freeze-dev":
        return freeze("dev")
    if args.command == "freeze-confirm":
        return freeze("confirm", args.threshold)
    if args.command.startswith("pixel-"):
        return run_pixel(args.command.split("-")[1])
    return run(args.command, args.workers)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
