"""Stage runner for SCM-4, the image level of SCM family v2 (spec memo/2026-09-20-scm4-shapes3d-reader-spec-v1.md).

    python3 -B run_scm4.py freeze-dev               write results/family_v2/scm4/dev_protocol_v1.json (write-once)
    python3 -B run_scm4.py dev [--workers 3]        run the development data sets
    python3 -B run_scm4.py freeze-confirm T         freeze the screen threshold chosen on development data
    python3 -B run_scm4.py confirm [--workers 3]    run the confirmation data sets

SCM-4 is SCM-1 with one change: each of the six observed coordinates of W reaches the learner as an original
Shapes3D image, and the frozen reader turns it back into a number.  Everything after that is the sealed SCM-1
code, imported unchanged: PROBE, the AIPW baselines, proximal 2SLS, Park's single proxy control and CEVAE all
receive the same six readings.  One comparator is new, the pixel CNN of scm4_pixel_cnn.py.

The sealed family-v2 files are not edited here, because their hashes are part of the record behind SCM-1 to 3
and of the reader's final check.  A task refuses to start unless the frozen reader passed that check and the
files it was checked against are unchanged (run_scm4_images.load_frozen_reader).
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
RESULTS = HERE.parent / "results" / "family_v2" / "scm4"
SPEC = HERE.parent / "memo" / "2026-09-20-scm4-shapes3d-reader-spec-v1.md"
SOURCES = ["family_v2_dgp.py", "family_v2_probe.py", "family_v2_competitors.py", "family_v2_images.py",
           "probe_structured_scm.py", "positive_control_scm.py", "scm4_pixel_cnn.py", "run_scm4_images.py",
           "run_scm4.py"]
LEVEL_NAME = "SCM-4"
BASE_LEVEL = "SCM-1"                                  # the data generating process and the learner's block layout
N = 24_000
DEV_REPS, CONFIRM_REPS = 3, 20
DEV_SEED, CONFIRM_SEED = 97_300_000, 97_400_000       # a range no earlier run has used
FOLD_SEED_OFFSET, IMAGE_STREAM_OFFSET = 7, 78
PROTOCOLS = {"dev": RESULTS / "dev_protocol_v1.json", "confirm": RESULTS / "confirm_protocol_v1.json"}
SHARDS = {"dev": RESULTS / "dev", "confirm": RESULTS / "confirm"}


def design_snapshot() -> dict:
    import run_family_v2 as rf
    import run_scm4_images as ri
    import scm4_pixel_cnn as px
    base = rf.design_snapshot()
    sds = {k: v.tolist() for k, v in im.population_sd(g.LEVELS[BASE_LEVEL]).items()}
    return {"family_v2": {"coefficients": base["coefficients"], "level": base["levels"][BASE_LEVEL],
                          "probe": base["probe"], "competitors": base["competitors"]},
            "recording": {"levels": im.LEVELS, "population_sd": sds, "range_sd": 3.0, "image_stream_offset": IMAGE_STREAM_OFFSET},
            "bank": {"split": im.SPLIT, "split_seed": im.SPLIT_SEED, "source_md5": im.H5_MD5},
            "reader": {"seed": im.READER_SEED, "training": im.READER, "gate": im.GATE, "digests": ri.current_digests()},
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
    payload = {"protocol_id": f"family-v2-scm4-{stage}-v1", "stage": stage, "frozen_at_utc": rf.now(),
               "spec": {"path": str(SPEC.relative_to(HERE.parent)), "sha256": rf.sha256(SPEC)},
               "n": N, "design": design_snapshot(), "tasks": tasks_for(stage),
               "source_sha256": {f: rf.sha256(HERE / f) for f in SOURCES},
               "screen_thresholds": {LEVEL_NAME: threshold} if threshold else None,
               "gates": "family v2 PRD section 10, plus: the paired 95% upper bound of PROBE's absolute error minus the "
                        "pixel CNN's is below zero, and every comparator returns a finite value on every data set",
               "notes": {"dev": "the threshold is chosen after development by the sealed family-v2 rule; shards keep "
                                "every rotation-level screen statistic so the aggregate is replayed",
                         "confirm": "threshold, sources, reader and design are frozen; nothing is chosen on these data"}[stage]}
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
        raise SystemExit("the design, the bank or the reader differs from the frozen protocol")
    return proto


def shard_path(stage: str, task: dict) -> Path:
    return SHARDS[stage] / f"{task['level']}_seed{task['seed']}.json"


def image_view(data: dict, net, images, dev, groups: list, seed: int) -> tuple[dict, dict, np.ndarray, dict]:
    """What the learner of SCM-4 receives, the matching evaluator record, the bank rows of the six images, and the
    reading errors.  The image streams are the ones run_scm4_images.build_views uses, so the two agree exactly."""
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


def run_task(stage: str, task: dict, proto_sha: str, tolerance: float | None) -> dict:
    import torch
    torch.set_num_threads(1)
    import family_v2_competitors as fc
    import family_v2_probe as pr
    import run_family_v2 as rf
    import run_scm4_images as ri
    import scm4_pixel_cnn as px

    out_path = shard_path(stage, task)
    if out_path.exists():
        return {"task": task, "status": "exists"}
    t_all = time.time()
    net, dev, excluded, _ = ri.load_frozen_reader()
    images = im.load_images()
    groups = im.bank_by_level(np.load(im.BANK_DIR / "split_rows.npz")["causal_bank"], excluded)
    lv = g.LEVELS[BASE_LEVEL]
    seed = task["seed"]
    fold_seed = seed + FOLD_SEED_OFFSET
    shard = {"protocol_sha256": proto_sha, "stage": stage, "task": task, "n": N, "generated_utc": rf.now(),
             "timing_s": {}, "device": str(dev)}

    t0 = time.time()
    data = g.generate(lv, N, seed)
    view, data4, src, shard["reading"] = image_view(data, net, images, dev, groups, seed)
    shard["timing_s"]["images_and_reading"] = round(time.time() - t0, 1)
    if any(s["nonfinite"] for s in shard["reading"].values()):
        raise RuntimeError("a reading is not finite; the task stops before any estimator runs")

    t0 = time.time()
    res = pr.run_dataset(view, lv, 1.0 if tolerance is None else tolerance, fold_seed, keep_b=True)
    problems = ri.validate(res)
    if tolerance is None:                              # in development the aggregate is replayed later
        problems = [p for p in problems if not p.startswith(("return kind", "no finite estimate"))]
    shard["probe"], shard["probe_problems"] = res, problems
    shard["timing_s"]["probe"] = round(time.time() - t0, 1)

    t0 = time.time()
    shard["baselines"] = fc.aipw_baselines(data4, fold_seed)
    shard["timing_s"]["baselines"] = round(time.time() - t0, 1)
    t0 = time.time()
    shard["proximal"] = fc.proximal_rows(view)
    shard["timing_s"]["proximal"] = round(time.time() - t0, 2)
    t0 = time.time()
    cev = fc.cevae_ate(view, seed + 11)
    shard["cevae"] = {"ate": cev["ate"], "final_loss": cev["final_loss"]}
    shard["timing_s"]["cevae"] = round(time.time() - t0, 1)
    t0 = time.time()
    reader_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    shard["pixel_cnn"] = px.estimate(images, src, view["X"], view["A"], view["Y"], reader_state, fold_seed, dev)
    shard["timing_s"]["pixel_cnn"] = round(time.time() - t0, 1)
    shard["timing_s"]["total"] = round(time.time() - t_all, 1)

    shard = rf._jsonable(shard)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(shard, allow_nan=False) + "\n")
    tmp.rename(out_path)
    return {"task": task, "status": "written", "seconds": shard["timing_s"]["total"],
            "estimate": res.get("estimate"), "problems": problems, "cevae": shard["cevae"]["ate"],
            "pixel_cnn": shard["pixel_cnn"].get("ate"), "raw": shard["baselines"]["raw"]}


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
                print(f"  seed {t['seed']}: {r['status']} {r.get('seconds')} s | PROBE {r.get('estimate')} "
                      f"raw {r.get('raw')} pixel CNN {r.get('pixel_cnn')} CEVAE {r.get('cevae')} "
                      f"| problems {r.get('problems')}", flush=True)
            except Exception:
                failures += 1                           # the seed stays in the task list; nothing replaces it
                print(f"  seed {t['seed']}: FAILED\n{traceback.format_exc()}", flush=True)
    print(f"{stage} finished with {failures} failures", flush=True)
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["freeze-dev", "dev", "freeze-confirm", "confirm"])
    ap.add_argument("threshold", nargs="?", type=float)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv[1:])
    if args.command == "freeze-dev":
        return freeze("dev")
    if args.command == "freeze-confirm":
        return freeze("confirm", args.threshold)
    return run(args.command, args.workers)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
