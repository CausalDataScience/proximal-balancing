"""Stage runner for the direct image balancing experiment
(execution PRD memo/2026-09-21-direct-image-balancing-figure-run-prd-v1.md).

    python3 -B direct_balance_run.py freeze-dev BENCH                 write the development protocol
    python3 -B direct_balance_run.py run BENCH dev|report --rep R     one data set: thirty splits, then comparators
    python3 -B direct_balance_run.py gate BENCH                       entry check on development, freeze the report
    python3 -B direct_balance_run.py status BENCH

BENCH is shapes3d or mnist.  One invocation of `run` is one data set, which is one Slurm array task on DeltaAI
and one loop step on the Mac.  A split's shard is written the moment the split is finished, so a job that dies
or hits its time cap loses one split and the next invocation picks up where it stopped.  A split that times out
or goes non-finite is RECORDED as having no value; it is never skipped silently and never retried with other
settings.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import direct_balance_data as db
import family_v2_dgp as g

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "direct_balance"
PRDS = [HERE.parent / "memo" / "2026-09-21-direct-image-balancing-prd-v1.md",
        HERE.parent / "memo" / "2026-09-21-direct-image-balancing-figure-run-prd-v1.md"]
SOURCES = ["family_v2_dgp.py", "family_v2_probe.py", "family_v2_images.py", "probe_structured_scm.py",
           "summarize_family_v2.py", "direct_balance_data.py", "direct_balance_net.py", "direct_balance_eval.py",
           "direct_balance_run.py", "summarize_direct_balance.py"]
DEV_SEED, REPORT_SEED = 96_300_000, 96_400_000        # ranges no earlier run has used; both benchmarks share them
DEV_REPS, REPORT_LIST = 2, 20                          # the report list is fixed at twenty; the first k are run
SPLIT_SECONDS_CAP = 1_800                              # a split that needs longer than this is recorded as no value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def tasks_for(stage: str) -> list[dict]:
    base, reps = (DEV_SEED, DEV_REPS) if stage == "dev" else (REPORT_SEED, REPORT_LIST)
    return [{"rep": rep, "seed": base + 1_000 * rep} for rep in range(reps)]


def protocol_path(bench: str, stage: str) -> Path:
    return RESULTS / bench / f"{stage}_protocol_v1.json"


def dataset_dir(bench: str, stage: str, seed: int) -> Path:
    return RESULTS / bench / stage / f"seed{seed}"


def bank_digest(bench: str) -> dict:
    import family_v2_images as im
    if bench == "shapes3d":
        return {"split_rows_sha256": sha256(im.BANK_DIR / "split_rows.npz"),
                "manifest_sha256": sha256(im.BANK_DIR / "manifest.json"),
                "cache_bytes": (im.BANK_DIR / im.CACHE_NAME).stat().st_size}
    return {"mnist_npz_sha256": sha256(db.MNIST_FILE)}


def design(bench: str) -> dict:
    import direct_balance_eval as de
    import direct_balance_net as dn
    return {"benchmark": bench, "levels": db.BENCHMARKS[bench], "n": db.N, "row_split": db.ROLES,
            "block_images": list(db.BLOCK_IMAGES), "image_stream_offset": db.IMAGE_STREAM_OFFSET,
            "base_level": db.BASE_LEVEL, "train": {k: (list(v) if isinstance(v, tuple) else v)
                                                   for k, v in dn.TRAIN.items()},
            "train_correction": dn.TRAIN_CORRECTION, "initialisation": "random, fixed by seed; no pretraining",
            "representation_dim": dn.REPRESENTATION_DIM, "feature_dim": dn.FEATURE_DIM, "link": list(dn.LINK),
            "image_spec": dn.SPECS[bench], "check": de.CHECK, "nuisance": {k: (list(v) if isinstance(v, tuple) else v)
                                                                          for k, v in de.NUISANCE.items()},
            "raw_cnn": {k: (list(v) if isinstance(v, tuple) else v) for k, v in de.RAW_CNN.items()},
            "split_seconds_cap": SPLIT_SECONDS_CAP, "bank": bank_digest(bench)}


def write_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists; it is write-once")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    tmp.rename(path)


def freeze_dev(bench: str) -> int:
    payload = {"protocol_id": f"direct-balance-{bench}-dev-v1", "stage": "dev", "frozen_at_utc": now(),
               "prd_sha256": {p.name: sha256(p) for p in PRDS}, "design": design(bench), "tasks": tasks_for("dev"),
               "source_sha256": {f: sha256(HERE / f) for f in SOURCES}, "threshold": None,
               "note": "the threshold is chosen after development by the rule in summarize_direct_balance"}
    write_once(protocol_path(bench, "dev"), payload)
    print(f"frozen {protocol_path(bench, 'dev')} sha256 {sha256(protocol_path(bench, 'dev'))[:16]}")
    return 0


def check_protocol(bench: str, stage: str) -> dict:
    path = protocol_path(bench, stage)
    if not path.exists():
        raise SystemExit(f"no protocol at {path}")
    proto = json.loads(path.read_text())
    moved = [f for f, h in proto["source_sha256"].items() if sha256(HERE / f) != h]
    if moved:
        raise SystemExit(f"source files differ from the frozen protocol: {moved}")
    if json.dumps(design(bench), sort_keys=True) != json.dumps(proto["design"], sort_keys=True):
        raise SystemExit("the design or the bank differs from the frozen protocol")
    return proto


def purpose_seed(seed: int, split_index: int, purpose: int) -> int:
    return int(np.random.SeedSequence([seed, split_index, purpose]).generate_state(1)[0])


# ----------------------------------------------------------------------------- one data set

def run_dataset(bench: str, stage: str, rep: int, only: list[str] | None = None) -> int:
    import torch
    import direct_balance_eval as de
    import direct_balance_net as dn

    proto = check_protocol(bench, stage)
    proto_sha = sha256(protocol_path(bench, stage))
    task = next(t for t in proto["tasks"] if t["rep"] == rep)
    seed = task["seed"]
    folder = dataset_dir(bench, stage, seed)
    folder.mkdir(parents=True, exist_ok=True)
    dev = dn.device()
    spec = dn.SPECS[bench]
    t_all = time.time()
    images, groups = db.load_bank(bench)
    data = g.generate(g.LEVELS[db.BASE_LEVEL], db.N, seed)
    src = db.draw_images(data, groups, db.BENCHMARKS[bench], seed)
    rows = db.split_rows(db.N, seed)
    x, a, y = data["X"], data["A"], data["Y"]
    fingerprint = hashlib.sha256(np.ascontiguousarray(a).tobytes() + np.ascontiguousarray(y).tobytes()
                                 + np.ascontiguousarray(src).tobytes()).hexdigest()
    meta_path = folder / "dataset.json"
    if not meta_path.exists():
        write_once(meta_path, {"protocol_sha256": proto_sha, "benchmark": bench, "stage": stage, "task": task,
                               "data_fingerprint_sha256": fingerprint, "device": str(dev),
                               "torch": torch.__version__, "numpy": np.__version__, "started_utc": now()})
    batches = dn.Batches(images, src, x, a, dev, cache_rows=np.arange(db.N), standardise_rows=rows["represent"])
    print(f"{bench} {stage} seed {seed} on {dev} | pixel cache {batches.cache_report()} | "
          f"{time.time() - t_all:.0f} s to set up", flush=True)

    splits = db.all_splits()
    for index, split in enumerate(splits):
        name = db.split_name(split)
        if only and name not in only:
            continue
        shard_path = folder / f"{name}.json"
        if shard_path.exists():
            continue
        t0 = time.time()
        record = {"protocol_sha256": proto_sha, "benchmark": bench, "stage": stage, "seed": seed, "split": name,
                  "device": str(dev), "timing_s": {}}
        try:
            fit = dn.train_split(batches, rows, split, purpose_seed(seed, index, 0), x.shape[1], spec=spec,
                                 max_seconds=SPLIT_SECONDS_CAP)
            record["timing_s"]["train"] = fit["seconds"]
            record.update({"kept_images": fit["kept_images"], "held_images": fit["held_images"],
                           "train": {"chosen_checkpoint": fit["chosen_checkpoint"], "scheme": fit["scheme"],
                                     "checkpoint_select_gap": {str(k): v for k, v in fit["checkpoint_select_gap"].items()},
                                     "history": fit["history"]}})
            _, kept = db.image_columns(split)
            scores = {}
            for arm, state in (("after", fit["chosen_state"]), ("before", fit["initial_state"])):
                t1 = time.time()
                rep_net = dn.load_representation(state, kept, x.shape[1], dev, spec)
                chk = de.check(rep_net, batches, rows["check"], split, purpose_seed(seed, index, 1), spec)
                est = de.estimate(rep_net, batches, a, y, rows, split, purpose_seed(seed, index, 2))
                scores[arm] = est.pop("scores")
                record[arm] = {"check": chk, "estimate": est}
                record["timing_s"][arm] = round(time.time() - t1, 1)
            finite = all(record[arm]["check"]["finite"] and record[arm]["estimate"]["finite"] for arm in scores)
            record["status"] = "ok" if finite else "no value"
            if not finite:
                record["reason"] = "a non-finite check or estimate"
            np.savez_compressed(folder / f"{name}.npz", **{k: v.astype(np.float32) for k, v in scores.items()})
        except TimeoutError as err:
            record.update({"status": "no value", "reason": str(err)})
        except Exception:
            record.update({"status": "no value", "reason": "exception", "traceback": traceback.format_exc()})
        record["timing_s"]["total"] = round(time.time() - t0, 1)
        write_once(shard_path, record)
        after = record.get("after", {})
        print(f"  {name:6s} {record['status']:8s} {record['timing_s']['total']:6.0f} s | chosen "
              f"{record.get('train', {}).get('chosen_checkpoint')} | upper "
              f"{after.get('check', {}).get('upper', float('nan')):.2e} | theta "
              f"{after.get('estimate', {}).get('theta', float('nan')):.3f}", flush=True)

    if only:
        return 0
    comp_path = folder / "comparators.json"
    if not comp_path.exists():
        t0 = time.time()
        comp = de.feature_comparators(data, x, a, y, rows, purpose_seed(seed, 99, 3), dev)
        comp["raw_cnn"] = de.raw_cnn(batches, x, a, y, rows, purpose_seed(seed, 99, 4), spec)
        comp["seconds"] = round(time.time() - t0, 1)
        comp["protocol_sha256"] = proto_sha
        write_once(comp_path, comp)
        print(f"  comparators: naive {comp['naive']['theta']:.3f} X {comp['X']['theta']:.3f} raw CNN "
              f"{comp['raw_cnn']['theta']} oracle {comp['oracle']['theta']:.3f} | {comp['seconds']:.0f} s", flush=True)
    done = folder / "done.json"
    if not done.exists():
        shards = [json.loads((folder / f"{db.split_name(s)}.json").read_text()) for s in splits]
        write_once(done, {"protocol_sha256": proto_sha, "seed": seed, "finished_utc": now(),
                          "splits_ok": sum(sh["status"] == "ok" for sh in shards), "splits": len(shards),
                          "seconds_this_invocation": round(time.time() - t_all, 1),
                          "seconds_sum_of_splits": round(sum(sh["timing_s"]["total"] for sh in shards), 1)})
    print(f"data set finished in {time.time() - t_all:.0f} s (this invocation)", flush=True)
    return 0


# ----------------------------------------------------------------------------- the gate between dev and report

def gate(bench: str) -> int:
    import summarize_direct_balance as sd
    dev_proto = check_protocol(bench, "dev")
    seeds = [t["seed"] for t in dev_proto["tasks"]]
    folders = [dataset_dir(bench, "dev", s) for s in seeds]
    if not all((f / "done.json").exists() for f in folders):
        raise SystemExit("a development data set is not finished; nothing is judged early")
    result = sd.entry_check([sd.load_dataset(f) for f in folders], seeds)
    entry_path = RESULTS / bench / "dev_entry_check_v1.json"
    import run_family_v2 as rf
    write_once(entry_path, rf._jsonable({"artifact": "direct_balance_dev_entry_check", "benchmark": bench,
                                         "generated_utc": now(), "dev_protocol_sha256": sha256(protocol_path(bench, "dev")),
                                         **result}))
    print(f"threshold {result['threshold']} \nreplay {[(r['seed'], r['return_kind'], r['estimate']) for r in result['replay']]}"
          f"\ngate {result['gate']}")
    if not result["pass"]:
        print("ENTRY CHECK FAILED: the report seeds stay closed")
        return 1
    payload = dict(dev_proto, protocol_id=f"direct-balance-{bench}-report-v1", stage="report", frozen_at_utc=now(),
                   tasks=tasks_for("report"), threshold=result["threshold"],
                   dev_protocol_sha256=sha256(protocol_path(bench, "dev")), dev_entry_check_sha256=sha256(entry_path),
                   note="threshold, sources, settings and the report seed list are frozen; the first k of the list "
                        "are run, k decided by measured speed alone, and every one that is run is reported")
    write_once(protocol_path(bench, "report"), payload)
    print(f"ENTRY CHECK PASSED: frozen {protocol_path(bench, 'report')} with t = {result['threshold']['t']}")
    return 0


def status(bench: str) -> int:
    for stage in ("dev", "report"):
        for task in tasks_for(stage):
            folder = dataset_dir(bench, stage, task["seed"])
            if folder.exists():
                n = len(list(folder.glob("S*.json")))
                print(f"{bench} {stage} seed {task['seed']}: {n}/30 splits, comparators "
                      f"{(folder / 'comparators.json').exists()}, done {(folder / 'done.json').exists()}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["freeze-dev", "run", "gate", "status"])
    ap.add_argument("bench", choices=sorted(db.BENCHMARKS))
    ap.add_argument("stage", nargs="?", choices=["dev", "report"])
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--only", nargs="*", help="run just these splits and no comparators (timing and smoke runs)")
    args = ap.parse_args(argv[1:])
    if args.command == "freeze-dev":
        return freeze_dev(args.bench)
    if args.command == "gate":
        return gate(args.bench)
    if args.command == "status":
        return status(args.bench)
    if not args.stage:
        raise SystemExit("run needs a stage: dev or report")
    return run_dataset(args.bench, args.stage, args.rep, args.only)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
