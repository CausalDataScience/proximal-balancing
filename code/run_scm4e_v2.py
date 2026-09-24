"""Stage runner for SCM-4e v2: a label-free image embedding, an association front end, and the sealed PROBE
(PRD memo/2026-09-21-scm4e-v2-labelfree-association-prd-v1.md).

    python3 -B run_scm4e_v2.py precheck                   the three test data sets of PRD section 4, step 2
    python3 -B run_scm4e_v2.py freeze-dev                 write results/family_v2/scm4e_v2/dev_protocol_v1.json
    python3 -B run_scm4e_v2.py dev [--workers 3]          PROBE and the number-based comparators
    python3 -B run_scm4e_v2.py pixel-dev                  the pixel CNN on the same data sets, one at a time
    python3 -B run_scm4e_v2.py freeze-confirm T           freeze the threshold chosen on development data
    python3 -B run_scm4e_v2.py confirm [--workers 3]      the confirmation data sets
    python3 -B run_scm4e_v2.py pixel-confirm              the pixel CNN on the confirmation data sets

The data, the recording rule, the image bank and the image draws are those of SCM-4.  Two things change.  Every
image becomes the 32 numbers of a frozen autoencoder that was trained to rebuild Shapes3D images and nothing
else, so no label of any kind shaped what the learner receives.  And the block is reduced by the association
front end rather than by variance, because variance in this embedding is colour.  Everything after that is the
sealed code: representation learning, screening, AIPW, aggregation.
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
import scm4e_encoder as ec
import scm4e_frontend as fe

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "family_v2" / "scm4e_v2"
PRD = HERE.parent / "memo" / "2026-09-21-scm4e-v2-labelfree-association-prd-v1.md"
SOURCES = ["family_v2_dgp.py", "family_v2_probe.py", "family_v2_competitors.py", "family_v2_images.py",
           "probe_structured_scm.py", "positive_control_scm.py", "scm4_pixel_cnn.py", "run_scm4_images.py",
           "run_scm4.py", "run_scm4e.py", "summarize_family_v2.py", "summarize_scm4.py",
           "scm4e_encoder.py", "scm4e_frontend.py", "scm4e_pixel_cnn.py",
           "run_scm4e_v2.py", "summarize_scm4e_v2.py"]
LEVEL_NAME = "SCM-4e-v2"
BASE_LEVEL = "SCM-1"
EMBED_DIM = ec.LATENT
LEVEL = g.Level(LEVEL_NAME, 5, 2, (EMBED_DIM,) * 4 + (2 * EMBED_DIM,), 2)
N = 24_000
DEV_REPS, CONFIRM_REPS = 3, 20
DEV_SEED, CONFIRM_SEED = 97_700_000, 97_900_000       # ranges no earlier run has used
PRECHECK_SEEDS = (97_821_000, 97_822_000, 97_823_000)
PRECHECK_TOLERANCE = 2.0e-3                            # the cap of the sealed threshold rule
FOLD_SEED_OFFSET, IMAGE_STREAM_OFFSET = 7, 78          # the same folds and the same six images as SCM-4
PROTOCOLS = {"dev": RESULTS / "dev_protocol_v1.json", "confirm": RESULTS / "confirm_protocol_v1.json"}
SHARDS = {"dev": RESULTS / "dev", "confirm": RESULTS / "confirm"}
PIXEL_SHARDS = {"dev": RESULTS / "dev_pixel", "confirm": RESULTS / "confirm_pixel"}
PRECHECK_FILE = RESULTS / "precheck_v1.json"
PRECHECK_GATE = {"mean_abs_error": 0.05, "worst_abs_error": 0.10, "tpr_mean": 0.8, "fpr_mean": 0.2}


# ----------------------------------------------------------------------------- what the learner receives

def draw_sources(data: dict, groups: list, seed: int) -> tuple[dict, dict]:
    """Recorded levels and bank rows per block, with the image streams SCM-4 uses."""
    import run_scm4e as rse
    return rse.draw_sources(data, groups, seed)


def embedding_view(data: dict, enc, images, dev, groups: list, seed: int) -> tuple[dict, dict, np.ndarray, dict]:
    """The learner's view with every block a raw embedding, the evaluator record, the six images' bank rows, and
    the recorded levels (kept for the evaluator-only diagnostics, never handed to a learner)."""
    levels, src = draw_sources(data, groups, seed)
    data_e = dict(data)
    for key, rows in src.items():
        data_e[key] = np.column_stack([ec.embed(enc, images, rows[:, c], dev) for c in range(rows.shape[1])])
    if not all(np.isfinite(data_e[k]).all() for k in src):
        raise RuntimeError("an embedding is not finite; the task stops before any estimator runs")
    return g.learner_view(data_e), data_e, np.column_stack([src[k] for k in src]), levels


def frontend_diagnostics(view: dict, levels: dict, seed: int) -> dict:
    """Evaluator-only: per rotation, what the front end saw and kept, and how much of the recorded level the kept
    directions carry (fitted on the D fold, scored on the other three).  No gate reads this."""
    import family_v2_probe as pr

    fold_rows = pr.folds(len(view["A"]), seed + FOLD_SEED_OFFSET)
    out = {}
    for rot in range(4):
        rl = pr.roles(rot)
        fit = fold_rows[rl["D"]]
        rest = np.concatenate([fold_rows[rl[r]] for r in ("S", "N", "E")])
        prep = fe.AssociationPrep(view, fit, EMBED_DIM)
        rec = {}
        for j, key in enumerate(prep.report):
            f_fit, f_rest = prep.block(view, fit, j), prep.block(view, rest, j)
            r2 = []
            for c in range(levels[key].shape[1]):
                t = levels[key][:, c].astype(float)
                coef = np.linalg.lstsq(np.column_stack([np.ones(len(fit)), f_fit]), t[fit], rcond=None)[0]
                pred = np.column_stack([np.ones(len(rest)), f_rest]) @ coef
                r2.append(round(float(1 - np.mean((t[rest] - pred) ** 2) / np.var(t[rest])), 4))
            rec[key] = dict(prep.report[key], level_r2_from_kept_directions=r2)
        out[f"rotation{rot}"] = rec
    return out


# ----------------------------------------------------------------------------- protocol

def design_snapshot() -> dict:
    import run_family_v2 as rf
    import run_scm4_images as ri
    import scm4e_pixel_cnn as pxe
    base = rf.design_snapshot()
    sds = {k: v.tolist() for k, v in im.population_sd(g.LEVELS[BASE_LEVEL]).items()}
    return {"family_v2": {"coefficients": base["coefficients"], "base_level": base["levels"][BASE_LEVEL],
                          "probe": base["probe"], "competitors": base["competitors"]},
            "learner_blocks": {"dims": list(LEVEL.d_blocks), "k": LEVEL.k,
                               "embedding": f"the frozen label-free encoder's {EMBED_DIM} numbers per image; "
                                            "W5 is its two images side by side"},
            "front_end": {"rule": "canonical correlation of the block against (X, A, the other blocks), fitted on "
                                  "the representation fold of each rotation",
                          "threshold_multiple": fe.THRESHOLD_MULTIPLE, "permutations": fe.PERMUTATIONS,
                          "cap": "at most one direction per image in the block"},
            "recording": {"levels": im.LEVELS, "population_sd": sds, "range_sd": 3.0,
                          "image_stream_offset": IMAGE_STREAM_OFFSET},
            "bank": {"split": im.SPLIT, "split_seed": im.SPLIT_SEED, "source_md5": im.H5_MD5},
            "encoder": {"config": ec.ENCODER, "latent": ec.LATENT, "check": ec.CHECK,
                        "seeds": {"init": ec.INIT_SEED, "batch_order": ec.ORDER_SEED, "subset": ec.SUBSET_SEED},
                        "digests": current_digests()},
            "reader_bank_digests": {k: v for k, v in ri.current_digests().items() if "split_rows" in k or "manifest" in k},
            "pixel_cnn": pxe.CONFIG}


def current_digests() -> dict:
    import run_family_v2 as rf
    return {p.name: rf.sha256(p) for p in (ec.WEIGHTS, ec.RECORD, ec.CHECK_FILE) if p.exists()}


def tasks_for(stage: str) -> list[dict]:
    if stage == "dev":
        return [{"level": LEVEL_NAME, "rep": rep, "seed": DEV_SEED + rep} for rep in range(DEV_REPS)]
    return [{"level": LEVEL_NAME, "rep": rep, "seed": CONFIRM_SEED + 1_000 * rep} for rep in range(CONFIRM_REPS)]


def freeze(stage: str, threshold: float | None = None) -> int:
    import run_family_v2 as rf
    path = PROTOCOLS[stage]
    if stage == "confirm" and not (threshold and 0 < threshold <= PRECHECK_TOLERANCE):
        raise SystemExit(f"give the development threshold, a positive number no larger than {PRECHECK_TOLERANCE}")
    if path.exists():
        raise SystemExit(f"{path} exists; protocols are write-once")
    if stage == "dev" and not PRECHECK_FILE.exists():
        raise SystemExit("run the pre-check first; the development seeds stay closed until it passes")
    if stage == "dev":
        stale = precheck_is_stale(json.loads(PRECHECK_FILE.read_text()))
        if stale:
            raise SystemExit(f"the development seeds stay closed: {stale}")
    if stage == "confirm" and not PROTOCOLS["dev"].exists():
        raise SystemExit("the development protocol must exist first")
    ec.load()                                            # refuses unless the encoder check exists and passed
    payload = {"protocol_id": f"family-v2-scm4e-v2-{stage}-v1", "stage": stage, "frozen_at_utc": rf.now(),
               "prd": {"path": str(PRD.relative_to(HERE.parent)), "sha256": rf.sha256(PRD)},
               "n": N, "design": design_snapshot(), "tasks": tasks_for(stage),
               "source_sha256": {f: rf.sha256(HERE / f) for f in SOURCES},
               "precheck_sha256": rf.sha256(PRECHECK_FILE) if PRECHECK_FILE.exists() else None,
               "screen_thresholds": {LEVEL_NAME: threshold} if threshold else None,
               "gates": "the thirteen gates of SCM-4 (family v2 PRD section 10 plus the pixel CNN)",
               "notes": {"dev": "the threshold is chosen after development by the sealed family-v2 rule",
                         "confirm": "threshold, sources, encoder and design are frozen; nothing is chosen here"}[stage]}
    if stage == "confirm":
        payload["dev_protocol_sha256"] = rf.sha256(PROTOCOLS["dev"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    tmp.rename(path)
    print(f"frozen {path} ({len(payload['tasks'])} tasks), sha256 {rf.sha256(path)}")
    return 0


def precheck_is_stale(pre: dict) -> list[str]:
    """Why this pre-check cannot open the development seeds: it failed, or it ran on other code or another encoder."""
    import run_family_v2 as rf
    reasons = [] if pre.get("pass") else ["the pre-check did not pass"]
    moved = [f for f, h in pre.get("source_sha256", {}).items() if rf.sha256(HERE / f) != h]
    if moved or set(pre.get("source_sha256", {})) != set(SOURCES):
        reasons.append(f"source files changed since the pre-check: {moved or 'the list of sources differs'}")
    if pre.get("encoder_digests", {}).get("weights_sha256") != current_digests().get(ec.WEIGHTS.name):
        reasons.append("the encoder is not the one the pre-check used")
    if pre.get("prd_sha256") != rf.sha256(PRD):
        reasons.append("the PRD changed since the pre-check")
    return reasons


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
        raise SystemExit("the design, the bank or the encoder differs from the frozen protocol")
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
    """The frozen encoder and the image bank.  The reader's exclusion record still decides the bank grouping, so
    the six images of a seed are the ones SCM-4 drew."""
    import run_scm4_images as ri
    _, _, excluded, _ = ri.load_frozen_reader()
    enc, dev, digests = ec.load()
    images = im.load_images()
    groups = im.bank_by_level(np.load(im.BANK_DIR / "split_rows.npz")["causal_bank"], excluded)
    return enc, dev, images, groups, digests


# ----------------------------------------------------------------------------- one data set

def estimators(view: dict, data_e: dict, levels: dict, seed: int, tolerance: float, want_cevae: bool) -> dict:
    """Everything that runs on the embedding: PROBE under the association front end, the AIPW baselines, the
    proximal rows and CEVAE.  The pixel CNN is a separate pass."""
    import family_v2_competitors as fc
    import family_v2_probe as pr
    import run_scm4_images as ri

    fold_seed = seed + FOLD_SEED_OFFSET
    out = {"timing_s": {}}
    t0 = time.time()
    out["front_end"] = frontend_diagnostics(view, levels, seed)
    out["timing_s"]["front_end_diagnostics"] = round(time.time() - t0, 1)
    with fe.installed(EMBED_DIM):
        t0 = time.time()
        res = pr.run_dataset(view, LEVEL, tolerance, fold_seed, keep_b=True)
        out["probe"], out["timing_s"]["probe"] = res, round(time.time() - t0, 1)
        out["probe_problems"] = ri.validate(res)
        t0 = time.time()
        out["baselines"] = fc.aipw_baselines(data_e, fold_seed)     # X, raw, summary, oracle, naive
        out["timing_s"]["baselines"] = round(time.time() - t0, 1)
    t0 = time.time()
    view_v, report_all = fe.variate_view(view, EMBED_DIM)           # the methods without folds: one fit on all rows
    out["front_end_all_rows"] = report_all
    out["proximal"] = fc.proximal_rows(view_v)
    out["timing_s"]["proximal"] = round(time.time() - t0, 2)
    if want_cevae:
        t0 = time.time()
        variates = fc.cevae_ate(view_v, seed + 11)
        out["cevae"] = {"variates": {"ate": variates["ate"], "final_loss": variates["final_loss"]}}
        full = fc.cevae_ate(view, seed + 11)
        out["cevae"]["full_embedding"] = {"ate": full["ate"], "final_loss": full["final_loss"]}
        out["cevae"]["ate"] = variates["ate"]                       # the row the comparator table reads
        out["timing_s"]["cevae"] = round(time.time() - t0, 1)
    return out


def run_task(stage: str, task: dict, proto_sha: str, tolerance: float | None) -> dict:
    import torch
    torch.set_num_threads(1)
    import run_family_v2 as rf

    out_path = shard_path(stage, task)
    if out_path.exists():
        return {"task": task, "status": "exists"}
    t_all = time.time()
    enc, dev, images, groups, digests = _load_bank()
    seed = task["seed"]
    shard = {"protocol_sha256": proto_sha, "stage": stage, "task": task, "n": N, "generated_utc": rf.now(),
             "device": str(dev), "encoder_digests": digests}
    t0 = time.time()
    data = g.generate(g.LEVELS[BASE_LEVEL], N, seed)
    view, data_e, _, levels = embedding_view(data, enc, images, dev, groups, seed)
    embed_seconds = round(time.time() - t0, 1)
    del enc, images
    shard.update(estimators(view, data_e, levels, seed, 1.0 if tolerance is None else tolerance, want_cevae=True))
    if tolerance is None:
        shard["probe_problems"] = [p for p in shard["probe_problems"]
                                   if not p.startswith(("return kind", "no finite estimate"))]
    shard["timing_s"]["images_and_embedding"] = embed_seconds
    shard["timing_s"]["total"] = round(time.time() - t_all, 1)
    _write_once(out_path, shard)
    kept = {k: v["kept"] for k, v in shard["front_end"]["rotation0"].items()}
    return {"task": task, "status": "written", "seconds": shard["timing_s"]["total"],
            "estimate": shard["probe"].get("estimate"), "problems": shard["probe_problems"],
            "cevae": shard["cevae"]["ate"], "raw": shard["baselines"]["raw"], "kept": kept}


# ----------------------------------------------------------------------------- the passes

def precheck() -> int:
    """PRD section 4, step 2: the real code path on three test data sets, before any development seed is opened."""
    import run_family_v2 as rf
    import summarize_family_v2 as sm

    if PRECHECK_FILE.exists():
        raise SystemExit(f"{PRECHECK_FILE} exists; it is write-once")
    enc, dev, images, groups, digests = _load_bank()
    rows = []
    for seed in PRECHECK_SEEDS:
        t0 = time.time()
        data = g.generate(g.LEVELS[BASE_LEVEL], N, seed)
        view, data_e, _, levels = embedding_view(data, enc, images, dev, groups, seed)
        out = estimators(view, data_e, levels, seed, PRECHECK_TOLERANCE, want_cevae=False)
        rep = sm.replay({"probe": out["probe"], "task": {"seed": seed}}, PRECHECK_TOLERANCE)
        rep.update(sm.mechanism(rep))
        kept = [(rot, key, rec["kept"], rec["images"], rec["nothing_above_threshold"])
                for rot, block in out["front_end"].items() for key, rec in block.items()]
        records, sealed_estimate = out["probe"]["records"], out["probe"]["estimate"]
        targets = {name: {"estimate": records[name]["estimate"], "retained": name in rep["retained"],
                          "max_upper": max(r["screen"]["upper"] for r in records[name]["rotations"])}
                   for name in sm.TARGETS}
        agrees = (sealed_estimate is None and rep["estimate"] is None) or (
            sealed_estimate is not None and rep["estimate"] is not None
            and abs(sealed_estimate - rep["estimate"]) < 1e-9)
        rows.append({"seed": seed, "estimate": rep["estimate"], "return_kind": rep["return_kind"],
                     "component_sizes": rep["component_sizes"], "tpr": rep["tpr"], "fpr": rep["fpr"],
                     "purity": rep["purity"], "problems": out["probe_problems"],
                     "retained": rep["retained"], "target_splits": targets,
                     "sealed_run_estimate": sealed_estimate, "replay_agrees_with_sealed_run": bool(agrees),
                     "baselines": out["baselines"], "proximal": out["proximal"], "timing_s": out["timing_s"],
                     "front_end": out["front_end"], "kept_as_images": all(k == n for _, _, k, n, _ in kept),
                     "any_empty_threshold": any(e for *_, e in kept),
                     "level_r2_min": min(r for block in out["front_end"].values() for rec in block.values()
                                         for r in rec["level_r2_from_kept_directions"]),
                     "seconds": round(time.time() - t0, 1)})
        print(f"  seed {seed}: PROBE {rows[-1]['estimate']} sizes {rows[-1]['component_sizes']} "
              f"TPR {rows[-1]['tpr']:.3f} FPR {rows[-1]['fpr']:.3f} | kept as images "
              f"{rows[-1]['kept_as_images']} | level R2 min {rows[-1]['level_r2_min']} "
              f"| {rows[-1]['seconds']} s", flush=True)
    errs = [abs(r["estimate"] - 1.0) for r in rows if r["estimate"] is not None]
    gate = {"complete": all(not r["problems"] for r in rows) and len(errs) == len(rows),
            "unique_return": all(r["return_kind"] == "unique_largest" for r in rows),
            "mean_abs_error": bool(errs) and float(np.mean(errs)) <= PRECHECK_GATE["mean_abs_error"],
            "worst_abs_error": bool(errs) and max(errs) <= PRECHECK_GATE["worst_abs_error"],
            "front_end_kept_one_direction_per_image": all(r["kept_as_images"] for r in rows),
            "front_end_never_empty": not any(r["any_empty_threshold"] for r in rows),
            "screen_tpr": float(np.mean([r["tpr"] for r in rows])) >= PRECHECK_GATE["tpr_mean"],
            "screen_fpr": float(np.mean([r["fpr"] for r in rows])) <= PRECHECK_GATE["fpr_mean"]}
    payload = {"artifact": "family_v2_scm4e_v2_precheck", "prd_sha256": rf.sha256(PRD),
               "seeds": list(PRECHECK_SEEDS), "tolerance": PRECHECK_TOLERANCE, "n": N,
               "encoder_digests": digests, "source_sha256": {f: rf.sha256(HERE / f) for f in SOURCES},
               "requirement": PRECHECK_GATE, "results": rows,
               "mean_abs_error": float(np.mean(errs)) if errs else None,
               "gate": gate, "pass": all(gate.values()), "generated_utc": rf.now()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = PRECHECK_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(rf._jsonable(payload), indent=1, allow_nan=False) + "\n")
    tmp.rename(PRECHECK_FILE)
    print(f"pre-check {'PASS' if payload['pass'] else 'FAIL'}: {gate}\nwritten {PRECHECK_FILE}")
    return 0 if payload["pass"] else 1


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
    import scm4e_pixel_cnn as pxe

    proto = check_protocol(stage)
    proto_sha = rf.sha256(PROTOCOLS[stage])
    enc, dev, images, groups, _ = _load_bank()
    encoder_state = {k: v.detach().clone() for k, v in enc.state_dict().items()}
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
        result = pxe.estimate(images, src_all, view["X"], view["A"], view["Y"], encoder_state,
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
    ap.add_argument("command", choices=["precheck", "freeze-dev", "dev", "pixel-dev",
                                        "freeze-confirm", "confirm", "pixel-confirm"])
    ap.add_argument("threshold", nargs="?", type=float)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv[1:])
    if args.command == "precheck":
        return precheck()
    if args.command == "freeze-dev":
        return freeze("dev")
    if args.command == "freeze-confirm":
        return freeze("confirm", args.threshold)
    if args.command.startswith("pixel-"):
        return run_pixel(args.command.split("-")[1])
    return run(args.command, args.workers)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
