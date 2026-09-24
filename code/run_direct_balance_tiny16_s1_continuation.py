"""User-authorized exploratory continuation; unchanged Tiny16 learner, gates descriptive.

Preserves the original stopped run. Reuses its controls, evaluates initial, fresh-select
chosen, and the predeclared last checkpoint 40. No audit or effect-based selection.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

import run_direct_balance_tiny16_s1 as old

OUT = old.HERE.parent / "results/direct_balance/tiny16_s1_continuation_v1"
ORIGINAL = old.OUT
CAP = 3600


def write_once(name, payload):
    path = OUT / name
    with path.open("x") as stream:
        json.dump(old.rf._jsonable(payload), stream, indent=1, allow_nan=False)
        stream.write("\n")


def main():
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUT}")
    OUT.mkdir()
    started = time.monotonic()
    original_hashes = {str(p): old.sha(p) for p in sorted(ORIGINAL.iterdir()) if p.is_file()}
    report = {"status": "running", "exploratory": True,
              "amendment": "User requests performance despite A1/A2 flags; no calibration/statistical gate aborts.",
              "original_files_sha256": original_hashes, "seed": old.SEED,
              "wall_cap_seconds": CAP, "evaluation_order": ["initial", "chosen", "last40"]}

    def finish(status):
        report["status"] = status
        report["seconds_total"] = round(time.monotonic() - started, 2)
        report["original_files_unchanged"] = all(Path(p).exists() and old.sha(Path(p)) == h
                                                for p, h in original_hashes.items())
        write_once("result_v1.json", report)
        print(json.dumps(old.rf._jsonable(report), indent=1, allow_nan=False), flush=True)

    def timeout(_sig, _frame):
        raise TimeoutError("60-minute wall clock cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(CAP)
    try:
        prior = json.loads((ORIGINAL / "protocol_v1.json").read_text())
        controls = json.loads((ORIGINAL / "stageA_controls.json").read_text())
        tests = json.loads((ORIGINAL / "stage0_tests.json").read_text())
        if not tests["passed"]:
            raise RuntimeError("Original mechanical tests did not pass")
        for name, digest in prior["source_sha256"].items():
            if old.sha(old.HERE / name) != digest:
                raise RuntimeError(f"Source drift from stopped Tiny16 protocol: {name}")
        write_once("prior_controls.json", {"controls": controls, "tests": tests,
                    "status": "Referenced unchanged, diagnostic only; not rerun"})
        dev = old.dn.device()
        images, groups = old.db.load_bank(old.BENCH)
        full = old.g.generate(old.g.LEVELS[old.db.BASE_LEVEL], old.db.N, old.SEED)
        src = old.db.draw_images(full, groups, old.db.BENCHMARKS[old.BENCH], old.SEED)
        rows = old.db.split_rows(old.db.N, old.SEED)
        x, a, y = full["X"], full["A"], full["Y"]
        held, kept = old.db.image_columns(old.SPLIT)
        tiny = old.t16.TinyBatches(images, src, x, a, dev, standardise_rows=rows["represent"])
        if {k: old.sha_arr(v) for k, v in rows.items()} != prior["rows_sha256"]:
            raise RuntimeError("Rows differ from original protocol")
        fingerprint = old.sha_arr(np.concatenate([a, y, src.ravel().astype(np.float64)]))
        if fingerprint != prior["data_fingerprint_sha256"]:
            raise RuntimeError("Data differs from original protocol")
        protocol = dict(prior)
        protocol.update({"protocol_id": "tiny16-s1-exploratory-continuation-v1",
                         "amendment": report["amendment"], "frozen_at_utc": old.rf.now(),
                         "original_protocol_sha256": old.sha(ORIGINAL / "protocol_v1.json"),
                         "continuation_source_sha256": old.sha(Path(__file__)),
                         "evaluation_order": report["evaluation_order"]})
        write_once("protocol_v1.json", protocol)
        print("Provenance verified; existing A1/A2 recorded only. Starting unchanged 40 iterations.", flush=True)
        fit = old.t16.train_split_kfold(tiny, rows, old.SPLIT, old.dr.purpose_seed(old.SEED, 0, 0),
                                       x.shape[1], max_seconds=CAP-(time.monotonic()-started)-600,
                                       log=lambda text: print(text, flush=True))
        states = fit["checkpoint_states"]
        if old.sha_state(fit["initial_state"]) != prior["initial_weights_sha256"]:
            raise RuntimeError("Initial weights differ from original protocol")
        if 40 not in states:
            raise RuntimeError("Fixed final checkpoint 40 missing")
        with (OUT / "checkpoints_v1.pt").open("xb") as stream:
            torch.save(states, stream)
        training = {k: v for k, v in fit.items() if k not in ("checkpoint_states", "initial_state")}
        training["checkpoint_weight_sha256"] = {str(k): old.sha_state(v) for k, v in states.items()}
        write_once("stageB_training.json", training)
        selection = old.de.select_with_fresh_critics(states, tiny, rows, old.SPLIT, x.shape[1],
                         old.dr.purpose_seed(old.SEED, 0, 5), old.t16.SPEC16)
        chosen = selection["chosen"]
        write_once("selection.json", selection)
        report["chosen"] = chosen
        report["steps_taken"] = sum(h["stepped"] for h in fit["history"])
        report["iterations_done"] = len(fit["history"])
        print(f"Fresh select chose {chosen}; evaluating 0 / {chosen} / 40 without reselection.", flush=True)
        audit, effects, audit_rows = {}, {}, {}
        check_seed = old.dr.purpose_seed(old.SEED, 0, 1)
        eval_points = {"initial": 0, "chosen": chosen, "last40": 40}
        cache = {}
        for label, cp in eval_points.items():
            if cp not in cache:
                rep = old.dn.load_representation(states[cp], kept, x.shape[1], dev, old.t16.SPEC16)
                chk = old.de.check(rep, tiny, rows["check"], old.SPLIT, check_seed,
                                   old.t16.SPEC16, return_rows=True)
                diff = chk.pop("rows_d")
                est = old.de.estimate(rep, tiny, a, y, rows, old.SPLIT, old.dr.purpose_seed(old.SEED, 0, 2))
                scores = est.pop("scores")
                est["absolute_error"] = abs(est["theta"] - 1.0)
                est["score_se"] = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
                cache[cp] = (chk, est, diff)
                write_once(f"evaluation_checkpoint_{cp}.json", {"checkpoint": cp, "tiny16_audit": chk,
                           "effect": est, "audit_row_differences": diff, "effect_scores": scores})
            audit[label], effects[label], audit_rows[label] = cache[cp]
            print(f"{label}: cp {cp}, tiny gap={audit[label]['gap']:.6f}, upper={audit[label]['upper']:.6f}, "
                  f"theta={effects[label]['theta']:.6f}, SE={effects[label]['score_se']:.6f}", flush=True)
        report["tiny16_audit"] = audit
        report["effect"] = effects
        report["paired_audit_change"] = {}
        for label in ("chosen", "last40"):
            diff = audit_rows[label] - audit_rows["initial"]
            report["paired_audit_change"][label] = {"mean": float(diff.mean()),
                  "se": float(diff.std(ddof=1) / np.sqrt(len(diff)))}
        mixed = old.t16.TinyBatches(images, src, x, a, dev,
                                    standardise_rows=rows["represent"], held64=held)
        audit64, cache64 = {}, {}
        for label, cp in eval_points.items():
            if cp not in cache64:
                rep = old.dn.load_representation(states[cp], kept, x.shape[1], dev, old.t16.SPEC16)
                cache64[cp] = old.de.check(rep, mixed, rows["check"], old.SPLIT,
                                           check_seed, old.dn.SPECS["shapes3d"])
                write_once(f"original64_checkpoint_{cp}.json", {"checkpoint": cp, **cache64[cp]})
            audit64[label] = cache64[cp]
            print(f"Original64 {label}: gap={audit64[label]['gap']:.6f}, upper={audit64[label]['upper']:.6f}", flush=True)
        report["original64_audit"] = audit64
        report["statistical_flags_descriptive_only"] = {
            label: {"tiny_upper_at_most_t": audit[label]["upper"] <= old.THRESHOLD,
                    "original64_upper_at_most_t": audit64[label]["upper"] <= old.THRESHOLD,
                    "effect_error_at_most_original_tolerance": effects[label]["absolute_error"] <= old.EFFECT_TOL}
            for label in eval_points}
        signal.alarm(0)
        finish("COMPLETED_EXPLORATORY_NOT_CONFIRMATORY")
        return 0
    except Exception as error:
        signal.alarm(0)
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        finish("TIMEOUT" if isinstance(error, TimeoutError) else "NUMERICAL_OR_MECHANICAL_ERROR")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
