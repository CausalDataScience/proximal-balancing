"""Reproduce the existing low-sigma control, then evaluate its omitted effect candidates.

This is a development follow-up. It does not train a low-sigma neural encoder, change the SCM,
or qualify a causal estimate. Existing results, logs and checkpoints are immutable.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "2")

import argparse
import json
import platform
import resource
import shlex
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from . import folds, level_a, nets, phantom as ph, pipeline_v2 as pv
from .run_pipeline_v2 import PROJECT, json_default, protected_hashes, sha, source_hashes

REFERENCE = PROJECT / "results/single_image/pipeline_v2_local.json"
REFERENCE_LOG = PROJECT / "results/single_image/pipeline_v2_local.log"


def protected_artifacts():
    hashes = protected_hashes()
    for path in (REFERENCE, REFERENCE_LOG):
        hashes[str(path.relative_to(PROJECT))] = sha(path)
    return hashes


def assert_reproduction(observed, expected, tolerance=1e-12):
    keys = ("Z_sha256", "q0_prediction_sha256", "q1_prediction_sha256", "patient_rows_sha256")
    for key in keys:
        if observed[key] != expected[key]:
            raise RuntimeError(f"low-sigma reproduction failed: {key}")
    numbers = {key: abs(observed[key] - expected[key]) for key in ("gap", "se", "B2")}
    for q in ("q0", "q1"):
        numbers[q + "_brier"] = abs(observed[q]["brier"] - expected[q]["brier"])
    if any(value > tolerance for value in numbers.values()):
        raise RuntimeError("low-sigma reproduction failed: scoring risk differs")
    return {"matched": True, "fingerprints": {key: observed[key] for key in keys},
            "numeric_absolute_differences": numbers, "tolerance": tolerance,
            "scope": "previously inspected S patients; development reproducibility, not a new final audit"}


def run_experiment(args, reference, log):
    started = time.time()
    nets.setup_torch(2)
    cfg = ph.PhantomConfig(n=args.n, sigma=.3, seed=args.seed, image_only=True)
    old = reference["low_sigma_control"]
    if ph.config_dict(cfg) != old["config"]:
        raise ValueError("this follow-up must use the original low-sigma SCM, n and seed")
    old_args = reference["provenance"]["arguments"]
    for key in ("reader_epochs", "mae_epochs", "critic_epochs", "raw_epochs", "mc"):
        if getattr(args, key) != old_args[key]:
            raise ValueError(f"this follow-up must preserve original {key}")
    data = ph.generate(cfg)
    groups = folds.six_groups(cfg.n, cfg.seed)
    roles = {"R": groups["R"], **folds.roles(groups["groups"], 0)}
    rows = np.concatenate(list(roles.values()))
    if len(np.unique(rows)) != cfg.n or data.X.shape[1] != 0 or cfg.treat_x != 0 or cfg.out_x != 0:
        raise AssertionError("patient roles and image-only cause/input contract must hold")
    log(f"low-sigma n={cfg.n}, seed={cfg.seed}, roles=" + str({k: len(v) for k, v in roles.items()}))
    mode = pv.prepare_mode(data, roles, "corrected", cfg.seed + 2, args.reader_epochs, args.mae_epochs, log)
    encoder = pv.LinearImageEncoder(mode, roles["D"], data.A, 0, cfg.seed + 100)
    bundle = pv.fit_for_rows(mode, encoder, mode.frame, roles["P"], data.A, cfg.seed + 200,
                             args.critic_epochs, args.raw_epochs)
    audit = pv.evaluate_rows(mode, encoder, bundle, mode.frame, roles["S"], data.A)
    reproduction = assert_reproduction(audit, old["audit_S1"])
    log("original low-sigma Z/q0/q1/patient fingerprints and scoring risks match")
    mc_seed = reference["config"]["seed"] + 80001
    pv.add_fresh_bounds([("low_sigma_S1", mode, encoder, bundle, audit)], cfg, mc_seed,
                        args.mc, len(roles["S"]), log)
    # Candidate calculation is independent of all balance/overlap qualification flags.
    # E was unused in the original low-sigma control and is never used for model selection here.
    candidates = pv.effects(mode, data, roles, encoder, bundle, None, None, cfg.seed + 900,
                            args.critic_epochs, log)
    return {"artifact": "single_image_low_sigma_effect_followup", "computation_complete": True,
            "feature_mode": "image_only", "config": ph.config_dict(cfg),
            "primary_delta2": pv.PRIMARY, "sensitivity_delta2": pv.BUDGETS,
            "role_sizes": {k: len(v) for k, v in roles.items()},
            "role_rows_sha256": {k: level_a.array_sha(v) for k, v in roles.items()},
            "protocol": {"rotation": 0, "stage1_R_images_only": True, "P_internal_validation_then_S": True,
                         "P_nuisance_fit_then_E": True, "E_used_for_selection": False,
                         "low_sigma_neural_training_run": False, "new_SCM_or_threshold": False,
                         "raw_T_preserved": True, "candidate_calculation_independent_of_gate": True,
                         "truth_inputs_to_ordinary_learning": False},
            "stage1_corrected": mode.report, "encoder": encoder.report, "audit_S1": audit,
            "original_low_sigma_reproduction": reproduction,
            "same_E_effect_diagnostics": candidates, "returned_causal_estimate": None,
            "qualification_scope": "diagnostic candidates; finite balance, overlap, ORC and nuisance bias remain separate",
            "seconds": time.time() - started}


def parser():
    p = argparse.ArgumentParser(description="Reproduce the omitted low-sigma fixed-encoder effect diagnostics")
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--seed", type=int, default=20271010)
    p.add_argument("--reader-epochs", type=int, default=30)
    p.add_argument("--mae-epochs", type=int, default=10)
    p.add_argument("--critic-epochs", type=int, default=30)
    p.add_argument("--raw-epochs", type=int, default=60)
    p.add_argument("--mc", type=int, default=5000)
    p.add_argument("--out-json", type=Path, default=PROJECT / "results/single_image/pipeline_v2_low_sigma_effect_followup.json")
    p.add_argument("--log", type=Path, default=PROJECT / "results/single_image/pipeline_v2_low_sigma_effect_followup.log")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for output in (args.out_json, args.log):
        if output.exists():
            raise SystemExit(f"{output} exists: refusing to overwrite before learning")
        if not output.parent.is_dir():
            raise SystemExit(f"output directory must already exist: {output.parent}")
    reference = json.loads(REFERENCE.read_text())
    before, sources = protected_artifacts(), source_hashes()
    started = time.time()
    command = "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 " + shlex.join(
        [sys.executable, "-B", "-m", "single_image.run_low_sigma_effects", *(sys.argv[1:] if argv is None else argv)])
    with args.log.open("x") as logfile:
        def log(message):
            line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
            print(line, flush=True); logfile.write(line + "\n"); logfile.flush()
        def timeout(signum, frame):
            raise TimeoutError("approved local run budget of 600 seconds exhausted")
        previous_handler = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(600)
        log(f"command: {command}")
        log(f"PID={os.getpid()}; CPU threads2; original image-only SCM and primary .01 preserved")
        try:
            result = run_experiment(args, reference, log)
        except Exception as exc:
            result = {"artifact": "single_image_low_sigma_effect_followup", "computation_complete": False,
                      "failure": f"{type(exc).__name__}: {exc}", "returned_causal_estimate": None}
            log(result["failure"])
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)
        after = protected_artifacts()
        if before != after or source_hashes() != sources:
            raise RuntimeError("protected artifacts or executing sources changed")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["provenance"] = {"command": command, "cwd": os.getcwd(), "python": sys.version,
            "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "source_sha256": sources, "reference_artifact": str(REFERENCE), "reference_sha256": sha(REFERENCE),
            "protected_sha256_before": before, "protected_sha256_after": after,
            "protected_count": len(before), "protected_unchanged": True, "sources_unchanged_during_run": True,
            "max_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024), "run_budget_seconds": 600}
        result["seconds_with_setup"] = time.time() - started
        with args.out_json.open("x") as output:
            json.dump(result, output, indent=2, default=json_default, allow_nan=False); output.write("\n")
        log(f"completed={result['computation_complete']}: {args.out_json}; protected {len(before)} unchanged; seconds={time.time()-started:.1f}")
    return result


if __name__ == "__main__":
    raise SystemExit(0 if main()["computation_complete"] else 1)
