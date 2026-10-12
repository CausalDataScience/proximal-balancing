"""Local image-only v2 execution. Existing results/checkpoints are never overwritten.

Run from this project's code/ directory with python3 -B -m single_image.run_pipeline_v2.
No model weights are persisted; seed, input roles, command and source hashes are recorded.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "2")

import argparse
import hashlib
import json
import platform
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from . import pipeline_v2

PROJECT = Path(__file__).resolve().parents[2]
SEALED = ("family_v2_probe.py", "family_v2_dgp.py", "hidden_proxy_recipe.py", "noise_scaled.py",
          "run_twins.py", "rhc_pooled.py", "probe_structured_scm.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes():
    paths = [PROJECT / "code" / name for name in SEALED]
    paths += sorted((PROJECT / "results/single_image/phase1").glob("*.json"))
    paths += sorted((PROJECT / "results/single_image/phase1/weights").glob("*.pt"))
    paths += [PROJECT / "memo/2026-10-08-single-image-probe-plan-v1.md"]
    return {str(p.relative_to(PROJECT)): sha(p) for p in paths}


def source_hashes():
    return {str(p.relative_to(PROJECT)): sha(p) for p in sorted((PROJECT / "code/single_image").rglob("*.py"))}


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def parser():
    p = argparse.ArgumentParser(description="Image-only pipeline with independent patient-role evaluation")
    p.add_argument("--n", type=int, default=24000)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--encoder-steps", type=int, default=20)
    p.add_argument("--reader-epochs", type=int, default=30)
    p.add_argument("--mae-epochs", type=int, default=10)
    p.add_argument("--critic-epochs", type=int, default=30)
    p.add_argument("--raw-epochs", type=int, default=60)
    p.add_argument("--mc", type=int, default=5000)
    p.add_argument("--low-n", type=int, default=6000)
    p.add_argument("--device", choices=("cpu",), default="cpu")
    p.add_argument("--out-json", type=Path, default=PROJECT / "results/single_image/pipeline_v2_local.json")
    p.add_argument("--log", type=Path, default=PROJECT / "results/single_image/pipeline_v2_local.log")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.n, args.low_n) < 400 or min(args.rounds, args.encoder_steps, args.reader_epochs,
            args.mae_epochs, args.critic_epochs, args.raw_epochs, args.mc) < 1:
        raise SystemExit("n and low-n must be >=400; all training/MC counts must be positive")
    for output in (args.out_json, args.log):
        if output.exists():
            raise SystemExit(f"{output} exists: refusing to overwrite before training")
        if not output.parent.is_dir():
            raise SystemExit(f"output directory must already exist: {output.parent}")
    before = protected_hashes()
    sources = source_hashes()
    started = time.time()
    command = "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 " + shlex.join(
        [sys.executable, "-B", "-m", "single_image.run_pipeline_v2", *(sys.argv[1:] if argv is None else argv)])
    with args.log.open("x") as logfile:
        def log(message):
            line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
            print(line, flush=True)
            logfile.write(line + "\n"); logfile.flush()
        log(f"command: {command}")
        log(f"PID={os.getpid()}; image_only=True; primary_delta2=.01 fixed before evaluation")
        result = pipeline_v2.run_local(args, log)
        after = protected_hashes()
        if after != before or source_hashes() != sources:
            raise RuntimeError("protected artifacts or running Python sources changed during execution")
        result["provenance"] = {"command": command, "cwd": os.getcwd(), "python": sys.version,
            "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "source_sha256": sources, "protected_sha256_before": before, "protected_sha256_after": after,
            "protected_count": len(before), "protected_unchanged": True, "sources_unchanged_during_run": True}
        result["seconds_with_setup"] = time.time() - started
        with args.out_json.open("x") as output:
            json.dump(result, output, indent=2, default=json_default, allow_nan=False)
            output.write("\n")
        log(f"completed: {args.out_json}; protected {len(before)} artifacts unchanged; seconds={time.time()-started:.1f}")
    return result


if __name__ == "__main__":
    main()
