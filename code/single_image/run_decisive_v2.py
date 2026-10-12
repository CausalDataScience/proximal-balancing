"""CLI for the corrected bounded decisive pilot; refuses existing result paths.

New output is checkpointed atomically. Original protocols/results/weights remain immutable.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "2")

import argparse
import hashlib
import json
import platform
import resource
import shlex
import signal
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch

from . import decisive_v2 as dv
from .run_low_sigma_effects import protected_artifacts
from .run_pipeline_v2 import PROJECT, json_default, sha, source_hashes

ORIGINAL_PROTOCOL = PROJECT / "memo/2026-10-09-single-image-decisive-test-v1.md"
CORRECTED_PROTOCOL = PROJECT / "memo/2026-10-09-single-image-decisive-test-v2.md"
OUTPUT = PROJECT / "results/single_image/decisive_v2"


def protected_twenty():
    files = protected_artifacts()
    for path in (ORIGINAL_PROTOCOL,
                 PROJECT / "results/single_image/pipeline_v2_low_sigma_effect_followup.json",
                 PROJECT / "results/single_image/pipeline_v2_low_sigma_effect_followup.log"):
        files[str(path.relative_to(PROJECT))] = sha(path)
    return files


def peak_rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 ** 2 if platform.system() == "Darwin" else 1024)


def assert_frozen(sources, protected, protocol_hash):
    if source_hashes() != sources:
        raise RuntimeError("Python source changed during execution; fail closed")
    if protected_twenty() != protected:
        raise RuntimeError("protected original artifact changed during execution; fail closed")
    if sha(CORRECTED_PROTOCOL) != protocol_hash:
        raise RuntimeError("corrected pre-run protocol changed during execution; fail closed")


def atomic_json(path, payload):
    cache = Path("/tmp/probe-decisive-v2-cache")
    cache.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="checkpoint-", suffix=".json", dir=cache)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, default=json_default, indent=2, allow_nan=False)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def parser():
    p = argparse.ArgumentParser(description="Bounded image-only PROBE: independent raw audit and candidate estimation")
    p.add_argument("--n", type=int, default=48000)
    p.add_argument("--sigma", type=float, default=.3)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--encoder-steps", type=int, default=20)
    p.add_argument("--reader-epochs", type=int, default=30)
    p.add_argument("--mae-epochs", type=int, default=10)
    p.add_argument("--critic-epochs", type=int, default=30)
    p.add_argument("--raw-epochs", type=int, default=60)
    p.add_argument("--mc", type=int, default=100000)
    p.add_argument("--mc-batch", type=int, default=1000)
    p.add_argument("--family-size", type=int, default=45)
    p.add_argument("--alpha", type=float, default=.05)
    p.add_argument("--max-seconds", type=int, default=21600)
    p.add_argument("--rss-limit-mib", type=float, default=12288)
    p.add_argument("--device", choices=("cpu",), default="cpu")
    p.add_argument("--conditional-random", action="store_true")
    p.add_argument("--out-json", type=Path, default=OUTPUT / "decisive_sigma0.3_seed20261009.json")
    p.add_argument("--log", type=Path, default=OUTPUT / "decisive_sigma0.3_seed20261009.log")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.n, args.rounds, args.encoder_steps, args.mc_batch) < 1 or args.mc < 2:
        raise SystemExit("invalid positive run sizes")
    if args.family_size < (15 if args.conditional_random else 10):
        raise SystemExit("family-size must cover every possible reported model")
    for output in (args.out_json, args.log):
        if output.exists():
            raise SystemExit(f"{output} exists: refusing to overwrite before learning")
        if not output.parent.is_dir():
            raise SystemExit(f"output parent must already exist: {output.parent}")
    if args.out_json.resolve() == args.log.resolve():
        raise SystemExit("JSON and log paths must differ")
    sources, protected = source_hashes(), protected_twenty()
    protocol_hash = sha(CORRECTED_PROTOCOL)
    start = time.time()
    command = "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 " + shlex.join(
        [sys.executable, "-B", "-m", "single_image.run_decisive_v2", *(sys.argv[1:] if argv is None else argv)])
    args_dict = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    provenance = {"executed_command": command, "arguments": args_dict, "source_sha256": sources,
        "protected_before_sha256": protected, "corrected_protocol_sha256": protocol_hash,
        "original_protocol_sha256": sha(ORIGINAL_PROTOCOL),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "hardware": {"device": "cpu", "threads": 2, "platform": platform.platform()},
        "started_at": datetime.now().astimezone().isoformat(), "PID": os.getpid(),
        "runtime_cap_seconds": args.max_seconds, "RSS_cap_mib": args.rss_limit_mib}
    current = {"computation_complete": False, "provenance": provenance}
    with args.log.open("x") as logfile:
        def guard():
            if time.time() - start >= args.max_seconds:
                raise TimeoutError("predeclared local runtime cap exhausted")
            if peak_rss_mib() > args.rss_limit_mib:
                raise MemoryError("predeclared peak RSS cap exceeded")
            assert_frozen(sources, protected, protocol_hash)

        def checkpoint(result):
            nonlocal current
            current = result
            current["provenance"] = provenance
            provenance["runtime_seconds_so_far"] = time.time() - start
            provenance["peak_RSS_mib_so_far"] = peak_rss_mib()
            atomic_json(args.out_json, current)

        def log(message):
            guard()
            line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
            print(line, flush=True)
            logfile.write(line + "\n"); logfile.flush()
            if message.startswith("loop "):
                current["latest_unfinished_loop_progress"] = message

        def on_timeout(signum, frame):
            raise TimeoutError("predeclared local runtime cap exhausted")

        old_handler = signal.signal(signal.SIGALRM, on_timeout)
        signal.alarm(args.max_seconds)
        try:
            log(f"PID={os.getpid()}; image_only; bounded; primary .01; M={args.family_size}; CPU2threads")
            result = dv.run_pilot(args, log, guard, checkpoint)
            current = result
            current.pop("latest_unfinished_loop_progress", None)
        except Exception as error:
            current["computation_complete"] = False
            current["failure"] = {"type": type(error).__name__, "message": str(error),
                                  "scope": "partial run; completed stages preserved; never automatic causal return"}
            logfile.write(f"FAILED {type(error).__name__}: {error}\n"); logfile.flush()
            print(f"FAILED {type(error).__name__}: {error}", flush=True)
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
            provenance["runtime_seconds"] = time.time() - start
            provenance["peak_RSS_mib"] = peak_rss_mib()
            provenance["ended_at"] = datetime.now().astimezone().isoformat()
            protected_after = protected_twenty()
            provenance["protected_after_sha256"] = protected_after
            provenance["protected_all_unchanged"] = protected_after == protected
            provenance["source_all_unchanged"] = source_hashes() == sources
            provenance["corrected_protocol_unchanged"] = sha(CORRECTED_PROTOCOL) == protocol_hash
            current["provenance"] = provenance
            if not (provenance["protected_all_unchanged"] and provenance["source_all_unchanged"]
                    and provenance["corrected_protocol_unchanged"]):
                current["computation_complete"] = False
                current["provenance_failure"] = "original artifacts or executed source changed"
            atomic_json(args.out_json, current)
    print(f"saved {args.out_json}; complete={current['computation_complete']}; "
          f"seconds={provenance['runtime_seconds']:.1f}; peakRSS={provenance['peak_RSS_mib']:.1f}MiB", flush=True)
    return 0 if current["computation_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
