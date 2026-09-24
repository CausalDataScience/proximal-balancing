"""Run the designated fresh replicates of the frozen v3 protocol, by explicit identifier only
(memo/2026-09-22-v3-confirmation-prd-v1.md, section 4).

The existing runners keep their own default ranges untouched; this driver reads the frozen manifest and
calls each runner's `run_replicate` on exactly the listed identifiers.  It refuses to start until the
protocol exists, and it refuses any identifier the protocol does not name.

    python3 -B run_v3_fresh.py <dataset> [--workers 2]
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent / "results" / "v3_confirmation_protocol.json"


def run_one(dataset: str, r: int) -> dict:
    """Dispatch to the runner that owns the data set.  Each writes its own write-once shard."""
    if dataset == "twins":
        import run_twins as m
        return m.run_replicate(r)
    if dataset == "rhc_planted":
        import run_rhc_planted as m
        return m.run_replicate(r)
    if dataset == "ihdp":
        import run_ihdp_acic as m
        return m.run_replicate("ihdp", r)
    raise ValueError(f"unknown data set {dataset!r}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["twins", "rhc_planted", "ihdp"])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv[1:])

    if not PROTOCOL.exists():
        raise SystemExit(f"{PROTOCOL} does not exist; freeze the protocol before running")
    manifest = json.loads(PROTOCOL.read_text())["manifests"][args.dataset]
    ids = list(manifest["replicates"])
    print(f"{args.dataset}: {len(ids)} designated replicates {ids} with {args.workers} workers", flush=True)

    import multiprocessing
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_one, args.dataset, r): r for r in ids}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                res = fut.result()
                print(f"  {args.dataset} {r}: {res['status']} {res.get('seconds', '')}", flush=True)
            except Exception:
                failures += 1
                print(f"  {args.dataset} {r}: FAILED\n{traceback.format_exc()}", flush=True)
    print(f"{args.dataset}: {failures} failures", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
