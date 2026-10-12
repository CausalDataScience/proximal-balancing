"""CLI for the objective comparison v1 (memo/2026-10-09-single-image-objective-comparison-v1.md).

Run from this project's code/ directory:

    nice -n 10 python3 -B -m single_image.run_objective_v1 --arms O3 --splits 1,2,3,4,5 --level-a-fresh --tag O3
    nice -n 10 python3 -B -m single_image.run_objective_v1 --arms O1,O2 --splits 1 --tag O1O2-S1
    nice -n 10 python3 -B -m single_image.run_objective_v1 --merge

A run writes one part JSON/log, objective_sigma{s}_seed{seed}.part-{tag}.json, and refuses existing paths.
--merge combines the complete parts of one sigma/seed into objective_sigma{s}_seed{seed}.json and rebuilds
objective_summary_v1.json (the section-4 decision table) from every merged file.  Only the derived merge
outputs can be replaced, and only with --replace-derived.  Caches (stage 1, warm starts) live in scratch.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "2")

import argparse
import hashlib
import json
import math
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

from . import level_a, objective_v1 as ob
from .run_pipeline_v2 import PROJECT, json_default, sha, source_hashes

PROTOCOL = PROJECT / "memo/2026-10-09-single-image-objective-comparison-v1.md"
DECISIVE_PROTOCOL = PROJECT / "memo/2026-10-09-single-image-decisive-test-v2.md"
DECISIVE_JSON = PROJECT / "results/single_image/decisive_v2/decisive_sigma0.3_seed20261009.json"
OUTPUT = PROJECT / "results/single_image/objective_v1"
SCRATCH = Path("/private/tmp/claude-502/-Users-yonghanjung-paios/1935c64b-8526-4d67-8733-aa20e64b80b6/"
               "scratchpad/objective_v1")
IMPORTED = ("__init__.py", "objective_v1.py", "run_objective_v1.py", "decisive_v2.py", "pipeline_v2.py",
            "folds.py", "level_a.py", "nets.py", "phantom.py", "reader.py", "regions.py", "stage1.py", "mae.py",
            "summaries.py", "oracle.py", "run_pipeline_v2.py")


def imported_hashes():
    return {name: sha(PROJECT / "code/single_image" / name) for name in IMPORTED}


def peak_rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 ** 2 if platform.system() == "Darwin" else 1024)


def atomic_json(path: Path, payload):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="checkpoint-", suffix=".json", dir=SCRATCH)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, default=json_default, indent=1, allow_nan=False)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def stem(sigma, seed):
    return f"objective_sigma{sigma:g}_seed{seed}"


def int_list(text):
    return [int(x) for x in text.split(",") if x]


def parser():
    p = argparse.ArgumentParser(description="Objective comparison v1: O1 Brier gap, O2 log-loss gap, O3 KCI")
    p.add_argument("--n", type=int, default=48000)
    p.add_argument("--sigma", type=float, default=.3)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--arms", type=lambda s: s.split(","), default=["O1", "O2", "O3"])
    p.add_argument("--inits", type=lambda s: s.split(","), default=["mae", "warm"])
    p.add_argument("--splits", type=int_list, default=[1, 2, 3, 4, 5], help="1-based singleton splits")
    p.add_argument("--tag", default=None)
    p.add_argument("--level-a-fresh", action="store_true", help="also measure D2_fresh for level A")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--encoder-steps", type=int, default=20)
    p.add_argument("--o3-steps", type=int, default=600)
    p.add_argument("--kci-batch", type=int, default=ob.KCI_BATCH)
    p.add_argument("--kci-eps", type=float, default=ob.KCI_EPS)
    p.add_argument("--lr", type=float, default=ob.LR)
    p.add_argument("--mu", type=float, default=ob.MU)
    p.add_argument("--reader-epochs", type=int, default=30)
    p.add_argument("--mae-epochs", type=int, default=10)
    p.add_argument("--critic-epochs", type=int, default=30)
    p.add_argument("--raw-epochs", type=int, default=60)
    p.add_argument("--warm-epochs", type=int, default=15)
    p.add_argument("--warm-lr", type=float, default=1e-3)
    p.add_argument("--warm-patience", type=int, default=3)
    p.add_argument("--mc", type=int, default=100000, help="fresh draws for D2_fresh (0 disables)")
    p.add_argument("--fresh-batch", type=int, default=256)
    p.add_argument("--fresh-epochs", type=int, default=40)
    p.add_argument("--fresh-raw-epochs", type=int, default=20)
    p.add_argument("--fresh-patience", type=int, default=6)
    p.add_argument("--fresh-raw-patience", type=int, default=4)
    p.add_argument("--fresh-fit-seed", type=int, default=20261009 + 40000001)
    p.add_argument("--max-seconds", type=int, default=21600)
    p.add_argument("--rss-limit-mib", type=float, default=9000)
    p.add_argument("--out-dir", type=Path, default=OUTPUT)
    p.add_argument("--cache-dir", type=Path, default=SCRATCH / "cache")
    p.add_argument("--no-reference", action="store_true", help="skip the decisive_v2 level-A identity check")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--replace-derived", action="store_true")
    return p


def load_reference(args):
    if args.no_reference or not DECISIVE_JSON.exists():
        return None, "not used"
    ref = json.loads(DECISIVE_JSON.read_text())
    a = ref["provenance"]["arguments"]
    same = (ref["config"]["n"] == args.n and ref["config"]["sigma"] == args.sigma and ref["config"]["seed"] == args.seed
            and ref["config"]["link"] == "bounded" and a["reader_epochs"] == args.reader_epochs
            and a["mae_epochs"] == args.mae_epochs and a["critic_epochs"] == args.critic_epochs
            and a["raw_epochs"] == args.raw_epochs)
    return (ref, "config matches; level A recomputed and compared") if same else (None, "config differs; recomputed only")


def provenance_block(args, argv, start):
    command = "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 " + shlex.join(
        [sys.executable, "-B", "-m", "single_image.run_objective_v1", *(sys.argv[1:] if argv is None else argv)])
    return {"executed_command": command,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "source_sha256_all_single_image": source_hashes(), "imported_module_sha256": imported_hashes(),
            "sealed_project_sha256": level_a.sealed_hashes(),
            "protocol": {"path": str(PROTOCOL.relative_to(PROJECT)), "sha256": sha(PROTOCOL)},
            "decisive_v2_protocol": {"path": str(DECISIVE_PROTOCOL.relative_to(PROJECT)), "sha256": sha(DECISIVE_PROTOCOL)},
            "decisive_v2_reference_json_sha256": sha(DECISIVE_JSON) if DECISIVE_JSON.exists() else None,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__},
            "hardware": {"device": "cpu", "torch_threads": torch.get_num_threads(), "platform": platform.platform(),
                         "nice": os.getpriority(os.PRIO_PROCESS, 0)},
            "started_at": datetime.now().astimezone().isoformat(), "PID": os.getpid(),
            "runtime_cap_seconds": args.max_seconds, "RSS_cap_mib": args.rss_limit_mib, "_start": start}


def run(args, argv):
    if args.tag is None:
        args.tag = "-".join(args.arms) + "-S" + "".join(str(s) for s in args.splits)
    bad = [a for a in args.arms if a not in ob.ARMS] + [i for i in args.inits if i not in ("mae", "warm")]
    if bad or not args.splits or any(s not in range(1, 6) for s in args.splits):
        raise SystemExit(f"invalid arms/inits/splits: {bad} {args.splits}")
    args.splits = [s - 1 for s in args.splits]
    out_json = args.out_dir / f"{stem(args.sigma, args.seed)}.part-{args.tag}.json"
    out_log = out_json.with_suffix(".log")
    for output in (out_json, out_log):
        if output.exists():
            raise SystemExit(f"{output} exists: refusing to overwrite before learning")
    if not args.out_dir.is_dir():
        raise SystemExit(f"output directory must already exist: {args.out_dir}")
    start = time.time()
    provenance = provenance_block(args, argv, start)
    provenance.pop("_start")
    sources, protocol_hash = provenance["source_sha256_all_single_image"], provenance["protocol"]["sha256"]
    reference, ref_status = load_reference(args)
    provenance["level_A_reference_status"] = ref_status
    current = {"computation_complete": False, "provenance": provenance}
    with out_log.open("x") as logfile:
        def guard():
            if time.time() - start >= args.max_seconds:
                raise TimeoutError("predeclared runtime cap exhausted")
            if peak_rss_mib() > args.rss_limit_mib:
                raise MemoryError("predeclared peak RSS cap exceeded")
            if source_hashes() != sources or sha(PROTOCOL) != protocol_hash:
                raise RuntimeError("Python source or protocol changed during execution; fail closed")

        def checkpoint(result):
            nonlocal current
            current = result
            current["provenance"] = provenance
            provenance["runtime_seconds_so_far"] = time.time() - start
            provenance["peak_RSS_mib_so_far"] = peak_rss_mib()
            atomic_json(out_json, current)

        def log(message):
            guard()
            line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
            print(line, flush=True)
            logfile.write(line + "\n"); logfile.flush()

        def on_alarm(signum, frame):
            raise TimeoutError("predeclared runtime cap exhausted")

        old = signal.signal(signal.SIGALRM, on_alarm)
        signal.alarm(args.max_seconds)
        try:
            log(f"PID={os.getpid()}; command: {provenance['executed_command']}")
            log(f"level-A reference: {ref_status}")
            current = ob.run_part(args, log, guard, checkpoint, reference=reference, cache_dir=args.cache_dir)
        except Exception as error:
            current["computation_complete"] = False
            current["failure"] = {"type": type(error).__name__, "message": str(error),
                                  "scope": "partial part; completed models preserved"}
            logfile.write(f"FAILED {type(error).__name__}: {error}\n"); logfile.flush()
            print(f"FAILED {type(error).__name__}: {error}", flush=True)
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old)
            provenance["runtime_seconds"] = time.time() - start
            provenance["peak_RSS_mib"] = peak_rss_mib()
            provenance["ended_at"] = datetime.now().astimezone().isoformat()
            provenance["source_all_unchanged"] = source_hashes() == sources
            provenance["protocol_unchanged"] = sha(PROTOCOL) == protocol_hash
            current["provenance"] = provenance
            if not (provenance["source_all_unchanged"] and provenance["protocol_unchanged"]):
                current["computation_complete"] = False
                current["provenance_failure"] = "source or protocol changed"
            atomic_json(out_json, current)
    print(f"saved {out_json}; complete={current['computation_complete']}; seconds={provenance['runtime_seconds']:.1f};"
          f" peakRSS={provenance['peak_RSS_mib']:.1f}MiB", flush=True)
    return 0 if current["computation_complete"] else 1


# ----------------------------------------------------------------------------- merge / summary

def table_rows(merged):
    rows = []
    la = merged["level_A"]
    for split, rec in la.items():
        rows.append({"split": split, "arm": "levelA", "init": "fixed_reader",
                     "D2_fresh": None if rec["D2_fresh"] is None else rec["D2_fresh"]["D2_fresh"],
                     "audit_gap": rec["audit_S"]["gap"], "audit_gap_SE": rec["audit_S"]["se"],
                     "R0": rec["audit_S"]["R0_q0_brier"], "R1_reader": rec["audit_S"]["R1_reader_brier"],
                     "R1_raw": rec["audit_S"]["R1_raw_brier"],
                     "q0_outside_S": rec["audit_S"]["q0_outside_05_95"], "clipped_E": rec["clipped_fraction_E"],
                     "theta": rec["effect_E"]["candidate_estimate"], "theta_SE": rec["effect_E"]["sampling_SE"],
                     "diff_vs_levelA": 0., "diff_SE": 0., "encoder_seconds": rec["seconds"]["encoder_training"]})
    for name, rec in merged["models"].items():
        pair = rec["paired_vs_levelA_same_split"]
        rows.append({"split": f"S{rec['split']}", "arm": rec["arm"], "init": rec["init"],
                     "D2_fresh": None if rec["D2_fresh"] is None else rec["D2_fresh"]["D2_fresh"],
                     "audit_gap": rec["audit_S"]["gap"], "audit_gap_SE": rec["audit_S"]["se"],
                     "R0": rec["audit_S"]["R0_q0_brier"], "R1_reader": rec["audit_S"]["R1_reader_brier"],
                     "R1_raw": rec["audit_S"]["R1_raw_brier"],
                     "q0_outside_S": rec["audit_S"]["q0_outside_05_95"], "clipped_E": rec["clipped_fraction_E"],
                     "theta": rec["effect_E"]["candidate_estimate"], "theta_SE": rec["effect_E"]["sampling_SE"],
                     "diff_vs_levelA": pair["candidate_difference"], "diff_SE": pair["sampling_SE"],
                     "encoder_seconds": rec["seconds"]["encoder_training"],
                     "selected": rec["training"].get("selected_round", rec["training"].get("selected_step")),
                     "steps": rec["training"]["encoder_steps_total"]})
    order = {"levelA": 0, "O1": 1, "O2": 2, "O3": 3}
    rows.sort(key=lambda r: (r["split"], order[r["arm"]], r["init"]))
    return rows


def format_rows(rows):
    def f(x, spec):
        return "NA" if x is None else format(x, spec)
    head = (f"{'split':5} {'arm':6} {'init':12} {'D2_fresh':>9} {'gapS':>9} {'outS':>7} {'clipE':>7} {'theta':>8}"
            f" {'diff±SE':>17} {'enc_s':>6}")
    lines = [head]
    for r in rows:
        lines.append(f"{r['split']:5} {r['arm']:6} {r['init']:12} {f(r['D2_fresh'], '.6f'):>9} {r['audit_gap']:>9.6f}"
                     f" {r['q0_outside_S']:>7.4f} {r['clipped_E']:>7.4f} {r['theta']:>8.5f}"
                     f" {r['diff_vs_levelA']:>+8.5f}±{r['diff_SE']:.5f} {f(r['encoder_seconds'], '.0f'):>6}")
    return "\n".join(lines)


def merge(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    part_paths = sorted(args.out_dir.glob(f"{stem(args.sigma, args.seed)}.part-*.json"))
    parts = [json.loads(p.read_text()) for p in part_paths]
    complete = [(p, d) for p, d in zip(part_paths, parts) if d.get("computation_complete")]
    if not complete:
        raise SystemExit("no complete parts")
    merged = ob.merge_parts([d for _, d in complete])
    merged["parts_files"] = {p.name: sha(p) for p, _ in complete}
    merged["incomplete_parts_ignored"] = [p.name for p, d in zip(part_paths, parts) if not d.get("computation_complete")]
    merged["table"] = table_rows(merged)
    merged["provenance"] = {"merged_at": datetime.now().astimezone().isoformat(),
                            "imported_module_sha256": imported_hashes(),
                            "source_sha256_all_single_image": source_hashes(),
                            "protocol": {"path": str(PROTOCOL.relative_to(PROJECT)), "sha256": sha(PROTOCOL)},
                            "part_source_hashes_identical": len({json.dumps(d["provenance"]["source_sha256_all_single_image"],
                                                                            sort_keys=True) for _, d in complete}) == 1}
    target = args.out_dir / f"{stem(args.sigma, args.seed)}.json"
    if target.exists() and not args.replace_derived:
        raise SystemExit(f"{target} exists; pass --replace-derived to rebuild a derived merge output")
    if target.exists() and json.loads(target.read_text()).get("artifact") != "single_image_objective_v1":
        raise SystemExit(f"{target} is not a derived merge output; refusing")
    atomic_json(target, merged)
    print(format_rows(merged["table"]))
    summary(args)
    return 0


def summary(args):
    merged = {}
    for path in sorted(args.out_dir.glob("objective_sigma*_seed*.json")):
        if ".part-" in path.name:
            continue
        data = json.loads(path.read_text())
        merged[path.name] = data
    by_sigma = {}
    for name, data in merged.items():
        by_sigma.setdefault(data["config"]["sigma"], {})[data["config"]["seed"]] = data
    out = {"artifact": "single_image_objective_v1_summary", "protocol": str(PROTOCOL.relative_to(PROJECT)),
           "protocol_sha256": sha(PROTOCOL), "imported_module_sha256": imported_hashes(),
           "merged_files": {n: sha(args.out_dir / n) for n in merged}, "decision": {}, "trend": {}, "tables": {}}
    for sigma, seeds in by_sigma.items():
        table = ob.decision_table(seeds)
        key = f"sigma{sigma:g}"
        (out["decision"] if math.isclose(sigma, .3) else out["trend"])[key] = table
        out["tables"][key] = {str(seed): data["table"] for seed, data in seeds.items()}
    target = args.out_dir / "objective_summary_v1.json"
    if target.exists() and json.loads(target.read_text()).get("artifact") != "single_image_objective_v1_summary":
        raise SystemExit("summary path holds a different artifact; refusing")
    atomic_json(target, out)
    for key, table in {**out["decision"], **out["trend"]}.items():
        print(f"== {key}: seeds {table['seeds']}; level A {json.dumps(table['level_A'], default=str)}")
        for label, row in table["rows"].items():
            print(f"  {label:9} D2={row['D2_fresh_mean_S1_S4']} ratio={row['D2_ratio_to_level_A']} R1={row['R1_balance_pass']}"
                  f" | err={row['abs_error_mean_S1_S4']} SE={row['paired_SE_vs_level_A']} R2={row['R2_accuracy_pass']}"
                  f" | range={row['theta_range_mean_S1_S4']} R3={row['R3_stability_pass']} ({row['R3_status']})"
                  f" | R4={row['R4_device_pass']} gaps={row['audit_gap_by_split']}")
        print("  branches:", json.dumps(table["branches"], indent=1))
    return out


def main(argv=None):
    args = parser().parse_args(argv)
    if args.merge:
        return merge(args)
    return run(args, argv)


if __name__ == "__main__":
    raise SystemExit(main())
