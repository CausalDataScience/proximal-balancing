"""SCM-7-v4: fresh confirmation of the image design with an image-specific screen tolerance.

Everything except the screen tolerance, the protocol id, the seeds and the arms is SCM-7-v3, imported
unchanged from run_scm7_v3.  The tolerance comes from calibrate_scm7_tolerance.py, which set it on the
failed SCM-7-v3 confirmation reclassified as development data.  No v4 seed has been seen by any step.
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import minimal_suite_common as c
import run_scm7_v3 as r7

PROTOCOL_PATH = "../results/scm7_v4/protocol.json"
SHARD_DIR = "../results/scm7_v4/shards"
CALIBRATION = "../results/scm7_v4/tolerance_calibration.json"
SOURCES = r7.SOURCES + ["calibrate_scm7_tolerance.py", "run_scm7_v4.py"]


def _frozen() -> dict:
    cal = json.loads(Path(CALIBRATION).read_text())
    rep = cal["report"]["new_tolerance"]
    f = copy.deepcopy(r7.FROZEN)
    f["protocol_id"] = "scm7-v4-image-tolerance-confirmation"
    f["experiment_id"] = "SCM-7-v4"
    f["arms"] = {"confirmation": {"n": 12000, "seeds": [95300000 + 1000 * r for r in range(15)]}}
    f["screen"]["tolerance"] = cal["tolerance"]
    f["decision_record"] = {
        "decided_by": "user, 2026-09-15: recalibrate the screen tolerance on image development data and reconfirm on "
                      "fresh data within the GPU budget",
        "development_data": "SCM-7-v3 confirmation shards (protocol 2ae2898a60ee), reclassified as development after "
                            "that confirmation failed five of thirteen frozen gates",
        "tolerance": f"{cal['tolerance']:.2e} from calibrate_scm7_tolerance.py (artifact sha256 "
                     f"{c.file_sha256(CALIBRATION)[:16]}): 95th percentile of the one-sided bound over clean per-rotation "
                     f"rows of data sets 95200000-95209000 ({cal['t0']:.3e}), inside the population window "
                     f"(2.983e-5, 7.711e-3), rounded down to two significant figures. On the held-back half "
                     f"95210000-95219000: clean pass {rep['check']['clean_pass']:.3f}, scorer TPR {rep['check']['scorer_tpr']:.3f}, "
                     f"scorer FPR {rep['check']['scorer_fpr']:.3f}; replayed four-rotation retention keeps at least two clean "
                     f"choices in {rep['replay_check']['seeds_with_at_least_2_clean']} of 10 data sets.",
        "seeds": "15 fresh data sets. The user's total GPU budget is 10 hours; 370.8 minutes were used, leaving 229.2. "
                 "Per-data-set p95 under the old tolerance was 13.1 minutes, about 0.8 more is expected for the extra "
                 "learned-target evaluations a looser tolerance admits, job overhead is 4.8 minutes, and a 10-minute "
                 "margin is kept, so 15 fit. With 15 data sets the 0.95 return-rate gate requires every data set to answer.",
        "no_qualification_arm": "the code path is SCM-7-v3's, which completed 22 data sets on this node type inside the "
                                "runtime gate (warm 669-711 s). Only the tolerance, protocol id and seeds change. The "
                                "runner stops after its first data set if recorded screen decisions disagree with the "
                                "frozen tolerance.",
        "known_risks": "the tolerance targets return rate, screen TPR and unique-largest rate. Oracle excess and the "
                       "learned-target p90 failed in v3 through representation error, which the tolerance does not "
                       "change, and a looser tolerance admits more rotations whose learned target may be further from tau.",
    }
    f["gates"] = copy.deepcopy(r7.FROZEN["gates"])
    return f


FROZEN = _frozen()


def freeze() -> dict:
    p = Path(PROTOCOL_PATH)
    if p.exists():
        return json.loads(p.read_text())
    proto = dict(FROZEN)
    proto["source_snapshot"] = c.source_snapshot(SOURCES)
    proto["mnist_sha256"] = r7.im.MNIST_SHA256
    proto["frozen_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    arm = FROZEN["arms"]["confirmation"]
    proto["runs"] = [{"experiment_id": "SCM-7-v4", "mode": "operational_rotate4", "phase": "confirmation",
                      "arm": "confirmation", "n": arm["n"], "seeds": arm["seeds"]}]
    return c.seal_protocol(p, proto)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard-dir", default=SHARD_DIR)
    args = ap.parse_args()
    proto = freeze()
    arm = FROZEN["arms"]["confirmation"]
    c.check_protocol(proto, {"experiment_id": "SCM-7-v4", "mode": "operational_rotate4", "phase": "confirmation",
                             "arm": "confirmation", "n": arm["n"], "seeds": arm["seeds"]}, PROTOCOL_PATH, SOURCES)
    tol = FROZEN["screen"]["tolerance"]
    # one_replicate reads the tolerance, the protocol id and the evaluator spec from this module global at call time
    r7.FROZEN = FROZEN
    facts = r7.s6.truth()
    gate = r7.im.exact_channel_check(n=1500, seed=2468)
    if not gate["exact_channel"]:
        raise SystemExit(f"renderer failed its information gate: {gate['gates']}")
    dev = r7.ft.device()
    Path(args.shard_dir).mkdir(parents=True, exist_ok=True)
    print(f"SCM-7-v4 | confirmation | n={arm['n']} | {len(arm['seeds'])} seeds | tolerance {tol:.2e} | device {dev} | "
          f"protocol {proto[c.DIGEST_FIELD][:12]}", flush=True)
    for seed in arm["seeds"]:
        shard = Path(args.shard_dir) / f"confirmation_seed{seed}_n{arm['n']}.json"
        if shard.exists():
            print(f"  seed {seed}: shard exists, skipping", flush=True)
            continue
        if r7.torch.cuda.is_available():
            r7.torch.cuda.reset_peak_memory_stats()
        cell = r7.one_replicate(seed, arm["n"], facts, dev, workers=args.workers)
        cell["protocol"] = {"path": PROTOCOL_PATH, c.DIGEST_FIELD: proto[c.DIGEST_FIELD]}
        cell["source_snapshot"] = c.source_snapshot(SOURCES)
        cell["screen_tolerance"] = tol
        c.write_json_new(shard, cell)
        wrong = [r["split_id"] for r in cell.get("rows", [])
                 if r["screen"]["pass"] != (r["screen"]["upper"] < tol and r["screen"]["overlap_ok"])]
        if wrong:
            raise SystemExit(f"seed {seed}: {len(wrong)} screen decisions disagree with tolerance {tol:.2e}; stopping")
        print(f"  seed {seed}: {cell['return_kind']} est={cell['estimate']} retained={len(cell.get('retained_splits', []))} "
              f"purity={cell.get('largest_component_purity')} ({cell['timing_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
