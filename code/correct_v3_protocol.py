"""Correct one manifest field of the frozen v3 protocol, and prove the correction changes nothing else.

The first freeze recorded, for the planted-RHC manifest, the file list of the Twins runner rather than the
list that `run_rhc_planted` stamps into its own shards.  The integrity check therefore compared the wrong
names and refused to score ten complete, sound shards.  No source file had changed: every file common to
the two lists carried an identical hash.

This writes v3_confirmation_protocol_v2.json, which differs from v1 in exactly one place, and asserts that
the rule, the gates, the reference values, the replicate identifiers, the seeds and the benchmark input
hashes are byte-identical.  The original file is left as it was.

    python3 -B correct_v3_protocol.py
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import freeze_v3_protocol as fz
import run_family_v2 as rf

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
V1, V2 = RESULTS / "v3_confirmation_protocol.json", RESULTS / "v3_confirmation_protocol_v2.json"
UNCHANGED = ("frozen_utc", "prd", "protocol_sha256_of_prd", "rule", "z", "gates", "reference_file",
             "acic", "source_sha256", "verified_before_freezing")


def main() -> int:
    if V2.exists():
        raise SystemExit(f"{V2} exists; write-once")
    v1 = json.loads(V1.read_text())
    v2 = copy.deepcopy(v1)

    dataset = "rhc_planted"
    correct = [k for k in fz.RUNNER_SOURCES[dataset] if k in v1["source_sha256"]]
    before = v1["manifests"][dataset]["expected"]["11"]["source_sha256"]
    for r, entry in v2["manifests"][dataset]["expected"].items():
        entry["source_sha256"] = {k: v1["source_sha256"][k] for k in correct}
    after = v2["manifests"][dataset]["expected"]["11"]["source_sha256"]

    # the correction must not touch anything else
    for key in UNCHANGED:
        assert json.dumps(v1[key], sort_keys=True) == json.dumps(v2[key], sort_keys=True), key
    for name, m1 in v1["manifests"].items():
        m2 = v2["manifests"][name]
        for key in ("replicates", "seed_base", "fold_seed_offset", "x_coefficient_seed",
                    "reference_conditional_mae", "reference_replicates", "gate_limit",
                    "gate_return_minimum", "tpr_fpr_available", "benchmark_input_sha256"):
            assert json.dumps(m1[key], sort_keys=True) == json.dumps(m2[key], sort_keys=True), (name, key)
        for r in m1["expected"]:
            assert m1["expected"][r]["seed"] == m2["expected"][r]["seed"], (name, r)
        if name != dataset:
            assert json.dumps(m1["expected"], sort_keys=True) == json.dumps(m2["expected"], sort_keys=True), name

    # and every hash that appears in both lists must already have agreed
    shared = sorted(set(before) & set(after))
    disagreeing = [k for k in shared if before[k] != after[k]]
    assert not disagreeing, disagreeing

    v2["correction"] = {
        "corrected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "supersedes": V1.name, "supersedes_sha256": rf.sha256(V1),
        "what": "the planted-RHC manifest recorded the Twins runner's source list instead of the list that "
                "run_rhc_planted stamps into its shards",
        "why_it_is_neutral": "no source file changed: every file present in both lists carried an identical "
                             "hash; the rule, both gates, the reference values, the replicate identifiers, "
                             "the seeds and the benchmark input hashes are byte-identical to v1",
        "expected_before": sorted(before), "expected_after": sorted(after),
        "removed": sorted(set(before) - set(after)), "added": sorted(set(after) - set(before)),
        "shared_files_with_identical_hashes": shared,
        "affects": "the integrity comparison for the planted-RHC shards only; IHDP was already scored under "
                   "v1 with its own correct list and is unaffected, and Twins had not been run",
    }
    with open(V2, "x") as handle:
        handle.write(json.dumps(v2, indent=1, allow_nan=False) + "\n")
    c = v2["correction"]
    print(f"written {V2}")
    print(f"  removed from the expectation: {c['removed']}")
    print(f"  added to the expectation    : {c['added']}")
    print(f"  shared files, hashes agreed : {len(shared)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
