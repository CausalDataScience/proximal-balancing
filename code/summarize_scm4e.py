"""Summaries for SCM-4e.  The scoring is summarize_scm4's code, run unchanged on SCM-4e's shards.

    python3 -B summarize_scm4e.py dev        threshold by the sealed rule, replay, entry check
    python3 -B summarize_scm4e.py confirm    the thirteen gates at the frozen threshold

Two things are handled here and nowhere else.  The pixel CNN has its own shards (it runs as a separate pass), so
each data set's value is joined in before scoring, and a data set without one is scored as missing, never
skipped.  And the summary records how much of each recorded level the sealed front end kept, which is the
evaluator-only check that the embedding route handed PROBE the information at all.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import run_scm4e as rse
import summarize_scm4 as ss

SUMMARY = {"dev": "dev_summary_v1.json", "confirm": "confirm_summary_v1.json"}


def merged(stage: str) -> list[dict]:
    shards = []
    for path in sorted(rse.SHARDS[stage].glob("*.json")):
        shard = json.loads(path.read_text())
        pixel_path = rse.PIXEL_SHARDS[stage] / path.name
        if pixel_path.exists():
            extra = json.loads(pixel_path.read_text())
            if extra["protocol_sha256"] != shard["protocol_sha256"] or extra["task"] != shard["task"]:
                raise SystemExit(f"{pixel_path.name} does not belong to the same protocol and task")
            shard["pixel_cnn"] = extra["pixel_cnn"]
        else:
            shard["pixel_cnn"] = {"finite": False, "ate": None, "reason": "no pixel CNN shard"}
        shard.setdefault("reading", {})                     # SCM-4e reads no angle; the table of reading errors is empty
        shards.append(shard)
    return shards


def embedding_summary(shards: list[dict]) -> dict:
    kept = [v["components_kept"] for sh in shards for v in sh["embedding_check"].values()]
    r2 = [x for sh in shards for v in sh["embedding_check"].values() for x in v["level_r2_from_kept_components"]]
    return {"components_kept_min_max": [int(min(kept)), int(max(kept))],
            "level_r2_min": float(min(r2)), "level_r2_median": float(np.median(r2))}


def score(stage: str) -> int:
    out_path = rse.RESULTS / SUMMARY[stage]
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; summaries are write-once")
    shards = merged(stage)
    saved = (ss.rs, ss.load, rse.RESULTS)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            ss.rs, ss.load, rse.RESULTS = rse, (lambda _stage: shards), Path(tmp)
            code = ss.dev() if stage == "dev" else ss.confirm()
            scored = json.loads((Path(tmp) / SUMMARY[stage]).read_text())
        finally:
            ss.rs, ss.load, rse.RESULTS = saved
    scored["artifact"] = f"family_v2_scm4e_{stage}_summary"
    scored["level"] = rse.LEVEL_NAME
    scored["embedding_check"] = embedding_summary(shards)
    scored["pixel_cnn_shards"] = sum(1 for sh in shards if sh["pixel_cnn"].get("reason") != "no pixel CNN shard")
    out_path.write_text(json.dumps(scored, indent=1, allow_nan=False) + "\n")
    print(f"embedding check: {scored['embedding_check']}\nwritten {out_path}")
    return code


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in SUMMARY:
        raise SystemExit("usage: summarize_scm4e.py dev|confirm")
    raise SystemExit(score(sys.argv[1]))
