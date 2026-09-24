"""Summaries for SCM-4e v2.  The scoring is summarize_scm4's code, run unchanged on SCM-4e v2's shards.

    python3 -B summarize_scm4e_v2.py dev        threshold by the sealed rule, replay, entry check
    python3 -B summarize_scm4e_v2.py confirm    the thirteen gates at the frozen threshold

Three things are handled here and nowhere else.  The pixel CNN has its own shards, so each data set's value is
joined in before scoring and a data set without one is scored as missing, never skipped.  The summary records
what the association front end kept and how much of the recorded level the kept directions carry, which is the
evaluator-only check that the embedding route handed PROBE the information at all.  And the development summary
repeats the v1 warning: a development mean absolute error above 0.05, or three estimates leaning the same way by
more than 0.03, is reported at the top even when the entry check passes, because that is what a bias looks like
before the confirmation seeds are opened.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import run_scm4e_v2 as rv
import summarize_scm4 as ss

SUMMARY = {"dev": "dev_summary_v1.json", "confirm": "confirm_summary_v1.json"}
LEAN_WARNING = {"dev_mae": 0.05, "same_side_bias": 0.03}


def merged(stage: str) -> list[dict]:
    shards = []
    for path in sorted(rv.SHARDS[stage].glob("*.json")):
        shard = json.loads(path.read_text())
        pixel_path = rv.PIXEL_SHARDS[stage] / path.name
        if pixel_path.exists():
            extra = json.loads(pixel_path.read_text())
            if extra["protocol_sha256"] != shard["protocol_sha256"] or extra["task"] != shard["task"]:
                raise SystemExit(f"{pixel_path.name} does not belong to the same protocol and task")
            shard["pixel_cnn"] = extra["pixel_cnn"]
        else:
            shard["pixel_cnn"] = {"finite": False, "ate": None, "reason": "no pixel CNN shard"}
        shard.setdefault("reading", {})           # v2 reads no angle; the table of reading errors is empty
        shards.append(shard)
    return shards


def front_end_summary(shards: list[dict]) -> dict:
    records = [rec for sh in shards for block in sh["front_end"].values() for rec in block.values()]
    kept = [rec["kept"] for rec in records]
    as_images = [rec["kept"] == rec["images"] for rec in records]
    r2 = [v for rec in records for v in rec["level_r2_from_kept_directions"]]
    ratio = [rec["canonical_correlations"][0] / rec["threshold"] for rec in records if rec["threshold"] > 0]
    return {"fits": len(records) // 5, "kept_min_max": [int(min(kept)), int(max(kept))],
            "kept_one_direction_per_image": {"yes": int(sum(as_images)), "of": len(as_images)},
            "nothing_above_threshold": int(sum(rec["nothing_above_threshold"] for rec in records)),
            "hit_the_image_cap": int(sum(rec["hit_the_image_cap"] for rec in records)),
            "level_r2_min": float(min(r2)), "level_r2_median": float(np.median(r2)),
            "first_correlation_over_threshold_min": round(float(min(ratio)), 3)}


def cevae_rows(shards: list[dict]) -> dict:
    out = {}
    for row in ("variates", "full_embedding"):
        values = [sh["cevae"][row]["ate"] for sh in shards if row in sh.get("cevae", {})]
        finite = [v for v in values if v is not None and np.isfinite(v)]
        out[row] = {"values": values, "mean": float(np.mean(finite)) if finite else None,
                    "mae": float(np.mean(np.abs(np.array(finite) - 1.0))) if finite else None}
    return out


def score(stage: str) -> int:
    out_path = rv.RESULTS / SUMMARY[stage]
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; summaries are write-once")
    shards = merged(stage)
    saved = (ss.rs, ss.load, rv.RESULTS)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            ss.rs, ss.load, rv.RESULTS = rv, (lambda _stage: shards), Path(tmp)
            code = ss.dev() if stage == "dev" else ss.confirm()
            scored = json.loads((Path(tmp) / SUMMARY[stage]).read_text())
        finally:
            ss.rs, ss.load, rv.RESULTS = saved
    scored["artifact"] = f"family_v2_scm4e_v2_{stage}_summary"
    scored["level"] = rv.LEVEL_NAME
    scored["front_end"] = front_end_summary(shards)
    scored["cevae_rows"] = cevae_rows(shards)
    scored["pixel_cnn_shards"] = sum(1 for sh in shards if sh["pixel_cnn"].get("reason") != "no pixel CNN shard")
    if stage == "dev":
        errors = [r["error"] for r in scored["replay"] if r["error"] is not None]
        mae = scored["entry_check"]["dev_mae"]
        scored["bias_warning"] = {
            "dev_mae_above": bool(mae is not None and mae > LEAN_WARNING["dev_mae"]),
            "all_lean_the_same_way": bool(errors) and (all(e > LEAN_WARNING["same_side_bias"] for e in errors)
                                                       or all(e < -LEAN_WARNING["same_side_bias"] for e in errors)),
            "errors": errors, "rule": LEAN_WARNING,
            "note": "the v1 run passed the entry check with a mean absolute error of 0.067 and then predicted a "
                    "confirmation failure; if either flag is true the confirmation is a user decision"}
        print("bias warning:", {k: v for k, v in scored["bias_warning"].items() if k in ("dev_mae_above", "all_lean_the_same_way")})
    print(f"front end: {scored['front_end']}\nCEVAE rows: "
          f"{ {k: v['mean'] for k, v in scored['cevae_rows'].items()} }")
    out_path.write_text(json.dumps(scored, indent=1, allow_nan=False) + "\n")
    print(f"written {out_path}")
    return code


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in SUMMARY:
        raise SystemExit("usage: summarize_scm4e_v2.py dev|confirm")
    raise SystemExit(score(sys.argv[1]))
