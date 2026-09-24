"""Amendment of the SCM-4 confirmation (spec section 13): re-execute the pixel CNN, alone, on the data sets where
it was cut off by its wall-clock cap, and score the confirmation a second time with those values filled in.

    python3 -B rerun_scm4_pixel_cnn.py rerun     re-execute the comparator where the sealed shard says "time cap"
    python3 -B rerun_scm4_pixel_cnn.py amend     write confirm_summary_v1_amended.json next to the sealed summary

Nothing sealed is edited: the shards, the protocol and confirm_summary_v1.json stay byte for byte as they are.
A data set is eligible only if its sealed shard holds no pixel CNN value AND the recorded reason is the time cap.
The data, the six images and the fold seed are regenerated from the same data seed, the configuration is the
sealed one, and the regenerated data are checked against the sealed shard before the comparator runs.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import family_v2_dgp as g
import family_v2_images as im
import run_scm4 as rs

RERUN_DIR = rs.RESULTS / "confirm_pixel_rerun"
SEALED_SUMMARY = rs.RESULTS / "confirm_summary_v1.json"
AMENDED_SUMMARY = rs.RESULTS / "confirm_summary_v1_amended.json"
BASELINE_KEYS = ("X", "raw", "oracle", "naive")
BASELINE_TOL = 1e-6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible(shard: dict) -> bool:
    """Only a value the time cap cut off may be re-executed; a recorded value is never replaced."""
    p = shard["pixel_cnn"]
    return (not p.get("finite")) and p.get("ate") is None and "time cap" in str(p.get("reason", ""))


def sealed_shards() -> dict[int, tuple[Path, dict]]:
    out = {}
    for path in sorted(rs.SHARDS["confirm"].glob("*.json")):
        shard = json.loads(path.read_text())
        out[shard["task"]["seed"]] = (path, shard)
    return out


def rerun() -> int:
    import family_v2_competitors as fc
    import run_scm4_images as ri
    import scm4_pixel_cnn as px

    proto = rs.check_protocol("confirm")                       # sources, design, bank and reader are still the sealed ones
    proto_sha = sha256(rs.PROTOCOLS["confirm"])
    todo = [(seed, path, sh) for seed, (path, sh) in sealed_shards().items() if eligible(sh)]
    print(f"eligible data sets: {[seed for seed, _, _ in todo]}", flush=True)
    net, dev, excluded, _ = ri.load_frozen_reader()
    images = im.load_images()
    groups = im.bank_by_level(np.load(im.BANK_DIR / "split_rows.npz")["causal_bank"], excluded)
    RERUN_DIR.mkdir(parents=True, exist_ok=True)
    for seed, path, sealed in todo:
        out_path = RERUN_DIR / path.name
        if out_path.exists():
            print(f"seed {seed}: already re-executed", flush=True)
            continue
        if sealed["protocol_sha256"] != proto_sha:
            raise SystemExit(f"seed {seed}: the sealed shard belongs to another protocol")
        t0 = time.time()
        fold_seed = seed + rs.FOLD_SEED_OFFSET
        data = g.generate(g.LEVELS[rs.BASE_LEVEL], rs.N, seed)
        view, data4, src, reading = rs.image_view(data, net, images, dev, groups, seed)
        base = fc.aipw_baselines(data4, fold_seed)
        gaps = {k: abs(base[k] - sealed["baselines"][k]) for k in BASELINE_KEYS}
        if max(gaps.values()) > BASELINE_TOL or reading != sealed["reading"]:
            raise SystemExit(f"seed {seed}: the regenerated data differ from the sealed shard: {gaps}")
        reader_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        result = px.estimate(images, src, view["X"], view["A"], view["Y"], reader_state, fold_seed, dev)
        payload = {"artifact": "scm4_confirm_pixel_cnn_rerun", "seed": seed, "protocol_sha256": proto_sha,
                   "sealed_shard": {"file": path.name, "sha256": sha256(path),
                                    "pixel_cnn_as_sealed": sealed["pixel_cnn"].get("reason")},
                   "regenerated_data_check": {"baseline_abs_gaps": gaps, "tolerance": BASELINE_TOL,
                                              "reading_identical": True},
                   "execution": "alone, no other task sharing the GPU; configuration and seeds as sealed",
                   "pixel_cnn": result, "seconds": round(time.time() - t0, 1),
                   "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
        tmp.rename(out_path)
        print(f"seed {seed}: pixel CNN {result.get('ate')} finite {result.get('finite')} in {payload['seconds']} s "
              f"| per rotation {[round(r['theta'], 3) for r in result.get('rotations', []) if r.get('finite')]}", flush=True)
    return 0


def merged_shards() -> tuple[list[dict], list[dict]]:
    """The sealed shards with the re-executed value filled in where, and only where, the amendment allows it."""
    proto_sha = sha256(rs.PROTOCOLS["confirm"])
    shards, log = [], []
    for seed, (path, shard) in sealed_shards().items():
        rerun_path = RERUN_DIR / path.name
        if rerun_path.exists():
            extra = json.loads(rerun_path.read_text())
            if not eligible(shard):
                raise SystemExit(f"seed {seed}: a re-executed value exists for a data set that already had one")
            if extra["protocol_sha256"] != proto_sha or extra["sealed_shard"]["sha256"] != sha256(path):
                raise SystemExit(f"seed {seed}: the re-executed value does not belong to this sealed shard")
            shard = dict(shard, pixel_cnn=extra["pixel_cnn"])
            log.append({"seed": seed, "rerun_file": rerun_path.name, "rerun_sha256": sha256(rerun_path),
                        "value": extra["pixel_cnn"].get("ate"), "seconds": extra["seconds"]})
        shards.append(shard)
    return shards, log


def amend() -> int:
    import summarize_scm4 as ss
    if AMENDED_SUMMARY.exists():
        raise SystemExit(f"{AMENDED_SUMMARY} exists; summaries are write-once")
    shards, log = merged_shards()
    if not log:
        raise SystemExit("no re-executed value found; run `rerun` first")
    saved_load, saved_results = ss.load, rs.RESULTS
    with tempfile.TemporaryDirectory() as tmp:                 # the sealed scoring code, pointed at a scratch output
        try:
            ss.load = lambda stage: shards
            rs.RESULTS = Path(tmp)
            ss.confirm()
            scored = json.loads((Path(tmp) / "confirm_summary_v1.json").read_text())
        finally:
            ss.load, rs.RESULTS = saved_load, saved_results
    scored["artifact"] = "family_v2_scm4_confirm_summary_AMENDED"
    scored["amendment"] = {
        "what": "the pixel CNN was re-executed alone, with the sealed configuration and seeds, on the data sets where "
                "the sealed run recorded no value because its wall-clock cap was reached while three tasks shared the GPU",
        "spec": "memo/2026-09-20-scm4-shapes3d-reader-spec-v1.md section 13",
        "filled": log, "sealed_summary_sha256": sha256(SEALED_SUMMARY),
        "sealed_verdict": json.loads(SEALED_SUMMARY.read_text())["verdict"],
        "unchanged": "every sealed shard, the sealed protocol and the sealed summary"}
    AMENDED_SUMMARY.write_text(json.dumps(scored, indent=1, allow_nan=False) + "\n")
    print(f"written {AMENDED_SUMMARY}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("rerun", "amend"):
        raise SystemExit("usage: rerun_scm4_pixel_cnn.py rerun|amend")
    raise SystemExit(rerun() if sys.argv[1] == "rerun" else amend())
