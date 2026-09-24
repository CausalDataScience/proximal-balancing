"""SCM-4 stages up to the connection check (spec memo/2026-09-20-scm4-shapes3d-reader-spec-v1.md).

    python3 -B run_scm4_images.py prepare         # verify the download, build the one cache, write the manifest
    python3 -B run_scm4_images.py freeze-reader   # train, choose on reader_selection, freeze, read the bank once
    python3 -B run_scm4_images.py connect         # three test seeds: exact levels vs the reader's numbers
    python3 -B run_scm4_images.py connect2 --seed-index 0|1|2   # version 2 (spec section 10), one seed per call
    python3 -B run_scm4_images.py connect2-summary               # the rank test, once all three seeds are recorded

Nothing here opens a development or confirmation seed.  Each stage writes its result once and refuses to
overwrite it, and `connect` refuses to start unless the frozen reader passed its final check on the causal bank
and the files it was checked against are still the ones on disk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import family_v2_dgp as g
import family_v2_images as im
import family_v2_probe as pr

LEVEL = g.LEVELS["SCM-1"]
TOLERANCE = 1.5e-3                                   # the sealed SCM-1 threshold
CONNECT_SEEDS = [g.TEST_SEED + 4000, g.TEST_SEED + 5000, g.TEST_SEED + 6000]
LIMIT = {"truth_error": 0.10, "input_difference": 0.02, "splits_with_different_verdict": 1}
RESULT_DIR = im.ROOT / "results" / "family_v2" / "scm4_connect"
# connection check v2 (spec section 10): fixed before its seeds were opened
CONNECT2_SEEDS = [g.TEST_SEED + 7000, g.TEST_SEED + 8000, g.TEST_SEED + 9000]
NOISE_DRAWS = 5
RANK_SUM_REJECT = 16                                 # P(rank sum >= 16) = 10/216 when the reader behaves like noise
RESULT2_DIR = im.ROOT / "results" / "family_v2" / "scm4_connect_v2"
EXPECTED_SPLITS = {pr.split_name(s) for s in pr.all_splits()}
REQUIRED_ROTATION_KEYS = ("rotation", "gap_fit", "screen", "theta")
REQUIRED_SCREEN_KEYS = ("gap", "se", "upper", "overlap", "pass")


# ----------------------------------------------------------------------------- stage 1

def cmd_prepare(_: argparse.Namespace) -> int:
    manifest = im.prepare()
    print(json.dumps({k: manifest[k] for k in ("h5", "cache_bytes", "split", "seconds", "free_gib_after")}, indent=1))
    return 0


# ----------------------------------------------------------------------------- stage 2

def cmd_freeze_reader(_: argparse.Namespace) -> int:
    import torch

    im.check_interpreter()
    weights, choices = im.BANK_DIR / "reader_frozen.pt", im.BANK_DIR / "reader_frozen.json"
    final_path = im.BANK_DIR / "reader_final_check.json"
    for path in (weights, choices, final_path):
        if path.exists():
            raise SystemExit(f"{path.name} exists; the frozen reader is write-once")
    images = im.load_images()
    rows = np.load(im.BANK_DIR / "split_rows.npz")
    levels = im.factor_index(np.arange(im.N_IMAGES))[:, im.ORIENTATION]
    dev = im.device()
    net, fit_info = im.fit(images, levels, rows["reader_train"], rows["reader_selection"], dev)

    excluded: dict = {}
    selection = im.verification_table(net, images, rows["reader_selection"], dev, excluded)
    if not im.gate(selection)["pass"]:
        excluded = im.propose_exclusion(selection) or {}
        repaired = im.verification_table(net, images, rows["reader_selection"], dev, excluded) if excluded else None
        if not excluded or not im.gate(repaired)["pass"]:
            im.write_json_atomic(im.BANK_DIR / "reader_selection_FAILED.json",
                                 {"table": selection, "gate": im.gate(selection), "fit": fit_info})
            raise SystemExit("STOP: the selection gate fails and no allowed exclusion repairs it")
        selection = repaired

    torch.save(net.state_dict(), weights)             # frozen before the causal bank is touched
    im.write_json_atomic(choices, {"seed": im.READER_SEED, "fit": fit_info, "excluded": excluded,
                                   "selection_table": selection, "selection_gate": im.gate(selection),
                                   "device": str(dev)})
    final = im.verification_table(net, images, rows["causal_bank"], dev, excluded)   # the one read of the bank
    verdict = im.gate(final)
    im.write_json_atomic(final_path, {"table": final, "gate": verdict, "excluded": excluded,
                                      "digests": current_digests(), "checked_utc": utc()})
    print(json.dumps({"fit": fit_info, "excluded": excluded, "final_overall": final["overall"],
                      "final_pass": verdict["pass"], "failing": verdict["failing"][:5]}, indent=1))
    return 0 if verdict["pass"] else 1


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def current_digests() -> dict:
    """What the final bank check was performed against: the reader, the split, the bank and this code."""
    return {name: im.file_sha256(im.BANK_DIR / name) for name in
            ("reader_frozen.pt", "reader_frozen.json", "split_rows.npz", "manifest.json")} | {
        "family_v2_images.py": im.file_sha256(Path(__file__).resolve().parent / "family_v2_images.py"),
        "family_v2_probe.py": im.file_sha256(Path(__file__).resolve().parent / "family_v2_probe.py"),
        "family_v2_dgp.py": im.file_sha256(Path(__file__).resolve().parent / "family_v2_dgp.py")}


def load_frozen_reader():
    """Refuse to go on unless the causal bank check exists, passed, and covers exactly these files."""
    import torch

    final_path = im.BANK_DIR / "reader_final_check.json"
    if not final_path.exists():
        raise SystemExit("STOP: the reader has no final check on the causal bank; run freeze-reader first")
    final = json.loads(final_path.read_text())
    if not final.get("gate", {}).get("pass"):
        raise SystemExit(f"STOP: the reader failed its final check: {final.get('gate', {}).get('failing')}")
    now, then = current_digests(), final.get("digests", {})
    changed = {k: (then.get(k), v) for k, v in now.items() if then.get(k) != v}
    if changed:
        raise SystemExit(f"STOP: files changed since the final check: {sorted(changed)}")
    dev = im.device()
    net = im.build().to(dev)
    net.load_state_dict(torch.load(im.BANK_DIR / "reader_frozen.pt", map_location=dev))
    return net.eval(), dev, final.get("excluded", {}), final


# ----------------------------------------------------------------------------- stage 3

def validate(out: dict) -> list[str]:
    """Every reason a run does not count as a complete, finite, single return.  Missing fields are reported,
    never raised, and the split and rotation identities are checked, not just their counts."""
    problems = []
    if not isinstance(out, dict):
        return ["result is not a record"]
    names = out.get("split_names")
    records = out.get("records")
    if not isinstance(names, list) or not isinstance(records, dict):
        return ["no split records"]
    if set(names) != EXPECTED_SPLITS or len(names) != len(EXPECTED_SPLITS):
        problems.append(f"split ids are not the 30 expected ones: {sorted(set(names) ^ EXPECTED_SPLITS)}")
    for name in EXPECTED_SPLITS:
        rec = records.get(name)
        if not isinstance(rec, dict):
            problems.append(f"{name}: missing record")
            continue
        rots = rec.get("rotations")
        if not isinstance(rots, list) or {r.get("rotation") for r in rots} != {0, 1, 2, 3} or len(rots) != 4:
            problems.append(f"{name}: rotations are not 0-3 exactly")
            continue
        for r in rots:
            missing = [k for k in REQUIRED_ROTATION_KEYS if k not in r]
            screen = r.get("screen") if isinstance(r.get("screen"), dict) else {}
            missing += [f"screen.{k}" for k in REQUIRED_SCREEN_KEYS if k not in screen]
            if missing:
                problems.append(f"{name} rotation {r.get('rotation')}: missing {missing}")
                continue
            numbers = [r["theta"], r["gap_fit"], screen["gap"], screen["se"], screen["upper"]]
            if not np.all(np.isfinite(np.array(numbers, dtype=float))):
                problems.append(f"{name} rotation {r.get('rotation')}: non-finite value")
            if screen["se"] < 0:
                problems.append(f"{name} rotation {r.get('rotation')}: negative standard error")
        est = rec.get("estimate")
        if est is None or not np.isfinite(est):
            problems.append(f"{name}: non-finite candidate estimate")
    if out.get("return_kind") != "unique_largest":
        problems.append(f"return kind {out.get('return_kind')}")
    est = out.get("estimate")
    if est is None or not np.isfinite(est):
        problems.append("no finite estimate")
    return problems


def run_one(view: dict, seed: int) -> dict:
    """One pass of the sealed pipeline.  A crash, a missing field or a non-finite number is a recorded failure."""
    t0 = time.time()
    try:
        out = pr.run_dataset(view, LEVEL, TOLERANCE, seed + 9)
        problems = validate(out)
    except Exception:
        return {"valid": False, "problems": ["exception"], "traceback": traceback.format_exc(),
                "seconds": round(time.time() - t0)}
    return {"valid": not problems, "problems": problems, "estimate": out.get("estimate"),
            "return_kind": out.get("return_kind"), "retained": sorted(out.get("retained", [])),
            "component_sizes": out.get("component_sizes"), "rho": out.get("rho"),
            "records": out.get("records"),                     # every split and rotation is kept for later reading
            "seconds": round(time.time() - t0)}


def verdict(exact: dict, cnn: dict) -> dict:
    reasons = [f"{tag}: {p}" for tag, r in (("exact", exact), ("cnn", cnn)) for p in r["problems"]]
    if not reasons:
        for tag, r in (("exact", exact), ("cnn", cnn)):
            if abs(r["estimate"] - g.TAU) > LIMIT["truth_error"]:
                reasons.append(f"{tag} input misses the truth by {abs(r['estimate'] - g.TAU):.3f}")
        if abs(cnn["estimate"] - exact["estimate"]) > LIMIT["input_difference"]:
            reasons.append(f"inputs differ by {abs(cnn['estimate'] - exact['estimate']):.3f}")
        differing = sorted(set(exact["retained"]) ^ set(cnn["retained"]))
        if len(differing) > LIMIT["splits_with_different_verdict"]:
            reasons.append(f"screen verdict differs on {differing}")
    return {"pass": not reasons, "reasons": reasons}


def build_views(data: dict, net, images, dev, groups, seed: int) -> tuple[dict, dict, dict]:
    """The same causal data seen twice: with the exact levels, and with what the reader makes of the images."""
    exact_view, cnn_view, reading_stats = g.learner_view(data), g.learner_view(data), {}
    streams, used = np.random.SeedSequence(seed + 78).spawn(6), 0
    for key, sd in im.population_sd(LEVEL).items():
        levels = im.record(data[key], sd)
        src = np.column_stack([im.draw_images(levels[:, c], groups, np.random.default_rng(streams[used + c]))
                               for c in range(levels.shape[1])])
        used += levels.shape[1]
        read = im.reading(net, images, src.ravel(), dev).reshape(src.shape)
        finite = np.isfinite(read)
        hard = np.where(finite, np.clip(np.rint(np.nan_to_num(read)), 0, im.LEVELS - 1), -99).astype(np.int64)
        err = hard - levels
        reading_stats[key] = {"n": int(read.size), "nonfinite": int((~finite).sum()),
                              "adjacent": int((finite & (np.abs(err) == 1)).sum()),
                              "far": int((finite & (np.abs(err) >= 2)).sum())}
        exact_view[key], cnn_view[key] = levels.astype(float), read
    return exact_view, cnn_view, reading_stats


def cmd_connect(args: argparse.Namespace) -> int:
    im.check_interpreter()
    net, dev, excluded, final = load_frozen_reader()
    images = im.load_images()
    groups = im.bank_by_level(np.load(im.BANK_DIR / "split_rows.npz")["causal_bank"], excluded)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows, failures = [], 0
    for seed in CONNECT_SEEDS:
        shard = RESULT_DIR / f"seed{seed}.json"
        if shard.exists():
            rows.append(json.loads(shard.read_text()))
            print(f"seed {seed}: already recorded", flush=True)
            continue
        t0 = time.time()
        try:
            data = g.generate(LEVEL, args.n, seed)
            exact_view, cnn_view, reading_stats = build_views(data, net, images, dev, groups, seed)
            exact, cnn = run_one(exact_view, seed), run_one(cnn_view, seed)
            row = {"seed": seed, "n": args.n, "exact": exact, "cnn": cnn, "reading": reading_stats,
                   "verdict": verdict(exact, cnn)}
        except Exception:                                      # an image or bookkeeping failure is still a result
            row = {"seed": seed, "n": args.n, "verdict": {"pass": False, "reasons": ["exception"]},
                   "traceback": traceback.format_exc()}
        row |= {"seconds": round(time.time() - t0), "recorded_utc": utc()}
        im.write_json_atomic(shard, row)                        # written per seed, before the next one starts
        failures += not row["verdict"]["pass"]
        print(json.dumps({"seed": seed, **row["verdict"], "seconds": row["seconds"],
                          "estimates": [row.get("exact", {}).get("estimate"), row.get("cnn", {}).get("estimate")]}),
              flush=True)
        rows.append(row)
    summary = {"stage": "scm4_connect", "limits": LIMIT, "tolerance": TOLERANCE, "n": args.n,
               "seeds": CONNECT_SEEDS, "excluded": excluded, "reader_final_overall": final["table"]["overall"],
               "pass": all(r["verdict"]["pass"] for r in rows),
               "rows": [{k: v for k, v in r.items() if k != "traceback"} for r in rows],
               "digests": current_digests(), "generated_utc": utc()}
    for row in summary["rows"]:                                 # the summary keeps the verdicts, shards keep detail
        for tag in ("exact", "cnn"):
            if isinstance(row.get(tag), dict):
                row[tag] = {k: v for k, v in row[tag].items() if k != "records"}
    im.write_json_atomic(RESULT_DIR / "summary.json", summary)
    print(json.dumps({"pass": summary["pass"], "failures": failures}, indent=1))
    return 0 if summary["pass"] else 1


# ----------------------------------------------------------------------------- stage 3, version 2

def noise_sd_from_final_check(final: dict) -> float:
    """The noise reference has the frozen reader's own error size on the causal bank."""
    sd = float(final["table"]["overall"]["rms"])
    if not np.isfinite(sd) or sd <= 0:
        raise SystemExit("STOP: the final check holds no usable rms")
    return sd


def noisy_view(exact_view: dict, levels: dict, sd: float, seed: int, draw: int) -> dict:
    """Exact levels plus independent Gaussian noise, clipped to the scale exactly as a reading is."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4242, draw]))
    view = dict(exact_view)
    for key, lv in levels.items():
        view[key] = np.clip(lv.astype(float) + sd * rng.standard_normal(lv.shape), 0.0, im.LEVELS - 1.0)
    return view


def seed_conditions(runs: dict) -> dict:
    """The per-seed conditions of spec section 10, and the six absolute differences the rank test needs."""
    reasons = [f"{tag}: {p}" for tag, r in runs.items() for p in r["problems"]]
    out = {"pass": False, "reasons": reasons}
    if reasons:
        return out
    exact, cnn = runs["exact"], runs["cnn"]
    for tag in ("exact", "cnn"):
        if abs(runs[tag]["estimate"] - g.TAU) > LIMIT["truth_error"]:
            reasons.append(f"{tag} input misses the truth by {abs(runs[tag]['estimate'] - g.TAU):.3f}")
    differing = sorted(set(exact["retained"]) ^ set(cnn["retained"]))
    if len(differing) > LIMIT["splits_with_different_verdict"]:
        reasons.append(f"screen verdict differs on {differing}")
    diffs = {tag: abs(r["estimate"] - exact["estimate"]) for tag, r in runs.items() if tag != "exact"}
    return {"pass": not reasons, "reasons": reasons, "abs_diff_from_exact": diffs}


def rank_verdict(per_seed_diffs: list[dict]) -> dict:
    """Rank of the reader's difference among the six per seed (1 smallest, ties averaged), summed over seeds."""
    from scipy.stats import rankdata

    ranks = []
    for diffs in per_seed_diffs:
        tags = sorted(diffs)
        if "cnn" not in tags or len(tags) != NOISE_DRAWS + 1:
            return {"pass": False, "reason": f"a seed has {tags} instead of cnn plus {NOISE_DRAWS} noise draws"}
        values = np.array([diffs[t] for t in tags], dtype=float)
        if not np.all(np.isfinite(values)):
            return {"pass": False, "reason": "non-finite difference"}
        ranks.append(float(rankdata(values, method="average")[tags.index("cnn")]))
    total = float(sum(ranks))
    return {"pass": total < RANK_SUM_REJECT, "reader_ranks": ranks, "rank_sum": total,
            "reject_at": RANK_SUM_REJECT, "null_probability_of_rejection": "10/216"}


def cmd_connect2(args: argparse.Namespace) -> int:
    im.check_interpreter()
    seed = CONNECT2_SEEDS[args.seed_index]
    RESULT2_DIR.mkdir(parents=True, exist_ok=True)
    shard = RESULT2_DIR / f"seed{seed}.json"
    if shard.exists():
        raise SystemExit(f"{shard.name} exists; results are write-once")
    net, dev, excluded, final = load_frozen_reader()
    sd = noise_sd_from_final_check(final)
    images = im.load_images()
    groups = im.bank_by_level(np.load(im.BANK_DIR / "split_rows.npz")["causal_bank"], excluded)
    t0 = time.time()
    try:
        data = g.generate(LEVEL, args.n, seed)
        exact_view, cnn_view, reading_stats = build_views(data, net, images, dev, groups, seed)
        levels = {k: im.record(data[k], s) for k, s in im.population_sd(LEVEL).items()}
        runs = {"exact": run_one(exact_view, seed), "cnn": run_one(cnn_view, seed)}
        for k in range(1, NOISE_DRAWS + 1):
            runs[f"noise_{k}"] = run_one(noisy_view(exact_view, levels, sd, seed, k), seed)
            print(f"seed {seed}: noise draw {k} done", flush=True)
        row = {"seed": seed, "n": args.n, "noise_sd": sd, "runs": runs, "reading": reading_stats,
               "conditions": seed_conditions(runs)}
    except Exception:
        row = {"seed": seed, "n": args.n, "noise_sd": sd, "conditions": {"pass": False, "reasons": ["exception"]},
               "traceback": traceback.format_exc()}
    row |= {"seconds": round(time.time() - t0), "recorded_utc": utc(), "digests": current_digests()}
    im.write_json_atomic(shard, row)
    print(json.dumps({"seed": seed, **{k: v for k, v in row["conditions"].items()},
                      "estimates": {t: r.get("estimate") for t, r in row.get("runs", {}).items()}}), flush=True)
    return 0 if row["conditions"]["pass"] else 1


def cmd_connect2_summary(_: argparse.Namespace) -> int:
    out_path = RESULT2_DIR / "summary.json"
    if out_path.exists():
        raise SystemExit("summary.json exists; results are write-once")
    rows = []
    for seed in CONNECT2_SEEDS:
        shard = RESULT2_DIR / f"seed{seed}.json"
        if not shard.exists():
            raise SystemExit(f"STOP: {shard.name} is missing; every seed must be recorded before the verdict")
        rows.append(json.loads(shard.read_text()))
    conditions_ok = all(r["conditions"]["pass"] for r in rows)
    rank = (rank_verdict([r["conditions"]["abs_diff_from_exact"] for r in rows]) if conditions_ok
            else {"pass": False, "reason": "a seed failed its own conditions, so no rank test is made"})
    summary = {"stage": "scm4_connect_v2", "rule": "spec section 10", "seeds": CONNECT2_SEEDS,
               "limits": LIMIT, "noise_draws": NOISE_DRAWS, "noise_sd": rows[0].get("noise_sd"),
               "per_seed": [{"seed": r["seed"], "conditions": r["conditions"], "reading": r.get("reading"),
                             "estimates": {t: x.get("estimate") for t, x in r.get("runs", {}).items()},
                             "seconds": r["seconds"]} for r in rows],
               "rank_test": rank, "pass": bool(conditions_ok and rank["pass"]), "generated_utc": utc()}
    im.write_json_atomic(out_path, summary)
    print(json.dumps({k: summary[k] for k in ("pass", "rank_test")}, indent=1))
    return 0 if summary["pass"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare").set_defaults(func=cmd_prepare)
    sub.add_parser("freeze-reader").set_defaults(func=cmd_freeze_reader)
    connect = sub.add_parser("connect")
    connect.add_argument("--n", type=int, default=24_000)
    connect.set_defaults(func=cmd_connect)
    connect2 = sub.add_parser("connect2")
    connect2.add_argument("--seed-index", type=int, required=True, choices=range(len(CONNECT2_SEEDS)))
    connect2.add_argument("--n", type=int, default=24_000)
    connect2.set_defaults(func=cmd_connect2)
    sub.add_parser("connect2-summary").set_defaults(func=cmd_connect2_summary)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
