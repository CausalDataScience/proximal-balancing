"""PROBE on the WSC within-study comparison (Keller et al. 2025), seven real proxy blocks.

The data handling is the August design, imported unchanged from `wsc_population_ate_pilot.load_data`,
which checks every source file against `manifest.json`:
    2,200 people, math versus vocabulary training randomised over everyone (`mathGrp`).
    The randomised contrast over all 2,200 is the benchmark: 0.760 (se 0.151).
    The observational sample is the 1,105 whose assignment matched their stated preference, 288 of them
    treated, so within it treatment equals self-selection.
    Outcome `mathPost`; X the 14 demographic columns; W the 114 pre-treatment items.

What this file adds, fixed before it ran (2026-09-22 21:00 CDT):
    Blocks.  The 114 items split by their instrument, in the August order: Big Five, AMAS, BDI, GSES, math
        pre-test, vocabulary pre-test, pre-treatment MCS.  J = 7, so 126 held-out candidates and
        z = Phi^{-1}(1 - 0.05 / 126).
    Primary reading.  The v3 noise-scaled rule, the paper's frozen rule.  At J = 7 it is not degenerate,
        unlike at J = 2.  The significance rule and the no-rule aggregate are reported beside it.
    Samples.  The whole observational sample and the common-support band [0.05, 0.95] of a propensity
        fitted on X and all 114 items, both reported.  The band changes the population and says so.
    Comparators.  Naive, X only, X and all items, on PROBE's rotations.  Proximal 2SLS needs a role
        assignment that nobody has here, so all 42 ordered pairs of blocks are run and the whole range is
        reported.  COCA as an ATE from each of the seven blocks.  CEVAE is not run.
    Seed 99_000_000, rank k = 2 as everywhere in the paper.
    The estimand is the effect in the observational sample; the benchmark is the effect over everyone.
        Comparing them assumes the effect transports, which the caption must say.

    python3 -B wsc_probe.py
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import family_v2_competitors as fc
import park_ate as pa
import probe_j2 as pj
import run_family_v2 as rf
import run_two_proxy as rtp
import two_proxy_data as tp
import two_proxy_uncertainty as tpu
import wsc_population_ate_pilot as wp

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "wsc" / "probe_v1.json"
SEED = 99_000_000
K = 2
BAND = tp.BAND
BLOCK_ORDER = ["big5", "amas", "bdi", "gses", "math_pre", "vocab_pre", "mcs_pre"]
SOURCES = ("wsc_probe.py", "wsc_population_ate_pilot.py", "probe_j2.py", "family_v2_probe.py",
           "probe_structured_scm.py", "family_v2_competitors.py", "park_ate.py", "run_two_proxy.py",
           "two_proxy_data.py", "two_proxy_uncertainty.py")


def load_view() -> tuple[dict, dict]:
    data = wp.load_data()
    names = data["proxy_names"]
    prefix = np.array([n.split(":")[0] for n in names])
    if list(dict.fromkeys(prefix)) != BLOCK_ORDER:
        raise RuntimeError(f"block order drift: {list(dict.fromkeys(prefix))}")
    view = {"X": data["X"].astype(float), "A": data["A"].astype(float), "Y": data["Y"].astype(float)}
    for j, block in enumerate(BLOCK_ORDER):
        view[f"W{j + 1}"] = data["W"][:, prefix == block].astype(float)
    meta = {"rct_benchmark": data["rct_benchmark"], "rct_se": data["rct_standard_error"],
            "n": int(len(view["A"])), "treated": int(view["A"].sum()),
            "block_sizes": {b: int((prefix == b).sum()) for b in BLOCK_ORDER},
            "x_columns": list(wp.X_COLUMNS), "source_hashes": data["source_hashes"]}
    return view, meta


def propensity(view: dict, n_blocks: int) -> np.ndarray:
    z = np.column_stack([view["X"]] + [view[f"W{j + 1}"] for j in range(n_blocks)])
    z = (z - z.mean(0)) / (z.std(0) + 1e-12)
    return LogisticRegression(max_iter=5000).fit(z, view["A"]).predict_proba(z)[:, 1]


def restrict(view: dict, keep: np.ndarray) -> dict:
    return {k: v[keep] for k, v in view.items()}


def comparators(view: dict, seed: int, n_blocks: int) -> dict:
    out = rtp.baselines(view, seed, n_blocks)
    pairs = {}
    for zi, wi in itertools.permutations(range(1, n_blocks + 1), 2):
        try:
            pairs[f"W{zi}_treats_W{wi}_outcome"] = fc.p2sls(view, f"W{zi}", f"W{wi}")
        except np.linalg.LinAlgError as exc:               # recorded, never dropped silently
            pairs[f"W{zi}_treats_W{wi}_outcome"] = f"failed: {exc}"
    coca = {}
    for j in range(1, n_blocks + 1):
        try:
            coca[f"W{j}"] = pa.park_parts(view, f"W{j}")["ate"]
        except np.linalg.LinAlgError as exc:
            coca[f"W{j}"] = f"failed: {exc}"
    out["p2sls_all_role_assignments"] = pairs
    out["coca_ate_by_block"] = coca
    return out


def reading(view: dict, seed: int, label: str, n_blocks: int) -> dict:
    probe = pj.run(view, K, seed, n_blocks=n_blocks)
    rep = pj.propensity_report(view, seed, n_blocks)
    rep.pop("propensity")
    return {"label": label, "n": len(view["A"]), "overlap": rep,
            "probe": {k: v for k, v in probe.items() if k != "score_crossproduct"},
            "score_crossproduct": probe["score_crossproduct"],
            "comparators": comparators(view, seed, n_blocks)}


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; write-once")
    view, meta = load_view()
    n_blocks = len(BLOCK_ORDER)
    e = propensity(view, n_blocks)
    keep = (e >= BAND[0]) & (e <= BAND[1])
    banded, dropped = tp.drop_dependent_columns(restrict(view, keep), list(wp.X_COLUMNS))
    readings = {"whole_sample": reading(view, SEED, "the 1,105 whose training matched their preference",
                                        n_blocks),
                "common_support": reading(banded, SEED + 1, f"propensity in [{BAND[0]}, {BAND[1]}]", n_blocks)}
    for key, r in readings.items():
        spread = {w: tpu.spread(r, w, SEED + 50 + i) for i, w in enumerate(("noise_floor", "significance", "both"))}
        r["aggregate_uncertainty"] = spread
    payload = {"artifact": "wsc_probe", "generated_utc": rf.now(), "seed": SEED, "k": K, "band": list(BAND),
               "blocks": BLOCK_ORDER, "primary_reading": "noise_floor (v3)",
               "covariates_dropped_on_the_band": dropped, "reference": {
                   "estimand": "average effect over all 2,200, from the randomisation",
                   "estimate": meta["rct_benchmark"], "se": meta["rct_se"]},
               "meta": meta, "readings": readings,
               "transport_note": "PROBE estimates the effect in the observational sample (and, on the band, in "
                                 "the band); the benchmark is the effect over everyone randomised",
               "source_sha256": {f: rf.sha256(HERE / f) for f in SOURCES}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "x") as handle:
        handle.write(json.dumps(payload, indent=1, allow_nan=False, default=float) + "\n")

    truth = meta["rct_benchmark"]
    print(f"benchmark {truth:.3f} (se {meta['rct_se']:.3f}); n {meta['n']}, treated {meta['treated']}; "
          f"blocks {meta['block_sizes']}")
    for key, r in readings.items():
        ov, p, c = r["overlap"], r["probe"], r["comparators"]
        print(f"\n--- {key}: n = {r['n']}, treated ESS {ov['treated_ess']:.1f} of {ov['treated']}, "
              f"control ESS {ov['control_ess']:.1f} of {ov['control']}")
        for rn in ("noise_floor", "significance", "both"):
            rd, u = p["readings"][rn], r["aggregate_uncertainty"][rn]
            got = "nothing" if rd["output"] is None else (
                f"{rd['output']:+.3f} (error {rd['output'] - truth:+.3f}), sd {u['sd']:.3f}, "
                f"95% ({u['q025']:+.3f}, {u['q975']:+.3f})")
            print(f"  PROBE {rn:12s} kept {len(rd['retained']):3d} of {len(p['split_names'])} -> {got}")
        pairs = [v for v in c["p2sls_all_role_assignments"].values() if isinstance(v, float)]
        coca = [v for v in c["coca_ate_by_block"].values() if isinstance(v, float)]
        print(f"  naive {c['naive']:+.3f}  X {c['X']:+.3f}  X+all items {c['raw']:+.3f}")
        print(f"  2SLS over {len(pairs)} role assignments: min {min(pairs):+.3f}, median {np.median(pairs):+.3f}, "
              f"max {max(pairs):+.3f}")
        print(f"  COCA as ATE over 7 blocks: min {min(coca):+.3f}, median {np.median(coca):+.3f}, max {max(coca):+.3f}")
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
