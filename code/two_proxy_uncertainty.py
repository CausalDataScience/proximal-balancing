"""Sampling spread of what PROBE reports on the two-proxy data sets, by the same resampling as RHC.

The reported number is the median of the largest agreeing group, not any one candidate's estimate, so its
uncertainty is not one candidate's AIPW standard error.  This resamples the retained candidates from
N(observed, Sigma) with Sigma the stored score cross-product over n squared, re-applies the sealed linking
and largest-component median to every draw, and reports the spread.  It holds the retained set and the
linking radius at their observed values, so it does not carry the uncertainty of the screen itself.

    python3 -B two_proxy_uncertainty.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import probe_j2 as pj
import probe_structured_scm as ps
import run_family_v2 as rf

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "two_proxy"
OUT = RESULTS / "uncertainty_v1.json"
DRAWS = 20_000
SEED = 98_900_000


def spread(reading: dict, which: str, seed: int) -> dict | None:
    p = reading["probe"]
    rd = p["readings"][which]
    if rd["output"] is None:
        return {"return_kind": rd["return_kind"], "output": None}
    names, splits = p["split_names"], pj.all_splits(p["n_blocks"])
    keep = [names.index(nm) for nm in rd["retained"]]
    n = reading["n"]
    cov = np.array(reading["score_crossproduct"])[np.ix_(keep, keep)] / (n * n)
    centre = np.array([p["records"][names[i]]["estimate"] for i in keep])
    sample = np.random.default_rng(seed).multivariate_normal(centre, cov, size=DRAWS)
    labels = [splits[i] for i in keep]
    outs, kinds = [], {}
    for row in sample:
        agg = ps.aggregate(labels, row, rd["rho"])
        kinds[agg["return_kind"]] = kinds.get(agg["return_kind"], 0) + 1
        if agg["output"] is not None:
            outs.append(agg["output"])
    outs = np.array(outs)
    return {"return_kind": rd["return_kind"], "output": rd["output"], "retained": rd["retained"],
            "rho": rd["rho"], "returned": int(len(outs)), "of": DRAWS, "return_kinds": kinds,
            "mean": float(outs.mean()), "sd": float(outs.std(ddof=1)),
            "q025": float(np.quantile(outs, 0.025)), "q975": float(np.quantile(outs, 0.975)),
            "per_candidate_se": {names[i]: float(np.sqrt(cov[j, j])) for j, i in enumerate(keep)}}


def main_for(names: list[str]) -> int:
    """One write-once file per data set added after uncertainty_v1.json was written."""
    for i, name in enumerate(names):
        out = RESULTS / f"uncertainty_{name}_v1.json"
        if out.exists():
            raise SystemExit(f"{out} exists; write-once")
        d = json.loads((RESULTS / f"{name}_v1.json").read_text())
        block = {key: {w: spread(r, w, SEED + 1_000 + 31 * i + 7 * k)
                       for k, w in enumerate(("significance", "noise_floor"))}
                 for key, r in d["readings"].items()}
        payload = {"artifact": "two_proxy_uncertainty", "generated_utc": rf.now(), "draws": DRAWS,
                   "seed": SEED + 1_000 + 31 * i, "dataset": name,
                   "what": "the sampling spread of the median of the largest agreeing group, conditional on "
                           "the retained set and the linking radius produced on the observed data",
                   "does_not_include": "the uncertainty of the screen", "datasets": {name: block},
                   "source_sha256": {f: rf.sha256(HERE / f) for f in
                                     ("two_proxy_uncertainty.py", "probe_j2.py", "probe_structured_scm.py")}}
        with open(out, "x") as handle:
            handle.write(json.dumps(payload, indent=1, allow_nan=False) + "\n")
        for key, b in block.items():
            for w, s_ in b.items():
                got = "no return" if s_["output"] is None else (
                    f"{s_['output']:+10.2f}  sd {s_['sd']:8.2f}  95% ({s_['q025']:+10.2f}, {s_['q975']:+10.2f})")
                print(f"{name:13s} {key:14s} {w:13s} {got}")
        print(f"written {out}")
    return 0


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; write-once")
    payload = {"artifact": "two_proxy_uncertainty", "generated_utc": rf.now(), "draws": DRAWS, "seed": SEED,
               "what": "the sampling spread of the median of the largest agreeing group, conditional on the "
                       "retained set and the linking radius produced on the observed data",
               "does_not_include": "the uncertainty of the screen",
               "datasets": {}}
    for i, name in enumerate(("zika", "lalonde")):
        d = json.loads((RESULTS / f"{name}_v1.json").read_text())
        payload["datasets"][name] = {
            key: {w: spread(r, w, SEED + 31 * i + 7 * k) for k, w in enumerate(("significance", "noise_floor"))}
            for key, r in d["readings"].items()}
        for key, block in payload["datasets"][name].items():
            for w, s in block.items():
                if s["output"] is None:
                    print(f"{name:8s} {key:14s} {w:13s} no return ({s['return_kind']})")
                else:
                    print(f"{name:8s} {key:14s} {w:13s} {s['output']:+10.2f}  sd {s['sd']:8.2f}  "
                          f"95% ({s['q025']:+10.2f}, {s['q975']:+10.2f})")
    payload["source_sha256"] = {f: rf.sha256(HERE / f) for f in
                                ("two_proxy_uncertainty.py", "probe_j2.py", "probe_structured_scm.py")}
    with open(OUT, "x") as handle:
        handle.write(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main_for(sys.argv[1:]) if len(sys.argv) > 1 else main())
