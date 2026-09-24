"""Uncertainty for the quantity PROBE actually reports on RHC: the median of the largest agreeing group
(memo/2026-09-22-v3-confirmation-prd-v1.md, section 6, option (a)).

The number printed beside the PROBE point used to be one retained member's AIPW standard error, which is
the wrong quantity: the reported value is a median over a group of members chosen by a linking rule, not
any single member.  This resamples the whole aggregation.

    1. take the retained set and the linking radius that the v3 rule produced on the observed data
    2. draw the retained members' estimates from N(observed, Sigma) with Sigma the stored score
       cross-product restricted to those members and divided by n squared
    3. re-apply the sealed linking and largest-component median to each draw
    4. report the spread of the resulting outputs

This is the sampling spread of the aggregate conditional on the retained set and the radius.  It does not
carry the uncertainty of the screen itself, and the caption says so.

    python3 -B rhc_aggregate_uncertainty.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import family_v2_probe as pr
import noise_scaled as ns
import probe_structured_scm as ps
import rhc_data as rd
import run_family_v2 as rf

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = RESULTS / "rhc" / "aggregate_uncertainty_v1.json"
DRAWS = 20_000
SEED = 98_600_000


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; write-once")
    probe = json.loads((RESULTS / "rhc" / "diagnostic_v1_full_records.json").read_text())["probe"]
    decision = ns.decide(probe, rd.N, rd.SEED)
    names, splits = probe["split_names"], pr.all_splits()
    keep = [names.index(nm) for nm in decision["retained"]]
    if decision["output"] is None:
        raise SystemExit("the observed run returned nothing; there is no aggregate to resample")

    cov = np.array(probe["score_crossproduct"])[np.ix_(keep, keep)] / (rd.N * rd.N)
    centre = np.array([probe["records"][names[i]]["estimate"] for i in keep])
    rho = decision["rho"]
    rng = np.random.default_rng(SEED)
    sample = rng.multivariate_normal(centre, cov, size=DRAWS)

    outputs, kinds = [], {}
    labels = [splits[i] for i in keep]
    for row in sample:
        agg = ps.aggregate(labels, row, rho)
        kinds[agg["return_kind"]] = kinds.get(agg["return_kind"], 0) + 1
        if agg["output"] is not None:
            outputs.append(agg["output"])
    outputs = np.array(outputs)

    member_se = float(np.sqrt(np.median(np.diag(cov))))
    payload = {
        "artifact": "rhc_aggregate_uncertainty", "generated_utc": rf.now(),
        "what": "the sampling spread of the median of the largest agreeing group, conditional on the "
                "retained set and the linking radius that the v3 rule produced on the observed data",
        "draws": DRAWS, "seed": SEED,
        "retained": decision["retained"], "n_retained": decision["n_retained"], "rho": rho,
        "observed_output": decision["output"],
        "resampled": {"returned": int(len(outputs)), "of": DRAWS,
                      "return_kinds": kinds,
                      "mean": float(outputs.mean()), "sd": float(outputs.std(ddof=1)),
                      "q025": float(np.quantile(outputs, 0.025)),
                      "q975": float(np.quantile(outputs, 0.975))},
        "for_contrast_only": {"median_member_aipw_se": member_se,
                              "note": "the number the figure used to print; it is one member's standard "
                                      "error, not the standard error of the aggregate"},
        "does_not_include": "the uncertainty of the screen; the retained set and the radius are held at "
                            "their observed values",
        "source_sha256": {f: rf.sha256(HERE / f) for f in
                          ("rhc_aggregate_uncertainty.py", "noise_scaled.py", "probe_structured_scm.py")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "x") as handle:
        handle.write(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    r = payload["resampled"]
    print(f"observed output {decision['output']:.4f} over {decision['n_retained']} retained, rho {rho:.4f}")
    print(f"resampled aggregate: mean {r['mean']:.4f}, sd {r['sd']:.4f}, "
          f"95% interval ({r['q025']:.3f}, {r['q975']:.3f}), returned {r['returned']}/{DRAWS} {kinds}")
    print(f"for contrast, one member's AIPW se was {member_se:.4f}")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
