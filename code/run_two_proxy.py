"""PROBE on the two data sets whose proxy is a pair, with every reading written down.

    python3 -B run_two_proxy.py zika
    python3 -B run_two_proxy.py lalonde

Neither data set supplies an exact effect, so nothing here is scored the way the synthetic family is.
What is reported is: the overlap of the whole sample, the overlap of the declared common-support band,
PROBE's two candidates with their screen statistics, the three readings of the screen, and the same
comparators the rest of the paper uses.  LaLonde carries a real benchmark from the randomised arm of the
same experiment, and the reading is put beside it with the estimand difference stated, not hidden.

The band is declared in two_proxy_data.py before any estimate is formed.  Running on the whole sample as
well is the point: the untrimmed reading is what a bare application would give, and it is reported so the
reader can see what the band buys.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import family_v2_competitors as fc
import family_v2_probe as pr
import probe_j2 as pj
import run_family_v2 as rf
import two_proxy_data as tp

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "two_proxy"
K = 2                                   # the rank the whole paper uses
SOURCES = ("probe_j2.py", "two_proxy_data.py", "run_two_proxy.py", "family_v2_probe.py",
           "family_v2_competitors.py", "park_ate.py")


def baselines(view: dict, seed: int, n_blocks: int) -> dict[str, float]:
    """The X-only and the everything-in AIPW, on the same four rotations and the same code path as PROBE."""
    n = len(view["A"])
    fold_rows = pr.folds(n, seed)
    sums = {"X": 0.0, "raw": 0.0}
    for rot in range(4):
        rl = pr.roles(rot)
        rn, re = fold_rows[rl["N"]], fold_rows[rl["E"]]
        prep = pj.PrepJ(view, rn, n_blocks)
        one_n, one_e = np.ones((len(rn), 1)), np.ones((len(re), 1))
        w_all = lambda rows: np.column_stack([view[f"W{j + 1}"][rows] for j in range(n_blocks)])
        feats = {"X": (np.column_stack([one_n, prep.x(view, rn)]),
                       np.column_stack([one_e, prep.x(view, re)])),
                 "raw": (np.column_stack([one_n, view["X"][rn], w_all(rn)]),
                         np.column_stack([one_e, view["X"][re], w_all(re)]))}
        for name, (fn, fe) in feats.items():
            if name == "raw":
                mu, sd = fn[:, 1:].mean(0), fn[:, 1:].std(0) + 1e-12
                fn = np.column_stack([one_n, (fn[:, 1:] - mu) / sd])
                fe = np.column_stack([one_e, (fe[:, 1:] - mu) / sd])
            psi = fc.aipw_cv(fn, view["A"][rn], view["Y"][rn], fe, view["A"][re], view["Y"][re],
                             seed + 97 * rot)
            sums[name] += float(psi.mean()) / 4.0
    a, y = view["A"], view["Y"]
    sums["naive"] = float(y[a == 1].mean() - y[a == 0].mean())
    return sums


def comparators(view: dict, seed: int, n_blocks: int) -> dict:
    import park_ate as pa
    out = baselines(view, seed, n_blocks)
    out["p2sls_W1_treats_W2_outcome"] = fc.p2sls(view, "W1", "W2")
    out["p2sls_W2_treats_W1_outcome"] = fc.p2sls(view, "W2", "W1")
    for j in (1, 2):
        parts = pa.park_parts(view, f"W{j}")
        out[f"coca_W{j}_ett"] = parts["ett"]
        out[f"coca_W{j}_ate"] = parts["ate"]
    return out


def reading(view: dict, seed: int, label: str) -> dict:
    probe = pj.run(view, K, seed, n_blocks=2)
    rep = pj.propensity_report(view, seed, 2)
    rep.pop("propensity")
    return {"label": label, "n": len(view["A"]), "overlap": rep,
            "probe": {k: v for k, v in probe.items() if k != "score_crossproduct"},
            "score_crossproduct": probe["score_crossproduct"],
            "comparators": comparators(view, seed, 2)}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in tp.DATASETS:
        raise SystemExit(f"usage: run_two_proxy.py [{' | '.join(tp.DATASETS)}]")
    name = argv[1]
    spec = tp.DATASETS[name]
    out = RESULTS / f"{name}_v1.json"
    if out.exists():
        raise SystemExit(f"{out} exists; results are write-once")
    view, seed = spec["load"](), spec["seed"]
    labels = spec["covariate_labels"]
    e = tp.propensity(view)
    lo, hi = tp.BAND
    keep = (e >= lo) & (e <= hi)
    banded, dropped = tp.drop_dependent_columns(tp.restrict(view, keep), labels)

    payload = {"artifact": "two_proxy", "dataset": name, "generated_utc": rf.now(), "k": K, "seed": seed,
               "band": list(tp.BAND),
               "band_declared_before_any_estimate": "two_proxy_data.BAND, with the reason written beside it",
               "columns": view["names"], "reference": spec["reference"],
               "readings": {"whole_sample": reading(view, seed, "the sample as published"),
                            "common_support": reading(banded, seed + 1,
                                                      f"the units whose propensity lies in [{lo}, {hi}]")},
               "covariate_labels": labels,
               "covariates_dropped_on_the_band": dropped,
               "why_dropped": "the band makes these exactly a function of the intercept and the remaining "
                              "covariates, so every design built from them is singular; they are dropped "
                              "once for all estimators alike, never for one of them",
               "estimand_note": "PROBE's score is the one for the average effect over whoever is in the "
                                "sample.  On the band that is the average effect over the band, which is a "
                                "different population from the whole sample and from the treated.",
               "source_sha256": {f: rf.sha256(HERE / f) for f in SOURCES}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as handle:
        handle.write(json.dumps(payload, indent=1, allow_nan=False) + "\n")

    for key, r in payload["readings"].items():
        ov = r["overlap"]
        print(f"\n--- {name}: {r['label']}  (n = {r['n']}, treated {ov['treated']})")
        print(f"    propensity in [{ov['min']:.4f}, {ov['max']:.4f}], "
              f"effective sample size treated {ov['treated_ess']:.1f} of {ov['treated']}, "
              f"control {ov['control_ess']:.1f} of {ov['control']}")
        p = r["probe"]
        for nm in p["split_names"]:
            rec = p["records"][nm]
            print(f"    {nm}: estimate {rec['estimate']:+9.4f}  gap {rec['pooled_gap']:+.5f}  "
                  f"se {rec['pooled_se']:.5f}  gap - z se {rec['pooled_gap'] - p['z'] * rec['pooled_se']:+.5f}  "
                  f"overlap {rec['overlap_all_rotations']}")
        for rname, rd in p["readings"].items():
            got = "nothing" if rd["output"] is None else f"{rd['output']:+.4f}"
            print(f"    {rname:13s} retained {rd['retained'] or '[]'} -> {got} ({rd['return_kind']})")
        c = r["comparators"]
        print("    " + "  ".join(f"{k} {v:+.3f}" for k, v in c.items()))
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
