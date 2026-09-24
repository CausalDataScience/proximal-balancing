"""The randomised answer for exactly the WSC observational sample.

`mathGrp` was randomised over all 2,200 independently of the stated preference `mathSel` (about half of
each preference group got math). So within each preference group the math-versus-vocabulary contrast is a
randomised effect, and the observational sample, the 288 who preferred math and got it plus the 817 who
preferred vocabulary and got it, has the known effect

    tau_obs = (288 * tau_prefer_math + 817 * tau_prefer_vocab) / 1,105.

This replaces the full-sample contrast (0.760) as the answer for PROBE's whole-sample estimate, whose
target is the average effect over those 1,105.  The common-support band changes the population and has no
such exact answer.

    python3 -B wsc_exact_benchmark.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_family_v2 as rf

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "materials" / "real_world_data" / "wscdata" / "df2200_cleaned.csv"
OUT = HERE.parent / "results" / "wsc" / "exact_benchmark_v1.json"


def compute() -> dict:
    d = pd.read_csv(CSV)
    z, s, y = d.mathGrp.to_numpy(), d.mathSel.to_numpy(), d.mathPost.to_numpy()
    groups = {}
    for label, pref in (("prefer_math", 1), ("prefer_vocabulary", 0)):
        a, b = y[(s == pref) & (z == 1)], y[(s == pref) & (z == 0)]
        groups[label] = {"effect": float(a.mean() - b.mean()),
                         "se": float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))),
                         "n_math": int(len(a)), "n_vocabulary": int(len(b))}
    n1, n0 = int(((z == 1) & (s == 1)).sum()), int(((z == 0) & (s == 0)).sum())
    g1, g0 = groups["prefer_math"], groups["prefer_vocabulary"]
    est = (n1 * g1["effect"] + n0 * g0["effect"]) / (n1 + n0)
    se = float(np.sqrt((n1 * g1["se"]) ** 2 + (n0 * g0["se"]) ** 2) / (n1 + n0))
    return {"estimate": float(est), "se": se, "weights": {"prefer_math": n1, "prefer_vocabulary": n0},
            "by_preference": groups}


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; write-once")
    payload = {"artifact": "wsc_exact_benchmark", "generated_utc": rf.now(), **compute(),
               "estimand": "average effect of math training over the 1,105 whose assignment matched their preference",
               "source_sha256": {"wsc_exact_benchmark.py": rf.sha256(HERE / "wsc_exact_benchmark.py")}}
    with open(OUT, "x") as handle:
        handle.write(json.dumps(payload, indent=1) + "\n")
    print(f"exact benchmark {payload['estimate']:.3f} (se {payload['se']:.3f}); written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
