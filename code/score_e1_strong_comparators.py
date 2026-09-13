"""Score every saved E1 replicate against strong-learner comparators and save the result.

PROBE's numbers are the saved Algorithm 1 outputs of the E1 grid.  Only the comparators are recomputed, with
the logistic propensity and MLP outcome models that EXPERIMENTS.md requires for any reported claim.  The
earlier strong-comparator numbers lived only in a scratch log; this file is their durable, reproducible form.
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings

import numpy as np
from scipy.stats import t as tdist

warnings.filterwarnings("ignore")
import probe_scms as scms
import probe_e1_pipeline as P
from probe_e1_pipeline import seed_key, rng, baseline_aipw

P.NUISANCE_LEARNER["kind"] = "mlp"
GRID = [("s9", [12000]), ("s10", [12000, 6000]), ("s11", [12000, 6000]), ("s12", [12000, 6000]), ("s3", [1500])]
OUT = "../results/probe_e1_strong_comparators.json"


def main() -> int:
    out = {"note": __doc__, "learner": "logistic propensity + MLP outcomes, clip 0.05", "cells": {}, "complete": False}
    t0 = time.time()
    for scm, ns in GRID:
        saved = json.load(open(f"../results/probe_e1_{scm}.json"))
        coef = scms.coefficients(scm, seed_key("coef"))
        for n in ns:
            runs = [x for x in saved["runs"] if x["n"] == n and x["algorithm1"].get("output") is not None]
            if not runs:
                continue
            n_index = [1500, 3000, 6000, 12000].index(n)
            rows = []
            for x in runs:
                rep = x["rep"]
                d = scms.generate(scm, n, seed_key("data", rep), coef)
                perm = rng(*seed_key("split", rep, n_index)).permutation(n)
                third = n // 3
                idx = {"D": perm[:third], "N": perm[third:2 * third], "E": perm[2 * third:]}
                rows.append({"rep": rep, "probe": x["algorithm1"]["output"],
                             "oracle": baseline_aipw(scms.oracle_adjustment(scm, d), d, idx, 0.05, 3),
                             "raw": baseline_aipw(np.column_stack([d["X"], d["W"]]), d, idx, 0.05, 2),
                             "x_only": baseline_aipw(d["X"], d, idx, 0.05, 1)})
            e = {k: np.abs(np.array([r[k] for r in rows]) - 1.0) for k in ("probe", "oracle", "raw", "x_only")}
            q = tdist.ppf(0.975, len(rows) - 1)

            def ci(dd):
                m, se = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
                return {"mean": float(m), "lo": float(m - q * se), "hi": float(m + q * se)}
            vs_or, vs_raw = ci(e["probe"] - e["oracle"]), ci(e["probe"] - e["raw"])
            cell = {"n": n, "reps": len(rows), "mae": {k: float(v.mean()) for k, v in e.items()},
                    "vs_oracle": vs_or, "vs_raw": vs_raw,
                    "beats_raw": bool(vs_raw["hi"] < 0), "matches_oracle": bool(vs_or["hi"] <= 0.10),
                    "rows": rows}
            out["cells"][f"{scm}_n{n}"] = cell
            print(f"{scm} n={n} reps={len(rows)}: MAE PROBE {e['probe'].mean():.3f} | oracle {e['oracle'].mean():.3f} | "
                  f"raw {e['raw'].mean():.3f} | X-only {e['x_only'].mean():.3f} || vs oracle {vs_or['mean']:+.3f} "
                  f"[{vs_or['lo']:+.3f},{vs_or['hi']:+.3f}] matches={cell['matches_oracle']} | vs raw "
                  f"{vs_raw['mean']:+.3f} [{vs_raw['lo']:+.3f},{vs_raw['hi']:+.3f}] beats={cell['beats_raw']}", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)
    out["complete"] = True
    out["runtime_s"] = time.time() - t0
    json.dump(out, open(OUT, "w"), indent=1)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
