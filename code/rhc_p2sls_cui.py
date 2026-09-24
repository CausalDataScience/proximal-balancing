"""Proximal 2SLS on RHC with Cui et al.'s specification, and only that one
(the author's decision of 2026-09-23; memo/2026-09-23-replicate-extension-prd-v2.md, amendment 1).

Cui et al. (2022, Semiparametric proximal causal inference, section 6): Z = (pafi1, paco21), W = (ph1, hema1),
and "the remaining 67 variables in X", which include the other six physiological measures; the outcome bridge
is linear, h(W, A, X; b) = b0 + ba A + bw W + bx X (their equation 15), so the effect is ba.  Here W1 is their
Z, W2 their W, and X is the pre-registered covariate set plus the blocks W3, W4, W5 (wblc1, alb1, bili1, crea1,
sod1, pot1).  Our covariate set drops three variables that are more than half missing (cat2, adld3p, urin1),
so X is close to theirs but not identical.

The resamples are those of rhc_bootstrap.py, drawn by the same call with the same seeds; the check below
recomputes the frozen runner's own 2SLS (without the six measures) on resamples 1 to 20 and requires it to
equal the sealed shards exactly, which pins the resampling.

    python3 -B rhc_p2sls_cui.py            write results/rhc_p2sls_cui/v1.json, write-once
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "rhc_p2sls_cui" / "v1.json"
SEALED = HERE.parent / "results" / "rhc_bootstrap"
RESAMPLES = range(1, 101)
LABEL = "Proximal 2SLS, Cui et al.'s specification (Z = W1, W = W2, X = covariates and W3, W4, W5)"


def cui_view(view: dict) -> dict:
    return dict(view, X=np.column_stack([view["X"], view["W3"], view["W4"], view["W5"]]))


def estimate(view: dict) -> float:
    import family_v2_competitors as fc
    return float(fc.p2sls(cui_view(view), "W1", "W2"))


def resample(view: dict, n: int, b: int) -> dict:
    import rhc_bootstrap as rb
    idx = np.random.default_rng(rb.SEED + b).integers(0, n, n)       # exactly rhc_bootstrap.run_one
    return {k: v[idx] for k, v in view.items()}


def main() -> int:
    import family_v2_competitors as fc
    import rhc_data as rd
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; write-once")
    data = rd.load()
    view, n = data["view"], data["n"]
    for b in range(1, 21):                                             # the resampling is the sealed one
        sealed = json.loads((SEALED / f"boot_{b}.json").read_text())["comparators"]["p2sls_cui_roles"]
        mine = float(fc.p2sls(resample(view, n, b), "W1", "W2"))
        if mine != sealed:
            raise SystemExit(f"resample {b}: the runner's own 2SLS is {mine}, the sealed shard says {sealed}")
    values = {str(b): estimate(resample(view, n, b)) for b in RESAMPLES}
    payload = {"label": LABEL, "observed": estimate(view), "resamples": values,
               "check": "the runner's own 2SLS recomputed on resamples 1 to 20 equals the sealed shards exactly",
               "cui_et_al_2022_table_3": {"proximal_or": -1.80, "proximal_dr": -1.66},
               "source_sha256": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()
                                 for f in ("rhc_p2sls_cui.py", "family_v2_competitors.py", "rhc_data.py",
                                           "rhc_bootstrap.py")}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "x") as handle:
        handle.write(json.dumps(payload, indent=1) + "\n")
    v = np.array([values[str(b)] for b in RESAMPLES])
    s20 = v[:20]
    print(f"observed {payload['observed']:+.3f}; resamples 1-20 mean {s20.mean():+.3f} "
          f"(90% {np.quantile(s20, .05):+.3f} to {np.quantile(s20, .95):+.3f}); all 100 mean {v.mean():+.3f}")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
