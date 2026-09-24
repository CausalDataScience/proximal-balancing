"""Park's control outcome calibration approach (COCA) as an ATE, on every designated replicate.

The sealed comparator `family_v2_competitors.park_spc` returns the effect on the treated.  Scoring it
against an ATE needs the other half, which `park_ate.park_parts` supplies by fitting the same linear
bridge on the treated arm and combining

    ATE = P(A = 1) * ETT + P(A = 0) * ETC.

This recomputes both halves for the replicates named in the frozen protocol and for the ACIC realizations,
so the ground-truth figure can carry a COCA row scored on the same footing as every other method.  Only
linear solves are involved; nothing that produced the shards is rerun.

    python3 -B coca_ate_all.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import park_ate as pa
import run_family_v2 as rf

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = RESULTS / "coca_ate_all_v1.json"
PROTOCOL = RESULTS / "v3_confirmation_protocol_v2.json"
BLOCK = "W1"                    # the clean proxy block, as in the sealed comparator


def views_for(dataset: str, replicates) -> dict[int, tuple[dict, float]]:
    """The learner view and the exact effect of each replicate, rebuilt from its own data module."""
    out = {}
    if dataset == "twins":
        import twins_data as td
        real = td.load_real()
        for r in replicates:
            d = td.generate(real, r)
            out[r] = ({k: d[k] for k in ("X", "W1", "W2", "W3", "W4", "W5", "A", "Y")}, real["tau"])
    elif dataset == "rhc_planted":
        import rhc_planted_data as rp
        real = rp.load_real()
        for r in replicates:
            d = rp.generate(real, r)
            out[r] = ({k: d[k] for k in ("X", "W1", "W2", "W3", "W4", "W5", "A", "Y")}, d["tau"])
    elif dataset in ("ihdp", "acic"):
        import ihdp_acic_data as ia
        for r in replicates:
            d = ia.DATASETS[dataset]["load"](r)
            out[r] = ({k: d[k] for k in ("X", "W1", "W2", "W3", "W4", "W5", "A", "Y")}, d["tau"])
    else:
        raise ValueError(dataset)
    return out


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; write-once")
    protocol = json.loads(PROTOCOL.read_text())
    plan = {name: list(m["replicates"]) for name, m in protocol["manifests"].items()}
    plan["acic"] = json.loads((RESULTS / "acic" / "summary_v1.json").read_text())["replicates"]

    payload = {"artifact": "coca_ate_all", "generated_utc": rf.now(),
               "method": "Park, Richardson and Tchetgen Tchetgen's control outcome calibration approach; the "
                         "sealed implementation gives the effect on the treated, and the effect on the "
                         "controls is obtained by the same construction on the treated arm, so that "
                         "ATE = pi * ETT + (1 - pi) * ETC",
               "block": BLOCK, "protocol": PROTOCOL.name, "datasets": {}}
    for dataset, replicates in plan.items():
        rows = []
        for r, (view, tau) in views_for(dataset, replicates).items():
            parts = pa.park_parts(view, BLOCK)
            rows.append({"replicate": r, "tau": tau, **parts, "error": abs(parts["ate"] - tau)})
        errs = [x["error"] for x in rows]
        payload["datasets"][dataset] = {
            "replicates": rows, "mae": float(np.mean(errs)), "p90": float(np.quantile(errs, 0.9)),
            "mean_ate": float(np.mean([x["ate"] for x in rows])),
            "bridge_w_coefficient_range": [float(min(x["bridge_w_coefficient"]["control_arm"] for x in rows)),
                                           float(max(x["bridge_w_coefficient"]["control_arm"] for x in rows))],
        }
        print(f"{dataset:12s} n={len(rows):2d}  COCA ATE mean {payload['datasets'][dataset]['mean_ate']:+.4f}  "
              f"MAE {payload['datasets'][dataset]['mae']:.4f}")
    payload["source_sha256"] = {f: rf.sha256(HERE / f) for f in ("coca_ate_all.py", "park_ate.py",
                                                                 "family_v2_competitors.py")}
    with open(OUT, "x") as handle:
        handle.write(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
