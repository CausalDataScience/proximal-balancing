"""Recompute every round 6 population quantity under the corrected definitions, without retraining.

Two things changed in round 7 and neither of them touches a learned weight:
  R1  the benchmark now holds out T = (X, W_S), as the manuscript's Section 3 defines it, instead of W_S
      alone, and conditions through an orthonormal basis so that repeating X in (Z, T) is harmless;
  R3  every comparator gets its own population adjusted effect instead of being assigned 1.

So the round 6 checkpoints are reloaded, their representations are rebuilt exactly, and the population
numbers are recomputed.  Old and new values are written side by side so the change is auditable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import probe_dgp_ae as dgp
import probe_brier_neural as pbn
import provenance as prov
import staged_dgp_f as sf

S = (1,)
OUT = "../results/dgp_f_round6_rescored.json"
# (checkpoint tag, rows generated for that run, replicate) -- the generation settings of each round 6 run
SOURCES = [("stage2_n4000", 4000, sf.STAGE_SEED + 1, "stage 2, n=4000", 0.0040024),
           ("stage2_n16000", 16000, sf.STAGE_SEED + 1, "stage 2, n=16000", 0.0014776),
           ("stage2_n48000", 48000, sf.STAGE_SEED + 1, "stage 2, n=48000", 0.0000876),
           ("stage3_rep24000", 24000 + 32000, sf.STAGE_SEED + 2, "stage 3", 0.0002938)]


def main() -> int:
    pbn.ENCODER_FACTORY = sf.LinearRepresentation
    pbn.Z_DIM = sf.LINEAR_Z_DIM
    inputs = {tag: f"../results/checkpoints/{tag}.pt" for tag, *_ in SOURCES}
    inputs["stage3_result"] = "../results/dgp_f_staged_r6_s3.json"
    out = {"provenance": prov.header(
        run_id="rescore_round6", script="rescore_round6.py", args={}, training_mode="variant",
        seeds={"note": "no new training; the round 6 checkpoints are reloaded unchanged"},
        code_hash=sf.code_hash(), inputs=inputs,
        extra={"scope": "population quantities only: D2 under the corrected held-out set, and tau_Z for "
                        "every comparator; no weight is retrained and no estimate is recomputed",
               "checkpoints_predate_hashing": True}),
        "note": __doc__, "representations": [], "comparators": {}}
    maps = sf.f_linear_maps()

    print(f"{'run':20s} {'stored D2':>12} {'corrected D2':>13} {'tau_Z':>11} {'|tau_Z - 1|':>12}")
    for tag, n_rows, replicate, label, stored in SOURCES:
        ck = torch.load(Path("../results/checkpoints") / f"{tag}.pt", weights_only=False)
        data = dgp.generate("F", n_rows, replicate=replicate, phase="pilot", coef=dgp.coefficients())
        specs = pbn.block_specs(data, [b for b in range(dgp.N_BLOCKS) if b not in set(S)])
        enc = sf.LinearRepresentation(specs, data["X"].shape[1])
        enc.load_state_dict(ck["encoder"])
        b = sf.benchmark_rows(S, sf.learned_z_rows({"enc": enc}, ck["standardizers"]))
        out["representations"].append({"run": label, "checkpoint": tag, "D2_round6": stored,
                                       "D2_corrected": b["D2"], "tau_Z": b["tau_Z"],
                                       "representation_bias": abs(b["tau_Z"] - dgp.TRUE_ATE)})
        print(f"{label:20s} {stored:12.7f} {b['D2']:13.7f} {b['tau_Z']:+11.6f} "
              f"{abs(b['tau_Z'] - dgp.TRUE_ATE):12.6f}")

    print()
    for lab, Z in (("raw (X, W)", np.vstack([maps["X"][None, :]] + [maps[f"W{b}"] for b in range(5)])),
                   ("X only", maps["X"][None, :]),
                   ("oracle (X, U, C)", np.vstack([maps["X"], maps["U"], maps["C"]])),
                   ("PROBE, exact representation", sf.z_map("family", S, dgp.f_balance_coefficient(4), maps))):
        t = sf.benchmark_rows(S, Z)["tau_Z"]
        out["comparators"][lab] = {"tau_Z": t, "adjustment_bias": abs(t - dgp.TRUE_ATE)}
        print(f"population adjusted effect, {lab:28s} tau_Z = {t:+.6f}, "
              f"|tau_Z - 1| = {abs(t - dgp.TRUE_ATE):.6f}")

    # round 6 stage 3 stored |theta - tau_Z| for the comparators against a target of 1; recompute correctly
    s3 = json.load(open("../results/dgp_f_staged_r6_s3.json"))["stage3"]["rows"]
    fixed = []
    for r in s3:
        lab = r["adjustment"]
        tz = out["comparators"].get(lab, {}).get("tau_Z")
        if lab == "PROBE, learned representation":
            tz = [q for q in out["representations"] if q["run"] == "stage 3"][0]["tau_Z"]
        theta = float(r["theta"])
        fixed.append({"n_est": r["n_est"], "adjustment": lab, "theta": theta, "tau_Z": tz,
                      "err_total": abs(theta - dgp.TRUE_ATE),
                      "err_adjustment": None if tz is None else abs(tz - dgp.TRUE_ATE),
                      "err_estimation": None if tz is None else abs(theta - tz),
                      "err_estimation_round6": float(r["|theta - tau_Z|"])})
    out["stage3_recomputed"] = fixed
    print("\nstage 3 estimation error, round 6 value against the corrected one")
    print(f"{'n_est':>7} {'adjustment':30s} {'round 6':>9} {'corrected':>10}")
    for f in fixed:
        if f["err_estimation"] is not None and abs(f["err_estimation"] - f["err_estimation_round6"]) > 1e-6:
            print(f"{f['n_est']:7d} {f['adjustment']:30s} {f['err_estimation_round6']:9.4f} "
                  f"{f['err_estimation']:10.4f}")
    out["provenance"]["complete"] = True
    Path(OUT).write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
