"""Display-only replay of the sealed positive-control audit, for the mechanism figure.

The audit `results/positive_control/confirmation_v1.json` records the retained set, the selected
representations, rho and the pass rates, but not the two quantities a picture of the mechanism needs:
the pair of Brier risks the screen compared for one kept and one dropped split, and the estimates of a
single evaluation draw.  Both are deterministic functions of the sealed protocol, so they are replayed
here rather than recomputed under new settings.

Nothing in this file can change a decision.  The protocol, the learning seed, n_D, the threshold, the
retained set, the selected beta and rho all come from the sealed artifacts; the script only reads the
numbers back out at a finer grain.  Source hashes are checked by load_protocol.

Writes results/positive_control/screen_display_v1.json.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import positive_control_scm as pc
import run_positive_control_confirmation as conf

STATE = 0
DRAW = 0
KEPT_SPLIT = (0,)          # W_1 held out: a clean measurement of the confounder
DROPPED_SPLIT = (3,)       # W_4 held out: a measurement shifted by the treatment-only latent
OUTPUT = conf.RESULTS / "screen_display_v1.json"


def screen_pair(design, learning: dict, split: tuple[int, ...], threshold: float) -> dict:
    """The two Brier risks the screen compared for this split, at the beta it selected."""
    out = pc.select_and_screen(design, split, learning, threshold)
    beta = np.array(out["beta"])
    stat = pc.screen_statistic(design, split, beta, learning)
    return {"split": list(split), "beta": out["beta"],
            "risk_base": stat["risk_base"], "risk_augmented": stat["risk_augmented"],
            "objective": stat["objective"], "signed_gap": stat["signed_gap"],
            "threshold": threshold, "retained": out["retained"],
            "selection_margin": out["selection_margin"]}


def main() -> int:
    design, protocol = conf.load_protocol()
    audit = json.loads((conf.RESULTS / "confirmation_v1.json").read_text())
    state = audit["states"][STATE]

    n_d = protocol["screen"]["n_D"]
    threshold = protocol["screen"]["threshold_t"]
    n_e = protocol["evaluation"]["n_E"]
    d_seed = state["learning_seed"]
    e_seed = 5_400_000 + 10_000 * STATE + DRAW

    learning = pc.generate(design, n_d, d_seed)
    screen = [screen_pair(design, learning, KEPT_SPLIT, threshold),
              screen_pair(design, learning, DROPPED_SPLIT, threshold)]
    for row in screen:
        print(f"split {row['split']}  R_0 {row['risk_base']:.6f}  R_1 {row['risk_augmented']:.6f}  "
              f"objective {row['objective']:.3e}  retained {row['retained']}")

    # Every split's screen statistic, so the picture can show the whole census against the threshold.
    census = []
    for split in pc.all_splits(design.J):
        out = pc.select_and_screen(design, split, learning, threshold)
        census.append({"split": list(split), "objective": out["objective"],
                       "retained": out["retained"]})
    n_kept = sum(r["retained"] for r in census)
    print(f"census: {n_kept} of {len(census)} splits below t = {threshold}")

    retained = [(tuple(r["split"]), np.array(r["beta"])) for r in state["retained"]]
    rho = state["rho"]
    evaluation = pc.generate(design, n_e, e_seed)
    estimates = np.array([pc.aipw_scores(design, split, beta, evaluation).mean()
                          for split, beta in retained])
    result = pc.aggregate([s for s, _ in retained], estimates, rho)
    print(f"\ndraw {DRAW} (seed {e_seed}, n_E {n_e:,})  return {result['return_kind']}  "
          f"output {result['output']:.5f}  component sizes {result['component_sizes']}")

    payload = {
        "purpose": "display quantities for figures/section5/probe-mechanism; no decision depends on this file",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol": {"path": str(conf.PROTOCOL), "protocol_id": protocol["protocol_id"],
                     "sha256": hashlib.sha256(conf.PROTOCOL.read_bytes()).hexdigest()},
        "audit": {"path": str(conf.RESULTS / "confirmation_v1.json"),
                  "sha256": hashlib.sha256((conf.RESULTS / "confirmation_v1.json").read_bytes()).hexdigest()},
        "state": STATE, "learning_seed": d_seed, "n_D": n_d, "threshold_t": threshold,
        "screen": screen, "census": census,
        "evaluation_draw": {"draw": DRAW, "seed": e_seed, "n_E": n_e, "rho": rho,
                            "link_radius": 2.0 * rho,
                            "splits": [list(s) for s, _ in retained],
                            "theta": [r["theta"] for r in state["retained"]],
                            "estimates": estimates.tolist(),
                            "return_kind": result["return_kind"],
                            "output": result["output"],
                            "members": result["members"],
                            "component_sizes": result["component_sizes"]},
    }
    OUTPUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nwritten to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
