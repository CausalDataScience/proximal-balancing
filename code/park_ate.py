"""Park's single proxy control turned into an ATE estimator, so that it can be scored against the ATE.

The sealed comparator `family_v2_competitors.park_spc` targets the effect on the treated,
ETT = E[Y | A = 1] - E[b_0(W, X) | A = 1], with the linear bridge b_0 solved from the control-arm moments
E[(1, Y, X)(Y - b_0(W, X)) | A = 0] = 0.  Under joint latent exchangeability the same construction on the
treated arm gives the effect on the controls, ETC = E[b_1(W, X) | A = 0] - E[Y | A = 0], with b_1 solved from
E[(1, Y, X)(Y - b_1(W, X)) | A = 1] = 0, and

    ATE = P(A = 1) * ETT + P(A = 0) * ETC.

Nothing in the sealed file is changed; its ETT is reproduced here bit for bit and the ETC half is added.

    python3 -B park_ate.py          checks the construction on SCM-1 (ATE = ETT = 1) and scores it on Twins
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import family_v2_competitors as fc

HERE = Path(__file__).resolve().parent


def _bridge(view: dict, block: str, arm: int) -> tuple[np.ndarray, np.ndarray]:
    """The linear bridge fitted on one arm, returned as (coefficients, the design for every unit)."""
    a, y = view["A"], view["Y"]
    x = (view["X"] - view["X"].mean(0)) / (view["X"].std(0) + 1e-12)
    w = fc._pcs(view[block], 1)[:, 0]
    rows = a == arm
    inst = np.column_stack([np.ones(rows.sum()), y[rows], x[rows]])
    reg = np.column_stack([np.ones(rows.sum()), w[rows], x[rows]])
    eta = np.linalg.solve(inst.T @ reg, inst.T @ y[rows])
    return eta, np.column_stack([np.ones(len(a)), w, x])


def park_parts(view: dict, block: str) -> dict:
    a, y = view["A"], view["Y"]
    eta0, design = _bridge(view, block, 0)
    eta1, _ = _bridge(view, block, 1)
    trt, ctrl = a == 1, a == 0
    ett = float(y[trt].mean() - (design[trt] @ eta0).mean())
    etc = float((design[ctrl] @ eta1).mean() - y[ctrl].mean())
    pi = float(a.mean())
    return {"ett": ett, "etc": etc, "pi": pi, "ate": pi * ett + (1.0 - pi) * etc,
            "bridge_w_coefficient": {"control_arm": float(eta0[1]), "treated_arm": float(eta1[1])}}


def main() -> int:
    import family_v2_dgp as g
    import twins_data as td

    # 1. the construction on SCM-1, where the effect is homogeneous and ATE = ETT = 1
    print("SCM-1 confirmation seeds, Park on the clean block W1 and the contaminated block W4:")
    seeds = [97_200_000 + 1_000 * k for k in range(20)]
    rows = []
    for seed in seeds:
        data = g.generate(g.LEVELS["SCM-1"], 24_000, seed)
        view = g.learner_view(data)
        p1, p4 = park_parts(view, "W1"), park_parts(view, "W4")
        sealed = fc.park_spc(view, "W1")
        assert abs(sealed - p1["ett"]) < 1e-9, "the ETT half must reproduce the sealed comparator exactly"
        rows.append((p1, p4))
    for name, idx in (("clean W1", 0), ("contaminated W4", 1)):
        ett = np.array([r[idx]["ett"] for r in rows]); etc = np.array([r[idx]["etc"] for r in rows])
        ate = np.array([r[idx]["ate"] for r in rows])
        print(f"  {name:16s} ETT mean {ett.mean():+.4f}  ETC mean {etc.mean():+.4f}  ATE mean {ate.mean():+.4f}  "
              f"ATE MAE {np.mean(np.abs(ate - 1.0)):.4f}")

    # 2. Twins, scored against the ATE and against the true ETT and ETC, which both potential outcomes give
    real = td.load_real()
    print(f"\nTwins, ATE = {real['tau']:+.4f}:")
    out = []
    for r in range(1, 11):
        d = td.generate(real, r)
        view = {k: d[k] for k in ("X", "W1", "W2", "W3", "W4", "W5", "A", "Y")}
        a = d["A"]
        diff = real["Y1"] - real["Y0"]
        truth = {"ett": float(diff[a == 1].mean()), "etc": float(diff[a == 0].mean()), "ate": real["tau"]}
        p = park_parts(view, "W1")
        out.append({"replicate": r, "truth": truth, "park_clean_w1": p})
        print(f"  rep {r:2d}: true ETT {truth['ett']:+.4f} ETC {truth['etc']:+.4f} | Park ETT {p['ett']:+.4f} "
              f"ETC {p['etc']:+.4f} ATE {p['ate']:+.4f} | bridge coef on W: control {p['bridge_w_coefficient']['control_arm']:+.2f}, "
              f"treated {p['bridge_w_coefficient']['treated_arm']:+.2f}")
    ate = np.array([o["park_clean_w1"]["ate"] for o in out])
    print(f"  Park ATE: mean {ate.mean():+.4f}, MAE against tau {np.mean(np.abs(ate - real['tau'])):.4f}")
    path = HERE.parent / "results" / "twins" / "park_ate_v1.json"
    if not path.exists():
        path.write_text(json.dumps({"artifact": "twins_park_ate", "construction": __doc__.strip().splitlines()[0],
                                    "scm1_check": [{"clean": r[0], "contaminated": r[1]} for r in rows],
                                    "twins": out}, indent=1, allow_nan=False) + "\n")
        print(f"  written {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
