"""The shared SCM of the two original multi-image experiments (PRDs memo/2026-09-20-shapes3d-multi-image-prd-v1.md
and memo/2026-09-20-mnist-multi-image-prd-v1.md, sections 3 and 5).

Every scalar is a sign.  U = (U_c, U_t) is latent.  Four measurement signs M = (V_1, V_2, 2 V_3 - 1, N_0) choose
the four original images and are never shown to a learner:

    U_c, U_t, X_2, N_0 independent fair signs
    X_1 = U_c w.p. 0.65        V_1 = U_c w.p. 0.85        V_2 = U_c w.p. 0.70        V_3 = 1{U_c = 1 or U_t = 1}
    e = 0.5 + 0.25 V_1 + 0.15 U_t + 0.05 X_2,     A ~ Bernoulli(e)
    Y(a) = a - 2.5 U_c + 0.3 X_1 + 0.4 X_2 + 0.3 eps,     so tau = 1

The exact evaluator enumerates the 128 states (U_c, U_t, X_2, N_0, three flips), sums A with its true
probability and integrates eps at its mean.  Everything below `sample` is evaluator-only.

The census (P0, PRD section 5 item 7 and section 12) goes one step past the known witness.  A representation is
Z = (X, h(W_-S)), and at the population level h is a function of the retained measurement signs, so for one
split it ranges over the set partitions of {-1, +1}^{|-S|}: at most Bell(8) = 4,140.  For every one of them the
census records the population Brier gap and the adjustment functional, which is what decides whether a screen
threshold can separate the representations that target tau from the ones that do not.
"""
from __future__ import annotations

import itertools

import numpy as np

P_X1, P_V1, P_V2 = 0.65, 0.85, 0.70
TREAT = {"const": 0.5, "V1": 0.25, "Ut": 0.15, "X2": 0.05}
OUTCOME = {"Uc": -2.5, "X1": 0.3, "X2": 0.4}
OUTCOME_NOISE = 0.3
TAU = 1.0
J = 4
CHANNELS = ("V1", "V2", "M3", "N0")          # the sign that chooses image j = 1..4
REFERENCE = {"unadjusted": -0.607, "X": -0.627450261141, "X,V1": 1.0, "X,V1,V2,V3,N0": 1.291167121762,
             "oracle X,U": 1.0}
REFERENCE_TOL = 1e-9


def _derive(uc, ut, x2, n0, f_x1, f_v1, f_v2) -> dict[str, np.ndarray]:
    """Everything that is a deterministic function of the exogenous signs; a flip of 1 reverses the sign."""
    x1, v1, v2 = uc * (1 - 2 * f_x1), uc * (1 - 2 * f_v1), uc * (1 - 2 * f_v2)
    v3 = ((uc == 1) | (ut == 1)).astype(np.int64)
    e = TREAT["const"] + TREAT["V1"] * v1 + TREAT["Ut"] * ut + TREAT["X2"] * x2
    mu0 = OUTCOME["Uc"] * uc + OUTCOME["X1"] * x1 + OUTCOME["X2"] * x2
    return {"Uc": uc, "Ut": ut, "X1": x1, "X2": x2, "V1": v1, "V2": v2, "V3": v3, "M3": 2 * v3 - 1, "N0": n0,
            "e": e, "mu0": mu0}


def sample(n: int, seed: int) -> dict[str, dict[str, np.ndarray]]:
    """One data set, split into what a learner may read and what only the evaluator may read.

    Four generator streams: latent signs, measurement flips, treatment draw, outcome noise.  The image draw has
    its own streams in multi_image_banks, so a change there never moves (X, A, Y).
    """
    lat, flip, trt, out = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    signs = 2 * lat.integers(0, 2, size=(n, 4)) - 1
    flips = (flip.random((n, 3)) >= np.array([P_X1, P_V1, P_V2])).astype(np.int64)
    d = _derive(signs[:, 0], signs[:, 1], signs[:, 2], signs[:, 3], flips[:, 0], flips[:, 1], flips[:, 2])
    a = (trt.random(n) < d["e"]).astype(np.float64)
    y = d["mu0"] + TAU * a + OUTCOME_NOISE * out.standard_normal(n)
    learner = {"row_id": np.arange(n), "X": np.column_stack([d["X1"], d["X2"]]).astype(np.float64), "A": a, "Y": y}
    evaluator = {k: d[k] for k in ("Uc", "Ut", "V1", "V2", "V3", "N0", "e", "mu0")}
    evaluator["M"] = np.column_stack([d[c] for c in CHANNELS])
    return {"learner": learner, "evaluator": evaluator}


# ----------------------------------------------------------------------------- exact evaluator

def state_table() -> dict[str, np.ndarray]:
    """The 128 states with their probabilities."""
    grid = np.array(list(itertools.product((-1, 1), (-1, 1), (-1, 1), (-1, 1), (0, 1), (0, 1), (0, 1))))
    uc, ut, x2, n0, f_x1, f_v1, f_v2 = grid.T
    prob = np.full(len(grid), 1 / 16.0)
    for f, p in ((f_x1, P_X1), (f_v1, P_V1), (f_v2, P_V2)):
        prob = prob * np.where(f == 1, 1 - p, p)
    table = _derive(uc, ut, x2, n0, f_x1, f_v1, f_v2)
    table["p"] = prob
    return table


def labels(table: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    """An integer label per state for the joint value of the named columns."""
    if not keys:
        return np.zeros(len(table["p"]), dtype=np.int64)
    cols = np.column_stack([table[k] for k in keys])
    return np.unique(cols, axis=0, return_inverse=True)[1].reshape(-1)


def _join(*parts: np.ndarray) -> np.ndarray:
    return np.unique(np.column_stack(parts), axis=0, return_inverse=True)[1].reshape(-1)


def adjusted(table: dict[str, np.ndarray], z: np.ndarray) -> float:
    """sum_z P(z) {E[Y | A=1, z] - E[Y | A=0, z]} for a discrete representation given as state labels."""
    p, e, mu0 = table["p"], table["e"], table["mu0"]
    k = int(z.max()) + 1
    pz = np.bincount(z, p, k)
    m1 = np.bincount(z, p * e * (mu0 + TAU), k) / np.bincount(z, p * e, k)
    m0 = np.bincount(z, p * (1 - e) * mu0, k) / np.bincount(z, p * (1 - e), k)
    return float(pz @ (m1 - m0))


def brier_gap(table: dict[str, np.ndarray], z: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """Population Brier gap of the nested critics, E[{P(A=1 | Z, T) - P(A=1 | Z)}^2], with the variance of the
    per-row difference d = (A - q0)^2 - (A - q1)^2 that an audit of n rows would average."""
    p, e = table["p"], table["e"]
    zt = _join(z, t)
    q1 = (np.bincount(zt, p * e) / np.bincount(zt, p))[zt]
    q0 = (np.bincount(z, p * e) / np.bincount(z, p))[z]
    gap = float(p @ (q1 - q0) ** 2)
    d1, d0 = (1 - q0) ** 2 - (1 - q1) ** 2, q0 ** 2 - q1 ** 2          # d when A = 1, when A = 0
    second = float(p @ (e * d1 ** 2 + (1 - e) * d0 ** 2))
    return {"gap": gap, "d_sd": float(np.sqrt(max(second - gap ** 2, 0.0)))}


def reference_values() -> dict[str, float]:
    t = state_table()
    sets = {"unadjusted": (), "X": ("X1", "X2"), "X,V1": ("X1", "X2", "V1"),
            "X,V1,V2,V3,N0": ("X1", "X2", "V1", "V2", "V3", "N0"), "oracle X,U": ("X1", "X2", "Uc", "Ut")}
    return {name: adjusted(t, labels(t, keys)) for name, keys in sets.items()}


def overlap_range() -> tuple[float, float]:
    e = state_table()["e"]
    return float(e.min()), float(e.max())


# ----------------------------------------------------------------------------- splits and the witness

def all_splits() -> list[tuple[int, ...]]:
    """The fourteen nonempty proper subsets of the blocks 1..4, as held-out sets."""
    return [s for r in range(1, J) for s in itertools.combinations(range(1, J + 1), r)]


def split_id(split: tuple[int, ...]) -> str:
    return "S" + "".join(str(j) for j in split)


def _held(table, split) -> np.ndarray:
    """Audit target T_S = (X, M_S).  X is in every representation, so only M_S can add anything."""
    return labels(table, tuple(CHANNELS[j - 1] for j in split))


def _kept(split) -> tuple[str, ...]:
    return tuple(CHANNELS[j - 1] for j in range(1, J + 1) if j not in split)


def witness_report() -> dict:
    """PRD section 5 item 7 for Z* = (X, V_1) with S = {2} and S = {2, 4}."""
    t = state_table()
    z = labels(t, ("X1", "X2", "V1"))
    out = {"target": adjusted(t, z), "splits": {}}
    for split in ((2,), (2, 4)):
        held = _held(t, split)
        # common held-out channel (manuscript Assumption 2): given (U, X, W_-S), the held-out signs do not move A
        rest = labels(t, ("Uc", "Ut", "X1", "X2") + _kept(split))
        out["splits"][split_id(split)] = {"balance_gap": brier_gap(t, z, held)["gap"],
                                          "heldout_channel_gap": brier_gap(t, rest, held)["gap"]}
    # outcome-relevant completeness: inside every cell of Z*, the law of V_2 given U_c is the same full-rank
    # channel, because V_2 depends on the state through U_c alone and its flip is independent of (X, V_1)
    dets = []
    for cell in np.unique(z):
        rows = z == cell
        mat = np.array([[t["p"][rows & (t["Uc"] == u) & (t["V2"] == v)].sum() for v in (-1, 1)] for u in (-1, 1)])
        dets.append(float(np.linalg.det(mat / mat.sum(1, keepdims=True))))
    out["completeness_det_by_cell"] = dets
    # the outcome mean depends on the latent state through U_c alone
    out["outcome_free_of_Ut"] = bool(np.ptp(t["mu0"] - (OUTCOME["Uc"] * t["Uc"] + OUTCOME["X1"] * t["X1"]
                                                        + OUTCOME["X2"] * t["X2"])) == 0.0)
    return out


# ----------------------------------------------------------------------------- census

def set_partitions(n: int):
    """Every set partition of range(n) as a restricted growth string."""
    def grow(prefix, top):
        if len(prefix) == n:
            yield tuple(prefix)
            return
        for b in range(top + 2):
            yield from grow(prefix + [b], max(top, b))
    yield from grow([0], 0) if n else iter([()])


def census(thresholds=(2.5e-4, 5e-4, 1e-3), tolerance: float = 0.05) -> dict:
    """For every split, every population representation Z = (X, h(M_-S)): its Brier gap and its target.

    Reported per split: the smallest gap, and at each threshold how many representations pass, the range of
    their targets and how many of them miss tau by more than `tolerance`.  Two named representations are
    listed as well: `full` keeps every retained sign (what an information-preserving encoder yields) and
    `none` keeps nothing beyond X.
    """
    t = state_table()
    x = labels(t, ("X1", "X2"))
    out = {}
    for split in all_splits():
        kept, held = _kept(split), _held(t, split)
        cell = labels(t, kept)                                  # which retained-sign state each row is in
        values = np.unique(np.column_stack([t[k] for k in kept]), axis=0)
        rows = []
        for part in set_partitions(len(values)):
            z = _join(x, np.asarray(part)[cell])
            g = brier_gap(t, z, held)
            rows.append((g["gap"], adjusted(t, z), g["d_sd"], part))
        gaps, taus = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
        full = max(range(len(rows)), key=lambda i: len(set(rows[i][3])))
        none = min(range(len(rows)), key=lambda i: len(set(rows[i][3])))
        entry = {"kept": kept, "n_representations": len(rows), "min_gap": float(gaps.min()),
                 "tau_at_min_gap": float(taus[gaps.argmin()]),
                 "full": {"gap": rows[full][0], "tau": rows[full][1], "d_sd": rows[full][2]},
                 "none": {"gap": rows[none][0], "tau": rows[none][1], "d_sd": rows[none][2]}, "by_threshold": {}}
        for thr in thresholds:
            ok = gaps <= thr
            entry["by_threshold"][f"{thr:g}"] = (
                {"pass": int(ok.sum()), "tau_min": float(taus[ok].min()), "tau_max": float(taus[ok].max()),
                 "off_target": int((np.abs(taus[ok] - TAU) > tolerance).sum())} if ok.any() else {"pass": 0})
        out[split_id(split)] = entry
    return out


def main() -> int:
    ref = reference_values()
    print("reference functionals (PRD section 5)")
    ok = True
    for name, value in ref.items():
        good = abs(value - REFERENCE[name]) <= REFERENCE_TOL
        ok &= good
        print(f"  {name:16s} {value: .12f}   PRD {REFERENCE[name]: .12f}   {'ok' if good else 'MISMATCH'}")
    print("overlap range", overlap_range())
    print("witness", witness_report())
    print("census: split, representations, min gap, [gap, tau] of full and of none, pass / off-target at t = 5e-4")
    for sid, c in census().items():
        b = c["by_threshold"]["0.0005"]
        print(f"  {sid:5s} {c['n_representations']:5d}  min {c['min_gap']:.2e} (tau {c['tau_at_min_gap']:.3f})"
              f"  full [{c['full']['gap']:.2e}, {c['full']['tau']:.3f}]  none [{c['none']['gap']:.2e}, "
              f"{c['none']['tau']:.3f}]  pass {b['pass']}" + (f" off {b['off_target']} tau in "
              f"[{b['tau_min']:.3f}, {b['tau_max']:.3f}]" if b["pass"] else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
