"""P0 for SCM family v2: exact population census of the role-blind linear representation class.

Everything in family_v2_dgp is a fixed linear combination of independent standard normals and the treatment
link is a bounded probit, so for any representation Z whose coordinates are linear in the informative
coordinates:

    tau_Z = E_Z[ E(Y | A=1, Z) - E(Y | A=0, Z) ]              two-dimensional Gauss-Hermite, exact up to order
    D^2_S(Z) = E[ {P(A=1 | Z, T_S) - P(A=1 | Z)}^2 ]           same
    balance  <=>  Cov(L, t | Z) = 0 for every informative row t of the held-out blocks

where L is the treatment index.  Pure noise coordinates are independent of everything, so the population
census lives in the informative subspace: a balancing representation with noise-free rows exists whenever
one exists at all, and adding noise to Z does not change which splits admit balance.

The class for split S is Z = (X_1, C, B v_{-S}) with B a k x r matrix on the r informative coordinates of
the retained blocks.  X enters whole, which is what "X always enters the representation" means here.

For each split the census samples the balanced set {B : Cov(L, T_S | Z) = 0} from many random starts and
records tau_Z on it.  A split is TARGET if every sampled balanced point gives tau, WRONG if none does,
MIXED if both occur, and REJECTED if no balanced point is found; for rejected splits the smallest D^2 over
the class is searched as well, because that number bounds the screen threshold from above.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.stats import norm

import family_v2_dgp as g
import positive_control_scm as pc

LATENT = ("U", "I", "C", "eta", "e0", "e1", "e2", "e3", "e4")
GH_ORDER = 96
GH_CHECK = 144
D2_ORDER = 48
D2_SEARCH_ORDER = 32
BALANCE_TOL = 1e-10
TAU_TOL = 1e-8


def rows(c_in_x: bool = True) -> dict[str, np.ndarray]:
    e = {name: np.eye(len(LATENT))[i] for i, name in enumerate(LATENT)}
    r = {"U": e["U"], "I": e["I"], "C": e["C"], "X1": e["U"] + g.KAPPA * e["eta"]}
    for j in range(4):
        r[f"M{j}"] = e["U"] + g.PROXY_SD * e[f"e{j}"]
    r["M4"] = e["U"] + g.BAD_I * e["I"] + g.PROXY_SD * e["e4"]
    r["L"] = g.TREAT["U"] * r["U"] + g.TREAT["I"] * r["I"] + g.TREAT["X1"] * r["X1"] + g.TREAT["C"] * r["C"]
    r["Ylat"] = g.OUTCOME["U"] * r["U"] + g.OUTCOME["X1"] * r["X1"] + g.OUTCOME["C"] * r["C"]
    return r


R = rows()


def block_info(c_in_x=True) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Informative names of X and of W_1..W_5.  c_in_x=False is the PRD v1 layout, kept for the design check;
    c_in_x="1c" is the SCM-1c diagnostic layout (I inside X, every block a pure measurement of U)."""
    if c_in_x == "1c":
        return g.X_INFO_1C, g.BLOCK_INFO_1C
    if c_in_x:
        return g.X_INFO, g.BLOCK_INFO
    return ("X1",), g.BLOCK_INFO[:4] + (("M0", "I", "C"),)


def stack(names) -> np.ndarray:
    return np.vstack([R[n] for n in names]) if len(names) else np.zeros((0, len(LATENT)))


def tau_z(z: np.ndarray, order: int = GH_ORDER) -> float:
    """tau_Z for a representation whose rows in the latent basis are z (X_1 and C need not be inside)."""
    mu_y, _ = pc.conditional(R["Ylat"], z)
    mu_l, var_l = pc.conditional(R["L"], z)
    cov_yl = float((R["Ylat"] - mu_y) @ (R["L"] - mu_l))
    cov = np.array([[mu_y @ mu_y, mu_y @ mu_l], [mu_y @ mu_l, mu_l @ mu_l]])
    pts, weights = pc._gh2(cov, order)
    _, e1, e0 = pc._arm_means(pts[0], pts[1], var_l, cov_yl)
    return float(g.TAU + ((e1 - e0) * weights).sum())


def discrepancy(z: np.ndarray, t: np.ndarray, order: int = D2_ORDER) -> float:
    m0, v0 = pc.conditional(R["L"], z)
    m1, v1 = pc.conditional(R["L"], np.vstack([z, t]))
    cov = np.array([[m0 @ m0, m0 @ m1], [m0 @ m1, m1 @ m1]])
    pts, weights = pc._gh2(cov, order)
    p0 = 0.1 + 0.8 * norm.cdf(pts[0] / math.sqrt(1.0 + v0))
    p1 = 0.1 + 0.8 * norm.cdf(pts[1] / math.sqrt(1.0 + v1))
    return float((((p1 - p0) ** 2) * weights).sum())


def balance_residual(z: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cov(L, t_i | Z) for every held-out informative row; all zero exactly when D^2 = 0."""
    res_l = R["L"] - pc.conditional(R["L"], z)[0]
    return np.array([float(res_l @ (row - pc.conditional(row, z)[0])) for row in t])


def all_splits() -> list[tuple[int, ...]]:
    return [s for k in range(1, g.N_BLOCKS) for s in itertools.combinations(range(g.N_BLOCKS), k)]


def split_name(split: tuple[int, ...]) -> str:
    return "S" + "".join(str(j + 1) for j in split)


def split_rows(split: tuple[int, ...], c_in_x: bool = True):
    x_names, blocks = block_info(c_in_x)
    kept = [j for j in range(g.N_BLOCKS) if j not in split]
    v = stack([n for j in kept for n in blocks[j]])
    t = stack([n for j in split for n in blocks[j]])
    return stack(x_names), v, t


def z_of(x_rows: np.ndarray, v: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.vstack([x_rows, b @ v])


def _full_rank(z: np.ndarray, count: int) -> bool:
    sv = np.linalg.svd(z, compute_uv=False)
    return int((sv > 1e-6 * sv[0]).sum()) == count


def sample_balanced(split, k: int, starts: int, seed: int, c_in_x: bool = True) -> dict:
    """Random points of the balanced set of informative rank exactly min(k, r), and tau_Z at each.

    A representation with k outputs can have informative rank below k (dependent rows, or rows spent on
    noise coordinates), so the census calls this for every rank from 1 to k and takes the union."""
    x_rows, v, t = split_rows(split, c_in_x)
    r = v.shape[0]
    kk = min(k, r)
    rng = np.random.default_rng(seed)
    taus, residuals, d2s, bs = [], [], [], []
    if kk == r:                                          # the class holds one point: every retained coordinate
        b = np.eye(r)
        res = balance_residual(z_of(x_rows, v, b), t)
        if np.max(np.abs(res)) < BALANCE_TOL:
            taus.append(tau_z(z_of(x_rows, v, b)))
            residuals.append(float(np.max(np.abs(res))))
            bs.append(b)
        return {"k": kk, "r": r, "taus": taus, "residuals": residuals, "d2": d2s, "b": bs, "starts": 1}
    for _ in range(starts):
        b0 = rng.standard_normal((kk, r))
        fun = lambda flat: balance_residual(z_of(x_rows, v, flat.reshape(kk, r)), t)
        sol = least_squares(fun, b0.ravel(), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
        b = sol.x.reshape(kk, r)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        z = z_of(x_rows, v, b)
        res = balance_residual(z, t)
        if np.max(np.abs(res)) < BALANCE_TOL and _full_rank(z, x_rows.shape[0] + kk):
            taus.append(tau_z(z))
            residuals.append(float(np.max(np.abs(res))))
            bs.append(b)
    return {"k": kk, "r": r, "taus": taus, "residuals": residuals, "b": bs, "starts": starts}


def min_discrepancy(split, k: int, starts: int, seed: int, c_in_x: bool = True) -> dict:
    """Smallest D^2 found over the class, for splits with no balanced point."""
    x_rows, v, t = split_rows(split, c_in_x)
    r = v.shape[0]
    kk = min(k, r)
    if kk == r:
        z = z_of(x_rows, v, np.eye(r))
        return {"min_d2": discrepancy(z, t), "tau_at_min": tau_z(z), "starts": 1}
    rng = np.random.default_rng(seed)
    best, best_tau = math.inf, None
    for _ in range(starts):
        b0 = rng.standard_normal((kk, r))
        f = lambda flat: discrepancy(z_of(x_rows, v, flat.reshape(kk, r)), t, D2_SEARCH_ORDER)
        out = minimize(f, b0.ravel(), method="L-BFGS-B", options={"maxiter": 400, "ftol": 1e-14, "gtol": 1e-10})
        z = z_of(x_rows, v, out.x.reshape(kk, r))
        if not _full_rank(z, x_rows.shape[0] + kk):
            continue
        d2 = discrepancy(z, t, D2_ORDER)
        if d2 < best:
            best, best_tau = d2, tau_z(z)
    return {"min_d2": best, "tau_at_min": best_tau, "starts": starts}


def reference_values() -> dict:
    x = stack(("X1", "C"))
    raw = stack(("X1", "C", "M0", "M1", "M2", "M3", "M4", "I"))
    return {"naive": tau_z(np.zeros((0, len(LATENT)))),
            "X (X_1 and C)": tau_z(x),
            "X_1 only": tau_z(stack(("X1",))),
            "raw (X, W)": tau_z(raw),
            "oracle (X, U)": tau_z(stack(("X1", "C", "U"))),
            "gh_check_raw": abs(tau_z(raw, GH_ORDER) - tau_z(raw, GH_CHECK))}


MANUSCRIPT_TREAT = {"U": 1.0, "I": 2.5, "X1": 0.2, "C": 0.3}
MANUSCRIPT_BAD_I = -0.6


def reproduce_manuscript() -> dict:
    """The manuscript's population numbers for the role-aware family (its coefficients, not family v2's),
    from this module's algebra: a regression check of the algebra itself."""
    global R
    saved = (g.TREAT, g.BAD_I, R)
    g.TREAT, g.BAD_I = MANUSCRIPT_TREAT, MANUSCRIPT_BAD_I
    R = rows()
    try:
        return _reproduce_manuscript_body()
    finally:
        g.TREAT, g.BAD_I, R = saved


def _reproduce_manuscript_body() -> dict:
    x = stack(("X1", "C"))
    out = {}
    for label, kept, root, expected in (("held out W1, r = 0.2245", ("M0", "M2", "M3", "M4"), 0.2244682054491422, 1.0),
                                        ("held out W4, r = -0.6502", ("M0", "M1", "M2", "M3"), -0.6501708893114988, -0.4432),
                                        ("held out W4, r = 2.5502", ("M0", "M1", "M2", "M3"), 2.5502, 0.9530)):
        channel = stack(kept).mean(axis=0) + root * R["I"]
        z = np.vstack([x, channel])
        held = R["M1"] if "W1" in label else R["M4"]
        out[label] = {"tau": tau_z(z), "expected": expected,
                      "balance_residual": float(balance_residual(z, held[None, :])[0])}
    return out


def design_check_c_in_w5(k: int = 2, starts: int = 200, seed: int = 97_019_900) -> dict:
    """PRD v1 put C inside W_5.  Then a representation can balance a clean held-out block while leaving C out,
    and its adjustment misses tau: completeness fails because the outcome depends on a retained coordinate
    that Z need not contain.  This check is why Revision 1.1 moves C into X."""
    sample = sample_balanced((0,), k, starts, seed, c_in_x=False)
    x1 = stack(("X1",))
    omit = np.vstack([x1, stack(("M0", "M2", "M3", "M4")).mean(axis=0) + 0.2244682054491422 * R["I"]])
    # the root for the C-free channel differs; find it on a line
    from scipy.optimize import brentq
    def res(rr):
        z = np.vstack([x1, stack(("M0", "M2", "M3", "M4")).mean(axis=0) + rr * R["I"]])
        return balance_residual(z, R["M1"][None, :])[0]
    grid = np.linspace(-3, 3, 601)
    vals = [res(rr) for rr in grid]
    roots = [brentq(res, grid[i], grid[i + 1]) for i in range(len(grid) - 1) if vals[i] * vals[i + 1] < 0]
    c_free = [{"r": rr, "tau": tau_z(np.vstack([x1, stack(("M0", "M2", "M3", "M4")).mean(axis=0) + rr * R["I"]]))}
              for rr in roots]
    taus = np.array(sample["taus"])
    return {"split": "held out W1", "k": k, "balanced_points": len(taus),
            "tau_min": float(taus.min()) if len(taus) else None,
            "tau_max": float(taus.max()) if len(taus) else None,
            "share_within_1e-6_of_tau": float(np.mean(np.abs(taus - 1) < 1e-6)) if len(taus) else None,
            "c_free_channel_roots": c_free}


def census(level: g.Level, starts: int, min_starts: int, seed: int) -> dict:
    """Split census for one level.  The SCM-1c layout swaps the rows (no contamination) for the duration."""
    global R
    layout = "1c" if level.variant == "1c" else True
    saved = (g.BAD_I, R)
    if level.variant == "1c":
        g.BAD_I = 0.0
        R = rows()
    try:
        return _census(level, starts, min_starts, seed, layout)
    finally:
        g.BAD_I, R = saved


def _census(level: g.Level, starts: int, min_starts: int, seed: int, layout) -> dict:
    rows_out = []
    for idx, split in enumerate(all_splits()):
        t0 = time.time()
        r_info = split_rows(split, layout)[1].shape[0]
        per_rank, all_taus, all_res, first_b = {}, [], [], None
        for rank in range(1, min(level.k, r_info) + 1):
            smp = sample_balanced(split, rank, starts, seed + 100 * idx + rank, layout)
            per_rank[rank] = len(smp["taus"])
            all_taus += smp["taus"]
            all_res += smp["residuals"]
            if first_b is None and smp["b"]:
                first_b = smp["b"][0]
        taus = np.array(all_taus)
        smp = {"residuals": all_res, "b": [first_b] if first_b is not None else []}
        entry = {"split": split_name(split), "held_out": [f"W{j + 1}" for j in split],
                 "ranks_searched": list(per_rank), "balanced_points_by_rank": per_rank,
                 "retained_informative": r_info, "balanced_points": int(len(taus)), "starts_per_rank": starts}
        if len(taus):
            on_target = np.abs(taus - g.TAU) < TAU_TOL
            entry.update({"tau_min": float(taus.min()), "tau_max": float(taus.max()),
                          "tau_quantiles": [float(q) for q in np.quantile(taus, [0.05, 0.25, 0.5, 0.75, 0.95])],
                          "share_on_target": float(on_target.mean()),
                          "max_balance_residual": float(max(smp["residuals"])),
                          "kind": "target" if on_target.all() else ("wrong" if not on_target.any() else "mixed")})
            z = z_of(*split_rows(split, layout)[:2], smp["b"][0])
            entry["d2_at_first_point"] = discrepancy(z, split_rows(split, layout)[2])
        else:
            best = None
            for rank in range(1, min(level.k, r_info) + 1):
                md = min_discrepancy(split, rank, min_starts, seed + 100 * idx + 50 + rank, layout)
                if best is None or md["min_d2"] < best["min_d2"]:
                    best = dict(md, rank=rank)
            entry.update({"kind": "rejected", "min_d2_found": best["min_d2"], "tau_at_min_d2": best["tau_at_min"],
                          "rank_at_min_d2": best["rank"], "min_d2_starts_per_rank": min_starts})
        entry["seconds"] = round(time.time() - t0, 2)
        rows_out.append(entry)
        print(f"  {level.name} {entry['split']:6s} {entry['kind']:8s} "
              + (f"tau [{entry['tau_min']:+.4f}, {entry['tau_max']:+.4f}] n={entry['balanced_points']}"
                 if entry["kind"] != "rejected" else f"min D2 {entry['min_d2_found']:.3e} (tau there {entry['tau_at_min_d2']})")
              + f"  ({entry['seconds']} s)", flush=True)
    kinds = {k: [r["split"] for r in rows_out if r["kind"] == k] for k in ("target", "wrong", "mixed", "rejected")}
    rejected_min = min((r["min_d2_found"] for r in rows_out if r["kind"] == "rejected"), default=None)
    n_candidates = len(kinds["target"]) + len(kinds["wrong"]) + len(kinds["mixed"])
    return {"level": level.name, "k": level.k, "dim": level.dim, "class": "Z = (X_1, C, B v), informative rank of B <= k",
            "rows": rows_out, "kinds": kinds,
            "summary": {"target_splits": len(kinds["target"]), "wrong_splits": len(kinds["wrong"]),
                        "mixed_splits": len(kinds["mixed"]), "rejected_splits": len(kinds["rejected"]),
                        "p_tau_upper": len(kinds["target"]) / n_candidates if n_candidates else None,
                        "smallest_rejected_d2": rejected_min}}


def main(argv: list[str]) -> int:
    starts = int(argv[1]) if len(argv) > 1 else 60
    min_starts = int(argv[2]) if len(argv) > 2 else 12
    out_path = Path(__file__).resolve().parent.parent / "results" / "family_v2" / "p0_v3.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; P0 output is write-once")
    t0 = time.time()
    ref = reference_values()
    print("reference:", {k: round(v, 6) for k, v in ref.items()})
    repro = reproduce_manuscript()
    print("manuscript reproduction:", {k: (round(v["tau"], 4), v["expected"]) for k, v in repro.items()})
    check = design_check_c_in_w5()
    print("design check, C inside W5:", check)
    levels = {}
    for name, level in g.LEVELS.items():
        print(f"census {name} (k = {level.k})")
        levels[name] = census(level, starts, min_starts, 97_010_000 + 3_000 * level.index)
    payload = {"artifact": "family_v2_p0", "prd": "memo/2026-09-18-family-v2-role-blind-dimension-prd-v1.md",
               "supersedes": {"path": "results/family_v2/p0_v2.json",
                              "reason": "v2 used the manuscript coefficients (treatment index U + 2.5 I, contamination -0.6 I); "
                                        "v3 is the PRD v2 design: index 1.5 U + 2.0 I, contamination -3.0 I, C inside X, "
                                        "k = 2 at every level, census over informative rank <= k"},
               "coefficients": {"treat": g.TREAT, "bad_i": g.BAD_I, "proxy_sd": g.PROXY_SD, "noise_sd": g.NOISE_SD,
                                "outcome": g.OUTCOME, "outcome_noise": g.OUTCOME_NOISE, "kappa": g.KAPPA},
               "levels_spec": {n: {"d_x": l.d_x, "d_blocks": list(l.d_blocks), "k": l.k, "dim": l.dim} for n, l in g.LEVELS.items()},
               "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
               "method": "exact linear-Gaussian algebra with a bounded probit link; Gauss-Hermite order "
                         f"{GH_ORDER} for tau and {D2_ORDER} for D^2; balanced points by least squares on the "
                         f"balance residual from {starts} random starts per split; tolerance {BALANCE_TOL}",
               "reference_values": ref, "manuscript_reproduction": repro,
               "design_check_c_in_w5": check, "levels": levels,
               "rotation_digests": {n: g.rotation_digest(l) for n, l in g.LEVELS.items()},
               "seconds": round(time.time() - t0, 1)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(f"written {out_path} in {payload['seconds']} s")
    return 0


def main_variant(name: str, starts: int, min_starts: int) -> int:
    """P0 for a diagnostic variant (PRD section 7); written next to p0_v3.json, write-once."""
    level = g.level(name)
    out_path = Path(__file__).resolve().parent.parent / "results" / "family_v2" / f"p0_v3_{name.lower().replace('-', '')}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; P0 output is write-once")
    t0 = time.time()
    global R
    saved = (g.BAD_I, R)
    g.BAD_I = 0.0
    R = rows()
    try:
        x = stack(g.X_INFO_1C)
        raw = stack(g.X_INFO_1C + ("M0", "M1", "M2", "M3", "M4"))
        ref = {"naive": tau_z(np.zeros((0, len(LATENT)))), "X": tau_z(x), "raw (X, W)": tau_z(raw),
               "oracle (X, U)": tau_z(stack(g.X_INFO_1C + ("U",)))}
    finally:
        g.BAD_I, R = saved
    print("reference:", {k: round(v, 4) for k, v in ref.items()})
    cen = census(level, starts, min_starts, 97_019_000)
    payload = {"artifact": "family_v2_p0_variant", "variant": name,
               "prd": "memo/2026-09-18-family-v2-role-blind-dimension-prd-v1.md",
               "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
               "coefficients": {"treat": g.TREAT, "bad_i": 0.0, "proxy_sd": g.PROXY_SD},
               "layout": {"x": list(g.X_INFO_1C), "blocks": [list(b) for b in g.BLOCK_INFO_1C]},
               "reference_values": ref, "census": cen, "seconds": round(time.time() - t0, 1)}
    out_path.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(f"written {out_path}; summary {cen['summary']}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--variant":
        raise SystemExit(main_variant(sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))
    raise SystemExit(main(sys.argv))


# ----------------------------------------------------------------------------- learned-target evaluator

def _pad(row: np.ndarray, width: int) -> np.ndarray:
    return np.concatenate([row, np.zeros(width - len(row))]) if width > len(row) else row


def tau_z_ext(z: np.ndarray, order: int = GH_ORDER) -> float:
    """tau_Z when Z also loads on extra independent standard-normal coordinates (columns beyond the latent basis)."""
    w = z.shape[1]
    ylat, l = _pad(R["Ylat"], w), _pad(R["L"], w)
    mu_y, _ = pc.conditional(ylat, z)
    mu_l, var_l = pc.conditional(l, z)
    cov_yl = float((ylat - mu_y) @ (l - mu_l))
    cov = np.array([[mu_y @ mu_y, mu_y @ mu_l], [mu_y @ mu_l, mu_l @ mu_l]])
    pts, weights = pc._gh2(cov, order)
    _, e1, e0 = pc._arm_means(pts[0], pts[1], var_l, cov_yl)
    return float(g.TAU + ((e1 - e0) * weights).sum())


def discrepancy_ext(z: np.ndarray, t: np.ndarray, order: int = D2_ORDER) -> float:
    w = z.shape[1]
    l = _pad(R["L"], w)
    tt = np.column_stack([t, np.zeros((t.shape[0], w - t.shape[1]))]) if w > t.shape[1] else t
    m0, v0 = pc.conditional(l, z)
    m1, v1 = pc.conditional(l, np.vstack([z, tt]))
    cov = np.array([[m0 @ m0, m0 @ m1], [m0 @ m1, m1 @ m1]])
    pts, weights = pc._gh2(cov, order)
    p0 = 0.1 + 0.8 * norm.cdf(pts[0] / math.sqrt(1.0 + v0))
    p1 = 0.1 + 0.8 * norm.cdf(pts[1] / math.sqrt(1.0 + v1))
    return float((((p1 - p0) ** 2) * weights).sum())


def learned_rows(level: g.Level, prep, b: np.ndarray, kept: list[int]) -> np.ndarray:
    """Rows of the learned Z = (X features, V B^T) in the latent basis, with the pure-noise part folded into
    extra independent coordinates.  Evaluator only: it undoes the rotations the learner never saw."""
    q = g.rotations(level)
    lat_rows, noise_blocks = [], []
    # X features: (X - mu) / sd with X = Q_X [info; noise]
    qx = q["X"]
    load_x = qx.T / prep.x_sd[None, :]                  # stacked coordinate i -> feature column
    x_names = g.x_info(level)
    c = len(x_names)
    for col in range(level.d_x):
        lat_rows.append(sum(load_x[i, col] * R[x_names[i]] for i in range(c)))
    noise_x = load_x[c:, :]                              # (d_x - c) x d_x
    # block features, then B
    feat_lat, feat_noise = [], []
    for j in kept:
        mu, proj, sd = prep.blocks[j]
        qj = q[f"W{j + 1}"]
        load = (qj.T @ proj) / sd[None, :]               # stacked coordinate -> feature column
        names = g.block_info(level)[j]
        cj = len(names)
        for col in range(load.shape[1]):
            feat_lat.append(sum(load[i, col] * R[names[i]] for i in range(cj)))
        feat_noise.append(load[cj:, :])
    feat_lat = np.array(feat_lat)                        # q x 9
    z_lat = b @ feat_lat                                 # k x 9
    # noise: independent across blocks; build the joint loading matrix on all noise coordinates
    widths = [m.shape[0] for m in feat_noise]
    total_noise = noise_x.shape[0] + sum(widths)
    rows_x_noise = np.zeros((level.d_x, total_noise))
    rows_x_noise[:, :noise_x.shape[0]] = noise_x.T
    fn = np.zeros((feat_lat.shape[0], total_noise))
    r0, c0 = 0, noise_x.shape[0]
    for m in feat_noise:
        fn[r0:r0 + m.shape[1], c0:c0 + m.shape[0]] = m.T
        r0 += m.shape[1]
        c0 += m.shape[0]
    z_noise = np.vstack([rows_x_noise, b @ fn]) * g.NOISE_SD          # rows x noise coordinates (unit variance)
    z_latent = np.vstack([np.array(lat_rows), z_lat])
    if z_noise.shape[1] == 0 or not np.any(z_noise):
        return z_latent
    # compress the noise to an orthonormal basis of its row space so the width stays small
    u, s, vt = np.linalg.svd(z_noise, full_matrices=False)
    keep = s > 1e-12 * max(s[0], 1.0)
    compressed = u[:, keep] * s[keep]
    return np.hstack([z_latent, compressed])


def learned_target(level: g.Level, prep, b: np.ndarray, split: tuple[int, ...]) -> dict:
    """Population tau_Z and D^2 of a learned representation.  For SCM-1c the rows are rebuilt without the
    contamination for the duration of the call."""
    global R
    saved = (g.BAD_I, R)
    if level.variant == "1c":
        g.BAD_I = 0.0
        R = rows()
    try:
        kept = [j for j in range(g.N_BLOCKS) if j not in split]
        z = learned_rows(level, prep, b, kept)
        t = split_rows(split, "1c" if level.variant == "1c" else True)[2]
        return {"tau_z": tau_z_ext(z), "d2": discrepancy_ext(z, t)}
    finally:
        g.BAD_I, R = saved
