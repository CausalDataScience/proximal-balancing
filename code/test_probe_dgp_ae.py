#!/usr/bin/env python3
"""Generation unit tests for DGP A to E (plan section 16.1) and the split/leakage tests (16.2).

Run: python3 test_probe_dgp_ae.py
Every check prints PASS or FAIL and the script exits nonzero if anything fails.
"""
from __future__ import annotations

import math
import sys

import numpy as np

import probe_dgp_ae as dgp

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' | ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    coef = dgp.coefficients()
    check("16.1 unit norms of b_x, a_x, y_x, yt_x",
          all(abs(np.linalg.norm(coef[k]) - 1) < 1e-12 for k in ("b_x", "a_x", "y_x", "yt_x")))
    check("16.1 b_good matches the eight fixed values",
          np.allclose(coef["b_good"], dgp.B_GOOD_EXPECTED, atol=1e-12),
          np.array2string(coef["b_good"], precision=4))

    n = 200000
    for name in ["A", "B", "C", "D", "E"]:
        nn = 40000 if name in ("D", "E") else n
        d = dgp.generate(name, nn, replicate=0, phase="pilot", coef=coef)
        vu, vc = d["U"].var(), d["C"].var()
        corr = np.corrcoef(d["U"], d["C"])[0, 1]
        tol = 0.03 if name != "E" else 0.05
        check(f"16.1 {name}: Var(U) and Var(C) near 1, Corr(U,C) near 0.5",
              abs(vu - 1) < tol and abs(vc - 1) < tol and abs(corr - 0.5) < tol,
              f"Var(U)={vu:.3f} Var(C)={vc:.3f} Corr={corr:.3f}")
        check(f"16.1 {name}: empirical Y(1)-Y(0) equals 1 exactly",
              np.allclose(d["Y1"] - d["Y0"], 1.0, atol=1e-12))
        pr = d["propensity"]
        check(f"16.1 {name}: structural propensity inside [0.05, 0.95]",
              pr.min() >= 0.05 - 1e-12 and pr.max() <= 0.95 + 1e-12,
              f"[{pr.min():.4f}, {pr.max():.4f}]")
        naive = d["Y"][d["A"] == 1].mean() - d["Y"][d["A"] == 0].mean()
        check(f"17.1 {name}: naive contrast has the wrong sign (Simpson at the naive level)",
              naive < 0, f"naive={naive:+.3f}")
        check(f"16.1 {name}: five blocks with block 0 equal to (I1, I2, C)",
              len(d["blocks"]) == 5 and d["blocks"][0]["v"].shape[1] == 3
              and np.allclose(d["blocks"][0]["v"][:, 2], d["C"], atol=1e-5))

    # B: derivative lower bound of the first conditional mean (plan 4.4)
    lower = 1.0 - 3.0 * dgp.B_BETA / math.sqrt(6.0)
    check("16.1 B: first conditional mean is strictly increasing in u", bool((lower > 0).all()),
          f"min lower bound={lower.min():.4f}")

    # C: loading matrix rank (plan 5.3)
    load = np.column_stack([dgp.C_LAMBDA, dgp.C_KAPPA])
    check("16.1 C: log-rate loading matrix has rank 2", np.linalg.matrix_rank(load) == 2,
          f"singular values={np.round(np.linalg.svd(load, compute_uv=False), 4)}")

    # D: orthonormal templates (plan 6.1)
    tmpl = dgp.d_templates()
    gram = np.einsum("jrpq,jspq->jrs", tmpl, tmpl)
    err = float(np.abs(gram - np.eye(3)[None]).max())
    check("16.1 D: three templates per block are orthonormal to 1e-10", err < 1e-10, f"max error={err:.2e}")
    dD = dgp.generate("D", 4000, replicate=0, phase="pilot", coef=coef)
    dim = sum(dgp.block_matrix(b).shape[1] for b in dD["blocks"])
    check("6.2 D: observed proxy dimension is 3 + 4*1024 = 4099", dim == 4099, f"dim={dim}")

    # E: pool provenance, class frequency, corruption rate (plan 7.2, 7.3)
    pool = dgp.mnist_pool()
    check("16.1 E: pool has 1000 images per class", pool["images"].shape == (10, 1000, 28, 28))
    check("16.1 E: no exact pixel duplicate across classes", len(pool["cross_class_duplicates"]) == 0,
          f"{len(pool['cross_class_duplicates'])} collisions")
    dE = dgp.generate("E", 60000, replicate=0, phase="pilot", coef=coef)
    freq = np.bincount(dE["K"], minlength=10) / len(dE["K"])
    check("16.1 E: latent class is close to uniform", float(np.abs(freq - 0.1).max()) < 0.005,
          f"max deviation={np.abs(freq - 0.1).max():.4f}")
    rates = [float((b["corrupted_class"] != dE["K"]).mean()) for b in dE["blocks"][1:]]
    check("16.1 E: corruption rate is 0.25 in every good block",
          all(abs(r - 0.25) < 0.01 for r in rates), f"rates={np.round(rates, 4)}")
    ch = dgp.e_channel_matrix()
    sv = np.linalg.svd(ch, compute_uv=False)
    check("14.2 E: corrupted-label channel is full rank", np.linalg.matrix_rank(ch) == 10,
          f"min singular value={sv.min():.4f}, condition={sv.max()/sv.min():.2f}")

    # design F (review of 2026-09-11): the exact balancing representation and the feasibility notion
    import math as _m
    dF = dgp.generate("F", 60000, replicate=0, phase="pilot", coef=coef)
    check("F: five three-dimensional blocks and a scalar covariate",
          [b["v"].shape[1] for b in dF["blocks"]] == [3] * 5 and dF["X"].shape[1] == 1)
    check("F: structural propensity inside [0.05, 0.95]",
          dF["propensity"].min() >= 0.05 - 1e-12 and dF["propensity"].max() <= 0.95 + 1e-12)
    check("F: the naive contrast has the wrong sign",
          float(dF["Y"][dF["A"] == 1].mean() - dF["Y"][dF["A"] == 0].mean()) < 0)
    roots_ok = True
    for k in range(1, 6):
        r = dgp.f_balance_coefficient(k)
        roots_ok &= abs(r * r - 3 * r + 1.0 / k) < 1e-12
    check("F: the balancing coefficient solves r^2 - 3r + 1/k = 0", roots_ok,
          f"r*(k=1..5)={[round(dgp.f_balance_coefficient(k), 4) for k in range(1, 6)]}")
    z = dgp.f_exact_representation(dF, (1,))
    check("F: the verification representation is three-dimensional and finite",
          z.shape == (60000, 3) and np.isfinite(z).all())
    feas = [s for s in dgp.all_subsets() if dgp.balance_feasible("A", s)]
    check("balance feasibility is stricter than not holding out block 0",
          len(feas) == 14 and sum(dgp.structural_valid(s) for s in dgp.all_subsets()) == 15
          and not dgp.balance_feasible("A", (1, 2, 3, 4)),
          "holding out every measurement block leaves nothing to balance with")

    # subset family (plan 8.1)
    subs = dgp.all_subsets()
    valid = sum(dgp.structural_valid(s) for s in subs)
    check("16.3 thirty nonempty proper subsets, 15 without block 0",
          len(subs) == 30 and valid == 15 and len(set(subs)) == 30, f"valid={valid}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + "; ".join(FAILURES))
        return 1
    print("all generation tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
