#!/usr/bin/env python3
"""Algorithm tests for the Brier-neural implementation (plan section 16.2 and 16.3)."""
from __future__ import annotations

import math
import sys

import copy
import pathlib

import numpy as np
import torch

import probe_brier_neural as pbn
import probe_dgp_ae as dgp
import run_dgp_ae as runner

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' | ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    torch.manual_seed(0)
    d = dgp.generate("A", 900, replicate=0, phase="pilot")
    S = (0, 2)
    rest = [b for b in range(5) if b not in S]
    specs_v, specs_t = pbn.block_specs(d, rest), pbn.block_specs(d, list(S))

    # 16.3 and fourth review: the augmented critic shares the base trunk, so containment holds by construction
    base = pbn.BaseCritic(pbn.Z_DIM)
    aug = pbn.AugmentedCritic(base, specs_t, 20)
    shared = all(any(p is q for q in aug.parameters()) for p in base.parameters())
    check("8.5 the augmented critic shares every base parameter rather than copying them", shared)
    z = torch.randn(50, pbn.Z_DIM)
    st = pbn.build_standardizers(d, np.arange(900))
    tp = pbn.make_parts(d, list(S), np.arange(50), st)
    x = torch.as_tensor(st["X"](d["X"][:50]))
    with torch.no_grad():
        same = torch.allclose(aug(z, tp, x), base(z), atol=1e-7)
    check("16.3 augmented critic with zero residual equals the base critic", bool(same))

    # 16.2 split and leakage
    f = runner.folds(900, "A", 900, 0, "pilot")
    union = np.concatenate(f)
    check("16.2 the three folds are disjoint and cover the sample",
          len(np.unique(union)) == 900 and sum(len(x) for x in f) == 900)
    e_counts = np.zeros(900, dtype=int)
    for k in range(3):
        e_counts[f[(k + 2) % 3]] += 1
    check("16.2 every unit is the evaluation fold exactly once", bool((e_counts == 1).all()))
    d_fit, d_val, d_audit = runner.d_subsplit(f[0], "A", 900, 0, 0, "pilot")
    sizes = (len(d_fit), len(d_val), len(d_audit))
    check("16.2 D splits 50/25/25 into fit, val and audit",
          abs(sizes[0] / len(f[0]) - 0.5) < 0.02 and abs(sizes[1] / len(f[0]) - 0.25) < 0.02
          and len(np.unique(np.concatenate([d_fit, d_val, d_audit]))) == len(f[0]), f"sizes={sizes}")

    # 16.2 the standardizer is fitted once on D_fit and not refit
    st2 = pbn.build_standardizers(d, d_fit)
    mu_before = st2["X"].mu.copy()
    _ = pbn.make_parts(d, rest, f[2], st2)
    check("16.2 standardizer statistics are unchanged after use on other folds",
          bool(np.array_equal(mu_before, st2["X"].mu)))

    # 16.3 the candidate family
    subs = dgp.all_subsets()
    check("16.3 the family is the 30 nonempty proper subsets with no pruning",
          len(subs) == 30 and all(0 < len(s) < 5 for s in subs))

    # 16.3 aggregation behaviour
    agg0 = pbn.aggregate([], np.zeros(0), 0.1)
    check("16.3 no survivor returns no_return", agg0["return_kind"] == "no_return" and agg0["output"] is None)
    agg1 = pbn.aggregate([1], np.array([0.7]), 0.1)
    check("16.3 a single survivor is flagged as such",
          agg1["return_kind"] == "single_survivor" and abs(agg1["output"] - 0.7) < 1e-12)
    tie = pbn.aggregate([1, 2, 3, 4], np.array([0.0, 0.02, 1.0, 1.02]), 0.02)
    check("16.3 tied largest components return a set and no point value",
          tie["return_kind"] == "tied_largest" and tie["output"] is None and len(tie["candidates"]) == 2,
          f"candidates={np.round(tie['candidates'], 3)}")
    even = pbn.aggregate([1, 2, 3, 4], np.array([0.0, 0.1, 0.2, 0.3]), 1.0)
    check("16.3 an even component median is the mean of the middle two",
          abs(even["output"] - 0.15) < 1e-12, f"output={even['output']:.4f}")

    # 8.10 the multiplier bootstrap shares one draw across candidates
    rng = np.random.default_rng(0)
    psi = rng.standard_normal((4, 600))
    r1 = pbn.linking_radius(psi, psi.mean(1), 5)
    r2 = pbn.linking_radius(psi, psi.mean(1), 5)
    check("8.10 the linking radius is reproducible from its seed", abs(r1 - r2) < 1e-12, f"rho={r1:.4f}")
    check("8.10 the radius is on the scale of the largest standard error",
          0.5 * psi.std(1).max() / np.sqrt(600) < r1 < 6 * psi.std(1).max() / np.sqrt(600), f"rho={r1:.4f}")

    # 8.6 and the review of 2026-09-11: the selected checkpoint must come from a trained representation
    import probe_brier_neural as _pbn
    d_small = dgp.generate("A", 900, replicate=1, phase="pilot")
    fit, val = np.arange(0, 400), np.arange(400, 600)
    st_s = pbn.build_standardizers(d_small, fit)
    orig = _pbn.MAX_EPOCHS
    torch.manual_seed(0)
    baseline = _pbn.Representation(pbn.block_specs(d_small, [1, 2, 3, 4]), 20)
    before = copy.deepcopy([p.detach().clone() for p in baseline.parameters()])
    _pbn.MAX_EPOCHS = 60
    fitted = _pbn.train_representation(d_small, (0,), fit, val, st_s, (1, 2, 3))
    check("8.6 the accepted checkpoint carries at least the required number of encoder updates",
          fitted["encoder_updates"] >= _pbn.MIN_ENCODER_UPDATES and fitted["epoch"] >= 0,
          f"encoder updates at checkpoint={fitted['encoder_updates']} (floor {_pbn.MIN_ENCODER_UPDATES}), epoch={fitted['epoch']}")
    check("8.6 the reported count is the one at the checkpoint, not at the end of training",
          fitted["encoder_updates"] <= fitted["encoder_updates_at_end"],
          f"checkpoint={fitted['encoder_updates']} end={fitted['encoder_updates_at_end']}")
    _pbn.MAX_EPOCHS = 4
    short = _pbn.train_representation(d_small, (0,), fit, val, st_s, (1, 2, 3))
    check("8.6 a run too short to reach the floor is marked instead of silently accepted",
          short["epoch"] == -1, f"epoch={short['epoch']}, encoder updates={short['encoder_updates']}")
    _pbn.MAX_EPOCHS = orig
    after = [p.detach().clone() for p in fitted["enc"].parameters()]
    moved = any(not torch.allclose(a, b) for a, b in zip(after, before)) if len(after) == len(before) else True
    check("8.6 the selected parameters differ from a freshly initialised representation", moved)
    check("8.6 restart eligibility is reported", "any_restart_eligible" in fitted)

    # fourth review: the F15 test must separate the fix from the bug it replaced
    import probe_brier_neural as _p
    lo, hi = _p.PROPENSITY_BOUND
    lg = torch.tensor([-6.0, 0.0, 6.0])
    pb = _p.bounded_prob(lg).numpy()
    check("F15 the bounded map keeps probabilities inside the structural interval",
          bool((pb >= lo - 1e-6).all() and (pb <= hi + 1e-6).all()), f"p={np.round(pb, 4)}")
    # A narrow bound makes the difference between the fixed path and the old one large enough to see: under
    # [0.05, 0.95] the spurious map p -> 0.05 + 0.9 p moves a probability by at most 0.05, which the fit error
    # hides.  Under [0.3, 0.7] the same mistake moves it by up to 0.18.
    saved_bound = _p.PROPENSITY_BOUND
    _p.PROPENSITY_BOUND = (0.3, 0.7)
    lo2, hi2 = _p.PROPENSITY_BOUND
    rng = np.random.default_rng(0)
    feat = rng.standard_normal((8000, 3)).astype(np.float32)
    truth = np.clip(0.32 + 0.36 / (1 + np.exp(-(1.5 * feat[:, 0] + 0.8 * feat[:, 1] ** 2 - 1.0))), lo2, hi2)
    a = (rng.uniform(size=8000) < truth).astype(float)
    y = a + feat[:, 1] + rng.standard_normal(8000)
    nz = _p.fit_nuisances(feat, a, y, 5)
    est = _p.propensity(nz, feat)
    err = float(np.mean(np.abs(est - truth)))
    err_const = float(np.mean(np.abs(0.5 - truth)))
    err_double = float(np.mean(np.abs(lo2 + (hi2 - lo2) * est - truth)))
    _p.PROPENSITY_BOUND = saved_bound
    check("F15 the fitted propensity tracks the bounded truth", err < 0.03,
          f"mean absolute error={err:.4f}")
    check("F15 the test rejects a constant predictor", err_const > 3 * err,
          f"constant 0.5 error={err_const:.4f} against {err:.4f}")
    check("F15 the test rejects the transform applied twice",
          err_double > 2 * err and err_double > 0.05,
          f"double-transform error={err_double:.4f} against {err:.4f}, tolerance 0.03")

    # fourth review: the screen rule is tested through the function the runner calls
    good = dict(refit_gap=0.0, refit_se=0.001, overlap_pass=True, restart_eligible=True,
                epoch=5, encoder_updates=_p.MIN_ENCODER_UPDATES)
    check("F18 a well-trained candidate with a small audit gap passes",
          pbn.screen_decision(**good)["screen_pass"])
    check("F18 a candidate below the training floor is rejected",
          not pbn.screen_decision(**{**good, "encoder_updates": 4})["screen_pass"]
          and not pbn.screen_decision(**{**good, "epoch": -1})["screen_pass"])
    check("F18 a large audit gap is rejected",
          not pbn.screen_decision(**{**good, "refit_gap": 0.5})["screen_pass"])
    saved = _p.PROPENSITY_BOUND
    _p.PROPENSITY_BOUND = None
    unbounded = pbn.screen_decision(**{**good, "overlap_pass": False})
    _p.PROPENSITY_BOUND = saved
    check("F18 the overlap condition applies again once the bound is switched off",
          not unbounded["screen_pass"] and "overlap" in unbounded["reject_reasons"],
          f"reasons={unbounded['reject_reasons']}")
    check("F18 the overlap condition does not fire while the bound is in force",
          pbn.screen_decision(**{**good, "overlap_pass": False})["screen_pass"])

    # third review: feasibility is design specific
    check("F16 design F admits an exact representation when every measurement block is audited",
          dgp.balance_feasible("F", (1, 2, 3, 4)) and not dgp.balance_feasible("A", (1, 2, 3, 4))
          and sum(dgp.balance_feasible("F", s) for s in dgp.all_subsets()) == 15)

    # the review of 2026-09-11: the linking radius ignores units that were never evaluated
    m = np.full((3, 900), np.nan)
    m[:, :300] = 0.4
    r_partial = pbn.linking_radius(m, np.full(3, 0.4), 3)
    check("8.10 a radius over identical evaluated scores is zero", abs(r_partial) < 1e-9, f"rho={r_partial:.6f}")

    # the review of 2026-09-11, round 5: a degenerate audit statistic must not be read as a pass
    for label, bad in (("a non-finite gap", dict(refit_gap=float("nan"))),
                       ("a non-finite standard error", dict(refit_se=float("nan"))),
                       ("an infinite standard error beside a huge gap", dict(refit_gap=100.0,
                                                                             refit_se=float("inf"))),
                       ("a negative standard error", dict(refit_se=-1.0))):
        out = pbn.screen_decision(**{**good, **bad})
        check(f"F23 the screen rejects {label}",
              not out["screen_pass"] and "audit statistic not finite" in out["reject_reasons"],
              f"pass={out['screen_pass']} reasons={out['reject_reasons']}")

    # round 5: the checkpoint search must not prefer a more negative validation gap
    check("F24 a gap closer to zero outranks a more negative one",
          pbn.checkpoint_key(-0.01) < pbn.checkpoint_key(-0.10))
    check("F24 a small positive gap outranks every negative gap",
          pbn.checkpoint_key(0.001) < pbn.checkpoint_key(-0.10)
          and pbn.checkpoint_key(0.001) < pbn.checkpoint_key(-0.01))
    check("F24 among positive gaps the smaller one still wins",
          pbn.checkpoint_key(0.01) < pbn.checkpoint_key(0.20))
    check("F24 a non-finite gap is never selected",
          pbn.checkpoint_key(0.99) < pbn.checkpoint_key(float("nan")))
    ordered = sorted([-0.10, 0.20, -0.01, 0.001, float("nan"), 0.0], key=pbn.checkpoint_key)
    check("F24 the full ordering puts zero first and the worst negative gap fourth",
          ordered[0] == 0.0 and ordered[1] == 0.001 and ordered[2] == -0.01 and ordered[3] == -0.10,
          f"order={ordered}")

    # the review of 2026-09-12 (round 7), PRD-1: the comparison rule itself, in both modes
    saved_mode = _p.TRAINING_MODE
    md = _p.MIN_DELTA
    for mode in ("variant", "manuscript"):
        _p.TRAINING_MODE = mode
        check(f"PRD-1 [{mode}] the key is always a pair of floats, so two keys can be compared",
              all(isinstance(pbn.checkpoint_key(g), tuple) and len(pbn.checkpoint_key(g)) == 2
                  and all(isinstance(q, float) for q in pbn.checkpoint_key(g))
                  for g in (0.1, -0.1, 0.0, float("nan"))))
        check(f"PRD-1 [{mode}] positive to smaller positive is an improvement",
              pbn.checkpoint_improves(0.005, 0.020))
        check(f"PRD-1 [{mode}] positive to larger positive is not",
              not pbn.checkpoint_improves(0.020, 0.005))
        check(f"PRD-1 [{mode}] negative to a smaller magnitude negative is an improvement",
              pbn.checkpoint_improves(-0.001, -0.100))
        check(f"PRD-1 [{mode}] a non-finite candidate never wins",
              not pbn.checkpoint_improves(float("nan"), 0.5)
              and not pbn.checkpoint_improves(float("inf"), 0.5))
        check(f"PRD-1 [{mode}] any finite candidate beats a non-finite incumbent, which is the first epoch",
              pbn.checkpoint_improves(0.9, float("inf")) and pbn.checkpoint_improves(-0.9, float("nan")))
        check(f"PRD-1 [{mode}] an identical gap is not an improvement", not pbn.checkpoint_improves(0.01, 0.01))
        check(f"PRD-1 [{mode}] a change just under MIN_DELTA is not an improvement",
              not pbn.checkpoint_improves(0.01 - md * 0.5, 0.01))
        check(f"PRD-1 [{mode}] a change just over MIN_DELTA is an improvement",
              pbn.checkpoint_improves(0.01 - md * 2.0, 0.01))
    _p.TRAINING_MODE = "manuscript"
    check("PRD-1 [manuscript] a positive gap never beats a non-positive one",
          not pbn.checkpoint_improves(0.001, -0.010) and pbn.checkpoint_improves(-0.010, 0.001))
    check("PRD-1 [manuscript] among epochs at objective zero the one closest to zero wins",
          pbn.checkpoint_improves(-0.001, -0.010))
    check("PRD-1 [manuscript] the objective is the clipped gap",
          pbn.objective(-0.3) == 0.0 and pbn.objective(0.02) == 0.02)
    _p.TRAINING_MODE = "variant"
    check("PRD-1 [variant] a small positive gap beats a large negative one",
          pbn.checkpoint_improves(0.001, -0.010))
    _p.TRAINING_MODE = saved_mode

    # PRD-1 integration: both modes must run encoder updates and select a checkpoint, not just order keys
    import staged_dgp_f as sf
    _p.ENCODER_FACTORY = sf.LinearRepresentation
    _saved_z = _p.Z_DIM
    _p.Z_DIM = sf.LINEAR_Z_DIM
    small = dgp.generate("F", 2000, replicate=1, phase="pilot", coef=dgp.coefficients())
    i_fit, i_val, _ = sf.three_way(np.arange(2000), 1)
    st_small = pbn.build_standardizers(small, i_fit)
    for mode in ("variant", "manuscript"):
        _p.TRAINING_MODE = mode
        try:
            r = pbn.train_representation(small, (1,), i_fit, i_val, st_small, (1, 0))
            w0 = r["init_state"]["lin.weight"].numpy()
            w1 = r["enc"].lin.weight.detach().numpy()
            moved = float(np.abs(w1 - w0).max())
            ok = (r["epoch"] >= 0 and r["encoder_updates"] >= _p.MIN_ENCODER_UPDATES and moved > 1e-6
                  and math.isfinite(r["val_gap"]))
            check(f"PRD-1 [{mode}] training runs end to end and selects a trained checkpoint", ok,
                  f"epoch={r['epoch']}, encoder updates={r['encoder_updates']}, "
                  f"max weight change={moved:.4f}, val gap={r['val_gap']:+.6f}")
            check(f"PRD-1 [{mode}] the reported objective matches the selected gap",
                  abs(r["val_objective"] - pbn.objective(r["val_gap"])) < 1e-12)
        except Exception as exc:
            check(f"PRD-1 [{mode}] training runs end to end and selects a trained checkpoint", False,
                  f"{type(exc).__name__}: {exc}")
    _p.TRAINING_MODE = saved_mode
    _p.ENCODER_FACTORY = None
    _p.Z_DIM = _saved_z

    # PRD-2: parameter separation, independent refit, and the two objectives kept apart
    _p.ENCODER_FACTORY = sf.LinearRepresentation
    _p.Z_DIM = sf.LINEAR_Z_DIM
    mid = dgp.generate("F", 3000, replicate=9, phase="pilot", coef=dgp.coefficients())
    m_fit, m_val, m_aud = sf.three_way(np.arange(3000), 9)
    st_mid = pbn.build_standardizers(mid, m_fit)
    fits = {}
    for mode in ("variant", "manuscript"):
        _p.TRAINING_MODE = mode
        fits[mode] = pbn.train_representation(mid, (1,), m_fit, m_val, st_mid, (9, 0))
    check("PRD-2 in manuscript mode the augmented critic does not share the base trunk",
          fits["manuscript"].q1.base is not fits["manuscript"].q0 if False else
          fits["manuscript"]["q1"].base is not fits["manuscript"]["q0"])
    check("PRD-2 in variant mode it does share the base trunk",
          fits["variant"]["q1"].base is fits["variant"]["q0"])

    # a step on one critic's own loss must leave the other critic's parameters untouched
    _p.TRAINING_MODE = "manuscript"
    f = fits["manuscript"]
    z_all = torch.as_tensor(np.zeros((3000, sf.LINEAR_Z_DIM), dtype=np.float32))
    a_t = torch.as_tensor(mid["A"].astype(np.float32))
    before = [q.detach().clone() for q in f["q0"].parameters()]
    opt1 = torch.optim.SGD(f["q1"].parameters(), lr=0.1)
    tp = pbn.make_parts(mid, [1], np.arange(3000), st_mid)
    x_t = torch.as_tensor(st_mid["X"](mid["X"]))
    sel = torch.arange(256)
    opt1.zero_grad()
    ((a_t[sel] - f["q1"](z_all[sel], pbn.slice_parts(tp, sel), x_t[sel])) ** 2).mean().backward()
    opt1.step()
    after = [q.detach().clone() for q in f["q0"].parameters()]
    check("PRD-2 a step on the augmented critic's own loss does not move the base critic",
          all(bool(torch.equal(b, a)) for b, a in zip(before, after)))

    diag = pbn.critic_refit_risk_difference(mid, (1,), f, m_fit, m_val, m_aud, st_mid, 3, restarts=2)
    check("PRD-2 the refit diagnostic reports fit, validation and audit risk for both critics",
          all(f"{c}_refit_{n}" in diag and f"{c}_excess_{n}" in diag
              for c in ("R0", "R1") for n in ("fit", "val", "audit")))
    check("PRD-2 the refit diagnostic records how many restarts it searched and disclaims the class minimum",
          diag["restarts"] == 2 and "not a certificate" in diag["note"])
    check("PRD-2 the excess risk is the loop risk minus the refit risk on the same split",
          abs((diag["R1_loop_audit"] - diag["R1_refit_audit"]) - diag["R1_excess_audit"]) < 1e-12)

    rep = pbn.encoder_objective_report(mid, (1,), f, m_fit, st_mid)
    check("PRD-2 the manuscript objective and the minibatch surrogate are reported separately",
          {"full_sample_gap", "manuscript_objective", "minibatch_surrogate",
           "surrogate_minus_objective"} <= set(rep))
    check("PRD-2 the manuscript objective is the clipped full-sample gap",
          abs(rep["manuscript_objective"] - max(rep["full_sample_gap"], 0.0)) < 1e-12)
    check("PRD-2 the surrogate is at least the clipped full-sample gap, by Jensen",
          rep["minibatch_surrogate"] >= rep["manuscript_objective"] - 1e-9,
          f"surrogate {rep['minibatch_surrogate']:.6f}, objective {rep['manuscript_objective']:.6f}")
    _p.TRAINING_MODE = saved_mode
    _p.ENCODER_FACTORY = None
    _p.Z_DIM = _saved_z

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + "; ".join(FAILURES))
        return 1
    print("all algorithm tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
