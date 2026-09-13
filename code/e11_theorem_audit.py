"""E11: a literal audit of Algorithm 1 against Theorem thm:probe-search-error.

The point is not a new performance claim.  It is to connect each line of Algorithm 1 and each hypothesis of
the finite-sample theorem to one observable check, on a positive-control SCM where the theorem's quantities
can be computed rather than guessed.

What the theorem says.  Condition on the learning samples.  Let B-hat be the screened set, theta_S the
population adjustment value of the learned representation for split S, n(c) the count of screened splits with
theta_S = c, L the number of distinct non-tau values, and Delta = n(tau)/N - max_{c != tau} n(c)/N.  If
  (a) Delta > 0,
  (b) with probability at least 1 - delta over the evaluation sample, |theta-hat_S - theta_S| <= rho for all
      S in B-hat,
  (c) any two distinct values in {theta_S} differ by more than 4 rho,
then P(|tau-hat - tau| <= rho) >= 1 - delta - P(R=0) - L E[exp(-R Delta^2 / 2)], with R hypergeometric on
(T, N, m), and a candidate set counts as a failure.

How each piece is obtained here.
  theta_S      exact.  Every map in the encoder class is linear in nine independent standard normals and the
               treatment link is a probit, so the population adjustment value has a closed form.
  rho          the manuscript's own formula, sigma_S / sqrt(n_E delta/N) + q_e(q_0+q_1)/eta, with the
               nuisance errors and the score variance measured against the exact nuisances on an independent
               evaluator draw.  The evaluation sample is never used to choose rho.
  B            the exact finite sums of Equation (sampling-failure-terms).
  success      the literal event: a single returned value within rho of tau.  Candidate sets, ties and empty
               returns are failures.

The encoder class is finite and fixed before the run.  That is a deliberate restriction: the theorem needs
several target splits to carry exactly the same tau, which a continuous learned coefficient gives only
approximately.  This is a positive control for the search, screen, estimation and aggregation stages, not a
representation-learning experiment.
"""
from __future__ import annotations

import math

import numpy as np

import e8_population as e8
import probe_structured_scm as m

# ---------------------------------------------------------------- frozen before any confirmatory run
# Decoys sit far from every balancing coefficient of this design, which span [-0.66, +0.46].  A decoy close
# to a root would make the finite-sample objective unable to tell them apart, and the selected map would then
# carry a population value near but not equal to tau, which breaks the separation condition for a reason that
# has nothing to do with the theorem.  Chosen in a pilot on seeds disjoint from the confirmatory ones.
DECOYS = (-0.95, 0.95)                   # data-independent, identical for every split
CONTAMINATED_BLOCK = 4                   # W_4 carries bad_i * I; a design fact, never given to the learner
IDENTITY_TOLERANCE = 1e-6                # only ever used to check the analytic identity, never to cluster
ROOT_WINDOW = (-1.0, 1.0)                # the interval the production search covers
TIE_BREAK = "smallest |r|, then smallest r"


def finite_class(spec: m.SCMSpec, split: tuple[int, ...]) -> list[float | None]:
    """The pre-fixed finite set of encoder maps for one split.

    A split that holds out block 0 has no typed I channel, so its only map is the untyped one.  Every other
    split gets the balancing coefficients that lie inside the production search window, together with a fixed
    decoy grid that is the same for all splits.  The class is built from the design, never from tau or from a
    validity label, and the run selects inside it by the empirical objective alone.
    """
    if 0 in split:
        return [None]
    roots = [r for r in e8.clean_roots(spec, split) if ROOT_WINDOW[0] <= r <= ROOT_WINDOW[1]]
    return sorted(set([round(r, 12) for r in roots] + list(DECOYS)))


def select_map(d_view: dict, split: tuple[int, ...], candidates: list[float | None]) -> dict:
    """Line 2 of Algorithm 1: minimise the empirical nested Brier risk gap on I_D over the finite class."""
    target = m.heldout(d_view, split)
    rows = []
    for r in candidates:
        z = m.representation(d_view, split, r)
        beta0 = m._fit_brier_probit(z, d_view["A"])
        lifted = np.concatenate([beta0, np.zeros(target.shape[1])])
        beta1 = m._fit_brier_probit(np.column_stack([z, target]), d_view["A"],
                                    starts=[lifted, np.zeros(len(lifted))])
        gap = float(np.mean(m._brier_rows(beta0, z, d_view["A"])
                            - m._brier_rows(beta1, np.column_stack([z, target]), d_view["A"])))
        rows.append({"r": r, "signed_gap": gap, "objective": max(gap, 0.0)})
    best = min(rows, key=lambda q: (q["objective"], abs(q["r"]) if q["r"] is not None else 0.0,
                                    q["r"] if q["r"] is not None else 0.0))
    return {"r": best["r"], "objective": best["objective"], "signed_gap": best["signed_gap"],
            "class_size": len(rows), "all": rows}


def exact_theta(spec: m.SCMSpec, split: tuple[int, ...], r: float | None) -> float:
    """theta_S for the selected map, exactly."""
    maps = e8.linear_maps(spec)
    if 0 in split:
        kept = [j for j in range(5) if j not in split]
        mean_k = np.mean([maps[f"K{j}"] for j in kept], axis=0)
        Z = np.vstack([maps["X"], mean_k])
        return _adjusted_effect_rows(spec, Z)
    return e8.adjusted_effect(spec, split, r)


def _adjusted_effect_rows(spec: m.SCMSpec, Z: np.ndarray) -> float:
    """The same population adjustment value, for a representation given directly as coefficient rows."""
    maps = e8.linear_maps(spec)
    mu_u_row, _ = e8.conditional(maps["U"], Z)
    mu_v_row, s_v2 = e8.conditional(maps["V"], Z)
    cov_uv = float((maps["U"] - mu_u_row) @ (maps["V"] - mu_v_row))
    cov = np.array([[mu_u_row @ mu_u_row, mu_u_row @ mu_v_row],
                    [mu_u_row @ mu_v_row, mu_v_row @ mu_v_row]])
    pts, w = e8._gh2(cov)
    mu_u, mu_v = pts[0], pts[1]
    root = math.sqrt(1.0 + s_v2)
    scaled = mu_v / root
    from scipy.stats import norm
    p = 0.1 + 0.8 * norm.cdf(scaled)
    e_u_a1 = (0.1 * mu_u + 0.8 * (mu_u * norm.cdf(scaled) + cov_uv * norm.pdf(scaled) / root)) / p
    e_u_a0 = (mu_u - p * e_u_a1) / (1.0 - p)
    contrast = (1.0 + spec.effect_u * e_u_a1) - spec.outcome_u * (e_u_a1 - e_u_a0)
    return float((contrast * w).sum())


def representation_rows(spec: m.SCMSpec, split: tuple[int, ...], r: float | None) -> np.ndarray:
    maps = e8.linear_maps(spec)
    if 0 in split:
        kept = [j for j in range(5) if j not in split]
        return np.vstack([maps["X"], np.mean([maps[f"K{j}"] for j in kept], axis=0)])
    return e8.z_rows(maps, split, r)


def nuisance_quality(spec: m.SCMSpec, split: tuple[int, ...], r: float | None,
                     nuisance_view: dict, evaluator_view: dict, evaluator_raw: dict) -> dict:
    """sigma_S, q_e, q_0 and q_1 for the fitted nuisances, measured against the exact ones.

    The errors are L2 norms under the distribution of the representation, conditional on the learning
    samples, so they are computed on a large independent evaluator draw.  The evaluation sample of the
    algorithm is never touched here.  What comes back is a Monte Carlo reading of population quantities, and
    its own standard error is reported alongside.
    """
    z_n = m.representation(nuisance_view, split, r)
    z_v = m.representation(evaluator_view, split, r)
    beta = m._fit_scaled_probit(z_n, nuisance_view["A"])
    p_hat, eta_hat = m._probit_probability(beta, z_v)
    p_n, eta_n = m._probit_probability(beta, z_n)

    from scipy.stats import norm
    design_n = np.column_stack([np.ones(len(z_n)), z_n])
    design_v = np.column_stack([np.ones(len(z_v)), z_v])
    mu_hat = []
    for arm in (0, 1):
        corr_n = (0.8 * norm.pdf(eta_n) / p_n) if arm else (-0.8 * norm.pdf(eta_n) / (1.0 - p_n))
        corr_v = (0.8 * norm.pdf(eta_hat) / p_hat) if arm else (-0.8 * norm.pdf(eta_hat) / (1.0 - p_hat))
        mask = nuisance_view["A"] == arm
        coef = np.linalg.lstsq(np.column_stack([design_n, corr_n])[mask], nuisance_view["Y"][mask],
                               rcond=None)[0]
        mu_hat.append(np.column_stack([design_v, corr_v]) @ coef)

    truth = e8.conditional_arm_means(spec, representation_rows(spec, split, r), z_v)
    q_e = float(np.sqrt(np.mean((p_hat - truth["e"]) ** 2)))
    q_0 = float(np.sqrt(np.mean((mu_hat[0] - truth["m0"]) ** 2)))
    q_1 = float(np.sqrt(np.mean((mu_hat[1] - truth["m1"]) ** 2)))

    a_v, y_v = evaluator_raw["A"], evaluator_raw["Y"]
    score = (mu_hat[1] - mu_hat[0] + a_v * (y_v - mu_hat[1]) / p_hat
             - (1.0 - a_v) * (y_v - mu_hat[0]) / (1.0 - p_hat))
    sigma = float(score.std(ddof=1))
    n_v = len(score)
    return {"sigma": sigma, "q_e": q_e, "q_0": q_0, "q_1": q_1,
            "overlap_min": float(p_hat.min()), "overlap_max": float(p_hat.max()),
            "evaluator_n": int(n_v),
            "sigma_mc_se": float(sigma / math.sqrt(2.0 * (n_v - 1))),
            "score_mean": float(score.mean())}


def radius(quality: dict[tuple, dict], n_e: int, delta: float, n_screened: int, eta: float) -> dict:
    """rho = max over the screened splits of sigma_S / sqrt(n_E delta/N) + q_e(q_0+q_1)/eta.

    This is Equation (aipw-error-radius) with delta_E replaced by delta/N, exactly as the theorem's
    discussion of condition (b) prescribes.  It is a function of the learning samples only.
    """
    per = {}
    for s, q in quality.items():
        variance_term = q["sigma"] / math.sqrt(n_e * delta / n_screened)
        bias_term = q["q_e"] * (q["q_0"] + q["q_1"]) / eta
        per[s] = {"variance_term": variance_term, "bias_term": bias_term,
                  "epsilon": variance_term + bias_term}
    worst = max(per, key=lambda s: per[s]["epsilon"])
    return {"rho": per[worst]["epsilon"], "argmax_split": list(worst), "per_split": {str(k): v for k, v in per.items()}}


def sampling_terms(T: int, N: int, mm: int, delta_gap: float) -> dict:
    """P(R=0) and E[exp(-R Delta^2/2)] for R hypergeometric on (T, N, m), as exact finite sums."""
    from math import comb
    total = comb(T, mm)
    p_zero = comb(T - N, mm) / total if T - N >= mm else 0.0
    expectation = sum(comb(N, rr) * comb(T - N, mm - rr) * math.exp(-rr * delta_gap ** 2 / 2.0)
                      for rr in range(0, min(mm, N) + 1) if 0 <= mm - rr <= T - N) / total
    return {"P_R_zero": p_zero, "E_exp": expectation}


def theorem_bound(T: int, N: int, mm: int, delta: float, delta_gap: float, L: int) -> dict:
    s = sampling_terms(T, N, mm, delta_gap)
    return {"B": 1.0 - delta - s["P_R_zero"] - L * s["E_exp"], **s}


def outer_state(spec: m.SCMSpec, seed: int, n_role: int, t: float, eta: float, delta: float,
                evaluator_n: int) -> dict:
    """Build one conditioning state: draw I_D and I_N once, run lines 2 to 5, and compute (a), (c), rho, B.

    Nothing downstream is allowed to change any of this.  The split budget and the evaluation sample are the
    only things that vary afterwards, which is exactly the probability space the theorem speaks about.
    """
    raw_d = m.generate_role(spec, n_role, seed)
    raw_n = m.generate_role(spec, n_role, seed + 1)
    raw_v = m.generate_role(spec, evaluator_n, seed + 2)          # analytic evaluator, never the algorithm's
    transform = m.Transform.fit(raw_d, spec.observation)
    d_view = m.observed_view(raw_d, transform)
    n_view = m.observed_view(raw_n, transform, include_outcome=True)
    v_view = m.observed_view(raw_v, transform, include_outcome=True)

    screened, selected, thetas, quality = [], {}, {}, {}
    per_split = []
    for split in m.ALL_SPLITS:
        pick = select_map(d_view, split, finite_class(spec, split))
        z_n = m.representation(n_view, split, pick["r"])
        beta = m._fit_scaled_probit(z_n, n_view["A"])
        p_n, _ = m._probit_probability(beta, z_n)
        overlap = bool(p_n.min() >= eta and p_n.max() <= 1.0 - eta)
        keep = bool(pick["objective"] <= t and overlap)
        row = {"split": list(split), "r": pick["r"], "objective": pick["objective"],
               "class_size": pick["class_size"], "overlap_ok": overlap, "retained": keep,
               "category_eval_only": m.split_category(split)}
        if keep:
            screened.append(split)
            selected[split] = pick["r"]
            thetas[split] = exact_theta(spec, split, pick["r"])
            quality[split] = nuisance_quality(spec, split, pick["r"], n_view, v_view, raw_v)
            row["theta_S"] = thetas[split]
        per_split.append(row)

    N = len(screened)
    if N == 0:
        return {"seed": seed, "N": 0, "verdict": "NOT_APPLICABLE", "reason": "the screen retained nothing",
                "per_split": per_split}

    # (a) plurality.  Which splits carry tau is certified by the design's analytic identity, not by
    # clustering numbers within a tolerance: a split whose held-out set excludes the contaminated block and
    # whose selected map is a balancing root has theta_S = tau exactly, by the approximate-bias theorem with
    # zero residual discrepancy.  The high-precision value is then checked against that identity, and a
    # mismatch is an error rather than something to absorb into a wider tolerance.
    tau = 1.0
    def carries_tau(sp: tuple[int, ...]) -> bool:
        if 0 in sp or CONTAMINATED_BLOCK in sp:
            return False
        roots = e8.clean_roots(spec, sp)
        return any(abs(selected[sp] - rr) < 1e-6 for rr in roots)

    target_splits = [sp for sp in screened if carries_tau(sp)]
    for sp in target_splits:
        if abs(thetas[sp] - tau) > IDENTITY_TOLERANCE:
            raise RuntimeError(f"identity check failed for split {sp}: theta = {thetas[sp]!r}")
    for sp in screened:
        if sp not in target_splits and abs(thetas[sp] - tau) <= IDENTITY_TOLERANCE:
            raise RuntimeError(f"split {sp} hits tau without carrying it by identity: {thetas[sp]!r}")

    n_tau = len(target_splits)
    # distinct non-target values, grouped by the same identity logic: two non-target splits share a value
    # only when the design makes them equal, which here means the same selected map on the same block set
    others_vals = [thetas[sp] for sp in screened if sp not in target_splits]
    uniq_others: list[float] = []
    for v in others_vals:
        if not any(abs(v - u) <= IDENTITY_TOLERANCE for u in uniq_others):
            uniq_others.append(float(v))
    counts = {tau: n_tau}
    for u in uniq_others:
        counts[u] = int(sum(1 for v in others_vals if abs(v - u) <= IDENTITY_TOLERANCE))
    uniq = ([tau] if n_tau else []) + uniq_others
    L = len(uniq_others)
    p_tau = n_tau / N
    p_star = max((counts[u] for u in uniq_others), default=0) / N
    gap = p_tau - p_star

    # rho, from the learning samples only
    rad = radius(quality, n_e=n_role, delta=delta, n_screened=N, eta=eta)
    rho = rad["rho"]

    # (c) separation between distinct population values
    sep = min((abs(a - b) for i, a in enumerate(uniq) for b in uniq[i + 1:]), default=float("inf"))

    return {"seed": seed, "N": N, "screened": [list(s) for s in screened], "selected_r": {str(k): v for k, v in selected.items()},
            "theta_S": {str(k): v for k, v in thetas.items()}, "distinct_values": uniq, "counts": counts,
            "n_tau": n_tau, "L": L, "p_tau": p_tau, "p_star": p_star, "Delta": gap,
            "rho": rho, "radius_detail": rad, "separation": sep, "separation_ok": bool(sep > 4.0 * rho),
            "plurality_ok": bool(gap > 0.0), "per_split": per_split,
            "quality": {str(k): v for k, v in quality.items()},
            "transform": transform, "d_view": d_view, "n_view": n_view, "spec": spec}


def fixed_nuisance(n_view: dict, split: tuple[int, ...], r: float | None) -> dict:
    """Fit the propensity and the two outcome models once on I_N and freeze them.

    The theorem conditions on the learning samples, so the nuisance functions must not move when the
    evaluation sample is redrawn.  Refitting them inside the inner loop would change the object the theorem
    speaks about.
    """
    from scipy.stats import norm
    z_n = m.representation(n_view, split, r)
    beta = m._fit_scaled_probit(z_n, n_view["A"])
    p_n, eta_n = m._probit_probability(beta, z_n)
    design_n = np.column_stack([np.ones(len(z_n)), z_n])
    coefs = []
    for arm in (0, 1):
        corr = (0.8 * norm.pdf(eta_n) / p_n) if arm else (-0.8 * norm.pdf(eta_n) / (1.0 - p_n))
        mask = n_view["A"] == arm
        coefs.append(np.linalg.lstsq(np.column_stack([design_n, corr])[mask], n_view["Y"][mask],
                                     rcond=None)[0])
    return {"beta": beta, "outcome_coefs": coefs}


def apply_nuisance(model: dict, e_view: dict, split: tuple[int, ...], r: float | None) -> np.ndarray:
    """The honest AIPW score on I_E with the frozen nuisances."""
    from scipy.stats import norm
    z_e = m.representation(e_view, split, r)
    p_e, eta_e = m._probit_probability(model["beta"], z_e)
    design_e = np.column_stack([np.ones(len(z_e)), z_e])
    mu = []
    for arm, coef in enumerate(model["outcome_coefs"]):
        corr = (0.8 * norm.pdf(eta_e) / p_e) if arm else (-0.8 * norm.pdf(eta_e) / (1.0 - p_e))
        mu.append(np.column_stack([design_e, corr]) @ coef)
    a_e, y_e = e_view["A"], e_view["Y"]
    return mu[1] - mu[0] + a_e * (y_e - mu[1]) / p_e - (1.0 - a_e) * (y_e - mu[0]) / (1.0 - p_e)


def run_inner(state: dict, spec: m.SCMSpec, budget: int, reps: int, seed: int, n_role: int,
              tau: float = 1.0) -> dict:
    """The two things the theorem lets vary: the split draw and the evaluation sample.

    Every inner repetition draws `budget` splits uniformly without replacement from all T of them with a
    generator that never sees the data, draws a fresh I_E, and runs Algorithm 1 on exactly the splits it drew.
    The full screened set is estimated as well, but only to record whether the uniform-radius event of
    condition (b) held; the algorithm's return value never sees it.
    """
    screened = [tuple(s) for s in state["screened"]]
    rho, thetas = state["rho"], {tuple(eval(k)): v for k, v in state["theta_S"].items()}
    selected_r = {tuple(eval(k)): v for k, v in state["selected_r"].items()}
    models = {s: fixed_nuisance(state["n_view"], s, selected_r[s]) for s in screened}
    all_splits = list(m.ALL_SPLITS)
    draw_rng = np.random.default_rng(seed)              # independent of the data, as line 1 requires
    data_rng = np.random.default_rng(seed + 1)

    ok_radius = np.zeros(reps, dtype=bool)
    success = np.zeros(reps, dtype=bool)
    kinds: dict[str, int] = {}
    r_counts = np.zeros(reps, dtype=int)
    for i in range(reps):
        drawn = [all_splits[j] for j in draw_rng.choice(len(all_splits), size=budget, replace=False)]
        raw_e = m.generate_role(spec, n_role, int(data_rng.integers(1, 2 ** 31 - 1)))
        e_view = m.observed_view(raw_e, state["transform"], include_outcome=True)
        est = {s: float(apply_nuisance(models[s], e_view, s, selected_r[s]).mean()) for s in screened}
        ok_radius[i] = all(abs(est[s] - thetas[s]) <= rho for s in screened)
        retained = [s for s in drawn if s in set(screened)]
        r_counts[i] = len(retained)
        agg = m.aggregate(retained, np.array([est[s] for s in retained]), rho) if retained else \
            {"return_kind": "no_return", "output": None}
        kinds[agg["return_kind"]] = kinds.get(agg["return_kind"], 0) + 1
        success[i] = bool(agg["return_kind"] == "unique_largest" and agg["output"] is not None
                          and abs(agg["output"] - tau) <= rho)
    return {"reps": reps, "budget": budget, "uniform_radius_rate": float(ok_radius.mean()),
            "success_rate": float(success.mean()), "return_kinds": kinds,
            "R_mean": float(r_counts.mean()), "R_zero_rate": float((r_counts == 0).mean()),
            "R_counts": np.bincount(r_counts, minlength=min(budget, len(screened)) + 1).tolist()}


def lower_bound(successes: int, n: int, level: float = 0.95) -> float:
    """One-sided Clopper-Pearson lower limit for a binomial proportion."""
    from scipy.stats import beta as beta_dist
    if successes == 0:
        return 0.0
    return float(beta_dist.ppf(1.0 - level, successes, n - successes + 1))
