# Rerun Registration v2 (frozen before execution)

Date: 2026-08-19. Status: every parameter, seed, decision rule, and
instrumentation item below is fixed before any of the three experiments run.
The sha256 of this file is recorded in every result file. Deviations must be
logged as protocol violations. Audit context: this registration implements
the three redo items identified on 2026-08-19 (hard constraint, T4b-exact
world, x_signal_null resolution) with the instrumentation that the first
suite omitted (per-stratum shift vectors, ACIC sha256).

## E1. Hard-constrained arm rerun (tabular)

Question: does the theory-derived objective min D_W(h) subject to
D_X(h) <= c retain its efficacy when the constraint is actually enforced?

- Sources: acic, scm_dense, scm_mixed as in `code/benchmark_suite_tabular.py`;
  variants v1_linear, v2_sine, v3_interact, v4_hetero; regime primary only.
- n = 1200, reps = 6, steps = 120, strata M = 5, seed formula identical to
  the first suite (base 2026081901), so every cell is paired with the frozen
  soft-penalty run.
- Arms: x_only (anchor provider, identical to first suite) and
  constrained_hard. No joint arm.
- Constraint quantity: train-side soft D_X^2 at temperature 0.25. Anchor
  c = the x_only net's final value of that quantity, computed without
  outcome data.
- Enforcement: dual ascent. lambda_0 = 25; after each optimizer step,
  lambda <- lambda + 500 * relu(soft D_X^2 - c). After the 120 steps, up to
  3 feasibility rounds of 40 steps each, doubling lambda at each round
  start, until train soft D_X^2 <= c + 1e-6. Feasibility flag recorded.
- Primary decision rule (all three required): (1) train-side feasibility in
  100 percent of runs; (2) per source, paired win rate of simultaneous
  |oracle bias| and |tau error| improvement over x_only >= 0.70;
  (3) per source, one-sided Wilcoxon on paired |oracle bias| reductions
  p < 0.05.
- Diagnostics (not pass/fail): held-out hard D_X gap versus x_only per cell
  with exceedance counts, compared against the soft-penalty run's
  exceedances; held-out D_W.
- Instrumentation: per-stratum vectors delta_z, p_z, pi_z(1-pi_z) persisted
  per run; ACIC csv sha256 recorded in provenance.
- Output: results/rerun_e1_constrained_hard.json.

## E2. T4b-exact world validation (population-exact, no sampling)

Question: do the T4b sandwich and ordering criterion hold literally in a
world where Q1 to Q5 hold by construction?

- Finite X with S = 12 sites, p(x) uniform. m(x) = linspace(-1, 1, 12).
  Shift groups: d(x) = 0.8 for x < 6, d(x) = 0.3 for x >= 6 (both positive,
  so Q5 holds for every stratum of every map). e(x) = 0.5 constant, so
  within-stratum arm mixture weights match and Q4 is automatic. f = 0.
  U | X=x, A=a ~ Normal(m(x) + a d(x), 0.6^2) by direct construction;
  W = U + 0.5 eps (Q2 exact); Y = A + 2U + eps (Q1 exact, beta = 2).
- A stratum is Q3-exact iff its members lie in one shift group (then the
  treated law is the control law shifted by that group's d). Maps whose
  strata are all group-pure are the Q3-exact class.
- Evaluation: closed-form Gaussian-mixture MMD for D_W per stratum
  (Gaussian kernel k_W with bandwidth 1.0); T and S2 from the construction;
  bias = beta T exactly (T1, f = 0). s_z = RKHS norm of the embedded
  derivative of the control-arm proxy density, and C_z from the second
  derivative, both by 400-point grid quadrature on [-6, 6]; the band
  [s_min, s_max] and C_bar are taken over all evaluated strata.
- Map sets: 2,000 constructed group-pure maps and 5,000 uniform random maps,
  M = 3, base seed 2026081903.
- Decision rule (all required): (1) zero violations of the T4b sandwich
  s_min T - 0.5 C_bar S2 <= D_W <= s_max T + 0.5 C_bar S2 across all 7,000
  maps; (2) zero violations of the ordering-criterion implication (whenever
  the criterion's strict inequality holds for a pair, the bias ordering
  holds) over 50,000 random pairs; (3) on group-pure strata the per-stratum Taylor bound
  |Delta_W - s_z delta_z| <= 0.5 C_z delta_z^2 holds with zero violations,
  which is the exact content of the T4b proof at finite shift (amended
  before any execution: the first draft wrongly demanded the first-order
  identity at relative error 1e-3, which the theorem does not claim at
  finite delta).
- Output: results/rerun_e2_t4b_exact.json.

## E3. x_signal_null resolution (tabular)

Question: is the residual gain in the U-independent-of-X control explained
by the propensity-saturation channel, and does it vanish when that channel
is closed?

- Common: sources acic and scm_dense, variant v2_sine, n = 1200, reps = 9,
  steps = 120, M = 5, arms joint and x_only, base seed 2026081904.
- E3a (channel measurement): regime x_signal_null exactly as in the first
  suite. Instrumentation: per-stratum delta_z, p_z, and the saturation
  measure pi_z(1 - pi_z) on the held-out fold. Pre-registered tests:
  (1) pooled Spearman correlation between |delta_z| and pi_z(1 - pi_z)
  across joint-arm strata is positive with p < 0.05; (2) paired one-sided
  Wilcoxon that the joint arm's T = sum_z p_z delta_z is smaller than
  x_only's, p < 0.05. Both must hold to declare the saturation channel
  confirmed.
- E3b (channel closed): new regime x_signal_null_flat, identical except the
  propensity is clip(0.5 + 0.12 U + 0.12 q(X), 0.10, 0.90), which removes
  representation-accessible variation in the U-selection slope up to rare
  clipping. Pre-registered criterion for "gain vanishes": joint win rate
  over the 18 pairs < 0.70 and one-sided Wilcoxon p >= 0.05.
- Verdict rule: if E3a confirms and E3b vanishes, the original control
  failure is recorded as an identified secondary mechanism and the
  re-specified control passes; the tabular track FAIL under the first
  suite's rule stands unchanged in the historical record. If either part
  fails, the saturation explanation is dropped and the residual gain is
  recorded as unexplained.
- Output: results/rerun_e3_control_resolution.json.

## Shared

Runs are reported in full including failures; no post-result tuning; the
learner, optimizer, schedule, and capacity are identical across arms within
each experiment; only stated objectives differ.
