# Pre-Registration: Two-Track Benchmark Suite (Tabular + Image)

Date: 2026-08-18
Status: frozen before execution. No metric, decision rule, DGP grid, or loss may change after results are seen. Deviations must be logged as protocol violations in the results file.
Theory anchors: `../THEORY.md` T4 (order preservation), T4b (continuous severity), robust corollary (constrained objective derivation).
Executing track: codex `multi-agent:single-proxy-balancing + causalpy + math-theory-agent` thread; this memo is the contract it runs against.

## Tracks and data

1. Tabular: ACIC-covariate semi-synthetic augmentation. Real ACIC 2016 covariate matrix (4,802 x 58; provenance path and sha256 recorded in the results file), with $U$, $W$, $A$, $Y$ generated per DGP spec. Additional synthetic SCM families from local CausalPy. Multiple DGP draws (vary severity strength, proxy noise, selection strength, nonlinearity) x multiple data resamples; grid enumerated in the results file before the first run.
2. Image: two domains, (a) local handwritten-digit images, (b) fully synthetic lesion images. $U$ is a smooth function of true image factors; $W=U+\varepsilon_W$; $A\sim\mathrm{expit}(aU+q(X))$-type with clipping; $Y=\tau A+\beta U+\varepsilon$. CNN learns $H=h(X)$ with $M$ strata; identical architecture, initialization seeds, and schedule in both arms; only the balance loss differs.

## Arms

- Joint arm A1 (replication): additive joint-kernel discrepancy objective, unchanged from the high-dimensional run.
- Constrained arm A2 (new, theory-derived): minimize $\widehat D_W(H)$ subject to $\widehat D_X(H)\le c$ with the anchor $c=\widehat D_X(H_{X\text{-only}})$ computed on the same fold without outcome data. Lagrangian implementation allowed with pre-registered fixed multiplier.
- Baseline: $X$-only balance learner, same class and capacity.
- Estimation in every arm: identical stratified plug-in on the learned $H$ only, cross-fitted; representation learned on the training fold, $\widehat D$-metrics and $\widehat\tau$ on the held-out fold.

## Primary metrics and decision rule

Primary, paired per (DGP, seed) cell: $|\mathrm{bias}|$ of $\widehat\tau$, RMSE, held-out $\widehat D_W$. The joint discrepancy $D_{XW}$ and $\widehat D_X$ are diagnostics only; per the T4 lesson, $D_{XW}$ is not a pass/fail quantity.

Per track and per arm-vs-baseline comparison, PASS requires all three of:

1. Win rate of simultaneous bias and RMSE improvement over paired cells at least 70 percent.
2. One-sided Wilcoxon signed-rank test on paired bias reductions significant at $\alpha=0.05$.
3. Both negative controls behave as predicted: with uninformative $W$ the gain vanishes or reverses; with severity unpredictable from $X$ the gain vanishes.

Interpretation rule fixed in advance: if $\widehat D_W$ decreases but bias does not, diagnose the T4/T4b preconditions before any redesign, using the logged sign pattern of stratum shifts.

## Mandatory logging (adds T4b validation for free)

Per cell and per learned $H$: held-out $\widehat D_W$, $\widehat D_X$, $D_{XW}$; oracle per-stratum mean shifts $\delta_z$ and their signs; $T=\sum_zp_z\delta_z$ and $S_2=\sum_zp_z\delta_z^2$; realized bias. These verify the sign-uniformity condition Q5 and the first-order relation of T4b directly from the suite.

## Guardrails

- Mass-floor regularizer on stratum occupancy in both arms identically.
- Minibatch MMD noise: batch size and epochs pre-fixed; final $\widehat D$ values recomputed in a full pass.
- No per-arm tuning; any hyperparameter shared across arms or pre-registered per arm.
- Seeds enumerated in advance; all runs reported including failures.

## Execution deviations and verdict (post-hoc record, 2026-08-19)

Logged per the deviation rule above, after an independent audit recomputed all
means, win rates, and Wilcoxon tests from the raw CSVs and confirmed the
stored numbers.

1. This document predates the results by timestamp but did not fix $n$,
   repetitions, seeds, steps, or the penalty weight; the correct label for
   the suite is internally pre-specified, not preregistered.
2. Kernel mismatch: the memo says additive joint kernel; the code uses the
   product kernel $k_Xk_W$, matching the prior high-dimensional run.
3. The memo says cross-fitting; the code uses a single 50/50 split
   (learn on fold 1, evaluate on fold 2), with no fold swap.
4. Per-stratum shift vectors $\delta_z$ were not persisted; only the
   aggregates $T$, $S_2$, and the negative-shift count were saved.
5. The promised ACIC sha256 was not written into the results file.
6. The constrained arm is a soft penalty $D_W^2+25\max\{D_X^2-c,0\}$ with a
   train-side anchor; it does not enforce the constraint, and held-out
   $D_X$ exceeded the $X$-only value in scm_dense, scm_mixed, and digits.
7. Negative-control repetition counts were not fixed here; the 18-pair
   x_signal_null extension is post hoc and stored separately.

Verdicts under the frozen decision rule: tabular track FAIL (all six primary
efficacy decisions pass; the x_signal_null control gain did not vanish);
image track PASS (all four primary decisions and both controls pass).
Naming corrections: ACIC runs are an augmented synthetic DGP on real ACIC
2016 covariates; scm_dense and scm_mixed are newly written local synthetic
covariate distributions, not CausalPy DGPs; digits is semi-synthetic on real
pixels; lesions is fully synthetic. Results are robustness evidence adjacent
to T4b (Q3 and Q5 are violated by the logistic selection), not a theorem
validation.
