# WSC Overlap-Constrained Cross-Mask Follow-Up: Locked Protocol

Date locked: 2026-08-23 CDT

Status: locked before outcome and RCT evaluation of the follow-up

## Output Contract

- Deliverable: one paired WSC follow-up comparing the unchanged cross-mask learner with an outcome-free overlap-constrained version.
- Consumer: Yonghan, who will decide whether this method warrants a larger empirical study.
- Acceptance: exact paired folds and audit masks; 10 completed algorithmic repetitions; 900 main representation fits; outcome-free penalty selection; unchanged held-out overlap gate; explicit reporting of selection and final-gate failures; reproducible machine-readable output.
- Boundaries: this is not a Monte Carlo data-generating experiment; no outcome or randomized benchmark may select the penalty; no threshold may be lowered; prior pilot files must not be modified; no README, ARTIFACTS, dependency, or git operation.

Manager plan review: `pass`.

## Fixed data, estimand, and comparators

The follow-up inherits the data contract from `memo/2026-08-23-wsc-population-ate-pilot-preregistration.md`.

- Learned models and standardization use only the 1,105 QED rows satisfying $Z=S$, with 288 treated and 817 controls.
- The full randomized sample enters only the scalar randomized difference-in-means benchmark and its standard error.
- The population-ATE interpretation requires the explicit $P(Z=1)=1/2$ randomization condition stated in the pilot protocol.
- The five audit masks, their order, and seed 190602 are unchanged.
- Outer-fold generation, three folds, preprocessing, outcome learner, propensity diagnostic, held-out finite-RFF audit surrogate, overlap thresholds, no-trimming rule, and all integrity checks are unchanged.

The reported methods are:

1. full-sample randomized benchmark;
2. naive QED contrast;
3. direct $X$ adjustment;
4. direct $(X,W)$ adjustment;
5. unchanged unconstrained cross-mask representation adjustment;
6. overlap-constrained cross-mask representation adjustment.

All paired learned methods use identical QED rows, outer folds, audit masks, evaluation random features, outcome learner, and seeds.

## Why an overlap penalty is needed

Let $R$ denote the learned scalar representation. The existing conditional cross-covariance objective residualizes treatment $A$ on random features of $R$. If $R$ predicts treatment almost perfectly, the treatment residual becomes small and can make the dependence objective small even when the resulting adjustment has poor overlap.

For a training set, let $q(R)$ be the differentiable ridge prediction of $A$ from the same fixed random features of $R$ used in the conditional cross-covariance calculation. The added optimization penalty is

$$
\mathcal P_{\mathrm{ov}}(R)
=
\frac1n\sum_{i=1}^n
\left[
\{0.10-q(R_i)\}_{+}^{2}
+
\{q(R_i)-0.90\}_{+}^{2}
\right].
$$

For candidate $\lambda$, Adam minimizes

$$
\widehat D_{\mathrm{RFF}}^2(R)
+
\lambda\mathcal P_{\mathrm{ov}}(R)
+
\text{the unchanged score-variance penalty}.
$$

The interval $[0.10,0.90]$ is deliberately stricter than the unchanged final diagnostic interval $[0.05,0.95]$. The ridge-hinge penalty is only an optimization device. It is not an identification assumption, does not prove positivity, and cannot replace the final held-out overlap gate.

## Outcome-free selection of $\lambda$

The fixed candidate grid is

$$
\Lambda=\{0.1,1,10,100\}.
$$

Within each outer-training fold and audit orientation:

1. form a fixed stratified 80/20 selection-training/selection-validation split using only $A$;
2. fit all four candidates on selection-training rows with identical initialization, bandwidth, random features, and optimization budget;
3. predict $R$ in selection-validation rows;
4. estimate the validation propensity from selection-training $R$ and score the fraction of validation propensities in $[0.05,0.95]$;
5. compute the validation audit surrogate with the same inner-cross-fitting rule used by the pilot;
6. among candidates with central fraction at least 0.90, choose the candidate with the smallest validation audit surrogate, breaking ties toward the smaller $\lambda$;
7. if no candidate passes 0.90, choose the candidate with the largest central fraction, then the smallest audit surrogate, then the smaller $\lambda$;
8. refit the selected $\lambda$ on every row in the outer-training fold and evaluate only on the outer-held-out fold.

The outer evaluation fold is never used for candidate fitting or $\lambda$ selection. Candidate random features, bandwidths, propensity nuisance models, and audit nuisance models are fitted only inside the nested outer-training split. No $Y$, RCT benchmark, treatment-effect estimate, or benchmark gap enters this selection. Report the number of candidate-selection cells in which no $\lambda$ passes the inner 0.90 screen.

## Unchanged held-out overlap gate

The final gate is unchanged from the pilot. In each representation orientation, out-of-fold propensities must satisfy all of:

- at least 90% in $[0.05,0.95]$;
- treated and control ATE-weight ESS at least `max(25, 0.20 * observed arm size)`;
- at least five treated and five controls in each of five pooled propensity-quantile cells.

No trimming or weight clipping is allowed. A five-mask average passes a repetition only when every orientation passes. Report all final held-out failures separately from inner candidate-selection failures.

## Repetition and fit budget

WSCdata is one fixed observed dataset. Changing folds and optimization seeds does not generate a new dataset. Accordingly, the follow-up uses exactly:

- 10 paired **algorithmic repeated sample splits**;
- 3 outer folds per repetition;
- 5 fixed audit orientations;
- 1 unchanged unconstrained fit per repetition-fold-orientation;
- 4 constrained candidate fits plus 1 selected-$\lambda$ refit per repetition-fold-orientation.

The main analysis therefore contains exactly

$$
10\times3\times5\times(1+4+1)=900
$$

representation fits. Smoke or deterministic QA fits are counted and reported separately. These 10 repetitions must not be called Monte Carlo replications.

## Reporting and decision rule

For every method report the estimate, algorithmic split variability, signed and absolute gap from the noisy RCT benchmark, and held-out overlap. For the two cross-mask methods report paired held-out audit surrogates, all selected $\lambda$ values, inner selection failures, and final held-out failures.

The overlap intervention succeeds mechanistically only if the constrained five-mask method passes every unchanged held-out overlap component in at least 8 of 10 repetitions, thereby materially improving on the pilot's 0/3 pass rate and the paired unconstrained method's pass count. Estimation preserves the prior encouraging direction only if its benchmark gap is below direct $X$ adjustment in at least 8 of 10 paired repetitions and its mean gap is not larger than the unconstrained cross-mask mean gap by more than one RCT benchmark standard error. Per-repetition and per-orientation paired changes are retained in the machine-readable result. These evaluation rules do not influence $\lambda$ selection.

Regardless of the outcome, the experiment cannot establish proxy validity, independent proxy views, mixture injectivity, positivity, or point identification.
