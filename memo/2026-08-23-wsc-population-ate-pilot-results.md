# WSC Population-ATE Pilot: Results

Date: 2026-08-23 CDT

Protocol: `memo/2026-08-23-wsc-population-ate-pilot-preregistration.md`
Machine-readable result: `results/wsc_population_ate_pilot.json`

## Final pilot verdict

The result is **promising but does not pass the locked success rule**.

The proposed cross-mask estimator was substantially closer to the randomized benchmark than the naive QED and direct $X$ estimators in all three algorithmic repetitions. It was also closer than direct $(X,W)$ adjustment. However, every cross-mask orientation failed the preregistered overlap gate. Therefore, this pilot does not support a causal-performance claim.

## Benchmark and estimates

The full randomized sample contains 2,200 participants, with 1,064 assigned to mathematics training and 1,136 assigned to control. The unadjusted randomized benchmark is

$$
\widehat\tau_{\mathrm{RCT}}=0.760,
\qquad
\widehat{\operatorname{SE}}=0.151,
$$

with a Wald interval of $[0.464,1.056]$. This is a noisy benchmark, not the true causal effect.

| Method | Mean estimate | SD across 3 splits | Mean absolute benchmark gap | Overlap pass rate |
|---|---:|---:|---:|---:|
| Naive QED contrast | 3.451 | 0.000 | 2.690 | Not applicable |
| Direct $X$ adjustment | 2.493 | 0.041 | 1.733 | 3/3 |
| Direct $(X,W)$ adjustment | 1.481 | 0.008 | 0.720 | 0/3 |
| Learned $X$-only score, five-mask average | 2.409 | 0.046 | 1.649 | 3/3 |
| Proposed cross-mask score, five-mask average | 1.316 | 0.083 | 0.555 | 0/3 |

The three proposed estimates were 1.220, 1.365, and 1.362. Relative to the mean gaps, cross-mask reduced the gap by approximately 79% versus the naive QED contrast, 68% versus direct $X$ adjustment, and 23% versus direct $(X,W)$ adjustment. These percentages are descriptive only because the overlap gate failed and the benchmark itself is noisy.

## What the overlap failure means

The direct $X$ method passed every overlap component in all three repetitions. Its predicted propensity was mostly central, and both arm-specific inverse-probability effective sample sizes were comfortably above the locked thresholds.

Direct $(X,W)$ adjustment failed severely. Its out-of-fold propensities extended approximately from 0 to 1, only 59% to 61% were in $[0.05,0.95]$, and its arm-specific ESS fell below the locked threshold.

Cross-mask was less extreme than direct $(X,W)$ adjustment. All cross-mask orientations passed the ESS and local two-arm support requirements. Nevertheless, only about 76% to 82% of predicted propensities were in $[0.05,0.95]$, below the locked 90% requirement. All 15 orientation-by-repetition fits therefore failed the overall gate. The cross-mask estimate is an extrapolation-sensitive diagnostic, not a supported causal estimate under this protocol.

## Held-out audit discrepancy

The finite-RFF held-out surrogate is comparable only between each paired learned $X$-score and cross-mask score because they use the same untouched audit item, outer fold, bandwidth, random features, and inner cross-fitting contract.

Across 15 paired comparisons:

- mean learned-$X$ surrogate: 0.02126;
- mean cross-mask surrogate: 0.02046;
- cross-mask was lower in 10 of 15 comparisons.

The average decrease was about 3.8%, but it was not orientation-wise universal. This is weaker evidence than the benchmark-gap change and does not rescue the overlap failure. The quantity is a finite random-feature conditional cross-covariance surrogate, not exact KCI.

## Data and leakage checks

All integrity gates passed:

- all source hashes matched the frozen manifest;
- WSC counts and the 288/817 QED treatment cells were reproduced;
- the seven non-Big-Five aggregate scores were reconstructed exactly;
- the official BFI-44 reverse key, translated to the WSC grouped columns, reconstructed all five Big Five aggregate scores exactly for all 2,200 rows;
- only MCS `pre` rows entered $W$;
- `mathSel`, posttest variables, and post-training items were excluded from representation inputs;
- representation, outcome, propensity, and standardization calculations used QED rows only;
- the full randomized sample entered only the scalar randomized benchmark and its standard error;
- every learned effect and propensity prediction was outer-fold out of fold;
- paired discrepancies used evaluation-fold inner cross-fitting;
- no trimming or weight clipping was performed;
- a second full run reproduced all records exactly, excluding runtime metadata.

The Big Five scoring assertion was added after the first smoke computation but before interpreting the full pilot. The five audit masks were not changed. The canonical result was then regenerated with the amended code and registration hash.

## Important identification limitation

The QED subset represents the original population only if randomized assignment has probability $1/2$ independently of pretreatment variables and potential outcomes. Under that condition, the probability of satisfying $C=\mathbb 1(Z=S)=1$ is $1/2$ for every pretreatment profile, so keeping all $C=1$ observations preserves the population distribution. The frozen WSC package documentation calls `mathGrp` randomized assignment but does not state the exact assignment probability. Consequently, the population-ATE interpretation depends on an explicit design condition not verified in the frozen snapshot.

This experiment also cannot verify proxy validity, independent proxy views, mixture injectivity, or point identification.

## Decision for the next step

The empirical signal is worth preserving: cross-mask moved the estimate much closer to the randomized benchmark and outperformed direct $(X,W)$ regression descriptively. But the current learner appears to encode treatment selection too sharply. A next experiment should be separately preregistered to add an overlap constraint or penalty during representation learning and should keep the current masks, folds, outcome learner, and evaluation rule fixed. No such follow-up was run in this step.
