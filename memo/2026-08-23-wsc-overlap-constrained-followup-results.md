# WSC Overlap-Constrained Cross-Mask Follow-Up: Results

Date: 2026-08-23 CDT

Protocol: `memo/2026-08-23-wsc-overlap-constrained-followup-preregistration.md`

Machine-readable result: `results/wsc_overlap_constrained_followup.json`

## Exact repetition count

This experiment used one fixed WSC dataset. It did not generate new datasets.

- Monte Carlo data-generating repetitions: **0**
- Paired algorithmic repeated sample splits: **10**
- Outer folds per repetition: 3
- Fixed audit masks: 5
- Representation fits per repetition-fold-mask: 6
- Main representation fits: $10\times3\times5\times6=900$
- Deterministic smoke QA: two runs of 18 fits, or 36 additional fits
- Total representation fits executed in this follow-up: **936**

The ten repeated splits quantify algorithmic and sample-split sensitivity on this fixed dataset. They are not Monte Carlo replications and do not measure repeated-sampling performance over new populations.

## Final verdict

The preregistered result is **FAIL**.

The overlap penalty strongly improved practical overlap, but did not reach the locked all-mask threshold in enough repetitions. More importantly, the benchmark-gap advantage of the unconstrained representation largely disappeared. In this fixed penalty intervention, an empirical overlap-audit-discrepancy tradeoff was observed.

## Estimates and overlap

The noisy full-sample randomized benchmark remains 0.760 with standard error 0.151.

| Method | Mean estimate | SD over 10 splits | Mean absolute benchmark gap | Complete overlap pass |
|---|---:|---:|---:|---:|
| Naive QED | 3.451 | 0.000 | 2.690 | Not applicable |
| Direct $X$ | 2.465 | 0.038 | 1.704 | 10/10 |
| Direct $(X,W)$ | 1.433 | 0.042 | 0.672 | 0/10 |
| Unconstrained cross-mask | 1.302 | 0.065 | 0.541 | 0/10 |
| Overlap-constrained cross-mask | 2.423 | 0.186 | 1.663 | 6/10 |

The unconstrained cross-mask result reproduces the previous pilot direction and remains closest to the RCT benchmark, but it has no supported overlap. Direct $(X,W)$ has the next-smallest gap and also fails overlap in every repetition. The constrained method is only slightly closer to the benchmark than direct $X$ on average, and it has substantially larger algorithmic variability.

The constrained method had a smaller benchmark gap than direct $X$ in 7 of 10 paired repetitions. The preregistered requirement was at least 8 of 10. Its mean gap, 1.663, was also much larger than the unconstrained gap 0.541 plus one RCT standard error, 0.151.

## How much overlap improved

At the orientation level, the penalty improved the out-of-fold central propensity fraction in every one of the 50 paired comparisons.

- Unconstrained orientation overlap passes: 0/50
- Constrained orientation overlap passes: 45/50
- Repetitions in which all five constrained orientations passed: 6/10
- Mean central-fraction increase: 0.164

The five final failures were all failures of the unchanged 90% central-fraction condition. Their fractions were 0.868, 0.881, 0.888, 0.892, and 0.895. Every failed orientation still passed the arm-specific ESS and five-cell local-support components. Thus the intervention moved the learner very close to the practical gate, but the threshold was not changed after seeing the data.

## Cost in proxy balance

The mean held-out audit surrogate changed from

$$
0.01973\quad\text{to}\quad0.02511.
$$

That is an increase of approximately 27%. The constrained surrogate improved in only 1 of 50 paired orientation comparisons. Therefore, the overlap gain did not come for free. Constraining treatment predictability removed representation information that had been useful for balancing the untouched proxy audit view.

Three changes co-occurred under the fixed penalty intervention: practical overlap improved, held-out audit discrepancy worsened, and the estimate moved toward direct $X$ adjustment. This experiment does not establish that one of these changes caused either of the others.

## Outcome-free lambda selection

There were 150 nested outer-fold-mask selection cells. Selected values were:

| $\lambda$ | Count |
|---:|---:|
| 0.1 | 25 |
| 1 | 16 |
| 10 | 66 |
| 100 | 43 |

In 149 of 150 cells, at least one candidate passed the inner 90% central-fraction screen. One cell used the preregistered fallback rule because no candidate passed. This inner failure is distinct from the five final outer-held-out orientation failures.

No outcome, RCT estimate, causal estimate, or benchmark gap entered lambda selection. The outer evaluation fold was never used for selection. The ridge-hinge term was an optimization device only and was not treated as an identification assumption or proof of positivity.

## Preregistered decision components

| Component | Result |
|---|---:|
| Constrained complete overlap in at least 8/10 repetitions | Fail, 6/10 |
| More complete-overlap passes than unconstrained | Pass, 6 versus 0 |
| Smaller gap than direct $X$ in at least 8/10 repetitions | Fail, 7/10 |
| Mean gap within one RCT SE of unconstrained | Fail |

The combined preregistered decision is therefore `FAIL`.

## Validation

- All data-integrity and QED-only gates passed.
- The five masks, outer-fold rule, outcome learner, evaluation RFF map, audit view, and seeds were paired across the two cross-mask learners.
- All predictions were outer-fold out of fold.
- Nested selection fitted all random-feature, audit-nuisance, and propensity components without using the outer evaluation fold.
- No trimming or weight clipping was applied.
- Two smoke runs reproduced exactly after excluding runtime metadata.
- The first three unconstrained repetitions exactly matched the corresponding prior-pilot estimates for direct $X$, direct $(X,W)$, and cross-mask adjustment.

## Scientific interpretation

The follow-up is informative despite failing. In this fixed penalty intervention, practical overlap improved while held-out audit discrepancy worsened and the benchmark advantage largely disappeared. This is an empirical overlap-audit-discrepancy tradeoff in this experiment, not a general Pareto-frontier result and not evidence that the constrained estimator identifies the ATE.

The result does not establish the exact one-half randomization condition, proxy validity, independent proxy views, mixture injectivity, exact balance, positivity, or point identification. No additional experiment was run after this result.
