# WSC Population-ATE Pilot: Locked Protocol

Date locked: 2026-08-23 CDT

Status: core protocol locked before execution; Big Five integrity amendment added before full-result inspection, with masks and analysis unchanged
Scope: modest pilot only, not the final repeated benchmark

## Output Contract

- Deliverable: one executable WSC pilot, one machine-readable result, and a concise interpretation.
- Consumer: Yonghan, who will decide whether a larger Step 3 experiment is justified.
- Acceptance criteria: all data-integrity gates pass; every learned prediction is outer-fold out of fold; representation fitting is outcome-free and uses QED rows only; all methods use the same outer folds and outcome learner; overlap failures are reported without trimming; the full randomized experiment is used only as a noisy benchmark.
- Boundaries: no causal claim from this pilot, no claim that proxy validity or injectivity holds, no tuning after seeing the results, no README or ARTIFACTS edit, no dependency installation, and no commit.

Manager plan review: `pass`. The protocol below is the approved minimum implementation.

## Data and target

Let $Z$ be randomized assignment to mathematics training, $S$ the pretreatment self-selection indicator, and $Y$ the mathematics posttest score. The observational QED subset is

$$
C=\mathbb 1(Z=S)=1.
$$

Within this subset, define $A=Z=S$. It contains 288 treated observations and 817 controls. Only these 1,105 rows may train a representation, outcome regression, or propensity diagnostic.

The primary target is the population average treatment effect over all 2,200 randomized participants. The QED construction targets this population estimand only under the design condition

$$
P(Z=1)=P(Z=0)=\tfrac12,
\qquad Z\perp(S,X,W,Y(0),Y(1)).
$$

Under that condition,

$$
P(C=1\mid S,X,W,Y(0),Y(1))=\tfrac12,
$$

so retaining every row with $C=1$ does not change the population distribution of pretreatment variables or potential outcomes. No additional analysis deletion is allowed. The noisy benchmark is the unadjusted full-sample randomized contrast

$$
\widehat\tau_{\mathrm{RCT}}
=
\overline Y_{Z=1}-\overline Y_{Z=0}.
$$

No rows are removed from this benchmark. The package documentation identifies $Z$ as randomized assignment, but the frozen snapshot does not state the exact assignment probability in the included documentation. The population-ATE interpretation of the QED estimator therefore relies on the explicit $1/2$ design condition above. The benchmark remains a noisy randomized estimate, not the true causal effect.

For learned adjustments, nuisance functions are trained only on QED rows. Out-of-fold conditional-effect predictions are also averaged over the 1,105 QED rows only. Under the $1/2$ randomization condition, this QED distribution represents the full population. No variable from a randomized-discordant row is supplied to representation learning, outcome regression, propensity fitting, or standardization. The full 2,200 rows enter only the scalar randomized difference-in-means benchmark and its standard error.

## Pretreatment variables

The direct covariate vector $X$ contains 14 baseline variables:

`female`, `white`, `black`, `hisp`, `asian`, `married`, `age`, `collegeS`, `collegeM`, `collegeD`, `calc`, `books`, `mathLike`, and `income`.

The high-dimensional proxy vector $W$ contains 114 item responses:

- Big Five: 44 items
- Abbreviated Math Anxiety Scale: 9 items
- Beck Depression Inventory: 13 items
- General Self-Efficacy Scale: 6 items
- mathematics pretest: 12 items
- vocabulary pretest: 24 items
- Mathematical Confidence Scale pretest: 6 items

`mathSel` is not included in $X$, $W$, or any learned feature because $A=S$ in the QED subset. All posttest and post-training variables are excluded. The MCS file is filtered to `type == "pre"` before row alignment is checked.

Before analysis, the five Big Five aggregate scores must be reconstructed exactly for all 2,200 rows. Responses are reverse transformed as $6-x$. The grouped-column reverse keys are `O7,O9`; `C2,C4,C5,C9`; `E2,E5,E7`; `A1,A3,A6,A8`; and `N2,N5,N7`. These are the standard BFI-44 reverse keys after translating the original item numbers into the WSC trait-grouped column order. The key is grounded in the [University of Wisconsin BFI scoring page](https://arc.psych.wisc.edu/self-report/big-five-inventory-bfi/) and the WSC documentation, which states that `Big5_WSC` contains positive and negative items requiring reverse coding. The five reconstructed sums must exactly equal `big5O`, `big5C`, `big5E`, `big5A`, and `big5N` in the main table. This check is added before result interpretation and does not change the already locked audit masks.

## Locked audit masks

The ordered 114 proxy names are formed by the block order above and the frozen CSV column order. Five indices are selected without replacement by NumPy `default_rng(190602)`. The resulting audit items are fixed before outcomes are inspected:

1. `gses:item_2`
2. `big5:C8`
3. `big5:A4`
4. `big5:O3`
5. `mcs_pre:item_1`

For orientation $j$, the representation input is $(X,W_{-j})$ and the untouched audit view is $(X,W_j)$. The audit item is excluded before all preprocessing and representation training. The primary proposed estimate is the arithmetic mean of the five orientation-specific estimates. This cross-mask construction prevents literal copying of the scored audit dimension, but it does not prove conditional independence of proxy views.

## Methods

All learned methods use identical outer folds, outcome learner specification, preprocessing protocol, seeds, and training-row restrictions.

1. `naive_qed`: raw QED difference in means. This is not population-standardized.
2. `x_direct`: outcome adjustment using $X$ directly.
3. `full_xw_direct`: outcome adjustment using $(X,W)$ directly.
4. `x_score_average`: five $X$-only scalar representations, each trained against exactly the same untouched audit view as its paired cross-mask orientation; orientation estimates are averaged.
5. `cross_mask_average`: five scalar representations using $(X,W_{-j})$ and the paired untouched audit view; orientation estimates are averaged.

The direct and representation adjustments use the same two-arm gradient-boosting outcome-regression specification. For each outer fold, both outcome models are fit on QED rows outside the fold and predict QED rows inside the fold. Thus every standardized effect prediction is out of fold. No outcome enters representation fitting.

## Representation objective

The scalar score is initialized by treatment-only logistic loss and then optimized with Adam to reduce a finite random-feature conditional cross-covariance surrogate between $A$ and the untouched audit view given the score. The score variance floor, ridge value, random-feature counts, optimizer, learning rates, and step counts are fixed in code before execution. The surrogate is not called exact KCI.

## Held-out discrepancy

For each outer evaluation fold, finite random features and their bandwidth are fitted using only the corresponding outer-training audit data. Within the QED evaluation rows, two inner folds are formed. Residual regressions are fitted on one inner fold and cross-covariance is scored on the other, then reversed. The same evaluation specification is used for the paired `x_score` and `cross_mask` methods.

Discrepancies are comparable only within the same audit orientation and frozen evaluation specification. No discrepancy ranking is reported for `full_xw_direct`, because its inputs contain every audit item.

## Overlap and effective-sample-size gate

Propensities are estimated out of fold on QED rows in each method's adjustment space. No observation is trimmed and no weight is clipped for the gate. A method passes one repetition only if:

- at least 90% of predicted propensities are in $[0.05,0.95]$;
- ATE inverse-probability ESS is at least `max(25, 0.20 * observed arm size)` in each arm;
- each of five pooled propensity-quantile cells contains at least five treated and five control observations.

`cross_mask_average` passes only if all five orientations pass. Failed estimates remain in the JSON as diagnostics but are not presented as supported causal estimates.

## Pilot budget and reporting

- root seed: 190602
- algorithmic repetitions: 3, explicitly exploratory
- outer folds per repetition: 3
- inner evaluation folds: 2
- audit orientations: 5
- no outcome-based hyperparameter selection

For each method report the estimate, signed and absolute gap from $\widehat\tau_{\mathrm{RCT}}$, variation across the three algorithmic repetitions, propensity range, central-overlap fraction, arm-specific ESS, local-support gate, and failures. For paired `x_score` and `cross_mask` orientations, report the held-out discrepancy and whether the cross-mask method reduces it. The quantity

$$
\left|\widehat\tau_{\mathrm{method}}-\widehat\tau_{\mathrm{RCT}}\right|
$$

is called a benchmark gap, never true bias.

## Interpretation rule

This pilot is encouraging only if the proposed estimate has a smaller benchmark gap than both `naive_qed` and `x_direct`, the improvement is stable across repetitions, and the overlap gate passes. A smaller held-out surrogate without overlap or estimation improvement is not sufficient. No result can establish proxy validity, injectivity, or point identification in WSCdata.
