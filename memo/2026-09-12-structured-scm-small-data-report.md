# Small-data SCM experiment: failure diagnosis, fix, and five frozen settings

Date: 2026-09-12  
Status: complete for the structured-moment positive control and the restricted parametric Brier confirmation; neural representation learning remains unverified

**한국어 판정.** 총 표본 6,000명에서 구조화된 moment learner는 최종 G1–G5 모두 사전 기준을 두 개의 독립 20회 batch에서 각각 통과했다. 같은 표본 예산의 restricted parametric Brier learner도 새 독립 seed 20회에서 다섯 설정 모두 통과했다. 이는 같은 식별 topology를 공유하는 다섯 stress setting에 대한 확인이며, neural PROBE나 전역 ERM 최적화, 원고의 finite-sample certificate를 확인한 것은 아니다.

## 1. Bottom line

The project now has five reproducible SCM **stress-test settings** for which a data-learned representation, all 30 split candidates, independent screening, and Algorithm 1 aggregation work at a literal total sample size of 6,000. They are not five unrelated causal identification families: all five share the same proxy-availability topology and differ in outcome noise, effect heterogeneity, proxy heteroskedasticity, or observation nonlinearity.

The structured observed-moment learner does not know $U$, the valid subsets, the ATE, or outcomes during representation learning and screening. It is a useful positive control for Sections 2–4. A second implementation now confirms the manuscript's same-$D$ Brier objective within prespecified bounded-probit linear critic and scalar-encoder classes; it is not evidence that the earlier neural Brier implementation works.

The main empirical conclusion is therefore two-part:

1. The identification and voting construction is operationally realizable with data-learned representations and wrong surviving candidates.
2. The restricted parametric same-$D$ Brier implementation passes all frozen small-data performance gates in five settings after the combined rotate-four cross-fitting/sample-reuse fix. Neural-Brier success and the unrestricted theory-level guarantee remain unverified.

## 2. Exact common SCM

Draw mutually independent standard normal variables

$$
U,I,C,e_X,e_0,\ldots,e_4,e_Y\sim N(0,1).
$$

Set

$$
X=U+2e_X,
\qquad
M_j=U+\sigma_j e_j.
$$

The five observed proxy blocks are

$$
W_0=(I,C,T(M_0)),\quad
W_j=T(M_j),\ j=1,2,3,\quad
W_4=T(M_4-0.6I).
$$

Treatment follows the bounded probit model

$$
P(A=1\mid U,I,C,X)
=0.1+0.8\Phi(U+2.5I+0.2X+0.3C).
$$

The ATE is $1$ in every setting. The settings are:

| Setting | Outcome | Proxy noise | Observation map |
|---|---|---|---|
| G1 | $Y(a)=a-2.5U+0.2X-0.5C+0.3e_Y$ | all $\sigma_j=0.85$ | $T(x)=x$ |
| G2 | $Y(a)=a-2.5U+0.2X-0.5C+(0.3+0.2\lvert U\rvert)e_Y$ | all $\sigma_j=0.85$ | $T(x)=x$ |
| G3 | $Y(a)=a(1+0.3U)-2.7U+0.2X-0.5C+0.3e_Y$ | all $\sigma_j=0.85$ | $T(x)=x$ |
| G4 | $Y(a)=a-2.5U+0.2X-0.5C+0.3e_Y$ | $(0.8,0.9,1.0,0.85,0.95)$ | $T(x)=x$ |
| G5 | $Y(a)=a-2.5U+0.2X-0.5C+0.3e_Y$ | all $\sigma_j=0.85$ | $T(x)=\sinh x$ |

For G5, the learner estimates each marginal empirical CDF on its representation-training fold, uses ranks $r/(n_D+1)$, maps them through $\Phi^{-1}$, and divides by the observed covariance with $X$. Nothing is fitted on the screen, nuisance, or evaluation fold.

Lowering outcome noise was an adaptive second design round made after the first confirmation missed the predeclared 90th-percentile gate. It leaves every conditional outcome mean, ATE, population adjustment functional, balance root, and vote geometry unchanged. It changes only finite-sample outcome-regression and AIPW variance.

G3 has marginal ATE $E(1+0.3U)=1$, but $1+0.3U$ is negative in an extreme Gaussian tail. Its Simpson claim is therefore about reversal relative to the positive marginal ATE, not a treatment effect that is positive in every latent stratum.

## 3. Population checks

The population values were computed separately by Gaussian quadrature. These use the outcome conditional mean, so G1/G2/G5 share a row.

| Setting | Naive | $X$-only | Raw $(X,W)$ | Exact clean representation | Oracle |
|---|---:|---:|---:|---:|---:|
| G1/G2/G5 | -0.1742 | 0.0480 | 0.8256 | 1.0000 | 1.0000 |
| G3 | -0.1998 | 0.0303 | 0.8221 | 1.0000 | 1.0000 |
| G4 | -0.1742 | 0.0480 | 0.8110 | 1.0000 | 1.0000 |

Thus all five exhibit Simpson reversal, raw-adjustment bias above 0.15, and an exact clean representation equal to the oracle target.

For G1, the clean roots when the complement contains $k=2,3,4$ measurement coordinates are approximately $0.4540,0.3004,0.2245$. Their population moment slopes are $0.1580,0.1868,0.2036$, respectively. The deliberately wrong singleton has a retained root near $-0.6502$ and population adjusted value about $-0.4432$.

The standalone population checker evaluates Gauss–Hermite rules of orders 40, 80, and 160 and an independent adaptive quadrature. The maximum order-160 versus adaptive-quadrature difference is $1.04\times10^{-7}$. It also substitutes every clean root into the covariance equations and the same adjustment integral. Here $c_{UL}=\operatorname{Cov}(U,L\mid Z)$ is the conditional covariance between the latent confounder and treatment index. The maximum absolute root residual is $2.78\times10^{-16}$, maximum $|c_{UL}|$ is $3.33\times10^{-16}$, maximum clean $|\theta_Z-1|$ is $3.33\times10^{-16}$, and the computed oracle error is zero. These are numerical population checks, not finite-sample certificates.

## 4. Learner and Algorithm 1

All 30 nonempty proper subsets $S\subset\{0,1,2,3,4\}$ are processed. The evaluation-only structural categories are 7 clean subsets, the bad singleton $\{4\}$, 7 mixed subsets containing block 4, and 15 subsets containing block 0. These tags never enter fitting or selection.

When block 0 is in the complement, its typed channels $I,C$ are available and the encoder is

$$
Z_r=(X,C,\bar K_{-S}+rI),\qquad r\in[-1,1].
$$

When block 0 is held out, those channels are unavailable and $Z=(X,\bar K_{-S})$. The moment learner minimizes the symmetric sum of squared observed partial correlations between $A$ and every held-out coordinate given $Z_r$. It searches every local minimum bracketed by a fixed 201-point grid. Values within $10^{-12}$ of the minimum tie, followed by the truth-free rule: smallest $|r|$, then smallest $r$.

The independent screen keeps a candidate only if the maximum absolute studentized residual covariance over all held-out coordinates is at most 3.5. This is a zero-null nonrejection rule, not an equivalence test or causal-validity certificate.

Four independent folds of 1,500 units give exactly 6,000 generated units. Four fixed role rotations assign every fold once to representation, screen, nuisance, and evaluation. A split survives only if it passes all four screens. Its four disjoint evaluation score vectors are concatenated. The procedure then uses a common 5,000-draw Gaussian multiplier radius, joins candidate estimates at distance at most $2\rho$, and returns the median of every largest connected component. Tied largest components remain set-valued.

Because each evaluation fold participates in another rotation's fitting roles, this is an operational cross-fitted extension. It is not the manuscript's one-shot conditional finite-sample guarantee. The radius covers evaluation-score sampling only; it does not cover learned-representation, screening, or nuisance error.

## 5. What actually failed, and what was fixed

### 5.1 Implementation defects

The first transient moment code found roots using only the first held-out coordinate and then compared the full vector. That depended on coordinate order and was not the joint empirical minimizer. The persisted implementation instead minimizes the symmetric squared partial-correlation norm, precomputes the covariance matrix, and has an order-invariance test.

An attempted redesign that moved $I$ into globally observed baseline covariates was rejected before execution. With the manuscript target $T_S=(X,W_S)$, that construction would require simultaneous balance of $I$ and a noisy clean proxy and has no exact scalar-compression root. The rejected construction remains in the candidate ledger.

The first diagnostic protocol also contained a manually entered future `frozen_at` time. Its bytes did precede its three-replicate outputs, but the timestamp itself was false. The original files were preserved and a separate correction artifact records the actual filesystem times and hashes. Later protocols use the runtime UTC clock.

### 5.2 Statistical design defects

With a hard four-way split, only 1,500 evaluation units contribute to the final ATE. For the original Q2 setting, three development outputs were $1.0719,1.0188,0.8466$, giving MAE $0.0813$ and 90th-percentile error $0.1371$. Q2 also passed 4, 7, and 7 mixed invalid candidates; the common radius was $0.304$–$0.346$, allowing graph chaining.

The stronger F1 treatment instrument removed the mixed screen false positives, but hard-split MAE remained $0.1087$. Thus stronger screening alone did not solve small-sample ATE variation.

With the exact same total 6,000 units, fixed four-role rotation reduced the common radius to roughly $0.10$–$0.15$. Three development F1 outputs had MAE $0.0316$ and 90th-percentile error $0.0445$. This is evidence that combined cross-fitting and sample reuse addresses a major practical bottleneck; it does not isolate evaluation sample size from changes in representation, screening, and nuisance averaging.

### 5.3 The Brier issue is not resolved by switching objectives

The fact that the Brier discrepancy is squared does not itself force an $n^{-1/4}$ estimator rate. An $n^{-1/4}$ bound arises from taking the square root of a global $O_p(n^{-1/2})$ excess-risk bound. In a correctly specified, locally strongly curved one-dimensional parametric class, localized M-estimation can still estimate the root at $n^{-1/2}$.

The current evidence supports four possible Brier failure mechanisms: weak local curvature relative to empirical critic noise, multiple minima, critic approximation/optimization error, and amplification of root error into causal error. The moment learner exposes a first-order observed signal and is therefore a stable positive control; it does not prove Brier optimization is intrinsically impossible.

## 6. Confirmation results

### 6.1 First confirmation with standard outcome noise

All five F settings passed MAE, return, paired comparison, and oracle-excess criteria. Only F2 passed the 90th-percentile error criterion. The other values were:

| Setting | MAE | 90th percentile | Full verdict |
|---|---:|---:|---|
| F1 | 0.0416 | 0.1046 | Fail |
| F2 | 0.0349 | 0.0796 | Pass |
| F3 | 0.0418 | 0.1054 | Fail |
| F4 | 0.0440 | 0.1088 | Fail |
| F5 | 0.0423 | 0.1069 | Fail |

No threshold was changed and these failures remain part of the record.

### 6.2 Frozen G settings: two independent batches

Each cell below contains `MAE / 90th-percentile absolute error`.

| Setting | Batch 1, 20 reps | Batch 2, 20 reps | Pooled 40 |
|---|---:|---:|---:|
| G1 | 0.0295 / 0.0671 | 0.0230 / 0.0393 | 0.0262 / 0.0578 |
| G2 | 0.0315 / 0.0643 | 0.0262 / 0.0441 | 0.0288 / 0.0579 |
| G3 | 0.0316 / 0.0662 | 0.0239 / 0.0399 | 0.0277 / 0.0625 |
| G4 | 0.0311 / 0.0711 | 0.0248 / 0.0429 | 0.0280 / 0.0590 |
| G5 | 0.0298 / 0.0654 | 0.0232 / 0.0421 | 0.0265 / 0.0597 |

Every setting returned one unique-largest-component point in all 20 replicates of each batch. Every setting passed every frozen gate separately in batch 1, separately in batch 2, and pooled across 40 replicates.

For the pooled results, the paired absolute-error confidence intervals versus raw $(X,W)$ were:

| Setting | PROBE minus raw error, 95% CI | PROBE minus oracle error, 95% CI |
|---|---:|---:|
| G1 | [-0.164, -0.139] | [0.011, 0.026] |
| G2 | [-0.164, -0.136] | [0.008, 0.024] |
| G3 | [-0.166, -0.141] | [0.012, 0.027] |
| G4 | [-0.177, -0.152] | [0.012, 0.028] |
| G5 | [-0.164, -0.139] | [0.011, 0.026] |

The paired intervals use independent replicate seed-bases as the inference units and the usual two-sided $t$ critical value. They are per-comparison 95% intervals without multiplicity adjustment.

### 6.3 Voting evidence

Across both batches, the bad singleton survived in 37–38 of 40 replicates depending on setting, so the exercise was not reduced to screening away every wrong candidate. It entered the selected largest component zero times. Mixed and block-0 subsets were retained zero times. The selected component contained only clean candidates, usually 6 or 7 and at least 3. Algorithm 1 therefore actually defeated a plausible wrong survivor rather than using a known-valid label.

A conditional split-budget replay sampled $m=10,20,30$ candidates without replacement from each already fitted 30-candidate library, using 1,000 draws for each of the 40 data seeds and the stored full-library radius. Across G1–G5, the rates were:

| $m$ | No return | Tied largest | Unique return | Unique and $\lvert\widehat\tau-1\rvert\leq0.1$ |
|---:|---:|---:|---:|---:|
| 10 | 0.0316–0.0329 | 0.0749–0.0770 | 0.8906–0.8932 | 0.8452–0.8549 |
| 20 | 0.0001–0.0003 | 0.0070–0.0075 | 0.9923–0.9929 | 0.9651–0.9731 |
| 30 | 0 | 0 | 1.0000 | 0.9750 |

This is a conditional replay of fitted libraries, not fresh representation/nuisance fits and not an independent sampling distribution or confidence interval. It shows that the candidate budget materially affects return behavior; it does not validate a theorem-level budget formula.

## 7. Restricted Brier end-to-end development result

The corrected restricted Brier arm uses bounded-probit linear critic classes. For each of 101 fixed $r$ values, the base critic is approximately minimized from one zero start; the augmented critic is approximately minimized from the lifted-base and zero starts. The best optimized or unoptimized start is retained. This makes the lifted base critic with zero held-out coefficients an explicit fallback, so the empirical nested gap cannot be negative beyond numerical tolerance. Both risks are evaluated on the same representation fold. An independent screen fold is used only after encoder selection. BFGS now uses the exact analytic gradient; a fixed-seed comparison reproduced the previous two development outputs exactly, with maximum empirical-objective difference $1.32\times10^{-12}$.

The Brier screen retains a split only when $\max(\widehat D^2,0)\leq2\widehat{\operatorname{SE}}(\widehat D^2)$ on the independent screen fold. This is a one-sided zero-null nonrejection rule, not an equivalence test, a balance certificate, or a causal-validity test.

All final estimators fit a bounded-probit propensity and arm-specific linear outcome regressions augmented by the appropriate selection-correction regressor. $X$-only, raw $(X,W)$, oracle, and PROBE use the same nuisance/evaluation folds and budget; raw uses all observed features, with the same representation-fold-fitted marginal transform in G5. No true nuisance functions are supplied; latent $U$ is supplied only to the explicitly labeled oracle, never to PROBE or observed-data baselines.

On G1 with a hard four-way split and two development seeds, the selected components contained all seven clean candidates and no mixed or block-0 candidates. One bad singleton survived in the first replicate but did not enter the chosen component. The ATE outputs were $1.1806$ and $0.8759$. Thus correct component selection alone did not overcome the hard-split sampling variation.

A predeclared rotate-four comparison then produced $1.0278$ and $0.9793$. Both outputs were unique-largest-component medians of six clean candidates. The bad singleton survived in one replicate but entered neither selected component; mixed and block-0 subsets were rejected. Across 240 fitted split-rotation comparisons, every stored empirical nested gap was nonnegative, the minimum was $7.59\times10^{-9}$, and the unoptimized lifted-base fallback was never selected. This was development evidence; the fresh confirmation below is the performance endpoint.

### 7.1 Fresh restricted-Brier confirmation

The frozen confirmation used 20 new independent base seeds per setting, total $n=6{,}000$ per seed, the same 101-point grid, all-four screen, nuisance fits, common radius, and Algorithm 1. The table reports the mean estimate, PROBE MAE, 90th-percentile absolute error, raw and oracle MAE, and the upper endpoint of the paired 95% $t$ interval for the PROBE-minus-comparator absolute-error difference.

The frozen gates were MAE at most $0.05$, 90th-percentile absolute error at most $0.10$, unique-return rate at least $0.95$, paired absolute-error interval upper endpoint below zero versus both $X$-only and raw $(X,W)$, and oracle-excess interval upper endpoint at most $0.05$.

| Setting | Mean | PROBE MAE | 90th pct. | Raw MAE | Oracle MAE | Upper vs. raw | Upper oracle excess | Return | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| G1 | 0.9873 | 0.0280 | 0.0541 | 0.1854 | 0.0052 | -0.1386 | 0.0332 | 20/20 | Pass |
| G2 | 0.9859 | 0.0292 | 0.0597 | 0.1843 | 0.0094 | -0.1343 | 0.0308 | 20/20 | Pass |
| G3 | 0.9873 | 0.0294 | 0.0577 | 0.1892 | 0.0062 | -0.1407 | 0.0342 | 20/20 | Pass |
| G4 | 0.9913 | 0.0298 | 0.0563 | 0.2001 | 0.0052 | -0.1490 | 0.0355 | 20/20 | Pass |
| G5 | 0.9881 | 0.0285 | 0.0504 | 0.1859 | 0.0052 | -0.1374 | 0.0333 | 20/20 | Pass |

All five settings pass return rate, MAE, 90th-percentile error, paired superiority to $X$-only and raw $(X,W)$ adjustment, and the predeclared oracle-excess margin. The oracle comparison is a noninferiority margin of $0.05$, not a claim of statistical equality.

The naive estimate was negative in all 20 replicates of every setting. Its mean was $-0.1709$ in G1/G4/G5, $-0.1723$ in G2, and $-0.1963$ in G3, matching the population Simpson-reversal direction.

Every one of the 100 outputs was a unique-largest-component return. The bad singleton survived 16, 16, 16, 18, and 17 times in G1–G5 but entered a selected component zero times. Mixed and block-0 candidates entered zero selected components. Across 12,000 stored selected-point split-rotation diagnostics there were no gaps below $-10^{-10}$. Among the 36,000 stored base/augmented optimization attempts at those selected points, none reported failure; the lifted unoptimized fallback was selected four times. Optimizer statuses over all 101 scanned grid points were not retained, so this does not certify all-grid optimizer success or a global minimum. Thus approximate optimization occasionally used its prespecified nesting safeguard, while the full causal output remained well-defined.

## 8. Reproducibility and evidence boundaries

Canonical implementation:

- `code/probe_structured_scm.py`
- `code/run_probe_structured_scm.py`
- `code/test_probe_structured_scm.py`
- `code/check_probe_structured_population_v2.py`

The G confirmation used a frozen source copy in `results/probe_structured_scm_g_confirm_v1_source/`. Batch outputs are:

- `results/probe_structured_scm_g_confirm_v1.json`, SHA-256 `51cfcb8721d0fbb76a0c867febcaf6e3daa85a6219460ddf9542fb853344566f`
- `results/probe_structured_scm_g_confirm_v1b.json`, SHA-256 `a6e7f3c9ec61aa9a81ed6eaefafd2386c1af9dd68c1f5c64bbd408fd39fa5cdb`

Batch 2 used the same frozen source, settings, gates, and a nonoverlapping seed namespace. Conversation ordering records that it was specified before batch-1 results were inspected, but the durable protocol file's modification time is later than batch 1's output completion. The durable filesystem alone therefore proves pre-execution freezing for batch 2, not pre-completion freezing relative to batch 1.

The restricted-Brier confirmation is specified in `results/probe_structured_scm_brier_g_confirm_protocol_v2.json`, SHA-256 `cf5b0f3b9e678af3763993e044783584015af60d22d69f57f5b781849795fa65`. Its exact source copy is in `results/probe_structured_scm_brier_g_confirm_v2_source/`; the five immutable raw outputs and their SHA-256 values are indexed by `results/probe_structured_scm_brier_g_confirm_summary_v1.json`. The independent population calculation is `results/probe_structured_scm_population_checks_v2.json`. The five settings use the same 20 base seeds for paired comparability, so the inference unit is the seed within each setting; the 100 setting-by-seed rows must not be pooled as 100 independent observations.

Run tests with:

```bash
cd code
PYTHONDONTWRITEBYTECODE=1 python3 -B test_probe_structured_scm.py
```

The final run reported 13 passing tests, including a directional finite-difference check of the analytic Brier gradient. One remaining implementation limitation is that block 0 availability is still represented by numeric index 0 inside `representation` and `heldout`; the experiment follows the intended typed schema, but a full schema-permutation implementation test remains undone. This does not alter the frozen numeric results, but it limits the strength of the software-generality claim.

## 9. Final scientific assessment

The five G settings are suitable as a main positive-control suite for both the structured moment learner and the restricted parametric same-$D$ Brier implementation. They establish Simpson reversal, raw ordinary-adjustment failure, oracle-margin learned adjustment, and nontrivial voting at total $n=6{,}000$.

They should not be presented as five confirmations of a general neural algorithm, a global empirical-risk optimum, or the manuscript's full finite-sample theorem. The fixed 101-point grid can also leave a nonvanishing discretization floor, so this $n=6{,}000$ experiment does not establish asymptotic convergence. The correct empirical claim is narrower: a prespecified restricted Brier critic/encoder class, operational rotate-four reuse, and Algorithm 1 passed the complete frozen small-data protocol in all five stress settings. A full neural-encoder experiment remains the next separate question.
