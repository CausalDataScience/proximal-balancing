# Real-World Data Audit for Single Proxy Balancing

Date: 2026-08-23 CDT

## Audit conclusion

WSCdata can support the requested benchmark without simulating treatment or outcome. The construction uses the study's randomized assignment to obtain an experimental reference and its pretreatment training preference to form a selection-confounded observational comparison. The dataset also contains 114 pretreatment item responses that can serve as a high-dimensional proxy surface.

This construction does not prove that the proxy assumptions hold. The hidden variable is not observed, and neither conditional proxy validity nor injectivity is testable from WSCdata alone. WSCdata is valuable because the randomized cells reveal the causal target against which a proxy method can be evaluated.

LaLonde is a lower-dimensional supporting benchmark. RHC is an application dataset with no experimental causal truth and an unresolved timing problem for physiological proxy candidates.

## Source and reproducibility status

The local Zotero index was searched first and did not contain a directly usable entry for these dataset snapshots. The audit therefore uses the official public repositories and data pages.

| Dataset | Frozen upstream revision | Reuse status | Primary local file | Primary SHA-256 |
| --- | --- | --- | --- | --- |
| WSCdata | Git commit `6360a5247567c0ff632cd12aa2f6b3cfff92b2b6`, package version `0.1.1.9000` | MIT | `wscdata/df2200_cleaned.csv` | `cc67f453da430840a2262538ca8ba14cf2fc49bc10050225d570d649dca8cdaf` |
| LaLonde | Git commit `c179a80e397178185983c272e1c1105d7ca53bf3` | MIT repository license, with upstream data credit retained | `lalonde/nsw_dw.dta` | `d1bd2680a1c6f799f1c6d2455bf29633fdf19be01cb19490621c20a560b4e072` |
| RHC | HBiostat snapshot accessed 2026-08-23 | Public-use permission with requested attribution, no standard license stated | `rhc/rhc.csv` | `9ef4ab578be4b40ad5d97d3a7e08ffdc1f9f76aeeefee51b4996e4221556f8e8` |

The complete URL, byte-size, and SHA-256 record for all 33 stored source files is in `materials/real_world_data/manifest.json`.

## 1. WSCdata

### 1.1 Observed variables

The main table has 2,200 participants and 33 columns. It has no missing value. The variables needed for the math-training benchmark are:

- `mathGrp`: randomized assignment, with $1$ for math training and $0$ for vocabulary training.
- `mathSel`: pretreatment preference, with $1$ for choosing math training and $0$ for choosing vocabulary training.
- `mathPost`: post-treatment math score.
- `vocabPost`: post-treatment vocabulary score.
- Pretreatment summaries: demographics, education, books read, mathematics liking, personality, anxiety, depression, confidence, self-efficacy, vocabulary pretest, and mathematics pretest.

The four observed cells are:

| Randomized assignment `mathGrp` | Preference `mathSel` | Interpretation | Count |
| ---: | ---: | --- | ---: |
| $0$ | $0$ | preferred vocabulary, assigned vocabulary | 817 |
| $0$ | $1$ | preferred math, assigned vocabulary | 319 |
| $1$ | $0$ | preferred vocabulary, assigned math | 776 |
| $1$ | $1$ | preferred math, assigned math | 288 |

The assignment margins are 1,136 assigned to vocabulary and 1,064 assigned to math. The preference margins are 1,593 preferring vocabulary and 607 preferring math.

### 1.2 Exact observational-confounding construction

Let $Z$ denote the randomized assignment recorded as `mathGrp`. Let $S$ denote the pretreatment preference recorded as `mathSel`. Let $Y$ denote the post-treatment math score `mathPost`.

Define the concordance indicator

$$
C \triangleq \mathbb{1}(Z=S).
$$

The observational comparison retains only participants with $C=1$. It then defines the received treatment as

$$
A \triangleq Z.
$$

Within the retained sample, $A=S$. The treated group is the $(Z,S)=(1,1)$ cell with 288 participants. The control group is the $(Z,S)=(0,0)$ cell with 817 participants. The observational sample therefore has 1,105 participants.

This restriction creates treatment selection relative to the analysis. Any pretreatment latent trait $U$ that affects both preference $S$ and the post-treatment outcome $Y$ becomes a common cause of $A$ and $Y$ in the retained sample. Candidate components of $U$ include mathematics ability, motivation, mathematics anxiety, confidence, self-efficacy, and stable personality traits.

The construction creates a confounded comparison, not a known numerical latent variable. The data do not reveal the true value or dimension of $U$.

### 1.3 Experimental reference targets

Randomized assignment supplies three references:

1. The full-sample ATE reference compares $Z=1$ with $Z=0$, using 1,064 and 1,136 participants.
2. The effect among participants who preferred math compares $(Z,S)=(1,1)$ with $(Z,S)=(0,1)$, using 288 and 319 participants. Randomization within $S=1$ identifies the effect for the math-preferring subgroup.
3. The effect among participants who preferred vocabulary compares $(Z,S)=(1,0)$ with $(Z,S)=(0,0)$, using 776 and 817 participants. Randomization within $S=0$ identifies the effect for the vocabulary-preferring subgroup.

For a treated-population benchmark, the natural observational task uses $(1,1)$ as treated and $(0,0)$ as observational controls, then compares the resulting estimate with the randomized $S=1$ reference from $(1,1)$ versus $(0,1)$. The target and weighting scheme must be frozen before Step 2. The two-cell observational contrast should not be labeled an ATE without this target specification.

### 1.4 High-dimensional proxy surface

The public source contains the following pretreatment item tables:

| Block | Rows | Pretreatment items | Missing entries |
| --- | ---: | ---: | ---: |
| Big Five personality | 2,200 | 44 | 0 |
| Mathematics anxiety, AMAS | 2,200 | 9 | 0 |
| Depression, BDI | 2,200 | 13 | 0 |
| General self-efficacy, GSES | 2,200 | 6 | 0 |
| Mathematics pretest | 2,200 | 12 | 0 |
| Vocabulary pretest | 2,200 | 24 | 0 |
| Mathematics confidence, pretreatment portion | 2,200 | 6 | 0 |

These blocks provide 114 pretreatment item coordinates. A defensible first split is:

- $X$: demographics, age, education, income, prior calculus, books read, and mathematics liking.
- $W$: the 114 pretreatment item responses.
- $Y$: `mathPost` for the math intervention analysis.

When item responses are used in $W$, their aggregate scores should not also be copied into $X$ without a prespecified reason. Otherwise the same measurements enter both views.

The item CSV files contain no participant identifier. The package preparation script imports them in their stored row order, but it does not perform an identifier-based join. A rowwise audit nevertheless provides strong alignment evidence. For every participant, the AMAS, BDI, GSES, mathematics-pretest, vocabulary-pretest, pretreatment-MCS, and post-treatment-MCS item sums exactly equal the corresponding score in the main table. Big Five scores require reverse coding, so that block was not resolved by a raw-sum check. Step 2 must preserve source row order, assert these score equalities again after every merge, and separately verify the Big Five reverse-coding key.

The raw mathematics-confidence file has 4,400 rows, not 2,200. It contains 2,200 `pre` rows and 2,200 `post` rows. The package documentation reports 2,200 cases while also describing pre and post records. The rowwise sum check shows that the `pre` rows exactly reproduce `MCS` and the `post` rows exactly reproduce `MCSpost`. Only records with `type == "pre"` are admissible proxy candidates. This filtering rule must be an asserted preprocessing invariant.

Posttest items, mathematics-training-session responses, `mathPost`, `vocabPost`, and `MCSpost` are excluded from the proxy surface because they occur after treatment assignment or during treatment.

### 1.5 What WSCdata establishes and what it does not

WSCdata establishes that a method can be tested against randomized reference effects while operating on a selection-confounded observational comparison. It also provides rich pretreatment measurements that are plausible proxies for the traits producing selection.

WSCdata does not establish either of the following structural conditions:

$$
W \perp\!\!\!\perp A \mid (X,U)
$$

or injectivity of the conditional measurement channel from $U$ to $W$. The first condition may fail if a particular survey answer directly influences the reported training preference beyond the latent trait represented by $U$. Injectivity may fail if distinct latent trait distributions produce the same item-response distribution.

The randomized reference can detect whether the final causal estimate is accurate. It cannot prove why the proxy method succeeded.

## 2. LaLonde

### 2.1 Stored samples

| File | Rows | Treatment counts | Columns | Missing entries |
| --- | ---: | --- | ---: | ---: |
| `nsw.dta` | 722 | 297 treated, 425 controls | 10 | 0 |
| `nsw_dw.dta` | 445 | 185 treated, 260 controls | 11 | 0 |
| `cps_controls.dta` | 15,992 | all controls | 11 | 0 |
| `psid_controls.dta` | 2,490 | all controls | 11 | 0 |

The treatment is participation in the National Supported Work program. The outcome is 1978 earnings, `re78`. Baseline covariates include age, education, race, marital status, degree status, and pretreatment earnings.

### 2.2 Benchmark construction

The standard observational comparison combines NSW treated participants with CPS or PSID nonexperimental controls. Selection into the NSW treated sample versus an external survey control sample creates confounding. The experimental NSW controls provide an ATT reference.

The Dehejia-Wahba subset contains two repeated pretreatment earnings measurements, `re74` and `re75`. They can be treated as low-dimensional proxies for latent employability or earnings capacity. This dataset is useful as a supporting negative or stress-test benchmark because it has only two natural proxy coordinates and no independent high-dimensional proxy block.

Proxy validity and injectivity remain unverified. The repo has an MIT license, while the included source note also requests credit to Dehejia and Wahba and points to the NBER source.

## 3. Right Heart Catheterization

### 3.1 Observed variables

The RHC table has 5,735 patients and 62 data variables after excluding the exported row index. Treatment `swang1` has 2,184 RHC patients and 3,551 non-RHC patients. Candidate outcomes include:

- `dth30`: death within 30 days, with 1,918 deaths and 3,817 survivors.
- `t3d30`: days to death or censoring at 30 days, with no missing value.
- `surv2md1`: a survival summary, with no missing value.

The file contains 13,873 missing entries across five columns. Most occur in `cat2`, `adld3p`, `urin1`, and death or discharge dates. The treatment and listed outcome candidates are complete.

### 3.2 Proxy candidates and timing risk

Ten common physiological measurements are available without missing values in this processed file:

`sod1`, `pot1`, `crea1`, `bili1`, `alb1`, `pafi1`, `paco21`, `ph1`, `wblc1`, and `hema1`.

They are plausible measurements of latent acute severity. The source documentation says that the dataset pertains to the first qualifying hospital day and that treatment records whether RHC occurred on that day. It does not provide measurement timestamps that prove every physiological value preceded catheter placement. Therefore these variables cannot yet be labeled pretreatment proxies. Some cardiopulmonary measurements may also be affected by care delivered during the same day.

The complete laboratory fields may reflect preprocessing or imputation. That provenance was not established in this audit.

RHC has no randomized causal reference. It can serve as an application or sensitivity-analysis example after timing is resolved, but it cannot be the main accuracy benchmark.

## Dataset ranking for Step 2

| Rank | Dataset | Recommended role | Main reason |
| ---: | --- | --- | --- |
| 1 | WSCdata | Primary benchmark | Observational selection challenge, randomized reference, and 114 pretreatment item coordinates in one study |
| 2 | LaLonde | Supporting stress test | Experimental ATT reference and observational external controls, but only two natural repeated proxy measurements |
| 3 | RHC | Application only | Rich severity measurements, but no causal truth and unresolved same-day timing |

## Step 2 gates

No modeling should start until the following WSC decisions are frozen:

1. Preserve item-to-participant row order, rerun the exact aggregate-score checks after every merge, and verify the Big Five reverse-coding key. An identifier-bearing source or author confirmation would provide additional assurance.
2. Filter the 4,400-row mathematics-confidence file to `type == "pre"` and assert that its row sums equal `MCS`.
3. Select the primary estimand. The treated-population effect for $S=1$ is the cleanest within-study comparison target.
4. Freeze the $X$ and $W$ split without duplicating item aggregates across both views.
5. Prespecify the representation-training fold, audit fold, and randomized reference evaluation.
6. State that observed performance evaluates the method but does not test conditional proxy validity or injectivity.

This audit stopped before model fitting, causal estimation, or Monte Carlo analysis.
