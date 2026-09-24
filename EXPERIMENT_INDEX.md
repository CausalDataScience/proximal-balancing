# Experiment index

Created 2026-09-22 CDT at the author's instruction; IHDP and ACIC added the same evening. **Only the seven experiments below are final.** Anything
not listed here is not final and is not cited as evidence in the paper. Every entry names its frozen
record, its code, its result files and its figure, so each number can be traced to a file.

MAE is the mean absolute error of the estimate against the known effect, over the listed replicates.
"Role-requiring" marks methods that must be told which proxy block plays which part.

| ID | Experiment | Replicates | Known answer |
|---|---|---|---|
| E1 | SCM-1, SCM-2, SCM-3 sealed confirmation | 20 per level, n = 24,000 | effect = 1 |
| E2 | SCM-4 sealed confirmation, Shapes3D images | 20, n = 24,000 | effect = 1 |
| E3 | Observed RHC, pre-registered | one data set, n = 5,735 | none |
| E5 | Twins, prospective replicates 11 to 20 | 10, n = 11,984 | exact, −0.0252 |
| E6 | IHDP, prospective replicates 12 to 21 | 10, n = 747 | exact, 3.12 to 12.65 by replicate |
| E7 | ACIC 2016, realizations 2 to 10, exploratory | 9, n = 4,802 | exact, by realization |
| E8 | KSPC benchmark, binary treatment | 20, n = 3,000 | effect = 1 |

E4 is retired: the KSPC benchmark was renumbered E8 on 2026-09-23 at the author's instruction. E5 to E7
keep their numbers because frozen folder names use them (`results/kspc_comparator/E5` and so on).

**Rows marked † (all experiments except E8).** KSPC of Xu and Gretton with observed covariates, from Appendix
A of their paper, in our implementation (`code/kspc_wrapper.py`, the authors' code having no covariate
argument), estimator SKPV only, run on data rebuilt from each frozen seed after the protocols were frozen
(`memo/2026-09-22-kspc-comparator-prd-v2.md`). Gates: `results/kspc_comparator/gates_v1.json` (gate 2 failed
for SPMMR, which is therefore not run); integrity: `results/kspc_comparator/code_integrity_v1.json` and a
per-replicate check against each sealed shard; results and freeze: `results/kspc_comparator/`,
`FROZEN_v1.json`; memo `memo/2026-09-23-kspc-comparator-results-v1.md`. KSPC is role-requiring: someone
chooses its proxy block.

## E1. SCM-1, SCM-2, SCM-3 sealed confirmation

- **Design.** The family v2 generator, `code/family_v2_dgp.py`. Covariates X of dimension 2, 8 and 32; five
  proxy blocks, one of them contaminated by a treatment-side variable. Proxy coordinates in total: 8, 88,
  1,312.
- **Frozen record.** `results/family_v2/confirm_protocol_v1.json`, frozen 2026-09-19T03:16:18.425788Z UTC, with the sha256 of
  every source file.
- **Code.** `code/run_family_v2.py`, `code/family_v2_probe.py`, `code/family_v2_competitors.py`,
  `code/probe_structured_scm.py`.
- **Results.** `results/family_v2/confirm/SCM-{1,2,3}_seed*.json`, `results/family_v2/confirm_summary_v1.json`.
  Every gate passed at every level.
- **Figures.** `ICLR/paper/figures/probe-estimates.pdf` (Section 5); `probe-aggregation.pdf`,
  `probe-coverage.pdf`, `probe-learned.pdf` (appendix). Source `code/make_section5_v2_figures.py`.

| MAE | SCM-1 | SCM-2 | SCM-3 |
|---|---|---|---|
| **PROBE** | **0.020** | **0.017** | **0.014** |
| CEVAE | 0.030 | 0.280 | 0.409 |
| 2SLS, roles given (role-requiring) | 0.031 | 0.159 | 0.120 |
| COCA, clean block (role-requiring) | 0.033 | 0.035 | 0.036 |
| adjust for X and all of W | 0.271 | 0.277 | 0.446 |
| adjust for X | 1.442 | 1.430 | 1.430 |
| 2SLS, roles swapped | 5.186 | 5.312 | 5.225 |
| COCA, contaminated block | 5.258 | 5.228 | 5.249 |
| KSPC with X, clean block† | 0.875 | 0.854 | 0.813 |
| KSPC with X, contaminated block† | 1.881 | 1.656 | 1.796 |
| oracle (adjusts for U) | 0.003 | 0.003 | 0.004 |

## E2. SCM-4 sealed confirmation, Shapes3D images

- **Design.** SCM-1's observed coordinates delivered as original Shapes3D images and read back by a frozen
  reader, `code/family_v2_images.py`.
- **Frozen record.** `results/family_v2/scm4/confirm_protocol_v1.json`, frozen 2026-09-21T05:34:06.321510Z UTC.
- **Code.** `code/run_scm4_images.py`, `code/run_scm4.py`, `code/scm4_pixel_cnn.py`.
- **Results.** `results/family_v2/scm4/confirm/SCM-4_seed*.json`,
  `results/family_v2/scm4/confirm_summary_v1_amended.json`. All 13 gates passed. The amendment fills the
  pixel CNN value for the two data sets whose first run timed out (`confirm_pixel_rerun/`); no PROBE number
  changed.
- **Figures.** The SCM-4 panel of `probe-estimates.pdf`. Figure 2, `probe-mechanism.pdf`, shows images from
  the same Shapes3D bank for illustration only; it reports no result.

| MAE | SCM-4 |
|---|---|
| **PROBE** | **0.018** |
| 2SLS, roles given | 0.033 |
| COCA, clean block | 0.038 |
| CEVAE | 0.047 |
| pixel CNN | 0.281 |
| adjust for X and all of W | 0.306 |
| adjust for X | 1.433 |
| 2SLS, roles swapped / COCA, contaminated | 5.136 / 5.216 |
| KSPC with X, clean / contaminated block† | 0.870 / 1.807 |
| oracle | 0.003 |

## E3. Observed RHC, pre-registered

- **Design.** The right heart catheterisation study, 5,735 patients, outcome days survived within 30.
  Ten physiological measurements in five proxy blocks, `code/rhc_data.py`. No known answer.
- **Frozen record.** `memo/2026-09-22-rhc-real-data-prereg-v1.md`.
- **Rule history, stated plainly.** The pre-registered analysis, with the sealed screen threshold, returned
  no estimate (`memo/2026-09-22-rhc-real-data-results-v1.md`). The reported number uses the v3
  noise-scaled rule, `code/noise_scaled.py`, adopted after that result and recorded in
  `memo/2026-09-22-noise-scaled-threshold-v3.md`. The v3 rule was then confirmed prospectively on E5.
- **Results.** `results/rhc/diagnostic_v1_full_records.json`, `results/noise_scaled_v3.json` (entry `rhc`),
  `results/rhc/aggregate_uncertainty_v1.json`.
- **Figure.** `ICLR/paper/figures/probe-rhc.pdf`, source `code/make_rhc_figure.py`.

| effect on days survived | estimate |
|---|---|
| **PROBE, no roles given** (15 of 30 candidates kept) | **−1.29**, resampled 95% −1.88 to −0.70 |
| standard doubly robust, published by Cui et al. | −1.17 |
| proximal doubly robust with roles assigned, published by Cui et al. | −1.66 |
| difference in means, no adjustment | −1.52 |
| KSPC with X, each of the five blocks† | −0.93 to −2.18 |

## E5. Twins, prospective replicates 11 to 20

- **Design.** Real twin-birth outcomes with an exactly known effect. A real covariate, gestational age, is
  hidden as U; proxies and treatment are generated from it by the recipe in `code/hidden_proxy_recipe.py`;
  `code/twins_data.py`. n = 11,984, 42 covariates.
- **Frozen record.** `results/v3_confirmation_protocol.json` and its neutral correction
  `results/v3_confirmation_protocol_v2.json` (frozen 2026-09-22T21:37:01Z UTC; the correction records that same freeze time). Replicates 11 to 20 were named before
  any of them was generated, and the v3 rule was frozen before they ran.
- **Code.** `code/run_v3_fresh.py`, `code/run_twins.py`, `code/v3_fresh_scorer.py`, `code/noise_scaled.py`;
  COCA as an ATE from `code/coca_ate_all.py`.
- **Results.** `results/twins/replicate_{11..20}.json`, `results/twins/summary_v3_fresh.json`,
  `results/coca_ate_all_v1.json`. Both pre-registered gates passed: 10 of 10 returned, MAE 0.0077 against
  a limit of 0.0143.
- **Figure.** The Twins panel of `figures/section5/realdata-prospective.pdf`
  (`ICLR/paper/figures/probe-realdata.pdf`), source `code/make_prospective_figure.py`.

| MAE | Twins 11 to 20 |
|---|---|
| **PROBE** | **0.0077** |
| 2SLS, roles given | 0.0156 |
| adjust for X and all of W | 0.0207 |
| adjust for X | 0.0749 |
| CEVAE | 0.0894 |
| 2SLS, roles swapped | 0.2453 |
| COCA as an ATE | 0.7318 |
| KSPC with X, clean block† | 0.0099 |
| KSPC with X, contaminated block† | 0.0815 |
| oracle | 0.0074 |

## E6. IHDP, prospective replicates 12 to 21

- **Design.** The IHDP covariates of Hill (2011), `materials/real_world_data/ihdp/ihdp_npci_1-100.{train,test}.npz`.
  One covariate is hidden as U and the proxies and treatment are generated from it by the same recipe as
  E5, `code/hidden_proxy_recipe.py`; `code/ihdp_acic_data.py`. n = 747, 24 covariates. The effect is known
  exactly and differs by replicate.
- **Frozen record.** The same v3 protocol as E5, `results/v3_confirmation_protocol.json` and its neutral
  correction `results/v3_confirmation_protocol_v2.json`. Replicates 12 to 21 were named before any of them
  was generated. The v3 rule was chosen after seeing IHDP's development replicates 2 to 11, not these.
- **Code.** `code/run_v3_fresh.py`, `code/run_ihdp_acic.py`, `code/v3_fresh_scorer.py`, `code/noise_scaled.py`,
  `code/coca_ate_all.py`.
- **Results.** `results/ihdp/replicate_{12..21}.json`, `results/ihdp/summary_v3_fresh.json`,
  `results/coca_ate_all_v1.json`. Both pre-registered gates passed: 10 of 10 returned, MAE 0.3054 against a
  limit of 0.5043. The gates compare PROBE with its own earlier accuracy, not with other methods.
- **Figure.** The IHDP panel of `ICLR/paper/figures/probe-realdata.pdf`, source `code/make_prospective_figure.py`.

| MAE | IHDP 12 to 21 |
|---|---|
| CEVAE | **0.1828** |
| 2SLS, roles given | 0.2328 |
| **PROBE** | 0.3054 |
| adjust for X and all of W | 0.3927 |
| adjust for X | 0.7221 |
| 2SLS, roles swapped | 1.5856 |
| COCA as an ATE | 5.9838 |
| KSPC with X, clean block† | 0.6214 |
| KSPC with X, contaminated block† | 1.2499 |
| oracle | 0.4868 |

PROBE is third. Against CEVAE, the most accurate method that needs no roles, its error ratio is 1.67 with a
90% paired-bootstrap interval of 0.78 to 2.78, so the difference is not resolved by ten replicates.

## E7. ACIC 2016, realizations 2 to 10, exploratory

- **Design.** The ACIC 2016 covariates and realizations, `materials/real_world_data/acic2016/`
  (`x.csv`, `zymu_{r}.csv`). One covariate is hidden as U, same recipe as E5; `code/ihdp_acic_data.py`.
  n = 4,802, 78 covariates. The effect is known exactly for each realization.
- **Status, stated plainly.** This is not a prospective test. The v3 rule was chosen after seeing these
  realizations (`results/noise_scaled_v3.json`, field `adopted_after_seeing`). Realization 1 was the
  development set of the earlier design, `memo/2026-09-22-ihdp-acic-prereg-v1.md`. Every figure labels ACIC
  "rule chosen here".
- **Code.** `code/run_ihdp_acic.py`, `code/noise_scaled.py`, `code/coca_ate_all.py`.
- **Results.** `results/acic/summary_v1.json` (comparators), `results/noise_scaled_v3.json` (entry `acic`,
  PROBE under the v3 rule), `results/coca_ate_all_v1.json`. All 9 realizations returned.
- **Figure.** The ACIC panel of `ICLR/paper/figures/probe-realdata.pdf`.

| MAE | ACIC 2 to 10 |
|---|---|
| 2SLS, roles given (role-requiring) | **0.1016** |
| adjust for X | 0.1757 |
| **PROBE** | 0.2025 |
| adjust for X and all of W | 0.2239 |
| CEVAE | 0.2575 |
| 2SLS, roles swapped | 0.8015 |
| COCA as an ATE | 34.6486 |
| KSPC with X, clean block† | 0.5622 |
| KSPC with X, contaminated block† | 0.9318 |
| oracle | 0.1336 |

PROBE is third. Against adjusting for X, the most accurate method that needs no roles, its error ratio is
1.15 with a 90% paired-bootstrap interval of 0.69 to 2.66; 2SLS handed the true roles is clearly better,
1.99 with an interval of 1.24 to 3.06.

## E8. KSPC benchmark, binary treatment (formerly E4)

- **Design.** The generator of Xu and Gretton (arXiv 2308.04585) with one change, a binary treatment on their
  own index: $P(A=1 \mid U) = 0.1 + 0.8\{1 + \mathrm{erf}(U)\}/2$. The outcome stays deterministic,
  $Y = \sin(\pi U/2) + A - 0.3$, so the effect is exactly 1. PROBE gets their two proxies as two blocks,
  $e^U + N(0, 0.1^2)$ and an MNIST image of $\lfloor 5U+5 \rfloor$; KSPC gets one at a time.
  `code/kspc_design.py`, fixed before any run.
- **Frozen record.** `results/kspc_design/FROZEN_v1.json`, 2026-09-22T22:41:43-05:00, the sha256 of 29 files; `FROZEN_v1_addendum.json` records that the current code,
  which gained the covariate classes afterwards, reproduces the frozen numbers exactly. The authors'
  code: `materials/external_code/KernelSingleProxy/PROVENANCE.md`, commit a5283d62, unchanged.
- **Code.** `code/run_kspc_design.py`, `code/kspc_wrapper.py` (sanity gate on the authors' first
  experiment, passed; a stand-in for the unused `torchvision` import; a bandwidth rule for a binary
  treatment, inactive on continuous data).
- **Results.** `results/kspc_design/replicate_{1..20}.json`, `results/kspc_design/summary_v1.json`,
  `memo/2026-09-22-kspc-benchmark-results-v1.md`.
- **Figure.** `ICLR/paper/figures/probe-kspc.pdf`, source `code/make_kspc_figure.py`.

| MAE | |
|---|---|
| KSPC (SPMMR / SKPV), given the low-dimensional proxy (role-requiring) | 0.007 / 0.023 |
| COCA, given the low-dimensional proxy | 0.017 |
| 2SLS, W2 treats / W1 treats | 0.025 / 0.100 |
| **PROBE** | **0.054** |
| adjust for all of W | 0.104 |
| COCA, given the image | 0.140 |
| KSPC (SKPV / SPMMR), given the image | 0.273 / 0.555 |
| naive difference | 0.617 |
| oracle | 0.004 |

## Not final

These are recorded and kept, and none is cited as evidence: planted RHC and its extension 21 to 50
(PROBE behind adjusting for everything), LaLonde's ten samples, Zika, WSC, the search-budget control e11,
the SCM-1 sample-size curve e12, and the earlier structured-SCM and SCM-6 confirmations.

## Consistency with the paper, as of this index

- `ICLR/paper/figures/probe-budget.pdf`, in the appendix, is drawn from e11, which is not final.
- `ICLR/paper/figures/probe-realdata.pdf` draws E5, E7 and E6 (Twins, ACIC, IHDP), all final. No section
  includes it at present.
- 2026-09-23 evening: Figure 2 is now `ICLR/paper/figures/probe-mechanism-v2.pdf`, whose panels (c) and (d) draw E2 replicate 0 (`code/make_probe_mechanism_figure_v2.py`); the positive-control version is no longer included. `probe-budget.pdf` (e11) was removed from the appendix on acceptance.
