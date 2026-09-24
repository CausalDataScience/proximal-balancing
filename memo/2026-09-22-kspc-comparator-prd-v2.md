# PRD v2: KSPC as a comparator on the final experiments E1 to E7

2026-09-22 CDT. Draft for the author's approval. Supersedes workstream A of
`memo/2026-09-22-kspc-lalonde-wsc-prd-v1.md`, which is wrong in one respect: it ran KSPC "without X"
because the authors' code has no covariate argument. The method itself handles observed covariates, in
Appendix A of the paper. Nothing below runs before section 0 is answered. ICLR deadline 2026-09-26 06:59 CDT.

The final experiments are those of `EXPERIMENT_INDEX.md`: E1 SCM-1 to 3, E2 SCM-4, E3 observed RHC, E4 the
KSPC benchmark, E5 Twins 11 to 20, E6 IHDP 12 to 21, E7 ACIC 2 to 10.

## 0. Decisions for the author

| | decision | recommendation |
|---|---|---|
| D1 | Implement KSPC with covariates from the authors' Appendix A (Theorems A.6 and A.7) in our wrapper, labelled "KSPC, Appendix A, our implementation" | yes |
| D2 | Main figures show one KSPC estimator, SKPV, with the clean and the contaminated block, as COCA is shown; SPMMR goes to an appendix table | yes |
| D3 | E3 has no answer: report KSPC with each of its five blocks as a descriptive range, appendix only | yes |
| D4 | E2 regenerates the SCM-4 images and readings with the frozen pipeline, about 1 to 5 minutes per data set | yes |
| D5 | Every KSPC row is labelled "added after the protocol was frozen"; no sealed file is edited | yes, required |

## 1. Why

- **The straw-man risk.** The only single-proxy comparator in E1, E2 and E5 to E7 is our linear COCA. It is
  accurate on the clean block in the simulations (MAE 0.033 to 0.038) and it breaks down on the real data
  (E5 0.73, E6 5.98, E7 34.6). A reviewer who knows Xu and Gretton will ask for the kernel version, which
  the introduction already cites.
- **E4 is not enough on its own.** It shows KSPC on its home ground, without covariates. The main claim
  concerns data with covariates and several proxy blocks, where KSPC has not been run.
- **The earlier reason for leaving KSPC out was wrong.** In E1 the covariates alone move the estimate by
  about 0.31 (naive −0.75 against X-adjusted −0.44, truth 1), so KSPC without X would fail for a reason
  unrelated to proxies. Appendix A removes that reason.

## 2. The method with covariates, as the paper states it

Appendix A of arXiv 2308.04585 adds the observed confounder $X$ to every condition:

- proxy condition $W \perp\!\!\!\perp (A, Y) \mid U, X$;
- completeness: $\mathbb{E}[l(U) \mid A = a, W = w, X = x] = 0$ for all $w$ implies $l = 0$;
- deterministic outcome $Y = \gamma_0(A, U, X)$;
- target $f(\tilde a) = \mathbb{E}_{X,U}\big[\mathbb{E}[Y \mid A = \tilde a, X, U]\big]$, through a bridge $h_0$ with
  $\mathbb{E}[h_0(a, x, W) \mid A = a, X = x, Y = y] = y$.

The estimators multiply every kernel matrix elementwise by a kernel on $X$. Two-stage regression
(Theorem A.6): $M = K_{\dot A \dot A} \odot (B^\top K_{WW} B) \odot K_{\dot X \dot X}$ and
$h(a, w, x) = \alpha^\top\{k_{\dot A}(a) \odot (B^\top k_W(w)) \odot k_{\dot X}(x)\}$. Maximum moment restriction
(Theorem A.7): $L = K_{AA} \odot K_{WW} \odot K_{XX}$ and $G = K_{AA} \odot K_{YY} \odot K_{XX}$. The effect is
$f(1) - f(0)$ with $f(a)$ averaged over the empirical joint distribution of $(W, X)$.

**Before any code is written**, the wrapper's author reads Theorems A.6 and A.7 in full and records in the
wrapper's docstring every step not spelled out above, in particular the first-stage kernel. The paper has
no experiment with $X$, and the authors' code has no $X$, so our implementation is the only one; it is
labelled as ours everywhere.

**The outcome condition fails in E1 to E3 and E5 to E7.** Their outcomes carry noise; KSPC runs outside
its identifying condition there, as COCA does, and every caption says so.

## 3. Implementation

- `code/kspc_wrapper.py` gains `kspc_effect(..., x=None)`. With `x`, it subclasses the authors'
  `SKPVModel` and `SPMMRModel` in the wrapper and multiplies by $K_{XX}$ as in section 2; the authors'
  files stay unedited. Standardised $X$, Gaussian kernel, the authors' median bandwidth, and the binary
  bandwidth rule already in the wrapper for discrete columns.
- Hyperparameters: the authors' configurations, `skpv.json` and `spmmr.json`, unchanged. Nothing is tuned
  against a known effect.
- Data: rebuilt from each frozen generator and seed, never read from or written into a sealed file.
  E1 `family_v2_dgp.generate`; E2 the frozen `run_scm4_images` pipeline; E5 to E7
  `coca_ate_all.views_for`; E3 `rhc_data.load`.
- Kernel cost is cubic in n. Any data set above 3,000 rows is subsampled to 3,000 rows with seed
  `99_400_000 + replicate`, fixed here. E6 (747 rows) uses every row.

| experiment | replicates | blocks given to KSPC | rows |
|---|---|---|---|
| E1 SCM-1, 2, 3 | the 20 confirmation seeds each | W1 clean, W4 contaminated | 3,000 of 24,000 |
| E2 SCM-4 | the 20 confirmation seeds | W1, W4 (reader outputs, as COCA gets) | 3,000 of 24,000 |
| E3 RHC | the one data set | each of the five blocks | 3,000 of 5,735 |
| E4 KSPC benchmark | already done, no rerun | W1, the image | 3,000 |
| E5 Twins | 11 to 20 | W1 clean, W4 contaminated | 3,000 of 11,984 |
| E6 IHDP | 12 to 21 | W1, W4 | all 747 |
| E7 ACIC | 2 to 10 | W1, W4 | 3,000 of 4,802 |

Estimated compute with two workers at `nice 10`: E1 about 25 minutes, E2 about 30 to 100 minutes (the
image regeneration dominates), E3, E5, E6 and E7 about 15 minutes together.

## 4. Gates, before our data is touched

1. **Reduction.** With $X$ constant the kernel on $X$ is 1, so the covariate version must equal the authors'
   code to within $10^{-8}$ on an E4 replicate.
2. **Validity.** On the authors' first experiment with an observed confounder added, fixed here:
   $X \sim N(0,1)$, $U \sim \mathrm{Unif}(-1,1)$, $A = \mathrm{erf}(U) + 0.5X + N(0, 0.1^2)$,
   $W = e^U + N(0, 0.1^2)$, $Y = \sin(\pi U/2) + A^2 - 0.3 + 0.5X$, n = 1,000, seed 42. The structural-function
   error of KSPC with $X$ must be at most half that of KSPC without $X$, for both estimators.

If either gate fails, KSPC with covariates is not run on E1 to E7, and the memo says why. The fallback is
not "KSPC without X".

## 5. Reporting

- `results/kspc_comparator/<experiment>/replicate_<r>.json`, write-once, one per replicate, both estimators
  and both blocks; a summary per experiment; a freeze record with sha256 like E4's.
- Scoring uses the same replicate identifiers and the same known effects as the frozen sets; E3 reports
  estimates only.
- Figures: SKPV rows, clean and contaminated, next to the COCA rows in `probe-estimates.pdf` and
  `probe-realdata.pdf`; captions carry "added after the protocol was frozen" and the outcome caveat.
  SPMMR and E3 go to an appendix table.
- `EXPERIMENT_INDEX.md` gains the KSPC rows under each experiment, with their own file paths.

## 6. What is not claimed

- The KSPC rows are not part of any pre-registration. They are a comparator added after freezing, on data
  regenerated from the frozen seeds.
- A good KSPC result on the clean block does not say KSPC needs no roles: someone chose the block. The
  contaminated row is there to show what the wrong choice costs, exactly as for COCA.

## 7. Acceptance

- Both gates pass, or the memo records the failure and nothing else runs.
- Every replicate in section 3 has a result file or a recorded failure reason.
- The figures and the index are updated, the sealed files are byte-identical to their frozen hashes, and
  all tests pass, including new ones for the covariate version and both gates.

## 8. Status, 2026-09-22 23:58 CDT: gate 2 failed; nothing has run on E1 to E7

Approved by the author. The covariate version is implemented in `code/kspc_wrapper.py` (`SKPVWithX`,
`SPMMRWithX`), subclassing the authors' models; their files are unedited. Gate record:
`results/kspc_comparator/gates_v1.json`, write-once.

| gate | SKPV | SPMMR | verdict |
|---|---|---|---|
| 1, reduction: largest difference from the authors' estimator with X constant | 2.2e-16 | 0.0 | pass |
| 2, validity: error with X over error without X, must be at most 0.5 | 0.434 | **0.511** | **fail** |

As section 4 requires, KSPC with covariates has not been run on E1 to E7.

A diagnostic run after the failure, which changes neither the gate nor its verdict
(`results/kspc_comparator/gate2_seed_diagnostic_v1.json`), repeats gate 2 at seeds 43 to 47:

| seed | 43 | 44 | 45 | 46 | 47 |
|---|---|---|---|---|---|
| SKPV ratio | 0.448 | 0.413 | 0.419 | 0.454 | 0.466 |
| SPMMR ratio | 0.493 | 0.497 | 0.486 | 0.699 | 0.498 |

SKPV halves the error at all six seeds. SPMMR sits at the threshold and misses badly at seed 46. The
implementation reduces exactly to the authors' estimator, so the failure is SPMMR's, not the wrapper's.

**Decision needed from the author.** (i) Keep the rule: no KSPC rows on E1 to E7, KSPC stays in E4 only.
(ii) Amend, after seeing gate 2 and labelled so: run SKPV only, the estimator that passed at every seed and
the one D2 already puts in the main figures; SPMMR is reported as having failed its gate. Relaxing the 0.5
threshold is not offered.

## 9. Amendment, decided by the author after seeing gate 2 (2026-09-23 00:05 CDT)

Option (ii). KSPC with covariates runs on E1 to E7 with **SKPV only**, the estimator that passed gate 2 at
all six seeds. SPMMR is recorded as having failed its gate and is not run on E1 to E7. The 0.5 threshold is
unchanged. Every KSPC row and every result file carries the label "SKPV only, decided after gate 2".
Everything else in sections 3 to 7 holds as written.

## 10. Note, 2026-09-23

The KSPC benchmark, called E4 above, is E8 in `EXPERIMENT_INDEX.md` from 2026-09-23; the number E4 is retired.
Nothing else in this document changes.
