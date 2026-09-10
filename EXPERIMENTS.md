# PROBE experiments: handoff plan

Written: 2026-09-09. Owner of the theory and the manuscript: Yonghan Jung. This document is written so that a
new collaborator can run, extend, and finish the experimental program without any other context. Read it top
to bottom once; afterwards Sections 4 and 7 are the working references.

## 1. What is being tested

PROBE (PROxy Blockwise Exclusion) estimates the average treatment effect $\tau = \mathbb E\{Y(1) - Y(0)\}$
from observational data $(X, W, A, Y)$ when an unobserved confounder $U$ affects both the binary treatment
$A$ and the outcome $Y$, and the observed high-dimensional proxy $W = (W_1, \ldots, W_J)$ carries information
about $U$ in blocks (groups of coordinates). $X$ are ordinary observed covariates.

The method holds one block set $S$ out, learns a representation $Z = \phi(X, W_{-S})$ from the covariates and
the remaining blocks, and uses the held-out block $W_S$ as an audit: the representation is acceptable when
$(X, W_S)$ carries no information about $A$ beyond $Z$. The audit statistic is the residual discrepancy

$$
D^2_{\mathrm{res}}(\phi) = \mathbb E\bigl[\{P(A=1 \mid X, W_S, Z) - P(A=1 \mid Z)\}^2\bigr],
$$

estimated as the difference of two Brier risks (one critic predicts $A$ from $Z$, one from $(X, W_S, Z)$).
Adjusting for a $Z$ with zero discrepancy identifies $\tau$ (manuscript Theorem 1) under the assumptions of
manuscript Section 3, of which the important one is outcome-relevant completeness of the held-out block
(Assumption 3, sufficient conditions in Proposition 1). Algorithm 1 (manuscript Section 4.3) does not know
which block sets are valid: it samples $m$ block sets, screens each by discrepancy and overlap, estimates
$\tau$ by honest AIPW for each survivor, links survivors whose estimates are within $2\rho$ of each other, and
returns the median of the largest linked group. Theorem 6 bounds its error.

The experiments must show four things: (i) naive and covariate-only adjustment fail; (ii) adjusting for the
raw proxy also fails when the proxy is many weak coordinates; (iii) PROBE recovers $\tau$ at the accuracy of
an oracle that observes $U$; (iv) the audit rejects invalid representations and block sets.

Manuscript sources: `manuscript/main/3.tex` (identification), `manuscript/main/4.tex` (learning,
estimation, Algorithm 1), `manuscript/appendix/proof.tex` (proofs). The manuscript in this package is the
last accepted state; edits under review are kept in the authoring workspace and are not needed to run
experiments.

## 2. Files

| Path | Role |
|---|---|
| `code/probe_scms.py` | All synthetic data-generating processes (SCMs) `s1`..`s12`, defaults, block layout, validity of held-out sets, an ORC check (`check_orc`). |
| `code/probe_e1_pipeline.py` | The full PROBE pipeline (encoder, critics, discrepancy, screening, honest AIPW, linking, median) plus baselines, the DGP audit table (`--audit-dgps`), and the replicate driver. |
| `code/probe_e4_mnist.py` | Stage 1 of the MNIST image-proxy experiment (E4); reuses the pipeline components. |
| `results/probe_e1_<scm>.json` | E1 grid output per SCM; written after every replicate; `complete` is false until the grid finishes. |
| `materials/benchmarks/mnist/mnist.npz` | MNIST (Keras packaging), provenance in `materials/benchmarks/manifest.json`. |
| `materials/real_world_data/` | Twins (CEVAE files), RHC, LaLonde, WSC, with `manifest.json`. |
| `memo/2026-09-09-e4-mnist-proxy-plan.md` | Fixed design of E4. |

Environment: Python 3.14, `numpy`, `scipy`, `scikit-learn`, `torch` (CPU is enough), versions in
`requirements.txt`. Nothing uses a GPU.

## 3. How to run

Sanity table of an SCM (propensity range, naive contrast, AIPW estimates of the fixed adjustment sets, audit
power of one held-out block for $Z = X$ and for the oracle $Z = (X, U)$, ORC check):

```bash
cd code && python3 probe_e1_pipeline.py --scm s10 --audit-dgps --n-audit 12000
```

One replicate of Algorithm 1 on one SCM (prints retained splits, the Algorithm 1 output, the linking radius,
component sizes, and the baselines):

```bash
cd code && python3 probe_e1_pipeline.py --scm s10 --n 3000 --reps 1 --m 10
```

The E1 grid for one SCM (what is running now; about 1.6 hours per SCM on a laptop core pair):

```bash
cd code && OMP_NUM_THREADS=2 python3 probe_e1_pipeline.py --scm s10 --n 1500 3000 6000 12000 --reps 20 --m 10 --threads 2
```

Any SCM parameter can be overridden with `--scm-param key=value` (repeatable); dimensions with `--d-x` and
`--d-w`; encoder with `--d-z`, `--enc-steps`, `--enc-hidden`, `--lr`, `--lam`, `--keep-x`,
`--pretrain-steps`; screening with `--t` (default two standard errors), `--eta`, `--overlap-min`; linking with
`--rho`, `--rho-rule`, `--delta`; the split family with `--max-held`; the nuisance learners with `--nuisance {gbm,mlp}`
(use `mlp` for every paper result, see 7.6); three-way fold rotation with `--crossfit`; alternative representations
with `--enc-arch {mlp,set,index,index2}` and warm starts with `--pretrain-mode {none,treat,mask}` (ablations only,
all inferior to the default audit-gap MLP).

E4 stage 1:

```bash
cd code && OMP_NUM_THREADS=2 python3 probe_e4_mnist.py --sigma 1.0 --occ 10 --n 6000 --reps 5 --m 6
```

## 4. The pipeline, step by step

For one replicate with $n$ units and one sampled block set $S$:

1. Three-way split of the units: discrepancy sample $\mathcal I_D$, nuisance sample $\mathcal I_N$, evaluation
   sample $\mathcal I_E$, one third each.
2. Inputs $V = (X, W_{-S})$ and $T = (X, W_S)$, standardized with the means and standard deviations of
   $\mathcal I_D$.
3. Encoder $\phi$: an MLP (`Encoder`, hidden width 32) from $V$ to $Z \in \mathbb R^{d_z}$, $d_z = 3$ by
   default. With `--keep-x`, $Z = (X, \psi(V))$ instead.
4. Critics (`NestedCritics`): closed-form ridge regressions of $A - 1/2$ on random Fourier features. $h_0$ uses
   128 features of $Z$; $h_1$ uses the same 128 features plus 128 features of $(T, Z)$, so the $h_1$ class
   contains the $h_0$ class and the in-sample Brier gap $R_0 - R_1$ is nonnegative. A control critic $h_{1c}$
   has the geometry of $h_1$ but sees $T$ with the held-out coordinates replaced by independent Gaussian
   noise; its gap is the optimism floor that comes from feature count alone.
5. Training (`train_encoder`): Adam on the in-sample nested gap plus a variance floor on $Z$, computed on 75%
   of $\mathcal I_D$, with early stopping on the two-fold cross-fitted gap of the other 25% (checked every 10
   steps, 300 steps maximum). Critic weights are detached at every step. `--pretrain-steps k` first trains
   the encoder for $k$ steps to predict $A$ (Brier loss, linear head, head discarded); see Section 7.1 before
   using it.
6. Discrepancy (`empirical_discrepancy`): two-fold cross-fitted per-unit gaps on $\mathcal I_D$; the reported
   statistic is $\widehat D^2 = \max\{\text{mean}(\text{gap} - \text{control gap}), 0\}$ with its standard
   error.
7. Screen: the split survives when $\widehat D^2 \le t$ (default $t = 2\,\widehat{\mathrm{se}}$) and at least
   98% of the evaluation units have an estimated propensity inside $[\eta, 1 - \eta]$, $\eta = 0.05$.
8. Nuisances (`fit_nuisances`): gradient boosting for $\mathbb E[Y \mid Z, A = a]$ and $P(A = 1 \mid Z)$ on
   $\mathcal I_N$; honest AIPW (`aipw`) on $\mathcal I_E$ with propensities clipped to $[\eta, 1 - \eta]$;
   returns $\theta_S$, its influence-function standard deviation, and the overlap share.
9. Algorithm 1 over the $m$ sampled block sets (`run_replicate`): linking radius
   $\rho = z_{1 - \delta/(2m)}\,\hat\sigma / \sqrt{n_E}$ (or Chebyshev with `--rho-rule chebyshev`),
   $\delta = 0.1$; union-find links survivors with $|\theta_S - \theta_{S'}| \le 2\rho$; the output is the
   median of the largest component (ties are reported as `candidates`). `mean_over_retained` and
   `median_over_retained` are stored for reference only.
10. Baselines (`baseline_aipw`), all honest AIPW with the same learners, clipping, and evaluation sample, with
    nuisances fitted on $\mathcal I_D \cup \mathcal I_N$: `naive` (difference in means), `baseline_X`
    ($X$ only), `baseline_XW` ($X$ and all of $W$), `baseline_X_Wmean` ($X$ and the mean of $W$),
    `oracle_XU` ($X$ and $U$).

Split family: all block sets of size 1 to $\lfloor J/2 \rfloor$ (`all_splits`); $m$ of them are sampled per
replicate. Sets holding out more than half of the blocks leave too little information for the
representation (see Section 7.3).

Result JSON: `config`, resolved `scm_params`, `summary[n]` (mean absolute error of every estimator, the
single-output rate and mean number of retained splits of Algorithm 1), and `runs`, one record per replicate
with every split's `S`, `D2_hat`, `gap_cf`, `null_floor`, `gap_se`, `threshold`, `screen_pass`, `theta`,
`sigma`, `overlap_frac`, `best_step`, and `valid_heldout` (whether the set satisfies Assumption 2 and the
sufficient conditions of Proposition 1).

## 5. The data-generating processes

Common skeleton (`probe_scms.py`; $\tau = 1$ in every SCM):

$$
X = b_x U + \epsilon_X \in \mathbb R^{20},\qquad
P(A = 1 \mid U, X) = c + (1 - 2c)\,\mathrm{expit}\{u_{\mathrm{tr}} U + 0.6\, a_x^\top X\},\qquad
Y = A + 0.7\, y_x^\top X + 0.3\,(y_t^\top X)^2 + u_{\mathrm{out}} U + \epsilon_Y .
$$

The main five SCMs use the Simpson configuration $u_{\mathrm{tr}} = 1.2$, $u_{\mathrm{out}} = -3$, $c = 0.05$
(s9: $1.5$, $-4$): the treated have larger $U$ and larger $U$ lowers $Y$, so the naive and the $X$-adjusted
contrasts have the wrong sign while the oracle recovers 1.

| SCM | Proxy channel $W_k = g_k(U) + \text{noise}$ | Blocks | ORC status (Proposition 1) |
|---|---|---|---|
| s10 | linear, 200 coordinates, loadings $0.35 \times$ Unif(0.35, 1) with random signs, $N(0,1)$ noise | 10 blocks of 20 | (i) and (b) hold for every block |
| s12 | four modalities of 40 coordinates: linear; Poisson with rate $\exp(0.3 + m_\ell U)$; Bernoulli with logit $m_\ell U$; $\tanh(m_\ell U) + 0.5\,N(0,1)$ | 4 blocks | linear (b), Poisson (c), tanh (b); the binary block alone is not complete for a continuous $U$, so `heldout_set_valid` marks $S = \{2\}$ invalid |
| s3 | finite latent $U \in \{0..7\}$ with values linspace(-1.5, 1.5); each block a noisy 8-bit one-hot code, bits flipped with probability 0.25 | 10 blocks of 8 | finite-state completeness: channel matrix of one code block has rank 8 (a single bit has rank 2, so whole code blocks are the unit) |
| s9 | block 0 is an instrument ($N(0,1)$, enters the treatment index with coefficient 2, no $U$ information); blocks 1..4 linear proxies with loadings $1.5 \times$ Unif(0.35, 1) | 5 blocks of 2 | blocks 1..4 satisfy (i) and (b); any set containing block 0 violates Assumption 2 |
| s11 | decoder $W_j = A_{2,j}^\top \tanh(a_1 U + b_1) + B_j N_j + 1.5\,\epsilon_j$, block-specific nuisance factors $N_j \sim N(0, I_{20})$, nuisance scale 2 | 10 blocks of 30 | (i) and (b) hold; injectivity of $u \mapsto A_{2,j}^\top\tanh(a_1 u + b_1)$ checked numerically by `check_orc` (minimum secant ratio 0.26 to 0.42) |

Sanity table at $n = 12000$ (estimates of $\tau = 1$; $\widehat D^2$ is the audit statistic of block 1 with
$Z = X$, standard error in parentheses):

| SCM | naive | X-only | (X, W) | oracle (X, U) | $\widehat D^2$ (Z = X) |
|---|---|---|---|---|---|
| s10 | -1.53 | -0.50 | +0.22 | +0.92 | 0.0067 (0.0006) |
| s12 | -1.53 | -0.50 | +0.82 | +0.92 | 0.0168 (0.0009) |
| s3 | -1.65 | -0.37 | -0.05 | +0.99 | 0.0017 (0.0003) |
| s9 | -1.39 | -0.43 | +0.65 (valid blocks only: +0.78) | +0.99 | 0.0029 (0.0003) |
| s11 | -1.53 | -0.50 | +0.25 | +0.92 | 0.0035 (0.0005) |

The oracle is 0.92 rather than 1.00 in the three high-dimensional SCMs because the gradient-boosting
nuisances are imperfect under this strong confounding; PROBE cannot be expected to beat the oracle.

Appendix SCMs: s5 (a second latent $U_2$ that no proxy measures: the audit is blind by design), s4 (scalar
injective blocks, for the block-count sweep), s8 (limited overlap), s1 and s2 (linear and nonlinear controls),
s6 and s7 (a block that affects $A$ or $Y$ directly).

## 6. Experiment list and status

| Id | Content | Status on 2026-09-09 |
|---|---|---|
| E1 | Main grid: five SCMs, $n \in \{1500, 3000, 6000, 12000\}$, 20 replicates, $m = 10$; figure: absolute error of naive, X-only, (X, W), PROBE, oracle against $n$ | Original recipe complete for s10, s12, s9, s11 (`results/probe_e1_<scm>.json`); s3 stopped (7.1). To be re-run with the re-tuned strengths, `--nuisance mlp`, and `--crossfit` (7.6, 8). |
| E2 | Block-count sweep on s4: $J \in \{2, 4, 8, 16\}$ at $n = 6000$, 20 replicates; shows that exact balance is approached as $J$ grows | Not started; runs with existing code (`--scm s4 --n-blocks J`). |
| E3 | Comparators on the tabular SCMs: classify-then-Kuroki-Pearl for s3, Single Proxy Control (Park, Richardson, Tchetgen Tchetgen), two-proxy proximal doubly robust estimator, all with the same sample splits | Not implemented. |
| E4 | MNIST image proxies (design fixed in the memo): degradation sweep and $n$ sweep | Stage 1 script written; pilot stopped because the encoder failed, see 7.2. |
| E5 | Twins semi-synthetic benchmark (CEVAE protocol: latent = same-sex/birth-weight class, proxies = noisy copies of the latent) | Data present, protocol not implemented. |
| Appendix | s5 audit blindness, s8 overlap, s1/s2 controls, s6/s7 invalid blocks | Not started; runs with existing code. |

E1 grid results with the original recipe (tree nuisances, original proxy strengths; mean absolute error of
Algorithm 1 / raw $(X,W)$ / oracle, 20 replicates; s3 was stopped, see 7.1):

| SCM | $n=1500$ | $n=3000$ | $n=6000$ | $n=12000$ |
|---|---|---|---|---|
| s10 | 0.57 / 0.66 / 0.11 | 0.52 / 0.71 / 0.09 | 0.42 / 0.78 / 0.07 | 0.34 / 0.78 / 0.05 |
| s12 | 0.19 / 0.20 / 0.11 | 0.19 / 0.16 / 0.09 | 0.13 / 0.17 / 0.07 | 0.11 / 0.17 / 0.05 |
| s9 | 0.44 / 0.46 / 0.09 | 0.31 / 0.39 / 0.09 | 0.16 / 0.38 / 0.05 | 0.10 / 0.39 / 0.03 |
| s11 | 0.98 / 0.68 / 0.11 | 0.82 / 0.70 / 0.09 | 0.68 / 0.72 / 0.07 | 0.53 / 0.73 / 0.05 |

These files are the "original recipe" record. Section 7.6 explains why they are not the paper's evidence: the
$(X,W)$ column uses a tree learner that cannot add up many weak coordinates, and the proxy strengths put the
information ceiling far below the oracle.

## 7. Known problems, what was tried, and what to do next

### 7.1 s3: the encoder does not recover a categorical latent from many weak binary codes

Symptom: at $n = 1500$ all splits pass the screen ($\widehat D^2$ median 0) while the estimates are far from
1 (median -0.65). The audit is blind because one code block carries little information about $A$ given a
partial $Z$, and the encoder learns only a linear approximation of $U$. The Bayes rule for $U$ from the
remaining nine blocks is the position with the most ones (an argmax of eight counts); a linear function of the
counts explains about 81% of the variance of $\mu(U)$, and PROBE removes about 70% of the confounding bias,
which matches a linear-level representation.

Per-split estimates $\theta_S$ (three fixed sets, $n = 3000$; oracle about 0.98):

| Change tried | PROBE $\theta_S$ (mean, range) | (X, W) | Note |
|---|---|---|---|
| none (K = 8, flip 0.25) | -0.04 (-0.20 to +0.13) | -0.29 | current default |
| flip 0.15 | +0.51 (+0.29 to +0.75) | +0.25 | audit signal doubles |
| flip 0.10 | +0.67 (+0.45 to +0.94) | +0.51 | |
| K = 6, flip 0.15 | +0.80 (+0.47 to +1.11) | +0.51 | |
| K = 4, flip 0.15 | +0.79 (+0.63 to +0.86) | +0.64 | |
| 15 blocks, flip 0.15 | +0.52 (+0.23 to +0.82) | +0.31 | more blocks do not help |
| `--keep-x` | worse in every setting | | |
| $n = 6000$, flip 0.15, default encoder | 0.53, 0.29, 0.52 | +0.30 | larger $n$ alone does not help |
| $n = 6000$, hidden 64, 1000 steps, lr 0.001 | 0.49, 0.60, 0.19 | | larger encoder does not help |
| $n = 12000$, default encoder | 0.64, 0.55, 0.48 | +0.41 | oracle 0.99; the plateau persists at the largest $n$ of the grid |
| $n = 12000$, hidden 64, 1000 steps (lr 0.001 or 0.003) | 0.59, 0.63, 0.45 and 0.62, 0.50, 0.44 | | same |
| `--pretrain-steps 300` (treatment-predictive warm start) | -1.04, -0.66, -0.96 | | harmful: the warm start overfits $A$ on 1000 units |
| $d_z = 8$ | 0.66, 0.27, 0.55 | | $R^2$ of $\mu(U)$ given $Z$ on fresh units rises from 0.75 to 0.91 (block 1 held out) but the estimates do not follow |
| $d_z = 16$ | 0.63, 0.43, 0.55 | | $R^2$ 0.88 to 0.92; same conclusion |
| set encoder (`--enc-arch set`), $d_z = 3$ | 0.20, 0.35, 0.12 | | worse: the architecture can represent the counts, but the balance objective does not train it to |
| set encoder, $d_z = 8$ | 0.28, 0.31, 0.26 | | same |
| set encoder at $n = 6000$, $d_z = 3$ or 8 | 0.41, 0.47, 0.30 and 0.52, 0.43, 0.28 | +0.30 | oracle 0.89; same at flip 0.25 (about 0) |

Why every change so far failed: the encoder's only training signal is the audit gap, and the audit block
is one weak code (8 bits with flip probability 0.15 to 0.25). Once $Z$ carries the linear part of $U$, the
information left in a single held-out block about $A$ given $Z$ is small, the ridge critics barely detect it,
and the gradient toward the missing nonlinear part (the argmax over eight position counts) vanishes. Enlarging
the representation or giving the encoder the right architecture does not create a signal that is not there.

Next steps, in order: (a) learn the representation body from the proxy structure itself, without the audit
block: an unsupervised warm start such as cross-block prediction (predict one remaining block from the others;
because blocks are conditionally independent given $U$, the shared information is exactly $U$) or an
autoencoder of $W_{-S}$, followed by the balance objective for fine-tuning and by the audit for screening.
This is consistent with the manuscript: $\phi$ may be any learner, the theory concerns the audited $Z$. Do not
use a treatment-predictive warm start (row above) or any use of $Y$. (b) If (a) works on s3, apply the same
recipe to E4 (Section 7.2) with a shared per-image map. (c) Report the $U$ content of $Z$ ($R^2$ of $\mu(U)$
given $Z$ on fresh units, as in the table) next to the estimates in every s3 experiment; it separates
representation failures from estimation failures. Acceptance: at $n = 12000$ the valid-split $\theta_S$ within 0.1 of the oracle, and at smaller
$n$ the screen rejecting splits whose $\theta_S$ is off by more than $2\rho$.

### 7.2 E4: the same failure with MNIST proxies through PCA features

Pilot replicate ($n = 6000$, $\sigma = 1$, occlusion 10 px, 50 principal components per image): Algorithm 1
output -0.48, X-only -0.42, raw feature adjustment -0.17, oracle +0.91. The digit identity has to be decoded
nonlinearly from 150 PCA coordinates of three images; the balance objective alone does not train the MLP to
do it. Fixes (a) to (c) of 7.1 apply, with the set encoder being the most natural (a shared per-image map with
pooling, later a shared CNN). Do not use a supervised digit classifier as the feature map: its output is
essentially $\hat U$ and would leak the latent into every method.

### 7.3 Held-out sets that leave too little behind

In s12 the set that holds out three of the four blocks gave $\theta_S$ of 0.51 and 0.66 at $n = 12000$
while the screen caught it only once in two replicates. The split family is therefore limited to sets of at
most half of the blocks (`--max-held`). Keep this restriction unless there is a reason to study the full
family.

### 7.4 Screen floor calibration

With the raw oracle $Z = (X, U)$ the floor-corrected $\widehat D^2$ is 0.0016 (s10), 0.0041 (s12), 0.0008
(s11) at $n = 12000$ instead of 0, because $h_1$'s extra features of $(T, Z)$ also approximate
$P(A \mid Z)$ better than $h_0$'s features of $Z$ alone. A synthetic null ($A^\ast \sim
\mathrm{Bernoulli}(\hat h_0(Z))$) gives a floor of 0 and a conditional-permutation control gives the same
floor as the Gaussian-noise control, so neither fixes it. Learned representations pass the screen
($\widehat D^2 \le 0.001$) because the encoder chooses coordinates that $h_0$ fits well. Consequence: the
oracle is a yardstick for the estimate, not for the screen; report screen pass rates for learned $Z$ only.
Open item: a floor that reproduces the approximation gain.

### 7.5 Small-sample linking radius

At $n = 1500$ the evaluation sample has 500 units and $\rho \approx 0.7$, so every survivor links into one
component and the median cannot separate good from bad splits. This is expected; the $n$ curve is the result.

## 7.6 First-principles audit of the design (2026-09-09, late evening)

Two findings change how Sections 5 to 8 must be read.

1. **The raw proxy adjustment fails only with the tree learner.** With an L2 logistic propensity and MLP outcome
   models (`--nuisance mlp`), adjusting for $(X, W)$ reaches 0.78 in s10 and s11, 0.74 in s9, 0.94 in s12
   ($n = 12000$), against 0.14 to 0.61 with gradient boosting. The oracle itself moves from 0.91 to 0.99.
   Every claim that "(X, W) fails" in the tabular SCMs was an artifact of the learner.
2. **The oracle is out of reach by information, not by learning.** Adjusting for $(X, E[U \mid X, W_{-1}])$,
   with the posterior mean approximated on 100k units, gives 0.77 (s10), 0.74 (s11), 0.75 (s9), 0.91 (s12),
   0.78 (s3, a lower bound) at the current proxy strengths, because $R^2(U \mid X, W_{-1})$ is 0.87 to 0.96 and
   the Simpson strength $u_{\mathrm{out}} = -3$ turns a residual variance of $1 - R^2$ into a bias of about
   $3 \times 1.2 \times 0.8 \times (1 - R^2)$. Matching the oracle within 0.1 needs $R^2 \gtrsim 0.97$.

Re-tuned candidates (ceiling scan, $n = 12000$, MLP nuisances): s10 loading 0.6 ($R^2$ 0.965, ceiling 0.91,
$(X,W)$ 0.92); s11 $\sigma = 0.7$, nuisance scale 1.0 (0.973, 0.94, 0.89); s9 loading 2.5 (0.944, 0.89, 0.89);
s12 loading 0.7 (0.983, 0.96, 0.94); s3 flip 0.15 (0.998, 0.997, $(X,W)$ 0.74).

PROBE on the candidates (audit-gap MLP encoder, MLP nuisances, $n = 12000$, one replicate, three splits):
s10 0.85 to 0.89; s11 0.83; s12 0.87 to 0.92; s9 0.83 to 1.03 (with instrument coefficient 4: 0.86 to 0.99,
while $(X,W)$ with MLP nuisances is 1.00); s3 0.45 to 0.65. PROBE therefore sits at or slightly below the
strong $(X, W)$ adjustment and about 0.1 below the oracle in the valid SCMs.

Representation ablation (same nuisances): a ridge-logistic propensity index of $(X, W_{-S})$ gives 0.05 to 0.22
and fails the overlap gate in s9 (it keeps the instrument); propensity plus outcome indices give 0.47 to 0.75;
the audit-gap encoder gives 0.62 to 0.98 with intact overlap. Treatment-predictive representations amplify the
residual confounding; the balance objective is instrument-averse. Keep the audit-gap encoder.

Consequences. In valid tabular SCMs PROBE cannot beat a strong $(X, W)$ adjustment; the honest claim is
equivalence with the proxy ceiling plus the audit certificate. "Beats $(X, W)$" is attainable only where the
raw adjustment fails for structural reasons: nonlinear code structure (s3, ceiling 0.997 versus $(X,W)$ 0.74,
provided the encoder decodes it), and image proxies (E4). The current E1 grid (tree nuisances, old strengths)
is kept only as the "old recipe" record.

## 8. Success criterion (set by Yonghan, 2026-09-09; status after the audit above)

**First cell that passes the (X, W) condition with strong baselines: s9 at $n = 12000$.** With the saved
grid outputs (tree nuisances inside PROBE) scored against logistic+MLP baselines on the same 20 replicates:
MAE PROBE 0.100, oracle 0.030, $(X, W)$ 0.304; paired PROBE minus $(X,W)$ = $-0.20$ [$-0.25$, $-0.16$];
PROBE minus oracle = $+0.07$ [$+0.04$, $+0.10$]. Re-running PROBE with `--nuisance mlp`
(`results/probe_e1_s9_n12000_mlp.json`): MAE PROBE 0.119, oracle 0.021, $(X,W)$ 0.287; PROBE minus
$(X,W)$ = $-0.17$ [$-0.21$, $-0.12$]; PROBE minus oracle = $+0.10$ [$+0.04$, $+0.16$]. The instrument story
holds against strong baselines; the remaining 0.1 to the oracle is in the representation stage (3-dimensional
$Z$ learned on one third of the sample), not in the nuisance stage.

**s12 at $n = 12000$ (valid SCM) against strong baselines:** MAE PROBE 0.106, oracle 0.026, $(X, W)$ 0.068;
PROBE minus $(X,W)$ = $+0.04$ [$+0.02$, $+0.06$] (PROBE loses), PROBE minus oracle = $+0.08$ [$+0.06$, $+0.10$].
This is the expected picture in a valid SCM with strong proxies: the raw adjustment with a good learner is the
ceiling and PROBE sits slightly below it. Against the tree baselines of the saved grid, s12 passed both
conditions from $n = 6000$ on, which shows how much the criterion depends on the baseline learner.

For every main SCM at the largest sample size, over the 20 replicates, with paired differences of absolute
error (PROBE minus comparator) and t-based 95% intervals (`paired_criterion` in the pipeline, stored under
`summary[n]["criterion"]`):

1. PROBE beats the raw proxy adjustment: the interval of $|\hat\tau_{\mathrm{PROBE}} - \tau| - |\hat\tau_{(X,W)} - \tau|$ lies
   below 0. A tie with $(X, W)$ is a failure.
2. PROBE matches the oracle: the upper limit of $|\hat\tau_{\mathrm{PROBE}} - \tau| - |\hat\tau_{(X,U)} - \tau|$ is at most
   0.10 (equivalence margin in units of $\tau = 1$).

Status of the saved cells on 2026-09-09: no cell meets both conditions. s12 ties with $(X, W)$; s9 and s10
beat $(X, W)$ but are far from the oracle; s11 and s3 lose to $(X, W)$. The encoder fix of Section 7.1 is
therefore the critical path; the unsupervised masked-block warm start is implemented as
`--pretrain-mode mask --pretrain-steps 500` and under test.

## 9. Reporting conventions

- The estimand is $\tau = 1$; report absolute error in units of $\tau$ and, for Simpson settings, also the
  outcome standard deviation (about 3.2 in s10) so that the error can be read relative to the noise.
- Every method uses the same replicate, the same sample split, the same evaluation sample, and the same
  gradient-boosting learners; only the adjustment set differs.
- Seeds: `ROOT_SEED` and the namespaced `seed_key` in `probe_e1_pipeline.py`; structural coefficients are
  drawn once per SCM from `seed_key("coef")`, data from `seed_key("data", rep)`.
- Keep `complete: false` results out of figures.

## 10. Questions to settle with Yonghan before changing the design

1. Whether s3 stays a main SCM after the encoder fix or moves to the appendix as the finite-state case.
2. The comparator set of E3 and which implementation of Single Proxy Control to use.
3. Whether E4 stage 2 (end-to-end CNN) is needed for the paper or stage 1 with a set encoder suffices.
