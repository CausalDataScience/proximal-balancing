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
`--rho`, `--rho-rule`, `--delta`; the split family with `--max-held`.

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
| E1 | Main grid: five SCMs, $n \in \{1500, 3000, 6000, 12000\}$, 20 replicates, $m = 10$; figure: absolute error of naive, X-only, (X, W), PROBE, oracle against $n$ | Running for s10, s12, s9, s11 (partial results in `results/`). s3 stopped, see 7.1. |
| E2 | Block-count sweep on s4: $J \in \{2, 4, 8, 16\}$ at $n = 6000$, 20 replicates; shows that exact balance is approached as $J$ grows | Not started; runs with existing code (`--scm s4 --n-blocks J`). |
| E3 | Comparators on the tabular SCMs: classify-then-Kuroki-Pearl for s3, Single Proxy Control (Park, Richardson, Tchetgen Tchetgen), two-proxy proximal doubly robust estimator, all with the same sample splits | Not implemented. |
| E4 | MNIST image proxies (design fixed in the memo): degradation sweep and $n$ sweep | Stage 1 script written; pilot stopped because the encoder failed, see 7.2. |
| E5 | Twins semi-synthetic benchmark (CEVAE protocol: latent = same-sex/birth-weight class, proxies = noisy copies of the latent) | Data present, protocol not implemented. |
| Appendix | s5 audit blindness, s8 overlap, s1/s2 controls, s6/s7 invalid blocks | Not started; runs with existing code. |

Early E1 results (mean absolute error at $n = 1500$, 20 replicates): Algorithm 1 / X-only / (X, W) / oracle:
s9 0.44 / 1.29 / 0.46 / 0.09; s12 0.19 / 1.41 / 0.20 / 0.11; s10 0.57 / 1.41 / 0.66 / 0.11;
s11 0.98 / 1.41 / 0.68 / 0.11; s3 1.63 / 1.52 / 1.21 / 0.11 (stopped).

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
| `--pretrain-steps 300` (treatment-predictive warm start) | -1.04, -0.66, -0.96 | | harmful: the warm start overfits $A$ on 1000 units |

Next steps, in order: (a) representation dimension $d_z \in \{8, 16\}$ so that the count vector fits
linearly and the nuisance learners do the argmax (test running when this document was written; result goes
into the log); (b) a set encoder with a shared per-block map and sum pooling (`DeepSets`), which yields the
count vector by construction and is the natural architecture when blocks are exchangeable given $U$; (c) an
unsupervised warm start of the encoder body (autoencoder or cross-block prediction, which extracts exactly the
shared latent because blocks are conditionally independent given $U$), instead of the treatment-predictive
warm start. Acceptance: at $n = 12000$ the valid-split $\theta_S$ within 0.1 of the oracle, and at smaller
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

## 8. Reporting conventions

- The estimand is $\tau = 1$; report absolute error in units of $\tau$ and, for Simpson settings, also the
  outcome standard deviation (about 3.2 in s10) so that the error can be read relative to the noise.
- Every method uses the same replicate, the same sample split, the same evaluation sample, and the same
  gradient-boosting learners; only the adjustment set differs.
- Seeds: `ROOT_SEED` and the namespaced `seed_key` in `probe_e1_pipeline.py`; structural coefficients are
  drawn once per SCM from `seed_key("coef")`, data from `seed_key("data", rep)`.
- Keep `complete: false` results out of figures.

## 9. Questions to settle with Yonghan before changing the design

1. Whether s3 stays a main SCM after the encoder fix or moves to the appendix as the finite-state case.
2. The comparator set of E3 and which implementation of Single Proxy Control to use.
3. Whether E4 stage 2 (end-to-end CNN) is needed for the paper or stage 1 with a set encoder suffices.
