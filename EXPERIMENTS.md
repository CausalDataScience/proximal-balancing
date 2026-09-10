# PROBE experiments: status and plan

Updated 2026-09-10. Owner of the theory and the manuscript: Yonghan Jung. This document is written to be read
without any other context. Sections 1 to 3 are the status and the plan. Sections 4 to 6 are the reference.
The appendices record what was tried and what it showed.

## 1. The question and the bar

PROBE estimates the average treatment effect $\tau = \mathbb E\{Y(1) - Y(0)\}$ when an unobserved confounder
$U$ moves both the binary treatment $A$ and the outcome $Y$, and the only trace of $U$ is a high-dimensional
proxy $W = (W_1, \ldots, W_J)$ observed in blocks. It holds out a block set $S$, learns a representation
$Z = \phi(X, W_{-S})$ from the covariates and the remaining blocks, and uses the held-out block as an audit:
the representation is acceptable when $(X, W_S)$ says nothing more about $A$ once $Z$ is known. Algorithm 1
does this for $m$ sampled block sets, screens each by that audit and by overlap, estimates $\tau$ by honest
AIPW for each survivor, links survivors whose estimates agree, and returns the median of the largest linked
group.

**The bar, set on 2026-09-09 and unchanged.** In every experiment, over 20 replicates, with paired differences
of absolute error and t-based 95% intervals (`paired_criterion` in the pipeline, stored under
`summary[n]["criterion"]`):

1. PROBE beats the raw proxy adjustment. The interval for
   $|\hat\tau_{\mathrm{PROBE}} - \tau| - |\hat\tau_{(X,W)} - \tau|$ lies below zero. A tie is a failure.
2. PROBE matches the oracle that sees $U$. The upper limit of
   $|\hat\tau_{\mathrm{PROBE}} - \tau| - |\hat\tau_{(X,U)} - \tau|$ is at most 0.10 in units of $\tau = 1$.

Every comparator is the same honest AIPW with the same sample splits, the same clipping level and the same
learners; only the adjustment set differs. Since 2026-09-10 all comparator numbers use strong learners, an L2
logistic propensity with MLP outcome models. Tree-based comparators understate the raw adjustment badly and
must not be used for any reported claim (Appendix A.1).

## 2. Where we stand today

### 2.1 Headline performance

Mean absolute error over 20 replicates at $n = 12000$, $\tau = 1$. Comparators use strong learners.

| SCM | PROBE | raw $(X,W)$ | $X$ only | oracle $(X,U)$ | vs oracle | vs $(X,W)$ | verdict |
|---|---|---|---|---|---|---|---|
| s9 with strength 2.0, cross-fitting, strong nuisances | **0.055** | 0.172 | 1.219 | 0.016 | +0.04 [+0.02, +0.06] | $-0.12$ [$-0.15$, $-0.09$] | **passes both** |
| s9 as first run | 0.100 | 0.304 | 1.300 | 0.030 | +0.07 [+0.04, +0.10] | $-0.20$ [$-0.25$, $-0.16$] | beats, does not match |
| s12 mixed modality | 0.106 | 0.068 | 1.450 | 0.026 | +0.08 [+0.06, +0.10] | +0.04 [+0.02, +0.06] | loses |
| s10 many weak proxies | 0.338 | 0.187 | 1.334 | 0.026 | +0.31 [+0.29, +0.34] | +0.15 [+0.13, +0.18] | loses |
| s11 decoder proxy | 0.530 | 0.209 | 1.334 | 0.026 | +0.51 [+0.47, +0.54] | +0.32 [+0.29, +0.35] | loses |
| s3 finite codes, $n = 1500$ | 1.628 | 0.575 | 1.432 | 0.135 | +1.49 | +1.05 | loses |
| E4 MNIST images, $n = 6000$, 5 replicates | 1.120 | 0.717 | 1.272 | 0.041 | +1.08 | +0.40 | loses |

One configuration clears the bar: the instrument-contaminated SCM at the tuned strength, with cross-fitting
and strong nuisances everywhere. Everything else loses to the raw adjustment once that adjustment is fitted
with a learner that can add up weak signals.

### 2.2 Does more data close the gap?

PROBE's mean output and its standard deviation across replicates, by sample size, on the original recipe.

| SCM | $n = 1500$ | $n = 3000$ | $n = 6000$ | $n = 12000$ | raw $(X,W)$ at 6000 and 12000 |
|---|---|---|---|---|---|
| s9 | 0.577 (0.25) | 0.718 (0.25) | 0.902 (0.17) | 0.952 (0.11) | 0.39, 0.30 error |
| s12 | 0.835 (0.15) | 0.821 (0.13) | 0.873 (0.08) | 0.894 (0.05) | flat |
| s10 | 0.429 (0.17) | 0.482 (0.13) | 0.580 (0.12) | 0.662 (0.06) | 0.205, 0.187 error |
| s11 | 0.025 (0.21) | 0.177 (0.15) | 0.320 (0.13) | 0.470 (0.07) | 0.201, 0.209 error |

Three readings.

- **s9 converges.** The error falls 0.44, 0.31, 0.16, 0.10, close to the $n^{-1/2}$ rate, the spread falls with
  it, and the mean reaches 0.95. With the tuned recipe the mean at $n = 12000$ is 1.011 with spread 0.074, so
  what remains is noise, not bias. More data keeps helping.
- **s10, s11, s12 do not converge to the oracle.** The spread collapses (s10 from 0.17 to 0.06) while the mean
  stays far from 1, so what remains is systematic. The mean does drift upward, because the representation is
  learned on a growing sample, but it drifts slowly: s10 moves from 0.43 to 0.66 while $n$ grows eight times.
- **The raw adjustment has already converged** in those SCMs, so waiting does not change the ranking.

The reason s10 cannot converge is measurable. Adjusting for $X$ together with the best possible reconstruction
of $U$ from the same blocks PROBE uses returns 0.69 at that setting. PROBE returns 0.662. PROBE is already
getting everything the proxy contains; the proxy simply does not contain enough, because 200 coordinates with
loading 0.35 leave 9% of the variance of $U$ unexplained and the outcome multiplies that residual by three.
The fix is a proxy with more information, not a bigger sample: at loading 0.6 the same reconstruction returns
0.91 and PROBE returns 0.85 to 0.89 in a single-replicate check.

### 2.3 What is running or pending

- `results/probe_e1_s9_strength2_crossfit_ncurve.json`: the passing recipe at $n \in \{1500, 3000, 6000\}$,
  20 replicates, to complete the convergence curve for the configuration that clears the bar. Running.
- The re-tuned SCM defaults from Appendix A.3 are not yet the module defaults, pending a decision.

## 3. Experiment by experiment

| Id | Content | Status | Next action |
|---|---|---|---|
| E1 | Five synthetic SCMs, $n \in \{1500, 3000, 6000, 12000\}$, 20 replicates, $m = 10$ | Complete on the original recipe for s10, s12, s9, s11; s3 stopped. Only s9 passes. | Re-run all five with the tuned strengths, `--nuisance mlp`, `--crossfit`. About 2 hours per SCM. |
| E2 | Block-count sweep on s4, $J \in \{2, 4, 8, 16\}$ at $n = 6000$ | Not started; runs with existing code | `--scm s4 --n-blocks J`, 20 replicates, about 40 minutes per $J$ |
| E3 | Comparators: classify-then-Kuroki-Pearl for the finite-code SCM, Single Proxy Control, two-proxy proximal | Not implemented | Implement after E1 is settled |
| E4 | MNIST image proxies | Pilot failed twice; the setting was too degraded to be informative | Redesign in 3.1 |
| E5 | Twins semi-synthetic benchmark | Data in place, protocol not implemented | After E4 |

### 3.1 E4 redesign, in the order it should be done

The current setting recovers the digit from three degraded images to $R^2 = 0.72$, and with the outcome
coefficient at $-3$ that residual alone costs about 0.8 of bias. PROBE returns $-0.15$ to $-0.27$ and the raw
four-image adjustment returns $+0.21$ to $+0.33$; both are useless, so the experiment as built cannot answer
anything. Three changes, each with the number it is expected to move.

1. **Stop degrading so hard.** Drop the occlusion and use pixel noise 0.5 or less. Measured effect on how well
   the digit is recovered from three images: $R^2$ goes from 0.72 to 0.91, and the value obtained by adjusting
   on that reconstruction goes from 0.21 to 0.78.
2. **Replace the feature map.** Fifty principal components of the pixels recover the digit only to $R^2 = 0.94$
   even on clean images. A small convolutional network trained on the MNIST test split as a digit classifier,
   used as a frozen feature extractor, should reach 0.99. This is the change that matters most and it costs
   about an hour of preparation.
3. **Lower the confounding coefficient** from $-3$ to about $-1.5$, so that a few percent of unrecovered
   latent variance costs 0.07 rather than 0.15.

Decision rule after the redesign: adjusting on the reconstructed digit must return at least 0.9, and the raw
four-image adjustment must stay below it. If both hold, the experiment can answer whether PROBE reaches the
same place on image proxies. If the raw adjustment also reaches 0.9, the honest report is equivalence.

### 3.2 Open decisions for Yonghan

1. Adopt the re-tuned SCM defaults (Appendix A.3) as the module defaults.
2. Keep one bar for every SCM, or split it: beat the raw adjustment where the proxy is contaminated, match the
   proxy's own limit where it is clean.
3. s3: try a latent-class decoder as the representation, or move it to the appendix as the finite-state case.
4. Approve the E4 redesign in 3.1.

## 4. Why the four failures happen

**The representation learner is the bottleneck.** PROBE learns $Z$ from one third of the sample using the audit
gap, a weak signal. The comparator's nuisance models get two thirds of the sample and a direct target. In a
clean SCM the raw adjustment is therefore the one to beat and it is hard to beat, because
$Z = \phi(X, W_{-S})$ uses a subset of the proxy and can never hold more information about $U$ than $W$ does.

**So PROBE can win only where the raw adjustment is structurally wrong, not merely imprecise.** Four channels
exist and only two are usable.

1. *Contaminated blocks.* A block that drives treatment without carrying $U$ makes the raw adjustment amplify
   whatever confounding is left, no matter which learner is used. This is s9 and it is the demonstrated win.
2. *Collider blocks.* Conditioning on a block that mixes a treatment cause with an outcome cause creates
   confounding: the raw adjustment returns $-0.05$ against $\tau = 1$. PROBE identifies $\tau$ in that world
   under its assumptions, but the audit cannot tell an admissible representation input from an inadmissible
   one, so the search does not find the right configuration (Appendix B.1).
3. *Positivity.* Refuted. Augmented IPW with a good outcome model survives a propensity pushed to 0.001, and a
   forcing variable that depends on $U$ breaks the representation's overlap as well (Appendix A.5).
4. *Statistical efficiency.* Real room exists when the proxy is coded nonlinearly, but our representation
   learner cannot take it (Appendix A.4).

**The audit certifies balance, not admissibility.** This belongs in the manuscript as a limitation.

## 5. How to run

Environment: Python 3.14 with numpy, scipy, scikit-learn and torch, CPU only.

Sanity table for one SCM, including the check of the sufficient conditions of Proposition 1:

```bash
cd code && python3 probe_e1_pipeline.py --scm s10 --audit-dgps --n-audit 12000
```

One replicate of Algorithm 1:

```bash
cd code && python3 probe_e1_pipeline.py --scm s9 --n 3000 --reps 1 --m 10
```

The recipe that should be used for every reported result:

```bash
cd code && OMP_NUM_THREADS=2 python3 probe_e1_pipeline.py --scm s9 --scm-param proxy_strength=2.0 \
  --n 1500 3000 6000 12000 --reps 20 --m 10 --crossfit --nuisance mlp --threads 2
```

MNIST, stage one:

```bash
cd code && python3 probe_e4_mnist.py --sigma 0.5 --occ 0 --n 6000 --reps 5 --m 6 --enc-arch prog --nuisance mlp
```

Options that matter: `--nuisance {gbm,mlp}` for the learners, `--crossfit` for the three-way fold rotation,
`--d-x` and `--d-w` for dimensions, `--scm-param key=value` for any SCM parameter, `--max-held` and
`--max-excluded` for the candidate family, `--t`, `--eta`, `--overlap-min` for the screen, `--rho`,
`--rho-rule`, `--delta` for linking, `--enc-arch {mlp,set,index,index2,prog}` and
`--pretrain-mode {none,treat,mask}` and `--z-noise` for representation ablations.

## 6. Reference

### 6.1 Files

| Path | Role |
|---|---|
| `code/probe_scms.py` | All synthetic data-generating processes, defaults, block layout, validity of a configuration, and the numerical check of Proposition 1 |
| `code/probe_e1_pipeline.py` | The pipeline: representation, critics, discrepancy, screening, honest AIPW, linking, median, baselines, the sanity table and the replicate driver |
| `code/probe_e4_mnist.py` | MNIST image proxies, stage one |
| `results/probe_e1_<scm>.json` | Grid output, written after every replicate; `complete` is false while a grid is running |
| `materials/benchmarks/mnist/` | MNIST with provenance in `materials/benchmarks/manifest.json` |
| `materials/real_world_data/` | Twins, right heart catheterization, LaLonde, WSC with a manifest |
| `memo/2026-09-09-e4-mnist-proxy-plan.md` | The fixed E4 design |

### 6.2 The pipeline, step by step

For one replicate and one configuration:

1. Three-way split into a discrepancy sample, a nuisance sample and an evaluation sample. With `--crossfit`
   the three roles rotate and the three answers are averaged.
2. Inputs $V = (X, W_{-S})$ and $T = (X, W_S)$, standardized on the discrepancy sample.
3. Representation: an MLP from $V$ to $Z \in \mathbb R^{d_z}$, $d_z = 3$ by default, trained on the in-sample
   nested gap with early stopping on a held-out quarter. Alternatives exist for ablations.
4. Critics: closed-form ridge regressions of $A - 1/2$ on random Fourier features. One critic sees $Z$, the
   other sees the same features plus features of $(T, Z)$, so its class contains the first and the gap is
   nonnegative. A control critic has the same geometry but sees the held-out coordinates replaced by noise;
   its gap is the optimism that comes from feature count alone and is subtracted.
5. Screen: the configuration survives when the corrected discrepancy is within two standard errors of zero and
   at least 98% of evaluation units have an estimated propensity inside $[\eta, 1 - \eta]$ with $\eta = 0.05$.
6. Estimation: gradient boosting or the strong learners for the nuisances on the nuisance sample, honest AIPW
   on the evaluation sample.
7. Algorithm 1: linking radius $\rho = z_{1 - \delta/(2m)}\hat\sigma/\sqrt{n_E}$ with $\delta = 0.1$,
   union-find over $|\theta_S - \theta_{S'}| \le 2\rho$, output the median of the largest component.

### 6.3 The data-generating processes

All share the skeleton $X = b_x U + \epsilon_X \in \mathbb R^{20}$,
$P(A = 1 \mid U, X) = c + (1 - 2c)\,\mathrm{expit}\{u_{\mathrm{tr}} U + 0.6 a_x^\top X\}$,
$Y = A + 0.7 y_x^\top X + 0.3 (y_t^\top X)^2 + u_{\mathrm{out}} U + \epsilon_Y$ with $\tau = 1$. The five main
SCMs use $u_{\mathrm{tr}} = 1.2$, $u_{\mathrm{out}} = -3$, $c = 0.05$ (s9 uses 1.5 and $-4$), under which the
naive contrast and the $X$-adjusted contrast have the wrong sign.

| SCM | Proxy channel | Blocks | Proposition 1 |
|---|---|---|---|
| s9 | block 0 is an instrument with no $U$ content; blocks 1 to 4 are linear proxies | 5 of 2 | holds for blocks 1 to 4; block 0 is an invalid held-out block by construction |
| s10 | linear, 200 coordinates with small loadings | 10 of 20 | holds for every block |
| s11 | decoder $A_2^\top \tanh(a_1 U + b_1)$ plus block-specific nuisance factors and noise | 10 of 30 | holds, injectivity checked numerically |
| s12 | four modalities: linear, Poisson counts, binary codes, tanh | 4 of 40 | linear, Poisson and tanh are covered; the binary block alone is not |
| s3 | finite latent in $\{0..7\}$, each block a noisy 8-bit one-hot code | 10 of 8 | finite-state completeness, rank 8 verified |
| s13 | $a_k \sin(f_k U + p_k)$ plus noise, 600 to 1200 coordinates | 6 | holds, injectivity checked numerically |
| s14 | block 0 measures $L_1 + L_2$ with $L_1 \to A$ and $L_2 \to Y$ | 5 of 4 | block 0 is inadmissible in both roles |

Appendix SCMs: s5 has a second latent no proxy measures, s4 sweeps the block count, s8 limits overlap, s1 and
s2 are controls, s6 and s7 place a direct effect in one block.

### 6.4 Reporting conventions

- Report absolute error in units of $\tau = 1$, and for the Simpson settings also the outcome standard
  deviation, about 3.2 in s10, so the error can be read against the noise.
- Seeds: `ROOT_SEED` with the namespaced `seed_key`; structural coefficients once per SCM, data per replicate.
- Never put a result with `complete: false` in a figure.

## Appendix A: diagnostics and dead ends

### A.1 The comparator learner changes the conclusion

At $n = 12000$, adjusting for $(X, W)$ with gradient boosting versus with an L2 logistic propensity and MLP
outcomes: s10 gives $+0.14$ against $+0.80$, s11 $+0.20$ against $+0.80$, s3 $-0.09$ against $+0.58$, s9
$+0.61$ against $+0.74$. The oracle itself moves from 0.91 to 0.99. Trees cannot add up two hundred weak
coordinates. Every claim in this project must therefore be scored against the strong learners.

### A.2 Where PROBE's remaining error lives

At $n = 12000$ on s9 with strong nuisances, PROBE has mean 0.912 and spread 0.139 while the oracle has spread
0.027; roughly half the error is bias and half is noise. Cross-fitting cut the spread from 0.139 to 0.079 and
the error from 0.119 to 0.085. Restricting held-out sets to a single block made it worse, 0.145, because too
few configurations survive for the median to be stable. Neither a larger representation nor longer training
helped.

### A.3 Re-tuned SCM settings

Measured at $n = 12000$: how well $U$ is recovered from the blocks feeding the representation, the value
obtained by adjusting on that reconstruction, and the raw adjustment with strong learners.

| SCM | change | recovery $R^2$ | value on the reconstruction | raw $(X,W)$ |
|---|---|---|---|---|
| s10 | loading 0.35 to 0.6 | 0.91 to 0.965 | 0.765 to 0.906 | 0.783 to 0.915 |
| s11 | noise 1.5 to 0.7, nuisance scale 2 to 1 | 0.908 to 0.973 | 0.740 to 0.935 | 0.742 to 0.887 |
| s9 | strength 1.5 to 2.5 | 0.871 to 0.944 | 0.751 to 0.886 | 0.738 to 0.891 |
| s12 | loading 0.35 to 0.7 | 0.964 to 0.983 | 0.908 to 0.958 | 0.938 to 0.943 |
| s3 | flip 0.25 to 0.15 | 0.929 to 0.998 | 0.781 to 0.997 | 0.408 to 0.742 |

PROBE on the re-tuned settings, one replicate, three configurations: s10 0.85 to 0.89, s11 0.83, s12 0.87 to
0.92, s9 0.83 to 1.03, s3 0.45 to 0.65.

### A.4 s3 and s13: the representation learner cannot decode

s3 asks the representation to find the position of the largest count among eight noisy code bits. Eight
changes were tried: code strength (flip 0.25, 0.15, 0.10), code length (8, 6, 4), block count 15, keeping $X$
in $Z$, sample size to 12000, a wider and longer-trained network, representation dimension 8 and 16, a shared
per-block map with pooling, a treatment-predictive warm start and an unsupervised masked-block warm start. The
best result was 0.64 against an oracle of 0.99. The measured content of the representation explains it: $Z$
carries 75% to 95% of the variance of the latent score, which is what a linear read of the counts gives.

s13 codes the latent through coordinate-specific sine functions. There the raw adjustment fails too, returning
0.63 to 0.85 while adjusting on a good reconstruction returns 0.95. PROBE returns 0.55 to 0.90 with the audit
gap representation and 0.81 to 0.90 with a prognostic representation built from outcome regressions. The room
is real and our learner cannot take it.

### A.5 Refuted mechanisms

*Positivity.* With an eligibility flag in block 0 that forces treatment, the raw adjustment returns 0.42 while
adjusting on the valid blocks returns 0.83 and the oracle 0.92, but every PROBE configuration fails the
overlap gate, and excluding the flag block also removes the latent information it carries, leaving 0.53 to
0.62. Adding exogenous noise to $Z$ to buy overlap makes it worse: at noise levels 0, 0.25, 0.5, 1, 2 the
overlap share rises 0.86, 0.88, 0.90, 0.95, 0.995 while the estimate falls 0.93, 0.81, 0.54, $-0.17$, $-1.04$.

*Stronger instruments.* Raising the instrument coefficient from 2 to 6 improves the raw adjustment from 0.84
to 1.02, because a strong instrument makes treatment more random with respect to $U$ and leaves less
confounding to amplify. The win has an interior maximum and s9's default sits near it.

### A.6 Other findings

The screen's floor correction does not fully remove the optimism of the critics: with the raw oracle as the
representation the corrected discrepancy is 0.0016 in s10, 0.0041 in s12, 0.0008 in s11 instead of zero. A
synthetic null and a conditional-permutation control both failed to fix it. Learned representations pass, so
screen pass rates should be reported for learned representations only.

Configurations that hold out more than half the blocks leave too little for the representation: in s12 the
three-block held-out set returned 0.51 and 0.66 while the screen caught it once in two replicates. The
candidate family is therefore capped at half the blocks by default.

## Appendix B: theory notes

### B.1 A clean separation exists but is not certifiable

Let $U, L_1, L_2, \epsilon_1, \epsilon_Y$ be independent standard normals, $W_1 = U$, $W_2 = U + \epsilon_1$,
$W_3 = L_1 + L_2$, $A \sim \mathrm{Bernoulli}\{\mathrm{expit}(U + L_1)\}$ and
$Y = \tau A + U + L_2 + \epsilon_Y$. With block 2 audited, block 1 feeding the representation, block 3 used in
neither role and $Z = W_1$, all assumptions of Section 3 hold, balance and overlap hold, and Theorem 1
identifies $\tau$; the raw adjustment conditions on a collider and does not. The same observed distribution
also admits the configuration that puts block 3 into the representation, which passes both screens and returns
a different value, so balance and overlap cannot separate them. The advantage rests on the untestable
assumption that the representation input is admissible.

### B.2 Randomized representations

If $Z = \phi(X, W_{-S}, \xi)$ with $\xi$ exogenous, identification is unchanged: the assumptions survive the
extra conditioning and the theorem never used determinism. Appending noise as a coordinate changes nothing.
Garbling trades balance for overlap: in
$D^2_{\mathrm{res}} = \mathbb E\{\mathrm{Var}(A \mid Z)\} - \mathbb E\{\mathrm{Var}(A \mid T, Z)\}$ the value
is zero at no noise for a balancing $\phi$ and rises with the noise level, while the overlap constant improves,
so Theorem 3's bound can be minimized at a strictly stochastic representation. Numerically the optimum stayed
at zero noise (A.5). A pure-noise representation has a strictly positive discrepancy, so randomization cannot
fool the audit.

### B.3 The three-role search

`all_configs(n_blocks, max_held, max_excluded)` searches over (audited, representation, excluded) rather than
(audited, rest). This is inside the theory, since Assumption 2 refers to the blocks that actually enter the
representation, and it only changes the counts in Theorem 6. It does not rescue the collider case: the family
is dominated by configurations that keep the collider in the representation and pass the audit, and the
correct configurations disagree with each other depending on how much valid proxy is left.
