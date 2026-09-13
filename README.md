# PROBE: Single Proxy Balancing

Created: 2026-07-06
Held-out-block direction locked: 2026-08-24
Manuscript theory refreshed: 2026-09-09

This project studies causal adjustment with observed pretreatment covariates $X$, one high-dimensional pretreatment proxy $W=(W_1,\ldots,W_J)$ of an unobserved confounder $U$, a binary treatment $A$, and an outcome $Y$. The method, PROxy Blockwise Exclusion (PROBE), withholds one proxy block, learns an adjustment representation from the observed covariates and the remaining blocks, and uses the withheld block to measure the treatment imbalance that remains.

## Method

For a held-out block $j$, write $V_j=(X,W_{-j})$ for the representation input and $T_j=(X,W_j)$ for the held-out pair. A representation is $Z_j=\phi_j(V_j)$. Its residual treatment discrepancy is

$$
D_{\mathrm{res},j}^2(\phi)
\triangleq
\mathbb E\bigl[\{P(A=1\mid T_j,Z_j^\phi)-P(A=1\mid Z_j^\phi)\}^2\bigr],
$$

which equals zero exactly when $(X,W_j)\perp\!\!\!\perp A\mid Z_j^\phi$. The empirical version is the difference of two Brier risks: one critic predicts $A$ from $Z_j^\phi$ alone, the other from $(T_j,Z_j^\phi)$. The encoder minimizes this discrepancy on a discrepancy-learning sample, nuisance functions are fitted on a second sample, and an honest AIPW estimate is computed on a third sample. When the valid held-out block is unknown, Algorithm 1 of the manuscript samples proxy splits, screens them by discrepancy and overlap, links estimates within a radius, and returns the median of the largest linked group.

## Canonical Theory

The manuscript is the canonical theory source: [manuscript/main/3.tex](manuscript/main/3.tex) (identification), [manuscript/main/4.tex](manuscript/main/4.tex) (learning, AIPW estimation, proxy-split search), and [manuscript/appendix/proof.tex](manuscript/appendix/proof.tex) (all proofs). As of the 2026-09-09 draft, the results are:

- Section 3: Assumptions 1 to 4 (consistency, latent exchangeability with a common held-out channel, outcome-relevant completeness, $L_2$ outcome-relevant stability); Proposition 1 (sufficient conditions for completeness); Theorem 1 (exact identification under exact held-out balance); Proposition 2 (non-nesting with the two-proxy proximal conditions); Theorem 2 and Corollary 1 (plurality and majority identification over proxy splits); Proposition 3 (zero discrepancy equals exact balance); Theorem 3 (bias bound $\Gamma_j D_{\mathrm{res},j}/\{\eta(1-\eta)\}$); Proposition 4 (population sensitivity region).
- Section 4: Brier-risk residual discrepancy; Proposition 5 (uniform deviation); Proposition 6 (oracle inequality); Theorem 4 (causal error of the learned representation); the AIPW score and Proposition 7 (exact remainder); Theorem 5 (finite-sample causal error of honest AIPW); Corollary 2 (consistency); Algorithm 1 and Theorem 6 (error of Algorithm 1); Corollary 3 (budget for a target failure level); Corollary 4 (median under majority).

[THEORY.md](THEORY.md) is the historical cross-view theory document. It predates the residual-discrepancy formulation and is retained for provenance only.

Open items are the neural implementation of the Brier-risk critics, the soft-to-hard gap of earlier experiments, calibration of $\Gamma_j$, and inference after adaptive representation selection.

## Lean Scope

[proofs/single_proxy_balance_core.lean](proofs/single_proxy_balance_core.lean) checks only an abstract audit-channel injectivity implication. It does not formalize conditional probability, the audit factorization, causal identification, RKHS stability, or finite-sample results.

## Manuscript

The canonical authoring source is [manuscript](manuscript). Publishing follows the recorded one-way flow from the PAIOS manuscript to the standalone Dropbox mirror `/Users/yonghanjung/Dropbox/Personal/Research/Code/single-proxy-balancing` and then to the private GitHub repository `CausalDataScience/single-proxy-balancing`, only after user authorization.

## Experiment Program (2026-09)

The active experimental program (five main SCMs, Algorithm 1 pipeline, MNIST proxies, comparators, Twins) is described for collaborators in [EXPERIMENTS.md](EXPERIMENTS.md). Code: [code/probe_scms.py](code/probe_scms.py), [code/probe_e1_pipeline.py](code/probe_e1_pipeline.py), [code/probe_e4_mnist.py](code/probe_e4_mnist.py).

## Accepted Experimental Evidence

The manuscript evidence currently consists of the following studies. They use a random-Fourier-feature surrogate objective and mask averaging, and are scheduled to be superseded by experiments that run the Brier-risk critics and Algorithm 1 of the manuscript.

### Honest Approximate SCM1 and SCM2

- Code: [code/honest_canonical_representation_experiment.py](code/honest_canonical_representation_experiment.py)
- SCM1 results:
  - [results/honest_canonical_representation_original_scm1_20rep_summary.json](results/honest_canonical_representation_original_scm1_20rep_summary.json)
  - [results/honest_canonical_representation_original_scm1_n2000_5000_10000_20rep_summary.json](results/honest_canonical_representation_original_scm1_n2000_5000_10000_20rep_summary.json)
- SCM2 results:
  - [results/honest_canonical_representation_nonlinear_scm2_20rep_summary.json](results/honest_canonical_representation_nonlinear_scm2_20rep_summary.json)
  - [results/honest_canonical_representation_nonlinear_scm2_n2000_5000_10000_20rep_summary.json](results/honest_canonical_representation_nonlinear_scm2_n2000_5000_10000_20rep_summary.json)
- Design: 20 Monte Carlo repetitions per SCM with nested sample sizes $n\in\{100,200,500,1000,2000,5000,10000\}$.

### Exact-Assumption Track A and Track B

- Track A code: [code/exact_injectivity_finite_strata_experiment.py](code/exact_injectivity_finite_strata_experiment.py)
- Track A result: [results/exact_injectivity_finite_strata_summary.json](results/exact_injectivity_finite_strata_summary.json)
- Track B code: [code/exact_injectivity_continuous_cci_experiment.py](code/exact_injectivity_continuous_cci_experiment.py)
- Track B result: [results/exact_injectivity_continuous_cci_summary.json](results/exact_injectivity_continuous_cci_summary.json)
- Both tracks use root seed 190602 and 20 repetitions.

### WSC Observational Benchmark

- Shared implementation: [code/wsc_population_ate_pilot.py](code/wsc_population_ate_pilot.py)
- Final implementation: [code/wsc_overlap_constrained_followup.py](code/wsc_overlap_constrained_followup.py)
- Final result: [results/wsc_overlap_constrained_followup.json](results/wsc_overlap_constrained_followup.json)
- Result interpretation: [memo/2026-08-23-wsc-overlap-constrained-followup-results.md](memo/2026-08-23-wsc-overlap-constrained-followup-results.md)

WSC uses one fixed observational dataset and 10 algorithmic sample-split repetitions, not 10 Monte Carlo datasets. The randomized contrast is a noisy descriptive benchmark rather than causal ground truth. Only the unconstrained cross-mask learner is the proposed method. This benchmark does not establish proxy validity, exact balance, population positivity, or point identification.

## Data

Real-world data live under [materials/real_world_data](materials/real_world_data) with provenance and SHA-256 hashes in [materials/real_world_data/manifest.json](materials/real_world_data/manifest.json):

- `wscdata/`: Within-Study Comparison data (Keller et al., 2025), the accepted observational benchmark.
- `lalonde/`: LaLonde (1986) data as packaged by Imbens and Xu (2024).
- `rhc/`: Right heart catheterization data (Connors et al., 1996), used in the proximal inference literature.
- `twins/`: CEVAE Twins files (Louizos et al., 2017), the planned semi-synthetic benchmark with a categorical latent confounder and noisy proxy blocks.

## Final Figures

- [figures/final/figure_approximate_scms.pdf](figures/final/figure_approximate_scms.pdf)
- [figures/final/figure_exact_tracks.pdf](figures/final/figure_exact_tracks.pdf)
- [figures/final/wsc_benchmark.pdf](figures/final/wsc_benchmark.pdf)

## Historical Boundary

Earlier $Z=\phi(X)$ theory, same-view $R=\phi(X,W)$ proposals, finite-state certificate theory, front-door sketches, and exploratory high-dimensional or image experiments are not active theoretical or manuscript evidence. Their code, results, figures, and memos remain in the project as historical development artifacts and are not deleted by this cleanup.

KMU and Zika are excluded from the final evidence. Their retained artifacts record negative or invalidated benchmark attempts rather than support for the method.

## Reproducibility Utilities

The project-local final-experiment utilities are under [code/final_experiments](code/final_experiments). Their metadata may still contain historical package entries until a separately approved publication-package refresh. The current scientific authority is this README, [THEORY.md](THEORY.md), the canonical manuscript, and the exact result files listed above.
