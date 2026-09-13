# Current Artifact Manifest

Created: 2026-07-06
Cross-view manifest refreshed: 2026-08-24

This manifest identifies the active artifacts for the final cross-view and cross-mask project. It does not list every historical development file retained in the directory.

## Canonical Theory

- The manuscript Sections 3 and 4 with the proofs in [manuscript/appendix/proof.tex](manuscript/appendix/proof.tex) are the canonical theory as of 2026-09-09.
- [THEORY.md](THEORY.md): historical cross-view theory document, retained for provenance.
- [proofs/single_proxy_balance_core.lean](proofs/single_proxy_balance_core.lean): an abstract audit-channel injectivity lemma only. It is not a formal proof of the probability, causal, RKHS, or finite-sample theorems.

## Canonical Manuscript

- [manuscript/main.tex](manuscript/main.tex): manuscript entrypoint.
- [manuscript/main/1.tex](manuscript/main/1.tex): introduction and related work (draft).
- [manuscript/main/3.tex](manuscript/main/3.tex): identification theory source.
- [manuscript/main/4.tex](manuscript/main/4.tex): representation learning, AIPW estimation, and proxy-split search (Algorithm 1).
- [manuscript/main/5.tex](manuscript/main/5.tex): accepted experiments source.
- [manuscript/appendix/proof.tex](manuscript/appendix/proof.tex): proofs of every stated result.
- [manuscript/main.pdf](manuscript/main.pdf): generated manuscript PDF.

The manuscript authoring authority is the PAIOS project path above. The standalone Dropbox checkout and private GitHub repository are publication mirrors, not reverse-sync sources.

## Active Experiment Program (2026-09-09)

- [EXPERIMENTS.md](EXPERIMENTS.md): handoff plan with the pipeline description, the five main SCMs, status, and known problems.
- [code/probe_scms.py](code/probe_scms.py), [code/probe_e1_pipeline.py](code/probe_e1_pipeline.py), [code/probe_e4_mnist.py](code/probe_e4_mnist.py): active code.
- [results/probe_e1_s10.json](results/probe_e1_s10.json) and siblings: E1 grid output (in progress; `complete` flag).
- [materials/benchmarks/manifest.json](materials/benchmarks/manifest.json): MNIST provenance for E4.
- [memo/2026-09-09-e4-mnist-proxy-plan.md](memo/2026-09-09-e4-mnist-proxy-plan.md): fixed E4 design.

## Accepted Approximate-SCM Experiments

### Code

- [code/honest_canonical_representation_experiment.py](code/honest_canonical_representation_experiment.py)

### Results

- [results/honest_canonical_representation_original_scm1_20rep_summary.json](results/honest_canonical_representation_original_scm1_20rep_summary.json)
- [results/honest_canonical_representation_original_scm1_n2000_5000_10000_20rep_summary.json](results/honest_canonical_representation_original_scm1_n2000_5000_10000_20rep_summary.json)
- [results/honest_canonical_representation_nonlinear_scm2_20rep_summary.json](results/honest_canonical_representation_nonlinear_scm2_20rep_summary.json)
- [results/honest_canonical_representation_nonlinear_scm2_n2000_5000_10000_20rep_summary.json](results/honest_canonical_representation_nonlinear_scm2_n2000_5000_10000_20rep_summary.json)

Each SCM has 20 Monte Carlo repetitions. The sample sizes $100$, $200$, $500$, $1000$, $2000$, $5000$, and $10000$ are nested within each repetition.

## Accepted Exact-Assumption Experiments

### Track A

- Code: [code/exact_injectivity_finite_strata_experiment.py](code/exact_injectivity_finite_strata_experiment.py)
- Result: [results/exact_injectivity_finite_strata_summary.json](results/exact_injectivity_finite_strata_summary.json)

### Track B

- Code: [code/exact_injectivity_continuous_cci_experiment.py](code/exact_injectivity_continuous_cci_experiment.py)
- Result: [results/exact_injectivity_continuous_cci_summary.json](results/exact_injectivity_continuous_cci_summary.json)

Both tracks record root seed 190602 and 20 repetitions in their result provenance.

## Accepted WSC Observational Benchmark

### Data and provenance

- [materials/real_world_data/manifest.json](materials/real_world_data/manifest.json)
- [materials/real_world_data/wscdata](materials/real_world_data/wscdata)

Additional real-world data registered in the same manifest on 2026-09-09: [materials/real_world_data/twins](materials/real_world_data/twins) (CEVAE Twins; planned semi-synthetic benchmark), [materials/real_world_data/rhc](materials/real_world_data/rhc), and [materials/real_world_data/lalonde](materials/real_world_data/lalonde).

### Code and results

- [code/wsc_population_ate_pilot.py](code/wsc_population_ate_pilot.py): shared implementation imported by the follow-up.
- [code/wsc_overlap_constrained_followup.py](code/wsc_overlap_constrained_followup.py): final ten-split implementation.
- [results/wsc_overlap_constrained_followup.json](results/wsc_overlap_constrained_followup.json): machine-readable final result.
- [memo/2026-08-23-wsc-overlap-constrained-followup-preregistration.md](memo/2026-08-23-wsc-overlap-constrained-followup-preregistration.md): locked protocol.
- [memo/2026-08-23-wsc-overlap-constrained-followup-results.md](memo/2026-08-23-wsc-overlap-constrained-followup-results.md): interpretation.

This is one fixed observational dataset with 10 algorithmic repeated sample splits and zero Monte Carlo data-generating repetitions. The randomized contrast is a noisy descriptive reference. Only the unconstrained cross-mask learner is the proposed method.

## Final Manuscript Figures

- [figures/final/figure_approximate_scms.pdf](figures/final/figure_approximate_scms.pdf)
- [figures/final/figure_exact_tracks.pdf](figures/final/figure_exact_tracks.pdf)
- [figures/final/wsc_benchmark.pdf](figures/final/wsc_benchmark.pdf)

PNG renderings with the same stems are retained for visual QA.

## Final-Experiment Utilities

- [code/final_experiments/README.md](code/final_experiments/README.md)
- [code/final_experiments/manifest.json](code/final_experiments/manifest.json)
- [code/final_experiments/verify_manifest.py](code/final_experiments/verify_manifest.py)
- [code/final_experiments/make_manuscript_figures.py](code/final_experiments/make_manuscript_figures.py)
- [code/final_experiments/build_reproducibility_package.py](code/final_experiments/build_reproducibility_package.py)

These utilities are retained unchanged in this theory cleanup. Their package metadata may still list historical finite-state inputs until a separately approved publication-package refresh.

## Historical Boundary

All remaining code, results, figures, memos, copied idea notes, temporary bundles, and negative benchmark artifacts outside the active lists above are retained for provenance and development history. They are not current theoretical or manuscript evidence unless promoted by a later reviewed update.

In particular:

- earlier $Z=\phi(X)$ and same-view theories are obsolete;
- finite-state certificate artifacts are historical and are not manuscript evidence;
- exploratory high-dimensional and image experiments are historical;
- KMU and Zika are excluded from final evidence;
- copied front-door notes belong to historical idea development, not the final cross-view theorem.
