# Current Artifact Manifest

Created: 2026-07-06
Cross-view manifest refreshed: 2026-08-24

This manifest identifies the active artifacts for the final cross-view and cross-mask project. It does not list every historical development file retained in the directory.

- **Frozen for writing the paper, 2026-09-23:** [results/FINAL_FREEZE_v1.json](results/FINAL_FREEZE_v1.json) holds the sha256 of every result file of E1 to E8, the seven paper figures and `EXPERIMENT_INDEX.md` (271 files). None is edited. The replicate extension PRD is shelved, not run.
- E3 supplement, 20 bootstrap resamples of RHC: [code/rhc_bootstrap.py](code/rhc_bootstrap.py), `results/rhc_bootstrap/`, [figures/section5/rhc-bootstrap.pdf](figures/section5/rhc-bootstrap.pdf) (also `ICLR/paper/figures/probe-rhc-bootstrap.pdf`), [memo/2026-09-23-rhc-bootstrap-v1.md](memo/2026-09-23-rhc-bootstrap-v1.md); frozen as `results/FINAL_FREEZE_v1_addendum_1.json`. The frozen index and `probe-rhc.pdf` are unchanged.
- Extension to about 100 data sets per experiment, run on NCSA DeltaAI after the freeze: [EXPERIMENT_INDEX_EXTENSION_v1.md](EXPERIMENT_INDEX_EXTENSION_v1.md) (sealed, added and pooled numbers), [memo/2026-09-23-replicate-extension-prd-v2.md](memo/2026-09-23-replicate-extension-prd-v2.md) (plan, amendments 1 to 3, run log), `results/extension_v1/` (protocol, 648 data sets, 488 KSPC rows, gates, summary), `results/rhc_p2sls_cui/v1.json` (RHC 2SLS with Cui et al.'s specification), `figures/section5/extension-*.pdf`; code `code/run_extension_v1.py`, `code/summarise_extension_v1.py`, `code/make_extension_figures.py`, `code/write_extension_index.py`, `code/extension_v1_gate.py`, `code/rhc_p2sls_cui.py`; frozen as `results/FINAL_FREEZE_v1_addendum_2.json`. The frozen index and figures are unchanged. Paper figures from the pooled data in the v4 design: `ICLR/paper/figures/probe-{synthetic,realdata}-v4-ext.pdf`, source [code/make_section5_v4_extension.py](code/make_section5_v4_extension.py) (v4's drawing and checks, unchanged; two axes widened so every point shows; RHC panel without Cui et al.'s standard DR).
- [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md): **the seven final experiments** (E1 SCM-1 to 3, E2 SCM-4, E3 observed RHC, E5 Twins 11 to 20, E6 IHDP 12 to 21, E7 ACIC 2 to 10, exploratory, E8 KSPC benchmark; E4 retired), each with its frozen record, code, result files, figure and numbers. Everything else is recorded but not final.

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

Registered on 2026-09-22: [materials/real_world_data/ihdp](materials/real_world_data/ihdp), the ACIC 2016 realizations read from the TransPFN workspace, and [materials/real_world_data/zika](materials/real_world_data/zika) (the municipality panel distributed with Park, Richardson and Tchetgen Tchetgen, 2024).

### Two real proxies: Zika and LaLonde

These are the only two data sets in the paper that are real throughout; the treatment, the covariates, both proxies and the outcome are all published columns. Each has exactly two proxies and no agreed assignment of roles between them.

- [code/probe_j2.py](code/probe_j2.py): Algorithm 1 over J blocks, reusing the sealed `learn_representation`, `screen`, `aipw`, `common_radius` and `aggregate` unchanged.
- [code/two_proxy_data.py](code/two_proxy_data.py): both views, and the propensity band declared in the source before any estimate was formed.
- [code/run_two_proxy.py](code/run_two_proxy.py), [code/two_proxy_uncertainty.py](code/two_proxy_uncertainty.py), [code/make_two_proxy_figure.py](code/make_two_proxy_figure.py), [code/test_two_proxy.py](code/test_two_proxy.py) (14 checks).
- [results/two_proxy/](results/two_proxy): `zika_v1.json`, `lalonde_v1.json`, `uncertainty_v1.json`, all write-once.
- [memo/2026-09-22-two-proxy-zika-lalonde-v1.md](memo/2026-09-22-two-proxy-zika-lalonde-v1.md): what the runs showed, including a limitation of the screen's overlap check that they exposed.
- KSPC (Xu and Gretton, arXiv 2308.04585): authors' code in `materials/external_code/KernelSingleProxy/` (commit a5283d62, see its PROVENANCE.md), [code/kspc_wrapper.py](code/kspc_wrapper.py) with a sanity gate on the authors' first experiment; the KSPC benchmark with a binary treatment [code/kspc_design.py](code/kspc_design.py), [code/run_kspc_design.py](code/run_kspc_design.py), `results/kspc_design/`, [figures/section5/kspc-benchmark.pdf](figures/section5/kspc-benchmark.pdf), [memo/2026-09-22-kspc-benchmark-results-v1.md](memo/2026-09-22-kspc-benchmark-results-v1.md), [code/test_kspc.py](code/test_kspc.py). WSC and LaLonde are no longer drawn (author's decision).
- WSC's exact answer for the observational sample, 0.787 (se 0.142), from the randomisation within each preference group: [code/wsc_exact_benchmark.py](code/wsc_exact_benchmark.py), `results/wsc/exact_benchmark_v1.json`. The figures now score WSC's whole-sample reading against it and no longer draw LaLonde (author's decision).
- WSC with PROBE (J = 7, 126 candidates): [code/wsc_probe.py](code/wsc_probe.py), [code/test_wsc_probe.py](code/test_wsc_probe.py) (4 checks), [results/wsc/probe_v1.json](results/wsc/probe_v1.json), [memo/2026-09-22-wsc-probe-results-v1.md](memo/2026-09-22-wsc-probe-results-v1.md). PROBE 2.248 on the band against the benchmark 0.760; adjusting for X and all items 1.229. Drawn as the fourth panel of `nonwinning-real` and a row of `win-boundary`.
- LaLonde, ten real samples (the two early random-assignment AFDC women samples added; section 10 of the two-proxy memo). Previously: CPS-1 to CPS-3 and PSID-1 to PSID-3 against the 185 NSW men, and the NSW AFDC women against PSID-1 and PSID-2 women (`two_proxy_data.load_lalonde_group`, `load_afdc`, `afdc_benchmark`); downloads registered in `materials/real_world_data/manifest.json`; results `results/two_proxy/*_v1.json`; section 9 of the two-proxy memo. `nonwinning-real` draws them as one dot cloud per method.
- LaLonde against the PSID-1 controls: `two_proxy_data.load_lalonde_psid`, [results/two_proxy/lalonde_psid_v1.json](results/two_proxy/lalonde_psid_v1.json), `uncertainty_lalonde_psid_v1.json`; recorded in section 8 of the two-proxy memo. The boundary figures now omit planted RHC (author's decision) and draw LaLonde with both comparison groups.
- KSPC comparator on E1 to E7 (SKPV with covariates): [code/run_kspc_comparator.py](code/run_kspc_comparator.py), [code/kspc_comparator_integrity.py](code/kspc_comparator_integrity.py), [code/finalize_kspc_comparator.py](code/finalize_kspc_comparator.py), [code/test_kspc_comparator.py](code/test_kspc_comparator.py); `results/kspc_comparator/` (frozen, 120 files); [memo/2026-09-23-kspc-comparator-results-v1.md](memo/2026-09-23-kspc-comparator-results-v1.md). The KSPC rows are in `probe-estimates.pdf` and `probe-realdata.pdf`.
- Section 5 main figure v3 (E1, E2), 2026-09-23: [code/make_section5_v3_figures.py](code/make_section5_v3_figures.py) writes `figures/section5/section5-v3-estimates.{pdf,png}`, copied to `ICLR/paper/figures/probe-estimates-v3.pdf`. It reads the same frozen E1 and E2 shards and `results/kspc_comparator/E1`, `E2` through the v2 loaders, plots estimate minus the true effect on a symmetric log scale with role groups and a per-row MAE column, and checks that all 45 MAE cells match `EXPERIMENT_INDEX.md`. It is proposed to replace `probe-estimates.pdf` in Section 5; the v2 line stays as a source comment until the author accepts.
- Section 5 compact figures v4 (E1, E2, E5 to E7, E3), 2026-09-23: [code/make_section5_v4_figures.py](code/make_section5_v4_figures.py) writes `figures/section5/section5-v4-{synthetic,realdata}.{pdf,png}`, copied to `ICLR/paper/figures/probe-{synthetic,realdata}-v4.pdf`. Same style as v3 at 2.34 in and 2.12 in tall, with correct and wrong roles merged into one row. The script checks 75 MAE cells against `EXPERIMENT_INDEX.md`, the RHC values against E3, and the 110 KSPC files against `FROZEN_v1.json`. In the rewritten Section 5, v4 is the main-text figure and v3 moves to Appendix D as the detailed version.
- 2026-09-23: at the author's request, `code/make_section5_v4_figures.py` no longer draws the dagger on KSPC or the "run after the protocol was frozen" footnote; `code/make_section5_v4_extension.py` was rerun (all checks passed), so `ICLR/paper/figures/probe-{synthetic,realdata}-v4-ext.pdf` changed only in those labels. The post-freeze status of the KSPC runs stays recorded in `EXPERIMENT_INDEX.md`.
- Section 5 Figure 3 v5 (design A, chosen by the author on 2026-09-23): [code/make_section5_v5_figure.py](code/make_section5_v5_figure.py) writes `figures/section5/section5-v5-experiments.{pdf,png}`, copied to `ICLR/paper/figures/probe-experiments-v5.pdf`. One row of seven panels (SCM-1 to SCM-4, Twins, IHDP, ACIC), 5.5 x 2.72 in, 10.5 pt text; same pooled data and checks as the v4 extension figures, which move to Appendix D as the detailed versions with MAE numbers and the RHC panel. Revised 2026-09-23 at the author's request: PROBE's row stands apart under its own solid separator, PROBE's pooled MAE (2 significant digits, decimal half-up, checked against summary_v1.json) is written in gray under each panel title, and "NA" in gray marks the pixel CNN outside SCM-4; now 5.5 x 2.92 in (height cap raised from 2.9 to 3.0 in), same data and dot positions.
- Figure 1 v2 (flatter, 2026-09-23 at the author's request): [figures/2026-09-23-proximal-balancing-structures-v2.tex](figures/2026-09-23-proximal-balancing-structures-v2.tex) compiles with pdflatex to the pdf of the same stem, copied to `ICLR/paper/figures/causal-structures-3panel-v2.pdf` and included at natural size (363 x 72 pt, against 293 x 97 pt scaled to 0.91 textwidth before). Same nodes and edges as version 1 (`2026-09-23-proximal-balancing-structures.tex`, `causal-structures-3panel.pdf`, kept); rows closer, panel labels at the top left.
- Figure 2 v2 (2026-09-23, final experiments only): [code/make_probe_mechanism_figure_v2.py](code/make_probe_mechanism_figure_v2.py) writes `figures/section5/probe-mechanism-v2.{pdf,png}`, copied to `ICLR/paper/figures/probe-mechanism-v2.pdf`. Panels (c) and (d) now come from the E2 sealed shard `results/family_v2/scm4/confirm/SCM-4_seed97400000.json` (rep 0, lowest seed): the screen statistic is the maximum over the four rotations of max(gap, 0) + z SE, t = 0.0017, 8 of 30 splits kept (S4 among them), linking radius 2 rho = 0.117, components [7, 1], returned 0.992. Panels (a) and (b) show J = 5 blocks. Version 1 (`code/make_probe_mechanism_figure.py`, `probe-mechanism.pdf`, positive-control audit, not final) is kept. Labels reworded on 2026-09-24 to match the main text (splits, held-out blocks, component, minimize); same data and numbers.
- 2026-09-24: [code/make_section5_v2_figures.py](code/make_section5_v2_figures.py) (only `fig_aggregation`, `fig_coverage`, `fig_learned` rerun) redrew `figures/section5/section5-v2-{aggregation,coverage,learned}.{pdf,png}`, copied to `ICLR/paper/figures/probe-{aggregation,coverage,learned}.pdf`: labels 'choice' changed to 'split'; data unchanged (same sealed shards; plotted values and axes positions identical to the previous run).
- [memo/2026-09-22-kspc-comparator-prd-v2.md](memo/2026-09-22-kspc-comparator-prd-v2.md): PRD v2, KSPC with covariates (Appendix A) as a comparator on E1 to E7. Awaiting approval. Supersedes workstream A of v1.
- [memo/2026-09-22-kspc-lalonde-wsc-prd-v1.md](memo/2026-09-22-kspc-lalonde-wsc-prd-v1.md): PRD draft for the kernel single proxy control comparator, more LaLonde samples, and the WSC benchmark. Awaiting approval.
- [code/extend_rhc_planted.py](code/extend_rhc_planted.py): thirty more planted RHC replicates (21-50), fixed in advance, run with the code the frozen protocol hashed, scored with the v3 rule; [results/rhc_planted/extension_v1.json](results/rhc_planted/extension_v1.json); [memo/2026-09-22-rhc-planted-extension-v1.md](memo/2026-09-22-rhc-planted-extension-v1.md); [code/test_rhc_planted_extension.py](code/test_rhc_planted_extension.py) (3 checks). The boundary figures read replicates 11-50 when the extension exists.
- [code/make_boundary_figures.py](code/make_boundary_figures.py): [figures/section5/nonwinning-real.pdf](figures/section5/nonwinning-real.pdf) (the four data sets with a known effect where PROBE is not the most accurate) and [figures/section5/win-boundary.pdf](figures/section5/win-boundary.pdf) (all nine, as PROBE's error over the best role-free rival's, with a 90% replicate bootstrap). Reads existing results only. Not referenced by any section.
- [figures/section5/two-proxy.pdf](figures/section5/two-proxy.pdf), copied to `ICLR/paper/figures/probe-two-proxy.pdf`. Not yet referenced by any section.

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
