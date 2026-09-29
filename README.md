# Proximal Balancing for Causal Effect Estimation under Unmeasured Confounding

Code, results and paper for **Proximal Balancing for Causal Effect Estimation under Unmeasured Confounding**
(Yonghan Jung, University of Illinois Urbana-Champaign, 2026). The paper is in [`paper/main.pdf`](paper/main.pdf).

Causal effects are not identified from observational data when confounders are unmeasured. Proximal causal
inference addresses this with proxies of the unmeasured confounders. Existing proxy-based methods either designate
proxy roles and solve an ill-posed inverse problem, or fit a latent-variable model and are biased when the learned
latent does not match the hidden confounder. *Proximal balancing* carries covariate balancing to confounders that
are observed only through proxies: it learns a low-dimensional summary of the covariates and proxies that makes the
treatment groups comparable, and then adjusts for that summary. It needs no designated proxy roles, no inverse
problem and no latent model. The paper gives identification theory, finite-sample guarantees and a practical
algorithm, **PROBE**, and evaluates it on low-dimensional, high-dimensional and image proxies and on real data.

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | LaTeX source and PDF of the paper (CC BY 4.0) |
| `code/` | PROBE, the data-generating processes, the competitors, every experiment runner, the summarizers, the figure scripts and the tests |
| `results/` | The frozen results behind every number, table and figure in the paper (about 43 MB of JSON) |
| `memo/` | The protocols and preregistrations the experiments were run under; the runners check their hashes |
| `figures/` | TikZ sources of the three diagrams and the figure files the scripts write |
| `EXPERIMENT_INDEX.md`, `EXPERIMENT_INDEX_EXTENSION_v1.md` | Index of the final experiments; the figure scripts cross-check their numbers against it |
| `materials/` | Created by `scripts/download_data.py`; holds the third-party data and the Shapes3D reader we trained |
| `scripts/` | `download_data.py`, `reproduce_figures.sh`, `run_tests.sh` |

## Setup

The results in the paper were computed with Python 3.14.7 on an Apple M4 (macOS); the replicates added after the
original protocol (appendix, "Added data sets") were computed with Python 3.11.9 on NCSA DeltaAI.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py          # about 55 MB; add --only shapes3d for the 268 MB Shapes3D file
```

`pdflatex` (with `latexmk`) is needed only for the TikZ diagrams and the paper; `pdftotext` (poppler) is used for an
optional cross-check in one figure script.

## Data

No dataset is stored in this repository. `scripts/download_data.py` fetches each file from its original source at a
pinned version, checks its sha256 and places it where the code expects it. `--list` shows every file and checksum,
`--check-urls` checks that the sources are reachable, and `--from-local DIR` copies from a local tree instead of
downloading.

| Dataset | Source | Terms |
|---|---|---|
| Twins | CEVAE repository (AMLab-Amsterdam/CEVAE, commit 9081f863) | No licence file; derived from NBER public-use linked birth and infant death data. Cite Louizos et al. (2017) and Almond, Chay and Lee (2005). |
| IHDP | fredjo.com (Johansson et al.) | No terms stated. Cite Hill (2011) and Johansson et al. (2016). |
| RHC | hbiostat.org (Vanderbilt Biostatistics) | Use allowed with citation of Connors et al. (1996). |
| ACIC 2016 | causallib (BiomedSciAI/causallib, commit 267e0c72) | CDLA-Sharing-1.0. Cite Dorie et al. (2019). |
| LaLonde / NSW | xuyiqing/lalonde (commit c179a80e); Dehejia's NBER page as fallback | MIT. |
| Zika | qkrcks0218/SingleProxyControl (commit 37270e8e) | Built from Harvard Dataverse doi:10.7910/DVN/ENG0IY (CC0) and IBGE census data. |
| MNIST | Keras datasets bucket | CC BY-SA 3.0. |
| Shapes3D | DeepMind 3D Shapes (optional, 268 MB) | Apache-2.0. |
| KSPC code | liyuan9988/KernelSingleProxy, commit a5283d62 (fetched with `git`) | No licence; fetched from the authors, not redistributed. |

## Reproducing the paper

### Figures

`bash scripts/reproduce_figures.sh` rebuilds every figure from the shipped results.

| Paper figure (file in `paper/figures/`) | Built by |
|---|---|
| Three causal structures, Section 1 (`causal-structures-3panel-v2.pdf`) | `figures/2026-09-23-proximal-balancing-structures-v2.tex` (pdflatex) |
| PROBE mechanism, Section 4 (`probe-mechanism-v2.pdf`) | `code/make_probe_mechanism_figure_v2.py`; needs the Shapes3D image cache (see below) |
| Experiments, Section 5 (`probe-experiments-v5.pdf`) | `code/make_section5_v5_figure.py`; run after the v4 script, whose PDFs it checks |
| Proximal causal structures, appendix B (`causal-graphs.pdf`) | `figures/2026-08-27-proximal-balancing-causal-graphs.tex` |
| Majority-valid orientations, appendix B (`majority-orientations.pdf`) | `figures/2026-08-31-majority-valid-orientations.tex` |
| Synthetic estimates, appendix D (`probe-estimates-v3.pdf`) | `code/make_section5_v3_figures.py` |
| KSPC benchmark, appendix D (`probe-kspc.pdf`) | `code/make_kspc_figure.py` |
| All replicates, synthetic and real data, appendix D (`probe-synthetic-v4-ext.pdf`, `probe-realdata-v4-ext.pdf`) | `code/make_section5_v4_extension.py` |
| Aggregation, coverage and learned summary, appendix D (`probe-aggregation.pdf`, `probe-coverage.pdf`, `probe-learned.pdf`) | `code/make_section5_v2_figures.py` |

The Python scripts write `figures/section5/<name>.{pdf,png}`; the v3, v4, v5 and mechanism scripts also copy their
output into `ICLR/paper/figures/`, the folder layout of the conference submission, which `reproduce_figures.sh`
creates. We checked the release this way: the v3, v4 and v5 figures come out byte-identical to the paper's, and the
v2, KSPC and TikZ figures differ only in the creation date embedded in the PDF (identical pixels at 100 dpi). The
mechanism figure needs the Shapes3D image cache (5.9 GB): run `python scripts/download_data.py --only shapes3d`, then
build the cache with `family_v2_images.build_cache()` from `code/`.

### Tables

No script writes the tables; their numbers are copied from these files.

| Table | Source |
|---|---|
| Verified results (appendix D) | `results/family_v2/confirm_summary_v1.json` (`levels.SCM-k`: `mae`, `p90`, `paired_upper.oracle_excess_upper`) and `results/family_v2/scm4/confirm_summary_v1_amended.json` |
| Benchmarks (appendix D) | `results/{twins,ihdp}/summary_v3_fresh.json`, `results/acic/summary_v1.json`, `results/noise_scaled_v3.json`, `results/coca_ate_all_v1.json`, `results/kspc_comparator/summary_v1.json` |
| Folds by setting (appendix D) | The design, as listed in `EXPERIMENT_INDEX.md`; the noise floors in the next paragraph come from `results/noise_scaled_v3.json` |

### Rerunning the experiments

The commands below produced the shipped results (run from `code/`). The runners write into `results/`; to recompute
an experiment instead of reusing the shipped files, run it in a separate copy of the repository with that
experiment's result folder removed.

| Experiment (paper) | Commands |
|---|---|
| E1, synthetic SCM-1 to SCM-3 | `run_family_v2.py freeze-dev`, `run_family_v2.py dev --workers 6`, `summarize_family_v2.py dev`, `run_family_v2.py freeze-confirm 0.0015 0.0018 0.0016`, `run_family_v2.py confirm --workers 6`, `summarize_family_v2.py confirm` |
| E2, SCM-4 with image proxies | Shapes3D as above; `run_scm4_images.py freeze-reader`, `connect`, `connect2`; `run_scm4.py freeze-dev`, `dev`; `summarize_scm4.py dev`; `run_scm4.py freeze-confirm 0.0017`, `confirm --workers 3`; `summarize_scm4.py confirm`; `rerun_scm4_pixel_cnn.py rerun`, `amend` |
| E3, RHC | `run_rhc.py`, `rhc_pooled.py`, `noise_scaled.py`, `rhc_aggregate_uncertainty.py`; the bootstrap and the proximal 2SLS row: `rhc_bootstrap.py run`, `rhc_bootstrap.py summarise`, `rhc_p2sls_cui.py` |
| E5 to E7, Twins, IHDP, ACIC 2016 | `run_twins.py run --replicates 1-10`, `run_twins.py summarise`; `run_ihdp_acic.py run ihdp`, `summarise ihdp`, and the same for `acic`; `freeze_v3_protocol.py`, `correct_v3_protocol.py`; `run_v3_fresh.py twins`, `ihdp`; `v3_fresh_scorer.py twins`, `ihdp`; `coca_ate_all.py` |
| E8, KSPC design | `run_kspc_design.py run --workers 2`, `run_kspc_design.py summarise` |
| KSPC comparator rows | `run_kspc_comparator.py run E1 E3 E5 E6 E7`, `run E2`, `summarise`; `kspc_comparator_integrity.py` |
| Added replicates (appendix, "Added data sets") | `run_extension_v1.py check-sources`, `freeze`, `run --cores 144 --gpus 2 --per-gpu 4 --time-limit-s 12400` (Slurm script: `code/deltaai/extension_v1_main.sbatch`); `extension_v1_gate.py`; `summarise_extension_v1.py --write`; `write_extension_index.py` |

`noise_scaled.py` reads every `results/<name>/replicate_*.json`, including the prospective Twins (11 to 20) and IHDP
(12 to 21) replicates that were written after `results/noise_scaled_v3.json`. To recompute that file, move those
replicates aside first; the result is then identical apart from its `generated_utc` timestamp.
`make_extension_figures.py` and `make_rhc_bootstrap_figure.py` draw diagnostic figures of the added replicates; they
are listed in the freeze record but not used in the paper.

Two result files have no writer script: `results/rhc/diagnostic_v1_full_records.json` (a rerun of `run_rhc.py` that
kept every bootstrap draw) and the `results/kspc_design/FROZEN_v1*.json` protocol files. Do not run
`finalize_kspc_comparator.py` when rebuilding the figures: it overwrites two earlier figure versions.

PROBE's iterative fits are not bit-for-bit reproducible across machines. On the replicates we checked, the data were
identical on an Apple M4 and on DeltaAI and the final PROBE estimates differed by 0.0002 to 0.009; the appendix
section "Added data sets" describes the check we used before pooling the added replicates.

### Frozen protocols

Every experiment was frozen before it was run: the protocol files in `memo/` and `results/**/FROZEN*.json` record
sha256 hashes of the source files, the seeds and the thresholds, and the runners refuse to run when a hashed file has
changed. For that reason the code is shipped exactly as it was run, including a few machine-specific strings:
`PYTHON_PREFIX` in `code/family_v2_images.py` (checked only when the Shapes3D cache is rebuilt with `prepare`; use
`family_v2_images.build_cache()` to rebuild the cache alone), the Slurm account in `code/deltaai/*.sbatch`, and the
absolute paths recorded in the `sealed` field of some result files, which only say where a file was first written.
`results/FINAL_FREEZE_v1.json` and its two addenda list every final result file with its sha256.

## Tests

`bash scripts/run_tests.sh` runs the unit tests of PROBE, the data-generating processes, the loaders, the scorers and
the extension driver.

The unit-test files run as `python3 -B test_<name>.py` from `code/`; `test_kspc.py`, `test_kspc_comparator.py` and
`test_two_proxy.py` run under `pytest`. The whole suite (15 files, 222 tests, one skipped by design) takes about
4 minutes with two threads and needs the default datasets from `scripts/download_data.py`.

#### Hashes recorded before later edits

The release is byte-identical to the project the paper was written from, but a few files were edited after a
protocol recorded their hash, so a check against that protocol reports a mismatch: `EXPERIMENT_INDEX.md` (against
`results/FINAL_FREEZE_v1.json`), the protocol documents `memo/2026-09-18-family-v2-role-blind-dimension-prd-v1.md`,
`memo/2026-09-20-scm4-shapes3d-reader-spec-v1.md`, `memo/2026-09-22-kspc-comparator-prd-v2.md` and
`memo/2026-09-23-replicate-extension-prd-v2.md` (edited after freezing, for example to append status logs and
amendments), and
`code/freeze_v3_protocol.py` and `code/v3_fresh_scorer.py` (against the v3 protocols; the scorer change is disclosed
in the extension protocol). `EXPERIMENT_INDEX.md` also cites internal result memos that are not part of this
release; the numbers they summarize are in `results/`.

## Citation

```bibtex
@misc{jung2026proximal,
  title        = {Proximal Balancing for Causal Effect Estimation under Unmeasured Confounding},
  author       = {Jung, Yonghan},
  year         = {2026},
  howpublished = {\url{https://github.com/CausalDataScience/proximal-balancing}}
}
```

## License

The code is released under the MIT License (`LICENSE`). The paper in `paper/` is released under CC BY 4.0
(`paper/LICENSE`). The datasets and the KSPC authors' code keep their own terms (see [Data](#data)).
