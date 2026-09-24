# KSPC as a comparator on the final experiments: results

2026-09-23, against `results/kspc_comparator/summary_v1.json` (frozen in `FROZEN_v1.json`, 120 files).
Protocol: `memo/2026-09-22-kspc-comparator-prd-v2.md`, including the amendment of section 9 (SKPV only).

## What ran

KSPC with covariates (Appendix A of Xu and Gretton, our implementation), estimator SKPV, on every data set
of E1, E2, E3, E5, E6 and E7, rebuilt from its frozen seed and subsampled to 3,000 rows where larger. Each
rebuilt data set reproduced its sealed shard's two COCA values exactly, and every generator file still
hashes as its frozen protocol recorded (`code_integrity_v1.json`), so the data are the sealed data. 108
tasks, no failure.

## Result

| MAE | clean block | contaminated block | PROBE, for reference |
|---|---|---|---|
| E1 SCM-1 / 2 / 3 | 0.875 / 0.854 / 0.813 | 1.881 / 1.656 / 1.796 | 0.020 / 0.017 / 0.014 |
| E2 SCM-4 | 0.870 | 1.807 | 0.018 |
| E5 Twins | **0.0099** | 0.0815 | 0.0077 |
| E6 IHDP | 0.621 | 1.250 | 0.305 |
| E7 ACIC | 0.562 | 0.932 | 0.203 |

E3 (no answer): the five blocks give −0.93 to −2.18 days, against PROBE's −1.29.

## Reading

- **Twins.** Given the clean block, KSPC is second only to PROBE (0.0099 against 0.0077) and ahead of 2SLS
  with the roles given (0.0156). Given the contaminated block it is ten times worse.
- **The simulations.** Even given the clean block, KSPC stops near 0.13 to 0.19 against a true effect of 1,
  while COCA given the same block is within 0.04. The implementation is not the reason: it reproduces the
  authors' estimator exactly with a constant X (gate 1), and on a binary treatment with an observed
  confounder and a known effect of 1 it is accurate to 0.021
  (`results/kspc_comparator/binary_x_diagnostic_v1.json`, a diagnostic run after gate 2). The outcome noise
  is small (sd 0.3 against 2.5 for U), so the deterministic-outcome condition is barely violated either.
  **A hypothesis, not tested:** the simulations' clean proxy is noisy, $U + 0.85\varepsilon$, reliability
  about 0.58, where the authors' proxy has noise sd 0.1; undoing that much measurement error is an ill-posed
  inverse problem, and kernel ridge regularisation stops short of it. The same proxy recipe is used in E5
  to E7.
- **IHDP and ACIC.** KSPC is behind PROBE with either block.

KSPC's rows are labelled "added after the protocol was frozen" in both figures and in `EXPERIMENT_INDEX.md`.
