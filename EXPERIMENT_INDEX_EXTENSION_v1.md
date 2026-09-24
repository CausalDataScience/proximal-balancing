# Experiment index, extension v1: added replicates

Companion to the frozen `EXPERIMENT_INDEX.md`, which is not edited. Plan and decisions:
`memo/2026-09-23-replicate-extension-prd-v2.md` (amendments 1 to 3). Protocol, frozen before any added
replicate ran: `results/extension_v1/protocol_v1.json`. Summary: `results/extension_v1/summary_v1.json`.

## How the added replicates were produced

- Every added replicate ran through the frozen runner of its experiment, unchanged, with identifiers and
  seeds fixed in the protocol. The sealed replicates are untouched; tables show them, the added ones and
  both together.
- The added replicates ran on NCSA DeltaAI (Arm Neoverse V2 cores; the SCM-4 reader and pixel CNN on an
  H100), the sealed ones on an Apple M4. Rerunning sealed data sets on DeltaAI gave the same data and
  PROBE outputs that differ from the Mac values by 0.0002 to 0.009 (E3: 0.028), each less than half the
  spread of that experiment's sealed outputs (`results/extension_v1/gate_*.json`). The gate first fixed,
  equality to $10^{-6}$, failed; the author replaced it after seeing that (amendment 3).
- RHC's proximal 2SLS is reported only with Cui et al.'s specification (amendment 1); Cui et al.'s
  standard DR is not drawn (amendment 2).
- Seeds of the sealed E1 set: SCM-1 replicates 10 to 19 share seeds with SCM-2 replicates 0 to 9, and
  SCM-2 10 to 19 with SCM-3 0 to 9, so those pairs share random draws; comparisons within a level are
  unaffected. The added seeds have no such overlap.
- IHDP has 88 data sets, not the planned 89. The file indexes its realizations 0 to 99; the protocol
  named 22 to 100, and 100 does not exist (that task failed with an index error). Realization 0, never
  used, was not in the protocol and was not run.
- SCM-4's pixel CNN finished within its cap on all 80 added data sets. Two sealed data sets had passed
  the cap on the Mac and were filled by the amended summary, as before.
- Charge on DeltaAI, GPU-hours: smoke test 1.01 (job 3198134), recheck 0.26 (3198540), first main run 1.36 (3198612, cancelled after the GPU-memory failures), main run 2.71 (3198776); 5.33 in all, against a ceiling of 10.

Figures: `figures/section5/extension-estimates.pdf`, `extension-realdata.pdf`,
`extension-kspc-benchmark.pdf`, `extension-rhc-bootstrap.pdf` (source `code/make_extension_figures.py`).
† KSPC with covariates, our implementation of its Appendix A, added after the protocol was frozen.

## E1, SCM-1

| MAE ± Monte Carlo SE | sealed (20) | added (80) | pooled (100) |
|---|---|---|---|
| oracle (adjusts for X and U) | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.000 |
| **PROBE** | 0.020 ± 0.003 | 0.018 ± 0.002 | 0.019 ± 0.001 |
| 2SLS, roles given | 0.031 ± 0.005 | 0.024 ± 0.002 | 0.025 ± 0.002 |
| COCA, clean block | 0.033 ± 0.004 | 0.029 ± 0.003 | 0.030 ± 0.002 |
| CEVAE | 0.030 ± 0.005 | 0.041 ± 0.003 | 0.039 ± 0.003 |
| adjust for X and all of W | 0.271 ± 0.004 | 0.273 ± 0.002 | 0.273 ± 0.002 |
| KSPC with X, clean block† | 0.875 ± 0.005 | 0.872 ± 0.004 | 0.873 ± 0.004 |
| adjust for X | 1.442 ± 0.005 | 1.437 ± 0.003 | 1.438 ± 0.003 |
| KSPC with X, contaminated block† | 1.881 ± 0.009 | 1.871 ± 0.006 | 1.873 ± 0.005 |
| 2SLS, roles swapped | 5.186 ± 0.020 | 5.188 ± 0.010 | 5.188 ± 0.009 |
| COCA, contaminated block | 5.258 ± 0.022 | 5.280 ± 0.010 | 5.275 ± 0.009 |

## E1, SCM-2

| MAE ± Monte Carlo SE | sealed (20) | added (80) | pooled (100) |
|---|---|---|---|
| oracle (adjusts for X and U) | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 |
| **PROBE** | 0.017 ± 0.003 | 0.019 ± 0.002 | 0.018 ± 0.001 |
| COCA, clean block | 0.035 ± 0.004 | 0.030 ± 0.003 | 0.031 ± 0.002 |
| 2SLS, roles given | 0.159 ± 0.032 | 0.176 ± 0.080 | 0.173 ± 0.064 |
| adjust for X and all of W | 0.277 ± 0.004 | 0.277 ± 0.002 | 0.277 ± 0.002 |
| CEVAE | 0.280 ± 0.008 | 0.283 ± 0.004 | 0.283 ± 0.004 |
| KSPC with X, clean block† | 0.854 ± 0.013 | 0.855 ± 0.005 | 0.855 ± 0.005 |
| adjust for X | 1.430 ± 0.004 | 1.438 ± 0.003 | 1.436 ± 0.002 |
| KSPC with X, contaminated block† | 1.656 ± 0.009 | 1.660 ± 0.005 | 1.659 ± 0.004 |
| COCA, contaminated block | 5.228 ± 0.019 | 5.259 ± 0.010 | 5.253 ± 0.009 |
| 2SLS, roles swapped | 5.312 ± 0.219 | 5.620 ± 0.374 | 5.559 ± 0.302 |

## E1, SCM-3

| MAE ± Monte Carlo SE | sealed (20) | added (80) | pooled (100) |
|---|---|---|---|
| oracle (adjusts for X and U) | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 |
| **PROBE** | 0.014 ± 0.003 | 0.019 ± 0.002 | 0.018 ± 0.001 |
| COCA, clean block | 0.036 ± 0.006 | 0.033 ± 0.002 | 0.033 ± 0.002 |
| 2SLS, roles given | 0.120 ± 0.040 | 0.163 ± 0.037 | 0.155 ± 0.031 |
| CEVAE | 0.409 ± 0.007 | 0.418 ± 0.005 | 0.416 ± 0.005 |
| adjust for X and all of W | 0.446 ± 0.004 | 0.444 ± 0.002 | 0.445 ± 0.002 |
| KSPC with X, clean block† | 0.813 ± 0.012 | 0.840 ± 0.007 | 0.835 ± 0.006 |
| adjust for X | 1.430 ± 0.005 | 1.444 ± 0.003 | 1.441 ± 0.003 |
| KSPC with X, contaminated block† | 1.796 ± 0.027 | 1.812 ± 0.011 | 1.809 ± 0.011 |
| 2SLS, roles swapped | 5.225 ± 0.133 | 5.218 ± 0.067 | 5.220 ± 0.059 |
| COCA, contaminated block | 5.249 ± 0.020 | 5.264 ± 0.012 | 5.261 ± 0.010 |

## E2, SCM-4 (Shapes3D images)

| MAE ± Monte Carlo SE | sealed (20) | added (80) | pooled (100) |
|---|---|---|---|
| oracle (adjusts for X and U) | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.000 |
| **PROBE** | 0.018 ± 0.004 | 0.019 ± 0.002 | 0.019 ± 0.002 |
| 2SLS, roles given | 0.033 ± 0.004 | 0.030 ± 0.002 | 0.030 ± 0.002 |
| COCA, clean block | 0.038 ± 0.006 | 0.037 ± 0.003 | 0.037 ± 0.003 |
| CEVAE | 0.047 ± 0.010 | 0.052 ± 0.004 | 0.051 ± 0.004 |
| pixel CNN | 0.281 ± 0.005 | 0.287 ± 0.003 | 0.286 ± 0.002 |
| adjust for X and all of W | 0.306 ± 0.004 | 0.311 ± 0.002 | 0.310 ± 0.002 |
| KSPC with X, clean block† | 0.870 ± 0.009 | 0.876 ± 0.004 | 0.875 ± 0.003 |
| adjust for X | 1.433 ± 0.006 | 1.435 ± 0.003 | 1.435 ± 0.003 |
| KSPC with X, contaminated block† | 1.807 ± 0.012 | 1.818 ± 0.006 | 1.816 ± 0.005 |
| 2SLS, roles swapped | 5.136 ± 0.025 | 5.178 ± 0.012 | 5.170 ± 0.011 |
| COCA, contaminated block | 5.216 ± 0.026 | 5.244 ± 0.011 | 5.238 ± 0.010 |

## E5, Twins

| MAE ± Monte Carlo SE | sealed (10) | added (90) | pooled (100) |
|---|---|---|---|
| oracle (adjusts for X and U) | 0.0074 ± 0.0018 | 0.0059 ± 0.0005 | 0.0061 ± 0.0005 |
| **PROBE** | 0.0077 ± 0.0016 | 0.0073 ± 0.0006 | 0.0073 ± 0.0006 |
| KSPC with X, clean block† | 0.0099 ± 0.0018 | 0.0085 ± 0.0006 | 0.0086 ± 0.0006 |
| 2SLS, roles given | 0.0156 ± 0.0024 | 0.0138 ± 0.0008 | 0.0139 ± 0.0007 |
| adjust for X and all of W | 0.0207 ± 0.0033 | 0.0219 ± 0.0008 | 0.0218 ± 0.0008 |
| CEVAE | 0.0894 ± 0.0142 | 0.0717 ± 0.0042 | 0.0735 ± 0.0040 |
| adjust for X | 0.0749 ± 0.0021 | 0.0759 ± 0.0006 | 0.0758 ± 0.0006 |
| KSPC with X, contaminated block† | 0.0815 ± 0.0032 | 0.0800 ± 0.0010 | 0.0801 ± 0.0009 |
| difference in means | 0.0940 ± 0.0015 | 0.0931 ± 0.0006 | 0.0932 ± 0.0005 |
| 2SLS, roles swapped | 0.2453 ± 0.0030 | 0.2475 ± 0.0009 | 0.2473 ± 0.0009 |
| COCA as an ATE | 0.7318 ± 0.0159 | 0.7267 ± 0.0047 | 0.7272 ± 0.0045 |

## E6, IHDP

| MAE ± Monte Carlo SE | sealed (10) | added (78) | pooled (88) |
|---|---|---|---|
| 2SLS, roles given | 0.2328 ± 0.1515 | 0.2248 ± 0.0386 | 0.2257 ± 0.0379 |
| **PROBE** | 0.3054 ± 0.1197 | 0.4220 ± 0.0885 | 0.4088 ± 0.0796 |
| adjust for X and all of W | 0.3927 ± 0.1583 | 0.4615 ± 0.0840 | 0.4536 ± 0.0764 |
| oracle (adjusts for X and U) | 0.4868 ± 0.2168 | 0.4758 ± 0.0985 | 0.4770 ± 0.0904 |
| CEVAE | 0.1828 ± 0.0463 | 0.6039 ± 0.1358 | 0.5561 ± 0.1212 |
| adjust for X | 0.7221 ± 0.3840 | 0.6303 ± 0.1436 | 0.6407 ± 0.1339 |
| KSPC with X, clean block† | 0.6214 ± 0.1604 | 0.6852 ± 0.1185 | 0.6779 ± 0.1064 |
| difference in means | 0.7026 ± 0.4404 | 0.7973 ± 0.1538 | 0.7865 ± 0.1444 |
| KSPC with X, contaminated block† | 1.2499 ± 0.6210 | 1.3720 ± 0.2545 | 1.3581 ± 0.2353 |
| 2SLS, roles swapped | 1.5856 ± 0.7368 | 1.7401 ± 0.4317 | 1.7226 ± 0.3907 |
| COCA as an ATE | 5.9838 ± 2.1562 | 27.3473 ± 11.1400 | 24.9196 ± 9.8963 |

## E8, KSPC benchmark

| MAE ± Monte Carlo SE | sealed (20) | added (80) | pooled (100) |
|---|---|---|---|
| oracle (adjusts for X and U) | 0.004 ± 0.001 | 0.003 ± 0.000 | 0.003 ± 0.000 |
| KSPC spmmr_W1 | 0.007 ± 0.001 | 0.008 ± 0.001 | 0.008 ± 0.001 |
| COCA, given W1 | 0.017 ± 0.002 | 0.013 ± 0.001 | 0.014 ± 0.001 |
| KSPC skpv_W1 | 0.023 ± 0.001 | 0.021 ± 0.001 | 0.021 ± 0.001 |
| 2SLS, W2 treats | 0.025 ± 0.002 | 0.029 ± 0.001 | 0.028 ± 0.001 |
| **PROBE** (reported rule) | 0.054 ± 0.003 | 0.059 ± 0.001 | 0.058 ± 0.001 |
| PROBE, v3 rule | 0.054 ± 0.003 | 0.059 ± 0.001 | 0.058 ± 0.001 |
| adjust for X and all of W | 0.104 ± 0.006 | 0.113 ± 0.003 | 0.111 ± 0.002 |
| 2SLS, W1 treats | 0.100 ± 0.021 | 0.133 ± 0.011 | 0.126 ± 0.010 |
| COCA, given the image | 0.140 ± 0.028 | 0.216 ± 0.029 | 0.201 ± 0.024 |
| KSPC skpv_W2 | 0.273 ± 0.004 | 0.279 ± 0.003 | 0.278 ± 0.002 |
| KSPC spmmr_W2 | 0.555 ± 0.005 | 0.561 ± 0.003 | 0.559 ± 0.003 |
| difference in means | 0.617 ± 0.006 | 0.616 ± 0.003 | 0.616 ± 0.002 |
| adjust for X | 0.617 ± 0.006 | 0.616 ± 0.003 | 0.616 ± 0.003 |

## E3, observed RHC, bootstrap resamples

| effect on days survived | sealed resamples | added resamples | all |
|---|---|---|---|
| **PROBE**, v3 rule | -1.20 (sd 0.27; 90% -1.61 to -0.72) | -1.31 (sd 0.33; 90% -1.80 to -0.82) | -1.29 (sd 0.32; 90% -1.72 to -0.81) |
| difference in means | -1.49 (sd 0.23; 90% -1.88 to -1.15) | -1.56 (sd 0.33; 90% -2.07 to -1.02) | -1.55 (sd 0.31; 90% -2.03 to -1.03) |
| adjust for X | -1.21 (sd 0.27; 90% -1.54 to -0.73) | -1.33 (sd 0.34; 90% -1.86 to -0.84) | -1.31 (sd 0.33; 90% -1.78 to -0.82) |
| adjust for X and all ten proxies | -1.15 (sd 0.21; 90% -1.48 to -0.91) | -1.24 (sd 0.31; 90% -1.71 to -0.78) | -1.22 (sd 0.30; 90% -1.71 to -0.78) |
| Proximal 2SLS, Cui et al.'s specification | -1.55 (sd 0.40; 90% -2.43 to -1.06) | -1.46 (sd 3.18; 90% -2.95 to -0.99) | -1.48 (sd 2.85; 90% -2.83 to -1.00) |

