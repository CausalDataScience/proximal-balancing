# PROBE on WSC: results

2026-09-22, written after the run, against `results/wsc/probe_v1.json` (code `code/wsc_probe.py`, design
fixed in its docstring and in section 4 of `memo/2026-09-22-kspc-lalonde-wsc-prd-v1.md` before it ran).

## Design

The August WSC design, imported unchanged: 1,105 people whose randomised training matched their stated
preference, 288 of them in math; outcome `mathPost`; 14 demographic covariates; 114 pre-treatment items.
New here: the items split by instrument into J = 7 blocks (Big Five 44, AMAS 9, BDI 13, GSES 6, math
pre-test 12, vocabulary pre-test 24, pre-treatment MCS 6), so 126 held-out candidates. The benchmark is
the randomised contrast over all 2,200: **0.760 (se 0.151)**. Primary reading, declared: the v3 rule.

## Result

| | whole sample (1,105) | band [0.05, 0.95] (766) |
|---|---|---|
| treated effective sample size | 37.7 of 288 | 139.6 of 273 |
| naive | 3.451 | 2.431 |
| adjust for X | 2.489 | 1.909 |
| **adjust for X and all 114 items** | **1.336** | **1.229** |
| 2SLS over all 42 role assignments | median 1.370, range −6.2 to 18.9 | median 1.202, range −3.2 to 28.3 |
| COCA as ATE, over the 7 blocks | median −1.48 | median 0.54 |
| **PROBE, v3 (primary)** | **1.865**, kept 45 of 126, 95% 1.16 to 2.60 | **2.248**, kept 38 of 126, 95% 1.28 to 3.19 |
| PROBE, significance rule | 1.968, kept all 126 | 2.305, kept all 126 |

Error against 0.760 on the band: X and all items +0.47, adjust for X +1.15, PROBE +1.49. PROBE's interval
excludes the benchmark in both readings. **WSC is a data set on which PROBE does not work.** For
reference, the August cross-mask run got 1.302 on the whole sample.

## Why, as far as these numbers show

1. **No block is harmful to include.** Every item is measured before treatment and none is moved by the
   treatment-side variable. Adjusting for all 114 items is safe, and it wins, as on planted RHC and
   LaLonde.
2. **Rank 2 may be too small.** PROBE summarises the kept blocks in two dimensions. Self-selection into
   math training plausibly depends on several distinct traits at once: math skill, math anxiety, math
   self-concept, personality. This is a hypothesis; it was not tested, and testing it by varying the rank
   against the known benchmark would be tuning.
3. **Few rows per covariate.** About 14 on the band, below the range where PROBE has won.

The screen did discriminate: the v3 rule kept 45 and 38 of 126 candidates. The significance rule found
no imbalance anywhere and kept all 126, which says its power is low at this sample size.

## What the 42 role assignments show

Proximal 2SLS moves from −6.2 to +18.9 on the whole sample depending on which block is called the
treatment-side proxy and which the outcome-side one. On data with seven plausible proxies and no agreed
roles, the choice of roles is not a detail. That is the problem PROBE addresses, and it is visible here
even though PROBE's own estimate is off.

## Addendum, 22:00 CDT: the exact answer, and LaLonde dropped

`mathGrp` was randomised over everyone independently of the stated preference, so the randomised effect
is available within each preference group: 0.795 (se 0.278) among those preferring math and 0.785
(se 0.166) among those preferring vocabulary. Weighting by the 288 and 817 in the observational sample
gives the answer for exactly that sample: **0.787 (se 0.142)**, `results/wsc/exact_benchmark_v1.json`.
The whole-sample reading is scored against it from now on; the band has no exact answer.

| whole sample, against 0.787 | estimate | error |
|---|---|---|
| X and all 114 items | 1.336 | **+0.55** |
| PROBE, v3 | 1.865 | +1.08 |
| adjust for X | 2.489 | +1.70 |

LaLonde was dropped from the figures by the author's decision: its benchmark is the experimental effect
on the treated men, not the average effect over the propensity band that PROBE estimates, so it is not
the answer to PROBE's question. Its results stay in `results/two_proxy/`.

Zotero holds none of the within-study comparison literature (checked by title and by the authors
Shadish, Steiner, Keller and Wong). The Keller et al. (2025) citation is from `manifest.json`.
