# Planted RHC: thirty more replicates

2026-09-22. Written after the run, against `results/rhc_planted/extension_v1.json`.

## Why

The prospective check scored replicates 11-20. Ten points could not separate PROBE (mean absolute error
0.351) from adjusting for X and all of W (0.305): the ratio was 1.15 with a 90% bootstrap interval of
0.78 to 1.56, which contains 1.

## What was fixed before running

- The identifiers 21-50, thirty replicates, set before any was drawn. Every one is reported.
- The code. All nine source files hash to the values the frozen protocol recorded for 11-20, and
  `extend_rhc_planted.py` refuses to run or score otherwise.
- The rule. The v3 noise-scaled rule through `noise_scaled.decide`, unchanged.

21-50 were not named in the frozen protocol and were run after 11-20 had been scored. They are an
extension, reported beside the designated ten and pooled with them, never in place of them.

## Result

All thirty ran, none failed, all forty returned.

| mean absolute error | 11-20 | 21-50 | 11-50 |
|---|---|---|---|
| adjust for X and all of W | 0.305 | 0.291 | **0.295** |
| PROBE | 0.351 | 0.422 | **0.404** |
| adjust for X | 0.436 | 0.519 | 0.498 |
| CEVAE | 1.173 | 1.631 | 1.517 |
| 2SLS, roles swapped | 2.037 | 2.065 | 2.058 |
| 2SLS, roles given | 4.270 | 3.849 | 3.954 |
| COCA as ATE | 53.8 | 40.2 | 43.6 |
| oracle | 0.260 | 0.235 | 0.241 |

On the forty, PROBE's error divided by that of X and all of W is **1.37, with a 90% interval of 1.18 to
1.59**. The interval excludes 1. The tie at ten replicates was imprecision; with forty, PROBE is less
accurate than adjusting for everything on this design. It remains second of seven.

## Why PROBE loses here

The loss is bias, not noise. Over the forty, PROBE's mean error is −0.367 and that of X and all of W is
−0.177; their standard deviations are 0.305 and 0.300.

The design explains it. The five proxy blocks are the real blood-test pairs of the first 24 hours, used
as recorded. No block is contaminated by the treatment-side variable and none is a collider, so including
every block harms nothing. Adjusting for X and all of W then uses all the proxy information there is.
PROBE holds one block out for the screen and summarises the rest in a rank-2 representation, so it uses
less of that information and leaves more of the confounding in place.

This is the price of PROBE's protection, measured. PROBE is built for data in which using a proxy the
wrong way does harm. When no proxy can do harm, the protection costs bias, here about 0.19 on an effect of
−1.
