# E3 supplement: RHC on 20 bootstrap resamples

2026-09-23, at the author's request. `code/rhc_bootstrap.py`, design fixed in its docstring before any
resample: B = 20, rows drawn with replacement, seed 99_600_000 + b; PROBE by the path that produced E3's
number (sealed run, then the v3 rule); comparators: difference in means, AIPW adjusting for X (E3's
reproduction-gate function), AIPW adjusting for X and all ten proxies, proximal 2SLS with Cui et al.'s roles.
Results `results/rhc_bootstrap/`, figure `figures/section5/rhc-bootstrap.pdf` (a new file; the frozen
`probe-rhc.pdf` is untouched), freeze `results/FINAL_FREEZE_v1_addendum_1.json`. 20 of 20 ran; PROBE
returned on all 20.

| effect on days survived | mean over resamples | sd | 90% of resamples |
|---|---|---|---|
| PROBE, no roles | −1.20 | 0.27 | −1.61 to −0.72 |
| proximal 2SLS, Cui et al.'s roles | −1.36 | 0.28 | −1.74 to −0.92 |
| adjust for X and all ten proxies | −1.15 | 0.21 | −1.48 to −0.91 |
| adjust for X | −1.21 | 0.27 | −1.54 to −0.73 |
| difference in means | −1.49 | 0.23 | −1.88 to −1.15 |

Published by Cui et al.: standard DR −1.17 (se 0.32), proximal DR −1.66 (se 0.43).

A resample is not a Monte Carlo replicate: RHC has no known answer, and every resample is drawn from the
same patients. A resample can also put copies of one patient in different cross-fitting folds, so each
point is somewhat optimistic. The spread is descriptive.
