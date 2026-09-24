# PRD: kernel single proxy control, more LaLonde samples, and the WSC benchmark (v1)

2026-09-22 20:05 CDT. Draft for the author's approval. Nothing in workstreams A to C runs before
section 0 is answered. ICLR deadline: 2026-09-26 06:59 CDT.

## 0. Decisions the author makes first

| | decision | recommendation |
|---|---|---|
| D1 | Download the authors' code for kernel single proxy control, `github.com/liyuan9988/KernelSingleProxy`, pinned to one commit, sha256 recorded | yes |
| D2 | Download the other LaLonde comparison groups: CPS-2, CPS-3, PSID-2, PSID-3 from the Dehejia data page (`users.nber.org/~rdehejia/data/`), and the NSW AFDC women sample `NSW_AFDC_CS.dta` from the Imbens and Xu repository at the commit already in `manifest.json` | yes |
| D3 | Add the WSC within-study comparison, already in the workspace | yes |
| D4 | Add the Cui et al. (2023) proximal simulation and the Demand design | only if time remains after A to C |

Each approved download goes to `materials/real_world_data/` with its source, revision and sha256 in
`manifest.json`, the same way Zika was registered.

## 1. Why

Three gaps a reviewer will press.

1. **The only single-proxy comparator is our linear COCA**, which breaks down on every real data set
   (ACIC 34.6, IHDP 6.0). A reviewer who knows Xu and Gretton will call it a straw man.
2. **LaLonde rests on two real samples.** CPS-1 and PSID-1 give PROBE errors of +185 and −3,149 dollars
   against the randomised benchmark of +1,794.
3. **No real data set with a randomised answer has more than two proxy blocks.** At two blocks the v3
   rule degenerates and the aggregation reduces to an average, so the real-data evidence never
   exercises the part of PROBE that the simulations show working.

## 2. Workstream A: kernel single proxy control (KSPC)

**Method.** Xu and Gretton, *Kernel Single Proxy Control for Deterministic Confounding*, arXiv
2308.04585, v4 of 18 March 2025. One proxy suffices when the outcome is a deterministic function of the
treatment and the confounder. Two estimators: two-stage regression and maximum moment restriction. The
target is the average causal effect, so for a binary treatment it is the ATE directly and needs no
conversion. The wrapper must confirm from the code which quantity is returned before it is used.

**Its assumption fails on our data.** Every data set here has outcome noise, so every KSPC run is
outside its identifying condition. The caption says so, as it does for COCA.

**Which proxy it gets.** KSPC takes one proxy, so someone must choose it, and it is coloured as a
role-requiring method. It gets the same blocks COCA gets.

| data set | clean block | second row |
|---|---|---|
| SCM-1 to SCM-4 | W1 | W4, the contaminated block |
| Twins, IHDP, ACIC | W1 | W4 |
| LaLonde, every sample | W1 = 1974 earnings | W2 = 1975 earnings |
| WSC | decided in section 4 before running | |

**Covariates.** The paper's main setting has no X. If the code accepts covariates, standardised X is
passed. If it does not, the row is reported as "KSPC without X" and the caption says so. X is not
appended to the proxy.

**Hyperparameters.** The authors' defaults and their own selection procedure. Nothing is tuned against
our known effects.

**Computation.** The exact kernel solve is cubic in n. Every data set larger than 3,000 rows is
subsampled to 3,000 rows with seed `99_200_000 + replicate`, fixed here. Two workers, two threads each,
`nice 10`. Estimated 3 to 6 CPU hours. If the first SCM level takes more than 90 minutes, SCM seeds drop
from 20 to the first 10, and that fallback is fixed here too.

**Replicates, fixed here.** SCM-1 to SCM-4: the 20 confirmation seeds already scored. Twins 11 to 20,
IHDP 12 to 21, ACIC realizations 2 to 10, every LaLonde sample, WSC.

**Sanity gate before our data.** The wrapper reproduces one of the authors' own synthetic examples to
within the error they report. If it cannot, KSPC is not run on our data and the memo says why.

**Output.** `results/kspc/<dataset>_v1.json`, write-once. One new row in the estimates figure and the
real-data figures, in the role-requiring colour.

## 3. Workstream B: more LaLonde samples

**Done.** CPS-1: 697 in the band, PROBE error +185. PSID-1: 438 in the band, PROBE error −3,149, with a
95% interval of −4,406 to +1,726 dollars around the estimate, so the benchmark sits just outside it.

**Added after D2.** CPS-2, CPS-3, PSID-2, PSID-3, the nested comparison groups Dehejia and Wahba report,
and the NSW AFDC women sample, which has its own randomised benchmark.

**Rules, unchanged.** The band [0.05, 0.95] declared in `two_proxy_data.py`, the same dependent-column
rule, a seed per sample fixed in the data module before it runs.

**What is not claimed.** CPS-3 lies inside CPS-2 inside CPS-1, and likewise for PSID, so the samples
are not independent. Each is reported on its own. No interval is formed across them. A bootstrap may be
shown as spread within one sample, never as extra samples.

## 4. Workstream C: the WSC benchmark

**Data.** Keller et al. (2025), a four-arm within-study comparison, already in
`materials/real_world_data/wscdata/`: 2,200 people, randomised and self-selected arms of the same math
versus vocabulary training. The randomised arm gives the benchmark for the math post-test,
0.760 (se 0.151).

**Why it fits PROBE.** The pre-tests come as item-level files. The August design (`load_data` in
`code/wsc_population_ate_pilot.py`) used seven of them as proxy blocks: Big Five, AMAS, BDI, GSES, math
pre-test, vocabulary pre-test, and the pre-treatment MCS items; only the post-treatment MCS items are
excluded. That is 114 items. So J = 7, with 126 candidates. The screen and the aggregation have real work
to do, unlike at J = 2. (An earlier edit of this section said six blocks; that was wrong and is corrected.)

**Design, fixed before running.** Observational sample: the self-selected arm. Treatment: choosing math
training. Outcome: the math post-test. X: the demographic columns of `df2200_cleaned.csv`. Blocks: the
seven item-level blocks of the August design, one block each. Estimand: the ATE in the self-selected arm, against the
randomised-arm benchmark, stated as a transport assumption in the caption. Band: [0.05, 0.95].

**Risk, stated now.** About 1,100 self-selected people and some 14 covariates give roughly 20 rows per
fold per covariate. That is between IHDP (8) and LaLonde CPS-1 (35), where PROBE did not win. WSC is a
test, not a showcase.

**Prior run on the same data.** On 2026-08-23 a different method, cross-mask balancing, was
pre-registered on this data set and failed. Its data handling is reused. Its results are not.

## 5. Honesty rules

- Seeds, replicate identifiers, sample definitions, the subsample size and hyperparameters are fixed in
  this document before any run.
- Every result is written once and recorded in a memo, favourable or not.
- Every adaptation carries its label, for example "KSPC without X" or "outside its identifying
  condition".
- **Planted RHC.** Its results stay in `results/rhc_planted/` and in the memos. The paper does not report
  it, by the author's decision of 2026-09-22. If the paper reports the frozen prospective protocol, it must
  say that one of the three designated data sets is not shown and where it is recorded. A reader who
  compares the protocol with the paper would otherwise find the gap.

## 6. Order of work

| step | work | new runs | estimate |
|---|---|---|---|
| 1 | Screen accuracy figure: true and false selection rates per data set | none | 1 h |
| 2 | KSPC wrapper and its sanity gate | none on our data | 3 h |
| 3 | KSPC on every data set in section 2 | yes | 3 to 6 CPU h |
| 4 | LaLonde samples after D2 | yes | 1 h |
| 5 | WSC data module and run | yes | 4 h |
| 6 | Figures, captions, appendix text | none | 3 h |
| 7 | D4, only if time remains | yes | open |

## 7. Acceptance

- KSPC passes its sanity gate, or the memo records why it was not run.
- Every data set and replicate named here has a result file or a recorded failure reason.
- Every figure shows KSPC in the role-requiring colour and carries the deterministic-outcome caveat.
- All tests pass, including new ones for each wrapper and data module.

## 8. Status, 20:50 CDT

**D2 done.** The author asked for as many LaLonde samples as IHDP and ACIC have. Five files were
downloaded from the pinned Imbens and Xu commit and registered. LaLonde now has eight real samples: six
comparison groups for the men and two for the AFDC women. That is every real comparison group these data
provide; more points could only come from resampling. Results are in section 9 of the two-proxy memo.
PROBE has the smallest median error (360 dollars) and the third mean error (811); 2SLS with W2 on the
treatment side has the smallest mean (620).

**WSC, established.** The 2026-08-23 run used the predecessor method, cross-mask balancing, and failed
its pre-registration. PROBE has never been run on WSC. The design is fixed by that run's code: 1,105
people in the self-selected arm, 288 of them treated; outcome `mathPost`; 14 demographic covariates; seven
item-level proxy blocks (Big Five, AMAS, BDI, GSES, math pre-test, vocabulary pre-test, pre-treatment MCS).
J = 7 gives 126 candidates. The benchmark is 0.760 (se 0.151).

**Demand design, established.** From the authors' generator, `src/data/ate/demand_pv.py` in
`github.com/liyuan9988/DeepFeatureProxyVariable`:

- hidden demand $D \sim \mathrm{Unif}[0, 10]$, and $\psi(t) = 2\{(t-5)^4/600 + e^{-4(t-5)^2} + t/10 - 2\}$
- cost proxies $C_1 = 2\sin(2\pi D/10) + \varepsilon_1$ and $C_2 = 2\cos(2\pi D/10) + \varepsilon_2$
- price $P = 35 + (C_1 + 3)\,\psi(D) + C_2 + \varepsilon_P$, continuous
- page views $V = 7\psi(D) + 45 + \varepsilon_V$
- sales $Y = \min\{e^{(V-P)/10}, 5\}\,P - 5\psi(D) + \varepsilon_Y$
- all noise $N(0, 1)$; no covariates; truth is the average sales at each of ten prices from 10 to 30

The treatment is continuous, so PROBE's binary-treatment score does not apply as published. The honest
adaptation is our DGP on their design: a binary fare choice driven by the same price index, the potential
outcomes at two fixed fares, and the truth by Monte Carlo. It would be labelled that way. It has two
proxy blocks, the costs and the views. Holding out the costs is invalid, because costs move the price
directly; holding out the views is valid. So the screen has a real decision to make even at J = 2. But
no block is harmful to adjust for, the situation in which adjusting for everything beat PROBE on the
planted RHC design, so it may not favour PROBE. Its value is that it is the Gretton group's own benchmark
for KSPC.

## 9. Superseded, 23:30 CDT

Workstream A (KSPC as a comparator) is superseded by `memo/2026-09-22-kspc-comparator-prd-v2.md`. Its rule
"KSPC without X" rested on the authors' code, which has no covariate argument; the method handles observed
covariates in Appendix A of the paper, and v2 implements that. Workstreams B and C are closed: LaLonde and
WSC were dropped by the author, and the KSPC benchmark (E4) is done and frozen.
