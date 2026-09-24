# Zika and LaLonde with two real proxies

2026-09-22. Written after the runs, against `results/two_proxy/zika_v1.json`,
`results/two_proxy/lalonde_v1.json` and `results/two_proxy/uncertainty_v1.json`.

## 1. Why these two

Every other real data set in the paper is real in part. Twins carries real outcomes but a proxy we built;
the planted RHC design carries real proxies but an effect we chose; IHDP and ACIC are synthetic outcomes
over real covariates. Zika and LaLonde are real all the way through. The treatment, the covariates, the two
proxies and the outcome are all columns of the published file, and nothing is hidden and reintroduced.

Both have exactly two proxies, and in both the literature disagrees about which of the two may act on the
treatment and which may act on the outcome. That is the question PROBE is built not to need answered.

| | Zika | LaLonde |
|---|---|---|
| units | 673 Brazilian municipalities | 185 NSW men and 15,992 CPS-SSA-1 controls |
| treated | 185 heavily exposed to the 2015-16 outbreak | 185 assigned to National Supported Work |
| `W1` | 2013 birth rate, before the outbreak | 1974 earnings, before the programme |
| `W2` | 2014 birth rate, before the outbreak | 1975 earnings, before the programme |
| `Y` | 2016 birth rate | 1978 earnings |
| `X` | three published municipality covariates | six demographic covariates |
| answer | none | the randomised arm: **+$1,794 (se 671)** on the treated |

## 2. What had to change, and what did not

`family_v2_probe.run_dataset` loops over five blocks, so it cannot be pointed at a data set with two. The
new `code/probe_j2.py` supplies the loop; every piece that computes a number is the sealed code, imported
unchanged: `learn_representation`, `screen`, `aipw`, `common_radius`, `aggregate`. The preprocessing class
is the sealed `Prep` rule rewritten for J blocks, line for line. Nothing sealed was edited.

Two blocks give two candidates rather than thirty, so the multiplicity correction follows the candidate
count: `z = 1.960` rather than `2.935`.

**The v3 rule is degenerate at two candidates.** Its threshold is the median over candidates of `z SE`,
and the median of two numbers is their average, so it can keep the more precise candidate or neither, never
both. It is still reported, labelled, and it did return nothing twice. The reading treated as primary is a
significance rule with no free threshold at all: keep a candidate when `gap - z SE <= 0`, that is when
there is no evidence of imbalance.

**The aggregation has almost nothing to do at two candidates either.** When both are kept and their
estimates fall inside one linking radius, the largest agreeing group is the pair, and the median of two
numbers is their mean. On LaLonde's band PROBE's +1,979 is exactly the average of its two candidates,
+2,291 and +1,667. The linking and largest-component machinery earns its keep at thirty candidates; here it
degenerates to an average, and the paper should not claim otherwise. What is still doing real work at two
candidates is the screen, which is what decides whether a candidate is used at all.

## 3. The overlap band, declared before any estimate

`two_proxy_data.BAND = (0.05, 0.95)`, fixed in the source with its reason, before a single estimate was
formed. PROBE's score is the one for the average effect, whose weights are `1/e` and `1/(1-e)`, so the band
bounds every weight by 20. It uses the treatment and the covariates. It never touches the outcome.

It had to exist. On the published samples the treated arm has all but vanished:

| effective sample size, treated arm | of | |
|---|---|---|
| Zika, whole sample | **1.0** | 185 |
| Zika, on the band | 79.5 | 140 |
| LaLonde, whole sample | **16.1** | 185 |
| LaLonde, on the band | 101.3 | 155 |

One Brazilian municipality carries the entire treated arm of the untrimmed Zika sample. No estimator that
reweights can work there, and the sections below show that none does.

The band changes the population. The reading on it is the average effect over the units in the band, not
over everyone and not over the treated. Restricting LaLonde to the band also makes `black` and `hispanic`
exactly complementary, so those two and the intercept become linearly dependent; `hispanic` is dropped once,
for every estimator alike, and recorded. Zika drops nothing.

## 4. LaLonde, against a real answer

Benchmark **+$1,794 (se 671)**.

| | whole sample | on the band |
|---|---|---|
| naive difference | −8,498 | +465 |
| adjust for X | −6,472 | +1,344 |
| adjust for X and both proxies | −6,281 | +1,639 |
| proximal 2SLS, W1 treats | +1,185 | +1,917 |
| proximal 2SLS, W2 treats | +829 | +1,942 |
| COCA on the treated, W1 / W2 | +5,190 / +5,967 | +9,335 / +10,505 |
| COCA as an average effect, W1 / W2 | +376,284 / +531,299 | −17,675 / −323,284 |
| **PROBE, significance reading** | **−5,025** | **+1,979** |
| PROBE, v3 reading | nothing | +1,667 |

PROBE's two candidates on the band are +2,291 and +1,667; it keeps both and reports their median.
Resampling the aggregation gives sd 932 and a 95% interval of (+125, +3,789). The benchmark sits inside it.

**What this does and does not show.** PROBE lands $185 from an answer produced by randomisation, without
being told which pre-programme year is the treatment proxy and which is the outcome proxy. It does not beat
the simpler adjustments: at sd 932, PROBE's +1,979, the everything-in +1,639 and the benchmark +1,794 are
not distinguishable. What separates the methods here is not accuracy, it is the two failures. COCA is off by
thousands in both readings and by hundreds of thousands once converted to an average effect. And on the
whole sample every reweighting method, PROBE included, is confidently wrong: PROBE reports −5,025 with a
resampled sd of 105, which is a tight interval around a badly wrong number.

**Estimand.** The benchmark is the effect on the treated. PROBE returns the average effect over the band.
These agree only if the effect is constant over that population. That is an assumption and it is not tested
here.

## 5. Zika, with no answer

Park, Richardson and Tchetgen Tchetgen report **−2.180 (se 0.342)** on the treated using both proxies. That
is a number to sit beside, not a truth to be scored against.

| | whole sample | on the band |
|---|---|---|
| naive difference | +3.384 | +0.966 |
| adjust for X | +3.217 | +1.409 |
| adjust for X and both proxies | +0.215 | −0.432 |
| proximal 2SLS, W1 treats / W2 treats | −0.942 / −1.185 | −0.900 / −2.128 |
| COCA as an average effect, W1 / W2 | −2.990 / −2.057 | −4.906 / −2.734 |
| **PROBE, significance reading** | **+0.314** | **−0.269** |
| PROBE, v3 reading | +0.233 | nothing |

The naive +3.384 reproduces the published crude contrast exactly, which is the check that the data were
read correctly.

On the band PROBE agrees in sign with the proximal literature and is much smaller in magnitude: −0.269 with
sd 0.34 and a 95% interval of (−0.94, +0.39), which contains zero and does not contain −2.180. On the whole
sample PROBE gives +0.314, the wrong sign, and the effective treated sample size of 1.0 says why.

## 6. A limitation this run exposed

The screen's own overlap check reported `True` on both whole samples, where the propensity effective sample
size is 1.0 and 16.1. The check is about the critic's fitted probabilities lying inside [0.05, 0.95] on the
learned representation, and it is silent about the propensity the AIPW step actually divides by. A reader
who takes `overlap: True` as a licence to trust the estimate would be misled here. The fix is not internal
to the screen: the propensity report belongs beside it. Both runs now record one.

## 7. Files

New: `code/probe_j2.py`, `code/two_proxy_data.py`, `code/run_two_proxy.py`,
`code/two_proxy_uncertainty.py`, `code/test_two_proxy.py` (14 checks, all pass).
Written: `results/two_proxy/zika_v1.json`, `results/two_proxy/lalonde_v1.json`,
`results/two_proxy/uncertainty_v1.json`. Data already in the workspace under
`materials/real_world_data/zika/` and `materials/real_world_data/lalonde/`, both with recorded provenance.

## 8. Addendum, 20:00 CDT: LaLonde against the PSID-1 controls

The same 185 NSW men against the 2,490 PSID-1 controls, the second comparison group that LaLonde and
Dehejia and Wahba report. Same columns, same benchmark of +1,794 dollars, same declared band
[0.05, 0.95]. `results/two_proxy/lalonde_psid_v1.json`, `results/two_proxy/uncertainty_lalonde_psid_v1.json`.

The whole sample fails overlap as badly as CPS-1: the treated effective sample size is 4.1 of 185. The
band keeps 438 people, 177 of them treated, and raises it to 50.4.

| on the band | estimate | error against +1,794 |
|---|---|---|
| adjust for X | −886 | −2,680 |
| adjust for X and both proxies | −474 | −2,268 |
| 2SLS, W1 treats | +7,454 | +5,660 |
| 2SLS, W2 treats | +1,032 | −762 |
| **PROBE** | **−1,355** | **−3,149** |

PROBE's two candidates disagree by about 2,900 dollars, −2,798 and +89, and are still linked, so the
median is again their average. The resampled 95% interval is −4,406 to +1,726, and the benchmark lies
just outside it. On PSID-1 every method except 2SLS with W2 on the treatment side misses by more than
2,000 dollars, and PROBE misses by the most among the methods that need no role assignment.

Across the two samples the mean absolute error is 1,667 for PROBE and 1,212 for adjusting for X and both
proxies. LaLonde is a data set where PROBE does not win.

## 9. Addendum, 20:40 CDT: eight real LaLonde samples

Downloaded from the Imbens and Xu repository at the commit already pinned in `manifest.json`, sha256
recorded: CPS-2, CPS-3, PSID-2, PSID-3, and Calonico and Smith's NSW AFDC women. The four subsets pair
with the same 185 men and the benchmark of +1,794. The women form two more samples, the experimental
treated women against PSID-1 and PSID-2 women, scored against their own randomised benchmark of +801
(se 330). Fixed before running: 1979 earnings as the women's outcome, complete cases on 1974, 1975 and
1979 earnings, and no `afdc75` covariate, because every experimental woman received AFDC. The samples are
nested (CPS-3 in CPS-2 in CPS-1, PSID-3 in PSID-2 in PSID-1), so they are not independent.

| error against each sample's benchmark | CPS-1 | CPS-2 | CPS-3 | PSID-1 | PSID-2 | PSID-3 | AFDC, PSID-1 | AFDC, PSID-2 |
|---|---|---|---|---|---|---|---|---|
| adjust for X | −450 | −1,510 | −233 | −2,680 | −1,946 | −1,395 | −857 | −555 |
| X and both proxies | −155 | −1,214 | −458 | −2,268 | −733 | +500 | +252 | +122 |
| 2SLS, W1 treats | +122 | −742 | −440 | +5,660 | +391 | +141 | +875 | +270 |
| 2SLS, W2 treats | +148 | −351 | −89 | −762 | +1,094 | +670 | +1,188 | +655 |
| **PROBE** | +185 | +504 | −64 | −3,149 | −216 | +1,527 | −0 | +844 |

| over the eight | mean absolute error | median absolute error |
|---|---|---|
| 2SLS, W2 treats | 620 | 662 |
| X and both proxies | 713 | 479 |
| **PROBE** | 811 | **360** |
| 2SLS, W1 treats | 1,080 | 415 |
| adjust for X | 1,203 | 1,126 |
| COCA as an ATE | 37,383 | 25,189 |

PROBE has the smallest median error and a mean error near zero, −46 dollars, but the third mean absolute
error, pulled up by PSID-1. The two 2SLS orientations need a role assignment that no one supplies; picking
one at random would give an expected mean absolute error of about 850. LaLonde does not separate PROBE
from the simple adjustments.

## 10. Addendum, 21:40 CDT: ten real LaLonde samples

The early random-assignment AFDC women (Smith and Todd's restriction, as Calonico and Smith use it), 565
complete cases with their own randomised benchmark of +1,268 (se 465), against PSID-1 and PSID-2 women.
They lie inside the women of section 9, so they are nested as well. The original LaLonde men's file has no
1974 earnings, so it has one proxy and cannot be used. These ten are every real sample the files support.

| over the ten | mean absolute error | median absolute error | mean signed error |
|---|---|---|---|
| 2SLS, W2 treats | 601 | 662 | +361 |
| X and both proxies | 636 | 443 | −461 |
| **PROBE** | 825 | 524 | **−104** |
| 2SLS, W1 treats | 909 | 380 | +657 |
| adjust for X | 1,167 | 1,042 | −1,167 |

PROBE's signed error is the smallest, so it is the least biased on average; its absolute error is third.
