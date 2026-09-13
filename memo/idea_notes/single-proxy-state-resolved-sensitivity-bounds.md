# State-Resolved Sensitivity Bounds: One Repeated Proxy Recovers the Latent Mixture, and a State-by-State Rosenbaum Cap Sharply Bounds the Residual Within-State Confounding

**Date:** 2026-06-28
**Status:** sharp partial-ID only (point-ID recovered only in the testable limit $\Gamma=1$). The headline interval as originally written is valid but NOT sharp; it over-covers because it tilts only the control arm. A verified two-arm fix restores genuine sharpness.
**Rank among the 9:** #1 of 9. The only idea the matched adversary could not break.
**Mechanism family:** repeated single-proxy mixture recovery (Deaner / Kruskal / Hu / AMR) composed with a state-resolved marginal sensitivity model (Rosenbaum / Tan / Zhao / Dorn-Guo-Kallus).
**Tags:** proximal-inference, single-proxy, negative-control, panel-data, latent-mixture, sensitivity-analysis, marginal-sensitivity-model, Rosenbaum-Gamma, partial-identification, sharp-bounds, over-identification-test.

---

> One proxy read at least twice across a panel recovers the latent confounder's mixture and within-state $X$-conditional outcome means; capping residual within-state treatment selection by a Rosenbaum odds-ratio $\Gamma$ applied separately to each recovered latent state then yields ATE bounds that collapse to the back-door point at $\Gamma=1$ and come with an over-identification test that data-informs whether $\Gamma>1$ is required.

---

## Glossary

Every symbol and term below is used later in the note. Definitions appear before first use.

- $X \in \{0,1\}$: the binary treatment whose causal effect we want.
- $Y_1 \in [0,1]$: the outcome, measured at the treatment wave $t=1$, bounded without loss of generality to $[0,1]$.
- $Y_1(x)$: the potential outcome under intervention $\mathrm{do}(X=x)$, for $x\in\{0,1\}$.
- $U^* $: the time-invariant unmeasured confounder. It is the true latent state that drives both $X$ and $Y_1$.
- $U=\phi(U^*)$: a **coarsening** of $U^*$, taking $K$ discrete values $u\in\{1,\dots,K\}$. This is the part of the true confounder that the proxy can resolve. $\phi$ is an unknown many-to-one map.
- $V := U^*\setminus U$: the **residual sub-state**, the part of $U^*$ the proxy cannot resolve. This is the object the sensitivity parameter bounds.
- $W$: the **single proxy** of $U$, a **negative control** read once per wave. "Negative control" means a variable that responds to the latent confounder but is excluded from directly causing $X$ or $Y$.
- $W_{0a}, W_{0b}, W_1$: the proxy reads at pre-treatment waves $t=0a, 0b$ and at the treatment wave $t=1$.
- **Proxy / negative-control exclusion:** $W \perp X \mid U^*$ and $W \perp Y \mid U^*$, and $W$ depends on $U^*$ only through $U$. So $W$ carries information about $U$ but injects no direct path to treatment or outcome.
- $M[u,w] := P(W=w\mid U=u)$: the wave-stationary **emission matrix** (the measurement mechanism), of size $K\times|\mathcal{W}|$.
- $p_U(u) := P(U=u)$: the latent **mixture weights**.
- $m_{\mathrm{obs}}[u,x] := E[Y_1\mid U=u, X=x]$: the within-state, $X$-conditional observed outcome mean.
- $p_X(u) := P(X=1\mid U=u)$: the within-state treatment probability.
- **ATE:** the average treatment effect $\tau := E[Y_1(1)-Y_1(0)]$.
- $\tau(u) := E[Y_1(1)-Y_1(0)\mid U=u]$: the per-recovered-state effect.
- **Bridge function:** in proximal inference, a function solving an integral equation that links the proxy to the confounded outcome. Here the analogous object is a finite linear system, not an infinite-dimensional Fredholm equation.
- **Completeness:** a rank/injectivity condition ensuring a conditional-expectation operator is invertible, so distinct latent states leave distinct observable signatures. For discrete $U$ it reduces to $\mathrm{rank}(M)=K$.
- **Rosenbaum $\Gamma$ / marginal sensitivity model (MSM):** a cap stating that, after conditioning on observed covariates (here, on $U$), the odds of treatment between two units differ by at most a factor $\Gamma\ge 1$. $\Gamma=1$ means no unmeasured selection; $\Gamma\to\infty$ means arbitrary selection.
- $m0_u := m_{\mathrm{obs}}[u,0]$, $m1_u := m_{\mathrm{obs}}[u,1]$: shorthand for the within-state control and treated outcome means.
- $L(\Gamma), H(\Gamma)$: the lower and upper endpoints of the ATE bound at sensitivity level $\Gamma$.
- $\Gamma^*$: the **breakdown value**, the smallest $\Gamma$ at which the ATE interval first includes $0$.

---

## Problem and motivation

Standard causal identification assumes no unmeasured confounding: conditional on observed covariates, treatment is as good as randomly assigned. When a confounder $U^*$ is unmeasured, the back-door adjustment is biased, and the bias is not signed or bounded without further structure. The two-proxy proximal framework (Miao, Geng and Tchetgen Tchetgen 2018; Cui et al. 2024) restores point identification using **two** proxies of $U^*$ that satisfy complementary exclusion and completeness conditions. The two proxies jointly certify, through a bridge function, that the latent confounder is fully accounted for.

The **single-proxy regime** is the interesting and harder case: we have only one proxy variable. Deaner (2018) showed that in panel data a single proxy read across multiple periods behaves like multiple proxies, because repeated conditionally independent reads of a time-invariant latent identify the latent **mixture** by the Kruskal / Hu / Allman-Matias-Rhodes machinery. Deaner then point-identifies heterogeneous effects.

The gap this idea targets is precise. Deaner's point estimate silently requires that the proxy-resolvable latent $U$ **exhausts** the $X\to Y$ confounding, that is, there is no residual sub-state $V$ inside $U^*$ that still tilts treatment selection within a recovered state. Repeated reads genuinely deliver the mixture, but they do **not** certify exhaustiveness. If $V$ exists, the back-door point on the recovered posterior is biased, and the repeated-reads design gives no honest accounting of how biased. This note prices that residual confounding rather than assuming it away.

## Setup and causal graph

We observe a panel with two pre-treatment waves $t=0a, 0b$ (where $X$ is structurally absent) and one treatment wave $t=1$. The single proxy $W$ is read once per wave. The directed edges are:

- $U^* \to X$ (latent confounds treatment),
- $U^* \to Y_1$ (latent confounds outcome),
- $X \to Y_1$ (the causal effect of interest),
- $U^* \to U \to W$ (the proxy reads $U$, a coarsening of $U^*$; $W$ touches $U^*$ only through $U$),
- $V \to X$ and $V \to Y_1$ (the residual sub-state may still confound),
- with $V \perp W \mid U$ (the residual is invisible to the proxy).

$U^*$ is **unmeasured**. $W$ is the **single proxy**. $V=U^*\setminus U$ is the **auxiliary residual state** that the sensitivity parameter bounds. There is no second proxy and no unconfounded mediator.

```
            U*  (UNMEASURED, time-invariant)
          /  |  \
         /   |   \
        U    |    V        U = phi(U*)  (proxy-resolvable coarsening)
        |    |   / \       V = U* \ U   (residual, proxy CANNOT resolve)
        |    |  /   \
        W    v v     v
     (proxy) X ----> Y_1
       reads:        ^
     W_0a,W_0b,W_1   |
                     X -> Y_1  (target effect)

  Exclusion:  W _||_ X | U*,   W _||_ Y | U*,   V _||_ W | U
```

## Target estimand

The primary estimand is the average treatment effect
$$\tau = E[Y_1(1)-Y_1(0)],$$
together with the per-recovered-state effect
$$\tau(u) = E[Y_1(1)-Y_1(0)\mid U=u].$$
Because residual confounding by $V$ is bounded but not eliminated, both are reported as **partial-identification intervals** $[L(\Gamma), H(\Gamma)]$ indexed by the sensitivity cap $\Gamma\ge 1$. The point-identified special case is $\Gamma=1$, the testable limit in which $U$ exhausts confounding, where $\tau(u)=m1_u-m0_u$, reproducing Deaner's back-door point. The full deliverable is the **bound curve** over $\Gamma$ plus the breakdown value $\Gamma^*$ where $0$ first enters the ATE interval.

## Assumptions

**A1. Time-invariant latent confounder with a single repeated negative-control proxy.**
Formal: $U^*$ is time-invariant; $W$ is read once per wave with reads conditionally independent given $U=\phi(U^*)$; the emission is wave-stationary; $W\perp X\mid U^*$ and $W\perp Y\mid U^*$ (negative-control exclusion); and $V:=U^*\setminus U$ satisfies $V\perp W\mid U$.
Plain reading: the proxy reflects only the stable latent state, carries no direct path to treatment or outcome, and is blind to the residual sub-state.
Mildness: this is the standard single-proxy / negative-control exclusion plus the design fact that the proxy is recorded each wave. The one conceptual addition is honesty: $U$ is a coarsening, not the whole confounder.
Testable: the exclusion is partly checkable through the over-identification test (A6); the time-invariance and stationarity are checkable through a co-occurrence stationarity diagnostic.

**A2. At least two pre-treatment waves; reads conditionally i.i.d. given $U$ (mixture recovery, granted to Deaner).**
Formal: at $t=0a, 0b$ treatment is structurally absent, so with the $t=1$ read we have at least three conditionally i.i.d. views of $U$ with a wave-stationary full-column-rank emission; hence $\{p_U, M\}$ and the within-state means $m_{\mathrm{obs}}[u,x]$ are identified up to label permutation.
Plain reading: three independent looks at the same stable latent pin down the mixture and the per-state outcome means.
Mildness: this is exactly Deaner / Kruskal / Hu / AMR; we claim no novelty for it. Two pre-waves are a routine run-in design.
Testable: indirectly, through the over-identification restriction in A6.

**A3. Full-column-rank (proxy completeness) emission.**
Formal: $\mathrm{rank}(M)=K$ for discrete $U$, equivalently $W$ is complete for $U$; this enables the mixture inversion in A2.
Plain reading: distinct latent states produce distinguishable proxy distributions.
Mildness: carried honestly as the price of recovering the mixture, identical to Deaner. It buys the mixture, not exhaustiveness.
Testable: partly, via the eigenvalue gap of the empirical pairwise read matrix.

**A4. Bounded residual within-state selection (THE key assumption, sensitivity cap $\Gamma$).**
Formal: within each recovered state $u$, for all residual values $v, v'$,
$$\frac{1}{\Gamma}\ \le\ \frac{P(X=1\mid U=u,V=v)/P(X=0\mid U=u,V=v)}{P(X=1\mid U=u,V=v')/P(X=0\mid U=u,V=v')}\ \le\ \Gamma,\qquad \Gamma\ge 1.$$
Plain reading: the unresolved sub-state can tilt treatment odds within a state by at most a factor $\Gamma$.
Mildness: one interpretable scalar, the Rosenbaum / Tan scale practitioners already report, applied per recovered state. It is strictly milder than assuming "$U$ exhausts confounding" ($=\Gamma=1$ by fiat).
Testable: partly data-informed through A6.

**A5. Bounded outcome; control counterfactual anchored at the treatment wave, not the pre-period.**
Formal: $Y$ is bounded (WLOG $[0,1]$); $E[Y_1(0)\mid U=u]$ is anchored by the treatment-wave within-state control mean $m0_u$ (sharpened by $\Gamma$), not by any pre-period mean $g(u)$. The pre-period reads serve only as the third view for mixture recovery.
Plain reading: we never use the pre-period outcome as the counterfactual baseline, so we need no parallel-trends or no-drift restriction.
Mildness: boundedness is standard for any Manski or Rosenbaum bound; anchoring at the treatment wave eliminates the secular-drift failure mode.
Testable: boundedness is a known range; the placebo third-pre-wave check probes stationarity.

**A6. Over-identification test of the exhaustiveness restriction (data-informed $\Gamma$).**
Formal: with at least four conditionally i.i.d. views, the mixture is over-determined; a rejected over-id restriction (the recovered $K$-state model cannot jointly reproduce the higher-order read tensor and the $X$-conditional outcome moments) is evidence that residual $V$ is present, forcing $\Gamma>1$.
Plain reading: an extra read lets the data argue that $\Gamma=1$ is untenable.
Mildness: converts $\Gamma$ from a free knob into a partially testable quantity.
Testable: yes, this is the distinctive test.

## The single mild assumption that replaces the second proxy

The assumption that stands in for the missing second proxy is **A4**, the state-by-state Rosenbaum cap $\Gamma$.

In two-proxy proximal inference, the second proxy $Z$ supplies a completeness condition that certifies the latent confounder is fully captured, so the bridge function exists and the effect is point-identified. Deaner reduces that second variable to a repeated read, but his point estimate still needs the recovered latent to **exhaust** the $X\to Y$ confounding. That exhaustiveness is exactly what a second proxy's completeness would certify, and exactly what repeated reads cannot certify.

A4 replaces that certification with a quantified bound. Instead of a variable asserting "the latent is complete", we posit a single scalar $\Gamma$ that caps how far the recovered latent may **fall short** of exhausting confounding. The proxy mixture is what lets this cap be imposed **per recovered state**: each state's own weight $p_U(u)$, control mean $m0_u$, and outcome support are identified, so the Rosenbaum model acts state-by-state rather than marginally. $\Gamma=1$ recovers exhaustiveness (and Deaner's point); $\Gamma\to\infty$ recovers the assumption-free Manski bound.

## Identification result

**Granted input (A2-A3, Deaner / Kruskal / Hu / AMR).** From three conditionally i.i.d. reads,
$$P(W_{0a}=i, W_{0b}=j, W_1=k)=\sum_{u} p_U(u)\,M[u,i]\,M[u,j]\,M[u,k]$$
identifies $\{p_U, M\}$ up to label permutation. Adjoining $(X, Y_1)$ as a further view identifies $m_{\mathrm{obs}}[u,x]=E[Y_1\mid U=u, X=x]$ and $p_X(u)=P(X=1\mid U=u)$. This holds even when $V$ is present, because $V\perp W\mid U$ keeps $(X, Y_1)\perp W_{\text{reads}}\mid U$.

**Partial-ID layer (A4-A5).** The only object not identified is $E[Y_1(0)\mid U=u]$ (and, after the soundness audit below, also $E[Y_1(1)\mid U=u]$), because residual $V$ breaks within-state ignorability. Under the state-wise marginal sensitivity model A4 with bounded outcome A5, the Tan / Zhao sharp image of a single arm's mean $m\in[0,1]$ under odds-ratio tilts of magnitude at most $\Gamma$ is the closed interval with endpoints
$$\underline{T}(m;\Gamma)=\frac{m}{\,m+\Gamma(1-m)\,},\qquad \overline{T}(m;\Gamma)=\frac{\Gamma\,m}{\,\Gamma\,m+(1-m)\,}.$$

**Claim (as originally stated, one-arm version).** Setting $L_u(\Gamma)=\underline{T}(m0_u;\Gamma)$ and $H_u(\Gamma)=\overline{T}(m0_u;\Gamma)$ and pinning the treated arm at $m1_u$, the per-state effect interval is $\tau(u)\in[\,m1_u-H_u(\Gamma),\ m1_u-L_u(\Gamma)\,]$ and the ATE interval is
$$L(\Gamma)=\sum_u p_U(u)\big(m1_u-H_u(\Gamma)\big),\qquad H(\Gamma)=\sum_u p_U(u)\big(m1_u-L_u(\Gamma)\big).$$
This interval is **valid** (it always contains the truth) but, as Section 14 shows, it is **not sharp**: it over-covers because it tilts only the control arm.

**Two-arm correction (the verified fix).** The genuinely sharp set tilts both arms under one shared per-stratum propensity tilt. For each admissible within-state tilt of odds-ratio spread at most $\Gamma$, both $E[Y_1(1)\mid U=u]$ and $E[Y_1(0)\mid U=u]$ move under mirror-image Tan / Zhao maps applied to $m1_u$ and $m0_u$ respectively, coupled by the constraint that the same tilt drives both. Taking $\tau(u)=E[Y_1(1)\mid U=u]-E[Y_1(0)\mid U=u]$ over this joint admissible set gives the sharp per-state interval; aggregating with the identified $p_U(u)$ gives the sharp ATE interval. Numerically the corrected interval is strictly inside the one-arm interval (for example $(0.514, 0.657)$ instead of $(0.467, 0.689)$).

**$\Gamma=1$ collapse.** $\underline{T}(m;1)=\overline{T}(m;1)=m$, so $L_u=H_u=m0_u$, the treated pinning becomes exact (since $V$ is absent), and $\tau(u)=m1_u-m0_u$, recovering Deaner's point.

**Strict-sharpening over the marginal bound.** For any $\Gamma>1$ with heterogeneous $(p_X(u), m0_u)$ across states,
$$H(\Gamma)-L(\Gamma) < H_{\mathrm{marg}}(\Gamma)-L_{\mathrm{marg}}(\Gamma),$$
where the marginal no-proxy bound applies the same $\Gamma$ to the pooled control mean $m0=\sum_u p_U(u) m0_u$. The width function of the MSM map is not affine in $m$, so by Jensen the per-state-then-average width is strictly smaller than the average-then-MSM width. Reproduced width ratios were about $0.87$ to $0.89$ across $\Gamma\in[1.2, 3.0]$ for a two-state heterogeneous design. This argument is about the per-state-versus-pooled control map and is unaffected by the one-arm-versus-two-arm correction.

## Proof sketch (Lean-style)

**Step 1. Mixture recovery (A2-A3).**
Input: at least three conditionally i.i.d. reads, $\mathrm{rank}(M)=K$.
Derived: $p_U, M, m_{\mathrm{obs}}[u,x], p_X(u)$ up to label permutation.
Justification: Kruskal / Hu / AMR uniqueness of the three-way decomposition; $V\perp W\mid U$ ensures $(X, Y_1)\perp W_{\text{reads}}\mid U$, so adjoining $(X, Y_1)$ as a view is legitimate even with $V$ present. Granted, not novel.

**Step 2. Localize the unidentified object.**
Input: the recovered $\{p_U, M, m_{\mathrm{obs}}, p_X\}$.
Derived: the only quantities not identified are the within-state counterfactual means $E[Y_1(x)\mid U=u]$, because residual $V$ breaks $Y_1(x)\perp X\mid U$. We refuse to point-identify them.
Justification: under exhaustiveness ($\Gamma=1$) these equal $m_{\mathrm{obs}}[u,x]$; without it they do not.

**Step 3. Sharp single-arm interval under A4.**
Input: $m_{\mathrm{obs}}[u,x]$ and the cap $\Gamma$.
Derived: the reachable set of each counterfactual arm mean is the Tan / Zhao image $[\underline{T}(m;\Gamma), \overline{T}(m;\Gamma)]$.
Justification: brute-force sweep over all admissible odds-ratio tilts $t\in[1/\Gamma, \Gamma]$ matched the closed-form endpoints to $10^{-6}$.

**Step 4. The single proxy enters here.**
Input: the per-state arm intervals and the identified weights $p_U(u)$.
Derived: the per-state effect interval and, by $p_U$-weighting, the ATE interval.
Justification: $\tau(u)$ and ATE are monotone affine in the counterfactual arm means, so their sharp intervals are the images of the per-state intervals. The proxy is load-bearing precisely because it supplies the identified $p_U(u)$ weights and the per-state $m0_u, m1_u$ that let $\Gamma$ act state-by-state. Without the proxy, $\Gamma$ could only act on the pooled means, giving the strictly wider marginal bound (Step 6).

**Step 5. $\Gamma=1$ collapse.**
Input: the closed-form maps at $\Gamma=1$.
Derived: $L_u(1)=H_u(1)=m0_u$, treated pinning exact, $\tau(u)=m1_u-m0_u$.
Justification: $\underline{T}(m;1)=\overline{T}(m;1)=m$; the prior point-ID result is the $\Gamma=1$ slice, now exposed as an assumption rather than a theorem.

**Step 6. Strict sharpening over marginal.**
Input: heterogeneous $(p_X(u), m0_u)$.
Derived: $H(\Gamma)-L(\Gamma)<H_{\mathrm{marg}}(\Gamma)-L_{\mathrm{marg}}(\Gamma)$.
Justification: the MSM width is non-affine in $m$, so Jensen gives a strict gap; verified ratios $0.87$ to $0.89$.

**Step 7. Over-id test (A6).**
Input: a fourth read.
Derived: the $K$-state model's predicted fourth-order read tensor and joint outcome moments are testable; rejection certifies $V$ present, hence $\Gamma>1$.
Justification: over-determination of the mixture under A2-A3.

**Step 8. Drift fixed (A5).**
Input: the anchoring at the treatment-wave control mean.
Derived: the secular-drift counterexample does not apply because the pre-period mean $g(u)$ is never used as the baseline.

**Correction note.** Steps 3-4 must use the **two-arm** sharp set (Section 14), not the one-arm version, to obtain genuine sharpness. The collapse, the proxy-enters argument, and the strict-sharpening argument all survive the correction unchanged.

## Why a single proxy suffices

Two-proxy proximal inference needs a second variable to certify, through completeness, that the latent fully mediates confounding. Deaner reduces that second variable to a repeated read but still needs the resolved latent to exhaust the $X\to Y$ confounding for his point estimate.

One proxy suffices here because we ask it to do strictly less. The single repeated proxy buys only the latent **mixture**, which a proxy read at least twice genuinely delivers (Step 1). We do not ask it to certify exhaustiveness. The residual confounding the proxy cannot resolve is not assumed away but **bounded** by $\Gamma$ and reported. The proxy is still essential to the partial-ID layer: it supplies the identified weights $p_U(u)$ and per-state means that make the state-resolved bound strictly tighter than the no-proxy marginal bound (Step 6). The honest one-line statement is that one proxy point-identifies the mixture and sharply partially identifies the causal effect, with point ID recovered only in the testable limit $\Gamma=1$.

## Estimator

**Stage 1 (mixture, granted).** Tensor / Kruskal decomposition of the empirical three-read tensor yields $\widehat{p_U}, \widehat{M}$. An eigenvalue-gap rank check probes A3.

**Stage 2 (within-state moments).** Compute posteriors $P(U=u\mid \text{reads})$, then $\widehat{m}_{\mathrm{obs}}[u,x]$ and $\widehat{p_X}(u)$ by posterior-weighted regression, with cross-fitting to avoid overfitting the posterior.

**Stage 3 (sensitivity bounds).** For a grid of $\Gamma\ge 1$, compute $L(\Gamma), H(\Gamma)$ via the **two-arm** closed form. Report the bound curve and the breakdown $\Gamma^*$ where $0$ enters the ATE interval.

**Stage 4 (over-id test, A6).** With a fourth read, form a GMM or likelihood-ratio over-id statistic on the joint read-tensor and outcome moments. A rejection lower-bounds $\Gamma$ and flags that the $\Gamma=1$ point is unsafe; inverting the test gives a data-driven minimal $\Gamma$.

**Inference.** Use a percentile bootstrap, or the MSM efficient-influence-function bounds in the Dorn-Guo-Kallus style, for $\sqrt{n}$ confidence intervals on $[L(\Gamma), H(\Gamma)]$ that account for Stage-1 and Stage-2 estimation error.

**Diagnostics.** Rank check (A3), wave-stationarity co-occurrence check, third-pre-wave placebo, and the over-id test (A6). When A3 fails (rank-deficient emission), fall back to mixing over all admissible mixtures (Manski), a weaker but still valid report.

## Worked intuition (binary example)

Take a single recovered state ($K=1$, which isolates the residual-confounding layer) with observed within-state law $p_X(u)=0.5$, $m1_u=0.8$, $m0_u=0.2$, at $\Gamma=2$.

The Tan / Zhao control-arm image of $m0_u=0.2$ is
$$\underline{T}(0.2;2)=\frac{0.2}{0.2+2(0.8)}=0.111,\qquad \overline{T}(0.2;2)=\frac{2(0.2)}{2(0.2)+0.8}=0.333.$$
The **one-arm** interval pins the treated arm at $0.8$ and reports $\tau\in[0.8-0.333,\,0.8-0.111]=[0.467, 0.689]$, width $0.222$. But the same $V$ that confounds selection also confounds the treated arm, so $E[Y_1(1)\mid U=u]\ne m1_u$ in general. The **two-arm** sharp set, tilting both arms under one shared propensity tilt, gives $\tau\in(0.514, 0.657)$, width $0.143$. The one-arm interval is about $55\%$ too wide here. The mechanism is tangible: ignoring the treated-arm tilt does not bias the answer, it merely loosens the bound.

At $\Gamma=1$ both maps collapse to $0.2$, both arms pin exactly, and $\tau=0.8-0.2=0.6$, the back-door point.

## Soundness audit

The matched adversary **could not construct a break**: no two admissible structural models $M_1, M_2$ share the same observed law over $(W\text{-reads}, X, Y_1)$ yet yield an ATE outside the claimed interval. The adversary searched exhaustively over $K=1$ (the decisive unit, since the bound is built state-by-state) with residual $V$ of two to three levels, propensity odds-ratio tilt capped at $\Gamma$, both arms confounded by $V$, matching the full observed binary law. In every configuration the true sharp ATE set was strictly **contained** in the claimed interval. Representative numbers:

- observed $(p_X=0.5, m1=0.8, m0=0.2)$, $\Gamma=2$: claimed $(0.467, 0.689)$ width $0.222$; true sharp $(0.514, 0.657)$ width $0.143$.
- observed $(p_X=0.3, m1=0.7, m0=0.5)$, $\Gamma=3$: claimed $(-0.05, 0.45)$ width $0.500$; true sharp $(-0.049, 0.354)$ width $0.404$.
- observed $(p_X=0.8, m1=0.6, m0=0.4)$, $\Gamma=2.5$: claimed $(-0.025, 0.389)$ width $0.414$; true sharp $(-0.015, 0.365)$ width $0.380$.

The claimed interval is therefore a **valid, over-covering** bound: it always contains the truth, which is why no two same-observed-law models escape it. The coverage claim survives the decisive adversarial test.

**The real defect is a labeling error, not a break.** The central technical claim that the interval is **sharp** is false. The functional applies the Tan / Zhao tilt only to the control counterfactual $E[Y_1(0)\mid U=u]$ and silently point-identifies the treated counterfactual as $E[Y_1(1)\mid U=u]=m1_u$. But under A4 the same unresolved $V$ that confounds selection within state $u$ also confounds the treated arm, so $E[Y_1\mid U=u, X=1]\ne E[Y_1(1)\mid U=u]$ unless $Y_1(1)\perp X\mid U$. Inflating only one arm makes the interval strictly **wider** than sharp (about $35\%$ to $55\%$ too wide in asymmetric cases). The brute-force "sharpness" check swept only the control-arm tilt, which is why the non-sharpness was missed. The severity is **patchable**: conservative but correct, not anti-conservative.

The granted upstream steps are clean. Since $V\perp W\mid U$ and reads depend on $U^*$ only through $U$, we have $(X, Y_1)\perp W_{\text{reads}}\mid U$, so $m_{\mathrm{obs}}[u,x]$ and $p_X(u)$ are genuinely identified even with $V$ present. The $\Gamma=1$ collapse is correct.

## Honest status and the fix

**Final status: sharp partial-ID only.** The idea does not achieve point identification; it achieves a valid partial-identification interval that, with the fix, becomes genuinely sharp. Point ID is recovered only in the testable limit $\Gamma=1$.

**The fix (verified).** Replace the asymmetric control-arm-only MSM bound with a **two-arm MSM** that tilts both potential-outcome means within each recovered state under a single shared propensity tilt. Concretely, do not set $E[Y_1(1)\mid U=u]=m1_u$. Instead, for each admissible within-state propensity tilt with odds-ratio spread at most $\Gamma$, jointly bound $E[Y_1(1)\mid U=u]$ and $E[Y_1(0)\mid U=u]$ (the treated arm gets the mirror-image Tan / Zhao map applied to $m1_u$ with the same $\Gamma$), and intersect with the constraint that the same tilt drives both arms. Take $\tau(u)$ over this joint admissible set. This yields the genuinely sharp interval, for example $(0.514, 0.657)$ rather than $(0.467, 0.689)$.

The fix **strengthens** the paper. The interval tightens, the $\Gamma=1$ collapse is preserved, the strict-sharpening-over-marginal story is untouched (the Jensen-gap argument concerns the per-state-versus-pooled control map, not the within-arm count), and the over-id test layer (A6) is unaffected. The honest framing either drops the word "sharp" from the one-arm construction or, better, adopts the two-arm construction and re-verifies sharpness by brute force over both arms' tilts jointly.

## Novelty vs prior work

The single closest prior work is **Deaner (2018), "Proxy Controls and Panel Data."** Deaner uses repeated-period reads as proxies for time-invariant heterogeneity and point-identifies heterogeneous effects.

The delta is precise. We **grant** Deaner's mixture and point-ID and add the layer his point estimator does not provide: the sharp partial-ID region for residual within-state confounding, the exact gap that the adversary proved fatal for point ID. Three increments: (1) the deliverable lives in Deaner's failure mode, supplying the bound where he assumes the gap away; (2) the proxy is load-bearing for partial ID, since applying $\Gamma$ per recovered state with identified weights $p_U(u)$ strictly narrows the no-proxy marginal bound (about $12\%$), and $\Gamma=1$ collapses to his point; (3) a constructive over-identification test from at least four reads that data-informs whether $\Gamma>1$ is required, a finite-sample diagnostic for the conditional-exhaustiveness restriction Deaner assumes and cannot test.

## AISTATS significance and positioning

This is the best AISTATS bet of the nine. It is the only idea the adversary could not break, and its residual flaw is a labeling error with a verified strengthening fix, not a structural failure. The positioning is sharp: proximal inference and panel proxy controls promise point identification under conditions practitioners cannot check; this work reframes the single-proxy panel design as a partial-identification problem with one interpretable, partly testable sensitivity parameter, recovering point ID as a testable limiting case. It bridges two communities, latent-mixture proxy identification and Rosenbaum-style sensitivity analysis, with a concrete state-resolved bound, a strict-sharpening result, and an over-identification test. The deliverable, a bound curve plus a breakdown value $\Gamma^*$, is exactly the robustness summary applied researchers want.

## Open problems and next steps

1. Implement and brute-force-verify the two-arm sharp construction over both arms' tilts jointly, confirming the closed form against the grid to high precision.
2. Derive the efficient influence function for the two-arm state-resolved bound endpoints to obtain valid $\sqrt{n}$ confidence intervals that absorb Stage-1 and Stage-2 estimation error.
3. Formalize the over-identification test's power: characterize which residual-$V$ structures it can detect and quantify the implied data-driven lower bound on $\Gamma$.
4. Extend from discrete $K$-state $U$ to continuous latent states, replacing the finite Kruskal system with a completeness condition and a bridge equation.
5. Study the interaction between rank-deficient emission (A3 failure) and the sensitivity bound, characterizing the graceful degradation to the Manski fallback.
6. Empirically benchmark the state-resolved bound width against the marginal bound across realistic heterogeneity profiles to validate the $0.87$ to $0.89$ width-ratio range.

## References

- Deaner, B. (2018). "Proxy Controls and Panel Data." arXiv:1810.00283. Also "Many Proxy Controls," arXiv:2204.13815.
- Rosenbaum, P. (2002). *Observational Studies*. Springer.
- Tan, Z. (2006). "A distributional approach for causal inference using propensity scores." *JASA*.
- Zhao, Q., Small, D., and Bhattacharya, B. (2019). "Sensitivity analysis for inverse probability weighting estimators via the percentile bootstrap." *JRSS-B*.
- Dorn, J., and Guo, K. (2023). "Sharp sensitivity analysis for inverse propensity weighting." *JASA*.
- Dorn, J., Guo, K., and Kallus, N. (2024). Sharp MSM bounds with efficient influence functions.
- Manski, C. (1990). "Nonparametric bounds on treatment effects." *American Economic Review*.
- Hu, Y. (2008). "Identification and estimation of nonlinear models with misclassification error using instrumental variables." *Journal of Econometrics*.
- Allman, E., Matias, C., and Rhodes, J. (2009). "Identifiability of parameters in latent structure models with many observed variables." *Annals of Statistics*.
- Kruskal, J. (1977). "Three-way arrays: rank and uniqueness of trilinear decompositions." *Linear Algebra and its Applications*.
- Miao, W., Geng, Z., and Tchetgen Tchetgen, E. (2018). "Identifying causal effects with proxy variables of an unmeasured confounder." *Biometrika*.
- Cui, Y., et al. (2024). "Semiparametric proximal causal inference." *JRSS-B*.
- Kuroki, M., and Pearl, J. (2014). "Measurement bias and effect restoration in causal inference." *Biometrika*.
