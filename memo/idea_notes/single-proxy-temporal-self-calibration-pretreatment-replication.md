# Temporal Self-Calibration: Single-Proxy Point Identification via a Pre-Treatment Replication Period

**Date:** 2026-06-28
**Status:** ELIMINATED (broken). Exact reason: the transport step assumes the pre-period proxy-to-outcome slope equals the treatment-period $X$-conditional slope, which fails whenever the single proxy's latent map is non-injective (two latent states share the same proxy and baseline-outcome law). An explicit counterexample produces two structural models that agree on the full observed law to machine precision yet have different ATEs (0.10 vs 0.20), and the proposed estimator returns 0.180 (wrong for both). The stated identifying functional additionally collapses algebraically to a no-adjustment formula.
**Rank among the 9:** last tier (eliminated; one of the broken ideas, salvageable only by importing the very completeness condition it claims to discharge).
**Mechanism family:** temporal / panel self-calibration. A pre-treatment replication period is used to learn the confounder-to-proxy and confounder-to-baseline-outcome measurement model and transport it into the treatment period.
**Tags:** single-proxy, proximal causal inference, negative controls, panel data, difference-in-differences, completeness, non-injective proxy, cautionary mechanism.

## One-line idea

> If a single proxy $W$ and the outcome $Y$ are each measured once before treatment exists and once after, and the confounder's effect on $W$ and on the baseline outcome is stationary across the two periods, then the pre-period acts as a self-contained calibration sample that pins down the confounding bridge, point-identifying the treatment effect from one proxy. (This claim is shown below to be false without an added completeness condition.)

## Glossary

- $X$: treatment (exposure), assigned only in the treatment period. Contrast values written $x, x'$.
- $Y$: outcome. $Y_0$ is the pre-period (baseline) outcome, $Y_1$ the treatment-period outcome.
- $U$: latent confounder, time-invariant across the two periods, UNMEASURED, possibly vector-valued.
- $W$: the single proxy of $U$. Observed twice, written $W_0$ at $t=0$ and $W_1$ at $t=1$. It is a negative-control / pre-outcome-type variable: no direct effect of or on treatment or outcome except through $U$.
- $t$: period index, $t \in \{0, 1\}$. $t=0$ is the pre-treatment (placebo) period; $t=1$ is the treatment period.
- $Y_1(x)$: potential outcome in the treatment period under intervention $X = x$.
- ATE (average treatment effect): $\mathbb{E}[Y_1(x) - Y_1(x')]$ for a contrast $(x, x')$. The dose-response is $\mathbb{E}[Y_1(x)]$.
- Proxy / negative control: a variable driven by $U$ that carries information about $U$ but does not itself cause $Y$ or $X$ except through $U$.
- Completeness: a condition stating that conditioning on a variable retains all information about a latent. Formally $\mathbb{E}[h(U) \mid W = w] = 0$ for all $w$ implies $h = 0$. It guarantees that any bounded function of $U$ is recoverable from $W$.
- Bridge function: a function of observables solving an integral (Fredholm) equation that algebraically removes confounding, the central object of proximal causal inference.
- Confounding bridge $b_0(w)$: the function $\mathbb{E}[g(U) \mid W = w]$, the part of the proxy-outcome association attributable purely to $U$.
- $m(U) := \mathbb{E}[W_t \mid U]$: the latent-to-proxy regression (same in both periods by stationarity).
- $g(U) := \mathbb{E}[Y_0 \mid U]$: the latent-to-baseline-outcome regression.
- $\tau(X)$: the additive treatment effect map at $t=1$, with $\tau(0) = 0$.
- $\beta_U$: the composite "confounding slope", the coefficient of $W$ in the linear projection of $Y_0$ on $W$.
- Injective map: a function taking distinct inputs to distinct outputs. An injective $m$ means distinct $U$ values yield distinct proxy distributions.

## Problem and motivation

When a treatment $X$ and an outcome $Y$ share an unmeasured common cause $U$, the back-door criterion cannot be satisfied with observed covariates, and the naive regression of $Y$ on $X$ confounds the causal association with the spurious one induced by $U$. Proximal causal inference repairs this using negative controls (proxies) of $U$. The established result is that TWO proxies, a treatment-side proxy $Z$ and an outcome-side proxy $W$, suffice for nonparametric point identification, provided a completeness condition links them strongly enough to invert a Fredholm integral equation for the outcome bridge.

The single-proxy regime is the interesting frontier because two valid proxies are often unavailable: many studies have at most one negative-control variable. The question this family of ideas asks is whether one proxy, combined with some additional structure, can still deliver point identification. The structure exploited here is a DESIGN feature rather than a second variable: a pre-treatment period in which the treatment node does not yet exist. The hope is that the pre-period gives a clean, treatment-free read of how $U$ contaminates the proxy-outcome relationship, and that this read can be transported into the treatment period to subtract exactly the confounding. The gap targeted is precisely the second-proxy requirement: can TIME play the role of the second negative control?

## Setup and causal graph

The latent $U$ is time-invariant and unmeasured. In period 0 there is no treatment node. In period 1 the treatment $X$ switches on. The single proxy $W$ is recorded in both periods. The directed edges are:

- Period 0: $U \to W_0$, $U \to Y_0$.
- Period 1: $U \to X$, $U \to W_1$, $U \to Y_1$, $X \to Y_1$.

There is NO edge $X \to W_1$ (the proxy is a pre-outcome-type negative control, unaffected by treatment). There is NO direct edge $W_0 \to Y_1$ or $W_1 \to Y_1$ except through $U$. There is NO lagged $Y_0 \to Y_1$ state dependence beyond $U$ (relaxed in A5). There is NO second proxy and NO unconfounded mediator.

```
                 U  (UNMEASURED, time-invariant)
        ________/ | \________________________
       /     /    |        \        \         \
      v     v     v         v        v         v
    W0     Y0    W1         X        Y1        (X exists only at t=1)
                            |________^
                              X -> Y1

 Period 0 (pre-treatment):   U -> W0,  U -> Y0
 Period 1 (treatment):       U -> W1,  U -> X,  U -> Y1,  X -> Y1
 Single proxy W observed at t=0 (W0) and t=1 (W1).
```

## Target estimand

The target is the treatment-period dose-response and the contrast built from it:
$$
\mathbb{E}[Y_1(x)] \;=\; \mathbb{E}_U\!\big[\, \mathbb{E}(Y_1 \mid X = x, U) \,\big], \qquad \text{ATE} \;=\; \mathbb{E}[Y_1(x) - Y_1(x')].
$$
Here $Y_1(x)$ is the potential outcome under setting $X = x$ in the treatment period.

## Assumptions

**A1. Single time-invariant confounder and proxy.**
Formal: there exists a latent $U$ (scalar or vector), time-invariant across the two periods, with a single proxy $W$ observed at $t = 0$ and $t = 1$ such that $W_t \perp X \mid U$ and $W_t \perp Y_t \mid U$.
Plain reading: one hidden cause drives everything, and $W$ is a pure proxy of it, with no direct link to treatment or outcome.
Mildness: this is exactly the standard single-proxy exclusion restriction of negative-control work. The only addition is that the same proxy is recorded twice, automatic in any panel.
Testable: the exclusion restrictions are not directly testable; the time-invariance of $U$ is a modeling assumption.

**A2. Pre-treatment period (design feature).**
Formal: at $t = 0$ treatment is not assigned (or is held at a common reference $x^\* = 0$ for all units), so $Y_0 = Y_0(0)$ and the $X$ node has no causal path into $Y_0$ or $W_0$. Structurally $X \perp \{Y_0, W_0\}$.
Plain reading: before the policy or exposure starts, nobody is treated, so the baseline outcome reflects only confounding.
Mildness: routinely available as baseline waves, pre-policy years, or run-in periods. This is the same design fact difference-in-differences exploits.
Testable: holds by design when the exposure start date is known.

**A3. Channel stationarity (the key extra assumption).**
Formal: $\mathbb{E}[W_1 \mid U] = \mathbb{E}[W_0 \mid U] =: m(U)$, and $\mathbb{E}[Y_1 \mid U, X] = g(U) + \tau(X)$ with $\mathbb{E}[Y_0 \mid U] = g(U)$ and $\tau(0) = 0$.
Plain reading: the meaning of the proxy and the baseline confounding structure do not drift over the short panel; only the treatment channel switches on.
Mildness: weaker than additive equi-confounding on the treatment-effect side, because $\tau(X)$ may be fully heterogeneous and the confounding bias is never assumed equal across treatment arms.
Testable: partially checkable; a placebo pre-period (3-wave panel) or a within-unit proxy-change test can probe drift.

**A4. Linearity-in-$U$ of the proxy/outcome maps OR a known link (relevance).**
Formal: in the working model $m(U) = a + bU$ and $g(U) = c + dU$ with $b \ne 0$, $d$ unrestricted; the nonparametric version posits an injective $m$ and a completeness-in-time analogue.
Plain reading: the proxy carries information about $U$, and in the simplest model that information is linear.
Mildness as stated: $b \ne 0$ is just proxy relevance. THIS IS THE ASSUMPTION THAT THE COUNTEREXAMPLE BREAKS: a scalar slope $b \ne 0$ does not guarantee that $m$ is injective or full rank.
Testable: relevance ($b \ne 0$) is testable from the proxy-outcome association; injectivity is not testable without knowing $U$.

**A5. No outcome state-dependence beyond $U$.**
Formal: $Y_0$ does not enter the structural equation for $Y_1$ except through $U$. Relaxable: if $Y_1 = \rho Y_0 + g(U) + \tau(X) + e$ with known or identified $\rho$, the argument proceeds after partialling out $\rho Y_0$.
Plain reading: no lagged-outcome feedback; the only thing carried across periods is $U$.
Mildness: standard in two-wave panel identification, with an explicit relaxation provided.
Testable: not directly; the relaxed model identifies $\rho$ under added structure.

## The single mild assumption that replaces the second proxy

The proposal isolates A3 (channel stationarity) as the lever meant to replace the second proxy. The claim is that the second proxy in standard proximal inference does two jobs: (i) locate $U$ in proxy space, and (ii) separate the confounding association from the causal one, the separation that requires completeness on $(Z, W)$. The pre-treatment period is supposed to supply job (ii) for free, because at $t = 0$ there is no treatment, so the regression of $Y_0$ on $W$ is "pure confounding" with no causal contamination. A3 then transports this calibrated confounding bridge into the treatment period, where the single proxy only has to do job (i). In short, the proposal claims that the completeness condition of two-proxy proximal inference is "discharged by time".

The soundness audit below shows this claim is FALSE. Job (i), locating $U$, itself requires a completeness / full-rank condition on the single proxy, and A4 as stated does not deliver it.

## Identification result

**Claimed theorem (as proposed, later shown to require a stronger A4).** Under A1 to A5, the treatment-period dose-response is identified by
$$
\mathbb{E}[Y_1(x)] \;=\; \mathbb{E}_W\!\big[\, \mathbb{E}(Y_1 \mid X = x, W) - b_0(W) \,\big] + \mathbb{E}_W\!\big[\, b_0(W) \,\big],
$$
where the confounding bridge $b_0(w) = c' + \beta_U w$ is read off from the pre-period regression of $Y_0$ on $W$, with $\beta_U := d/b$ the composite confounding slope. Equivalently the proposal writes
$$
\mathbb{E}[Y_1(x)] \;=\; \mathbb{E}_W\!\big[\, \mathbb{E}(Y_1 \mid X = x, W) \,\big] - \beta_U \,\big( \mathbb{E}[W \mid X = x] - \mathbb{E}[W] \big).
$$
The pre-period normal equation $\mathbb{E}[Y_0 - b_0(W) \mid \text{span}] = 0$ is presented as a degenerate, finite-dimensional, hence completeness-free Fredholm equation.

Two flaws in this statement are documented in the soundness audit: the first displayed functional collapses algebraically (the $-b_0$ and $+\mathbb{E}_W[b_0]$ cancel under the marginal), and the "completeness-free" claim is false.

## Proof sketch (Lean-style, with the broken step flagged)

**Step 1 (calibration identifies the confounding map).**
Inputs: A2 (no treatment at $t=0$), A3, A4.
Derivation: at $t=0$, $Y_0 = g(U) + e_0$ with $\mathbb{E}[e_0 \mid U, W] = 0$. With $g(U) = c + dU$ and $\mathbb{E}[W_0 \mid U] = a + bU$, we get $\mathbb{E}[Y_0 \mid W_0 = w] = c + d\,\mathbb{E}[U \mid W_0 = w]$. In the linear-Gaussian working model $\mathbb{E}[U \mid W_0 = w]$ is affine in $w$, so $\mathbb{E}[Y_0 \mid W_0 = w] = c' + \beta_U w$.
Justification: at $t = 0$ the regression is guaranteed to be pure confounding because treatment is absent by design.
Flag: this step silently assumes $\mathbb{E}[U \mid W = w]$ is a faithful read of $U$. That holds only if $m$ is injective. The counterexample violates it.

**Step 2 (transport via stationarity).**
Inputs: A3 (same $(a,b)$ and $(c,d)$ at $t=1$), Step 1.
Derivation: at $t=1$, $\mathbb{E}[Y_1 \mid X = x, W = w] = \tau(x) + (\text{confounding projection of } Y_1 \text{ on } w)$. By A3 the confounding projection equals the same affine map $c' + \beta_U w$ from Step 1. Hence $\mathbb{E}[Y_1 \mid X = x, W = w] - (c' + \beta_U w) = \tau(x)$ plus a pure treatment shift.
Flag: this is the load-bearing and incorrect step. It equates the $X$-conditional confounding slope at $t=1$ with the marginal slope at $t=0$. When $W$ is a noisy / non-sufficient proxy, $\mathbb{E}[g(U) \mid X = x, W = w] \ne \mathbb{E}[g(U) \mid W = w]$, because conditioning on $X$ reweights the latent posterior. The two differ exactly when the proxy fails to be a sufficient statistic for $U$.

**Step 3 (averaging / g-formula).**
Inputs: Step 2, A1.
Derivation: marginalize over $W$ to undo selection into $X$ through $U$, yielding the stated functional.
Flag: marginalizing the stated functional cancels the adjustment (see audit), so even granting Steps 1 to 2 the written formula performs no correction.

**Step 4 (why it is claimed to be an honest bridge solution).**
The proposal argues the standard outcome bridge solves a Fredholm-I equation needing completeness, but here the pre-period gives a direct regression read of $b_0$, so the integral equation "degenerates" and completeness is "vacuous".
Flag: false. The degeneracy claim assumes the regression of $Y_0$ on $W$ recovers $g(U)$ per latent state. It recovers only $\mathbb{E}[g(U) \mid W]$, which loses information whenever $m$ is non-injective.

## Why a single proxy was claimed to suffice

The crux argument was: two proxies are needed because one proxy cannot simultaneously locate $U$ and separate confounding from causation; with one proxy these demands collide and completeness fails. The pre-treatment period was supposed to supply the separation for free, since the $t=0$ regression of $Y_0$ on $W$ is pure confounding, directly tracing $b_0(\cdot)$. Time was cast as the "second negative control": $W_0$ as the calibrating negative-control outcome, $W_1$ as the locating proxy.

The flaw is that locating $U$ from a single proxy is itself a completeness problem. Two i.i.d. reads of $W$ identify the latent mixture only when the latent-to-proxy map has full column rank (Allman-Matias-Rhodes, Kruskal, Hu). A scalar nonzero slope does not imply full rank. So the "second negative control via time" does not discharge completeness; it merely relocates it onto the single proxy, where it must still be assumed.

## Estimator (as proposed)

Stage A (calibrate): regress $Y_0$ on $W$ in the pre-period to estimate $\hat{b}_0(w) = \hat{c}' + \hat{\beta}_U w$, or a flexible $\hat{b}_0(\cdot)$ in the nonparametric link version.
Stage B (transport and average): estimate
$$
\widehat{\mathbb{E}[Y_1(x)]} = \frac{1}{n}\sum_i \big[\hat{\mu}_1(x, W_i) - \hat{b}_0(W_i)\big] + \frac{1}{n}\sum_i \hat{b}_0(W_i),
$$
with $\hat{\mu}_1(x, w) = \widehat{\mathbb{E}}[Y_1 \mid X = x, W = w]$, then $\widehat{\text{ATE}} = \widehat{\mathbb{E}[Y_1(x)]} - \widehat{\mathbb{E}[Y_1(x')]}$.
A doubly robust version augments with a propensity term $\pi(x \mid W)$ and the residual $Y_1 - \hat{\mu}_1$, with cross-fitting for machine-learning nuisances. A proposed falsification test regresses the within-unit proxy change $W_1 - W_0$ on covariates and checks for a zero slope, and uses a second pre-period as a placebo for $\hat{\tau} = 0$.

Two estimator problems are confirmed below: the Stage-B formula reduces to $\frac{1}{n}\sum_i \hat{\mu}_1(x, W_i)$ (no adjustment), and the whole pipeline is not consistent when $m$ is non-injective.

## Worked intuition: a small concrete example

Take $U \in \{0, 1, 2\}$, $W$ binary read i.i.d. at each period with $\mathbb{P}(W = 1 \mid U = u) = m[u]$, baseline $\mathbb{P}(Y_0 = 1 \mid U = u) = g[u]$, treatment $\mathbb{P}(X = 1 \mid U = u) = \pi[u]$, and outcome $\mathbb{P}(Y_1 = 1 \mid U = u, X = x) = g[u] + \tau[u]\,x$. Suppose
$$
m = [0.2,\; 0.8,\; 0.8], \qquad g = [0.3,\; 0.6,\; 0.6].
$$
States $U = 1$ and $U = 2$ have IDENTICAL proxy law and IDENTICAL baseline-outcome law. The pre-period regression of $Y_0$ on $W$ sees only the proxy distribution and the baseline outcome, so it cannot tell $U = 1$ from $U = 2$. Any treatment heterogeneity hidden inside the $\{U = 1, U = 2\}$ block is invisible to the calibration. This is the mechanism by which the transport step fails: the pre-period "self-calibration" resolves $U$ only up to the proxy's coarsening, and $\tau$ can hide in the unresolved directions.

## Soundness audit

The adversary found a patchable but decisive counterexample. Two structural causal models, both satisfying A1 to A5 literally, produce IDENTICAL observed laws over $(W_0, Y_0, W_1, X, Y_1)$ (maximum cell difference $1.4 \times 10^{-17}$ across all 32 cells) but DIFFERENT ATEs.

Shared lever: $m = [0.2, 0.8, 0.8]$ and $g = [0.3, 0.6, 0.6]$, so states $U = 1$ and $U = 2$ are observationally fused. The proxy map is NON-INJECTIVE.

Model A (ATE = 0.10): $p_U = [0.4, 0.3, 0.3]$, $\pi = [0.3, 0.8, 0.2]$, $\tau = [0.1, 0.2, 0.0]$. Then $\text{ATE} = 0.4(0.1) + 0.3(0.2) + 0.3(0.0) = 0.10$.

Model B (ATE = 0.20): $p_U = [0.4, 0.2, 0.4]$, $\pi = [0.3, 0.9, 0.3]$, $\tau = [0.1, 0.0, 0.4]$. Then $\text{ATE} = 0.4(0.1) + 0.2(0.0) + 0.4(0.4) = 0.20$.

Inside the indistinguishable $\{U = 1, U = 2\}$ block the two models re-allocate latent mass, selection, and effect while holding fixed the only two functionals the data can see in that block: the selection mass $a_1\pi_1 + a_2\pi_2 = 0.30$ and the treated-outcome mass $a_1\pi_1\tau_1 + a_2\pi_2\tau_2 = 0.048$. Everything observable is a function of (block mass, $g$, selection mass, treated-outcome mass), all matched. But $\mathbb{E}_U[\tau(U)]$ is NOT a function of those quantities, so it moves freely between 0.10 and 0.20. Numerically verified: identical 32-cell table, ATEs 0.10 versus 0.20, and the proposal's own estimator returns 0.180 on BOTH, wrong for each.

Severity: patchable in the sense that the mechanism is sound under an added condition, but the idea as written is broken. The break shows the claim that "completeness is vacuous / discharged by time" is false. Two i.i.d. reads of $W$ replace the second proxy ONLY when the latent-to-proxy map is injective / full column rank ($|\text{support}(W)| \ge |\text{support}(U)|$ with independent columns, or injective $m$ in the continuous case). A4 as stated, a scalar slope $b \ne 0$, does not guarantee this.

A second, independent defect: the stated identifying functional $\mathbb{E}_W[\mathbb{E}(Y_1 \mid X = x, W) - b_0(W)] + \mathbb{E}_W[b_0(W)]$ collapses to $\mathbb{E}_W[\mathbb{E}(Y_1 \mid X = x, W)]$, because $-b_0$ and $+\mathbb{E}_W[b_0]$ cancel in the marginal. That formula performs NO confounding adjustment and is biased even within a single correctly specified model (verified: returns 2.717 against a true 2.0). The correct bridge must subtract $\mathbb{E}[g(U) \mid X = x, W = w]$, the $X$-conditional latent posterior, not $b_0(w) = \mathbb{E}[g(U) \mid W = w]$, the marginal posterior; these differ whenever $W$ is non-sufficient.

Corroborating evidence that the temporal idea is sound under the missing condition: Jacobian-rank local-identification tests with a FULL-RANK proxy were all full rank (deficiency 0), across discrete $K = L = 3$, a coarse proxy $K = 3, L = 2$, a hidden extra latent direction resolvable via the $X / Y$ channels, and heterogeneous $\tau(U)$; the linear-Gaussian version over-determines $\tau$. The failure is specifically the non-injective / rank-deficient proxy, not the temporal mechanism per se.

## Honest status and the fix

**Final status: ELIMINATED (broken).** The transport step assumes the pre-period proxy slope equals the $X$-conditional slope in the treatment period, which fails under a non-injective proxy and under drift. Once the proxy is forced to be full-rank, the construction reduces to a Sofer-style difference-in-differences with negative-control pre-outcome, carrying no novelty over the additive-equi-confounding sibling (idea 2 in this family). The note is retained as a CAUTIONARY MECHANISM.

Concrete repair path (each item makes the idea correct but removes its claimed advantage):

1. Discrete case: replace A4 with $|\text{support}(W)| \ge |\text{support}(U)|$ and require the matrix $M[u, w] = \mathbb{P}(W = w \mid U = u)$ to have full column rank (no two latent states share a proxy distribution). This rules out the $m[1] = m[2]$ collision.
2. Continuous / nonparametric case: require $m$ injective AND completeness-in-$W$ ($\mathbb{E}[h(U) \mid W = w] = 0 \Rightarrow h = 0$). This is the SAME completeness that two-proxy proximal needs, now imposed on the single proxy's two-read structure. With it, the two i.i.d. reads identify the latent mixture (Hu; Allman-Matias-Rhodes; Kruskal), the pre-period identifies $g(U)$ per state, and $\tau$ is over-identified.
3. Linear-Gaussian working model: forbid any confounding latent direction that loads on $(X, Y)$ but not on $W$. State a "no unproxied confounding direction" assumption ($U$ one-dimensional and fully proxied).
4. Fix the functional: subtract $\mathbb{E}[g(U) \mid X = x, W = w]$, not $b_0(w)$. Re-derive via latent-state weights recovered from the full-rank two-read mixture, e.g. $\text{ATE} = \sum_u p_U(u)\,\tau(u)$ with $(p_U, \tau)$ obtained by inverting the full-rank $M$ at each stage.

The sibling that salvages the temporal lever is the additive-equi-confounding / negative-control difference-in-differences idea (COCA and Sofer-style), which is honest about its parallel-trends-type assumption and does not claim to discharge completeness.

## Novelty vs prior work

The intended novelty was point identification from EXACTLY ONE proxy with no known measurement mechanism, no second proxy, no completeness, no unconfounded mediator, and no additive equi-confounding across arms, using temporal self-calibration as the new lever.

The closest prior work is **Single Proxy Control (Tchetgen Tchetgen, Park et al., 2024)**. The intended delta was: that line still requires a strong proxy-relevance / completeness-type condition and an i.i.d. single-period design, whereas this idea claimed to use the panel design to discharge completeness. The audit shows this delta does not hold: completeness cannot be discharged by time, only relocated onto the single proxy's two-read structure. Once relocated, the requirement is the same completeness Single Proxy Control already needs, so the claimed delta evaporates. Against the difference-in-differences sibling, after the full-rank patch the construction collapses to a Sofer-style negative-control DiD, so it is not novel relative to that line either.

## AISTATS significance and positioning

A correct version would not be a new identification result; it would be a panel restatement of existing single-proxy or two-proxy completeness conditions, which is below the bar for a methods venue. The genuine scientific contribution of this note is NEGATIVE and pedagogical: it pins down, with an explicit machine-precision counterexample, the exact reason a tempting "time replaces the second proxy" intuition fails. The precise lesson is that relevance ($b \ne 0$) is strictly weaker than sufficiency / full rank, and that pre-period calibration recovers only the marginal latent posterior $\mathbb{E}[g(U) \mid W]$, not the $X$-conditional posterior the bridge actually needs. This is worth documenting as a cautionary case study supporting the surviving ideas in the family, not as a standalone submission.

## Open problems and next steps

1. Characterize the sharp partial-identification region for $\mathbb{E}[Y_1(x)]$ when the proxy is non-injective: the counterexample shows the ATE is set-valued, and the bounds (driven by the unresolved within-block reallocation of mass, selection, and effect) are worth deriving explicitly.
2. Quantify how much the multi-read panel structure tightens those bounds as the number of waves grows, and whether three or more i.i.d. reads can substitute for full rank at $K = 2$.
3. Make the drift channel (a violation of A3) a sensitivity parameter and trace its effect on the bias, separating the non-injectivity failure from the stationarity failure.
4. Test whether combining the pre-period calibration with a weak instrument or a partial second proxy restores point identification with a strictly weaker condition than full two-proxy completeness.

## References

- Tchetgen Tchetgen, Park, et al. (2024). Single Proxy Control. Biometrics; and Single Proxy Synthetic Control, arXiv:2307.16353.
- Miao, Geng, Tchetgen Tchetgen (2018). Identifying causal effects with proxy variables of an unmeasured confounder. Biometrika.
- Cui, Pu, Shi, Miao, Tchetgen Tchetgen (2024). Semiparametric proximal causal inference. JRSS-B.
- Ghassami et al. Proximal causal inference (minimax bridge estimation).
- Tchetgen Tchetgen (2014). Control Outcome Calibration Approach (COCA).
- Sofer et al. Negative-control / pre-outcome difference-in-differences and parallel trends.
- Kuroki, Pearl (2014). Measurement bias and effect restoration. Biometrika.
- Pearl (1995). Front-door criterion; Dukes, Shi, Fulcher, Tchetgen Tchetgen, proximal mediation.
- Allman, Matias, Rhodes (2009). Identifiability of latent-class mixtures (Kruskal's theorem).
- Hu (2008). Identification of nonclassical measurement-error models.
- Athey, Imbens; panel factor-model and long-term causal inference under persistent confounding, arXiv:2202.07234.
