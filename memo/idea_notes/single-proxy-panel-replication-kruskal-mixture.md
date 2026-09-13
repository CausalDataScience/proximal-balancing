# Panel Proxy Replication: One Repeated Proxy in Place of the Second Proxy, via Repeated iid Reads and a Kruskal Mixture

**Date:** 2026-06-28
**Status:** ELIMINATED. Exact reason: the identifying functional silently requires that the proxy-resolvable latent states exhaust the confounding of the treatment-outcome relation (no residual within-state confounding) and, separately, that the control baseline does not drift across waves. Two structural models satisfying all stated assumptions A1 to A5 reproduce the full observed law to machine precision yet disagree on the ATE (0.050 versus 0.195). Point identification fails.
**Rank among the 9:** eliminated tier (one of two fatally broken ideas; salvaged by sibling Idea 7, which replaces the false exhaustiveness with a Gamma sensitivity cap).
**Mechanism family:** panel proxy replication. Three or more conditionally independent reads of a single proxy recover a finite latent mixture by a Kruskal three-way decomposition, followed by an attempt at point identification.
**Tags:** single-proxy, proximal causal inference, negative control, finite mixture, Kruskal decomposition, Hu 2008, Allman-Matias-Rhodes, panel data, difference-in-differences, residual confounding, eliminated.

## One-line idea

> If a single proxy $W$ of a time-invariant latent confounder $U$ is read at least three times across a short panel (two pre-treatment waves where treatment is structurally absent, plus one treatment wave) and the proxy emission has full column rank, then the repeated iid reads identify the latent mixture and one hopes to point-identify a fully heterogeneous treatment effect. The hope is false: the recovered latent states need not exhaust the confounder, so the effect is not identified.

## Glossary

- $X$: treatment, taken binary unless stated otherwise. $X=1$ treated, $X=0$ control.
- $Y$: outcome. $Y_0$ is a pre-treatment (baseline) outcome read, $Y_1$ is the treatment-wave outcome.
- $U$: unmeasured confounder, time-invariant across the panel window, discrete with $K$ latent states unless the continuous case is named.
- $W$: the single proxy of $U$. It is read once per wave: $W_{0a}, W_{0b}$ pre-treatment and $W_1$ in the treatment wave.
- wave: a measurement occasion in the panel. Waves are $t = 0a, 0b$ (pre-treatment) and $t = 1$ (treatment).
- proxy / negative control: a variable driven by $U$ that has no direct causal link to $X$ or $Y$ except through $U$. Formally $W_t \perp X \mid U$ and $W_t \perp Y_s \mid U$.
- emission map $M$: in the discrete case the matrix $M[u,w] = P(W = w \mid U = u)$, the conditional law of the proxy given the latent state.
- full column rank / proxy completeness: $\mathrm{rank}(M) = K$ in the discrete case, meaning distinct latent states have distinct proxy laws. In the continuous case the operator $h \mapsto E[h(W) \mid U = u]$ is injective.
- ATE: average treatment effect, $E_U[\tau(U)]$ for binary $X$.
- $\tau(u)$: the per-latent-state (CATE-by-latent-state) effect within state $U = u$.
- $p_U(u)$: the latent state prior, $P(U = u)$.
- bridge function: in standard proximal inference a function $b_0(W)$ solving an integral equation that performs confounding adjustment. Here the analogue is the finite Kruskal factorization system, not a Fredholm inverse.
- completeness: an injectivity condition on a conditional expectation operator, the technical engine that lets proxies stand in for $U$.
- DiD: difference-in-differences. NOC: negative-outcome control. SCM: structural causal model.

## Problem and motivation

When a confounder $U$ of the $X \to Y$ relation is unmeasured, naive adjustment on observed covariates leaves a bias term that no amount of data removes. Proximal causal inference addresses this with proxies of $U$. The standard machinery (Miao, Geng and Tchetgen Tchetgen 2018; Cui et al. 2024) needs two proxies: a treatment-inducing proxy $Z$ and an outcome-inducing proxy $W$, plus completeness conditions on each, so that a bridge function can be solved and the latent emission inverted.

The interesting and harder regime is a single proxy. One proxy variable does not, on its own, supply enough conditionally independent views of $U$ to invert the emission. The present idea attacks this by trading a second proxy variable for a design feature: a panel. If the same proxy is recorded again in another wave and its meaning does not drift, the repeated reads are conditionally independent measurements of the same time-invariant $U$. Three such reads meet the Kruskal threshold for finite-mixture identification, so the latent mixture is recoverable from one proxy variable read three times. The gap this idea targets is therefore real: reduce the second proxy variable to a second and third read of one proxy, while being honest that proxy completeness is still paid.

The note documents why this attractive reduction fails to deliver point identification. The mixture machinery works exactly as advertised, yet the causal estimand still escapes, because recovering $U$'s mixture is not the same as recovering the full confounder of $X$ on $Y$.

## Setup and causal graph

The latent $U$ is time-invariant with $K$ states. There are three waves. The directed edges are:

- $U \to W_{0a}$, $U \to W_{0b}$, $U \to W_1$: the one proxy variable $W$ read once per wave, with the three reads conditionally independent given $U$ and no direct read-to-read carryover.
- $U \to Y_{0a}$ (and optionally $U \to Y_{0b}$): pre-treatment baseline outcome reads, which are pure $Y(0)$ because treatment is structurally absent in the pre-period.
- $U \to X$, $U \to Y_1$, $X \to Y_1$: confounding by $U$ into both treatment and treatment-wave outcome, plus the causal arrow of interest.

Exclusion: $W_t \perp X \mid U$ and $W_t \perp Y_s \mid U$ for all $t, s$, so $W$ is a pure negative-control proxy. There is no second proxy variable and no unconfounded mediator. The only multiplicity is in time, not in variables. Crucially, $X$ does not exist as a node at $t = 0a, 0b$.

```
            U  (UNMEASURED, time-invariant, K states)
        /  / | \  \   \   \
       /  /  |  \   \    \    \
   W_0a W_0b W_1  Y_0a(Y_0b)  X   Y_1
                                  ^
                                  |
                                  X ---> Y_1

  pre-period (t=0a, 0b): treatment absent, Y_0 is pure Y(0)
  treatment period (t=1): X assigned, X -> Y_1
  reads W_0a, W_0b, W_1 conditionally independent given U
```

## Target estimand

The primary target is the per-latent-state effect and its average. For binary $X$:

$$\tau(u) = E[Y_1(1) - Y_1(0) \mid U = u], \qquad \mathrm{ATE} = E_U[\tau(U)] = \sum_u p_U(u)\, \tau(u).$$

For general dose $x$ the target is the dose-response $E[Y_1(x)] = \sum_u p_U(u)\, E[Y_1 \mid U = u, X = x]$. Because the method claims to recover the full latent mixture, the ambitious deliverable is the full heterogeneity profile $\tau(u)$, which scalar-outcome DiD cannot recover. The fallback target under rank deficiency is sharp partial-identification bounds on $E_U[\tau(U)]$ obtained by ranging over all latent mixtures consistent with the observed read tensor.

## Assumptions

**A1. Time-invariant latent confounder with a single repeated proxy.**
Formal: $U$ is constant across the panel window; $W$ is read once per wave with reads conditionally independent given $U$, and $W_t \perp X \mid U$, $W_t \perp Y_s \mid U$.
Plain reading: the same hidden trait persists, and the proxy reacts only to that trait, never directly to treatment or outcome.
Mildness: the exclusion is the standard single-proxy / negative-control restriction. Recording the same proxy each wave is automatic in longitudinal designs.
Testable: the exclusion is not directly testable; the conditional-independence-of-reads structure is partially checkable through wave co-occurrence patterns.

**A2. At least two pre-treatment waves with treatment structurally absent.**
Formal: at $t = 0a, 0b$ the $X$ node does not exist, so $Y_{0a}, Y_{0b}$ equal $Y(0)$; at $t = 1$, $X$ is assigned and $X \to Y_1$.
Plain reading: there are genuine baseline waves before any treatment, like pre-policy years or run-in periods.
Mildness: routinely available. The two pre-waves (versus one) are the only design cost beyond DiD, and they supply the third read.
Testable: the structural absence of treatment is a design fact, not a data test.

**A3. Wave-stationary emission.**
Formal: $P(W_{0a} = w \mid U) = P(W_{0b} = w \mid U) = P(W_1 = w \mid U) =: M[u,w]$ for all waves.
Plain reading: the proxy means the same thing in each wave; the measurement instrument does not drift.
Mildness: this restricts the measurement, not the confounding-outcome relation, so it differs in kind from parallel trends.
Testable: partially. Equality of read marginals and of pairwise read co-occurrence tensors across waves is observable.

**A4. Proxy completeness / full-column-rank emission (the stated price).**
Formal: discrete, $\mathrm{rank}(M) = K$; continuous, $m$ injective and $W$ complete for $U$.
Plain reading: distinct latent states leave distinct proxy fingerprints, so no two states are confused.
Mildness: this is the binding cost, equal in strength to the completeness that two-proxy proximal already demands, here imposed on one proxy read several times.
Testable: partially refutable by a rank or eigenvalue-gap check on the empirical read co-occurrence matrix.

**A5. No outcome state-dependence beyond $U$ within the window.**
Formal: $Y_1$ depends on $(U, X)$ only, not on lagged $Y_0$ except through the time-invariant $U$.
Plain reading: the only thing tying baseline and treatment-wave outcomes together is the persistent trait.
Mildness: standard in short-panel latent-variable identification; the extra pre-wave provides moments to test it.
Testable: partially, via a third pre-wave placebo.

## The single mild assumption that replaces the second proxy

The replacement is A4, proxy completeness, combined with A3, wave-stationarity. In two-proxy proximal inference, the second proxy variable exists to supply a second conditionally independent measurement of $U$ so that the latent emission can be inverted, which requires a completeness condition on that second proxy. Here, A3 makes the repeated reads of the single proxy conditionally iid given the time-invariant $U$, and A4 supplies the full-column-rank condition that makes inversion of the single emission solvable.

The honest statement is that A4 stands in for the second-proxy completeness, not for the second-proxy variable plus its completeness for free. Time converts the need for a second variable into the need for a second and third read, but it does not remove the richness requirement. That reduction, second variable down to second read, is the intended contribution.

## Identification result (claimed, later refuted)

**Claim (as proposed).** Under A1 to A5, the latent mixture $\{p_U, M\}$ is identified up to label permutation from the three reads, the per-state baseline $g(u) = E[Y_0 \mid U = u]$ is identified from the pre-period, and the per-state effect

$$\tau(u) = E[Y_1 \mid U = u, X = 1] - E[Y_0 \mid U = u]$$

is identified, giving $\mathrm{ATE} = \sum_u p_U(u)\, \tau(u)$ and $E[Y_1(x)] = \sum_u p_U(u)\, E[Y_1 \mid U = u, X = x]$.

The bridge equation is the finite Kruskal factorization system

$$P(W_{0a} = i, W_{0b} = j, W_1 = k) = \sum_u p_U(u)\, M[u,i]\, M[u,j]\, M[u,k],$$

equivalently, in pairwise form, $P_{ab} = M^\top \, \mathrm{diag}(p_U)\, M$. This is a finite eigen / tensor problem, solvable exactly when A4 holds. Completeness is paid here, not avoided. The claim is later shown to be false: the functional above is not identified because the latent states $U$ may fail to exhaust the confounding of $X$ on $Y_1$, and because $g(u)$ may differ from the treatment-wave control baseline.

## Proof sketch (Lean-style, with the broken step flagged)

**Step 1 (mixture recovery).** Inputs: the three reads $W_{0a}, W_{0b}, W_1$; A3 (shared emission); A4 (rank $K$). Derived: $\{p_U, M\}$ up to label permutation. Justification: Kruskal 1977, Allman, Matias and Rhodes 2009, Hu 2008. Three conditionally independent finite views with full-column-rank emission identify the mixture. The label permutation is irrelevant to the ATE. This step is sound.

**Step 2 (per-state baseline).** Inputs: $\{p_U, M\}$ from Step 1; the pre-period outcome $Y_0$; A2 (pure $Y(0)$ at baseline). Derived: $g(u) = E[Y_0 \mid U = u]$ via the same factorization with the two pre-reads as anchors. Justification: $Y_0$ is a fourth conditionally independent view at baseline. A scalar $Y_0$ alone cannot separate two latent states sharing a baseline mean, but the full-rank proxy reads can resolve them, so $g(\cdot)$ is identified per state. This step is sound as a recovery of $g(u)$.

**Step 3 (per-state effect, the load-bearing and broken step).** Inputs: $\{p_U, M\}$; the treatment-wave $(X, Y_1)$; the recovered $g(u)$. Derived (claimed): $E[Y_1 \mid U = u, X = x] = g(u) + \tau(u) x$, hence $\tau(u) = E[Y_1 \mid U = u, X = 1] - g(u)$.

This is where the single proxy enters as the supposed engine of heterogeneity, and it is where the proof breaks. The step silently assumes two things that A1 to A5 do not deliver. First, it assumes $Y_1(0) \perp X \mid U$: the proxy-resolvable $K$-state $U$ is the entire confounder of the $X$-$Y_1$ relation. A single proxy can certify completeness only relative to the latent states it emits from. Nothing forces those states to exhaust the $X$-$Y$ confounding. Second, it equates the treatment-wave control baseline with the pre-period baseline, $E[Y_1(0) \mid U = u] = g(u)$, ruling out secular drift, which is an outcome-side parallel-baselines restriction that A3 deliberately did not carry.

**Step 4 (necessity check, honest).** A Jacobian-rank check confirms local identification of the full three-read model under A4 (rank $17$ with $K = 3$). An explicit construction shows that when A4 fails, the same read tensor admits two models with different ATE, so A4 is necessary. These checks are correct, but they verify the mixture recovery, not the causal estimand. They do not detect the failure in Step 3, which is a confounding failure invisible to the read tensor.

## Why a single proxy suffices (for the mixture, not for the effect)

The crux argument the idea offers is genuine for the mixture. A panel supplies the second and third conditionally independent measurement of $U$ from the same proxy: under A3 the repeated reads are conditionally iid given the time-invariant $U$, which is exactly the Kruskal / Hu multi-view hypothesis. The pre-period further guarantees that one read's companion outcome $Y_0$ is pure confounding, anchoring per-state baselines without treatment contamination. So time replaces the second proxy variable with extra reads, but only when the single proxy is complete.

The honest limit of this argument is that recovering $U$'s mixture is necessary but not sufficient for the causal effect. The crux argument certifies that $W$'s repeated reads pin down $U$. It says nothing about whether $U$ is the whole confounder of $X$ on $Y_1$. The single proxy suffices to identify the latent mixture and fails to identify the effect.

## Estimator (as proposed)

Three stages, all finite-dimensional. Stage 1: from the empirical three-read tensor, recover $\{\hat p_U, \hat M\}$ by a tensor or Kruskal decomposition, for example simultaneous diagonalization of two read-sliced pairwise matrices, with the rank chosen from the eigenvalue gap of the pairwise read matrix (which doubles as the A4 diagnostic). Stage 2: regress the pre-period outcome on the recovered posterior $P(U = u \mid W_{0a}, W_{0b})$ to obtain $\hat g(u)$. Stage 3: within each latent state, $\hat\tau(u) = \hat E[Y_1 \mid U = u, X = 1] - \hat g(u)$, then $\widehat{\mathrm{ATE}} = \sum_u \hat p_U(u)\hat\tau(u)$.

A doubly robust, EM-augmented variant cross-fits the emission and outcome models and uses the efficient influence function of the finite mixture for Wald confidence intervals at the $\sqrt{n}$ rate. Diagnostics: a rank check on the pairwise read matrix (tests A4), a wave-stationarity check on read marginals and co-occurrence (tests A3), and a third pre-wave placebo expecting $\hat\tau = 0$ (tests A5). The fatal point is that none of these diagnostics can detect the residual within-state confounding that breaks Step 3. The estimator is internally consistent and targets a non-identified functional.

## Worked intuition with a concrete binary example

Take $U$ binary with $p_U = (0.5, 0.5)$, a proxy $W \in \{0,1,2\}$ with wave-stationary emission

$$M = \begin{pmatrix} 0.7 & 0.2 & 0.1 \\ 0.1 & 0.3 & 0.6 \end{pmatrix}, \qquad \mathrm{rank}(M) = 2,$$

so A3 and A4 hold and the three reads are conditionally iid given $U$. Pre-period baselines are $g(u) = E[Y_0 \mid U = u] = (0.30, 0.60)$. The proxy is generated from $U$ alone, so $W \perp X \mid U$ and $W \perp Y \mid U$ hold exactly. The mixture machinery recovers $p_U$, $M$, and $g(u)$ perfectly. The reader who stops here is convinced the estimand is in hand. The next section shows it is not.

## Soundness audit (the adversary counterexample, stated explicitly)

The adversary finding is fatal, with two independent fatal breaks. Both are verified numerically: the two models reproduce the full observed law over $(W_{0a}, W_{0b}, W_1, Y_0, X, Y_1)$ to within a maximum per-cell probability difference of $2 \times 10^{-17}$, machine zero, while disagreeing on the ATE.

**Primary break: residual within-state confounding.** Both models share the latent setup above: $U$ binary, $p_U = (0.5, 0.5)$, the same $M$, the same $g(u) = (0.30, 0.60)$.

Model $M_1$ adds a hidden sub-state $V \in \{0,1\}$ with $V \perp U$, $V \perp W$, $V \perp Y_0$, where $V$ affects only $X$ and $Y_1$. Set $p_V = (0.5, 0.5)$, $P(X = 1 \mid U, V) = \begin{pmatrix} 0.2 & 0.6 \\ 0.4 & 0.9 \end{pmatrix}$, and $Y_1(0)$ means $(U{=}0,V{=}0){:}\,0.20$, $(0,1){:}\,0.50$, $(1,0){:}\,0.50$, $(1,1){:}\,0.80$, with a small homogeneous true effect $0.05$, so $P(Y_1 = 1 \mid U, V, X = 1) = \text{base} + 0.05$. Marginalizing over $V$ gives observed $P(X = 1 \mid U) = (0.40, 0.65)$ and observed $m_{\mathrm{obs}}[u,x] = P(Y_1 = 1 \mid U = u, X = x) = \begin{pmatrix} 0.300 & 0.475 \\ 0.5429 & 0.7577 \end{pmatrix}$. True ATE in $M_1$ is $0.050$.

Model $M_2$ takes $V$ degenerate. Set $P(X = 1 \mid U) = (0.40, 0.65)$ and $P(Y_1 = 1 \mid U, X) = m_{\mathrm{obs}}$ directly, with the same proxy, the same $g$, and the same $Y_0$. True ATE in $M_2$ is $\sum_u p_U(u)(m_{\mathrm{obs}}[u,1] - m_{\mathrm{obs}}[u,0]) = 0.195$.

Because the $V$-marginalized conditionals of $W, Y_0, X, Y_1$ coincide cell by cell with $M_2$, the two observed laws are identical. The mixture machinery (Steps 1 and 2) is fully granted and correctly recovers $p_U$, $M$, $g(u)$, and the $X$-conditional posteriors in both models. It simply cannot see $V$. The idea's own functional $\sum_u p_U(u)(E[Y_1 \mid U = u, X = 1] - E[Y_0 \mid U = u])$ returns $0.166$, wrong for both. The unstated requirement of Step 3 is $Y_1(0) \perp X \mid U$, which fails in $M_1$: $E[Y_1(0) \mid U = 0, X = 0] = 0.300$ versus $E[Y_1(0) \mid U = 0, X = 1] = 0.425$. A single proxy certifies completeness only relative to the states it emits from, and nothing forces those states to exhaust the $X$-$Y$ confounding. This is exactly the gap that a second, treatment-inducing proxy $Z$ closes in standard proximal inference.

**Secondary independent break: secular drift.** Even granting $Y_1(0) \perp X \mid U$, Step 3 anchors the treatment-period control baseline with the pre-period outcome $g(u) = E[Y_0 \mid U]$ rather than the treatment-period control mean. With $g = (0.30, 0.60)$ and observed $m_{\mathrm{obs}} = \begin{pmatrix} 0.45 & 0.55 \\ 0.50 & 0.80 \end{pmatrix}$, a per-state secular drift $\delta(u) = m_{\mathrm{obs}}[u,0] - g(u) = (0.15, -0.10)$ leaves the observed law unchanged for any $\delta$. The true ATE is $\sum_u p_U(u)(m_{\mathrm{obs}}[u,1] - m_{\mathrm{obs}}[u,0]) = 0.200$, while the idea's functional gives $\sum_u p_U(u)(m_{\mathrm{obs}}[u,1] - g(u)) = 0.225$. So a no-drift restriction $E[Y_1(0) \mid U = u] = E[Y_0 \mid U = u]$ is also required and was unstated.

Severity: fatal, and doubly so, since the two breaks are independent. Either alone destroys point identification.

## Honest status and the fix

Status: ELIMINATED. The functional is not identified. The failure mode is precise. Step 3 of the proof smuggles in two assumptions that A1 to A5 do not supply.

1. No residual within-state confounding: $Y_1(0) \perp X \mid U$, that is, the proxy-resolvable $U$ is the entire confounder of $X$ on $Y_1$. This is not refutable from the observed data (no diagnostic separates $M_1$ from $M_2$) and is as strong as assuming the full confounder is measured up to the proxy. It is precisely the role the missing second, treatment-inducing proxy plays in standard proximal inference, so requiring it concedes the central novelty claim.
2. No secular drift in the control baseline: $E[Y_1(0) \mid U = u] = E[Y_0 \mid U = u]$. This is a parallel-baselines restriction on the outcome's confounding map across waves, exactly the kind of assumption A3's measurement-only stationarity was meant to avoid carrying.

Partial repair: if assumption 1 is granted, then $E[Y_1(0) \mid U = u]$ is identified directly as the treatment-wave control mean $E[Y_1 \mid U = u, X = 0]$, the pre-period $g(u)$ becomes unnecessary, and the correct functional is $\tau(u) = E[Y_1 \mid U = u, X = 1] - E[Y_1 \mid U = u, X = 0]$. This is back-door adjustment on the recovered latent posterior. It still needs assumption 1, which a single proxy cannot certify. There is no patch that preserves both one proxy and point identification of a heterogeneous $\tau(U)$ without importing an uncheckable no-residual-confounding (completeness-on-the-true-confounder) assumption.

Cautionary lesson and salvage path: treat this mechanism as a warning that mixture recovery of a latent confounder is not the same as identification of the causal estimand. The sibling Idea 7 salvages the construction by replacing the false exhaustiveness with a Gamma sensitivity cap that bounds the residual within-state confounding, trading point identification for a sensitivity analysis that remains honest about what one proxy can and cannot certify.

## Novelty vs prior work

The single closest prior work is Sofer, Richardson, Colicino, Schwartz and Tchetgen Tchetgen (2016), which frames negative-outcome control as a generalization of difference-in-differences. The intended delta was to make the repeated single proxy load-bearing for a fully heterogeneous $\tau(U)$ and a nonparametric baseline $g(U)$, which their scalar negative-outcome control cannot identify, since a scalar $Y_0$ cannot separate latent states that share a baseline mean while full-rank proxy reads can.

The honest delta after the audit is narrower. The mixture-recovery contribution (one proxy read three times pins the latent mixture by Kruskal / Hu) stands. The causal-identification contribution does not, because it inherits the same no-residual-confounding requirement that the second proxy in Sofer et al. and in two-proxy proximal exists to certify. The idea does not beat the baseline on the estimand; it relocates the unverifiable assumption rather than removing it.

## AISTATS significance and positioning

The negative result is itself instructive for a methods audience. It draws a clean line between two things often conflated in the proximal and measurement-error literatures: identifying the law of a latent confounder, and identifying a causal effect through it. The Kruskal / Hu machinery solves the former; the latter additionally requires that the recovered latent variable exhaust the confounding, which no single-proxy panel design can verify. Positioned honestly, this idea is a cautionary companion to a sensitivity-analysis method (Idea 7), not a standalone identification result. As a submission it would need to be reframed around the sharp partial-identification fallback and the Gamma cap, with the point-ID claim withdrawn.

## Open problems and next steps

- Characterize the sharp partial-identification region for $E_U[\tau(U)]$ when only the mixture is recovered and within-state confounding is unrestricted, then study how a Gamma sensitivity cap on the residual confounding tightens it (the Idea 7 direction).
- Quantify how an extra pre-wave or an identified autoregressive coefficient could bound, rather than assume away, the secular drift, and whether any observable signature of drift exists.
- Determine whether a partial second proxy (a weak or coarsened treatment-inducing proxy) can certify exhaustiveness without a full second variable, interpolating between this design and two-proxy proximal.
- Develop a formal diagnostic, if any exists, that could in principle detect residual within-state confounding; the present analysis suggests none does from the observed law alone, which would be worth proving.

## References

- Sofer, T., Richardson, D., Colicino, E., Schwartz, J., and Tchetgen Tchetgen, E. (2016). On negative outcome control of unobserved confounding as a generalization of difference-in-differences. Statistical Science.
- Hu, Y. (2008). Identification and estimation of nonlinear models with misclassification error using instrumental variables. Journal of Econometrics.
- Allman, E. S., Matias, C., and Rhodes, J. A. (2009). Identifiability of latent class models with many observed variables. Annals of Statistics.
- Kruskal, J. B. (1977). Three-way arrays: rank and uniqueness of trilinear decompositions. Linear Algebra and its Applications.
- Miao, W., Geng, Z., and Tchetgen Tchetgen, E. (2018). Identifying causal effects with proxy variables of an unmeasured confounder. Biometrika.
- Cui, Y., Pu, H., Shi, X., Miao, W., and Tchetgen Tchetgen, E. (2024). Semiparametric proximal causal inference. Journal of the Royal Statistical Society, Series B.
- Tchetgen Tchetgen, E., Park, C., et al. (2024). Single proxy control. Biometrics.
- Kuroki, M., and Pearl, J. (2014). Measurement bias and effect restoration in causal inference. Biometrika.
- Kasahara, H., and Shimotsu, K. (2009). Nonparametric identification of finite mixture models of dynamic discrete choices. Econometrica.
- Bonhomme, S., Jochmans, K., and Robin, J.-M. (2016). Nonparametric estimation of finite mixtures from repeated measurements. Journal of the Royal Statistical Society, Series B.
