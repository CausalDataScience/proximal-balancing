# Variance-Anchored Single Proxy Control: internal identification of the negative-control scale from cross-arm second moments

**Date:** 2026-06-28
**Status:** ELIMINATED (absorbed into the sibling proxy-side idea). Exact reason: the published identifying functional anchors the latent confounding scale on the cross-arm difference of the outcome variance $\mathrm{Var}(Y\mid X,C)$, and that denominator is contaminated by arm-dependent idiosyncratic outcome heteroskedasticity that every listed assumption permits. A concrete two-model counterexample drives the stated functional far from the truth while satisfying all assumptions. The estimand itself stays point-identified, but only through a different anchor (the proxy variance), so this note documents the outcome-side route as the cautionary precursor that motivates moving the anchor to the proxy side.
**Rank among the 9:** eliminated, but kept as the motivating precursor for the surviving proxy-side idea (idea 5).
**Mechanism family:** single negative-control-outcome proxy, additive (location-shift) confounding, second-moment (heteroskedasticity-based) scale anchor.
**Tags:** proximal causal inference, single proxy, negative control outcome, additive confounding, difference-in-differences generalization, heteroskedasticity identification, partial identification, sensitivity analysis.

## One-line idea

> Under additive single-factor confounding with one negative-control-outcome proxy, recover the unknown proxy-to-outcome confounding scale from cross-arm second moments rather than from a second proxy or external calibration, then read the ATE off a closed-form bridge. The published outcome-variance anchor is wrong; the proxy-variance anchor (sibling idea) is the repair.

## Glossary

- $X$: binary treatment, $X\in\{0,1\}$.
- $Y$: scalar outcome.
- $U$: a single unmeasured (latent) scalar confounder summary. It is never observed.
- $C$: observed baseline covariates, measured before treatment.
- $W$: the single proxy, a negative-control outcome (NCO) of $U$. It is a measured variable driven by $U$ but not caused by $X$ and not causing $X$ or $Y$.
- $\gamma(U,C)$: the scalar latent confounding channel through which $U$ enters both $Y$ and $W$. We write $\gamma$ for short.
- ATE (average treatment effect): $\theta = E[Y(1)] - E[Y(0)]$, where $Y(x)$ is the potential outcome under treatment level $x$.
- $\theta(C)$: the covariate-conditional treatment effect, $E[Y(1)-Y(0)\mid C]$, so $\theta = E_C[\theta(C)]$.
- proxy (negative control): a variable that carries information about $U$ but is shielded from $X$ by the causal structure, used to back out confounding bias.
- bridge function / identifying functional: a closed-form expression for the causal target written purely in terms of the observed law of $(X,W,Y,C)$.
- completeness: an injectivity condition on a conditional-expectation operator that two-proxy proximal inference uses to invert $U$'s effect; it is the requirement this note tries to avoid.
- proxy relevance: the requirement that $W$ actually carries information about $U$, formally $\mathrm{Cov}(W,U\mid C)\neq 0$.
- location shift / additive confounding: $U$ enters the conditional mean of $Y$ and of $W$ additively through $\gamma$, with no interaction between $U$ and $X$ in the $Y$ mean.
- loading: the per-unit coefficient with which $\gamma$ enters a mean. $b_Y(C)$ is $\gamma$'s loading on $Y$; $b_W(C)$ is $\gamma$'s loading on $W$.
- $\rho(C)$: the proxy-to-outcome confounding-scale ratio, $\rho(C) = b_W(C)/b_Y(C)$, the single unknown the second-moment anchor is meant to supply.
- $D_\gamma(C)$: the cross-arm location shift of the confounder, $E[\gamma\mid X=1,C]-E[\gamma\mid X=0,C]$.
- $D_v(C)$: the cross-arm dispersion shift of the confounder, $\mathrm{Var}(\gamma\mid X=1,C)-\mathrm{Var}(\gamma\mid X=0,C)$.
- $\varepsilon_Y,\varepsilon_W$: idiosyncratic (non-$U$) noise in $Y$ and $W$.

## Problem and motivation

Unmeasured confounding breaks the naive contrast. If $U$ affects both $X$ and $Y$, then $E[Y\mid X=1,C]-E[Y\mid X=0,C]$ mixes the causal effect with a confounding term, because the treated and control groups differ in their latent $U$ composition even within a covariate stratum $C$. Standard proximal causal inference repairs this with two proxies of $U$: a treatment-side proxy and an outcome-side proxy. The two-proxy machinery inverts an integral operator to recover $U$'s effect, and that inversion needs a completeness (injectivity) condition supplied by the second proxy.

The single-proxy regime is the interesting one because in practice an analyst often has exactly one credible negative control, for instance a pre-treatment repeated measurement of the outcome, or a second outcome known to be untouched by the treatment. One proxy is cheap and common. Two valid proxies plus completeness is a strong ask.

The gap that one proxy leaves is a counting gap. With additive confounding, the contamination of the $X$-contrast collapses to a single scalar per stratum, namely the confounder location shift $D_\gamma(C)$. The crude mean contrasts give two equations in three unknowns: the causal effect $\theta(C)$, the location shift $D_\gamma(C)$, and the calibration scale that links $W$'s units to $Y$'s units. Prior single-proxy work closes this by importing the scale: difference-in-differences sets it to one (parallel trends), validation-data methods read it from external calibration, and rank or monotonicity assumptions pin it structurally. This note targets exactly that imported scale and asks whether one proxy can supply it internally, from the observed law alone.

## Setup and causal graph

Vertices are $X$ (treatment), $Y$ (outcome), $U$ (unmeasured scalar confounder), $W$ (single NCO proxy), and $C$ (observed baseline covariates). The directed edges are:

- $U\to X$, $U\to Y$, $U\to W$ (the confounder drives treatment, outcome, and proxy).
- $X\to Y$ (the causal effect of interest).
- $C\to X$, $C\to Y$, $C\to U$, $C\to W$ (covariates feed everything).

The forbidden edges carry the structural content of a single NCO: $X\not\to W$ (treatment does not cause the proxy) and $W\not\to Y$, $W\not\to X$ (the proxy causes neither outcome nor treatment). Equivalently $W\perp\!\!\!\perp X\mid (U,C)$. So $U$ is the only non-$C$ common parent of $(X,Y)$ and the only non-$C$ parent of $W$. Single proxy means exactly one $W$ and no treatment-side proxy $Z$.

The heteroskedastic-confounding annotation is the load-bearing extra structure: the $U\to X$ edge is such that conditioning on $X$ changes not only the mean of $U$ but also its variance within a $C$-stratum. Selection on $U$ compresses or inflates the spread of $U$ differently in the two arms. This is the signal the second-moment anchor was meant to exploit.

```
            C
       ./   |    \.
      v     v      v
      U ---------> W
     /| \
    v |  v
   X  |   (U->W shown above)
    \ v
     vY <---- X
```

A cleaner edge listing of the same DAG:

```
mermaid
graph LR
  C --> X
  C --> Y
  C --> U
  C --> W
  U --> X
  U --> Y
  U --> W
  X --> Y
  %% forbidden: X-/->W, W-/->Y, W-/->X  (so W _||_ X | U,C)
```

## Target estimand

The target is the average treatment effect
$$\theta = E[Y(1)] - E[Y(0)],$$
built from covariate-conditional contrasts $\theta(C) = \mu(1,C) - \mu(0,C)$, where the interventional mean factors additively as
$$E[Y\mid do(x), C] = \mu(x,C) + E[\gamma(U,C)\mid C],$$
so that $\theta = E_C[\theta(C)]$. The dose-response curve $\beta(x) = E[Y(x)]$ is identified the same way arm by arm against a reference level.

## Assumptions

**A1. Consistency and positivity.** Formal: $Y = Y(X)$ and $0 < P(X=1\mid C) < 1$ almost surely. Plain reading: the observed outcome is the potential outcome at the realized treatment, and both arms occur in every stratum. Mildness: standard, shared by two-proxy proximal, COCA, Sofer-NCO, and DiD. Testable: positivity is checkable from data; consistency is structural.

**A2. Latent ignorability given $(U,C)$.** Formal: $Y(x)\perp\!\!\!\perp X\mid U,C$ for all $x$. Plain reading: once we condition on the latent confounder and covariates, treatment is as good as randomized. Mildness: exactly the proximal/NCO premise; $(U,C)$ block all backdoor paths. No extra cost over prior work. Testable: not testable, since $U$ is unobserved.

**A3. Valid single negative-control-outcome proxy.** Formal: $X\not\to W$ and $W\not\to(X,Y)$, i.e. $W\perp\!\!\!\perp X\mid (U,C)$ with $W$ having no causal effect on $X$ or $Y$. Plain reading: $W$ reflects $U$ but is causally inert with respect to the treatment-outcome system. Mildness: identical to the single-NCO validity requirement in COCA, Single Proxy Control, and Sofer et al.; a pre-treatment repeated measure is a canonical instance. Testable: the conditional-independence content is partly checkable through $W\perp\!\!\!\perp X\mid C$ implications, but the structural no-effect claims are not fully testable.

**A4. Additive single-factor confounding (no $U\times X$ interaction).** Formal: $E[Y\mid X=x,U,C] = \mu(x,C) + b_Y(C)\,\gamma(U,C)$ and $E[W\mid U,C] = \alpha(C) + b_W(C)\,\gamma(U,C)$, with the $X$-effect $\mu(1,C)-\mu(0,C)$ not depending on $U$, and $\gamma$ a scalar summary. Plain reading: $U$ shifts the level of $Y$ and $W$ through one shared scalar channel, and the treatment effect is constant across $U$. Mildness: the same location-shift homogeneity used in DiD and Sofer-NCO; weaker than COCA rank preservation. Single-factor means $U$ acts through one scalar direction shared by $Y$ and $W$, standard in factor-model confounding. Testable: not directly testable.

**A5. Constant loadings across arms.** Formal: $b_Y(C)$ and $b_W(C)$ do not depend on $x$. Plain reading: $U$'s per-unit effect on $Y$ and on $W$ is the same in both arms within $C$. Mildness: for $b_Y$ this is implied by A4 (no $U$-by-$X$ interaction), and for $b_W$ it follows from A3 because the law of $W$ given $U$ does not depend on $X$. It is therefore made explicit rather than added. Testable: inherits the testability of A3 and A4.

**A6. Proxy relevance.** Formal: $b_W(C)\neq 0$ and $\gamma(\cdot,C)$ non-degenerate, i.e. $\mathrm{Cov}(W,U\mid C)\neq 0$. Plain reading: $W$ actually carries information about $U$. Mildness: the single-proxy analogue of proxy relevance; if it fails no method can help. Testable: partly, through observed $\mathrm{Cov}(W,Y\mid C)$ being nonzero.

**A7. Treatment-induced confounder location shift.** Formal: $E[\gamma(U,C)\mid X=1,C]\neq E[\gamma(U,C)\mid X=0,C]$, i.e. $D_\gamma(C)\neq 0$. Plain reading: the confounder distribution shifts in mean between arms; this is the bias being corrected. Mildness: if it were zero the naive regression would already be unbiased. Testable: not directly, but its presence is reflected in a nonzero $E[W\mid X=1,C]-E[W\mid X=0,C]$.

**A8. KEY. Arm-varying confounder dispersion (the anchor).** Formal: $\mathrm{Var}(\gamma(U,C)\mid X=1,C)\neq \mathrm{Var}(\gamma(U,C)\mid X=0,C)$, i.e. $D_v(C)\neq 0$. Published claim: equivalently the observable $\mathrm{Var}(Y\mid X=1,C)\neq \mathrm{Var}(Y\mid X=0,C)$. Plain reading: the spread of the confounder, not just its mean, differs across arms, and this was claimed to show up as outcome heteroskedasticity. Mildness: generic under confounder-driven selection, with equal variance the knife-edge case. Testability: this is where the idea fails. The equivalence between confounder-dispersion change and observed outcome heteroskedasticity is false, because idiosyncratic outcome noise can itself be a function of $\gamma$ without violating any other assumption. The honest restatement appears in the soundness audit and the fix.

## The single mild assumption that replaces the second proxy

The assumption meant to do the work of the second proxy is A8, arm-varying confounder dispersion, $D_v(C)\neq 0$. In two-proxy proximal inference the second (treatment-side) proxy supplies completeness, the injectivity that lets one invert the operator mapping the latent $U$ to its effect on $Y$. Under additivity that operator collapses: the only thing left to pin is a single scalar, the proxy-to-outcome scale $\rho(C) = b_W(C)/b_Y(C)$. A8 was intended to supply exactly that one scalar equation, by making the cross-arm difference of second moments nonzero so the scale could be read off as a ratio of those differences. In other words A8 was advertised as standing in precisely for the completeness condition of the second proxy, trading an injectivity requirement on an operator for a non-degeneracy requirement on a variance contrast.

The catch, developed below, is that the second-moment channel the idea taps on the outcome side does not isolate $\mathrm{Var}(\gamma\mid X)$. The surviving repair taps the same kind of channel but on the proxy side, where the noise is forced constant by A3.

## Identification result (as published, before the audit)

Normalize $b_Y(C) = 1$ without loss of generality, since $\theta$ is invariant to rescaling $U$. Define $\rho(C) = b_W(C)/b_Y(C) = b_W(C)$.

Claimed two-moment identification of the scale:
$$\rho(C) = \frac{\mathrm{Cov}(W,Y\mid X=1,C) - \mathrm{Cov}(W,Y\mid X=0,C)}{\mathrm{Var}(Y\mid X=1,C) - \mathrm{Var}(Y\mid X=0,C)},$$
well-defined exactly when A8 makes the denominator nonzero.

First-moment contrasts. With $a(C) = E[Y\mid X=1,C]-E[Y\mid X=0,C]$ and $b(C) = E[W\mid X=1,C]-E[W\mid X=0,C]$, additivity gives
$$a(C) = \theta(C) + b_Y(C)\,D_\gamma(C), \qquad b(C) = b_W(C)\,D_\gamma(C).$$

Eliminating $D_\gamma(C)$ through $\rho$ yields the published bridge:
$$\theta(C) = a(C) - \frac{b(C)}{\rho(C)} = a(C) - b(C)\,\frac{\mathrm{Var}(Y\mid X=1,C)-\mathrm{Var}(Y\mid X=0,C)}{\mathrm{Cov}(W,Y\mid X=1,C)-\mathrm{Cov}(W,Y\mid X=0,C)},$$
and $\theta = E_C[\theta(C)]$. Every quantity on the right is a functional of the observed law of $(X,W,Y,C)$, with no operator inverted, no completeness invoked, and no external scale supplied. The soundness audit shows that this specific bridge is incorrect because the $\mathrm{Var}(Y\mid X)$ denominator is not a clean function of $\mathrm{Var}(\gamma\mid X)$.

## Proof sketch (Lean-style, step by step)

Throughout set $b_Y = 1$ by the scale normalization.

**Step 1 (means).** Inputs: A2, A4, A5. Derive $E[Y\mid X=x,C] = \mu(x,C) + E[\gamma\mid X=x,C]$, hence $a(C) = \theta(C) + D_\gamma(C)$, where $\theta(C) = \mu(1,C)-\mu(0,C)$ is the causal contrast because A4's no-$U$-modification gives $E[Y(1)-Y(0)\mid C] = \mu(1,C)-\mu(0,C)$. Justification: additive mean plus latent ignorability. By A3 and A5, $E[W\mid X=x,C] = \alpha(C) + b_W(C)\,E[\gamma\mid X=x,C]$, hence $b(C) = b_W(C)\,D_\gamma(C)$.

**Step 2 (the proxy enters: second moments).** Inputs: the decompositions $Y = \mu(x,C) + \gamma + \varepsilon_Y$ and $W = \alpha + b_W\gamma + \varepsilon_W$. The published argument assumes $\varepsilon_Y\perp\!\!\!\perp(X,U,W)\mid C$ with constant variance and $\varepsilon_W\perp\!\!\!\perp(X,U)\mid C$ with constant variance. Under those assumptions, within arm $x$,
$$\mathrm{Var}(Y\mid X=x,C) = \mathrm{Var}(\gamma\mid X=x,C) + \mathrm{Var}(\varepsilon_Y\mid C),$$
$$\mathrm{Cov}(W,Y\mid X=x,C) = b_W(C)\,\mathrm{Var}(\gamma\mid X=x,C).$$
This is the step where the single proxy enters: its covariance with the outcome tracks $b_W\,\mathrm{Var}(\gamma\mid X)$. Differencing across arms cancels the constant noise variances:
$$\mathrm{Var}(Y\mid 1,C)-\mathrm{Var}(Y\mid 0,C) = D_v(C), \qquad \mathrm{Cov}(W,Y\mid 1,C)-\mathrm{Cov}(W,Y\mid 0,C) = b_W(C)\,D_v(C),$$
so the ratio is $b_W(C) = \rho(C)$. A8 makes $D_v(C)\neq 0$, so the ratio is well-defined. The hidden premise is constant $\mathrm{Var}(\varepsilon_Y\mid C)$, and the audit shows that premise is not implied by A1 through A8.

**Step 3 (solve).** From $b(C) = \rho(C)\,D_\gamma(C)$, recover $D_\gamma(C) = b(C)/\rho(C)$, hence $\theta(C) = a(C) - b(C)/\rho(C)$.

**Step 4 (target).** $\theta = E_C[\theta(C)]$.

**Counting and well-posedness.** At the contrast-plus-second-moment level the observed law supplies $\{a, b, D_v, \text{the cross-arm covariance difference}\}$, while the unknowns are $\{\theta, D_\gamma, b_W\}$. The second-moment pair pins $b_W$, leaving a just-identified scalar equation for $D_\gamma$. A brute-force search over 360 in-class structural models matching a fixed observed law returned $\theta$ constant to $10^{-14}$ under the constant-noise restriction, which is exactly the restriction the adversary removes.

## Why a single proxy suffices

In two-proxy proximal inference $U$'s effect on $Y$ is an unknown function, and recovering it inverts an integral operator whose injectivity is the completeness the second proxy supplies. Additivity (A4) reduces the contamination of the $X$-contrast to a single scalar per cell, the location shift $D_\gamma(C)$. A scalar unknown needs one scalar equation. The crude mean contrasts $(a,b)$ give only two equations for three scalars $(\theta, D_\gamma, \text{scale})$, which is precisely why pure mean-contrast single-proxy methods import the scale.

The intended new observation is that one proxy carries a second, independent signature of $U$ through cross-arm second moments. The mean of $W$ reveals the location shift, and the variance/covariance of $W$ with $Y$ was meant to reveal the scale. The single proxy thus plays a double role, so no second proxy and no external calibration are required, provided the confounder spread differs across arms. The qualifier in the audit is that the outcome-side reading of this second signature is corruptible, while the proxy-side reading is not.

## Estimator

Plug-in with cross-fitting. (1) Estimate the conditional first moments $m_Y^x(C)=E[Y\mid X=x,C]$, $m_W^x(C)=E[W\mid X=x,C]$ and the conditional second moments $V_Y^x(C)=\mathrm{Var}(Y\mid X=x,C)$, $\mathrm{Cov}^x(C)=\mathrm{Cov}(W,Y\mid X=x,C)$ by flexible regression (boosting, series, or kernels), fitting conditional means then regressing squared and cross residuals on $C$. (2) Form $\hat\rho(C) = (\mathrm{Cov}^1-\mathrm{Cov}^0)/(V_Y^1-V_Y^0)$ and $\hat\theta(C) = (m_Y^1-m_Y^0) - (m_W^1-m_W^0)/\hat\rho(C)$. (3) Average, $\hat\theta = P_n\,\hat\theta(C)$.

Inference. $\theta(C)$ is a smooth functional of conditional-moment nuisances. Build the efficient influence function by stacking the AIPW score for the pseudo-outcome ATE on $\tilde Y = Y - W/\rho$ and the score contributions of the second-moment nuisances entering $\rho$, with the extra term from differentiating $\theta(C)$ in $\rho$, namely $+\,b(C)/\rho(C)^2$ times the $\rho$-score. Cross-fitting plus a product-of-rates Neyman-orthogonality condition over all nuisances yields root-$n$, asymptotically normal, multiply robust estimation.

Stability and partial ID. $\hat\rho$ is weak-anchor-sensitive when $|V_Y^1-V_Y^0|$ is small, an analogue of weak instruments. Report the first-stage statistic and use a weak-anchor-robust (Anderson-Rubin-style) confidence interval in $\rho$. If the anchor is weak, report the sharp partial-ID set $\theta(\rho) = a - b/\rho$ over a user interval of $\rho>0$. Critically, after the audit, the operative estimator replaces $V_Y$ by $V_W = \mathrm{Var}(W\mid X)$ in the scale, since $V_W$ is immune to outcome-noise heteroskedasticity.

## Worked intuition (Gaussian)

Take one $C$-stratum. Let $Y = \mu(x) + \gamma + \varepsilon_Y$ and $W = 0.5 + 2\gamma + \varepsilon_W$ with $\varepsilon_W\sim N(0,0.9)$. In arm 0, $\gamma\sim N(0.4, 0.5)$; in arm 1, $\gamma\sim N(1.3, 1.4)$. Set $\mu(1)-\mu(0)=0.8$, so the true $\theta = 0.80$, with $P(X=1)=0.5$.

Here $b_W = 2$, so $\rho = 2$. The location shift is $D_\gamma = 1.3 - 0.4 = 0.9$, and the dispersion shift is $D_v = 1.4 - 0.5 = 0.9 \neq 0$, satisfying A8. The mean contrasts give $a = \theta + D_\gamma = 0.8 + 0.9 = 1.7$ and $b = b_W D_\gamma = 2\times 0.9 = 1.8$. With constant outcome noise, $\mathrm{Cov}(W,Y\mid x) = b_W\mathrm{Var}(\gamma\mid x)$ and $\mathrm{Var}(Y\mid x) = \mathrm{Var}(\gamma\mid x)+\mathrm{Var}(\varepsilon_Y)$, so the cross-arm ratio returns $\rho = 2$ and $\theta = a - b/\rho = 1.7 - 1.8/2 = 1.7 - 0.9 = 0.8$. The mechanism is tangible: the proxy's covariance climbs with the confounder spread, the outcome variance climbs in lockstep, and their slopes against arm differ only by the scale $b_W$. The next section shows why this clean reading collapses when outcome noise itself depends on $\gamma$.

## Soundness audit

The adversary produced a patchable counterexample: two structural models in one $C$-stratum that satisfy every listed assumption, agree on the relevant means and covariances, yet drive the published functional away from the truth.

Common structure: $Y = \mu(x) + \gamma + \varepsilon_Y$, $W = 0.5 + 2\gamma + \varepsilon_W$, $\varepsilon_W\sim N(0,0.9)$. Arm 0: $\gamma\sim N(0.4,0.5)$. Arm 1: $\gamma\sim N(1.3,1.4)$. $\mu(1)-\mu(0)=0.8$, so true $\theta = 0.80$, $P(X=1)=0.5$.

Model M1 (homoskedastic outcome noise): $\mathrm{Var}(\varepsilon_Y\mid\gamma)=0.7$ constant.

Model M2 (heteroskedastic outcome noise, still satisfying every assumption): $\mathrm{Var}(\varepsilon_Y\mid\gamma)=0.7 + k\gamma^2$ with $k>0$. This obeys $\varepsilon_Y\perp\!\!\!\perp X\mid (\gamma,C)$ because the noise law depends only on $\gamma$, not on $X$. It obeys A4 because $U$ enters the $Y$ mean only through $b_Y\gamma$. It obeys A3 (NCO validity). No listed assumption excludes it, since latent ignorability and additivity constrain only the conditional mean structure, never the conditional variance of the idiosyncratic term as a function of $\gamma$.

Effect on the observed law. The covariance $\mathrm{Cov}(W,Y\mid x) = b_W\mathrm{Var}(\gamma\mid x)$ is identical in M1 and M2, because the $\gamma$-heteroskedastic noise is mean-zero given $\gamma$ and so does not enter the covariance. But
$$\mathrm{Var}(Y\mid X=x,C) = \mathrm{Var}(\gamma\mid x) + E[\mathrm{Var}(\varepsilon_Y\mid\gamma)\mid x] = \mathrm{Var}(\gamma\mid x) + 0.7 + k\big(\mathrm{Var}(\gamma\mid x) + E[\gamma\mid x]^2\big)$$
is inflated in M2, and the inflation differs across arms because $E[\gamma\mid x]$ and $\mathrm{Var}(\gamma\mid x)$ differ across arms. Hence the published $\rho = (\mathrm{Cov}^1-\mathrm{Cov}^0)/(\mathrm{Var}(Y)^1-\mathrm{Var}(Y)^0)$ is biased and the published $\theta = a - b/\rho$ is wrong.

Monte Carlo with $N=6\times 10^6$ and true $\theta = 0.80$: $k=0$ gives the published functional $0.7993$; $k=0.6$ gives $-0.6568$; $k=1.2$ gives $-2.1114$. The functional is decisively broken by a structurally permitted perturbation that no listed assumption rules out, and the claimed test (observed outcome heteroskedasticity) cannot tell it apart from genuine confounder-dispersion change, because both inflate $\mathrm{Var}(Y\mid X)$.

Severity: patchable, not point-falsifying. The estimand $\theta$ stays identified from the full observed law. The unused moment $\mathrm{Var}(W\mid x)$ pins the scale, because NCO validity forces constant proxy noise: $b_W = (\mathrm{Var}(W\mid 1)-\mathrm{Var}(W\mid 0))/(\mathrm{Cov}(W,Y\mid 1)-\mathrm{Cov}(W,Y\mid 0))$, giving $\theta = a - b\,(\mathrm{Cov}(W,Y\mid 1)-\mathrm{Cov}(W,Y\mid 0))/(\mathrm{Var}(W\mid 1)-\mathrm{Var}(W\mid 0))$, verified to return $0.80$ for $k=0,0.6,1.2$. So M1 and M2 cannot simultaneously match the full observed law including $\mathrm{Var}(W)$ and differ in $\theta$. They only break the specific $\mathrm{Var}(Y)$-based functional the idea publishes.

## Honest status and the fix

Final status: ELIMINATED as a standalone point-ID claim. The published bridge that anchors the scale on $\mathrm{Var}(Y\mid X)$ is incorrect, and the advertised testability (read arm-varying confounder dispersion off observed outcome heteroskedasticity) must be retracted, because observed outcome heteroskedasticity conflates a genuine $\mathrm{Var}(\gamma)$ shift with a $\gamma$-heteroskedastic-noise shift.

The repair, which is the sibling idea (idea 5), replaces the outcome-variance anchor with the proxy-variance anchor:
$$b_W(C) = \frac{\mathrm{Var}(W\mid 1,C) - \mathrm{Var}(W\mid 0,C)}{\mathrm{Cov}(W,Y\mid 1,C) - \mathrm{Cov}(W,Y\mid 0,C)},$$
$$\theta(C) = \big(E[Y\mid 1,C]-E[Y\mid 0,C]\big) - \big(E[W\mid 1,C]-E[W\mid 0,C]\big)\,\frac{\mathrm{Cov}(W,Y\mid 1,C)-\mathrm{Cov}(W,Y\mid 0,C)}{\mathrm{Var}(W\mid 1,C)-\mathrm{Var}(W\mid 0,C)}.$$
This never touches $\mathrm{Var}(Y)$, so it is immune to $\gamma$-heteroskedastic outcome noise. It relies only on listed assumptions, because NCO validity forces $\mathrm{Var}(\varepsilon_W)$ constant across arms. It uses the same anchor, arm-varying confounder dispersion, now read through $\mathrm{Cov}(W,Y\mid 1)\neq\mathrm{Cov}(W,Y\mid 0)$ and $\mathrm{Var}(W\mid 1)\neq\mathrm{Var}(W\mid 0)$, which are pure functionals of $(X,W,Y)$.

Two secondary repairs. First, the testable surrogate for A8 is no longer outcome heteroskedasticity but $\mathrm{Cov}(W,Y\mid 1)\neq\mathrm{Cov}(W,Y\mid 0)$ together with $\mathrm{Var}(W\mid 1)\neq\mathrm{Var}(W\mid 0)$. Second, the overidentification remark must be recast honestly: the system is overidentified precisely because $\mathrm{Var}(W)$, $\mathrm{Cov}(W,Y)$, and $\mathrm{Var}(Y)$ give two competing scale estimates, and their equality is the falsifiable specification test for constant outcome noise. Reporting both $b_W = (\mathrm{Var}(W)\text{-diff})/(\mathrm{Cov}\text{-diff})$ and $b_W = (\mathrm{Cov}\text{-diff})/(\mathrm{Var}(Y)\text{-diff})$ and testing their equality is the correct check. As a cautionary mechanism, this note records that putting the heteroskedasticity anchor on the noisy side (the outcome) is the failure mode, and that moving it to the structurally protected side (the proxy) is the salvage.

## Novelty vs prior work

The single closest prior work is Sofer, Wang, Richardson and Tchetgen Tchetgen (2016), negative outcome control as a generalization of difference-in-differences. They use a single NCO with additive equi-confounding and a closed form, but the calibration scale is either set to one (parallel trends) or estimated from a known/assumed scale ratio using first moments. The intended delta was to identify the scale internally from second-moment cross-arm contrasts under one testable condition. The delta survives the audit only on the proxy side: the surviving claim is internal scale identification from $\mathrm{Var}(W\mid X)$ and $\mathrm{Cov}(W,Y\mid X)$ cross-arm contrasts, not from $\mathrm{Var}(Y\mid X)$. Against Sofer et al. the scale is not assumed or externally calibrated; the precise published outcome-side equation, however, is the part this note retracts.

## AISTATS significance and positioning

The positive content for a venue like AISTATS is a clean message about where a heteroskedasticity-based anchor can and cannot live in a single-proxy graph. The result that the outcome-variance anchor is corruptible while the proxy-variance anchor is protected is itself a contribution, because it draws a sharp line through a natural family of single-proxy point-ID attempts and explains, structurally, why one side works and the other does not. The honest framing pairs a cautionary outcome-side mechanism with a working proxy-side identifier and a built-in specification test (equality of the two competing scale estimates), which is exactly the kind of identify-plus-falsify package that referees reward. The outcome-side route, presented as the eliminated precursor, motivates the proxy-side identifier and gives the paper its narrative spine.

## Open problems and next steps

First, establish the efficient influence function and the multiply robust estimator for the proxy-side functional, including the weak-anchor regime where $\mathrm{Var}(W\mid 1)-\mathrm{Var}(W\mid 0)$ is small, with an Anderson-Rubin-style interval in $b_W$. Second, characterize the sharp partial-ID set when A8 is weak or doubtful, tracing $\theta(\rho)$ over admissible $\rho$. Third, relax single-factor additivity toward low-rank or mildly nonlinear confounding and ask how much of the proxy-variance anchor survives. Fourth, formalize the specification test that compares the two scale estimates and study its power against $\gamma$-heteroskedastic outcome noise. Fifth, connect the proxy-side anchor to Lewbel-style heteroskedasticity identification to see whether the confounder-spread channel admits a unified treatment with internal-instrument constructions.

## References

- Sofer, T., Wang, L., Richardson, T. S., and Tchetgen Tchetgen, E. J. (2016). Negative outcome control as a generalization of difference-in-differences. Journal of the Royal Statistical Society, Series B.
- Tchetgen Tchetgen, E. J. (2014). The Control Outcome Calibration Approach (COCA). American Journal of Epidemiology.
- Kuroki, M., and Pearl, J. (2014). Measurement bias and effect restoration in causal inference. Biometrika.
- Park, C., Richardson, T. S., and Tchetgen Tchetgen, E. J. (2024). Single Proxy Control. Biometrics.
- Miao, W., Geng, Z., and Tchetgen Tchetgen, E. J. (2018). Identifying causal effects with proxy variables of an unmeasured confounder. Biometrika.
- Cui, Y., et al. (2024). Semiparparametric proximal causal inference. Journal of the Royal Statistical Society, Series B.
- Lewbel, A. (2012). Using heteroscedasticity to identify and estimate mismeasured and endogenous regressor models. Journal of Business and Economic Statistics.
- Difference-in-differences / parallel-trends literature, the $\lambda=1$ special case of the present functional.
