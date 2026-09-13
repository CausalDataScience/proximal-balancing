# Location-Shift Confounding Control: Single-Proxy Identification via Additively Separable $U$ and a Calibrated Proxy Slope

Date: 2026-06-28
Status: sharp partial-ID only. Point-ID claim collapses to difference-in-differences. The reason: at the contrast level the system has 2 observable numbers and 3 structural unknowns ($\theta$, the confounder shift, and the gauge $\lambda$); the observed law never reveals $\lambda$, so $\lambda$ carries all of the identifying power from outside the data. Point identification holds only in the special case $\lambda(C)=1$, which is parallel trends, and there the estimand is exactly difference-in-differences.
Rank among the 9: #5 of 9.
Mechanism family: location-shift / additive-separability structural restriction that replaces the second proxy's completeness with a finite-dimensional calibration scalar.
Tags: single-proxy, proximal causal inference, negative control outcome, location-shift, difference-in-differences, sharp partial identification, sensitivity analysis, completeness replacement.

## One-line idea

> If the unmeasured confounder shifts both the outcome mean and a single proxy mean by the same location amount (up to a scalar gauge $\lambda$), then the treatment-induced change in the proxy mean reveals the treatment-induced change in the confounder distribution, and subtracting a calibrated multiple of it from the crude outcome contrast removes confounding bias. The catch: $\lambda$ is not recoverable from the observed law, so point identification requires fixing $\lambda$ from outside the data, and $\lambda=1$ is just difference-in-differences.

## Glossary

Every symbol used in this note is defined here before it appears in the argument.

- $X$: the treatment (or exposure) variable. We treat it as binary, $X \in \{0,1\}$, unless a dose-response is stated.
- $Y$: the outcome variable.
- $U$: the unmeasured (latent) confounder. It is never observed in the data.
- $W$: the single proxy of $U$. It is one observed variable that carries information about $U$. There is exactly one such proxy. There is no second, treatment-side proxy.
- $C$: observed baseline covariates, measured before treatment.
- $Y(x)$: the potential outcome under the intervention that sets $X=x$ (the do-operator). Consistency means $Y = Y(X)$.
- ATE: the average treatment effect, here $\theta = E[Y(1)] - E[Y(0)]$.
- Dose-response: the curve $\beta(x) = E[Y(x)]$ across treatment levels $x$.
- $\theta(C)$: the conditional treatment effect given covariates, $E[Y(1) - Y(0)\mid C]$. Then $\theta = E_C[\theta(C)]$.
- Proxy (negative control outcome, NCO): an observed surrogate $W$ of $U$ that is NOT caused by treatment and does NOT cause treatment or outcome. It moves only because $U$ moves.
- $\gamma(U,C)$: the one-dimensional confounding summary, the way $U$ enters the outcome additively. This is the latent nuisance the method must control for.
- $\mu(x,C)$: the structural (counterfactual) outcome mean under $do(X=x)$, with $U$ marginalized in the form $E[Y\mid do(x),C] = \mu(x,C) + E[\gamma(U,C)\mid C]$.
- $\lambda(C)$: the scalar gauge, the slope with which $\gamma(U,C)$ enters the proxy mean $E[W\mid U,C]$. The crux object of this note.
- $\alpha(C)$: the proxy intercept, $E[W\mid U,C] = \alpha(C) + \lambda(C)\gamma(U,C)$. It cancels in every contrast and is therefore harmless.
- $\Delta\gamma(x,C)$: the treatment-induced confounder shift, $E[\gamma(U,C)\mid X=x,C] - E[\gamma(U,C)\mid X=0,C]$. This is the bias term.
- $\delta_W(x,C)$: the observable proxy contrast, $E[W\mid X=x,C] - E[W\mid X=0,C]$.
- Completeness: a statistical injectivity condition. In two-proxy proximal causal inference, completeness of the treatment-side proxy in $U$ is what makes the relevant integral (Fredholm) equation uniquely solvable for the bridge function. It is the condition this idea tries to replace.
- Bridge function / bridge equation: in proximal inference, a function $b$ solving $E[b(W)\mid Z,X,C] = E[Y\mid Z,X,C]$; here the analogue is the closed-form contrast that maps the observed law to $\theta$.
- $e(C)$: the propensity score $P(X=1\mid C)$, used by the efficient estimator.
- DiD: difference-in-differences, the panel-data design whose identifying assumption is parallel trends.

## Problem and motivation

When an unmeasured confounder $U$ affects both treatment $X$ and outcome $Y$, the naive regression contrast $E[Y\mid X=1,C] - E[Y\mid X=0,C]$ is biased for the causal effect, because part of the difference reflects that treated and untreated units differ in $U$. Standard adjustment fails since $U$ is never measured.

Proximal causal inference solves this by using two proxies of $U$: one negative-control outcome and one negative-control exposure. Point identification then rests on a completeness condition, an injectivity property of an integral operator that the second proxy supplies. Completeness is strong, untestable, and demands two valid proxies, which are often hard to find.

The single-proxy regime is the interesting frontier precisely because one proxy is far more common in practice (for example, a single pre-treatment measurement) yet a lone proxy generically cannot deliver nonparametric point identification. The question this idea targets is whether a mild semiparametric structural restriction can supply the identifying power that the second proxy's completeness normally extracts, so that one proxy suffices. The honest answer, developed below, is that the restriction shifts the burden onto a single scalar that the data cannot reveal.

## Setup and causal graph

The vertices are $X$ (treatment), $Y$ (outcome), $U$ (unmeasured confounder), $W$ (single proxy), and $C$ (observed baseline covariates). The directed edges are:

- $U \to X$, $U \to Y$, $U \to W$ (confounder drives treatment, outcome, and proxy)
- $X \to Y$ (the causal effect of interest)
- $C \to X$, $C \to Y$, $C \to U$, $C \to W$ (baseline covariates)

The forbidden edges carry the structural content. There is no $X \to W$ edge: treatment does not cause the proxy, so $W$ is a negative control outcome. There is no $W \to Y$ and no $W \to X$ edge: the proxy is not a cause of outcome or treatment. Thus $U$ is the only non-$C$ parent of $W$ that is also a parent of $(X,Y)$. "Single proxy" means there is exactly one $W$ and no second, treatment-side proxy $Z$.

```
        C
     /  | | \
    v   v v  v
    X   U .  W
    |  /| \   ^
    | / |  \  |
    vv  |   \ |
    Y<--+    (U->W)
         (U->Y, U->X)

Edges: U->X, U->Y, U->W, X->Y,
       C->X, C->Y, C->U, C->W.
Forbidden: X->W, W->Y, W->X.
U is UNMEASURED. W is the single proxy (negative control outcome).
```

## Target estimand

The target is the average treatment effect and, more generally, the dose-response curve. Writing the structural outcome mean as $E[Y\mid do(x),C] = \mu(x,C) + E[\gamma(U,C)\mid C]$, the conditional contrast is

$$\theta(C) = \mu(1,C) - \mu(0,C) = E[Y(1) - Y(0)\mid C],$$

and the marginal target is

$$\theta = E_C[\theta(C)] = E[Y(1)] - E[Y(0)].$$

The general dose-response is $\beta(x) = E[Y(x)]$ with conditional version $\mu(x,C) - \mu(0,C)$ averaged over $C$.

## Assumptions

Each assumption is stated formally, read in plain language, justified for mildness, and labeled testable or not.

1. Consistency and positivity. Formal: $Y = Y(X)$ and $0 < P(X=x\mid C) < 1$ almost surely. Plain: observed outcomes equal potential outcomes at the realized treatment, and both treatment arms occur at every covariate value. Mildness: standard, assumed by every method in the comparison set including two-proxy proximal and COCA. Testable: positivity is partly checkable from data; consistency is not.

2. Latent ignorability given $(U,C)$. Formal: $Y(x) \perp\!\!\!\perp X \mid U, C$ for all $x$. Plain: once we know $U$ and $C$, treatment is as good as randomly assigned, so $U,C$ block all backdoor paths. Mildness: this is exactly the proximal / negative-control premise; it adds no cost relative to prior work. Testable: no, because $U$ is unobserved.

3. Negative-control proxy (no anticipation). Formal: $X \not\to W$ and $W \not\to (X,Y)$; equivalently $W \perp\!\!\!\perp X \mid (U,C)$ and $W$ has no causal effect on $Y$ or $X$. Plain: treatment does not move the proxy except through $U$, and the proxy is not itself a cause. Mildness: identical to the validity requirement on a single NCO in COCA and Single Proxy Control; $W$ can be a pre-treatment repeated measure. Testable: the conditional independence given $(U,C)$ is not testable since $U$ is latent.

4. Additive (location-shift) outcome in $U$. Formal: $E[Y\mid X=x,U,C] = \mu(x,C) + \gamma(U,C)$, so the contrast $\mu(1,C) - \mu(0,C)$ does not depend on $U$. Plain: $U$ shifts the outcome level but does not modify the size of the treatment effect on the additive scale. Mildness: this is the no-effect-modification-by-$U$ assumption used throughout DiD and parallel-trends designs, and it is weaker than COCA's rank preservation (which assumes constant individual effects). Testable: not directly, since it constrains a latent regression.

5. Calibrated proxy gauge. Formal: $E[W\mid U,C] = \alpha(C) + \lambda(C)\gamma(U,C)$ with $\lambda(C) \neq 0$ known or anchored. Plain: the proxy tracks the same one-dimensional confounding summary $\gamma$ that contaminates $Y$, with slope $\lambda$. Mildness: when $W$ is a lagged measurement of $Y$ in the same units, $\lambda=1$ automatically; otherwise a placebo or validation stratum is needed to pin $\lambda$. Testable: the form is not testable; critically, $\lambda$ itself is NOT identified from the observed law (this is the failure documented below).

6. Proxy relevance. Formal: $\mathrm{Cov}(W,U\mid C) \neq 0$, equivalently $\lambda(C) \neq 0$ and $\gamma(\cdot,C)$ non-degenerate. Plain: the proxy actually responds to the confounder. Mildness: the single-proxy analogue of proxy relevance; if it fails, $U$ is undetectable and no method can help. Testable: indirectly, through observed association of $W$ with treatment within strata.

7. Treatment-induced confounder shift is detectable. Formal: $E[\gamma(U,C)\mid X=x,C]$ varies with $x$. Plain: treated and untreated units genuinely differ in the confounder. Mildness: this is the bias being corrected; if it is zero, naive regression already works. Testable: not directly, but $\delta_W \neq 0$ is its observable footprint.

## The single mild assumption that replaces the second proxy

Assumptions 4 and 5 together are the structural restriction that is meant to stand in for the second proxy. In two-proxy proximal inference, the second (treatment-side) proxy $Z$ supplies an L2-completeness condition: completeness of $Z$ in $U$ is exactly what makes the outcome-bridge Fredholm integral equation $E[b(W)\mid Z,X,C] = E[Y\mid Z,X,C]$ uniquely solvable. That completeness is the price of recovering the confounder's influence as an unknown function.

The location-shift restriction is supposed to pay that price differently. Under additivity (Assumption 4) the only latent quantity that contaminates the treatment contrast is the scalar shift $\Delta\gamma(x,C)$ in each $(x,C)$ cell, not the whole function $\gamma(\cdot)$. The calibrated gauge (Assumption 5) ties the proxy to that same scalar via a known slope. So the second proxy's completeness is replaced by: (additive outcome model) + (a finite-dimensional, anchorable scalar $\lambda(C)$). Instead of inverting an integral operator, identification is meant to reduce to one scalar equation per cell. The soundness audit below shows this replacement is incomplete: the scalar $\lambda$ that does the work is not pinned by the data.

## Identification result

Claim (as proposed). Define the observable proxy contrast $\delta_W(x,C) = E[W\mid X=x,C] - E[W\mid X=0,C]$. Under Assumptions 1 through 7, because $X \not\to W$, all of $\delta_W$ comes from the confounder-distribution difference:

$$\delta_W(x,C) = \lambda(C)\,\big\{ E[\gamma(U,C)\mid X=x,C] - E[\gamma(U,C)\mid X=0,C] \big\} = \lambda(C)\,\Delta\gamma(x,C).$$

The crude outcome contrast is

$$E[Y\mid X=x,C] - E[Y\mid X=0,C] = \big[\mu(x,C) - \mu(0,C)\big] + \Delta\gamma(x,C).$$

Eliminating $\Delta\gamma$ gives the bridge equation / identifying functional:

$$\mu(x,C) - \mu(0,C) = \big\{ E[Y\mid X=x,C] - E[Y\mid X=0,C] \big\} - \frac{1}{\lambda(C)}\big\{ E[W\mid X=x,C] - E[W\mid X=0,C] \big\},$$

and the marginal target is

$$\theta = E_C\!\left[ \big\{ E[Y\mid X=1,C] - E[Y\mid X=0,C] \big\} - \frac{1}{\lambda(C)}\big\{ E[W\mid X=1,C] - E[W\mid X=0,C] \big\} \right].$$

Everything on the right-hand side is a function of the observed distribution of $(Y,W,X,C)$ plus the gauge $\lambda(C)$. No integral operator is inverted and no completeness is invoked. The decisive qualifier, established in the soundness audit, is that $\theta$ is point-identified only once $\lambda(C)$ is known from outside the observed law.

## Proof sketch

The proof has four steps. The single proxy enters at Step 2, which is flagged explicitly.

Step 1 (decompose $Y$). Input: Assumption 2 (latent ignorability) and Assumption 4 (additive outcome). Derive: $E[Y\mid X=x,C] = E_U[ E[Y\mid X=x,U,C] \mid X=x,C ] = \mu(x,C) + E[\gamma(U,C)\mid X=x,C]$. Subtracting the $x=0$ arm yields $E[Y\mid X=x,C] - E[Y\mid X=0,C] = (\mu(x,C) - \mu(0,C)) + \Delta\gamma(x,C)$. Justification: iterated expectation plus the additive form; no $U \times X$ interaction means $\mu(x,C) - \mu(0,C)$ equals the causal contrast $E[Y(x) - Y(0)\mid C]$.

Step 2 (read the confounder shift off the single proxy). Input: Assumption 3 ($X \not\to W$, so $W \perp\!\!\!\perp X \mid (U,C)$) and Assumption 5 (calibrated gauge). This is where the single proxy enters. Derive: $E[W\mid X=x,C] = E_U[ E[W\mid U,C] \mid X=x,C ] = \alpha(C) + \lambda(C)\,E[\gamma(U,C)\mid X=x,C]$. Hence $\delta_W(x,C) = \lambda(C)\,\Delta\gamma(x,C)$, and the intercept $\alpha(C)$ cancels in the difference, so the unknown proxy intercept is irrelevant. Justification: because $W$ depends on $X$ only through $U$, the entire treatment-arm difference in $W$ is the $U$-distribution difference, scaled by $\lambda$.

Step 3 (solve the scalar equation). Input: Step 2 and Assumption 6 ($\lambda(C) \neq 0$). Derive: $\Delta\gamma(x,C) = \delta_W(x,C) / \lambda(C)$. Substitute into Step 1 to get $\mu(x,C) - \mu(0,C) = (E[Y\mid X=x,C] - E[Y\mid X=0,C]) - \delta_W(x,C)/\lambda(C)$. Justification: one scalar unknown, one scalar equation, provided $\lambda$ is known.

Step 4 (target). Input: Step 3. Derive: average over $C$ to obtain $\theta$. Justification: the map from observed law to $\theta$ is a difference of regression contrasts, continuous, with no operator inversion, hence well posed conditional on $\lambda$.

The hidden gap. Steps 3 and 4 both require $\lambda(C)$ as a known input. The proof never derives $\lambda(C)$ from the observed law. The soundness audit shows it cannot be so derived.

## Why a single proxy suffices (the crux argument, and where it breaks)

In two-proxy proximal inference the confounder's effect on $Y$ is an unknown function $\gamma(U)$, and recovering it requires inverting an integral operator whose injectivity is the completeness condition the second proxy supplies. The location-shift restriction is meant to remove the need to recover $\gamma(\cdot)$ as a function: under additivity the only thing that contaminates the $X$-contrast is the scalar $\Delta\gamma(x,C)$ per cell, and one calibrated proxy provides exactly the one equation $\delta_W = \lambda\,\Delta\gamma$ that pins that scalar. The structural restriction is supposed to convert an ill-posed inverse problem into a just-identified algebraic one, with $\lambda(C)$ playing the role of the second proxy's loadings.

This is where the argument breaks. The "one equation" $\delta_W = \lambda\,\Delta\gamma$ has two unknowns on its right side, $\lambda$ and $\Delta\gamma$. The observed law supplies $\delta_W$, not the split between $\lambda$ and $\Delta\gamma$. So the just-identified count is illusory: one observed number cannot resolve two latent numbers. The gauge $\lambda$ is a property of the latent map $E[W\mid U,C]$, and since $U$ is unobserved while $\alpha(C)$ absorbs the location, nothing in the joint law of $(X,W,Y,C)$ reveals $\lambda$. The "richness" that completeness extracts from a second proxy is not actually supplied by the assumed functional form; it is supplied by fixing $\lambda$ from outside the data.

## Estimator

Conditional on a value or interval for $\lambda(C)$, estimation is straightforward. Plug-in: (1) estimate the four conditional means $m_Y^x(C) = E[Y\mid X=x,C]$ and $m_W^x(C) = E[W\mid X=x,C]$ by any flexible regression (gradient boosting, kernels, or series). (2) Form $\hat\theta(C) = (m_Y^1 - m_Y^0) - (m_W^1 - m_W^0)/\hat\lambda(C)$. (3) Average: $\hat\theta = \mathbb{P}_n \hat\theta(C)$.

Efficiency and inference. The functional is a difference of regression contrasts, so a one-step / AIPW correction is available. With $e(C) = P(X=1\mid C)$, the influence function has the form

$$\psi = \left( \frac{\mathbb{1}(X=1)}{e(C)} - \frac{\mathbb{1}(X=0)}{1 - e(C)} \right)\left( Y - \frac{1}{\lambda(C)} W \right) - \text{(regression-adjustment terms)},$$

which yields a doubly robust, cross-fit, root-$n$, asymptotically normal estimator under a product-of-rates condition on the nuisances, matching the inferential quality of proximal DR estimators but with one proxy.

Handling $\lambda$. If $W$ is a repeated pre-period measurement of $Y$ in the same units, set $\lambda = 1$. Otherwise, $\lambda$ must come from an external source (a placebo stratum with a known null $X$-effect, or a validation anchor). Since this external source is a second identifying input, the honest deliverable is a sensitivity analysis: report $\theta(\lambda) = a - b/\lambda$ as $\lambda$ ranges over a user-specified interval, where $a$ and $b$ are the marginal $Y$- and $W$-contrasts. That interval is exactly the sharp partial-identification set.

## Worked intuition (binary / Gaussian example)

Drop $C$ and let $\gamma(U) = U$ with linear-Gaussian structure. Suppose the observed law is: $P(X=1) = 0.5$; in arm $X=1$, $E[Y]=4$ and $E[W]=1$; in arm $X=0$, $E[Y]=-1$ and $E[W]=-1$; and within each arm $\mathrm{Var}(Y) = 1.49$, $\mathrm{Var}(W) = 3.56$, $\mathrm{Cov}(W,Y) = 1$. Then the observed contrasts are $a = E[Y\mid 1] - E[Y\mid 0] = 5$ and $b = E[W\mid 1] - E[W\mid 0] = 2$.

If you believe $\lambda = 1$ (the proxy moves one-for-one with the confounder), the corrected effect is $\theta = a - b/\lambda = 5 - 2/1 = 3$. The proxy says the treated arm is $2$ units higher in $U$, that $2$ contaminates $Y$ directly, so removing it from the crude $5$ leaves $3$ as the causal effect.

If instead you believe $\lambda = 2$ (the proxy is twice as sensitive to $U$ as $Y$ is), the same observed $b=2$ now implies only a $\Delta\gamma = 1$ unit shift in $U$, so $\theta = 5 - 2/2 = 4$. The mechanism is tangible: the proxy tells you how far apart the arms are in $U$ only after you commit to how loudly the proxy speaks, which is $\lambda$. The data fixes what the proxy says ($b=2$); it does not fix how to translate that into confounder units ($\lambda$).

## Soundness audit

The adversary produced an explicit, numerically verified counterexample. Severity: patchable. The construction exhibits two valid structural causal models with identical observed law of $(X,W,Y)$ but different $\theta$.

Common observed law (linear-Gaussian, so the listed moments pin the full joint): $P(X=1) = 0.5$; arm $X=1$ has $E[Y]=4$, $E[W]=1$; arm $X=0$ has $E[Y]=-1$, $E[W]=-1$; both arms have $\mathrm{Var}(Y)=1.49$, $\mathrm{Var}(W)=3.56$, $\mathrm{Cov}(W,Y)=1$. Hence $a=5$, $b=2$.

Model M1 (gauge $\lambda=1$): $U\mid X=1 \sim N(1,1)$, $U\mid X=0 \sim N(-1,1)$, so $\Delta\gamma = 2$. Outcome $E[Y\mid x,U] = \mu(x) + U$ with $\mu(1)-\mu(0) = \theta_1 = 3$. Proxy $E[W\mid U] = 0 + 1\cdot U$. Noise sds $0.7$ (outcome) and $1.6$ (proxy). Then $a = \theta_1 + \Delta\gamma = 3 + 2 = 5$ and $b = \lambda\,\Delta\gamma = 1\cdot 2 = 2$. The true effect is $\theta_1 = 3$.

Model M2 (gauge $\lambda=2$): $U\mid X=1 \sim N(0.5, 1/2)$, $U\mid X=0 \sim N(-0.5, 1/2)$, so $\Delta\gamma' = 1$. Outcome $E[Y\mid x,U] = \mu'(x) + U$ with $\mu'(1)-\mu'(0) = \theta_2 = 4$. Proxy $E[W\mid U] = 0 + 2\cdot U$. Noise sds $0.995$ (outcome) and $1.249$ (proxy), both real and positive, so the SCM is valid. Then $a = \theta_2 + \Delta\gamma' = 4 + 1 = 5$ and $b = \lambda\,\Delta\gamma' = 2\cdot 1 = 2$. The true effect is $\theta_2 = 4$.

Every observed moment (arm probabilities, $E[Y\mid x]$, $E[W\mid x]$, $\mathrm{Var}(Y)$, $\mathrm{Var}(W)$, $\mathrm{Cov}(W,Y)$ in both arms) matches between M1 and M2 to machine precision, yet $\theta_1 = 3 \neq \theta_2 = 4$. Both models satisfy every stated assumption: consistency and positivity; latent ignorability ($U$ is the only confounder); negative-control proxy ($W$ is generated from $U$ alone, no $X$ path); additive outcome with no $U\times X$ interaction; calibrated-gauge proxy form $E[W\mid U] = \alpha + \lambda U$; proxy relevance ($\lambda \neq 0$); and a nonzero treatment-induced shift. They differ only in $\lambda$, a property of the latent map $E[W\mid U]$ that the observed law cannot see, because $U$ is unobserved and $\alpha$ absorbs the location.

Root cause (counting). At the contrast level there are 2 observable numbers $(a,b)$ but 3 structural unknowns $(\theta, \Delta\gamma, \lambda)$. The system $a = \theta + \Delta\gamma$, $b = \lambda\,\Delta\gamma$ is underdetermined unless $\lambda$ is fixed from outside the observed law. The bridge functional $\theta = a - b/\lambda$ returns $5 - 2/1 = 3$ or $5 - 2/2 = 4$ purely as a function of the assumed $\lambda$. The proposal's claim that $\lambda$ is "identified up to sign by an anchor" is doing 100 percent of the identification, and that anchor is not supplied by the single proxy.

Why the anchoring routes do not rescue point-ID from one proxy. The placebo-stratum and validation-anchor routes require external data, which is a second identifying source, contradicting the "single proxy, no second source" framing. The $\lambda=1$ route ("$W$ is a lagged measurement of $Y$ in the same units") silently imposes a second structural assumption: $U$ loads on the pre-period outcome with the same coefficient as on the post-period outcome. That is exactly parallel trends, and under it $\theta = (E[Y\mid 1] - E[Y\mid 0]) - (E[W\mid 1] - E[W\mid 0])$ is identified, but the method has collapsed into difference-in-differences with the pre-period outcome as the proxy.

## Honest status and the fix

Final status: sharp partial-ID only. Point identification holds only in the $\lambda(C)=1$ special case, where it is difference-in-differences. Rank #5 of 9.

The repair path has three options. Option 1: restrict to $\lambda(C)=1$ and present the result honestly as a proxy-based, cross-sectional difference-in-differences identity. There $\theta = (Y\text{-contrast}) - (W\text{-contrast})$ is genuinely point-identified, but the novelty claim must be downgraded from "replaces completeness, beats two-proxy and Kuroki-Pearl" to "a proxy-based DiD with a doubly robust cross-fit estimator and a built-in falsification test." Option 2 (recommended primary contribution): keep general $\lambda$, drop the point-ID claim, and present sharp partial identification. As $\lambda$ ranges over a user-specified interval $[\lambda_{\mathrm{lo}}, \lambda_{\mathrm{hi}}]$ with $\lambda > 0$, the curve $\theta(\lambda) = a - b/\lambda$ traces an interval. That set is exactly the counterexample family and is genuinely sharp, so the sensitivity-analysis framing is sound and honest. Option 3: supply a genuine within-observed-law anchor for $\lambda$, for instance an instrument-like variable or a second moment restriction tying $\lambda$ to observables without a second proxy or external validation. The proposal currently offers none, and the counting argument shows one is necessary.

The salvaging sibling is the sharp-partial-ID framing itself: ship Option 2 as the headline (sharp partial-ID plus sensitivity in $\lambda$) with Option 1 as the point-ID special case, explicitly acknowledging the DiD reduction.

## Novelty vs prior work

Closest prior work: Tchetgen Tchetgen, Park and Richardson (2024), "Single Proxy Control," Biometrics. That work also uses a single NCO but relies on completeness for nonparametric identification or on rank preservation. The proposed delta was to remove both via location-shift plus a calibrated gauge. The honest delta after the audit is smaller: the location-shift functional is essentially the negative-outcome-control identity of Sofer et al. (2016), with $1/\lambda$ playing the role of the ratio of confounding loadings $s_Y / s_N$, and point-ID holding only under the parallel-trends specialization. So relative to Single Proxy Control, the genuine contribution is not removing completeness for free, but reframing the single-proxy problem as a sharp sensitivity analysis over the unidentified gauge $\lambda$, with a doubly robust estimator and a built-in falsification test.

## AISTATS significance and positioning

The positioning that survives scrutiny is methodological honesty about a widely made but under-examined move. Many applied single-proxy analyses implicitly fix a gauge (often $\lambda=1$) without naming it, which silently imposes parallel trends. The contribution is to make that gauge explicit, to prove that it carries the entire identification burden via the 2-equations-3-unknowns counting argument, and to deliver the sharp partial-identification interval $\theta(\lambda) = a - b/\lambda$ together with a doubly robust, cross-fit estimator and a falsification test for the location-shift structure. Framed as sensitivity analysis rather than point identification, this is a clean, defensible AISTATS contribution. Framed as "single proxy beats two-proxy and Kuroki-Pearl," it is false and would not survive review.

## Open problems and next steps

1. Characterize when a within-observed-law anchor for $\lambda$ exists, for example via a third moment, heteroskedasticity in $W$, or an auxiliary variable that is conditionally independent of $\Delta\gamma$ but informative about $\lambda$.
2. Extend the sharp partial-ID interval to the dose-response curve $\beta(x)$ and to continuous treatments, where the counting argument generalizes to one gauge against many contrasts.
3. Develop the falsification test formally: in a stratum where $\gamma$ is known constant, Step 2 predicts $\delta_W = 0$, so a test of $\delta_W = 0$ there is a specification test for location-shift plus gauge.
4. Quantify the efficiency cost of the gauge uncertainty, deriving the width of the partial-ID interval as a function of the proxy signal-to-noise ratio.
5. Relate the gauge interval to standard DiD sensitivity tools (for example, honest parallel-trends bounds) so practitioners can import existing calibration intuitions.

## References

- Tchetgen Tchetgen, E. J., Park, C., and Richardson, D. (2024). Single Proxy Control. Biometrics.
- Miao, W., Geng, Z., and Tchetgen Tchetgen, E. J. (2018). Identifying causal effects with proxy variables of an unmeasured confounder. Biometrika.
- Cui, Y., et al. (2024). Semiparametric proximal causal inference. Journal of the Royal Statistical Society, Series B.
- Kuroki, M., and Pearl, J. (2014). Measurement bias and effect restoration in causal inference. Biometrika.
- Tchetgen Tchetgen, E. J. (2013). The control outcome calibration approach (COCA) for causal inference with unobserved confounding. American Journal of Epidemiology.
- Sofer, T., Richardson, D. B., Colicino, E., Schwartz, J., and Tchetgen Tchetgen, E. J. (2016). On negative outcome control of unobserved confounding as a generalization of difference-in-differences. Statistical Science.
- Pearl, J. (1995). Causal diagrams for empirical research (front-door criterion). Biometrika.
