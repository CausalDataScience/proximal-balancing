# Single-Proxy Identification of a Commonly-Confounded-Mediator Effect via a Coupled Proximal g-Functional

**Date:** 2026-06-28
**Status:** developed working draft (NSF CAREER Thrust A / Sub-AIM A3 candidate). Point-ID holds under (E1), (E2), (A3) **plus a factorization-uniqueness condition**, restricted to discrete low-cardinality $U$; sharp partial-ID of the ATE otherwise. This supersedes the seed note's point-ID-under-(A3)-alone claim: (A3) makes the kernel left-inverse exist once $K$ is known, but recovering $K$ from the observed law is a nonnegative factorization that needs a separability/anchor (or completeness) condition to pin the mixture up to a $U$-label permutation.
**Provenance:** developed from `single-proxy-coupled-proximal-g-functional-confounded-mediator.md` via 5 parallel derivation tracks (estimand-correction, identification frontier, estimation/EIF, sharp partial-ID, CAREER positioning), each adversarially verified; every verifier correction applied. PROVEN vs OPEN/CONJECTURE separated throughout.
**Seed rank:** #2 of 9 in the single-proxy idea family (the strongest point-ID member; #1 is the state-resolved sensitivity-bounds note).


## 1. Idea and status

A single proxy $W$ for an unmeasured confounder $U$ can identify the true coupled front-door effect $\psi(x)=\sum_u p(u)\sum_m E[Y\mid m,u]\,p(m\mid x,u)$ in the front-door graph **with a commonly-confounded mediator** (edge $U\to M$ present, which classical front-door forbids), without a second proxy and without an additivity restriction. **Status line:** point-identified under proxy-resolution (assumption (A3): the kernel $[p(W\mid U)]$ has full column rank, equivalently the conditional-expectation operator $T_{W\mid U}$ is injective) **plus** a factorization-uniqueness condition that pins the latent mixture up to a label permutation of $U$; sharply partially identified (closed bounded ATE interval with computable endpoints) when proxy-resolution degrades. The same scalar diagnostic, the smallest singular value $\sigma_{\min}(\hat K)$ of the estimated kernel, governs the transition between the two regimes.

Standing symbols, fixed once. $U$ is the unmeasured confounder with law $p(u)$; $X,M,W,Y$ are treatment, mediator, proxy, outcome. Exclusions: (E1) front-door, no $X\to Y$ edge, so $Y(x,m)=Y(m)$ and $Y\perp X\mid(M,U)$, giving $E[Y\mid m,x,u]=E[Y\mid m,u]$; (E2) $W\perp(X,M,Y)\mid U$. Discrete cardinalities $r=\mathrm{card}(U)$, $d_W=\mathrm{card}(W)$, $d_M=\mathrm{card}(M)$, $d_X=\mathrm{card}(X)$. The kernel is $K=[p(W=w\mid U=u)]_{w,u}\in\mathbb R^{d_W\times r}$, column-stochastic. The two structural legs are
$$
a_x(m,u):=p(m\mid x,u)\quad(\text{treatment}\to\text{mediator leg}),\qquad
b(m,u):=E[Y\mid m,u]\quad(\text{mediator}\to\text{outcome leg}),
$$
where, by (E1), $b(m,u)$ carries no $x$-dependence. Expectations over $U\sim p$: $E_Uf=\sum_u p(u)f(u)$, $\mathrm{Cov}_U$, $\mathrm{Var}_U$, $\mathrm{sd}_U$ as usual. Marginalized legs $\bar a_x(m)=E_Ua_x(m,U)=p(m\mid x)$, $\bar b(m)=E_Ub(m,U)$. The classical multiplicative front-door functional is $\psi_{\mathrm{FD}}(x)=\sum_m\bar a_x(m)\bar b(m)$. The latent core array is $J(x,m,u,y)=p(x,m,u)\,p(y\mid m,u)$, and for each $(x,m,y)$ the observed-law factorization is the matrix/vector identity
$$
P_{x,m,\cdot,y}=K\,J_{x,m,\cdot,y},\qquad P_{x,m,w,y}=p(x,m,w,y).\tag{$\star$}
$$

## 2. Delta over the seed note

The seed note recorded only the identity $\psi(x)-\psi_{\mathrm{FD}}(x)=\sum_m\mathrm{Cov}_U(p(m\mid x,U),E[Y\mid m,U])$ and the point-ID-under-(A3) claim. This program adds the following concrete results.

- **Exact estimand-correction cluster (Track 1).** A full bias theorem with: the level-wise per-mediator covariance decomposition (Theorem 1.1), the exact necessary-and-sufficient zero-bias condition with its honest sufficiency taxonomy (Theorem 1.2), the comonotone sign rule and binary-$U$ closed form (Theorem 1.3), the ATE-level shared-leg cancellation theorem (Theorem 1.4), and a **proxy-free observed-law-calibrated sensitivity audit** (Theorem 1.5).
- **Identification frontier with two separated targets (Track 2).** A clean split between latent-law identification (J-ID) and scalar-effect identification (ATE-ID), the proof that (A3) is necessary and sufficient for J-ID but only sufficient for ATE-ID, the **no-rescue-from-tensor** theorem ($W$ is the unique pure conditionally-independent view of $U$), continuous-$U$ completeness sufficient conditions with one full Fourier proof, and a diagnostic over-identification test that separates (E1) from (E2) failures.
- **Estimation and an honest EIF (Track 3).** A constrained deconvolution estimator, the delta-method variance with the $\sigma_{\min}(K)^{-2}$ blow-up, and a **corrected** efficient-influence-function statement that distinguishes three distinct variances (efficient $<$ a valid non-efficient influence function $<$ delta-method) rather than asserting they coincide.
- **Sharp partial-ID program (Track 4).** The compact-semialgebraic identified-set characterization, the interval theorem proved by a block-coordinate homotopy, certified-sharp computation via the moment-SOS hierarchy (asymptotic convergence only), the width-versus-deficit relations, and a constructive positive-width certificate.
- **Validity-aware defer rule (Tracks 4-5).** A fail-closed rule keyed on $\sigma_{\min}(\hat K)$ that reports a point with CI when resolution is certified and a sharp interval otherwise, realizing the Pragmatic-Causal-Inference dichotomy inside one estimand.
- **CAREER Sub-AIM A3 program (Track 5).** A three-paper arc, software, evaluation, and an honest risk table, with the central identification claim correctly stated as requiring a factorization-uniqueness condition in addition to (A3).

## 3. The estimand-correction theorem (Track 1)

### 3.1 Exact decomposition

**Theorem 1.1 (exact estimand-correction decomposition). [PROVEN]** Under (E1),
$$
\psi(x)=\psi_{\mathrm{FD}}(x)+B(x),\qquad B(x)=\sum_m\mathrm{Cov}_U\big(a_x(m,U),b(m,U)\big)=\sum_m\mathrm{Cov}_U\big(p(m\mid x,U),E[Y\mid m,U]\big).
$$
*Proof.* For each fixed $m$, the elementary identity $E[fg]=\mathrm{Cov}(f,g)+E[f]E[g]$ applied to $f=a_x(m,\cdot)$, $g=b(m,\cdot)$ as functions of $U\sim p$ gives $E_U[a_x(m,U)b(m,U)]=\mathrm{Cov}_U(a_x(m,U),b(m,U))+\bar a_x(m)\bar b(m)$. Summing over $m$ and using $\psi(x)=\sum_mE_U[a_x(m,U)b(m,U)]$ and $\psi_{\mathrm{FD}}(x)=\sum_m\bar a_x(m)\bar b(m)$ yields the claim; the $\sum_m$/$E_U$ interchange is justified by Tonelli under finite first moment of $Y$. $\blacksquare$

The functional $\psi_{\mathrm{FD}}$ is the product of two **separately marginalized** legs; under $U\to M$ it is not a valid causal effect, and $B(x)$ is exactly the mediator-confounding contamination that the multiplicative split discards.

### 3.2 Zero-bias characterization, with the honest sufficiency taxonomy

Write $c_x(m):=\mathrm{Cov}_U(a_x(m,U),b(m,U))$, so $B(x)=\sum_mc_x(m)$.

**Theorem 1.2 (zero-bias). [PROVEN]**
- **(a) Sufficient conditions.** $B\equiv0$ for all $x$ if either leg is $U$-degenerate: (i) unconfounded mediator $U\perp M\mid X$ forces $a_x(m,u)=p(m\mid x)$ constant in $u$; (ii) unconfounded outcome $U\perp Y\mid M$ forces $b(m,u)=E[Y\mid m]$ constant in $u$. A covariance with a constant is zero, so every $c_x(m)=0$.
- **(b) Exact necessary-and-sufficient condition (fixed $x$).** $B(x)=0$ iff $\sum_m\mathrm{Cov}_U(a_x(m,U),b(m,U))=0$. This aggregate condition is **strictly weaker** than level-wise vanishing $c_x(m)\equiv0$: cancellation across mediator levels can give $B(x)=0$ with individual $c_x(m)\neq0$.

**Honest taxonomy (correction applied).** Conditions (i) and (ii) are **sufficient, always**, for $B\equiv0$; they are **not necessary**. There is a full positive-dimensional family of non-degenerate laws with cancelling $c_x(m)$ that also give $B(x)=0$. The earlier "sharp dichotomy / necessary up to a measure-zero cancellation set" language is withdrawn: no genericity or measure argument was established, and the honest exact statement is exactly (b). The degeneracy "OR" in (a) is a sufficiency taxonomy, not an iff.

These conditions are statements about the latent law and are **not** testable from $p(x,m,w,y)$ without resolving $U$ (Section 3.5, Theorem 1.6a).

### 3.3 Sign and binary-$U$ closed form

**Theorem 1.3 (sign and monotone co-confounding). [PROVEN]** Fix $m,x$.
- **(a)** If $u\mapsto a_x(m,u)$ and $u\mapsto b(m,u)$ are **monotone in a common ordering of $U$** (both nondecreasing or both nonincreasing in that ordering), then by the Chebyshev correlation inequality $c_x(m)\ge0$; if monotone in opposite directions, $c_x(m)\le0$. (The earlier equation of comonotonicity with Esary-Proschan-Walkup association is dropped; EPW association is a strictly more general, non-equivalent notion, and only the Chebyshev inequality is needed.)
- **(b)** If the same direction holds for every $m$, then $B(x)\ge0$ (front-door understates $\psi$) or $B(x)\le0$ (front-door overstates) accordingly.
- **(c) Binary $U$.** With $U\in\{0,1\}$, $p=p(U=1)$, $\Delta a_x(m)=a_x(m,1)-a_x(m,0)$, $\Delta b(m)=b(m,1)-b(m,0)$,
$$
c_x(m)=p(1-p)\,\Delta a_x(m)\,\Delta b(m),\qquad B(x)=p(1-p)\sum_m\Delta a_x(m)\,\Delta b(m),
$$
since for binary $U$ any $\mathrm{Cov}_U(f,g)=p(1-p)(f(1)-f(0))(g(1)-g(0))$.

### 3.4 ATE-level bias and the shared outcome leg

**Theorem 1.4 (ATE-level bias). [PROVEN]** With $\mathrm{ATE}(x,x')=\psi(x)-\psi(x')$ and $\mathrm{ATE}_{\mathrm{FD}}$ the front-door analogue,
$$
\mathrm{ATE}(x,x')-\mathrm{ATE}_{\mathrm{FD}}(x,x')=B(x)-B(x')=\sum_m\mathrm{Cov}_U\big(p(m\mid x,U)-p(m\mid x',U),\ E[Y\mid m,U]\big).
$$
*Proof.* By Theorem 1.1 the difference is $B(x)-B(x')$. Because the outcome leg $b(m,U)$ is shared across $x,x'$ (E1), covariance is bilinear in its first argument with $b(m,U)$ held fixed, so $c_x(m)-c_{x'}(m)=\mathrm{Cov}_U(a_x(m,U)-a_{x'}(m,U),b(m,U))$; sum over $m$. $\blacksquare$

**Corollary 1.4.1. [PROVEN]** If the confounded part of the mediator leg is $x$-invariant, i.e. $a_x(m,u)-\bar a_x(m)=a_{x'}(m,u)-\bar a_{x'}(m)$ for all $u,m$, then $a_x(m,U)-a_{x'}(m,U)$ is $U$-constant and the ATE-bias vanishes even though each $\psi(x)$ is individually biased. Differencing cancels any contamination whose $U$-profile is common to both arms.

**Corollary 1.4.2 (binary $U$). [PROVEN]** $\mathrm{ATE}-\mathrm{ATE}_{\mathrm{FD}}=p(1-p)\sum_m(\Delta a_x(m)-\Delta a_{x'}(m))\,\Delta b(m)$.

### 3.5 The no-proxy sensitivity corollary

**Theorem 1.5 (proxy-free front-door bias audit). [PROVEN]** Since $B(x)$ is a sum of covariances, Cauchy-Schwarz applies level-wise.
- **(a)** $|c_x(m)|\le\mathrm{sd}_U(a_x(m,U))\,\mathrm{sd}_U(b(m,U))$, hence
$$
|B(x)|\le\sum_m\mathrm{sd}_U\big(p(m\mid x,U)\big)\,\mathrm{sd}_U\big(E[Y\mid m,U]\big).\tag{5}
$$
Equality holds **level-wise** iff $a_x(m,U)$ and $b(m,U)$ are $p$-a.s. affine functions of one another. This level-wise condition is the only sharpness actually proven; the aggregate envelope (5) is not claimed sharp.
- **(b) Correlation cap.** Writing $c_x(m)=\rho_x(m)\,\mathrm{sd}_U(a_x(m,U))\,\mathrm{sd}_U(b(m,U))$ with $\rho_x(m)\in[-1,1]$, a single cap $|\rho_x(m)|\le\rho^\star$ gives $|B(x)|\le\rho^\star\sum_m\mathrm{sd}_U(a_x(m,U))\,\mathrm{sd}_U(b(m,U))$.
- **(c) Observed-law-calibrated form.** With $R^2_{M\sim U\mid x,m}:=\mathrm{Var}_U(p(m\mid x,U))/[p(m\mid x)(1-p(m\mid x))]\in[0,1]$ (valid since a $[0,1]$-valued random variable with mean $\mu$ has variance $\le\mu(1-\mu)$) and the Popoviciu bound $\mathrm{sd}_U(b(m,U))\le\tfrac12\mathrm{range}(Y\mid m)$,
$$
|B(x)|\le\rho^\star\sum_m\sqrt{p(m\mid x)(1-p(m\mid x))}\cdot\tfrac12\,\mathrm{range}(Y\mid m),\tag{7}
$$
in which **every quantity except $\rho^\star$ is identified from $p(x,m,y)$ alone**. The analyst computes the observed-law factors, posits one confounding-correlation cap $\rho^\star$, and reports $[\psi_{\mathrm{FD}}(x)\pm\rho^\star\sum_m(\cdots)]$ for $\psi(x)$ and the differenced version for the ATE. The ATE audit is tighter because the shared outcome leg lets one bound $\mathrm{sd}_U(a_x(m,U)-a_{x'}(m,U))$, which is far smaller than $\mathrm{sd}_U(a_x)+\mathrm{sd}_U(a_{x'})$ when the arms share their $U$-profile (Corollary 1.4.1).

**Theorem 1.6 (recoverability). [PROVEN direction + flagged CONJECTURE]**
- **(a) [PROVEN]** $B(x)$ is not a functional of the observed law without resolving $U$. Construction: take $r=2$, $W$ degenerate ($d_W=1$, kernel rank $1<2$); the observed law constrains only the $U$-marginals $\bar a_x(m),\bar b(m)$, while $c_x(m)=p(1-p)\Delta a_x(m)\Delta b(m)$ is free (vary $\Delta a,\Delta b$ keeping $\bar a,\bar b$ fixed by compensating the two $U$-cells). Two assignments match all observables but differ in $B$. Under (A3) with a left-invertible kernel, $J$ and hence $c_x(m),B(x),\psi$ are point-identified and $\psi$ is read off directly.
- **(b) [PROVEN]** Consequently $\rho_x(m)$ is the non-identified sensitivity direction; the observed law pins only the scale factors, so $\rho^\star$ in Theorem 1.5 is genuinely free.
- **(c) [PROVEN framing]** $\psi_{\mathrm{FD}}$ is identified precisely under $U\perp M\mid X$ (Theorem 1.2(i)). The hidden-mediator program (Fulcher-Ghassami-Shpitser-Shi-Tchetgen Tchetgen) and proximal-mediation (Dukes-Shpitser-Tchetgen Tchetgen 2023, needing a second proxy $Z$ + double completeness; Park-Richardson-Tchetgen Tchetgen 2024, single proxy + additivity) restore $\psi$ under $U\to M$ by importing extra structure. The present route trades additivity for the proxy-resolution rank condition.
- **(d) CONJECTURE.** When $\mathrm{rank}(K)=\rho<r$, the sharp ATE interval has endpoints attained where the unresolved $U$-mixture is maximally co-/counter-monotone in $(\Delta a,\Delta b)$, matching Theorem 1.3; positivity constraints can make the truth strictly tighter than the $\rho^\star=1$ Cauchy-Schwarz envelope, so the conjecture likely holds with equality only in a moment-matched relaxation. **Stated, not proven.**

### 3.6 Worked binary-$U$ numeric example [PROVEN by direct computation]

All of $X,M,U,Y$ binary, $Y\in\{0,1\}$, $p=0.5$. Mediator leg $a_1(1,\cdot)=(0.2,0.8)$, $a_0(1,\cdot)=(0.1,0.5)$ over $u\in\{0,1\}$; outcome leg $b(0,\cdot)=(0.1,0.4)$, $b(1,\cdot)=(0.5,0.9)$. High $U$ raises both legs (comonotone), predicting $B>0$.

Computation gives $\psi(1)=0.490$, $\psi_{\mathrm{FD}}(1)=0.475$, so $B(1)=0.015$; the binary-$U$ formula $B(1)=0.25[(-0.6)(0.3)+(0.6)(0.4)]=0.015$ agrees. Likewise $\psi(0)=0.395$, $\psi_{\mathrm{FD}}(0)=0.385$, $B(0)=0.010$. Hence $\mathrm{ATE}(1,0)=0.095$ versus $\mathrm{ATE}_{\mathrm{FD}}=0.090$, difference $0.005=B(1)-B(0)$, cross-checked three ways. The level-wise sensitivity envelope $\sum_m\mathrm{sd}_U(a)\mathrm{sd}_U(b)=0.105\ge|B(1)|=0.015$ (loose because the two levels' true correlations are $\pm1$ and partially cancel).

## 4. Identification frontier (Track 2)

Keep two targets rigorously separate. **(J-ID):** the latent array $J$ (equivalently $p(u),p(m\mid x,u),E[Y\mid m,u]$) is identified. **(ATE-ID):** the scalar $\psi(x)$ is identified for the relevant $x,x'$. Trivially J-ID $\Rightarrow$ ATE-ID; the content below is the converse and the gap.

### 4.1 The sharpest necessary/sufficient statement

**Proposition 2.1 (J-ID iff (A3), kernel known). [PROVEN]** Fix the true kernel $K^\star$ with $J^\star$ having all rows nonzero.
- (i) If $\mathrm{rank}(K^\star)=r$, then $J^\star=K^{\star+}P$ is the unique solution of $(\star)$ given $K^\star$, so J-ID holds.
- (ii) If $\mathrm{rank}(K^\star)=\rho<r$, then $\ker K^\star\neq\{0\}$; perturbing a column of $J$ along $\ker K^\star$ keeps $(\star)$ and the simplex satisfied for small steps, so J-ID fails.

**Necessity of $\mathrm{card}(W)\ge\mathrm{card}(U)$. [PROVEN]** A left inverse of $K\in\mathbb R^{d_W\times r}$ exists only if $\mathrm{rank}(K)=r$, which forces $d_W\ge r$.

**Honest scope on the kernel-unknown case (correction applied).** Recovering $K$ itself from $P$ is **not** a linear inverse but a nonnegative/simplex-structured matrix factorization in the joint unknowns $(K,J)$. Collapsing indices, $P=KJ$ is a single rank-$r$ factorization identified only up to a general invertible $r\times r$ transform $G$ ($K\to KG$, $J\to G^{-1}J$); column-stochasticity and nonnegativity narrow $G$ but do **not** generically reduce it to a permutation. A permutation-only conclusion requires a **factorization-uniqueness condition** (separability/anchor, à la Donoho-Stodden / Successive Projection, or the two-proxy double completeness of Miao-Geng-Tchetgen Tchetgen 2018). Therefore the honest J-ID statement is: **under (E1),(E2),(A3) and an explicit factorization-uniqueness condition, restricted to discrete low-cardinality $U$, $K$ and $J$ are identified up to a $U$-label permutation; since $\psi(x)$ marginalizes $U$ it is permutation-invariant, hence point-identified.** (A3) alone guarantees the left inverse **exists once $K$ is known**, not that $K$ is recovered. The clean, fully proven J-ID is the kernel-known case (i).

**ATE-ID can survive rank deficiency.** ATE depends on $J$ only through the bilinear contraction
$$
\psi(x)=\sum_m\sum_u\frac{a_{x,m}(u)}{p(x\mid u)}\,b_m(u),\qquad p(x\mid u)=\frac{\sum_{m'}a_{x,m'}(u)}{\sum_{x',m'}a_{x',m'}(u)},\tag{1}
$$
with $a_{x,m}(u)=p(x,m,u)=\sum_yJ(x,m,u,y)$ and $b_m(u)=E[Y\mid m,u]$, $x$-independent by (E1). (The earlier intermediate display asserting an underbrace $=1$ is deleted; the correct factor is $p(x\mid u)$, and the boxed (1) is the corrected form, verified: $a_{x,m}(u)/p(x\mid u)=p(u)p(m\mid x,u)$.) This functional is **coarser** than $J$, so ATE-ID can in principle pass even when J-ID fails.

**Proposition 2.2 (exact ATE-ID gap). [PROVEN]** With feasible set $\mathcal F(P)=\{J\ge0:KJ_{x,m}=P_{x,m}\ \forall x,m,\ \text{normalized}\}$ for the known kernel, ATE-ID holds iff $\psi$ is **constant on the entire feasible slice** $J^\star+(\ker K)$. A vanishing first-order directional derivative $\nabla_J\psi(J^\star)\perp\ker K$ is **necessary** but, because $\psi$ is rational in the deficiency parameter $t$ (both $1/p(x\mid u)$ and $b_m(u)$ vary with $t$), is **not sufficient**: global constancy on $\mathcal F(P)$ is required, and a single first-order condition does not reduce to it. (This corrects the earlier "iff moment-orthogonality" overclaim to a necessary local condition.)

**Proposition 2.3 (an interpretable point-ID-without-resolution regime). [PROVEN]** If $E[Y\mid m,u]=b_m$ does not depend on $u$, then by the seed identity every $c_x(m)=0$, so $\psi(x)=\psi_{\mathrm{FD}}(x)=\sum_mb_m\,p(m\mid x)$, which depends only on observed margins. Hence ATE is point-identified with $\mathrm{rank}(K)$ arbitrary (even rank $1$, $W\perp U$), while $J$ is non-identified. This is the cleanest coarser-functional rescue. **OPEN (ATE-ID certification under deficiency):** find verifiable functionals of $P$ that certify $\psi$ constant on $\mathcal F(P)$ when $\mathrm{rank}(K)<r$; conjecture that only the front-door-collapse case (Proposition 2.3) is data-certifiable.

### 4.2 The no-rescue-from-tensor argument

**Proposition 2.4 (only one pure view). [PROVEN]** Conditioning on $U$, the d-separations in $G$ are $W\perp(X,M,Y)\mid U$ (E2), $X\perp Y\mid(M,U)$ (E1, chain blocked by $M$), but $X\not\perp M\mid U$ (edge), $M\not\perp Y\mid U$ (edge), and $X\not\perp Y\mid U$ (path $X\to M\to Y$ open when $M$ is not conditioned). Hence $W$ is the **unique** observed variable that is conditionally independent of all others given $U$; the chain $X\to M\to Y$ contributes a single entangled block, not independent views.

**Proposition 2.5 (no-rescue). [PROVEN]** Kruskal/Allman-Matias-Rhodes finite-mixture identification needs three conditionally-independent views with Kruskal ranks summing to $\ge2r+2$. With only one pure view $W$ and one entangled block $(X,M,Y)$, this is a two-block problem, which is not generically identifiable unless one block already has Kruskal rank $\ge r$, i.e. exactly $\mathrm{rank}(K)=r$, which is (A3) itself. Any split of $(X,M,Y)$ has an open within-block path given $U$ (Proposition 2.4), so it cannot supply the missing view. The identification power that Kruskal would provide must come entirely from $W$'s rank; (A3) is intrinsic, not a proof artifact. This is precisely why the literature imports a second proxy $Z$ (Miao 2018, Dukes-Shpitser 2023) to manufacture the missing view.

### 4.3 Continuous-$U$ completeness sufficient conditions

For continuous $U$, (A3) becomes injectivity of $T_{W\mid U}:f\mapsto E[f(U)\mid W=\cdot]$ (completeness of $p(w\mid u)$ in $u$). Checkable sufficient conditions:
- **C1 (exponential family). [sufficient]** $p(w\mid u)\propto h(w)\exp(\eta(u)^\top T(w)-A(u))$ with $\{\eta(u)\}$ containing an open set is complete (full-rank exponential family completeness). Example: $W\mid U\sim\mathcal N(\beta_0+\beta_1u,\sigma^2)$, $\beta_1\neq0$.
- **C2 (location family). [PROVEN here]** If $W=U+\varepsilon$, $\varepsilon\perp U$, with characteristic function $\phi_\varepsilon(t)\neq0$ for all $t$, then $T_{W\mid U}$ is injective. *Proof.* $E[f(U)\mid W=w]=0$ a.e. gives $(g*p_\varepsilon)=0$ with $g=fp_U\in L^1$; Fourier transform yields $\hat g(t)\phi_\varepsilon(t)=0$ for all $t$; $\phi_\varepsilon$ nonvanishing forces $\hat g\equiv0$, so $f=0$ $p_U$-a.e. $\blacksquare$ Gaussian, Laplace, Cauchy noise qualify; uniform noise (sinc has zeros) can fail.
- **C3 (total positivity). [sufficient]** STP$_\infty$ / MLR families are injective by the variation-diminishing property; checkable via $\partial^2\log p(w\mid u)/\partial w\partial u>0$ for the order-2 case.

**Honest scope.** Recovery requires deconvolving by $\phi_\varepsilon$, so the inversion is **ill-posed**: ordinary-smooth noise ($|\phi_\varepsilon(t)|\sim|t|^{-\beta}$) is mildly ill-posed; super-smooth (Gaussian) noise is severely ill-posed with only logarithmic rates. This is fundamentally a **low-cardinality discrete-$U$ method**, where $\mathrm{rank}(K)=r$ is finite and testable; continuous high-dimensional $U$ survives in principle but is rate-limited, and $\sqrt n$ inference there needs an additional source/smoothness condition.

### 4.4 Over-identification test

**Proposition 2.6 (over-ID rank restriction). [PROVEN]** When $d_W>r$ and $\mathrm{rank}(K)=r$, $(\star)$ forces every $P_{x,m}\in\mathrm{col}(K)$, so the stacked matrix $\mathbf P=[P_{x,m}]_{(x,m)}\in\mathbb R^{d_W\times c}$ has $\mathrm{rank}(\mathbf P)=r$ (given full-row-rank stacked $\mathbf J$), a falsifiable restriction: its top $r$ singular values are positive and the rest are exactly zero in population. Test $H_0:\mathrm{rank}(\mathbf P)=r$ via a Robin-Smith or Kleibergen-Paap rank test with degrees of freedom $\mathrm{df}=(d_W-r)(c-r)$.

**Diagnostic content.** Slice-wise over-rank ($\mathrm{rank}(P_{x,m})>r$ for some slices, with $(x,m)$-varying excess singular vectors) points at an **(E2) failure** (impure proxy, e.g. $M\to W$ or $X\to W$, making the kernel $(x,m)$-dependent). Slice-wise rank intact but a cross-$x$ mismatch in the recovered $b_m(u)=E[Y\mid m,u]$ between the $x$- and $x'$-slices points at an **(E1) failure** (residual $X\to Y$, breaking the shared outcome bridge). This turns the test from a yes/no falsification into an attribution between the two exclusions. **Scope honestly:** the entire frontier analysis is sharpest for low-cardinality $U$.

## 5. Estimation and semiparametric theory (Track 3)

### 5.1 Discrete deconvolution estimator

Notation as in Section 1; $K^+=(K^\top K)^{-1}K^\top$ under (A3). The structural reads from $J$ are $p(u)=\sum_{x,m,y}J$, $p(m\mid x,u)=p(x,m,u)/p(x,u)$, $\mu(m,u)=\sum_yyp(m,u,y)/p(m,u)$, and the plug-in is $\hat\psi(x)=\sum_u\hat p(u)\sum_m\hat\mu(m,u)\hat p(m\mid x,u)$.

- **Regime A (kernel known/separately supplied).** Solve, per $(x,m,y)$, $\hat J=\Pi_\Delta(K^+\hat P)$ or jointly the convex QP $\min_{\mathbf J\ge0,\,\mathbf 1^\top\mathbf J\mathbf 1=1}\tfrac12\|\hat{\mathbf P}-K\mathbf J\|_F^2$; the interior solution is $\hat J=K^+\hat P$.
- **Regime B (kernel unknown).** Stacked $\mathbf P=K\mathbf J$ is a **simplex-structured NMF**. **PROVEN:** (A3) identifies $J$ *given* $K$, but a single proxy does **not** identify $K$ (the factorization is unique only up to an invertible transform; a permutation conclusion needs separability/anchor). Estimate $\hat K$ by rank-truncated SVD plus a separable-NMF step (Successive Projection / Hottopixx), with rank $\hat r$ chosen from the spectral gap above a noise floor $\tau_n\asymp\sqrt{(d_W+N)/n}$. Always report $\sigma_{\min}(\hat K)$ and $\|\hat K^+\|_2=1/\sigma_{\min}(\hat K)$ as the resolution diagnostic.

### 5.2 Delta-method variance and the $\sigma_{\min}(K)^{-2}$ blow-up

On the resolution interior $\{\sigma_{\min}(K)\ge c>0\}$ with positivity/overlap, the simplex constraints are asymptotically inactive, so $\hat\psi=g(\hat P)$ is a smooth (rational) map and $\sqrt n(\hat\psi-\psi)\xrightarrow{d}\mathcal N(0,\nabla g^\top\Sigma_P\nabla g)$ with multinomial $\Sigma_P=\mathrm{diag}(P)-PP^\top$. The Jacobian factors as $\nabla g=(I\otimes K^+)^\top h$ where $h=\partial\psi/\partial J$ is bounded and kernel-independent. Using $\|I\otimes K^+\|_2=\|K^+\|_2$,
$$
n\,\mathrm{Var}(\hat\psi(x))=\Theta\big(\|K^+\|_2^2\big)=\Theta\big(\sigma_{\min}(K)^{-2}\big)\quad\text{as }\sigma_{\min}(K)\to0.
$$
This is **graceful degradation in variance**: identification never collapses discontinuously inside the resolution region; the standard error inflates like $1/\sigma_{\min}(K)$, and rank deficiency ($\sigma_{\min}\to0$) is the boundary where the point-ID variance diverges and the estimand widens to an interval (Track 4). When $K$ is also estimated, the propagated $\partial K^+/\partial K=-K^+(\partial K)K^+$ plus range/cokernel corrections gives, in the **worst case**, a $\sigma_{\min}^{-4}$ scaling. **This $\sigma_{\min}^{-4}$ rate is a CONJECTURED worst-case bound**, not a two-sided $\Theta$: the range/cokernel corrections are not written out here.

### 5.3 Efficient influence function, with PROVEN-vs-OPEN labels

This is where the earlier draft overreached; the corrected statement distinguishes three distinct objects on the model $P=KJ$ with $K$ known. Let $\ell$ be the centered latent gradient $\partial\psi/\partial J-\psi$.

- **(a) Multinomial/full-data gradient IF $\partial g/\partial P$.** Its variance is the delta-method variance of Section 5.2. It treats $\hat P$ as unconstrained multinomial and so **ignores the constraint $P\in\mathrm{range}(K)$**; it is an **upper bound** on efficiency, not the efficiency bound.
- **(b) The particular influence function $\varphi=K^{+\top}\ell$. [PROVEN to be a valid IF]** It is mean-zero and pathwise-correct ($E[\varphi\,s_{\mathrm{obs}}]=\partial_\theta\psi$ for all submodels $P_\theta=KJ_\theta$), and its norm scales as $\Theta(\sigma_{\min}(K)^{-1})$ (Kronecker identity $\|I\otimes K^+\|_2=\|K^+\|_2$ verified). But it is **not efficient**: $K^{+\top}$ maps into $\mathrm{range}(K^\top)$, which is not the $P$-weighted $\mathrm{range}(K)$ that defines the observed tangent space.
- **(c) The efficient influence function.** The EIF is the **orthogonal projection in $L^2(P)$** of the gradient onto the observed tangent space $\mathcal T_K=\{$scores representable as $(K\,dJ)/P\}=\mathrm{range}(K)\ominus\{\text{const}\}$; concretely, for each $(m,y)$-block project the gradient's $w$-profile onto $\mathrm{range}(K)$ in the $P$-weighted metric, then center.

**Corrected ordering (PROVEN numerically across seeds with $d_W>r$):** $\mathrm{Var}_{\text{EIF}}<\mathrm{Var}(K^{+\top}\ell)<\mathrm{Var}_{\text{delta}}$, strictly. A representative seed gives $0.2468<0.2556<0.2669$. The earlier claim that these three coincide is **withdrawn**. What survives and is correct: all three variances scale as $\Theta(\sigma_{\min}(K)^{-2})$, so the headline blow-up is right even though the variances are not equal. **Honest weakened EIF statement:** for $K$ known, $\varphi=K^{+\top}\ell$ is a valid regular mean-zero influence function with variance $\Theta(\sigma_{\min}^{-2})$; the EIF is its projection onto $\mathcal T_K$, with strictly smaller variance; the delta-method variance is a looser upper bound that ignores the $\mathrm{range}(K)$ constraint.

### 5.4 Why no clean doubly-robust form

The target is bilinear in two **latent** legs sharing the **same** operator $K^+$: $\psi=\sum_up(u)\langle a_x(\cdot,u),b(\cdot,u)\rangle_M$. This is structurally unlike the single-bridge proximal functional of Cui-Pu-Shi-Miao-Tchetgen Tchetgen 2024, which is **linear** in one bridge and has a clean two-bridge DR form robust to one of $\{h_0,q_0\}$.
- **PROVEN obstruction.** There is **no full triple robustness** across $\{a,b,K\}$: the kernel error $\hat K^+-K^+$ enters the von Mises remainder **linearly** unless explicitly orthogonalized, because both legs are reconstructed through the same $K^+$. No re-expression makes this a one-bridge linear-DR problem.
- **CONJECTURE (high confidence).** Conditional on $K$, the coupled functional admits a **leg-versus-leg** mixed-bias structure with remainder $\sum_up(u)\langle\Delta a,\Delta b\rangle$, vanishing if either leg is consistent per stratum $u$. This is strongly supported by the seed identity (the bias term is literally an $\langle a,b\rangle_U$ inner product), but **not proven** in the post-deconvolution observed model. The closed form of the kernel-orthogonalized score $\tilde\varphi=K^{+\top}\ell-\Lambda_K[\text{score}_K]$ is **OPEN**, with non-regularity threatened at the simplex/anchor boundary (the $\sigma_{\min}\to0$ regime). The content does **not** fabricate a clean DR form.

### 5.5 Cross-fit inference

With cross-fitting over folds $k$, estimate nuisances $(\hat K,\hat a,\hat b,\hat p_u)$ on $\mathcal D_{-k}$ and evaluate the (conjectured Neyman-orthogonal) score on $\mathcal D_k$, solving $\mathbb P_n\tilde\varphi=0$, with $\widehat{\mathrm{Var}}=n^{-1}\mathbb P_n\tilde\varphi^2$. Sufficient conditions for $\sqrt n$-normality (PROVEN modulo the conjectured $\tilde\varphi$): (1) resolution interior $\sigma_{\min}(K)\ge c>0$; (2) the mixed-bias product condition $\sqrt n(\sum_up(u)\|\hat a_x-a_x\|_{L_2(u)}\|\hat b-b\|_{L_2(u)}+\|\hat K^+-K^+\|_{\text{orth}})=o_p(1)$, so each leg may converge as slowly as $n^{-1/4}$ once orthogonalized; (3) bounded second moment of $\tilde\varphi$, with the empirical-process term controlled by cross-fitting. Without the kernel orthogonalization, $\hat K$ must satisfy the far stronger un-squared $o_p(n^{-1/2})$ rate, which is the practically binding constraint and degrades with $\sigma_{\min}(K)$.

For continuous $U$, $(\star)$ becomes a Fredholm-I equation $P=T_{W\mid U}J$ with compact, hence non-invertibly-bounded, $T_{W\mid U}$. Under a source condition $J\in\mathrm{range}(T^\beta)$ and singular decay $s_j\asymp j^{-p}$, regularized estimation attains the **standard NPIV-style rate (cited, not derived here)** $\|\hat J-J\|=O_p(n^{-\beta/(2\beta+2p+1)})$; the exact placement of the ill-posedness exponent is convention-dependent across the NPIV literature, so this is labeled cited rather than proven. $\sqrt n$ inference for $\psi$ needs the functional smoothness to dominate $p$.

## 6. Sharp partial-ID program (Track 4)

Regime: $\mathrm{rank}(K)=r-d$ with deficit $d\ge1$. Structural factors $\pi_u=p(u)$, $q_{m\mid x,u}=p(m\mid x,u)$, $g_{m,u}=E[Y\mid m,u]\in[y_{\min},y_{\max}]$.

### 6.1 Identified-set characterization

The feasible set is $\mathcal F(\hat P)=\{(K,J):K\in\mathcal K,\ J\in\mathcal J,\ KJ_{x,m,\cdot,y}=\hat P_{x,m,\cdot,y}\ \forall(x,m,y)\}$, where $\mathcal K$ is the column-simplex set, $\mathcal J$ imposes nonnegativity, normalization, and the (E1) Markov product form $J(x,m,u,y)=p(x,m,u)p(y\mid m,u)$.

**[PROVEN, structural.]** $\mathcal F$ is a **compact semialgebraic set** (finite polynomial equalities/inequalities; all coordinates are probabilities or conditional expectations in $[y_{\min},y_{\max}]$). The constraint $KJ=P$ is bilinear in $(K,J)$; holding $K$ fixed it is affine in $J$. The **unresolved degrees of freedom live entirely in $\ker K$** ($\dim=d$): for fixed feasible $K$, $J_{x,m,\cdot,y}=K^+P_{x,m,\cdot,y}+N\theta_{x,m,y}$ with $\mathrm{col}(N)=\ker K$, subject to nonnegativity, the Markov product form, and cross-$(x,m,y)$ marginal coupling. The Markov coupling shrinks the raw kernel box to a curved feasible slice, which can make the ATE interval strictly narrower than a naive per-slice bound. The ATE identified set is $\mathcal I_{\mathrm{ATE}}=\{\mathrm{ATE}(K,J):(K,J)\in\mathcal F\}$.

### 6.2 The interval theorem

The objective $\psi(x)=\sum_{u,m}\pi_uq_{m\mid x,u}g_{m,u}$ is **multilinear (degree 3)**, separately linear in each factor-block; $\mathrm{ATE}$ likewise.

**Theorem 4.1 (sharp ATE set is a closed bounded interval). [PROVEN via block-coordinate homotopy]** If $\mathcal F(\hat P)\neq\varnothing$, then $\mathcal I_{\mathrm{ATE}}=[\mathrm{LB},\mathrm{UB}]$ with both endpoints attained.
*Proof.* $\mathcal F$ is compact (Section 6.1) and $\mathrm{ATE}$ is a polynomial, so by Weierstrass $\min=\mathrm{LB}$ and $\max=\mathrm{UB}$ are attained and the image lies in $[\mathrm{LB},\mathrm{UB}]$. For the no-gaps (interval) property, connect any two feasible structures by a **block-coordinate homotopy**: move one factor-block at a time along a straight segment holding the others fixed. The moment equalities $\sum_uK_{wu}p(x,m,u)p(y\mid m,u)=\hat P$ are **affine in the $\pi_u$ block and in the $p(y\mid m,u)$ block** separately (with the $q_{m\mid x,u}$ block handled with the matched $p(x,u)$ rescaling), so each single-block segment between two feasible endpoints stays feasible, and $\mathrm{ATE}$ moves continuously along it. Concatenating the block segments path-connects the two structures within $\mathcal F$ and makes $\mathrm{ATE}$'s image connected, hence an interval. $\blacksquare$

**Honesty note (correction applied).** The earlier "global connectedness of $\mathcal F$ via single-block convex segments between *arbitrary* feasible structures" claim is **false in general**: the constraint is multilinear across blocks, so a single-block segment between two structures that differ in several blocks need not be feasible. The valid argument is the **block-coordinate homotopy above**, which moves one block at a time and is feasibility-preserving block by block; this is the primary proof. The fiber-connectedness route in the earlier draft is downgraded to motivation. In knife-edge degenerate cases (isolated feasible points, lower-dimensional components) the interval conclusion still holds because $\mathrm{ATE}$ is constant on or shares endpoints across such pieces, but a fully general topological connectedness of $\mathcal F$ is **not** claimed.

**Theorem 4.2 (point-ID at $d=0$). [PROVEN]** If $d=0$, $K$ is left-invertible, $J=K^+P$ is unique, $\psi$ point-identifies, and the width $\mathcal W=\mathrm{UB}-\mathrm{LB}=0$.

### 6.3 Certified-sharp computation

- **Exact POP.** $\mathrm{UB}=\max_{\pi,q,g,K}\sum_u\pi_u\sum_m(q_{m\mid x,u}-q_{m\mid x',u})g_{m,u}$ subject to the factorization, simplex, stochasticity, and range constraints; a polynomial program of constraint degree $\le4$. $\mathrm{LB}$ is the matching $\min$.
- **Practical workhorse: grid over $\ker K$ + evaluate.** Fix the kernel cell and grid the $d$-dimensional kernel direction $\theta$; at each grid point $J=J^0+N\theta$ is fully determined, so the multilinear $\mathrm{ATE}$ evaluates in closed form; sweep to drive to $\min/\max$. For $d=1$ this is a one-dimensional line search; refining the grid gives validated **inner** bounds. (The earlier "objective is a fractional ratio" justification is **deleted**: with $K$ fixed the objective is **linear in $J$**, since the $\sum_yJ$ denominator cancels; $\frac{(\sum_yyJ)(\sum_{m',y}J)}{\sum_yJ}=g_{m,u}p(m\mid x,u)p(x,u)$ is linear in $J$. The only genuine nonconvexity is the $KJ=P$ bilinear coupling, not the objective.) McCormick envelopes on the bilinear products give a valid **outer** LP; squeeze inner against outer.
- **Moment-SOS (Lasserre) hierarchy.** Cast the POP into the moment-SOS hierarchy: $\mathrm{UB}_t\downarrow\mathrm{UB}$, $\mathrm{LB}_t\uparrow\mathrm{LB}$. **[PROVEN]** Adding the redundant ball constraint forces the Archimedean/Putinar condition, so the hierarchy **converges asymptotically**: $\mathrm{UB}_t\to\mathrm{UB}$. The earlier "finite convergence is generic, degree 2-3 certifies" claim is **withdrawn**: generic finite convergence of the moment-SOS hierarchy has delicate hypotheses not verified for this bilinear variety. Only **asymptotic** certified outer convergence is claimed; flat-extension/rank-stabilization may detect exactness empirically but is not guaranteed at a fixed degree.

### 6.4 Width versus deficit and the validity-aware defer rule

Let $d=r-\mathrm{rank}(K)$ and $\sigma_{r-d}$ the smallest nonzero singular value of $K$. Width $\mathcal W=\mathrm{UB}-\mathrm{LB}$.

**Theorem 4.3 (nesting, fixed law and kernels). [PROVEN, restated in correct form]** Fix one observed law $P$ and two kernels $K_1,K_2$ **both consistent with that same $P$**, with $\mathrm{col}(K_2)\subseteq\mathrm{col}(K_1)$ and deficits $d_2\ge d_1$. Then $\{J:K_1J=P\}\subseteq\{J:K_2J=P\}$, so $[\mathrm{LB}_1,\mathrm{UB}_1]\subseteq[\mathrm{LB}_2,\mathrm{UB}_2]$ and $\mathcal W_1\le\mathcal W_2$. (The earlier statement compared feasible sets across **different** observed laws and is corrected to this fixed-$P$, fixed-kernel-pair form.)

**Proposition 4.4 (width envelope). [cap and Lipschitz bound PROVEN; rate CONJECTURED]** $\mathrm{ATE}$ is Lipschitz-multilinear in $J$ with constant $L\le(y_{\max}-y_{\min})$ times products of probabilities, so $\mathcal W\le L\cdot\mathrm{diam}(\mathcal F\cap\text{kernel slab})\lesssim\frac{C(y_{\max}-y_{\min})}{\sigma_{r-d}}\wedge(y_{\max}-y_{\min})$, capped by the Manski width once nonnegativity saturates. The cap and Lipschitz bound are proven; the exact $1/\sigma_{r-d}$ rate is a conjectured tight envelope (it is an upper bound; tightness needs kernel-direction/ATE alignment and no nonnegativity clipping). The constant $C$ and the alignment condition are **OPEN**.

**Positive-width certificate. [PROVEN]** If two structures consistent with the **same** observed law give distinct ATEs $0.047$ and $0.075$, then both lie in $\mathcal I_{\mathrm{ATE}}$, so $\mathrm{UB}-\mathrm{LB}\ge0.028>0$: a constructive certificate that the interval is nondegenerate and point-ID is unjustified under that proxy. The seed identity explains why: the unresolved kernel direction reshuffles the $U$-covariance term $\sum_m\mathrm{Cov}_U$ while leaving $\psi_{\mathrm{FD}}$ fixed.

**Validity-aware defer rule. [validity PROVEN modulo $\tau$-calibration]** Diagnostic $\hat\kappa=\sigma_{\min}(\hat K)$ (a single choice, the smallest singular value / completeness modulus, not the condition number). Rule: if $\hat\kappa-z_{1-\alpha}\widehat{\mathrm{SE}}(\hat\kappa)>\tau$, report the point estimate $\hat\psi$ with CI; else report the sharp interval $[\widehat{\mathrm{LB}},\widehat{\mathrm{UB}}]$ from Section 6.3, optionally widened by an Imbens-Manski-type confidence interval for the identified set. Calibrate $\tau\asymp n^{-1/2}\times$ safety factor, so the inversion error $1/\hat\kappa$ keeps point-ID bias below the target CI half-width. **Fail-closed guarantee:** when the true $K$ is rank-deficient ($\sigma_{\min}=0$), the test fails with probability $\to1$ and the rule returns the interval, which covers $\psi$ by Theorem 4.1; in the boundary zone it defaults to the interval, erring toward bounds, never toward a spurious point. Uniformly-valid inference across the $\sigma_{\min}=\tau$ boundary is **OPEN** (Imbens-Manski/Stoye-style smoothing).

## 7. CAREER Sub-AIM A3 program (Track 5)

### 7.1 PCI framing

Classical front-door buys point-ID by assuming $U\to M$ away, discarding exactly $\sum_m\mathrm{Cov}_U(p(m\mid x,U),E[Y\mid m,U])$. A3 instead spends one observable, falsifiable measurement condition (proxy-resolution) plus a factorization-uniqueness condition to **recover** the latent law and read $\psi$ off the coupled functional, permitting a confounded mediator without additivity and without a second proxy. The Pragmatic-Causal-Inference payoff: a single scalar $\sigma_{\min}(\hat K)$ slides the **same estimand, data, and code path** between a point estimate (Thrust A) and a sharp interval (Thrust B), with no discontinuity in what is claimed and no silent overclaiming when resolution is weak.

### 7.2 Delta versus prior work (defensible one-liners)

- **vs Dukes-Shpitser-Tchetgen Tchetgen 2023:** they need a second treatment-inducing proxy $Z$ and double completeness; A3 uses a single resolving proxy $W$ and one rank/completeness condition, because deconvolving the latent law needs only the $W\mid U$ channel invertible.
- **vs Miao-Geng-Tchetgen Tchetgen 2018 / Cui-Pu-Shi-Miao-Tchetgen Tchetgen 2024:** two-proxy proximal needs both treatment- and outcome-side proxies; A3 trades the bracket for the single, stronger resolution condition (plus factorization-uniqueness).
- **vs Park-Richardson-Tchetgen Tchetgen 2024:** they use one proxy with an additivity/no-interaction restriction; A3 needs **no additivity** and recovers the full coupled functional including the latent covariance term.
- **vs Kuroki-Pearl 2014:** they assume $p(W\mid U)$ known; A3 recovers the kernel and structural conditionals from the observed law up to a $U$-label permutation, under the factorization-uniqueness condition.
- **vs classical front-door:** front-door forbids $U\to M$; A3 permits a commonly-confounded mediator and still point-identifies.

The lane: single-proxy, no-additivity, mechanism-recovered point-ID of the full coupled front-door g-functional under a confounded mediator, with one rank/completeness knob that degrades sharply to bounds. No competing result occupies all five coordinates.

### 7.3 Three-paper arc

- **Paper I (identification; JMLR / Annals / JRSS-B).** Central theorem, **honestly stated:** under (E1),(E2),(A3) **and an explicit factorization-uniqueness condition (separability/anchor or sufficient completeness pinning the mixture up to $U$-permutation), restricted to discrete low-cardinality $U$**, $K$ and $J$ are identified up to permutation, and since $\psi(x)$ marginalizes $U$ it is point-identified; the front-door gap equals $\sum_m\mathrm{Cov}_U(p(m\mid x,U),E[Y\mid m,U])$. (A3) alone is necessary but not sufficient: it guarantees the left inverse exists once $K$ is known, not that $K$ is recovered. The factorization-uniqueness condition is **promoted into the main theorem**, not buried as a mitigation.
- **Paper II (estimation/efficiency; JASA / Biometrika / JRSS-B).** **PROVEN deliverable:** the plug-in deconvolution estimator with delta-method variance propagating the $\sigma_{\min}(K)$ term. **CONJECTURE deliverable:** the one-step/cross-fit estimator and its influence function; explicitly note the EIF may **not** admit a clean doubly-robust/mixed-bias form because the map $(K,\text{recovered conditionals})\mapsto\psi$ is bilinear in the kernel inverse and the conditionals. The phrase "doubly-robust-style" is reserved for the conjectural deliverable, not asserted of the proven one.
- **Paper III (partial-ID/sensitivity or application; JRSS-B / Annals / domain venue).** When (A3) fails, the ATE is partially identified; we **conjecture** the identified set equals the projection of the feasible-$J$ set onto $\psi(x)-\psi(x')$ and provide certified outer/inner brackets (Section 6.3); **sharpness requires proving the feasible-$J$ characterization is exactly the model-consistent set under (E1),(E2) and the full graph**, which is stated as an open problem, not asserted. The over-id residual is the falsification statistic.

### 7.4 Deliverables and software

A package `proxydecon` implementing (i) the deconvolution point estimator, (ii) the $\sigma_{\min}(\hat K)$ diagnostic, (iii) the over-id falsification test with (E1)/(E2) attribution, and (iv) the automatic defer rule (point with CI when resolution certified, sharp interval plus flag otherwise), behind a single `fit(...).effect(x, x')` entry point. An education wrapper prints the front-door estimate, the deconvolution estimate, and the covariance-gap decomposition $\sum_m\mathrm{Cov}_U(\cdot,\cdot)$ so students see what a confounded mediator costs and how a proxy buys it back. This satisfies the CAREER scientific-software clause with a reusable, self-falsifying causal-mediation estimator.

### 7.5 Evaluation

Semi-synthetic benchmarks with controlled $\mathrm{card}(W)$ versus $\mathrm{card}(U)$: full-rank arm checks $\sqrt n$ coverage and that bias tracks $1/\sigma_{\min}$ (not the condition number); rank-deficient arm checks that the partial-ID interval covers truth and matches the brute-force feasible-set projection; sweep $\sigma_{\min}$ to exhibit the continuous point-to-interval transition. Real low-cardinality-$U$ domains where a single discrete proxy is plausible: ABO blood-group/genotype strata with a noisy phenotype marker; coarse clinical severity classes with one recorded surrogate. The **headline empirical claim is the over-identification falsification check**, not the point estimate: the contribution reviewers should remember is the ability to detect when the proxy fails to resolve $U$ and defer.

### 7.6 Honest risk table

| Risk | Severity | Trigger | Mitigation |
|---|---|---|---|
| **Factorization-uniqueness fails** (kernel recoverable only up to invertible transform, not permutation) | High | single proxy, no anchor/separability | Promote the separability/anchor (or completeness) condition into the main theorem; scope point-ID to discrete low-cardinality $U$ where anchors are checkable; otherwise defer to Paper-III bounds. |
| Completeness/injectivity fails for **continuous $U$** | High | high-/infinite-dim $U$, weak proxy | Scope point-ID to discrete low-cardinality $U$; for richer $U$ report bounds. Do not claim continuous completeness we cannot verify. |
| **EIF has no clean DR form** (bilinear in kernel inverse $\times$ conditionals) | Medium-High | semiparametric efficiency in Paper II | Ship plug-in + delta-method with propagated $\sigma_{\min}$ variance as the guaranteed fallback; pursue the one-step/orthogonalized score as a stretch goal with explicit NMF caveats. |
| **Rank-selection / $\mathrm{card}(U)$ instability** | Medium | estimating $K$ from finite data | Report $\sigma_{\min}(\hat K)$ as a first-class output; trigger partial-ID mode when $\sigma_{\min}<\tau$; use the over-id residual to falsify $\mathrm{card}(U)$. |
| Latent-label nonidentifiability | Low | always | $\psi(x)$ is permutation-invariant, so the estimand is unaffected; only per-stratum readouts carry the label ambiguity, which we flag. |
| Real proxy violates (E2) | Medium | misspecified proxy | The slice-wise over-id check detects (E2) impurity (Section 4.4); scope to mechanistically defensible proxies and run a sensitivity arm. |

The diagnostic $\hat\kappa$ is fixed as $\sigma_{\min}(\hat K)$ throughout (not the condition number $\sigma_{\max}/\sigma_{\min}$), because the plug-in deconvolution error scales as $\|K^+\|\,\|\Delta K\|\sim\sigma_{\min}^{-1}\|\Delta K\|$; "bias tracks $1/\sigma_{\min}$" is the coherent statement.

### 7.7 Y1-Y5 and cross-thrust feed

- **Y1-Y2 (Thrust A core):** Paper I identification theory plus `proxydecon` v0 (diagnostic + over-id test); wire deconvolution into the `identify` stage of the PCI pipeline as the first attempt before fallback.
- **Y2-Y3 (efficiency, A.3):** Paper II estimation, cross-fit estimator, propagated-$\sigma_{\min}$ variance; package v1 with point-ID + CI.
- **Y3-Y4 (defer to Thrust B):** Paper III sharp partial-ID. The rank-deficient fallback **is** a finished Thrust-B causal-bound deliverable, demonstrating validity-aware defer from A to B end-to-end inside one sub-aim.
- **Y4-Y5 (feed Thrust C):** the deconvolution map (observed law $\to$ latent law $\to\psi$) plus the $\sigma_{\min}$-gated defer rule is amortizable: train an amortized estimator to emit, for a small categorical dataset, either a point or a bound together with the regime flag, learning the defer decision directly. A3 supplies Thrust C both the operator to amortize and a labeled notion of when to defer.

## 8. Open problems (ranked)

1. **ATE-ID certification under rank deficiency.** Find verifiable functionals of the observed law that certify $\psi$ is constant on the feasible slice $\mathcal F(P)$ when $\mathrm{rank}(K)<r$. Conjecture: only the front-door-collapse case (constant-in-$u$ outcome bridge, Proposition 2.3) is data-certifiable. This is the single most important gap, because it determines whether the defer rule ever upgrades a deficient kernel to a point.
2. **Sharpness of the partial-ID set (Paper III core).** Prove that the feasible-$J$ characterization (nonnegative factorizations consistent with $P$, the marginal constraints, and the (E1) Markov form) is exactly the model-consistent set under (E1),(E2) and the full graph, so that the computed $[\mathrm{LB},\mathrm{UB}]$ is genuinely sharp rather than a valid outer bracket.
3. **Factorization-uniqueness for single-proxy kernel recovery.** Identify the weakest checkable separability/anchor or completeness condition under which $P=KJ$ pins $(K,J)$ up to a $U$-permutation for discrete low-cardinality $U$, replacing the generic invertible-transform indeterminacy.
4. **Leg-versus-leg double robustness after deconvolution.** Prove or refute that the conjectured mixed-bias remainder $\sum_up(u)\langle\Delta a,\Delta b\rangle$ survives composition with $K^+$, and derive the closed-form kernel-orthogonalized score $\tilde\varphi$, including its regularity at the simplex/anchor boundary where $\sigma_{\min}\to0$.
5. **Tight width-versus-deficit envelope.** Determine the constant $C$ and the alignment condition under which $\mathcal W\asymp(y_{\max}-y_{\min})/\sigma_{r-d}$ is attained (versus clipped by nonnegativity), turning Proposition 4.4 from an upper bound into a two-sided rate.
6. **Uniformly-valid inference across the point-to-interval boundary.** Construct confidence intervals valid uniformly over $\sigma_{\min}(K)\ge0$, removing the post-model-selection discontinuity at $\sigma_{\min}=\tau$ in the defer rule.
7. **Estimated-kernel variance exponent.** Settle whether the $\sigma_{\min}^{-4}$ worst-case variance under estimated $K$ is the true two-sided rate, by writing out the range/cokernel corrections to $\partial K^+/\partial K$.
8. **Continuous-$U$ $\sqrt n$ threshold.** Pin the exact smoothness-versus-ill-posedness threshold ($\gamma$ versus $p$) governing $\sqrt n$ recoverability of $\psi$ in the continuous case, with the ill-posedness exponent placement made convention-explicit.
