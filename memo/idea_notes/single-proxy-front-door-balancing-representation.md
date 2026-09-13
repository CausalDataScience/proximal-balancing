# Single-Proxy Front-Door Balancing Representation

Date: 2026-07-06

Status: theorem seed, Lean-style verified. The algebraic proxy-to-latent core was compiled in Lean 4 at `/tmp/paios_single_proxy_frontdoor_core.lean`. The correct statement requires joint proxy balance. The naive version with separate marginal balance is false.

## Source Map

- Zotero first check:
  - Ghassami, Shpitser, and Tchetgen Tchetgen, "Proximal Causal Inference with Hidden Mediators: Front-Door and Related Mediation Problems", Zotero key `CMNU9JMN`, arXiv:2111.02927.
  - Tchetgen Tchetgen, Ying, Cui, Shi, and Miao, "An Introduction to Proximal Causal Learning", Zotero key `ZJTVIFF4`, arXiv:2009.10982.
  - Cui, Pu, Shi, Miao, and Tchetgen Tchetgen, "Semiparametric proximal causal inference", Zotero key `LZYHKCES`, arXiv:2011.08411.
  - Kallus, Mao, and Uehara, "Causal Inference Under Unmeasured Confounding With Negative Controls", Zotero key `FLXKGE87`, arXiv:2103.14029.
  - Ben-Michael, Feller, Hirshberg, and Zubizarreta, "The Balancing Act in Causal Inference", Zotero key `ZNLDURX3`, arXiv:2110.14831.
- Local anchors:
  - `research/ideas/paper_idea/proxy-balancing-achieving-ignorability-via-a-single-negative-control-outcome.md`.
  - `research/ideas/paper_idea/front-door-with-single-proxy-orthogonalized-single-proxy-front-door-ospf.md`.
  - `research/ideas/paper_idea/front-door-estimation-using-covariate-balancing.md`.
  - `research/papers/fdcate/manuscript/main/2.tex` and `research/papers/fdcate/manuscript/main/3.tex`.
  - `research/material/Pearl_r60.pdf`, `research/material/Purdue_Thesis_Yonghan.pdf`, `research/material/Complete_Identification_Methods_for_the_Causal_Hierarchy.pdf`.

## One-Line Claim

A single negative-control proxy can be used to learn a front-door balancing representation, but only if the representation balances the joint proxy-covariate object needed to resolve the latent confounder at both front-door legs.

## Standing Objects

Let \(A\) be the treatment, \(M\) the mediator, \(Y\) the outcome, \(X\) observed pre-treatment covariates, \(U\) an unobserved confounder, and \(W\) a single negative-control proxy for \(U\). Let

\[
H=h_\theta(X)
\]

be the learned representation used by the front-door estimator. The target is

\[
\Psi(a)=\mathbb E[Y\mid \mathrm{do}(A=a)].
\]

For the first leg \(A\to M\), the relevant assignment is \(A\). For the second leg \(M\to Y\), the relevant assignment is \(M\) inside strata of \((A,H)\).

## Key Correction

The following separate marginal balances are not enough:

\[
X\perp A\mid H,\qquad W\perp A\mid H,
\]

and

\[
X\perp M\mid A,H,\qquad W\perp M\mid A,H.
\]

The load-bearing conditions are joint balances:

\[
(X,W)\perp A\mid H,
\]

and

\[
(X,W)\perp M\mid A,H.
\]

Equivalently, the implementation must balance rich test functions of the pair \((X,W)\), not just all test functions of \(X\) and all test functions of \(W\) separately. If \(W\mid U,H\) is already invariant and complete without conditioning on \(X\), then \(W\)-only balance can be sufficient. If \(W\mid U,X,H\) depends on \(X\), joint balance is the minimal safe statement.

## Why Separate Marginal Balance Is False

Let \(H\) be constant. Let \(A\sim\mathrm{Bernoulli}(1/2)\). Let \(U=1-A\). Let \(X\sim\mathrm{Bernoulli}(1/2)\) independently of \(A\). Define

\[
W =
\begin{cases}
X, & U=1,\\
1-X, & U=0.
\end{cases}
\]

Then \(X\perp A\) and \(W\perp A\) both hold marginally, because both \(X\) and \(W\) are uniform in both treatment arms. However, the joint law changes:

\[
P(X=W\mid A=0)=1,\qquad P(X=W\mid A=1)=0.
\]

The proxy channel \(W\mid X,U\) is deterministic and resolves \(U\) for each \(X\), but separate marginal balance misses the latent imbalance because the signal is stored in the dependence pattern between \(X\) and \(W\). Thus separate \(X\)-balance plus separate \(W\)-balance does not imply \(U\perp A\mid H\).

The same construction applies to the mediator leg after replacing \(A\) by \(M\) and conditioning on \(A,H\).

## Assumptions for the Correct Theorem

Assumption A0, representation and positivity. \(H=h_\theta(X)\) is pre-treatment. The conditional probabilities or densities used below are positive on their supports.

Assumption A1, latent sufficiency for the two legs. For all \(a,m\),

\[
M(a)\perp A\mid X,U,
\]

and

\[
Y(m)\perp M\mid A,X,U.
\]

Assumption A2, front-door exclusion. The treatment affects the outcome only through the mediator:

\[
Y(a,m)=Y(m).
\]

Assumption A3, proxy-channel invariance. For the treatment-to-mediator leg,

\[
W\perp A\mid X,U,H.
\]

For the mediator-to-outcome leg,

\[
W\perp M\mid A,X,U,H.
\]

Assumption A4, proxy-channel injectivity. For each \(x,h\), the linear operator

\[
T_{x,h}: p(u)\mapsto \sum_u p(w\mid x,u,h)p(u)
\]

is injective on the relevant simplex. For each \(a,x,h\), the analogous operator

\[
T_{a,x,h}: p(u)\mapsto \sum_u p(w\mid a,x,u,h)p(u)
\]

is injective on the relevant simplex.

Assumption A5, exact joint balancing representation. The representation \(H\) satisfies

\[
(X,W)\perp A\mid H,
\]

and

\[
(X,W)\perp M\mid A,H.
\]

## Theorem

Under A0 through A5,

\[
M(a)\perp A\mid H,
\]

and

\[
Y(m)\perp M\mid A,H.
\]

Consequently, the front-door functional is identified by the representation-adjusted front-door formula

\[
\Psi(a)
=
\mathbb E_H
\left[
\sum_m
p(m\mid a,H)
\sum_{a'}
\mathbb E[Y\mid M=m,A=a',H]p(a'\mid H)
\right].
\]

## Proof

Lemma 1, joint proxy balance implies latent balance on the \(A\to M\) leg. Fix \(h\) and \(x\) with positive probability. From \((X,W)\perp A\mid H\), for every \(a\),

\[
p(x,w\mid a,h)=p(x,w\mid h).
\]

Summing over \(w\) gives \(p(x\mid a,h)=p(x\mid h)\). Dividing the previous display by these positive quantities gives

\[
p(w\mid a,x,h)=p(w\mid x,h).
\]

By A3,

\[
p(w\mid a,x,h)=\sum_u p(w\mid x,u,h)p(u\mid a,x,h),
\]

and

\[
p(w\mid x,h)=\sum_u p(w\mid x,u,h)p(u\mid x,h).
\]

By injectivity of \(T_{x,h}\) in A4,

\[
p(u\mid a,x,h)=p(u\mid x,h).
\]

Together with \(p(x\mid a,h)=p(x\mid h)\), this gives

\[
(X,U)\perp A\mid H.
\]

Lemma 2, joint proxy balance implies latent balance on the \(M\to Y\) leg. Fix \(a,h,x\) with positive probability. From \((X,W)\perp M\mid A,H\), for every \(m\),

\[
p(x,w\mid m,a,h)=p(x,w\mid a,h).
\]

The same division and injectivity argument using \(T_{a,x,h}\) gives

\[
p(u\mid m,a,x,h)=p(u\mid a,x,h).
\]

Together with \(p(x\mid m,a,h)=p(x\mid a,h)\), this gives

\[
(X,U)\perp M\mid A,H.
\]

Lemma 3, latent balance collapses the two ignorability assumptions to \(H\). By A1,

\[
M(a)\perp A\mid X,U.
\]

By Lemma 1, \((X,U)\perp A\mid H\). Marginalizing over \((X,U)\) therefore gives

\[
M(a)\perp A\mid H.
\]

Similarly, A1 gives

\[
Y(m)\perp M\mid A,X,U.
\]

By Lemma 2, \((X,U)\perp M\mid A,H\). Marginalizing over \((X,U)\) gives

\[
Y(m)\perp M\mid A,H.
\]

Lemma 4, representation-adjusted front-door formula. By A2,

\[
Y(a)=Y(M(a)).
\]

Thus

\[
\mathbb E[Y(a)\mid H]
=
\sum_m \mathbb E[Y(m)\mid H]p(M(a)=m\mid H).
\]

By \(M(a)\perp A\mid H\),

\[
p(M(a)=m\mid H)=p(m\mid a,H).
\]

Also,

\[
\mathbb E[Y(m)\mid H]
=
\sum_{a'}\mathbb E[Y(m)\mid A=a',H]p(a'\mid H).
\]

By \(Y(m)\perp M\mid A,H\) and consistency,

\[
\mathbb E[Y(m)\mid A=a',H]
=
\mathbb E[Y\mid M=m,A=a',H].
\]

Substituting these three displays gives the formula in the theorem.

## Lean-Compiled Core

The nontrivial algebraic core is that an injective proxy operator turns proxy-mixture equality into latent-mixture equality. This was compiled in Lean 4:

```lean
universe u v r

variable {U : Type u} {S : Type v} {R : Type r}

def MixOp (U : Type u) (S : Type v) (R : Type r) := (U -> R) -> (S -> R)

theorem latent_balance_from_proxy_balance
    (T : MixOp U S R)
    (h_inj : Function.Injective T)
    (p_treated p_target : U -> R)
    (h_proxy_balance : T p_treated = T p_target) :
    p_treated = p_target := by
  exact h_inj h_proxy_balance

theorem two_stage_latent_balance_from_proxy_balance
    (T_A T_M : MixOp U S R)
    (hA_inj : Function.Injective T_A)
    (hM_inj : Function.Injective T_M)
    (p_A p_A0 p_M p_M0 : U -> R)
    (hA_proxy_balance : T_A p_A = T_A p_A0)
    (hM_proxy_balance : T_M p_M = T_M p_M0) :
    p_A = p_A0 ∧ p_M = p_M0 := by
  constructor
  · exact hA_inj hA_proxy_balance
  · exact hM_inj hM_proxy_balance
```

Command:

```bash
lean /tmp/paios_single_proxy_frontdoor_core.lean
```

Result: compiled with exit code 0.

This Lean theorem does not encode measure theory or do-calculus. It verifies the algebraic step used in Lemmas 1 and 2. The causal part is carried by A1 through A5 and the front-door derivation in Lemma 4.

## Estimation Blueprint

Learn \(H=h_\theta(X)\) with two conditional joint-balance losses:

\[
\mathcal L_A(\theta)
=
\sup_{f\in\mathcal F_A}
\left|
\mathbb E[f(X,W)\mid A=1,H_\theta]
-
\mathbb E[f(X,W)\mid A=0,H_\theta]
\right|,
\]

and

\[
\mathcal L_M(\theta)
=
\sup_{g\in\mathcal F_M}
\left|
\mathbb E[g(X,W)\mid M=m,A,H_\theta]
-
\mathbb E[g(X,W)\mid A,H_\theta]
\right|,
\]

with the second display implemented either separately for discrete \(m\) or as a conditional IPM/adversarial loss for general \(M\). The total representation loss is

\[
\mathcal L(\theta)
=
\mathcal L_A(\theta)
+
\lambda_M\mathcal L_M(\theta)
+
\lambda_{\mathrm{ov}}\mathcal L_{\mathrm{overlap}}(\theta)
+
\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{reg}}(\theta).
\]

After learning \(\hat H\), plug it into the existing front-door learner by estimating

\[
e(a\mid \hat H),\qquad q(m\mid a,\hat H),\qquad \mu(m,a,\hat H)=\mathbb E[Y\mid M=m,A=a,\hat H],
\]

and then using the FD-DR or FD-R machinery already developed in the FDCATE project.

## Relation to OSPF

This note assumes an invariant negative-control proxy channel after conditioning on the variables in A3. If the observed proxy is contaminated by treatment or mediator leakage, as in

\[
W=h(U,C)+q(A,M,C)+\eta,
\]

then raw \(W\) does not satisfy A3. In that regime, the OSPF residualization idea is the relevant preprocessing step. One can replace \(W\) by a residualized proxy \(\widetilde W\), but the theorem then requires the residualized channel \(\widetilde W\mid X,U,H\) and \(\widetilde W\mid A,X,U,H\) to be invariant and injective. Mean residualization alone is not enough unless it restores the full conditional proxy channel needed by A3 and A4.

## When the Front-Door Proxy Helps

This note distinguishes two representation classes. A single-proxy representation uses \(X\) and a single proxy \(W\), without using the treatment-mediator response pattern \((A,M)\), to learn a balancing object \(r(W,X)\). A single front-door proxy representation may also use \((A,M)\) as an additional view, while retaining the option to ignore \(M\) when it is not useful.

The single-proxy representation fails when the proxy is many-to-one in the outcome-relevant latent state. A canonical separation example is

\[
U=(B,C),\qquad W=B.
\]

If \(C\) affects both treatment assignment and outcome, then balancing on \(W\) leaves residual imbalance:

\[
P(C\mid A=1,W)\neq P(C\mid A=0,W).
\]

In that case, no representation of \(W\) alone can remove the confounding carried by \(C\).

The single front-door proxy representation can succeed when the mediator response reveals the missing latent coordinate. In the binary example

\[
M=A\oplus C,
\]

the missing coordinate is recovered from the observed pair \((A,M)\):

\[
C=A\oplus M.
\]

Thus \(W\) supplies \(B\), while the treatment-mediator response supplies \(C\). The mediator helps because it is an informative measurement view of the unobserved state, not because post-treatment adjustment is automatically valid.

The sufficient structural pattern is:

\[
W \text{ alone is many-to-one in } U,
\]

but

\[
(W,A,M) \text{ separates the outcome-relevant part of } U.
\]

Equivalently, the conditional mediator channel must distinguish latent states that the proxy channel collapses. If two latent states induce the same proxy distribution and the same mediator response distribution, then the mediator cannot rescue the single-proxy failure.

If a single-proxy representation already succeeds, then the inclusive single front-door proxy class also succeeds. Formally, if there exists \(r(W,X)\) such that

\[
Y(a)\perp A\mid r(W,X),
\]

then the front-door-proxy class can choose

\[
s(W,X,A,M)=r(W,X)
\]

and ignore the mediator. This implication is an inclusion statement about representation classes. It does not justify a naive estimator that forcibly conditions on \(M\). Since \(M\) is post-treatment, forced adjustment on \(M\) can block part of the causal effect or introduce post-treatment selection bias.

The experiment labels should therefore avoid biomedical-looking names such as subtype, adherence, or resistance unless those terms are explicitly defined in the data-generating process. For CAREER-style figures, neutral labels such as DGP 1, DGP 2, and DGP 3 are safer. The scientific message is that \(W\) alone can be incomplete, while the mediator response can reveal the missing latent state.

## Diagnostics

- Joint balance diagnostics must test functions of \((X,W)\), not only marginal balance for \(X\) and \(W\).
- Proxy-resolution diagnostics should estimate whether the empirical proxy channel has a non-negligible smallest singular value or an analogous completeness modulus.
- Overlap diagnostics are needed for \(p(A=a\mid H)\), \(p(M=m\mid A,H)\), and the conditional joint-balance strata.
- If exact balance is replaced by approximate balance, the theorem becomes a bias-bound statement, not point identification.

## Failure Modes

- If \(W\) is affected by \(A\) or \(M\) after conditioning on the variables in A3, the proxy-balance implication can fail.
- If the proxy channel is not injective, proxy balance can hold while latent imbalance remains.
- If only separate marginal balance is enforced, dependence-pattern confounding can remain, as shown by the counterexample.
- If \(H\) includes \(W\) directly, \(W\)-balance can become vacuous and should not be interpreted as evidence that \(U\) is balanced.
- If \(H\) is too coarse to support joint balance and overlap simultaneously, the method should fail closed and report non-identification or sensitivity bounds.

## Paper Seed

The defensible paper claim is:

> A single negative-control proxy can support front-door estimation through representation learning when the learned representation enforces two joint proxy-covariate balance conditions. Under proxy-channel injectivity, these balances recover the two local ignorability conditions needed by the front-door formula.

The novelty relative to existing notes is not "single proxy magically identifies front-door." The novelty is a precise representation-learning interface:

1. joint proxy balance is the learned object,
2. injectivity/completeness is the load-bearing causal measurement assumption,
3. the output plugs directly into FD-DR or FD-R learners,
4. the method fails closed when joint balance or proxy resolution fails.
