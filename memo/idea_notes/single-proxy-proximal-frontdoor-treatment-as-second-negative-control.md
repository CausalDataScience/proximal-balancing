# Proximal Front-Door with One Confounder Proxy: the Treatment as a Second Negative-Control Exposure on the Mediator-Outcome Leg

Date: 2026-06-28
Status: eliminated, with the exact reason: conditioning the leg-2 outcome bridge on the treatment $X$ reproduces the proximal mediation construction of Dukes, Shi, Shpitser, and Tchetgen Tchetgen (2023), so "treatment $X$ as a free second negative-control exposure" is not novel, and a leg-2 completeness condition is hidden inside the bridge claim rather than discharged for free.
Rank among the 9: lowest tier (eliminated, absorbed into idea 8).
Mechanism family: proximal front-door, single confounder proxy, treatment reused as negative-control exposure (NCE).
Tags: front-door, proximal causal inference, negative control, outcome bridge, mediation, partial identification, unmeasured confounding.

> Inside a front-door factorization with a confounded mediator, the single proxy $W$ acts as a negative-control outcome and the treatment $X$ is claimed to act as a free negative-control exposure on the mediator-to-outcome leg, so that leg is said to be point-identified with one measured proxy and no additivity; the remaining treatment-to-mediator margin leg is an honest single-proxy backdoor problem with point identification under a named rank condition and sharp bounds otherwise.

## One-line idea

> The front-door exclusion is invoked to turn the treatment $X$ into a second negative-control exposure on the $M \to Y$ leg, hoping that one measured proxy $W$ then suffices for the whole effect; the leg-2 step that this enables is, on inspection, exactly proximal mediation, so the "single proxy" framing collapses to a known two-control construction with one of the controls renamed.

## Glossary

- $X$: the treatment (exposure). Discrete or continuous.
- $Y$: the outcome. Discrete or continuous.
- $U$: the unmeasured (latent) common cause that confounds $X$, the mediator $M$, and $Y$. Never observed.
- $M$: an observed mediator on the path $X \to M \to Y$. It is confounded with $Y$ by $U$, because $U \to M$ and $U \to Y$ are both present. This is what breaks classical front-door.
- $W$: the single observed proxy of $U$. It is a pure child of $U$ and carries information about $U$ only.
- $C$: optional measured covariates, conditioned on throughout. Suppressed in notation for brevity.
- ATE: average treatment effect, the contrast $\psi(x) - \psi(x')$ defined below.
- $\psi(x)$: the interventional mean $\psi(x) = \mathbb{E}[Y(\mathrm{do}\,X=x)]$, the expected outcome under setting treatment to level $x$.
- $Y(x,m)$, $Y(m)$: potential outcomes. $Y(x,m)$ is the outcome under setting $X=x$ and $M=m$; front-door exclusion forces it to depend on $m$ alone, written $Y(m)$.
- $\mathbb{E}[Y(\mathrm{do}\,M=m)]$: the mean outcome under intervening on the mediator to level $m$.
- proxy: an observed variable carrying information about $U$ without itself being a cause of the variables on the causal path.
- negative-control outcome (NCO): a variable, here $W$, known to be unaffected by the relation under study, used to detect and adjust for confounding.
- negative-control exposure (NCE): a variable, here proposed to be $X$, known not to affect $Y$ once the relevant variables are conditioned, used as a second control.
- bridge function (outcome bridge): a function $h(m,w)$ solving a conditional-moment (Fredholm integral) equation, which converts a proxy into an adjustment that recovers an interventional mean.
- completeness: a statistical condition stating that a family of conditional distributions is rich enough to identify any integrable function from its conditional expectations. It is the proximal analogue of an instrument-relevance condition and is the crux that this note alleges is hidden.
- proximal causal inference: identification of causal effects under unmeasured confounding using two proxies and completeness, via bridge functions.

## Problem and motivation

Unmeasured confounding by $U$ breaks two things at once in this graph. First, the mediator $M$ is no longer an unconfounded mediator, because $U \to M$ and $U \to Y$ create a backdoor path $M \leftarrow U \to Y$. Classical front-door identification of Pearl (1995) requires exactly that this path be absent, so front-door fails. Second, the treatment $X$ is confounded with $Y$ through $U \to X$ and $U \to Y$, so naive adjustment fails as well.

The proximal program of Miao, Geng, and Tchetgen Tchetgen (2018) repairs hidden confounding using two proxies of $U$, one negative-control outcome and one negative-control exposure, plus completeness. The single-proxy regime is the interesting frontier precisely because two valid proxies are often a luxury. The recurring question across this family of ideas is whether structural features of a specific graph can donate a second control for free, so that one measured proxy carries the load.

This note targets that gap by asking whether the front-door exclusion itself supplies the missing second control. The hope is that, on the $M \to Y$ leg, the treatment $X$ is a legitimate negative-control exposure, leaving only one measured proxy $W$ to find. The honesty requirement of this family means we must check whether that hope survives contact with the existing proximal mediation literature. It does not.

## Setup and causal graph

Directed edges:

- $U \to X$ (latent confounder affects treatment).
- $U \to M$ (latent confounder affects mediator; this is the edge that breaks front-door).
- $U \to Y$ (latent confounder affects outcome).
- $X \to M$ (treatment affects mediator).
- $M \to Y$ (mediator affects outcome).
- $U \to W$ (latent confounder generates the single proxy).

$U$ is unmeasured. $W$ is the single proxy. $M$ is the observed but confounded mediator. There is no direct edge $X \to Y$ (front-door exclusion E1). $W$ has no edge into or out of $\{X,M,Y\}$ (E2), and $W \perp (X,M,Y) \mid U$ (E3).

```
        U  (unmeasured)
      / | | \
     /  | |  \
    v   v v   v
    X   M Y    W   (W = single proxy, pure child of U)
     \  ^ ^
      \ | |
       vX->M  M->Y
   X --> M --> Y      (front-door path; no direct X -> Y edge)
```

Reading the DAG: $U$ fans out to $X$, $M$, $Y$, and $W$. The causal chain $X \to M \to Y$ is the front-door path. $W$ hangs off $U$ alone. The proposal additionally claims that, conditional on $(M,U)$, the treatment $X$ has no open path to $Y$, so $X$ can serve as a negative-control exposure on the $M \to Y$ leg.

## Target estimand

The target is the ATE contrast
$$
\mathrm{ATE}(x,x') = \psi(x) - \psi(x'), \qquad \psi(x) = \mathbb{E}[Y(\mathrm{do}\,X=x)] = \sum_m \mathbb{E}[Y(\mathrm{do}\,M=m)]\, p(M=m \mid \mathrm{do}\,X=x).
$$
The right-hand factorization is the front-door decomposition: the effect of $X$ on $Y$ is the mediator dose-response $\mathbb{E}[Y(\mathrm{do}\,M=m)]$ weighted by the interventional mediator law $p(M=m \mid \mathrm{do}\,X=x)$. Levels $\psi(x)$ may carry one global additive nuisance if the leg-2 bridge has a constant null direction; all contrasts cancel that nuisance.

## Assumptions

1. Front-door exclusion (E1).
   Formal: $Y(x,m) = Y(m)$, hence $\psi(x) = \sum_m \mathbb{E}[Y(\mathrm{do}\,M=m)]\, p(M=m \mid \mathrm{do}\,X=x)$.
   Plain reading: the treatment affects the outcome only through the mediator.
   Mildness: standard front-door, assumed by every front-door or proximal-mediation competitor.
   Testable: no, exclusion restrictions are not testable from the observed law alone.

2. Single-proxy validity (E2, E3).
   Formal: $W$ is a child of $U$ only, with $W \perp (X,M,Y) \mid U$ and no arrows $W \to \{X,M,Y\}$ or back.
   Plain reading: $W$ reflects $U$ and nothing on the causal path.
   Mildness: the standard negative-control-outcome condition, strictly weaker than requiring a second separate proxy.
   Testable: partially; some implied conditional independences can be checked, the latent-conditional part cannot.

3. $X$ is a valid NCE on the $M \to Y$ leg (the key structural claim).
   Formal: $X \perp Y \mid (M,U)$, alleged to follow from E1 plus the graph, together with $X \to M$ relevance ($p(M \mid X,U)$ varies in $X$).
   Plain reading: once $M$ and $U$ are fixed, the treatment carries no further information about the outcome, so $X$ behaves like a second negative control.
   Mildness: claimed free. The audit below shows the graphical statement is correct but that using it inside a bridge is exactly proximal mediation, so it is not a new donation.
   Testable: no, it conditions on $U$.

4. Outcome-bridge existence and completeness for leg 2.
   Formal: there exists $h(m,w)$ with $\mathbb{E}[\,Y - h(M,W) \mid M=m, X=x\,] = 0$ for all $x,m$. Existence and uniqueness modulo a constant require $p(W\mid U)$ and $p(X\mid M,U)$ to be jointly complete for $U$ on the $M \to Y$ leg.
   Plain reading: the proxy $W$ and the treatment $X$ jointly probe $U$ richly enough to solve for the bridge.
   Mildness: claimed equal to the baseline proximal mediation completeness. This is the condition the verdict flags as hidden; it is real and load-bearing.
   Testable: no in general.

5. Leg-1 proxy-richness / rank condition (the named cost).
   Formal: for point identification of $p(M=m \mid \mathrm{do}\,X=x)$, the family $\{p(W \mid U=u)\}$ must be injective in $u$ on the channels through which $U$ acts on $(X,M)$, equivalently the operator induced by $p(W \mid X,U)$ has trivial null space. Discrete sufficient form: $\mathrm{rank}$ of the $p(M,W\mid X)$ tables within $X$-strata at least the cardinality of effective $U$, so $|W| \ge |U_{\mathrm{eff}}|$.
   Plain reading: the proxy must separate the relevant latent states.
   Mildness: stated as a first-class cost. Untestable in general, with a checkable discrete sufficient version.
   Testable: the discrete rank version is checkable from $p(M,W\mid X)$; the latent injectivity is not.

6. Positivity.
   Formal: $0 < p(M=m \mid X=x) < 1$ and $p(X=x\mid U)$ bounded away from $0$ and $1$ on the support used.
   Plain reading: every treatment and mediator level occurs with positive probability.
   Mildness: routine, required by any front-door or adjustment estimator.
   Testable: the observed-data parts are checkable.

## The single mild assumption that replaces the second proxy

The proposal advertises Assumption 3 as the single mild condition that replaces the usual second proxy. It states that the treatment $X$ is a valid negative-control exposure on the $M \to Y$ leg, because $X \perp Y \mid (M,U)$ follows from front-door exclusion plus the DAG. The claim is that this lets $X$ play the role of the treatment-inducing proxy $Z$ in two-proxy proximal mediation, so the outcome bridge $h(m,w)$ is identified by the moment $\mathbb{E}[Y - h(M,W)\mid X=x, M\text{-stratum}] = 0$ with no $U$-additivity.

The specific two-proxy requirement it claims to stand in for is the existence of a separate treatment-inducing proxy $Z$ that, together with completeness, makes the leg-2 bridge well-posed. The graphical fact $X \perp Y \mid (M,U)$ is genuinely true. The fatal point, developed in the audit, is that conditioning the bridge on $X$ in this way is not a new substitution: it is precisely the construction Dukes, Shi, Shpitser, and Tchetgen Tchetgen (2023) already use, so the second proxy is not eliminated, it is renamed, and Assumption 4 (leg-2 completeness) is the same completeness the baseline needs.

## Identification result

Claim (as proposed, before audit). Under Assumptions 1 through 6, the effect decomposes into two legs built from the single proxy $W$, with leg 2 additionally conditioning on $X$.

Leg 2 ($M \to Y$, claimed point-identified, no additivity). With $W$ as NCO and $X$ as NCE for the single relation $M \to Y$, define the outcome bridge $h(m,w)$ by
$$
\mathbb{E}[\,Y - h(M,W)\mid M=m,\, X=x\,] = 0 \quad \text{for all } x,m,
\tag{BRIDGE-2}
$$
equivalently $\mathbb{E}[Y\mid M=m,X=x] = \mathbb{E}[h(m,W)\mid M=m,X=x]$ for all $x$. Then
$$
\mathbb{E}[Y(\mathrm{do}\,M=m)] = \mathbb{E}[h(m,W)] = \sum_w h(m,w)\, p(w).
\tag{LEG-2 ID}
$$

Leg 1 ($X \to M$ margin). With pseudo-outcome $\mathbf{1}\{M=m\}$, define $\phi_M(m,x,w)$ by
$$
\mathbb{E}[\mathbf{1}\{M=m\}\mid X=x, W=w] = \mathbb{E}[\phi_M(m,x,W')\mid X=x,W=w],
\tag{BRIDGE-1}
$$
then $p(M=m\mid \mathrm{do}\,X=x) = \mathbb{E}[\phi_M(m,x,W)]$, valid under Assumption 5.

Composition.
$$
\psi(x) = \sum_m \Big(\sum_w h(m,w)\,p(w)\Big)\, \mathbb{E}[\phi_M(m,x,W)], \qquad
\mathrm{ATE}(x,x') = \sum_m \Big(\sum_w h(m,w)\,p(w)\Big)\big(\mathbb{E}[\phi_M(m,x,W)] - \mathbb{E}[\phi_M(m,x',W)]\big).
$$
Point-identified when Assumption 5 holds; sharp partial identification otherwise, with leg 2 fixed and the leg-1 do-margin ranging over a polytope of latent structures consistent with $p(X,M,W)$.

Status of the claim. The leg-2 identification (BRIDGE-2) and (LEG-2 ID) are correct as mathematics, but they are not new and they do depend on Assumption 4 completeness. The verdict therefore eliminates the idea: the novel-looking step is a relabeling of proximal mediation.

## Proof sketch

Step 1 (front-door composition). Inputs: E1. Derived: $\psi(x) = \sum_m \mathbb{E}[Y(\mathrm{do}\,M=m)]\,p(M=m\mid \mathrm{do}\,X=x)$. Justification: standard do-calculus on the front-door path, valid as an algebraic identity even though each factor is not a naive conditional because $U$ confounds both legs.

Step 2 (leg 2 as a double-negative-control problem, where the single proxy enters). Inputs: E2, E3 give $W \perp Y\mid (M,U)$ with $W$ associated with $U$; E1 plus the graph give $X \perp Y\mid (M,U)$ with $X$ associated with $U$ through $M$. Derived: the pair $(W,X)$ is a valid two-control system for the lone relation $M \to Y$, so (BRIDGE-2) has a solution $h$ and (LEG-2 ID) holds. Justification: this is the Miao-Geng-Tchetgen Tchetgen and Cui et al. outcome-bridge theorem applied to the relation $M \to Y$ with controls $(W,X)$. The single measured proxy $W$ enters here as the NCO; the second control is $X$. This step requires Assumption 4 completeness for existence and uniqueness, contradicting the claim that completeness is avoided.

Step 3 (leg 1, honest). Inputs: Assumption 5 injectivity of $\{p(W\mid U)\}$ on the relevant channels. Derived: (BRIDGE-1) is well-posed and $p(M=m\mid \mathrm{do}\,X=x) = \mathbb{E}[\phi_M(m,x,W)]$. Justification: single-proxy proximal-backdoor identification. When injectivity fails, the set of latent structures matching $p(X,M,W)$ is a polytope, the do-margin is its image under a linear map, so its range is a sharp interval.

Step 4 (level versus contrast). Inputs: leg-2 bridge null direction. Derived: $\psi(x)$ carries at most one global additive nuisance, which is $m$- and $x$-free and cancels in $\mathrm{ATE}(x,x') = \psi(x) - \psi(x')$. Justification: linearity of the composition in the nuisance constant.

The chain is internally consistent. The defect is not an arithmetic gap but a novelty and accounting failure exposed in Step 2: completeness is used, not removed, and the construction is prior art.

## Why a single proxy suffices

The proposal's crux argument is that the front-door exclusion makes $X$ a second negative-control exposure on the $M \to Y$ leg, so only one measured proxy $W$ is needed. The graphical statement $X \perp Y\mid (M,U)$ is correct: conditioning on $M$ closes the only $X$-to-$Y$ route, and conditioning on $U$ blocks the backdoor $X \leftarrow U \to Y$.

The argument fails as a novelty claim. In proximal mediation the treatment is already the variable on which the outcome bridge for the $M \to Y$ leg is conditioned. Writing $X$ into (BRIDGE-2) and calling it a "free second NCE" does not reduce the number of measured proxies below the proximal-mediation baseline, because that baseline also uses one measured outcome proxy and conditions on the treatment. The single measured proxy $W$ does not suffice for anything that proximal mediation did not already deliver with the same single measured outcome proxy. So the crux argument describes existing machinery in new words rather than weakening the requirement.

## Estimator

The proposed estimator is three-stage and cross-fit. Stage 2 solves (BRIDGE-2) for $h(m,w)$ by GMM or minimax kernel-IV using the moment $\mathbb{E}[(Y-h(M,W))\rho(X,M)] = 0$ over test functions $\rho$, with $X$ as the conditioning instrument; for discrete $(M,W,X)$ this is the linear solve $\mathbb{E}[Y\mid m,x] = \sum_w h(m,w)\,p(w\mid m,x)$, then $\theta(m) = \mathbb{E}_n[h(m,W)]$. Stage 1 solves the proximal-backdoor bridge $\phi_M(m,x,w)$ by GMM with $W$ as proxy and sets $\pi(m,x) = \mathbb{E}_n[\phi_M(m,x,W)]$; for discrete low-cardinality $U$ this is matrix inversion of the $p(M\mid X,W)$ blocks, and the empirical rank or condition number diagnoses Assumption 5. If rank-deficient, the estimator switches to partial-ID mode and solves two linear programs for $\min$ and $\max$ of $\sum_m \theta(m)[\pi(m,x)-\pi(m,x')]$ over latent $\{p(U), p(X\mid U), p(M\mid X,U)\}$ subject to observed-moment constraints, returning sharp bounds. Stage 3 sets $\widehat{\mathrm{ATE}} = \sum_m \widehat\theta(m)(\widehat\pi(m,x)-\widehat\pi(m,x'))$. Inference stacks the moments and uses the front-door efficient influence function with $(h,\phi_M,\text{$M$-density})$ as Neyman-orthogonal cross-fit nuisances. This pipeline is sound, but it is the proximal-mediation estimator plus a single-proxy backdoor stage, not a new estimand-enabling device.

## Worked intuition or a small concrete example

Take all of $X,M,W,U,Y$ binary. On leg 2, fixing $M=m$ and varying $X$ probes the latent mixture $p(U\mid M=m,X=x)$. The bridge $h(m,w)$ solves a $2\times 2$ linear system per $m$, recovering $\mathbb{E}[Y(\mathrm{do}\,M=m)] = \sum_w h(m,w)p(w)$. This is exactly the binary proximal calculation with the "second control" being $X$. The intuition that makes the failure tangible: an analyst who already ran binary proximal mediation would write the identical $2\times 2$ system; renaming $Z$ to $X$ changes no numbers. On leg 1, if $W$ and $U$ are both binary and $p(W\mid U)$ is non-degenerate (distinct rows), the $2\times 2$ $p(M\mid X,W)$ block is invertible and the do-margin is recovered; if the two $U$-states share $p(W\mid U)$, the block is rank-deficient and only bounds follow.

## Soundness audit

The matched adversary finding in the staging data is null, so the controlling objection is reconstructed from the ranking verdict, which is decisive.

Primary objection (duplication, fatal). Conditioning the leg-2 outcome bridge on the treatment $X$ is exactly the proximal mediation construction of Dukes, Shi, Shpitser, and Tchetgen Tchetgen (2023). In that work the outcome bridge for the $M \to Y$ leg is identified using a measured outcome proxy and conditioning on the treatment. The present idea's "$X$ as a free second negative-control exposure" is the same object with the treatment-inducing proxy relabeled as $X$. There are therefore not two structural models that agree on the observed law and disagree on the estimand; rather, the two "models" are the proposed method and the published proximal mediation method, which agree on everything, including the estimand. The disagreement is purely in the novelty narrative. Severity: maximal, because it removes the claimed contribution rather than the validity of the formulas.

Secondary objection (hidden completeness). The proposal repeatedly asserts that leg 2 needs no $U$-additivity and frames this as evading the usual price. It does evade additivity, but it silently relies on Assumption 4, the joint completeness of $p(W\mid U)$ and $p(X\mid M,U)$ for $U$ on the $M \to Y$ leg. That is the standard proximal completeness, present and load-bearing. The claim of "no completeness on leg 2" is false; the claim should read "the same completeness as proximal mediation." This is the leg-2 completeness the verdict says is hidden.

Residual genuine content. The leg-1 single-proxy backdoor stage with an explicitly named rank condition, and the linear-program sharp bounds when only relevance holds, are honest and non-trivial, but they are not enough to carry a paper once leg 2 is recognized as prior art.

## Honest status and the fix

Final status: eliminated. The exact failure mode is that the load-bearing leg-2 step is identical to Dukes, Shi, Shpitser, and Tchetgen Tchetgen (2023) proximal mediation, with the treatment relabeled as the second negative-control exposure, and the leg-2 completeness condition is real and undisclosed rather than removed. The "single measured proxy" framing does not reduce the proxy count below that paper's baseline, which also uses a single measured outcome proxy while conditioning on the treatment.

Cautionary lesson: a graphical conditional independence that holds for free ($X \perp Y\mid (M,U)$) does not by itself create a new identification result. Whether it yields novelty depends on whether the resulting moment equation differs from existing constructions. Here it does not.

Repair path and sibling. The genuinely new content that survives this critique is the coupled $g$-functional that identifies the full $X \to Y$ effect through the confounded mediator without introducing any separate measurement proxy $Z$, which is developed as idea 8. This note should be read as the non-novel sibling that motivates idea 8: idea 8 keeps the single-measured-variable ambition but couples the two legs into one functional rather than re-deriving proximal mediation on leg 2. Anyone tempted by "treatment as a free second NCE" should go directly to idea 8.

## Novelty vs prior work

Closest prior work: Dukes, Shi, Shpitser, and Tchetgen Tchetgen, "Proximal mediation analysis" (arXiv:2109.11904; Biometrika 110(4):973, 2023). That paper identifies mediated effects under unmeasured confounding using proxies and bridge functions, conditioning the outcome bridge on the treatment. The proposed delta was "use one measured proxy by recognizing $X$ as the second NCE on the $M \to Y$ leg." The audit shows the delta is null on leg 2: the bridge moment, the conditioning on $X$, and the completeness requirement coincide with that paper. The only residual delta is the leg-1 single-proxy backdoor stage with named rank condition and the linear-program partial-ID fallback, which is too thin to constitute a separate contribution and is better folded into idea 8.

## AISTATS significance and positioning

As a standalone submission this idea is not viable: a reviewer familiar with proximal mediation will identify the leg-2 step as a relabeling and reject on novelty, exactly as the internal verdict did. The constructive positioning is to cite this mechanism as a foil inside the idea-8 paper. There it demonstrates why the naive "treatment as free second control" route does not improve on existing proximal mediation, motivating the coupled-functional alternative that needs no separate measurement proxy. Positioned that way, the failed attempt sharpens the contribution of the sibling work rather than competing with it.

## Open problems and next steps

- Make precise, inside idea 8, the exact sense in which a coupled $g$-functional escapes the leg-2 duplication that sinks this note.
- Characterize when the leg-1 linear-program bounds are informative (strictly inside the no-assumptions Manski bounds) as a function of the proxy strength of $W$, since that partial-ID content is the only salvageable piece here.
- Determine whether any front-door variant exists where the treatment supplies a control that is provably not equivalent to the proximal-mediation conditioning, which would be the only way to revive the headline claim.
- State a checkable discrete diagnostic for Assumption 4 completeness so future single-proxy proposals cannot hide it.

## References

- Pearl, J. (1995). Causal diagrams for empirical research. Front-door criterion. Requires an unconfounded mediator.
- Miao, W., Geng, Z., and Tchetgen Tchetgen, E. (2018). Identifying causal effects with proxy variables of an unmeasured confounder. Biometrika. Canonical two-proxy point identification with completeness.
- Cui, Y., et al. (2024). Semiparametric proximal causal inference. JRSS-B. Two-proxy point identification.
- Dukes, O., Shi, X., Shpitser, I., and Tchetgen Tchetgen, E. (2023). Proximal mediation analysis. arXiv:2109.11904; Biometrika 110(4):973. Closest prior work; the leg-2 construction here coincides with it.
- Tchetgen Tchetgen, E., Ying, A., Cui, Y., Shi, X., and Miao, W. (2024). Single proxy control. Biometrics 80(2):ujae027. Single proxy via additive equiconfounding.
- Sofer, T., et al. (2016). Additive equiconfounding negative-control methods.
- Fulcher, I., Shi, X., and Tchetgen Tchetgen, E. (2025). Causal inference with hidden mediators. Biometrika 112(1):asae037; arXiv:2111.02927. Proxies the mediator under mediator-outcome unconfoundedness.
- Kuroki, M., and Pearl, J. (2014). Measurement bias and effect restoration. Biometrika. Single proxy with known $p(W\mid U)$ or rank preservation.
- Manski, C. (sharp partial identification tradition). Basis for the linear-program bounds used in the leg-1 fallback.
