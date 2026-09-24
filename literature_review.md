# Literature Review: Proximal Inference, COCA, and Cross-View Proxy Balancing

Created: 2026-08-26

Source provenance: the Zotero library was checked first. The comparison was then verified against the primary journal articles, their legal open-access versions, and the project-local bibliography.

## 1. Scope and bottom line

This review compares three ways to use proxy measurements when treatment and outcome share an unobserved cause. The methods solve related but different identification problems.

Let $A\in\{0,1\}$ denote treatment, $X$ observed pretreatment covariates, and $Y(a)$ the outcome that would be observed if treatment were set to $a$. Let $U$ denote an unobserved pretreatment variable that can affect both treatment and outcome. Two causal targets will appear:

$$
\tau_{\mathrm{ATE}}
\triangleq
\mathbb E\{Y(1)-Y(0)\}
$$

and

$$
\tau_{\mathrm{ETT}}
\triangleq
\mathbb E\{Y(1)-Y(0)\mid A=1\}.
$$

The first is the average treatment effect in the population. The second is the effect of treatment on the treated, also called the ATT.

The main conclusion is that no method uniformly dominates the others.

- Standard proximal causal inference has the most developed general theory for population ATE under unmeasured confounding. It requires scientifically distinct treatment-inducing and outcome-inducing proxy roles, a bridge existence condition, and completeness or related operator conditions. Estimation can require solving an ill-posed inverse problem.
- COCA and Single Proxy Control use one negative-control outcome as a proxy for the untreated potential outcome $Y(0)$. This economy is attractive, but the basic nonparametric target is the ETT rather than the population ATE. Its shielding assumption is generally untestable, and its bridge existence condition is nonautomatic.
- Cross-view proxy balancing targets the population ATE. It learns a low-dimensional representation from part of a high-dimensional proxy and audits the representation with a held-out proxy block. Exact observed balance becomes ordinary adjustment only under cross-view structural assumptions and outcome-relevant completeness. Approximate balance requires an additional stability budget.

The defensible claim for cross-view proxy balancing is therefore a different tradeoff, not uniformly weaker assumptions. It exchanges direct bridge estimation for representation learning, held-out balance, balance attainability, and an outcome-relevant operator condition.

## 2. Standard proximal causal inference

### 2.1 Two proxy roles

Standard proximal causal inference uses two proxy roles for the hidden confounder $U$. The literature usually denotes them by $Z$ and $W$. In this section only, $Z$ is a treatment-inducing proxy and $W$ is an outcome-inducing proxy. The learned representation in our method will later be denoted by $Z_j$.

A common point-treatment formulation combines latent exchangeability,

$$
Y(a)\perp\!\!\!\perp A\mid(U,X),
$$

with proxy restrictions such as

$$
Y\perp\!\!\!\perp Z\mid(A,U,X)
$$

and

$$
W\perp\!\!\!\perp(A,Z)\mid(U,X).
$$

The first proxy restriction says that after $(A,U,X)$ is fixed, the treatment-inducing proxy $Z$ carries no additional information about the outcome. The second says that after $(U,X)$ is fixed, the outcome-inducing proxy $W$ has a common measurement law across treatment and $Z$. These statements involve the unobserved variable $U$, so the observed data cannot generally verify them.

### 2.2 The outcome bridge

An outcome bridge is a function $h$ satisfying the observed conditional moment equation

$$
\mathbb E(Y\mid Z,A,X)
=
\mathbb E\{h(W,A,X)\mid Z,A,X\}.
\tag{P1}
$$

Equation (P1) is a Fredholm integral equation of the first kind. Bridge existence means that the observed outcome regression on the left belongs to the range of the conditional expectation operator on the right. This range condition is not automatic.

A representative latent completeness condition is

$$
\mathbb E\{g(U)\mid Z,A=a,X=x\}=0
\quad\Longrightarrow\quad
g(U)=0
$$

for every square-integrable function $g$ and relevant $(a,x)$. This condition makes the treatment-inducing proxy sufficiently informative about functions of $U$. Under the complete assumption package, the observed bridge can be transferred to the latent level, leading to

$$
\mathbb E\{Y(a)\}
=
\mathbb E\{h(W,a,X)\}.
\tag{P2}
$$

The population ATE is then the difference between equation (P2) at $a=1$ and $a=0$.

Bridge existence, completeness, and bridge uniqueness are separate questions. Completeness can establish an injectivity property or uniqueness in a chosen function space. Identification of the causal functional does not always require the bridge itself to be unique, because multiple bridge solutions can yield the same value of equation (P2).

### 2.3 Estimation and practical assessment

Standard proximal methods estimate an outcome bridge, a treatment bridge, or both. Semiparametric theory provides proximal inverse-weighted, outcome-regression, and doubly robust estimators. These are substantial advantages because the framework directly handles ATE and, in extensions, ATT under unmeasured confounding.

The practical costs are also substantial.

1. The analyst must assign scientifically defensible treatment-inducing and outcome-inducing proxy roles.
2. The proxy restrictions condition on the hidden $U$ and are generally untestable.
3. Bridge existence is a nonautomatic range condition.
4. Completeness requires proxies to contain sufficiently rich variation relative to the hidden confounder.
5. Bridge estimation can be unstable because Fredholm inverse problems are often ill-posed.

Thus, standard proximal inference is broad and mature, but it is not a free consequence of observing two variables correlated with $U$.

Primary sources for this section are [Miao, Geng, and Tchetgen Tchetgen (2018)](https://doi.org/10.1093/biomet/asy038), [Tchetgen Tchetgen et al. (2024)](https://doi.org/10.1214/23-STS911), and [Cui et al. (2024)](https://doi.org/10.1080/01621459.2023.2191817).

## 3. COCA and Single Proxy Control

### 3.1 What it means for $W$ to proxy $Y(0)$

COCA uses one negative-control outcome $W$. Its core proxy conditions can be written as

$$
W(a)=W,
$$

$$
W\not\!\perp\!\!\!\perp Y(0)\mid X,
$$

and

$$
W\perp\!\!\!\perp A\mid\{Y(0),X\}.
\tag{C1}
$$

The first condition says that treatment does not causally change the negative-control outcome. The second requires $W$ to contain information about the untreated potential outcome. Equation (C1) is the load-bearing shielding condition. Once $Y(0)$ and $X$ are fixed, $W$ must provide no additional information about treatment assignment.

Calling $W$ a proxy for $Y(0)$ does not require $W$ to be a literal noisy measurement of $Y(0)$. It means that $W$ predicts $Y(0)$ and that $Y(0)$ shields $W$ from treatment selection after conditioning on $X$. A pretreatment measurement of the same outcome process can be a plausible candidate. The condition becomes implausible when treatment selection directly uses $W$, or when a hidden variable affects both $A$ and $W$ through information not fully summarized by $(Y(0),X)$.

This proxy interpretation is more outcome-specific than saying that $W$ measures a generic hidden confounder $U$. The specificity is both a strength and a limitation. It allows one proxy to be useful because $Y(0)$ is observed among untreated units, but it ties validity to one potential outcome and one causal target.

### 3.2 The COCA bridge

The COCA bridge approach assumes that a function $b$ exists such that

$$
y
=
\mathbb E\{b(W,X)\mid Y=y,X=x,A=0\}.
\tag{C2}
$$

The variable $y$ appears on both sides because equation (C2) is a calibration equation. Among untreated units, consistency gives $Y=Y(0)$. The equation asks for a transformation of $(W,X)$ whose conditional mean, at each untreated outcome value $y$, returns that same value $y$. Equivalently, at the structural level,

$$
Y(0)
=
\mathbb E\{b(W,X)\mid Y(0),X\}.
$$

Equation (C2) is not guaranteed for an arbitrary proxy. Let $T_x$ map a function of $W$ to its conditional expectation given $(Y,X=x,A=0)$. Bridge existence requires the identity function $y\mapsto y$ to belong to the range of $T_x$. This is another Fredholm range condition.

There are important cases where the bridge is natural. If $W=Y(0)+\varepsilon$ with conditionally mean-zero measurement error, the identity transformation can supply a bridge. Informative binary $W$ and binary $Y(0)$ also admit an explicit bridge under nondegeneracy. In a general continuous model, existence can fail. A solution can also be nonunique or unstable.

Bridge uniqueness is not required for the COCA target once a valid solution exists. Completeness is a sufficient route to uniqueness, but the identified ETT functional can remain unique across multiple bridge solutions.

### 3.3 Target and practical assessment

The basic nonparametric COCA target is the ETT. The bridge gives

$$
\mathbb E\{Y(0)\mid A=1\}
=
\mathbb E\{b(W,X)\mid A=1\},
$$

and therefore

$$
\tau_{\mathrm{ETT}}
=
\mathbb E\{Y-b(W,X)\mid A=1\}.
\tag{C3}
$$

COCA also provides an extended-propensity-score route and a doubly robust estimator. Its central practical advantage is that one negative-control outcome can suffice for the ETT. Population ATE generally requires additional structure, such as rank preservation or a second proxy appropriate for $Y(1)$.

The control-arm equation (C2) involves observed variables and can be fitted or challenged empirically. A good fit does not verify equation (C1). Treated units' $Y(0)$ values are unobserved, so the core shielding assumption remains generally untestable.

The primary source for the nonparametric formulation is [Park, Richardson, and Tchetgen Tchetgen (2024)](https://doi.org/10.1093/biomtc/ujae027). The earlier parametric COCA formulation is [Tchetgen Tchetgen (2014)](https://doi.org/10.1093/aje/kwt303).

## 4. Cross-view proxy balancing

### 4.1 Representation and held-out balance

Our project begins with one high-dimensional candidate proxy divided into prespecified blocks:

$$
W=(W_1,\ldots,W_J).
$$

For a block index $j$, let $W_{-j}$ contain all blocks except $W_j$. The representation is

$$
Z_j=\phi_j(X,W_{-j}).
$$

The block $W_j$ is excluded from the input to $\phi_j$. It is used with $X$ as the held-out audit view. The population exact-balance endpoint is

$$
(X,W_j)\perp\!\!\!\perp A\mid Z_j.
\tag{B1}
$$

This separation prevents a representation from satisfying its audit merely by copying the audited block. Equation (B1) is an observed-distribution property for a fixed representation. In finite data, it can only be assessed with a statistical tolerance or equivalence procedure.

### 4.2 Structural cross-view assumptions

The current theory requires, for each treatment level $a$,

$$
A\perp\!\!\!\perp\{Y(a),W_{-j}\}\mid(X,U)
\tag{B2}
$$

and

$$
W_j\perp\!\!\!\perp(A,W_{-j})\mid(X,U).
\tag{B3}
$$

Equation (B2) preserves latent exchangeability after conditioning on a representation built from $W_{-j}$. Equation (B3) makes the held-out block follow the same measurement law in both treatment arms and separates it from the representation blocks after $(X,U)$ is fixed.

Although the data contain one object named $W$, equations (B2) and (B3) require meaningful multiple views inside that object. High dimension alone does not create valid views. Repeated measurements, separate sensors, or scientifically distinct measurement channels provide a more credible basis than an arbitrary coordinate split.

### 4.3 Outcome-relevant completeness

Fix a representation value $z$. Let $Q$ denote a candidate conditional distribution of $(X,U)$ among units with $Z_j=z$. The held-out measurement mechanism maps $Q$ to the corresponding observed distribution of $(X,W_j)$. Denote this mapping by $\mathcal K_jQ$.

For treatment level $a$, define

$$
\mu_{a,j,z}(x,u)
\triangleq
\mathbb E\{Y(a)\mid X=x,U=u,Z_j=z\}.
$$

Outcome-relevant completeness requires, for every pair $Q,Q'$ in a prespecified admissible class,

$$
\mathcal K_jQ=\mathcal K_jQ'
\quad\Longrightarrow\quad
\int\mu_{a,j,z}\,dQ
=
\int\mu_{a,j,z}\,dQ'
\tag{B4}
$$

for both $a=0$ and $a=1$.

Equation (B4) says that if the held-out audit view cannot distinguish two latent distributions, then those distributions must agree on the two potential-outcome means needed for the ATE. It does not require $Q=Q'$. It also does not imply

$$
(X,U)\perp\!\!\!\perp A\mid Z_j.
$$

To see the operator structure, define a signed contrast $\nu=Q-Q'$ and the outcome functional

$$
\ell_{a,j,z}(\nu)
\triangleq
\int\mu_{a,j,z}\,d\nu.
$$

Then equation (B4) is the null-space inclusion

$$
\ker(\mathcal K_j)
\subseteq
\ker(\ell_{a,j,z})
$$

over the prespecified contrast class. This condition is class-wide and therefore is not a restatement about only the realized treated-control contrast. It is not tautological. It is nevertheless strong when the admissible class contains all integrable probability laws supported on the representation stratum. In that case, every latent direction invisible to the held-out proxy must also be irrelevant to each potential-outcome mean.

Full mixture injectivity would require

$$
\mathcal K_jQ=\mathcal K_jQ'
\quad\Longrightarrow\quad
Q=Q'.
$$

Full injectivity implies outcome-relevant completeness. The reverse implication does not hold. Outcome-relevant completeness protects the target functional rather than recovering the full latent distribution.

### 4.4 Identification and approximate balance

For a fixed representation, define its ordinary adjustment functional by

$$
\tau_{Z_j}
\triangleq
\sum_zP(Z_j=z)
\left[
\mathbb E(Y\mid A=1,Z_j=z)
-
\mathbb E(Y\mid A=0,Z_j=z)
\right].
$$

Under consistency, overlap, equations (B2) and (B3), outcome-relevant completeness, and exact balance in equation (B1), this observed-data functional equals the population ATE:

$$
\tau_{\mathrm{ATE}}
=
\tau_{Z_j}.
\tag{B5}
$$

This is the central practical benefit. The final effect estimator adjusts for the learned low-dimensional $Z_j$ and need not estimate a proximal bridge.

Let $D_j(Z_j)$ denote a nonnegative population discrepancy for the two treatment-arm distributions of $(X,W_j)$ within $Z_j$, chosen so that $D_j(Z_j)=0$ exactly when equation (B1) holds. The point-identification theorem is conditional on attaining exact balance. A restricted representation class can satisfy

$$
\inf_{\phi_j}D_j(\phi_j)>0,
$$

where $D_j$ is the population discrepancy that measures violation of equation (B1). The theory does not guarantee that an exact representation exists or that an optimizer finds it.

Approximate balance requires a separate quantitative condition. If a finite stability constant $\Gamma_j$ controls every outcome-relevant latent contrast by its observed audit discrepancy, then

$$
|\tau_{Z_j}-\tau_{\mathrm{ATE}}|
\le
\Gamma_jD_j(Z_j).
\tag{B6}
$$

Outcome-relevant completeness alone proves only the exact zero-to-zero implication. It does not prove equation (B6). The stability constant $\Gamma_j$ is not generally identified from the observed single-proxy law, so equation (B6) is normally reported as a sensitivity analysis over scientifically plausible values of $\Gamma_j$.

The method can rotate the held-out block and aggregate mask-specific estimates. Robust median aggregation requires a strict majority of masks to satisfy the relevant structural conditions. High dimension supplies candidate redundancy, not majority validity.

## 5. Cold comparison

| Axis | Standard proximal inference | COCA / Single Proxy Control | Cross-view proxy balancing |
|---|---|---|---|
| Primary target | Population ATE, with ATT extensions | ETT or ATT | Population ATE |
| Proxy interpretation | Treatment-inducing and outcome-inducing proxies for $U$ | One negative-control proxy for $Y(0)$ | Conditionally separated blocks inside a high-dimensional proxy for $U$ |
| Number of proxy roles | Two scientifically distinct roles | One outcome-side role | Representation blocks plus one held-out audit block per mask |
| Main observed relation | Proximal bridge moment equation | COCA bridge or extended propensity score equation | Conditional balance of $(X,W_j)$ given $Z_j$ |
| Main hidden restriction | Latent proxy restrictions and completeness | $W\perp A\mid\{Y(0),X\}$ | Representation safety, view separation, and outcome-relevant completeness |
| Identification mechanism | Solve a bridge and evaluate the proximal g-formula | Recover the treated units' mean $Y(0)$ | Transfer exact held-out balance to zero outcome-relevant bias |
| Final effect calculation | Proximal bridge functional | Bridge, extended propensity score, or doubly robust ETT estimator | Ordinary adjustment or AIPW using $Z_j$ |
| Full latent recovery required | No | No | No |
| Main inverse problem | Bridge estimation | Bridge or extended propensity score estimation | ORC is a null-space restriction; approximate claims require stability |
| Population-observable component | Bridge moment restrictions | Control-arm bridge equation | Exact balance and overlap for a fixed representation |
| Main practical benefit | Mature general ATE theory and semiparametric estimation | One proxy can identify ETT | High-dimensional representation learning plus ordinary ATE adjustment |
| Main practical cost | Two valid proxy roles and potentially ill-posed bridge estimation | Outcome-specific shielding, bridge existence, and ETT target | Valid view separation, balance attainability, ORC, and sensitivity calibration |

### 5.1 The operator assumptions are different

The three central operator conditions should not be treated as interchangeable.

Standard proximal completeness is often an injectivity condition for a conditional expectation operator acting on every suitable function of $U$. It is outcome-agnostic and can be stronger than preserving only the two outcome means.

Cross-view outcome-relevant completeness asks only that the held-out proxy operator preserve the two linear functionals needed for $Y(0)$ and $Y(1)$. It is therefore weaker than full latent mixture injectivity. With a support-wide admissible class, it remains a demanding structural condition.

The COCA bridge is a range condition. It asks whether the identity function of $Y(0)$ can be represented as a conditional expectation of a transformation of $W$. It does not require full recovery of the latent distribution.

In finite-dimensional linear algebra, null-space inclusion and factorization through an operator are closely related. In infinite-dimensional models, obtaining an actual measurable or square-integrable bridge from a null-space condition can require continuity and closed-range regularity. It is therefore inaccurate to say that cross-view ORC removes all bridge-like structure. A precise statement is:

> Outcome-relevant completeness expresses only the target-relevant part of the inverse problem in null-space form, and the proposed estimator does not explicitly estimate a bridge.

### 5.2 There is no global ordering of assumption strength

Comparing only operator conditions, outcome-relevant completeness is narrower than full mixture injectivity and often narrower than latent completeness. Comparing the full identification packages, no general ordering exists.

- Standard proximal inference pays for two scientifically designated proxy roles and bridge inversion. It does not require the analyst to learn an exactly balanced representation.
- COCA pays for a strong $Y(0)$ shielding condition and targets ETT. It does not require a second proxy view.
- Cross-view balancing pays for representation-safe exchangeability, conditional separation of proxy blocks, exact-balance attainability, and outcome-relevant completeness. It avoids direct bridge estimation at the final adjustment stage and targets population ATE.

The project should therefore avoid saying that its assumptions are weaker overall. They are different and attractive in a different data regime.

## 6. What can be checked from data

The phrase "testable assumption" needs careful qualification.

| Condition | Population relation uses only observed variables? | Can finite data prove it exactly? |
|---|---:|---:|
| Representation overlap | Yes | No, but violations and weak overlap can be diagnosed |
| Exact cross-view balance for fixed $Z_j$ | Yes | No, but equivalence tests and held-out discrepancies can assess it |
| Standard proximal bridge equation | Yes | No, but its observable moment fit can be assessed |
| COCA control-arm bridge equation | Yes | No, but its observable moment fit can be assessed |
| Proximal proxy restrictions given $U$ | No | No |
| COCA shielding given $Y(0)$ | No for treated units | No |
| Cross-view separation given $U$ | No | No |
| Outcome-relevant completeness | No | No |
| Approximate stability budget $\Gamma_j$ | Generally no | No |
| Strict-majority mask validity | No | No |

Exact observed balance is therefore an observable endpoint, not a complete set of testable identification assumptions. A large held-out discrepancy can refute exact balance. A small empirical discrepancy cannot establish view validity, outcome-relevant completeness, or a finite stability budget.

## 7. Practical decision guide

Use standard proximal inference when the scientific design provides defensible treatment-inducing and outcome-inducing proxies, the population ATE is the target, and bridge estimation can be regularized and stress-tested.

Use COCA when a credible negative-control outcome measures the untreated outcome process, the shielding condition is scientifically plausible, and the ETT is the relevant target. A baseline or parallel outcome measurement is more convincing than an arbitrary correlate of a generic latent confounder.

Use cross-view proxy balancing when the data contain a high-dimensional pretreatment measurement with scientifically defensible redundant views, a low-dimensional balancing representation is plausible, and ordinary adjustment after representation learning is operationally valuable. Separate sensors, repeated assays, repeated measurements, or prespecified measurement blocks provide a stronger basis than arbitrary coordinate masking.

Use sensitivity or partial identification when the proxy views are weak, bridge existence is doubtful, exact balance is unattainable, or the structural assumptions cannot be defended. In those regimes, a point estimate produced by any one of the three methods can be more precise than the scientific assumptions justify.

## 8. Defensible novelty and exact non-claims

The broad idea of using one proxy for causal identification is not new. The project should not claim the first single-proxy identification result. COCA, Kernel Single Proxy Control, and recent work with a known proxy measurement mechanism already prevent that claim.

The defensible contribution is the following combination:

1. partition a high-dimensional candidate proxy into representation and held-out audit views;
2. learn a low-dimensional representation using an observable conditional balance objective;
3. use outcome-relevant completeness to turn exact held-out balance into population ATE identification by ordinary adjustment;
4. use quantitative stability to turn approximate balance into a sensitivity-indexed bias certificate;
5. rotate the held-out block and use robust aggregation when a strict majority of masks are valid.

This positioning must retain the following non-claims.

- One arbitrary scalar proxy does not generally identify the population ATE.
- High dimension does not automatically produce conditionally independent proxy views.
- Exact balance does not establish causal identification without the structural cross-view and outcome-relevance assumptions.
- Outcome-relevant completeness does not imply full latent balance or $U\perp\!\!\!\perp A\mid Z_j$.
- Outcome-relevant completeness alone does not make approximate discrepancy control causal bias.
- The stability constant $\Gamma_j$ is not generally identified from the observed data.
- The method is not uniformly weaker than standard proximal inference or COCA.
- The theory does not guarantee that the representation class contains an exact solution or that optimization recovers it.

A concise positioning statement is:

> Standard proximal inference solves an inverse bridge problem using treatment-inducing and outcome-inducing proxies. COCA obtains single-proxy identification of the effect on the treated by treating a negative-control outcome as a proxy for the untreated potential outcome. Cross-view proxy balancing instead learns a low-dimensional adjustment representation from one set of proxy blocks and audits it with a held-out block. Exact cross-view balance and outcome-relevant completeness identify the population ATE by ordinary adjustment, while approximate balance has causal meaning only under an additional stability budget.

## 9. Primary references

1. Miao, W., Geng, Z., and Tchetgen Tchetgen, E. J. (2018). Identifying Causal Effects With Proxy Variables of an Unmeasured Confounder. *Biometrika*, 105(4), 987-993. [https://doi.org/10.1093/biomet/asy038](https://doi.org/10.1093/biomet/asy038)
2. Tchetgen Tchetgen, E. J., Ying, A., Cui, Y., Shi, X., and Miao, W. (2024). An Introduction to Proximal Causal Inference. *Statistical Science*, 39(3), 375-390. [https://doi.org/10.1214/23-STS911](https://doi.org/10.1214/23-STS911)
3. Cui, Y., Pu, H., Shi, X., Miao, W., and Tchetgen Tchetgen, E. J. (2024). Semiparametric Proximal Causal Inference. *Journal of the American Statistical Association*, 119(546), 1348-1359. [https://doi.org/10.1080/01621459.2023.2191817](https://doi.org/10.1080/01621459.2023.2191817)
4. Park, C., Richardson, D. B., and Tchetgen Tchetgen, E. J. (2024). Single Proxy Control. *Biometrics*, 80(2), ujae027. [https://doi.org/10.1093/biomtc/ujae027](https://doi.org/10.1093/biomtc/ujae027)
5. Tchetgen Tchetgen, E. J. (2014). The Control Outcome Calibration Approach for Causal Inference With Unobserved Confounding. *American Journal of Epidemiology*, 179(5), 633-640. [https://doi.org/10.1093/aje/kwt303](https://doi.org/10.1093/aje/kwt303)
6. Kallus, N., Mao, X., and Uehara, M. (2021). Causal Inference Under Unmeasured Confounding With Negative Controls: A Minimax Learning Approach. [https://arxiv.org/abs/2103.14029](https://arxiv.org/abs/2103.14029)
7. Xu, L. and Gretton, A. (2023). Kernel Single Proxy Control for Deterministic Confounding. [https://arxiv.org/abs/2308.04585](https://arxiv.org/abs/2308.04585)
8. Ghassami, A., Zhang, Y., Shpitser, I., and Tchetgen Tchetgen, E. J. (2026 version). Partial Identification of Causal Effects Using Proxy Variables. [https://arxiv.org/abs/2304.04374](https://arxiv.org/abs/2304.04374)
9. Vollmer, S., Pfister, N., and Weichwald, S. (2026). Identifying Causal Effects Using a Single Proxy Variable. [https://arxiv.org/abs/2604.09135](https://arxiv.org/abs/2604.09135)

The first five references provide the main comparison. References 6 through 9 delimit adjacent estimation, single-proxy, and partial-identification boundaries. The two broad single-proxy references use assumptions different from cross-view balance, so they restrict firstness claims without subsuming the present method.
