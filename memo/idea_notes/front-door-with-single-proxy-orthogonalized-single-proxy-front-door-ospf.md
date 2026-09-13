# Front-door with Single-Proxy (Orthogonalized Single-Proxy Front-Door (OSPF))

Pillar: Pillar 1: Causal Inference under Real-World Imperfections

Sources:
- projects/research/ideas/paper_idea/sources/251104-Main-Idea.txt
- projects/research/ideas/paper_idea/sources/251104-Main-Idea-list.txt

External Links:
- https://chatgpt.com/c/68e55659-b7d4-8329-89fe-149baf8479d2

---

**Summary** 

Orthogonalized Single-Proxy Front-Door) extends the classical front-door identification strategy to cases where the mediator–outcome path is confounded by an unobserved variable $U$, but only a single noisy proxy $W$ for $U$ is available. Instead of requiring two shielded proxies (as in proximal causal inference), OSPF residualizes $W$ to remove contamination from treatment and mediator effects, then constructs a sufficient control $R=r(\tilde W,X,C)$ that captures the information of $U$ conditional on observed covariates $C$. Under completeness and structural assumptions, OSPF identifies

$$
\Psi(x)=\mathbb E[Y\mid \mathrm{do}(X{=}x)]
$$

through an orthogonalized front-door decomposition that combines:

1. an inner repair of the confounded $Z\to Y$ path using $R$, and
2. an outer mixing step integrating over the interventional law of $R$.

---

**Practical Motivation**

Classical front-door identification fails when a latent $U$ confounds both $Z$ and $Y$. Proximal mediation methods fix this with two proxies—one predictive of $U$ in $Z$ and another predictive of $U$ in $Y$—and assume those proxies are unaffected by treatment or mediator (shielding). In practice, however, modern proxies such as risk scores, embeddings, or learned features depend on $(X,Z)$:

$$
W = h(U,C) + q(X,Z,C) + \eta_W,
$$

where $h(U,C)$ carries the latent signal and $q(X,Z,C)$ captures leakage. OSPF residualizes out this contamination in mean and reuses the same proxy for both confounding control and effect mediation. This makes single-proxy front-door feasible in real systems like healthcare EHRs (disease severity proxy contaminated by treatment choices) or advertising (user engagement proxies contaminated by exposure).

---

**Input**

- Observed data $(Y_i, X_i, Z_i, W_i, C_i)_{i=1}^n$.
- Structural assumptions:
    - $Y(x)=Y(Z(x))$ (no direct $X\to Y$).
    - $Y(z)\perp Z\mid X,C,U$.
    - Proxy model $W=h(U,C)+q(X,Z,C)+\eta_W$, with $\mathbb E[\eta_W\mid U,X,Z,C]=0$.
    - Completeness: the channel $U\mapsto\tilde W=W-\mathbb E[W\mid X,Z,C]$ is rich enough that $\mathbb E[g(U)\mid \tilde W,X,C]=0$ implies $g(U)=0$.

---

**Output**

- Residualized proxy $\tilde W=W-\mathbb E[W\mid X,Z,C]$.
- Sufficient control $R=r(\tilde W,X,C)$, measurable with respect to $(U,X,C)$ via completeness.
- Identification formula:
    
    $$
    \begin{align}
    \Psi(x)=\int\!\!\Bigg[\underbrace{\int\!\mathbb E[Y\mid Z{=}z,X{=}x',C{=}c,R{=}r]\,f(x'\mid r,c)\,dx'}_{\text{inner repaired map } \mu(z,r,c)}\Bigg]f(z\mid X{=}x,R{=}r,C{=}c)\,f(r\mid \mathrm{do}(x),C{=}c)\,f(c)\,dz\,dr\,dc.
    \end{align}
    $$
    
    The last term $f(r\mid \mathrm{do}(x),C)$ is identified by inverse propensity weighting: $\mathbb E[g(R)\mid \mathrm{do}(X{=}x),C]=\mathbb E\!\left[g(R)\frac{\mathbf 1_{X=x}}{f(X{=}x\mid C)}\,\Big|\,C\right].$
    
- If $\mathcal{L}(\tilde{W} \mid U,C)$ is invariant to $(X,Z)$ (a stronger assumption), then $R=r(\tilde{W},C)$ and $f(r \mid \mathrm{do}(x),C)=f(r \mid C)$, giving a simpler expression.

---

**Contribution** 

- **Single-proxy identification:** Demonstrates that one residualized proxy can identify a front-door effect even when the proxy depends on treatment and mediator.
- **Orthogonalization step:** Subtracting $\mathbb E[W\mid X,Z,C]$ removes leakage in mean, leaving a proxy for $U$ robust to post-treatment contamination.
- **Control-function sufficiency:** Constructs $R=r(\tilde W,X,C)$ using completeness, guaranteeing $Y(z)\perp Z\mid X,C,R$.
- **Interventional weighting:** Uses a formally derived IPW representation to identify $f(r\mid \mathrm{do}(x),C)$ without new assumptions on $U$.
- **Fully nonparametric form:** All components—$\mathbb E[Y\mid Z,X,C,R]$, $f(Z\mid X,C,R)$, $f(X\mid C,R)$—are estimable from observational data via flexible ML with cross-fitting.

---

**Novelty**

- **Beyond classical front-door:** Traditional front-door assumes no $Z!\leftrightarrow!Y$ confounding. OSPF allows latent confounding on the mediator–outcome path.
- **Beyond two-proxy proximal methods:** Standard proximal mediation needs two shielded proxies unaffected by treatment. OSPF achieves identification with a *single, contaminated proxy* through residualization and completeness.
- **Beyond COCA or proximal causal inference for ATE:** COCA and related single-proxy approaches identify total effects only. OSPF uniquely extends the front-door decomposition to mediation structures.
- **Orthogonalization innovation:** The additive residualization $W-\mathbb E[W\mid X,Z,C]$ parallels orthogonal moments in semiparametric inference, ensuring robustness to small misspecifications of the contamination model.

---

**Impact**

- **Scientific:** OSPF establishes that front-door identification can survive real-world proxy contamination. This removes the strongest bottleneck in mediation analysis with learned or noisy features.
- **Methodological:** It introduces a general blueprint—residualize → construct control → plug into orthogonalized front-door formula—that extends to multi-proxy or high-dimensional cases.\
- **Practical:** The method aligns with how modern ML systems create proxies: embeddings or scores depending on $X$ or $Z$. OSPF allows causal estimation without retraining them for shielding.
- **Theoretical:** The proof clarifies where completeness and invariance enter, supplying formal IPW representation for the interventional law of the proxy control.

---

## Related

### Paper Ideas
- [[causal-catalyst-identification-via-an-internal-negative-control-stratum]]
- [[copula-invariant-identification-of-probabilities-of-causation-from-multi-environment-marginals]]
- [[cross-world-linear-programming-cw-lp-copula-sharp-bounds-for-probabilities-of-causation]]
- [[deconfounding-with-replicated-proxy-outcomes-identification-and-estimation-without-ignorability]]
- [[front-door-cate-estimation]]
- [[front-door-estimation-using-covariate-balancing]]
- [[optimal-causal-estimand]]
- [[proximal-inference-with-high-dimensional-proxies]]
- [[proximal-joint-treatment-effect-estimation]]
- [[proximal-transportability-a-framework-for-generalizing-causal-effects-under-unmeasured-confounding]]
- [[proxy-balancing-achieving-ignorability-via-a-single-negative-control-outcome]]
- [[weak-iv-with-one-proxy]]
- [[weakfd-weak-mediator-in-front-door-analysis]]

### Grant Ideas
- [[../grants/Causal Inference with Front-door Models]]
