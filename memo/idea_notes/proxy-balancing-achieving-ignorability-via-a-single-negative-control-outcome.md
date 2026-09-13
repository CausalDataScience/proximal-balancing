# Proxy Balancing: Achieving Ignorability via a Single Negative-Control Outcome

Pillar: Pillar 1: Causal Inference under Real-World Imperfections

Sources:
- projects/research/ideas/paper_idea/sources/251104-Main-Idea.txt
- projects/research/ideas/paper_idea/sources/251104-Main-Idea-list.txt

External Links:
- (none)

---

 [[GPT](https://chatgpt.com/c/68b277f6-dccc-832d-adf0-f0ceba78ae76), [Overleaf](https://www.overleaf.com/project/68b29b1ada26eddde731ed88), [GPTPro](https://chatgpt.com/c/68e5cb72-baf0-832c-b8ed-6ae1845b6b1b)]

**Summary**

Proxy Balancing provides a new route to causal identification when treatment $A$ and outcome $Y$ are confounded by an unobserved variable $U$, yet a *single* negative-control outcome (NCO) $W$ is available. The method learns a low-dimensional representation $h(X)$ of covariates $X$ such that:

1. $A\perp X\mid h(X)$ (Bal-X) and
2. $W\perp A\mid h(X)$ (Bal-W).If $W$ is sufficiently informative about $U$ given $X$ (a completeness or mixture-injectivity condition), then these two balances imply $A\perp U\mid h(X)$ and therefore $Y(a)\perp A\mid h(X)$.The causal effect is thus identified by $h$-adjustment: $\mathbb E[Y(a)] = \mathbb E\!\left[\mathbb E[Y\mid A=a,\,h(X)]\right].$

This approach replaces the two-proxy requirement of proximal causal inference with a learnable balancing representation that exploits one NCO.

---

**Motivation**

In many observational domains—biomedicine, public policy, economics—latent behavioral or risk factors $U$ drive both treatment and outcomes. For example, in healthcare, a "health-seeking" trait makes patients both more likely to start preventive medication $A$ and less likely to be hospitalized $Y$. Pre-treatment behaviors like annual flu shots or dental visits form natural proxies $W$: they reveal $U$ but cannot be caused by $A$.

**Traditional methods**:

- Propensity scores balance measured $X$ but not unmeasured $U$.
- Proximal causal inference needs both a negative-control outcome $W$ and exposure $Z$.

**Proxy Balancing** instead learns $h(X)$ to equalize the distribution of $W$ across treatment groups while keeping $A$ balanced in $X$. This forces the hidden $U$ to be balanced—turning pre-treatment proxies into confounding "meters."

---

**Input**

- Observed data $(Y_i, A_i, X_i, W_i)_{i=1}^n$, where $X\in\mathbb R^p$ (covariates), $A\in{0,1}$ (treatment), $Y\in\mathbb R$ (outcome), $W\in\mathbb R^d$ (NCO).
- Structural model: 
 $A=f_A(X,U,\varepsilon_A),\quad Y(a)=f_Y(a,X,U,\varepsilon_Y),\quad W=f_W(X,U,\varepsilon_W),\quad W(a)=W.$
- **Assumptions**
    1. (Positivity) $\Pr(A=a\mid h(X))>0$.
    2. (NCO) $W(a)=W$ and $W\perp A\mid (X,U)$.
    3. (Latent ignorability) $Y(a)\perp A\mid (X,U)$.
    4. (Completeness) Within $X=x$, the mapping from the mixing law of $U$ to the law of $W$ is injective (mixture-injectivity).
    5. (Representation balances) Learn $h(X)$ so that both Bal-X and Bal-W hold approximately.

---

**Output**

- A learned representation $h(X)=[e_\theta(X),r_\theta(X)]$, where $e_\theta(X)$ is a learned propensity component, and $r_\theta(X)$ is trained to make $W\perp A\mid h(X)$ via distributional alignment.
- An identified causal estimand: 
 $\tau=\mathbb E[Y(1)-Y(0)] =\mathbb E\!\left[\mathbb E[Y\mid A=1,h]-\mathbb E[Y\mid A=0,h]\right],$  estimated by outcome-regression or AIPW on $h$.
- Diagnostics: conditional two-sample tests for Bal-W, added-value tests for Bal-X, and overlap checks on $\hat e_h(h)$.

---

**Contribution**

1. **New sufficient path to ignorability**: Bal-X + Bal-W + completeness ⇒ $U\perp A\mid h$ ⇒ $Y(a)\perp A\mid h$.
2. **Joint ML representation**: formulates Bal-X and Bal-W as adversarial or divergence-based losses optimized jointly:
    
    $$
    \mathcal L(\theta)
     = \mathrm{CE}_{\text{adv}}(A\!\text{ on }[R,h])
     + \lambda_X\Delta\mathrm{CE}X
     + \lambda{\text{reg}}\|\theta\|^2.
    $$
    
3. **Approximate-ignorability theorem:** quantifies residual bias as $O(\varepsilon+\kappa)$ when the empirical Bal-W and Bal-X discrepancies are $\varepsilon,\kappa$.
4. **Empirical procedure:** balancing via negative-control alignment is operationalized with modern ML (adversarial or IPM training).
5. **Diagnostic transparency:** every assumption (NCO validity, completeness, positivity, approximate independence) has a testable empirical analogue.

---

**Novelty (Gap from Existing Works)**

- **Single proxy sufficiency**: unlike Proximal Causal Inference or two-proxy methods (Kuroki–Pearl; Miao–Geng–Tchetgen), Proxy Balancing requires only one NCO $W$.
- **Bridge-free identification**: existing single-proxy controls (COCA/SPC) solve bridge equations; here ignorability is achieved by distributional balancing, not equation solving.
- **Learned representation**: extends balancing-score learning beyond $X$ to include an auxiliary NCO, connecting representation learning and negative-control theory.
- **Quantitative theory**: provides explicit bias bounds tied to measurable divergences $\delta_W(h)$ and $\delta_X(h)$ controlled by ML losses.

---

**Impact and Significance**

- **Theoretical impact.** Shows that a single informative NCO can suffice for point identification when combined with distributional balancing, bridging the gap between classic propensity methods and proximal identification.
- **Methodological impact.** Offers an ML-ready loss function that enforces two causal balances jointly, yielding bias-diagnostic, interpretable representations.
- **Practical impact.** Addresses real observational settings (e.g., healthy-user bias in medicine, engagement bias in online experiments) where pre-treatment proxies exist but no valid negative-control exposure $Z$ can be found.
- **Empirical evidence.** Simulations with high-dimensional $X$ and learned $h$ demonstrate recovery of the correct causal effect under Simpson's paradox and robustness that improves with proxy informativeness $\eta$.

[Proxy_and_Balancing.pdf](attachment:7ecef946-25ad-4769-8bbf-3672dfdbea8f:Proxy_and_Balancing.pdf)

---

## Related

### Paper Ideas
- [[causal-catalyst-identification-via-an-internal-negative-control-stratum]]
- [[copula-invariant-identification-of-probabilities-of-causation-from-multi-environment-marginals]]
- [[cross-world-linear-programming-cw-lp-copula-sharp-bounds-for-probabilities-of-causation]]
- [[deconfounding-with-replicated-proxy-outcomes-identification-and-estimation-without-ignorability]]
- [[front-door-cate-estimation]]
- [[front-door-estimation-using-covariate-balancing]]
- [[front-door-with-single-proxy-orthogonalized-single-proxy-front-door-ospf]]
- [[optimal-causal-estimand]]
- [[proximal-inference-with-high-dimensional-proxies]]
- [[proximal-joint-treatment-effect-estimation]]
- [[proximal-transportability-a-framework-for-generalizing-causal-effects-under-unmeasured-confounding]]
- [[weak-iv-with-one-proxy]]
- [[weakfd-weak-mediator-in-front-door-analysis]]

### Grant Ideas
- [[../grants/Causal inference Under Hidden Confounders with Practical Imperfectness]]
