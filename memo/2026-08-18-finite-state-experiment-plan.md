# Finite-State Population Experiment Plan

Date: 2026-08-18
Status: approved plan (conditional GO, sequential execution step 1)
Theory anchor: `../THEORY.md` (T1, T2, T3, finite-$M$ attainability, rescaling invariance, finite-$\Phi$ deviation)
Code target: `../code/finite_state_bridge_experiment.py`
Results target: `../results/finite_state_bridge_summary.json`

## Purpose

Verify the theory at population level with every assumption holding by construction, separating theory failure from optimization failure. All representations are enumerated, so $\varepsilon_{\mathrm{opt}}=0$ and every oracle quantity is exact.

## DGP (bridge-first construction)

- $\mathcal X=\{0,\ldots,9\}$ ($S=10$), $U,W\in\{0,1\}$, $M=3$, $|\Phi_M|=3^{10}=59049$.
- $p(x)$ near-uniform; $P(U=1\mid X=x)$ varies across $x$ (range roughly $0.15$ to $0.85$).
- Proxy channel, treatment-free: $P(W=1\mid U=1)=0.8$, $P(W=1\mid U=0)=0.2$ (full-rank, injective).
- Propensity: $P(A=1\mid x,u)=\mathrm{clip}(\cdot,\eta,1-\eta)$ with $\eta=0.1$. Clipping propagates overlap to every $\phi$ and stratum by iterated expectation.
- Kernel: fixed Gaussian on the 20-point support of $(x,w)$ with $k(v,v)=1$ ($\kappa=1$); relative $X$/$W$ weighting fixed before analysis.
- Outcome via bridge witness: choose $\alpha_a$, set $g_a=K\alpha_a$ on the support, $\mu_a\triangleq Tg_a$, $Y(a)=\mu_a(X,U)+\varepsilon_Y$. Then A3 and the global bridge hold by construction, $\Gamma_{\mathrm{true}}=\max_a\|g_a\|_{\mathcal H_k}=\max_a\sqrt{\alpha_a^\top K\alpha_a}$ is known, and $\tau=\mathbb E[g_1(X,W)]-\mathbb E[g_0(X,W)]$ is closed-form.

## Regimes and acceptance criteria

1. Exact regime (R1): propensity depends on $x$ only and takes exactly 3 values. Predicted: the $D(\phi)=0$ set equals the propensity-level partition up to the $3!$ relabelings; realized bias is 0 there; attainability proposition verified in both directions. Interpretation guard: this regime is necessarily a world with $U\perp A\mid X$; the demonstration value is certification, not bias removal.
2. Approximate regime (R2): propensity depends on $(x,u)$. Predicted: $\inf_\phi D(\phi)>0$; zero violations of $|\tau_\phi-\tau|\le\Gamma_{\mathrm{true}}D(\phi)$ over all admissible $\phi$; the per-stratum bound $\sum_z p_z\{(1-\pi_z)\Gamma^\star_{1,\phi,z}+\pi_z\Gamma^\star_{0,\phi,z}\}\Delta_z$ is tighter and also never violated; T1 identity checks numerically; brute-force $\arg\min_\phi D$ realizes the T3 conclusion with $\varepsilon_{\mathrm{opt}}=0$.
3. Ill-posedness sweep (R4): channel $P(W=1\mid U=1)\in\{0.8,0.65,0.55,0.51,0.5\}$ with a fixed $u$-dependent $\mu_a$ chosen outside the bridge construction. Predicted: $\max_z\Gamma^\star$ diverges as the channel weakens while $\min_\phi D$ stays small; at $0.5$ the $\Gamma^\star=\infty$ flag fires and the certificate is honestly vacuous.
4. Finite-sample block: $n=20{,}000$, 30 repetitions; verify the explicit $r_n(\delta)$ event with the deviation computed exhaustively over all enumerated $\Phi$ for repetition 0, and over a 400-map random subset for every repetition. Subset values only lower-bound the supremum, so only the exhaustive repetition is a complete event check. Also report coverage of the combined interval $\varepsilon_{\mathrm{est}}+\Gamma\{\widehat D+r_n\}$ with a conservative variance term.

Admissibility: all three strata nonempty with $p_z\ge p_{\min}=0.05$; $\pi_z\in[\eta,1-\eta]$ is automatic from clipping.

$\Gamma^\star_{a,\phi,z}$ computation: restrict mass-zero contrasts to the stratum support, form $G=C^\top KC$ with $C$ the channel matrix, and evaluate the generalized Rayleigh value $\widetilde m^\top\widetilde G^{+}\widetilde m$ on the restricted subspace, with an $\infty$ flag when $\widetilde m$ has a component outside $\mathrm{range}(\widetilde G)$.

Main comparison discipline (carried to the later neural stage): $W$-supervised learner vs $X$-only learner at the same $M$ and class; full-$X$ adjustment reported as reference only; population metrics are actual bias and certificate; finite-sample metrics are bias, variance, MSE, interval coverage.

Dependencies: numpy only.
