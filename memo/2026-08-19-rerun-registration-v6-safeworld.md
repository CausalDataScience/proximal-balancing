# Registration v6: KMU Rescue + Ignorability World (frozen before execution)

Date: 2026-08-19. Two questions, one registration.

## T1. What succeeds in the KMU world (rescue cell)

Reveal the hidden treatment-side proxy $Z$ and run the linear proximal
two-stage estimator (P2SLS: stage 1 regress $W$ on $(Z,A,X)$; stage 2
regress $Y$ on $(\widehat W,A,X)$; ATE = coefficient of $A$), the standard
linear instantiation of double-proxy proximal identification. Run it on
(a) the untransformed system $(X_p,Z_p,W_p)$ where the linear bridge is
exact by construction, and (b) the observed transformed system $(X,Z,W)$
where linearity is misspecified. KMU transforms sin and cube, $n=1200$,
6 repetitions, seeds identical to the ablation.

Frozen predictions: (a) near-unbiased (this is what "succeeds" in the KMU
world); (b) biased but far below the single-proxy Park bridge's 280 to 642.

## T2. Ignorability world comparison

World SAFE-X: $X\in\mathbb R^{20}$ Gaussian, directions $b_c,b_t,b_y$ as
in registration v3. $U=r+0.5\sin r+0.5\varepsilon_U$;
$W=U+0.5\varepsilon_W$; $Z=U+0.5\varepsilon_Z$ (second proxy for the
double-proxy arm); propensity depends on $X$ only:
$A\sim\mathrm{Bern}(0.05+0.90\,\mathrm{expit}(0.8r+1.5q))$, so
$A\perp U\mid X$ holds and $X$ satisfies ignorability;
$Y=\tau A+2U+0.5(q+0.5b_y^\top X)+\varepsilon$, $\tau=1$.
$n=2000$, reps 10, steps 120, $M=5$, seed base 2026081910.

Estimators: our joint, constrained, x_only arms (protocol unchanged);
Park linear bridge (ETT form; ETT = ATE here by effect homogeneity);
linear P2SLS double proxy with $(Z,W,X)$; naive difference of means.
Metrics: oracle bias, noisy RMSE per estimator; our arms additionally log
held-out $\widehat D_W$ (frozen prediction: near zero, certifying safety).

Frozen predictions: (1) x_only and our arms near-unbiased with the lowest
RMSE band; (2) Park unbiased in expectation but with inflated variance
from the unnecessary reverse-bridge inversion; (3) P2SLS unbiased with
inflated variance from the unnecessary first stage; (4) naive biased
(X-confounding through $q$ and $r$); (5) our $\widehat D_W$ near zero.
All outcomes reported as found.

Output: results/rerun_v6_safeworld.json.
