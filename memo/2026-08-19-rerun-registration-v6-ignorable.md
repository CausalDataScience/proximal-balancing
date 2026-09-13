# Registration v6: Ignorable-X World Comparison (frozen before execution)

Date: 2026-08-19. User directive: assume $X$ is a sufficient covariate
satisfying ignorability, and compare Park's estimator, a standard
double-proxy proximal estimator, and our arms on known-covariate benchmarks
(ACIC). All parameters frozen before execution.

## World

$X$: real ACIC 2016 covariates (sha256 0d6387ad...) and scm_dense (25-dim
Gaussian) as a secondary source. $U = r(X)$-free noise... precisely:
$U = 0.7\,\varepsilon_U + 0.3\,r(X)$ with $r=b_c^\top X$ std (U may relate
to X but not to A beyond X); $A\sim\mathrm{Bern}(0.05+0.9\,
\mathrm{expit}(1.5q))$ with $q=b_t^\top X$ std, so $A\perp U\mid X$ HOLDS
(ignorability): treatment depends on X only. $W=U+0.5\varepsilon$;
$Z=U+0.5\varepsilon'$ (generated only for the double-proxy baseline).
$Y=\tau A+f(X)+2U+\varepsilon_Y$ with $f(X)=q+0.5\,b_y^\top X$, $\tau=1$.
Observed X-confounding is real (q in both A and Y); latent confounding is
absent by construction.

## Estimators

1. naive: difference of means.
2. full_x: g-formula ridge regression of Y on (X, A) (base module helper).
3. Our three arms (joint, constrained, x_only), protocol as v3.
4. park_x: Park linear bridge WITH covariates: $b(W,X)=\eta_0+\eta_1W+
   \eta_X^\top X$, control-arm moments against $r=(1,Y,X)$, exactly
   identified linear solve; ETT $=\overline Y_1-\overline{b(W,X)}_1$.
5. p2sls: parametric double-proxy proximal outcome bridge
   $h(W,X,A)=c_0+c_1W+c_X^\top X+\tau A$ solved by exactly identified
   linear GMM with instruments $(1,Z,X,A)$ (proximal two-stage least
   squares); ATE estimate is the $\tau$ coefficient.

## Grid and metrics

Sources acic, scm_dense; $n=2000$; B=20 reps; steps 120; $M=5$; seed base
2026081910. Metrics per estimator: mean and median $|$oracle bias$|$, RMSE
of the noisy estimate, and for our arms the held-out $\widehat D_W$
(certificate readout). Full run list saved.

## Frozen predictions

1. full_x, x_only, joint, constrained all near-unbiased; our arms within
   noise of x_only (no-harm property).
2. Our $\widehat D_W$ is small and certifies the safe world (the Simpson
   SAFE-world readout on real covariates).
3. p2sls suffers weak identification because $Z\perp A\mid X$ under
   ignorability: unstable estimates (heavy tails: median moderate, mean and
   RMSE inflated).
4. park_x approximately unbiased in this linear-bridge-feasible world, with
   higher variance than covariate methods.
All outcomes reported as found.

Output: results/ignorable_world_comparison.json.
