# Rerun Registration v4: Dimension Sweep (frozen before execution)

Date: 2026-08-19. User directive: sweep the covariate dimension
$d\in\{5,10,20,30,40,50,100\}$ in the corrected direct-X template.
All parameters below fixed before execution; sha256 recorded in results.

## Design

Synthetic Gaussian covariates $X\sim N(0,I_d)$ (the controllable-dimension
source). DGP is the registration-v3 corrected template, variant v2_sine:

$U=r+0.5\sin r+0.5\varepsilon_U$ with $r=b_c^\top X$ standardized;
$W=U+0.5\varepsilon_W$;
$A\sim\mathrm{Bern}(0.05+0.90\,\mathrm{expit}(0.8U+1.5q))$, $q=b_t^\top X$;
$Y=\tau A+2U+\lambda_f\{q+0.5\,b_y^\top X\}+\varepsilon_Y$, $\tau=1$.
$b_c,b_t,b_y$ orthonormal, drawn per $d$ from the registered seed.

Grid: $d\in\{5,10,20,30,40,50,100\}\times\lambda_f\in\{0,0.5,1.0\}\times$
6 repetitions. $n=1200$, steps 120, $M=5$, base seed 2026081907. Arms and
protocol identical to registration v3 (joint, constrained soft-penalty with
the $X$-only anchor, $X$-only; identical learner per cell; held-out
$H$-stratified plug-in; per-stratum $\delta_z$, $D_W$, $D_X$, $D_{XW}$
logged).

## Frozen decision rules

1. Per $\lambda_f$, pooled over $d$ (42 pairs): win rate $\ge0.70$ and
   one-sided Wilcoxon $p<0.05$ on $|$oracle bias$|$ reductions constitutes
   an advantage claim at that $\lambda_f$; otherwise the advantage is
   recorded as absent at that level.
2. Primary deliverable: the dimension-response curves of mean $|$oracle
   bias$|$, RMSE, and held-out $D_W$ per arm per $\lambda_f$, reported per
   $d$ with no per-$d$ pass/fail (6 pairs per cell is a curve point, not a
   test).
3. Pre-registered prediction (secondary): at $\lambda_f=0$ the advantage
   over $X$-only does not shrink as $d$ grows, because the $X$-only
   objective spreads capacity over increasingly many outcome-irrelevant
   directions. The trend at $\lambda_f=1$ is the open question this sweep
   maps; both outcomes are recorded.

Output: results/benchmark_suite_dimsweep.json.
