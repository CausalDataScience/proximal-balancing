# Rerun Registration v3: Direct X-to-Y Effects (frozen before execution)

Date: 2026-08-19. Fixes the identified flaw of the first suites: the outcome
had no direct X path ($f=0$), which structurally favors proxy-supervised
learners because sacrificing X-balance costs nothing. This rerun adds the
direct effect and sweeps its strength. Everything below is fixed before
execution; sha256 recorded in the result files.

## Design

Setting confirmed with the user: $X$ is high-dimensional (real ACIC
covariates, synthetic 25-dim, or images); $W$ is a scalar proxy; $U$ is
unmeasured everywhere. Identical three arms (joint product kernel,
constrained soft-penalty with the $X$-only anchor, $X$-only), identical
learner/seed/schedule per cell, $H=h(X)$ only, estimation by held-out
$H$-stratified plug-in. The single change is the outcome:

Tabular: $Y=\tau A+2U+\lambda_f\{q(X)+0.5\,b_y^\top X\}+\varepsilon_Y$,
where $q=b_t^\top X$ is the treatment-driving direction (so for
$\lambda_f>0$, $q$ is a genuine observed confounder that every method must
balance) and $b_y$ is a third direction orthogonal to $b_c$ and $b_t$
(outcome-relevant, treatment-irrelevant). Image: the position factor $q$
enters $Y$ the same way with the $b_y$ term omitted.

Sweep: $\lambda_f\in\{0.0,0.5,1.0\}$. $\lambda_f=0$ replicates the original
template.

## Grid, seeds, parameters

- Tabular: sources acic, scm_dense, scm_mixed; variants v1_linear, v2_sine;
  $n=1200$; reps 6; steps 120; $M=5$; base seed 2026081905.
- Image: domains digits, lesions; variant v1_std; $n=1200$; reps 4;
  steps 100; $M=5$; base seed 2026081906.
- All other constants identical to the first suites (kernels, penalties,
  floors, anchor definition, 50/50 split).

## Frozen decision rules

Per track and per $\lambda_f$ level, for each arm versus $X$-only:
win rate of simultaneous $|$oracle bias$|$ and $|\tau$ error$|$ improvement
over paired cells, and one-sided Wilcoxon on $|$oracle bias$|$ reductions.

1. Replication check at $\lambda_f=0$: win rate $\ge0.70$ and $p<0.05$
   (both tracks, both arms), else the rerun itself is inconsistent.
2. Primary question at $\lambda_f\in\{0.5,1.0\}$: the advantage persists
   (win rate $\ge0.70$ and $p<0.05$) or it does not; both outcomes are
   recorded as the answer, and a shrinking-but-significant advantage counts
   as persists. This is a boundary-mapping rerun, so there is no suite-level
   PASS/FAIL beyond honest reporting of where the advantage holds.
3. Comparative prediction (pre-registered, secondary): the constrained arm's
   degradation from $\lambda_f=0$ to $\lambda_f=1$ in mean $|$bias$|$ gap
   over $X$-only is no worse than the joint arm's, because the constraint
   protects the $X$-balance that now carries outcome weight.
4. Instrumentation: per-stratum $\delta_z$ vectors, held-out $D_W$, $D_X$,
   $D_{XW}$ per cell; ACIC sha256
   0d6387ad45d23e54b11cbf967248fb68ad5e578db80d193208b8b4c98b1ed0c1.

Outputs: results/benchmark_suite_tabular_v2_directx.json,
results/benchmark_suite_image_v2_directx.json (plus runs JSON per track).
