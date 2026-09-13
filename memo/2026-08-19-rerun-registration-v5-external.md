# Registration v5: External Benchmarks (frozen before execution)

Date: 2026-08-19. Two user-approved external-benchmark experiments. All
parameters fixed before execution; sha256 recorded in results.

## T1. Park single-proxy-control comparison (2x2 cross design)

Park, Richardson, Tchetgen Tchetgen (arXiv:2302.06054, Biometrics 2024)
contains no simulation study, so the honest design instantiates BOTH
frameworks' assumptions and crosses worlds with estimators.

Worlds ($d=20$, $n=2000$, reps 10, seed base 2026081908, $\tau=1$):

- DGP-P (Park-conforming): $Y_0=r+0.5\sin r+0.5\varepsilon$ with
  $r=b_c^\top X$ std; $A\sim\mathrm{Bern}(0.05+0.9\,\mathrm{expit}(0.8\,
  \mathrm{std}(Y_0)+1.5q))$ (selection directly on the treatment-free
  outcome); $W=Y_0+0.5\varepsilon$; $Y=Y_0+\tau A$. Then Park's Assumption
  $W\perp A\mid Y_0$ holds exactly and the linear bridge $b(w)=w$ solves
  their Condition 1, while our A3 holds with latent $U\equiv Y_0$.
- DGP-O (our registration-v3 template, $\lambda_f=0.5$): as in v3 with
  Gaussian $X$; Park's assumption fails there because $A$ depends on $U$,
  not on $Y_0$, and $Y$ carries independent noise.

Estimators in every cell: Park bridge estimator (linear families
$m_w(w)=(1,w)$, $r_W(y)=(1,y)$, control-arm GMM exactly as in their
Result 3, ETT $=\bar Y_1-\overline{b(W)}_1$), and our three arms with the
treated-weighted stratified estimator (ETT weighting). Estimand ETT $=1$
in both worlds by construction.

Frozen predictions: (1) DGP-P: Park near-unbiased and better than all our
arms (they use $W$ in estimation, we do not); our arms still beat $X$-only.
(2) DGP-O: Park biased; our arms beat both Park and $X$-only. Deliverable
is the 2x2 table with |bias| and RMSE plus paired Wilcoxon of constrained
vs Park in each world; all outcomes reported as found.

Output: results/external_park_crossworld.json.

## T2. dSprites-as-X image track

Public dataset: DeepMind dsprites,
dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz, sha256
857dc001287e8efa226e0025fac6b29e5745c621b380b0d3363e4e33f4e28fa2,
737,280 images 64x64, latents (color, shape, scale, orientation, posX,
posY). Framing fixed: OUR DGP layered on THEIR public images; not their
benchmark task.

$X=$ image; severity driver $r=$ std(scale latent); treatment factor
$q=$ std(posX latent); $U=r+0.5\varepsilon_U$; $W=U+0.5\varepsilon_W$;
$A\sim\mathrm{Bern}(0.05+0.9\,\mathrm{expit}(0.8U+1.5q))$;
$Y=\tau A+2U+\lambda_f q+\varepsilon$, $\tau=1$,
$\lambda_f\in\{0,1\}$. $n=1200$ subsampled per rep, reps 4, steps 100,
$M=5$, seed base 2026081909. CNN arms exactly as the image suite (joint,
constrained soft-penalty with $X$-only anchor, $X$-only).

Frozen decision rule: per $\lambda_f$ (8 pairs), win rate $\ge0.70$ and
one-sided Wilcoxon $p<0.05$ for an advantage claim; constrained expected
more robust at $\lambda_f=1$ (secondary prediction); all outcomes reported.

Output: results/external_dsprites_image.json.
