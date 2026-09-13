# Proximal Causal Inference Benchmark Catalog

Date: 2026-08-19. Zotero connector unavailable this session; compiled from
the local workspace and web verification (arXiv, GitHub). Zotero intake of
the cited papers is pending. Purpose: complete inventory of benchmarks used
in the proximal/negative-control literature, with availability and the
adaptation each needs to run under our interface (high-dim covariate $X$,
binary $A$, single pre-treatment scalar proxy $W$, unmeasured $U$).

## A. The standard ML proximal benchmark suite

Introduced by Deep Feature Proxy Variable (Xu, Kanagawa, Gretton 2021,
github.com/liyuan9988/DeepFeatureProxyVariable) and reused by KPV/PMMR
(Mastouri et al. 2021, github.com/yuchen-zhu/kernel_proxies), doubly robust
proximal (ICLR 2024), MMR-deep (arXiv:2205.09824), and the density-ratio-free
line (arXiv:2503.08371, 2505.19807). All double-proxy ($Z$ treatment-
inducing, $W$ outcome-inducing).

1. Demand design. Synthetic airline-ticket demand: $A$ = price
   (continuous), $U$ = demand level, $Z$ = cost shifters, $W$ = web views.
   Code public in both repos. Our fit: single-proxy ablation (drop $Z$,
   keep $W$), binarize $A$ or defer; adaptation must be labeled.
2. dSprite proxy task. Image-valued TREATMENT (4096-dim $A$ and $W$),
   $U$ = position latents. Dataset public (DeepMind; local copy sha256
   857dc001...). Our fit: not direct (image is $A$, not $X$); we instead
   layered our DGP on the images as $X$ (registration v5 T2).
3. Legalized abortion and crime. Real data in kernel_proxies repo:
   $A$ = effective abortion rate, $Y$ = murder rate, $Z$ = AFDC
   generosity, $W$ = beer consumption, log prisoners, gun-law indicators.
   Our fit: single-proxy ablation with one $W$, rest to $X$; continuous
   $A$ needs binarization; no ground-truth effect (agreement/certificate
   demo only).
4. Grade retention (Deaner). Real education data in kernel_proxies repo:
   $A$ = grade retention, $Z$ = elementary scores, $W$ = kindergarten
   cognitive scores. Our fit: good single-proxy ablation candidate
   (binary-ish treatment, natural $X$); no ground truth.

## B. Statistics-literature simulations (paper-specified DGPs)

5. Miao, Geng, Tchetgen Tchetgen 2018 double-NC Gaussian DGPs
   (arXiv:1808.04945): equations in paper, no packaged code. Our fit:
   reproducible from equations; single-proxy ablation.
6. Cui, Pu, Shi, Miao, Tchetgen Tchetgen 2023 semiparametric proximal
   simulations (JRSS-B): equations in paper/supplement. Same fit.
7. Kallus, Mao, Uehara 2021 (arXiv:2103.14029) d=60 Gaussian system.
   ALREADY IMPLEMENTED locally as
   `code/kmu_2022_single_proxy_ablation_experiment.py` with provenance of
   completions/deviations.
8. Ghassami et al. minimax kernel proximal simulations: equations in
   papers; also RHC application.

## C. Recurring real-data applications

9. Right heart catheterization (RHC / SUPPORT). THE canonical proximal
   real benchmark (Tchetgen Tchetgen et al. 2020 intro, Cui et al. 2023,
   Ghassami et al., censored variants e.g. MDPI Stats 2025). Public
   (Vanderbilt biostat mirrors; 5,735 patients, binary $A$ = RHC, ~70
   covariates; proximal papers split ten physiological covariates into
   $Z$/$W$ sets). Our fit: BEST real-data demo; take their $W$ set
   (or one), rest to $X$; report estimate plus certificate; no ground
   truth.
10. Brazil Zika birth-rate panel (Park single proxy control and single
    proxy synthetic control; github.com/qkrcks0218/SingleProxyControl).
    Public. Our fit: ATT demo with $W$ = 2014 birth rate.

## D. Single-proxy line

11. Park, Richardson, Tchetgen Tchetgen (arXiv:2302.06054, Biometrics
    2024): no simulation study; Zika application; estimators EPS, outcome
    bridge, doubly robust. Our cross-world 2x2 comparison implemented as
    `code/external_park_crossworld.py` (registration v5 T1).
12. Xu, Gretton kernel single proxy (arXiv:2308.04585,
    github.com/liyuan9988/KernelSingleProxy): synthetic deterministic-
    outcome DGPs + dSprites; no covariate $X$ in the main setting. Our
    fit: their estimator as a baseline in a deterministic-outcome cell;
    not a direct port.

## Status and recommended intake order

Implemented already: KMU 2021 ablation (B7), Park cross-world (D11),
dSprites-as-X (A2 adaptation). Recommended next: RHC single-proxy demo
(C9, highest citation value), then Demand and grade-retention ablations
(A1, A4) via the public kernel_proxies data, then Miao/Cui equation
reproductions (B5, B6). Every adaptation must carry the label
"single-proxy ablation of a double-proxy benchmark" or "our DGP on their
data", never "their benchmark as published".
