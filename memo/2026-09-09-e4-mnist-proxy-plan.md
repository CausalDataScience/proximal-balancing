# E4 plan: MNIST image proxies (fixed 2026-09-09)

Status: approved by 용한 on 2026-09-09. This file fixes the design; changes require a new approval.

## Goal

Show that PROBE recovers the average treatment effect when the proxy blocks are real images of a finite latent
confounder, in the regime where one image is a weak proxy and several images together are a strong proxy.
The experiment substantiates the Section 1 claim that PROBE covers proxies that are images rather than
low-cardinality codes, and separates PROBE from the classify-then-Kuroki-Pearl route.

## Data-generating process (one unit)

1. Latent state $U \in \{0,\ldots,9\}$ uniform, unobserved. Score $s(U) = (U - 4.5)/2.87$.
2. Proxy blocks $W_1,\ldots,W_4$: four distinct MNIST training images with label $U$, drawn independently from
   the class pool, then degraded: pixel noise $N(0,\sigma^2)$ on the $[0,1]$ scale and a random square occlusion
   of side $o$ pixels. Blocks are conditionally independent given $U$ by construction.
3. Covariates $X = b_x s(U) + \epsilon_X \in \mathbb R^{20}$ with the unit-norm loading of the tabular SCMs.
4. Treatment $P(A=1 \mid U, X) = 0.05 + 0.9\,\mathrm{expit}\{1.2\,s(U) + 0.6\,a_x^\top X\}$.
5. Outcome $Y = A + 0.7\,y_x^\top X + 0.3\,(y_t^\top X)^2 - 3\,s(U) + \epsilon_Y$, so $\tau = 1$ and the naive
   and $X$-adjusted contrasts have the wrong sign (Simpson configuration shared with s10, s12, s3, s9, s11).

Images come from `materials/benchmarks/mnist/mnist.npz` (train split only for proxies; test split reserved
for the external classifier of the Kuroki-Pearl comparator). Provenance in `materials/benchmarks/manifest.json`.

## Methods compared

- naive: difference in means.
- X-only: honest AIPW on $X$.
- raw image adjustment: honest AIPW on $(X, \psi(W_1), \ldots, \psi(W_4))$ with $\psi$ = 50 principal
  components fitted on the discrepancy sample.
- PROBE: encoder = shared CNN over the remaining images, embeddings concatenated with $X$, MLP head to
  $Z \in \mathbb R^3$; critics = nested RFF ridge on $Z$ and on $(X, \psi(W_S), Z)$; screening, overlap gate,
  honest AIPW, linking and largest-component median exactly as in `code/probe_e1_pipeline.py`.
- oracle: honest AIPW on $(X, U)$.
- classify-then-Kuroki-Pearl: a digit classifier trained on the MNIST test split maps each image to a predicted
  label; Kuroki and Pearl (2014) matrix restoration with two conditionally independent discrete proxies.

All AIPW variants share the sample split, the GBM learners, and the clipping level $\eta = 0.05$.

## Design grid

- Degradation sweep (main figure): $\sigma \in \{0, 0.5, 1.0, 1.5, 2.0\}$ with occlusion $o = 10$ px; feasibility
  reference (PCA-50 + logistic, one image): accuracy 0.91 at $\sigma = 0$, 0.70 at $\sigma = 1$, 0.42 at $\sigma = 2$.
- Sample sizes: $n \in \{3000, 6000, 12000\}$ at the main setting $\sigma = 1$, $o = 10$.
- Replicates: 10 per cell (pilot 5). Budget $m = 6$ of the 10 held-out sets with at most two blocks.
- Metrics: absolute error of each method, PROBE screen pass rate, audit $\widehat D^2$ for $Z = X$ and for the
  learned $Z$, single-output rate of Algorithm 1.

## Compute stages

1. Pilot (now, CPU, 2 threads): encoder on frozen unsupervised features. The feature map $\psi$ is the
   50-component PCA of the degraded pixels fitted on the discrepancy sample; the PROBE encoder is the MLP on
   $(X, \psi(W_{-S}))$ and the raw-image baseline adjusts for $(X, \psi(W_1), \ldots, \psi(W_4))$. A supervised
   digit classifier is not used for features because its output is essentially $\hat U$ and would leak the
   latent label into every method. $n = 6000$, $\sigma = 1$, $o = 10$, $m = 6$, 5 replicates.
   Script: `code/probe_e4_mnist.py`; result: `results/probe_e4_mnist_pilot_sigma1.0_occ10.json`.
2. End-to-end: the CNN is trained jointly with the balance objective (minibatches, nested critics refit per
   batch). Run only if the pilot shows the expected ordering (X-only fails, raw adjustment partially recovers,
   PROBE near oracle).
3. Full grid as in the design grid.

## Theory notes to carry into the paper

- Finite latent state: completeness holds because the ten class-conditional image laws are linearly independent,
  the mechanism of the finite-state case; the paper positions this regime as outside Kuroki-Pearl because the
  proxies are images, not discrete codes.
- The critic sees the held-out image through the fixed map $\psi$, so the screened quantity is $D_\psi \le D$;
  the screen is anti-conservative with respect to the population discrepancy. State as a limitation.
- Real multi-image proxies (several scans of one patient) need not be independent given $U$; state as a limitation.

## Files

- Code: `code/probe_e4_mnist.py` (stage 1, written 2026-09-09; reuses `probe_e1_pipeline.py` components). The classify-then-Kuroki-Pearl comparator is implemented after the pilot (stage 3).
- Results: `results/probe_e4_mnist_*.json`.
- Log: `memory/logs/paios/2026-09-09-paios.md` entries.
