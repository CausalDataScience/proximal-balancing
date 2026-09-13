# Final Experiments

This directory is the canonical allowlist for manuscript experiments. It does not duplicate experiment sources or results. `manifest.json` fixes their paths, roles, and SHA-256 hashes; `verify_manifest.py` checks them; `make_manuscript_figures.py` renders only those registered results; and `build_reproducibility_package.py` assembles the independent package.

Included experiment families are:

1. Honest Approximate SCM1 and SCM2;
2. Exact-Assumption Track A and Track B;
3. finite-state certificate verification; and
4. the descriptive WSC observational benchmark.

KMU, Zika, exploratory high-dimensional/image experiments, and superseded pilot outputs are excluded. WSC uses the final ten-algorithmic-repeat result. Its randomized contrast is a noisy reference rather than causal ground truth.

Validate the registered inputs with:

```bash
python3 code/final_experiments/verify_manifest.py
python3 code/final_experiments/make_manuscript_figures.py --check-only
```
