# Single Proxy Balancing: Reproducibility Package

This package contains only the experiments accepted for the manuscript:

1. honest approximate SCM1 and SCM2;
2. exact-assumption Tracks A and B;
3. finite-state certificate verification; and
4. the descriptive WSC observational benchmark.

KMU, Zika, exploratory high-dimensional/image suites, and superseded pilots are intentionally excluded.

## Quick verification

```bash
python3 verify_package.py
python3 code/final_experiments/make_manuscript_figures.py --root . --check-only
```

To regenerate the figures:

```bash
python3 code/final_experiments/make_manuscript_figures.py --root .
```

To compile the manuscript:

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The WSC randomized contrast is a noisy reference, not causal ground truth. Its ten repetitions are algorithmic data splits on one fixed data set, not Monte Carlo data-generating repetitions.
