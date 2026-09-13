# Code Notes

The experiment scripts in this folder were copied from `/tmp` on 2026-07-06.

The two main scripts are:

- `single_proxy_balancing_experiment.py`: scalar proxy experiment.
- `single_proxy_highdim_image_experiment.py`: high-dimensional image-like proxy experiment.

These scripts currently preserve their original behavior and write generated outputs to `/tmp`. The copied output artifacts are already stored in this project under `results/` and `figures/`.

For a quick syntax check:

```bash
python3 -m py_compile research/papers/single_proxy_balancing/code/*.py
```

For the surviving abstract audit-channel proof check:

```bash
lean research/papers/single_proxy_balancing/proofs/single_proxy_balance_core.lean
```
