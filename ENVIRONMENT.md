# Environment

The final artifacts were validated on macOS with Python 3 and the versions pinned in `requirements.txt`.
The manuscript additionally requires a LaTeX installation providing `latexmk` and the packages loaded by `manuscript/main.tex`.
No dependency was added for packaging. PDF and PNG figures are generated with a noninteractive Matplotlib backend.
The WSC source files are exposed under `data/wsc/` and mirrored under their original `materials/real_world_data/` paths so that the frozen analysis scripts run without modification.

The active PROBE pipeline (`code/probe_e1_pipeline.py`, `code/probe_scms.py`, `code/probe_e4_mnist.py`) needs `torch`, `scipy`, and `scikit-learn` in addition; it runs on CPU (set `--threads`). MNIST for E4 is packaged under `materials/benchmarks/mnist/`.
