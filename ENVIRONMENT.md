# Environment

The final artifacts were validated on macOS with Python 3 and the versions pinned in `requirements.txt`.
The manuscript additionally requires a LaTeX installation providing `latexmk` and the packages loaded by `manuscript/main.tex`.
No dependency was added for packaging. PDF and PNG figures are generated with a noninteractive Matplotlib backend.
The WSC source files are exposed under `data/wsc/` and mirrored under their original `materials/real_world_data/` paths so that the frozen analysis scripts run without modification.
