"""Section 5 figure: PROBE across three synthetic settings and the two learned renderings.

Five panels, one per setting, named Sim-1 to Sim-5.  Sim-1 to Sim-3 share the seven-number
proxy and vary the model; Sim-4 and Sim-5 keep the base model and enrich the proxy.  The
five synthetic models of the family agree to within 0.002 in mean estimate, so only the
base model and the two structural variants appear here.  Every row is defined in every panel
except the learned summary, which only the two learned renderings have.  That row builds a
scalar summary of every block, holds none of them out, and never tests balance: it isolates
"a representation was learned" from "the representation was checked for balance".

Sources, all sealed:
  synthetic   results/t1_fixed_block_scm_family_v1.json      n = 6,000,  20 data sets each
  SCM-6       results/scm6_v3/shards/larger_n_*_n24000.json  n = 24,000, 10 data sets
  SCM-7       results/scm7_v4/shards/confirmation_*.json     n = 12,000, 15 data sets
The no-balance summary is the comparator the sealed scoring named `best_no_balance`:
blockwise PCA channels for SCM-6 and CNN channels for SCM-7.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import make_section5_figures as m

LIGHT = "#b9b8b2"
TAU = 1.0


def shard_values(pattern: str, getter) -> list[float]:
    out = []
    for path in sorted((m.RESULTS / "scm6_v3" / "shards").glob(pattern)) if "larger_n" in pattern \
            else sorted((m.RESULTS / "scm7_v4" / "shards").glob(pattern)):
        payload = json.loads(path.read_text())
        value = getter(payload)
        if value is not None:
            out.append(float(value))
    return out


def synthetic(scm: str) -> dict[str, list[float]]:
    cells = [c for c in json.loads((m.RESULTS / "t1_fixed_block_scm_family_v1.json").read_text())["cells"]
             if c["scm"] == scm]
    return {"x": [c["baselines"]["X"] for c in cells],
            "raw": [c["baselines"]["raw_XW"] for c in cells],
            "summary": None,
            "probe": [c["probe_output"] for c in cells],
            "oracle": [c["baselines"]["oracle"] for c in cells]}


def learned(pattern: str, raw_key: str, summary_key: str) -> dict[str, list[float]]:
    return {"x": shard_values(pattern, lambda p: p["baselines"]["X"]),
            "raw": shard_values(pattern, lambda p: p["baselines"][raw_key]),
            "summary": shard_values(pattern, lambda p: p["baselines"][summary_key]),
            "probe": shard_values(pattern, lambda p: p.get("estimate")),
            "oracle": shard_values(pattern, lambda p: p["baselines"]["oracle"])}


ROWS = [("Adjust for $X$", "x", m.GRAY),
        ("Adjust for $X$\nand all of $W$", "raw", m.GRAY),
        ("Learned summary (all $W$),\nno balance check", "summary", m.GRAY),
        ("PROBE (ours)", "probe", m.BLUE),
        ("Oracle\n(uses hidden $U$)", "oracle", LIGHT)]

COLUMNS = [
    ("Sim-1\nbase", synthetic("SCM-1"), "$n=6{,}000$"),
    ("Sim-2\nvarying effect", synthetic("SCM-3"), "$n=6{,}000$"),
    ("Sim-3\nnonlinear proxy", synthetic("SCM-5"), "$n=6{,}000$"),
    ("Sim-4\nhigh-dim", learned("larger_n_*_n24000.json", "raw_ridge", "blockwise_pca"),
     "$n=24{,}000$"),
    ("Sim-5\nimages", learned("confirmation_*.json", "raw_cnn", "cnn_channels"),
     "$n=12{,}000$"),
]

W, H = 6.5, 2.85
with plt.rc_context({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "savefig.bbox": "standard"}):
    fig, axes = plt.subplots(1, 5, figsize=(W, H), sharey=True)
    for ax, (title, data, size_note) in zip(axes, COLUMNS):
        for i, (label, key, colour) in enumerate(ROWS):
            y = len(ROWS) - 1 - i
            values = data[key]
            if values is None:
                ax.text(0.5, y, "not applicable", ha="center", va="center", fontsize=7.5, color=LIGHT)
            else:
                m.dots(ax, np.asarray(values), y, colour, size=11)
        ax.axvline(TAU, color=m.INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.set_xlim(-0.12, 1.25)
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        ax.set_xticks([0, 1])
        ax.set_title(title + "\n" + size_note, fontsize=8, linespacing=1.25)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(ROWS)))
    axes[0].set_yticklabels([label for label, _, _ in reversed(ROWS)], fontsize=8, linespacing=1.1)
    fig.text(0.5, 0.035, "Estimated treatment effect  (dashed line: true effect)",
             ha="center", va="bottom", fontsize=8.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_w = max(t.get_window_extent(renderer).width for t in axes[0].get_yticklabels()) / fig.dpi
    fig.subplots_adjust(left=(label_w + 0.09) / W, right=0.995, bottom=0.19, top=0.815, wspace=0.14)
    m.save(fig, "section5-five-settings")
