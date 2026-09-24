"""Summary of the Sim5 CNN development run: the three-method comparison and the per-split balance diagnostics.

Reads results/sim5_cnn/dev_v1.json, prints the tables, and draws a 1x3 diagnostic figure to
figures/sim5_cnn/dev_v1_diagnostics.{pdf,png}.  Truth is tau = 1.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_section5_figures as m

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "results" / "sim5_cnn" / "dev_v1.json").read_text())
cfg, meth, sp = d["config"], d["methods"], d["splits"]
t = cfg["t"]

print("method        estimate   |error|   note")
for k in ("raw_cnn", "probe_warm", "probe_final", "x_only"):
    v = meth[k]
    est = v["estimate"]
    note = ""
    if "R" in v:
        note = (f"R={v['R']}/{len(sp)}  {v['return_kind']}  rho={v['rho']:.3f}  comps={v['component_sizes'][:4]}  "
                f"screened mean={v['screened_mean']:.3f} median={v['screened_median']:.3f}  "
                f"all-split median={v['all_splits_median']:.3f}") if v["rho"] is not None else f"R={v['R']}"
    print(f"{k:12s} {'  none ' if est is None else f'{est:+.4f}'}   {'' if est is None else f'{abs(est - 1):.4f}'}   {note}")

w_err = np.array([s["warm"]["abs_error"] for s in sp]); f_err = np.array([s["final"]["abs_error"] for s in sp])
w_gap = np.array([s["warm"]["gap"] for s in sp]); f_gap = np.array([s["final"]["gap"] for s in sp])
sel = np.array([s["balance"]["selected_update"] for s in sp])
zero = np.array([s["balance"]["zero_gradient_updates"] for s in sp])
wchg = np.array([next(c["weight_change"] for c in s["balance"]["checkpoints"] if c["update"] == s["balance"]["selected_update"])
                 for s in sp])
print(f"\nper split (n={len(sp)}): |error| warm median {np.median(w_err):.4f}  final median {np.median(f_err):.4f}  "
      f"final better in {np.mean(f_err < w_err):.2f} of splits")
print(f"gap warm median {np.median(w_gap):+.5f}  final median {np.median(f_gap):+.5f}  "
      f"retained at t={t}: warm {np.sum(np.maximum(w_gap, 0) <= t)}, final {np.sum(np.maximum(f_gap, 0) <= t)}")
print(f"selected update counts {dict(zip(*np.unique(sel, return_counts=True)))}  "
      f"zero-gradient updates median {np.median(zero):.0f}/{cfg['balance_updates']}  "
      f"relative head change median {np.median(wchg):.3f}")
print(f"timing {d['timing_s']}  ledger {d['ledger']}")

W, H = 6.5, 2.1
with plt.rc_context({"font.size": 7.5}):
    fig, axs = plt.subplots(1, 3, figsize=(W, H), constrained_layout=True)
    ax = axs[0]
    lim = max(w_err.max(), f_err.max()) * 1.05
    ax.scatter(w_err, f_err, s=9, color=m.BLUE, alpha=0.7, linewidths=0)
    ax.plot([0, lim], [0, lim], color=m.GRAY, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("warm $|\\widehat\\theta_S-1|$"); ax.set_ylabel("final $|\\widehat\\theta_S-1|$")
    ax.set_title("(a) per split error", fontsize=7.5)
    ax = axs[1]
    ax.scatter(w_gap, f_gap, s=9, color=m.BLUE, alpha=0.7, linewidths=0)
    lo, hi = min(w_gap.min(), f_gap.min()), max(w_gap.max(), f_gap.max())
    ax.plot([lo, hi], [lo, hi], color=m.GRAY, lw=0.8, ls=(0, (3, 2)))
    ax.axhline(t, color=m.INK, lw=0.7, ls=(0, (1, 1))); ax.axvline(t, color=m.INK, lw=0.7, ls=(0, (1, 1)))
    ax.set_xlabel("warm gap"); ax.set_ylabel("final gap"); ax.set_title("(b) Brier gap on $D_{val}$", fontsize=7.5)
    ax = axs[2]
    names = [("x_only", "$X$ only"), ("raw_cnn", "raw CNN"), ("probe_warm", "PROBE warm"), ("probe_final", "PROBE final")]
    vals = [meth[k]["estimate"] for k, _ in names]
    ys = np.arange(len(names))[::-1]
    for y, (k, lab), v in zip(ys, names, vals):
        if v is not None:
            ax.scatter([v], [y], s=22, color=m.BLUE if k == "probe_final" else m.GRAY, zorder=3)
    ax.axvline(1.0, color=m.INK, lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(ys, [lab for _, lab in names]); ax.set_xlabel("estimate (dashed: $\\tau=1$)")
    ax.set_title("(c) methods", fontsize=7.5)
    for a_ in axs:
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    out = ROOT / "figures" / "sim5_cnn"; out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"dev_v1_diagnostics.{ext}", dpi=200)
print("wrote figures/sim5_cnn/dev_v1_diagnostics.{pdf,png}")
