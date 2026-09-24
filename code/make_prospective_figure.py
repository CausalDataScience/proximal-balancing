"""Main-text figure A: the three benchmarks with an exactly known effect, on the replicates that were
designated before the v3 rule was frozen (memo/2026-09-22-v3-confirmation-prd-v1.md, section 6).

Only the designated identifiers are drawn: Twins 11-20, the planted RHC design 11-20, IHDP 12-21.  Each
panel carries the return count out of ten and the identifiers of any replicate that returned nothing.  The
rows and the colours follow the simulation figure so the reader compares like with like.

The x axis is the estimate minus the exact effect of that replicate.  IHDP's effect differs from one
replicate to the next (3.12 to 12.65 across these ten), so a shared "no error" line at zero is the only
honest way to put the three panels side by side.

Source: results/<dataset>/summary_v3_fresh*.json, written by v3_fresh_scorer.py against the frozen
protocol.  Nothing is recomputed here.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import make_section5_figures as m
import make_section5_v2_figures as v2

LIGHT = "#b9b8b2"
NO_AUDIT, NEEDS_ROLES, LATENT, PROBE = v2.NO_AUDIT, v2.NEEDS_ROLES, v2.LATENT, m.BLUE
# The middle panel is a design we built on the RHC covariates, not the RHC study as observed: the
# treatment is generated from a covariate we hid, and the effect is planted, so both the truth and the
# oracle exist here.  The observed RHC study, which has neither, is a separate figure.
PANELS = [("twins", "Twins\nreal outcomes"),
          ("acic", "ACIC 2016"),
          ("ihdp", "IHDP")]
XLIM = {"twins": (-0.125, 0.045), "acic": (-1.5, 1.5), "ihdp": (-2.6, 3.4)}
TICKS = {"twins": [-0.1, -0.05, 0], "acic": [-1, 0, 1], "ihdp": [-2, 0, 2]}
# ACIC is the set on which the v3 threshold rule was chosen, so its panel is exploratory and says so.
EXPLORATORY = {"acic"}
ROWS = [("Adjust for $X$", "X", NO_AUDIT),
        ("Adjust for $X$\nand all of $W$", "raw", NO_AUDIT),
        ("Proximal 2SLS,\nroles given", "p2sls_given_roles", NEEDS_ROLES),
        ("Proximal 2SLS,\nroles swapped", "p2sls_swapped_roles", NEEDS_ROLES),
        ("Park COCA as ATE,\nlinear bridge", "coca", NEEDS_ROLES),
        ("KSPC$^\\dagger$,\nclean proxy", "kspc_W1", NEEDS_ROLES),
        ("KSPC$^\\dagger$,\ncontaminated proxy", "kspc_W4", NEEDS_ROLES),
        ("CEVAE", "cevae", LATENT),
        ("PROBE (ours)", "probe", PROBE),
        ("Oracle\n(uses hidden $U$)", "oracle", LIGHT)]


def summary(name: str) -> dict:
    """The prospective score where one exists; for ACIC, the v3 reading of the exploratory realizations."""
    if name == "acic":
        v3 = json.loads((m.RESULTS / "noise_scaled_v3.json").read_text())["real_data"]["acic"]
        old = json.loads((m.RESULTS / "acic" / "summary_v1.json").read_text())
        by_rep = {r["replicate"]: r for r in v3["replicates"]}
        order = old["replicates"]
        rows = []
        for i, r in enumerate(order):
            rows.append({"replicate": r, "tau": by_rep[r]["tau"], "output": by_rep[r]["output"],
                         "comparators": {k: old["methods"][k]["values"][i]
                                         for k in ("naive", "X", "raw", "oracle", "cevae",
                                                   "p2sls_given_roles", "p2sls_swapped_roles")}})
        return {"status": "complete", "replicates": rows, "returned": v3["returned"],
                "return_denominator": v3["of"], "no_return_ids": []}
    corrected = m.RESULTS / name / "summary_v3_fresh_after_correction.json"
    path = corrected if corrected.exists() else m.RESULTS / name / "summary_v3_fresh.json"
    s = json.loads(path.read_text())
    if s["status"] != "complete":
        raise ValueError(f"{name}: the scored set is {s['status']}, so it is not drawn")
    return s


def coca(name: str) -> dict[int, float]:
    """Park's COCA turned into an ATE, keyed by replicate (coca_ate_all.py)."""
    d = json.loads((m.RESULTS / "coca_ate_all_v1.json").read_text())["datasets"][name]
    return {r["replicate"]: r["ate"] for r in d["replicates"]}


def errors(s: dict, key: str, name: str) -> np.ndarray:
    """Estimate minus the exact effect of that replicate; the PROBE row skips any non-return."""
    if key == "probe":
        return np.array([r["output"] - r["tau"] for r in s["replicates"] if r["output"] is not None])
    if key.startswith("kspc_"):
        folder = m.RESULTS / "kspc_comparator" / {"twins": "E5", "ihdp": "E6", "acic": "E7"}[name]
        block = key.split("_")[1]
        vals = []
        for r in s["replicates"]:
            path = folder / f"{name}_replicate_{r['replicate']}.json"
            if not path.exists():
                return np.array([])
            vals.append(json.loads(path.read_text())["kspc_skpv_with_x"][block]["ate"] - r["tau"])
        return np.array(vals)
    if key == "coca":
        table = coca(name)
        return np.array([table[r["replicate"]] - r["tau"] for r in s["replicates"]])
    return np.array([r["comparators"][key] - r["tau"] for r in s["replicates"]])


def main() -> int:
    W, H = 5.5, 3.75
    with plt.rc_context({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "savefig.bbox": "standard"}):
        fig, axes = plt.subplots(1, 3, figsize=(W, H), sharey=True)
        for ax, (name, title) in zip(axes, PANELS):
            s = summary(name)
            xlim = XLIM[name]
            span = xlim[1] - xlim[0]
            for i, (label, key, colour) in enumerate(ROWS):
                y = len(ROWS) - 1 - i
                ax.axhline(y, color=m.HAIR, lw=0.4, zorder=0)
                values = errors(s, key, name)
                if len(values) == 0:
                    continue
                if np.all(values < xlim[0]) or np.all(values > xlim[1]):
                    left = np.all(values < xlim[0])
                    xe = xlim[0] + 0.04 * span if left else xlim[1] - 0.04 * span
                    ax.scatter([xe], [y], marker="<" if left else ">", s=22, color=colour,
                               edgecolors="white", linewidths=0.4, zorder=3)
                    ax.text(xe + (-0.02 if left else 0.02) * span, y + 0.24, f"{values.mean():+.2f}",
                            ha="left" if left else "right", va="bottom", fontsize=6.2, color=m.INK2)
                else:
                    m.dots(ax, values, y, colour, size=9)
                    outside = int(np.sum((values < xlim[0]) | (values > xlim[1])))
                    if outside:
                        ax.text(xlim[1] - 0.02 * span, y + 0.28, f"{outside} off", ha="right", va="bottom",
                                fontsize=5.5, color=m.INK2)
            ax.axvline(0.0, color=m.INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
            ax.set_xlim(*xlim)
            ax.set_xticks(TICKS[name])
            ax.set_ylim(-0.6, len(ROWS) - 0.4)
            note = f"{s['returned']}/{s['return_denominator']} returned"
            if name in EXPLORATORY:
                note += "\n(rule chosen here)"
            if s["no_return_ids"]:
                note += "\nno return: " + ", ".join(str(i) for i in s["no_return_ids"])
            ax.set_title(title, fontsize=7.2, linespacing=1.25)
            ax.text(0.5, -0.155, note, transform=ax.transAxes, ha="center", va="top", fontsize=6.2,
                    color=m.INK2, linespacing=1.15)
            ax.tick_params(axis="y", length=0)
            ax.grid(False)
        axes[0].set_yticks(range(len(ROWS)))
        axes[0].set_yticklabels([label for label, _, _ in reversed(ROWS)], fontsize=6.6, linespacing=1.05)
        for tick, (_, _, colour) in zip(axes[0].get_yticklabels(), reversed(ROWS)):
            tick.set_color(m.INK if colour is LIGHT else colour)
        fig.text(0.5, 0.012, "estimate minus the exact effect  (dashed line: no error)\n"
                             "$^\\dagger$KSPC with covariates (App. A; our implementation), added after the protocol was frozen",
                 ha="center", va="bottom", fontsize=7.2, linespacing=1.3)
        v2.fit_labels(fig, axes, W, H)
        fig.subplots_adjust(bottom=0.26)
        m.save(fig, "realdata-prospective")

    for name, _ in PANELS:
        s = summary(name)
        line = "  ".join(f"{k} {np.mean(np.abs(errors(s, k, name))):.4f}"
                         for k in ("X", "raw", "coca", "cevae", "probe", "oracle"))
        print(f"{name:12s} {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
