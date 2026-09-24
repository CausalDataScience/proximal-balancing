"""Figures with the sealed and the added replicates together (memo/2026-09-23-replicate-extension-prd-v2.md,
section 6).  New files only, `figures/section5/extension-*.{pdf,png}`; the frozen figures are not touched.
Each figure keeps the design of the frozen one it extends and states its count of data sets.

    python3 -B make_extension_figures.py
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import make_kspc_figure as kf
import make_prospective_figure as pf
import make_rhc_bootstrap_figure as rbf
import make_section5_figures as m
import make_section5_v2_figures as v2
import summarise_extension_v1 as se

NOTE = "sealed and added data sets together; the added ones ran after freezing, on DeltaAI"


def pooled_values(pairs: dict, key: str, minus_truth: bool = False) -> list[float]:
    out = []
    for part in ("sealed", "extension"):
        for v, t in pairs[part].get(key, []):
            if v is not None and np.isfinite(v):
                out.append(v - t if minus_truth else v)
    return out


# ----------------------------------------------------------------------------------------------- E1, E2
def ext_row(ax, values, y, colour, xlim) -> None:
    """One row of the estimates figure.  With a hundred data sets some rows are no longer all on one side of the
    axis, which the frozen rule (a triangle only when every value is below) assumed: a row mostly below the axis
    gets the triangle labelled with its median, and values beyond either edge are counted ("k off")."""
    values = np.asarray(values, dtype=float)
    lo, hi = xlim
    span = hi - lo
    below, above = values < lo, values > hi
    inside = values[~below & ~above]
    if below.sum() >= len(values) / 2:
        ax.scatter([lo + 0.05], [y], marker="<", s=26, color=colour, edgecolors="white", linewidths=0.4, zorder=3)
        ax.text(lo + 0.13, y, f"{np.median(values):.1f}", ha="left", va="center", fontsize=6.5, color=m.INK2)
        if len(inside):
            jitter = np.random.default_rng(int(1000 * y) + len(inside)).uniform(-0.17, 0.17, len(inside))
            ax.scatter(inside, y + jitter, s=10, color=colour, edgecolors="white", linewidths=0.4, zorder=3)
        if above.sum():
            ax.text(hi - 0.02 * span, y + 0.28, f"{int(above.sum())} off", ha="right", va="bottom", fontsize=5.5,
                    color=m.INK2)
        return
    m.dots(ax, values, y, colour, size=10)
    for count, x, ha in ((int(below.sum()), lo + 0.02 * span, "left"), (int(above.sum()), hi - 0.02 * span, "right")):
        if count:
            ax.text(x, y + 0.28, f"{count} off", ha=ha, va="bottom", fontsize=5.5, color=m.INK2)


def strip_ext(rows, columns, xlim, xticks, xlabel, stem, height, role_bracket=()) -> None:
    """v2.strip_figure with ext_row in place of its row rule; everything else is the frozen design."""
    W = v2.WIDTH
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "savefig.bbox": "standard"}):
        fig, axes = plt.subplots(1, len(columns), figsize=(W, height), sharey=True)
        for ax, (title, data) in zip(axes, columns):
            for i in range(len(rows)):
                ax.axhline(len(rows) - 1 - i, color=m.HAIR, lw=0.4, zorder=0)
            for i, (label, key, colour) in enumerate(rows):
                y = len(rows) - 1 - i
                if data[key] is None:
                    ax.text(np.mean(xlim), y, "images only", ha="center", va="center", fontsize=6.5, color=v2.LIGHT)
                    continue
                ext_row(ax, data[key], y, colour, xlim)
            ax.axvline(v2.TAU, color=m.INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
            ax.set_xlim(*xlim)
            ax.set_ylim(-0.6, len(rows) - 0.4)
            ax.set_xticks(xticks)
            ax.set_title(title, fontsize=8, linespacing=1.25)
            ax.tick_params(axis="y", length=0)
            ax.grid(False)
        axes[0].set_yticks(range(len(rows)))
        axes[0].set_yticklabels([label for label, _, _ in reversed(rows)], fontsize=7.5, linespacing=1.1)
        for tick, (_, _, colour) in zip(axes[0].get_yticklabels(), reversed(rows)):
            tick.set_color(m.INK if colour is v2.LIGHT else colour)
        fig.text(0.5, 0.02, xlabel, ha="center", va="bottom", fontsize=8)
        v2.fit_labels(fig, axes, W, height, right=0.955 if role_bracket else 0.995,
                      bottom_in=0.5 + 0.17 * xlabel.count("\n"))
        if role_bracket:
            v2.bracket(axes[-1], rows, role_bracket)
        m.save(fig, stem)


def fig_estimates() -> None:
    columns, counts = [], []
    for level in v2.LEVELS:
        pairs = {"sealed": se.scm_pairs(level, "sealed"), "extension": se.scm_pairs(level, "extension")}
        data = {key: (pooled_values(pairs, key) if (key != "pixel" or level == "SCM-4") else None)
                for _, key, _ in v2.ROWS_A}
        counts.append(f"{level} {len(pairs['sealed']['probe']) + len(pairs['extension']['probe'])}")
        columns.append((v2.TITLES[level], data))
    sizes = {c.split()[1] for c in counts}
    count_line = (f"{sizes.pop()} data sets per level, sealed and added after freezing (DeltaAI)" if len(sizes) == 1
                  else "Data sets: " + ", ".join(counts) + "; sealed and added after freezing (DeltaAI)")
    strip_ext(v2.ROWS_A, columns, v2.XLIM_A, [0, 1],
                    "Estimated treatment effect  (dashed line: true effect $\\tau=1$)\n"
                    "$\\blacktriangleleft$: median of a row mostly off the axis; \"k off\": values beyond the axis\n"
                    "$^\\dagger$KSPC with covariates, our implementation of its App. A; added after freezing\n"
                    + count_line,
                    "extension-estimates", height=4.75,
                    role_bracket=("p2sls_given", "p2sls_swapped", "park_clean", "park_contaminated", "kspc_clean",
                                  "kspc_contaminated"))


# ----------------------------------------------------------------------------------------------- E5, E6 (+ ACIC)
KEYS = {"X": "X", "raw": "raw", "p2sls_given_roles": "p2sls_given_roles",
        "p2sls_swapped_roles": "p2sls_swapped_roles", "coca": "coca_ate", "kspc_W1": "kspc_clean",
        "kspc_W4": "kspc_contaminated", "cevae": "cevae", "probe": "probe", "oracle": "oracle"}


def fig_realdata() -> None:
    W, H = 5.5, 3.9
    pooled = {exp: {"sealed": se.real_pairs(exp, "sealed"), "extension": se.real_pairs(exp, "extension")}
              for exp in ("E5", "E6")}
    exp_of = {"twins": "E5", "ihdp": "E6"}
    with plt.rc_context({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "savefig.bbox": "standard"}):
        fig, axes = plt.subplots(1, 3, figsize=(W, H), sharey=True)
        for ax, (name, title) in zip(axes, pf.PANELS):
            xlim = pf.XLIM[name]
            span = xlim[1] - xlim[0]
            if name in exp_of:
                pairs = pooled[exp_of[name]]
                n_sets = len(pairs["sealed"]["probe"]) + len(pairs["extension"]["probe"])
                returned = sum(v is not None for part in ("sealed", "extension") for v, _ in pairs[part]["probe"])
                note = f"{returned}/{n_sets} returned"
                get = lambda key: np.array(pooled_values(pairs, KEYS[key], minus_truth=True))
            else:
                s = pf.summary(name)
                note = f"{s['returned']}/{s['return_denominator']} returned\n(rule chosen here)"
                get = lambda key, s=s, name=name: pf.errors(s, key, name)
            for i, (label, key, colour) in enumerate(pf.ROWS):
                y = len(pf.ROWS) - 1 - i
                ax.axhline(y, color=m.HAIR, lw=0.4, zorder=0)
                values = get(key)
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
                    m.dots(ax, values, y, colour, size=7)
                    outside = int(np.sum((values < xlim[0]) | (values > xlim[1])))
                    if outside:
                        ax.text(xlim[1] - 0.02 * span, y + 0.28, f"{outside} off", ha="right", va="bottom",
                                fontsize=5.5, color=m.INK2)
            ax.axvline(0.0, color=m.INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
            ax.set_xlim(*xlim)
            ax.set_xticks(pf.TICKS[name])
            ax.set_ylim(-0.6, len(pf.ROWS) - 0.4)
            ax.set_title(title, fontsize=7.2, linespacing=1.25)
            ax.text(0.5, -0.155, note, transform=ax.transAxes, ha="center", va="top", fontsize=6.2,
                    color=m.INK2, linespacing=1.15)
            ax.tick_params(axis="y", length=0)
            ax.grid(False)
        axes[0].set_yticks(range(len(pf.ROWS)))
        axes[0].set_yticklabels([label for label, _, _ in reversed(pf.ROWS)], fontsize=6.6, linespacing=1.05)
        for tick, (_, _, colour) in zip(axes[0].get_yticklabels(), reversed(pf.ROWS)):
            tick.set_color(m.INK if colour is pf.LIGHT else colour)
        fig.text(0.5, 0.012, "estimate minus the exact effect  (dashed line: no error)\n"
                             "$^\\dagger$KSPC with covariates (App. A; our implementation), added after the protocol was frozen\n"
                             "Twins and IHDP: " + NOTE,
                 ha="center", va="bottom", fontsize=6.8, linespacing=1.3)
        v2.fit_labels(fig, axes, W, H)
        fig.subplots_adjust(bottom=0.29)
        m.save(fig, "extension-realdata")


# ----------------------------------------------------------------------------------------------- E8
def fig_kspc() -> None:
    shards = [json.loads(p.read_text()) for p in sorted((m.RESULTS / "kspc_design").glob("replicate_*.json"))]
    shards += [json.loads(p.read_text()) for p in sorted((se.EXT / "E8").glob("replicate_*.json"))]
    W, H = 3.2, 3.4
    xlim = (-0.45, 0.8)
    with plt.rc_context({"font.size": 7.0, "xtick.labelsize": 6.6, "ytick.labelsize": 6.4, "savefig.bbox": "standard"}):
        fig, ax = plt.subplots(figsize=(W, H))
        for i, (label, get, colour) in enumerate(kf.ROWS):
            y = len(kf.ROWS) - 1 - i
            ax.axhline(y, color=m.HAIR, lw=0.4, zorder=0)
            vals = np.array([get(s) - s["tau"] for s in shards if get(s) is not None])
            m.dots(ax, vals, y, colour, size=5)
            out = int(np.sum((vals < xlim[0]) | (vals > xlim[1])))
            if out:
                ax.text(xlim[1] - 0.02, y + 0.28, f"{out} off", ha="right", va="bottom", fontsize=5.2, color=m.INK2)
            ax.text(xlim[1] + 0.03, y, f"{np.mean(np.abs(vals)):.3f}", ha="left", va="center", fontsize=5.8,
                    color=m.INK2, clip_on=False)
        ax.text(xlim[1] + 0.03, len(kf.ROWS) - 0.55, "MAE", ha="left", va="bottom", fontsize=5.8, color=m.INK2,
                clip_on=False)
        ax.axvline(0.0, color=m.INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.6, len(kf.ROWS) - 0.4)
        ax.set_yticks(range(len(kf.ROWS)))
        ax.set_yticklabels([r[0] for r in reversed(kf.ROWS)])
        for tick, (_, _, colour) in zip(ax.get_yticklabels(), reversed(kf.ROWS)):
            tick.set_color(m.INK if colour == "#b9b8b2" else colour)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("estimate minus the exact effect", fontsize=6.8)
        ax.set_title(f"KSPC benchmark, binary treatment\n{len(shards)} data sets", fontsize=7.0, pad=8, linespacing=1.2)
        ax.grid(False)
        fig.subplots_adjust(left=0.47, right=0.87, top=0.89, bottom=0.13)
        m.save(fig, "extension-kspc-benchmark")


# ----------------------------------------------------------------------------------------------- E3
def fig_rhc_bootstrap() -> None:
    import rhc_data as rd
    boots = [json.loads(p.read_text()) for p in sorted(rbf.BOOT.glob("boot_*.json"))]
    boots += [json.loads(p.read_text()) for p in sorted((se.EXT / "E3").glob("boot_*.json"))]
    obs = rbf.observed()
    cui = json.loads((m.RESULTS / "rhc_p2sls_cui" / "v1.json").read_text())
    obs["p2sls_cui_roles"] = cui["observed"]                  # amendment 1: Cui et al.'s specification only
    W, H = 3.1, 2.9
    xlim = (-3.2, 0.2)
    rows = [(("Proximal 2SLS,\nCui et al.'s specification" if key == "p2sls_cui_roles" else label), key, colour)
            for label, key, colour in rbf.ROWS]
    rows += [(label, key, m.INK2) for label, key in rbf.PUBLISHED if key != "standard_dr"]   # the author removed it
    with plt.rc_context({"font.size": 7.0, "xtick.labelsize": 6.6, "ytick.labelsize": 6.4, "savefig.bbox": "standard"}):
        fig, ax = plt.subplots(figsize=(W, H))
        for i, (label, key, colour) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.axhline(y, color=m.HAIR, lw=0.4, zorder=0)
            if key in rd.PUBLISHED:
                est, se_ = rd.PUBLISHED[key]
                ax.plot([est - 1.96 * se_, est + 1.96 * se_], [y, y], color=colour, lw=1.0, zorder=3)
                ax.scatter([est], [y], marker="D", s=16, color=colour, zorder=4)
                continue
            vals = np.array([(b["probe_v3"]["output"] if key == "probe" else
                              cui["resamples"][str(b["b"])] if key == "p2sls_cui_roles" else b["comparators"][key])
                             for b in boots], dtype=float)
            vals = vals[np.isfinite(vals)]
            m.dots(ax, vals, y, colour, size=5)
            ax.scatter([obs[key]], [y], s=26, facecolors="white", edgecolors=m.INK, linewidths=0.9, zorder=5)
            off = int(np.sum((vals < xlim[0]) | (vals > xlim[1])))
            if off:
                ax.text(xlim[1] - 0.05, y + 0.28, f"{off} off", ha="right", va="bottom", fontsize=5.5, color=m.INK2)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in reversed(rows)], linespacing=1.05)
        for tick, (_, _, colour) in zip(ax.get_yticklabels(), reversed(rows)):
            tick.set_color(colour)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("effect on days survived in 30", fontsize=6.8)
        ax.set_title(f"RHC, {len(boots)} bootstrap resamples", fontsize=7.2)
        fig.text(0.04, 0.015, "dots: bootstrap resamples; black tick: their mean;\n"
                              "open circle: the data as observed; diamond: published\n"
                              "by Cui et al., with its 95% bar.  RHC has no known answer.",
                 ha="left", va="bottom", fontsize=5.8, color=m.INK2, linespacing=1.25)
        ax.grid(False)
        fig.subplots_adjust(left=0.44, right=0.97, top=0.91, bottom=0.30)
        m.save(fig, "extension-rhc-bootstrap")


def main() -> int:
    fig_estimates()
    fig_realdata()
    fig_kspc()
    fig_rhc_bootstrap()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
