"""Section 5 figures from the sealed family v2 confirmation (SCM-1, SCM-2, SCM-3) and the sealed SCM-4
Shapes3D confirmation, one script for the four figure slots the manuscript defines.

    section5-v2-estimates     Theorem 1.  What each method returns, one dot per data set, four panels.
    section5-v2-aggregation   Theorem 2.  Three ways to combine the retained held-out choices.
    section5-v2-learned       Theorems 4 and 5.  Representation gap and total error of PROBE against the baselines.
    section5-v2-budget        Theorem 6.  Guarantee and observed success against the search budget (wrapfigure).

Sources, all sealed and read-only here:
    results/family_v2/confirm/SCM-{1,2,3}_seed*.json     n = 24,000, 20 data sets per level
    results/family_v2/scm4/confirm/SCM-4_seed*.json       n = 24,000, 20 data sets; the pixel CNN of the two
                                                           timed-out data sets comes from the amended summary
    results/family_v2/scm4/confirm_summary_v1_amended.json
    results/e11_v2_mcurve_v1.json                          the exactly solvable control for the budget curve

Layout follows make_section5_five_settings_figure.py: rows are methods, x is the estimate, a black bar is the
mean over data sets, and the dashed line is the true effect.  Comparator rows whose values fall outside the
axis are drawn as one triangle at the axis edge carrying the mean, so the axis keeps its resolution near tau.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import make_section5_figures as m

LIGHT = "#b9b8b2"
TAU = 1.0
WIDTH = 5.5   # the ICLR 2027 text width, 397.485 pt, so a figure at \linewidth is drawn at true size
LEVELS = ["SCM-1", "SCM-2", "SCM-3", "SCM-4"]
TITLES = {"SCM-1": "SCM-1\n8 numbers", "SCM-2": "SCM-2\n88 numbers",
          "SCM-3": "SCM-3\n1,312 numbers", "SCM-4": "SCM-4\n6 images"}
TARGETS = ("S1", "S2", "S3", "S12", "S13", "S23", "S123")


def shards(level: str) -> list[dict]:
    if level == "SCM-4":
        return m.shards("family_v2/scm4/confirm/SCM-4_seed*.json")
    return m.shards(f"family_v2/confirm/{level}_seed*.json")


def pixel_cnn_amended() -> dict[int, float]:
    """The pixel CNN value per seed, the two timed-out data sets filled from the amended summary."""
    summary = json.loads((m.RESULTS / "family_v2" / "scm4" / "confirm_summary_v1_amended.json").read_text())
    seeds = [r["seed"] for r in summary["replay"]]
    return dict(zip(seeds, summary["comparators"]["pixel_cnn"]["values"]))


def collect(level: str) -> dict[str, list[float] | None]:
    """Every row of the estimates figure for one level, in data-set order."""
    rows = shards(level)
    out = {
        "x": [s["baselines"]["X"] for s in rows],
        "raw": [s["baselines"]["raw"] for s in rows],
        "pixel": None,
        "p2sls_given": [s["proximal"]["p2sls_given_roles"] for s in rows],
        "p2sls_swapped": [s["proximal"]["p2sls_swapped_roles"] for s in rows],
        "park_clean": [s["proximal"]["park_spc_clean_proxy"] for s in rows],
        "park_contaminated": [s["proximal"]["park_spc_contaminated_proxy"] for s in rows],
        "cevae": [s["cevae"]["ate"] for s in rows],
        "probe": [s["probe"]["estimate"] for s in rows],
        "oracle": [s["baselines"]["oracle"] for s in rows],
    }
    if level == "SCM-4":
        filled = pixel_cnn_amended()
        out["pixel"] = [filled[s["task"]["seed"]] for s in rows]
    out.update(kspc_rows(level, rows))
    if any(v is None for v in out["probe"]):
        raise ValueError(f"{level}: a data set returned nothing; the figure does not silently drop it")
    return out


def kspc_rows(level: str, rows: list[dict]) -> dict:
    """KSPC with covariates (Appendix A, our implementation, SKPV only), added after the protocol was frozen
    (memo/2026-09-22-kspc-comparator-prd-v2.md), matched to the sealed data sets by replicate.  All or
    nothing: a partial set raises rather than drawing a partial row."""
    folder = m.RESULTS / "kspc_comparator" / ("E2" if level == "SCM-4" else "E1")
    paths = [folder / f"{level}_replicate_{s['task']['rep']}.json" for s in rows]
    present = [q.exists() for q in paths]
    if not any(present):
        return {"kspc_clean": None, "kspc_contaminated": None}
    if not all(present):
        raise FileNotFoundError(f"{level}: {present.count(False)} KSPC results missing; not drawing a partial row")
    shards = [json.loads(q.read_text())["kspc_skpv_with_x"] for q in paths]
    return {"kspc_clean": [k["W1"]["ate"] for k in shards], "kspc_contaminated": [k["W4"]["ate"] for k in shards]}


def retained_estimates(shard: dict) -> tuple[list[float], float]:
    """Estimates of the retained held-out choices, and PROBE's own output."""
    rec = shard["probe"]["records"]
    values = [rec[name]["estimate"] for name in shard["probe"]["retained"]]
    return values, shard["probe"]["estimate"]


# One hue per family of methods, not one per row: the row label already carries identity, so colour is a
# redundant channel that groups methods by what they need.  Every pair is at least 15.8 apart in OKLab under
# normal vision and 15.3 under a deuteranope simulation, and the five grey levels differ, so the figure also
# reads in print.
NO_AUDIT, NEEDS_ROLES, LATENT = m.GRAY, m.ORANGE, "#166534"

ROWS_A = [("Adjust for $X$", "x", NO_AUDIT),
          ("Adjust for $X$\nand all of $W$", "raw", NO_AUDIT),
          ("CNN on all pixels,\nno balance check", "pixel", NO_AUDIT),
          ("Proximal 2SLS,\nroles given", "p2sls_given", NEEDS_ROLES),
          ("Proximal 2SLS,\nroles swapped", "p2sls_swapped", NEEDS_ROLES),
          ("Park SPC,\nclean proxy", "park_clean", NEEDS_ROLES),
          ("Park SPC,\ncontaminated proxy", "park_contaminated", NEEDS_ROLES),
          ("KSPC$^\\dagger$,\nclean proxy", "kspc_clean", NEEDS_ROLES),
          ("KSPC$^\\dagger$,\ncontaminated proxy", "kspc_contaminated", NEEDS_ROLES),
          ("CEVAE", "cevae", LATENT),
          ("PROBE (ours)", "probe", m.BLUE),
          ("Oracle\n(uses hidden $U$)", "oracle", LIGHT)]
XLIM_A = (-0.72, 1.32)


def offscale(ax, values, y, colour):
    """One triangle at the left edge with the mean, for a row whose every value lies below the axis."""
    ax.scatter([XLIM_A[0] + 0.05], [y], marker="<", s=26, color=colour, edgecolors="white", linewidths=0.4, zorder=3)
    ax.text(XLIM_A[0] + 0.13, y, f"{np.mean(values):.1f}", ha="left", va="center", fontsize=6.5, color=m.INK2)


def fit_labels(fig, axes, W: float, height: float, right: float = 0.995, bottom_in: float = 0.5) -> None:
    """Left margin from the measured width of the y labels.  The axes are first pushed right so that no label is
    cut by the canvas while it is measured."""
    fig.subplots_adjust(left=0.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_w = max(t.get_window_extent(renderer).width for t in axes[0].get_yticklabels()) / fig.dpi
    fig.subplots_adjust(left=(label_w + 0.2) / W, right=right, bottom=bottom_in / height, top=1 - 0.42 / height,
                        wspace=0.14)


def bracket(ax, rows, keys: tuple[str, ...]) -> None:
    """A thin bracket to the right of the last panel over the rows named in `keys`, with a short label."""
    ys = [len(rows) - 1 - i for i, (_, key, _) in enumerate(rows) if key in keys]
    lo, hi = min(ys) - 0.35, max(ys) + 0.35
    ax.plot([1.04, 1.04], [lo, hi], transform=ax.get_yaxis_transform(), color=m.AXIS, lw=0.8, clip_on=False)
    ax.text(1.075, (lo + hi) / 2, "need the\nproxy roles", transform=ax.get_yaxis_transform(), rotation=90,
            ha="center", va="center", fontsize=6.5, color=m.INK2, linespacing=1.1, clip_on=False)


def strip_figure(rows, columns, xlim, xticks, xlabel, stem, height, not_applicable="images only",
                 role_bracket: tuple[str, ...] = (), reference: float = TAU):
    """Rows of dot strips, one panel per level, shared y labels on the left."""
    W = WIDTH
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "savefig.bbox": "standard"}):
        fig, axes = plt.subplots(1, len(columns), figsize=(W, height), sharey=True)
        for ax, (title, data) in zip(axes, columns):
            for i in range(len(rows)):
                ax.axhline(len(rows) - 1 - i, color=m.HAIR, lw=0.4, zorder=0)
            for i, (label, key, colour) in enumerate(rows):
                y = len(rows) - 1 - i
                values = data[key]
                if values is None:
                    ax.text(np.mean(xlim), y, not_applicable, ha="center", va="center", fontsize=6.5, color=LIGHT)
                    continue
                values = np.asarray(values, dtype=float)
                if np.all(values < xlim[0]):
                    offscale(ax, values, y, colour)
                else:
                    m.dots(ax, values, y, colour, size=10)
            ax.axvline(reference, color=m.INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
            ax.set_xlim(*xlim)
            ax.set_ylim(-0.6, len(rows) - 0.4)
            ax.set_xticks(xticks)
            ax.set_title(title, fontsize=8, linespacing=1.25)
            ax.tick_params(axis="y", length=0)
            ax.grid(False)
        axes[0].set_yticks(range(len(rows)))
        axes[0].set_yticklabels([label for label, _, _ in reversed(rows)], fontsize=7.5, linespacing=1.1)
        for tick, (_, _, colour) in zip(axes[0].get_yticklabels(), reversed(rows)):
            tick.set_color(m.INK if colour is LIGHT else colour)
        fig.text(0.5, 0.02, xlabel, ha="center", va="bottom", fontsize=8)
        fit_labels(fig, axes, W, height, right=0.955 if role_bracket else 0.995,
                   bottom_in=0.5 + 0.17 * xlabel.count("\n"))
        if role_bracket:
            bracket(axes[-1], rows, role_bracket)
        m.save(fig, stem)


def fig_estimates() -> None:
    columns = [(TITLES[level], collect(level)) for level in LEVELS]
    strip_figure(ROWS_A, columns, XLIM_A, [0, 1],
                 "Estimated treatment effect  (dashed line: true effect $\\tau=1$)\n"
                 "$^\\dagger$KSPC with covariates, our implementation of its App. A; added after freezing",
                 "section5-v2-estimates", height=4.15,
                 role_bracket=("p2sls_given", "p2sls_swapped", "park_clean", "park_contaminated", "kspc_clean",
                               "kspc_contaminated"))


ROWS_B = [("Average of the\nretained choices", "average", m.GRAY),
          ("Median of the\nretained choices", "median", m.GRAY),
          ("PROBE: median of the\nlargest agreeing group", "probe", m.BLUE)]


def fig_aggregation() -> None:
    columns = []
    for level in LEVELS:
        avg, med, probe = [], [], []
        for shard in shards(level):
            values, output = retained_estimates(shard)
            avg.append(float(np.mean(values)))
            med.append(float(np.median(values)))
            probe.append(output)
        columns.append((TITLES[level], {"average": avg, "median": med, "probe": probe}))
    strip_figure(ROWS_B, columns, (0.78, 1.14), [0.8, 0.9, 1.0, 1.1],
                 "Combined estimate over the retained held-out choices  (dashed line: $\\tau=1$)",
                 "section5-v2-aggregation", height=1.9)


ROWS_C = [("Adjust for $X$", "x", m.GRAY),
          ("Adjust for $X$\nand all of $W$", "raw", m.GRAY),
          ("CNN on all pixels,\nno balance check", "pixel", m.GRAY),
          ("One held-out choice:\nrepresentation gap $|\\tau_{\\widehat Z}-\\tau|$", "gap", m.BLUE),
          ("One held-out choice:\nestimation gap $|\\widehat\\theta-\\tau_{\\widehat Z}|$", "est", m.BLUE),
          ("One held-out choice:\nerror $|\\widehat\\theta-\\tau|$", "one", m.BLUE),
          ("PROBE output:\nerror $|\\widehat\\tau-\\tau|$", "total", m.BLUE)]


def per_choice(shard: dict) -> dict[str, float | None]:
    """Medians over the target held-out choices and rotations of the three per-representation quantities.

    tau_{Z_hat} is the population effect of the learned representation, exact for the linear representations of
    SCM-1 to SCM-3 and not recorded for SCM-4 (its spec, section 12); theta_hat is that rotation's AIPW estimate.
    """
    rots = [r for name in TARGETS for r in shard["probe"]["records"][name]["rotations"]]
    one = [abs(r["theta"] - TAU) for r in rots]
    with_pop = [r for r in rots if "learned_tau" in r]
    gap = [abs(r["learned_tau"] - TAU) for r in with_pop]
    est = [abs(r["theta"] - r["learned_tau"]) for r in with_pop]
    return {"gap": float(np.median(gap)) if gap else None, "est": float(np.median(est)) if est else None,
            "one": float(np.median(one))}


def fig_learned() -> None:
    """Absolute errors on a log axis.  The representation gap is the population value of the learned
    representation, which SCM-4 does not record, so that cell says so."""
    columns = []
    for level in LEVELS:
        data = collect(level)
        choice = [per_choice(s) for s in shards(level)]
        column = {"x": [abs(v - TAU) for v in data["x"]],
                  "raw": [abs(v - TAU) for v in data["raw"]],
                  "pixel": None if data["pixel"] is None else [abs(v - TAU) for v in data["pixel"]],
                  "total": [abs(v - TAU) for v in data["probe"]]}
        for key in ("gap", "est", "one"):
            values = [c[key] for c in choice]
            column[key] = None if any(v is None for v in values) else values
        columns.append((TITLES[level], column))
    W, height = WIDTH, 2.9
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "savefig.bbox": "standard"}):
        fig, axes = plt.subplots(1, len(columns), figsize=(W, height), sharey=True)
        for ax, (title, data) in zip(axes, columns):
            ax.set_xscale("log")
            for i in range(len(ROWS_C)):
                ax.axhline(len(ROWS_C) - 1 - i, color=m.HAIR, lw=0.4, zorder=0)
            for i, (label, key, colour) in enumerate(ROWS_C):
                y = len(ROWS_C) - 1 - i
                values = data[key]
                if values is None:
                    note = "images only" if key == "pixel" else "not recorded"
                    ax.text(10 ** np.mean(np.log10([0.002, 2.0])), y, note, ha="center", va="center",
                            fontsize=6.5, color=LIGHT)
                    continue
                values = np.asarray(values, dtype=float)
                jitter = np.random.default_rng(int(1000 * y) + len(values)).uniform(-0.17, 0.17, len(values))
                ax.scatter(values, y + jitter, s=10, color=colour, edgecolors="white", linewidths=0.4, zorder=3)
                med = np.median(values)
                ax.plot([med] * 2, [y - 0.32, y + 0.32], color=m.INK, lw=1.4, zorder=4, solid_capstyle="butt")
            ax.set_xlim(0.002, 2.0)
            ax.set_xticks([0.01, 0.1, 1])
            ax.set_xticklabels(["0.01", "0.1", "1"])
            ax.set_ylim(-0.6, len(ROWS_C) - 0.4)
            ax.set_title(title, fontsize=8, linespacing=1.25)
            ax.tick_params(axis="y", length=0)
            ax.grid(False)
        axes[0].set_yticks(range(len(ROWS_C)))
        axes[0].set_yticklabels([label for label, _, _ in reversed(ROWS_C)], fontsize=7.5, linespacing=1.1)
        fig.text(0.5, 0.02, "Absolute error, log scale  (bar: median over data sets)", ha="center", va="bottom",
                 fontsize=8)
        fit_labels(fig, axes, W, height)
        m.save(fig, "section5-v2-learned")


# ----------------------------------------------------------------------------- coverage of the interval

ROWS_D = [("Covers $\\tau_{\\widehat Z}$,\nwhat the representation targets", "z", m.BLUE),
          ("Covers $\\tau$,\nthe causal effect", "tau", m.BLUE)]
NOMINAL = 0.95


def coverage(level: str) -> dict[str, list[float] | None]:
    """Per data set, the share of the seven target held-out choices whose nominal 95% interval covers.

    The interval is the AIPW estimate plus or minus $1.96$ times its own standard error, and that standard
    error comes from the score cross-product the sealed run already stored, so nothing is refitted here.
    $\tau_{\widehat Z}$ is the population effect of the learned representation, averaged over the four
    rotations exactly as the estimate is; SCM-4 does not record it.
    """
    tau, z = [], []
    for shard in shards(level):
        probe = shard["probe"]
        names = probe["split_names"]
        se = np.sqrt(np.diag(np.array(probe["score_crossproduct"]))) / shard["n"]
        hit_tau, hit_z = [], []
        for j, name in enumerate(names):
            if name not in TARGETS:
                continue
            record = probe["records"][name]
            estimate, half = record["estimate"], 1.96 * se[j]
            hit_tau.append(abs(estimate - TAU) <= half)
            rotations = record["rotations"]
            if all("learned_tau" in r for r in rotations):
                target = float(np.mean([r["learned_tau"] for r in rotations]))
                hit_z.append(abs(estimate - target) <= half)
        tau.append(float(np.mean(hit_tau)))
        if hit_z:
            z.append(float(np.mean(hit_z)))
    return {"tau": tau, "z": z if len(z) == len(tau) else None}


def fig_coverage() -> None:
    columns = [(TITLES[level], coverage(level)) for level in LEVELS]
    strip_figure(ROWS_D, columns, (0.50, 1.07), [0.6, 0.8, 1.0],
                 "Share of the seven target held-out choices covered  (dashed line: nominal $0.95$)",
                 "section5-v2-coverage", height=1.65, not_applicable="not recorded", reference=NOMINAL)
    for level in LEVELS:
        c = coverage(level)
        z = "not recorded" if c["z"] is None else f"{np.mean(c['z']):.3f}"
        print(f"{level}: covers tau {np.mean(c['tau']):.3f}, covers tau_Zhat {z}, "
              f"{len(c['tau']) * len(TARGETS)} intervals")


def fig_budget() -> None:
    """The budget curve of the exactly solvable control: observed success with its 95% interval, and the bound."""
    d = json.loads((m.RESULTS / "e11_v2_mcurve_v1.json").read_text())
    rows = d["rows"]
    mm = np.array([r["m"] for r in rows])
    success = np.array([r["success"] for r in rows])
    lo = np.array([r["ci95"][0] for r in rows])
    hi = np.array([r["ci95"][1] for r in rows])
    bound = np.array([r["bound"] for r in rows])
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "savefig.bbox": "tight"}):
        fig, ax = plt.subplots(figsize=(2.55, 1.85))
        ax.fill_between(mm, lo, hi, color=m.BLUE, alpha=0.18, lw=0)
        ax.plot(mm, success, "-o", color=m.BLUE, lw=1.3, ms=3.2, markeredgecolor="white", markeredgewidth=0.4,
                zorder=3)
        ax.plot(mm, bound, color=m.INK2, lw=1.1, ls=(0, (3, 2)), zorder=2)
        ax.text(mm[-1], success[-1] + 0.01, "  observed", ha="left", va="bottom", fontsize=7, color=m.BLUE)
        ax.text(mm[-1], bound[-1] - 0.02, "  bound", ha="left", va="top", fontsize=7, color=m.INK2)
        ax.set_xlim(0.5, mm[-1] + 3.2)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 5, 10, 14])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("search budget $m$")
        ax.set_ylabel(r"$P(|\widehat\tau-\tau|\leq\rho)$")
        ax.grid(False)
        m.save(fig, "section5-v2-budget")


def report() -> None:
    """The numbers the captions and text cite, printed once so they are copied and not retyped."""
    for level in LEVELS:
        data = collect(level)
        line = {k: (None if v is None else round(float(np.mean(v)), 3)) for k, v in data.items()}
        errs = [abs(v - TAU) for v in data["probe"]]
        print(f"{level}: means {line} | PROBE MAE {np.mean(errs):.4f} p90 {np.quantile(errs, 0.9):.4f}")
    for level in LEVELS:
        avg, med, out = [], [], []
        for shard in shards(level):
            values, output = retained_estimates(shard)
            avg.append(np.mean(values)); med.append(np.median(values)); out.append(output)
        print(f"{level}: retained average {np.mean(avg):.3f}, median {np.mean(med):.3f}, PROBE {np.mean(out):.3f}, "
              f"retained per data set {[len(retained_estimates(s)[0]) for s in shards(level)]}")


def main() -> int:
    fig_estimates()
    fig_aggregation()
    fig_coverage()
    fig_learned()
    fig_budget()
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
