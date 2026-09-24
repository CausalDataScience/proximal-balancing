"""Two figures that show where PROBE wins, where it does not, and what separates the two.

    nonwinning-real   the three data sets with a known effect on which PROBE is not the most accurate:
                      ACIC 2016, IHDP and LaLonde, the last with its two real comparison groups (CPS-1 and
                      PSID-1) against one randomised benchmark.  Same rows, colours and axes grammar as the
                      prospective figure, so the two can be read side by side.

    win-boundary      all eight data sets with a known effect in one panel.  Each row is PROBE's mean
                      absolute error divided by that of the most accurate method that needs no role
                      assignment (adjust for X, adjust for X and all of W, CEVAE, and the pixel CNN on
                      SCM-4), on a log scale, with a 90% paired bootstrap interval over replicates.  Rows are
                      ordered by the rows in one fold divided by the number of covariates.  A second mark
                      compares with proximal 2SLS given the true roles, which no real analysis has.

Nothing is recomputed from data.  Every number comes from the results that the other figures read:
    results/family_v2/confirm, results/family_v2/scm4/confirm           the four simulations
    results/<dataset>/summary_v3_fresh*.json, noise_scaled_v3.json      Twins, IHDP, ACIC
    results/coca_ate_all_v1.json                                        COCA as an average effect
    results/two_proxy/lalonde{,_psid}_v1.json, uncertainty*.json       LaLonde, CPS-1 and PSID-1

    python3 -B make_boundary_figures.py
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import make_prospective_figure as pf
import make_section5_figures as m
import make_section5_v2_figures as v2

NO_AUDIT, NEEDS_ROLES, LATENT, PROBE, LIGHT = v2.NO_AUDIT, v2.NEEDS_ROLES, v2.LATENT, m.BLUE, pf.LIGHT
TWO = m.RESULTS / "two_proxy"
BOOT, BOOT_SEED = 5_000, 99_100_000
FOLD_ROWS_PER_COVARIATE = {"SCM-1": 6000 / 2, "SCM-4": 6000 / 2, "SCM-2": 6000 / 8, "SCM-3": 6000 / 32,
                           "twins": 11984 / 4 / 42,
                           "acic": 4802 / 4 / 78, "ihdp": 747 / 4 / 24}
FOLD_LABEL = {"lalonde": "11–35"}
# LaLonde's ten real samples: the Dehejia-Wahba men against six comparison groups (CPS-2 and CPS-3 lie
# inside CPS-1, PSID-2 and PSID-3 inside PSID-1), and two groups of NSW AFDC women, all with 1974 earnings
# and the early random-assignment subset, each against PSID-1 and PSID-2 women.
# Nested, so not independent; each is scored against its own randomised benchmark.
LALONDE_SAMPLES = ["lalonde", "lalonde_cps2", "lalonde_cps3", "lalonde_psid", "lalonde_psid2", "lalonde_psid3",
                   "afdc_psid1", "afdc_psid2", "afdc_early_psid1", "afdc_early_psid2"]

ROWS = [("Adjust for $X$", "X", NO_AUDIT),
        ("Adjust for $X$\nand all of $W$", "raw", NO_AUDIT),
        ("Proximal 2SLS,\nroles given", "p2sls_given_roles", NEEDS_ROLES),
        ("Proximal 2SLS,\nroles swapped", "p2sls_swapped_roles", NEEDS_ROLES),
        ("Park COCA as ATE,\nlinear bridge", "coca", NEEDS_ROLES),
        ("CEVAE", "cevae", LATENT),
        ("PROBE (ours)", "probe", PROBE),
        ("Oracle\n(uses hidden $U$)", "oracle", LIGHT)]

PANELS = [("acic", "ACIC 2016", (-1.5, 1.5), [-1, 0, 1]),
          ("ihdp", "IHDP", (-2.6, 3.4), [-2, 0, 2]),
]
WSC_FILE = m.RESULTS / "wsc" / "probe_v1.json"
WSC_NOTES = {"p2sls_given_roles": "all 42 role assignments", "p2sls_swapped_roles": "no roles known",
             "cevae": "not run", "oracle": "no hidden $U$"}
SHORT = {"X": "adjust for $X$", "raw": "$X$ and all of $W$", "p2sls_given_roles": "2SLS, roles given",
         "p2sls_swapped_roles": "2SLS, roles swapped", "coca": "COCA", "cevae": "CEVAE", "probe": "PROBE"}
def wsc_rows() -> tuple[dict[str, np.ndarray], float]:
    """One real sample.  Band reading, v3 rule as declared primary.  2SLS: every ordered pair of blocks,
    since no roles are known; COCA: every block."""
    d = json.loads(WSC_FILE.read_text())
    r = d["readings"]["whole_sample"]
    t = json.loads((m.RESULTS / "wsc" / "exact_benchmark_v1.json").read_text())["estimate"]
    c = r["comparators"]
    num = lambda xs: np.array([x - t for x in xs if isinstance(x, (int, float))])
    out = {"X": num([c["X"]]), "raw": num([c["raw"]]),
           "p2sls_given_roles": num(c["p2sls_all_role_assignments"].values()),
           "p2sls_swapped_roles": np.array([]), "coca": num(c["coca_ate_by_block"].values()),
           "cevae": np.array([]), "probe": num([r["probe"]["readings"]["noise_floor"]["output"]]),
           "oracle": np.array([])}
    out["probe_candidates"] = num([rec["estimate"] for rec in r["probe"]["records"].values()])
    return out, float(r["aggregate_uncertainty"]["noise_floor"]["sd"])


def panel_errors(name: str) -> dict[str, np.ndarray]:
    if name == "wsc":
        return wsc_rows()[0]
    if name == "lalonde":
        return lalonde_rows()
    s = pf.summary(name)
    return {key: pf.errors(s, key, name) for _, key, _ in ROWS}


def rank_note(errs: dict[str, np.ndarray], name: str) -> str:
    """LaLonde has no known roles, so its two 2SLS rows are named by which proxy is put on the treatment side."""
    short = dict(SHORT)
    if name == "lalonde":
        short.update({"p2sls_given_roles": "2SLS, $W_1$ treats", "p2sls_swapped_roles": "2SLS, $W_2$ treats"})
    if name == "wsc":
        short.update({"p2sls_given_roles": "2SLS, any roles"})
    mae = {k: float(np.mean(np.abs(v))) for k, v in errs.items()
           if k not in ("oracle", "probe_candidates") and len(v)}
    order = sorted(mae, key=mae.get)
    place = order.index("probe") + 1
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(place, "th")
    return f"PROBE {place}{suffix} of {len(order)}\nlowest: {short[order[0]]}"


def lalonde_rows() -> dict[str, np.ndarray]:
    """Estimate minus that sample's randomised benchmark, one value per real sample.  No roles are known, so
    the two 2SLS rows are W1 treats and W2 treats; no CEVAE was run and no oracle exists."""
    out = {k: [] for _, k, _ in ROWS}
    for name in LALONDE_SAMPLES:
        d = json.loads((TWO / f"{name}_v1.json").read_text())
        r, truth = d["readings"]["common_support"], d["reference"]["estimate"]
        c = r["comparators"]
        values = {"X": c["X"], "raw": c["raw"], "p2sls_given_roles": c["p2sls_W1_treats_W2_outcome"],
                  "p2sls_swapped_roles": c["p2sls_W2_treats_W1_outcome"], "coca": c["coca_W1_ate"],
                  "probe": r["probe"]["readings"]["significance"]["output"]}
        for k, v in values.items():
            out[k].append(v - truth)
    return {k: np.array(v) for k, v in out.items()}


LALONDE_NOTES = {"p2sls_given_roles": "$W_1$ treats", "p2sls_swapped_roles": "$W_2$ treats",
                 "cevae": "not run", "oracle": "no hidden $U$"}


def draw_row(ax, values, y, colour, xlim, sd=None, note=None):
    span = xlim[1] - xlim[0]
    if len(values) == 0:
        if note:
            ax.text(xlim[0] + 0.04 * span, y, note, ha="left", va="center", fontsize=5.4, color=m.INK2,
                    style="italic")
        return
    outside_all = np.all((values < xlim[0]) | (values > xlim[1]))
    if outside_all and not (np.all(values < xlim[0]) or np.all(values > xlim[1])):
        ax.text(xlim[0] + 0.04 * span, y, f"all {len(values)} off scale", ha="left", va="center",
                fontsize=5.3, color=m.INK2, style="italic", zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
        return
    if outside_all:
        left = bool(np.all(values < xlim[0]))
        xe = xlim[0] + 0.04 * span if left else xlim[1] - 0.04 * span
        ax.scatter([xe], [y], marker="<" if left else ">", s=20, color=colour, edgecolors="white",
                   linewidths=0.4, zorder=3)
        mean = float(values.mean())
        label = f"{mean:+,.0f}" if abs(mean) >= 100 else f"{mean:+.1f}"
        ax.text(xe + (-0.02 if left else 0.02) * span, y + 0.24, label, ha="left" if left else "right",
                va="bottom", fontsize=5.4, color=m.INK2)
        return
    if len(values) == 1:
        if sd is not None:
            ax.plot([values[0] - sd, values[0] + sd], [y, y], color=colour, lw=1.1, zorder=3,
                    solid_capstyle="butt")
        ax.scatter(values, [y], s=22, color=colour, edgecolors="white", linewidths=0.5, zorder=4)
    else:
        m.dots(ax, values, y, colour, size=9)
        outside = int(np.sum((values < xlim[0]) | (values > xlim[1])))
        if outside:
            ax.text(xlim[1] - 0.02 * span, y + 0.28, f"{outside} off", ha="right", va="bottom", fontsize=5.4,
                    color=m.INK2)
    if note:
        ax.text(xlim[1] - 0.02 * span, y - 0.30, note, ha="right", va="top", fontsize=5.2, color=m.INK2)


def nonwinning() -> None:
    W, H = 5.5, 3.35
    with plt.rc_context({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.6,
                         "ytick.labelsize": 7, "savefig.bbox": "standard"}):
        fig, axes = plt.subplots(1, 2, figsize=(W, H), sharey=True)
        for ax, (name, title, xlim, ticks) in zip(axes, PANELS):
            errs = panel_errors(name)
            note = rank_note(errs, name)
            span = xlim[1] - xlim[0]
            for i, (_, key, colour) in enumerate(ROWS):
                y = len(ROWS) - 1 - i
                ax.axhline(y, color=m.HAIR, lw=0.4, zorder=0)
                row_note = (LALONDE_NOTES.get(key) if name == "lalonde" else
                            WSC_NOTES.get(key) if name == "wsc" else None)
                if len(errs[key]) == 0:
                    ax.text(xlim[0] + 0.04 * span, y, row_note, ha="left", va="center", fontsize=5.4,
                            color=m.INK2, style="italic")
                    continue
                if name == "wsc" and key == "probe":
                    cand = errs["probe_candidates"]
                    inside = cand[(cand > xlim[0]) & (cand < xlim[1])]
                    jit = np.random.default_rng(7).uniform(-0.2, 0.2, len(inside))
                    ax.scatter(inside, y + jit, s=5, color=colour, alpha=0.3, linewidths=0, zorder=2)
                    row_note = f"{len(cand)} candidates, faint"
                bar = wsc_rows()[1] if (name == "wsc" and key == "probe") else None
                draw_row(ax, errs[key], y, colour, xlim, sd=bar)
                if row_note:
                    ax.text(xlim[0] + 0.03 * span, y - 0.30, row_note, ha="left", va="top", fontsize=5.2,
                            color=m.INK2)
            ax.axvline(0.0, color=m.INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
            ax.set_xlim(*xlim)
            ax.set_xticks(ticks)
            if name == "lalonde":
                ax.set_xticklabels(["$-$3k", "0", "+3k"])
            ax.set_ylim(-0.6, len(ROWS) + 0.05)
            ax.set_title(title, fontsize=7.0, linespacing=1.2)
            ax.text(0.5, -0.13, note, transform=ax.transAxes, ha="center", va="top", fontsize=6.0,
                    color=m.INK2, linespacing=1.15)
            ax.tick_params(axis="y", length=0)
            ax.grid(False)
        axes[0].set_yticks(range(len(ROWS)))
        axes[0].set_yticklabels([label for label, _, _ in reversed(ROWS)], fontsize=6.4, linespacing=1.05)
        for tick, (_, _, colour) in zip(axes[0].get_yticklabels(), reversed(ROWS)):
            tick.set_color(m.INK if colour is LIGHT else colour)
        fig.text(0.5, 0.012, "estimate minus the known effect  (dashed line: no error)\n"
                             "",
                 ha="center", va="bottom", fontsize=6.8, linespacing=1.25)
        v2.fit_labels(fig, axes, W, H)
        fig.subplots_adjust(bottom=0.265, wspace=0.12)
        m.save(fig, "nonwinning-real")

# ----------------------------------------------------------------------------- figure 2

ROLE_FREE = {"x": "$X$ only", "raw": "$X$ + all $W$", "cevae": "CEVAE", "pixel": "pixel CNN"}
REAL_KEY = {"x": "X", "raw": "raw", "cevae": "cevae", "p2sls_given": "p2sls_given_roles", "probe": "probe"}


def per_replicate_errors() -> dict[str, dict[str, np.ndarray]]:
    """Absolute errors aligned by replicate, for every data set that has replicates."""
    out = {}
    for level in v2.LEVELS:
        c = v2.collect(level)
        out[level] = {k: np.abs(np.array(v) - v2.TAU) for k, v in c.items() if v is not None}
    for name in ("twins", "acic", "ihdp"):
        s = pf.summary(name)
        if s["no_return_ids"]:
            raise ValueError(f"{name}: a replicate returned nothing, so the pairing would be broken")
        out[name] = {k: np.abs(pf.errors(s, real, name)) for k, real in REAL_KEY.items()}
    return out


def ratio_with_interval(probe: np.ndarray, rival: np.ndarray, seed: int) -> tuple[float, float, float]:
    point = float(probe.mean() / rival.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(probe), size=(BOOT, len(probe)))
    draws = probe[idx].mean(1) / rival[idx].mean(1)
    return point, float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))


LABELS = {"SCM-1": "SCM-1, simulated", "SCM-4": "SCM-4, simulated images", "SCM-2": "SCM-2, simulated",
          "SCM-3": "SCM-3, simulated", "twins": "Twins, real outcomes", "lalonde": "LaLonde, 10 real samples", "wsc": "WSC, one real sample", "acic": "ACIC 2016", "ihdp": "IHDP"}


def boundary() -> list[dict]:
    errs = per_replicate_errors()
    order = sorted(FOLD_ROWS_PER_COVARIATE, key=lambda k: (-FOLD_ROWS_PER_COVARIATE[k], k))
    rows = []
    for i, name in enumerate(order):
        if name == "wsc":
            e = {k: np.abs(v) for k, v in panel_errors(name).items() if len(v) and k != "probe_candidates"}
            rivals = {"x": float(e["X"].mean()), "raw": float(e["raw"].mean())}
            best = min(rivals, key=rivals.get)
            rows.append({"name": name, "rival": best, "ratio": float(e["probe"].mean()) / rivals[best],
                         "lo": None, "hi": None, "given": None, "given_lo": None, "given_hi": None})
            continue
        e = errs[name]
        rivals = {k: float(e[k].mean()) for k in ROLE_FREE if k in e}
        best = min(rivals, key=rivals.get)
        r, lo, hi = ratio_with_interval(e["probe"], e[best], BOOT_SEED + 17 * i)
        g, glo, ghi = ratio_with_interval(e["probe"], e["p2sls_given"], BOOT_SEED + 17 * i + 1)
        rows.append({"name": name, "rival": best, "ratio": r, "lo": lo, "hi": hi,
                     "given": g, "given_lo": glo, "given_hi": ghi})
    return rows


def win_boundary() -> None:
    rows = boundary()
    W, H = 5.5, 2.75
    with plt.rc_context({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "savefig.bbox": "standard"}):
        fig, ax = plt.subplots(figsize=(W, H))
        n = len(rows)
        ax.axvspan(1e-3, 1.0, color="#eef4fb", zorder=0, lw=0)
        ax.axvline(1.0, color=m.INK, lw=0.9, zorder=2)
        for j, r in enumerate(rows):
            y = n - 1 - j
            ax.axhline(y, color=m.HAIR, lw=0.4, zorder=0)
            dy = 0.14 if r["given"] is not None else 0.0
            if r["lo"] is not None:
                ax.plot([r["lo"], r["hi"]], [y + dy] * 2, color=PROBE, lw=1.2, zorder=3, solid_capstyle="butt")
            ax.scatter([r["ratio"]], [y + dy], s=26, color=PROBE, edgecolors="white", linewidths=0.5, zorder=4)
            if r["given"] is not None:
                ax.plot([r["given_lo"], r["given_hi"]], [y - dy] * 2, color=NEEDS_ROLES, lw=0.9, zorder=3,
                        solid_capstyle="butt")
                ax.scatter([r["given"]], [y - dy], s=20, facecolors="white", edgecolors=NEEDS_ROLES,
                           linewidths=1.0, zorder=4)
            ax.text(1.02, y, FOLD_LABEL.get(r["name"], f"{FOLD_ROWS_PER_COVARIATE[r['name']]:,.0f}"),
                    transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=6.4, color=m.INK2)
            ax.text(1.20, y, ROLE_FREE[r["rival"]], transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=6.2, color=m.INK2)
        ax.text(1.02, n - 0.35, "rows ÷\ncovariates", transform=ax.get_yaxis_transform(), ha="left",
                va="bottom", fontsize=6.0, color=m.INK2, linespacing=1.1)
        ax.text(1.20, n - 0.35, "role-free\nrival", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=6.0, color=m.INK2, linespacing=1.1)
        ax.text(0.0, -0.215, "bars: 90% interval over repeated draws.",
                transform=ax.transAxes, linespacing=1.2,
                ha="left", va="top", fontsize=5.8, color=m.INK2)
        ax.set_xscale("log")
        ax.set_xlim(0.02, 6.0)
        ticks = [0.03, 0.1, 0.3, 1, 3]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks(range(n))
        ax.set_yticklabels([LABELS[r["name"]] for r in reversed(rows)], fontsize=6.8)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
        ax.set_xlabel("PROBE's mean absolute error ÷ the rival's  (left of 1: PROBE is more accurate)",
                      fontsize=7)
        marks = [Line2D([], [], marker="o", ls="none", markerfacecolor=PROBE, markeredgecolor="white",
                        markersize=5, label="vs the most accurate method that needs no roles"),
                 Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor=NEEDS_ROLES,
                        markersize=4.5, label="vs proximal 2SLS handed the true roles")]
        m.key(fig, marks, y=1.0, ncol=2)
        fig.subplots_adjust(left=0.235, right=0.72, top=0.86, bottom=0.25)
        m.save(fig, "win-boundary")
    print(f"{'data set':14s} {'rows/cov':>8s}  {'ratio':>6s}  {'90% interval':>16s}  {'rival':10s}  vs given 2SLS")
    for r in rows:
        iv = "" if r["lo"] is None else f"({r['lo']:.3f}, {r['hi']:.3f})"
        gv = "" if r["given"] is None else f"{r['given']:.3f} ({r['given_lo']:.3f}, {r['given_hi']:.3f})"
        print(f"{r['name']:14s} {FOLD_ROWS_PER_COVARIATE[r['name']]:8.0f}  {r['ratio']:6.3f}  {iv:>16s}  "
              f"{r['rival']:10s}  {gv}")


if __name__ == "__main__":
    nonwinning()
    win_boundary()
