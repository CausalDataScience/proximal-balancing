"""The four figures for the minimal suite, drawn from stored results only.

Nothing is recomputed here beyond reading the result files and the SCM-7 renderer, so a figure cannot
drift away from the numbers it is supposed to show.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parent.parent / "figures"
RESULTS = Path(__file__).resolve().parent.parent / "results"
TRUE_EFFECT = 1.0
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 160, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def require(*names: str) -> bool:
    """Skip a figure whose inputs have not been produced yet rather than stopping the whole script."""
    missing = [n for n in names if not (RESULTS / n).exists()]
    if missing:
        print(f"  skipped, missing {missing}")
    return not missing


def save(fig, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{suffix}")
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf and .png")


def figure_e12() -> None:
    """Where the error sits as the sample grows, and how it splits into its two sources."""
    if not require("e12_scm1_ncurve_confirm_v2.json"):
        return
    data = json.loads((RESULTS / "e12_scm1_ncurve_confirm_v2.json").read_text())
    by_n = {}
    for cell in data["cells"]:
        if cell["estimate"] is not None:
            by_n.setdefault(cell["n_total"], []).append(cell)
    sizes = sorted(by_n)
    err = [np.array([abs(c["estimate"] - TRUE_EFFECT) for c in by_n[n]]) for n in sizes]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    left.boxplot(err, positions=range(len(sizes)), widths=0.5, showfliers=False,
                 medianprops={"color": "#1f4e79"})
    for i, e in enumerate(err):
        left.scatter(np.full(len(e), i) + np.random.default_rng(i).normal(0, 0.06, len(e)), e,
                     s=7, alpha=0.45, color="#1f4e79", zorder=3)
    left.set_xticks(range(len(sizes)), [f"{n:,}" for n in sizes])
    left.set_xlabel("total sample size $n$")
    left.set_ylabel(r"$|\hat\tau - \tau|$")
    left.set_title("PROBE error per replicate", fontsize=9)
    names = {"X": "$X$ only", "raw_XW": "raw $(X, W)$", "oracle": "oracle $(X, U, C)$"}
    right.plot(sizes, [np.mean(e) for e in err], "o-", color="#1f4e79", label="PROBE", lw=1.6)
    right.plot(sizes, [np.median([c["representation_error"] for c in by_n[n]]) for n in sizes],
               "s--", color="#4c8fbd", label="representation part", lw=1.1, ms=4)
    right.plot(sizes, [np.median([c["estimation_error"] for c in by_n[n]]) for n in sizes],
               "^--", color="#8fbcd9", label="estimation part", lw=1.1, ms=4)
    for key, style in zip(names, ("-.", ":", "-")):
        right.plot(sizes, [np.mean([abs(c["baselines"][key] - TRUE_EFFECT) for c in by_n[n]])
                           for n in sizes], style, color="#999999", lw=1.1, label=names[key])
    right.set_xscale("log"); right.set_yscale("log")
    right.set_xticks(sizes, [f"{n:,}" for n in sizes])
    right.minorticks_off()
    right.set_xlabel("total sample size $n$")
    right.set_ylabel("mean absolute error")
    right.set_title("PROBE against the comparators", fontsize=9)
    right.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, "minimal-suite-e12-sample-size")


def figure_scm6_objective() -> None:
    """The objective that should be minimised, and the one the protocol's sample size actually gives."""
    d = json.loads((RESULTS / "scm6_objective_curves_v1.json").read_text())
    grid, root = np.array(d["r_grid"]), d["balance_root"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    left.plot(grid, d["population_d2"], color="#1f4e79", lw=1.8, label="population $D^2_{res}$")
    left.axvline(root, color="#b03a2e", ls="--", lw=1.1,
                 label=f"balance root $r^\\star={root:+.4f}$")
    left.set_xlabel("coefficient $r$")
    left.set_ylabel("$D^2_{res}$")
    left.set_title("exact population objective", fontsize=9)
    left.legend(fontsize=7)
    shades = ["#c44e52", "#5f9ea0", "#1f4e79"]
    for (n, curve), colour in zip(sorted(d["by_sample"].items(), key=lambda kv: int(kv[0])), shades):
        right.plot(grid, curve, color=colour, lw=1.5, label=f"$n_\\phi = {int(n):,}$")
        best = grid[int(np.argmin(curve))]
        right.scatter([best], [min(curve)], color=colour, s=26, zorder=4)
    right.axvline(root, color="#b03a2e", ls="--", lw=1.1)
    right.set_xlabel("coefficient $r$")
    right.set_ylabel("estimated nested Brier gap")
    right.set_title("what Algorithm 1 minimises", fontsize=9)
    right.legend(fontsize=7)
    save(fig, "minimal-suite-scm6-objective")


def figure_scm6_levers() -> None:
    """Two levers on the same bias: rows given to the critics, and the width of the balance target."""
    d = json.loads((RESULTS / "scm6_objective_curves_v1.json").read_text())
    grid, root = np.array(d["r_grid"]), d["balance_root"]
    at_root = int(np.argmin(np.abs(grid - root)))
    truth = d["population_d2"][at_root]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ns = sorted(int(k) for k in d["by_sample"])
    bias = [abs(d["by_sample"][str(n)][at_root] - truth) for n in ns]
    left.loglog(ns, bias, "o-", color="#1f4e79", lw=1.6)
    ref = bias[0] * ns[0] / np.array(ns, dtype=float)
    left.loglog(ns, ref, ":", color="#999999", lw=1.2, label=r"$\propto 1/n_\phi$")
    left.set_xticks(ns, [f"{n:,}" for n in ns])
    left.minorticks_off()
    left.set_xlabel(r"rows given to the critics, $n_\phi$")
    left.set_ylabel(r"$|\,$estimated gap $-\ D^2_{res}|$ at $r^\star$")
    left.set_title("lever 1: critic sample", fontsize=9)
    left.legend(fontsize=7)
    widths = sorted(int(k) for k in d["by_width"])
    wbias = [abs(d["by_width"][str(w)][at_root] - truth) for w in widths]
    right.loglog(widths, wbias, "s-", color="#c44e52", lw=1.6)
    right.axhline(np.max(d["population_d2"]), color="#1f4e79", ls="--", lw=1.1,
                  label="full range of the signal")
    right.set_xticks(widths, [str(w) for w in widths])
    right.minorticks_off()
    right.set_xlabel("columns in the balance target")
    right.set_ylabel(r"$|\,$estimated gap $-\ D^2_{res}|$ at $r^\star$")
    right.set_title(r"lever 2: target width at $n_\phi = 1{,}500$", fontsize=9)
    right.legend(fontsize=7)
    save(fig, "minimal-suite-scm6-bias-levers")


def figure_scm7_renderer() -> None:
    """The image channel: the digit slides along one axis with K and nothing else moves that moment."""
    import probe_image_w as im
    values = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    lat = im.latents(len(values), 20260913)
    images = im.render(values, lat["digit"][:, 0], lat["template"][:, 0], np.zeros(len(values)))
    fig = plt.figure(figsize=(7.4, 3.1))
    spec = fig.add_gridspec(2, len(values), height_ratios=[1.15, 1.0], hspace=0.45)
    # Images are shown in the usual MNIST orientation.  The channel moves mass along the canvas's
    # first axis, which the plan writes as x and which appears vertical in that orientation, so the
    # white reference line at 48 is horizontal here.
    for i, value in enumerate(values):
        ax = fig.add_subplot(spec[0, i])
        ax.imshow(images[i], cmap="magma", origin="upper")
        ax.axhline(im.CENTRE, color="white", lw=0.6, alpha=0.6)
        ax.set_title(f"$K={value:+.1f}$", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    bottom = fig.add_subplot(spec[1, :])
    sweep = np.linspace(-6, 6, 61)
    batch = im.latents(len(sweep), 777)
    rendered = im.render(sweep, batch["digit"][:, 0], batch["template"][:, 0],
                         batch["style"][:, 0])
    bottom.plot(sweep, im.horizontal_centroid(rendered) - im.CENTRE, "o", ms=3,
                color="#1f4e79", label="rendered images")
    bottom.plot(sweep, im.SHIFT_SCALE * np.tanh(sweep / im.CHANNEL_SCALE), "-", color="#c44e52",
                lw=1.2, label=r"$8\tanh(K/3)$")
    bottom.set_xlabel("latent channel $K_j$")
    bottom.set_ylabel("first-axis centroid $-\\,48$")
    bottom.set_title("the first moment recovers the channel exactly, whatever the template or shift",
                     fontsize=8)
    bottom.legend(fontsize=7)
    save(fig, "minimal-suite-scm7-renderer")


def figure_scm6_repair() -> None:
    """Where the selected coefficient lands before and after the repair, against the true roots."""
    if not require("scm6_highdim_confirm_v1.json", "scm6_v2_development_v1.json"):
        return
    v1 = json.loads((RESULTS / "scm6_highdim_confirm_v1.json").read_text())
    v2 = json.loads((RESULTS / "scm6_v2_development_v1.json").read_text())
    roots = {k: v["root"] for k, v in v2["population_truth"].items() if v["root"] is not None}
    old_r = [row["r"] for cell in v1["cells"] for rot in cell["per_rotation"] for row in rot
             if row["r"] is not None and row["split_id"] in roots]
    new_rows = [r for cell in v2["cells"] for r in cell["rows"]
                if r["r"] is not None and r["split_id"] in roots]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    bins = np.linspace(-1.05, 1.05, 43)
    # The two runs have different cell counts, so the comparison is by share of selections.
    left.hist(old_r, bins=bins, density=True, color="#c44e52",
              label=f"v1, in-sample gap, full target ({len(old_r)})")
    left.hist([r["r"] for r in new_rows], bins=bins, density=True, color="#1f4e79", alpha=0.8,
              label=f"v2, cross-fitted gap, compressed target ({len(new_rows)})")
    for root in sorted(set(roots.values())):
        left.axvline(root, color="#444444", ls="--", lw=0.9)
    left.set_xlabel("selected coefficient $r$")
    left.set_ylabel("share of selections")
    left.set_title("dashed lines are the true balancing roots", fontsize=9)
    left.legend(fontsize=7)
    order = sorted(roots, key=lambda k: (len(k), k))
    med = [np.median([r["r"] for r in new_rows if r["split_id"] == k]) for k in order]
    right.scatter([roots[k] for k in order], med, s=34, color="#1f4e79", zorder=3)
    lim = (-0.8, 0.6)
    right.plot(lim, lim, "--", color="#888888", lw=1.0, label="exact agreement")
    seen = {}
    for k, m in zip(order, med):
        rank = seen[roots[k]] = seen.get(roots[k], -1) + 1
        right.annotate(k, (roots[k], m), fontsize=7, xytext=(6, -4 - 9 * rank),
                       textcoords="offset points", color="#444444")
    right.set_xlim(*lim); right.set_ylim(*lim)
    right.set_xlabel("true balancing root $r^\\star$")
    right.set_ylabel("median selected $r$ under v2")
    right.set_title("v2 is centred on every root", fontsize=9)
    right.legend(fontsize=7)
    save(fig, "minimal-suite-scm6-selection-repair")


def figure_comparators() -> None:
    """PROBE against every comparator on the two high-dimensional designs, development seeds."""
    if not require("scm6_v2_development_v3.json", "scm7_image_development_v2.json"):
        return
    panels = [("SCM-6, 1,282-dim orthogonal blocks", "scm6_v2_development_v3.json",
               [("X", "X only"), ("raw_ridge", "raw (X, W), ridge"),
                ("learned_projection", "learned channels, no balancing"),
                ("blockwise_pca", "blockwise PCA"), ("PROBE", "PROBE"), ("oracle", "oracle (X, U, C)")]),
              ("SCM-7, 46,082-dim MNIST images", "scm7_image_development_v2.json",
               [("X", "X only"), ("pca_channels", "blockwise PCA"),
                ("cnn_channels", "CNN channels, no balancing"),
                ("statistics_channels", "image statistics, no balancing"),
                ("PROBE", "PROBE"), ("oracle", "oracle (X, U, C)")])]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax, (title, name, order) in zip(axes, panels):
        data = json.loads((RESULTS / name).read_text())
        cells = [c for c in data["cells"] if c.get("estimate") is not None]
        errs = {}
        for key, _ in order:
            if key == "PROBE":
                errs[key] = np.array([abs(c["estimate"] - 1.0) for c in cells])
            else:
                errs[key] = np.array([abs(c["baselines"][key] - 1.0) for c in cells
                                      if key in c["baselines"]])
        labels = [lab for key, lab in order if len(errs[key])]
        keys = [key for key, _ in order if len(errs[key])]
        means = [errs[k].mean() for k in keys]
        colors = ["#1f4e79" if k == "PROBE" else ("#888888" if k == "oracle" else "#c44e52")
                  for k in keys]
        y = np.arange(len(keys))
        ax.barh(y, means, color=colors, height=0.62)
        for yi, k in zip(y, keys):
            ax.scatter(errs[k], np.full(len(errs[k]), yi), s=9, color="#222222", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("mean |estimate - 1|, five development seeds")
        ax.set_title(title, fontsize=9)
        ax.axvline(0.05, color="#444444", ls=":", lw=0.8)
    save(fig, "minimal-suite-comparators")


def _shards(pattern: str) -> list[dict]:
    import glob
    return [json.loads(Path(q).read_text()) for q in sorted(glob.glob(str(RESULTS / pattern)))]


def figure_v3_paired() -> None:
    """Main Figure 1: per-seed paired absolute errors by modality, with the mean and a simultaneous
    95 percent upper bound on each contrast against PROBE.  Dots show seed-level stability and paired
    superiority in a way bars cannot."""
    from scipy.stats import t as tdist
    panels = [("SCM-6-v3, 1,282-dim vector proxies", "scm6_v3/shards/confirmation_*.json",
               [("X", "X only"), ("raw_ridge", "raw (X, W) ridge"), ("blockwise_pca", "blockwise PCA"),
                ("learned_projection", "learned channels, no balance"), ("PROBE", "PROBE"),
                ("oracle", "oracle (X, U, C)")]),
              ("SCM-7-v3, 46,082-dim image proxies", "scm7_v3/shards/confirmation_*.json",
               [("X", "X only"), ("raw_cnn", "raw (X, W) CNN"), ("pca_channels", "blockwise PCA"),
                ("statistics_channels", "image statistics, no balance"),
                ("cnn_channels", "CNN channels, no balance"), ("PROBE", "PROBE"),
                ("oracle", "oracle (X, U, C)")])]
    cells = {name: _shards(pat) for name, pat, _ in panels}
    if not any(cells.values()):
        print("  skipped, no v3 confirmation shards yet")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
    for ax, (title, pat, order) in zip(axes, panels):
        qs = [q for q in cells[title] if q.get("estimate") is not None]
        if not qs:
            ax.set_title(title + " (pending)", fontsize=9); ax.axis("off"); continue
        probe = np.array([abs(q["estimate"] - 1.0) for q in qs])
        keys = [k for k, _ in order if k == "PROBE" or all(k in q["baselines"] for q in qs)]
        labels = [lab for k, lab in order if k in keys]
        errs = {k: (probe if k == "PROBE" else np.array([abs(q["baselines"][k] - 1.0) for q in qs]))
                for k in keys}
        n_con = max(1, len(keys) - 2)
        y = np.arange(len(keys))
        for yi, k in zip(y, keys):
            e = errs[k]
            colour = "#1f4e79" if k == "PROBE" else ("#888888" if k == "oracle" else "#c44e52")
            ax.scatter(e, np.full(len(e), yi) + np.random.default_rng(0).uniform(-0.18, 0.18, len(e)),
                       s=8, color=colour, alpha=0.7, zorder=3)
            ax.plot([e.mean(), e.mean()], [yi - 0.32, yi + 0.32], color="#111111", lw=1.6, zorder=4)
            if k not in ("PROBE",):
                d = probe - e
                ub = d.mean() + tdist.ppf(1 - 0.05 / n_con, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                ax.annotate(f"UB {ub:+.3f}", (ax.get_xlim()[1] if False else e.max(), yi),
                            xytext=(6, -3), textcoords="offset points", fontsize=6.5, color="#333333")
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5); ax.invert_yaxis()
        ax.set_xscale("log"); ax.set_xlabel("|estimate - 1|, one dot per seed; bar is the mean")
        ax.set_title(f"{title}, {len(qs)} seeds", fontsize=9)
        ax.axvline(0.05, color="#444444", ls=":", lw=0.8)
    save(fig, "minimal-suite-v3-paired")


def figure_v3_aggregation() -> None:
    """Main Figure 2: why aggregation is needed, on one confirmation replicate of SCM-6-v3.
    Candidate estimates on the axis, coloured by their population target, linked within 2 rho, the
    largest component highlighted, the exactly balanced wrong singleton marked."""
    qs = [q for q in _shards("scm6_v3/shards/confirmation_*.json") if q.get("estimate") is not None]
    if not qs:
        print("  skipped, no SCM-6-v3 confirmation shards yet")
        return
    # The mechanism is visible only when a balanced-but-wrong candidate survived the screen alongside
    # the true-target ones: choose the replicate with the most retained candidates among those where at
    # least one retained candidate targets another value, else the one with the most candidates.
    def other_target(q):
        truth = {r["split_id"]: r["evaluator_only"]["targets_truth"] for r in q["rows"]}
        return sum(not truth[k] for k in q["candidate_estimates"])
    with_other = [q for q in qs if other_target(q) > 0]
    pool = with_other if with_other else qs
    q = max(pool, key=lambda q: len(q["candidate_estimates"]))
    est = q["candidate_estimates"]; rho = q["rho"]; members = set(q["largest_component"])
    truth = {r["split_id"]: r["evaluator_only"]["targets_truth"] for r in q["rows"]}
    names = sorted(est, key=lambda k: est[k])
    xs = np.array([est[k] for k in names])
    # Two panels on one axis line: the full range showing the exactly balanced wrong candidate far
    # from the rest, and a zoom around the true-target cluster where the links and the component live.
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 2.6), gridspec_kw={"width_ratios": [1, 1.6]})
    rng = np.random.default_rng(1)
    ypos = {k: (float(rng.uniform(-0.3, 0.3)) if truth[k] else 0.0) for k in names}
    for ax, (lo, hi), zoom in ((left, (xs.min() - 0.15, xs.max() + 0.15), False),
                               (right, (min(x for x in xs if x > 0.5) - 0.03, max(xs) + 0.03), True)):
        y = {k: (ypos[k] if zoom else 0.0) for k in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if abs(est[a] - est[b]) <= 2 * rho:
                    ax.plot([est[a], est[b]], [y[a], y[b]], color="#bbbbbb", lw=1.0, zorder=1)
        for k, x in zip(names, xs):
            ax.scatter(x, y[k], s=70 if k in members else 34, zorder=3,
                       color="#1f4e79" if truth[k] else "#c44e52",
                       edgecolor="#111111" if k in members else "none", linewidth=1.2)
            if zoom or not truth[k]:
                ax.annotate(k, (x, y[k]), xytext=(0, 8 if y[k] >= 0 else -12),
                            textcoords="offset points", ha="center", fontsize=6.5)
        ax.axvline(1.0, color="#444444", ls="--", lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
        ax.set_xlabel("candidate estimate", fontsize=8)
    left.set_title("all retained candidates", fontsize=8.5, pad=4)
    right.set_title(f"true-target cluster; grey links within 2 rho = {2 * rho:.3f}", fontsize=8.5, pad=4)
    fig.suptitle(f"SCM-6-v3 seed {q['data_seed']}: {len(names)} retained, largest component "
                 f"{len(members)} ringed, returned {q['estimate']:.3f}.  Blue: population target 1; "
                 f"red: another value.", fontsize=8.5, y=1.02)
    save(fig, "minimal-suite-v3-aggregation")


def main() -> int:
    FIGURES.mkdir(exist_ok=True)
    for name, fn in (("E12", figure_e12), ("SCM-6 objective", figure_scm6_objective),
                     ("SCM-6 levers", figure_scm6_levers),
                     ("SCM-6 repair", figure_scm6_repair),
                     ("comparators", figure_comparators),
                     ("v3 paired", figure_v3_paired), ("v3 aggregation", figure_v3_aggregation),
                     ("SCM-7 renderer", figure_scm7_renderer)):
        print(f"{name}:")
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
