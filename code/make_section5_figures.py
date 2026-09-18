"""Section 5 figures.  One message per figure, one theorem per figure, as little text as the message allows.

Colour carries one job across every figure; row names and direct labels carry identity, so the figures read
in grayscale:
  blue              PROBE, or a held-out block that satisfies the theorem's assumptions
  orange            the contaminated block W4, or a candidate built on it
  gray              every other method, the oracle, and theoretical references
  thin dashed ink   the true effect tau = 1
Blue and orange are slots 1 and 2 of the validated reference palette (all-pairs CVD dE 24.7, normal 33.6).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS, FIGDIR = HERE.parent / "results", HERE.parent / "figures" / "section5"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
INK, INK2, HAIR, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
TAU = 1.0
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def text_collisions(fig, pad: float = 1.0) -> list[tuple[str, str]]:
    """Every pair of visible text boxes that overlap on the rendered canvas.

    Guessing label widths from font size failed twice on these figures, so the layout is measured instead:
    draw the canvas, take each text artist's window extent, and report intersecting pairs.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # tick labels for ticks outside the view limits exist as artists but are never drawn
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            span = hi - lo
            for tick in axis.get_major_ticks():
                if not (lo - 1e-9 * span <= tick.get_loc() <= hi + 1e-9 * span):
                    hidden.update({id(tick.label1), id(tick.label2)})
    boxes = []
    for t in fig.findobj(matplotlib.text.Text):
        if id(t) in hidden or not t.get_visible() or not t.get_text().strip():
            continue
        try:
            bb = t.get_window_extent(renderer)
        except Exception:
            continue
        if bb.width < 1 or bb.height < 1:
            continue
        boxes.append((t.get_text().replace("\n", " ")[:32], bb.expanded(1.0, 1.0).padded(-pad)))
    hits = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                hits.append((boxes[i][0], boxes[j][0]))
    return hits


def save(fig, stem: str) -> None:
    hits = text_collisions(fig)
    if hits:
        print(f"  TEXT COLLISIONS in {stem}: {hits}")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/section5/{stem}.pdf and .png")


def shards(pattern: str) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(RESULTS / pattern)))]


def truth_line(ax, label: bool = False) -> None:
    ax.axvline(TAU, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
    if label:
        ax.annotate("true effect", (TAU, 1.0), xycoords=("data", "axes fraction"), xytext=(-3, -1),
                    textcoords="offset points", ha="right", va="top", fontsize=7, color=INK2)


def dots(ax, x, y, colour: str, size: float = 14) -> None:
    jitter = np.random.default_rng(int(1000 * y) + len(x)).uniform(-0.17, 0.17, len(x))
    ax.scatter(x, y + jitter, s=size, color=colour, edgecolors="white", linewidths=0.4, zorder=3)
    ax.plot([np.mean(x)] * 2, [y - 0.32, y + 0.32], color=INK, lw=1.4, zorder=4, solid_capstyle="butt")


def key(fig, handles, y=1.0, ncol=None):
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(handles),
               frameon=False, fontsize=7, handlelength=1.6, columnspacing=1.4, handletextpad=0.5)


from matplotlib.lines import Line2D


def h_dot(colour, label, marker="o"):
    return Line2D([], [], marker=marker, lw=0, markersize=5, markerfacecolor=colour, markeredgecolor="white", label=label)


H_MEAN = Line2D([], [], marker="|", lw=0, markersize=9, markeredgewidth=1.4, color=INK, label="average over data sets")
H_TRUTH = Line2D([], [], color=INK, lw=0.9, ls=(0, (3, 2)), label="true effect = 1")


# ----------------------------------------------------------------------------- Figure 1
def fig1_estimates() -> None:
    """Only PROBE recovers the true effect, whatever form the proxy takes."""
    rows = ["Naive comparison", "Adjust for X", "Adjust for X and W", "Learned summary of W,\nno balance check",
            "PROBE (ours)", "Oracle: adjust for\nhidden U"]
    e12 = json.loads((RESULTS / "e12_scm1_ncurve_confirm_v2.json").read_text())
    scm1 = [q for q in e12["cells"] if q["n_total"] == 12000 and q["estimate"] is not None]
    scm6 = shards("scm6_v3/shards/confirmation_*.json")
    # SCM-7-v4 is the image confirmation of record once its sealed score exists; v3 became calibration data.
    scm7 = (shards("scm7_v4/shards/confirmation_*.json") if (RESULTS / "scm7_v4/scores_confirmation_v1.json").exists()
            else shards("scm7_v3/shards/confirmation_*.json"))
    note = ""
    if not scm7:
        scm7, note = shards("scm7_v3/deltaai_qualification_2ae2898a/*.json"), "\n(preview: 2 data sets)"

    def best(cells, keys):
        present = [k for k in keys if all(k in q["baselines"] for q in cells)]
        return min(present, key=lambda k: np.mean([abs(q["baselines"][k] - TAU) for q in cells])) if present else None

    cols = [("W = 7 numbers", scm1, "raw_XW", None),
            ("W = 1,282-dim vector", scm6, "raw_ridge", best(scm6, ["learned_projection", "blockwise_pca"])),
            ("W = five 96×96 images" + note, scm7, "raw_cnn",
             best(scm7, ["cnn_channels", "statistics_channels", "pca_channels"]))]
    fig, axes = plt.subplots(1, 3, figsize=(5.8, 2.75), sharey=True)
    ypos = {r: len(rows) - 1 - i for i, r in enumerate(rows)}
    for ax, (title, cells, raw, nb) in zip(axes, cols):
        get = lambda k: [q["baselines"][k] for q in cells]
        dots(ax, get("naive"), ypos[rows[0]], GRAY)
        dots(ax, get("X"), ypos[rows[1]], GRAY)
        if all(raw in q["baselines"] for q in cells):
            dots(ax, get(raw), ypos[rows[2]], GRAY)
        if nb:
            dots(ax, get(nb), ypos[rows[3]], GRAY)
        answered = [q["estimate"] for q in cells if q["estimate"] is not None]
        dots(ax, answered, ypos[rows[4]], BLUE)
        if len(answered) < len(cells):
            # a data set where the majority vote tied or nothing passed the balance check returns no answer
            ax.annotate(f"answered\n{len(answered)} of {len(cells)}", (min(answered), ypos[rows[4]]),
                        xytext=(-9, 0), textcoords="offset points", ha="right", va="center", fontsize=6.5, color=INK2)
        dots(ax, get("oracle"), ypos[rows[5]], GRAY)
        ax.axvline(TAU, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(-0.45, 1.3); ax.set_xticks([0, 0.5, 1]); ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(title, fontsize=8, pad=4)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=HAIR, lw=0.5, zorder=0); ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels(rows[::-1])
    axes[0].get_yticklabels()[1].set_color(BLUE); axes[0].get_yticklabels()[1].set_fontweight("bold")
    axes[1].set_xlabel("Estimated treatment effect")
    key(fig, [h_dot(GRAY, "one data set, other method"), h_dot(BLUE, "one data set, PROBE"), H_MEAN, H_TRUTH],
        y=1.07)
    save(fig, "section5-fig1-estimates")


# ----------------------------------------------------------------------------- Figure 2
def _runs(mask: np.ndarray) -> list[slice]:
    """Contiguous True runs, so a line never jumps across points that fall outside the view."""
    out, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append(slice(start, i)); start = None
    return out


def fig2_certificate() -> None:
    """If the held-out blocks are clean, removing the imbalance removes the bias.  If one is contaminated, it does not."""
    path = RESULTS / "population_certificate_curve_v1.json"
    if not path.exists():
        print("  skipped fig2"); return
    data = json.loads(path.read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.8, 2.55), sharey=True, gridspec_kw={"wspace": 0.12})
    panels = ((a, [k for k, v in data["splits"].items() if v["label"] == "valid"], BLUE,
               "(a) Held-out blocks are clean proxies of U"),
              (b, ["S4"], ORANGE, "(b) Held-out block is contaminated"))
    for ax, sids, colour, title in panels:
        for sid in sids:
            pts = sorted(data["splits"][sid]["points"], key=lambda q: q["r"])
            d = np.array([q["d_res"] for q in pts]); bias = np.array([abs(q["tau_z"] - TAU) for q in pts])
            for run in _runs(d <= 0.1):
                ax.plot(d[run], bias[run], color=colour, lw=0.7, alpha=0.45, zorder=2)
                ax.scatter(d[run], bias[run], s=11, color=colour, edgecolors="white", linewidths=0.3, zorder=3)
        ax.set_title(title, fontsize=8, pad=4, loc="left")
        ax.set_xlim(-0.004, 0.1); ax.set_ylim(-0.06, 1.9)
        ax.set_xlabel("Leftover imbalance $D_{\\mathrm{res}}$")
        ax.grid(color=HAIR, lw=0.5, zorder=0); ax.set_axisbelow(True)
    a.scatter([0], [0], s=46, color=BLUE, edgecolors=INK, linewidths=0.9, zorder=5)
    a.annotate("zero imbalance,\nzero bias", (0, 0), xytext=(0.05, 0.1), textcoords="data", fontsize=7,
               color=INK2, va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkB=4))
    s4 = data["splits"]["S4"]; at = next(q for q in s4["points"] if abs(q["r"] - s4["root"]) < 1e-5)
    b.scatter([0], [abs(at["tau_z"] - TAU)], s=46, color=ORANGE, edgecolors=INK, linewidths=0.9, zorder=5)
    b.annotate(f"zero imbalance,\nbut bias {abs(at['tau_z'] - TAU):.2f}", (0, abs(at["tau_z"] - TAU)),
               xytext=(0.012, 0.55), textcoords="data", fontsize=7, color=INK2, va="center", ha="left",
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkB=4))
    a.set_ylabel("Bias of the adjusted effect\n$|\\tau_Z-\\tau|$")
    fig.text(0.5, 1.01, "each dot = one candidate representation $Z$;   "
             "imbalance = how much the held-out block still predicts treatment",
             ha="center", va="bottom", fontsize=7, color=INK2)
    save(fig, "section5-fig2-certificate")


# ----------------------------------------------------------------------------- Figure 3
def fig3_aggregation() -> None:
    """A contaminated held-out block can pass the balance check; the majority vote ignores it, an average does not."""
    cells = shards("scm6_v3/shards/confirmation_*.json")
    if not cells:
        print("  skipped fig3"); return

    def split_truth(q):
        return {r["split_id"]: r["evaluator_only"]["targets_truth"] for r in q["rows"]}

    with_bad = [q for q in cells if any(not split_truth(q)[k] for k in q["candidate_estimates"])]
    ex = max(with_bad, key=lambda q: len(q["candidate_estimates"]))
    tr = split_truth(ex)
    clean = sorted(v for k, v in ex["candidate_estimates"].items() if tr[k])
    bad = [v for k, v in ex["candidate_estimates"].items() if not tr[k]]
    avg_ex = float(np.mean(clean + bad))

    fig, (a, b) = plt.subplots(1, 2, figsize=(5.8, 2.45), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.55})
    for i, v in enumerate(clean):
        a.scatter(v, 2 + (i - (len(clean) - 1) / 2) * 0.09, s=16, color=BLUE, edgecolors="white", linewidths=0.4, zorder=3)
    a.scatter(bad, [2] * len(bad), s=16, color=ORANGE, edgecolors="white", linewidths=0.4, zorder=3)
    # two labels on one row: the contaminated one above its dot, the clean one below the stack
    a.annotate("1 held-out choice,\ncontaminated", (bad[0], 2), xytext=(-4, 7), textcoords="offset points",
               ha="left", va="bottom", fontsize=7, color=INK2)
    a.annotate(f"{len(clean)} held-out\nchoices, all clean", (min(clean), 2 - (len(clean) - 1) / 2 * 0.09),
               xytext=(-8, -4), textcoords="offset points", ha="right", va="top", fontsize=7, color=INK2)
    a.scatter([avg_ex], [1], s=34, marker="D", color=GRAY, zorder=4)
    a.annotate(f"{avg_ex:.2f}", (avg_ex, 1), xytext=(-7, 0), textcoords="offset points", ha="right", va="center", fontsize=7, color=INK2)
    a.scatter([ex["estimate"]], [0], s=64, marker="*", color=BLUE, zorder=4)
    a.annotate(f"{ex['estimate']:.2f}", (ex["estimate"], 0), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=7, color=INK2)
    a.axvline(TAU, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=1)
    a.set_yticks([2, 1, 0]); a.set_yticklabels(["Estimates that passed\nthe balance check", "Average of them", "PROBE: majority vote"])
    a.tick_params(axis="y", length=0); a.set_ylim(-0.6, 2.75); a.set_xlim(-0.7, 1.3); a.set_xticks([-0.5, 0, 0.5, 1])
    a.set_xlabel("Estimated treatment effect")
    a.set_title("(a) One simulated data set", fontsize=8, pad=4, loc="left")
    a.grid(axis="x", color=HAIR, lw=0.5, zorder=0); a.set_axisbelow(True)
    avg = [float(np.mean(list(q["candidate_estimates"].values()))) for q in cells]
    dots(b, avg, 1, GRAY); dots(b, [q["estimate"] for q in cells], 0, BLUE)
    b.axvline(TAU, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=1)
    b.set_yticks([1, 0]); b.set_yticklabels(["Average of passing\nestimates", "PROBE: majority vote"])
    b.tick_params(axis="y", length=0); b.set_ylim(-0.6, 1.6); b.set_xlim(0.55, 1.25)
    b.set_xlabel("Estimated treatment effect")
    b.set_title(f"(b) {len(cells)} simulated data sets, one dot each", fontsize=8, pad=4, loc="left")
    b.grid(axis="x", color=HAIR, lw=0.5, zorder=0); b.set_axisbelow(True)
    key(fig, [H_MEAN, H_TRUTH], y=1.08)
    save(fig, "section5-fig3-aggregation")


# ----------------------------------------------------------------------------- Figure 4
def fig4_finite_sample() -> None:
    """The guaranteed success probability holds and rises with the number of held-out choices; error falls with n."""
    mpath = RESULTS / "e11_v2_mcurve_v1.json"
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.8, 2.45), gridspec_kw={"wspace": 0.5})
    rho_txt = "ρ"
    if mpath.exists():
        mc = json.loads(mpath.read_text())
        rho_txt = f"ρ = {mc['rho']:.2f}"
        m = np.array([r["m"] for r in mc["rows"]]); bound = np.clip([r["bound"] for r in mc["rows"]], 0, 1)
        obs = np.array([r["success"] for r in mc["rows"]])
        lo = np.array([r["ci95"][0] for r in mc["rows"]]); hi = np.array([r["ci95"][1] for r in mc["rows"]])
        a.plot(m, bound, color=GRAY, lw=1.5, zorder=2, label="theorem's lower bound")
        a.vlines(m, lo, hi, color=BLUE, lw=0.8, zorder=3)
        a.scatter(m, obs, s=14, color=BLUE, zorder=4, label=f"observed ({mc['reps']:,} runs)")
        a.set_ylim(-0.03, 1.05); a.set_xlim(0.4, m.max() + 0.6); a.set_xticks([1, 2, 4, 6, 8, 10, 12, 14])
        a.legend(loc="lower right", frameon=False, fontsize=6.5)
    a.set_xlabel("Number of held-out choices tried, $m$")
    a.set_ylabel(f"Chance the estimate is\nwithin {rho_txt} of the true effect")
    a.set_title("(a) Guarantee vs. observed", fontsize=8, pad=4, loc="left")
    a.grid(color=HAIR, lw=0.5, zorder=0); a.set_axisbelow(True)
    e12 = json.loads((RESULTS / "e12_scm1_ncurve_confirm_v2.json").read_text())
    ns = sorted({q["n_total"] for q in e12["cells"]})

    def mae(k, n):
        cs = [q for q in e12["cells"] if q["n_total"] == n and q["estimate"] is not None]
        return np.mean([abs((q["estimate"] if k == "probe" else q["baselines"][k]) - TAU) for q in cs])
    for k, colour, label in (("raw_XW", GRAY, "Adjust for X and W"), ("probe", BLUE, "PROBE (ours)"),
                              ("oracle", "#b9b8b2", "Oracle: adjust for hidden U")):
        ys = [mae(k, n) for n in ns]
        b.plot(ns, ys, color=colour, lw=1.4, marker="o", markersize=3.5, zorder=3)
        b.annotate(label, (ns[-1], ys[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=6.5, color=INK2)
    b.set_xscale("log"); b.set_yscale("log"); b.minorticks_off()
    b.set_xticks(ns); b.set_xticklabels([f"{n:,}" for n in ns]); b.set_xlim(ns[0] * 0.8, ns[-1] * 4.5)
    b.set_xlabel("Sample size $n$"); b.set_ylabel("Average error  $|\\hat\\tau-\\tau|$")
    b.set_title("(b) More data, less error", fontsize=8, pad=4, loc="left")
    b.grid(color=HAIR, lw=0.5, zorder=0); b.set_axisbelow(True)
    save(fig, "section5-fig4-finite-sample")


# ----------------------------------------------------------------------------- Theorem 1
def fig_thm1_fixed_block() -> None:
    """Theorem 1, one panel per SCM: balance a clean held-out block and the effect comes back; a contaminated one, not."""
    path = RESULTS / "t1_fixed_block_scm_family_v1.json"
    if not path.exists():
        print("  skipped thm1"); return
    cells = json.loads(path.read_text())["cells"]
    scms = [("SCM-1", "base"), ("SCM-2", "outcome noise"), ("SCM-3", "varying effect"), ("SCM-4", "proxy noise"),
            ("SCM-5", "nonlinear proxy")]
    rows = [("Adjust for $X$", lambda q: q["baselines"]["X"], GRAY),
            ("Adjust for $(X,W)$", lambda q: q["baselines"]["raw_XW"], GRAY),
            ("PROBE, held-out $W_1$\n(clean)", lambda q: q["fixed_blocks"]["W1"]["estimate"], BLUE),
            ("PROBE, held-out $W_4$\n(contaminated)", lambda q: q["fixed_blocks"]["W4"]["estimate"], ORANGE)]
    width, height = 6.5, 2.4
    # five panels at full text width: 9 pt text, margins measured so nothing is rescaled when placed
    with plt.rc_context({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "savefig.bbox": "standard"}):
        fig, axes = plt.subplots(1, len(scms), figsize=(width, height), sharey=True)
        for k, (ax, (scm, what)) in enumerate(zip(axes, scms)):
            sel = [q for q in cells if q["scm"] == scm]
            for i, (_, get, colour) in enumerate(rows):
                xs = [get(q) for q in sel]
                jitter = np.random.default_rng(100 * k + i).uniform(-0.16, 0.16, len(xs))
                ax.scatter(xs, len(rows) - 1 - i + jitter, s=16, color=colour, edgecolors="white", linewidths=0.35,
                           zorder=3)
            ax.axvline(TAU, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
            ax.set_xlim(-0.75, 1.25)
            ax.set_xticks([0, 1], ["0", "1"])
            ax.set_ylim(-0.6, len(rows) - 0.4)
            ax.set_title(f"{scm}\n{what}", fontsize=9, pad=3)
            ax.tick_params(axis="y", length=0, pad=4)
            ax.spines["left"].set_visible(False)
            ax.grid(axis="x", color=HAIR, lw=0.5, zorder=0); ax.set_axisbelow(True)
        axes[0].set_yticks(range(len(rows)), [label for label, _, _ in rows][::-1])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        label_w = max(t.get_window_extent(renderer).width for t in axes[0].get_yticklabels()) / fig.dpi
        fig.subplots_adjust(left=(label_w + 0.1) / width, right=0.99, bottom=0.55 / height, top=1 - 0.42 / height,
                            wspace=0.1)
        x0, x1 = axes[0].get_position().x0, axes[-1].get_position().x1
        fig.text((x0 + x1) / 2, 0.04 / height, "Estimated treatment effect  (dashed line: true effect)",
                 ha="center", va="bottom", fontsize=9)
        fig.canvas.draw()
        bb = fig.get_tightbbox(renderer)
        if bb.x0 < -0.01 or bb.y0 < -0.01 or bb.x1 > width + 0.01 or bb.y1 > height + 0.01:
            raise RuntimeError(f"thm1 figure content leaves the {width} x {height} in page: {bb}")
        save(fig, "section5-thm1-fixed-block")


def main() -> int:
    for name, fn in (("fig1", fig1_estimates), ("fig2", fig2_certificate), ("fig3", fig3_aggregation),
                     ("fig4", fig4_finite_sample), ("thm1", fig_thm1_fixed_block)):
        print(name)
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
