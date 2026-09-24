"""PROBE as a mechanism, 1x4 at text width, drawn against one sealed data set of experiment E2.

Panels (a) and (b) are schematic.  The blocks are drawn as images to make the point that a block may be
an image.  The thumbnails are original Shapes3D images, drawn by the SCM-4 rule: one image per recorded
level, uniform within that level's group of the causal bank, with every other generative factor free.  They
use a figure-only seed and belong to no data set of any experiment reported here.  There is one thumbnail
per proxy block of SCM-4, J = 5, so (a) counts 2^5 - 2 = 30 splits.

Panels (c) and (d) carry real numbers from experiment E2 (SCM-4 sealed confirmation, Shapes3D images; see
EXPERIMENT_INDEX.md).  They come from the sealed shard results/family_v2/scm4/confirm/SCM-4_seed97400000.json,
replicate 0, the replicate with the lowest seed.  It is picked by its seed alone, never by its outcome.

    (c) one dot per split S of the five blocks.  Its screen statistic is the largest, over the four
        cross-fitting rotations, of the screen's upper bound  max(gap, 0) + z SE,  where gap is the held-out
        Brier risk difference of fresh nested critics refitted on the screen fold (family_v2_probe.screen).
        A split is kept when it passes the screen in all four rotations; t is the shard's tolerance.
    (d) the estimates of the kept splits (each the mean over the four rotations), linked within 2 rho; the
        returned tau-hat is the median of the unique largest component (probe_structured_scm.aggregate).

Before anything is drawn the script checks, on the shard itself, that every split's pass_all is the
conjunction of its four rotation passes, that the kept splits are the shard's retained list, that every
rotation's pass equals (upper <= t and overlap), that linking within 2 rho reproduces the shard's component
sizes and members, and that tau-hat is the median of the largest component.  It stops if any check fails.

The earlier version, code/make_probe_mechanism_figure.py, drew (c) and (d) from the positive-control audit
(results/positive_control/screen_display_v1.json), which is not a final experiment, and drew J = 7 blocks.
Writes figures/section5/probe-mechanism-v2.{pdf,png} and copies the pdf to
ICLR/paper/figures/probe-mechanism-v2.pdf.
"""
import json
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle
from matplotlib.ticker import NullLocator

import make_section5_figures as m
import family_v2_images as fim

SHARD_DIR = m.RESULTS / "family_v2" / "scm4" / "confirm"
PAPER_FIGDIR = m.HERE.parent / "ICLR" / "paper" / "figures"
STEM = "probe-mechanism-v2"
J = 5
VALID, INVALID = "S1", "S4"   # W1 alone held out is a valid block; W4 alone is balanceable, not outcome-complete
W, H = 6.5, 1.55                         # the float is \linewidth wide; keep it shallow
pw, ph, gx, margin = 1.44, 1.43, 0.20, 0.05
FONT, SMALL = 7.2, 6.2


def stop(message: str) -> None:
    raise SystemExit(f"STOP, nothing drawn: {message}")


# ----------------------------------------------------------------------------- E2, replicate 0 (lowest seed)

def seed_of(path) -> int:
    return int(path.stem.split("_seed")[1])


shard_paths = sorted(SHARD_DIR.glob("SCM-4_seed*.json"), key=seed_of)
if not shard_paths:
    stop(f"no SCM-4 shard under {SHARD_DIR}")
SHARD = shard_paths[0]
d = json.loads(SHARD.read_text())
task = d["task"]
if (d["stage"], task["level"], task["rep"], task["seed"]) != ("confirm", "SCM-4", 0, seed_of(SHARD)):
    stop(f"{SHARD.name} is not the sealed SCM-4 confirmation replicate 0: {d['stage']}, {task}")
probe = d["probe"]
t = probe["tolerance"]
names = probe["split_names"]
records = probe["records"]
if list(records) != names or len(names) != 2 ** J - 2 or any(len(records[s]["rotations"]) != 4 for s in names):
    stop("the shard does not carry 2^J - 2 splits with four rotations each")
print(f"shard: {SHARD.relative_to(m.HERE.parent)} (rep {task['rep']}, seed {task['seed']}, "
      f"lowest of {len(shard_paths)}), n = {d['n']}, t = {t}")

# ----------------------------------------------------------------------------- checks against the shard

failures = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  check  {label}: {'ok' if ok else 'FAILS'}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


bad = [s for s in names if records[s]["pass_all"] != all(r["screen"]["pass"] for r in records[s]["rotations"])]
check("pass_all == all four rotation passes, every split", not bad, str(bad) if bad else "")
kept_names = [s for s in names if records[s]["pass_all"]]
retained = probe["retained"]
check("the pass_all splits == retained", set(kept_names) == set(retained),
      f"pass_all {kept_names}, retained {retained}")
rotations = [(s, r["rotation"], r["screen"]) for s in names for r in records[s]["rotations"]]
agree = sum(sc["pass"] == (sc["upper"] <= t and sc["overlap"]) for _, _, sc in rotations)
check("pass == (upper <= t and overlap), every rotation", agree == len(rotations),
      f"{agree} of {len(rotations)} rotations agree")
strict = sum(sc["pass"] == (sc["upper"] < t and sc["overlap"]) for _, _, sc in rotations)
print(f"         the code's own rule is strict, upper < t and overlap: {strict} of {len(rotations)} agree; "
      f"overlap holds in {sum(sc['overlap'] for _, _, sc in rotations)} of {len(rotations)}")

est = np.array([records[s]["estimate"] for s in retained])
rho, tau_hat = probe["rho"], probe["estimate"]
parent = list(range(len(est)))


def find(i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for i in range(len(est)):
    for j in range(i + 1, len(est)):
        if abs(est[i] - est[j]) <= 2.0 * rho:
            parent[find(i)] = find(j)
groups = {}
for i in range(len(est)):
    groups.setdefault(find(i), []).append(i)
components = sorted(groups.values(), key=lambda c: (-len(c), c[0]))
sizes = [len(c) for c in components]
check("linking within 2 rho reproduces component_sizes", sizes == probe["component_sizes"],
      f"{sizes} vs {probe['component_sizes']}")
largest = [c for c in components if len(c) == sizes[0]]
unique = len(largest) == 1 and probe["return_kind"] == "unique_largest" and len(probe["members"]) == 1
check("the largest component is unique", unique, probe["return_kind"])
big = largest[0]
big_blocks = sorted([int(ch) - 1 for ch in retained[i][1:]] for i in big)
check("the largest component reproduces members", unique and big_blocks == sorted(probe["members"][0]),
      "{" + ", ".join(retained[i] for i in big) + "}")
median = float(np.median(est[big]))
check("tau-hat == median of the largest component", abs(median - tau_hat) <= 1e-12,
      f"{median!r} vs {tau_hat!r}")
others = sorted(retained[i] for c in components[1:] for i in c)
w4 = sorted(s for s in retained if "4" in s[1:])
check("outside the largest component == kept splits holding out W4 (orange means W4)", others == w4,
      f"{others} vs {w4}")
if failures:
    stop("; ".join(failures))

stat = np.array([max(r["screen"]["upper"] for r in records[s]["rotations"]) for s in names])
kept = np.array([records[s]["pass_all"] for s in names])
index = {s: k for k, s in enumerate(names)}
in_big = np.zeros(len(retained), dtype=bool)
in_big[big] = True
print(f"screen: {kept.sum()} of {len(names)} splits kept at t = {t}: {', '.join(retained)}; statistic "
      f"{stat.min():.2e} to {stat.max():.2e}; {VALID} {stat[index[VALID]]:.2e} "
      f"({'kept' if kept[index[VALID]] else 'dropped'}), {INVALID} {stat[index[INVALID]]:.2e} "
      f"({'kept' if kept[index[INVALID]] else 'dropped'})")
print(f"aggregate: 2rho = {2 * rho:.4f}; components {sizes}; largest "
      + ", ".join(f"{retained[i]} {est[i]:.4f}" for i in big)
      + "; others " + ", ".join(f"{retained[i]} {est[i]:.4f}" for i in range(len(est)) if not in_big[i])
      + f"; returned {tau_hat:.5f}")

THUMB_SEED = 96_920_000                  # a stream no experiment uses; these images decide nothing


def shapes3d_thumbs(count: int, seed: int) -> list[np.ndarray]:
    """`count` original Shapes3D images at distinct orientation levels, drawn the way SCM-4 draws them."""
    groups = fim.bank_by_level(np.load(fim.BANK_DIR / "split_rows.npz")["causal_bank"])
    rng = np.random.default_rng(seed)
    levels = rng.choice(fim.LEVELS, size=count, replace=False)
    picks = fim.draw_images(levels, groups, rng)
    images = fim.load_images()
    print(f"thumbnails: orientation levels {levels.tolist()}, bank rows {picks.tolist()}")
    return [np.asarray(images[row]) for row in picks]


thumbs = shapes3d_thumbs(J, THUMB_SEED)

TITLES = ["(a) Split proxy $W$ into\nblocks; hold some out",
          "(b) Learn $Z$ from $X$ and\nthe kept blocks",
          "(c) Given $Z$, do the\nheld-out blocks predict $A$?",
          "(d) Repeat; the largest\ncomponent gives $\\widehat\\tau$"]


def frame(inset, colour=None, dashed=False):
    inset.set_xticks([]); inset.set_yticks([])
    for sp in inset.spines.values():
        sp.set_visible(True); sp.set_color(colour or m.AXIS); sp.set_linewidth(1.2 if colour else 0.5)
        if dashed:
            sp.set_linestyle((0, (2, 1.2)))


def bare(ax):
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(m.AXIS)
    ax.set_facecolor("none")
    ax.set_yticks([])


with plt.rc_context({"font.size": FONT, "savefig.bbox": "standard"}):
    fig = plt.figure(figsize=(W, H))
    y0 = (H - margin - ph) / H
    axs = [fig.add_axes([(margin + k * (pw + gx)) / W, y0, pw / W, ph / H]) for k in range(4)]
    for ax, title in zip(axs, TITLES):
        ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc="#f7f7f5", ec=m.AXIS, lw=0.6, zorder=0))
        ax.text(0.5, 0.97, title, ha="center", va="top", fontsize=FONT)
    for k in range(3):
        p, q = axs[k].get_position(), axs[k + 1].get_position()
        y = p.y0 + 0.45 * p.height
        fig.add_artist(FancyArrowPatch((p.x1 + 0.006, y), (q.x0 - 0.006, y), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=8, color=m.INK2, lw=0.9))
    ratio = pw / ph

    # (a) the proxy in J blocks, one group held out; every group is a candidate
    a = axs[0]
    s, row, step = 0.122, 0.40, 0.133
    left = 0.5 - ((J - 1) * step + s) / 2
    for j in range(J):
        x0 = left + j * step
        inset = a.inset_axes([x0, row, s, s * ratio])
        inset.imshow(thumbs[j], interpolation="nearest")
        frame(inset, m.BLUE if j == 0 else None, dashed=(j == 0))
        a.text(x0 + s / 2, row - 0.02, "$W_%d$" % (j + 1), ha="center", va="top",
               fontsize=SMALL, color=m.BLUE if j == 0 else m.INK2)
    a.text(left + s / 2, row + s * ratio + 0.02, "held out", ha="center", va="bottom",
           fontsize=SMALL, color=m.BLUE)
    a.text(0.5, 0.05, "$2^%d-2 = %d$ splits in all" % (J, 2 ** J - 2), ha="center", va="bottom",
           fontsize=SMALL, color=m.INK2)

    # (b) the kept blocks and X give Z; phi is chosen to make the held-out group look balanced
    b = axs[1]
    sb, step_b = 0.115, 0.135
    left_b = 0.5 - ((J - 2) * step_b + sb) / 2
    for k in range(J - 1):
        inset = b.inset_axes([left_b + k * step_b, 0.64, sb, sb * ratio])
        inset.imshow(thumbs[k + 1], interpolation="nearest"); frame(inset)
    b.text(0.5, 0.625, "$W_2,\\ldots,W_%d$ and $X$" % J, ha="center", va="top", fontsize=SMALL, color=m.INK2)
    b.add_patch(FancyArrowPatch((0.5, 0.560), (0.5, 0.525), arrowstyle="-|>", mutation_scale=6,
                                color=m.INK2, lw=0.8))
    b.add_patch(Polygon([(0.06, 0.515), (0.94, 0.515), (0.74, 0.285), (0.26, 0.285)], closed=True,
                        fc="#dde9f8", ec=m.BLUE, lw=0.9))
    b.text(0.5, 0.450, "pick $\\phi$ to minimize", ha="center", va="center", fontsize=SMALL)
    b.text(0.5, 0.340, "$\\widetilde D_S^2$", ha="center", va="center", fontsize=FONT + 1)
    b.add_patch(FancyArrowPatch((0.5, 0.275), (0.5, 0.240), arrowstyle="-|>", mutation_scale=6,
                                color=m.INK2, lw=0.8))
    for k in range(3):
        b.add_patch(Rectangle((0.38 + k * 0.07, 0.140), 0.06, 0.09, fc=m.BLUE,
                              alpha=0.3 + 0.2 * k, ec="white", lw=0.5))
    b.text(0.62, 0.185, "$Z$", ha="left", va="center", fontsize=FONT)
    b.text(0.5, 0.04, "never sees the held-out blocks", ha="center", va="bottom",
           fontsize=SMALL, color=m.INK2)

    # (c) the screen statistic of every group against the threshold.  Both named examples are kept and lie
    # 0.15 decade apart inside a tight kept cluster, so their rings sit at the top and bottom edges of the
    # band rather than on the centre line, each over a thin white halo that keeps it visible among the
    # kept dots.  Each label ends at its own ring, so the W4 label stays on the kept side of t.
    xlo, xhi, ylo, yhi = 1.5e-4, 1.5e-1, -1.80, 2.50
    cx = axs[2].inset_axes([0.13, 0.31, 0.82, 0.44])
    rng = np.random.default_rng(11)
    jit = rng.uniform(-0.55, 0.55, size=len(names))
    jit[index[VALID]], jit[index[INVALID]] = 0.42, -0.42
    cx.scatter(stat[~kept], jit[~kept], s=6, color=m.GRAY, alpha=0.55, linewidths=0, zorder=2)
    cx.scatter(stat[kept], jit[kept], s=8, color=m.BLUE, linewidths=0, zorder=3)
    for split, colour in ((VALID, m.BLUE), (INVALID, m.ORANGE)):
        xy = ([stat[index[split]]], [jit[index[split]]])
        cx.scatter(*xy, s=30, facecolors="none", edgecolors="white", linewidths=1.6, zorder=4.5)
        cx.scatter(*xy, s=30, facecolors="none", edgecolors=colour, linewidths=0.9, zorder=5)
    cx.axvline(t, color=m.INK, lw=0.9, ls=(0, (3, 2)), zorder=4)
    cx.set_xscale("log"); cx.set_xlim(xlo, xhi); cx.set_ylim(ylo, yhi)
    cx.set_xticks([1e-3, 1e-1], ["$10^{-3}$", "$10^{-1}$"])
    cx.xaxis.set_minor_locator(NullLocator())      # as in the old panel: decade ticks only
    cx.tick_params(axis="x", labelsize=SMALL, length=2, pad=1.5); bare(cx)
    cx.text(xlo * 1.2, 2.42, "keep %d" % kept.sum(), ha="left", va="top", fontsize=SMALL, color=m.BLUE)
    cx.text(xhi * 0.75, 2.42, "drop %d" % (~kept).sum(), ha="right", va="top", fontsize=SMALL, color=m.GRAY)
    nudge = 10 ** 0.05
    cx.text(stat[index[VALID]] * nudge, 0.80, "$W_1$ out", ha="right", va="bottom", fontsize=SMALL,
            color=m.BLUE)
    cx.text(stat[index[INVALID]] * nudge, -0.80, "$W_4$ out", ha="right", va="top", fontsize=SMALL,
            color=m.ORANGE)
    axs[2].text(0.5, 0.05, "screen statistic; keep below $t$", ha="center", va="bottom",
                fontsize=SMALL, color=m.INK2)

    # (d) one estimate per kept group; the largest agreeing group gives the answer
    dx = axs[3].inset_axes([0.11, 0.31, 0.83, 0.38])
    n_big, n_other = int(in_big.sum()), int((~in_big).sum())
    ry = np.linspace(-0.42, 0.42, n_big)
    wy = np.linspace(-0.18, 0.18, n_other) if n_other > 1 else np.zeros(n_other)
    dx.scatter(est[in_big], ry, s=15, color=m.BLUE, edgecolors="white", linewidths=0.4, zorder=3)
    dx.scatter(est[~in_big], wy, s=15, color=m.ORANGE, edgecolors="white", linewidths=0.4, zorder=3)
    dx.scatter([tau_hat], [2.00], marker="*", s=60, color=m.INK, zorder=4)
    dx.text(tau_hat - 0.06, 2.00, "$\\widehat\\tau$ = %.2f" % tau_hat, ha="right", va="center",
            fontsize=SMALL + 0.5)
    dx.axvline(m.TAU, color=m.INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
    wrong_x = float(est[~in_big].min())
    dx.annotate("", xy=(wrong_x, 0.80), xytext=(wrong_x + 2 * rho, 0.80),
                arrowprops=dict(arrowstyle="<->", color=m.INK2, lw=0.7, shrinkA=0, shrinkB=0))
    dx.text(wrong_x + rho, 0.90, "$2\\rho$", ha="center", va="bottom", fontsize=SMALL, color=m.INK2)
    dx.text(float(est[in_big].max()) + 0.07, 0.0, "%d" % n_big, ha="left", va="center",
            fontsize=SMALL, color=m.BLUE)
    if INVALID in retained:              # the lone orange dot is named rather than counted
        dx.text(est[retained.index(INVALID)] - 0.07, 0.0, "$W_4$ out", ha="right", va="center",
                fontsize=SMALL, color=m.ORANGE)
    dx.set_xlim(0.14, 1.30); dx.set_ylim(-0.60, 2.40)
    dx.set_xticks([0.5, 1.0], ["0.5", "1"])
    dx.tick_params(labelsize=SMALL, length=2, pad=1.5); bare(dx)
    axs[3].text(0.5, 0.05, "one dot per split; dashed $\\tau$", ha="center", va="bottom",
                fontsize=SMALL, color=m.INK2)
    m.save(fig, STEM)

shutil.copyfile(m.FIGDIR / f"{STEM}.pdf", PAPER_FIGDIR / f"{STEM}.pdf")
print(f"  copied the pdf to ICLR/paper/figures/{STEM}.pdf")
