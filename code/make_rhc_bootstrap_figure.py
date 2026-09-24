"""E3 with 20 bootstrap points per method (results/rhc_bootstrap/, rhc_bootstrap.py).

A new figure beside the frozen `probe-rhc.pdf`, which is not touched.  Each small dot is the method's
estimate on one resample of the 5,735 patients; the black tick is their mean; the open circle is the
estimate on the data as observed.  The two bottom rows are the values Cui et al. published, with their
reported standard errors, drawn as points with 95% bars because they cannot be resampled here.  RHC has no
known answer, so there is no reference line.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import make_section5_figures as m
import make_section5_v2_figures as v2

BOOT = m.RESULTS / "rhc_bootstrap"
ROWS = [("PROBE, no roles given", "probe", m.BLUE),
        ("Proximal 2SLS,\nroles of Cui et al.", "p2sls_cui_roles", v2.NEEDS_ROLES),
        ("Adjust for $X$\nand all ten proxies", "adjust_X_and_all_W", v2.NO_AUDIT),
        ("Adjust for $X$", "adjust_X", v2.NO_AUDIT),
        ("$E[Y|A{=}1]-E[Y|A{=}0]$", "naive", v2.NO_AUDIT)]
PUBLISHED = [("Cui et al., standard DR\n(published)", "standard_dr"),
             ("Cui et al., proximal DR\n(published)", "proximal_dr")]


def observed() -> dict[str, float]:
    import family_v2_competitors as fc
    import rhc_data as rd
    import run_rhc as rr
    import run_two_proxy as rtp
    view = rd.load()["view"]
    a, y = view["A"], view["Y"]
    probe = json.loads((m.RESULTS / "noise_scaled_v3.json").read_text())["rhc"]["output"]
    return {"probe": probe, "p2sls_cui_roles": float(fc.p2sls(view, "W1", "W2")),
            "adjust_X_and_all_W": rtp.baselines(view, rd.SEED, 5)["raw"],
            "adjust_X": rr.x_only_aipw(view, rd.SEED)["estimate"],
            "naive": float(y[a == 1].mean() - y[a == 0].mean())}


def main() -> int:
    import rhc_data as rd
    boots = [json.loads(p.read_text()) for p in sorted(BOOT.glob("boot_*.json"))]
    if len(boots) != 20:
        raise SystemExit(f"{len(boots)} resamples found, 20 expected")
    obs = observed()
    W, H = 3.1, 2.9
    xlim = (-3.2, 0.2)
    rows = ROWS + [(label, key, m.INK2) for label, key in PUBLISHED]
    with plt.rc_context({"font.size": 7.0, "xtick.labelsize": 6.6, "ytick.labelsize": 6.4, "savefig.bbox": "standard"}):
        fig, ax = plt.subplots(figsize=(W, H))
        for i, (label, key, colour) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.axhline(y, color=m.HAIR, lw=0.4, zorder=0)
            if key in rd.PUBLISHED:
                est, se = rd.PUBLISHED[key]
                ax.plot([est - 1.96 * se, est + 1.96 * se], [y, y], color=colour, lw=1.0, zorder=3)
                ax.scatter([est], [y], marker="D", s=16, color=colour, zorder=4)
                continue
            vals = np.array([(b["probe_v3"]["output"] if key == "probe" else b["comparators"][key]) for b in boots],
                            dtype=float)
            vals = vals[np.isfinite(vals)]
            m.dots(ax, vals, y, colour, size=7)
            ax.scatter([obs[key]], [y], s=26, facecolors="white", edgecolors=m.INK, linewidths=0.9, zorder=5)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in reversed(rows)], linespacing=1.05)
        for tick, (_, _, colour) in zip(ax.get_yticklabels(), reversed(rows)):
            tick.set_color(colour)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("effect on days survived in 30", fontsize=6.8)
        ax.set_title("RHC, 20 bootstrap resamples", fontsize=7.2)
        fig.text(0.04, 0.015, "dots: bootstrap resamples; black tick: their mean;\n"
                              "open circle: the data as observed; diamonds: published\n"
                              "by Cui et al., with 95% bars.  RHC has no known answer.",
                 ha="left", va="bottom", fontsize=5.8, color=m.INK2, linespacing=1.25)
        ax.grid(False)
        fig.subplots_adjust(left=0.44, right=0.97, top=0.91, bottom=0.30)
        m.save(fig, "rhc-bootstrap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
