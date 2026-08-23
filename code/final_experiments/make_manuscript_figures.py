#!/usr/bin/env python3
"""Create the provenance-matched final manuscript figures.

Only the experiment artifacts allowlisted in ``manifest.json`` are read.  The
script never runs or rewrites an experiment.  It validates the registered
seeds, replication counts, sample-size grids, and key result invariants before
rendering any figure.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_N = [100, 200, 500, 1000, 2000, 5000, 10000]

RESULTS = {
    "scm1_small": "results/honest_canonical_representation_original_scm1_20rep_summary.json",
    "scm1_large": "results/honest_canonical_representation_original_scm1_n2000_5000_10000_20rep_summary.json",
    "scm2_small": "results/honest_canonical_representation_nonlinear_scm2_20rep_summary.json",
    "scm2_large": "results/honest_canonical_representation_nonlinear_scm2_n2000_5000_10000_20rep_summary.json",
    "track_a": "results/exact_injectivity_finite_strata_summary.json",
    "track_b": "results/exact_injectivity_continuous_cci_summary.json",
    "finite_state": "results/finite_state_bridge_summary.json",
    "wsc": "results/wsc_overlap_constrained_followup.json",
}

COLORS = {
    "proposed": "#0072B2",
    "x": "#D55E00",
    "xw": "#009E73",
    "oracle": "#6F4E7C",
    "negative": "#8C8C8C",
    "bias": "#CC79A7",
    "gamma": "#E69F00",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_inputs(root: Path) -> dict[str, dict[str, Any]]:
    loaded = {key: load_json(root / rel) for key, rel in RESULTS.items()}

    for key in ("scm1_small", "scm1_large", "scm2_small", "scm2_large"):
        require(loaded[key]["provenance"]["root_seed"] == 190702, f"{key}: unexpected seed")
        require(loaded[key]["provenance"]["replicates_executed"] == 20,
                f"{key}: expected 20 replications")

    track_a = loaded["track_a"]
    require(track_a["provenance"]["root_seed"] == 190602, "Track A: unexpected seed")
    require(track_a["provenance"]["replicates"] == 20, "Track A: expected 20 replications")
    require(track_a["pipeline_pass"], "Track A: registered pipeline did not pass")

    track_b = loaded["track_b"]
    require(track_b["provenance"]["root_seed"] == 190602, "Track B: unexpected seed")
    require(track_b["all_gates_pass"], "Track B: registered gates did not pass")
    require(track_b["primary_comparison"]["paired_abs_error_improvement_x_minus_learned"]["total"] == 20,
            "Track B: expected 20 paired replications")

    finite = loaded["finite_state"]
    require(finite["R1_exact"]["n_exact"] == 6, "Finite state: expected exactly six exact maps")
    require(finite["R1_exact"]["attainability_ok"], "Finite state: attainability check failed")
    require(finite["R2_approximate"]["violations_common_full"] == 0,
            "Finite state: common certificate violation found")
    require(finite["finite_sample"]["r_n_event_holds_subset_all_reps"],
            "Finite state: finite-sample event failed")

    wsc = loaded["wsc"]
    require(wsc["provenance"]["root_seed"] == 190602, "WSC: unexpected seed")
    require(wsc["provenance"]["algorithmic_repetitions"] == 10,
            "WSC: expected ten algorithmic repetitions")
    require(wsc["provenance"]["monte_carlo_data_generating_repetitions"] == 0,
            "WSC: repetitions must not be labeled Monte Carlo")
    require(wsc["all_integrity_gates_pass"], "WSC: integrity gates failed")

    referenced = "\n".join(RESULTS.values()).lower()
    require("kmu" not in referenced and "zika" not in referenced,
            "Rejected KMU or Zika artifact entered the final figure allowlist")
    return loaded


def merge_rows(*artifacts: dict[str, Any], field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for artifact in artifacts:
        for row in artifact[field]:
            rep = row.get("replicate", row.get("rep"))
            key = (row["n"], row["method"], rep)
            require(key not in seen, f"Duplicate result row: {key}")
            seen.add(key)
            rows.append(row)
    return rows


def mean_ci(rows: list[dict[str, Any]], value_key: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["n"]))].append(float(row[value_key]))
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for method in sorted({key[0] for key in grouped}):
        ns = np.asarray(sorted(n for candidate, n in grouped if candidate == method), dtype=float)
        means, half_widths = [], []
        for n in ns.astype(int):
            values = np.asarray(grouped[(method, n)], dtype=float)
            require(len(values) == 20, f"{method}, n={n}: expected 20 replications")
            means.append(float(values.mean()))
            half_widths.append(float(1.96 * values.std(ddof=1) / math.sqrt(len(values))))
        out[method] = (ns, np.asarray(means), np.asarray(half_widths))
    return out


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def draw_error_curves(
    ax: plt.Axes,
    summaries: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    methods: list[tuple[str, str, str, str]],
    title: str,
) -> None:
    for method, label, color_key, linestyle in methods:
        ns, means, half_widths = summaries[method]
        require(ns.astype(int).tolist() == EXPECTED_N, f"{title}, {method}: unexpected n grid")
        color = COLORS[color_key]
        ax.plot(ns, means, marker="o", markersize=3.2, linewidth=1.5,
                linestyle=linestyle, color=color, label=label)
        ax.fill_between(ns, np.maximum(0.0, means - half_widths), means + half_widths,
                        color=color, alpha=0.12, linewidth=0)
    ax.set_xscale("log")
    ax.set_xticks(EXPECTED_N)
    ax.set_xticklabels(["100", "200", "500", "1k", "2k", "5k", "10k"])
    ax.set_xlabel("Sample size $n$", fontsize=9)
    ax.set_ylabel("Mean absolute ATE error", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    style_axis(ax)


def draw_approx_panel(ax: plt.Axes, data: dict[str, dict[str, Any]], scm: str) -> None:
    small = data[f"{scm}_small"]
    large = data[f"{scm}_large"]
    summaries = mean_ci(merge_rows(small, large, field="aggregate_results"), "absolute_error")
    draw_error_curves(
        ax,
        summaries,
        [
            ("proposed_cross_mask_v1", "Proposed cross-mask", "proposed", "-"),
            ("adjust_x", "$X$ adjustment", "x", "-"),
            ("adjust_xw_full_observed", "Full observed $(X,W)$", "xw", "-"),
            ("observed_xu_fitted_reference", "Oracle $(X,U)$", "oracle", "--"),
            ("heuristic_self_conditioned_v2", "Self-conditioned", "negative", ":"),
        ],
        "SCM1" if scm == "scm1" else "SCM2",
    )


def draw_track_a(ax: plt.Axes, track_a: dict[str, Any]) -> None:
    rows = [{**row, "absolute_error": abs(float(row["error"]))}
            for row in track_a["aggregate_results"]]
    summaries = mean_ci(rows, "absolute_error")
    draw_error_curves(
        ax,
        summaries,
        [
            ("learned_fixed_rep_audit", "Learned representation", "proposed", "-"),
            ("adjust_x", "$X$ adjustment", "x", "-"),
            ("adjust_xw_full_observed", "Full observed $(X,W)$", "xw", "-"),
            ("oracle_h0_stratified", "Oracle score", "oracle", "--"),
        ],
        "Track A: finite exact score",
    )


def draw_track_b(ax: plt.Axes, track_b: dict[str, Any]) -> None:
    summaries = mean_ci(track_b["records"], "abs_error")
    draw_error_curves(
        ax,
        summaries,
        [
            ("learned_continuous_r", "Learned representation", "proposed", "-"),
            ("x_fitted", "$X$ adjustment", "x", "-"),
            ("full_observed_xw_fitted", "Full observed $(X,W)$", "xw", "-"),
            ("oracle_r0_adjustment", "Oracle score", "oracle", "--"),
        ],
        "Track B: continuous exact score",
    )


def draw_paired_wins(ax: plt.Axes, track_a: dict[str, Any], track_b: dict[str, Any]) -> None:
    a_x = next(row for row in track_a["paired_vs_x"] if row["n"] == 10000)
    a_xw = next(row for row in track_a["paired_vs_full_xw"] if row["n"] == 10000)
    b = track_b["primary_comparison"]
    values = [
        a_x["wins"] / 20,
        a_xw["wins"] / 20,
        b["paired_abs_error_improvement_x_minus_learned"]["wins"] / 20,
        b["paired_abs_error_improvement_full_xw_minus_learned"]["wins"] / 20,
    ]
    labels = ["A vs\n$X$", "A vs\n$(X,W)$", "B vs\n$X$", "B vs\n$(X,W)$"]
    colors = [COLORS["proposed"], COLORS["proposed"], COLORS["oracle"], COLORS["oracle"]]
    bars = ax.bar(np.arange(4), values, color=colors, width=0.68)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025,
                f"{round(value * 20):d}/20", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel("Paired win rate at $n=10{,}000$", fontsize=9)
    ax.set_title("Learned method wins", fontsize=10, fontweight="bold")
    style_axis(ax)


def draw_finite_state(ax: plt.Axes, finite: dict[str, Any]) -> None:
    sweep = finite["R4_illposedness"]["sweep"]
    x = np.asarray([row["c1"] for row in sweep], dtype=float)
    min_d = np.asarray([row["min_D"] for row in sweep], dtype=float)
    bias = np.asarray([abs(row["bias_at_argmin"]) for row in sweep], dtype=float)
    gamma = np.asarray([
        np.nan if row["gamma_star_infinite_at_argmin"] else row["max_gamma_star_at_argmin"]
        for row in sweep
    ], dtype=float)
    ax.plot(x, min_d, marker="o", color=COLORS["proposed"], label=r"$\inf_\phi D(\phi)$")
    ax.plot(x, bias, marker="s", color=COLORS["bias"], label=r"$|\mathrm{bias}(\phi^\star)|$")
    ax.set_xlabel("Proxy channel parameter $c_1$", fontsize=9)
    ax.set_ylabel("Discrepancy or absolute bias", fontsize=9)
    ax.invert_xaxis()
    style_axis(ax)
    ax2 = ax.twinx()
    finite_mask = np.isfinite(gamma)
    ax2.plot(x[finite_mask], gamma[finite_mask], marker="^", color=COLORS["gamma"],
             linestyle="--", label=r"$\Gamma^\star$")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"Stability $\Gamma^\star$ (log)", fontsize=8, color=COLORS["gamma"], labelpad=2)
    ax2.tick_params(axis="y", labelsize=8, colors=COLORS["gamma"])
    ax2.spines["top"].set_visible(False)
    ax2.text(0.98, 0.96, r"$\Gamma^\star=\infty$ at $c_1=0.5$",
             transform=ax2.transAxes, ha="right", va="top", fontsize=7.5,
             color=COLORS["gamma"])
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=7, frameon=False)
    ax.set_title("Finite-state certificate", fontsize=10, fontweight="bold")
    ax.text(0.02, 0.04, "6 exact maps; 0 certificate violations",
            transform=ax.transAxes, fontsize=7.5)


def draw_wsc(ax: plt.Axes, wsc: dict[str, Any]) -> None:
    summary = wsc["summary"]
    methods = ["naive_qed", "x_direct", "full_xw_direct", "cross_mask_unconstrained"]
    labels = ["Naive\nQED", "Direct\n$X$", "Direct\n$(X,W)$", "Proposed\ncross-mask"]
    means = np.asarray([summary[key]["estimate_mean"] for key in methods])
    sds = np.asarray([summary[key]["estimate_sd_across_algorithmic_repeats"] for key in methods])
    positions = np.arange(len(methods))
    ax.errorbar(positions, means, yerr=sds, fmt="o", color=COLORS["proposed"],
                ecolor=COLORS["proposed"], capsize=3, markersize=5, linewidth=1.1,
                label=r"Estimate $\pm$ algorithmic-split SD")
    rct = float(wsc["estimand"]["rct_benchmark"])
    rct_se = float(wsc["estimand"]["rct_standard_error_unpooled"])
    ax.axhspan(rct - 1.96 * rct_se, rct + 1.96 * rct_se,
               color=COLORS["negative"], alpha=0.18, label="Noisy RCT 95% reference band")
    ax.axhline(rct, color=COLORS["negative"], linestyle="--", linewidth=1.2)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Estimated mathematics-training ATE", fontsize=9)
    ax.set_title("WSC observational benchmark", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.text(0.02, 0.04, "One fixed data set; no causal ground truth",
            transform=ax.transAxes, fontsize=7.5)
    style_axis(ax)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata={"Creator": "single-proxy-balancing"})
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight",
                metadata={"Software": "single-proxy-balancing"})
    plt.close(fig)


def save_single_panel(draw: Callable[[plt.Axes], None], stem: Path, width: float = 3.4) -> None:
    fig, ax = plt.subplots(figsize=(width, 2.65))
    draw(ax)
    save_figure(fig, stem)


def render_all(data: dict[str, dict[str, Any]], output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.7,
        "legend.handlelength": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    save_single_panel(lambda ax: draw_approx_panel(ax, data, "scm1"), output_dir / "approx_scm1")
    save_single_panel(lambda ax: draw_approx_panel(ax, data, "scm2"), output_dir / "approx_scm2")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    draw_approx_panel(axes[0], data, "scm1")
    draw_approx_panel(axes[1], data, "scm2")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend:
            legend.remove()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.24, wspace=0.28)
    save_figure(fig, output_dir / "figure_approximate_scms")

    save_single_panel(lambda ax: draw_track_a(ax, data["track_a"]), output_dir / "exact_track_a")
    save_single_panel(lambda ax: draw_track_b(ax, data["track_b"]), output_dir / "exact_track_b")
    save_single_panel(lambda ax: draw_paired_wins(ax, data["track_a"], data["track_b"]),
                      output_dir / "exact_paired_wins")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.95))
    draw_track_a(axes[0], data["track_a"])
    draw_track_b(axes[1], data["track_b"])
    draw_paired_wins(axes[2], data["track_a"], data["track_b"])
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[:2]:
        legend = ax.get_legend()
        if legend:
            legend.remove()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.25, wspace=0.34)
    save_figure(fig, output_dir / "figure_exact_tracks")

    save_single_panel(lambda ax: draw_finite_state(ax, data["finite_state"]),
                      output_dir / "finite_state_certificate")
    save_single_panel(lambda ax: draw_wsc(ax, data["wsc"]), output_dir / "wsc_benchmark")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.95))
    draw_finite_state(axes[0], data["finite_state"])
    draw_wsc(axes[1], data["wsc"])
    fig.subplots_adjust(bottom=0.21, wspace=0.52)
    save_figure(fig, output_dir / "figure_certificate_wsc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT,
                        help="Project or standalone package root")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to ROOT/figures/final")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate all registered inputs without writing figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    data = load_inputs(root)
    if not args.check_only:
        render_all(data, (args.output_dir or root / "figures" / "final").resolve())
    print(json.dumps({
        "status": "validated" if args.check_only else "rendered",
        "root": str(root),
        "inputs": RESULTS,
    }, indent=2))


if __name__ == "__main__":
    main()
