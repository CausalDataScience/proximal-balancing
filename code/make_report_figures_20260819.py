#!/usr/bin/env python3
"""Figures for the 2026-08-19 experiment set, drawn from saved result JSONs."""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "results"
F = ROOT / "figures"
COL = {"joint": "#0f766e", "constrained": "#7c3aed", "x_only": "#d97706",
       "park": "#dc2626"}
ARMS = ("joint", "constrained", "x_only")


def load(name):
    return json.load(open(R / name))


def fig_dimsweep():
    d = load("benchmark_suite_dimsweep.json")
    dims = [5, 10, 20, 30, 40, 50, 100]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, lam in zip(axes, ("0.0", "0.5", "1.0")):
        curves = d["per_lambda"][lam]["curves"]
        for arm in ARMS:
            ys = [curves[str(k)][arm]["mean_abs_bias"] for k in dims]
            ax.plot(dims, ys, "o-", color=COL[arm], label=arm)
        ax.set_xscale("log")
        ax.set_xticks(dims)
        ax.set_xticklabels(dims)
        ax.set_title(f"$\\lambda_f={lam}$")
        ax.set_xlabel("covariate dimension $d$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean |oracle bias|")
    axes[0].legend()
    fig.suptitle("Dimension sweep (registration v4): 6 reps per cell, n=1200")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(F / f"report_20260819_dimsweep.{ext}", dpi=200)
    plt.close(fig)


def fig_directx():
    t = load("benchmark_suite_tabular_v2_directx.json")
    i = load("benchmark_suite_image_v2_directx.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, d, title, lams in (
            (axes[0], t, "Tabular direct-X (36 pairs/level)",
             ("0.0", "0.5", "1.0")),
            (axes[1], i, "Image direct-X (4 pairs/level)",
             ("0.0", "0.5", "1.0"))):
        xs = [float(v) for v in lams]
        for arm in ARMS:
            ys = [d["per_lambda"][v]["aggregates"][arm]["mean_abs_bias"]
                  for v in lams]
            ax.plot(xs, ys, "o-", color=COL[arm], label=arm)
        ax.set_xlabel("direct-effect strength $\\lambda_f$")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean |oracle bias|")
    axes[0].legend()
    fig.suptitle("Direct X-to-Y sweep (registration v3)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(F / f"report_20260819_directx.{ext}", dpi=200)
    plt.close(fig)


def fig_suites():
    t = load("benchmark_suite_tabular_summary.json")
    i = load("benchmark_suite_image_summary.json")
    labels = ["ACIC", "scm_dense", "scm_mixed", "digits", "lesions"]
    vals = {arm: [] for arm in ARMS}
    for src in ("acic", "scm_dense", "scm_mixed"):
        for arm in ARMS:
            vals[arm].append(
                t["per_source"][src]["aggregates"][arm]["mean_abs_bias"])
    for dom in ("digits", "lesions"):
        for arm in ARMS:
            vals[arm].append(
                i["per_domain"][dom]["aggregates"][arm]["mean_abs_bias"])
    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, arm in enumerate(ARMS):
        ax.bar(x + (k - 1) * w, vals[arm], w, color=COL[arm], label=arm)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean |oracle bias|")
    ax.set_title("First benchmark suites (2026-08-19): 5 input distributions")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(F / f"report_20260819_suites.{ext}", dpi=200)
    plt.close(fig)


def fig_external_e1():
    pk = load("external_park_crossworld_B20.json")
    ds = load("external_dsprites_image_B20.json")
    e1 = load("rerun_e1_constrained_hard.json")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))

    ests = ("park", "joint", "constrained", "x_only")
    x = np.arange(2)
    w = 0.2
    for k, est in enumerate(ests):
        ys = [pk[wld]["aggregates"][est]["mean_abs_bias"]
              for wld in ("park", "ours")]
        axes[0].bar(x + (k - 1.5) * w, ys, w, color=COL[est], label=est)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Park world", "our world"])
    axes[0].set_title("Park cross-world 2x2 (B=20)")
    axes[0].set_ylabel("mean |ETT bias|")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    x = np.arange(2)
    w = 0.26
    for k, arm in enumerate(ARMS):
        ys = [ds["per_lambda"][v]["aggregates"][arm]["mean_abs_bias"]
              for v in ("0.0", "1.0")]
        axes[1].bar(x + (k - 1) * w, ys, w, color=COL[arm], label=arm)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["$\\lambda_f=0$", "$\\lambda_f=1$"])
    axes[1].set_title("dSprites-as-X (B=20)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    srcs = ("acic", "scm_dense", "scm_mixed")
    x = np.arange(3)
    w = 0.35
    hard = [e1["per_source"][s]["mean_abs_bias_hard"] for s in srcs]
    xo = [e1["per_source"][s]["mean_abs_bias_xonly"] for s in srcs]
    axes[2].bar(x - w / 2, hard, w, color=COL["constrained"],
                label="constrained (hard)")
    axes[2].bar(x + w / 2, xo, w, color=COL["x_only"], label="x_only")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(srcs)
    axes[2].set_title("E1 hard constraint (feasibility 72/72)")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("External benchmarks and hard-constraint rerun (2026-08-19)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(F / f"report_20260819_external.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    F.mkdir(exist_ok=True)
    fig_dimsweep()
    fig_directx()
    fig_suites()
    fig_external_e1()
    print("figures written")
