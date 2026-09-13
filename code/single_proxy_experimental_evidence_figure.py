#!/usr/bin/env python3
"""Create a graph-plus-experiment evidence figure from saved summaries.

The first panel draws the causal graph used to explain the image proxy setting:
X -> A -> Y, U -> X/A/Y, and X -> W.  The W proxy box contains a small image
thumbnail from the saved image-proxy experiment assets.  The second panel plots
the image-proxy Monte Carlo RMSE curves from ../results.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
CODE_DIR = PROJECT_ROOT / "code"

IMAGE_RESULT = RESULTS_DIR / "single_proxy_highdim_image_summary.json"
IMAGE_EXAMPLE = FIGURES_DIR / "single_proxy_highdim_image_examples.png"
PLOT_DATA = RESULTS_DIR / "single_proxy_experimental_evidence_plot_data.csv"
METADATA = RESULTS_DIR / "single_proxy_experimental_evidence_metadata.json"
OUT_PNG = FIGURES_DIR / "single_proxy_experimental_evidence_square.png"
OUT_PDF = FIGURES_DIR / "single_proxy_experimental_evidence_square.pdf"
CAUSAL_PNG = FIGURES_DIR / "single_proxy_causal_diagram.png"
CAUSAL_PDF = FIGURES_DIR / "single_proxy_causal_diagram.pdf"
EXPERIMENT_PNG = FIGURES_DIR / "single_proxy_image_proxy_experiment_plot.png"
EXPERIMENT_PDF = FIGURES_DIR / "single_proxy_image_proxy_experiment_plot.pdf"

DISPLAYED_IMAGE_ESTIMATORS = {
    "Balance X only",
    "Severity-preserving image rep",
}


def read_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records_for_plot(
    summary: list[dict[str, Any]],
    experiment: str,
    x_key: str,
    x_label: str,
    displayed_estimators: set[str],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for row in summary:
        if row["estimator"] not in displayed_estimators:
            continue
        records.append(
            {
                "experiment": experiment,
                "x_key": x_key,
                "x_label": x_label,
                "x_value": f"{float(row[x_key]):.6g}",
                "estimator": str(row["estimator"]),
                "mean_estimate": f"{float(row['mean_estimate']):.12g}",
                "bias": f"{float(row['bias']):.12g}",
                "rmse": f"{float(row['rmse']):.12g}",
                "mae": f"{float(row['mae']):.12g}",
                "reps": str(int(row["reps"])),
            }
        )
    return records


def write_plot_data(image: list[dict[str, Any]]) -> None:
    records = records_for_plot(
        image,
        "image_proxy",
        "image_signal",
        "severity signal in image W",
        DISPLAYED_IMAGE_ESTIMATORS,
    )
    fieldnames = [
        "experiment",
        "x_key",
        "x_label",
        "x_value",
        "estimator",
        "mean_estimate",
        "bias",
        "rmse",
        "mae",
        "reps",
    ]
    with PLOT_DATA.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_metadata(image: list[dict[str, Any]]) -> None:
    payload = {
        "combined_figure_png": str(OUT_PNG.relative_to(PROJECT_ROOT)),
        "combined_figure_pdf": str(OUT_PDF.relative_to(PROJECT_ROOT)),
        "causal_diagram_png": str(CAUSAL_PNG.relative_to(PROJECT_ROOT)),
        "causal_diagram_pdf": str(CAUSAL_PDF.relative_to(PROJECT_ROOT)),
        "experiment_plot_png": str(EXPERIMENT_PNG.relative_to(PROJECT_ROOT)),
        "experiment_plot_pdf": str(EXPERIMENT_PDF.relative_to(PROJECT_ROOT)),
        "plot_data": str(PLOT_DATA.relative_to(PROJECT_ROOT)),
        "figure_design": [
            "Causal diagram: X -> A -> Y, U -> X/A/Y, X -> W proxy.",
            "Experiment plot: image-proxy experiment only; naive and irrelevant-image curves are omitted.",
        ],
        "displayed_estimators": sorted(DISPLAYED_IMAGE_ESTIMATORS),
        "generated_from": [
            {
                "path": str(IMAGE_RESULT.relative_to(PROJECT_ROOT)),
                "sha256": sha256(IMAGE_RESULT),
                "rows": len(image),
                "experiment": "16x16 image proxy W",
                "n": 2200,
                "reps_per_grid_value": sorted({int(row["reps"]) for row in image}),
            },
        ],
        "visual_asset": {
            "path": str(IMAGE_EXAMPLE.relative_to(PROJECT_ROOT)),
            "sha256": sha256(IMAGE_EXAMPLE),
            "usage": "thumbnail inside the W proxy node",
        },
        "source_code": [
            {
                "path": str((CODE_DIR / "single_proxy_highdim_image_experiment.py").relative_to(PROJECT_ROOT)),
                "sha256": sha256(CODE_DIR / "single_proxy_highdim_image_experiment.py"),
            },
            {
                "path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
                "sha256": sha256(Path(__file__).resolve()),
            },
        ],
        "note": (
            "This is a deterministic Matplotlib figure. The graph panel is a schematic "
            "of the DGP; the experiment panel is generated from the saved image-proxy "
            "Monte Carlo summary JSON. It does not use image generation."
        ),
    }
    METADATA.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def series(summary: list[dict[str, Any]], x_key: str, estimator: str) -> tuple[list[float], list[float]]:
    rows = [row for row in summary if row["estimator"] == estimator]
    rows.sort(key=lambda row: float(row[x_key]))
    return [float(row[x_key]) for row in rows], [float(row["rmse"]) for row in rows]


def crop_proxy_thumbnail() -> Any:
    img = mpimg.imread(IMAGE_EXAMPLE)
    height, width = img.shape[:2]
    # Crop the rightmost "strong proxy" square from the saved experiment image.
    y0 = int(0.17 * height)
    y1 = int(0.98 * height)
    x0 = int(0.70 * width)
    x1 = int(0.96 * width)
    return img[y0:y1, x0:x1]


def draw_node(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    edgecolor: str = "#111827",
    facecolor: str = "white",
    fontsize: float = 13.5,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.45,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize)


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#111827",
    rad: float = 0.0,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "linewidth": 1.45,
            "color": color,
            "shrinkA": 6,
            "shrinkB": 6,
            "connectionstyle": f"arc3,rad={rad}",
        },
    )


def draw_graph_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.96, "Image 1. data-generating graph", fontsize=9.2, fontweight="bold", va="top")

    u = (0.22, 0.78)
    x = (0.22, 0.47)
    a = (0.53, 0.47)
    y = (0.82, 0.47)
    w = (0.22, 0.17)

    draw_node(ax, u, 0.14, 0.12, "$U$", edgecolor="#6b7280", facecolor="#f3f4f6")
    draw_node(ax, x, 0.14, 0.12, "$X$", edgecolor="#2563eb", facecolor="#eff6ff")
    draw_node(ax, a, 0.14, 0.12, "$A$", edgecolor="#111827", facecolor="white")
    draw_node(ax, y, 0.14, 0.12, "$Y$", edgecolor="#dc2626", facecolor="#fef2f2")

    w_box = FancyBboxPatch(
        (0.075, 0.045),
        0.29,
        0.22,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.45,
        edgecolor="#047857",
        facecolor="#ecfdf5",
    )
    ax.add_patch(w_box)
    ax.text(0.132, 0.215, "$W$ proxy", fontsize=10.5, ha="left", va="center", color="#065f46")
    ax.text(0.255, 0.102, "image", fontsize=8.0, ha="center", va="center", color="#065f46")
    inset = ax.inset_axes([0.112, 0.068, 0.105, 0.130])
    inset.imshow(crop_proxy_thumbnail())
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#065f46")

    draw_arrow(ax, (0.29, 0.47), (0.46, 0.47))  # X -> A
    draw_arrow(ax, (0.60, 0.47), (0.75, 0.47))  # A -> Y
    draw_arrow(ax, (0.22, 0.72), (0.22, 0.54), "#6b7280")  # U -> X
    draw_arrow(ax, (0.285, 0.755), (0.48, 0.53), "#6b7280", rad=-0.05)  # U -> A
    draw_arrow(ax, (0.30, 0.80), (0.78, 0.54), "#6b7280", rad=-0.12)  # U -> Y
    draw_arrow(ax, (0.22, 0.40), (0.22, 0.275), "#047857")  # X -> W


def make_causal_diagram() -> None:
    fig, ax = plt.subplots(figsize=(4.75, 1.85), dpi=320)
    draw_graph_panel(ax)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.93, bottom=0.035)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CAUSAL_PNG, dpi=320)
    fig.savefig(CAUSAL_PDF)
    plt.close(fig)


def plot_image_panel(
    ax: plt.Axes,
    summary: list[dict[str, Any]],
    order: list[tuple[str, str, str, str]],
    label_offsets: dict[str, float],
) -> None:
    for estimator, color, marker, label in order:
        xs, ys = series(summary, "image_signal", estimator)
        ax.plot(xs, ys, marker=marker, linewidth=2.8, markersize=6.5, color=color, label=label)
    ax.set_xlabel(r"$W$ informativeness", labelpad=3, fontsize=18.5)
    ax.set_ylabel("RMSE", labelpad=4, fontsize=18.5)
    ax.yaxis.set_label_coords(-0.09, 0.50)
    ax.set_xlim(-0.025, 0.925)
    ax.set_ylim(0.0, 1.20)
    ax.grid(True, axis="y", linewidth=0.75, alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12.5, width=0.90, length=3.3, pad=1.6)
    ax.legend(
        loc="upper right",
        fontsize=7.6,
        frameon=True,
        framealpha=0.84,
        borderpad=0.25,
        labelspacing=0.22,
        handlelength=0.95,
        handletextpad=0.34,
        markerscale=0.78,
    )


def make_experiment_plot(image: list[dict[str, Any]]) -> None:
    image_order = [
        ("Balance X only", "#2563eb", "s", "$X$ only"),
        ("Severity-preserving image rep", "#047857", "^", "image proxy"),
    ]
    fig, ax = plt.subplots(figsize=(4.75, 2.18), dpi=320)
    plot_image_panel(
        ax,
        image,
        image_order,
        {
            "$X$ only": -0.035,
            "image proxy": 0.050,
        },
    )
    fig.subplots_adjust(left=0.27, right=0.985, top=0.88, bottom=0.40)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(EXPERIMENT_PNG, dpi=320, bbox_inches="tight", pad_inches=0.055)
    fig.savefig(EXPERIMENT_PDF, bbox_inches="tight", pad_inches=0.055)
    plt.close(fig)


def make_figure(image: list[dict[str, Any]]) -> None:
    image_order = [
        ("Balance X only", "#2563eb", "s", "$X$ only"),
        ("Severity-preserving image rep", "#047857", "^", "image proxy"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(4.75, 4.75), dpi=320, height_ratios=[0.90, 1.10])
    draw_graph_panel(axes[0])
    plot_image_panel(
        axes[1],
        image,
        image_order,
        {
            "$X$ only": -0.035,
            "image proxy": 0.050,
        },
    )
    fig.suptitle("Single Proxy Balancing with an image proxy", fontsize=10.8, y=0.982)
    fig.text(
        0.50,
        0.946,
        "Causal graph plus Monte Carlo evidence from the image-proxy experiment",
        ha="center",
        va="center",
        fontsize=7.2,
    )
    fig.text(
        0.50,
        0.018,
        "Saved summary: image n=2200, 60 reps/grid. Lower RMSE is better.",
        ha="center",
        va="center",
        fontsize=6.6,
    )
    fig.subplots_adjust(left=0.115, right=0.81, top=0.900, bottom=0.085, hspace=0.35)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=320)
    fig.savefig(OUT_PDF)
    plt.close(fig)


def main() -> None:
    image = read_json(IMAGE_RESULT)
    write_plot_data(image)
    make_causal_diagram()
    make_experiment_plot(image)
    make_figure(image)
    write_metadata(image)
    print(f"Wrote {CAUSAL_PNG}")
    print(f"Wrote {CAUSAL_PDF}")
    print(f"Wrote {EXPERIMENT_PNG}")
    print(f"Wrote {EXPERIMENT_PDF}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {PLOT_DATA}")
    print(f"Wrote {METADATA}")


if __name__ == "__main__":
    main()
