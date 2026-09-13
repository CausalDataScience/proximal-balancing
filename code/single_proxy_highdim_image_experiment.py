#!/usr/bin/env python3
"""High-dimensional/image-like W simulation for Single Proxy Balancing.

Question:
  Does the Single Proxy Balancing theory still make sense when the proxy W is
  high-dimensional, like a baseline image?

Answer tested here:
  Yes, if the image representation R = G(W) preserves the latent severity
  mixture.  No, if the image encoder collapses or misses the severity signal.

DGP:
  U: hidden severity.
  X: weak measured covariate.
  W: 16 x 16 baseline image.  A severity template is added with strength rho.
  A: treatment selected by U and X.
  Y: outcome with true effect tau and hidden-confounding by U.

Estimators:
  ATT by entropy balancing controls to treated units.
  - Naive: no balancing.
  - Balance X only: measured covariate only.
  - Severity-preserving image rep: balance the image score in the severity
    direction. This is the simulation stand-in for a learned image encoder that
    preserves the proxy channel required by the theorem.
  - Noise image score: balance a fixed image direction orthogonal to severity.
  - Oracle U: balance the unobserved severity, for reference only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, root


OUT_DIR = Path("/tmp")
TRUE_TAU = 1.0
IMAGE_SIDE = 16
IMAGE_DIM = IMAGE_SIDE * IMAGE_SIDE


@dataclass(frozen=True)
class SimResult:
    image_signal: float
    estimator: str
    estimate: float
    error: float


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_severity_template() -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, IMAGE_SIDE)
    xx, yy = np.meshgrid(grid, grid)
    blob_pos = np.exp(-((xx + 0.35) ** 2 + (yy - 0.25) ** 2) / 0.08)
    blob_neg = np.exp(-((xx - 0.35) ** 2 + (yy + 0.25) ** 2) / 0.10)
    ridge = 0.45 * np.exp(-(yy**2) / 0.035)
    template = blob_pos - 0.80 * blob_neg + ridge
    template = template - template.mean()
    return template.reshape(-1) / np.linalg.norm(template)


SEVERITY_TEMPLATE = make_severity_template()


def fixed_noise_direction() -> np.ndarray:
    rng = np.random.default_rng(271828)
    v = rng.normal(size=IMAGE_DIM)
    v = v - (v @ SEVERITY_TEMPLATE) * SEVERITY_TEMPLATE
    return v / np.linalg.norm(v)


NOISE_DIRECTION = fixed_noise_direction()


def balance_features(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    cols = [z]
    if z.shape[1] <= 4:
        cols.append(z**2)
    return np.column_stack(cols)


def entropy_balance_weights(control_features: np.ndarray, target_mean: np.ndarray) -> np.ndarray:
    f = np.asarray(control_features, dtype=float)
    target = np.asarray(target_mean, dtype=float)
    center = f.mean(axis=0)
    scale = f.std(axis=0)
    scale[scale < 1e-8] = 1.0
    fs = (f - center) / scale
    ts = (target - center) / scale

    def weights(lam: np.ndarray) -> np.ndarray:
        scores = fs @ lam
        scores -= np.max(scores)
        raw = np.exp(scores)
        return raw / raw.sum()

    def moment_error(lam: np.ndarray) -> np.ndarray:
        return weights(lam) @ fs - ts

    sol = root(moment_error, np.zeros(fs.shape[1]), method="hybr")
    if sol.success and np.linalg.norm(moment_error(sol.x), ord=2) < 1e-6:
        return weights(sol.x)

    def objective(lam: np.ndarray) -> float:
        err = moment_error(lam)
        return float(err @ err + 1e-4 * (lam @ lam))

    opt = minimize(objective, np.zeros(fs.shape[1]), method="BFGS")
    return weights(opt.x)


def simulate_once(rng: np.random.Generator, n: int, image_signal: float) -> dict[str, np.ndarray]:
    u = rng.normal(0.0, 1.0, size=n)
    x = 0.20 * u + rng.normal(0.0, 1.0, size=n)

    nuisance = rng.normal(0.0, 1.0, size=(n, IMAGE_DIM))
    common_texture_score = rng.normal(0.0, 1.0, size=n)
    common_texture = fixed_noise_direction()
    # The texture component makes the image high-dimensional and imperfect:
    # an encoder must recover the severity direction, not just any image signal.
    image = (
        image_signal * u[:, None] * SEVERITY_TEMPLATE[None, :]
        + 0.45 * common_texture_score[:, None] * common_texture[None, :]
        + np.sqrt(max(1.0 - image_signal**2, 1e-8)) * nuisance / np.sqrt(IMAGE_DIM)
    )

    p_a = sigmoid(-0.15 + 1.20 * u + 0.35 * x)
    a = rng.binomial(1, p_a, size=n)
    y = TRUE_TAU * a + 1.05 * u + 0.30 * x + rng.normal(0.0, 1.0, size=n)
    return {"U": u, "X": x, "W_img": image, "A": a, "Y": y}


def estimate_att(data: dict[str, np.ndarray], mode: str) -> float:
    a = data["A"].astype(bool)
    y = data["Y"]
    yt = y[a].mean()
    yc = y[~a]
    if mode == "Naive":
        return float(yt - yc.mean())
    if mode == "Balance X only":
        z = data["X"]
    elif mode == "Severity-preserving image rep":
        z = data["W_img"] @ SEVERITY_TEMPLATE
    elif mode == "Noise image score":
        z = data["W_img"] @ NOISE_DIRECTION
    elif mode == "Oracle balance U":
        z = data["U"]
    else:
        raise ValueError(mode)
    ft = balance_features(z[a])
    fc = balance_features(z[~a])
    weights = entropy_balance_weights(fc, ft.mean(axis=0))
    return float(yt - weights @ yc)


def run_experiment(n: int = 2200, reps: int = 60, seed: int = 20260706) -> list[SimResult]:
    rng = np.random.default_rng(seed)
    signal_grid = np.array([0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90])
    estimators = [
        "Naive",
        "Balance X only",
        "Noise image score",
        "Severity-preserving image rep",
        "Oracle balance U",
    ]
    rows: list[SimResult] = []
    for signal in signal_grid:
        for _ in range(reps):
            data = simulate_once(rng, n=n, image_signal=float(signal))
            treated = int(data["A"].sum())
            if treated < 30 or treated > n - 30:
                continue
            for estimator in estimators:
                est = estimate_att(data, estimator)
                rows.append(SimResult(float(signal), estimator, est, est - TRUE_TAU))
    return rows


def summarize(rows: list[SimResult]) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    keys = sorted({(r.image_signal, r.estimator) for r in rows})
    for signal, estimator in keys:
        errors = np.array([r.error for r in rows if r.image_signal == signal and r.estimator == estimator])
        estimates = np.array([r.estimate for r in rows if r.image_signal == signal and r.estimator == estimator])
        out.append(
            {
                "image_signal": signal,
                "estimator": estimator,
                "mean_estimate": float(estimates.mean()),
                "bias": float(errors.mean()),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mae": float(np.mean(np.abs(errors))),
                "reps": int(errors.size),
            }
        )
    return out


def make_figure(summary: list[dict[str, float | str]]) -> None:
    order = [
        ("Naive", "#6b7280", "o", "naive"),
        ("Balance X only", "#2563eb", "s", "$X$ only"),
        ("Noise image score", "#a855f7", "v", "wrong image score"),
        ("Severity-preserving image rep", "#047857", "^", "image proxy rep"),
        ("Oracle balance U", "#dc2626", "D", "oracle $U$"),
    ]
    fig, ax = plt.subplots(figsize=(3.15, 2.10), dpi=260)
    last_points: list[tuple[float, float, str, str]] = []
    for estimator, color, marker, label in order:
        xs = [float(r["image_signal"]) for r in summary if r["estimator"] == estimator]
        ys = [float(r["rmse"]) for r in summary if r["estimator"] == estimator]
        ax.plot(xs, ys, marker=marker, linewidth=1.75, markersize=3.3, color=color)
        last_points.append((xs[-1], ys[-1], color, label))
    ax.set_xlabel("image proxy signal", labelpad=1.5, fontsize=7.0)
    ax.set_ylabel("effect RMSE", labelpad=1.5, fontsize=7.0)
    ax.set_xlim(-0.02, 1.18)
    ax.set_ylim(0.0, 1.28)
    ax.grid(True, axis="y", linewidth=0.45, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=6.2, width=0.65, length=2.2, pad=1.5)
    offsets = {
        "naive": 0.070,
        "$X$ only": -0.030,
        "wrong image score": -0.035,
        "image proxy rep": 0.055,
        "oracle $U$": -0.040,
    }
    for x, y, color, label in last_points:
        ax.text(
            x + 0.025,
            y + offsets[label],
            label,
            color=color,
            fontsize=5.4,
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.3},
        )
    fig.tight_layout(pad=0.30)
    fig.savefig(OUT_DIR / "single_proxy_highdim_image_wrapfigure.pdf")
    fig.savefig(OUT_DIR / "single_proxy_highdim_image_wrapfigure.png", dpi=300)


def save_example_images() -> None:
    rng = np.random.default_rng(777)
    nuisance = rng.normal(0.0, 1.0, size=IMAGE_DIM)
    texture_score = 0.30
    hidden_severity = 1.35

    def demo_image(signal: float) -> np.ndarray:
        return (
            signal * hidden_severity * SEVERITY_TEMPLATE
            + 0.45 * texture_score * NOISE_DIRECTION
            + np.sqrt(max(1.0 - signal**2, 1e-8)) * nuisance / np.sqrt(IMAGE_DIM)
        )

    weak = demo_image(0.10)
    strong = demo_image(0.90)
    fig, axes = plt.subplots(1, 3, figsize=(3.9, 1.25), dpi=220)
    images = [
        (SEVERITY_TEMPLATE.reshape(IMAGE_SIDE, IMAGE_SIDE), "severity\npattern"),
        (weak.reshape(IMAGE_SIDE, IMAGE_SIDE), "weak\nproxy"),
        (strong.reshape(IMAGE_SIDE, IMAGE_SIDE), "strong\nproxy"),
    ]
    for ax, (img, title) in zip(axes, images):
        lim = float(np.max(np.abs(img)))
        ax.imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=6.5, pad=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
    fig.tight_layout(pad=0.20)
    fig.savefig(OUT_DIR / "single_proxy_highdim_image_examples.png", dpi=300)


def main() -> None:
    rows = run_experiment()
    summary = summarize(rows)
    (OUT_DIR / "single_proxy_highdim_image_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    make_figure(summary)
    save_example_images()
    for row in summary:
        if row["image_signal"] in {0.0, 0.45, 0.9}:
            print(row)
    print(f"Wrote {OUT_DIR / 'single_proxy_highdim_image_wrapfigure.pdf'}")
    print(f"Wrote {OUT_DIR / 'single_proxy_highdim_image_wrapfigure.png'}")
    print(f"Wrote {OUT_DIR / 'single_proxy_highdim_image_examples.png'}")
    print(f"Wrote {OUT_DIR / 'single_proxy_highdim_image_summary.json'}")


if __name__ == "__main__":
    main()
