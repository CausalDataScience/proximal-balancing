#!/usr/bin/env python3
"""Mini experiment for NSF CAREER Aim A.2: single-proxy balancing.

The experiment is deliberately small and proposal-facing.  It asks whether a
single pre-treatment proxy W can remove hidden-confounding bias when ordinary
measured covariates X are insufficient.

DGP:
  U: latent severity, unmeasured by the analyst.
  X: weak measured covariate.
  W: proxy for U, with tunable informativeness.
  A: treatment, selected by U and X.
  Y: outcome, with constant treatment effect tau and confounding by U.

Estimator:
  ATT by entropy balancing of controls to treated units.  The proxy method uses
  moments of W; the oracle uses moments of U and serves only as a benchmark.
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


@dataclass(frozen=True)
class SimResult:
    proxy_strength: float
    estimator: str
    estimate: float
    error: float


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_features(values: np.ndarray) -> np.ndarray:
    """Use low-order moments so the figure represents balancing, not ML power."""
    z = np.asarray(values, dtype=float)
    return np.column_stack([z, z**2])


def entropy_balance_weights(control_features: np.ndarray, target_mean: np.ndarray) -> np.ndarray:
    """Return entropy-balancing weights over control units.

    The weights sum to one and approximately match target feature means.  A
    root solve is attempted first; if the moment target is near the convex-hull
    boundary, a smooth least-squares fallback is used.
    """
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
        w = weights(lam)
        return w @ fs - ts

    sol = root(moment_error, np.zeros(fs.shape[1]), method="hybr")
    if sol.success and np.linalg.norm(moment_error(sol.x), ord=2) < 1e-6:
        return weights(sol.x)

    def objective(lam: np.ndarray) -> float:
        err = moment_error(lam)
        return float(err @ err + 1e-4 * (lam @ lam))

    opt = minimize(objective, np.zeros(fs.shape[1]), method="BFGS")
    return weights(opt.x)


def simulate_once(rng: np.random.Generator, n: int, proxy_strength: float) -> dict[str, np.ndarray]:
    u = rng.normal(0.0, 1.0, size=n)
    x = 0.25 * u + rng.normal(0.0, 1.0, size=n)
    proxy_noise_sd = np.sqrt(max(1.0 - proxy_strength**2, 1e-8))
    w = proxy_strength * u + proxy_noise_sd * rng.normal(0.0, 1.0, size=n)
    p_a = sigmoid(-0.1 + 1.15 * u + 0.35 * x)
    a = rng.binomial(1, p_a, size=n)
    y = TRUE_TAU * a + 1.0 * u + 0.30 * x + rng.normal(0.0, 1.0, size=n)
    return {"U": u, "X": x, "W": w, "A": a, "Y": y}


def estimate_att(data: dict[str, np.ndarray], balance_on: str | None) -> float:
    a = data["A"].astype(bool)
    y = data["Y"]
    yt = y[a].mean()
    yc = y[~a]
    if balance_on is None:
        return float(yt - yc.mean())
    ft = make_features(data[balance_on][a])
    fc = make_features(data[balance_on][~a])
    weights = entropy_balance_weights(fc, ft.mean(axis=0))
    return float(yt - weights @ yc)


def run_experiment(n: int = 1800, reps: int = 80, seed: int = 20260705) -> list[SimResult]:
    rng = np.random.default_rng(seed)
    proxy_grid = np.array([0.00, 0.20, 0.40, 0.60, 0.75, 0.90, 0.97])
    estimators = {
        "Naive": None,
        "Balance X only": "X",
        "Single-proxy balance W": "W",
        "Oracle balance U": "U",
    }
    rows: list[SimResult] = []
    for s in proxy_grid:
        for _ in range(reps):
            data = simulate_once(rng, n=n, proxy_strength=float(s))
            # skip rare degenerate treatment splits
            treated = data["A"].sum()
            if treated < 20 or treated > n - 20:
                continue
            for name, covar in estimators.items():
                est = estimate_att(data, covar)
                rows.append(SimResult(float(s), name, est, est - TRUE_TAU))
    return rows


def summarize(rows: list[SimResult]) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    keys = sorted({(r.proxy_strength, r.estimator) for r in rows})
    for s, name in keys:
        vals = np.array([r.error for r in rows if r.proxy_strength == s and r.estimator == name])
        ests = np.array([r.estimate for r in rows if r.proxy_strength == s and r.estimator == name])
        out.append(
            {
                "proxy_strength": s,
                "estimator": name,
                "mean_estimate": float(ests.mean()),
                "bias": float(vals.mean()),
                "rmse": float(np.sqrt(np.mean(vals**2))),
                "mae": float(np.mean(np.abs(vals))),
                "reps": int(vals.size),
            }
        )
    return out


def make_figure(summary: list[dict[str, float | str]]) -> None:
    order = [
        ("Naive", "#6b7280", "o", "naive"),
        ("Balance X only", "#2563eb", "s", "$X$ only"),
        ("Single-proxy balance W", "#059669", "^", "single $W$"),
        ("Oracle balance U", "#dc2626", "D", "oracle $U$"),
    ]
    fig, ax = plt.subplots(figsize=(2.45, 1.78), dpi=260)
    last_points: list[tuple[float, float, str, str]] = []
    for name, color, marker, label in order:
        xs = [float(r["proxy_strength"]) for r in summary if r["estimator"] == name]
        ys = [float(r["rmse"]) for r in summary if r["estimator"] == name]
        ax.plot(xs, ys, marker=marker, linewidth=1.9, markersize=3.6, color=color, label=name)
        last_points.append((xs[-1], ys[-1], color, label))

    ax.set_xlabel("proxy informativeness", labelpad=1.5, fontsize=6.7)
    ax.set_ylabel("effect RMSE", labelpad=1.5, fontsize=6.7)
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(0.0, 1.18)
    ax.grid(True, axis="y", linewidth=0.45, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=6.2, width=0.65, length=2.2, pad=1.5)

    offsets = {
        "naive": 0.00,
        "$X$ only": 0.015,
        "single $W$": 0.075,
        "oracle $U$": -0.030,
    }
    for x, y, color, label in last_points:
        ax.text(
            x + 0.025,
            y + offsets[label],
            label,
            color=color,
            fontsize=5.6,
            va="center",
            ha="left",
        )
    fig.tight_layout(pad=0.28)
    fig.savefig(OUT_DIR / "single_proxy_balancing_wrapfigure.pdf")
    fig.savefig(OUT_DIR / "single_proxy_balancing_wrapfigure.png", dpi=300)


def main() -> None:
    rows = run_experiment()
    summary = summarize(rows)
    (OUT_DIR / "single_proxy_balancing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    make_figure(summary)
    for row in summary:
        if row["proxy_strength"] in {0.0, 0.6, 0.97}:
            print(row)
    print(f"Wrote {OUT_DIR / 'single_proxy_balancing_wrapfigure.pdf'}")
    print(f"Wrote {OUT_DIR / 'single_proxy_balancing_wrapfigure.png'}")
    print(f"Wrote {OUT_DIR / 'single_proxy_balancing_summary.json'}")


if __name__ == "__main__":
    main()
