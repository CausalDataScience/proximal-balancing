#!/usr/bin/env python3
"""DGPs where single-proxy balancing fails but mediator-assisted proxy recovery works.

The construction uses U=(B,C). The single proxy W reveals B but not C. Treatment
A is confounded by C, and Y depends on C, so W-only adjustment remains biased.
The mediator M is generated as A xor C, so the pair (A,M) reveals C. Thus one
external proxy plus the mediator response recovers U=(W, A xor M), enabling a
front-door/g-functional estimator even though W alone is insufficient.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#1f77b4"
RED = "#d62728"
GRAY = "#7f7f7f"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class DGP:
    name: str
    p_b: float
    p_c: float
    a0: float
    a_b: float
    a_c: float
    tau_m: float
    y_b: float
    y_c: float
    noise_sd: float
    story: str


DGPS = [
    DGP(
        name="DGP1_subtype_response",
        p_b=0.45,
        p_c=0.30,
        a0=-0.15,
        a_b=0.45,
        a_c=1.65,
        tau_m=1.30,
        y_b=0.40,
        y_c=1.20,
        noise_sd=0.80,
        story="Clinical response subtype: W captures baseline inflammation B, M reveals hidden responder subtype C through treatment-specific early response.",
    ),
    DGP(
        name="DGP2_adherence_response",
        p_b=0.55,
        p_c=0.38,
        a0=-0.30,
        a_b=0.85,
        a_c=1.25,
        tau_m=1.05,
        y_b=0.70,
        y_c=1.55,
        noise_sd=1.00,
        story="Care management: W captures prior engagement B, M reveals hidden adherence barrier C after the intervention is assigned.",
    ),
    DGP(
        name="DGP3_resistance_response",
        p_b=0.35,
        p_c=0.24,
        a0=0.05,
        a_b=0.35,
        a_c=2.05,
        tau_m=1.70,
        y_b=0.25,
        y_c=0.95,
        noise_sd=0.90,
        story="Treatment resistance: W captures measured severity B, M reveals unmeasured resistance phenotype C via early biomarker movement.",
    ),
]


def iter_states(dgp: DGP) -> Iterable[Dict[str, float | int]]:
    for b in [0, 1]:
        p_b = dgp.p_b if b else 1.0 - dgp.p_b
        for c in [0, 1]:
            p_c = dgp.p_c if c else 1.0 - dgp.p_c
            p_u = p_b * p_c
            p_a1 = float(sigmoid(dgp.a0 + dgp.a_b * (2 * b - 1) + dgp.a_c * (2 * c - 1)))
            for a in [0, 1]:
                p_a = p_a1 if a else 1.0 - p_a1
                m = a ^ c
                w = b
                y_mean = dgp.tau_m * m + dgp.y_b * b + dgp.y_c * c
                yield {
                    "B": b,
                    "C": c,
                    "W": w,
                    "A": a,
                    "M": m,
                    "p": p_u * p_a,
                    "y_mean": y_mean,
                }


def true_psi(dgp: DGP, a: int) -> float:
    total = 0.0
    for b in [0, 1]:
        p_b = dgp.p_b if b else 1.0 - dgp.p_b
        for c in [0, 1]:
            p_c = dgp.p_c if c else 1.0 - dgp.p_c
            m = a ^ c
            y_mean = dgp.tau_m * m + dgp.y_b * b + dgp.y_c * c
            total += p_b * p_c * y_mean
    return float(total)


def exact_cond_mean(dgp: DGP, condition: Dict[str, int]) -> float:
    num = 0.0
    den = 0.0
    for row in iter_states(dgp):
        if all(int(row[k]) == v for k, v in condition.items()):
            num += float(row["p"]) * float(row["y_mean"])
            den += float(row["p"])
    return num / den


def exact_prob(dgp: DGP, condition: Dict[str, int]) -> float:
    return sum(float(row["p"]) for row in iter_states(dgp) if all(int(row[k]) == v for k, v in condition.items()))


def w_only_psi(dgp: DGP, a: int) -> float:
    total = 0.0
    for w in [0, 1]:
        p_w = dgp.p_b if w else 1.0 - dgp.p_b
        total += p_w * exact_cond_mean(dgp, {"A": a, "W": w})
    return float(total)


def mediator_assisted_psi(dgp: DGP, a: int) -> float:
    total = 0.0
    for w in [0, 1]:
        p_w = dgp.p_b if w else 1.0 - dgp.p_b
        for c in [0, 1]:
            p_c = dgp.p_c if c else 1.0 - dgp.p_c
            m = a ^ c
            # C is decoded from observed data by C_hat=A xor M. The conditional
            # mean below is identified because M=m appears when A=m xor C.
            y_mean = dgp.tau_m * m + dgp.y_b * w + dgp.y_c * c
            total += p_w * p_c * y_mean
    return float(total)


def tv_for_discrete(dist0: Dict[Tuple[int, int], float], dist1: Dict[Tuple[int, int], float]) -> float:
    keys = set(dist0) | set(dist1)
    return 0.5 * sum(abs(dist0.get(k, 0.0) - dist1.get(k, 0.0)) for k in keys)


def normalize(counts: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    total = sum(counts.values())
    if total <= 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def residual_imbalance_w(dgp: DGP) -> float:
    score = 0.0
    for w in [0, 1]:
        p_w = dgp.p_b if w else 1.0 - dgp.p_b
        dist = {}
        for a in [0, 1]:
            counts: Dict[Tuple[int, int], float] = {}
            for row in iter_states(dgp):
                if int(row["W"]) == w and int(row["A"]) == a:
                    counts[(int(row["B"]), int(row["C"]))] = counts.get((int(row["B"]), int(row["C"])), 0.0) + float(row["p"])
            dist[a] = normalize(counts)
        score += p_w * tv_for_discrete(dist[0], dist[1])
    return float(score)


def residual_imbalance_mediator_state(dgp: DGP) -> float:
    # R=(W,C_hat) with C_hat=A xor M. In this DGP R exactly recovers U=(B,C),
    # so the weighted residual imbalance is zero up to arithmetic precision.
    score = 0.0
    for w in [0, 1]:
        for c_hat in [0, 1]:
            counts_r = sum(float(row["p"]) for row in iter_states(dgp) if int(row["W"]) == w and (int(row["A"]) ^ int(row["M"])) == c_hat)
            if counts_r == 0:
                continue
            dist = {}
            for a in [0, 1]:
                counts: Dict[Tuple[int, int], float] = {}
                for row in iter_states(dgp):
                    if int(row["A"]) == a and int(row["W"]) == w and (int(row["A"]) ^ int(row["M"])) == c_hat:
                        counts[(int(row["B"]), int(row["C"]))] = counts.get((int(row["B"]), int(row["C"])), 0.0) + float(row["p"])
                dist[a] = normalize(counts)
            score += counts_r * tv_for_discrete(dist[0], dist[1])
    return float(score)


def exact_summary(dgps: Sequence[DGP]) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for dgp in dgps:
        true_ate = true_psi(dgp, 1) - true_psi(dgp, 0)
        w_ate = w_only_psi(dgp, 1) - w_only_psi(dgp, 0)
        med_ate = mediator_assisted_psi(dgp, 1) - mediator_assisted_psi(dgp, 0)
        rows.append(
            {
                "dgp": dgp.name,
                "true_ate": true_ate,
                "w_only_ate": w_ate,
                "mediator_assisted_ate": med_ate,
                "w_only_abs_error": abs(w_ate - true_ate),
                "mediator_abs_error": abs(med_ate - true_ate),
                "w_only_u_imbalance": residual_imbalance_w(dgp),
                "mediator_state_u_imbalance": residual_imbalance_mediator_state(dgp),
                "story": dgp.story,
            }
        )
    return rows


def sample_dgp(dgp: DGP, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    B = (rng.random(n) < dgp.p_b).astype(int)
    C = (rng.random(n) < dgp.p_c).astype(int)
    p_a = sigmoid(dgp.a0 + dgp.a_b * (2 * B - 1) + dgp.a_c * (2 * C - 1))
    A = (rng.random(n) < p_a).astype(int)
    M = A ^ C
    W = B
    y_mean = dgp.tau_m * M + dgp.y_b * B + dgp.y_c * C
    Y = y_mean + rng.normal(0.0, dgp.noise_sd, size=n)
    return {"B": B, "C": C, "W": W, "A": A, "M": M, "Y": Y}


def smoothed_mean(y: np.ndarray, mask: np.ndarray, fallback: float, kappa: float = 3.0) -> float:
    count = int(mask.sum())
    if count == 0:
        return float(fallback)
    return float((y[mask].sum() + kappa * fallback) / (count + kappa))


def estimate_w_only(data: Dict[str, np.ndarray]) -> float:
    A, W, Y = data["A"], data["W"], data["Y"]
    n = len(A)
    global_mean = float(Y.mean())
    psi = {}
    for a in [0, 1]:
        val = 0.0
        for w in [0, 1]:
            mask_w = W == w
            p_w = float(mask_w.mean())
            mu = smoothed_mean(Y, mask_w & (A == a), global_mean)
            val += p_w * mu
        psi[a] = val
    return float(psi[1] - psi[0])


def estimate_mediator_assisted(data: Dict[str, np.ndarray]) -> float:
    A, M, W, Y = data["A"], data["M"], data["W"], data["Y"]
    C_hat = A ^ M
    global_mean = float(Y.mean())
    psi = {}
    for a in [0, 1]:
        val = 0.0
        for w in [0, 1]:
            for c in [0, 1]:
                mask_u = (W == w) & (C_hat == c)
                p_u = float(mask_u.mean())
                m = a ^ c
                # Estimate E[Y | M=m, W=w, C_hat=c]. The data contain this
                # cell through the observed A=m xor c arm.
                mu = smoothed_mean(Y, mask_u & (M == m), global_mean)
                val += p_u * mu
        psi[a] = val
    return float(psi[1] - psi[0])


def run_simulation(dgps: Sequence[DGP], ns: Sequence[int], reps: int, seed: int) -> List[Dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    true_ates = {dgp.name: true_psi(dgp, 1) - true_psi(dgp, 0) for dgp in dgps}
    rows: List[Dict[str, float | int | str]] = []
    for dgp in dgps:
        for n in ns:
            for rep in range(reps):
                data = sample_dgp(dgp, n, rng)
                for method, estimator in [
                    ("W-only proxy balancing", estimate_w_only),
                    ("single proxy + mediator", estimate_mediator_assisted),
                ]:
                    ate = estimator(data)
                    err = ate - true_ates[dgp.name]
                    rows.append(
                        {
                            "dgp": dgp.name,
                            "n": n,
                            "rep": rep,
                            "method": method,
                            "ate": ate,
                            "true_ate": true_ates[dgp.name],
                            "error": err,
                            "abs_error": abs(err),
                            "sq_error": err * err,
                        }
                    )
    return rows


def aggregate(rows: List[Dict[str, float | int | str]], keys: Sequence[str]) -> List[Dict[str, float | int | str]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out: List[Dict[str, float | int | str]] = []
    for key, vals in sorted(groups.items()):
        abs_errors = np.asarray([float(v["abs_error"]) for v in vals])
        sq_errors = np.asarray([float(v["sq_error"]) for v in vals])
        rec = {k: key[i] for i, k in enumerate(keys)}
        rec.update(
            {
                "count": len(vals),
                "mean_abs_error": float(abs_errors.mean()),
                "se_abs_error": float(abs_errors.std(ddof=1) / math.sqrt(len(abs_errors))),
                "rmse": float(math.sqrt(sq_errors.mean())),
            }
        )
        out.append(rec)
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def savefig(fig, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=220)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_exact_bars(rows: List[Dict[str, float | str]], out_base: Path) -> None:
    labels = [str(r["dgp"]).replace("DGP", "") for r in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(x - width, [float(r["true_ate"]) for r in rows], width, color=GRAY, label="true ATE")
    ax.bar(x, [float(r["w_only_ate"]) for r in rows], width, color=RED, label="W-only")
    ax.bar(x + width, [float(r["mediator_assisted_ate"]) for r in rows], width, color=BLUE, label="W + mediator")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("ATE")
    ax.set_title("Single proxy fails; mediator response recovers the target")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    savefig(fig, out_base)


def plot_error_vs_n(summary: List[Dict[str, float | int | str]], out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for method, color, label in [
        ("W-only proxy balancing", RED, "W-only proxy balancing"),
        ("single proxy + mediator", BLUE, "single proxy + mediator"),
    ]:
        vals = [r for r in summary if r["method"] == method]
        xs = np.asarray([int(r["n"]) for r in vals])
        ys = np.asarray([float(r["mean_abs_error"]) for r in vals])
        ses = np.asarray([float(r["se_abs_error"]) for r in vals])
        order = np.argsort(xs)
        xs, ys, ses = xs[order], ys[order], ses[order]
        ax.plot(xs, ys, marker="o", linewidth=2.2, color=color, label=label)
        ax.fill_between(xs, ys - 1.96 * ses, ys + 1.96 * ses, color=color, alpha=0.14)
    ax.set_xscale("log")
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Mean absolute ATE error")
    ax.set_title("Finite-sample error across three DGPs")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, out_base)


def plot_imbalance(rows: List[Dict[str, float | str]], out_base: Path) -> None:
    labels = [str(r["dgp"]).replace("DGP", "") for r in rows]
    x = np.arange(len(rows))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - width / 2, [float(r["w_only_u_imbalance"]) for r in rows], width, color=RED, label="condition on W")
    ax.bar(x + width / 2, [float(r["mediator_state_u_imbalance"]) for r in rows], width, color=BLUE, label=r"condition on $(W,A\oplus M)$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Residual U imbalance across A")
    ax.set_title("Mediator response supplies the missing latent bit")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, out_base)


def plot_mechanism_heatmap(dgp: DGP, out_base: Path) -> None:
    # Show p(C=1 | A,W) for W-only and p(C=1 | A,W,C_hat) after mediator decoding.
    w_only = np.zeros((2, 2))
    for a in [0, 1]:
        for w in [0, 1]:
            den = exact_prob(dgp, {"A": a, "W": w})
            num = sum(float(row["p"]) for row in iter_states(dgp) if int(row["A"]) == a and int(row["W"]) == w and int(row["C"]) == 1)
            w_only[a, w] = num / den
    decoded = np.array([[0.0, 1.0], [0.0, 1.0]])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)
    im0 = axes[0].imshow(w_only, cmap="Reds", vmin=0, vmax=1)
    axes[0].set_title(r"W-only: $P(C=1\mid A,W)$")
    axes[0].set_xlabel("W")
    axes[0].set_ylabel("A")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    for i in [0, 1]:
        for j in [0, 1]:
            axes[0].text(j, i, f"{w_only[i, j]:.2f}", ha="center", va="center", color="black")
    im1 = axes[1].imshow(decoded, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title(r"with mediator: $C=A\oplus M$")
    axes[1].set_xlabel(r"$C_{\rm decoded}$")
    axes[1].set_ylabel("A")
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    for i in [0, 1]:
        for j in [0, 1]:
            axes[1].text(j, i, f"{decoded[i, j]:.0f}", ha="center", va="center", color="black")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    savefig(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="/tmp/paios_single_proxy_mediator_dgps")
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--ns", default="200,500,1000,2000,5000")
    parser.add_argument("--seed", type=int, default=20260706)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ns = [int(x) for x in args.ns.split(",") if x.strip()]

    exact_rows = exact_summary(DGPS)
    sim_rows = run_simulation(DGPS, ns, args.reps, args.seed)
    sim_summary = aggregate(sim_rows, ["n", "method"])
    per_dgp_summary = aggregate(sim_rows, ["dgp", "n", "method"])

    write_csv(outdir / "exact_dgp_summary.csv", exact_rows)
    write_csv(outdir / "simulation_raw.csv", sim_rows)
    write_csv(outdir / "simulation_summary.csv", sim_summary)
    write_csv(outdir / "per_dgp_simulation_summary.csv", per_dgp_summary)
    with (outdir / "metadata.json").open("w") as f:
        json.dump(
            {
                "dgps": [asdict(d) for d in DGPS],
                "ns": ns,
                "reps": args.reps,
                "seed": args.seed,
                "mechanism": "U=(B,C), W=B, M=A xor C, C decoded by A xor M",
            },
            f,
            indent=2,
            sort_keys=True,
        )

    plot_exact_bars(exact_rows, outdir / "single_proxy_vs_mediator_exact_ates")
    plot_error_vs_n(sim_summary, outdir / "single_proxy_vs_mediator_error_vs_n")
    plot_imbalance(exact_rows, outdir / "single_proxy_vs_mediator_u_imbalance")
    plot_mechanism_heatmap(DGPS[0], outdir / "single_proxy_vs_mediator_mechanism_heatmap")

    print(f"outdir={outdir}")
    for row in exact_rows:
        print(
            "{dgp}: true={true_ate:.3f} W-only={w_only_ate:.3f} med={mediator_assisted_ate:.3f} "
            "imb_W={w_only_u_imbalance:.3f} imb_med={mediator_state_u_imbalance:.3f}".format(
                **row
            )
        )
    for row in sim_summary:
        if int(row["n"]) == max(ns):
            print(
                "n={n} method={method} mae={mean_abs_error:.4f} rmse={rmse:.4f}".format(
                    **row
                )
            )


if __name__ == "__main__":
    main()
