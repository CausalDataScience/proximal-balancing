#!/usr/bin/env python3
"""E4: MNIST image proxies of a finite latent confounder (plan: memo/2026-09-09-e4-mnist-proxy-plan.md).

One unit: latent digit U ~ Unif{0..9}, score s(U) = (U - 4.5)/2.87; four proxy blocks W_1..W_4 = four distinct
MNIST training images of digit U, occluded by a random o x o square and perturbed by N(0, sigma^2) pixel noise;
X = b_x s(U) + eps_X in R^{d_x}; P(A=1|U,X) = c + (1-2c) expit(1.2 s(U) + 0.6 a_x'X); Y = A + 0.7 y_x'X +
0.3 (yt_x'X)^2 - 3 s(U) + eps_Y (tau = 1, Simpson configuration of the tabular SCMs).

Stage 1 (this script): every image enters the pipeline through an unsupervised feature map psi = the first
`pcs` principal components of the degraded pixels, fitted on the discrepancy sample of the replicate.  PROBE's
encoder, the critics, the raw-image adjustment baseline, and the audit all see psi(image) only, so this stage
is the tabular pipeline with W_j = psi(W_j).  Stage 2 (end-to-end CNN encoder) is a separate script.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

import probe_scms as scms
import probe_e1_pipeline as P
from probe_e1_pipeline import (TRUE_ATE, aggregate, all_splits, baseline_aipw, linking_radius, rng, run_split, seed_key, summarize, torch_seed)

ROOT = Path(__file__).resolve().parents[1]
MNIST = ROOT / "materials" / "benchmarks" / "mnist" / "mnist.npz"
RESULTS_DIR = ROOT / "results"
J = 4


def load_pool():
    d = np.load(MNIST)
    x = d["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
    y = d["y_train"].astype(int)
    return [x[y == k] for k in range(10)]


def degrade(img: np.ndarray, sigma: float, occ: int, g: np.random.Generator) -> np.ndarray:
    out = img.copy()
    if occ > 0:
        n = len(out); im = out.reshape(n, 28, 28)
        r = g.integers(0, 28 - occ + 1, size=n); c = g.integers(0, 28 - occ + 1, size=n)
        for i in range(n):
            im[i, r[i]:r[i] + occ, c[i]:c[i] + occ] = 0.0
        out = im.reshape(n, 784)
    if sigma > 0:
        out = out + sigma * g.standard_normal(out.shape).astype(np.float32)
    return out


def generate(pool, n: int, key: tuple[int, ...], coef: dict, sigma: float, occ: int, d_x: int, prop_floor: float = 0.05):
    ru, rx, rw, ra, ry, rimg = [np.random.default_rng(c) for c in np.random.SeedSequence(key).spawn(6)]
    u_cat = ru.integers(0, 10, size=n); s = (u_cat - 4.5) / 2.87
    x = s[:, None] * coef["b_x"] + rx.standard_normal((n, d_x))
    images = np.zeros((n, J, 784), dtype=np.float32)
    for k in range(10):
        idx = np.where(u_cat == k)[0]
        if len(idx) == 0:
            continue
        # J distinct images per unit, drawn without replacement within the unit
        picks = np.stack([rimg.choice(len(pool[k]), size=J, replace=False) for _ in idx])
        images[idx] = pool[k][picks]
    images = degrade(images.reshape(n * J, 784), sigma, occ, rw).reshape(n, J, 784)
    index = 1.2 * s + 0.6 * (x @ coef["a_x"])
    propensity = prop_floor + (1 - 2 * prop_floor) * scms.expit(index)
    a = (ra.uniform(size=n) < propensity).astype(float)
    y0 = 0.7 * (x @ coef["y_x"]) + 0.3 * (x @ coef["yt_x"]) ** 2 - 3.0 * s + ry.standard_normal(n)
    return {"U": s, "U_cat": u_cat, "X": x, "images": images, "A": a, "Y": y0 + TRUE_ATE * a, "propensity": propensity}


def features(images: np.ndarray, fit_idx: np.ndarray, pcs: int, seed: int) -> np.ndarray:
    """Unsupervised per-image feature map psi: PCA on the degraded pixels of all blocks of the fitting units."""
    n = images.shape[0]
    pca = PCA(pcs, random_state=seed).fit(images[fit_idx].reshape(-1, 784))
    return pca.transform(images.reshape(n * J, 784)).reshape(n, J * pcs).astype(np.float32)


def run_replicate(args, pool, coef, rep: int, n: int, n_index: int) -> dict:
    data = generate(pool, n, seed_key("data", rep, 4), coef, args.sigma, args.occ, args.d_x)
    perm = rng(*seed_key("split", rep, n_index)).permutation(n); third = n // 3
    idx = {"D": perm[:third], "N": perm[third:2 * third], "E": perm[2 * third:]}
    data["W"] = features(data["images"], idx["D"], args.pcs, rep)
    blk = [np.arange(b * args.pcs, (b + 1) * args.pcs) for b in range(J)]
    splits = all_splits(J, args.max_held)
    g = rng(*seed_key("sample_splits", rep, n_index))
    chosen = [splits[i] for i in sorted(g.choice(len(splits), size=min(args.m, len(splits)), replace=False))]
    records = [run_split(data, idx, S, args, rep, n_index, None, blk) for S in chosen]
    for r in records:
        r["valid_heldout"] = True
    retained = [r for r in records if r["screen_pass"]]
    out = {"scm": "e4_mnist", "sigma": args.sigma, "occ": args.occ, "pcs": args.pcs, "rep": rep, "n": n, "T": len(splits), "m": len(chosen), "splits": records}
    if retained:
        rho = linking_radius(retained, args, len(chosen))
        out["algorithm1"] = aggregate(retained, rho)
        th = np.array([r["theta"] for r in retained])
        out["mean_over_retained"] = float(th.mean()); out["median_over_retained"] = float(np.median(th))
    else:
        out["algorithm1"] = {"n_retained": 0, "output": None, "tie": False, "candidates": []}
    X, W, A, Y = data["X"], data["W"], data["A"], data["Y"]
    s = torch_seed(*seed_key("nuisance", rep, n_index, 999)) % (2 ** 31)
    out["naive"] = float(Y[A == 1].mean() - Y[A == 0].mean())
    out["baseline_X"] = baseline_aipw(X, data, idx, args.eta, s)
    out["baseline_XW"] = baseline_aipw(np.column_stack([X, W]), data, idx, args.eta, s + 1)  # raw image adjustment through psi
    out["oracle_XU"] = baseline_aipw(np.column_stack([X, data["U"]]), data, idx, args.eta, s + 2)
    out["baseline_X_Wmean"] = baseline_aipw(np.column_stack([X, W.reshape(n, J, args.pcs).mean(1)]), data, idx, args.eta, s + 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=1.0, help="pixel noise standard deviation on the [0,1] scale")
    ap.add_argument("--occ", type=int, default=10, help="side of the random occlusion square in pixels")
    ap.add_argument("--pcs", type=int, default=50, help="principal components per image (feature map psi)")
    ap.add_argument("--n", type=int, nargs="+", default=[6000])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--max-held", type=int, default=2)
    ap.add_argument("--d-x", type=int, default=20)
    ap.add_argument("--d-z", type=int, default=3)
    ap.add_argument("--enc-steps", type=int, default=300)
    ap.add_argument("--enc-hidden", type=int, default=32)
    ap.add_argument("--enc-arch", default="mlp", choices=["mlp", "set", "index", "index2", "prog"], help="representation: audit-gap MLP, or one built by a strong learner (prog)")
    ap.add_argument("--nuisance", default="gbm", choices=["gbm", "mlp"], help="nuisance learners for every AIPW")
    ap.add_argument("--pretrain-steps", type=int, default=0, help="treatment-predictive warm start of the encoder")
    ap.add_argument("--m-rff", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--keep-x", action="store_true")
    ap.add_argument("--t", type=float, default=None)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--overlap-min", type=float, default=0.98)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--rho-rule", default="normal", choices=["normal", "chebyshev"])
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(); args.diagnostics = False; args.n_diag = 0
    torch.set_num_threads(args.threads)
    P.NUISANCE_LEARNER["kind"] = args.nuisance
    pool = load_pool()
    g = np.random.default_rng(np.random.SeedSequence(seed_key("coef")))
    coef = {k: (lambda v: v / np.linalg.norm(v))(g.standard_normal(args.d_x)) for k in ("b_x", "a_x", "y_x", "yt_x")}
    out = Path(args.out) if args.out else RESULTS_DIR / f"probe_e4_mnist_sigma{args.sigma}_occ{args.occ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); runs = []

    def save(done: bool):
        summary = {n: summarize([r for r in runs if r["n"] == n]) for n in args.n if any(r["n"] == n for r in runs)}
        payload = {"config": vars(args), "true_ate": TRUE_ATE, "complete": done, "summary": summary, "runs": runs, "runtime_s": time.time() - t0}
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o))
        tmp.replace(out)
        return summary

    for n_index, n in enumerate(args.n):
        for rep in range(args.reps):
            r = run_replicate(args, pool, coef, rep, n, n_index); runs.append(r)
            a = r["algorithm1"]
            print(f"[e4 sigma={args.sigma} occ={args.occ} n={n} rep={rep}] retained {a.get('n_retained')}/{r['m']} | alg1 out={a.get('output')} rho={a.get('rho')} comps={a.get('component_sizes')} | "
                  f"naive={r['naive']:.3f} X={r['baseline_X']:.3f} XW={r['baseline_XW']:.3f} XU={r['oracle_XU']:.3f} | {time.time()-t0:.0f}s", flush=True)
            save(False)
    summary = save(True)
    print(json.dumps(summary, indent=2)); print("saved", out)


if __name__ == "__main__":
    main()
