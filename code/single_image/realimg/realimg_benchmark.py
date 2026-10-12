"""Semi-synthetic real-image benchmark v1 (memo/2026-10-10-pneumoniamnist-benchmark-v1.md).

Real PneumoniaMNIST 64x64 chest X-rays; U = standardised pneumonia label (hidden from every learner);
A, Y and the block-5 overlays are simulated per replicate.  The paper's sealed PROBE (family_v2_probe) is run
unchanged on two block definitions (pixels, maereg) and compared with naive, oracle, CNN_all and all-blocks
adjustment on the same sealed folds and AIPW.

  python realimg_benchmark.py --prepare                 # train/cache MAE, render the block/overlay figure
  python realimg_benchmark.py --scenarios 0,1 --tag A   # run replicates
  python realimg_benchmark.py --summarize               # pool all result files
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from single_image import level_a, mae as mae_mod, nets  # noqa: E402
import family_v2_probe as pr  # noqa: E402

SP = Path(__file__).resolve().parent
DATA = SP.parents[2] / "data/medmnist/pneumoniamnist_64.npz"
OUT = SP.parents[2] / "results/single_image/pneumoniamnist_v1"
BLOCKS = [((8, 30), (4, 27)), ((33, 55), (4, 27)), ((8, 30), (36, 59)), ((33, 55), (36, 59)), ((0, 63), (29, 34))]
MAE_EPOCHS, MAE_SEED = 30, 20261010
DELTA2 = 0.005
R_DEFAULT = 10


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def masks() -> np.ndarray:
    m = np.zeros((5, 64, 64), bool)
    for k, ((r0, r1), (c0, c1)) in enumerate(BLOCKS):
        m[k, r0:r1 + 1, c0:c1 + 1] = True
    return m


def load():
    z = np.load(DATA, allow_pickle=False)
    x = np.concatenate([z["train_images"], z["val_images"], z["test_images"]]).astype(np.float32) / 255.0
    u = np.concatenate([z["train_labels"], z["val_labels"], z["test_labels"]]).ravel().astype(np.float64)
    return x, u


def get_mae(x):
    path = OUT / f"mae_pneu_e{MAE_EPOCHS}_s{MAE_SEED}.pt"
    if path.exists():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        trunk = nets.ConvTrunk(ck["width"])
        trunk.load_state_dict(ck["trunk"])
        return trunk.eval(), tuple(ck["stats"]), ck["report"]
    res = mae_mod.train_mae(x, MAE_SEED, torch.device("cpu"), epochs=MAE_EPOCHS, log=log)
    report = {k: v for k, v in res.items() if k != "model"}
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"trunk": res["model"].trunk.state_dict(), "width": res["model"].trunk.width,
                "stats": res["stats"], "report": report}, path)
    return res["model"].trunk.eval(), res["stats"], report


def trunk_features(trunk, stats, imgs, vis_mask=None) -> np.ndarray:
    vis = np.ones((64, 64), np.float32) if vis_mask is None else vis_mask.astype(np.float32)
    out = []
    with torch.no_grad():
        for s in range(0, len(imgs), 512):
            z = nets.standardise(imgs[s:s + 512], stats).astype(np.float32)
            xin = np.stack([z * vis, np.broadcast_to(vis, z.shape)], axis=1).astype(np.float32)
            x = torch.from_numpy(xin)
            f = trunk(x)
            v = torch.nn.functional.avg_pool2d(x[:, 1:2], 8)
            mean = (f * v).sum((2, 3)) / (v.sum((2, 3)) + 1e-6)
            out.append(torch.cat([mean, f.amax((2, 3))], 1).numpy())
    return np.concatenate(out).astype(np.float64)


def draw(c: int, r: int, x: np.ndarray, u_star: np.ndarray, m: np.ndarray) -> dict:
    rng = np.random.default_rng(20261100 + 100 * c + r)
    n = len(x)
    u = (u_star - u_star.mean()) / u_star.std()
    i = (rng.random(n) < 0.3).astype(float) if c == 1 else np.zeros(n)
    a = (rng.random(n) < 0.1 + 0.8 * norm.cdf(1.5 * u + 2.0 * i)).astype(float)
    y = 1.0 * a + u + rng.standard_normal(n)
    img = x.copy()
    if c == 1:
        line = np.zeros((64, 64), np.float32)
        line[4:45, 31:33] = 0.35
        img[i == 1] = np.clip(img[i == 1] + line[None] * m[4][None], 0, 1)
    if c == 2:
        rr = u - 1.5 * a + 0.5 * rng.standard_normal(n)
        rt = np.clip((rr - rr.mean()) / rr.std(), -2.5, 2.5)
        yy, xx = np.mgrid[0:64, 0:64]
        blob = (np.exp(-((yy - 20) ** 2 + (xx - 31.5) ** 2) / (2 * 2.0 ** 2)) * m[4]).astype(np.float32)
        img = np.clip(img + 0.08 * rt[:, None, None].astype(np.float32) * blob[None], 0, 1)
    return {"U": u, "I": i, "A": a, "Y": y, "img": img}


def aipw_features(view, feats, seed) -> float:
    n = len(view["A"])
    fold_rows = pr.folds(n, seed)
    tot = 0.0
    for rot in range(4):
        rl = pr.roles(rot)
        rn, re = fold_rows[rl["N"]], fold_rows[rl["E"]]
        mu, sd = feats[rn].mean(0), feats[rn].std(0) + 1e-12
        fn = np.column_stack([np.ones(len(rn)), (feats[rn] - mu) / sd])
        fe = np.column_stack([np.ones(len(re)), (feats[re] - mu) / sd])
        tot += float(pr.aipw(fn, view["A"][rn], view["Y"][rn], fe, view["A"][re], view["Y"][re]).mean()) / 4
    return tot


def all_blocks_prep(view, seed) -> float:
    """AIPW on all five blocks after the sealed per-block PCA (Prep fitted on fold D): PROBE's features, no balance."""
    n = len(view["A"])
    fold_rows = pr.folds(n, seed)
    tot = 0.0
    for rot in range(4):
        rl = pr.roles(rot)
        rd_, rn, re = fold_rows[rl["D"]], fold_rows[rl["N"]], fold_rows[rl["E"]]
        prep = pr.Prep(view, rd_)
        f = lambda rows: np.column_stack([np.ones(len(rows))] + [prep.block(view, rows, j) for j in range(5)])
        tot += float(pr.aipw(f(rn), view["A"][rn], view["Y"][rn], f(re), view["A"][re], view["Y"][re]).mean()) / 4
    return tot


def probe_summary(rec: dict) -> dict:
    est = {nm: r["estimate"] for nm, r in rec["per_split"].items()}
    retained = level_a.rethreshold(rec, DELTA2, "all_rotations")["retained"]
    gaps = {nm: r["pooled_gap"] for nm, r in rec["per_split"].items()}
    return {"per_split_theta": est, "per_split_gap": gaps,
            "relaxed_retained": retained,
            "relaxed_estimate": float(np.median([est[s] for s in retained])) if retained else None,
            "v3_estimate": rec["v3"]["estimate"], "v3_retained": rec["v3"]["retained"],
            "sealed_estimate": rec["sealed"]["estimate"], "seconds": rec["seconds"]}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    x, u = load()
    trunk, stats, rep = get_mae(x)
    log(f"MAE: val masked MSE {rep['val_masked_mse']:.4f} vs mean-image {rep['reference_mean_image_mse']:.4f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = masks()
    d1, d2 = draw(1, 0, x, u, m), draw(2, 0, x, u, m)
    k1 = int(np.where(d1["I"] == 1)[0][0])
    hi, lo = int(np.argmax(np.abs(d2["img"] - x).sum((1, 2)))), 0
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
    ax[0].imshow(x[0], cmap="gray")
    for k in range(5):
        ax[0].contour(m[k], levels=[.5], colors=["c", "c", "y", "y", "r"][k], linewidths=.8)
    ax[0].set_title("blocks W1-W4 (lungs), W5 (red)")
    ax[1].imshow(d1["img"][k1], cmap="gray"); ax[1].set_title("C1: device line (I=1)")
    ax[2].imshow(d2["img"][hi], cmap="gray", vmin=0, vmax=1); ax[2].set_title("C2: collider blob (largest)")
    ax[3].imshow(np.abs(d2["img"][hi] - x[hi]), cmap="magma"); ax[3].set_title("C2: |overlay|")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "blocks_overlays.png", dpi=90)
    log(f"saved {OUT / 'blocks_overlays.png'}")


def run(scenarios, reps, tag):
    x, u_star = load()
    m = masks()
    trunk, stats, _ = get_mae(x)
    base_region = [trunk_features(trunk, stats, x, m[k]) for k in range(4)]
    base_all = trunk_features(trunk, stats, x)
    out_path = OUT / f"results_{tag}.jsonl"
    for c in scenarios:
        for r in range(reps):
            t0 = time.time()
            d = draw(c, r, x, u_star, m)
            seed = 7000 + 100 * c + r
            img = d["img"]
            region5 = trunk_features(trunk, stats, img, m[4])
            feats_all = base_all if c == 0 else trunk_features(trunk, stats, img)
            rec = {"scenario": c, "rep": r, "seed": seed, "naive": float(d["Y"][d["A"] == 1].mean() - d["Y"][d["A"] == 0].mean()),
                   "treated_fraction": float(d["A"].mean())}
            view_any = None
            for kind in ("pixels", "maereg"):
                if kind == "pixels":
                    blocks = [img[:, m[k]].astype(np.float64) for k in range(5)]
                else:
                    blocks = base_region + [region5]
                view = level_a.make_view(np.zeros((len(img), 0)), blocks, d["A"], d["Y"])
                view_any = view
                probe = level_a.run_probe(view, seed, f"REALIMG_{kind.upper()}")
                rec[kind] = {"probe": probe_summary(probe), "all_blocks_prep": all_blocks_prep(view, seed)}
            comps = level_a.comparators(view_any, d["U"], seed)
            rec["oracle"] = comps["aipw_X_U"]
            rec["CNN_all"] = aipw_features(view_any, feats_all, seed)
            rec["seconds"] = time.time() - t0
            with out_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            p, q = rec["pixels"]["probe"], rec["maereg"]["probe"]
            log(f"C{c} r{r}: naive {rec['naive']:.3f} oracle {rec['oracle']:.3f} CNN_all {rec['CNN_all']:.3f} | "
                f"pixels all {rec['pixels']['all_blocks_prep']:.3f} PROBE {p['relaxed_estimate']} "
                f"(S1-4 {[round(p['per_split_theta'][s], 3) for s in ('S1', 'S2', 'S3', 'S4')]}) | "
                f"maereg all {rec['maereg']['all_blocks_prep']:.3f} PROBE {q['relaxed_estimate']} | {rec['seconds']:.0f}s")


def summarize():
    rows = [json.loads(l) for f in sorted(OUT.glob("results_*.jsonl")) for l in f.read_text().splitlines() if l]
    summ = {}
    for c in sorted({r["scenario"] for r in rows}):
        rr = [r for r in rows if r["scenario"] == c]
        cols = {"naive": [r["naive"] for r in rr], "oracle": [r["oracle"] for r in rr], "CNN_all": [r["CNN_all"] for r in rr]}
        for kind in ("pixels", "maereg"):
            cols[f"{kind}: all blocks, no balance"] = [r[kind]["all_blocks_prep"] for r in rr]
            for s in ("S1", "S2", "S3", "S4", "S5"):
                cols[f"{kind}: PROBE {s}"] = [r[kind]["probe"]["per_split_theta"][s] for r in rr]
            cols[f"{kind}: PROBE mean of S1-S4"] = [np.mean([r[kind]["probe"]["per_split_theta"][s] for s in ("S1", "S2", "S3", "S4")]) for r in rr]
            rel = [r[kind]["probe"]["relaxed_estimate"] for r in rr]
            cols[f"{kind}: PROBE relaxed rule"] = [v for v in rel if v is not None]
            cols[f"{kind}: S5 gap / mean S1-S4 gap"] = [r[kind]["probe"]["per_split_gap"]["S5"] / np.mean([r[kind]["probe"]["per_split_gap"][s] for s in ("S1", "S2", "S3", "S4")]) for r in rr]
            cols[f"{kind}: relaxed rule returned (count)"] = [float(len([1 for r in rr if r[kind]["probe"]["relaxed_estimate"] is not None]))]
        summ[f"C{c}"] = {k: {"mean": float(np.mean(v)) if v else None,
                             "mc_se": float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
                             "rmse_vs_tau": float(np.sqrt(np.mean((np.array(v) - 1.0) ** 2))) if v and "gap" not in k and "count" not in k else None,
                             "n": len(v)} for k, v in cols.items()}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    for c, s in summ.items():
        print(f"== {c}")
        for k, v in s.items():
            if v["mean"] is not None:
                print(f"  {k:42s} {v['mean']:.4f}  (MC SE {v['mc_se'] if v['mc_se'] is None else round(v['mc_se'], 4)}, RMSE {v['rmse_vs_tau'] if v['rmse_vs_tau'] is None else round(v['rmse_vs_tau'], 4)}, n={v['n']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--scenarios", type=lambda s: [int(v) for v in s.split(",")], default=[0, 1, 2])
    ap.add_argument("--reps", type=int, default=R_DEFAULT)
    ap.add_argument("--tag", default="all")
    a = ap.parse_args()
    nets.setup_torch(2)
    if a.prepare:
        prepare()
    elif a.summarize:
        summarize()
    else:
        run(a.scenarios, a.reps, a.tag)
