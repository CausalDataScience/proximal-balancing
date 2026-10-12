"""The paper's sealed PROBE (family_v2_probe.run_dataset: all 30 held-out subsets, 4 rotations, rank-2 linear
encoder, screen, AIPW, aggregation) applied to the single-image phantom with three block definitions:

    reader   W_k = reader CNN outputs of region k (8 numbers)                    [the paper's method as run so far]
    pixels   W_k = brightness-corrected raw pixels of region k (~800 numbers);
             the sealed Prep reduces them to principal components above the MP edge  [the paper's method + PCA]
    maereg   W_k = whole-image MAE trunk applied to the image with only region k visible, pooled (128 numbers);
             the sealed Prep reduces them the same way                           [CNN_all split by region]

Comparators on the sealed rotations: naive, all blocks without balance, CNN_all (whole-image MAE features,
128 numbers, no balance), oracle (true U).  Nothing in the sealed code is changed.
Usage (from code/):  python sealed_blocks.py --blocks reader,pixels --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from single_image import decisive_v2 as dv, folds, level_a, nets, objective_v1 as ob, phantom as ph, regions as rg
from single_image import run_objective_v1 as ro
from single_image.run_pipeline_v2 import json_default
import family_v2_probe as pr
import run_twins as rtw


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def trunk_features(mode, images_processed, visible_regions: tuple[int, ...]):
    """Pooled MAE-trunk features (128) of the standardised image with only `visible_regions` (1-based) shown."""
    model, stats = mode.fit["mae"]["model"], mode.fit["mae"]["stats"]
    held = tuple(j for j in range(1, 6) if j not in visible_regions)
    trunk = model.trunk
    trunk.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(images_processed), 256):
            x = torch.from_numpy(rg.holdout_view(nets.standardise(images_processed[s:s + 256], stats), held))
            f = trunk(x)
            vis = torch.nn.functional.avg_pool2d(x[:, 1:2], 8)
            mean = (f * vis).sum((2, 3)) / (vis.sum((2, 3)) + 1e-6)
            out.append(torch.cat([mean, f.amax((2, 3))], 1).numpy())
    return np.concatenate(out).astype(np.float64)


def sealed_aipw_on_features(view, feats, seed):
    """AIPW with the sealed nuisances on the sealed N/E rotations for an arbitrary feature matrix (no X here)."""
    n = len(view["A"])
    fold_rows = pr.folds(n, seed)
    total = 0.0
    for rot in range(4):
        rl = pr.roles(rot)
        rn, re = fold_rows[rl["N"]], fold_rows[rl["E"]]
        mu, sd = feats[rn].mean(0), feats[rn].std(0) + 1e-12
        fn = np.column_stack([np.ones(len(rn)), (feats[rn] - mu) / sd])
        fe = np.column_stack([np.ones(len(re)), (feats[re] - mu) / sd])
        psi = pr.aipw(fn, view["A"][rn], view["Y"][rn], fe, view["A"][re], view["Y"][re])
        total += float(psi.mean()) / 4.0
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", type=lambda s: s.split(","), default=["reader", "pixels", "maereg"])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    args = ro.parser().parse_args([])
    nets.setup_torch(2)
    cfg = ph.PhantomConfig(n=args.n, sigma=args.sigma, seed=args.seed, image_only=True)
    data = dv.bounded_generate(cfg)
    g = folds.six_groups(args.n, args.seed)
    roles = {"R": g["R"], **folds.roles(g["groups"], 0)}
    mode, info = ob.stage1(data, roles, cfg, args, log, args.cache_dir)
    assert info["loaded"], "stage-1 cache missing"
    frame = mode.frame
    pseed = level_a.probe_seed(cfg.seed)
    masks = [m.astype(bool) for m in rg.masks()]
    result = {"artifact": "sealed_probe_on_single_image_blocks", "sigma": cfg.sigma, "seed": cfg.seed, "n": cfg.n,
              "tau": data.tau, "probe_seed": pseed, "runs": {}}

    log("CNN_all comparator (whole-image MAE features, 128)")
    full = trunk_features(mode, frame.processed, (1, 2, 3, 4, 5))
    view_any = level_a.make_view(data.X, [frame.psi[k] for k in range(5)], data.A, data.Y)
    result["CNN_all_sealed_aipw"] = sealed_aipw_on_features(view_any, full, pseed)
    log(f"CNN_all sealed AIPW = {result['CNN_all_sealed_aipw']:.4f}")
    a.out.write_text(json.dumps(result, indent=1, default=json_default))

    for kind in a.blocks:
        t0 = time.time()
        if kind == "reader":
            blocks = [frame.psi[k] for k in range(5)]
        elif kind == "pixels":
            blocks = [frame.processed[:, masks[k]].astype(np.float64) for k in range(5)]
        elif kind == "maereg":
            blocks = [trunk_features(mode, frame.processed, (k + 1,)) for k in range(5)]
        else:
            raise SystemExit(kind)
        view = level_a.make_view(data.X, blocks, data.A, data.Y)
        log(f"{kind}: block widths {[b.shape[1] for b in blocks]}; running sealed PROBE")
        probe = level_a.run_probe(view, pseed, f"SINGLE_IMAGE_{kind.upper()}")
        comps = level_a.comparators(view, data.U, pseed)
        est = {nm: r["estimate"] for nm, r in probe["per_split"].items()}
        valid = [nm for nm in est if "5" not in nm]
        probe["summary"] = {"singletons": {nm: est[nm] for nm in est if len(nm) == 2},
                            "pairs": {nm: est[nm] for nm in est if len(nm) == 3},
                            "median_valid": float(np.median([est[nm] for nm in valid])),
                            "median_valid_singletons_pairs": float(np.median([est[nm] for nm in valid if len(nm) <= 3])),
                            "v3_estimate": probe["v3"]["estimate"], "v3_retained": probe["v3"]["retained"],
                            "sealed_estimate": probe["sealed"]["estimate"], "sealed_retained": probe["sealed"]["retained"],
                            "seconds": time.time() - t0}
        result["runs"][kind] = {"probe": probe, "comparators": comps}
        log(f"{kind}: singletons {probe['summary']['singletons']}")
        log(f"{kind}: pairs {probe['summary']['pairs']}")
        log(f"{kind}: v3 {probe['v3']['estimate']} retained {probe['v3']['retained']}; sealed {probe['sealed']['estimate']} "
            f"retained {probe['sealed']['retained']}; comparators {comps}; {time.time() - t0:.0f}s")
        a.out.write_text(json.dumps(result, indent=1, default=json_default))
    log(f"saved {a.out}")


if __name__ == "__main__":
    main()
