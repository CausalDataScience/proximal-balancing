"""Z0 fine-tuning plan v1, step 1 (memo/2026-10-09-z0-balance-finetune-v1.md): the "before" embeddings.

For each split S, three unbalanced CNN embeddings of the image with region S masked are evaluated with the same
P-fit / S-audit / E-effect pipeline and the same fresh-sample D2 as objective_v1:

    Z0_head  MAE trunk + random 2-d head  (bitwise the start of the O1/O3 "mae" fine-tuning runs)
    Z0_pca   MAE trunk pooled features (128) -> PCA(2) fitted on R
    Z0_full  MAE trunk pooled features (128)

Each gets a brightness-reference column, as every objective_v1 encoder does. No balance objective is used.
The "after" scores are read from the objective_v1 part files (same E rows) and compared pairwise.

Run from code/:  python -m single_image.run_z0_before_v1 --splits 1,2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import decisive_v2 as dv, folds, level_a, objective_v1 as ob, phantom as ph, pipeline_v2 as pv
from . import run_objective_v1 as ro
from .run_pipeline_v2 import PROJECT, json_default

OUT = PROJECT / "results/single_image/z0_finetune_v1"
AFTER = PROJECT / "results/single_image/objective_v1"
SEED_OFFSET = 77000


class TrunkFeatureEncoder:
    """Pooled MAE-trunk features of the masked view (the RegionEncoder input before its head), optionally PCA."""

    def __init__(self, base: pv.NeuralImageEncoder, pca: tuple[np.ndarray, np.ndarray] | None = None):
        self.base, self.split, self.mode, self.pca = base, base.split, base.mode, pca

    def features(self, frame, rows):
        model = self.base.model
        model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(rows), 256):
                x = torch.from_numpy(self.base.view(frame, rows[start:start + 256]))
                f = model.trunk(x)
                vis = torch.nn.functional.avg_pool2d(x[:, 1:2], 8)
                mean = (f * vis).sum((2, 3)) / (vis.sum((2, 3)) + 1e-6)
                out.append(torch.cat([mean, f.amax((2, 3))], 1).numpy())
        return np.concatenate(out).astype(np.float64)

    def encode(self, frame, rows):
        feat = self.features(frame, rows)
        if self.pca is not None:
            mu, comps = self.pca
            feat = (feat - mu) @ comps.T
        ref = np.asarray(frame.reference[rows], float).reshape(len(rows), -1)
        return np.column_stack([ref, feat])


def fit_pca(enc: TrunkFeatureEncoder, frame, rows, k=2):
    feat = enc.features(frame, rows)
    mu = feat.mean(0)
    _, s, vt = np.linalg.svd(feat - mu, full_matrices=False)
    explained = (s ** 2) / (s ** 2).sum()
    return (mu, vt[:k]), [float(v) for v in explained[:5]]


def paired(score_a, score_b):
    d = np.asarray(score_a) - np.asarray(score_b)
    return {"difference": float(d.mean()), "SE": float(d.std(ddof=1) / math.sqrt(len(d)))}


def load_after(sigma, seed, split):
    """E scores of the fine-tuned (mae init) models and level A for this split from complete objective_v1 parts."""
    found = {}
    for path in sorted(AFTER.glob(f"{ro.stem(sigma, seed)}.part-*.json")):
        d = json.loads(path.read_text())
        if not d.get("computation_complete"):
            continue
        la = d["level_A"].get(f"S{split + 1}")
        if la is not None and "levelA" not in found:
            found["levelA"] = {"theta": la["effect_E"]["candidate_estimate"], "scores": la["E_scores"],
                               "D2_fresh": None if la["D2_fresh"] is None else la["D2_fresh"]["D2_fresh"],
                               "file": path.name}
        if la is not None and la["D2_fresh"] is not None:
            found["levelA"]["D2_fresh"] = la["D2_fresh"]["D2_fresh"]
        for name, rec in d["models"].items():
            if rec["split"] == split + 1 and rec["init"] == "mae":
                found[name] = {"theta": rec["effect_E"]["candidate_estimate"], "scores": rec["E_scores"],
                               "D2_fresh": None if rec["D2_fresh"] is None else rec["D2_fresh"]["D2_fresh"],
                               "initial_encoder_sha256": rec["training"]["initial_encoder_sha256"],
                               "file": path.name}
    return found


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=ro.int_list, default=[1, 2])
    p.add_argument("--sigma", type=float, default=.3)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--no-fresh", action="store_true")
    a = p.parse_args(argv)
    args = ro.parser().parse_args(["--sigma", str(a.sigma), "--seed", str(a.seed)])
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / f"z0_before_sigma{a.sigma}_seed{a.seed}.json"

    def log(msg):
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)

    t_start = time.time()
    torch.set_num_threads(2)
    cfg = ph.PhantomConfig(n=args.n, sigma=args.sigma, seed=args.seed, image_only=True)
    data = dv.bounded_generate(cfg)
    grouped = folds.six_groups(args.n, args.seed)
    roles = {"R": grouped["R"], **folds.roles(grouped["groups"], 0)}
    mode, cache_info = ob.stage1(data, roles, cfg, args, log, args.cache_dir)
    if not cache_info["loaded"]:
        raise SystemExit("stage-1 cache missing; refuse to refit stage 1 for a 'before' measurement")
    fresh = None if a.no_fresh else ob.FreshSample(mode, cfg, args.mc, args.seed + 30000001, log)
    frame, p_rows, e_rows = mode.frame, roles["P"], roles["E"]

    # full-image reader adjustment (no balancing), recomputed so it can be paired on the same E rows
    full = np.column_stack([frame.reference, *frame.psi])
    fb = pv.CriticBundle(full[p_rows], frame.psi[0][p_rows], mode.crop(frame, p_rows, 0), data.A[p_rows],
                         args.seed + SEED_OFFSET + 99, args.critic_epochs, 1)
    qf, _, _ = fb.predict(full[e_rows], frame.psi[0][e_rows], mode.crop(frame, e_rows, 0))
    full_score = dv.aipw_scores(full[p_rows], full[e_rows], data.A[p_rows], data.Y[p_rows], data.A[e_rows],
                                data.Y[e_rows], qf)
    a_e, y_e = data.A[e_rows], data.Y[e_rows]
    result = {"artifact": "single_image_z0_before_v1", "protocol": "memo/2026-10-09-z0-balance-finetune-v1.md",
              "sigma": a.sigma, "seed": a.seed, "tau": data.tau, "stage1_cache": cache_info,
              "fresh_sample": None if fresh is None else fresh.report,
              "naive_E": float(y_e[a_e == 1].mean() - y_e[a_e == 0].mean()),
              "full_image_reader_adjustment": float(full_score.mean()),
              "splits": {}}
    log(f"naive {result['naive_E']:.4f}; full-image reader adjustment {full_score.mean():.4f}")

    for split in [s - 1 for s in a.splits]:
        sname = f"S{split + 1}"
        base = ob.base_encoder(mode, split, cfg)
        head_sha = ob.state_sha(base.model.state_dict())
        after = load_after(a.sigma, a.seed, split)
        sha_match = {k: v["initial_encoder_sha256"] == head_sha for k, v in after.items()
                     if "initial_encoder_sha256" in v}
        log(f"{sname}: Z0_head sha {head_sha[:12]}; matches fine-tuning start: {sha_match}")
        trunk = TrunkFeatureEncoder(base)
        pca, explained = fit_pca(trunk, frame, roles["R"])
        encoders = {"Z0_head": base, "Z0_pca": TrunkFeatureEncoder(base, pca), "Z0_full": trunk}
        rec = {"Z0_head_state_sha256": head_sha, "Z0_head_equals_finetune_start": sha_match,
               "pca_explained_top5": explained, "before": {}, "after": {}, "paired": {}}
        scores = {}
        for i, (name, enc) in enumerate(encoders.items()):
            audit, effect, score, t_eval = ob.final_evaluation(mode, data, roles, enc,
                                                               args.seed + SEED_OFFSET + 10 * split + i, args,
                                                               f"{name}_{sname}")
            scores[name] = score
            entry = {"theta": effect["candidate_estimate"], "SE": effect["sampling_SE"],
                     "abs_error": abs(effect["candidate_estimate"] - data.tau),
                     "S_gap": audit["gap"], "S_gap_SE": audit["se"],
                     "clipped_fraction_E": effect["propensity_tail_fraction"],
                     "q0_outside_05_95_S": audit["q0"]["fraction_outside_05_95"], "z_dim": int(enc.encode(frame, e_rows[:2]).shape[1]),
                     "seconds_eval": t_eval}
            if fresh is not None:
                fr = ob.fresh_d2(mode, enc, split, fresh, args, log, f"{name}_{sname}")
                entry["D2_fresh"] = fr["D2_fresh"]
                entry["D2_fresh_replicates"] = fr["replicate_D2_out_of_fold"]
            rec["before"][name] = entry
            log(f"{name}_{sname}: theta={entry['theta']:.4f} (SE {entry['SE']:.4f}), S gap={entry['S_gap']:.5f}, "
                f"D2_fresh={entry.get('D2_fresh', float('nan')):.6f}, dim={entry['z_dim']}")
            result["splits"][sname] = rec
            out_json.write_text(json.dumps(result, indent=1, default=json_default))
        for name, v in after.items():
            rec["after"][name] = {k: v[k] for k in ("theta", "D2_fresh", "file")}
            rec["after"][name]["abs_error"] = abs(v["theta"] - data.tau)
        refs = {**{k: np.asarray(v["scores"]) for k, v in after.items()}, "full_image_reader": full_score}
        for b, sb in scores.items():
            for r, sr in refs.items():
                rec["paired"][f"{r} - {b}"] = paired(sr, sb)
        result["splits"][sname] = rec
        out_json.write_text(json.dumps(result, indent=1, default=json_default))
    result["seconds"] = time.time() - t_start
    out_json.write_text(json.dumps(result, indent=1, default=json_default))
    log(f"saved {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
