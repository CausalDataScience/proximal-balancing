"""Z0 fine-tuning plan v1, step 2 (memo/2026-10-09-z0-balance-finetune-v1.md, approved 2026-10-10).

Per split S and start point, the same network is evaluated before and after O1 (Brier-gap) balance fine-tuning:

    start "head"  MAE trunk + random 2-d MLP head        (Z0_head; the objective_v1 O1/mae start)
    start "pca"   MAE trunk + linear head set to PCA(2)  (Z0_pca exactly; PCA fitted on R)

Changes from objective_v1 O1: stop-rule patience (default 8 instead of 3). Everything else is objective_v1's
train_gap_arm, final_evaluation (P fit, S audit, E AIPW) and fresh-sample D2.

Standing comparator (user instruction 2026-10-10): the unbalanced CNN that sees everything. Here X is empty
(image-only phantom), so CNN_all = MAE trunk pooled features (128) of the whole, unmasked image. Every
candidate is paired against it on the same E rows.

Run from code/:  python -m single_image.run_z0_finetune_v1 --splits 1,2 --tag A
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time

import numpy as np
import torch
from torch import nn

from . import decisive_v2 as dv, folds, nets, objective_v1 as ob, phantom as ph, regions as rg
from . import run_objective_v1 as ro
from .run_pipeline_v2 import json_default
from .run_z0_before_v1 import OUT, TrunkFeatureEncoder, fit_pca, paired

START_OFFSET = {"head": 0, "pca": 600}


class FullImageTrunkEncoder(TrunkFeatureEncoder):
    """CNN_all: MAE trunk features of the unmasked image (nothing held out)."""

    def __init__(self, base):
        super().__init__(base)

    def features(self, frame, rows):
        model, stats = self.base.model, self.base.stats
        model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(rows), 256):
                x = torch.from_numpy(rg.holdout_view(nets.standardise(frame.processed[rows[start:start + 256]],
                                                                       stats), ()))
                f = model.trunk(x)
                vis = torch.nn.functional.avg_pool2d(x[:, 1:2], 8)
                mean = (f * vis).sum((2, 3)) / (vis.sum((2, 3)) + 1e-6)
                out.append(torch.cat([mean, f.amax((2, 3))], 1).numpy())
        return np.concatenate(out).astype(np.float64)


def start_encoder(mode, split, cfg, start, roles):
    enc = ob.base_encoder(mode, split, cfg)
    if start == "pca":
        (mu, comps), _ = fit_pca(TrunkFeatureEncoder(enc), mode.frame, roles["R"])
        lin = nn.Linear(comps.shape[1], comps.shape[0])
        with torch.no_grad():
            lin.weight.copy_(torch.as_tensor(comps, dtype=torch.float32))
            lin.bias.copy_(torch.as_tensor(-comps @ mu, dtype=torch.float32))
        enc.model.head = lin
    return enc


def summarize(audit, effect, tau, fresh=None):
    out = {"theta": effect["candidate_estimate"], "SE": effect["sampling_SE"],
           "abs_error": abs(effect["candidate_estimate"] - tau), "S_gap": audit["gap"], "S_gap_SE": audit["se"],
           "clipped_fraction_E": effect["propensity_tail_fraction"]}
    if fresh is not None:
        out.update({"D2_fresh": fresh["D2_fresh"], "D2_fresh_replicates": fresh["replicate_D2_out_of_fold"]})
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=ro.int_list, default=[1, 2, 3, 4])
    p.add_argument("--starts", type=lambda s: s.split(","), default=["head", "pca"])
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--tag", required=True)
    a = p.parse_args(argv)
    args = ro.parser().parse_args([])
    ob.StopRule.__init__.__defaults__ = (a.patience, ob.STOP_RELATIVE)

    def log(msg):
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)

    out_json = OUT / f"step2_sigma{args.sigma}_seed{args.seed}_{a.tag}.json"
    if out_json.exists():
        raise SystemExit(f"{out_json} exists")
    t0 = time.time()
    nets.setup_torch(2)
    cfg = ph.PhantomConfig(n=args.n, sigma=args.sigma, seed=args.seed, image_only=True)
    data = dv.bounded_generate(cfg)
    g = folds.six_groups(args.n, args.seed)
    roles = {"R": g["R"], **folds.roles(g["groups"], 0)}
    mode, info = ob.stage1(data, roles, cfg, args, log, args.cache_dir)
    if not info["loaded"]:
        raise SystemExit("stage-1 cache missing")
    fresh = ob.FreshSample(mode, cfg, args.mc, args.seed + 30000001, log)
    tau = data.tau

    cnn_all = FullImageTrunkEncoder(ob.base_encoder(mode, 0, cfg))
    audit, effect, all_score, _ = ob.final_evaluation(mode, data, roles, cnn_all, args.seed + 88000, args, "CNN_all")
    result = {"artifact": "single_image_z0_finetune_step2", "protocol": "memo/2026-10-09-z0-balance-finetune-v1.md",
              "sigma": args.sigma, "seed": args.seed, "tau": tau, "stop_patience": a.patience,
              "rounds_max": args.rounds, "fresh_sample": fresh.report,
              "CNN_all": summarize(audit, effect, tau), "splits": {}}
    log(f"CNN_all (unbalanced, whole image, 128 features): theta={effect['candidate_estimate']:.4f}")
    out_json.write_text(json.dumps(result, indent=1, default=json_default))

    for split in [s - 1 for s in a.splits]:
        sname = f"S{split + 1}"
        rec = {}
        for start in a.starts:
            name = f"{start}_{sname}"
            enc = start_encoder(mode, split, cfg, start, roles)
            init_state = copy.deepcopy(enc.model.state_dict())
            seed = args.seed + ob.ARM_OFFSET["O1"] + START_OFFSET[start] + split
            audit, effect, before_score, _ = ob.final_evaluation(mode, data, roles, enc, seed + 9000, args,
                                                                 f"before_{name}")
            before = summarize(audit, effect, tau, ob.fresh_d2(mode, enc, split, fresh, args, log, f"before_{name}"))
            log(f"before {name}: theta={before['theta']:.4f} D2={before['D2_fresh']:.6f}")
            enc.model.load_state_dict(init_state)
            training = ob.train_gap_arm(mode, data, roles, enc, split, "brier", seed, args, log, f"after_{name}")
            audit, effect, after_score, _ = ob.final_evaluation(mode, data, roles, enc, seed + 9000, args,
                                                                f"after_{name}")
            after = summarize(audit, effect, tau, ob.fresh_d2(mode, enc, split, fresh, args, log, f"after_{name}"))
            after["training"] = {k: v for k, v in training.items() if k != "history"}
            after["training"]["C_criterion_by_round"] = [h.get("C_criterion") for h in training["history"]]
            log(f"after {name}: theta={after['theta']:.4f} D2={after['D2_fresh']:.6f} "
                f"rounds={training['rounds_completed']} selected={training['selected_round']}")
            rec[start] = {"before": before, "after": after,
                          "paired": {"after - before": paired(after_score, before_score),
                                     "after - CNN_all": paired(after_score, all_score),
                                     "before - CNN_all": paired(before_score, all_score)}}
            result["splits"][sname] = rec
            out_json.write_text(json.dumps(result, indent=1, default=json_default))
    result["seconds"] = time.time() - t0
    out_json.write_text(json.dumps(result, indent=1, default=json_default))
    log(f"saved {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
