"""SCM-7-v3: full-pixel functional audit of the cross-fitted PROBE implementation on MNIST proxies.

The name is exact.  Representation search keeps the development selector, which is cross-fitted and
works on a compressed target, because refitting a convolutional critic at every grid point of every
split is out of any budget.  The final screen is where every pixel enters: a residual critic reads the
candidate representation together with trunk features of the entire held-out image block, fitted on
the same rows as the search and checked on an independent fold.  Literal Algorithm 1 is audited by
E11-v2, not here.

Caching.  Each seed renders its five image blocks once.  Each rotation fits the PROBE encoder once and
the shared pixel trunk once, then encodes every row once; the thirty candidates only fit small heads.
The raw-image comparator fits its own trunk on the nuisance rows and reuses nothing from PROBE.

Sampling mirrors SCM-6-v3: one latent draw per seed at n = 24,000, the n = 12,000 arm its fold-wise
prefix.  Rows inside the D fold at 12,000: 1,200 encoder, 300 calibration, 1,500 search and critics.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import learned_target as lt
import minimal_suite_common as c
import probe_full_target as ft
import probe_highdim_w as hd
import probe_highdim_w_v2 as v2
import probe_image_w as im
import probe_pixel_critic as pc
import probe_structured_scm as ps
import run_scm6_v2_development as s6
import run_scm7_image_w as s7
import scm6_population as sp

SOURCES = ["minimal_suite_common.py", "probe_image_w.py", "probe_highdim_w.py", "probe_highdim_w_v2.py",
           "probe_full_target.py", "probe_pixel_critic.py", "learned_target.py", "scm6_population.py",
           "run_scm7_image_w.py", "run_scm7_v3.py"]
PROTOCOL_PATH = "../results/scm7_v3/protocol.json"
SHARD_DIR = "../results/scm7_v3/shards"
LARGEST_N = 24000
FROZEN = {
    "protocol_id": "scm7-v3-full-pixel-audit",
    "experiment_id": "SCM-7-v3", "mode": "operational_rotate4",
    "arms": {"qualification": {"n": 12000, "seeds": [95000000, 95001000]},
             "confirmation": {"n": 12000, "seeds": [95200000 + 1000 * r for r in range(20)]},
             "larger_n": {"n": 24000, "seeds": [95200000 + 1000 * r for r in range(5)]}},
    "sampling": "one latent draw per seed at n = 24,000; the n = 12,000 arm is its fold-wise prefix",
    "d_fold_rows_at_12000": {"encoder_fit": 1200, "encoder_cal": 300, "search_and_critics": 1500},
    "encoder": dict(im.TRAIN),
    "selection": "development cross-fitted selector on the compressed target (encoder channel of the "
                 "held-out block plus four principal directions), unchanged",
    "pixel_trunk": {"widths": list(pc.TRUNK_WIDTHS), "stride": 2, "norm": "GroupNorm", "act": "SiLU",
                    "pooling": "none, grid flattened", "feature_dim": pc.FEATURE_DIM,
                    "inputs": "grayscale plus two normalised coordinate channels",
                    "fitted_as": "raw-image propensity model on the D_phi rows, once per rotation",
                    "epochs_max": pc.TRUNK_EPOCHS, "patience": pc.TRUNK_PATIENCE},
    "screen": {"critics": "base MLP on Z; augmented = base + residual MLP on (Z, X, trunk features of "
                          "every held-out block, held-out mask), hidden 64, SiLU, output in "
                          "[0.05, 0.95], 2 restarts, early stopping, lifted base a candidate",
               "statistic": "held-out Brier risk difference on the S fold",
               "rule": "max(gap, 0) + z_{1 - alpha/N} SE < tolerance, and no saturation",
               "tolerance": 2e-3, "alpha": 0.05, "N": 30,
               "population_window": [2.98e-5, 7.71e-3], "retention": "pass in all four rotations"},
    "discrimination_check": "reflection about the horizontal centroid must move the untrained trunk "
                            "feature by at least a tenth of the typical between-image movement",
    "comparators": ["naive", "X", "oracle", "cnn_channels", "statistics_channels", "pca_channels",
                    "raw_cnn"],
    "learned_target_evaluator": {"n": 50000, "seed_rule": "424242 + data seed", "chunk": 5000},
    "decision_record": {
        "screen_tolerance": "2e-3, set after one development seed (94000000) showed the one-sided bound's "
                            "slack z*SE reaches 2e-3 to 4e-3 whenever the residual branch moves at all, so "
                            "the earlier 1e-3 retained only 2 of 8 balance-feasible splits across four "
                            "rotations. From the recorded U values, 2e-3 retains 4 of 8 and passes no "
                            "infeasible row (their minimum U was 3.36e-3). Inside the population window "
                            "(2.99e-5, 7.72e-3). Chosen by the user on 2026-09-14 from options 1e-3, 2e-3, "
                            "2.5e-3 before any confirmation seed was opened.",
        "learned_target_gate": "median 0.10, p90 0.15, relaxed from the reviewer's 0.05 / 0.10 by the "
                               "user's decision on 2026-09-14 after the same development seed gave "
                               "0.076 / 0.122 over the splits that target 1 (encoder error 0.040 at the "
                               "exact root with 1,200 encoder rows, plus selection error). This is a "
                               "threshold set with knowledge of one development value; it is recorded "
                               "here so the sealed protocol cannot be read as pre-specified in the "
                               "stricter sense. No confirmation seed had been run.",
    },
    "gates": {
        "performance": {"return_rate_min": 0.95, "mae_max": 0.05, "p90_max": 0.10,
                        "oracle_excess_upper_max": 0.05, "beats": ["X", "raw_cnn", "best_no_balance"],
                        "alpha": 0.05, "bonferroni_contrasts": 4},
        "mechanism": {"screen_tpr_min": 0.80, "screen_fpr_max": 0.20,
                      "unique_largest_rate_min": 0.90, "component_purity_min": 0.80,
                      "learned_target_error_median_max": 0.10,
                      "learned_target_error_p90_max": 0.15},
        "runtime_to_open_confirmation": {"warm_seed_seconds_max": 1200, "peak_gpu_gb_max": 100},
        "secondary_only": ["root_recovery", "median_root_error", "channel_correlation",
                           "fitted_trunk_arrangement_sensitivity", "runtime", "gpu_memory"]},
}


def freeze() -> dict:
    p = Path(PROTOCOL_PATH)
    if p.exists():
        return json.loads(p.read_text())
    proto = dict(FROZEN)
    proto["source_snapshot"] = c.source_snapshot(SOURCES)
    proto["mnist_sha256"] = im.MNIST_SHA256
    proto["frozen_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    proto["runs"] = [{"experiment_id": "SCM-7-v3", "mode": "operational_rotate4",
                      "phase": "qualification" if arm == "qualification" else "confirmation",
                      "arm": arm, "n": spec["n"], "seeds": spec["seeds"]}
                     for arm, spec in FROZEN["arms"].items()]
    return c.seal_protocol(p, proto)


def nested_latents(seed: int, n: int) -> tuple[dict, list[np.ndarray]]:
    lat = im.latents(LARGEST_N, seed)
    folds_big = c.four_folds(LARGEST_N, seed + 1)
    folds = c.nested_prefix(folds_big, n // 4)
    keep = np.sort(np.concatenate(folds))
    pos = {row: i for i, row in enumerate(keep)}
    sliced = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == LARGEST_N else v)
              for k, v in lat.items()}
    return sliced, [np.array([pos[r] for r in f]) for f in folds]


def component_labels(agg: dict) -> list[str]:
    """Split ids of the largest component.  The aggregator explodes each string label into a list of
    characters, so ['S', '1', '2'] has to be joined back to 'S12' rather than re-encoded."""
    if not agg.get("members"):
        return []
    return ["".join(m) if not isinstance(m, str) else m for m in agg["members"][0]]


class SeedCache:
    """Everything rendered or encoded once per seed or per rotation, shared by all candidates."""

    def __init__(self, lat: dict, dev: torch.device) -> None:
        self.lat, self.dev = lat, dev
        n = len(lat["X"])
        t0 = time.time()
        self.blocks = [im.block(lat, j, np.arange(n)).astype(np.float32) for j in range(im.N_BLOCKS)]
        self.render_seconds = time.time() - t0
        self.numeric = {k: lat[k] for k in ("X", "I", "C", "A", "Y")}

    def rows(self, idx: np.ndarray) -> dict:
        return {k: v[idx] for k, v in self.numeric.items()}

    def block_rows(self, idx: np.ndarray) -> list[np.ndarray]:
        return [b[idx] for b in self.blocks]


class EvaluatorCache:
    """The learned-target evaluator sample, drawn once per seed and encoded once per rotation.

    The first version re-rendered and re-encoded 50,000 rows of five blocks for every retained split of
    every rotation, which is two dozen passes over a quarter of a million images per seed and was the
    dominant cost.  Only the encoder's channel estimates are needed, and they depend on the rotation's
    encoder alone, so one encoding per rotation serves every split.
    """

    def __init__(self, seed: int) -> None:
        spec = FROZEN["learned_target_evaluator"]
        self.big = im.latents(spec["n"], 424242 + seed)
        self.num = {k: self.big[k] for k in ("X", "I", "C", "A", "Y")}
        self.chunk, self.n = spec["chunk"], spec["n"]
        self._k, self._enc_id = None, None

    def channels(self, enc: dict) -> np.ndarray:
        if self._enc_id != id(enc):
            parts = [im.apply_encoder(enc, self.big, np.arange(s, min(s + self.chunk, self.n)))
                     for s in range(0, self.n, self.chunk)]
            self._k, self._enc_id = np.concatenate(parts), id(enc)
        return self._k

    def learned_target(self, enc: dict, split: tuple[int, ...], r, seed: int, dev) -> dict:
        z = hd.representation(self.channels(enc), self.num, split, r)
        return lt.learned_target(z, self.num["A"], self.num["Y"], seed, dev)


def _attach(spec: tuple) -> tuple:
    """A numpy view onto a shared-memory block; the segment is returned so the caller can close it."""
    from multiprocessing import shared_memory
    name, shape, dtype = spec
    seg = shared_memory.SharedMemory(name=name)
    return np.ndarray(shape, dtype=dtype, buffer=seg.buf), seg


def _select_one(args: tuple) -> tuple:
    """Selection for one split: pure numpy and scipy, so it runs in a spawned worker.

    The r search is the dominant CPU cost, about 1,200 seconds per seed, and never touches the GPU.
    Splits are independent and each carries its own optimizer seed, so running them in parallel changes
    nothing in the numbers.  Workers are spawned rather than forked, because a forked child of a process
    that has initialised the GPU runtime is not safe on every platform, and they read the rendered blocks
    and channel estimates from shared memory the parent filled once per rotation, so nothing large is
    pickled.  Nothing in a worker touches torch.
    """
    split, oseed, num_phi, num_s, k_specs, block_specs, pca_specs = args
    segs = []
    k_phi, seg = _attach(k_specs["phi"]); segs.append(seg)
    k_s, seg = _attach(k_specs["s"]); segs.append(seg)
    try:
        def view(which, num):
            b = s7.Blocks.__new__(s7.Blocks)
            b.lat, b.rows, b._cache = None, np.arange(len(num["X"])), {}
            for j in split:
                arr, seg_j = _attach(block_specs[which][j]); segs.append(seg_j)
                b._cache[j] = arr
            b.numeric = lambda: num
            return b

        phi_blocks, s_blocks = view("phi", num_phi), view("s", num_s)
        # Each block's principal directions are fitted on the D_phi rows of that block alone, so they are
        # the same for every split holding the block out.  The qualification run on the GPU node spent
        # most of its search time recomputing the same 1,500 x 9,216 SVDs; they are now computed once per
        # rotation in the parent and passed in, which is bit-identical to computing them here.
        dirs = {}
        for j in split:
            arr, seg_j = _attach(pca_specs[j]); segs.append(seg_j)
            dirs[j] = arr
        proj = {"directions": dirs, "components": s7.TARGET_COMPONENTS,
                "digest": c.array_sha256(np.concatenate([dirs[j].ravel() for j in split]))}
        z_ref = hd.representation(k_phi, num_phi, split, None if 0 in split else 0.0)
        pen = hd.choose_penalty(lambda: hd.Scaler(z_ref)(z_ref), num_phi["A"], s7.PENALTY_GRID, oseed)
        sel = s7.gap_and_screen(phi_blocks, k_phi, s_blocks, k_s, split, proj, pen, oseed)
        # the selection result carries only scalars and a digest, so it pickles back cheaply
        return split, pen, sel
    finally:
        for seg in segs:
            seg.close()


def block_directions(block_rows: np.ndarray) -> np.ndarray:
    """Leading principal directions of one block on the D_phi rows: the same arithmetic as
    fit_image_projection, applied once per block instead of once per split."""
    flat = block_rows.reshape(len(block_rows), -1).astype(np.float64)
    centred = flat - flat.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return np.ascontiguousarray(vt[:s7.TARGET_COMPONENTS].T)


def _share(arr: np.ndarray, registry: list) -> tuple:
    from multiprocessing import shared_memory
    arr = np.ascontiguousarray(arr)
    seg = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    np.ndarray(arr.shape, dtype=arr.dtype, buffer=seg.buf)[...] = arr
    registry.append(seg)
    return seg.name, arr.shape, arr.dtype.str


def run_selection(splits, phi_idx, s_idx, cache, k_all, seed, rot_index, workers: int) -> dict:
    """All thirty selections for one rotation, in spawned workers when workers > 1."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    num_phi, num_s = cache.rows(phi_idx), cache.rows(s_idx)
    jobs_meta = [(split, c.optimizer_seed(FROZEN["protocol_id"], seed, rot_index, c.split_id(split), 0))
                 for split in splits]
    if workers <= 1:
        # the serial path uses plain arrays through the same worker function, via private segments
        registry = []
        try:
            k_specs = {"phi": _share(k_all[phi_idx], registry), "s": _share(k_all[s_idx], registry)}
            block_specs = {"phi": {j: _share(cache.blocks[j][phi_idx], registry) for j in range(im.N_BLOCKS)},
                           "s": {j: _share(cache.blocks[j][s_idx], registry) for j in range(im.N_BLOCKS)}}
            pca_specs = {j: _share(block_directions(cache.blocks[j][phi_idx]), registry)
                         for j in range(im.N_BLOCKS)}
            results = [_select_one((split, oseed, num_phi, num_s, k_specs, block_specs, pca_specs))
                       for split, oseed in jobs_meta]
        finally:
            for seg in registry:
                seg.close(); seg.unlink()
        return {split: (pen, sel) for split, pen, sel in results}
    registry = []
    try:
        k_specs = {"phi": _share(k_all[phi_idx], registry), "s": _share(k_all[s_idx], registry)}
        block_specs = {"phi": {j: _share(cache.blocks[j][phi_idx], registry) for j in range(im.N_BLOCKS)},
                       "s": {j: _share(cache.blocks[j][s_idx], registry) for j in range(im.N_BLOCKS)}}
        pca_specs = {j: _share(block_directions(cache.blocks[j][phi_idx]), registry)
                     for j in range(im.N_BLOCKS)}
        jobs = [(split, oseed, num_phi, num_s, k_specs, block_specs, pca_specs)
                for split, oseed in jobs_meta]
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(_select_one, jobs, chunksize=1))
    finally:
        for seg in registry:
            seg.close(); seg.unlink()
    return {split: (pen, sel) for split, pen, sel in results}


def one_replicate(seed: int, n: int, facts: dict, dev: torch.device, workers: int = 1) -> dict:
    t0 = time.time()
    lat, folds = nested_latents(seed, n)
    cache = SeedCache(lat, dev)
    evaluator = EvaluatorCache(seed)
    scale = n // 12000
    emb_fit, emb_cal = 1200 * scale, 300 * scale
    splits = c.oriented_splits()
    scr_spec = FROZEN["screen"]
    rows_out, corrs, trunk_info, disc_info, raw_diag = [], [], [], [], []
    scores = {s: [] for s in splits}
    passes = {s: 0 for s in splits}
    base_scores: dict[str, list] = {}
    learned: dict[str, list] = {}
    timing = {"render": cache.render_seconds, "encoder": 0.0, "trunk": 0.0, "encode": 0.0,
              "selection": 0.0, "screen": 0.0, "raw_cnn": 0.0, "comparators": 0.0,
              "learned_target": 0.0}
    for rot in c.rotations(folds):
        c.assert_roles_disjoint(rot.roles)
        d_idx = rot.roles["D"]
        fit_idx, cal_idx = d_idx[:emb_fit], d_idx[emb_fit:emb_fit + emb_cal]
        phi_idx = d_idx[emb_fit + emb_cal:]
        t1 = time.time()
        enc = im.fit_encoder(lat, fit_idx, cal_idx, seed=seed)
        timing["encoder"] += time.time() - t1
        if enc["fail_closed"]:
            return {"data_seed": seed, "n_total": n, "return_kind": "no_return", "estimate": None,
                    "fail_closed": {"rotation": rot.index, "reason": enc["fail_closed"]},
                    "complete": True, "timing_seconds": time.time() - t0}
        corrs.append(float(np.median(im.channel_correlation(enc, lat, rot.roles["E"]))))
        t1 = time.time()
        k_all = im.apply_encoder(enc, lat, np.arange(n))
        timing["encode"] += time.time() - t1
        # shared pixel trunk on the D_phi rows, then features for every row
        t1 = time.time()
        num_phi = cache.rows(phi_idx)
        trunk = pc.fit_trunk(np.column_stack([num_phi["X"], num_phi["I"], num_phi["C"]]),
                             cache.block_rows(phi_idx), num_phi["A"], seed + rot.index, dev)
        timing["trunk"] += time.time() - t1
        trunk_info.append({"val_brier": trunk["val_brier"], "epochs": trunk["epochs"]})
        disc_info.append(pc.discrimination_check(trunk["trunk"], cache.blocks[1][phi_idx[:64]], dev))
        t1 = time.time()
        feats = pc.encode_blocks(trunk["trunk"], cache.blocks, dev)
        timing["encode"] += time.time() - t1
        s_idx, n_idx, e_idx = rot.roles["S"], rot.roles["N"], rot.roles["E"]
        phi_blocks = s7.Blocks(lat, phi_idx)
        num_s, num_n, num_e = cache.rows(s_idx), cache.rows(n_idx), cache.rows(e_idx)
        # comparators on the N and E rows
        t1 = time.time()
        fitted_cmp = s7.fit_comparators(lat, fit_idx, cal_idx)
        ch_n = s7.comparator_channels(lat, enc, fit_idx, cal_idx, n_idx, fitted_cmp)
        ch_e = s7.comparator_channels(lat, enc, fit_idx, cal_idx, e_idx, fitted_cmp)
        feat_n, feat_e = s7.comparator_features(lat, n_idx, ch_n), s7.comparator_features(lat, e_idx, ch_e)
        for name in feat_n:
            if name in feat_e:
                base_scores.setdefault(name, []).append(hd.aipw_from_features(
                    feat_n[name], num_n["A"], num_n["Y"], feat_e[name], num_e["A"], num_e["Y"]))
        timing["comparators"] += time.time() - t1
        t1 = time.time()
        raw = pc.raw_image_aipw(np.column_stack([num_n["X"], num_n["I"], num_n["C"]]),
                                cache.block_rows(n_idx), num_n["A"], num_n["Y"],
                                np.column_stack([num_e["X"], num_e["I"], num_e["C"]]),
                                cache.block_rows(e_idx), num_e["A"], num_e["Y"],
                                seed + 31 * rot.index, dev)
        base_scores.setdefault("raw_cnn", []).append(raw["scores"])
        raw_diag.append(raw["diagnostics"] | {"val_loss": raw["val_loss"]})
        timing["raw_cnn"] += time.time() - t1
        t1 = time.time()
        selected = run_selection(splits, phi_idx, s_idx, cache, k_all, seed, rot.index, workers)
        timing["selection"] += time.time() - t1
        for split in splits:
            sid = c.split_id(split)
            oseed = c.optimizer_seed(FROZEN["protocol_id"], seed, rot.index, sid, 0)
            pen, sel = selected[split]
            t1 = time.time()
            z_phi = hd.representation(k_all[phi_idx], num_phi, split, sel["r"])
            z_s = hd.representation(k_all[s_idx], num_s, split, sel["r"])
            t_phi = pc.image_target(feats[phi_idx], num_phi, split)
            t_s = pc.image_target(feats[s_idx], num_s, split)
            base = ft.fit_base(z_phi, num_phi["A"], oseed, dev)
            aug = ft.fit_augmented(base, z_phi, t_phi, num_phi["A"], oseed, dev)
            nest = ft.nesting_check(base, aug, z_phi, t_phi, num_phi["A"], dev)
            scr = ft.screen_full_target(base, aug, z_s, t_s, num_s["A"], scr_spec["tolerance"],
                                        scr_spec["alpha"], scr_spec["N"], dev)
            timing["screen"] += time.time() - t1
            fact = facts[sid]
            row = {"rotation": rot.index, "split_id": sid, "r": sel["r"], "penalty": pen,
                   "selection_objective": sel["objective"],
                   "compressed_screen_d2": sel["screen_d2"],
                   "screen": {k: scr[k] for k in ("gap", "se", "z", "upper", "pass", "overlap_ok",
                                                  "saturated_fraction", "augmented_kept_lifted_base",
                                                  "squared_difference")},
                   "nesting": nest,
                   "evaluator_only": {"balance_root": fact["root"], "feasible": fact["feasible"],
                                      "targets_truth": fact["targets_truth"],
                                      "tau_at_root": fact["tau_at_root"],
                                      "root_error": (None if fact["root"] is None or sel["r"] is None
                                                     else abs(sel["r"] - fact["root"]))}}
            if scr["pass"]:
                passes[split] += 1
                scores[split].append(hd.aipw_from_features(
                    hd.representation(k_all[n_idx], num_n, split, sel["r"]), num_n["A"], num_n["Y"],
                    hd.representation(k_all[e_idx], num_e, split, sel["r"]), num_e["A"], num_e["Y"]))
                t1 = time.time()
                lt_out = evaluator.learned_target(enc, split, sel["r"], seed, dev)
                timing["learned_target"] += time.time() - t1
                row["evaluator_only"]["learned_target"] = lt_out["theta"]
                learned.setdefault(sid, []).append(lt_out["theta"])
            rows_out.append(row)
        del phi_blocks

    retained = [s for s in splits if passes[s] == 4]
    est = {c.split_id(s): float(np.concatenate(scores[s]).mean()) for s in retained}
    if retained:
        mat = np.column_stack([np.concatenate(scores[s]) for s in retained])
        rho = float(ps.common_radius(mat, seed + 5))
        agg = ps.aggregate([c.split_id(s) for s in retained],
                           np.array([est[c.split_id(s)] for s in retained]), rho)
    else:
        rho, agg = None, {"return_kind": "no_return", "output": None, "members": []}
    members = component_labels(agg)
    purity = float(np.mean([facts[m]["targets_truth"] for m in members])) if members else None
    baselines = {k: float(np.concatenate(v).mean()) for k, v in base_scores.items()}
    baselines["naive"] = float(lat["Y"][lat["A"] == 1].mean() - lat["Y"][lat["A"] == 0].mean())
    gpu_peak = (torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None)
    return {"data_seed": seed, "n_total": n, "rows": rows_out,
            "retained_splits": [c.split_id(s) for s in retained], "candidate_estimates": est,
            "rho": rho, "return_kind": agg["return_kind"], "estimate": agg["output"],
            "largest_component": members, "largest_component_purity": purity,
            "learned_target_by_split": {k: float(np.median(v)) for k, v in learned.items()},
            "baselines": baselines,
            "representation_diagnostics": {"channel_correlation_median": float(np.median(corrs)),
                                           "trunk": trunk_info,
                                           "discrimination": disc_info},
            "comparator_diagnostics": {"raw_cnn_per_rotation": raw_diag},
            "timing": timing, "peak_gpu_gb": gpu_peak, "device": str(dev),
            "complete": True, "timing_seconds": time.time() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(FROZEN["arms"]), required=True)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--only-seed", type=int, default=None)
    ap.add_argument("--shard-dir", default=SHARD_DIR,
                    help="a scratch directory keeps a local function check out of the frozen shards")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes for the CPU-bound r search; the numbers do not depend on it")
    args = ap.parse_args()
    proto = freeze()
    arm = FROZEN["arms"][args.arm]
    c.check_protocol(proto, {"experiment_id": "SCM-7-v3", "mode": "operational_rotate4",
                             "phase": "qualification" if args.arm == "qualification" else "confirmation",
                             "arm": args.arm, "n": arm["n"], "seeds": arm["seeds"]}, PROTOCOL_PATH, SOURCES)
    seeds = arm["seeds"][:args.seeds] if args.seeds else arm["seeds"]
    if args.only_seed is not None:
        if args.only_seed not in arm["seeds"]:
            raise SystemExit(f"seed {args.only_seed} is not in the frozen {args.arm} arm")
        seeds = [args.only_seed]
    facts = s6.truth()
    gate = im.exact_channel_check(n=1500, seed=2468)
    if not gate["exact_channel"]:
        raise SystemExit(f"renderer failed its information gate: {gate['gates']}")
    dev = ft.device()
    Path(args.shard_dir).mkdir(parents=True, exist_ok=True)
    print(f"SCM-7-v3 | {args.arm} | n={arm['n']} | {len(seeds)} seeds | device {dev} | "
          f"protocol {proto[c.DIGEST_FIELD][:12]}", flush=True)
    for seed in seeds:
        shard = Path(args.shard_dir) / f"{args.arm}_seed{seed}_n{arm['n']}.json"
        if shard.exists():
            print(f"  seed {seed}: shard exists, skipping", flush=True)
            continue
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        cell = one_replicate(seed, arm["n"], facts, dev, workers=args.workers)
        cell["protocol"] = {"path": PROTOCOL_PATH, c.DIGEST_FIELD: proto[c.DIGEST_FIELD]}
        cell["source_snapshot"] = c.source_snapshot(SOURCES)
        c.write_json_new(shard, cell)
        print(f"  seed {seed}: {cell['return_kind']} est={cell['estimate']} "
              f"retained={len(cell.get('retained_splits', []))} "
              f"purity={cell.get('largest_component_purity')} "
              f"corr={cell.get('representation_diagnostics', {}).get('channel_correlation_median')} "
              f"gpu={cell.get('peak_gpu_gb')} ({cell['timing_seconds']:.0f}s "
              f"{ {k: round(v) for k, v in cell.get('timing', {}).items()} })", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
