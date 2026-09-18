"""SCM-7, development only: the whole pipeline on 46,082-dimensional image proxies.

The latent SCM is byte-identical to SCM-6's: the same U, I, C, X, K, A and Y with the same constants.
Only the rendering differs, and the renderer conserves the first moment exactly, so conditioning on a
block is conditioning on its channel.  The population facts therefore transfer without change, and the
same three criteria can be checked against them: does the selected coefficient land on the balancing
root, does the screen separate the feasible splits from the rest, and does the returned effect approach
the truth.  No confirmation seeds are opened here.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import minimal_suite_common as c
import probe_highdim_w as hd
import probe_highdim_w_v2 as v2
import probe_image_w as im
import run_scm6_v2_development as s6
import scm6_population as sp

SD_K = float(np.sqrt(1.0 + im.PROXY_SD ** 2))

SOURCES = ["minimal_suite_common.py", "probe_image_w.py", "probe_highdim_w_v2.py",
           "scm6_population.py", "run_scm7_image_w.py"]
DEV_SEEDS = [95000000 + 1000 * r for r in range(5)]
TOTAL_N = 12000
# Encoder budget chosen on development seeds before the run.  At 600 fit rows the stopping epoch is
# unstable: one seed stopped at 9 and the channel correlation collapsed to 0.672, another stopped at 30
# and reached 0.945.  At 1,200 fit rows both seeds cleared the gate, at 0.934 and 0.957.  The remaining
# 1,500 rows of the D fold go to the r search, which SCM-6 showed is the step short of data.
EMB_FIT, EMB_CAL = 1200, 300
PENALTY_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
# The target for an image block cannot be its leading principal direction.  Pixel variance is dominated
# by the template and by the independent vertical shift, not by the channel: PC1 correlates 0.93 with the
# style shift and only 0.08 with K, and it takes five components to reach a multiple correlation of 0.89.
# The encoder's own estimate of the held-out channel reaches 0.93 to 0.96 in a single column, so the
# target is that estimate plus a few unsupervised directions to catch what it misses.  Both are functions
# of the held-out block alone, so the check stays a balance check against W_S, compressed.
TARGET_COMPONENTS = 4
TARGET_USES_ENCODER_CHANNEL = True
# The threshold has to sit inside the population separation window and above what this estimator
# actually produces on a balancing split.  For the image design the window is (2.98e-5, 7.71e-3), the
# same as SCM-6's, but the finite-sample statistic is far more inflated: on development rows a feasible
# split reaches 1.9e-3 at its selected coefficient while the best an infeasible split does is 5.1e-3.
# SCM-6 could use 1e-3 because its feasible values sat at 1e-5 to 1e-4, two orders below.  Here that
# would reject balancing splits, so the threshold is 3e-3, still inside the window and above the
# observed feasible values.  Fixed on development rows before the development run.
POPULATION_WINDOW = (2.983e-5, 7.711e-3)
SCREEN_THRESHOLD = 3e-3
ROOT_TOLERANCE = 0.10
CHANNEL_GATE = 0.90
GATES = dict(s6.GATES, channel_correlation_min=CHANNEL_GATE)


class Blocks:
    """Rendered images for one set of rows, produced on demand and cached for the rotation only.

    Holding every block for every unit would be 2.2 GB at this sample size, so nothing is rendered until
    a split actually asks for it.
    """

    def __init__(self, lat: dict, rows: np.ndarray) -> None:
        self.lat, self.rows, self._cache = lat, rows, {}

    def __call__(self, j: int) -> np.ndarray:
        if j not in self._cache:
            self._cache[j] = im.block(self.lat, j, self.rows).astype(np.float32)
        return self._cache[j]

    def numeric(self) -> dict:
        return {k: self.lat[k][self.rows] for k in ("X", "I", "C", "A", "Y")}


def image_target(blocks: Blocks, split: tuple[int, ...], projection: dict | None,
                 channels: np.ndarray | None = None) -> np.ndarray:
    """T_S for image blocks: X, the numeric channels when block 0 is held out, and the compressed images.

    `channels` carries the encoder's estimate of every block, from which the held-out columns are taken.
    """
    num = blocks.numeric()
    cols = [num["X"][:, None]]
    for j in split:
        if j == 0:
            cols.extend([num["I"][:, None], num["C"][:, None]])
        if channels is not None and TARGET_USES_ENCODER_CHANNEL:
            cols.append(channels[:, j][:, None])
        flat = blocks(j).reshape(len(blocks.rows), -1)
        cols.append(flat if projection is None else flat @ projection["directions"][j])
    return np.column_stack(cols)


def fit_image_projection(blocks: Blocks, split: tuple[int, ...], components: int) -> dict:
    """Leading principal directions of each held-out block's pixels, fitted on the D_phi rows only."""
    directions = {}
    for j in split:
        flat = blocks(j).reshape(len(blocks.rows), -1).astype(np.float64)
        centred = flat - flat.mean(0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        directions[j] = vt[:components].T
    return {"directions": directions, "components": components,
            "digest": c.array_sha256(np.concatenate([directions[j].ravel() for j in split]))}


def gap_and_screen(phi: Blocks, k_phi: np.ndarray, s_blocks: Blocks, k_s: np.ndarray,
                   split: tuple[int, ...], proj: dict, penalty: float, seed: int,
                   grid: tuple = c.GRID) -> dict:
    """Cross-fitted selection and the non-negative screen, both against the compressed image target."""
    num_phi, num_s = phi.numeric(), s_blocks.numeric()
    t_phi = image_target(phi, split, proj, k_phi)
    t_s = image_target(s_blocks, split, proj, k_s)
    candidates = [None] if 0 in split else list(grid)
    values = []
    for r in candidates:
        z_raw = hd.representation(k_phi, num_phi, split, r)
        parts = v2.crossfit_folds(len(num_phi["A"]), 5, seed)
        fold_values = []
        for f in range(5):
            te = parts[f]
            tr = np.concatenate([parts[g] for g in range(5) if g != f])
            zs, ts = hd.Scaler(z_raw[tr]), hd.Scaler(t_phi[tr])
            b0, b1 = v2._fit_pair(zs(z_raw[tr]), ts(t_phi[tr]), num_phi["A"][tr], penalty)
            fold_values.append(np.mean(
                hd.brier_rows(b0, zs(z_raw[te]), num_phi["A"][te])
                - hd.brier_rows(b1, np.column_stack([zs(z_raw[te]), ts(t_phi[te])]),
                                num_phi["A"][te])))
        values.append(float(np.mean(fold_values)))
    best = int(np.argmin(values))
    r_hat = candidates[best]
    z_fit = hd.representation(k_phi, num_phi, split, r_hat)
    zs, ts = hd.Scaler(z_fit), hd.Scaler(t_phi)
    b0, b1 = v2._fit_pair(zs(z_fit), ts(t_phi), num_phi["A"], penalty)
    z_s = zs(hd.representation(k_s, num_s, split, r_hat))
    q = v2._propensity(b1, np.column_stack([z_s, ts(t_s)]))
    e = v2._propensity(b0, z_s)
    d = (q - e) ** 2
    return {"r": r_hat, "objective": values[best],
            "objective_vector_sha256": c.array_sha256(np.asarray(values)),
            "screen_d2": float(d.mean()), "screen_se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "overlap": [float(e.min()), float(e.max())], "target_digest": proj["digest"],
            "pass": bool(d.mean() <= SCREEN_THRESHOLD and e.min() >= c.OVERLAP_RANGE[0]
                         and e.max() <= c.OVERLAP_RANGE[1])}


def noise_from_correlation(corr: float) -> float:
    """Independent channel error implied by a measured correlation between K-hat and K.

    corr = sd(K) / sqrt(sd(K)^2 + sd(error)^2), so sd(error) = sd(K) sqrt(1/corr^2 - 1).
    """
    corr = min(max(corr, 1e-6), 1.0)
    return 0.0 if corr >= 1.0 else SD_K * float(np.sqrt(1.0 / corr ** 2 - 1.0))


def adjusted_truth(corr: float) -> dict:
    """Balancing roots for an encoder of this quality, not for a perfect one.

    A recovered channel carrying independent error is a noisier measure of the latent, and that moves the
    coefficient at which a split balances.  At correlation 0.98 the shift is at most 0.016, but at 0.93 it
    reaches 0.063, which is a large fraction of the tolerance the gate uses.  The gate therefore measures
    against the root this encoder can actually reach, and the exact-channel root is reported alongside.
    """
    noise = noise_from_correlation(corr)
    out = {}
    for split in c.oriented_splits():
        root = sp.operative_root(split, channel_noise=noise)
        exact = sp.operative_root(split)
        out[c.split_id(split)] = {"split": split, "root": root, "exact_channel_root": exact,
                                  "feasible": root is not None,
                                  "channel_noise": noise,
                                  "shift": None if (root is None or exact is None)
                                  else abs(root - exact)}
    return out


def comparator_channels(lat: dict, enc: dict, emb_fit: np.ndarray, emb_cal: np.ndarray,
                       rows: np.ndarray, pcs: dict) -> dict:
    """Channel estimates from each comparator, all fitted on the same embedding rows as the network."""
    out = {"cnn": im.apply_encoder(enc, lat, rows)}
    stat_cols, pca_cols = [], []
    for j in range(im.N_BLOCKS):
        stats_rows = im.image_statistics(im.block(lat, j, rows))
        fitted = pcs["statistics"][j]
        stat_cols.append(np.full(len(rows), np.nan) if fitted is None
                         else im.apply_statistic_channel(fitted, stats_rows))
        flat = im.block(lat, j, rows).reshape(len(rows), -1)
        pca_cols.append((flat - pcs["pca_mean"][j]) @ pcs["pca_dir"][j])
    out["statistics"] = np.column_stack(stat_cols)
    out["pca"] = np.column_stack(pca_cols)
    return out


def fit_comparators(lat: dict, emb_fit: np.ndarray, emb_cal: np.ndarray) -> dict:
    """Every comparator gets the same fit and calibration rows the network head got."""
    statistics, pca_dir, pca_mean = [], [], []
    for j in range(im.N_BLOCKS):
        fit_img, cal_img = im.block(lat, j, emb_fit), im.block(lat, j, emb_cal)
        statistics.append(im.fit_statistic_channel(im.image_statistics(fit_img), lat["X"][emb_fit],
                                                   im.image_statistics(cal_img), lat["X"][emb_cal]))
        flat = fit_img.reshape(len(emb_fit), -1)
        mean = flat.mean(0)
        _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
        pca_dir.append(vt[0])
        pca_mean.append(mean)
    return {"statistics": statistics, "pca_dir": pca_dir, "pca_mean": pca_mean}


def comparator_features(lat: dict, rows: np.ndarray, channels: dict) -> dict:
    """Feature maps the comparators adjust for, all built from observed data except the oracle."""
    x, i, cc = lat["X"][rows][:, None], lat["I"][rows][:, None], lat["C"][rows][:, None]
    out = {"X": x, "oracle": np.column_stack([x, lat["U_eval_only"][rows], cc])}
    for name in ("cnn", "statistics", "pca"):
        block = channels[name]
        if np.isfinite(block).all():
            out[f"{name}_channels"] = np.column_stack([x, cc, block])
    return out


def one_replicate(seed: int, facts: dict) -> dict:
    t0 = time.time()
    lat = im.latents(TOTAL_N, seed)
    folds = c.four_folds(TOTAL_N, seed + 1)
    splits = c.oriented_splits()
    rows_out, corrs = [], []
    scores = {sp_: [] for sp_ in splits}
    passes = {sp_: 0 for sp_ in splits}
    base_scores: dict[str, list] = {}
    comparator_corr: dict[str, list] = {}
    for rot in c.rotations(folds):
        c.assert_roles_disjoint(rot.roles)
        d_idx = rot.roles["D"]
        enc = im.fit_encoder(lat, d_idx[:EMB_FIT], d_idx[EMB_FIT:EMB_FIT + EMB_CAL], seed=seed)
        if enc["fail_closed"]:
            # One rotation is enough to abstain, which is what the algorithm specifies, but the record
            # has to say which rotation and which block so the cause is visible afterwards.
            return {"data_seed": seed, "return_kind": "no_return", "estimate": None,
                    "fail_closed": {"rotation": rot.index, "reason": enc["fail_closed"],
                                    "rotations_completed": rot.index},
                    "complete": True, "timing_seconds": time.time() - t0}
        rotation_corr = im.channel_correlation(enc, lat, rot.roles["E"])
        corrs.append(rotation_corr)
        # The truth this rotation can reach depends on the encoder it just fitted.
        rot_facts = adjusted_truth(float(np.median(rotation_corr)))
        phi_idx = d_idx[EMB_FIT + EMB_CAL:]
        phi, s_blocks = Blocks(lat, phi_idx), Blocks(lat, rot.roles["S"])
        k_phi, k_s = im.apply_encoder(enc, lat, phi_idx), im.apply_encoder(enc, lat, rot.roles["S"])
        k_n = im.apply_encoder(enc, lat, rot.roles["N"])
        k_e = im.apply_encoder(enc, lat, rot.roles["E"])
        num_n = {k: lat[k][rot.roles["N"]] for k in ("X", "I", "C", "A", "Y")}
        num_e = {k: lat[k][rot.roles["E"]] for k in ("X", "I", "C", "A", "Y")}
        fitted_cmp = fit_comparators(lat, d_idx[:EMB_FIT], d_idx[EMB_FIT:EMB_FIT + EMB_CAL])
        ch_n = comparator_channels(lat, enc, d_idx[:EMB_FIT], d_idx[EMB_FIT:EMB_FIT + EMB_CAL],
                                   rot.roles["N"], fitted_cmp)
        ch_e = comparator_channels(lat, enc, d_idx[:EMB_FIT], d_idx[EMB_FIT:EMB_FIT + EMB_CAL],
                                   rot.roles["E"], fitted_cmp)
        for name, block in ch_e.items():
            if np.isfinite(block).all():
                comparator_corr.setdefault(name, []).append(float(np.median(
                    [abs(np.corrcoef(block[:, j], lat["K_eval_only"][rot.roles["E"], j])[0, 1])
                     for j in range(im.N_BLOCKS)])))
        feat_n = comparator_features(lat, rot.roles["N"], ch_n)
        feat_e = comparator_features(lat, rot.roles["E"], ch_e)
        for name in feat_n:
            if name in feat_e:
                base_scores.setdefault(name, []).append(hd.aipw_from_features(
                    feat_n[name], num_n["A"], num_n["Y"], feat_e[name], num_e["A"], num_e["Y"]))
        for sp_ in splits:
            sid = c.split_id(sp_)
            oseed = c.optimizer_seed("scm7-image-development", seed, rot.index, sid, 0)
            proj = fit_image_projection(phi, sp_, TARGET_COMPONENTS)
            z_ref = hd.representation(k_phi, phi.numeric(), sp_, None if 0 in sp_ else 0.0)
            pen = hd.choose_penalty(lambda: hd.Scaler(z_ref)(z_ref), phi.numeric()["A"],
                                    PENALTY_GRID, oseed)
            out = gap_and_screen(phi, k_phi, s_blocks, k_s, sp_, proj, pen, oseed)
            fact = rot_facts[sid]
            rows_out.append({
                "rotation": rot.index, "split_id": sid, "r": out["r"], "penalty": pen,
                "objective": out["objective"], "screen_d2": out["screen_d2"],
                "screen_se": out["screen_se"], "screen_pass": out["pass"],
                "overlap": out["overlap"], "target_digest": out["target_digest"],
                "evaluator_only": {"balance_root": fact["root"], "feasible": fact["feasible"],
                                   "exact_channel_root": fact["exact_channel_root"],
                                   "channel_noise": fact["channel_noise"],
                                   "root_shift_from_exact": fact["shift"],
                                   "root_error": (None if fact["root"] is None or out["r"] is None
                                                  else abs(out["r"] - fact["root"])),
                                   "root_error_against_exact_channel":
                                       (None if fact["exact_channel_root"] is None or out["r"] is None
                                        else abs(out["r"] - fact["exact_channel_root"]))}})
            if out["pass"]:
                passes[sp_] += 1
                scores[sp_].append(hd.aipw_from_features(
                    hd.representation(k_n, num_n, sp_, out["r"]), num_n["A"], num_n["Y"],
                    hd.representation(k_e, num_e, sp_, out["r"]), num_e["A"], num_e["Y"]))
        del phi, s_blocks

    retained = [sp_ for sp_ in splits if passes[sp_] == 4]
    est = {c.split_id(sp_): float(np.concatenate(scores[sp_]).mean()) for sp_ in retained}
    if retained:
        mat = np.column_stack([np.concatenate(scores[sp_]) for sp_ in retained])
        rho = float(s6.v1.ps_common_radius(mat, seed + 5))
        agg = s6.v1.hd_aggregate([c.split_id(sp_) for sp_ in retained],
                                 np.array([est[c.split_id(sp_)] for sp_ in retained]), rho)
    else:
        rho, agg = None, {"return_kind": "no_return", "output": None, "members": []}
    baselines = {k: float(np.concatenate(v).mean()) for k, v in base_scores.items()}
    baselines["naive"] = float(lat["Y"][lat["A"] == 1].mean() - lat["Y"][lat["A"] == 0].mean())
    return {"data_seed": seed, "rows": rows_out, "baselines": baselines,
            "comparator_channel_correlation": {k: float(np.median(v))
                                               for k, v in comparator_corr.items()},
            "representation_diagnostics": {"channel_correlation_per_rotation": corrs,
                                           "channel_correlation_median": float(np.median(
                                               np.array(corrs)))},
            "retained_splits": [c.split_id(s) for s in retained], "candidate_estimates": est,
            "rho": rho, "return_kind": agg["return_kind"], "estimate": agg["output"],
            "complete": True, "timing_seconds": time.time() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(DEV_SEEDS))
    ap.add_argument("--out", default="../results/scm7_image_development_v1.json")
    args = ap.parse_args()
    facts = s6.truth()
    gate = im.exact_channel_check(n=1500, seed=2468)
    if not gate["exact_channel"]:
        raise SystemExit(f"the renderer failed its information gate: {gate['gates']}")
    print(f"SCM-7 | development only | raw W dim {2 + 5 * im.CANVAS ** 2} | n={TOTAL_N} | "
          f"{args.seeds} seeds | encoder {EMB_FIT} fit rows, early stopping")
    root = {"schema_version": c.SCHEMA_VERSION, "experiment_id": "SCM-7", "phase": "development",
            "mode": "operational_rotate4", "total_n": TOTAL_N, "seeds": DEV_SEEDS[:args.seeds],
            "emb_fit": EMB_FIT, "emb_cal": EMB_CAL, "train": dict(im.TRAIN),
            "screen_threshold": SCREEN_THRESHOLD, "population_window": list(POPULATION_WINDOW),
            "root_tolerance": ROOT_TOLERANCE,
            "target_components": TARGET_COMPONENTS,
            "target_uses_encoder_channel": TARGET_USES_ENCODER_CHANNEL, "gates": GATES,
            "renderer_gate": gate, "mnist_sha256": im.MNIST_SHA256,
            "population_truth_exact_channel": {k: {kk: vv for kk, vv in v.items() if kk != "split"}
                                               for k, v in facts.items()},
            "truth_rule": "root recovery is measured against the root implied by the encoder's own "
                          "measured channel correlation in that rotation; the exact-channel root is "
                          "recorded next to it",
            "source_snapshot": c.source_snapshot(SOURCES), "environment": c.environment(),
            "cells": [], "complete": False}
    for seed in DEV_SEEDS[:args.seeds]:
        cell = one_replicate(seed, facts)
        root["cells"].append(cell)
        got = [r for r in cell.get("rows", []) if r["evaluator_only"]["root_error"] is not None]
        near = sum(r["evaluator_only"]["root_error"] <= ROOT_TOLERANCE for r in got)
        corr = cell.get("representation_diagnostics", {}).get("channel_correlation_median")
        print(f"  seed {seed}: {cell['return_kind']} est={cell['estimate']} "
              f"retained={len(cell.get('retained_splits', []))} corr={corr} "
              f"root recovery {near}/{len(got)} ({cell['timing_seconds']:.0f}s)", flush=True)
    summary = s6.score(root["cells"], facts)
    corr = [cell["representation_diagnostics"]["channel_correlation_median"]
            for cell in root["cells"] if "representation_diagnostics" in cell]
    summary["channel_correlation_median"] = float(np.median(corr)) if corr else float("nan")
    summary["gates"]["channel_correlation"] = summary["channel_correlation_median"] >= CHANNEL_GATE
    summary["verdict"] = "PASS" if all(summary["gates"].values()) else "FAIL"
    summary["confirmation_may_open"] = summary["verdict"] == "PASS"
    root["summary"], root["complete"] = summary, True
    c.write_json_new(args.out, root)
    print(f"\nchannel corr    {summary['channel_correlation_median']:.3f}  (gate >= {CHANNEL_GATE})")
    print(f"root recovery   {summary['root_recovery_rate']:.3f}  "
          f"(gate >= {GATES['root_recovery_rate_min']})")
    print(f"screen TPR      {summary['screen_tpr']:.3f} | FPR {summary['screen_fpr']:.3f}")
    print(f"mean |error|    {summary['mean_abs_error']:.4f}  (gate <= {GATES['mean_abs_error_max']})")
    print(f"verdict {summary['verdict']} | confirmation may open: {summary['confirmation_may_open']}")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
