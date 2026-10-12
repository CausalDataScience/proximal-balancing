"""Level A baseline (plan section 8.1): the paper's PROBE, unchanged, on blocks = reader outputs.

Called exactly as the sealed real-data runners call it (run_ihdp_acic.py, rhc_bootstrap.py):

    probe    = family_v2_probe.run_dataset(view, hidden_proxy_recipe.level(d_x, NAME), run_twins.THRESHOLD, seed)
    estimate = noise_scaled.decide(probe, n, seed)["output"]          (the v3 noise-floor rule)

with view = {X, W1..W5, A, Y}.  Level: hidden_proxy_recipe.level(d_x, NAME) is the Level the IHDP/ACIC/Twins
runners use (k = 2, i.e. Z_S = (X, B V_{-S}) with B of rank 2, as section 8.1 asks; its d_blocks field is
only recorded, never read).  Threshold: run_twins.THRESHOLD = 0.0015 (SCM-1's sealed value) for the sealed
rule; the v3 rule needs no threshold.  Inside, PROBE does its own unsupervised preprocessing (each block
with more than two coordinates is cut to its principal components above the Marchenko-Pastur edge, at most
4), its four rotating folds D, S, N, E, its screen, its AIPW and its aggregation.  Nothing is changed.

Comparators on the same rows (run_twins.baselines, the sealed AIPW on the sealed rotations): naive
difference in means, AIPW on X, AIPW on (X, all blocks), AIPW on (X, U) (oracle).

Screen bookkeeping for this phantom: a split is valid when lobe 5 is retained (held-out blocks free of
the device); `noise_scaled.decide` reports TPR/FPR against the family-v2 targets, which do not apply here,
so the valid/invalid counts below are computed separately and `kept_any_w5` is the relevant flag.
"""
from __future__ import annotations

import sys
import time
import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from torch import nn

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import family_v2_probe as pr            # noqa: E402  (sealed)
import hidden_proxy_recipe as hp        # noqa: E402  (sealed)
import noise_scaled as ns               # noqa: E402  (sealed)
import run_twins as rtw                 # noqa: E402  (sealed)

SEED_BASE = 20_261_008
SEALED_SOURCES = ("family_v2_probe.py", "family_v2_dgp.py", "hidden_proxy_recipe.py", "noise_scaled.py",
                  "run_twins.py", "rhc_pooled.py", "probe_structured_scm.py")


def probe_seed(data_seed: int) -> int:
    return SEED_BASE + 1000 * data_seed + rtw.FOLD_SEED_OFFSET


def make_view(X: np.ndarray, blocks: list[np.ndarray], A: np.ndarray, Y: np.ndarray) -> dict:
    if len(blocks) != 5:
        raise ValueError("PROBE's sealed code fixes J = 5 blocks")
    view = {"X": np.asarray(X, float).reshape(len(A), -1), "A": np.asarray(A, float), "Y": np.asarray(Y, float)}
    for j, b in enumerate(blocks):
        view[f"W{j + 1}"] = np.asarray(b, float).reshape(len(A), -1)
    return view


def _valid(name: str) -> bool:
    return "5" not in name


def run_probe(view: dict, seed: int, name: str) -> dict:
    t0 = time.time()
    n = len(view["A"])
    d_x = view["X"].shape[1]
    probe = pr.run_dataset(view, hp.level(d_x, name), rtw.THRESHOLD, seed)
    decision = ns.decide(probe, n, seed)
    names = probe["split_names"]
    kept = set(decision["retained"])
    valid = {nm for nm in names if _valid(nm)}
    per_split = {}
    for nm in names:
        rec = probe["records"][nm]
        g, se = ns.pooled(rec)
        per_split[nm] = {"estimate": rec["estimate"], "valid": _valid(nm),
                         "pooled_gap": g, "pooled_se": se, "pooled_upper": g + ns.Z * se,
                         "sealed_pass_all": rec["pass_all"],
                         "gap_fit": [r["gap_fit"] for r in rec["rotations"]],
                         "screen": [{k: r["screen"][k] for k in ("gap", "se", "upper", "overlap", "pass")}
                                    for r in rec["rotations"]]}
    return {"name": name, "n": n, "seed": seed, "d_x": d_x,
            "feature_mode": "legacy_X" if d_x else "image_only",
            "audit_channel": "oracle_L_plus_true_I_positive_control" if "DEVICE" in name else
                             ("oracle_L" if "ORACLE" in name else "compressed_reader"),
            "critic_probability_range": [pr.LINK_LO, pr.LINK_HI],
            "overlap_scope": "bounded-link automatic flag; not a population overlap certificate",
            "discrepancy_scope": "finite fitted critic risk gap; not a certified upper bound on true D2",
            "block_widths": [int(view[f"W{j + 1}"].shape[1]) for j in range(5)],
            "v3": {"estimate": decision["output"], "return_kind": decision["return_kind"],
                   "floor": decision["floor"], "n_retained": decision["n_retained"],
                   "retained": decision["retained"], "rho": decision["rho"],
                   "component_sizes": decision["component_sizes"],
                   "kept_any_w5": decision["kept_any_w5"],
                   "n_valid_kept": len(kept & valid), "n_invalid_kept": len(kept - valid),
                   "n_valid": len(valid), "n_invalid": len(names) - len(valid)},
            "sealed": {"estimate": probe["estimate"], "return_kind": probe["return_kind"],
                       "retained": probe["retained"], "rho": probe["rho"],
                       "component_sizes": probe["component_sizes"], "threshold": rtw.THRESHOLD},
            "median_all_splits": float(np.median([per_split[nm]["estimate"] for nm in names])),
            "per_split": per_split, "seconds": round(time.time() - t0, 1)}


def comparators(view: dict, U: np.ndarray, seed: int) -> dict:
    """naive, AIPW-X, AIPW-(X, all blocks) ("raw"), AIPW-(X, U) ("oracle"); U is evaluation-only."""
    t0 = time.time()
    b = rtw.baselines(view, U, seed)
    return {"naive": b["naive"], "aipw_X": b["X"], "aipw_X_blocks": b["raw"], "aipw_X_U": b["oracle"],
            "seconds": round(time.time() - t0, 1)}


def sealed_hashes() -> dict:
    import hashlib
    return {f: hashlib.sha256((CODE / f).read_bytes()).hexdigest() for f in SEALED_SOURCES}


def rethreshold(record: dict, threshold: float, rule: str = "all_rotations") -> dict:
    """Reanalyse saved screens. This does not refit critics or produce a new PROBE effect estimate.

    The legacy fitted-overlap flag is automatic because its probability link lies in [.1, .9].
    These are empirical retained sets, not certification of D2, overlap, or a causal effect.
    """
    if threshold <= 0 or rule not in ("all_rotations", "pooled"):
        raise ValueError("positive threshold and all_rotations/pooled rule required")
    retained = []
    for name, rec in record["per_split"].items():
        if rule == "all_rotations":
            passed = all(s["upper"] < threshold and s["overlap"] for s in rec["screen"])
        else:
            passed = rec["pooled_upper"] < threshold and all(s["overlap"] for s in rec["screen"])
        if passed:
            retained.append(name)
    return {"threshold_D2": threshold, "rule": rule, "retained": retained, "n_retained": len(retained),
            "singleton_retained": [s for s in retained if len(s) == 2],
            "S5_retained": "S5" in retained,
            "scope": "legacy saved-result empirical reanalysis; no causal/overlap certification"}


def array_sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def probability_diagnostics(q: np.ndarray, a: np.ndarray) -> dict:
    """Observed prediction quality only. These statistics do not bound Bayes approximation error."""
    q, a = np.asarray(q, float), np.asarray(a, float)
    tails = (q < .05) | (q > .95)
    bins = np.minimum((q * 10).astype(int), 9)
    calibration = sum(np.mean(bins == b) * abs(q[bins == b].mean() - a[bins == b].mean())
                      for b in range(10) if np.any(bins == b))
    return {"brier": float(np.mean((q - a) ** 2)),
            "auc": float(roc_auc_score(a, q)) if len(np.unique(a)) == 2 else None,
            "calibration_bin_absolute_error": float(calibration),
            "range": [float(q.min()), float(q.max())],
            "fraction_outside_05_95": float(tails.mean()), "empirical_all_inside_05_95": bool(~tails.any()),
            "population_overlap_certified": False, "bayes_approximation_error_certified": False}


class _ProbabilityResidual(nn.Module):
    """A sigmoid probability. A zero residual contains the fixed base candidate exactly."""
    def __init__(self, width: int, raw: bool = False):
        super().__init__()
        if raw:
            from .reader import RegionReaderNet
            self.crop_net = RegionReaderNet(8, width=16)
        else:
            self.crop_net = None
        self.head = nn.Sequential(nn.Linear(width + (8 if raw else 0), 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, base_logit, crop=None):
        if self.crop_net is not None:
            features = torch.cat([features, self.crop_net(crop)], dim=1)
        return torch.sigmoid(base_logit + self.head(features).squeeze(1))


def _predict_probability(model, features, base_logit, crop=None) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(features), 512):
            sl = slice(start, start + 512)
            out.append(model(features[sl], base_logit[sl], None if crop is None else crop[sl]).cpu().numpy())
    return np.concatenate(out).astype(float)


def _fit_probability(model, features, base_logit, a, train, val, crop, epochs, seed, patience=6) -> dict:
    """Fit only on train, select checkpoints only on internal val. Scoring rows are never supplied."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=1e-4)
    y = torch.as_tensor(a, dtype=torch.float32)
    initial = _predict_probability(model, features[val], base_logit[val], None if crop is None else crop[val])
    best = float(np.mean((initial - a[val]) ** 2))
    best_state = copy.deepcopy(model.state_dict())
    initial_risk, best_epoch, stale = best, 0, 0
    for ep in range(epochs):
        model.train()
        order = rng.permutation(train)
        for start in range(0, len(order), 128):
            idx = order[start:start + 128]
            q = model(features[idx], base_logit[idx], None if crop is None else crop[idx])
            loss = ((q - y[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        pred = _predict_probability(model, features[val], base_logit[val], None if crop is None else crop[val])
        risk = float(np.mean((pred - a[val]) ** 2))
        if not np.isfinite(risk):
            raise RuntimeError("nonfinite internal validation risk")
        if risk < best - 1e-6:
            best, best_epoch, stale = risk, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return {"epochs_run": ep + 1, "selected_epoch": best_epoch, "initial_val_brier": initial_risk,
            "selected_val_brier": best, "zero_residual_selected": best_epoch == 0}


def _tensor_features(features: np.ndarray, train: np.ndarray) -> torch.Tensor:
    scale = features[train].std(0) + 1e-6
    return torch.as_tensor((features - features[train].mean(0)) / scale, dtype=torch.float32)


def _logit(q: np.ndarray) -> torch.Tensor:
    # Numerical log only. This does not force the probability head into an overlap interval.
    q = np.clip(q, 1e-7, 1 - 1e-7)
    return torch.as_tensor(np.log(q) - np.log1p(-q), dtype=torch.float32)


def fresh_critic_contrast(z: np.ndarray, reader_t: np.ndarray, a: np.ndarray, *, raw_crop=None,
                          oracle_i=None, threshold=.01, seed=0, epochs=30, raw_epochs=60,
                          reference_e=None, reference_p=None) -> dict:
    """Two-way independent scoring at one locked Z and one common q0 per fold.

    Reader q1 includes q0; raw/I q1 includes the fitted reader q1. Each training half contains its own
    80/20 train/validation split. Final scores never select a model. Oracle probabilities are accepted
    only for an explicitly supplied true-L posterior-reference, not for a learned B or reader Z.
    """
    from . import nets
    from .reader import RAW_TEXTURE_FEATURES, raw_texture_features
    nets.setup_torch(2)
    z, reader_t, a = np.asarray(z, float), np.asarray(reader_t, float), np.asarray(a, float)
    if len(a) < 40 or epochs < 1 or raw_epochs < 1:
        raise ValueError("at least 40 scoring rows and positive epochs required")
    if (reference_e is None) != (reference_p is None):
        raise ValueError("reference_e and reference_p must be supplied together")
    variants = ["reader"] + (["raw_image"] if raw_crop is not None else []) + (["oracle_I_positive_control"]
                                                                                 if oracle_i is not None else [])
    texture = raw_texture_features(raw_crop) if raw_crop is not None else None
    q0_all = np.empty(len(a)); q1_all = {name: np.empty(len(a)) for name in variants}
    folds = []
    half = np.arange(len(a)) % 2 == 0
    for fold, (fit, score) in enumerate(((half, ~half), (~half, half))):
        rng = np.random.default_rng(seed + fold)
        fit_rows, score_rows = np.flatnonzero(fit), np.flatnonzero(score)
        perm = rng.permutation(fit_rows); nv = max(4, len(perm) // 5)
        val, train = perm[:nv], perm[nv:]
        zf = _tensor_features(z, train)
        zeros = torch.zeros(len(a), dtype=torch.float32)
        nets.seed_all(seed + 10 * fold)
        q0 = _ProbabilityResidual(zf.shape[1])
        with torch.no_grad():
            rate = np.clip(a[train].mean(), .001, .999)
            q0.head[-1].bias.fill_(math.log(rate / (1 - rate)))
        q0_fit = _fit_probability(q0, zf, zeros, a, train, val, None, epochs, seed + fold)
        base_q = _predict_probability(q0, zf, zeros)
        base_logit = _logit(base_q)
        q0_all[score] = base_q[score]
        ft = _tensor_features(np.column_stack([z, reader_t]), train)
        nets.seed_all(seed + 10 * fold + 1)
        qreader = _ProbabilityResidual(ft.shape[1])
        reader_fit = _fit_probability(qreader, ft, base_logit, a, train, val, None, epochs, seed + fold + 100)
        reader_q = _predict_probability(qreader, ft, base_logit)
        q1_all["reader"][score] = reader_q[score]
        fits = {"reader": reader_fit}
        if raw_crop is not None:
            crop = torch.as_tensor(raw_crop, dtype=torch.float32)
            fr = _tensor_features(np.column_stack([z, reader_t, texture]), train)
            nets.seed_all(seed + 10 * fold + 2)
            raw = _ProbabilityResidual(fr.shape[1], raw=True)
            fits["raw_image"] = _fit_probability(raw, fr, _logit(reader_q), a, train, val, crop,
                                                 raw_epochs, seed + fold + 200, patience=12)
            q1_all["raw_image"][score] = _predict_probability(raw, fr[score], _logit(reader_q)[score], crop[score])
        if oracle_i is not None:
            fi = _tensor_features(np.column_stack([z, reader_t, oracle_i]), train)
            nets.seed_all(seed + 10 * fold + 3)
            qi = _ProbabilityResidual(fi.shape[1])
            fits["oracle_I_positive_control"] = _fit_probability(qi, fi, _logit(reader_q), a, train, val,
                                                                 None, epochs, seed + fold + 300)
            q1_all["oracle_I_positive_control"][score] = _predict_probability(qi, fi[score], _logit(reader_q)[score])
        folds.append({"fold": fold, "n_train": len(train), "n_internal_val": len(val), "n_score": len(score_rows),
                      "train_rows_sha256": array_sha(train), "validation_rows_sha256": array_sha(val),
                      "scoring_rows_sha256": array_sha(score_rows), "q0_fit": q0_fit, "augmented_fits": fits})
    zcrit = float(norm.ppf(1 - .05 / 5))
    contrasts = {}
    for name, q1 in q1_all.items():
        losses = (q0_all - a) ** 2 - (q1 - a) ** 2
        gap, se = float(losses.mean()), float(losses.std(ddof=1) / math.sqrt(len(a)))
        upper = max(gap, 0.) + zcrit * se
        rec = {"q1": probability_diagnostics(q1, a), "q1_prediction_sha256": array_sha(q1),
               "gap_unpenalized_brier": gap, "gap_se": se, "empirical_upper": upper,
               "prediction_disagreement_B2": float(np.mean((q1 - q0_all) ** 2)),
               "empirical_balance_below_threshold": upper < threshold,
               "certified_D2_upper": None, "causal_balance_certified": False,
               "epsilon0_squared": None, "epsilon1_squared": None,
               "status": "empirical_only; Bayes approximation error unavailable"}
        if reference_e is not None:
            e, p = np.asarray(reference_e), np.asarray(reference_p)
            d2 = (p - e) ** 2; eps0 = (q0_all - e) ** 2; eps1 = (q1 - p) ** 2
            residual = losses - d2 - eps0 + eps1
            rec.update({"reference_D2_scoring_mean": float(d2.mean()), "epsilon0_squared": float(eps0.mean()),
                        "epsilon1_squared": float(eps1.mean()), "identity_sampling_residual": float(residual.mean()),
                        "identity_sampling_residual_se": float(residual.std(ddof=1) / math.sqrt(len(a))),
                        "status": "true-L posterior-reference oracle diagnostic only; not learned-B certification"})
        if name == "raw_image" and all(f["augmented_fits"][name]["zero_residual_selected"] for f in folds):
            rec["status"] = "unresolved; raw CNN did not improve internal validation over the reader candidate"
        contrasts[name] = rec
    return {"n": len(a), "threshold_D2": threshold, "zcrit_five_singletons": zcrit,
            "raw_fixed_texture_features": list(RAW_TEXTURE_FEATURES) if texture is not None else None,
            "raw_feature_selection": "predetermined image-only features; no latent/outcome/device label selection",
            "Z_sha256": array_sha(z), "q0_prediction_sha256": array_sha(q0_all),
            "common_Z_rows_q0": True, "scoring_used_for_checkpoint_selection": False,
            "q0": probability_diagnostics(q0_all, a), "folds": folds, "contrasts": contrasts,
            "scope": "one fixed-Z fresh-critic diagnostic; no Stage2 or accepted PROBE output"}


def lock_reader_representation(view: dict, seed: int, max_screen_rows=4000) -> dict:
    """Fit the existing rank-2 encoder once on D for S5, rotation 0, then lock it for every audit channel."""
    rows = pr.folds(len(view["A"]), seed)
    d_rows, s_rows = rows[0], rows[1][:max_screen_rows]
    prep = pr.Prep(view, d_rows)
    blocks_d = [prep.block(view, d_rows, j) for j in range(5)]
    blocks_s = [prep.block(view, s_rows, j) for j in range(5)]
    vd, vs = np.column_stack(blocks_d[:4]), np.column_stack(blocks_s[:4])
    fit = pr.learn_representation(prep.x(view, d_rows), vd, blocks_d[4], view["A"][d_rows], 2, seed + 4)
    z = np.column_stack([prep.x(view, s_rows), vs @ fit["b"].T])
    return {"Z": z, "reader_T": blocks_s[4], "screen_rows": s_rows,
            "metadata": {"split": "S5", "rotation": 0, "D_n": len(d_rows), "S_n": len(s_rows),
                         "D_rows_sha256": array_sha(d_rows), "S_rows_sha256": array_sha(s_rows),
                         "B": fit["b"].tolist(), "B_sha256": array_sha(fit["b"]),
                         "prep_widths": prep.widths(), "legacy_gap_fit": fit["gap_fit"],
                         "encoder_training": "sealed rank-2 bounded-critic baseline, D only; locked afterwards"}}
