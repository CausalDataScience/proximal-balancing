"""Image-only local pipeline. Training, independent audit and candidate estimation are separate.

U/L/I truths are used only by explicitly marked evaluation controls, never encoder or ordinary critics.
The original held-out pixels remain available to the augmented critic after brightness correction.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.stats import norm
from sklearn.decomposition import FactorAnalysis
from torch import nn

from . import folds, level_a, nets, oracle, phantom as ph, reader as rd, regions as rg, stage1

PRIMARY = .01
BUDGETS = (.005, .01, .02)
FAMILY_SIZE = 12
ALPHA = .05


def body_reference_mask() -> np.ndarray:
    mask = binary_erosion(rg.body_mask() & ~rg.lung_mask(), iterations=2)
    if not mask.any() or np.any(mask & (rg.masks().sum(0) > 0)):
        raise AssertionError("reference must be nonempty and disjoint from every held-out region")
    return mask


def body_offset(images: np.ndarray, body_base: float = .55) -> np.ndarray:
    return (images[:, body_reference_mask()].mean(1) - body_base).astype(np.float32)[:, None]


def corrected_images(images: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return (images - offset[:, :, None]).astype(np.float32)


def overlap_hinge(q: torch.Tensor) -> torch.Tensor:
    return torch.relu(q - .95).square() + torch.relu(.05 - q).square()


def finite_screen(gap: float, se: float, n_s: int, pi_mse: float, m: int,
                  family_size: int = FAMILY_SIZE, alpha: float = ALPHA) -> dict:
    """Hoeffding/union bound for independent S and fresh MC, conditional on all fitted models.

    Loss difference has range [-1,1]; squared pi error has range [0,1]. Pi MSE bounds actual epsilon1^2
    conservatively and includes latent uncertainty. It is not the exact approximation error.
    """
    logterm = math.log(2 * family_size / alpha)
    radius_s = math.sqrt(2 * logterm / n_s)
    radius_pi = math.sqrt(logterm / (2 * m))
    b1 = min(1., pi_mse + radius_pi)
    finite_upper = min(1., max(gap, 0.) + radius_s + b1)
    normal_upper = max(gap, 0.) + float(norm.ppf(1 - alpha / family_size)) * se
    return {"normal_empirical_upper": normal_upper, "finite_risk_radius": radius_s,
            "fresh_pi_mse": pi_mse, "pi_sampling_radius": radius_pi, "b1_conservative_upper": b1,
            "b1_scope": "actual frozen q1 versus latent propensity; includes irreducible latent uncertainty",
            "finite_D2_upper": finite_upper, "family_size": family_size, "alpha": alpha,
            "finite_radius_alone_exceeds_primary": radius_s + radius_pi > PRIMARY,
            "cutoffs": {str(t): {"empirical_gap_pass": normal_upper <= t,
                                  "finite_discrepancy_pass": finite_upper <= t} for t in BUDGETS},
            "uniform_overlap_certified": False, "causal_identification_verified": False}


def tensor_state_sha(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def prediction_risks(q0, q1, a) -> dict:
    losses = (q0 - a) ** 2 - (q1 - a) ** 2
    return {"q0": level_a.probability_diagnostics(q0, a), "q1": level_a.probability_diagnostics(q1, a),
            "gap": float(losses.mean()), "se": float(losses.std(ddof=1) / math.sqrt(len(a))),
            "B2": float(np.mean((q1 - q0) ** 2))}


@dataclass
class ImageFrame:
    raw: np.ndarray
    processed: np.ndarray
    reference: np.ndarray
    psi: list[np.ndarray]
    chi: np.ndarray


@dataclass
class Mode:
    name: str
    fit: dict
    frame: ImageFrame
    raw_stats: tuple[float, float]
    ref_stats: tuple[float, float]
    body_base: float
    report: dict

    def new_frame(self, images: np.ndarray) -> ImageFrame:
        offset = body_offset(images, self.body_base)
        processed = images if self.name == "raw" else corrected_images(images, offset)
        reference = (offset - self.ref_stats[0]) / self.ref_stats[1]
        psi = self.fit["bank"].transform(processed, torch.device("cpu"))
        return ImageFrame(images, processed, reference, psi, self.fit["summaries"].transform(processed))

    def crop(self, frame: ImageFrame, rows: np.ndarray, split: int) -> np.ndarray:
        return rg.region_crops(nets.standardise(frame.raw[rows], self.raw_stats), split + 1)


def _ridge_prediction_score(features, target, train, score) -> dict:
    mean, scale = features[train].mean(0), features[train].std(0) + 1e-6
    fp = np.column_stack([np.ones(len(train)), (features[train] - mean) / scale])
    fs = np.column_stack([np.ones(len(score)), (features[score] - mean) / scale])
    coef = level_a.pr.ridge_ols(fp, target[train], 1e-3)
    pred = fs @ coef
    r2 = 1 - np.mean((pred - target[score]) ** 2) / np.var(target[score])
    return {"R2": float(r2), "n_fit": len(train), "n_score": len(score),
            "scope": "truth-supervised evaluation-only regression; never an input to PROBE"}


def prepare_mode(data, roles: dict, name: str, seed: int, reader_epochs: int, mae_epochs: int, log) -> Mode:
    start = time.time()
    offset = body_offset(data.images, data.config.body_base)
    r = roles["R"]
    ref_stats = float(offset[r].mean()), float(offset[r].std() + 1e-6)
    processed = data.images if name == "raw" else corrected_images(data.images, offset)
    log(f"stage1[{name}]: R={len(r)}; images only")
    fit = stage1.run_stage1(processed[r], seed, torch.device("cpu"), reader_epochs=reader_epochs,
                           mae_epochs=mae_epochs, n_boot=30, auto_raise=False, log=log)
    psi = fit["bank"].transform(processed)
    frame = ImageFrame(data.images, processed, (offset - ref_stats[0]) / ref_stats[1], psi,
                       fit["summaries"].transform(processed))
    p, s = roles["P"], roles["S"]
    f = np.column_stack(psi)
    severity = _ridge_prediction_score(f, data.U, p, s)
    brightness = _ridge_prediction_score(f, data.G, p, s)
    true_l = _ridge_prediction_score(data.L, data.U, p, s)
    pcs_r, _ = rd.first_pcs([block[r] for block in psi], [frame.chi[r, k, 0] for k in range(5)])
    fa = FactorAnalysis(n_components=2, random_state=seed).fit(pcs_r)
    report = {"mode": name, "R_rows_sha256": level_a.array_sha(r), "stage1_seconds": time.time() - start,
              "reader": fit["bank"].report, "mae": {k: v for k, v in fit["mae"].items() if k != "model"},
              "one_factor_R": fit["factor_R"], "two_factor_R": {"loadings": fa.components_.tolist(),
                  "noise_variance": fa.noise_variance_.tolist(), "scope": "diagnostic; factors are not identified as U/G"},
              "reader_features_linear_U_R2_evaluation_only": severity,
              "reader_features_linear_G_R2_evaluation_only": brightness,
              "true_L_to_U_evaluation_only": true_l,
              "true_L_R2_ceiling": 5 / (5 + data.config.sigma ** 2),
              "reference_pixels": int(body_reference_mask().sum()), "reference_in_Z": True,
              "raw_T_preserved": True, "proxy_ORC_or_stability_certified": False}
    log(f"stage1[{name}] complete: reader-features→U linear R2={severity['R2']:.4f}; trueL R2={true_l['R2']:.4f}")
    return Mode(name, fit, frame, (float(data.images[r].mean()), float(data.images[r].std())),
                ref_stats, data.config.body_base, report)


class LinearImageEncoder:
    def __init__(self, mode: Mode, rows: np.ndarray, a: np.ndarray, split: int, seed: int):
        self.split = split
        self.keep = [j for j in range(5) if j != split]
        frame = mode.frame
        view = level_a.make_view(np.empty((len(a), 0)), frame.psi, a, np.zeros(len(a)))
        self.prep = level_a.pr.Prep(view, rows)
        blocks = [self.prep.block(view, rows, j) for j in range(5)]
        fit = level_a.pr.learn_representation(frame.reference[rows], np.column_stack([blocks[j] for j in self.keep]),
                                             blocks[split], a[rows], 2, seed)
        self.b = fit["b"]
        self.report = {"kind": "legacy rank2 search; fresh proper critics used only for the new audit",
                       "D_rows_sha256": level_a.array_sha(rows), "B": self.b.tolist(),
                       "B_sha256": level_a.array_sha(self.b), "reference_in_Z": True,
                       "legacy_training_gap": fit["gap_fit"]}

    def encode(self, frame: ImageFrame, rows: np.ndarray) -> np.ndarray:
        # Same frozen preprocessing also transforms fresh patient arrays.
        scores = []
        for j in self.keep:
            mu, proj, scale = self.prep.blocks[j]
            scores.append(((frame.psi[j][rows] - mu) @ proj) / scale)
        return np.column_stack([frame.reference[rows], np.column_stack(scores) @ self.b.T])


class NeuralImageEncoder:
    def __init__(self, mode: Mode, split: int, seed: int, init: str = "mae"):
        nets.seed_all(seed)
        self.mode, self.split = mode, split
        if init not in ("mae", "random"):
            raise ValueError("init must be mae or random")
        original = mode.fit["mae"]["model"].trunk
        trunk = copy.deepcopy(original) if init == "mae" else nets.ConvTrunk(original.width)
        self.model = nets.RegionEncoder(trunk, d_out=2)
        self.init = init
        self.stats = mode.fit["mae"]["stats"]

    def view(self, frame: ImageFrame, rows: np.ndarray) -> np.ndarray:
        return rg.holdout_view(nets.standardise(frame.processed[rows], self.stats), (self.split + 1,))

    def forward(self, frame: ImageFrame, rows: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(self.view(frame, rows))
        return torch.cat([torch.as_tensor(frame.reference[rows], dtype=torch.float32), self.model(x)], dim=1)

    def encode(self, frame: ImageFrame, rows: np.ndarray) -> np.ndarray:
        self.model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(rows), 256):
                out.append(self.forward(frame, rows[start:start + 256]).numpy())
        return np.concatenate(out)


class CriticBundle:
    """Fitted q0, nested reader q1 and raw q1. All feature scales come from fit-training rows only."""
    def __init__(self, z, t, crop, a, seed, epochs, raw_epochs):
        n = len(a)
        order = np.random.default_rng(seed).permutation(n)
        nv = max(4, n // 5); self.val, self.train = order[:nv], order[nv:]
        texture = rd.raw_texture_features(crop)
        self.scales = []
        arrays = (z, np.column_stack([z, t]), np.column_stack([z, t, texture]))
        tensors = []
        for arr in arrays:
            mu, sd = arr[self.train].mean(0), arr[self.train].std(0) + 1e-6
            self.scales.append((mu.astype(np.float32), sd.astype(np.float32)))
            tensors.append(torch.as_tensor((arr - mu) / sd, dtype=torch.float32))
        f0, fr, fraw = tensors
        zeros = torch.zeros(n)
        nets.seed_all(seed)
        self.q0 = level_a._ProbabilityResidual(f0.shape[1])
        with torch.no_grad():
            rate = np.clip(a[self.train].mean(), .001, .999)
            self.q0.head[-1].bias.fill_(math.log(rate / (1 - rate)))
        report0 = level_a._fit_probability(self.q0, f0, zeros, a, self.train, self.val, None, epochs, seed)
        q0 = level_a._predict_probability(self.q0, f0, zeros)
        nets.seed_all(seed + 1)
        self.reader = level_a._ProbabilityResidual(fr.shape[1])
        reportr = level_a._fit_probability(self.reader, fr, level_a._logit(q0), a, self.train, self.val,
                                          None, epochs, seed + 1)
        qr = level_a._predict_probability(self.reader, fr, level_a._logit(q0))
        nets.seed_all(seed + 2)
        self.raw = level_a._ProbabilityResidual(fraw.shape[1], raw=True)
        reportraw = level_a._fit_probability(self.raw, fraw, level_a._logit(qr), a, self.train, self.val,
                                            torch.as_tensor(crop), raw_epochs, seed + 2, patience=12)
        self.report = {"q0": report0, "reader": reportr, "raw": reportraw,
                       "n_train": len(self.train), "n_internal_val": len(self.val),
                       "train_rows_sha256": level_a.array_sha(self.train),
                       "internal_val_rows_sha256": level_a.array_sha(self.val),
                       "scoring_rows_used": False}
        self.freeze()

    def freeze(self):
        for module in (self.q0, self.reader, self.raw):
            module.eval()
            module.requires_grad_(False)
            for parameter in module.parameters():
                parameter.grad = None

    def tensor_predict(self, z, t, texture, crop):
        fr = torch.cat([z, t], 1); fw = torch.cat([z, t, texture], 1)
        features = []
        for f, (mu, sd) in zip((z, fr, fw), self.scales):
            features.append((f - torch.as_tensor(mu)) / torch.as_tensor(sd))
        f0, fr, fw = features
        q0 = self.q0(f0, torch.zeros(len(f0)))
        # Match the numerical logit clamp used by _fit_probability exactly.
        # This is (0,1) stabilization, not the [.05,.95] overlap criterion.
        qr = self.reader(fr, torch.logit(q0.clamp(1e-7, 1 - 1e-7)))
        q1 = self.raw(fw, torch.logit(qr.clamp(1e-7, 1 - 1e-7)), crop)
        return q0, q1, qr

    def predict(self, z, t, crop):
        texture = rd.raw_texture_features(crop)
        out = [[], [], []]
        with torch.no_grad():
            for start in range(0, len(z), 512):
                sl = slice(start, start + 512)
                values = self.tensor_predict(torch.as_tensor(z[sl], dtype=torch.float32),
                    torch.as_tensor(t[sl], dtype=torch.float32), torch.as_tensor(texture[sl]), torch.as_tensor(crop[sl]))
                for dst, value in zip(out, values):
                    dst.append(value.numpy())
        return tuple(np.concatenate(v).astype(float) for v in out)


def fit_for_rows(mode, encoder, frame, rows, a, seed, epochs, raw_epochs) -> CriticBundle:
    return CriticBundle(encoder.encode(frame, rows), frame.psi[encoder.split][rows],
                        mode.crop(frame, rows, encoder.split), a[rows], seed, epochs, raw_epochs)


def evaluate_rows(mode, encoder, bundle, frame, rows, a, independent=True) -> dict:
    z = encoder.encode(frame, rows)
    q0, q1, qr = bundle.predict(z, frame.psi[encoder.split][rows], mode.crop(frame, rows, encoder.split))
    rec = prediction_risks(q0, q1, a[rows])
    rec["reader_only"] = prediction_risks(q0, qr, a[rows])
    rec.update({"q0_prediction_sha256": level_a.array_sha(q0), "q1_prediction_sha256": level_a.array_sha(q1),
                "Z_sha256": level_a.array_sha(z), "patient_rows_sha256": level_a.array_sha(rows),
                "epoch_selection": bundle.report, "fit_to_score_independent": independent,
                "epsilon1_exact": None, "scope": "observed raw-image audit; not exact Bayes error"})
    return rec


def oracle_i_positive_control(mode, encoder, bundle, frame, roles, data, seed, epochs) -> dict:
    if encoder.split != 4:
        return {"run": False, "reason": "true I is outside the held-out image; no oracle-I input allowed"}
    p, s = roles["P"], roles["S"]
    z = encoder.encode(frame, np.arange(len(data.A)))
    crop = mode.crop(frame, p, 4)
    _, base, _ = bundle.predict(z[p], frame.psi[4][p], crop)
    _, base_s, _ = bundle.predict(z[s], frame.psi[4][s], mode.crop(frame, s, 4))
    a = data.A[p]
    features = np.column_stack([z[p], frame.psi[4][p], data.I[p]])
    mean, scale = features[bundle.train].mean(0), features[bundle.train].std(0) + 1e-6
    ft = torch.as_tensor((features - mean) / scale, dtype=torch.float32)
    nets.seed_all(seed)
    net = level_a._ProbabilityResidual(ft.shape[1])
    report = level_a._fit_probability(net, ft, level_a._logit(base), a, bundle.train, bundle.val,
                                     None, epochs, seed)
    fs = torch.as_tensor((np.column_stack([z[s], frame.psi[4][s], data.I[s]]) - mean) / scale,
                         dtype=torch.float32)
    pred = level_a._predict_probability(net, fs, level_a._logit(base_s))
    q0, _, _ = bundle.predict(z[s], frame.psi[4][s], mode.crop(frame, s, 4))
    return {"run": True, "scope": "true heldout I positive control only; empirical, not the production predictor",
            "scores": prediction_risks(q0, pred, data.A[s]), "fit": report}


def run_fixed_audits(mode, data, roles, seed, epochs, raw_epochs, log):
    records, models = {}, []
    for split in range(5):
        name = f"S{split + 1}"
        encoder = LinearImageEncoder(mode, roles["D"], data.A, split, seed + split)
        bundle = fit_for_rows(mode, encoder, mode.frame, roles["P"], data.A, seed + 100 + split, epochs, raw_epochs)
        rec = evaluate_rows(mode, encoder, bundle, mode.frame, roles["S"], data.A)
        rec["encoder"] = encoder.report
        rec["oracle_I_evaluation_only"] = oracle_i_positive_control(mode, encoder, bundle, mode.frame, roles,
                                                                   data, seed + 200 + split, epochs)
        records[name] = rec; models.append((f"{mode.name}_{name}", mode, encoder, bundle, rec))
        log(f"raw audit[{mode.name}/{name}]: gap={rec['gap']:.6f}, SE={rec['se']:.6f}, "
            f"q1 Brier={rec['q1']['brier']:.6f}, tail={rec['q0']['fraction_outside_05_95']:.3%}")
    return records, models


def train_alternating(mode, data, roles, seed, rounds, encoder_steps, epochs, raw_epochs, log,
                      *, split=0, init="mae", enforce_tail_gate=True):
    """Every B update is followed by new-phi critic refitting before C/dual/checkpoint decisions."""
    encoder = NeuralImageEncoder(mode, split, seed, init)
    decoder = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 80))
    optimizer = torch.optim.Adam(list(encoder.model.parameters()) + list(decoder.parameters()), lr=2e-4)
    frame = mode.frame
    d, c = roles["D"], roles["C"]
    texture_d = rd.raw_texture_features(mode.crop(frame, d, split))
    crops_d = mode.crop(frame, d, split)
    rng = np.random.default_rng(seed)
    dual, mu = 10., 10.
    history, suitable_best, diagnostic_best, stable = [], None, None, 0
    bundle = fit_for_rows(mode, encoder, frame, d, data.A, seed + 10, epochs, raw_epochs)
    for iteration in range(rounds):
        encoder.model.train(); decoder.train()
        before = tensor_state_sha(encoder.model)
        losses = []
        for step in range(encoder_steps):
            local = rng.choice(len(d), size=min(128, len(d)), replace=False)
            rows = d[local]
            z = encoder.forward(frame, rows)
            q0, q1, _ = bundle.tensor_predict(z, torch.as_tensor(frame.psi[split][rows], dtype=torch.float32),
                                            torch.as_tensor(texture_d[local]), torch.as_tensor(crops_d[local]))
            target = torch.as_tensor(frame.chi[rows].reshape(len(rows), -1), dtype=torch.float32)
            keep = (decoder(z) - target).square().mean()
            a = torch.as_tensor(data.A[rows], dtype=torch.float32)
            gap = ((q0 - a).square() - (q1 - a).square()).mean()
            overlap = overlap_hinge(q0).mean()
            loss = keep + dual * (gap - PRIMARY) + mu * overlap
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.model.parameters()) + list(decoder.parameters()), 5.)
            optimizer.step(); losses.append(float(loss.detach()))
        # Mandatory new-phi refit; no checkpoint sees a stale-critic gap.
        bundle = fit_for_rows(mode, encoder, frame, d, data.A, seed + 1000 + iteration, epochs, raw_epochs)
        scored_d = evaluate_rows(mode, encoder, bundle, frame, d, data.A, independent=False)
        scored = evaluate_rows(mode, encoder, bundle, frame, c, data.A)
        encoder.model.eval(); decoder.eval()
        with torch.no_grad():
            zc = torch.as_tensor(encoder.encode(frame, c), dtype=torch.float32)
            keep_c = float((decoder(zc) - torch.as_tensor(frame.chi[c].reshape(len(c), -1))).square().mean())
        tail = scored["q0"]["fraction_outside_05_95"]
        eligible = scored["gap"] <= PRIMARY and (not enforce_tail_gate or tail <= .01)
        item = {"round": iteration + 1, "encoder_steps": encoder_steps, "encoder_changed": before != tensor_state_sha(encoder.model),
                "critic_refit_after_B": True, "checkpoint_evaluated_after_refit": True,
                "loss_mean": float(np.mean(losses)), "D_in_sample": scored_d, "C": scored, "C_keep": keep_c,
                "dual_before": dual, "C_empirically_suitable": eligible}
        dual = max(0., dual + 1000 * (scored["gap"] - PRIMARY))
        if scored["gap"] <= PRIMARY:
            dual *= .9
        item["dual_after"] = dual
        snapshot = {"score": keep_c, "model": copy.deepcopy(encoder.model.state_dict()),
                    "decoder": copy.deepcopy(decoder.state_dict()), "round": iteration + 1}
        if eligible and (suitable_best is None or keep_c < suitable_best["score"]):
            suitable_best = snapshot
        diagnostic_score = keep_c + 100 * max(scored["gap"] - PRIMARY, 0.) + tail
        if diagnostic_best is None or diagnostic_score < diagnostic_best["criterion"]:
            diagnostic_best = {**snapshot, "criterion": diagnostic_score}
        if history and eligible and abs(keep_c - history[-1]["C_keep"]) <= .01 * max(history[-1]["C_keep"], 1e-8):
            stable += 1
        else:
            stable = 0
        history.append(item)
        log(f"loop S{split + 1}/{init} round {iteration + 1}/{rounds}: Cgap={scored['gap']:.6f}, "
            f"Ctail={tail:.3%}, keep={keep_c:.4f}, dual={dual:.3f}, suitable={eligible}")
        if eligible and stable >= 3:
            break
    chosen = suitable_best if suitable_best is not None else diagnostic_best
    encoder.model.load_state_dict(chosen["model"]); decoder.load_state_dict(chosen["decoder"])
    encoder.model.eval()
    capped = len(history) == rounds and stable < 3
    report = {"history": history, "rounds_completed": len(history), "encoder_steps_total": sum(x["encoder_steps"] for x in history),
              "capped": capped, "converged_by_C_rule": not capped, "suitable_C_checkpoint_exists": suitable_best is not None,
              "selected_round": chosen["round"], "selected_checkpoint_kind": "C_suitable" if suitable_best else "diagnostic_fallback",
              "selected_encoder_sha256": tensor_state_sha(encoder.model), "S_or_E_used_for_selection": False,
              "algorithmic_convergence_proved": False, "split": split + 1, "initialization": init,
              "C_tail_gate_enforced": enforce_tail_gate}
    # The selected encoder is fixed; fit brand-new P critics, never reuse the D critics for the final audit.
    final = fit_for_rows(mode, encoder, frame, roles["P"], data.A, seed + 9000, epochs, raw_epochs)
    scored_p = evaluate_rows(mode, encoder, final, frame, roles["P"], data.A, independent=False)
    audit = evaluate_rows(mode, encoder, final, frame, roles["S"], data.A)
    selected = history[chosen["round"] - 1]
    report["refit_stability_diagnostic"] = {
        "selected_D_in_sample_gap": selected["D_in_sample"]["gap"],
        "selected_C_gap_from_D_predictors": selected["C"]["gap"],
        "fresh_P_in_sample_risks": scored_p,
        "independent_S_gap_from_P_predictors": audit["gap"],
        "S_minus_C_gap": audit["gap"] - selected["C"]["gap"],
        "scope": "same selected encoder; differences combine sampling and predictor refitting, not exact drift error"}
    log(f"selected S{split + 1}/{init} checkpoint round{chosen['round']}, capped={capped}; independent S gap={audit['gap']:.6f}")
    return encoder, final, report, audit


def add_fresh_bounds(models, data_cfg, seed, m, n_s, log):
    fresh = ph.generate(ph.with_changes(data_cfg, n=m, seed=seed))
    rows = np.arange(m)
    frames = {}
    for name, mode, encoder, bundle, rec in models:
        if mode.name not in frames:
            frames[mode.name] = mode.new_frame(fresh.images)
        frame = frames[mode.name]
        z = encoder.encode(frame, rows)
        q0, q1, _ = bundle.predict(z, frame.psi[encoder.split], mode.crop(frame, rows, encoder.split))
        pi_error = (q1 - fresh.propensity) ** 2
        pi_error0 = (q0 - fresh.propensity) ** 2
        rec["finite"] = finite_screen(rec["gap"], rec["se"], n_s, float(pi_error.mean()), m)
        rec["fresh_MC"] = {"seed": seed, "n": m, "model_frozen_before_MC": True,
                           "pi_error_SE": float(pi_error.std(ddof=1) / math.sqrt(m)),
                           "q0_pi_error_mean": float(pi_error0.mean()),
                           "q0_pi_error_SE": float(pi_error0.std(ddof=1) / math.sqrt(m)),
                           "q0_pi_error_scope": "diagnostic conservative latent-propensity error; not exact epsilon0",
                           "q1_prediction_sha256": level_a.array_sha(q1),
                           "scope": "phantom truth evaluation only; no refit/choice on these patients"}
        rec["returned_causal_estimate"] = None
        rec["qualification_failures"] = {"finite_gate": rec["finite"]["finite_D2_upper"] > PRIMARY,
            "empirical_gap": rec["finite"]["normal_empirical_upper"] > PRIMARY,
            "empirical_tail": rec["q0"]["fraction_outside_05_95"] > .01,
            "uniform_overlap_or_ORC_unverified": True}
        log(f"finite[{name}]: piMSE={pi_error.mean():.5f}, upper={rec['finite']['finite_D2_upper']:.5f}; "
            "conservative bound and causal assumptions reported separately")


def _aipw_result(zp, ze, ap, yp, ae, ye, e, tau, label):
    mean, scale = zp.mean(0), zp.std(0) + 1e-6
    fp = np.column_stack([np.ones(len(zp)), (zp - mean) / scale])
    fe = np.column_stack([np.ones(len(ze)), (ze - mean) / scale])
    m1 = fe @ level_a.pr.ridge_ols(fp[ap == 1], yp[ap == 1], 1e-3)
    m0 = fe @ level_a.pr.ridge_ols(fp[ap == 0], yp[ap == 0], 1e-3)
    clipped = np.clip(e, .05, .95)
    score = m1 - m0 + ae * (ye - m1) / clipped - (1 - ae) * (ye - m0) / (1 - clipped)
    return _effect_summary(score, tau, label, e)


def _effect_summary(score, tau, label, propensity, *, p_fitted=True):
    estimate = float(np.mean(score)); se = float(np.std(score, ddof=1) / math.sqrt(len(score)))
    return {"candidate_estimate": estimate, "sampling_SE": se,
            "normal_sampling_interval": [estimate - 1.96 * se, estimate + 1.96 * se],
            "error_vs_known_tau_evaluation_only": estimate - tau, "label": label,
            "propensity_tail_fraction": float(np.mean((propensity < .05) | (propensity > .95))),
            "P_fit_E_score": p_fitted, "P_nuisance_fit": p_fitted, "E_score": True,
            "propensity_clipping_scope": "numerical stabilization, not overlap certification",
            "causal_confidence_interval": None, "returned_causal_estimate": None}


def effects(mode, data, roles, baseline_encoder, baseline_bundle, encoder, bundle, seed, epochs, log):
    p, e = roles["P"], roles["E"]
    frame = mode.frame
    out = {}
    if (encoder is None) != (bundle is None):
        raise ValueError("trained encoder and predictor must both be present or both be absent")
    model_pairs = [("baseline_S1", baseline_encoder, baseline_bundle)]
    if encoder is not None:
        model_pairs.append(("trained_S1", encoder, bundle))
    for name, enc, crit in model_pairs:
        zp, ze = enc.encode(frame, p), enc.encode(frame, e)
        prob, _, _ = crit.predict(ze, frame.psi[0][e], mode.crop(frame, e, 0))
        out[name] = _aipw_result(zp, ze, data.A[p], data.Y[p], data.A[e], data.Y[e], prob, data.tau, name)
    # Same P/E patients for all ordinary and truth-only comparators.
    full = np.column_stack([frame.reference, *frame.psi])
    fitted = CriticBundle(full[p], frame.psi[0][p], mode.crop(frame, p, 0), data.A[p], seed, epochs, 1)
    q, _, _ = fitted.predict(full[e], frame.psi[0][e], mode.crop(frame, e, 0))
    out["full_image_reader_adjustment"] = _aipw_result(full[p], full[e], data.A[p], data.Y[p], data.A[e], data.Y[e],
                                                       q, data.tau, "image features only; no balancing")
    ml, vl = oracle._posterior(data.L[e], tuple(range(5)), data.config.sigma)
    el = oracle._p_given_post(ml, vl, np.zeros(len(e)), None, data.config)
    out["true_L_evaluation_control"] = _aipw_result(data.L[p], data.L[e], data.A[p], data.Y[p], data.A[e], data.Y[e],
                                                   el, data.tau, "true L evaluation-only; not an information ceiling")
    eu = oracle._p_given_post(data.U[e], 0., np.zeros(len(e)), None, data.config)
    m0 = ph.outcome_shape(data.U[e], data.config); m1 = m0 + data.tau
    ec = np.clip(eu, .05, .95)
    score = m1 - m0 + data.A[e] * (data.Y[e] - m1) / ec - (1 - data.A[e]) * (data.Y[e] - m0) / (1 - ec)
    out["oracle_U_evaluation_control"] = _effect_summary(score, data.tau,
        "known U-only propensity/outcome; evaluation-only", eu, p_fitted=False)
    a, y = data.A[e], data.Y[e]
    diff = float(y[a == 1].mean() - y[a == 0].mean())
    se = float(math.sqrt(y[a == 1].var(ddof=1) / np.sum(a == 1) + y[a == 0].var(ddof=1) / np.sum(a == 0)))
    out["naive_same_E"] = {"estimate": diff, "sampling_SE": se, "error_vs_tau": diff - data.tau}
    out["scope"] = "same heldout E; diagnostic candidates and controls, not automatically returned ATE"
    log("E effects: " + ", ".join(f"{k}={v.get('candidate_estimate', v.get('estimate')):.4f}" for k, v in out.items()
                                     if isinstance(v, dict)))
    return out


def run_local(args, log):
    started = time.time()
    if args.device != "cpu":
        raise ValueError("this local implementation uses the tested CPU path")
    nets.setup_torch(2)
    cfg = ph.PhantomConfig(n=args.n, sigma=.85, seed=args.seed, image_only=True)
    data = ph.generate(cfg)
    groups = folds.six_groups(args.n, args.seed)
    roles = {"R": groups["R"], **folds.roles(groups["groups"], 0)}
    if len(np.unique(np.concatenate(list(roles.values())))) != args.n or data.X.shape[1] != 0:
        raise AssertionError("patient roles must be disjoint and feature contract image-only")
    log(f"main n={args.n}, seed={args.seed}, roles=" + str({k: len(v) for k, v in roles.items()}))
    raw = prepare_mode(data, roles, "raw", args.seed + 1, args.reader_epochs, args.mae_epochs, log)
    raw_records, raw_models = run_fixed_audits(raw, data, roles, args.seed + 100, args.critic_epochs, args.raw_epochs, log)
    corrected = prepare_mode(data, roles, "corrected", args.seed + 1, args.reader_epochs, args.mae_epochs, log)
    corr_records, corr_models = run_fixed_audits(corrected, data, roles, args.seed + 100, args.critic_epochs, args.raw_epochs, log)
    encoder, critics, training, trained_audit = train_alternating(corrected, data, roles, args.seed + 300,
        args.rounds, args.encoder_steps, args.critic_epochs, args.raw_epochs, log)
    main_models = raw_models + corr_models + [("trained_S1", corrected, encoder, critics, trained_audit)]
    add_fresh_bounds(main_models, cfg, args.seed + 70001, args.mc, len(roles["S"]), log)
    baseline_encoder, baseline_bundle = corr_models[0][2], corr_models[0][3]
    effect = effects(corrected, data, roles, baseline_encoder, baseline_bundle, encoder, critics,
                     args.seed + 900, args.critic_epochs, log)
    low_cfg = ph.PhantomConfig(n=args.low_n, sigma=.3, seed=args.seed + 10001, image_only=True)
    low_data = ph.generate(low_cfg)
    lg = folds.six_groups(args.low_n, low_cfg.seed)
    low_roles = {"R": lg["R"], **folds.roles(lg["groups"], 0)}
    low = prepare_mode(low_data, low_roles, "corrected", low_cfg.seed + 2, args.reader_epochs, args.mae_epochs, log)
    low_encoder = LinearImageEncoder(low, low_roles["D"], low_data.A, 0, low_cfg.seed + 100)
    low_bundle = fit_for_rows(low, low_encoder, low.frame, low_roles["P"], low_data.A, low_cfg.seed + 200,
                              args.critic_epochs, args.raw_epochs)
    low_rec = evaluate_rows(low, low_encoder, low_bundle, low.frame, low_roles["S"], low_data.A)
    add_fresh_bounds([("low_sigma_S1", low, low_encoder, low_bundle, low_rec)], low_cfg, args.seed + 80001,
                     args.mc, len(low_roles["S"]), log)
    paired = {s: {"gap_change_corrected_minus_raw": corr_records[s]["gap"] - raw_records[s]["gap"],
                  "raw_q1_risk_change": corr_records[s]["q1"]["brier"] - raw_records[s]["q1"]["brier"],
                  "same_patient_roles": True, "raw_T_same_pixels": True} for s in raw_records}
    return {"artifact": "single_image_pipeline_v2_local", "computation_complete": True,
            "feature_mode": "image_only", "config": ph.config_dict(cfg), "primary_delta2": PRIMARY,
            "sensitivity_delta2": BUDGETS, "role_sizes": {k: len(v) for k, v in roles.items()},
            "role_rows_sha256": {k: level_a.array_sha(v) for k, v in roles.items()},
            "protocol": {"raw_T": "original heldout region pixels; constant R-only scale",
                         "reference": "fixed nonlung body mask; reference coordinate in every Z",
                         "P_internal_validation_then_S": True, "E_used_for_selection": False,
                         "MC_used_for_selection": False, "primary_or_architecture_changed_after_S": False,
                         "truth_inputs_to_training": False, "rotation": 0, "full_plurality_run": False},
            "stage1": {"raw": raw.report, "corrected": corrected.report},
            "five_split_raw_audits": raw_records, "five_split_corrected_audits": corr_records,
            "paired_brightness_comparison": paired, "alternating_training": training,
            "trained_independent_audit": trained_audit, "same_E_effect_diagnostics": effect,
            "low_sigma_control": {"config": ph.config_dict(low_cfg), "mode": low.report, "audit_S1": low_rec},
            "returned_causal_estimate": None,
            "qualification_scope": "finite discrepancy, uniform overlap, ORC/stability and nuisance bias remain separate",
            "seconds": time.time() - started}
