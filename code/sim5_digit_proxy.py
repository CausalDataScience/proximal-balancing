#!/usr/bin/env python3
"""Observed-data primitives for Sim-5 digit-proxy PRD Revision 3.

The latent generator is deliberately separate from the validated learner view.
These primitives do not assert that a representation or a theorem has passed.
"""
from __future__ import annotations

import itertools
import hashlib
import copy
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.ndimage import affine_transform
import torch
from torch import nn

FULL_SIZES = {"D_fit": 2400, "D_select": 800, "D_audit": 800, "N": 4000, "E": 40000}
BLOCK_DIMS = (324, 324, 324, 308, 16)
HEAD_UPDATES = 100
CRITIC_REFRESH_EVERY = 5
CHECKPOINT_UPDATES = (0, 25, 50, 100)


def block_masks() -> list[np.ndarray]:
    masks = []
    for rs, cs in ((0, 0), (0, 18), (18, 0), (18, 18)):
        mask = np.zeros((36, 36), dtype=bool)
        mask[rs:rs + 18, cs:cs + 18] = True
        masks.append(mask)
    masks[3][32:36, 32:36] = False
    marker = np.zeros((36, 36), dtype=bool)
    marker[32:36, 32:36] = True
    return masks + [marker]


def all_splits() -> list[tuple[int, ...]]:
    return [s for k in range(1, 5) for s in itertools.combinations(range(5), k)]


def outer_rows(sizes: dict[str, int], seed: int) -> dict[str, np.ndarray]:
    if any(not isinstance(n, int) or n <= 0 for n in sizes.values()):
        raise ValueError("all split sizes must be positive integers")
    perm = np.random.default_rng(seed).permutation(sum(sizes.values()))
    rows, start = {}, 0
    for name, n in sizes.items():
        rows[name] = perm[start:start + n]
        start += n
    return rows


def generate_latents(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    uc = rng.integers(1, 10, size=n)
    ut, ex, ey = rng.normal(size=(3, n))
    x = .5 * (uc - 5) + 2 * ex
    p = .1 + .8 * ndtr(.5 * (uc - 5) + 1.5 * ut + .2 * x)
    a = rng.binomial(1, p)
    y = a - .8 * (uc - 5) + .2 * x + .3 * ey
    return {"X": x, "A": a, "Y": y, "p_eval_only": p,
            "U_c_eval_only": uc, "U_t_eval_only": ut}


def load_mnist(path: Path | None = None) -> dict:
    """Read the local training bank; test-bank labels never enter training."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "materials/benchmarks/mnist/mnist.npz"
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as bank:
        images = np.asarray(bank["x_train"], dtype=np.float32) / 255.0
        labels = np.asarray(bank["y_train"], dtype=np.int64)
    keep = (labels >= 1) & (labels <= 9)
    images, labels = images[keep], labels[keep]
    if images.shape[1:] != (28, 28) or set(np.unique(labels)) != set(range(1, 10)):
        raise ValueError("MNIST training bank must contain all digits 1 through 9")
    return {"images": images, "labels": labels, "sha256": digest, "path": str(path)}


def render_images(latents: dict, seed: int, bank: dict | None = None) -> np.ndarray:
    """Render deterministic observed images. Latents are simulator-only inputs."""
    bank = load_mnist() if bank is None else bank
    uc = np.asarray(latents["U_c_eval_only"])
    ut = np.asarray(latents["U_t_eval_only"])
    if uc.ndim != 1 or ut.shape != uc.shape or not np.isin(uc, np.arange(1, 10)).all():
        raise ValueError("invalid generator digit/marker arrays")
    rng = np.random.default_rng(seed)
    by_digit = {d: np.flatnonzero(bank["labels"] == d) for d in range(1, 10)}
    if any(len(indices) == 0 for indices in by_digit.values()):
        raise ValueError("missing digit in rendering bank")
    n = len(uc)
    templates = [int(rng.choice(by_digit[int(d)])) for d in uc]
    angles = np.deg2rad(rng.uniform(-20, 20, size=n))
    scales = rng.uniform(.9, 1.1, size=n)
    result = np.zeros((n, 36, 36), dtype=np.float32)
    center = np.full(2, 13.5)
    for i, (template, angle, scale) in enumerate(zip(templates, angles, scales)):
        c, s = np.cos(angle), np.sin(angle)
        inverse = np.array([[c, s], [-s, c]]) / scale
        result[i, 4:32, 4:32] = affine_transform(
            bank["images"][template], inverse, offset=center - inverse @ center,
            output_shape=(28, 28), order=1, mode="constant", cval=0., prefilter=False)
    result[:, 32:36, 32:36] = (.5 + .4 * np.tanh(ut / 2))[:, None, None]
    # Draw per image to cap temporary memory at one canvas even for E=40,000.
    for image in result:
        image += rng.normal(0, .05, size=(36, 36)).astype(np.float32)
    np.clip(result, 0., 1., out=result)
    return result


def to_tensor(value) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def learner_view(data: dict) -> dict:
    allowed = {"X", "A", "Y", "images", "row_id"}
    if set(data) != allowed:
        raise ValueError("learner input must contain exactly X,A,Y,images,row_id; latent keys forbidden")
    x, a, y, rows = (np.asarray(data[k]) for k in ("X", "A", "Y", "row_id"))
    images = np.asarray(data["images"], dtype=np.float32)
    n = len(x)
    if images.shape != (n, 36, 36) or any(v.shape != (n,) for v in (x, a, y, rows)):
        raise ValueError("invalid observed-data shapes")
    if len(np.unique(rows)) != n or not all(np.isfinite(v).all() for v in (x, a, y, images)):
        raise ValueError("duplicate rows or nonfinite observations")
    return {"X": x, "A": a, "Y": y, "row_id": rows,
            "blocks": [images[:, mask] for mask in block_masks()]}


def _mlp(dims: list[int]) -> nn.Sequential:
    layers = []
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(din, dout))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class Standardizer:
    """Fit X/Y transforms exclusively on the supplied optimizer-training rows."""
    def fit(self, x: np.ndarray, y: np.ndarray, train_rows: np.ndarray):
        rows = np.asarray(train_rows, dtype=int)
        if len(rows) < 2 or len(np.unique(rows)) != len(rows):
            raise ValueError("need distinct training rows")
        self.fit_rows = rows.copy()
        self.x_mean = float(np.asarray(x)[rows].mean())
        self.x_std = max(float(np.asarray(x)[rows].std()), 1e-8)
        self.y_mean = float(np.asarray(y)[rows].mean())
        self.y_std = max(float(np.asarray(y)[rows].std()), 1e-8)
        return self

    def x(self, value):
        return (np.asarray(value) - self.x_mean) / self.x_std

    def y(self, value):
        return (np.asarray(value) - self.y_mean) / self.y_std

    def inverse_y(self, value):
        return np.asarray(value) * self.y_std + self.y_mean


def bounded_probability(logit: torch.Tensor) -> torch.Tensor:
    return .05 + .90 * torch.sigmoid(logit)


class BaseCritic(nn.Module):
    def __init__(self, z_dim: int = 9):
        super().__init__()
        self.z_dim = z_dim
        self.net = _mlp([z_dim, 64, 64, 1])

    def logit(self, z):
        return self.net(z).squeeze(-1)

    def forward(self, z):
        return bounded_probability(self.logit(z))


class AugmentedCritic(nn.Module):
    """Own lifted base plus zero-initialized residual, never shared parameters."""
    def __init__(self, base: BaseCritic, heldout_dim: int):
        super().__init__()
        self.base = copy.deepcopy(base)
        self.residual = _mlp([base.z_dim + heldout_dim, 64, 64, 1])
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, z, heldout):
        residual = self.residual(torch.cat([z, heldout], dim=1)).squeeze(-1)
        return bounded_probability(self.base.logit(z) + residual)


def paired_gap(a, base, augmented) -> dict:
    def array(value):
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    a, base, augmented = [array(v).astype(np.float64).reshape(-1) for v in (a, base, augmented)]
    if len(a) < 2 or base.shape != a.shape or augmented.shape != a.shape:
        raise ValueError("paired gap requires matched rows and at least two observations")
    if not all(np.isfinite(v).all() for v in (a, base, augmented)):
        raise ValueError("nonfinite paired-risk inputs")
    r0, r1 = (a - base) ** 2, (a - augmented) ** 2
    d = r0 - r1
    gap, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"risk_base": float(r0.mean()), "risk_augmented": float(r1.mean()),
            "signed_gap": gap, "clipped_gap": max(gap, 0.), "se": se, "n": len(d),
            "reliability_fail": gap < -2 * se}


def state_digest(model_or_state) -> str:
    state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        arr = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((arr.dtype, arr.shape)).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def fit_predictor(model: nn.Module, inputs: tuple[torch.Tensor, ...], target: torch.Tensor,
                  train_rows: np.ndarray, val_rows: np.ndarray, *, seed: int, kind: str = "brier",
                  max_epochs: int = 40, patience: int = 5, batch_size: int = 128,
                  learning_rate: float = 1e-3, min_delta: float = 1e-4) -> dict:
    """Same early-stopping contract for independent critics; returns fit evidence."""
    train_rows, val_rows = np.asarray(train_rows), np.asarray(val_rows)
    if np.intersect1d(train_rows, val_rows).size or len(train_rows) < 2 or len(val_rows) < 2:
        raise ValueError("train/validation rows must be nonempty and disjoint")
    if kind not in ("brier", "mse", "bce", "warm"):
        raise ValueError("unknown loss")
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    def loss(rows):
        prediction = model(*[x[rows] for x in inputs])
        if kind == "bce":
            return nn.functional.binary_cross_entropy(prediction, target[rows])
        if kind == "warm":
            return (nn.functional.binary_cross_entropy(prediction[:, 0], target[rows, 0])
                    + ((prediction[:, 1] - target[rows, 1]) ** 2).mean())
        return ((prediction - target[rows]) ** 2).mean()
    best, best_state, best_epoch, stale, updates = float("inf"), None, -1, 0, 0
    trace = []
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(train_rows)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            opt.zero_grad()
            value = loss(batch)
            if not torch.isfinite(value):
                raise ValueError("nonfinite training loss")
            value.backward()
            opt.step()
            updates += 1
        model.eval()
        with torch.no_grad():
            value = float(loss(val_rows))
        trace.append(value)
        if value < best - min_delta:
            best, best_state, best_epoch, stale = value, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise ValueError("no finite fitted checkpoint")
    model.load_state_dict(best_state)
    with torch.no_grad():
        fit_loss = float(loss(train_rows))
    return {"best_epoch": best_epoch, "epochs_run": len(trace), "optimizer_updates": updates,
            "fit_loss": fit_loss, "val_loss": best, "val_trace": trace,
            "state_sha256": state_digest(model), "seed": seed}


class SplitRepresentation(nn.Module):
    """All retained blocks enter phi(X,W); response/treatment are not inputs."""
    def __init__(self, retained: list[int]):
        super().__init__()
        if not retained or len(set(retained)) != len(retained) or any(j not in range(5) for j in retained):
            raise ValueError("invalid retained blocks")
        self.retained = tuple(retained)
        self.encoders = nn.ModuleList([_mlp([BLOCK_DIMS[j], 64, 64, 8]) for j in retained])
        self.fusion = _mlp([1 + 8 * len(retained), 64, 32, 8])

    def forward(self, x: torch.Tensor, retained_blocks: list[torch.Tensor]) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 1 or len(retained_blocks) != len(self.encoders):
            raise ValueError("phi expects one X column and every retained block")
        features = [enc(w) for enc, w in zip(self.encoders, retained_blocks)]
        return torch.cat([x, self.fusion(torch.cat([x, *features], dim=1))], dim=1)


class _WarmPrediction(nn.Module):
    """A is a response-head input; it is never supplied to representation.forward."""
    def __init__(self, representation: SplitRepresentation):
        super().__init__()
        self.representation = representation
        self.treatment = _mlp([9, 32, 1])
        self.outcome = _mlp([10, 32, 1])

    def forward(self, x, *blocks_and_treatment):
        blocks, a = list(blocks_and_treatment[:-1]), blocks_and_treatment[-1]
        z = self.representation(x, blocks)
        p = bounded_probability(self.treatment(z).squeeze(-1))
        m = self.outcome(torch.cat([a[:, None], z], dim=1)).squeeze(-1)
        return torch.stack([p, m], dim=1)


class _BlockWarmPrediction(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.treatment = _mlp([9, 32, 1])
        self.outcome = _mlp([10, 32, 1])

    def forward(self, x, w, a):
        z = torch.cat([x, self.encoder(w)], dim=1)
        p = bounded_probability(self.treatment(z).squeeze(-1))
        m = self.outcome(torch.cat([a[:, None], z], dim=1)).squeeze(-1)
        return torch.stack([p, m], dim=1)


def _warm_budget(budget):
    cfg = {"max_epochs": 40, "patience": 5, "batch_size": 128,
           "learning_rate": 1e-3, "min_delta": 1e-4}
    if budget:
        if set(budget) - set(cfg):
            raise ValueError("unknown warm budget")
        cfg.update(budget)
    return cfg


def fit_block_encoder_bank(view: dict, train_idx: np.ndarray, val_idx: np.ndarray,
                            seed: int, budget: dict | None = None) -> dict:
    """Fit every anonymous block once per dataset, independently of held-out split."""
    if set(view) != {"X", "A", "Y", "row_id", "blocks"} or len(view["blocks"]) != 5:
        raise ValueError("bank accepts only the five-block observed-data view")
    cfg = _warm_budget(budget)
    st = Standardizer().fit(view["X"], view["Y"], train_idx)
    x, a = to_tensor(st.x(view["X"])[:, None]), to_tensor(view["A"])
    target = torch.stack([a, to_tensor(st.y(view["Y"]))], dim=1)
    encoders, metadata = {}, {}
    for j in range(5):
        block_seed = seed + 1009 * j
        torch.manual_seed(block_seed)
        encoder = _mlp([BLOCK_DIMS[j], 64, 64, 8])
        predictor = _BlockWarmPrediction(encoder)
        fit = fit_predictor(predictor, (x, to_tensor(view["blocks"][j]), a), target,
                            train_idx, val_idx, seed=block_seed, kind="warm", **cfg)
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        encoders[j] = encoder
        metadata[j] = {"fit": fit, "digest": state_digest(encoder), "seed": block_seed}
    return {"encoders": encoders, "metadata": metadata, "standardizer": st, "seed": seed,
            "train_idx": np.asarray(train_idx).copy(), "val_idx": np.asarray(val_idx).copy(),
            "budget": cfg, "block_fit_count": 5}


def representation_from_bank(bank: dict, retained: list[int]) -> SplitRepresentation:
    model = SplitRepresentation(retained)
    model.encoders = nn.ModuleList([copy.deepcopy(bank["encoders"][j]) for j in retained])
    for parameter in model.encoders.parameters():
        parameter.requires_grad_(False)
    return model


def fit_warm_fusion(view: dict, bank: dict, retained: list[int], train_idx: np.ndarray,
                    val_idx: np.ndarray, seed: int, budget: dict | None = None) -> dict:
    """Keep the shared pretrained encoders fixed; fit fusion and auxiliary heads."""
    if set(view) != {"X", "A", "Y", "row_id", "blocks"}:
        raise ValueError("warm fusion accepts only observed X/W/A/Y")
    if not np.array_equal(train_idx, bank["train_idx"]) or not np.array_equal(val_idx, bank["val_idx"]):
        raise ValueError("fusion must use the bank's declared training/validation rows")
    cfg, st = _warm_budget(budget), bank["standardizer"]
    x = to_tensor(st.x(view["X"])[:, None])
    blocks = [to_tensor(view["blocks"][j]) for j in retained]
    a = to_tensor(view["A"])
    target = torch.stack([a, to_tensor(st.y(view["Y"]))], dim=1)
    torch.manual_seed(seed)
    representation = representation_from_bank(bank, retained)
    before = copy.deepcopy(representation.state_dict())
    encoder_digest = state_digest(representation.encoders)
    predictor = _WarmPrediction(representation)
    fit = fit_predictor(predictor, (x, *blocks, a), target, train_idx, val_idx,
                        seed=seed, kind="warm", **cfg)
    if state_digest(representation.encoders) != encoder_digest:
        raise AssertionError("fusion warmstart changed the pretrained bank")
    warm_state = copy.deepcopy(representation.state_dict())
    movement = sum(float(((warm_state[k] - before[k]) ** 2).sum()) for k in before) ** .5
    representation.eval()
    return {"model": representation, "standardizer": st, "warm_state": warm_state,
            "warm_digest": state_digest(warm_state), "initial_digest": state_digest(before),
            "representation_parameter_change_l2": movement, "fit": fit, "budget": cfg,
            "train_idx": np.asarray(train_idx).copy(), "val_idx": np.asarray(val_idx).copy(),
            "bank_encoder_digests": {j: bank["metadata"][j]["digest"] for j in retained},
            "seed": seed, "inference_inputs": ["X", "retained_blocks"]}


def fit_warm_representation(view: dict, retained: list[int], train_idx: np.ndarray,
                            val_idx: np.ndarray, seed: int, budget: dict | None = None,
                            *, bank: dict) -> dict:
    """Compatibility wrapper requiring the once-per-dataset pretrained bank."""
    return fit_warm_fusion(view, bank, retained, train_idx, val_idx, seed, budget)


def embed_view(model: SplitRepresentation, view: dict, standardizer: Standardizer,
               rows: np.ndarray | None = None) -> torch.Tensor:
    rows = np.arange(len(view["X"])) if rows is None else np.asarray(rows)
    return model(to_tensor(standardizer.x(view["X"][rows])[:, None]),
                 [to_tensor(view["blocks"][j][rows]) for j in model.retained])


def fit_critic_pair(z: torch.Tensor, heldout: torch.Tensor, a: torch.Tensor,
                    train_idx: np.ndarray, val_idx: np.ndarray, seed: int,
                    budget: dict | None = None) -> dict:
    cfg = {"max_epochs": 20, "patience": 3, "batch_size": 128,
           "learning_rate": 1e-3, "min_delta": 1e-4}
    if budget:
        if set(budget) - set(cfg):
            raise ValueError("unknown critic budget")
        cfg.update(budget)
    torch.manual_seed(seed)
    q0 = BaseCritic(z.shape[1])
    fit0 = fit_predictor(q0, (z.detach(),), a, train_idx, val_idx, seed=seed, **cfg)
    torch.manual_seed(seed + 1)
    q1 = AugmentedCritic(q0, heldout.shape[1])
    fit1 = fit_predictor(q1, (z.detach(), heldout.detach()), a, train_idx, val_idx,
                         seed=seed + 1, **cfg)
    return {"base": q0, "augmented": q1, "fit_base": fit0, "fit_augmented": fit1,
            "seed": seed, "budget": cfg,
            "optimizer_updates": fit0["optimizer_updates"] + fit1["optimizer_updates"]}


def balance_updates(warm: dict, view: dict, train_idx: np.ndarray, val_idx: np.ndarray,
                     seed: int, critic_budget: dict | None = None) -> dict:
    """Exactly 100 fusion optimizer steps; fresh critics every five steps.

    No selection or audit is performed here. Checkpoints are actual saved states.
    """
    model = copy.deepcopy(warm["model"])
    for p in model.encoders.parameters():
        p.requires_grad_(False)
    st = warm["standardizer"]
    x = to_tensor(st.x(view["X"])[:, None])
    with torch.no_grad():
        features = torch.cat([x, *[enc(to_tensor(view["blocks"][j]))
                                   for enc, j in zip(model.encoders, model.retained)]], dim=1)
    held_indices = [j for j in range(5) if j not in model.retained]
    if not held_indices:
        raise ValueError("balance requires at least one held-out block")
    heldout = to_tensor(np.concatenate([view["blocks"][j] for j in held_indices], axis=1))
    a = to_tensor(view["A"])
    opt = torch.optim.Adam(model.fusion.parameters(), lr=1e-4)
    initial_fusion = copy.deepcopy(model.fusion.state_dict())
    enc_digest = state_digest(model.encoders)
    checkpoints, updates, refreshes = {}, [], []
    zero_clipped = zero_gradient = critic_updates = 0
    def z_current():
        return torch.cat([x, model.fusion(features)], dim=1)
    def checkpoint(count):
        state = copy.deepcopy(model.state_dict())
        checkpoints[count] = {"head_update_count": count, "state": state,
                              "state_sha256": state_digest(state),
                              "critic_update_count": critic_updates}
    checkpoint(0)
    pair = None
    for count in range(HEAD_UPDATES + 1):
        if count % CRITIC_REFRESH_EVERY == 0:
            with torch.no_grad():
                z = z_current()
            pair = fit_critic_pair(z, heldout, a, train_idx, val_idx,
                                   seed=seed + 1009 * count, budget=critic_budget)
            critic_updates += pair["optimizer_updates"]
            for critic in (pair["base"], pair["augmented"]):
                critic.eval()
                for p in critic.parameters():
                    p.requires_grad_(False)
            with torch.no_grad():
                gap = paired_gap(a[train_idx], pair["base"](z[train_idx]),
                                 pair["augmented"](z[train_idx], heldout[train_idx]))
            refreshes.append({"head_update_count": count, "critic_update_count": critic_updates,
                              "seed": pair["seed"], "fit_base": pair["fit_base"],
                              "fit_augmented": pair["fit_augmented"], **gap})
        if count == HEAD_UPDATES:
            break
        opt.zero_grad()
        z = z_current()[train_idx]
        p0 = pair["base"](z)
        p1 = pair["augmented"](z, heldout[train_idx])
        signed = (((a[train_idx] - p0) ** 2) - ((a[train_idx] - p1) ** 2)).mean()
        objective = torch.clamp(signed, min=0.)
        objective.backward()
        grad_sq = sum(float((p.grad.detach() ** 2).sum()) for p in model.fusion.parameters()
                      if p.grad is not None)
        zero_clipped += int(float(signed.detach()) <= 0)
        zero_gradient += int(grad_sq == 0.)
        opt.step()
        updates.append({"head_update_count": count + 1, "signed_gap": float(signed.detach()),
                        "clipped_objective": float(objective.detach()), "gradient_l2": grad_sq ** .5})
        if count + 1 in CHECKPOINT_UPDATES:
            checkpoint(count + 1)
    final_fusion = model.fusion.state_dict()
    movement = sum(float(((final_fusion[k] - initial_fusion[k]) ** 2).sum())
                   for k in initial_fusion) ** .5
    if state_digest(model.encoders) != enc_digest:
        raise AssertionError("frozen block encoders changed during balance")
    return {"model": model, "checkpoints": checkpoints, "updates": updates, "refreshes": refreshes,
            "head_update_count": len(updates), "critic_update_count": critic_updates,
            "clipped_zero_count": zero_clipped, "zero_gradient_count": zero_gradient,
            "fusion_parameter_change_l2": movement, "frozen_encoders_digest": enc_digest,
            "standardizer": st, "warm_digest": warm["warm_digest"]}


def _best_two_refits(z, heldout, a, train_idx, val_idx, seed, budget=None):
    cfg = {"max_epochs": 40, "patience": 5}
    cfg.update(budget or {})
    trials = [fit_critic_pair(z, heldout, a, train_idx, val_idx, seed + 193 * j, cfg)
              for j in range(2)]
    ib = min(range(2), key=lambda j: trials[j]["fit_base"]["val_loss"])
    ia = min(range(2), key=lambda j: trials[j]["fit_augmented"]["val_loss"])
    return {"base": trials[ib]["base"], "augmented": trials[ia]["augmented"],
            "fit_base": trials[ib]["fit_base"], "fit_augmented": trials[ia]["fit_augmented"],
            "chosen_base_init": ib, "chosen_augmented_init": ia, "seed": seed,
            "trial_seeds": [t["seed"] for t in trials]}


def _fixed_critic_inputs(model, view, standardizer):
    with torch.no_grad():
        z = embed_view(model, view, standardizer)
    held = [j for j in range(5) if j not in model.retained]
    t = to_tensor(np.concatenate([view["blocks"][j] for j in held], axis=1))
    return z, t, to_tensor(view["A"])


def select_checkpoint(balance: dict, view: dict, train_idx: np.ndarray, val_idx: np.ndarray,
                       select_idx: np.ndarray, seed: int, budget: dict | None = None,
                       restart_id: int = 0) -> dict:
    """Rank saved states using Dselect only, after fresh independent refits."""
    if np.intersect1d(select_idx, np.concatenate([train_idx, val_idx])).size:
        raise ValueError("checkpoint-selection rows overlap fitting rows")
    records = []
    for count, saved in sorted(balance["checkpoints"].items()):
        model = copy.deepcopy(balance["model"])
        model.load_state_dict(saved["state"])
        z, heldout, a = _fixed_critic_inputs(model, view, balance["standardizer"])
        pair = _best_two_refits(z, heldout, a, train_idx, val_idx, seed, budget)
        with torch.no_grad():
            gap = paired_gap(a[select_idx], pair["base"](z[select_idx]),
                             pair["augmented"](z[select_idx], heldout[select_idx]))
        records.append({"head_update_count": count, "restart_id": restart_id,
                        "state_sha256": saved["state_sha256"], **gap,
                        "refit_seeds": pair["trial_seeds"],
                        "fit_base": pair["fit_base"], "fit_augmented": pair["fit_augmented"]})
    minimum = min(r["clipped_gap"] for r in records)
    eligible = [r for r in records if r["clipped_gap"] <= minimum + 1e-6]
    chosen = min(eligible, key=lambda r: (r["head_update_count"], r["restart_id"]))
    model = copy.deepcopy(balance["model"])
    model.load_state_dict(balance["checkpoints"][chosen["head_update_count"]]["state"])
    return {"model": model, "selected": chosen, "selection_records": records,
            "standardizer": balance["standardizer"], "select_idx": np.asarray(select_idx).copy(),
            "warm_state": copy.deepcopy(balance["checkpoints"][0]["state"]),
            "warm_digest": balance["warm_digest"], "restart_id": restart_id}


def overlap_diagnostics(predictions) -> dict:
    p = predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else np.asarray(predictions)
    p = p.astype(float).reshape(-1)
    if len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("nonfinite/empty propensity predictions")
    qs = np.quantile(p, [.01, .05, .50, .95, .99])
    return {"min": float(p.min()), "q01": float(qs[0]), "q05": float(qs[1]),
            "q50": float(qs[2]), "q95": float(qs[3]), "q99": float(qs[4]), "max": float(p.max()),
            "fraction_outside_0p1_0p9": float(np.mean((p < .1) | (p > .9))),
            "structural_eta": .05, "structural_range_pass": bool(np.all((p >= .05 - 1e-7) & (p <= .95 + 1e-7))),
            "interpretation": "actual bounded model output; range diagnostic is not evidence of proxy validity"}


def audit_warm_final(selected: dict, view: dict, train_idx: np.ndarray, val_idx: np.ndarray,
                      audit_idx: np.ndarray, seed: int, budget: dict | None = None) -> dict:
    """Independent warm/final refits on the same untouched audit rows, no reselection."""
    used = np.concatenate([train_idx, val_idx, selected["select_idx"]])
    if np.intersect1d(audit_idx, used).size:
        raise ValueError("audit rows overlap fitting or selection rows")
    before = state_digest(selected["model"])
    seeds = audit_refit_seeds(seed)
    results = {}
    for name in ("warm", "final"):
        model = copy.deepcopy(selected["model"])
        if name == "warm":
            model.load_state_dict(selected["warm_state"])
        z, heldout, a = _fixed_critic_inputs(model, view, selected["standardizer"])
        pair = _best_two_refits(z, heldout, a, train_idx, val_idx, seeds[name], budget)
        with torch.no_grad():
            p0 = pair["base"](z[audit_idx])
            p1 = pair["augmented"](z[audit_idx], heldout[audit_idx])
        results[name] = {**paired_gap(a[audit_idx], p0, p1), "seed": seeds[name],
                         "refit_seeds": pair["trial_seeds"], "state_sha256": state_digest(model),
                         "overlap": overlap_diagnostics(p0),
                         "fit_base": pair["fit_base"], "fit_augmented": pair["fit_augmented"]}
    if before != state_digest(selected["model"]):
        raise AssertionError("audit changed the selected representation")
    return {"warm": results["warm"], "final": results["final"],
            "selected_state_sha256": before, "selection_unchanged": True,
            "audit_row_ids": view["row_id"][audit_idx].tolist()}


def embed_view_batched(model: SplitRepresentation, view: dict, standardizer: Standardizer,
                       rows: np.ndarray, batch_size: int = 1024) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([embed_view(model, view, standardizer, rows[start:start + batch_size])
                          for start in range(0, len(rows), batch_size)], dim=0)


def fit_nuisance(model: SplitRepresentation, view: dict, standardizer: Standardizer,
                  n_idx: np.ndarray, seed: int, budget: dict | None = None) -> dict:
    """Fit only on N, with fixed phi and its existing X standardizer."""
    n_idx = np.asarray(n_idx, dtype=int)
    if len(n_idx) < 10 or len(np.unique(n_idx)) != len(n_idx):
        raise ValueError("nuisance needs at least ten distinct N rows")
    before = state_digest(model)
    z = embed_view_batched(model, view, standardizer, n_idx)
    a, y = to_tensor(view["A"][n_idx]), np.asarray(view["Y"])[n_idx]
    order = np.random.default_rng(seed).permutation(len(n_idx))
    cut = int(.8 * len(order))
    tr, va = order[:cut], order[cut:]
    y_mean, y_std = float(y[tr].mean()), max(float(y[tr].std()), 1e-8)
    cfg = {"max_epochs": 60, "patience": 8, "batch_size": 128,
           "learning_rate": 1e-3, "min_delta": 1e-4}
    if budget:
        if set(budget) - set(cfg):
            raise ValueError("unknown nuisance budget")
        cfg.update(budget)
    torch.manual_seed(seed)
    propensity = BaseCritic(9)
    fit_p = fit_predictor(propensity, (z,), a, tr, va, seed=seed, kind="bce", **cfg)
    torch.manual_seed(seed + 1)
    outcome = _mlp([10, 64, 64, 1])
    fit_m = fit_predictor(outcome, (torch.cat([a[:, None], z], dim=1),),
                          to_tensor((y - y_mean) / y_std)[:, None], tr, va,
                          seed=seed + 1, kind="mse", **cfg)
    if state_digest(model) != before:
        raise AssertionError("nuisance fitting changed phi")
    return {"propensity": propensity, "outcome": outcome, "y_mean": y_mean, "y_std": y_std,
            "fit_propensity": fit_p, "fit_outcome": fit_m, "budget": cfg,
            "fit_row_ids": view["row_id"][n_idx[tr]].tolist(),
            "val_row_ids": view["row_id"][n_idx[va]].tolist(),
            "phi_digest": before, "propensity_seed": seed, "outcome_seed": seed + 1}


def aipw_scores(a, y, propensity, m0, m1) -> np.ndarray:
    a, y, e, m0, m1 = [np.asarray(v, dtype=np.float64).reshape(-1)
                        for v in (a, y, propensity, m0, m1)]
    if not all(v.shape == a.shape and np.isfinite(v).all() for v in (y, e, m0, m1)):
        raise ValueError("nonfinite or mismatched AIPW inputs")
    if not np.isin(a, [0, 1]).all() or np.any((e <= 0) | (e >= 1)):
        raise ValueError("invalid treatment or propensity in AIPW")
    return m1 - m0 + a * (y - m1) / e - (1 - a) * (y - m0) / (1 - e)


def aipw_evaluate(model: SplitRepresentation, view: dict, standardizer: Standardizer,
                   nuisance: dict, e_idx: np.ndarray, batch_size: int = 1024) -> dict:
    e_idx = np.asarray(e_idx, dtype=int)
    fit_ids = nuisance["fit_row_ids"] + nuisance["val_row_ids"]
    if np.intersect1d(view["row_id"][e_idx], fit_ids).size:
        raise ValueError("E overlaps nuisance fit or validation rows")
    if state_digest(model) != nuisance["phi_digest"]:
        raise ValueError("phi changed after nuisance fitting")
    z = embed_view_batched(model, view, standardizer, e_idx, batch_size)
    p, m0, m1 = [], [], []
    with torch.no_grad():
        for start in range(0, len(e_idx), batch_size):
            zb = z[start:start + batch_size]
            p.append(nuisance["propensity"](zb).numpy())
            for treatment, target in ((0, m0), (1, m1)):
                az = torch.cat([torch.full((len(zb), 1), float(treatment)), zb], dim=1)
                pred = nuisance["outcome"](az).squeeze(-1).numpy()
                target.append(pred * nuisance["y_std"] + nuisance["y_mean"])
    p, m0, m1 = [np.concatenate(v) for v in (p, m0, m1)]
    scores = aipw_scores(view["A"][e_idx], view["Y"][e_idx], p, m0, m1)
    if not np.isfinite(scores).all() or len(scores) < 2:
        raise ValueError("invalid evaluation scores")
    return {"theta": float(scores.mean()), "score_sd": float(scores.std(ddof=1)),
            "n_e": len(scores), "scores": scores, "propensity": overlap_diagnostics(p),
            "eval_row_ids": view["row_id"][e_idx].tolist(), "phi_digest": state_digest(model)}


def evaluate_warm_final(selected: dict, view: dict, n_idx: np.ndarray, e_idx: np.ndarray,
                         seed: int, budget: dict | None = None) -> dict:
    if np.intersect1d(n_idx, e_idx).size:
        raise ValueError("N and E overlap")
    out = {}
    for name in ("warm", "final"):
        model = copy.deepcopy(selected["model"])
        if name == "warm":
            model.load_state_dict(selected["warm_state"])
        nuisance = fit_nuisance(model, view, selected["standardizer"], n_idx, seed, budget)
        evaluation = aipw_evaluate(model, view, selected["standardizer"], nuisance, e_idx)
        out[name] = {"evaluation": evaluation,
                     "nuisance_metadata": {k: v for k, v in nuisance.items()
                                           if k not in ("propensity", "outcome")}}
    return out


def training_event_schedule() -> list[dict]:
    return [{"head_update": u, "checkpoint": u in CHECKPOINT_UPDATES,
             "critic_refresh": u % CRITIC_REFRESH_EVERY == 0}
            for u in range(HEAD_UPDATES + 1)]


def audit_refit_seeds(seed: int) -> dict[str, int]:
    children = np.random.SeedSequence([seed, 731]).spawn(2)
    return {name: int(child.generate_state(1)[0]) for name, child in zip(("warm", "final"), children)}


def write_json_atomic(path: Path, payload: dict) -> None:
    """Publish a complete JSON file atomically, refusing existing destinations."""
    text = json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".sim5-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)
