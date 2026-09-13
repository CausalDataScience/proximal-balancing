#!/usr/bin/env python3
"""PROBE-Brier-neural: the main method of the fixed experiment plan (sections 8 and 9).

Every constant below is fixed by memo/2026-09-11-dgp-a-e-experiment-plan.md.  Three implementation decisions
that the plan leaves open are marked with the tag IMPLEMENTATION DECISION and are listed in the run report.

Pipeline for one replicate:
  unit permutation -> three folds -> for each rotation (D, N, E):
    for each of the 30 candidate held-out sets S:
      D is split 50/25/25 into fit, val, audit
      representation phi_S trained on D_fit to minimise the nested Brier risk gap, early stopped on D_val
      audit risk gap and its standard error computed on D_audit -> discrepancy screen
      nuisances fitted on N (80/20 fit/early stop) -> overlap screen on N
      AIPW scores evaluated on E
    candidates passing both screens in all three rotations are retained, their unit scores pooled,
    a shared Gaussian multiplier bootstrap gives one linking radius, components are formed and the
    largest one is reduced to its median; ties return a set; no survivor returns nothing.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn

import probe_dgp_ae as dgp

# ----------------------------------------------------------------------------- fixed hyperparameters (plan 8)
Z_DIM = 4                      # plan 8.4
BLOCK_EMBED = 16               # plan 8.4 and 6.3
ENC_LR, CRITIC_LR = 1e-3, 2e-3  # plan 8.6
WEIGHT_DECAY = 1e-4
BATCH = 256
CRITIC_STEPS_PER_ENCODER = 5
MAX_EPOCHS = 300
GRAD_CLIP = 5.0
PATIENCE = 20
MIN_DELTA = 1e-5
N_RESTARTS = 3
SCREEN_CONST = 0.02            # plan 8.7, a calibration constant frozen in the pilot
OVERLAP_LO, OVERLAP_HI = 0.05, 0.95   # plan 8.8
OVERLAP_MIN_SHARE = 0.98
CLIP_LO, CLIP_HI = 0.02, 0.98
NUIS_MAX_EPOCHS = 500          # plan 8.9
NUIS_PATIENCE = 20
NUIS_RESTARTS = 3
BOOTSTRAP_DRAWS = 5000         # plan 8.10
BOOTSTRAP_LEVEL = 0.95
RESTART_OVERLAP_TOL = 0.02     # plan 8.6
MIN_ENCODER_UPDATES = 20       # review of 2026-09-11: one update is not evidence of a trained representation
PROPENSITY_BOUND = (0.05, 0.95)  # structural bounds of the generators; set to None to fit unconstrained
ENCODER_FACTORY = None         # set to a callable(specs, d_x) to train a representation class other than
                               # `Representation`; the staged design F study uses it for a linear encoder
# Which arrangement of Section 4 the training loop implements.  "variant" is the implementation this project
# has been running: the two critics share a trunk, are trained on the summed loss, and the checkpoint is the
# epoch whose validation gap is closest to zero.  "manuscript" is the arrangement Section 4 actually states:
# each critic has its own parameters and minimises its own risk, the encoder minimises max(R0 - R1, 0), and
# the checkpoint minimises that same clipped objective.  Neither is a superset of the other, so both are kept
# and reported side by side rather than one being called a fix for the other (round 7 review, R5).
TRAINING_MODE = "variant"      # "variant" or "manuscript"


def bounded_prob(logit: torch.Tensor) -> torch.Tensor:
    """Treatment probability on the scale the generator actually uses.

    Every design bounds the structural propensity inside PROPENSITY_BOUND, and the conditional expectation of a
    bounded quantity obeys the same bounds, so the true P(A = 1 | Z) is inside that interval for every Z.  The
    map is applied inside the model, so the same quantity is used by the training loss, the validation loss and
    the prediction; applying it only at prediction time would shift every fitted probability.
    """
    p = torch.sigmoid(logit)
    if PROPENSITY_BOUND is None:
        return p
    lo, hi = PROPENSITY_BOUND
    return lo + (hi - lo) * p


def _mlp(sizes: list[int], out_act: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class ImageEncoder(nn.Module):
    """Plan 6.3, shared weights across image blocks within a representation."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4))
        self.head = nn.Sequential(nn.Linear(1024, BLOCK_EMBED), nn.LayerNorm(BLOCK_EMBED), nn.ReLU())

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(img).flatten(1))


class BlockStack(nn.Module):
    """Encodes a fixed list of blocks into concatenated embeddings.

    Tabular blocks get their own MLP(d, 64, 64, 16); image blocks share one CNN; the numeric channel of an
    image-plus-numeric block gets its own MLP (plan 8.4).
    """

    def __init__(self, specs: list[dict]) -> None:
        super().__init__()
        self.specs = specs
        self.tab = nn.ModuleDict()
        self.num = nn.ModuleDict()
        self.cnn = ImageEncoder() if any(s["kind"] in ("image", "image_numeric") for s in specs) else None
        for s in specs:
            key = str(s["index"])
            if s["kind"] == "tabular":
                self.tab[key] = _mlp([s["dim"], 64, 64, BLOCK_EMBED])
            elif s["kind"] == "image_numeric":
                self.num[key] = _mlp([s["num_dim"], 64, 64, BLOCK_EMBED])
        self.out_dim = sum(BLOCK_EMBED * (2 if s["kind"] == "image_numeric" else 1) for s in specs)

    def forward(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        cols = []
        for s in self.specs:
            key = str(s["index"])
            if s["kind"] == "tabular":
                cols.append(self.tab[key](parts[f"tab{key}"]))
            elif s["kind"] == "image":
                cols.append(self.cnn(parts[f"img{key}"]))
            else:
                cols.append(self.cnn(parts[f"img{key}"]))
                cols.append(self.num[key](parts[f"num{key}"]))
        return torch.cat(cols, 1) if cols else torch.zeros(0)


class Representation(nn.Module):
    """Z = fusion(concat(block embeddings, standardized X)) in R^4 (plan 8.4)."""

    def __init__(self, specs: list[dict], d_x: int) -> None:
        super().__init__()
        self.stack = BlockStack(specs)
        self.fusion = _mlp([self.stack.out_dim + d_x, 64, 32, Z_DIM])

    def forward(self, parts: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        emb = self.stack(parts)
        return self.fusion(torch.cat([emb, x], 1) if emb.numel() else x)


class BaseCritic(nn.Module):
    """q0(Z) in [0,1], width (128, 64) (plan 8.5)."""

    def __init__(self, d_z: int) -> None:
        super().__init__()
        self.net = _mlp([d_z, 128, 64, 1])

    def logit(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return bounded_prob(self.logit(z))


class AugmentedCritic(nn.Module):
    """q1(Z, T) whose class contains the base class (plan 8.5).

    The trunk is the base critic itself, shared by reference rather than copied, plus a residual branch on
    (T, Z) whose last layer starts at zero.

    What this does and does not establish.  Containment at the level of function classes held in the earlier
    implementation as well: that trunk had the same architecture, so copying the base weights and zeroing the
    residual reproduced the base function.  What sharing adds is tighter: with the residual at zero the
    augmented critic equals the base critic *currently being trained*, at every step.  It does not by itself
    make the empirical gap nonnegative, since gradient descent with early stopping need not reach the point
    where the residual would be set to zero; a base critic predicting 0.5 and an augmented critic predicting
    0.9 on balanced treatment gives risks 0.25 and 0.41, hence a gap of -0.16, and both are inside this class.
    Sharing also changes the training method: the shared parameters now receive gradient from both losses, so
    the base critic is no longer fitted to its own risk alone, which is what Section 4 of the manuscript asks
    for.  `decoupled_critic_fit` provides the comparison against fitting each critic to its own risk.

    IMPLEMENTATION DECISION.  When the held-out set contains image blocks the residual branch needs a
    featurizer for T; it is given its own copy of the plan 6.3 CNN plus MLPs for numeric parts, so the critic
    and the representation see images through the same architecture.
    """

    def __init__(self, base: "BaseCritic", t_specs: list[dict], d_x: int, d_z: int | None = None) -> None:
        super().__init__()
        # Resolved at call time, not as a default argument: a default is evaluated once when the class is
        # defined, so `d_z = Z_DIM` would freeze the plan's four dimensions and silently mismatch any run
        # that sets a different Z_DIM.
        d_z = Z_DIM if d_z is None else d_z
        self.base = base
        self.t_stack = BlockStack(t_specs)
        self.residual = _mlp([self.t_stack.out_dim + d_x + d_z, 128, 64, 1])
        with torch.no_grad():
            self.residual[-1].weight.zero_()
            self.residual[-1].bias.zero_()

    def own_parameters(self):
        """Parameters that are not shared with the base critic."""
        return list(self.t_stack.parameters()) + list(self.residual.parameters())

    def logit(self, z: torch.Tensor, t_parts: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        emb = self.t_stack(t_parts)
        feat = torch.cat([emb, x, z], 1) if emb.numel() else torch.cat([x, z], 1)
        return self.base.logit(z) + self.residual(feat).squeeze(1)

    def forward(self, z, t_parts, x) -> torch.Tensor:
        return bounded_prob(self.logit(z, t_parts, x))


# ----------------------------------------------------------------------------- tensor assembly
def block_specs(data: dict, blocks: list[int]) -> list[dict]:
    specs = []
    for b in blocks:
        blk = data["blocks"][b]
        if blk["kind"] == "tabular":
            specs.append({"index": b, "kind": "tabular", "dim": blk["v"].shape[1]})
        elif blk["kind"] == "image":
            specs.append({"index": b, "kind": "image"})
        else:
            specs.append({"index": b, "kind": "image_numeric", "num_dim": blk["num"].shape[1]})
    return specs


class Standardizer:
    """Column statistics taken once on the fit split and never refit (plan 16.2).

    A plain class rather than a closure so that the worker processes can receive it by pickle.
    """

    def __init__(self, v: np.ndarray, log1p: bool = False) -> None:
        self.log1p = log1p
        w = np.log1p(v) if log1p else v
        self.mu = w.mean(0)
        self.sd = np.maximum(w.std(0), 1e-6)

    def __call__(self, v: np.ndarray) -> np.ndarray:
        w = np.log1p(v) if self.log1p else v
        return ((w - self.mu) / self.sd).astype(np.float32)


def make_parts(data: dict, blocks: list[int], idx: np.ndarray, standardizers: dict) -> dict[str, torch.Tensor]:
    parts = {}
    for b in blocks:
        blk = data["blocks"][b]
        if blk["kind"] == "tabular":
            parts[f"tab{b}"] = torch.as_tensor(standardizers[f"tab{b}"](blk["v"][idx]))
        elif blk["kind"] == "image":
            parts[f"img{b}"] = torch.as_tensor(blk["img"][idx])
        else:
            parts[f"img{b}"] = torch.as_tensor(blk["img"][idx])
            parts[f"num{b}"] = torch.as_tensor(standardizers[f"num{b}"](blk["num"][idx]))
    return parts


def build_standardizers(data: dict, fit_idx: np.ndarray) -> dict:
    """Counts get the log1p transform of plan 5.2 before standardizing; everything else is standardized."""
    st = {"X": Standardizer(data["X"][fit_idx])}
    for b, blk in enumerate(data["blocks"]):
        if blk["kind"] == "tabular":
            st[f"tab{b}"] = Standardizer(blk["v"][fit_idx], log1p=bool(blk.get("counts")))
        elif blk["kind"] == "image_numeric":
            st[f"num{b}"] = Standardizer(blk["num"][fit_idx])
    return st


def slice_parts(parts: dict[str, torch.Tensor], sel: torch.Tensor) -> dict[str, torch.Tensor]:
    return {k: v[sel] for k, v in parts.items()}


# ----------------------------------------------------------------------------- representation training
def objective(gap: float) -> float:
    """The manuscript's empirical objective, max(R0 - R1, 0), evaluated at a signed risk gap."""
    return max(gap, 0.0) if math.isfinite(gap) else math.inf


def checkpoint_key(gap: float) -> tuple[float, float]:
    """Sort key for a candidate checkpoint, smallest first; both entries are plain floats.

    In "manuscript" mode the primary entry is the clipped objective max(gap, 0) that Section 4 minimises,
    and distance from zero breaks ties among the epochs that reach zero.  In "variant" mode the primary entry
    is distance from zero, which is what this project has been running.  A non-finite gap sorts last in both.
    The key is always a pair, so callers never have to branch on the mode to compare two of them.
    """
    if not math.isfinite(gap):
        return (math.inf, math.inf)
    return (max(gap, 0.0), abs(gap)) if TRAINING_MODE == "manuscript" else (abs(gap), abs(gap))


def checkpoint_improves(new_gap: float, best_gap: float, min_delta: float = None) -> bool:
    """Is `new_gap` a better checkpoint than `best_gap` by more than the tolerance?

    The rule is one function so that epoch selection and restart selection cannot drift apart, and so that
    the tolerance is applied to the quantity actually being minimised.  Round 7 selected on a key that is a
    pair in manuscript mode and then subtracted two pairs, which raised a TypeError on the first eligible
    epoch and aborted every manuscript-mode run.

    Order of tests:
      1. a non-finite candidate never wins, and any finite candidate beats a non-finite incumbent;
      2. the primary entry of the key must improve by more than `min_delta`;
      3. only when the primary entries are equal to within `min_delta` does the secondary entry decide, and
         it too must improve by more than `min_delta`.
    """
    d = MIN_DELTA if min_delta is None else min_delta
    if not math.isfinite(new_gap):
        return False
    if not math.isfinite(best_gap):
        return True
    (p_new, s_new), (p_old, s_old) = checkpoint_key(new_gap), checkpoint_key(best_gap)
    if p_old - p_new > d:
        return True
    if p_new - p_old > d:
        return False
    return s_old - s_new > d


def train_representation(data: dict, S: tuple[int, ...], d_fit: np.ndarray, d_val: np.ndarray,
                         st: dict, seed_parts: tuple, verbose: bool = False) -> dict:
    """Plan 8.6.  Alternating AdamW on the nested Brier risk gap; the encoder minimises the fit gap only."""
    rest = [b for b in range(dgp.N_BLOCKS) if b not in S]
    v_specs, t_specs = block_specs(data, rest), block_specs(data, list(S))
    x_all = torch.as_tensor(st["X"](data["X"]))
    a_all = torch.as_tensor(data["A"].astype(np.float32))
    v_parts = make_parts(data, rest, np.arange(len(data["A"])), st)
    t_parts = make_parts(data, list(S), np.arange(len(data["A"])), st)
    fit_t, val_t = torch.as_tensor(d_fit), torch.as_tensor(d_val)

    best = None
    for restart in range(N_RESTARTS):
        torch.manual_seed(int(np.random.SeedSequence(seed_parts + (restart,)).generate_state(1)[0]))
        enc = (ENCODER_FACTORY or Representation)(v_specs, x_all.shape[1])
        q0 = BaseCritic(Z_DIM)
        # In manuscript mode the augmented critic owns its trunk, so the two risks are minimised over two
        # separate parameter sets as Section 4 states.  In variant mode the trunk is shared by reference.
        q1 = AugmentedCritic(BaseCritic(Z_DIM) if TRAINING_MODE == "manuscript" else q0, t_specs,
                             x_all.shape[1])
        critic_params = (list(q0.parameters()) + list(q1.parameters()) if TRAINING_MODE == "manuscript"
                         else list(q0.parameters()) + q1.own_parameters())
        opt_e = torch.optim.AdamW(enc.parameters(), lr=ENC_LR, weight_decay=WEIGHT_DECAY)
        opt_c = torch.optim.AdamW(critic_params, lr=CRITIC_LR, weight_decay=WEIGHT_DECAY)
        gen = torch.Generator().manual_seed(restart + 17)
        # The state at initialisation is kept so that "improvement over initialisation" can be measured
        # against the starting point of the restart that was actually selected, not a fresh random draw.
        init_state = copy.deepcopy(enc.state_dict())
        best_gap, best_state, best_epoch, stale = float("inf"), None, 0, 0
        best_enc_updates = best_critic_updates = 0
        best_gap_fit = float("nan")
        # The encoder update cadence counts critic updates across epochs.  A per-epoch counter would never
        # reach five when the fit split holds fewer than five batches, leaving the encoder untrained.
        critic_updates = encoder_updates = 0
        for epoch in range(MAX_EPOCHS):
            perm = fit_t[torch.randperm(len(fit_t), generator=gen)]
            for start in range(0, len(perm), BATCH):
                sel = perm[start:start + BATCH]
                if len(sel) < 8:
                    continue
                zs = enc(slice_parts(v_parts, sel), x_all[sel])
                opt_c.zero_grad()
                zc = zs.detach()
                l0 = ((a_all[sel] - q0(zc)) ** 2).mean()
                l1 = ((a_all[sel] - q1(zc, slice_parts(t_parts, sel), x_all[sel])) ** 2).mean()
                # The summed loss is the same gradient for each parameter set when the sets are disjoint,
                # so this line implements both arrangements; what differs is whether they are disjoint.
                (l0 + l1).backward()
                nn.utils.clip_grad_norm_(critic_params, GRAD_CLIP)
                opt_c.step()
                critic_updates += 1
                if critic_updates % CRITIC_STEPS_PER_ENCODER == 0:
                    encoder_updates += 1
                    opt_e.zero_grad()
                    g0 = ((a_all[sel] - q0(zs)) ** 2).mean()
                    g1 = ((a_all[sel] - q1(zs, slice_parts(t_parts, sel), x_all[sel])) ** 2).mean()
                    # max(R0 - R1, 0) has zero gradient once the batch gap is already non-positive.
                    loss_e = torch.clamp(g0 - g1, min=0.0) if TRAINING_MODE == "manuscript" else g0 - g1
                    loss_e.backward()
                    nn.utils.clip_grad_norm_(enc.parameters(), GRAD_CLIP)
                    opt_e.step()
            with torch.no_grad():
                zv = enc(slice_parts(v_parts, val_t), x_all[val_t])
                gap = float(((a_all[val_t] - q0(zv)) ** 2 - (a_all[val_t] - q1(zv, slice_parts(t_parts, val_t), x_all[val_t])) ** 2).mean())
                zf = enc(slice_parts(v_parts, fit_t), x_all[fit_t])
                gap_fit = float(((a_all[fit_t] - q0(zf)) ** 2 - (a_all[fit_t] - q1(zf, slice_parts(t_parts, fit_t), x_all[fit_t])) ** 2).mean())
            if encoder_updates < MIN_ENCODER_UPDATES:
                # The checkpoint must never be an untrained representation.  With a small fit split an epoch
                # can finish before the encoder has had any update, and its gap is near zero only because both
                # critics are untrained.  A single update is not enough either, so a floor is required.
                continue
            # The tolerance applies to the improvement in the ranking key itself.  Testing the signed
            # difference |gap - best_gap| would let a move from +0.010 to -0.0099 count as an improvement of
            # 0.02 although the distance from zero barely changed (round 6 review).
            if checkpoint_improves(gap, best_gap):
                best_gap, best_epoch, stale = gap, epoch, 0
                best_gap_fit = gap_fit
                best_enc_updates, best_critic_updates = encoder_updates, critic_updates
                best_state = (copy.deepcopy(enc.state_dict()), copy.deepcopy(q0.state_dict()), copy.deepcopy(q1.state_dict()))
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        if best_state is None:
            # No eligible epoch: the run never updated the encoder.  Keep the final state and record it.
            best_state = (copy.deepcopy(enc.state_dict()), copy.deepcopy(q0.state_dict()), copy.deepcopy(q1.state_dict()))
            best_gap, best_epoch = float("nan"), -1
            best_enc_updates, best_critic_updates = encoder_updates, critic_updates
        enc.load_state_dict(best_state[0]); q0.load_state_dict(best_state[1]); q1.load_state_dict(best_state[2])
        with torch.no_grad():
            zv = enc(slice_parts(v_parts, val_t), x_all[val_t])
            prop_val = q0(zv).numpy()
        # Both critics predict the treatment probability, which the generator bounds, so they use the same
        # bounded map as the nuisance propensity.  The share below is then zero by construction and the
        # restart rule of plan 8.6 is vacuous in these designs; it is kept as a record, and when the bound is
        # switched off it recovers its original meaning.  Restart selection falls back to the smallest
        # validation gap, which is the primary rule of plan 8.6.
        viol = float(np.mean((prop_val < OVERLAP_LO - 1e-9) | (prop_val > OVERLAP_HI + 1e-9)))
        cand = {"enc": enc, "q0": q0, "q1": q1, "val_gap": best_gap, "fit_gap": best_gap_fit,
                "val_objective": objective(best_gap), "training_mode": TRAINING_MODE,
                "epoch": best_epoch, "init_state": init_state,
                "optimizer_seed": int(np.random.SeedSequence(seed_parts + (restart,)).generate_state(1)[0]),
                "restart": restart, "val_overlap_violation": viol,
                "encoder_updates": best_enc_updates, "critic_updates": best_critic_updates,
                "encoder_updates_at_end": encoder_updates,
                "restart_eligible": bool(viol <= RESTART_OVERLAP_TOL)}
        if best is None:
            best = cand
        else:
            ok_new, ok_old = viol <= RESTART_OVERLAP_TOL, best["val_overlap_violation"] <= RESTART_OVERLAP_TOL
            # Restart selection uses the same rule as epoch selection, with the same tolerance.
            if (ok_new and not ok_old) or (
                    ok_new == ok_old and checkpoint_improves(best_gap, best["val_gap"])):
                best = cand
    best["specs"] = (v_specs, t_specs)
    best["rest"] = rest
    best["any_restart_eligible"] = bool(best["restart_eligible"])
    return best


def refit_critics(data: dict, S: tuple[int, ...], fitted: dict, d_fit: np.ndarray, d_val: np.ndarray,
                  d_audit: np.ndarray, st: dict, seed: int, d_z: int | None = None) -> dict:
    """Diagnostic required to tell an overfitted augmented critic from an undertrained one.

    The representation is frozen at its selected checkpoint, both critics are trained again from scratch on
    D_fit with early stopping on D_val, and the two Brier risks are reported separately on the fit, validation
    and audit splits.  The plan's own audit (`audit_gap`) stays the primary quantity; this one is recorded
    alongside it so that a negative gap can be attributed.
    """
    rest = fitted["rest"]
    x_all = torch.as_tensor(st["X"](data["X"]))
    a_all = torch.as_tensor(data["A"].astype(np.float32))
    with torch.no_grad():
        z_all = fitted["enc"](make_parts(data, rest, np.arange(len(a_all)), st), x_all)
    t_parts = make_parts(data, list(S), np.arange(len(a_all)), st)
    fit_t, val_t, aud_t = (torch.as_tensor(i) for i in (d_fit, d_val, d_audit))
    d_z = z_all.shape[1] if d_z is None else d_z
    torch.manual_seed(seed % (2 ** 31))
    q0 = BaseCritic(d_z)
    # In manuscript mode the two risks are minimised over disjoint parameter sets, here as in training.
    # Round 7 review, PRD-2: the round 6 refit shared the trunk and minimised the summed loss, so calling its
    # output "the minimum risk of each class" was wrong in either mode.
    q1 = AugmentedCritic(BaseCritic(d_z) if TRAINING_MODE == "manuscript" else q0,
                         fitted["specs"][1], x_all.shape[1], d_z)
    refit_params = (list(q0.parameters()) + list(q1.parameters()) if TRAINING_MODE == "manuscript"
                    else list(q0.parameters()) + q1.own_parameters())
    opt = torch.optim.AdamW(refit_params, lr=CRITIC_LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed % (2 ** 20))
    best, stale, state = float("inf"), 0, None
    for _ in range(MAX_EPOCHS):
        perm = fit_t[torch.randperm(len(fit_t), generator=gen)]
        for start in range(0, len(perm), BATCH):
            sel = perm[start:start + BATCH]
            if len(sel) < 8:
                continue
            opt.zero_grad()
            l0 = ((a_all[sel] - q0(z_all[sel])) ** 2).mean()
            l1 = ((a_all[sel] - q1(z_all[sel], slice_parts(t_parts, sel), x_all[sel])) ** 2).mean()
            (l0 + l1).backward()
            nn.utils.clip_grad_norm_(refit_params, GRAD_CLIP)
            opt.step()
        with torch.no_grad():
            v = float(((a_all[val_t] - q0(z_all[val_t])) ** 2).mean()
                      + ((a_all[val_t] - q1(z_all[val_t], slice_parts(t_parts, val_t), x_all[val_t])) ** 2).mean())
        if v < best - MIN_DELTA:
            best, stale = v, 0
            state = (copy.deepcopy(q0.state_dict()), copy.deepcopy(q1.state_dict()))
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if state is not None:
        q0.load_state_dict(state[0]); q1.load_state_dict(state[1])
    out = {}
    with torch.no_grad():
        for name, idx in (("fit", fit_t), ("val", val_t), ("audit", aud_t)):
            r0 = ((a_all[idx] - q0(z_all[idx])) ** 2)
            r1 = ((a_all[idx] - q1(z_all[idx], slice_parts(t_parts, idx), x_all[idx])) ** 2)
            out[f"refit_R0_{name}"] = float(r0.mean())
            out[f"refit_R1_{name}"] = float(r1.mean())
            out[f"refit_gap_{name}"] = float((r0 - r1).mean())
        l0 = ((a_all[aud_t] - q0(z_all[aud_t])) ** 2).numpy()
        l1 = ((a_all[aud_t] - q1(z_all[aud_t], slice_parts(t_parts, aud_t), x_all[aud_t])) ** 2).numpy()
    diff = l0 - l1
    out["refit_se_audit"] = float(diff.std(ddof=1) / math.sqrt(len(diff)))
    # Row-level output.  Two representations scored on the same rows give a *paired* comparison; combining
    # their separate standard errors as if independent both ignores the pairing and misses the fact that the
    # evaluation sample itself is finite.  Callers align these by `audit_rows` and difference them row by row
    # (round 7 review, R2).
    out["audit_rows"] = np.asarray(d_audit, dtype=np.int64)
    out["audit_loss_base"] = l0.astype(np.float64)
    out["audit_loss_augmented"] = l1.astype(np.float64)
    out["audit_gap_rows"] = diff.astype(np.float64)
    return out


def common_critic_evaluate(data: dict, S: tuple[int, ...], fitted: dict, d_fit: np.ndarray,
                           d_stop: np.ndarray, d_select: np.ndarray, d_audit: np.ndarray,
                           st: dict, seed: int) -> dict:
    """Score a frozen representation with the same independently stopped critics for every candidate.

    The base and augmented critics own disjoint parameters and each is early-stopped on its own Brier risk.
    Both are fitted once on ``d_fit`` using ``d_stop``.  Their frozen predictions are then scored on the
    selection rows and the untouched audit rows.  The caller supplies the same split ids, initialisation seed,
    architecture and budget for every grid value and learned coefficient.
    """
    assert not (set(map(int, d_select)) & set(map(int, d_fit))
                or set(map(int, d_select)) & set(map(int, d_stop)))
    assert not (set(map(int, d_audit)) & set(map(int, d_fit))
                or set(map(int, d_audit)) & set(map(int, d_stop))
                or set(map(int, d_audit)) & set(map(int, d_select)))
    rest = fitted["rest"]
    x_all = torch.as_tensor(st["X"](data["X"]))
    a_all = torch.as_tensor(data["A"].astype(np.float32))
    with torch.no_grad():
        z_all = fitted["enc"](make_parts(data, rest, np.arange(len(a_all)), st), x_all)
    t_parts = make_parts(data, list(S), np.arange(len(a_all)), st)
    fit_t, stop_t = torch.as_tensor(d_fit), torch.as_tensor(d_stop)
    torch.manual_seed(seed % (2 ** 31))
    q0 = BaseCritic(z_all.shape[1])
    q1 = AugmentedCritic(BaseCritic(z_all.shape[1]), fitted["specs"][1], x_all.shape[1], z_all.shape[1])

    def _fit_one(params, loss_fn, generator_seed: int) -> float:
        params = list(params)
        opt = torch.optim.AdamW(params, lr=CRITIC_LR, weight_decay=WEIGHT_DECAY)
        gen = torch.Generator().manual_seed(generator_seed % (2 ** 31))
        best, stale, state = float("inf"), 0, None
        for _ in range(MAX_EPOCHS):
            perm = fit_t[torch.randperm(len(fit_t), generator=gen)]
            for start in range(0, len(perm), BATCH):
                sel = perm[start:start + BATCH]
                if len(sel) < 8:
                    continue
                opt.zero_grad(); loss_fn(sel).backward()
                nn.utils.clip_grad_norm_(params, GRAD_CLIP); opt.step()
            with torch.no_grad():
                risk = float(loss_fn(stop_t))
            if risk < best - MIN_DELTA:
                best, stale = risk, 0
                state = [q.detach().clone() for q in params]
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        if state is None:
            raise RuntimeError("common critic produced no finite early-stopping checkpoint")
        with torch.no_grad():
            for q, saved in zip(params, state):
                q.copy_(saved)
        return best

    def loss0(sel):
        return ((a_all[sel] - q0(z_all[sel])) ** 2).mean()

    def loss1(sel):
        return ((a_all[sel] - q1(z_all[sel], slice_parts(t_parts, sel), x_all[sel])) ** 2).mean()

    stop0 = _fit_one(q0.parameters(), loss0, seed + 101)
    stop1 = _fit_one(q1.parameters(), loss1, seed + 202)
    out = {"q0_stop_risk": float(stop0), "q1_stop_risk": float(stop1),
           "critic_protocol": "independent parameters; each early-stopped on its own Brier risk"}
    with torch.no_grad():
        for name, rows in (("select", d_select), ("audit", d_audit)):
            idx = torch.as_tensor(rows)
            r0 = (a_all[idx] - q0(z_all[idx])) ** 2
            r1 = (a_all[idx] - q1(z_all[idx], slice_parts(t_parts, idx), x_all[idx])) ** 2
            gap = r0 - r1
            out[f"R0_{name}"] = float(r0.mean())
            out[f"R1_{name}"] = float(r1.mean())
            out[f"gap_{name}"] = float(gap.mean())
            out[f"se_{name}"] = float(gap.numpy().std(ddof=1) / math.sqrt(len(idx)))
            out[f"rows_{name}"] = np.asarray(rows, dtype=np.int64)
            out[f"gap_rows_{name}"] = gap.numpy().astype(np.float64)
    return out


def decoupled_critic_fit(data: dict, S: tuple[int, ...], fitted: dict, d_fit: np.ndarray, d_val: np.ndarray,
                         d_audit: np.ndarray, st: dict, seed: int) -> dict:
    """Fit the base critic to its own risk, then start the augmented critic from it and fit only the residual.

    This is the arrangement Section 4 describes: each critic close to the minimum risk of its own class.  It is
    reported next to the shared-parameter fit so that the two can be compared rather than assumed equivalent.
    """
    rest = fitted["rest"]
    x_all = torch.as_tensor(st["X"](data["X"]))
    a_all = torch.as_tensor(data["A"].astype(np.float32))
    with torch.no_grad():
        z_all = fitted["enc"](make_parts(data, rest, np.arange(len(a_all)), st), x_all)
    t_parts = make_parts(data, list(S), np.arange(len(a_all)), st)
    fit_t, val_t, aud_t = (torch.as_tensor(i) for i in (d_fit, d_val, d_audit))
    torch.manual_seed((seed + 77) % (2 ** 31))

    def _train(params, loss_fn):
        opt = torch.optim.AdamW(params, lr=CRITIC_LR, weight_decay=WEIGHT_DECAY)
        gen = torch.Generator().manual_seed((seed + 5) % (2 ** 20))
        best, stale, state = float("inf"), 0, None
        for _ in range(MAX_EPOCHS):
            perm = fit_t[torch.randperm(len(fit_t), generator=gen)]
            for start in range(0, len(perm), BATCH):
                sel = perm[start:start + BATCH]
                if len(sel) < 8:
                    continue
                opt.zero_grad(); loss_fn(sel).backward()
                nn.utils.clip_grad_norm_(params, GRAD_CLIP); opt.step()
            with torch.no_grad():
                v = float(loss_fn(val_t))
            if v < best - MIN_DELTA:
                best, stale = v, 0
                state = [q.detach().clone() for q in params]
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        if state is not None:
            with torch.no_grad():
                for q, saved in zip(params, state):
                    q.copy_(saved)
        return best

    d_z = z_all.shape[1]
    q0 = BaseCritic(d_z)
    _train(list(q0.parameters()), lambda sel: ((a_all[sel] - q0(z_all[sel])) ** 2).mean())
    # Section 4 asks for two separate risk minimisations over two classes.  The augmented critic therefore
    # owns an independent copy of the trunk, initialised at the fitted base critic so that it starts inside
    # the base class, and every one of its parameters is trained on its own risk.  Freezing the copied trunk
    # and fitting only the residual, as the first version did, is a smaller problem and does not test the
    # manuscript's arrangement (round 6 review).
    q1 = AugmentedCritic(BaseCritic(d_z), fitted["specs"][1], x_all.shape[1], d_z)
    q1.base.load_state_dict(q0.state_dict())
    _train(list(q1.parameters()),
           lambda sel: ((a_all[sel] - q1(z_all[sel], slice_parts(t_parts, sel), x_all[sel])) ** 2).mean())
    out = {}
    with torch.no_grad():
        for name, idx in (("fit", fit_t), ("val", val_t), ("audit", aud_t)):
            r0 = ((a_all[idx] - q0(z_all[idx])) ** 2).mean()
            r1 = ((a_all[idx] - q1(z_all[idx], slice_parts(t_parts, idx), x_all[idx])) ** 2).mean()
            out[f"decoupled_gap_{name}"] = float(r0 - r1)
    return out


def critic_refit_risk_difference(data: dict, S: tuple[int, ...], fitted: dict, d_fit: np.ndarray,
                                 d_val: np.ndarray, d_audit: np.ndarray, st: dict, seed: int,
                                 restarts: int = 3) -> dict:
    """Risk of each training-loop critic minus the lowest risk found by refitting that critic class.

    What this is.  With the representation frozen, each critic class is refitted from scratch `restarts`
    times from different initialisations, with early stopping on the validation split, and the lowest
    validation risk is kept.  The difference from the training-loop critic's risk on the same rows is
    reported, on the fit, validation and audit splits separately.

    What this is not.  It is not the optimisation error term of Section 4.  That term is the distance to the
    infimum over the whole critic class, and a finite number of restarts of the same optimiser certifies
    nothing about that infimum.  A positive value here is evidence that the training-loop critic left risk on
    the table; a value near zero is the absence of such evidence, not a proof of near-optimality.  The name
    says what is computed (round 7 review, PRD-2).
    """
    rest = fitted["rest"]
    x_all = torch.as_tensor(st["X"](data["X"]))
    a_all = torch.as_tensor(data["A"].astype(np.float32))
    with torch.no_grad():
        z_all = fitted["enc"](make_parts(data, rest, np.arange(len(a_all)), st), x_all)
    t_parts = make_parts(data, list(S), np.arange(len(a_all)), st)
    idx = {"fit": torch.as_tensor(d_fit), "val": torch.as_tensor(d_val), "audit": torch.as_tensor(d_audit)}

    out = {"restarts": restarts, "training_mode": TRAINING_MODE}
    with torch.no_grad():
        for name, sel in idx.items():
            out[f"R0_loop_{name}"] = float(((a_all[sel] - fitted["q0"](z_all[sel])) ** 2).mean())
            out[f"R1_loop_{name}"] = float(((a_all[sel] - fitted["q1"](
                z_all[sel], slice_parts(t_parts, sel), x_all[sel])) ** 2).mean())

    best = {}
    for k in range(restarts):
        ref = refit_critics(data, S, fitted, d_fit, d_val, d_audit, st, seed + 977 * k)
        for critic in ("R0", "R1"):
            key = f"{critic}_val"
            if key not in best or ref[f"refit_{critic}_val"] < best[key][f"refit_{critic}_val"]:
                best[key] = ref
    for critic in ("R0", "R1"):
        ref = best[f"{critic}_val"]
        for name in ("fit", "val", "audit"):
            out[f"{critic}_refit_{name}"] = ref[f"refit_{critic}_{name}"]
            out[f"{critic}_excess_{name}"] = out[f"{critic}_loop_{name}"] - ref[f"refit_{critic}_{name}"]
    for name in ("fit", "val", "audit"):
        out[f"gap_loop_{name}"] = out[f"R0_loop_{name}"] - out[f"R1_loop_{name}"]
        out[f"objective_loop_{name}"] = objective(out[f"gap_loop_{name}"])
    out["note"] = ("excess risk against the best of a finite number of refits; not a certificate of the "
                   "class minimum and not the Section 4 optimisation error")
    return out


def encoder_objective_report(data: dict, S: tuple[int, ...], fitted: dict, d_fit: np.ndarray,
                             st: dict) -> dict:
    """The manuscript objective at the selected checkpoint, next to the surrogate actually minimised.

    Section 4 minimises max(Rhat0 - Rhat1, 0) over the whole fitting sample.  The training loop takes a
    gradient step on max(g0 - g1, 0) computed on one minibatch, which is not the same function: two batches
    with gaps -0.02 and +0.02 give 0 for the full-sample objective and 0.01 for the average of the clipped
    batch values.  Both numbers are reported so the difference is visible rather than assumed away
    (round 7 review, PRD-2).
    """
    rest = fitted["rest"]
    x_all = torch.as_tensor(st["X"](data["X"]))
    a_all = torch.as_tensor(data["A"].astype(np.float32))
    fit_t = torch.as_tensor(d_fit)
    t_parts = make_parts(data, list(S), np.arange(len(a_all)), st)
    with torch.no_grad():
        z = fitted["enc"](make_parts(data, rest, np.arange(len(a_all)), st), x_all)
        per_row = (((a_all[fit_t] - fitted["q0"](z[fit_t])) ** 2)
                   - ((a_all[fit_t] - fitted["q1"](z[fit_t], slice_parts(t_parts, fit_t),
                                                   x_all[fit_t])) ** 2)).numpy()
    full = float(per_row.mean())
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(per_row))
    batches = [per_row[perm[i:i + BATCH]] for i in range(0, len(perm), BATCH) if len(perm[i:i + BATCH]) >= 8]
    surrogate = float(np.mean([max(b.mean(), 0.0) for b in batches])) if batches else float("nan")
    return {"full_sample_gap": full, "manuscript_objective": objective(full),
            "minibatch_surrogate": surrogate, "surrogate_minus_objective": surrogate - objective(full),
            "batches": len(batches), "batch_size": BATCH,
            "note": "the encoder minimises the minibatch surrogate; the manuscript objective is the "
                    "full-sample clipped gap"}


def audit_gap(data: dict, S: tuple[int, ...], fitted: dict, d_audit: np.ndarray, st: dict) -> dict:
    """Plan 8.5 and 8.7.  Row-level Brier loss differences on the independent audit split."""
    rest = fitted["rest"]
    x_all = torch.as_tensor(st["X"](data["X"]))
    a_all = torch.as_tensor(data["A"].astype(np.float32))
    idx = torch.as_tensor(d_audit)
    with torch.no_grad():
        z = fitted["enc"](slice_parts(make_parts(data, rest, d_audit, st), torch.arange(len(d_audit))), x_all[idx])
        t_parts = make_parts(data, list(S), d_audit, st)
        d0 = (a_all[idx] - fitted["q0"](z)) ** 2
        d1 = (a_all[idx] - fitted["q1"](z, t_parts, x_all[idx])) ** 2
        diff = (d0 - d1).numpy()
    gap = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(len(diff)))
    thr = SCREEN_CONST + 2.0 * se
    return {"d2_raw": gap, "d2_screen": max(0.0, gap), "se": se, "threshold": thr,
            "discrepancy_pass": bool(max(0.0, gap) <= thr), "margin": float(thr - max(0.0, gap))}


def embed(data: dict, fitted: dict, idx: np.ndarray, st: dict) -> np.ndarray:
    x_all = torch.as_tensor(st["X"](data["X"][idx]))
    with torch.no_grad():
        parts = make_parts(data, fitted["rest"], idx, st)
        return fitted["enc"](parts, x_all).numpy().astype(np.float32)


# ----------------------------------------------------------------------------- nuisances and AIPW
def _fit_head(xf, yf, xv, yv, loss_kind: str, seed: int) -> nn.Module:
    best_model, best_loss = None, float("inf")
    for restart in range(NUIS_RESTARTS):
        torch.manual_seed(seed + 101 * restart)
        net = _mlp([xf.shape[1], 128, 64, 1])
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
        gen = torch.Generator().manual_seed(seed + restart)
        cur, stale, state = float("inf"), 0, None
        for _ in range(NUIS_MAX_EPOCHS):
            perm = torch.randperm(len(xf), generator=gen)
            for start in range(0, len(perm), BATCH):
                sel = perm[start:start + BATCH]
                if len(sel) < 8:
                    continue
                opt.zero_grad()
                out = net(xf[sel]).squeeze(1)
                loss = (nn.functional.binary_cross_entropy(bounded_prob(out), yf[sel]) if loss_kind == "bce"
                        else ((out - yf[sel]) ** 2).mean())
                loss.backward(); opt.step()
            with torch.no_grad():
                out = net(xv).squeeze(1)
                vl = float(nn.functional.binary_cross_entropy(bounded_prob(out), yv) if loss_kind == "bce"
                           else ((out - yv) ** 2).mean())
            if vl < cur - MIN_DELTA:
                cur, stale, state = vl, 0, copy.deepcopy(net.state_dict())
            else:
                stale += 1
                if stale >= NUIS_PATIENCE:
                    break
        if cur < best_loss:
            net.load_state_dict(state); best_loss, best_model = cur, net
    return best_model


def fit_nuisances(feat_n: np.ndarray, a_n: np.ndarray, y_n: np.ndarray, seed: int) -> dict:
    """Plan 8.9.  80/20 fit and early-stop split of N; identical heads for every adjustment set."""
    g = np.random.default_rng(np.random.SeedSequence((seed, 3)))
    perm = g.permutation(len(a_n)); cut = int(0.8 * len(a_n))
    fi, vi = perm[:cut], perm[cut:]
    st = Standardizer(feat_n[fi])
    xf, xv = torch.as_tensor(st(feat_n[fi])), torch.as_tensor(st(feat_n[vi]))
    af, av = torch.as_tensor(a_n[fi].astype(np.float32)), torch.as_tensor(a_n[vi].astype(np.float32))
    yf, yv = torch.as_tensor(y_n[fi].astype(np.float32)), torch.as_tensor(y_n[vi].astype(np.float32))
    prop = _fit_head(xf, af, xv, av, "bce", seed)
    heads = {}
    for arm in (1, 0):
        mf, mv = af == arm, av == arm
        heads[arm] = _fit_head(xf[mf], yf[mf], xv[mv], yv[mv], "mse", seed + 7 + arm)
    return {"prop": prop, "out": heads, "st": st}


def propensity(nuis: dict, feat: np.ndarray) -> np.ndarray:
    """Fitted treatment probability.  When the generator bounds the true propensity, the fitted one is mapped
    into the same interval: the true P(A=1|Z) is an average of e(U,X,...) and therefore obeys the same bounds
    for every Z, so an unconstrained network can only leave the interval by estimation error.  Reported overlap
    is then a statement about the model, not a test that a candidate is valid."""
    with torch.no_grad():
        return bounded_prob(nuis["prop"](torch.as_tensor(nuis["st"](feat))).squeeze(1)).numpy()


def aipw_scores(nuis: dict, feat: np.ndarray, a: np.ndarray, y: np.ndarray) -> dict:
    e = propensity(nuis, feat)
    clip_rate = float(np.mean((e < CLIP_LO) | (e > CLIP_HI)))
    e = np.clip(e, CLIP_LO, CLIP_HI)
    with torch.no_grad():
        z = torch.as_tensor(nuis["st"](feat))
        m1 = nuis["out"][1](z).squeeze(1).numpy()
        m0 = nuis["out"][0](z).squeeze(1).numpy()
    psi = m1 - m0 + a * (y - m1) / e - (1 - a) * (y - m0) / (1 - e)
    return {"psi": psi, "theta": float(psi.mean()), "clip_rate": clip_rate,
            "sigma": float(psi.std(ddof=1))}


def overlap_pass(nuis: dict, feat_n: np.ndarray) -> dict:
    e = propensity(nuis, feat_n)
    share = float(np.mean((e >= OVERLAP_LO) & (e <= OVERLAP_HI)))
    return {"overlap_share": share, "overlap_pass": bool(share >= OVERLAP_MIN_SHARE)}


class BlockNet(nn.Module):
    """End-to-end network over raw blocks plus X, used by the image raw baselines (plan 9.2).

    Same CNN architecture, restart count and epoch cap as the representation, so that the raw comparator on
    DGP D and E is not handicapped by the feature map.
    """

    def __init__(self, specs: list[dict], d_x: int) -> None:
        super().__init__()
        self.stack = BlockStack(specs)
        self.head = _mlp([self.stack.out_dim + d_x, 128, 64, 1])

    def forward(self, parts, x) -> torch.Tensor:
        emb = self.stack(parts)
        return self.head(torch.cat([emb, x], 1) if emb.numel() else x).squeeze(1)


def _fit_block_net(data, specs, blocks, idx_fit, idx_val, st, target, loss_kind, seed):
    x_all = torch.as_tensor(st["X"](data["X"]))
    parts_fit = make_parts(data, blocks, idx_fit, st)
    parts_val = make_parts(data, blocks, idx_val, st)
    yf = torch.as_tensor(target[idx_fit].astype(np.float32))
    yv = torch.as_tensor(target[idx_val].astype(np.float32))
    xf, xv = x_all[torch.as_tensor(idx_fit)], x_all[torch.as_tensor(idx_val)]
    best_model, best_loss = None, float("inf")
    for restart in range(NUIS_RESTARTS):
        torch.manual_seed(seed + 211 * restart)
        net = BlockNet(specs, x_all.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
        gen = torch.Generator().manual_seed(seed + restart)
        cur, stale, state = float("inf"), 0, None
        for _ in range(NUIS_MAX_EPOCHS):
            perm = torch.randperm(len(yf), generator=gen)
            for start in range(0, len(perm), BATCH):
                sel = perm[start:start + BATCH]
                if len(sel) < 8:
                    continue
                opt.zero_grad()
                out = net(slice_parts(parts_fit, sel), xf[sel])
                loss = (nn.functional.binary_cross_entropy(bounded_prob(out), yf[sel]) if loss_kind == "bce"
                        else ((out - yf[sel]) ** 2).mean())
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
                opt.step()
            with torch.no_grad():
                out = net(parts_val, xv)
                vl = float(nn.functional.binary_cross_entropy(bounded_prob(out), yv) if loss_kind == "bce"
                           else ((out - yv) ** 2).mean())
            if vl < cur - MIN_DELTA:
                cur, stale, state = vl, 0, copy.deepcopy(net.state_dict())
            else:
                stale += 1
                if stale >= NUIS_PATIENCE:
                    break
        if cur < best_loss:
            net.load_state_dict(state); best_loss, best_model = cur, net
    return best_model


def fit_nuisances_blocks(data: dict, idx_n: np.ndarray, st: dict, seed: int) -> dict:
    """Raw baseline on every block, learned end to end (plan 9.2).  Used for DGP D and E."""
    blocks = list(range(dgp.N_BLOCKS))
    specs = block_specs(data, blocks)
    g = np.random.default_rng(np.random.SeedSequence((seed, 5)))
    perm = idx_n[g.permutation(len(idx_n))]
    cut = int(0.8 * len(perm))
    fi, vi = perm[:cut], perm[cut:]
    a, y = data["A"], data["Y"]
    prop = _fit_block_net(data, specs, blocks, fi, vi, st, a, "bce", seed)
    heads = {}
    for arm in (1, 0):
        fa, va = fi[a[fi] == arm], vi[a[vi] == arm]
        heads[arm] = _fit_block_net(data, specs, blocks, fa, va, st, y, "mse", seed + 7 + arm)
    return {"prop": prop, "out": heads, "st": st, "blocks": blocks, "kind": "blocks"}


def _block_predict(net, data, blocks, idx, st) -> np.ndarray:
    x_all = torch.as_tensor(st["X"](data["X"][idx]))
    with torch.no_grad():
        return net(make_parts(data, blocks, idx, st), x_all).numpy()


def aipw_scores_blocks(nuis: dict, data: dict, idx: np.ndarray) -> dict:
    st, blocks = nuis["st"], nuis["blocks"]
    e = bounded_prob(torch.as_tensor(_block_predict(nuis["prop"], data, blocks, idx, st))).numpy()
    clip_rate = float(np.mean((e < CLIP_LO) | (e > CLIP_HI)))
    e = np.clip(e, CLIP_LO, CLIP_HI)
    m1 = _block_predict(nuis["out"][1], data, blocks, idx, st)
    m0 = _block_predict(nuis["out"][0], data, blocks, idx, st)
    a, y = data["A"][idx], data["Y"][idx]
    psi = m1 - m0 + a * (y - m1) / e - (1 - a) * (y - m0) / (1 - e)
    return {"psi": psi, "theta": float(psi.mean()), "clip_rate": clip_rate, "sigma": float(psi.std(ddof=1))}


def overlap_pass_blocks(nuis: dict, data: dict, idx: np.ndarray) -> dict:
    e = bounded_prob(torch.as_tensor(_block_predict(nuis["prop"], data, nuis["blocks"], idx, nuis["st"]))).numpy()
    share = float(np.mean((e >= OVERLAP_LO) & (e <= OVERLAP_HI)))
    return {"overlap_share": share, "overlap_pass": bool(share >= OVERLAP_MIN_SHARE)}


def screen_decision(refit_gap: float, refit_se: float, overlap_pass: bool, restart_eligible: bool,
                    epoch: int, encoder_updates: int) -> dict:
    """The single place where a candidate is accepted or rejected.

    A candidate must pass the audit computed with the representation frozen and the critics refitted, must
    come from a restart that met the validation tolerance, and must have been trained at all.  The overlap
    screen is required only when the propensity is left unbounded; with the structural bound in force it is
    satisfied by construction and would otherwise be a rule that never fires.
    """
    reasons = []
    # A comparison against a NaN is false, so a degenerate statistic would otherwise slip through every test.
    if not (math.isfinite(refit_gap) and math.isfinite(refit_se)) or refit_se < 0:
        reasons.append("audit statistic not finite")
    elif refit_gap > 0.02 + 2 * refit_se:
        reasons.append("refit audit")
    if not restart_eligible:
        reasons.append("restart tolerance")
    if epoch < 0 or encoder_updates < MIN_ENCODER_UPDATES:
        reasons.append("training floor")
    if PROPENSITY_BOUND is None and not overlap_pass:
        reasons.append("overlap")
    return {"screen_pass": not reasons, "reject_reasons": reasons,
            "overlap_screen_binding": PROPENSITY_BOUND is None}


# ----------------------------------------------------------------------------- linking and aggregation
def linking_radius(psi_matrix: np.ndarray, thetas: np.ndarray, seed: int) -> float:
    """Plan 8.10.  One Gaussian multiplier bootstrap shared across candidates and units.

    Columns that no candidate evaluated are dropped, so a single-rotation run divides by the number of units
    it actually scored rather than by the whole sample.
    """
    if psi_matrix.shape[0] == 0:
        return 0.0
    keep = ~np.all(np.isnan(psi_matrix), axis=0)
    psi_matrix = np.nan_to_num(psi_matrix[:, keep], nan=0.0)
    if psi_matrix.shape[1] == 0:
        return 0.0
    g = np.random.default_rng(np.random.SeedSequence((seed, 11)))
    n = psi_matrix.shape[1]
    centered = psi_matrix - thetas[:, None]
    maxima = np.empty(BOOTSTRAP_DRAWS)
    chunk = 250
    for start in range(0, BOOTSTRAP_DRAWS, chunk):
        size = min(chunk, BOOTSTRAP_DRAWS - start)
        gm = g.standard_normal((n, size))
        maxima[start:start + size] = np.abs(centered @ gm / n).max(0)
    return float(np.quantile(maxima, BOOTSTRAP_LEVEL))


def aggregate(labels: list, thetas: np.ndarray, rho: float, extra: float = 0.0) -> dict:
    """Plan 8.10.  Components by |theta - theta'| <= 2 rho + extra; largest component median; ties kept."""
    k = len(thetas)
    if k == 0:
        return {"return_kind": "no_return", "output": None, "candidates": [], "component_sizes": [],
                "rho": rho, "members": []}
    parent = list(range(k))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for i in range(k):
        for j in range(i + 1, k):
            if abs(thetas[i] - thetas[j]) <= 2 * rho + extra:
                parent[find(i)] = find(j)
    comps: dict[int, list[int]] = {}
    for i in range(k):
        comps.setdefault(find(i), []).append(i)
    sizes = {key: len(v) for key, v in comps.items()}
    top = max(sizes.values())
    largest = [key for key, s in sizes.items() if s == top]
    medians = [float(np.median(thetas[comps[key]])) for key in largest]
    if k == 1:
        kind = "single_survivor"
    elif len(largest) == 1:
        kind = "unique_largest"
    else:
        kind = "tied_largest"
    return {"return_kind": kind, "output": medians[0] if len(largest) == 1 else None,
            "candidates": medians, "component_sizes": sorted(sizes.values(), reverse=True), "rho": rho,
            "members": [[labels[i] for i in comps[key]] for key in largest]}
