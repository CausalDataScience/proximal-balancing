"""Direct image balancing: the networks and the alternating fit
(PRD memo/2026-09-21-direct-image-balancing-prd-v1.md, sections 3 and 4).

For one candidate split S the learner is handed the kept images and X, and must produce a two-dimensional
representation h such that, on rows it has never seen, knowing the held-out images adds nothing to what Z = (X, h)
already says about the treatment.  That is the whole training signal.  No orientation, no digit, no
reconstruction, no outcome.

    representation   one small CNN per image position, then (kept image features, X) -> MLP -> h (2 numbers);
                     the module returns Z = (X, h), so X reaches every critic and every estimate untouched
    base critic      q0(Z) -> P(A = 1)
    augmented critic q1(Z, held-out pixels) -> P(A = 1), built so that zeroing the held-out branch's output
                     weights reproduces q0 exactly, which is what makes the gap a gap and not a difference of
                     two unrelated fits

The objective is the manuscript's clipped Brier gap on the WHOLE representation fold, never a mean of per-batch
clipped gaps: those are different functions (PRD section 4), and the second one rewards a network for making the
gap negative on some batches.  So the gradient is accumulated over the fold, and the CNN moves only when the
fold's signed gap is positive.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

import direct_balance_data as db
import family_v2_images as im

REPRESENTATION_DIM = 2
FEATURE_DIM = 32
CRITIC_HIDDEN = (64, 64)
TRAIN = {"critic_initial": 100, "outer_iters": 40, "critic_per_iter": 25, "batch": 256,
         "lr_critic": 1e-3, "lr_cnn": 1e-3, "checkpoints": (0, 10, 20, 30, 40), "grad_batch": 512}
TRAIN_CORRECTION = {
    "what": "the representation's learning rate is 1e-3, not the 1e-4 the PRD started from, and there are forty "
            "outer iterations, not twenty",
    "why": "at 1e-4 the gap on an independent fold fell from 2.0e-2 only to 1.0e-2 in twenty updates, while the "
           "screen threshold is 1.5e-3.  At 1e-3 it crosses the threshold at iteration 35 and reaches -2.2e-3 "
           "by forty.  Five updates per refit instead of one reached the same place for more compute, so the "
           "update count stays at one per refit.",
    "checked_not_to_break_the_screen": "at this setting S1 and S123, which are balanceable in SCM-1, end below "
                                       "the threshold (-7.8e-3 and -3.6e-2) and S5, which is not, ends above it "
                                       "(+2.6e-2).  A faster rate of 3e-3 with twenty iterations was rejected "
                                       "because it left S1 above the threshold, a false negative on a split "
                                       "that should pass.",
    "authority": "PRD section 4 calls its settings a starting point, and section 8 (P3) allows one narrow "
                 "correction when development finds an optimisation problem.  This is that one correction.",
    "diagnostics": "results/direct_balance/p1_checks_v1.json and the P1 memo"}
LINK = (0.1, 0.8)                     # the sealed bounded link: 0.1 + 0.8 * sigmoid, so a critic never says 0 or 1


def link(logit: torch.Tensor) -> torch.Tensor:
    return LINK[0] + LINK[1] * torch.sigmoid(logit)


SPECS = {"shapes3d": {"channels": 3, "size": 64}, "mnist": {"channels": 1, "size": 28}}


def device() -> torch.device:
    """CUDA on DeltaAI, MPS on the Mac, the CPU otherwise.  `family_v2_images.device` is sealed and knows only
    the last two."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ImageTower(nn.Module):
    """A small convolutional stack, one per image position.  Deliberately smaller than the SCM-4 reader: every
    candidate split trains its own copy, so the cost of this stack is paid thirty times per data set."""

    def __init__(self, spec: dict | None = None, out_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        spec = spec or SPECS["shapes3d"]
        if spec.get("numeric"):                             # diagnosis only: the "image" is one exact number
            self.body = nn.Sequential(nn.Linear(spec.get("width", 1), out_dim))
            return
        layers, channels, side = [], spec["channels"], spec["size"]
        while side > 4:                                     # stride-two convolutions down to a 4 x 4 map
            out = 16 if not layers else 32
            layers += [nn.Conv2d(channels, out, 3, 2, 1), nn.ReLU()]
            channels, side = out, (side + 1) // 2
        self.body = nn.Sequential(*layers, nn.Flatten(), nn.Linear(channels * side * side, out_dim))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.body(pixels)


class Representation(nn.Module):
    """(kept images, X) -> Z = (X, h), where h is the CNN's two numbers.

    X is carried into the output untouched.  Feeding X to the head is not the same thing: a head may compress X
    or drop it, and then a critic that is handed the head's output alone is judging balance given h, not given
    (X, h), while the effect estimate adjusts for (X, h).  That mismatch was a real bug in the first version of
    this file.  Now this forward pass is the ONLY place Z is assembled, and training, checkpoint selection, the
    independent check and the effect estimate all consume what it returns.

    It never receives a held-out image: `kept` fixes which image positions exist at all.
    """

    def __init__(self, kept: np.ndarray, x_dim: int, dim: int = REPRESENTATION_DIM,
                 spec: dict | None = None) -> None:
        super().__init__()
        self.kept = np.asarray(kept)
        self.towers = nn.ModuleList([ImageTower(spec) for _ in self.kept])
        self.head = nn.Sequential(nn.Linear(len(self.kept) * FEATURE_DIM + x_dim, 64), nn.ReLU(),
                                  nn.Linear(64, dim))

    def forward(self, pixels: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        if len(pixels) != len(self.towers):
            raise ValueError(f"{len(pixels)} image columns for {len(self.towers)} kept positions")
        feats = [tower(p) for tower, p in zip(self.towers, pixels)]
        return torch.cat([x, self.head(torch.cat(feats + [x], dim=1))], dim=1)


class BaseCritic(nn.Module):
    """q0: the treatment probability from the representation alone."""

    def __init__(self, z_dim: int) -> None:
        super().__init__()
        h1, h2 = CRITIC_HIDDEN
        self.net = nn.Sequential(nn.Linear(z_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1))

    def logit(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return link(self.logit(z))


class AugmentedCritic(nn.Module):
    """q1: the same function of the representation, plus what the held-out pixels add.

    The held-out contribution is a separate CNN whose output layer starts at zero, and the representation branch
    starts as a copy of the fitted q0.  So at the start of its fit q1 IS q0, and the gap it measures is what the
    held-out images add, not the difference between two independently initialised networks.
    """

    def __init__(self, base: BaseCritic, n_held: int, z_dim: int, spec: dict | None = None) -> None:
        super().__init__()
        h1, h2 = CRITIC_HIDDEN
        self.net = nn.Sequential(nn.Linear(z_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1))
        self.net.load_state_dict(base.net.state_dict())
        self.towers = nn.ModuleList([ImageTower(spec) for _ in range(n_held)])
        self.extra = nn.Sequential(nn.Linear(n_held * FEATURE_DIM + z_dim, h1), nn.ReLU(), nn.Linear(h1, 1))
        nn.init.zeros_(self.extra[-1].weight)
        nn.init.zeros_(self.extra[-1].bias)

    def logit(self, z: torch.Tensor, held: list[torch.Tensor]) -> torch.Tensor:
        feats = [tower(p) for tower, p in zip(self.towers, held)]
        return self.net(z).squeeze(1) + self.extra(torch.cat(feats + [z], dim=1)).squeeze(1)

    def forward(self, z: torch.Tensor, held: list[torch.Tensor]) -> torch.Tensor:
        return link(self.logit(z, held))


def brier(pred: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return ((a - pred) ** 2).mean()


def to_input(pixels_u8: torch.Tensor) -> torch.Tensor:
    """uint8 (n, 64, 64, 3) -> float32 (n, 3, 64, 64) in [0, 1], laid out the way a convolution wants it.

    Numerically this is `family_v2_images.to_input`, which is sealed and stays as it is.  The difference is the
    final `.contiguous()`: without it the tensor keeps the permuted layout, and convolutions on this device run
    3.2 times slower on it (921 ms against 288 ms for 512 rows by 5 images, with the copy itself costing 77 ms).
    """
    if pixels_u8.dim() == 3:                                 # grey images carry no channel axis
        return (pixels_u8.unsqueeze(1).to(dtype=torch.float32) / 255.0).contiguous()
    return (pixels_u8.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0).contiguous()


CACHE_LIMIT_BYTES = 3 * 2 ** 30


class Batches:
    """Pixels for a set of rows.

    Training walks over the same few thousand rows twenty times, so the images those rows need are copied to the
    device once, as uint8, and every batch is then a gather that never touches the host.  Rows outside the cache
    (the estimation folds, which are read once) fall back to gathering from the memory-mapped bank.
    """

    def __init__(self, images, src: np.ndarray, x: np.ndarray, a: np.ndarray, dev,
                 cache_rows: np.ndarray | None = None, standardise_rows: np.ndarray | None = None) -> None:
        self.images, self.src, self.dev = images, np.asarray(src), dev
        x = np.asarray(x, dtype=np.float64)
        ref = x if standardise_rows is None else x[np.asarray(standardise_rows)]
        self.x_mu, self.x_sd = ref.mean(0), ref.std(0) + 1e-12      # one scaling of X, used by every stage
        self.x = torch.as_tensor((x - self.x_mu) / self.x_sd, dtype=torch.float32, device=dev)
        self.a = torch.as_tensor(a, dtype=torch.float32, device=dev)
        self.cache_index, self.cache = None, None
        if cache_rows is not None:
            wanted = np.unique(self.src[np.asarray(cache_rows)].ravel())
            if wanted.size * int(np.prod(self.images.shape[1:])) <= CACHE_LIMIT_BYTES:
                self.cache_index = wanted
                self.cache = torch.as_tensor(np.asarray(self.images[wanted]), device=dev)

    def cache_report(self) -> dict:
        if self.cache is None:
            return {"cached_images": 0, "megabytes": 0}
        return {"cached_images": int(self.cache_index.size),
                "megabytes": round(self.cache.numel() / 1e6, 1)}

    def _gather(self, bank_rows: np.ndarray) -> torch.Tensor:
        if self.cache is not None:
            at = np.searchsorted(self.cache_index, bank_rows)
            if at.max(initial=0) < self.cache_index.size and np.array_equal(self.cache_index[at], bank_rows):
                return self.cache[torch.as_tensor(at, device=self.dev)]
        uniq, inverse = np.unique(bank_rows, return_inverse=True)
        block = torch.as_tensor(np.asarray(self.images[uniq]), device=self.dev)
        return block[torch.as_tensor(inverse, device=self.dev)]

    def pixels(self, rows: np.ndarray, columns: np.ndarray) -> list[torch.Tensor]:
        return [to_input(self._gather(self.src[rows, c])) for c in columns]


def fit_critics(rep: Representation, data: Batches, rows: np.ndarray, kept: np.ndarray, held: np.ndarray,
                steps: int, rng: np.random.Generator, state: dict | None, batch: int = TRAIN["batch"],
                spec: dict | None = None):
    """Fit q0 and then q1 with the representation held fixed.

    The representation is frozen here, so its output is computed once and reused for every critic step.  q1 is
    warm-started from the q0 just fitted, which is what makes it nest q0.
    """
    rep.eval()
    z_all = representation_values(rep, data, rows, kept, grad=False)
    a_all = data.a[torch.as_tensor(rows, device=data.dev)]
    z_dim = z_all.shape[1]
    base = BaseCritic(z_dim).to(data.dev)
    if state is not None:
        base.load_state_dict(state["base"])
    opt0 = torch.optim.Adam(base.parameters(), lr=TRAIN["lr_critic"])
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, len(rows), batch), device=data.dev)
        opt0.zero_grad(set_to_none=True)
        brier(base(z_all[idx]), a_all[idx]).backward()
        opt0.step()
    aug = AugmentedCritic(base, len(held), z_dim, spec).to(data.dev)
    if state is not None and state.get("aug") is not None:
        aug.load_state_dict(state["aug"])
        aug.net.load_state_dict(base.net.state_dict())
    opt1 = torch.optim.Adam(aug.parameters(), lr=TRAIN["lr_critic"])
    for _ in range(steps):
        pick = rng.integers(0, len(rows), batch)
        idx = torch.as_tensor(pick, device=data.dev)
        held_px = data.pixels(rows[pick], held)
        opt1.zero_grad(set_to_none=True)
        brier(aug(z_all[idx], held_px), a_all[idx]).backward()
        opt1.step()
    base.eval()
    aug.eval()
    # The sealed `nested_critics` keeps the lifted base when the augmented fit cannot improve on it.  Without
    # that comparison a warm-started q1 can come back WORSE than q0 on the very rows it was fitted on; the gap
    # is then negative for no reason to do with balance, and the representation can lower its objective by
    # keeping q1 broken instead of by balancing.
    with torch.no_grad():
        s0 = s1 = 0.0
        for s_ in range(0, len(rows), TRAIN["grad_batch"]):
            r = rows[s_:s_ + TRAIN["grad_batch"]]
            sl = slice(s_, s_ + len(r))
            s0 += float(((a_all[sl] - base(z_all[sl])) ** 2).sum())
            s1 += float(((a_all[sl] - aug(z_all[sl], data.pixels(r, held))) ** 2).sum())
    risk_base, risk_aug = s0 / len(rows), s1 / len(rows)
    lifted = bool(risk_base < risk_aug)
    if lifted:                                   # q1 := q0 exactly; the towers it has learned are kept for next time
        aug.net.load_state_dict(base.net.state_dict())
        nn.init.zeros_(aug.extra[-1].weight)
        nn.init.zeros_(aug.extra[-1].bias)
    info = {"risk_base_fit": risk_base, "risk_augmented_fit": risk_aug, "kept_the_lifted_base": lifted}
    return base, aug, {"base": {k: v.detach().clone() for k, v in base.state_dict().items()},
                       "aug": {k: v.detach().clone() for k, v in aug.state_dict().items()}, "info": info}


CRITIC_V = {"initial_steps": 300, "steps_per_iter": 100, "eval_every": 10, "val_fraction": 0.2}


def fit_critics_validated(rep: Representation, data: Batches, rows: np.ndarray, kept: np.ndarray, held: np.ndarray,
                          steps: int, rng: np.random.Generator, state: dict | None, spec: dict | None = None,
                          batch: int = TRAIN["batch"], val_fraction: float = CRITIC_V["val_fraction"],
                          eval_every: int = CRITIC_V["eval_every"]):
    """The training critics, made to generalise, and the nesting decided where it can be trusted.

    The diagnosed failure: a critic fitted by plain minibatch steps memorises its rows; on other rows the
    augmented critic then predicts WORSE than the base, the gap goes negative, the clip turns it into zero, and
    the representation stops moving at a zero that means "the critic failed", not "balance was reached".  The
    lifted-base rule could not catch this because it compared risks on the fit rows, where the memorising critic
    always wins.

    Two changes, nothing else.  A fifth of the rows given here is held back; each critic keeps the state that
    did best on that fifth (early stopping, exactly as the fresh critics of the independent check do).  And the
    augmented critic is replaced by the lifted base when it is not better than the base ON THAT FIFTH.  So a
    positive gap now means the held-out images carry information the critic could use on rows it never fitted,
    and a zero means it could not: both are honest, and only the first moves the representation.
    """
    rep.eval()
    rows = np.asarray(rows)
    z_all = representation_values(rep, data, rows, kept, grad=False)
    a_all = data.a[torch.as_tensor(rows, device=data.dev)]
    order = rng.permutation(len(rows))
    n_val = max(1, int(round(val_fraction * len(rows))))
    val_idx, fit_idx = np.sort(order[:n_val]), np.sort(order[n_val:])
    val_t = torch.as_tensor(val_idx, device=data.dev)
    z_dim = z_all.shape[1]

    def snapshot(m):
        return {k: v.detach().clone() for k, v in m.state_dict().items()}

    base = BaseCritic(z_dim).to(data.dev)
    if state is not None:
        base.load_state_dict(state["base"])
    opt0 = torch.optim.Adam(base.parameters(), lr=TRAIN["lr_critic"])
    best0 = {"loss": math.inf, "state": None, "step": 0}
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            base.eval()
            with torch.no_grad():
                loss = float(brier(base(z_all[val_t]), a_all[val_t]))
            if loss < best0["loss"] - 1e-9:
                best0 = {"loss": loss, "state": snapshot(base), "step": step}
            base.train()
        if step == steps:
            break
        idx = torch.as_tensor(fit_idx[rng.integers(0, len(fit_idx), batch)], device=data.dev)
        opt0.zero_grad(set_to_none=True)
        brier(base(z_all[idx]), a_all[idx]).backward()
        opt0.step()
    base.load_state_dict(best0["state"])
    base.eval()

    aug = AugmentedCritic(base, len(held), z_dim, spec).to(data.dev)
    if state is not None and state.get("aug") is not None:
        aug.load_state_dict(state["aug"])
        aug.net.load_state_dict(base.net.state_dict())        # the z branch is the current base; the towers warm-start
    opt1 = torch.optim.Adam(aug.parameters(), lr=TRAIN["lr_critic"])
    val_px = data.pixels(rows[val_idx], held)
    best1 = {"loss": math.inf, "state": None, "step": 0}
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            aug.eval()
            with torch.no_grad():
                loss = float(brier(aug(z_all[val_t], val_px), a_all[val_t]))
            if loss < best1["loss"] - 1e-9:
                best1 = {"loss": loss, "state": snapshot(aug), "step": step}
            aug.train()
        if step == steps:
            break
        pick = fit_idx[rng.integers(0, len(fit_idx), batch)]
        idx = torch.as_tensor(pick, device=data.dev)
        opt1.zero_grad(set_to_none=True)
        brier(aug(z_all[idx], data.pixels(rows[pick], held)), a_all[idx]).backward()
        opt1.step()
    aug.load_state_dict(best1["state"])
    aug.eval()
    lifted = bool(best0["loss"] <= best1["loss"])              # decided on the validation fifth, never on fit rows
    if lifted:
        aug.net.load_state_dict(base.net.state_dict())
        nn.init.zeros_(aug.extra[-1].weight)
        nn.init.zeros_(aug.extra[-1].bias)
    fit_t = torch.as_tensor(fit_idx, device=data.dev)
    with torch.no_grad():
        r0_fit = float(brier(base(z_all[fit_t]), a_all[fit_t]))
        r1_fit = float(brier(aug(z_all[fit_t], data.pixels(rows[fit_idx], held)), a_all[fit_t]))
        r1_val = float(brier(aug(z_all[val_t], val_px), a_all[val_t]))
    info = {"risk_base_fit": r0_fit, "risk_augmented_fit": r1_fit,
            "risk_base_val": best0["loss"], "risk_augmented_val": r1_val,
            "gap_val": best0["loss"] - r1_val, "kept_the_lifted_base": lifted,
            "base_step": best0["step"], "augmented_step": best1["step"], "n_fit": int(len(fit_idx)), "n_val": int(n_val)}
    return base, aug, {"base": snapshot(base), "aug": snapshot(aug), "info": info}


def representation_values(rep: Representation, data: Batches, rows: np.ndarray, kept: np.ndarray,
                          grad: bool, batch: int = TRAIN["grad_batch"]) -> torch.Tensor:
    chunks = []
    with torch.set_grad_enabled(grad):
        for s in range(0, len(rows), batch):
            r = rows[s:s + batch]
            idx = torch.as_tensor(r, device=data.dev)
            chunks.append(rep(data.pixels(r, kept), data.x[idx]))
    return torch.cat(chunks)


@torch.no_grad()
def signed_gap(rep: Representation, base: BaseCritic, aug: AugmentedCritic, data: Batches, rows: np.ndarray,
               kept: np.ndarray, held: np.ndarray, batch: int = TRAIN["grad_batch"]) -> dict:
    """The fold's signed Brier gap R0 - R1 and the two risks, computed on whole rows, not per batch."""
    rep.eval()
    s0 = s1 = 0.0
    for s in range(0, len(rows), batch):
        r = rows[s:s + batch]
        idx = torch.as_tensor(r, device=data.dev)
        z = rep(data.pixels(r, kept), data.x[idx])
        a = data.a[idx]
        s0 += float(((a - base(z)) ** 2).sum())
        s1 += float(((a - aug(z, data.pixels(r, held))) ** 2).sum())
    r0, r1 = s0 / len(rows), s1 / len(rows)
    return {"risk_base": r0, "risk_augmented": r1, "signed_gap": r0 - r1, "clipped_gap": max(r0 - r1, 0.0)}


def cnn_step(rep: Representation, groups: list, data: Batches, kept: np.ndarray, held: np.ndarray, opt,
             batch: int = TRAIN["grad_batch"]) -> dict:
    """One update of the representation from the signed gap pooled over `groups`.

    A group is (rows, base critic, augmented critic): the rows the gap is measured on and the critic pair that
    measures it.  The standing scheme has one group, the whole representation fold scored by the critics fitted
    on that same fold.  The cross-fitted scheme has two: each half scored by the critics fitted on the OTHER
    half, so what a critic memorised about its own rows never reaches the gradient.

    The gradient is accumulated over every row of every group, and the step is taken only if the pooled gap is
    positive: the clipped objective is flat below zero, so the network is never pushed to make it more negative.
    """
    rep.train()
    critics = [p for _, base, aug in groups for p in list(base.parameters()) + list(aug.parameters())]
    for p in critics:
        p.requires_grad_(False)
    opt.zero_grad(set_to_none=True)
    n_all = sum(len(rows) for rows, _, _ in groups)
    total0 = total1 = 0.0
    for rows, base, aug in groups:
        for s_ in range(0, len(rows), batch):
            r = rows[s_:s_ + batch]
            idx = torch.as_tensor(r, device=data.dev)
            z = rep(data.pixels(r, kept), data.x[idx])
            a = data.a[idx]
            part0 = ((a - base(z)) ** 2).sum() / n_all
            part1 = ((a - aug(z, data.pixels(r, held))) ** 2).sum() / n_all
            (part0 - part1).backward()
            total0 += float(part0.detach())
            total1 += float(part1.detach())
    gap = total0 - total1
    grad_norm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in rep.parameters() if p.grad is not None)))
    if gap > 0:
        opt.step()
    else:
        opt.zero_grad(set_to_none=True)
    for p in critics:
        p.requires_grad_(True)
    return {"risk_base": total0, "risk_augmented": total1, "signed_gap": gap, "stepped": bool(gap > 0),
            "grad_norm": grad_norm}


def train_split(data: Batches, rows: dict, split, seed: int, x_dim: int, spec: dict | None = None,
                max_seconds: float | None = None, log=None, crossfit: bool = False) -> dict:
    """The alternating fit of PRD section 4 for one candidate split.

    `crossfit=False` is the PRD's scheme: critics fitted on the representation fold, gap and gradient on that
    same fold.  `crossfit=True` halves the fold: the critics fitted on one half score the other half, in both
    directions.  That is a different training surrogate for the same population balance, NOT the manuscript's
    empirical objective, which fits the critics and measures the gap on the same sample (manuscript section 4);
    nothing the manuscript proves about that objective transfers to it by itself.

    The representation starts from a random initialisation fixed by `seed`.  Every checkpoint's weights are
    returned, with the training critics' risks on the rows they were fitted on and on the `select` fold, so a
    caller can choose a checkpoint with critics of its own instead of trusting these.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    held, kept = db.image_columns(split)
    dev = data.dev
    rep = Representation(kept, x_dim, spec=spec).to(dev)
    opt = torch.optim.Adam(rep.parameters(), lr=TRAIN["lr_cnn"])
    t0 = time.time()
    fold = np.asarray(rows["represent"])
    if crossfit:
        order = np.random.default_rng(seed + 17).permutation(len(fold))
        halves = [np.sort(fold[order[:len(fold) // 2]]), np.sort(fold[order[len(fold) // 2:]])]
        fit_sets, score_sets = halves, [halves[1], halves[0]]          # fitted on one half, scored on the other
    else:
        fit_sets, score_sets = [fold], [fold]
    states = [None] * len(fit_sets)

    def refit(steps: int) -> list:
        pairs = []
        for i, fit_rows in enumerate(fit_sets):
            base, aug, states[i] = fit_critics(rep, data, fit_rows, kept, held, steps, rng, states[i], spec=spec)
            pairs.append((base, aug))
        return pairs

    pairs = refit(TRAIN["critic_initial"])
    history, checkpoints = [], {}

    def take(iteration: int) -> None:
        if iteration in TRAIN["checkpoints"]:
            on_select = [signed_gap(rep, base, aug, data, rows["select"], kept, held) for base, aug in pairs]
            pooled = {k: float(np.mean([g_[k] for g_ in on_select])) for k in ("risk_base", "risk_augmented", "signed_gap")}
            pooled["clipped_gap"] = max(pooled["signed_gap"], 0.0)
            checkpoints[iteration] = {"state": {k: v.detach().cpu().clone() for k, v in rep.state_dict().items()},
                                      "select": pooled, "fit": [st["info"] for st in states]}

    take(0)
    for it in range(1, TRAIN["outer_iters"] + 1):
        if max_seconds is not None and time.time() - t0 > max_seconds:
            raise TimeoutError(f"split {db.split_name(split)} passed its {max_seconds:.0f} s cap at iteration {it}")
        groups = [(score, base, aug) for score, (base, aug) in zip(score_sets, pairs)]
        step = cnn_step(rep, groups, data, kept, held, opt)
        pairs = refit(TRAIN["critic_per_iter"])
        history.append(dict(step, iteration=it, seconds=round(time.time() - t0, 1),
                            critics=[st["info"] for st in states]))
        take(it)
        if log:
            log(f"    iter {it}: gap {step['signed_gap']:.3e} stepped {step['stepped']} "
                f"grad {step['grad_norm']:.2e} ({time.time() - t0:.0f} s)")
    order = sorted(checkpoints)
    best = min(order, key=lambda k: (checkpoints[k]["select"]["clipped_gap"], k))   # ties go to the earliest
    return {"split": db.split_name(split), "kept_images": kept.tolist(), "held_images": held.tolist(),
            "scheme": "cross-fitted halves" if crossfit else "whole fold",
            "initial_state": checkpoints[0]["state"], "chosen_state": checkpoints[best]["state"],
            "chosen_checkpoint": best, "checkpoint_states": {k: checkpoints[k]["state"] for k in order},
            "checkpoint_select_gap": {k: checkpoints[k]["select"] for k in order},
            "checkpoint_fit_risks": {k: checkpoints[k]["fit"] for k in order},
            "history": history, "seconds": round(time.time() - t0, 1)}


def load_representation(state: dict, kept: np.ndarray, x_dim: int, dev, spec: dict | None = None) -> Representation:
    rep = Representation(kept, x_dim, spec=spec)
    rep.load_state_dict(state)
    return rep.to(dev).eval()
