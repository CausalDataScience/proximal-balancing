"""Direct image balancing v3: fixed critic folds, an exact gradient through the whole fold, and an optional
standardisation of the representation that the critics are carried across.

Three changes from v2, and nothing else.

1.  FIXED VALIDATION ROWS.  v2 redrew each critic's early-stopping rows at every outer iteration while carrying
    the critic's weights forward, so a row that trained the critic at iteration t could be its "held-out" row at
    t + 1.  Here every fold's fit rows and validation rows are drawn once, before training, and keep their role
    for the whole run.  A critic never sees its own validation rows in any iteration.

2.  THE WHOLE-FOLD GRADIENT IS EXACT.  The representation of every row is computed once per update, the pooled
    clipped objective is built from it as a function of that matrix alone, differentiated, and the resulting
    dL/dZ is pushed back through the CNN batch by batch.  The augmented critics' image towers are frozen during
    an update, so their features of the held-out images are precomputed and the loss really is a function of Z.
    This makes the statistics of change 3 differentiable rather than treated as constants.

3.  OPTIONAL STANDARDISATION (run B only).  The two numbers the CNN outputs are centred and scaled by their mean
    and standard deviation over the representation fold, with those statistics inside the gradient.  Because a
    warm-started critic would otherwise receive the same information in different units after the statistics
    move, the critics' input layers are transformed so that their function of the RAW representation is
    unchanged; `retune_critic` does that and a test checks the predictions are identical.

Run A uses changes 1 and 2 with the identity normaliser.  Run B adds change 3.  Everything else, including the
data, the split, the initial weights, the learning rate and the number of updates, is the same in both.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

import direct_balance_data as db
import direct_balance_net as dn

CRITIC = {"initial_steps": 300, "steps_per_iter": 100, "eval_every": 10, "val_fraction": 0.2, "batch": 256}
SD_FLOOR = 1e-3
ITERS = 100
CHECKPOINTS = (0, 10, 20, 40, 70, 100)
DIAG_PAIRS = ((0, 1), (9, 10), (29, 30), (59, 60), (99, 100))     # (before, after) of one update, fixed in advance
DIAG_REFIT_STEPS = 300


# ----------------------------------------------------------------------------- the representation's scale

class Normaliser:
    """Mean and standard deviation of the h columns of Z = (X, h).  `identity` leaves Z untouched (run A)."""

    def __init__(self, mu: torch.Tensor | None, sd: torch.Tensor | None, n_x: int = 2) -> None:
        self.mu, self.sd, self.n_x = mu, sd, n_x

    @property
    def identity(self) -> bool:
        return self.mu is None

    @staticmethod
    def fit(z: torch.Tensor, n_x: int = 2, enabled: bool = True) -> "Normaliser":
        if not enabled:
            return Normaliser(None, None, n_x)
        h = z[:, n_x:]
        return Normaliser(h.mean(0), torch.clamp(h.std(0, unbiased=False), min=SD_FLOOR), n_x)

    def apply(self, z: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return z
        return torch.cat([z[:, :self.n_x], (z[:, self.n_x:] - self.mu) / self.sd], dim=1)

    def frozen(self) -> "Normaliser":
        if self.identity:
            return Normaliser(None, None, self.n_x)
        return Normaliser(self.mu.detach().clone(), self.sd.detach().clone(), self.n_x)

    def record(self) -> dict:
        if self.identity:
            return {"identity": True}
        return {"identity": False, "mean": self.mu.detach().cpu().tolist(), "sd": self.sd.detach().cpu().tolist()}


class NormalisedRepresentation(nn.Module):
    """The CNN with a fixed normalisation attached, so every downstream evaluator sees the same Z the critics saw."""

    def __init__(self, rep: dn.Representation, norm: Normaliser) -> None:
        super().__init__()
        self.rep, self.norm = rep, norm
        self.kept = rep.kept

    def forward(self, pixels, x):
        return self.norm.apply(self.rep(pixels, x))


def load_normalised(state: dict, kept: np.ndarray, x_dim: int, dev, spec: dict, stats: dict | None):
    rep = dn.load_representation(state, kept, x_dim, dev, spec)
    if stats is None or stats.get("identity", True):
        norm = Normaliser(None, None)
    else:
        norm = Normaliser(torch.as_tensor(stats["mean"], dtype=torch.float32, device=dev),
                          torch.as_tensor(stats["sd"], dtype=torch.float32, device=dev))
    return NormalisedRepresentation(rep, norm).to(dev).eval()


def _first_layers(critic: nn.Module) -> list[tuple[nn.Linear, slice]]:
    """Every linear layer that reads Z, with the slice of its input that holds the h columns."""
    out = [(critic.net[0], slice(2, None))]
    if hasattr(critic, "extra"):
        z_dim = critic.net[0].in_features
        first = critic.extra[0]
        start = first.in_features - z_dim
        out.append((first, slice(start + 2, None)))
    return out


@torch.no_grad()
def retune_critic(critic: nn.Module, old: Normaliser, new: Normaliser) -> None:
    """Rewrite the layers that read Z so the critic is the same function of the RAW h after the statistics move.

    With z_old = (h - mu_old) / sd_old and z_new = (h - mu_new) / sd_new, W z_old + b equals
    (W * sd_new / sd_old) z_new + (b + W ((mu_new - mu_old) / sd_old)), which is what this writes back.
    """
    if old.identity or new.identity:
        return
    ratio = (new.sd / old.sd).to(dtype=torch.float32)
    shift = ((new.mu - old.mu) / old.sd).to(dtype=torch.float32)
    for layer, sl in _first_layers(critic):
        w = layer.weight[:, sl]
        layer.bias.add_(w @ shift)
        layer.weight[:, sl] = w * ratio


# ----------------------------------------------------------------------------- critics on fixed rows

def fit_critics_v3(rep: dn.Representation, norm: Normaliser, data: dn.Batches, fit_rows: np.ndarray,
                   val_rows: np.ndarray, kept: np.ndarray, held: np.ndarray, steps: int, rng: np.random.Generator,
                   state: dict | None, spec: dict | None, old_norm: Normaliser | None = None):
    """Both critics on `fit_rows`, early stopped on `val_rows`, and the lifted base decided on `val_rows`.

    The two row sets are given by the caller and never change during a run.  A warm-started critic is first
    retuned for the new normalisation so that carrying it forward does not silently change what it computes.
    """
    rep.eval()
    fit_rows, val_rows = np.asarray(fit_rows), np.asarray(val_rows)
    with torch.no_grad():
        z_fit = norm.apply(dn.representation_values(rep, data, fit_rows, kept, grad=False))
        z_val = norm.apply(dn.representation_values(rep, data, val_rows, kept, grad=False))
    a_fit = data.a[torch.as_tensor(fit_rows, device=data.dev)]
    a_val = data.a[torch.as_tensor(val_rows, device=data.dev)]
    z_dim = z_fit.shape[1]
    snap = lambda m: {k: v.detach().clone() for k, v in m.state_dict().items()}

    base = dn.BaseCritic(z_dim).to(data.dev)
    if state is not None:
        base.load_state_dict(state["base"])
        if old_norm is not None:
            retune_critic(base, old_norm, norm)
    opt = torch.optim.Adam(base.parameters(), lr=dn.TRAIN["lr_critic"])
    best0 = {"loss": math.inf, "state": None, "step": 0}
    for step in range(steps + 1):
        if step % CRITIC["eval_every"] == 0 or step == steps:
            base.eval()
            with torch.no_grad():
                v = float(dn.brier(base(z_val), a_val))
            if v < best0["loss"] - 1e-9:
                best0 = {"loss": v, "state": snap(base), "step": step}
            base.train()
        if step == steps:
            break
        idx = torch.as_tensor(rng.integers(0, len(fit_rows), CRITIC["batch"]), device=data.dev)
        opt.zero_grad(set_to_none=True)
        dn.brier(base(z_fit[idx]), a_fit[idx]).backward()
        opt.step()
    base.load_state_dict(best0["state"])
    base.eval()

    aug = dn.AugmentedCritic(base, len(held), z_dim, spec).to(data.dev)
    if state is not None and state.get("aug") is not None:
        aug.load_state_dict(state["aug"])
        if old_norm is not None:
            retune_critic(aug, old_norm, norm)
        aug.net.load_state_dict(base.net.state_dict())
    opt = torch.optim.Adam(aug.parameters(), lr=dn.TRAIN["lr_critic"])
    px_val = data.pixels(val_rows, held)
    best1 = {"loss": math.inf, "state": None, "step": 0}
    for step in range(steps + 1):
        if step % CRITIC["eval_every"] == 0 or step == steps:
            aug.eval()
            with torch.no_grad():
                v = float(dn.brier(aug(z_val, px_val), a_val))
            if v < best1["loss"] - 1e-9:
                best1 = {"loss": v, "state": snap(aug), "step": step}
            aug.train()
        if step == steps:
            break
        pick = rng.integers(0, len(fit_rows), CRITIC["batch"])
        idx = torch.as_tensor(pick, device=data.dev)
        opt.zero_grad(set_to_none=True)
        dn.brier(aug(z_fit[idx], data.pixels(fit_rows[pick], held)), a_fit[idx]).backward()
        opt.step()
    aug.load_state_dict(best1["state"])
    aug.eval()
    lifted = bool(best0["loss"] <= best1["loss"])
    if lifted:
        aug.net.load_state_dict(base.net.state_dict())
        nn.init.zeros_(aug.extra[-1].weight)
        nn.init.zeros_(aug.extra[-1].bias)
    with torch.no_grad():
        r0_fit = float(dn.brier(base(z_fit), a_fit))
        r1_fit = float(dn.brier(aug(z_fit, data.pixels(fit_rows, held)), a_fit))
        r1_val = float(dn.brier(aug(z_val, px_val), a_val))
    info = {"risk_base_fit": r0_fit, "risk_augmented_fit": r1_fit, "risk_base_val": best0["loss"],
            "risk_augmented_val": r1_val, "gap_val": best0["loss"] - r1_val, "kept_the_lifted_base": lifted,
            "base_step": best0["step"], "augmented_step": best1["step"]}
    return base, aug, {"base": snap(base), "aug": snap(aug), "info": info}


# ----------------------------------------------------------------------------- one update, exact

def _held_features(aug: dn.AugmentedCritic, data: dn.Batches, rows: np.ndarray, held: np.ndarray,
                   batch: int = 512) -> torch.Tensor:
    """The augmented critic's tower features of the held-out images.  Frozen during an update, so the loss below
    really is a function of the representation alone."""
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            px = data.pixels(rows[s:s + batch], held)
            out.append(torch.cat([t(p) for t, p in zip(aug.towers, px)], dim=1))
    return torch.cat(out)


def pooled_objective(z_raw: torch.Tensor, norm: Normaliser, groups: list, data: dn.Batches,
                     index_of: dict) -> tuple[torch.Tensor, dict]:
    """max(pooled signed gap, 0), where the pooling is over the scoring folds and the clip is applied once."""
    z = norm.apply(z_raw)
    n_all = sum(len(g["rows"]) for g in groups)
    total0 = total1 = 0.0
    parts = []
    for g in groups:
        at = index_of[id(g)]
        zg, a = z[at], data.a[torch.as_tensor(g["rows"], device=data.dev)]
        q0 = g["base"](zg)
        logit = g["aug"].net(zg).squeeze(1) + g["aug"].extra(torch.cat([g["feats"], zg], dim=1)).squeeze(1)
        q1 = dn.link(logit)
        s0 = ((a - q0) ** 2).sum()
        s1 = ((a - q1) ** 2).sum()
        total0 = total0 + s0
        total1 = total1 + s1
        parts.append({"n": len(g["rows"]), "risk_base": float(s0.detach()) / len(g["rows"]),
                      "risk_augmented": float(s1.detach()) / len(g["rows"])})
    signed = (total0 - total1) / n_all
    return torch.clamp(signed, min=0.0), {"signed_gap": float(signed.detach()), "folds": parts}


def update(rep: dn.Representation, data: dn.Batches, plan: dict, pairs: list, kept: np.ndarray, held: np.ndarray,
           opt, standardise: bool, batch: int = 512) -> dict:
    """One CNN update: Z for the fold, the objective as a function of Z, dL/dZ, then the CNN in batches."""
    rows_all = plan["represent"]
    z_leaf = dn.representation_values(rep, data, rows_all, kept, grad=False).clone().requires_grad_(True)
    norm = Normaliser.fit(z_leaf, enabled=standardise)
    pos = {int(r): i for i, r in enumerate(rows_all)}
    groups, index_of = [], {}
    for (base, aug), score_rows in zip(pairs, plan["score_sets"]):
        g = {"rows": score_rows, "base": base, "aug": aug,
             "feats": _held_features(aug, data, score_rows, held)}
        groups.append(g)
        index_of[id(g)] = torch.as_tensor([pos[int(r)] for r in score_rows], device=data.dev)
    loss, info = pooled_objective(z_leaf, norm, groups, data, index_of)
    stepped, grad_norm, first_conv = False, 0.0, 0.0
    if info["signed_gap"] > 0:
        loss.backward()
        upstream = z_leaf.grad
        rep.train()
        opt.zero_grad(set_to_none=True)
        for s in range(0, len(rows_all), batch):
            r = rows_all[s:s + batch]
            idx = torch.as_tensor(r, device=data.dev)
            rep(data.pixels(r, kept), data.x[idx]).backward(upstream[s:s + len(r)])
        grads = [(n, p.grad) for n, p in rep.named_parameters() if p.grad is not None]
        grad_norm = float(torch.sqrt(sum((g_ ** 2).sum() for _, g_ in grads)))
        first_conv = float(sum(g_.norm() for n, g_ in grads if n.endswith("towers.0.body.0.weight")))
        opt.step()
        rep.eval()
        stepped = True
    with torch.no_grad():
        h = z_leaf[:, 2:]
        scale = {"h_sd": h.std(0).tolist(), "h_mean": h.mean(0).tolist(), "h_absmax": float(h.abs().max())}
    return dict(info, stepped=stepped, grad_norm=grad_norm, first_conv_grad_norm=first_conv,
                dLdZ_norm=float(z_leaf.grad.norm()) if z_leaf.grad is not None else 0.0,
                normaliser=norm.record(), scale=scale), norm


def objective_with(rep: dn.Representation, norm: Normaliser, data: dn.Batches, plan: dict, pairs: list,
                   kept: np.ndarray, held: np.ndarray) -> float:
    """The same pooled objective, read off without touching anything (for the diagnostic)."""
    with torch.no_grad():
        z = dn.representation_values(rep, data, plan["represent"], kept, grad=False)
    pos = {int(r): i for i, r in enumerate(plan["represent"])}
    groups, index_of = [], {}
    for (base, aug), score_rows in zip(pairs, plan["score_sets"]):
        g = {"rows": score_rows, "base": base, "aug": aug, "feats": _held_features(aug, data, score_rows, held)}
        groups.append(g)
        index_of[id(g)] = torch.as_tensor([pos[int(r)] for r in score_rows], device=data.dev)
    with torch.no_grad():
        _, info = pooled_objective(z, norm, groups, data, index_of)
    return info["signed_gap"]


# ----------------------------------------------------------------------------- the run

def fold_plan(rows: dict, seed: int, folds: int = 3, val_fraction: float = CRITIC["val_fraction"]) -> dict:
    """Fit rows, validation rows and scoring rows for each fold, drawn once and kept for the whole run.

    Fold i is scored on part i and its critics are fitted on the other parts; a fifth of those fit rows is set
    aside for early stopping and never trains that fold's critics.
    """
    fold = np.asarray(rows["represent"])
    order = np.random.default_rng(seed).permutation(len(fold))
    parts = [np.sort(fold[order[i::folds]]) for i in range(folds)]
    fit_sets, val_sets, score_sets = [], [], []
    for i in range(folds):
        pool = np.sort(np.concatenate([p for j, p in enumerate(parts) if j != i]))
        inner = np.random.default_rng(seed + 100 + i).permutation(len(pool))
        n_val = max(1, int(round(val_fraction * len(pool))))
        val_sets.append(np.sort(pool[inner[:n_val]]))
        fit_sets.append(np.sort(pool[inner[n_val:]]))
        score_sets.append(parts[i])
    return {"represent": fold, "select": np.asarray(rows["select"]), "folds": folds,
            "fit_sets": fit_sets, "val_sets": val_sets, "score_sets": score_sets}


def plan_report(plan: dict) -> dict:
    import hashlib
    sha = lambda a: hashlib.sha256(np.ascontiguousarray(np.sort(a)).tobytes()).hexdigest()[:16]
    out = {"sizes": {"fit": [len(v) for v in plan["fit_sets"]], "val": [len(v) for v in plan["val_sets"]],
                     "score": [len(v) for v in plan["score_sets"]]},
           "sha": {"fit": [sha(v) for v in plan["fit_sets"]], "val": [sha(v) for v in plan["val_sets"]],
                   "score": [sha(v) for v in plan["score_sets"]]}, "leaks": []}
    for i in range(plan["folds"]):
        if np.intersect1d(plan["fit_sets"][i], plan["val_sets"][i]).size:
            out["leaks"].append(f"fold {i}: fit and validation rows overlap")
        if np.intersect1d(plan["fit_sets"][i], plan["score_sets"][i]).size:
            out["leaks"].append(f"fold {i}: fit and scoring rows overlap")
        if np.intersect1d(plan["val_sets"][i], plan["score_sets"][i]).size:
            out["leaks"].append(f"fold {i}: validation and scoring rows overlap")
        union = np.union1d(np.union1d(plan["fit_sets"][i], plan["val_sets"][i]), plan["score_sets"][i])
        if not np.array_equal(union, np.sort(plan["represent"])):
            out["leaks"].append(f"fold {i}: the three sets do not cover the representation fold")
    if np.intersect1d(np.concatenate(plan["score_sets"]), plan["select"]).size:
        out["leaks"].append("scoring rows overlap the selection fold")
    return out


def train(data: dn.Batches, rows: dict, split, seed: int, x_dim: int, spec: dict, standardise: bool,
          iters: int = ITERS, checkpoints=CHECKPOINTS, max_seconds: float | None = None, log=None) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    held, kept = db.image_columns(split)
    plan = fold_plan(rows, seed)
    rep = dn.Representation(kept, x_dim, spec=spec).to(data.dev)
    init_state = {k: v.detach().cpu().clone() for k, v in rep.state_dict().items()}
    opt = torch.optim.Adam(rep.parameters(), lr=dn.TRAIN["lr_cnn"])
    t0 = time.time()
    states = [None] * plan["folds"]
    norm = Normaliser.fit(dn.representation_values(rep, data, plan["represent"], kept, grad=False),
                          enabled=standardise).frozen()

    def refit(steps: int, old: Normaliser | None) -> list:
        pairs = []
        for i in range(plan["folds"]):
            base, aug, states[i] = fit_critics_v3(rep, norm, data, plan["fit_sets"][i], plan["val_sets"][i],
                                                  kept, held, steps, rng, states[i], spec, old)
            pairs.append((base, aug))
        return pairs

    pairs = refit(CRITIC["initial_steps"], None)
    history, saved, diag_states = [], {}, {}
    wanted = {i for pair in DIAG_PAIRS for i in pair}

    def snapshot(it: int) -> None:
        if it in checkpoints:
            saved[it] = {"state": {k: v.detach().cpu().clone() for k, v in rep.state_dict().items()},
                         "normaliser": norm.record(),
                         "critics": [st["info"] for st in states]}
        if it in wanted:
            diag_states[it] = {k: v.detach().cpu().clone() for k, v in rep.state_dict().items()}

    snapshot(0)
    for it in range(1, iters + 1):
        if max_seconds is not None and time.time() - t0 > max_seconds:
            raise TimeoutError(f"training passed its {max_seconds:.0f} s cap at iteration {it}")
        step, new_norm = update(rep, data, plan, pairs, kept, held, opt, standardise)
        old_norm, norm = norm, new_norm.frozen()
        pairs = refit(CRITIC["steps_per_iter"], old_norm if standardise else None)
        history.append(dict(step, iteration=it, seconds=round(time.time() - t0, 1),
                            critics=[st["info"] for st in states]))
        snapshot(it)
        if log:
            gv = float(np.mean([st["info"]["gap_val"] for st in states]))
            log(f"    iter {it:3d}: gap {step['signed_gap']:+.5f} stepped {step['stepped']} grad "
                f"{step['grad_norm']:.2e} | h sd {np.round(step['scale']['h_sd'], 3).tolist()} "
                f"| critic val gap {gv:+.5f} ({time.time() - t0:.0f} s)")
    return {"arm": "B (standardised)" if standardise else "A (bug fix only)", "split": db.split_name(split),
            "plan": plan, "plan_report": plan_report(plan), "initial_state": init_state,
            "checkpoint_states": {k: v["state"] for k, v in saved.items()},
            "checkpoint_normalisers": {k: v["normaliser"] for k, v in saved.items()},
            "checkpoint_critics": {k: v["critics"] for k, v in saved.items()},
            "diag_states": diag_states, "history": history, "seconds": round(time.time() - t0, 1),
            "steps_taken": sum(h["stepped"] for h in history)}


def refit_diagnostic(out: dict, data: dn.Batches, rows: dict, split, x_dim: int, spec: dict, standardise: bool,
                     seed: int, log=None) -> list:
    """For each saved update: did the objective fall with the critics fixed, and does it still fall when the
    critics are refitted from scratch on each representation?  Same rows, same budget, same initialisation."""
    held, kept = db.image_columns(split)
    plan = out["plan"]
    diag = []
    for before, after in DIAG_PAIRS:
        if before not in out["diag_states"] or after not in out["diag_states"]:
            continue
        reps, norms, objs = {}, {}, {}
        for tag, state in (("before", out["diag_states"][before]), ("after", out["diag_states"][after])):
            rep = dn.Representation(kept, x_dim, spec=spec).to(data.dev)
            rep.load_state_dict(state)
            rep.eval()
            reps[tag] = rep
            norms[tag] = Normaliser.fit(dn.representation_values(rep, data, plan["represent"], kept, grad=False),
                                        enabled=standardise).frozen()
        fixed_pairs = []
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        for i in range(plan["folds"]):
            b, a_, _ = fit_critics_v3(reps["before"], norms["before"], data, plan["fit_sets"][i], plan["val_sets"][i],
                                      kept, held, DIAG_REFIT_STEPS, rng, None, spec)
            fixed_pairs.append((b, a_))
        objs["fixed_before"] = objective_with(reps["before"], norms["before"], data, plan, fixed_pairs, kept, held)
        # `retune_critic` makes a critic the same function of the RAW h under either normalisation, so the fixed
        # comparison is done under the before-statistics for both: only the representation differs.
        objs["fixed_after"] = objective_with(reps["after"], norms["before"], data, plan, fixed_pairs, kept, held)
        refit_pairs = []
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        for i in range(plan["folds"]):
            b, a_, _ = fit_critics_v3(reps["after"], norms["after"], data, plan["fit_sets"][i], plan["val_sets"][i],
                                      kept, held, DIAG_REFIT_STEPS, rng, None, spec)
            refit_pairs.append((b, a_))
        objs["refit_after"] = objective_with(reps["after"], norms["after"], data, plan, refit_pairs, kept, held)
        rec = {"update": [before, after],
               "with_the_critics_fixed": {"before": objs["fixed_before"], "after": objs["fixed_after"],
                                          "change": objs["fixed_after"] - objs["fixed_before"]},
               "with_the_critics_refitted": {"before": objs["fixed_before"], "after": objs["refit_after"],
                                             "change": objs["refit_after"] - objs["fixed_before"]}}
        rec["both_fell"] = bool(rec["with_the_critics_fixed"]["change"] < 0 and rec["with_the_critics_refitted"]["change"] < 0)
        rec["only_the_fixed_critics_fell"] = bool(rec["with_the_critics_fixed"]["change"] < 0 <= rec["with_the_critics_refitted"]["change"])
        diag.append(rec)
        if log:
            log(f"  update {before}->{after}: fixed critics {rec['with_the_critics_fixed']['change']:+.5f} | "
                f"refitted critics {rec['with_the_critics_refitted']['change']:+.5f} | both fell {rec['both_fell']}")
    return diag
