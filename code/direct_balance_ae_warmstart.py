"""Tiny16 autoencoder warm start: two arms that differ only in where the CNN starts
(PRD memo/2026-09-22-tiny16-autoencoder-warmstart-prd-v1.md).

    R   the image towers start at random
    AE  the image towers start from an encoder that learned, with no label of any kind, to rebuild the 16 x 16
        images; the decoder is thrown away before balancing begins

After that the two arms are the same experiment: every weight of the representation is updated by the clipped
Brier gap and nothing else.  No reconstruction loss, no outcome, no angle, no latent.

Three things this module fixes or pins down relative to v3.

  normalisation lifecycle  v3 handed the critics the statistics of the representation BEFORE the update, so the
                           critics of iteration t + 1 were fitted against coordinates one step stale.  Here the
                           statistics are recomputed from the post-step encoder, the critics are carried to the
                           new coordinates, and a check records that encoder, statistics and critics share one
                           vintage (PRD section 6).
  purpose-separated seeds  every random draw comes from SeedSequence([data seed, purpose, fold, iteration]), so
                           training the autoencoder cannot shift the random state of anything the two arms are
                           supposed to share (PRD section 4.2).
  actual-update diagnosis  the decrease diagnostics are indexed by ACTUAL optimizer updates, not by outer
                           iterations, and the critics they hold fixed are the ones the update really used
                           (PRD section 7).  An iteration that took no step is `not_reached`, never a failure.
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np
import torch
from torch import nn

import direct_balance_data as db
import direct_balance_eval as de
import direct_balance_net as dn
import direct_balance_tiny16 as t16
import direct_balance_v3 as v3

DATA_SEED = 96_900_800
PURPOSE = {"ae_id_split": 101, "ae_init": 102, "ae_batch": 103, "r_tower": 201, "head": 202,
           "critic_init": 301, "critic_batch": 302, "fold_split": 303, "selection": 401,
           "check16": 402, "check64": 403, "nuisance": 404, "refit_diag": 501}
AE = {"epochs": 10, "batch": 256, "lr": 1e-3, "latent": dn.FEATURE_DIM, "val_fraction": 0.2, "improve": 1e-8}
CRITIC = {"initial_steps": 300, "steps_per_iter": 100, "eval_every": 10, "improve": 1e-7,
          "val_fraction": 0.2, "batch": 256}
TRAIN = {"iters": 100, "checkpoints": (0, 10, 20, 40, 70, 100), "lr": 1e-3, "folds": 3, "grad_batch": 512}
ACTUAL_UPDATE_ORDINALS = (1, 3, 7)
SELECT = {"steps": 300, "batch": 256, "lr": 1e-3, "val_fraction": 0.2, "eval_every": 25, "improve": 1e-7,
          "tolerance": 1e-6}
SD_FLOOR = v3.SD_FLOOR


def seed_for(purpose: str, fold: int = 0, iteration: int = 0) -> int:
    return int(np.random.SeedSequence([DATA_SEED, PURPOSE[purpose], fold, iteration]).generate_state(1)[0])


def sha_state(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode()); h.update(np.ascontiguousarray(state[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


class Deadline:
    def __init__(self, seconds: float) -> None:
        self.end = time.monotonic() + seconds

    def left(self) -> float:
        return self.end - time.monotonic()

    def over(self) -> bool:
        return self.left() <= 0


# ----------------------------------------------------------------------------- the autoencoder

def decoder(latent: int = AE["latent"]) -> nn.Module:
    """The mirror of the Tiny16 tower.  No sigmoid: the pixels it must reproduce are standardised and go negative."""
    return nn.Sequential(nn.Linear(latent, 512), nn.ReLU(), nn.Unflatten(1, (32, 4, 4)),
                         nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
                         nn.ConvTranspose2d(16, 3, 4, 2, 1))


def image_id_split(src: np.ndarray, represent_rows: np.ndarray) -> dict:
    """The distinct original images the representation rows draw, split into fit and validation by id.

    The split is by image, not by row, so no picture can be in both halves however often it is drawn.
    """
    ids = np.unique(np.asarray(src)[np.asarray(represent_rows)].ravel())
    order = np.random.default_rng(seed_for("ae_id_split")).permutation(len(ids))
    cut = int(math.floor(0.8 * len(ids)))
    return {"fit": np.sort(ids[order[:cut]]), "val": np.sort(ids[order[cut:]]), "all": ids}


def train_autoencoder(tiny: t16.TinyBatches, ids: dict, deadline: Deadline, log=None) -> dict:
    """Reconstruct the standardised 16 x 16 pixels and nothing else.  Every distinct image is seen once an epoch."""
    dev = tiny.dev
    torch.manual_seed(seed_for("ae_init"))
    enc = dn.ImageTower(t16.SPEC16).to(dev)
    dec = decoder().to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=AE["lr"])
    rng = np.random.default_rng(seed_for("ae_batch"))
    at = lambda bank_ids: torch.as_tensor(np.searchsorted(tiny.cache_index, bank_ids), device=dev)
    fit_at, val_at = at(ids["fit"]), at(ids["val"])

    def val_mse() -> float:
        enc.eval(); dec.eval()
        tot = 0.0
        with torch.no_grad():
            for s in range(0, len(val_at), 1024):
                x = tiny.small[val_at[s:s + 1024]]
                tot += float(((dec(enc(x)) - x) ** 2).mean()) * len(x)
        return tot / len(val_at)

    best = {"epoch": 0, "val_mse": val_mse(), "state": {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}}
    history = [{"epoch": 0, "train_mse": None, "val_mse": best["val_mse"]}]
    completed, stopped = 0, None
    t0 = time.monotonic()
    for epoch in range(1, AE["epochs"] + 1):
        enc.train(); dec.train()
        order = rng.permutation(len(fit_at))
        tot, seen = 0.0, 0
        for s in range(0, len(order), AE["batch"]):
            if deadline.over():
                stopped = f"time cap inside epoch {epoch}"
                break
            idx = fit_at[torch.as_tensor(order[s:s + AE["batch"]], device=dev)]
            x = tiny.small[idx]
            loss = ((dec(enc(x)) - x) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx); seen += len(idx)
        if stopped:
            break
        completed = epoch
        v = val_mse()
        history.append({"epoch": epoch, "train_mse": tot / seen, "val_mse": v})
        if v < best["val_mse"] - AE["improve"]:
            best = {"epoch": epoch, "val_mse": v,
                    "state": {k: t_.detach().cpu().clone() for k, t_ in enc.state_dict().items()}}
        if log:
            log(f"    AE epoch {epoch}: train {tot / seen:.5f} val {v:.5f} (best epoch {best['epoch']})")
    return {"images": {"fit": int(len(ids["fit"])), "val": int(len(ids["val"])), "all": int(len(ids["all"]))},
            "history": history, "chosen_epoch": best["epoch"], "chosen_val_mse": best["val_mse"],
            "epochs_completed": completed, "stopped_early": stopped, "state": best["state"],
            "state_sha256": sha_state(best["state"]), "seconds": round(time.monotonic() - t0, 1),
            "usable": completed >= 1 or best["epoch"] == 0}


# ----------------------------------------------------------------------------- paired initial representations

def build_representation(kept: np.ndarray, x_dim: int, dev, arm: str, ae_state: dict | None) -> dn.Representation:
    """Both arms share one head; only the image towers differ, and only in where they start."""
    torch.manual_seed(seed_for("head"))
    rep = dn.Representation(kept, x_dim, spec=t16.SPEC16)
    head_state = {k: v.detach().clone() for k, v in rep.head.state_dict().items()}
    if arm == "R":
        torch.manual_seed(seed_for("r_tower"))
        for tower in rep.towers:
            for m in tower.body:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
    else:
        if ae_state is None:
            raise RuntimeError("the AE arm needs a trained encoder")
        for tower in rep.towers:
            tower.load_state_dict({k: v.clone() for k, v in ae_state.items()})
    rep.head.load_state_dict(head_state)
    return rep.to(dev)


# ----------------------------------------------------------------------------- critics on fixed rows

def fit_critics(rep: dn.Representation, norm: v3.Normaliser, data: dn.Batches, fit_rows: np.ndarray,
                val_rows: np.ndarray, kept: np.ndarray, held: np.ndarray, steps: int, fold: int, iteration: int,
                state: dict | None, old_norm: v3.Normaliser | None, deadline: Deadline | None = None):
    """Both critics on `fit_rows`, early stopped on `val_rows`, seeded by (fold, iteration) and not by arm."""
    rep.eval()
    fit_rows, val_rows = np.asarray(fit_rows), np.asarray(val_rows)
    dev = data.dev
    with torch.no_grad():
        z_fit = norm.apply(dn.representation_values(rep, data, fit_rows, kept, grad=False))
        z_val = norm.apply(dn.representation_values(rep, data, val_rows, kept, grad=False))
    a_fit = data.a[torch.as_tensor(fit_rows, device=dev)]
    a_val = data.a[torch.as_tensor(val_rows, device=dev)]
    z_dim = z_fit.shape[1]
    rng = np.random.default_rng(seed_for("critic_batch", fold, iteration))
    snap = lambda m: {k: v.detach().clone() for k, v in m.state_dict().items()}
    torch.manual_seed(seed_for("critic_init", fold, iteration))
    base = dn.BaseCritic(z_dim).to(dev)
    aug_fresh = dn.AugmentedCritic(base, len(held), z_dim, t16.SPEC16).to(dev)
    if state is not None:
        base.load_state_dict(state["base"])
        if old_norm is not None:
            v3.retune_critic(base, old_norm, norm)
    opt = torch.optim.Adam(base.parameters(), lr=CRITIC["lr"] if "lr" in CRITIC else dn.TRAIN["lr_critic"])
    best0 = {"loss": math.inf, "state": None, "step": 0}
    for step in range(steps + 1):
        if step % CRITIC["eval_every"] == 0 or step == steps:
            base.eval()
            with torch.no_grad():
                v = float(dn.brier(base(z_val), a_val))
            if v < best0["loss"] - CRITIC["improve"]:
                best0 = {"loss": v, "state": snap(base), "step": step}
            base.train()
        if step == steps or (deadline is not None and deadline.over()):
            break
        idx = torch.as_tensor(rng.integers(0, len(fit_rows), CRITIC["batch"]), device=dev)
        opt.zero_grad(set_to_none=True)
        dn.brier(base(z_fit[idx]), a_fit[idx]).backward()
        opt.step()
    base.load_state_dict(best0["state"]); base.eval()

    aug = dn.AugmentedCritic(base, len(held), z_dim, t16.SPEC16).to(dev)
    aug.load_state_dict({k: v.clone() for k, v in aug_fresh.state_dict().items()})
    aug.net.load_state_dict(base.net.state_dict())
    nn.init.zeros_(aug.extra[-1].weight); nn.init.zeros_(aug.extra[-1].bias)
    if state is not None and state.get("aug") is not None:
        aug.load_state_dict(state["aug"])
        if old_norm is not None:
            v3.retune_critic(aug, old_norm, norm)
        aug.net.load_state_dict(base.net.state_dict())
    opt = torch.optim.Adam(aug.parameters(), lr=dn.TRAIN["lr_critic"])
    px_val = data.pixels(val_rows, held)
    best1 = {"loss": math.inf, "state": None, "step": 0}
    for step in range(steps + 1):
        if step % CRITIC["eval_every"] == 0 or step == steps:
            aug.eval()
            with torch.no_grad():
                v = float(dn.brier(aug(z_val, px_val), a_val))
            if v < best1["loss"] - CRITIC["improve"]:
                best1 = {"loss": v, "state": snap(aug), "step": step}
            aug.train()
        if step == steps or (deadline is not None and deadline.over()):
            break
        pick = rng.integers(0, len(fit_rows), CRITIC["batch"])
        idx = torch.as_tensor(pick, device=dev)
        opt.zero_grad(set_to_none=True)
        dn.brier(aug(z_fit[idx], data.pixels(fit_rows[pick], held)), a_fit[idx]).backward()
        opt.step()
    aug.load_state_dict(best1["state"]); aug.eval()
    lifted = bool(best0["loss"] <= best1["loss"])
    if lifted:
        aug.net.load_state_dict(base.net.state_dict())
        nn.init.zeros_(aug.extra[-1].weight); nn.init.zeros_(aug.extra[-1].bias)
        with torch.no_grad():
            best1["loss"] = float(dn.brier(aug(z_val, px_val), a_val))
    info = {"risk_base_val": best0["loss"], "risk_augmented_val": best1["loss"],
            "gap_val": best0["loss"] - best1["loss"], "kept_the_lifted_base": lifted,
            "base_step": best0["step"], "augmented_step": best1["step"]}
    return base, aug, {"base": snap(base), "aug": snap(aug), "info": info}


# ----------------------------------------------------------------------------- the objective and one update

def build_groups(rep: dn.Representation, data: dn.Batches, plan: dict, pairs: list, kept: np.ndarray,
                 held: np.ndarray, rows_all: np.ndarray):
    pos = {int(r): i for i, r in enumerate(rows_all)}
    groups, index_of = [], {}
    for (base, aug), score_rows in zip(pairs, plan["score_sets"]):
        g = {"rows": score_rows, "base": base, "aug": aug, "feats": v3._held_features(aug, data, score_rows, held)}
        groups.append(g)
        index_of[id(g)] = torch.as_tensor([pos[int(r)] for r in score_rows], device=data.dev)
    return groups, index_of


def objective_at(rep: dn.Representation, data: dn.Batches, plan: dict, pairs: list, kept: np.ndarray,
                 held: np.ndarray) -> dict:
    """The pooled clipped objective at this encoder, with the statistics recomputed from THIS encoder.

    Used by the diagnostics of PRD section 7.1: the critics stay exactly as they were, and only the encoder and
    its own statistics move, which is the function the gradient differentiated.
    """
    rows_all = plan["represent"]
    with torch.no_grad():
        z = dn.representation_values(rep, data, rows_all, kept, grad=False)
    norm = v3.Normaliser.fit(z, enabled=True)
    groups, index_of = build_groups(rep, data, plan, pairs, kept, held, rows_all)
    with torch.no_grad():
        clipped, info = v3.pooled_objective(z, norm, groups, data, index_of)
    return {"signed_gap": info["signed_gap"], "clipped_gap": float(clipped), "normaliser": norm.record()}


def update(rep: dn.Representation, data: dn.Batches, plan: dict, pairs: list, kept: np.ndarray, held: np.ndarray,
           opt, norm_now: v3.Normaliser) -> dict:
    """One update, then the statistics of the POST-step encoder.

    `norm_now` is the normalisation the critics were fitted against; the objective is differentiated with the
    statistics recomputed inside the graph, and the returned `norm_next` comes from the encoder after the step,
    so the next critic refit sees coordinates of the same vintage as the weights.
    """
    rows_all = plan["represent"]
    z_leaf = dn.representation_values(rep, data, rows_all, kept, grad=False).clone().requires_grad_(True)
    norm_dyn = v3.Normaliser.fit(z_leaf, enabled=True)
    groups, index_of = build_groups(rep, data, plan, pairs, kept, held, rows_all)
    loss, info = v3.pooled_objective(z_leaf, norm_dyn, groups, data, index_of)
    stepped, grad_norm, first_conv, changes = False, 0.0, 0.0, {}
    if info["signed_gap"] > 0:
        loss.backward()
        upstream = z_leaf.grad
        before = {n: p.detach().clone() for n, p in rep.named_parameters()}
        rep.train()
        opt.zero_grad(set_to_none=True)
        for s in range(0, len(rows_all), TRAIN["grad_batch"]):
            r = rows_all[s:s + TRAIN["grad_batch"]]
            idx = torch.as_tensor(r, device=data.dev)
            rep(data.pixels(r, kept), data.x[idx]).backward(upstream[s:s + len(r)])
        grads = {n: p.grad for n, p in rep.named_parameters() if p.grad is not None}
        grad_norm = float(torch.sqrt(sum((g ** 2).sum() for g in grads.values())))
        first_conv = float(sum(g.norm() for n, g in grads.items() if n.endswith("towers.0.body.0.weight")))
        opt.step()
        rep.eval()
        stepped = True
        changes = {n: float((p.detach() - before[n]).norm()) for n, p in rep.named_parameters()}
    with torch.no_grad():
        z_post = dn.representation_values(rep, data, rows_all, kept, grad=False)      # PRD section 6, step 4
        norm_next = v3.Normaliser.fit(z_post, enabled=True).frozen()
        h = z_post[:, 2:]
        scale = {"h_sd": h.std(0, unbiased=False).tolist(), "h_mean": h.mean(0).tolist(), "h_absmax": float(h.abs().max())}
    return {"raw_signed_gap": info["signed_gap"], "clipped_gap": float(loss.detach()), "stepped": stepped,
            "grad_norm": grad_norm, "first_conv_grad_norm": first_conv,
            "parameter_change_norm": changes, "fold_risks": info["folds"], "scale": scale,
            "normaliser_used": norm_now.record(), "normaliser_after": norm_next.record()}, norm_next


# ----------------------------------------------------------------------------- the arm

def fold_plan(rows: dict) -> dict:
    fold = np.asarray(rows["represent"])
    order = np.random.default_rng(seed_for("fold_split")).permutation(len(fold))
    parts = [np.sort(fold[order[i::TRAIN["folds"]]]) for i in range(TRAIN["folds"])]
    fit_sets, val_sets = [], []
    for i in range(TRAIN["folds"]):
        pool = np.sort(np.concatenate([p for j, p in enumerate(parts) if j != i]))
        inner = np.random.default_rng(seed_for("fold_split", i, 1)).permutation(len(pool))
        n_val = max(1, int(round(CRITIC["val_fraction"] * len(pool))))
        val_sets.append(np.sort(pool[inner[:n_val]]))
        fit_sets.append(np.sort(pool[inner[n_val:]]))
    return {"represent": fold, "select": np.asarray(rows["select"]), "folds": TRAIN["folds"],
            "fit_sets": fit_sets, "val_sets": val_sets, "score_sets": parts}


def run_arm(arm: str, rep: dn.Representation, data: dn.Batches, rows: dict, split, kept: np.ndarray,
            held: np.ndarray, deadline: Deadline, log=None) -> dict:
    plan = fold_plan(rows)
    opt = torch.optim.Adam(rep.parameters(), lr=TRAIN["lr"])
    t0 = time.monotonic()
    with torch.no_grad():
        norm = v3.Normaliser.fit(dn.representation_values(rep, data, plan["represent"], kept, grad=False),
                                 enabled=True).frozen()
    states = [None] * plan["folds"]

    def refit(steps: int, iteration: int, old: v3.Normaliser | None) -> list:
        pairs = []
        for i in range(plan["folds"]):
            b, a_, states[i] = fit_critics(rep, norm, data, plan["fit_sets"][i], plan["val_sets"][i], kept, held,
                                           steps, i, iteration, states[i], old, deadline)
            pairs.append((b, a_))
        return pairs

    pairs = refit(CRITIC["initial_steps"], 0, None)
    history, saved, actual = [], {}, 0
    diag_keep, pending = {}, None

    def snapshot(it: int) -> None:
        if it in TRAIN["checkpoints"]:
            saved[it] = {"state": {k: v.detach().cpu().clone() for k, v in rep.state_dict().items()},
                         "normaliser": norm.record(), "critics": [st["info"] for st in states]}

    snapshot(0)
    stopped = None
    for it in range(1, TRAIN["iters"] + 1):
        if deadline.over():
            stopped = f"time cap at iteration {it}"
            break
        pre_state = {k: v.detach().cpu().clone() for k, v in rep.state_dict().items()}
        pre_critics = [{"base": {k: v.cpu().clone() for k, v in st["base"].items()},
                        "aug": {k: v.cpu().clone() for k, v in st["aug"].items()}} for st in states]
        step, norm_next = update(rep, data, plan, pairs, kept, held, opt, norm)
        if step["stepped"]:
            actual += 1
            if actual in ACTUAL_UPDATE_ORDINALS:
                diag_keep[actual] = {"outer_iteration": it, "pre_state": pre_state,
                                     "post_state": {k: v.detach().cpu().clone() for k, v in rep.state_dict().items()},
                                     "critics": pre_critics, "norm_used": norm.record(),
                                     "norm_after": norm_next.record()}
        old_norm, norm = norm, norm_next
        pairs = refit(CRITIC["steps_per_iter"], it, old_norm)
        history.append(dict(step, iteration=it, actual_update=actual if step["stepped"] else None,
                            critics=[st["info"] for st in states], seconds=round(time.monotonic() - t0, 1),
                            vintage_ok=bool(norm.record() == step["normaliser_after"])))
        snapshot(it)
        if log and (it <= 3 or it % 10 == 0):
            log(f"    [{arm}] iter {it:3d}: raw gap {step['raw_signed_gap']:+.5f} stepped {step['stepped']} "
                f"(actual {actual}) grad {step['grad_norm']:.2e} | h sd "
                f"{np.round(step['scale']['h_sd'], 3).tolist()} | critic val gap "
                f"{np.mean([c['gap_val'] for c in history[-1]['critics']]):+.5f} ({time.monotonic() - t0:.0f} s)")
    done = len(history)
    return {"arm": arm, "plan": plan, "history": history, "actual_updates": actual,
            "iterations_planned": TRAIN["iters"], "iterations_done": done,
            "status": "complete" if done == TRAIN["iters"] else "partial_time_cap",
            "stopped": stopped, "checkpoint_states": {k: v["state"] for k, v in saved.items()},
            "checkpoint_normalisers": {k: v["normaliser"] for k, v in saved.items()},
            "checkpoint_critics": {k: v["critics"] for k, v in saved.items()},
            "diag": diag_keep, "seconds": round(time.monotonic() - t0, 1)}


# ----------------------------------------------------------------------------- selection and diagnostics

def select(out: dict, data: dn.Batches, rows: dict, split, kept: np.ndarray, held: np.ndarray, x_dim: int,
           dev, log=None) -> dict:
    """Fresh critics on the representation fold, scored on `select`; the smallest clipped gap wins, ties earliest."""
    table = {}
    saved = dict(de.CHECK)
    try:
        de.CHECK.update({"base_steps": SELECT["steps"], "augmented_steps": SELECT["steps"],
                         "eval_every": SELECT["eval_every"], "batch": SELECT["batch"],
                         "val_fraction": SELECT["val_fraction"]})
        for k in sorted(out["checkpoint_states"]):
            w = v3.load_normalised(out["checkpoint_states"][k], kept, x_dim, dev, t16.SPEC16,
                                   out["checkpoint_normalisers"][k])
            table[k] = de.fresh_gap(w, data, rows["represent"], rows["select"], split, seed_for("selection", 0, k),
                                    t16.SPEC16)
    finally:
        de.CHECK.clear(); de.CHECK.update(saved)
    best = None
    for k in sorted(table):
        if best is None or table[k]["clipped_gap"] < table[best]["clipped_gap"] - SELECT["tolerance"]:
            best = k
    if log:
        log(f"  [{out['arm']}] selection: {best} | gaps "
            f"{{{', '.join(f'{k}: {v['gap']:+.4f}' for k, v in sorted(table.items()))}}}")
    return {"chosen": best, "table": {str(k): v for k, v in table.items()}}


def actual_update_diagnostics(out: dict, data: dn.Batches, rows: dict, split, kept: np.ndarray, held: np.ndarray,
                              x_dim: int, dev, deadline: Deadline, log=None) -> list:
    """PRD section 7, indexed by actual optimizer updates.

    7.1 keeps the critics the update really used and recomputes the statistics from each encoder: that is the
        function the gradient saw.  The critics are NOT carried to new coordinates here, because carrying them
        would compare two different functions.
    7.2 fits brand new critics to each encoder with one budget and one seed.
    """
    plan = out["plan"]
    diag = []
    for ordinal in ACTUAL_UPDATE_ORDINALS:
        if ordinal not in out["diag"]:
            diag.append({"actual_update": ordinal, "status": "not_reached"})
            continue
        rec = out["diag"][ordinal]
        reps = {}
        for tag in ("pre", "post"):
            r = dn.Representation(kept, x_dim, spec=t16.SPEC16).to(dev)
            r.load_state_dict(rec[f"{tag}_state"])
            reps[tag] = r.eval()
        fixed_pairs = []
        for i in range(plan["folds"]):
            z_dim = 2 + dn.REPRESENTATION_DIM
            b = dn.BaseCritic(z_dim).to(dev); b.load_state_dict(rec["critics"][i]["base"]); b.eval()
            a_ = dn.AugmentedCritic(b, len(held), z_dim, t16.SPEC16).to(dev)
            a_.load_state_dict(rec["critics"][i]["aug"]); a_.eval()
            fixed_pairs.append((b, a_))
        entry = {"actual_update": ordinal, "outer_iteration": rec["outer_iteration"], "status": "done",
                 "with_the_actual_critics_fixed": {
                     tag: objective_at(reps[tag], data, plan, fixed_pairs, kept, held) for tag in ("pre", "post")}}
        w = entry["with_the_actual_critics_fixed"]
        entry["with_the_actual_critics_fixed"]["change_raw"] = w["post"]["signed_gap"] - w["pre"]["signed_gap"]
        entry["with_the_actual_critics_fixed"]["change_clipped"] = w["post"]["clipped_gap"] - w["pre"]["clipped_gap"]
        if deadline.over():
            entry["with_fresh_critics"] = {"status": "not run: time cap"}
        else:
            fresh = {}
            for tag in ("pre", "post"):
                with torch.no_grad():
                    norm = v3.Normaliser.fit(dn.representation_values(reps[tag], data, plan["represent"], kept, grad=False),
                                             enabled=True).frozen()
                pairs = []
                for i in range(plan["folds"]):
                    b, a_, _ = fit_critics(reps[tag], norm, data, plan["fit_sets"][i], plan["val_sets"][i], kept,
                                           held, CRITIC["initial_steps"], i, PURPOSE["refit_diag"], None, None,
                                           deadline)
                    pairs.append((b, a_))
                fresh[tag] = objective_at(reps[tag], data, plan, pairs, kept, held)
            fresh["change_raw"] = fresh["post"]["signed_gap"] - fresh["pre"]["signed_gap"]
            entry["with_fresh_critics"] = fresh
        diag.append(entry)
        if log:
            f = entry["with_the_actual_critics_fixed"]
            g = entry.get("with_fresh_critics", {})
            log(f"  [{out['arm']}] actual update {ordinal} (iter {rec['outer_iteration']}): actual critics "
                f"{f['change_raw']:+.6f} | fresh critics "
                f"{g.get('change_raw', float('nan')):+.6f}" if "change_raw" in g else
                f"  [{out['arm']}] actual update {ordinal}: actual critics {f['change_raw']:+.6f} | fresh: {g.get('status')}")
    return diag
