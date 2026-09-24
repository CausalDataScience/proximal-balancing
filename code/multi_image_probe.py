"""Learner of the multi-image experiments: encoder, nested critics, balance search, audit screen, nuisances,
honest AIPW and aggregation (PRD memo/2026-09-20-*-multi-image-prd-v1.md, sections 6 to 9).

Blocks and splits are numbered 1..4 as in the PRD; a split is the tuple of held-out blocks.  The representation
is Z = (X, H) with H = head(trunk features of the retained images, keep mask).  The encoder never fetches a
held-out pixel and the augmented critic's residual branch never fetches a retained one: both go through
BalanceView.images, which logs every block it serves.  BalanceView has no Y, so nothing that shapes the
representation or a critic can read the outcome.

Interpretation fixed here where the PRD leaves room (recorded in the run protocol as well):
- One outer iteration is 5 encoder steps against frozen critics, then 25 critic steps against the frozen
  encoder, then the fit/select record.  With the 100 initial critic steps this is the PRD's 600 critic and 100
  encoder steps, and every recorded gap comes from critics that were refitted to the encoder they judge.
- The frozen critics of an encoder step are the selected ones: the base critic, and for the augmented critic the
  better of its current state and the lifted base on D_select.  The augmented critic itself keeps training from
  its own state.
- Ties between an augmented checkpoint and the lifted base go to the lifted base, which is the null.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn

import multi_image_scm as scm
from sim5_cnn_algorithm1 import aggregate, aipw_scores
from sim5_digit_proxy import paired_gap, state_digest

J = scm.J
FEATURES, CODE, X_DIM = 16, 8, 2
Z_DIM = X_DIM + CODE
ROLES = {"Dfit": 4000, "Dselect": 1000, "Daudit": 1000, "Ntrain": 4800, "Nvalidation": 1200, "E": 12000}
SEARCH = {"critic_lr": 1e-3, "encoder_lr": 3e-4, "batch": 128, "clip": 5.0, "initial_critic_steps": 100,
          "outer": 20, "critic_steps": 25, "encoder_steps": 5, "encoder_min_delta": 1e-6}
AUDIT = {"steps": 600, "every": 25, "threshold": 5e-4}
NUISANCE = {"lr": 1e-3, "batch": 128, "max_steps": 1000, "every": 25, "min_steps": 100, "patience": 8,
            "min_delta": 1e-5, "hidden": 64}
RAW = {"pre_epochs": 10, "tune_epochs": 20, "patience": 5, "min_delta": 1e-5, "lr": 1e-3, "batch": 128}
ETA = 0.05
RHO = 0.10
EVAL_CHUNK = 1000
STAGES = {"encoder": 1, "search_critics": 2, "search_batches": 3, "audit_critics": 4, "audit_batches": 5,
          "nuisance": 6, "raw": 7, "reference": 8}


def stage_seed(algo_seed: int, split: tuple[int, ...], stage: str) -> int:
    code = sum(1 << (j - 1) for j in split)
    return int(np.random.SeedSequence([algo_seed, code, STAGES[stage]]).generate_state(1)[0])


def roles(n: int, seed: int) -> dict[str, np.ndarray]:
    """One random partition of the rows into the six honest roles (PRD section 6)."""
    if n != sum(ROLES.values()):
        scale = n / sum(ROLES.values())
        sizes = {k: int(round(v * scale)) for k, v in ROLES.items()}
        sizes["E"] += n - sum(sizes.values())
    else:
        sizes = dict(ROLES)
    perm = np.random.default_rng(seed).permutation(n)
    out, start = {}, 0
    for name, size in sizes.items():
        out[name] = np.sort(perm[start:start + size])
        start += size
    return out


def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class BalanceView:
    """What the representation and the critics may read: X, the four images and A.  There is no Y here."""

    def __init__(self, x: np.ndarray, a: np.ndarray, src: np.ndarray, bank_pixels: torch.Tensor, device) -> None:
        self.device = device
        self.x = torch.as_tensor(x, dtype=torch.float32, device=device)
        self.a = torch.as_tensor(a, dtype=torch.float32, device=device)
        self.src = torch.as_tensor(src, dtype=torch.long, device=device)
        self.bank = bank_pixels
        self.served: list[tuple[str, tuple[int, ...]]] = []
        self.log = False

    def images(self, rows: torch.Tensor, blocks: tuple[int, ...], reader: str) -> dict[int, torch.Tensor]:
        if self.log:
            self.served.append((reader, tuple(blocks)))
        out = {}
        for j in blocks:
            px = self.bank[self.src[rows, j - 1]]
            px = px.unsqueeze(1) if px.dim() == 3 else px.permute(0, 3, 1, 2)
            out[j] = px.to(torch.float32) / 255.0
        return out


class Trunk(nn.Module):
    """PRD 7.1: three stride-2 3x3 convolutions, global average pooling, a linear map to 16 features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(channels, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU())
        self.out = nn.Linear(32, FEATURES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.out(self.conv(images).mean(dim=(2, 3)))


def block_features(trunk: Trunk, images: dict[int, torch.Tensor], n: int, device) -> torch.Tensor:
    """(n, 64): the trunk's features in the slots of the served blocks, constant zero everywhere else."""
    out = torch.zeros(n, J * FEATURES, device=device)
    if images:
        order = sorted(images)
        feats = trunk(torch.cat([images[j] for j in order])).split(n)
        for j, f in zip(order, feats):
            out[:, (j - 1) * FEATURES:j * FEATURES] = f
    return out


def mask(blocks: tuple[int, ...], device) -> torch.Tensor:
    m = torch.zeros(1, J, device=device)
    for j in blocks:
        m[0, j - 1] = 1.0
    return m


class Encoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.trunk = Trunk(channels)
        self.head = mlp([J * FEATURES + J, 32, CODE])

    def forward(self, x: torch.Tensor, images: dict[int, torch.Tensor], keep: torch.Tensor) -> torch.Tensor:
        feats = block_features(self.trunk, images, len(x), x.device)
        return torch.cat([x, self.head(torch.cat([feats, keep.expand(len(x), -1)], dim=1))], dim=1)


class BaseCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = mlp([Z_DIM, 32, 32, 1])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class AugmentedCritic(nn.Module):
    """Its own base branch plus a residual logit from its own trunk on the held-out images (PRD 7.2)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.base = BaseCritic()
        self.trunk = Trunk(channels)
        self.residual = mlp([J * FEATURES + J + Z_DIM, 32, 1])
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, z: torch.Tensor, held_images: dict[int, torch.Tensor], held: torch.Tensor) -> torch.Tensor:
        feats = block_features(self.trunk, held_images, len(z), z.device)
        return self.base(z) + self.residual(torch.cat([feats, held.expand(len(z), -1), z], dim=1)).squeeze(-1)

    def lifted_state(self, base: BaseCritic) -> dict:
        """This critic's state with the base branch replaced by `base` and the residual's last layer at zero:
        as a function it is exactly `base`.  Tensors are copies; no parameter object is ever shared."""
        state = copy.deepcopy(self.state_dict())
        for key, value in base.state_dict().items():
            state[f"base.{key}"] = value.detach().clone()
        last = len(self.residual) - 1
        state[f"residual.{last}.weight"].zero_()
        state[f"residual.{last}.bias"].zero_()
        return state


def brier(a: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
    return ((a - torch.sigmoid(logit)) ** 2).mean()


def _seeded(factory, seed: int, device):
    torch.manual_seed(seed)
    return factory().to(device)


def _clone_state(module: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


class Split:
    """Everything one held-out set needs: its masks and the two disjoint block tuples."""

    def __init__(self, split: tuple[int, ...], device) -> None:
        self.held = tuple(split)
        self.kept = tuple(j for j in range(1, J + 1) if j not in split)
        self.keep_mask, self.held_mask = mask(self.kept, device), mask(self.held, device)
        self.id = scm.split_id(split)


def encode(encoder: Encoder, view: BalanceView, rows: torch.Tensor, sp: Split) -> torch.Tensor:
    return encoder(view.x[rows], view.images(rows, sp.kept, "encoder"), sp.keep_mask)


@torch.no_grad()
def encode_all(encoder: Encoder, view: BalanceView, rows: np.ndarray, sp: Split) -> torch.Tensor:
    idx = torch.as_tensor(rows, device=view.device)
    return torch.cat([encode(encoder, view, idx[s:s + EVAL_CHUNK], sp) for s in range(0, len(idx), EVAL_CHUNK)])


@torch.no_grad()
def predict(q0: BaseCritic, q1: AugmentedCritic, view: BalanceView, rows: np.ndarray, z: torch.Tensor,
            sp: Split) -> tuple[np.ndarray, np.ndarray]:
    """Both critics' probabilities on `rows`, whose representation `z` is given in the same order."""
    idx = torch.as_tensor(rows, device=view.device)
    p0, p1 = [], []
    for s in range(0, len(idx), EVAL_CHUNK):
        zz = z[s:s + EVAL_CHUNK]
        p0.append(torch.sigmoid(q0(zz)))
        p1.append(torch.sigmoid(q1(zz, view.images(idx[s:s + EVAL_CHUNK], sp.held, "critic"), sp.held_mask)))
    return torch.cat(p0).cpu().numpy().astype(np.float64), torch.cat(p1).cpu().numpy().astype(np.float64)


def _step(opt: torch.optim.Optimizer, loss: torch.Tensor, params) -> None:
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(params, SEARCH["clip"])
    opt.step()


def critic_steps(q0, q1, opt0, opt1, view: BalanceView, fit_rows: torch.Tensor, z_of, sp: Split, steps: int,
                 rng: np.random.Generator) -> None:
    """`steps` paired minibatch steps; both critics see the same rows, each descends its own Brier risk."""
    for _ in range(steps):
        rows = fit_rows[torch.as_tensor(rng.integers(0, len(fit_rows), SEARCH["batch"]), device=view.device)]
        z = z_of(rows)
        _step(opt0, brier(view.a[rows], q0(z)), list(q0.parameters()))
        logit1 = q1(z, view.images(rows, sp.held, "critic"), sp.held_mask)
        _step(opt1, brier(view.a[rows], logit1), list(q1.parameters()))


def select_augmented(q0: BaseCritic, q1: AugmentedCritic, view: BalanceView, select_rows: np.ndarray,
                     z_select: torch.Tensor, sp: Split) -> tuple[AugmentedCritic, str]:
    """The better of q1 and the lifted base on D_select, as a separate frozen module; ties go to the lifted."""
    lifted = copy.deepcopy(q1)
    lifted.load_state_dict(q1.lifted_state(q0))
    a = view.a[torch.as_tensor(select_rows, device=view.device)].cpu().numpy()
    _, p_own = predict(q0, q1, view, select_rows, z_select, sp)
    _, p_lift = predict(q0, lifted, view, select_rows, z_select, sp)
    own, lift = float(((a - p_own) ** 2).mean()), float(((a - p_lift) ** 2).mean())
    return (copy.deepcopy(q1), "own") if own < lift else (lifted, "lifted")


def gap_record(q0, q1, view: BalanceView, rows: np.ndarray, z: torch.Tensor, sp: Split) -> dict:
    p0, p1 = predict(q0, q1, view, rows, z, sp)
    a = view.a[torch.as_tensor(rows, device=view.device)].cpu().numpy()
    return paired_gap(a, p0, p1)


def search(view: BalanceView, role: dict[str, np.ndarray], split: tuple[int, ...], channels: int,
           algo_seed: int) -> dict:
    """Balance-only representation search for one held-out set (PRD 7.3)."""
    dev, sp = view.device, Split(split, view.device)
    encoder = _seeded(lambda: Encoder(channels), stage_seed(algo_seed, split, "encoder"), dev)
    initial = _clone_state(encoder)
    torch.manual_seed(stage_seed(algo_seed, split, "search_critics"))
    q0, q1 = BaseCritic().to(dev), AugmentedCritic(channels).to(dev)
    q1.load_state_dict(q1.lifted_state(q0))
    opt0 = torch.optim.Adam(q0.parameters(), lr=SEARCH["critic_lr"])
    opt1 = torch.optim.Adam(q1.parameters(), lr=SEARCH["critic_lr"])
    opt_e = torch.optim.Adam(encoder.parameters(), lr=SEARCH["encoder_lr"])
    rng = np.random.default_rng(stage_seed(algo_seed, split, "search_batches"))
    fit = torch.as_tensor(role["Dfit"], device=dev)

    def frozen_z(rows):
        with torch.no_grad():
            return encode(encoder, view, rows, sp)

    def delta(prefix: str = "") -> float:
        return float(torch.sqrt(sum(((v - initial[k]) ** 2).sum() for k, v in encoder.state_dict().items()
                                    if k.startswith(prefix))))

    history, best = [], None
    for k in range(SEARCH["outer"] + 1):
        grad_norms = []
        if k:
            for p in list(q0.parameters()) + list(chosen.parameters()):
                p.requires_grad_(False)
            for _ in range(SEARCH["encoder_steps"]):
                rows = fit[torch.as_tensor(rng.integers(0, len(fit), SEARCH["batch"]), device=dev)]
                z = encode(encoder, view, rows, sp)
                r0 = brier(view.a[rows], q0(z))
                r1 = brier(view.a[rows], chosen(z, view.images(rows, sp.held, "critic"), sp.held_mask))
                loss = torch.clamp(r0 - r1, min=0.0)
                opt_e.zero_grad(set_to_none=True)
                loss.backward()
                grad_norms.append(float(nn.utils.clip_grad_norm_(encoder.parameters(), SEARCH["clip"])))
                opt_e.step()
            for p in q0.parameters():
                p.requires_grad_(True)
        critic_steps(q0, q1, opt0, opt1, view, fit, frozen_z, sp,
                     SEARCH["critic_steps"] if k else SEARCH["initial_critic_steps"], rng)
        z_fit, z_sel = encode_all(encoder, view, role["Dfit"], sp), encode_all(encoder, view, role["Dselect"], sp)
        chosen, which = select_augmented(q0, q1, view, role["Dselect"], z_sel, sp)
        rec = {"outer": k, "encoder_updates": k * SEARCH["encoder_steps"], "augmented": which,
               "fit": gap_record(q0, chosen, view, role["Dfit"], z_fit, sp),
               "select": gap_record(q0, chosen, view, role["Dselect"], z_sel, sp),
               "parameter_delta": delta(), "trunk_delta": delta("trunk."), "encoder_grad_norms": grad_norms}
        history.append(rec)
        if best is None or rec["select"]["clipped_gap"] < best["clipped_gap"] - SEARCH["encoder_min_delta"]:
            best = {"clipped_gap": rec["select"]["clipped_gap"], "outer": k, "state": _clone_state(encoder)}
    final = copy.deepcopy(encoder)
    final.load_state_dict(best["state"])
    untrained = copy.deepcopy(encoder)
    untrained.load_state_dict(initial)
    return {"split": sp.id, "encoder": final.eval(), "untrained": untrained.eval(), "history": history,
            "selected_outer": best["outer"], "selected_updates": best["outer"] * SEARCH["encoder_steps"],
            "selected_was_updated": best["outer"] > 0, "total_encoder_updates": SEARCH["outer"] * SEARCH["encoder_steps"],
            "initial_digest": state_digest(initial), "selected_digest": state_digest(best["state"])}


def audit(view: BalanceView, role: dict[str, np.ndarray], split: tuple[int, ...], encoder: Encoder, channels: int,
          algo_seed: int, threshold: float = AUDIT["threshold"]) -> dict:
    """Fresh critics on the frozen encoder, chosen on D_select, then D_audit read once (PRD 8.1)."""
    dev, sp = view.device, Split(split, view.device)
    z = {name: encode_all(encoder, view, role[name], sp) for name in ("Dfit", "Dselect", "Daudit")}
    torch.manual_seed(stage_seed(algo_seed, split, "audit_critics"))
    q0, q1 = BaseCritic().to(dev), AugmentedCritic(channels).to(dev)
    q1.load_state_dict(q1.lifted_state(q0))
    opt0 = torch.optim.Adam(q0.parameters(), lr=SEARCH["critic_lr"])
    opt1 = torch.optim.Adam(q1.parameters(), lr=SEARCH["critic_lr"])
    rng = np.random.default_rng(stage_seed(algo_seed, split, "audit_batches"))
    fit = torch.as_tensor(role["Dfit"], device=dev)
    position = torch.full((int(fit.max()) + 1,), -1, dtype=torch.long, device=dev)
    position[fit] = torch.arange(len(fit), device=dev)
    a_sel = view.a[torch.as_tensor(role["Dselect"], device=dev)].cpu().numpy()
    best0 = {"risk": np.inf, "step": None, "state": None}
    best1 = {"risk": np.inf, "step": None, "state": None}
    for done in range(AUDIT["every"], AUDIT["steps"] + 1, AUDIT["every"]):
        critic_steps(q0, q1, opt0, opt1, view, fit, lambda rows: z["Dfit"][position[rows]], sp, AUDIT["every"], rng)
        p0, p1 = predict(q0, q1, view, role["Dselect"], z["Dselect"], sp)
        for best, p, module in ((best0, p0, q0), (best1, p1, q1)):
            risk = float(((a_sel - p) ** 2).mean())
            if risk < best["risk"]:
                best.update(risk=risk, step=done, state=_clone_state(module))
    q0.load_state_dict(best0["state"])
    q1.load_state_dict(best1["state"])
    chosen, which = select_augmented(q0, q1, view, role["Dselect"], z["Dselect"], sp)
    out = {"split": sp.id, "base_step": best0["step"], "augmented_step": best1["step"], "augmented": which,
           "encoder_digest": state_digest(encoder)}
    for name in ("Dfit", "Dselect", "Daudit"):
        out[name] = gap_record(q0, chosen, view, role[name], z[name], sp)
    p0, p1 = predict(q0, chosen, view, role["Daudit"], z["Daudit"], sp)
    a_aud = view.a[torch.as_tensor(role["Daudit"], device=dev)].cpu().numpy()
    out["audit_d"] = ((a_aud - p0) ** 2 - (a_aud - p1) ** 2).astype(np.float64)
    g = out["Daudit"]
    finite = bool(np.isfinite([g["signed_gap"], g["se"]]).all() and g["se"] >= 0)
    out["screen"] = {"threshold": threshold, "finite": finite, "pass": bool(finite and g["clipped_gap"] <= threshold),
                     "passed_with_negative_gap": bool(finite and g["signed_gap"] < 0)}
    return out


# ----------------------------------------------------------------------------- nuisances, AIPW, aggregation

def fit_head(inputs: torch.Tensor, target: torch.Tensor, train: np.ndarray, val: np.ndarray, kind: str,
             seed: int) -> dict:
    """One nuisance head, input -> 64 -> 64 -> 1, with the PRD 8.2 stopping rule; `kind` is brier or mse."""
    cfg = NUISANCE
    torch.manual_seed(seed)
    net = mlp([inputs.shape[1], cfg["hidden"], cfg["hidden"], 1])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(seed)
    tr, va = torch.as_tensor(train), torch.as_tensor(val)

    def risk(rows):
        out = net(inputs[rows]).squeeze(-1)
        out = torch.sigmoid(out) if kind == "brier" else out
        return ((target[rows] - out) ** 2).mean()

    best, bad, step = {"risk": np.inf, "step": 0, "state": None}, 0, 0
    while step < cfg["max_steps"]:
        for _ in range(cfg["every"]):
            rows = tr[torch.as_tensor(rng.integers(0, len(tr), cfg["batch"]))]
            opt.zero_grad(set_to_none=True)
            risk(rows).backward()
            opt.step()
        step += cfg["every"]
        with torch.no_grad():
            v = float(risk(va))
        if v < best["risk"] - cfg["min_delta"]:
            best, bad = {"risk": v, "step": step, "state": _clone_state(net)}, 0
        else:
            bad += 1
        if step >= cfg["min_steps"] and bad >= cfg["patience"]:
            break
    net.load_state_dict(best["state"])
    return {"net": net.eval(), "kind": kind, "val_risk": best["risk"], "selected_step": best["step"],
            "steps_run": step}


def aipw_from_features(feat: np.ndarray, a: np.ndarray, y: np.ndarray, role: dict[str, np.ndarray],
                       seed: int) -> dict:
    """Nuisances on N from fixed features, scores on E (PRD 8.2 and 8.3).  Runs on the CPU: the heads are tiny."""
    if not np.isfinite(feat).all():
        return {"finite": False, "theta": None}
    f = torch.as_tensor(feat, dtype=torch.float32)
    at, yt = torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    tr, va, ev = role["Ntrain"], role["Nvalidation"], role["E"]
    heads = {"e": fit_head(f, at, tr, va, "brier", seed)}
    for arm in (0, 1):
        heads[f"m{arm}"] = fit_head(f, yt, tr[a[tr] == arm], va[a[va] == arm], "mse", seed + 1 + arm)
    with torch.no_grad():
        fe = f[torch.as_tensor(ev)]
        e_raw = torch.sigmoid(heads["e"]["net"](fe).squeeze(-1)).numpy().astype(np.float64)
        m0, m1 = [heads[f"m{arm}"]["net"](fe).squeeze(-1).numpy().astype(np.float64) for arm in (0, 1)]
    finite = bool(np.isfinite(e_raw).all() and np.isfinite(m0).all() and np.isfinite(m1).all())
    if not finite:
        return {"finite": False, "theta": None}
    e = np.clip(e_raw, ETA, 1 - ETA)
    scores = aipw_scores(e, m0, m1, a[ev], y[ev])
    return {"finite": True, "theta": float(scores.mean()), "score_sd": float(scores.std(ddof=1)), "n_e": len(ev),
            "scores": scores,
            "overlap": {"outside_before_clip": float(((e_raw < ETA) | (e_raw > 1 - ETA)).mean()),
                        "raw_range": [float(e_raw.min()), float(e_raw.max())],
                        "clipped_range": [float(e.min()), float(e.max())]},
            "heads": {k: {"val_risk": h["val_risk"], "selected_step": h["selected_step"],
                          "steps_run": h["steps_run"]} for k, h in heads.items()}}


def candidate(view: BalanceView, y: np.ndarray, role: dict[str, np.ndarray], split: tuple[int, ...],
              encoder: Encoder, algo_seed: int) -> dict:
    """theta_S for a frozen encoder: Z on N and E, nuisances on N, AIPW on E."""
    sp = Split(split, view.device)
    before = state_digest(encoder)
    n = len(view.x)
    z = np.zeros((n, Z_DIM), dtype=np.float32)
    for name in ("Ntrain", "Nvalidation", "E"):
        z[role[name]] = encode_all(encoder, view, role[name], sp).cpu().numpy()
    out = aipw_from_features(z, view.a.cpu().numpy().astype(np.float64), y, role,
                             stage_seed(algo_seed, split, "nuisance"))
    out["encoder_digest_unchanged"] = state_digest(encoder) == before
    return out


def combine(estimates: dict[str, float], passed: dict[str, bool], rho: float = RHO) -> dict:
    """Radius graph over the retained candidates and the median of the unique largest component (PRD 8.3)."""
    kept = [sid for sid in estimates if passed[sid] and estimates[sid] is not None]
    return aggregate(kept, [estimates[sid] for sid in kept], rho)


def timed(fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, time.time() - t0
