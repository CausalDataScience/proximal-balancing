"""Direct image balancing: everything that happens after a representation exists
(execution PRD memo/2026-09-21-direct-image-balancing-figure-run-prd-v1.md, section 3).

    check       the independent balance check on rows the representation never saw, with FRESH critics
    estimate    the honest AIPW estimate from Z = (X, h): nuisances on one fold, scores on another
    combine     Algorithm 1's aggregation, by the sealed functions, from the splits that passed the check
    comparators X only, the oracle, and an end-to-end CNN on all six images

The check uses the sealed screen's statistic: per row d = (q0 - A)^2 - (q1 - A)^2 out of fold, pooled over the two
directions; gap = mean d, se = sd(d) / sqrt(n), upper = max(gap, 0) + z se with z for 0.05 / 30.

One thing is added to the sealed recipe, because these critics are networks and not ridge-penalised probits.  An
augmented critic that overfits its fitting half predicts WORSE than the base critic on the other half, the gap
goes negative, and a real imbalance is hidden behind it; P1 saw training critics reach -3.6e-2 on a fold they
were not fitted on.  So each fresh critic is stopped early on a fifth of its own fitting half, and the augmented
critic's starting point, where it still equals the base critic, is one of the candidates.  If the held-out images
carry nothing, the augmented critic stays the base critic and the gap stays at zero instead of below it.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import norm
from torch import nn

import direct_balance_data as db
import direct_balance_net as dn
import family_v2_probe as pr
import probe_structured_scm as ps

CHECK = {"base_steps": 300, "augmented_steps": 300, "eval_every": 25, "batch": 256, "val_fraction": 0.2}
NUISANCE = {"hidden": (64, 64), "epochs": 200, "patience": 15, "batch": 256, "lr": 1e-3, "weight_decay": 1e-4,
            "val_fraction": 0.2}
RAW_CNN = {"epochs": 10, "patience": 2, "batch": 128, "lr": 1e-3, "val_fraction": 0.2, "feature_dim": dn.FEATURE_DIM,
           "hidden": (128, 64), "eval_rows": 1024}
ALPHA, N_SPLITS, CLIP = 0.05, 30, (0.05, 0.95)


def _state(module: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


# ----------------------------------------------------------------------------- the independent balance check

def _fit_fresh_critics(z: torch.Tensor, a: torch.Tensor, data: dn.Batches, rows: np.ndarray, held: np.ndarray,
                       seed: int, spec: dict | None):
    """Fresh q0 and q1 on one half of the check fold, each stopped early on a fifth of that half."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    order = rng.permutation(len(rows))
    n_val = int(round(CHECK["val_fraction"] * len(rows)))
    val, fit = np.sort(order[:n_val]), order[n_val:]
    val_t = torch.as_tensor(val, device=data.dev)

    base = dn.BaseCritic(z.shape[1]).to(data.dev)
    opt = torch.optim.Adam(base.parameters(), lr=dn.TRAIN["lr_critic"])
    best = {"loss": math.inf, "state": None, "step": 0}
    for step in range(CHECK["base_steps"] + 1):
        if step % CHECK["eval_every"] == 0:
            with torch.no_grad():
                loss = float(dn.brier(base(z[val_t]), a[val_t]))
            if loss < best["loss"] - 1e-7:
                best = {"loss": loss, "state": _state(base), "step": step}
        if step == CHECK["base_steps"]:
            break
        idx = torch.as_tensor(fit[rng.integers(0, len(fit), CHECK["batch"])], device=data.dev)
        opt.zero_grad(set_to_none=True)
        dn.brier(base(z[idx]), a[idx]).backward()
        opt.step()
    base.load_state_dict(best["state"])
    base.eval()

    aug = dn.AugmentedCritic(base, len(held), z.shape[1], spec).to(data.dev)      # starts as q0 exactly
    opt = torch.optim.Adam(aug.parameters(), lr=dn.TRAIN["lr_critic"])
    val_px = data.pixels(rows[val], held)
    best_aug = {"loss": math.inf, "state": None, "step": 0}
    for step in range(CHECK["augmented_steps"] + 1):
        if step % CHECK["eval_every"] == 0:
            aug.eval()
            with torch.no_grad():
                loss = float(dn.brier(aug(z[val_t], val_px), a[val_t]))
            if loss < best_aug["loss"] - 1e-7:
                best_aug = {"loss": loss, "state": _state(aug), "step": step}
            aug.train()
        if step == CHECK["augmented_steps"]:
            break
        pick = fit[rng.integers(0, len(fit), CHECK["batch"])]
        idx = torch.as_tensor(pick, device=data.dev)
        opt.zero_grad(set_to_none=True)
        dn.brier(aug(z[idx], data.pixels(rows[pick], held)), a[idx]).backward()
        opt.step()
    aug.load_state_dict(best_aug["state"])
    aug.eval()
    fit_t = torch.as_tensor(np.sort(fit), device=data.dev)
    with torch.no_grad():
        risks = {"base_on_its_fit_rows": float(dn.brier(base(z[fit_t]), a[fit_t])),
                 "augmented_on_its_fit_rows": float(dn.brier(aug(z[fit_t], data.pixels(rows[np.sort(fit)], held)), a[fit_t])),
                 "base_on_its_validation_rows": best["loss"], "augmented_on_its_validation_rows": best_aug["loss"]}
    return base, aug, {"base_step": best["step"], "augmented_step": best_aug["step"], "risks": risks}


def check(rep: dn.Representation, data: dn.Batches, rows: np.ndarray, split, seed: int,
          spec: dict | None = None, return_rows: bool = False) -> dict:
    """The sealed screen's statistic on the check fold, with fresh critics fitted on one half and scored on the
    other, in both directions."""
    held, kept = db.image_columns(split)
    rows = np.asarray(rows)
    z_all = dn.representation_values(rep, data, rows, kept, grad=False)
    return check_from_z(z_all, data, rows, split, seed, spec, return_rows)


def check_from_z(z_all: torch.Tensor, data: dn.Batches, rows: np.ndarray, split, seed: int,
                 spec: dict | None = None, return_rows: bool = False) -> dict:
    """The same check on a representation handed over as numbers.  What the augmented critic sees of the
    held-out images is unchanged, so a fixed candidate is judged by exactly the device the learner faces."""
    held, _ = db.image_columns(split)
    rows = np.asarray(rows)
    a_all = data.a[torch.as_tensor(rows, device=data.dev)]
    half = np.arange(len(rows)) % 2 == 0
    d = np.empty(len(rows))
    q0_all = np.empty(len(rows))
    loss0 = np.empty(len(rows))
    stopped = []
    for direction, (fit_mask, eval_mask) in enumerate(((half, ~half), (~half, half))):
        fit_idx, eval_idx = np.flatnonzero(fit_mask), np.flatnonzero(eval_mask)
        fit_t = torch.as_tensor(fit_idx, device=data.dev)
        base, aug, info = _fit_fresh_critics(z_all[fit_t], a_all[fit_t], data, rows[fit_idx], held,
                                             seed + direction, spec)
        stopped.append(info)
        with torch.no_grad():
            for s in range(0, len(eval_idx), 1024):
                part = eval_idx[s:s + 1024]
                part_t = torch.as_tensor(part, device=data.dev)
                q0 = base(z_all[part_t])
                q1 = aug(z_all[part_t], data.pixels(rows[part], held))
                a = a_all[part_t]
                d[part] = ((q0 - a) ** 2 - (q1 - a) ** 2).cpu().numpy()
                q0_all[part] = q0.cpu().numpy()
                loss0[part] = ((q0 - a) ** 2).cpu().numpy()
    gap, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    z_crit = float(norm.ppf(1.0 - ALPHA / N_SPLITS))
    out = {"gap": gap, "se": se, "upper": max(gap, 0.0) + z_crit * se,
           "risk_base": float(loss0.mean()), "risk_augmented": float(loss0.mean() - gap),
           "overlap": bool(np.all((q0_all >= CLIP[0]) & (q0_all <= CLIP[1]))),
           "early_stopping": stopped, "finite": bool(np.isfinite(d).all())}
    if return_rows:
        out["rows_d"] = d
    return out


def fresh_gap(rep: dn.Representation, data: dn.Batches, fit_rows: np.ndarray, eval_rows: np.ndarray, split,
              seed: int, spec: dict | None = None) -> dict:
    """Fresh critics fitted on `fit_rows`, scored on `eval_rows`: one direction of the check, on rows of the
    caller's choosing.  Used to choose a checkpoint with critics the representation was never trained against."""
    held, kept = db.image_columns(split)
    fit_rows, eval_rows = np.asarray(fit_rows), np.asarray(eval_rows)
    z_fit = dn.representation_values(rep, data, fit_rows, kept, grad=False)
    a_fit = data.a[torch.as_tensor(fit_rows, device=data.dev)]
    base, aug, info = _fit_fresh_critics(z_fit, a_fit, data, fit_rows, held, seed, spec)
    z_eval = dn.representation_values(rep, data, eval_rows, kept, grad=False)
    a_eval = data.a[torch.as_tensor(eval_rows, device=data.dev)]
    d, l0 = [], []
    with torch.no_grad():
        for s in range(0, len(eval_rows), 1024):
            sl = slice(s, s + 1024)
            q0 = base(z_eval[sl])
            q1 = aug(z_eval[sl], data.pixels(eval_rows[sl], held))
            d.append(((q0 - a_eval[sl]) ** 2 - (q1 - a_eval[sl]) ** 2).cpu().numpy())
            l0.append(((q0 - a_eval[sl]) ** 2).cpu().numpy())
    d, l0 = np.concatenate(d), np.concatenate(l0)
    gap = float(d.mean())
    return {"gap": gap, "se": float(d.std(ddof=1) / math.sqrt(len(d))), "clipped_gap": max(gap, 0.0),
            "risk_base": float(l0.mean()), "risk_augmented": float(l0.mean() - gap), "critics": info}


def select_with_fresh_critics(states: dict, data: dn.Batches, rows: dict, split, x_dim: int, seed: int,
                              spec: dict | None = None) -> dict:
    """Choose a checkpoint by the gap FRESH critics find on the `select` fold.

    The training critics cannot be trusted with this choice: one that has memorised its fitting rows predicts
    worse than the base critic on any other rows, the gap there is negative, the clip turns it into a zero, and
    that zero wins.  Fresh critics are fitted on the representation fold with early stopping and scored on
    `select`, which no gradient and no critic has touched.  No true effect and no latent variable is read here.
    """
    _, kept = db.image_columns(split)
    table = {}
    for iteration, state in states.items():
        rep = dn.load_representation(state, kept, x_dim, data.dev, spec)
        table[iteration] = fresh_gap(rep, data, rows["represent"], rows["select"], split, seed, spec)
    chosen = min(sorted(table), key=lambda k: (table[k]["clipped_gap"], k))          # ties go to the earliest
    return {"chosen": chosen, "table": table}


# ----------------------------------------------------------------------------- the effect estimate

class _Mlp(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        h1, h2 = NUISANCE["hidden"]
        self.net = nn.Sequential(nn.Linear(in_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f).squeeze(1)


def _fit_mlp(f: np.ndarray, target: np.ndarray, binary: bool, seed: int, dev) -> _Mlp:
    """A small network on a handful of features, stopped early on a fifth of its rows."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    order = rng.permutation(len(f))
    n_val = max(1, int(round(NUISANCE["val_fraction"] * len(f))))
    val, fit = order[:n_val], order[n_val:]
    ft = torch.as_tensor(f, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(target, dtype=torch.float32, device=dev)
    net = _Mlp(f.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=NUISANCE["lr"], weight_decay=NUISANCE["weight_decay"])
    loss_fn = nn.functional.binary_cross_entropy_with_logits if binary else nn.functional.mse_loss
    best, bad = {"loss": math.inf, "state": None}, 0
    for _ in range(NUISANCE["epochs"]):
        net.train()
        perm = rng.permutation(fit)
        for s in range(0, len(perm), NUISANCE["batch"]):
            idx = torch.as_tensor(perm[s:s + NUISANCE["batch"]], device=dev)
            opt.zero_grad(set_to_none=True)
            loss_fn(net(ft[idx]), yt[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(ft[torch.as_tensor(val, device=dev)]), yt[torch.as_tensor(val, device=dev)]))
        if v < best["loss"] - 1e-6:
            best, bad = {"loss": v, "state": _state(net)}, 0
        else:
            bad += 1
            if bad >= NUISANCE["patience"]:
                break
    net.load_state_dict(best["state"])
    return net.eval()


def aipw_from_features(f_n: np.ndarray, a_n: np.ndarray, y_n: np.ndarray, f_e: np.ndarray, a_e: np.ndarray,
                       y_e: np.ndarray, seed: int, dev) -> dict:
    """Honest AIPW: flexible nuisances fitted on the nuisance fold, scores on the evaluation fold.

    Flexible, because a CNN's two numbers are an arbitrary warp of whatever they encode, and balance does not
    make E[Y | Z, A] linear in them.  The sealed linear AIPW on the same features is recorded next to it as a
    reference; the primary estimate was fixed to be this one before any data set was opened.
    """
    mu, sd = f_n.mean(0), f_n.std(0) + 1e-12
    fn, fe = (f_n - mu) / sd, (f_e - mu) / sd
    y_mu, y_sd = float(y_n.mean()), float(y_n.std() + 1e-12)
    e_net = _fit_mlp(fn, a_n, True, seed, dev)
    m1_net = _fit_mlp(fn[a_n == 1], (y_n[a_n == 1] - y_mu) / y_sd, False, seed + 1, dev)
    m0_net = _fit_mlp(fn[a_n == 0], (y_n[a_n == 0] - y_mu) / y_sd, False, seed + 2, dev)
    with torch.no_grad():
        fe_t = torch.as_tensor(fe, dtype=torch.float32, device=dev)
        e_raw = torch.sigmoid(e_net(fe_t)).cpu().numpy().astype(np.float64)
        m1 = m1_net(fe_t).cpu().numpy().astype(np.float64) * y_sd + y_mu
        m0 = m0_net(fe_t).cpu().numpy().astype(np.float64) * y_sd + y_mu
    e = np.clip(e_raw, *CLIP)
    psi = m1 - m0 + a_e * (y_e - m1) / e - (1 - a_e) * (y_e - m0) / (1 - e)
    ones_n, ones_e = np.ones((len(fn), 1)), np.ones((len(fe), 1))
    linear = pr.aipw(np.column_stack([ones_n, fn]), a_n, y_n, np.column_stack([ones_e, fe]), a_e, y_e)
    return {"theta": float(psi.mean()), "scores": psi, "theta_sealed_linear": float(linear.mean()),
            "outside_clip_before": float(((e_raw < CLIP[0]) | (e_raw > CLIP[1])).mean()),
            "finite": bool(np.isfinite(psi).all())}


def estimate(rep: dn.Representation, data: dn.Batches, a: np.ndarray, y: np.ndarray, rows: dict,
             split, seed: int) -> dict:
    _, kept = db.image_columns(split)
    feats = {}
    for role in ("nuisance", "evaluate"):                     # Z = (X, h) exactly as the critics saw it
        feats[role] = dn.representation_values(rep, data, rows[role], kept, grad=False).cpu().numpy().astype(np.float64)
    rn, re = rows["nuisance"], rows["evaluate"]
    return aipw_from_features(feats["nuisance"], a[rn], y[rn], feats["evaluate"], a[re], y[re], seed, data.dev)


# ----------------------------------------------------------------------------- aggregation

def combine(names: list[str], splits: list, estimates: list[float], scores: np.ndarray, passed: list[bool],
            seed: int) -> dict:
    """Algorithm 1's aggregation by the sealed functions.  `scores` is (evaluation rows, candidates)."""
    keep = [i for i, ok in enumerate(passed) if ok and estimates[i] is not None and np.isfinite(estimates[i])]
    if not keep:
        return {"return_kind": "no_return", "estimate": None, "retained": [], "rho": None,
                "component_sizes": [], "members": []}
    rho = float(ps.common_radius(scores[:, keep], seed))
    agg = ps.aggregate([splits[i] for i in keep], np.array([estimates[i] for i in keep]), rho)
    members = (["S" + "".join(str(j + 1) for j in m) for m in agg["members"][0]]
               if agg["return_kind"] == "unique_largest" else [])
    return {"return_kind": agg["return_kind"], "estimate": agg["output"], "retained": [names[i] for i in keep],
            "rho": rho, "component_sizes": agg.get("component_sizes", []), "members": members}


# ----------------------------------------------------------------------------- comparators

class RawNet(nn.Module):
    """Every pixel of all six images, and X, into propensity and outcome heads.  Same towers as the method."""

    def __init__(self, x_dim: int, spec: dict | None) -> None:
        super().__init__()
        self.towers = nn.ModuleList([dn.ImageTower(spec, RAW_CNN["feature_dim"]) for _ in range(db.N_IMAGES)])
        h1, h2 = RAW_CNN["hidden"]
        self.body = nn.Sequential(nn.Linear(db.N_IMAGES * RAW_CNN["feature_dim"] + x_dim, h1), nn.ReLU(),
                                  nn.Linear(h1, h2), nn.ReLU())
        self.e, self.m0, self.m1 = nn.Linear(h2, 1), nn.Linear(h2, 1), nn.Linear(h2, 1)

    def forward(self, pixels: list[torch.Tensor], x: torch.Tensor):
        h = self.body(torch.cat([t(p) for t, p in zip(self.towers, pixels)] + [x], dim=1))
        return self.e(h).squeeze(1), self.m0(h).squeeze(1), self.m1(h).squeeze(1)


def raw_cnn(data: dn.Batches, x: np.ndarray, a: np.ndarray, y: np.ndarray, rows: dict, seed: int,
            spec: dict | None = None) -> dict:
    """The strong baseline: adjust for whatever an end-to-end network extracts from all six images."""
    dev = data.dev
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    rn, re = rows["nuisance"], rows["evaluate"]
    order = rng.permutation(rn)
    n_val = int(round(RAW_CNN["val_fraction"] * len(rn)))
    val, fit = np.sort(order[:n_val]), order[n_val:]
    mu_x, sd_x = x[fit].mean(0), x[fit].std(0) + 1e-12
    y_mu, y_sd = float(y[fit].mean()), float(y[fit].std() + 1e-12)
    xt = torch.as_tensor((x - mu_x) / sd_x, dtype=torch.float32, device=dev)
    at = torch.as_tensor(a, dtype=torch.float32, device=dev)
    yt = torch.as_tensor((y - y_mu) / y_sd, dtype=torch.float32, device=dev)
    cols = np.arange(db.N_IMAGES)
    net = RawNet(x.shape[1], spec).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=RAW_CNN["lr"])

    def loss_on(r: np.ndarray) -> torch.Tensor:
        idx = torch.as_tensor(r, device=dev)
        logit, m0, m1 = net(data.pixels(r, cols), xt[idx])
        fitted = torch.where(at[idx] > 0.5, m1, m0)
        return (nn.functional.binary_cross_entropy_with_logits(logit, at[idx])
                + ((yt[idx] - fitted) ** 2).mean())

    best, bad, history = {"loss": math.inf, "state": None, "epoch": 0}, 0, []
    for epoch in range(1, RAW_CNN["epochs"] + 1):
        net.train()
        perm = rng.permutation(fit)
        for s in range(0, len(perm), RAW_CNN["batch"]):
            opt.zero_grad(set_to_none=True)
            loss_on(np.sort(perm[s:s + RAW_CNN["batch"]])).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = sum(float(loss_on(val[s:s + RAW_CNN["eval_rows"]])) * len(val[s:s + RAW_CNN["eval_rows"]])
                    for s in range(0, len(val), RAW_CNN["eval_rows"])) / len(val)
        history.append(round(v, 6))
        if v < best["loss"] - 1e-5:
            best, bad = {"loss": v, "state": _state(net), "epoch": epoch}, 0
        else:
            bad += 1
            if bad >= RAW_CNN["patience"]:
                break
    net.load_state_dict(best["state"])
    net.eval()
    e_raw, m0, m1 = [], [], []
    with torch.no_grad():
        for s in range(0, len(re), RAW_CNN["eval_rows"]):
            r = re[s:s + RAW_CNN["eval_rows"]]
            logit, b0, b1 = net(data.pixels(r, cols), xt[torch.as_tensor(r, device=dev)])
            e_raw.append(torch.sigmoid(logit)); m0.append(b0); m1.append(b1)
    e_raw, m0, m1 = [torch.cat(v).cpu().numpy().astype(np.float64) for v in (e_raw, m0, m1)]
    m0, m1 = m0 * y_sd + y_mu, m1 * y_sd + y_mu
    e = np.clip(e_raw, *CLIP)
    a_e, y_e = a[re], y[re]
    psi = m1 - m0 + a_e * (y_e - m1) / e - (1 - a_e) * (y_e - m0) / (1 - e)
    return {"theta": float(psi.mean()) if np.isfinite(psi).all() else None, "finite": bool(np.isfinite(psi).all()),
            "epoch": best["epoch"], "val_history": history,
            "outside_clip_before": float(((e_raw < CLIP[0]) | (e_raw > CLIP[1])).mean())}


def feature_comparators(data_full: dict, x: np.ndarray, a: np.ndarray, y: np.ndarray, rows: dict, seed: int,
                        dev) -> dict:
    """X only and the oracle, with the same nuisance structure and the same two folds as the method."""
    rn, re = rows["nuisance"], rows["evaluate"]
    oracle = np.column_stack([data_full["X1_eval_only"], data_full["C_eval_only"], data_full["U_eval_only"]])
    out = {}
    for name, f in (("X", x), ("oracle", oracle)):
        got = aipw_from_features(f[rn], a[rn], y[rn], f[re], a[re], y[re], seed, dev)
        out[name] = {"theta": got["theta"], "theta_sealed_linear": got["theta_sealed_linear"]}
    out["naive"] = {"theta": float(y[a == 1].mean() - y[a == 0].mean())}
    return out
