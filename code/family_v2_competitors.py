"""Comparators for SCM family v2 (PRD section 7).  All read the learner view except the oracle.

    naive          difference in means
    X              honest AIPW on X, same folds and nuisance family as PROBE
    raw            honest AIPW on (X, every coordinate of W), ridge chosen by cross-validation inside N
    summary        honest AIPW on (X, top principal components of every block): a learned summary of all of
                   W with no held-out block and no balance check
    oracle         honest AIPW on (X_1, C, U); evaluator baseline
    p2sls          linear proximal two-stage least squares with the roles given by the analyst
                   (Tchetgen Tchetgen et al. 2024): treatment proxy Z_tr, outcome proxy W_out, covariates X
    park_spc       single proxy control with the linear outcome bridge of Park, Richardson and Tchetgen
                   Tchetgen (2024, Result 3), covariates X; the target is the ETT, which equals the ATE here
                   because the effect is constant
    cevae          causal effect VAE (Louizos et al. 2017), reimplemented in PyTorch with the architecture and
                   default hyperparameters of the Pyro reference implementation (pyro.contrib.cevae)

Every proxy block with more than P coordinates enters p2sls and park_spc through its top principal
components, fitted without A or Y.
"""
from __future__ import annotations

import math

import numpy as np

import family_v2_dgp as g
import family_v2_probe as pr

RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


# ----------------------------------------------------------------------------- AIPW baselines

def _cv_ridge_ols(f: np.ndarray, y: np.ndarray, seed: int) -> float:
    idx = np.random.default_rng(seed).permutation(len(y))
    parts = np.array_split(idx, 3)
    best, best_lam = math.inf, RIDGE_GRID[0]
    for lam in RIDGE_GRID:
        err = 0.0
        for k in range(3):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(3) if j != k])
            beta = pr.ridge_ols(f[tr], y[tr], lam)
            err += float(np.sum((y[te] - f[te] @ beta) ** 2))
        if err < best:
            best, best_lam = err, lam
    return best_lam


def _cv_critic(f: np.ndarray, a: np.ndarray, seed: int) -> float:
    idx = np.random.default_rng(seed).permutation(len(a))
    parts = np.array_split(idx, 3)
    best, best_lam = math.inf, RIDGE_GRID[0]
    for lam in RIDGE_GRID:
        err = 0.0
        for k in range(3):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(3) if j != k])
            c, _ = pr.fit_critic(f[tr], a[tr], None, lam)
            err += float(np.sum((a[te] - pr.link(f[te] @ c)) ** 2))
        if err < best:
            best, best_lam = err, lam
    return best_lam


def aipw_cv(f_n, a_n, y_n, f_e, a_e, y_e, seed: int) -> np.ndarray:
    lam_e = _cv_critic(f_n, a_n, seed)
    c, _ = pr.fit_critic(f_n, a_n, None, lam_e)
    e = np.clip(pr.link(f_e @ c), *pr.CLIP)
    out = {}
    for arm in (0, 1):
        rows = a_n == arm
        lam = _cv_ridge_ols(f_n[rows], y_n[rows], seed + 1 + arm)
        out[arm] = f_e @ pr.ridge_ols(f_n[rows], y_n[rows], lam)
    return out[1] - out[0] + a_e * (y_e - out[1]) / e - (1 - a_e) * (y_e - out[0]) / (1 - e)


def aipw_baselines(data: dict[str, np.ndarray], seed: int) -> dict[str, float]:
    """X, raw, summary and oracle on the same four rotations as PROBE (nuisance on N, scores on E)."""
    view = g.learner_view(data)
    n = len(view["A"])
    fold_rows = pr.folds(n, seed)
    oracle_cols = np.column_stack([data["X1_eval_only"], data["C_eval_only"], data["U_eval_only"]])
    sums = {k: 0.0 for k in ("X", "raw", "summary", "oracle")}
    for rot in range(4):
        rl = pr.roles(rot)
        rn, re = fold_rows[rl["N"]], fold_rows[rl["E"]]
        prep = pr.Prep(view, rn)
        ones_n, ones_e = np.ones((len(rn), 1)), np.ones((len(re), 1))
        w_all = lambda rows: np.column_stack([view[f"W{j + 1}"][rows] for j in range(g.N_BLOCKS)])
        feats = {
            "X": (np.column_stack([ones_n, prep.x(view, rn)]), np.column_stack([ones_e, prep.x(view, re)])),
            "raw": (np.column_stack([ones_n, view["X"][rn], w_all(rn)]), np.column_stack([ones_e, view["X"][re], w_all(re)])),
            "summary": (np.column_stack([ones_n, prep.x(view, rn)] + [prep.block(view, rn, j) for j in range(g.N_BLOCKS)]),
                        np.column_stack([ones_e, prep.x(view, re)] + [prep.block(view, re, j) for j in range(g.N_BLOCKS)])),
            "oracle": (np.column_stack([ones_n, oracle_cols[rn]]), np.column_stack([ones_e, oracle_cols[re]])),
        }
        for name, (fn, fe) in feats.items():
            if name == "raw":                              # standardise raw columns on N for a fair ridge
                mu, sd = fn[:, 1:].mean(0), fn[:, 1:].std(0) + 1e-12
                fn = np.column_stack([ones_n, (fn[:, 1:] - mu) / sd])
                fe = np.column_stack([ones_e, (fe[:, 1:] - mu) / sd])
            psi = aipw_cv(fn, view["A"][rn], view["Y"][rn], fe, view["A"][re], view["Y"][re], seed + 97 * rot)
            sums[name] += float(psi.mean()) / 4.0
    a, y = view["A"], view["Y"]
    sums["naive"] = float(y[a == 1].mean() - y[a == 0].mean())
    return sums


# ----------------------------------------------------------------------------- proximal and single-proxy control

def _pcs(w: np.ndarray, p: int = pr.P_FEATURES) -> np.ndarray:
    wc = w - w.mean(0)
    if w.shape[1] <= p:
        return wc / (wc.std(0) + 1e-12)
    _, _, vt = np.linalg.svd(wc, full_matrices=False)
    s = wc @ vt[:p].T
    return s / (s.std(0) + 1e-12)


def p2sls(view: dict[str, np.ndarray], z_tr: str, w_out: str) -> float:
    """Linear proximal 2SLS.  Stage 1: W_out on (1, A, Z_tr, X); stage 2: Y on (1, A, fitted W_out, X)."""
    a, y = view["A"], view["Y"]
    x = (view["X"] - view["X"].mean(0)) / (view["X"].std(0) + 1e-12)
    zt, wo = _pcs(view[z_tr]), _pcs(view[w_out])
    if zt.shape[1] < wo.shape[1]:
        wo = wo[:, :zt.shape[1]]
    ones = np.ones((len(a), 1))
    s1 = np.column_stack([ones, a, zt, x])
    w_hat = s1 @ np.linalg.lstsq(s1, wo, rcond=None)[0]
    s2 = np.column_stack([ones, a, w_hat, x])
    return float(np.linalg.lstsq(s2, y, rcond=None)[0][1])


def park_spc(view: dict[str, np.ndarray], block: str) -> float:
    """ETT = E(Y | A=1) - E{b(W, X) | A=1}, b linear, from the control-arm moments E[(1, Y, X)(Y - b) | A=0] = 0."""
    a, y = view["A"], view["Y"]
    x = (view["X"] - view["X"].mean(0)) / (view["X"].std(0) + 1e-12)
    w = _pcs(view[block], 1)[:, 0]
    ctrl, trt = a == 0, a == 1
    inst = np.column_stack([np.ones(ctrl.sum()), y[ctrl], x[ctrl]])
    reg = np.column_stack([np.ones(ctrl.sum()), w[ctrl], x[ctrl]])
    eta = np.linalg.solve(inst.T @ reg, inst.T @ y[ctrl])
    b_trt = np.column_stack([np.ones(trt.sum()), w[trt], x[trt]]) @ eta
    return float(y[trt].mean() - b_trt.mean())


def proximal_rows(view: dict[str, np.ndarray]) -> dict[str, float]:
    return {"p2sls_given_roles": p2sls(view, "W4", "W2"),
            "p2sls_swapped_roles": p2sls(view, "W2", "W4"),
            "park_spc_clean_proxy": park_spc(view, "W1"),
            "park_spc_contaminated_proxy": park_spc(view, "W4")}


# ----------------------------------------------------------------------------- CEVAE

CEVAE_DEFAULTS = {"latent_dim": 20, "hidden_dim": 200, "num_layers": 3, "num_epochs": 30, "batch_size": 100,
                  "learning_rate": 1e-3, "learning_rate_decay": 0.1, "weight_decay": 1e-4, "num_samples": 100}


def cevae_ate(view: dict[str, np.ndarray], seed: int, **overrides) -> dict:
    """CEVAE on (x = standardised X and W, t = A, y = standardised Y); returns the in-sample ATE on the Y scale.

    Architecture and defaults follow pyro.contrib.cevae: ELU networks with num_layers hidden layers of
    hidden_dim units; model p(z) N(0, I), p(x | z) diagonal normal, p(t | z) Bernoulli, p(y | t, z) normal with
    one head per arm; guide q(t | x), q(y | t, x) with a shared trunk, q(z | t, y, x) with one head per arm;
    the loss is the negative ELBO plus the log-likelihood of the observed t and y under the guide; ClippedAdam-
    style exponential learning-rate decay to learning_rate_decay over training; the ITE averages
    E(y | t=1, z) - E(y | t=0, z) over num_samples draws of z from the guide with t and y also drawn from it."""
    import torch
    from torch import nn

    cfg = dict(CEVAE_DEFAULTS, **overrides)
    torch.manual_seed(seed)
    x_np = np.column_stack([view["X"]] + [view[f"W{j + 1}"] for j in range(g.N_BLOCKS)])
    x_np = (x_np - x_np.mean(0)) / (x_np.std(0) + 1e-12)
    y_mu, y_sd = float(view["Y"].mean()), float(view["Y"].std())
    x = torch.tensor(x_np, dtype=torch.float32)
    t = torch.tensor(view["A"], dtype=torch.float32)
    y = torch.tensor((view["Y"] - y_mu) / y_sd, dtype=torch.float32)
    d, h, L, k = x.shape[1], cfg["hidden_dim"], cfg["num_layers"], cfg["latent_dim"]

    def mlp(inp, out):
        layers, width = [], inp
        for _ in range(L):
            layers += [nn.Linear(width, h), nn.ELU()]
            width = h
        layers.append(nn.Linear(width, out))
        return nn.Sequential(*layers)

    class DiagNormal(nn.Module):
        def __init__(self, inp, out):
            super().__init__()
            self.net = mlp(inp, 2 * out)
            self.out = out

        def forward(self, v):
            o = self.net(v)
            return o[..., :self.out], nn.functional.softplus(o[..., self.out:]).clamp(1e-3, 1e6)

    x_nn, t_nn = DiagNormal(k, d), mlp(k, 1)
    y0_nn, y1_nn = DiagNormal(k, 1), DiagNormal(k, 1)
    gt_nn = mlp(d, 1)
    gy_trunk = nn.Sequential(*list(mlp(d, h).children())[:-1])     # hidden layers only
    gy0, gy1 = DiagNormal(h, 1), DiagNormal(h, 1)
    gz0, gz1 = DiagNormal(d + 1, k), DiagNormal(d + 1, k)
    modules = nn.ModuleList([x_nn, t_nn, y0_nn, y1_nn, gt_nn, gy_trunk, gy0, gy1, gz0, gz1])
    steps = cfg["num_epochs"] * math.ceil(len(t) / cfg["batch_size"])
    opt = torch.optim.Adam(modules.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg["learning_rate_decay"] ** (1.0 / steps))
    normal = torch.distributions.Normal
    bern = lambda logits: torch.distributions.Bernoulli(logits=logits)

    def pick(t_, a0, a1):
        return tuple(t_[:, None] * v1 + (1 - t_[:, None]) * v0 for v0, v1 in zip(a0, a1))

    gen = torch.Generator().manual_seed(seed + 1)
    losses = []
    for epoch in range(cfg["num_epochs"]):
        perm = torch.randperm(len(t), generator=gen)
        for s in range(0, len(t), cfg["batch_size"]):
            idx = perm[s:s + cfg["batch_size"]]
            xb, tb, yb = x[idx], t[idx], y[idx]
            hy = gy_trunk(xb)
            qy_loc, qy_scale = pick(tb, gy0(hy), gy1(hy))
            zin = torch.cat([xb, yb[:, None]], 1)
            qz_loc, qz_scale = pick(tb, gz0(zin), gz1(zin))
            z = qz_loc + qz_scale * torch.randn_like(qz_loc)
            px_loc, px_scale = x_nn(z)
            py_loc, py_scale = pick(tb, y0_nn(z), y1_nn(z))
            log_lik = (normal(px_loc, px_scale).log_prob(xb).sum(1)
                       + bern(t_nn(z)[:, 0]).log_prob(tb)
                       + normal(py_loc[:, 0], py_scale[:, 0]).log_prob(yb))
            kl = torch.distributions.kl_divergence(normal(qz_loc, qz_scale), normal(0.0, 1.0)).sum(1)
            aux = bern(gt_nn(xb)[:, 0]).log_prob(tb) + normal(qy_loc[:, 0], qy_scale[:, 0]).log_prob(yb)
            loss = -(log_lik - kl + aux).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        ite = torch.zeros(len(t))
        for _ in range(cfg["num_samples"]):
            t_s = bern(gt_nn(x)[:, 0]).sample()
            hy = gy_trunk(x)
            qy_loc, qy_scale = pick(t_s, gy0(hy), gy1(hy))
            y_s = qy_loc[:, 0] + qy_scale[:, 0] * torch.randn(len(t))
            qz_loc, qz_scale = pick(t_s, gz0(torch.cat([x, y_s[:, None]], 1)), gz1(torch.cat([x, y_s[:, None]], 1)))
            z = qz_loc + qz_scale * torch.randn_like(qz_loc)
            ite += (y1_nn(z)[0][:, 0] - y0_nn(z)[0][:, 0]) / cfg["num_samples"]
    return {"ate": float(ite.mean()) * y_sd, "final_loss": losses[-1], "config": cfg}
