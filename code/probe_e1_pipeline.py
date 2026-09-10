#!/usr/bin/env python3
"""E1: theory-faithful PROBE pipeline.

Implements the manuscript's Section 4 procedure end to end:
  * three-way split (discrepancy-learning, nuisance-learning, evaluation samples);
  * encoder phi_S trained to minimize the Brier-risk residual discrepancy
    D~^2 = R0(h0) - R1(h1) with two critics h0(Z) and h1(T, Z);
  * screening by empirical discrepancy and representation overlap;
  * honest AIPW estimate per retained proxy split;
  * Algorithm 1 aggregation: link estimates within 2*rho, return the median of
    the largest linked group (a candidate set on ties).
Baselines: AIPW on X only, on (X, W), and on the oracle (X, U); mean and plain
median over retained splits.  Oracle diagnostics (population candidate value of
each learned representation, population residual discrepancy) use a fresh
large sample and are labelled as oracle quantities.

SCM family (see --scm): s1 linear-proxy SCM1, s2 nonlinear SCM2 (both reuse the
structural equations of honest_canonical_representation_experiment.py), s6 =
s1 with block 0 affecting treatment directly (held-out sets containing block 0
are invalid), s7 = s1 with block 3 affecting the outcome directly (all splits
valid; a proximal outcome proxy would be invalid).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

ROOT_SEED = 190909
TRUE_ATE = 1.0
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

NAMESPACE = {"coef": 1, "data": 2, "diag": 3, "split": 4, "sample_splits": 5, "encoder": 6, "critic": 7, "nuisance": 8}


# ----------------------------------------------------------------------------- utilities
def seed_key(ns: str, *parts: int) -> tuple[int, ...]:
    return (ROOT_SEED, NAMESPACE[ns], *parts)


def rng(*key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(key))


def torch_seed(*key: int) -> int:
    return int(np.random.SeedSequence(key).generate_state(1, dtype=np.uint32)[0])


def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def unit_vector(g: np.random.Generator, d: int) -> np.ndarray:
    v = g.standard_normal(d)
    return v / np.linalg.norm(v)


def all_splits(n_blocks: int, max_held: int | None = None) -> list[tuple[int, ...]]:
    """Split family: held-out sets of 1..max_held blocks (default: at most half of the blocks, so that the
    remaining blocks keep a majority of the proxy information for the representation)."""
    top = max_held if max_held is not None else max(1, n_blocks // 2)
    out = []
    for r in range(1, min(top, n_blocks - 1) + 1):
        out.extend(itertools.combinations(range(n_blocks), r))
    return out


# ----------------------------------------------------------------------------- SCMs (see probe_scms.py)
import probe_scms as scms


def scm_params(args) -> dict:
    p = {}
    if args.proxy_strength is not None: p["proxy_strength"] = args.proxy_strength
    if args.n_blocks is not None: p["n_blocks"] = args.n_blocks
    if getattr(args, "d_x", None) is not None: p["d_x"] = args.d_x
    if getattr(args, "d_w", None) is not None: p["d_w"] = args.d_w
    for kv in (args.scm_param or []):
        k, v = kv.split("=", 1)
        try:
            p[k] = int(v)
        except ValueError:
            try:
                p[k] = float(v)
            except ValueError:
                p[k] = v
    return p


def coefficients(scm: str, params: dict) -> dict[str, np.ndarray]:
    return scms.coefficients(scm, seed_key("coef"), **params)


def generate(n: int, key: tuple[int, ...], coef: dict[str, np.ndarray], scm: str, params: dict) -> dict[str, np.ndarray]:
    return scms.generate(scm, n, key, coef, **params)


# ----------------------------------------------------------------------------- encoder and critics
class MLP(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


class Encoder(torch.nn.Module):
    """Z = phi(X, W_{-S}) in R^{d_z}; with keep_x, Z = (X, psi(X, W_{-S}))."""

    def __init__(self, d_x: int, d_v: int, d_z: int, keep_x: bool, hidden: int = 32):
        super().__init__()
        self.d_x, self.keep_x = d_x, keep_x
        self.psi = MLP(d_v, d_z, hidden)

    def forward(self, v):
        z = self.psi(v)
        return torch.cat([v[:, : self.d_x], z], 1) if self.keep_x else z


class RFF:
    """Random Fourier features of a standardized input; standardization statistics are taken from the fitting
    sample and detached, so fitting and evaluation share one feature map and the encoder cannot exploit it."""

    def __init__(self, d_in: int, m: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.omega = torch.randn(d_in, m, generator=g) / math.sqrt(d_in)
        self.phase = torch.rand(m, generator=g) * 2 * math.pi
        self.m = m

    @staticmethod
    def stats(x: torch.Tensor):
        with torch.no_grad():
            return x.mean(0, keepdim=True), x.std(0, keepdim=True) + 1e-3

    def __call__(self, x: torch.Tensor, st) -> torch.Tensor:
        return math.sqrt(2.0 / self.m) * torch.cos(((x - st[0]) / st[1]) @ self.omega + self.phase)


class NestedCritics:
    """Two ridge critics on random Fourier features.  h0 uses features of Z; h1 uses the same Z features plus
    features of (T, Z), so the class of h1 contains the class of h0 and the in-sample Brier gap R0 - R1 is
    nonnegative.  A control critic h1c has exactly the geometry of h1 but sees T with its held-out-block
    columns replaced by independent Gaussian noise (the X columns of T are kept); the gap R0 - R1c is the
    optimism that comes from feature count, shrinkage, and the duplicated X features rather than from the
    held-out block, and serves as the floor of the screen.  All minimizers are exact and closed-form."""

    def __init__(self, d_z: int, d_t: int, m: int, seed: int, lam: float = 1e-2, d_x: int = 0):
        self.fz, self.ftz, self.lam, self.d_x = RFF(d_z, m, seed), RFF(d_t + d_z, m, seed + 1), lam, d_x
        self.noise_seed = seed + 3

    def control_t(self, t: torch.Tensor) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.noise_seed + t.shape[0])
        noise = torch.randn(t.shape[0], t.shape[1] - self.d_x, generator=g)
        return torch.cat([t[:, : self.d_x], noise], 1)

    def design(self, z, t, st):
        pz = self.fz(z, st[0])
        return pz, torch.cat([pz, self.ftz(torch.cat([t, z], 1), st[1])], 1), torch.cat([pz, self.ftz(torch.cat([self.control_t(t), z], 1), st[1])], 1)

    def _solve(self, phi, a):
        return torch.linalg.solve(phi.T @ phi + self.lam * len(a) * torch.eye(phi.shape[1]), phi.T @ (a - 0.5))

    def fit(self, z, t, a):
        st = (RFF.stats(z), RFF.stats(torch.cat([t, z], 1)))
        with torch.no_grad():
            p0, p1, pc = self.design(z, t, st)
            return self._solve(p0, a), self._solve(p1, a), self._solve(pc, a), st

    def losses(self, z, t, a, w0, w1, wc, st):
        p0, p1, pc = self.design(z, t, st)
        return (a - (0.5 + p0 @ w0)) ** 2, (a - (0.5 + p1 @ w1)) ** 2, (a - (0.5 + pc @ wc)) ** 2


def insample_gap(c: NestedCritics, z, t, a) -> torch.Tensor:
    w0, w1, wc, st = c.fit(z.detach(), t, a)
    l0, l1, lc = c.losses(z, t, a, w0, w1, wc, st)
    return (l0 - l1).mean()


def crossfit_gaps(c: NestedCritics, z, t, a):
    """Two-fold cross-fitted per-unit gaps: (l0 - l1) for the held-out block and (l0 - lc) for the control."""
    n = len(a); half = n // 2; parts, ctrl = [], []
    with torch.no_grad():
        for fit, ev in [(slice(0, half), slice(half, n)), (slice(half, n), slice(0, half))]:
            w0, w1, wc, st = c.fit(z[fit], t[fit], a[fit])
            l0, l1, lc = c.losses(z[ev], t[ev], a[ev], w0, w1, wc, st)
            parts.append((l0 - l1).numpy()); ctrl.append((l0 - lc).numpy())
    return np.concatenate(parts), np.concatenate(ctrl)


def empirical_discrepancy(z: torch.Tensor, t: torch.Tensor, a: torch.Tensor, m_rff: int, seed: int, d_x: int) -> dict[str, float]:
    """In-sample nested gap (nonnegative), cross-fitted gap with its standard error, the control floor, and the
    floor-corrected discrepancy D2 = positive part of (cross-fitted gap - control gap) used for screening."""
    c = NestedCritics(z.shape[1], t.shape[1], m_rff, seed, d_x=d_x)
    with torch.no_grad():
        g_in = insample_gap(c, z, t, a).item()
    diff, ctrl = crossfit_gaps(c, z, t, a)
    corrected = diff - ctrl
    return {"D2_in": max(g_in, 0.0), "gap_in": g_in, "gap_cf": float(diff.mean()), "se": float(corrected.std(ddof=1) / math.sqrt(len(corrected))),
            "null_floor": float(ctrl.mean()), "D2": max(float(corrected.mean()), 0.0)}


def train_encoder(v: np.ndarray, t: np.ndarray, a: np.ndarray, d_x: int, d_z: int, keep_x: bool, steps: int, m_rff: int, seed: int, lr: float = 3e-3, lam: float = 1e-2, val_frac: float = 0.25, eval_every: int = 10, hidden: int = 32, pretrain_steps: int = 0) -> Encoder:
    """Minimize the in-sample nested Brier gap on a training part of the discrepancy sample, with early stopping
    on the held-out gap of a validation part (critics fitted on the training part, losses on the validation
    part).  Critic weights are detached at every step (envelope gradient); a variance floor prevents collapse."""
    import copy
    torch.manual_seed(seed)
    V, T, A = map(lambda x: torch.tensor(x, dtype=torch.float32), (v, t, a))
    n = len(A); g = torch.Generator().manual_seed(seed + 5); perm = torch.randperm(n, generator=g)
    n_tr = int(round((1 - val_frac) * n)); tr, va = perm[:n_tr], perm[n_tr:]
    enc = Encoder(d_x, V.shape[1], d_z, keep_x, hidden)
    d_zfull = (d_x if keep_x else 0) + d_z
    if pretrain_steps > 0:
        # warm start: a treatment-predictive representation (Brier loss of A given phi(V)) on the training part;
        # the head is discarded and the balance objective below takes over.  Y is never used.
        head = torch.nn.Linear(d_zfull, 1)
        opt0 = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=lr)
        for _ in range(pretrain_steps):
            opt0.zero_grad()
            loss0 = ((A[tr] - torch.sigmoid(head(enc(V[tr])).squeeze(1))) ** 2).mean()
            loss0.backward(); opt0.step()
    c = NestedCritics(d_zfull, T.shape[1], m_rff, seed + 11, lam=lam, d_x=d_x)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    best_gap, best_state, best_step = float("inf"), copy.deepcopy(enc.state_dict()), 0

    def validation_gap() -> float:
        with torch.no_grad():
            z = enc(V)
            w0, w1, wc, st = c.fit(z[tr], T[tr], A[tr])
            l0, l1, lc = c.losses(z[va], T[va], A[va], w0, w1, wc, st)
            return float((l0 - l1).mean())

    for step in range(steps):
        if step % eval_every == 0:
            gv = validation_gap()
            if gv < best_gap:
                best_gap, best_state, best_step = gv, copy.deepcopy(enc.state_dict()), step
        opt.zero_grad()
        z = enc(V[tr])
        gap = insample_gap(c, z, T[tr], A[tr])
        floor = torch.relu(0.5 - z.std(0)).sum()
        (gap + floor).backward(); opt.step()
    gv = validation_gap()
    if gv < best_gap:
        best_gap, best_state, best_step = gv, copy.deepcopy(enc.state_dict()), steps
    enc.load_state_dict(best_state); enc.eval()
    enc.best_step, enc.best_val_gap = best_step, best_gap
    return enc


def encode(enc: Encoder, v: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return enc(torch.tensor(v, dtype=torch.float32)).numpy()


# ----------------------------------------------------------------------------- nuisances and AIPW
GBR = dict(n_estimators=150, max_depth=2, learning_rate=0.05, min_samples_leaf=10)


def fit_nuisances(z: np.ndarray, a: np.ndarray, y: np.ndarray, seed: int):
    m1 = GradientBoostingRegressor(random_state=seed, **GBR).fit(z[a == 1], y[a == 1])
    m0 = GradientBoostingRegressor(random_state=seed + 1, **GBR).fit(z[a == 0], y[a == 0])
    e = GradientBoostingClassifier(random_state=seed + 2, n_estimators=150, max_depth=2, learning_rate=0.05, min_samples_leaf=10).fit(z, a)
    return m1, m0, e


def aipw(z: np.ndarray, a: np.ndarray, y: np.ndarray, nuis, eta: float) -> dict[str, float]:
    m1, m0, e = nuis
    mu1, mu0 = m1.predict(z), m0.predict(z)
    eh = np.clip(e.predict_proba(z)[:, 1], eta, 1 - eta)
    raw = e.predict_proba(z)[:, 1]
    uif = mu1 - mu0 + a * (y - mu1) / eh - (1 - a) * (y - mu0) / (1 - eh)
    return {"theta": float(uif.mean()), "sigma": float(uif.std(ddof=1)), "n": int(len(y)),
            "overlap_frac": float(np.mean((raw >= eta) & (raw <= 1 - eta))), "e_min": float(raw.min()), "e_max": float(raw.max())}


# ----------------------------------------------------------------------------- one proxy split
def run_split(data: dict[str, np.ndarray], idx: dict[str, np.ndarray], S: tuple[int, ...], args, rep: int, n_index: int, diag: dict[str, np.ndarray] | None, blk: list[np.ndarray]) -> dict:
    held = np.concatenate([blk[b] for b in S]); rem = np.concatenate([blk[b] for b in range(len(blk)) if b not in S])
    X, W, A, Y = data["X"], data["W"], data["A"], data["Y"]
    V = np.column_stack([X, W[:, rem]]); T = np.column_stack([X, W[:, held]])
    iD, iN, iE = idx["D"], idx["N"], idx["E"]
    mean_v, sd_v = V[iD].mean(0), V[iD].std(0) + 1e-8
    mean_t, sd_t = T[iD].mean(0), T[iD].std(0) + 1e-8
    Vs, Ts = (V - mean_v) / sd_v, (T - mean_t) / sd_t
    sid = int("".join(str(b) for b in S))
    enc = train_encoder(Vs[iD], Ts[iD], A[iD], X.shape[1], args.d_z, args.keep_x, args.enc_steps, args.m_rff, torch_seed(*seed_key("encoder", rep, n_index, sid)), lr=args.lr, lam=args.lam, hidden=getattr(args, "enc_hidden", 32), pretrain_steps=getattr(args, "pretrain_steps", 0))
    Z = encode(enc, Vs)
    zD = torch.tensor(Z[iD], dtype=torch.float32); tD = torch.tensor(Ts[iD], dtype=torch.float32); aD = torch.tensor(A[iD], dtype=torch.float32)
    disc = empirical_discrepancy(zD, tD, aD, args.m_rff, torch_seed(*seed_key("critic", rep, n_index, sid)), X.shape[1])
    nuis = fit_nuisances(Z[iN], A[iN], Y[iN], torch_seed(*seed_key("nuisance", rep, n_index, sid)) % (2 ** 31))
    est = aipw(Z[iE], A[iE], Y[iE], nuis, args.eta)
    rec = {"S": list(S), "D2_hat": disc["D2"], "D2_in": disc["D2_in"], "gap_in": disc["gap_in"], "gap_cf": disc["gap_cf"], "gap_se": disc["se"], "null_floor": disc["null_floor"], "best_step": int(enc.best_step), "best_val_gap": float(enc.best_val_gap), **est}
    thr = args.t if args.t is not None else 2.0 * disc["se"]  # D2 is already floor-corrected
    rec["threshold"] = thr
    rec["screen_pass"] = bool(disc["D2"] <= thr and est["overlap_frac"] >= args.overlap_min)
    if diag is not None:
        Vd = (np.column_stack([diag["X"], diag["W"][:, rem]]) - mean_v) / sd_v
        Td = (np.column_stack([diag["X"], diag["W"][:, held]]) - mean_t) / sd_t
        Zd = encode(enc, Vd)
        half = len(Zd) // 2
        zd, td, ad = (torch.tensor(Zd, dtype=torch.float32), torch.tensor(Td, dtype=torch.float32), torch.tensor(diag["A"], dtype=torch.float32))
        disc_d = empirical_discrepancy(zd, td, ad, args.m_rff, 7, X.shape[1])
        gap_pop = disc_d["gap_cf"]
        nuis_d = fit_nuisances(Zd[:half], diag["A"][:half], diag["Y"][:half], 11)
        est_d = aipw(Zd[half:], diag["A"][half:], diag["Y"][half:], nuis_d, args.eta)
        rec.update({"oracle_theta_S": est_d["theta"], "oracle_gap_pop": gap_pop, "oracle_D2_pop": max(gap_pop, 0.0), "oracle_overlap_frac": est_d["overlap_frac"]})
    return rec


# ----------------------------------------------------------------------------- Algorithm 1 aggregation
def linking_radius(records: list[dict], args, m: int) -> float:
    sig = max(r["sigma"] for r in records); nE = records[0]["n"]
    if args.rho is not None:
        return float(args.rho)
    if args.rho_rule == "chebyshev":
        return float(sig / math.sqrt(nE * args.delta / m))
    return float(norm.ppf(1 - args.delta / (2 * m)) * sig / math.sqrt(nE))


def aggregate(records: list[dict], rho: float) -> dict:
    th = np.array([r["theta"] for r in records]); k = len(th)
    parent = list(range(k))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for i in range(k):
        for j in range(i + 1, k):
            if abs(th[i] - th[j]) <= 2 * rho:
                parent[find(i)] = find(j)
    comps: dict[int, list[int]] = {}
    for i in range(k):
        comps.setdefault(find(i), []).append(i)
    sizes = sorted(((len(v), key) for key, v in comps.items()), reverse=True)
    largest = [key for size, key in sizes if size == sizes[0][0]]
    medians = [float(np.median(th[comps[key]])) for key in largest]
    return {"rho": rho, "n_retained": k, "component_sizes": sorted((len(v) for v in comps.values()), reverse=True),
            "tie": len(largest) > 1, "output": medians[0] if len(largest) == 1 else None, "candidates": medians,
            "largest_members": [records[i]["S"] for i in comps[largest[0]]]}


# ----------------------------------------------------------------------------- baselines
def baseline_aipw(feats: np.ndarray, data: dict, idx: dict, eta: float, seed: int) -> float:
    """Honest AIPW on a fixed adjustment set: nuisances on the discrepancy and nuisance samples together (the
    baselines need no representation stage, so they get both), evaluation on the same evaluation sample as PROBE."""
    fit = np.concatenate([idx["D"], idx["N"]])
    nuis = fit_nuisances(feats[fit], data["A"][fit], data["Y"][fit], seed)
    return aipw(feats[idx["E"]], data["A"][idx["E"]], data["Y"][idx["E"]], nuis, eta)["theta"]


# ----------------------------------------------------------------------------- driver
def run_replicate(args, coef, scm: str, params: dict, rep: int, n: int, n_index: int) -> dict:
    data = generate(n, seed_key("data", rep), coef, scm, params)
    blk = scms.blocks(scm, **params)
    perm = rng(*seed_key("split", rep, n_index)).permutation(n)
    third = n // 3
    idx = {"D": perm[:third], "N": perm[third:2 * third], "E": perm[2 * third:]}
    diag = generate(args.n_diag, seed_key("diag", rep), coef, scm, params) if args.diagnostics else None
    splits = all_splits(len(blk), args.max_held)
    g = rng(*seed_key("sample_splits", rep, n_index))
    chosen = [splits[i] for i in sorted(g.choice(len(splits), size=min(args.m, len(splits)), replace=False))]
    records = [run_split(data, idx, S, args, rep, n_index, diag, blk) for S in chosen]
    for r in records:
        r["valid_heldout"] = scms.heldout_set_valid(scm, r["S"], **params)
    retained = [r for r in records if r["screen_pass"]]
    out = {"scm": scm, "params": params, "rep": rep, "n": n, "T": len(splits), "m": len(chosen), "splits": records,
           "invalid_heldout_blocks": sorted(scms.invalid_heldout_blocks(scm, **params))}
    if retained:
        rho = linking_radius(retained, args, len(chosen))
        out["algorithm1"] = aggregate(retained, rho)
        th = np.array([r["theta"] for r in retained])
        out["mean_over_retained"] = float(th.mean()); out["median_over_retained"] = float(np.median(th))
    else:
        out["algorithm1"] = {"n_retained": 0, "output": None, "tie": False, "candidates": []}
    X, W = data["X"], data["W"]
    s = torch_seed(*seed_key("nuisance", rep, n_index, 999)) % (2 ** 31)
    A_, Y_ = data["A"], data["Y"]
    out["naive"] = float(Y_[A_ == 1].mean() - Y_[A_ == 0].mean())
    out["baseline_X"] = baseline_aipw(X, data, idx, args.eta, s)
    out["baseline_XW"] = baseline_aipw(np.column_stack([X, W]), data, idx, args.eta, s + 1)
    out["oracle_XU"] = baseline_aipw(scms.oracle_adjustment(scm, data), data, idx, args.eta, s + 2)
    out["baseline_X_Wmean"] = baseline_aipw(np.column_stack([X, W.mean(1)]), data, idx, args.eta, s + 3)
    return out


def summarize(runs: list[dict]) -> dict:
    def err(key):
        vals = [abs(r[key] - TRUE_ATE) for r in runs if r.get(key) is not None]
        return {"mae": float(np.mean(vals)) if vals else None, "count": len(vals)}
    alg = [r["algorithm1"] for r in runs]
    outs = [abs(a["output"] - TRUE_ATE) for a in alg if a.get("output") is not None]
    within = [abs(a["output"] - TRUE_ATE) <= a["rho"] for a in alg if a.get("output") is not None]
    return {"algorithm1": {"mae": float(np.mean(outs)) if outs else None, "single_output_rate": len(outs) / len(alg),
                           "within_rho_rate": float(np.mean(within)) if within else None,
                           "mean_retained": float(np.mean([a["n_retained"] for a in alg]))},
            "mean_over_retained": err("mean_over_retained"), "median_over_retained": err("median_over_retained"),
            "naive": err("naive"), "baseline_X": err("baseline_X"), "baseline_XW": err("baseline_XW"), "oracle_XU": err("oracle_XU"),
            "baseline_X_Wmean": err("baseline_X_Wmean")}


def audit_dgps(args) -> None:
    """Sanity table for every SCM at a large sample size: propensity range, AIPW bias of the fixed adjustment
    sets, and the audit power of the held-out block (cross-fitted Brier gap for Z = X and for the oracle Z)."""
    n = args.n_audit
    for name in (scms.SCM_NAMES if args.scm == "all" else [args.scm]):
        params = scm_params(args) if args.scm != "all" else {}
        coef = coefficients(name, params); data = generate(n, seed_key("data", 0), coef, name, params)
        blk = scms.blocks(name, **params); S = (1,) if len(blk) > 1 else (0,)
        held = np.concatenate([blk[b] for b in S])
        X, W, A, Y = data["X"], data["W"], data["A"], data["Y"]
        perm = rng(*seed_key("split", 0, 0)).permutation(n); third = n // 3
        idx = {"D": perm[:third], "N": perm[third:2 * third], "E": perm[2 * third:]}
        b_x = baseline_aipw(X, data, idx, args.eta, 1); b_xw = baseline_aipw(np.column_stack([X, W]), data, idx, args.eta, 2)
        b_or = baseline_aipw(scms.oracle_adjustment(name, data), data, idx, args.eta, 3)
        inv = scms.invalid_heldout_blocks(name, **params)
        keep = np.concatenate([blk[b] for b in range(len(blk)) if b not in inv]) if inv else None
        b_xw_valid = baseline_aipw(np.column_stack([X, W[:, keep]]), data, idx, args.eta, 4) if inv else float("nan")
        T = np.column_stack([X, W[:, held]]); T = (T - T.mean(0)) / (T.std(0) + 1e-8)
        def gap(Z):
            Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
            return empirical_discrepancy(torch.tensor(Zs, dtype=torch.float32), torch.tensor(T, dtype=torch.float32), torch.tensor(A, dtype=torch.float32), args.m_rff, 3, X.shape[1])
        gx, go = gap(X), gap(scms.oracle_adjustment(name, data))
        pr = data["propensity"]
        print(f"{name}: {scms.DESCRIPTIONS[name]}\n   orc: {scms.check_orc(name, coef, **params)}\n   blocks={len(blk)} d_x={X.shape[1]} d_w={W.shape[1]} | propensity [{pr.min():.3f}, {pr.max():.3f}] | naive {Y[A==1].mean()-Y[A==0].mean():+.3f}, estimates X-only {b_x:+.3f}, (X,W) {b_xw:+.3f}, (X,W valid) {b_xw_valid:+.3f}, oracle {b_or:+.3f} (truth {TRUE_ATE:+.1f}) | audit Z=X: gap {gx['gap_cf']:+.4f} floor {gx['null_floor']:+.4f} D2 {gx['D2']:.4f}\u00b1{gx['se']:.4f} | Z=oracle: gap {go['gap_cf']:+.4f} floor {go['null_floor']:+.4f} D2 {go['D2']:.4f}\u00b1{go['se']:.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scm", default="s1", choices=scms.SCM_NAMES + ["all"])
    ap.add_argument("--n", type=int, nargs="+", default=[1500])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--m", type=int, default=10, help="budget: number of sampled proxy splits")
    ap.add_argument("--max-held", type=int, default=None, help="largest number of blocks in a held-out set (default: half of the blocks)")
    ap.add_argument("--d-z", type=int, default=3, help="representation dimension")
    ap.add_argument("--enc-steps", type=int, default=300)
    ap.add_argument("--enc-hidden", type=int, default=32, help="hidden width of the encoder MLP")
    ap.add_argument("--pretrain-steps", type=int, default=0, help="warm-start steps of treatment-predictive encoder training before the balance objective")
    ap.add_argument("--m-rff", type=int, default=128, help="random Fourier features per critic")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam", type=float, default=1e-2, help="ridge penalty of the critics during training (times n)")
    ap.add_argument("--keep-x", action="store_true", help="use Z = (X, psi) instead of Z = phi(X, W_{-S})")
    ap.add_argument("--t", type=float, default=None, help="screen threshold on the cross-fitted discrepancy; default 2 standard errors")
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--overlap-min", type=float, default=0.98, help="required share of evaluation units with clipped-free propensity")
    ap.add_argument("--rho", type=float, default=None, help="linking radius; if omitted, derived by --rho-rule")
    ap.add_argument("--rho-rule", default="normal", choices=["normal", "chebyshev"])
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--n-diag", type=int, default=20000)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--threads", type=int, default=2, help="torch threads per process")
    ap.add_argument("--audit-dgps", action="store_true", help="print the DGP sanity table and exit")
    ap.add_argument("--n-audit", type=int, default=20000)
    ap.add_argument("--proxy-strength", type=float, default=None, help="override the SCM default loading multiplier of W on U")
    ap.add_argument("--n-blocks", type=int, default=None, help="override the number of proxy blocks (s4 sweep)")
    ap.add_argument("--d-x", type=int, default=None, help="covariate dimension (default 20)")
    ap.add_argument("--d-w", type=int, default=None, help="proxy dimension; block widths follow from d_w and the block count")
    ap.add_argument("--scm-param", action="append", help="override an SCM parameter, e.g. instrument_coef=3.0 (repeatable)")
    ap.add_argument("--baseline-proxy-mean", action="store_true", help="also report AIPW on (X, mean of W_{-S}) for the first sampled split")
    args = ap.parse_args()
    if args.audit_dgps:
        audit_dgps(args); return
    torch.set_num_threads(args.threads)
    params = scm_params(args)
    coef = coefficients(args.scm, params)
    out = Path(args.out) if args.out else RESULTS_DIR / f"probe_e1_{args.scm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); runs = []

    def save(done: bool):
        summary = {n: summarize([r for r in runs if r["n"] == n]) for n in args.n if any(r["n"] == n for r in runs)}
        payload = {"config": vars(args), "scm_params": scms.resolve(args.scm, **params), "true_ate": TRUE_ATE, "complete": done,
                   "summary": summary, "runs": runs, "runtime_s": time.time() - t0}
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o))
        tmp.replace(out)
        return summary

    for n_index, n in enumerate(args.n):
        for rep in range(args.reps):
            r = run_replicate(args, coef, args.scm, params, rep, n, n_index); runs.append(r)
            a = r["algorithm1"]
            print(f"[{args.scm} n={n} rep={rep}] retained {a.get('n_retained')}/{r['m']} | alg1 out={a.get('output')} rho={a.get('rho')} comps={a.get('component_sizes')} | "
                  f"mean={r.get('mean_over_retained')} | naive={r['naive']:.3f} X={r['baseline_X']:.3f} XW={r['baseline_XW']:.3f} XU={r['oracle_XU']:.3f} | {time.time()-t0:.0f}s", flush=True)
            save(False)
    summary = save(True)
    print(json.dumps(summary, indent=2)); print("saved", out)


if __name__ == "__main__":
    main()
