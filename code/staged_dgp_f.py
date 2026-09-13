"""Staged study of design F, the DGP whose exact balancing representation is known in closed form.

The review of 2026-09-11 (round 5) asked for the pipeline to be taken apart into checks that fail
independently; round 6 found that the first version of those checks evaluated the encoder on rows it had
trained on, compared learners on different rows, gave every comparator its own cross-fitting split, and
called a benchmark "exact" that was estimated from a sample.  This version fixes each of those.

  stage 1  evaluator   known exact representations and unbalanced controls, scored by the fitted critics
                       against a benchmark computed from the structural equations without sampling
  stage 2  learning    a linear representation is learned; learned, initial and exact representations are
                       scored on the same audit rows, which the encoder never trained or selected on
  stage 3  estimation  representation frozen, estimation sample grown, identical folds for every adjustment
  stage 4  convergence representation sample grown with nested training sets per data seed, optimizer seeds
                       varied separately from data seeds

Row usage is explicit and enforced.  Within a representation sample the rows are split 50/25/25 into
`fit` (encoder and critic training), `val` (checkpoint and restart selection) and `audit` (evaluation only).
Every evaluation asserts that its rows are disjoint from `fit` and `val`.  Causal estimates use a separate
estimation sample that never enters representation learning.

Design F.  Writing Z_3 = U + mean(e) + r I with k measurements kept, balance holds exactly when
r^2 - 3 r + 1/k = 0.  The linear encoder class contains that target: X, C, I and every measurement are raw
coordinates of the standardized block inputs.

Deviation from the plan, recorded deliberately: Z has three coordinates rather than the four of plan 8.4,
matching the exact representation.  This script is a diagnostic; its numbers are not confirmatory results.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit

import probe_dgp_ae as dgp
import provenance as prov
import probe_brier_neural as pbn

LINEAR_Z_DIM = 3
STAGE_SEED = 20260911
GH_NODES = 40                     # Gauss-Hermite order per dimension for the benchmark
GH_NODES_CHECK = 80               # doubled order, for the stability check
# Fixed per-adjustment nuisance seed offsets.  Python randomizes string hashing between processes, so a seed
# derived from hash(label) would not reproduce; the round 1 review caught exactly that in the main runner.
ADJ_SEED = {"PROBE, exact representation": 11, "PROBE, learned representation": 22, "raw (X, W)": 33,
            "oracle (X, U, C)": 44, "X only": 55}
CODE_FILES = ("staged_dgp_f.py", "probe_brier_neural.py", "probe_dgp_ae.py")
RUN_ID = ""                       # set in main(); scopes checkpoint filenames to one invocation
CHECKPOINT_DIR = Path("../results/checkpoints")
PROTOCOL_ID = "dgp-f-r10-p0-v1"


# ----------------------------------------------------------------------------- provenance
def code_hash() -> str:
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update(name.encode()); h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()[:16]


def split_id(idx: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.sort(idx.astype(np.int64))).tobytes()).hexdigest()[:16]


def assert_disjoint(name: str, eval_rows: np.ndarray, **used: np.ndarray) -> dict:
    """Refuse to evaluate on any row that took part in training or selection.  Returns the counts."""
    e = set(int(i) for i in eval_rows)
    counts = {k: len(e & set(int(i) for i in v)) for k, v in used.items()}
    if any(counts.values()):
        raise ValueError(f"{name}: evaluation rows overlap {counts}; the evaluation is not independent")
    return counts


def state_hash(state: dict) -> str:
    """Content hash of a state dict, so two runs can be compared for identical initial weights."""
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode()); h.update(np.ascontiguousarray(state[k].detach().numpy()).tobytes())
    return h.hexdigest()[:16]


def save_checkpoint(tag: str, fitted: dict, st: dict, extra: dict, run_id: str,
                    cell_key: tuple | list, protocol_id: str = PROTOCOL_ID) -> dict:
    """Write a checkpoint under a run-scoped name and return its path with the file's SHA256.

    Round 6 wrote `<tag>.pt` and recorded only the path, so a later run with different settings silently
    replaced the weights a stored result pointed at; two stage 4 runs already shared nine paths.  The run id
    is part of the filename, an existing file is never overwritten, and the digest lets `load_checkpoint`
    verify that the bytes are the ones the result was produced from (round 7 review, R6).
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = (CHECKPOINT_DIR / f"{run_id}_{tag}.pt").resolve()
    # Exclusive creation, so two concurrent runs cannot both pass an existence check and then both write.
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise FileExistsError(f"{path} already exists; a run id must not be reused") from None
    os.close(fd)
    checkpoint_provenance = {"protocol_id": protocol_id, "run_id": run_id,
                             "cell_key": list(cell_key)}
    torch.save({"encoder": fitted["enc"].state_dict(), "init": fitted["init_state"],
                "standardizers": st, "restart": fitted["restart"], "optimizer_seed": fitted["optimizer_seed"],
                "epoch": fitted["epoch"], "extra": extra, "run_id": run_id,
                "checkpoint_provenance": checkpoint_provenance}, path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "init_hash": state_hash(fitted["init_state"]), **checkpoint_provenance}


def load_checkpoint(record: dict) -> dict:
    """Load a checkpoint and verify its bytes against the digest stored beside the result."""
    path = Path(record["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise ValueError(f"{path} has changed since the result was written: stored {record['sha256'][:12]}, "
                         f"found {digest[:12]}")
    return torch.load(path, weights_only=False)


# ----------------------------------------------------------------------------- representations
class LinearRepresentation(nn.Module):
    """Z = W [standardized block values, standardized X] + b, a single affine map."""

    def __init__(self, specs: list[dict], d_x: int, d_z: int = LINEAR_Z_DIM) -> None:
        super().__init__()
        if any(s["kind"] != "tabular" for s in specs):
            raise ValueError("the linear encoder is defined for tabular blocks only")
        self.specs = specs
        self.lin = nn.Linear(sum(s["dim"] for s in specs) + d_x, d_z)

    def forward(self, parts: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        cols = [parts[f"tab{s['index']}"] for s in self.specs] + [x]
        return self.lin(torch.cat(cols, 1))


class CoefficientRepresentation(nn.Module):
    """Z_r = (X, C, mean of the kept measurements + r I) with r the only trainable parameter.

    The three output coordinates are read straight out of the standardized block inputs, undoing the
    standardization so that the map is the same family the benchmark scores: the first coordinate is X, the
    second is C, the third is the mean of the measurements left in the representation input plus r times the
    treatment-only cause I.  Nothing else can move, so a failure to reach a good r is a failure of the search
    over one number and not of the function class (round 7 review, PRD-3).
    """

    n_learnable = 1

    def __init__(self, specs: list[dict], d_x: int, r_init: float = 0.0, st: dict | None = None) -> None:
        super().__init__()
        if st is None:
            raise ValueError("the coefficient encoder needs the standardizers to undo their scaling")
        self.specs = specs
        self.r = nn.Parameter(torch.tensor(float(r_init)))
        kept = [sp["index"] for sp in specs]
        if 0 not in kept:
            raise ValueError("design F keeps block 0, which carries I, C and the first measurement")
        self.kept = kept
        # column j of block b in raw units is  standardized * sd + mu
        self._sd = {b: torch.as_tensor(st[f"tab{b}"].sd.astype(np.float32)) for b in kept}
        self._mu = {b: torch.as_tensor(st[f"tab{b}"].mu.astype(np.float32)) for b in kept}
        self._xsd = torch.as_tensor(st["X"].sd.astype(np.float32))
        self._xmu = torch.as_tensor(st["X"].mu.astype(np.float32))

    def _raw(self, parts: dict[str, torch.Tensor], b: int, col: int) -> torch.Tensor:
        return parts[f"tab{b}"][:, col] * self._sd[b][col] + self._mu[b][col]

    def forward(self, parts: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        i_cause = self._raw(parts, 0, 0)
        c_sev = self._raw(parts, 0, 1)
        ms = [self._raw(parts, 0, 2)] + [self._raw(parts, b, 0) for b in self.kept if b != 0]
        m_bar = torch.stack(ms, 1).mean(1)
        x_raw = x[:, 0] * self._xsd[0] + self._xmu[0]
        return torch.stack([x_raw, c_sev, m_bar + self.r * i_cause], 1)


class FrozenZ(nn.Module):
    """Serves a precomputed representation so a known Z can be pushed through the ordinary evaluator."""

    def __init__(self, z: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("z", torch.as_tensor(np.asarray(z, dtype=np.float32)))

    def forward(self, parts: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        if len(x) != len(self.z):
            raise ValueError("the frozen representation is only defined on the rows it was built from")
        return self.z


def f_representation(data: dict, S, r: float) -> np.ndarray:
    """(X, C, mean of the kept measurements + r I); r = r* reproduces `dgp.f_exact_representation`."""
    kept = [0] + [j for j in range(1, 5) if j not in set(int(b) for b in S)]
    m = np.column_stack([data["M"][:, j] for j in kept]).mean(1)
    return np.column_stack([data["X"][:, 0], data["C"], m + r * data["I1"]]).astype(np.float32)


# ----------------------------------------------------------------------------- exact benchmark (P1)
# Every design F variable is a fixed linear combination of independent standard normals, so the covariance of
# any collection of them is a product of coefficient matrices.  Nothing here is sampled.
NOISE = ["U", "I", "C", "eX"] + [f"e{j}" for j in range(5)] + [f"zeta{j}" for j in range(1, 5)] \
    + [f"N{j}" for j in range(1, 5)]
NIDX = {name: i for i, name in enumerate(NOISE)}


def _row(**coef) -> np.ndarray:
    v = np.zeros(len(NOISE))
    for k, c in coef.items():
        v[NIDX[k]] = c
    return v


def f_linear_maps() -> dict:
    """Coefficient rows over the noise vector for every design F quantity used by the benchmark."""
    maps = {"U": _row(U=1), "I": _row(I=1), "C": _row(C=1), "X": _row(U=1, eX=dgp.F_X_NOISE)}
    for j in range(5):
        maps[f"M{j}"] = _row(U=1, **{f"e{j}": 1})
    # V is the propensity index and L the outcome-relevant part of Y that does not involve A or eps_Y
    maps["V"] = (dgp.F_INDEX["U"] * maps["U"] + dgp.F_INDEX["I"] * maps["I"]
                 + dgp.F_INDEX["X"] * maps["X"] + dgp.F_INDEX["C"] * maps["C"])
    maps["L"] = dgp.F_OUTCOME["U"] * maps["U"] + dgp.F_OUTCOME["X"] * maps["X"] + dgp.F_OUTCOME["C"] * maps["C"]
    maps["W0"] = np.vstack([maps["I"], maps["C"], maps["M0"]])
    for j in range(1, 5):
        maps[f"W{j}"] = np.vstack([maps[f"M{j}"], maps["C"] + dgp.F_C_NOISE * _row(**{f"zeta{j}": 1}),
                                   _row(**{f"N{j}": 1})])
    return maps


def z_map(kind: str, S, r: float | None, maps: dict) -> np.ndarray:
    """Coefficient rows of a representation.  Mirrors the representations built from data in stage 1."""
    kept = [0] + [j for j in range(1, 5) if j not in set(int(b) for b in S)]
    mean_m = np.mean([maps[f"M{j}"] for j in kept], axis=0)
    if kind == "family":
        return np.vstack([maps["X"], maps["C"], mean_m + r * maps["I"]])
    if kind == "no measurement channel":
        return np.vstack([maps["X"], maps["C"], maps["I"]])
    if kind == "C dropped":
        return np.vstack([maps["X"], mean_m + r * maps["I"]])
    if kind == "X and C only":
        return np.vstack([maps["X"], maps["C"]])
    if kind == "constant":
        return np.zeros((0, len(NOISE)))
    raise ValueError(kind)


def _conditional(target: np.ndarray, given: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, float]:
    """Coefficient row of E[target | given] and the conditional variance, for mean-zero Gaussian rows.

    The conditioning set is reduced to an orthonormal basis of its row space by SVD before projecting.  A
    ridge on a singular Gram matrix would shrink the projection and silently understate the conditional mean,
    and the set is singular whenever a coordinate is repeated.  That happens by construction here: the
    manuscript's held-out variable is T = (X, W_S) and X is also a coordinate of most representations, so
    conditioning on (Z, T) conditions on X twice.  Working in the row space makes the result invariant to
    duplicated coordinates and to any invertible linear reparametrisation of the conditioning set.
    """
    if given.shape[0] == 0:
        return np.zeros(len(NOISE)), float(target @ target)
    u, sv, vt = np.linalg.svd(given, full_matrices=False)
    keep = sv > tol * max(sv[0], 1.0)
    basis = vt[keep]                                  # orthonormal rows spanning the same space
    coef = basis @ target
    mean_row = coef @ basis
    resid = float(target @ target - coef @ coef)
    return mean_row, max(resid, 0.0)


def _gh(nodes: int):
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    return x, w / w.sum()


def smoothed_prob(m, s: float, nodes: int):
    """g(m; s) = 0.05 + 0.9 E[expit(m + s xi)], xi standard normal, by Gauss-Hermite quadrature."""
    x, w = _gh(nodes)
    m = np.asarray(m, dtype=float)[..., None]
    return 0.05 + 0.9 * (expit(m + s * x) * w).sum(-1)


def benchmark(S, kind: str, r: float | None, nodes: int = GH_NODES) -> dict:
    return benchmark_rows(S, z_map(kind, S, r, f_linear_maps()), nodes)


def learned_z_rows(fitted: dict, st: dict) -> np.ndarray:
    """Coefficient rows over the noise vector of a learned *linear* representation.

    The encoder is an affine map of standardized inputs, each of which is an affine map of raw block values,
    which are linear in the noise.  Constant offsets do not affect conditional expectations and are dropped,
    so the population D^2 and tau_Z of a learned representation can be computed exactly, and the causal error
    can be split into representation error |tau_Z - 1| and estimation error |theta_hat - tau_Z| (P2).
    """
    if not isinstance(fitted["enc"], LinearRepresentation):
        raise TypeError("exact scoring of a learned representation is defined for the linear encoder only")
    maps = f_linear_maps()
    rows = []
    for spec in fitted["enc"].specs:
        b = spec["index"]
        rows.append(maps[f"W{b}"] / st[f"tab{b}"].sd[:, None])
    rows.append(maps["X"][None, :] / st["X"].sd[:, None])
    F = np.vstack(rows)
    W = fitted["enc"].lin.weight.detach().numpy().astype(float)
    return W @ F


def benchmark_rows(S, Z: np.ndarray, nodes: int = GH_NODES) -> dict:
    """Population D^2 and adjusted effect tau_Z for a linear representation, without sampling.

    D^2 = E[{P(A=1|T_S,Z) - P(A=1|Z)}^2].  With V the propensity index, P(A=1|Z) = g(m0; s0) where
    m0 = E[V|Z] and s0^2 = Var(V|Z), and P(A=1|Z,T_S) = g(m1; s1) likewise.  The pair (m0, m1) is bivariate
    Gaussian, so the outer expectation is a two-dimensional quadrature.

    tau_Z = E[E(Y|A=1,Z) - E(Y|A=0,Z)].  With L the outcome-relevant part of Y and beta the regression of L on
    V given Z, E(L|A=1,Z) - E(L|A=0,Z) = beta s0 E[xi p(m0 + s0 xi)] / [P(A=1|Z) P(A=0|Z)], and only m0 varies
    with Z, so the outer expectation is a one-dimensional quadrature.  Balance means beta = 0 and tau_Z = 1.
    """
    maps = f_linear_maps()
    # The manuscript (Section 3, "Fix a representation phi and write T_j = (X, W_j)") holds out X together
    # with the proxy block, and the fitted AugmentedCritic is given x as well.  Omitting X here made the
    # benchmark a different quantity from the thing being estimated; a representation that drops part of X
    # was scored as if that loss were invisible.  Round 7 review, R1.
    T = np.vstack([maps["X"][None, :]] + [maps[f"W{b}"] for b in S])
    ZT = np.vstack([Z, T])
    m0_row, s0sq = _conditional(maps["V"], Z)
    m1_row, s1sq = _conditional(maps["V"], ZT)
    s0, s1 = math.sqrt(s0sq), math.sqrt(s1sq)
    cov = np.array([[m0_row @ m0_row, m0_row @ m1_row], [m0_row @ m1_row, m1_row @ m1_row]])
    tower_defect = abs(cov[0, 1] - cov[0, 0])                 # E[m1 | m0] = m0 must hold exactly
    chol = np.linalg.cholesky(cov + 1e-14 * np.eye(2))
    x, w = _gh(nodes)
    g1, g2 = np.meshgrid(x, x, indexing="ij")
    m = np.stack([g1.ravel(), g2.ravel()])
    mm = chol @ m
    p0 = smoothed_prob(mm[0], s0, nodes)
    p1 = smoothed_prob(mm[1], s1, nodes)
    ww = np.outer(w, w).ravel()
    d2 = float(((p1 - p0) ** 2 * ww).sum())

    # tau_Z
    lz_row, _ = _conditional(maps["L"], Z)
    resid_l = maps["L"] - lz_row
    resid_v = maps["V"] - m0_row
    beta = float(resid_l @ resid_v / s0sq) if s0sq > 0 else 0.0
    sd_m0 = math.sqrt(cov[0, 0])
    m0_nodes = sd_m0 * x
    pz = smoothed_prob(m0_nodes, s0, nodes)
    xi_p = np.array([((expit(mi + s0 * x) * x) * w).sum() for mi in m0_nodes]) * 0.9
    inner = beta * s0 * xi_p / (pz * (1 - pz))
    tau_z = dgp.TRUE_ATE + float((inner * w).sum())
    return {"D2": d2, "tau_Z": tau_z, "beta": beta, "s0": s0, "s1": s1, "tower_defect": tower_defect,
            "nodes": nodes}


# ----------------------------------------------------------------------------- shared plumbing
def three_way(idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """50/25/25 fit / val / audit of the given rows."""
    p = np.random.default_rng(seed).permutation(idx)
    a, b = int(0.50 * len(p)), int(0.75 * len(p))
    return p[:a], p[a:b], p[b:]


def evaluate_representation(data: dict, S, z: np.ndarray, st: dict, seed: int, d_fit: np.ndarray,
                            d_val: np.ndarray, d_audit: np.ndarray, name: str = "evaluation") -> dict:
    """Fit both critics from scratch on a frozen Z and report the Brier risk gap on the audit rows.

    The three index arrays are the same ones the encoder used, so the audit rows are guaranteed to have been
    outside encoder training and checkpoint selection.  The assertion makes that a hard failure.
    """
    assert_disjoint(name, d_audit, fit=d_fit, val=d_val)
    rest = [b for b in range(dgp.N_BLOCKS) if b not in set(S)]
    fitted = {"enc": FrozenZ(z), "rest": rest,
              "specs": (pbn.block_specs(data, rest), pbn.block_specs(data, list(S)))}
    return pbn.refit_critics(data, tuple(S), fitted, d_fit, d_val, d_audit, st, seed)


def paired_gap_difference(a: dict, b: dict) -> dict:
    """Mean and standard error of (gap_a - gap_b) computed row by row on the shared evaluation rows.

    Both arguments are `refit_critics` outputs on the same audit rows.  The rows are aligned by their ids
    rather than by position, so a reordered save cannot silently mispair them.  The standard error is that of
    the per-row difference, which keeps the finite evaluation sample in the answer: if the two fits give
    identical predictions the difference and its standard error are both exactly zero, and if the predictions
    differ the uncertainty stays positive however many optimizer seeds agree.
    """
    ra, rb = np.asarray(a["audit_rows"]), np.asarray(b["audit_rows"])
    if set(ra.tolist()) != set(rb.tolist()):
        raise ValueError("paired comparison needs the same evaluation rows in both arguments")
    oa, ob = np.argsort(ra), np.argsort(rb)
    d = np.asarray(a["audit_gap_rows"])[oa] - np.asarray(b["audit_gap_rows"])[ob]
    n = len(d)
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0, "n": n}


def seed_averaged_paired_difference(fits_a: list[dict], fits_b: list[dict]) -> dict:
    """Paired difference between two representations, averaged over optimizer seeds row by row.

    Let d_ik be the paired per-row loss difference on evaluation row i for optimizer seed k.  The quantity
    being reported is the seed average, so the row-level statistic is

        dbar_i = (1/K) sum_k d_ik,

    and the standard error that accounts for the finite evaluation sample is sd(dbar_i)/sqrt(n_audit).  Every
    seed is scored on the same rows, so averaging over seeds adds no evaluation data and this standard error
    is not divided by K.  The spread across seeds is a different quantity and is returned separately rather
    than folded into one number; uncertainty from resampling the *training* data is a third quantity and
    needs independent data seeds, which this function does not see.  Round 6 added a row-level term and a
    seed term in quadrature, which has no such interpretation (round 7 review, PRD-4).
    """
    if len(fits_a) != len(fits_b) or not fits_a:
        raise ValueError("need the same non-zero number of optimizer seeds on both sides")
    rows = np.asarray(fits_a[0]["audit_rows"])
    order = np.argsort(rows)
    mat = []
    for fa, fb in zip(fits_a, fits_b):
        ra, rb = np.asarray(fa["audit_rows"]), np.asarray(fb["audit_rows"])
        if set(ra.tolist()) != set(rows.tolist()) or set(rb.tolist()) != set(rows.tolist()):
            raise ValueError("every seed must be scored on the same evaluation rows")
        mat.append(np.asarray(fa["audit_gap_rows"])[np.argsort(ra)]
                   - np.asarray(fb["audit_gap_rows"])[np.argsort(rb)])
    mat = np.vstack(mat)                       # K x n_audit
    dbar = mat.mean(0)
    n = len(dbar)
    per_seed = mat.mean(1)
    return {"mean": float(dbar.mean()),
            "se_evaluation": float(dbar.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "sd_across_seeds": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
            "n_audit": int(n), "n_seeds": int(mat.shape[0]), "per_seed_means": per_seed.tolist(),
            "note": "se_evaluation holds the evaluation sample only; sd_across_seeds is the optimizer "
                    "randomness; training-data randomness needs independent data seeds"}


def two_folds(idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.random.default_rng(seed).permutation(idx)
    return p[: len(p) // 2], p[len(p) // 2:]


def crossfit_theta(features: np.ndarray, a: np.ndarray, y: np.ndarray, folds: tuple, seed: int) -> dict:
    """Two-fold cross-fitted AIPW on rows already restricted to the estimation sample.

    `folds` is computed once per cell and shared by every adjustment, so the comparators differ only in
    what they adjust for (round 6 review, P0.5).
    """
    psi = np.concatenate([
        pbn.aipw_scores(pbn.fit_nuisances(features[tr], a[tr], y[tr], seed + i), features[te], a[te], y[te])["psi"]
        for i, (tr, te) in enumerate((folds, folds[::-1]))])
    return {"theta": float(psi.mean()), "se": float(psi.std(ddof=1) / math.sqrt(len(psi)))}


def assign_roles(pool: np.ndarray, data_seed: int) -> dict:
    """Assign fit / val / audit roles once per data seed, on the largest representation pool.

    Round 6 re-split the pool at every sample size, so the training rows of the small setting were not a
    subset of the training rows of the large one (851 of 1500 overlapped) and the audit rows moved as well.
    Roles are fixed here, and a smaller run takes a prefix of the same fit list, so growing the sample adds
    training rows and changes nothing else (round 7 review, R4).
    """
    f, v, a = three_way(pool, data_seed * 7919 + 1)
    return {"fit_pool": f, "val": v, "audit": a}


def learn_representation(data: dict, S, idx_rep: np.ndarray, data_seed: int, opt_seed: int,
                         roles: dict | None = None, n_train: int | None = None,
                         st: dict | None = None) -> dict:
    """Train the linear encoder on the fit rows, select on the val rows, and keep the audit rows untouched.

    With `roles` given the split is the fixed one from `assign_roles` and `n_train` takes a nested prefix of
    its fit list; otherwise the pool is split on its own.  The optimizer seed does not depend on the sample
    size, so the same `opt_seed` starts every size from identical initial weights.
    """
    if roles is None:
        roles = assign_roles(idx_rep, data_seed)
        d_fit = roles["fit_pool"]
    else:
        d_fit = roles["fit_pool"][:n_train] if n_train is not None else roles["fit_pool"]
    d_val, d_audit = roles["val"], roles["audit"]
    st = pbn.build_standardizers(data, d_fit) if st is None else st
    fitted = pbn.train_representation(data, S, d_fit, d_val, st, (STAGE_SEED, opt_seed))
    return {"fitted": fitted, "st": st, "fit": d_fit, "val": d_val, "audit": d_audit}

def embed_rows(enc: nn.Module, rest: list[int], data: dict, idx: np.ndarray, st: dict) -> np.ndarray:
    with torch.no_grad():
        return enc(pbn.make_parts(data, rest, idx, st), torch.as_tensor(st["X"](data["X"][idx]))).numpy()


def report(rows: list[dict], cols: list[str], head: str) -> None:
    # A repeated column name means the row dicts lost a value: the second assignment overwrote the first.
    # That is how a reported "its se" came to show the fitted critic's standard error instead of the oracle's
    # in the round 7 smoke run (round 7 review, PRD-4 problem B).
    if len(set(cols)) != len(cols):
        raise ValueError(f"duplicate column names in {cols}; the row dicts cannot hold both")
    missing = [c for c in cols if any(c not in r for r in rows)]
    if missing:
        raise ValueError(f"columns absent from some rows: {missing}")
    print(f"\n{head}")
    width = {c: max(len(c), *(len(f"{r[c]}") for r in rows)) for c in cols}
    print("  " + "  ".join(c.rjust(width[c]) for c in cols))
    for r in rows:
        print("  " + "  ".join(f"{r[c]}".rjust(width[c]) for c in cols))


# ----------------------------------------------------------------------------- stage 1: the evaluator (P1)
# The list of representations, the tolerances and the pass rules are fixed here, before anything runs.
# Nothing is dropped afterwards: a representation the evaluator cannot resolve is reported as unresolved.
# The index-critic rule is a z-test against its own standard error.  A relative tolerance was tried first
# in a smoke run at n = 6000 and cannot be met for benchmark D^2 below 1e-3 at any feasible n, because the
# critic's sampling error (about 4e-4 at that size) is larger than the quantity itself; a relative rule is
# not a statistical test.  The relative error is still reported.
STAGE1_TOL = {"exact_D2": 1e-10, "exact_tau": 1e-8, "quadrature_rel": 1e-3, "index_critic_z": 2.0}


def stage1_representations(S):
    k = 1 + len([j for j in range(1, 5) if j not in set(S)])
    r_star = dgp.f_balance_coefficient(k)
    r_other = (3.0 + math.sqrt(9.0 - 4.0 / k)) / 2.0
    return r_star, [
        ("exact r*", "family", r_star), ("second root", "family", r_other),
        ("r = 0", "family", 0.0), ("r*/2", "family", r_star / 2), ("2 r*", "family", 2 * r_star),
        ("r = 0.5", "family", 0.5), ("r = 1", "family", 1.0),
        ("no measurement channel", "no measurement channel", None),
        ("C dropped", "C dropped", r_star), ("X and C only", "X and C only", None)]


def build_from_data(kind: str, r, data: dict, S) -> np.ndarray:
    if kind == "family":
        return f_representation(data, S, r)
    if kind == "no measurement channel":
        return np.column_stack([data["X"][:, 0], data["C"], data["I1"]]).astype(np.float32)
    if kind == "C dropped":
        return f_representation(data, S, r)[:, [0, 2]]
    if kind == "X and C only":
        return np.column_stack([data["X"][:, 0], data["C"]]).astype(np.float32)
    raise ValueError(kind)


class IndexCritic(nn.Module):
    """A critic of the true functional form: p = g(w.f + b; s) with the smoothing s taken from the benchmark.

    Used only as a diagnostic.  If this critic recovers the benchmark D^2 while the neural critic does not, the
    benchmark is right and the neural critic has a detection floor; if neither does, the benchmark is suspect.
    """

    def __init__(self, d: int, s: float) -> None:
        super().__init__()
        self.lin = nn.Linear(d, 1)
        x, w = _gh(GH_NODES)
        self.register_buffer("x", torch.as_tensor(x, dtype=torch.float32) * s)
        self.register_buffer("w", torch.as_tensor(w, dtype=torch.float32))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        m = self.lin(f)
        return 0.05 + 0.9 * (torch.sigmoid(m + self.x) * self.w).sum(1)


def index_critic_gap(data: dict, S, z: np.ndarray, d_fit, d_val, d_audit, s0: float, s1: float,
                     seed: int) -> dict:
    a = torch.as_tensor(data["A"].astype(np.float32))
    # T = (X, W_S), matching the manuscript and `AugmentedCritic`, which also receives x.
    t = np.column_stack([data["X"]] + [data["blocks"][b]["v"] for b in S]).astype(np.float32)
    f0 = torch.as_tensor(z); f1 = torch.as_tensor(np.column_stack([z, t]))
    out = {}
    preds = {}
    for name, f, s in (("R0", f0, s0), ("R1", f1, s1)):
        torch.manual_seed(seed)
        net = IndexCritic(f.shape[1], s)
        opt = torch.optim.Adam(net.parameters(), lr=0.05)
        best, best_state, stale = float("inf"), None, 0
        for _ in range(2000):
            opt.zero_grad(); loss = ((a[d_fit] - net(f[d_fit])) ** 2).mean(); loss.backward(); opt.step()
            with torch.no_grad():
                v = float(((a[d_val] - net(f[d_val])) ** 2).mean())
            if v < best - 1e-9:
                best, stale, best_state = v, 0, {k: q.clone() for k, q in net.state_dict().items()}
            else:
                stale += 1
                if stale >= 50:
                    break
        net.load_state_dict(best_state)
        with torch.no_grad():
            preds[name] = net(f[d_audit])
    diff = ((a[d_audit] - preds["R0"]) ** 2 - (a[d_audit] - preds["R1"]) ** 2).numpy()
    return {"gap": float(diff.mean()), "se": float(diff.std(ddof=1) / math.sqrt(len(diff)))}


def stage1(args) -> dict:
    S = tuple(args.stage1_S)
    r_star, reps = stage1_representations(S)
    print(f"\n[stage 1] evaluator check, S={S}, r* = {r_star:.6f}; tolerances {STAGE1_TOL}")

    # (a) benchmark, with the quadrature stability check
    bench, bench_rows, ok_exact, ok_quad = {}, [], True, True
    for label, kind, r in reps:
        b = benchmark(S, kind, r, GH_NODES); b2 = benchmark(S, kind, r, GH_NODES_CHECK)
        rel = abs(b["D2"] - b2["D2"]) / max(b2["D2"], 1e-6)
        stable = rel < STAGE1_TOL["quadrature_rel"] and abs(b["tau_Z"] - b2["tau_Z"]) < 1e-4
        ok_quad &= stable
        if label in ("exact r*", "second root"):
            ok_exact &= b2["D2"] < STAGE1_TOL["exact_D2"] and abs(b2["tau_Z"] - 1) < STAGE1_TOL["exact_tau"]
        bench[label] = b2
        bench_rows.append({"representation": label, "D2": f"{b2['D2']:.3e}", "tau_Z": f"{b2['tau_Z']:+.5f}",
                           "|tau_Z - 1|": f"{abs(b2['tau_Z'] - 1):.4f}", "quadrature rel. change": f"{rel:.1e}",
                           "stable": "yes" if stable else "NO"})
    report(bench_rows, ["representation", "D2", "tau_Z", "|tau_Z - 1|", "quadrature rel. change", "stable"],
           "[stage 1] benchmark from the structural equations (no sampling)")
    print(f"[stage 1] (a) both exact representations give D2 < {STAGE1_TOL['exact_D2']} and |tau_Z - 1| < "
          f"{STAGE1_TOL['exact_tau']}: {'PASS' if ok_exact else 'FAIL'}")
    print(f"[stage 1] (b) quadrature stable at doubled order for every representation: "
          f"{'PASS' if ok_quad else 'FAIL'}")

    # (c) the fitted evaluators on data, with K optimizer seeds each
    rows, raw = [], {}
    for n in args.stage1_n:
        data = dgp.generate("F", n, replicate=STAGE_SEED, phase="pilot", coef=dgp.coefficients())
        d_fit, d_val, d_audit = three_way(np.arange(n), STAGE_SEED)
        assert_disjoint("stage 1", d_audit, fit=d_fit, val=d_val)
        st = pbn.build_standardizers(data, d_fit)
        for label, kind, r in reps:
            z = build_from_data(kind, r, data, S)
            gaps, r0s, r1s, ses, fits = [], [], [], [], []
            for k in range(args.stage1_seeds):
                res = evaluate_representation(data, S, z, st, STAGE_SEED + n + 101 * k, d_fit, d_val, d_audit,
                                              name=f"stage 1 {label}")
                gaps.append(res["refit_gap_audit"]); ses.append(res["refit_se_audit"])
                r0s.append(res["refit_R0_audit"]); r1s.append(res["refit_R1_audit"])
                fits.append(res)
            oracle = index_critic_gap(data, S, z, d_fit, d_val, d_audit, bench[label]["s0"], bench[label]["s1"],
                                      STAGE_SEED + n)
            g = np.array(gaps)
            raw[(n, label)] = {"neural_gaps": gaps, "neural_within_se": ses, "R0": r0s, "R1": r1s,
                               "index_critic": oracle, "fits": fits}
            rows.append({"n": n, "representation": label, "benchmark D2": f"{bench[label]['D2']:.3e}",
                         "neural gap mean": f"{g.mean():+.3e}", "sd over seeds": f"{g.std(ddof=1):.1e}",
                         "within-fit se": f"{np.mean(ses):.1e}",
                         "R0": f"{np.mean(r0s):.5f}", "R1": f"{np.mean(r1s):.5f}",
                         "index critic gap": f"{oracle['gap']:+.3e}", "its se": f"{oracle['se']:.1e}"})
    report(rows, ["n", "representation", "benchmark D2", "neural gap mean", "sd over seeds", "within-fit se",
                  "R0", "R1", "index critic gap", "its se"],
           f"[stage 1] fitted evaluators against the benchmark ({args.stage1_seeds} optimizer seeds each)")

    # (d) separation: which controls are told apart from the exact representation on this evaluation sample,
    # and does the index critic recover the benchmark where the neural critic does not.
    #
    # The comparison is paired row by row, because both representations are scored on the same audit rows.
    # The reported uncertainty combines the row-level standard error of one seed's paired difference, which
    # carries the finite evaluation sample, with the spread across optimizer seeds.  The round 6 version used
    # only the seed spread; that goes to zero when the seeds agree, which would let an arbitrarily small
    # difference count as separation.
    verdict, big = [], max(args.stage1_n)
    ex = raw[(big, "exact r*")]
    ex_mean, ex_sd = np.mean(ex["neural_gaps"]), np.std(ex["neural_gaps"], ddof=1)
    for label, kind, r in reps[2:]:
        c = raw[(big, label)]
        c_mean, c_sd = np.mean(c["neural_gaps"]), np.std(c["neural_gaps"], ddof=1)
        agg = seed_averaged_paired_difference(c["fits"], ex["fits"])
        diff, paired_se, seed_sd = agg["mean"], agg["se_evaluation"], agg["sd_across_seeds"]
        sep_sd = paired_se
        bench_d2 = bench[label]["D2"]
        idx_rel = abs(c["index_critic"]["gap"] - bench_d2) / max(bench_d2, 1e-6)
        idx_z = abs(c["index_critic"]["gap"] - bench_d2) / max(c["index_critic"]["se"], 1e-12)
        verdict.append({"control": label, "benchmark D2": f"{bench_d2:.3e}",
                        "paired diff from exact": f"{diff:+.3e}",
                        "evaluation se": f"{paired_se:.1e}", "sd across seeds": f"{seed_sd:.1e}",
                        "separated here": "yes" if diff > 2 * sep_sd else "no",
                        "neural bias": f"{c_mean - bench_d2:+.3e}",
                        "index critic rel. error": f"{idx_rel:.2f}", "index critic z": f"{idx_z:.2f}",
                        "index critic within tol": "yes" if idx_z <= STAGE1_TOL["index_critic_z"] else "no"})
    report(verdict, ["control", "benchmark D2", "paired diff from exact", "evaluation se", "sd across seeds",
                     "separated here", "neural bias", "index critic rel. error", "index critic z",
                     "index critic within tol"],
           f"[stage 1] observed separation at n={big} (this evaluation sample, these controls)")
    resolved = [float(v["benchmark D2"]) for v in verdict if v["separated here"] == "yes"]
    unresolved = [float(v["benchmark D2"]) for v in verdict if v["separated here"] == "no"]
    floor = min(resolved) if resolved else float("nan")
    ordered = not unresolved or (resolved and max(unresolved) < floor)
    print(f"[stage 1] (c) on this evaluation sample the neural evaluator separated every control with "
          f"benchmark D2 >= {floor:.2e} and none below {max(unresolved) if unresolved else 0:.2e}; "
          f"ordering consistent: {'yes' if ordered else 'NO'}.")
    print(f"[stage 1]     This is an observed separation on one sample with {len(reps) - 2} controls, not a "
          f"detection threshold.  A detection rate needs independent evaluation samples and both error rates.")
    idx_ok = all(v["index critic within tol"] == "yes" for v in verdict)
    print(f"[stage 1] (d) index critic of the true form agrees with every control's benchmark within "
          f"{STAGE1_TOL['index_critic_z']:.0f} of its own standard errors: {'PASS' if idx_ok else 'FAIL'}")
    keep = {f"{n}|{l}": {k: v for k, v in d.items() if k != "fits"} for (n, l), d in raw.items()}
    return {"S": list(S), "r_star": r_star, "tolerances": STAGE1_TOL, "benchmark": bench,
            "rows": rows, "separation": verdict, "raw": keep,
            "pass": {"exact": bool(ok_exact), "quadrature": bool(ok_quad), "index_critic": bool(idx_ok),
                     "observed_separation_floor": floor, "ordering": bool(ordered)},
            "note": "the separation floor is observed on one evaluation sample over a fixed control list; "
                    "it is not a detection threshold and no false-positive rate was measured"}


# ----------------------------------------------------------------------------- stage 2: the learner
def stage2(args) -> dict:
    """Does the objective improvement found during training reproduce on rows the trainer never touched?

    Learned, initial and exact representations are scored by the same evaluator on the same audit rows, and
    those rows are the encoder's own audit split, disjoint from its fit and val rows by construction and by
    assertion.  "Initial" is the starting state of the restart that was actually selected.
    """
    S = tuple(args.stage1_S)
    out_rows, detail, records = [], {}, []
    for n in args.stage2_n:
        data = dgp.generate("F", n, replicate=STAGE_SEED + 1, phase="pilot", coef=dgp.coefficients())
        d_fit, d_val, d_audit = three_way(np.arange(n), STAGE_SEED + 1)
        overlap = assert_disjoint("stage 2", d_audit, fit=d_fit, val=d_val)
        st = pbn.build_standardizers(data, d_fit)
        rest = [b for b in range(dgp.N_BLOCKS) if b not in set(S)]
        fitted = pbn.train_representation(data, S, d_fit, d_val, st, (STAGE_SEED, n, 2))
        z_learned = embed_rows(fitted["enc"], rest, data, np.arange(n), st)
        enc0 = LinearRepresentation(fitted["specs"][0], data["X"].shape[1])
        enc0.load_state_dict(fitted["init_state"])
        z_init = embed_rows(enc0, rest, data, np.arange(n), st)
        z_exact = dgp.f_exact_representation(data, S)

        scored = {name: evaluate_representation(data, S, z, st, STAGE_SEED + n + 7, d_fit, d_val, d_audit,
                                                name=f"stage 2 {name}")
                  for name, z in (("learned", z_learned), ("at initialisation", z_init), ("exact", z_exact))}
        dec = pbn.decoupled_critic_fit(data, S, fitted, d_fit, d_val, d_audit, st, STAGE_SEED + n)
        short = pbn.critic_shortfall(data, S, fitted, d_fit, d_val, d_audit, st, STAGE_SEED + n + 3)
        pop = benchmark_rows(S, learned_z_rows(fitted, st))
        pop0 = benchmark_rows(S, learned_z_rows({"enc": enc0}, st))
        ckpt = save_checkpoint(f"stage2_n{n}", fitted, st, {"S": S, "data_seed": STAGE_SEED + 1}, RUN_ID,
                               ("stage2", int(n)))
        records.append({"n": n, "data_seed": STAGE_SEED + 1, "optimizer_seed": fitted["optimizer_seed"],
                        "restart": fitted["restart"], "split_ids": {"fit": split_id(d_fit), "val": split_id(d_val),
                                                                     "audit": split_id(d_audit)},
                        "forbidden_overlap": overlap, "checkpoint": ckpt, "complete": True})
        # Paired on the shared audit rows: the difference of the two per-row gaps, not two separate means
        # combined as if independent (round 7 review, R2).
        pair = paired_gap_difference(scored["at initialisation"], scored["learned"])
        imp, se = pair["mean"], pair["se"]
        pair_exact = paired_gap_difference(scored["learned"], scored["exact"])
        detail[n] = {"training": {k: fitted[k] for k in ("fit_gap", "val_gap", "encoder_updates", "epoch")},
                     "decoupled": dec, "scored": {k: v["refit_gap_audit"] for k, v in scored.items()},
                     "population_learned": {"D2": pop["D2"], "tau_Z": pop["tau_Z"]},
                     "population_init": {"D2": pop0["D2"], "tau_Z": pop0["tau_Z"]},
                     "shortfall": short, "training_mode": pbn.TRAINING_MODE,
                     "improvement": {"value": imp, "se": se, "rows": pair["n"],
                                     "reproduced": bool(imp > 2 * se)},
                     "vs_exact": {"value": pair_exact["mean"], "se": pair_exact["se"]}}
        for label, res in scored.items():
            out_rows.append({"n": n, "representation": label, "audit gap": f"{res['refit_gap_audit']:+.5f}",
                             "se": f"{res['refit_se_audit']:.5f}",
                             "population D2": f"{ {'learned': pop, 'at initialisation': pop0}.get(label, {'D2': 0.0})['D2']:.2e}",
                             "population tau_Z": f"{ {'learned': pop, 'at initialisation': pop0}.get(label, {'tau_Z': 1.0})['tau_Z']:+.4f}"})
        print(f"[stage 2] n={n}: training fit gap {fitted['fit_gap']:+.5f}, val gap {fitted['val_gap']:+.5f}, "
              f"{fitted['encoder_updates']} encoder updates, restart {fitted['restart']}; audit gap learned "
              f"{scored['learned']['refit_gap_audit']:+.5f} | init {scored['at initialisation']['refit_gap_audit']:+.5f}"
              f" | exact {scored['exact']['refit_gap_audit']:+.5f}; decoupled full fit audit gap "
              f"{dec['decoupled_gap_audit']:+.5f}; population of learned Z: D2 {pop['D2']:.2e}, tau_Z {pop['tau_Z']:+.4f}")
        print(f"          critic shortfall against a refit of each class: base {short['shortfall_R0']:+.5f}, "
              f"augmented {short['shortfall_R1']:+.5f}; objective in the loop "
              f"{short['objective_training_loop']:.5f} against {short['objective_refit']:.5f} refit")
    report(out_rows, ["n", "representation", "audit gap", "se", "population D2", "population tau_Z"],
           "[stage 2] the same evaluator on three representations, same audit rows")
    big = max(args.stage2_n)
    i, v = detail[big]["improvement"], detail[big]["vs_exact"]
    print(f"[stage 2] at n={big}: paired improvement over the selected restart's initialisation "
          f"{i['value']:+.5f} (paired se {i['se']:.5f} over {i['rows']} rows), "
          + ("reproduced on untouched audit rows" if i["reproduced"] else
             "NOT reproduced on untouched audit rows"))
    print(f"[stage 2] paired difference from the exact representation on the same rows: {v['value']:+.5f} "
          f"(se {v['se']:.5f})")
    return {"rows": out_rows, "detail": detail, "records": records,
            "reproduced_at_largest": bool(detail[big]["improvement"]["reproduced"])}


# ----------------------------------------------------------------------------- stage 3: estimation
def stage3(args) -> dict:
    """Representation frozen, estimation sample grown, identical folds for every adjustment."""
    S = tuple(args.stage1_S)
    n_rep, n_max = args.stage3_rep, max(args.stage3_est)
    data = dgp.generate("F", n_rep + n_max, replicate=STAGE_SEED + 2, phase="pilot", coef=dgp.coefficients())
    order = np.random.default_rng(STAGE_SEED + 2).permutation(n_rep + n_max)
    idx_rep, idx_est_all = order[:n_rep], order[n_rep:]
    L = learn_representation(data, S, idx_rep, data_seed=STAGE_SEED + 2, opt_seed=3)
    fitted, st = L["fitted"], L["st"]
    gap = pbn.refit_critics(data, S, fitted, L["fit"], L["val"], L["audit"], st, STAGE_SEED + n_rep)
    pop = benchmark_rows(S, learned_z_rows(fitted, st))
    assert_disjoint("stage 3 estimation rows", idx_est_all, representation=idx_rep)
    ckpt = save_checkpoint(f"stage3_rep{n_rep}", fitted, st, {"S": S, "data_seed": STAGE_SEED + 2}, RUN_ID,
                           ("stage3", int(n_rep)))
    print(f"[stage 3] learned representation: audit gap {gap['refit_gap_audit']:+.5f}, population D2 "
          f"{pop['D2']:.2e}, population tau_Z {pop['tau_Z']:+.4f}, checkpoint {ckpt}")

    # Population adjusted effect of every comparator, not just the learned representation.  Storing 1 for the
    # others made their "estimation error" equal to their total error, which is wrong: the raw adjustment does
    # not target 1 (round 7 review, R3).
    m = f_linear_maps()
    tau_targets = {
        "PROBE, exact representation": dgp.TRUE_ATE,
        "PROBE, learned representation": pop["tau_Z"],
        "raw (X, W)": benchmark_rows(S, np.vstack([m["X"][None, :]] + [m[f"W{b}"] for b in range(5)]))["tau_Z"],
        "oracle (X, U, C)": benchmark_rows(S, np.vstack([m["X"], m["U"], m["C"]]))["tau_Z"],
        "X only": benchmark_rows(S, m["X"][None, :])["tau_Z"]}
    print("[stage 3] population adjusted effects: "
          + ", ".join(f"{k} {v:+.6f}" for k, v in tau_targets.items()))

    rows, cells = [], []
    for n_est in args.stage3_est:
        idx = idx_est_all[:n_est]
        folds = two_folds(np.arange(n_est), STAGE_SEED + n_est)
        a, y = data["A"][idx], data["Y"][idx]
        feats = {"PROBE, exact representation": dgp.f_exact_representation(data, S)[idx],
                 "PROBE, learned representation": embed_rows(fitted["enc"], fitted["rest"], data, idx, st),
                 "raw (X, W)": dgp.raw_matrix(data)[idx], "oracle (X, U, C)": dgp.oracle_matrix(data)[idx],
                 "X only": data["X"][idx]}
        for label, f in feats.items():
            est = crossfit_theta(f, a, y, folds, STAGE_SEED + n_est + ADJ_SEED[label])
            tz = tau_targets[label]
            cells.append({"n_est": n_est, "adjustment": label, "theta": est["theta"], "se": est["se"],
                          "tau_Z": tz, "tau_Z_reason": None if tz is not None else "not a linear map of the noise",
                          "err_total": abs(est["theta"] - dgp.TRUE_ATE),
                          "err_adjustment": None if tz is None else abs(tz - dgp.TRUE_ATE),
                          "err_estimation": None if tz is None else abs(est["theta"] - tz)})
            rows.append({"n_est": n_est, "adjustment": label, "theta": f"{est['theta']:+.4f}",
                         "se": f"{est['se']:.4f}",
                         "tau_Z": "null" if tz is None else f"{tz:+.6f}",
                         "|theta - 1|": f"{abs(est['theta'] - dgp.TRUE_ATE):.4f}",
                         "|tau_Z - 1|": "null" if tz is None else f"{abs(tz - dgp.TRUE_ATE):.4f}",
                         "|theta - tau_Z|": "null" if tz is None else f"{abs(est['theta'] - tz):.4f}"})
    report(rows, ["n_est", "adjustment", "theta", "se", "tau_Z", "|theta - 1|", "|tau_Z - 1|",
                  "|theta - tau_Z|"],
           f"[stage 3] representation frozen on {n_rep} rows, estimation sample grown, shared folds")
    bad = [c for c in cells if c["tau_Z"] is not None
           and c["err_total"] > c["err_adjustment"] + c["err_estimation"] + 1e-9]
    print(f"[stage 3] triangle inequality |theta-1| <= |tau_Z-1| + |theta-tau_Z| holds in "
          f"{len(cells) - len(bad)}/{len(cells)} cells")
    return {"n_rep": n_rep, "rows": rows, "cells": cells, "tau_targets": tau_targets,
            "learned_population": {"D2": pop["D2"], "tau_Z": pop["tau_Z"]},
            "learned_audit_gap": gap["refit_gap_audit"], "checkpoint": ckpt,
            "triangle_violations": len(bad),
            "folds_note": "two folds shared by all adjustments"}


# ----------------------------------------------------------------------------- stage 4: convergence
def stage4(args) -> dict:
    """Representation sample grown under a controlled design, with data and optimizer seeds separated.

    Controls, following R4 of the round 7 review.  For each data seed the row roles are assigned once on the
    largest representation pool; the fit list is used as a nested prefix, and the validation, audit and
    estimation rows are byte-identical across sizes.  The optimizer seed no longer contains the sample size,
    so the same seed index starts every size from the same initial weights.  The nuisance seed depends only
    on the data seed and the adjustment, so changing the optimizer seed changes the representation and
    nothing else.  Preprocessing is either refit per size (default, honest for a standalone run at that size)
    or shared from the largest fit pool (`--fixed-preprocessing`, which also makes the *initial function*
    identical across sizes, not just the initial weights).

    The statistical replicate is the data seed.  Optimizer seeds measure the variability of one training
    procedure on fixed data and are never counted as independent replicates.  No pass verdict is computed: a
    decrease across a few finite sizes is reported with its spread, not called convergence.
    """
    S = tuple(args.stage1_S)
    n_est, sizes = args.stage4_est, sorted(args.stage4_rep)
    n_max = sizes[-1]
    rows, cells = [], []
    m = f_linear_maps()
    tau_raw = benchmark_rows(S, np.vstack([m["X"][None, :]] + [m[f"W{b}"] for b in range(5)]))["tau_Z"]
    for ds in range(args.data_seeds):
        data_seed = STAGE_SEED + 300 + ds
        data = dgp.generate("F", n_max + n_est, replicate=data_seed, phase="pilot", coef=dgp.coefficients())
        order = np.random.default_rng(data_seed).permutation(n_max + n_est)
        idx_rep_max, idx_est = order[:n_max], order[n_max:]
        assert_disjoint("stage 4 estimation rows", idx_est, representation=idx_rep_max)
        roles = assign_roles(idx_rep_max, data_seed)
        shared_st = pbn.build_standardizers(data, roles["fit_pool"]) if args.fixed_preprocessing else None
        folds = two_folds(np.arange(n_est), data_seed)
        a, y = data["A"][idx_est], data["Y"][idx_est]
        fixed = {}
        for label, f in (("PROBE, exact representation", dgp.f_exact_representation(data, S)[idx_est]),
                         ("raw (X, W)", dgp.raw_matrix(data)[idx_est]),
                         ("oracle (X, U, C)", dgp.oracle_matrix(data)[idx_est])):
            fixed[label] = crossfit_theta(f, a, y, folds, data_seed + ADJ_SEED[label])
        for n_rep in sizes:
            n_train = int(round(len(roles["fit_pool"]) * n_rep / n_max))
            for os_ in range(args.opt_seeds):
                L = learn_representation(data, S, idx_rep_max, data_seed, os_, roles=roles, n_train=n_train,
                                         st=shared_st)
                fitted, st = L["fitted"], L["st"]
                gap = pbn.refit_critics(data, S, fitted, L["fit"], L["val"], L["audit"], st,
                                        data_seed + n_rep + os_)
                pop = benchmark_rows(S, learned_z_rows(fitted, st))
                z = embed_rows(fitted["enc"], fitted["rest"], data, idx_est, st)
                # the nuisance seed depends on the data seed and the adjustment only, never on os_
                est = crossfit_theta(z, a, y, folds,
                                     data_seed + ADJ_SEED["PROBE, learned representation"])
                ck = save_checkpoint(f"stage4_ds{ds}_n{n_rep}_os{os_}", fitted, st,
                                     {"S": S, "data_seed": data_seed, "n_rep": n_rep}, RUN_ID,
                                     (int(data_seed), int(n_rep), int(os_)))
                cell = {"data_seed": data_seed, "n_rep": n_rep, "n_train_rows": int(n_train),
                        "n_rows_used": int(n_train + len(roles["val"])), "opt_seed": os_,
                        "optimizer_seed_value": fitted["optimizer_seed"], "restart": fitted["restart"],
                        "init_hash": ck["init_hash"], "encoder_updates": fitted["encoder_updates"],
                        "audit_gap": gap["refit_gap_audit"], "audit_se": gap["refit_se_audit"],
                        "pop_D2": pop["D2"], "pop_tau_Z": pop["tau_Z"],
                        "theta": est["theta"], "theta_se": est["se"],
                        "err_total": abs(est["theta"] - dgp.TRUE_ATE),
                        "err_representation": abs(pop["tau_Z"] - dgp.TRUE_ATE),
                        "err_estimation": abs(est["theta"] - pop["tau_Z"]),
                        "err_raw": abs(fixed["raw (X, W)"]["theta"] - dgp.TRUE_ATE),
                        "tau_raw": tau_raw,
                        "err_exact": abs(fixed["PROBE, exact representation"]["theta"] - dgp.TRUE_ATE),
                        "err_oracle": abs(fixed["oracle (X, U, C)"]["theta"] - dgp.TRUE_ATE),
                        "split_ids": {k: split_id(L[k]) for k in ("fit", "val", "audit")},
                        "estimation_split_id": split_id(idx_est), "checkpoint": ck, "complete": True}
                cells.append(cell)
                print(f"[stage 4] ds{ds} n_rep={n_rep:6d} ({n_train} train rows) os{os_}: audit gap "
                      f"{cell['audit_gap']:+.5f}, pop D2 {pop['D2']:.1e}, |tau_Z-1| "
                      f"{cell['err_representation']:.4f}, |theta-tau_Z| {cell['err_estimation']:.4f}, "
                      f"|theta-1| {cell['err_total']:.4f} (raw {cell['err_raw']:.4f})")

    # controls actually achieved
    by_size = {n: [c for c in cells if c["n_rep"] == n] for n in sizes}
    # Initial weights are compared within the same restart index.  Which restart ends up selected is allowed
    # to differ between sizes; requiring the *selected* restarts to match would be a different claim
    # (round 7 review, PRD-5).
    controls_by_data_seed = {}
    for data_seed in sorted({int(c["data_seed"]) for c in cells}):
        selected = [c for c in cells if int(c["data_seed"]) == data_seed]
        groups = {}
        for c in selected:
            groups.setdefault((c["opt_seed"], c["restart"]), set()).add(c["init_hash"])
        controls_by_data_seed[str(data_seed)] = {
            "identical_initial_weights": all(len(v) == 1 for v in groups.values()),
            "val_rows_fixed": len({c["split_ids"]["val"] for c in selected}) == 1,
            "audit_rows_fixed": len({c["split_ids"]["audit"] for c in selected}) == 1,
            "estimation_rows_fixed": len({c["estimation_split_id"] for c in selected}) == 1,
        }
    controls = {name: all(x[name] for x in controls_by_data_seed.values())
                for name in ("identical_initial_weights", "val_rows_fixed", "audit_rows_fixed",
                             "estimation_rows_fixed")}
    print(f"\n[stage 4] controls: identical initial weights across sizes for a fixed seed: "
          f"{controls['identical_initial_weights']}; "
          f"validation rows fixed: {controls['val_rows_fixed']}; "
          f"audit rows fixed: {controls['audit_rows_fixed']}; "
          f"estimation rows fixed: {controls['estimation_rows_fixed']}; "
          f"preprocessing: {'shared from the largest fit pool' if args.fixed_preprocessing else 'refit per size'}")

    summary = []
    for n_rep in sizes:
        c = by_size[n_rep]
        tot = np.array([x["err_total"] for x in c]); rep = np.array([x["err_representation"] for x in c])
        estm = np.array([x["err_estimation"] for x in c]); d2 = np.array([x["pop_D2"] for x in c])
        by_ds = {}
        for x in c:
            by_ds.setdefault(x["data_seed"], []).append(x["err_representation"])
        rep_by_ds = np.array([np.median(v) for _, v in sorted(by_ds.items())])
        total_by_ds = np.array([np.median([x["err_total"] for x in c if x["data_seed"] == ds])
                                for ds in sorted(by_ds)])
        raw_by_ds = np.array([c[[x["data_seed"] for x in c].index(ds)]["err_raw"] for ds in sorted(by_ds)])
        within = math.sqrt(float(np.mean([np.var(v, ddof=1) if len(v) > 1 else 0.0 for v in by_ds.values()])))
        between = (math.sqrt(max(float(np.var([np.mean(v) for v in by_ds.values()], ddof=1))
                                 - within ** 2 / max(args.opt_seeds, 1), 0.0))
                   if len(by_ds) > 1 else float("nan"))
        summary.append({"n_rep": n_rep, "train rows": c[0]["n_train_rows"], "cells": len(c),
                        "median |theta-1|": f"{np.median(total_by_ds):.4f}",
                        "p90": f"{np.quantile(total_by_ds, 0.9):.4f}",
                        "median |tau_Z-1|": f"{np.median(rep_by_ds):.4f}",
                        "p90 |tau_Z-1|": f"{np.quantile(rep_by_ds, 0.9):.4f}",
                        "pooled-cell p90 |tau_Z-1| (diagnostic)": f"{np.quantile(rep, 0.9):.4f}",
                        "median |theta-tau_Z|": f"{np.median(estm):.4f}",
                        "median pop D2": f"{np.median(d2):.1e}",
                        "sd of |tau_Z-1| within data seed": f"{within:.4f}",
                        "sd between data seeds": "n/a" if math.isnan(between) else f"{between:.4f}",
                        "beats raw": f"{int((total_by_ds < raw_by_ds).sum())}/{len(total_by_ds)} data seeds",
                        "raw median": f"{np.median(raw_by_ds):.4f}"})
    report(summary, list(summary[0].keys()),
           f"[stage 4] {args.data_seeds} data seeds x {args.opt_seeds} optimizer seeds, estimation rows fixed "
           f"at {n_est}")
    print("[stage 4] the two spread columns are for |tau_Z - 1|, which involves no nuisance randomness; the "
          "between-seed column subtracts the within-seed component and is a moment estimate, not a test.")
    return {"n_est": n_est, "sizes": sizes, "data_seeds": args.data_seeds, "opt_seeds": args.opt_seeds,
            "fixed_preprocessing": bool(args.fixed_preprocessing),
            "controls": controls, "controls_by_data_seed": controls_by_data_seed,
            "cells": cells, "summary": summary, "tau_raw": tau_raw,
            "note": "no pass verdict: finite-size decrease reported with spread, not claimed as convergence"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="1,2,3,4")
    ap.add_argument("--stage1-S", type=int, nargs="+", default=[1])
    ap.add_argument("--stage1-n", type=int, nargs="+", default=[12000, 48000])
    ap.add_argument("--stage1-seeds", type=int, default=5, help="optimizer seeds per critic fit in stage 1")
    ap.add_argument("--stage2-n", type=int, nargs="+", default=[4000, 16000, 48000])
    ap.add_argument("--stage3-rep", type=int, default=24000)
    ap.add_argument("--stage3-est", type=int, nargs="+", default=[2000, 8000, 32000])
    ap.add_argument("--stage4-rep", type=int, nargs="+", default=[3000, 12000, 48000])
    ap.add_argument("--stage4-est", type=int, default=32000)
    ap.add_argument("--data-seeds", type=int, default=1)
    ap.add_argument("--opt-seeds", type=int, default=5)
    ap.add_argument("--fixed-preprocessing", action="store_true",
                    help="build the standardizers once from the largest fit pool and share them across "
                         "sizes, so the initial representation function is identical and not only the "
                         "initial weights")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--training-mode", choices=["variant", "manuscript"], default="variant",
                    help="'manuscript' gives each critic its own parameters, minimises max(R0-R1,0), and "
                         "selects checkpoints on that objective; 'variant' is the shared-trunk arrangement")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="../results/dgp_f_staged.json")
    args = ap.parse_args()

    global RUN_ID
    RUN_ID = args.run_id or time.strftime("%Y%m%dT%H%M%S")
    torch.set_num_threads(args.threads)
    pbn.ENCODER_FACTORY = LinearRepresentation
    pbn.Z_DIM = LINEAR_Z_DIM
    pbn.TRAINING_MODE = args.training_mode
    criteria_path = Path(__file__).with_name("frozen_criteria.json")
    criteria_sha = prov.file_sha256(criteria_path)
    source_manifest = prov.snapshot_run_sources(
        Path(args.out).with_suffix("") / f"run-{RUN_ID}",
        {name: Path(__file__).with_name(name) for name in
         ("staged_dgp_f.py", "probe_brier_neural.py", "probe_dgp_ae.py", "provenance.py",
          "frozen_criteria.json")})
    out = {"provenance": prov.header(
        run_id=RUN_ID, script="staged_dgp_f.py", args=vars(args), training_mode=pbn.TRAINING_MODE,
        code_hash=source_manifest["bundle_sha256"][:16],
        seeds={"stage_seed": STAGE_SEED, "data": [STAGE_SEED + 300 + i for i in range(args.data_seeds)],
               "optimizer": list(range(args.opt_seeds)), "adjustment_offsets": ADJ_SEED},
        extra={"protocol_id": PROTOCOL_ID, "criteria_sha256": criteria_sha,
               "source_manifest": source_manifest, "torch": torch.__version__, "numpy": np.__version__,
               "row_usage": "fit: encoder+critic training; val: checkpoint/restart selection; "
                            "audit: evaluation only; estimation rows: disjoint from all three"})}
    print(f"design F staged study | linear encoder, Z in R^{LINEAR_Z_DIM} | propensity bound "
          f"{pbn.PROPENSITY_BOUND} | training mode {pbn.TRAINING_MODE} | run {RUN_ID} | "
          f"code {out['provenance']['code_hash']}")
    wanted = [s.strip() for s in args.stages.split(",")]
    t0 = time.time()
    for name, fn in (("1", stage1), ("2", stage2), ("3", stage3), ("4", stage4)):
        if name in wanted:
            out[f"stage{name}"] = fn(args)
            Path(args.out).write_text(json.dumps(out, indent=1, default=float))
            print(f"[stage {name}] done at {time.time() - t0:.0f}s, written to {args.out}")
    out["provenance"]["complete"] = True
    out["provenance"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
