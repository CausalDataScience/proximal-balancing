"""Small-data structured-proxy SCM experiments.

This module is deliberately separate from the neural PROBE implementation.  It
provides a structured observed-moment positive control.  Neither representation
fitting nor screening receives latent variables, outcomes, causal truth, or
valid-block tags.  A manuscript-objective Brier arm is intentionally not claimed
until its separate implementation and tests exist.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm


ALL_SPLITS = tuple(
    s for k in range(1, 5) for s in itertools.combinations(range(5), k)
)
ROLE_NAMES = ("representation", "screen", "nuisance", "evaluation")
SCREEN_T = 3.5
ROOT_TIE_TOL = 1e-12
BOOTSTRAP_DRAWS = 5_000


@dataclass(frozen=True)
class SCMSpec:
    name: str
    treatment_i: float
    proxy_sd: float | tuple[float, float, float, float, float]
    outcome_u: float
    bad_i: float
    effect_u: float = 0.0
    observation: str = "linear"
    treatment_link: str = "probit"
    outcome_noise: str = "standard"


CONTROL_SPECS = {
    "q2_linear": SCMSpec("q2_linear", 1.5, 0.9, 2.7, -0.6),
    "q6_heterogeneous": SCMSpec(
        "q6_heterogeneous", 1.5, 0.9, 2.7, -0.6, effect_u=0.3
    ),
    "q7_sinh": SCMSpec("q7_sinh", 1.5, 0.9, 2.7, -0.6, observation="sinh"),
}

# Candidate family recommended from population calculations before any new
# finite-sample outputs.  It is not frozen until copied into a protocol file.
SMALL_DATA_CANDIDATES = {
    "f1_probit_linear": SCMSpec("f1_probit_linear", 2.5, 0.85, 2.5, -0.6),
    "f2_outcome_variance": SCMSpec(
        "f2_outcome_variance", 2.5, 0.85, 2.5, -0.6, outcome_noise="heteroscedastic_small"
    ),
    "f3_heterogeneous": SCMSpec("f3_heterogeneous", 2.5, 0.85, 2.7, -0.6, effect_u=0.3),
    "f4_heteroscedastic": SCMSpec(
        "f4_heteroscedastic", 2.5, (0.8, 0.9, 1.0, 0.85, 0.95), 2.5, -0.6
    ),
    "f5_sinh": SCMSpec("f5_sinh", 2.5, 0.85, 2.5, -0.6, observation="sinh"),
}

LOW_NOISE_CANDIDATES = {
    "g1_low_noise": SCMSpec("g1_low_noise", 2.5, 0.85, 2.5, -0.6, outcome_noise="constant_small"),
    "g2_heteroscedastic_noise": SCMSpec(
        "g2_heteroscedastic_noise", 2.5, 0.85, 2.5, -0.6, outcome_noise="heteroscedastic_small"
    ),
    "g3_heterogeneous_low_noise": SCMSpec(
        "g3_heterogeneous_low_noise", 2.5, 0.85, 2.7, -0.6, effect_u=0.3, outcome_noise="constant_small"
    ),
    "g4_proxy_heteroscedastic_low_noise": SCMSpec(
        "g4_proxy_heteroscedastic_low_noise", 2.5, (0.8, 0.9, 1.0, 0.85, 0.95), 2.5, -0.6,
        outcome_noise="constant_small",
    ),
    "g5_sinh_low_noise": SCMSpec(
        "g5_sinh_low_noise", 2.5, 0.85, 2.5, -0.6, observation="sinh", outcome_noise="constant_small"
    ),
}


# ----------------------------------------------------------------------------- the frozen SCM family
# The five settings confirmed on 2026-09-12 are the paper's SCM family from here on.  They are the same
# objects as LOW_NOISE_CANDIDATES above; this block fixes their public names and records the development key
# each one was confirmed under, so that a stored result written before the renaming stays traceable.
#
# Every member shares one causal structure.  U is the latent confounder, I and C are observed inside the
# proxy blocks, and U, I, C and all error terms are independent standard normals:
#
#     X   = U + 2 eps_X                       (observed covariate)
#     M_j = U + sigma_j eps_j,  j = 0..4      (latent measurements)
#     W_0 = (I, C, T(M_0))                    (proxy block 0)
#     W_j = T(M_j),            j = 1, 2, 3    (clean proxy blocks)
#     W_4 = T(M_4 - 0.6 I)                    (the contaminated block: bad_i = -0.6)
#     P(A = 1 | U, I, C, X) = 0.1 + 0.8 Phi(U + 2.5 I + 0.2 X + 0.3 C)
#     Y(a) = a - outcome_u * U + 0.2 X - 0.5 C + outcome noise
#
# The true average treatment effect is 1 in all five.  The unadjusted contrast is negative in all five, so
# each one carries the sign reversal the study is built around.
SCM_FAMILY = {
    "SCM-1": LOW_NOISE_CANDIDATES["g1_low_noise"],
    "SCM-2": LOW_NOISE_CANDIDATES["g2_heteroscedastic_noise"],
    "SCM-3": LOW_NOISE_CANDIDATES["g3_heterogeneous_low_noise"],
    "SCM-4": LOW_NOISE_CANDIDATES["g4_proxy_heteroscedastic_low_noise"],
    "SCM-5": LOW_NOISE_CANDIDATES["g5_sinh_low_noise"],
}

# What each member varies away from SCM-1, and the development key it was confirmed under.
SCM_FAMILY_NOTES = {
    "SCM-1": {"legacy_key": "g1_low_noise", "varies": "nothing; the base case",
              "outcome_noise": "0.3 eps_Y", "proxy_sd": "0.85 for every block",
              "observation": "T(x) = x", "outcome_u": 2.5, "effect_u": 0.0},
    "SCM-2": {"legacy_key": "g2_heteroscedastic_noise",
              "varies": "outcome noise depends on the latent confounder",
              "outcome_noise": "(0.3 + 0.2 |U|) eps_Y", "proxy_sd": "0.85 for every block",
              "observation": "T(x) = x", "outcome_u": 2.5, "effect_u": 0.0},
    "SCM-3": {"legacy_key": "g3_heterogeneous_low_noise",
              "varies": "the treatment effect itself depends on the latent confounder",
              "outcome_noise": "0.3 eps_Y", "proxy_sd": "0.85 for every block",
              "observation": "T(x) = x", "outcome_u": 2.7, "effect_u": 0.3},
    "SCM-4": {"legacy_key": "g4_proxy_heteroscedastic_low_noise",
              "varies": "the measurement blocks differ in how noisy they are",
              "outcome_noise": "0.3 eps_Y", "proxy_sd": "(0.8, 0.9, 1.0, 0.85, 0.95)",
              "observation": "T(x) = x", "outcome_u": 2.5, "effect_u": 0.0},
    "SCM-5": {"legacy_key": "g5_sinh_low_noise",
              "varies": "the proxies are seen through a nonlinear lens",
              "outcome_noise": "0.3 eps_Y", "proxy_sd": "0.85 for every block",
              "observation": "T(x) = sinh(x)", "outcome_u": 2.5, "effect_u": 0.0},
}

LEGACY_TO_SCM = {v["legacy_key"]: k for k, v in SCM_FAMILY_NOTES.items()}


def resolve_scm(name: str) -> SCMSpec:
    """Accept either the public name (SCM-3) or the development key (g3_heterogeneous_low_noise)."""
    if name in SCM_FAMILY:
        return SCM_FAMILY[name]
    if name in LEGACY_TO_SCM:
        return SCM_FAMILY[LEGACY_TO_SCM[name]]
    raise KeyError(f"{name!r} is not one of {sorted(SCM_FAMILY)} or {sorted(LEGACY_TO_SCM)}")


def source_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def role_sha256(data: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(data):
        digest.update(key.encode())
        digest.update(array_sha256(data[key]).encode())
    return digest.hexdigest()


def score_role_ledger(
    representation_folds: tuple[int, ...],
    screen_folds: tuple[int, ...],
    nuisance_fold: int,
    evaluation_fold: int,
    evaluation_n: int,
) -> dict:
    """Record and enforce row honesty for one score vector."""
    training = set(representation_folds) | set(screen_folds) | {nuisance_fold}
    if evaluation_fold in training:
        raise ValueError("evaluation fold overlaps representation, screen, or nuisance rows")
    return {
        "representation_folds": list(representation_folds),
        "screen_folds": list(screen_folds),
        "nuisance_fold": nuisance_fold,
        "evaluation_fold": evaluation_fold,
        "unique_evaluation_rows": evaluation_n,
    }


def average_score_vectors(scores: list[np.ndarray]) -> np.ndarray:
    """Average representations for the same evaluation units without duplicating rows."""
    if not scores:
        raise ValueError("at least one score vector is required")
    lengths = {len(np.asarray(score)) for score in scores}
    if len(lengths) != 1:
        raise ValueError("score vectors for one evaluation fold must have identical row counts")
    return np.mean(np.vstack([np.asarray(score) for score in scores]), axis=0)


def generate_role(spec: SCMSpec, n: int, seed: int) -> dict[str, np.ndarray]:
    """Generate one independent role.  Truth fields are evaluator-only."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 9))
    u, inst, cause, ex = z[:, :4].T
    proxy_sd = np.asarray(spec.proxy_sd)
    latent_m = u[:, None] + proxy_sd * z[:, 4:9]
    observed_m = latent_m.copy()
    observed_m[:, 4] += spec.bad_i * inst
    if spec.observation == "sinh":
        observed_m = np.sinh(observed_m)
    elif spec.observation != "linear":
        raise ValueError(f"unknown observation map {spec.observation}")
    x = u + 2.0 * ex
    index = u + spec.treatment_i * inst + 0.2 * x + 0.3 * cause
    if spec.treatment_link == "probit":
        link = norm.cdf(index)
    elif spec.treatment_link == "logistic":
        link = 1.0 / (1.0 + np.exp(-index))
    else:
        raise ValueError(f"unknown treatment link {spec.treatment_link}")
    p = 0.1 + 0.8 * link
    a = rng.binomial(1, p)
    y = a * (1.0 + spec.effect_u * u) - spec.outcome_u * u + 0.2 * x - 0.5 * cause
    outcome_error = rng.standard_normal(n)
    if spec.outcome_noise == "standard":
        scale = 1.0
    elif spec.outcome_noise == "constant_small":
        scale = 0.3
    elif spec.outcome_noise == "heteroscedastic_small":
        scale = 0.3 + 0.2 * np.abs(u)
    else:
        raise ValueError(f"unknown outcome noise {spec.outcome_noise}")
    y = y + scale * outcome_error
    return {
        "X": x,
        "I": inst,
        "C": cause,
        "M": observed_m,
        "A": a.astype(float),
        "Y": y,
        "U_eval_only": u,
        "p_eval_only": p,
    }


@dataclass
class Transform:
    observation: str
    sorted_columns: list[np.ndarray] | None = None
    scales: np.ndarray | None = None

    @classmethod
    def fit(cls, data: dict[str, np.ndarray], observation: str) -> "Transform":
        if observation == "linear":
            return cls(observation, None, np.ones(5))
        sorted_columns = [np.sort(data["M"][:, j]) for j in range(5)]
        g = cls._normal_scores(data["M"], sorted_columns)
        scales = np.array([np.cov(g[:, j], data["X"], ddof=0)[0, 1] for j in range(5)])
        if np.any(np.abs(scales) < 1e-8):
            raise RuntimeError("rank-Gaussian scale is numerically zero")
        return cls(observation, sorted_columns, scales)

    @staticmethod
    def _normal_scores(values: np.ndarray, sorted_columns: list[np.ndarray]) -> np.ndarray:
        out = np.empty_like(values)
        for j, ref in enumerate(sorted_columns):
            rank = np.searchsorted(ref, values[:, j], side="right")
            rank = np.clip(rank, 1, len(ref))
            out[:, j] = norm.ppf(rank / (len(ref) + 1.0))
        return out

    def apply(self, data: dict[str, np.ndarray]) -> np.ndarray:
        if self.observation == "linear":
            return data["M"].copy()
        assert self.sorted_columns is not None and self.scales is not None
        return self._normal_scores(data["M"], self.sorted_columns) / self.scales


def observed_view(
    data: dict[str, np.ndarray], transform: Transform, *, include_outcome: bool = False
) -> dict[str, np.ndarray]:
    """Strip evaluator-only fields before any learner is called."""
    keys = ["X", "I", "C", "A"] + (["Y"] if include_outcome else [])
    return {k: data[k] for k in keys} | {"K": transform.apply(data)}


def split_category(split: tuple[int, ...]) -> str:
    """Evaluation label only; never used by fitting, screening, or aggregation."""
    if 0 in split:
        return "S0"
    if split == (4,):
        return "bad_singleton"
    if 4 in split:
        return "mixed"
    return "clean"


def _rest(split: tuple[int, ...]) -> list[int]:
    return [j for j in range(5) if j not in split]


def representation(data: dict[str, np.ndarray], split: tuple[int, ...], r: float | None) -> np.ndarray:
    """Metadata-equivalent original topology, without consulting validity tags.

    Block 0 is the only block whose observed schema contains I and C.  Therefore
    I and C are available exactly when block 0 is in the complement.  This is an
    availability rule, not a hard-coded validity decision.
    """
    mean_k = data["K"][:, _rest(split)].mean(axis=1)
    if 0 in split:
        return np.column_stack([data["X"], mean_k])
    if r is None:
        raise ValueError("r is required when the typed I channel is available")
    return np.column_stack([data["X"], data["C"], mean_k + r * data["I"]])


def heldout(data: dict[str, np.ndarray], split: tuple[int, ...]) -> np.ndarray:
    cols: list[np.ndarray] = []
    for j in split:
        if j == 0:
            cols.extend([data["I"], data["C"], data["K"][:, 0]])
        else:
            cols.append(data["K"][:, j])
    return np.column_stack(cols)


def _residual(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), z])
    return y - design @ np.linalg.lstsq(design, y, rcond=None)[0]


def partial_covariance(a: np.ndarray, target: np.ndarray, z: np.ndarray) -> np.ndarray:
    values = np.column_stack([a, target, z])
    values -= values.mean(axis=0)
    cov = values.T @ values / len(values)
    q = 1 + target.shape[1]
    return cov[0, 1:q] - cov[0, q:] @ np.linalg.solve(cov[q:, q:], cov[q:, 1:q])


def moment_root(data: dict[str, np.ndarray], split: tuple[int, ...]) -> dict:
    if 0 in split:
        return {"r": None, "roots": [], "objective": None}
    target = heldout(data, split)
    # Precompute one covariance matrix.  Each Z_r is a linear transform of
    # (X,C,R,I), so the grid search never revisits the n rows.
    raw = np.column_stack(
        [data["A"], target, data["X"], data["C"], data["K"][:, _rest(split)].mean(axis=1), data["I"]]
    )
    raw -= raw.mean(axis=0)
    covariance = raw.T @ raw / len(raw)
    q = 1 + target.shape[1]
    aux = covariance[q:, q:]

    def moments(r: float) -> np.ndarray:
        linear = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, r]])
        var_z = linear.T @ aux @ linear
        inv_z = np.linalg.inv(var_z)
        cov_a_z = covariance[0, q:] @ linear
        cov_z_t = linear.T @ covariance[q:, 1:q]
        cov_res = covariance[0, 1:q] - cov_a_z @ inv_z @ cov_z_t
        var_a = covariance[0, 0] - cov_a_z @ inv_z @ cov_a_z
        cov_t_z = covariance[1:q, q:] @ linear
        var_t = np.diag(covariance[1:q, 1:q] - cov_t_z @ inv_z @ cov_z_t)
        return cov_res / np.sqrt(np.maximum(var_a * var_t, 1e-16))

    def objective(r: float) -> float:
        value = moments(r)
        return float(value @ value)

    grid = np.linspace(-1.0, 1.0, 201)
    values = np.array([objective(r) for r in grid])
    local = [i for i in range(1, len(grid) - 1) if values[i] <= values[i - 1] and values[i] <= values[i + 1]]
    candidates = [(-1.0, objective(-1.0)), (1.0, objective(1.0))]
    for i in local:
        fitted = minimize_scalar(objective, bounds=(grid[i - 1], grid[i + 1]), method="bounded")
        candidates.append((float(fitted.x), float(fitted.fun)))
    # Deduplicate numerical copies while keeping the exact, truth-free tie rule.
    unique: dict[float, float] = {}
    for r, value in candidates:
        key = round(r, 10)
        unique[key] = min(value, unique.get(key, float("inf")))
    roots = sorted(unique)
    objectives = np.array([unique[r] for r in roots])
    eligible = np.flatnonzero(objectives <= objectives.min() + ROOT_TIE_TOL)
    chosen = min(eligible, key=lambda i: (abs(roots[i]), roots[i]))
    return {
        "r": roots[chosen],
        "roots": roots,
        "root_objectives": objectives.tolist(),
        "objective": float(objectives[chosen]),
    }


def screen(data: dict[str, np.ndarray], split: tuple[int, ...], r: float | None) -> dict:
    z = representation(data, split, r)
    a_res = _residual(data["A"], z)
    target = heldout(data, split)
    statistics = []
    for j in range(target.shape[1]):
        product = a_res * _residual(target[:, j], z)
        statistics.append(float(product.mean() / (product.std(ddof=1) / math.sqrt(len(product)))))
    max_abs = max(abs(x) for x in statistics)
    return {"pass": bool(max_abs <= SCREEN_T), "statistics": statistics, "max_abs": max_abs}


def _fit_scaled_probit(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(a)), z])

    def risk(beta: np.ndarray) -> float:
        p = 0.1 + 0.8 * norm.cdf(np.clip(design @ beta, -8.0, 8.0))
        return float(-np.mean(a * np.log(p) + (1.0 - a) * np.log(1.0 - p)))

    fitted = minimize(risk, np.zeros(design.shape[1]), method="BFGS", options={"maxiter": 120})
    if not fitted.success and not np.isfinite(fitted.fun):
        raise RuntimeError(f"scaled probit fit failed: {fitted.message}")
    return fitted.x


def _fit_brier_probit(
    z: np.ndarray,
    a: np.ndarray,
    starts: Iterable[np.ndarray] | None = None,
    *,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Independently minimise empirical bounded-probit Brier risk."""
    design = np.column_stack([np.ones(len(a)), z])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        return _brier_risk_gradient_from_design(beta, design, a)

    def risk(beta: np.ndarray) -> float:
        return objective(beta)[0]

    starts = list(starts) if starts is not None else [np.zeros(z.shape[1] + 1)]
    attempts = []
    candidates = []
    for start in starts:
        start = np.asarray(start)
        candidates.append((risk(start), start.copy(), "unoptimized_start"))
        fitted = minimize(objective, start, jac=True, method="BFGS", options={"maxiter": 120})
        attempts.append({"success": bool(fitted.success), "status": int(fitted.status), "message": str(fitted.message), "risk": float(fitted.fun)})
        candidates.append((float(fitted.fun), fitted.x.copy(), "optimized"))
    best_risk, best_beta, source = min(candidates, key=lambda item: item[0])
    if not np.isfinite(best_risk):
        raise RuntimeError("Brier critic fit failed")
    info = {"risk": float(best_risk), "chosen_source": source, "attempts": attempts}
    return (best_beta, info) if return_info else best_beta


def _brier_risk_gradient(beta: np.ndarray, z: np.ndarray, a: np.ndarray) -> tuple[float, np.ndarray]:
    design = np.column_stack([np.ones(len(a)), z])
    return _brier_risk_gradient_from_design(beta, design, a)


def _brier_risk_gradient_from_design(
    beta: np.ndarray, design: np.ndarray, a: np.ndarray
) -> tuple[float, np.ndarray]:
    eta = design @ beta
    clipped = np.clip(eta, -8.0, 8.0)
    p = 0.1 + 0.8 * norm.cdf(clipped)
    derivative = 0.8 * norm.pdf(clipped) * ((eta > -8.0) & (eta < 8.0))
    residual = p - a
    risk = float(np.mean(residual**2))
    gradient = 2.0 * design.T @ (residual * derivative) / len(a)
    return risk, gradient


def _brier_rows(beta: np.ndarray, z: np.ndarray, a: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(a)), z])
    p = 0.1 + 0.8 * norm.cdf(np.clip(design @ beta, -8.0, 8.0))
    return (a - p) ** 2


def brier_select(
    data: dict[str, np.ndarray], split: tuple[int, ...], seed: int, grid_points: int = 101
) -> dict:
    """Restricted manuscript-objective encoder selection on the same D rows.

    q0 and q1 independently minimise their empirical Brier risks on D.  The
    augmented optimiser includes the fitted base critic with zero held-out
    coefficients as a start, enforcing the nested-class comparison numerically.
    This is a restricted linear/probit critic class, not the neural experiment.
    """
    candidates: list[float | None] = [None] if 0 in split else list(np.linspace(-1.0, 1.0, grid_points))
    rows = []
    target = heldout(data, split)
    for r in candidates:
        z = representation(data, split, r)
        beta0, info0 = _fit_brier_probit(z, data["A"], return_info=True)
        lifted = np.concatenate([beta0, np.zeros(target.shape[1])])
        beta1, info1 = _fit_brier_probit(
            np.column_stack([z, target]), data["A"], starts=[lifted, np.zeros(len(lifted))], return_info=True
        )
        loss0 = _brier_rows(beta0, z, data["A"])
        loss1 = _brier_rows(beta1, np.column_stack([z, target]), data["A"])
        signed_gap = float(np.mean(loss0 - loss1))
        if signed_gap < -1e-10:
            raise RuntimeError(f"nested empirical Brier gap is negative: {signed_gap}")
        rows.append(
            {
                "r": r,
                "signed_empirical_gap": signed_gap,
                "objective": max(signed_gap, 0.0),
                "beta0": beta0,
                "beta1": beta1,
                "critic0_fit": info0,
                "critic1_fit": info1,
            }
        )
    values = np.array([row["objective"] for row in rows])
    eligible = np.flatnonzero(values <= values.min() + ROOT_TIE_TOL)
    chosen = min(
        eligible,
        key=lambda i: (
            abs(rows[i]["r"]) if rows[i]["r"] is not None else 0.0,
            rows[i]["r"] if rows[i]["r"] is not None else 0.0,
        ),
    )
    selected = rows[chosen]
    return {
        "r": selected["r"],
        "signed_empirical_gap": selected["signed_empirical_gap"],
        "objective": selected["objective"],
        "grid_points": grid_points,
        "empirical_n": len(data["A"]),
        "beta0": selected["beta0"],
        "beta1": selected["beta1"],
        "critic0_fit": selected["critic0_fit"],
        "critic1_fit": selected["critic1_fit"],
        "objective_min_count": int(len(eligible)),
    }


def brier_audit(
    data: dict[str, np.ndarray], split: tuple[int, ...], selected: dict
) -> dict:
    z = representation(data, split, selected["r"])
    target = heldout(data, split)
    loss0 = _brier_rows(selected["beta0"], z, data["A"])
    loss1 = _brier_rows(selected["beta1"], np.column_stack([z, target]), data["A"])
    delta = loss0 - loss1
    gap, se = float(delta.mean()), float(delta.std(ddof=1) / math.sqrt(len(delta)))
    return {
        "signed_gap": gap,
        "clipped_gap": max(gap, 0.0),
        "se": se,
        "pass": bool(max(gap, 0.0) <= 2.0 * se),
        "rule": "clipped_gap_le_2se_zero_null_nonrejection",
    }


def run_brier_diagnostic_replicate(spec: SCMSpec, total_n: int, seed_base: int) -> dict:
    """All-30 restricted-Brier selection, independent screen, and ATE aggregation."""
    if spec.treatment_link != "probit":
        raise ValueError("restricted Brier critic is frozen to the probit-link settings")
    n = total_n // 4
    raw_d = generate_role(spec, n, seed_base)
    raw_s = generate_role(spec, n, seed_base + 1)
    raw_n = generate_role(spec, n, seed_base + 2)
    raw_e = generate_role(spec, n, seed_base + 3)
    transform = Transform.fit(raw_d, spec.observation)
    d, s = observed_view(raw_d, transform), observed_view(raw_s, transform)
    nuisance = observed_view(raw_n, transform, include_outcome=True)
    evaluation = observed_view(raw_e, transform, include_outcome=True)
    rows = []
    survivors = []
    for i, split in enumerate(ALL_SPLITS):
        selected = brier_select(d, split, seed_base + 100 + i)
        audit = brier_audit(s, split, selected)
        row = {
                "split": list(split),
                "r": selected["r"],
                "empirical_objective": selected["objective"],
                "signed_empirical_gap": selected["signed_empirical_gap"],
                "objective_min_count": selected["objective_min_count"],
                "critic0_fit": selected["critic0_fit"],
                "critic1_fit": selected["critic1_fit"],
                "audit": audit,
                "category_eval_only": split_category(split),
            }
        if audit["pass"]:
            scores = aipw_scores(nuisance, evaluation, split, selected["r"])
            row["estimate"] = float(scores.mean())
            row["score_se"] = float(scores.std(ddof=1) / math.sqrt(len(scores)))
            row["score_sha256"] = array_sha256(scores)
            survivors.append((split, scores))
        rows.append(row)
    if survivors:
        matrix = np.column_stack([scores for _, scores in survivors])
        rho = common_radius(matrix, seed_base + 9)
        aggregation = aggregate([split for split, _ in survivors], matrix.mean(axis=0), rho)
    else:
        rho, aggregation = None, aggregate([], np.array([]), 0.0)
    component_counts = []
    for component in aggregation["members"]:
        count = {category: 0 for category in ("clean", "bad_singleton", "mixed", "S0")}
        for split in component:
            count[split_category(tuple(split))] += 1
        component_counts.append(count)
    baseline_scores = baseline_score_arrays(raw_n, raw_e, transform)
    baselines = {name: float(scores.mean()) for name, scores in baseline_scores.items()}
    baselines["naive"] = float(raw_e["Y"][raw_e["A"] == 1].mean() - raw_e["Y"][raw_e["A"] == 0].mean())
    return {
        "spec": asdict(spec),
        "total_n": total_n,
        "role_n": n,
        "role_seeds": {name: seed_base + i for i, name in enumerate(ROLE_NAMES)},
        "role_sha256": [role_sha256(x) for x in (raw_d, raw_s, raw_n, raw_e)],
        "used_roles": ["representation_empirical_risk", "independent_screen", "nuisance", "evaluation"],
        "learner": "restricted_parametric_empirical_brier",
        "rows": rows,
        "rho": rho,
        "aggregation": aggregation,
        "largest_component_counts_eval_only": component_counts,
        "retained_counts_eval_only": {
            category: sum(row["audit"]["pass"] and row["category_eval_only"] == category for row in rows)
            for category in ("clean", "bad_singleton", "mixed", "S0")
        },
        "baseline_estimates": baselines,
    }


def _probit_probability(beta: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eta = np.column_stack([np.ones(len(z)), z]) @ beta
    return 0.1 + 0.8 * norm.cdf(eta), eta


def aipw_scores(
    nuisance: dict[str, np.ndarray], evaluation: dict[str, np.ndarray], split: tuple[int, ...], r: float | None
) -> np.ndarray:
    z_n = representation(nuisance, split, r)
    z_e = representation(evaluation, split, r)
    return aipw_feature_scores(z_n, nuisance["A"], nuisance["Y"], z_e, evaluation["A"], evaluation["Y"])


def aipw_feature_scores(
    z_n: np.ndarray,
    a_n: np.ndarray,
    y_n: np.ndarray,
    z_e: np.ndarray,
    a_e: np.ndarray,
    y_e: np.ndarray,
) -> np.ndarray:
    """Scaled-probit/inverse-Mills AIPW for a supplied observed feature map."""
    beta = _fit_scaled_probit(z_n, a_n)
    p_n, eta_n = _probit_probability(beta, z_n)
    p_e, eta_e = _probit_probability(beta, z_e)
    design_n = np.column_stack([np.ones(len(z_n)), z_n])
    design_e = np.column_stack([np.ones(len(z_e)), z_e])
    predictions = []
    for arm in (0, 1):
        corr_n = (0.8 * norm.pdf(eta_n) / p_n) if arm else (-0.8 * norm.pdf(eta_n) / (1.0 - p_n))
        corr_e = (0.8 * norm.pdf(eta_e) / p_e) if arm else (-0.8 * norm.pdf(eta_e) / (1.0 - p_e))
        features_n = np.column_stack([design_n, corr_n])
        features_e = np.column_stack([design_e, corr_e])
        mask = a_n == arm
        coef = np.linalg.lstsq(features_n[mask], y_n[mask], rcond=None)[0]
        predictions.append(features_e @ coef)
    mu0, mu1 = predictions
    return mu1 - mu0 + a_e * (y_e - mu1) / p_e - (1.0 - a_e) * (y_e - mu0) / (1.0 - p_e)


def baseline_score_arrays(
    nuisance_raw: dict[str, np.ndarray],
    evaluation_raw: dict[str, np.ndarray],
    transform: Transform,
) -> dict[str, np.ndarray]:
    """Matched X/raw/oracle AIPW score arrays for exactly the supplied N/E rows."""
    k_n, k_e = transform.apply(nuisance_raw), transform.apply(evaluation_raw)
    maps = {
        "X": (nuisance_raw["X"][:, None], evaluation_raw["X"][:, None]),
        "raw_XW": (
            np.column_stack([nuisance_raw["X"], nuisance_raw["I"], nuisance_raw["C"], k_n]),
            np.column_stack([evaluation_raw["X"], evaluation_raw["I"], evaluation_raw["C"], k_e]),
        ),
        "oracle": (
            np.column_stack([nuisance_raw["X"], nuisance_raw["U_eval_only"], nuisance_raw["C"]]),
            np.column_stack([evaluation_raw["X"], evaluation_raw["U_eval_only"], evaluation_raw["C"]]),
        ),
    }
    return {
        name: aipw_feature_scores(z_n, nuisance_raw["A"], nuisance_raw["Y"], z_e,
                                  evaluation_raw["A"], evaluation_raw["Y"])
        for name, (z_n, z_e) in maps.items()
    }


def common_radius(score_matrix: np.ndarray, seed: int) -> float:
    n = score_matrix.shape[0]
    centered = score_matrix - score_matrix.mean(axis=0)
    covariance = centered.T @ centered / (n * n)
    draws = np.random.default_rng(seed).multivariate_normal(
        np.zeros(score_matrix.shape[1]), covariance, size=BOOTSTRAP_DRAWS
    )
    return float(np.quantile(np.max(np.abs(draws), axis=1), 0.95))


def aggregate(labels: list[tuple[int, ...]], estimates: np.ndarray, rho: float) -> dict:
    if len(estimates) == 0:
        return {"return_kind": "no_return", "output": None, "candidates": [], "members": []}
    parent = list(range(len(estimates)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(estimates)):
        for j in range(i + 1, len(estimates)):
            if abs(estimates[i] - estimates[j]) <= 2.0 * rho:
                parent[find(i)] = find(j)
    components: dict[int, list[int]] = {}
    for i in range(len(estimates)):
        components.setdefault(find(i), []).append(i)
    largest_size = max(map(len, components.values()))
    largest = [v for v in components.values() if len(v) == largest_size]
    candidates = [float(np.median(estimates[idx])) for idx in largest]
    return {
        "return_kind": "unique_largest" if len(largest) == 1 else "tied_largest",
        "output": candidates[0] if len(candidates) == 1 else None,
        "candidates": candidates,
        "members": [[list(labels[i]) for i in idx] for idx in largest],
        "component_sizes": sorted((len(v) for v in components.values()), reverse=True),
    }


def run_moment_replicate(spec: SCMSpec, total_n: int, seed_base: int) -> dict:
    if total_n % 4:
        raise ValueError("total_n must divide exactly into four independent roles")
    n = total_n // 4
    generated = [generate_role(spec, n, seed_base + j) for j in range(4)]
    transform = Transform.fit(generated[0], spec.observation)
    roles = [
        observed_view(data, transform, include_outcome=(i >= 2))
        for i, data in enumerate(generated)
    ]
    survivors = []
    fits = []
    for split in ALL_SPLITS:
        root = moment_root(roles[0], split)
        if 0 not in split and root["r"] is None:
            fits.append({"split": list(split), "root": root, "screen": {"pass": False, "reason": "no_root"}})
            continue
        audit = screen(roles[1], split, root["r"])
        row = {"split": list(split), "root": root, "screen": audit}
        if audit["pass"]:
            scores = aipw_scores(roles[2], roles[3], split, root["r"])
            row["estimate"] = float(scores.mean())
            survivors.append((split, scores))
        fits.append(row)
    if survivors:
        matrix = np.column_stack([scores for _, scores in survivors])
        rho = common_radius(matrix, seed_base + 9)
        result = aggregate([split for split, _ in survivors], matrix.mean(axis=0), rho)
    else:
        rho, result = None, aggregate([], np.array([]), 0.0)
    retained_counts = {name: 0 for name in ("clean", "bad_singleton", "mixed", "S0")}
    for split, _ in survivors:
        retained_counts[split_category(split)] += 1
    component_counts = []
    for component in result["members"]:
        count = {name: 0 for name in retained_counts}
        for split in component:
            count[split_category(tuple(split))] += 1
        component_counts.append(count)
    # Baselines use the same nuisance and evaluation roles.  U appears only in
    # the privileged oracle; the other baselines use observed arrays.
    n_raw, e_raw = generated[2], generated[3]
    k_n, k_e = transform.apply(n_raw), transform.apply(e_raw)
    baseline_features = {
        "X": (n_raw["X"][:, None], e_raw["X"][:, None]),
        "raw_XW": (
            np.column_stack([n_raw["X"], n_raw["I"], n_raw["C"], k_n]),
            np.column_stack([e_raw["X"], e_raw["I"], e_raw["C"], k_e]),
        ),
        "oracle": (
            np.column_stack([n_raw["X"], n_raw["U_eval_only"], n_raw["C"]]),
            np.column_stack([e_raw["X"], e_raw["U_eval_only"], e_raw["C"]]),
        ),
    }
    baselines = {}
    for name, (z_n, z_e) in baseline_features.items():
        scores = aipw_feature_scores(z_n, n_raw["A"], n_raw["Y"], z_e, e_raw["A"], e_raw["Y"])
        baselines[name] = float(scores.mean())
    baselines["naive"] = float(e_raw["Y"][e_raw["A"] == 1].mean() - e_raw["Y"][e_raw["A"] == 0].mean())
    return {
        "spec": asdict(spec),
        "total_n": total_n,
        "role_n": n,
        "role_seeds": {name: seed_base + i for i, name in enumerate(ROLE_NAMES)},
        "role_sha256": {name: role_sha256(generated[i]) for i, name in enumerate(ROLE_NAMES)},
        "learner": "structured_observed_partial_covariance",
        "rho": rho,
        "aggregation": result,
        "retained_counts_eval_only": retained_counts,
        "largest_component_counts_eval_only": component_counts,
        "baseline_estimates": baselines,
        "split_fits": fits,
    }


def run_rotating_replicate(
    spec: SCMSpec, total_n: int, seed_base: int, *, learner: str = "moment"
) -> dict:
    """Four honest role rotations over the same four independent folds.

    A split is retained only if its independently fitted representation passes
    the screen in all four rotations.  Its evaluation score concatenates the
    four disjoint E folds.  This is a cross-fitted operational extension; since
    every E fold is used in another rotation's D/S/N role, it is not advertised
    as the manuscript's one-shot conditional finite-sample guarantee.
    """
    if total_n % 4:
        raise ValueError("total_n must divide exactly into four folds")
    n = total_n // 4
    folds = [generate_role(spec, n, seed_base + j) for j in range(4)]
    by_split: dict[tuple[int, ...], list[dict]] = {split: [] for split in ALL_SPLITS}
    baseline_scores: dict[str, list[np.ndarray]] = {k: [] for k in ("X", "raw_XW", "oracle")}
    naive_parts = []
    rotations = []
    for rotation in range(4):
        d_i, s_i, n_i, e_i = [(rotation + j) % 4 for j in range(4)]
        transform = Transform.fit(folds[d_i], spec.observation)
        d = observed_view(folds[d_i], transform)
        s = observed_view(folds[s_i], transform)
        nuisance = observed_view(folds[n_i], transform, include_outcome=True)
        evaluation = observed_view(folds[e_i], transform, include_outcome=True)
        rotations.append({"rotation": rotation, "fold_roles": {"D": d_i, "S": s_i, "N": n_i, "E": e_i}})
        for split in ALL_SPLITS:
            if learner == "moment":
                root = moment_root(d, split)
                if 0 not in split and root["r"] is None:
                    by_split[split].append({"root": root, "screen": {"pass": False, "reason": "no_root"}})
                    continue
                r = root["r"]
                audit = screen(s, split, r)
                row = {"root": root, "screen": audit}
            elif learner == "brier":
                selected = brier_select(d, split, seed_base + 1000 * rotation + ALL_SPLITS.index(split))
                r = selected["r"]
                audit = brier_audit(s, split, selected)
                row = {
                    "brier": {
                        "r": r,
                        "empirical_objective": selected["objective"],
                        "signed_empirical_gap": selected["signed_empirical_gap"],
                        "objective_min_count": selected["objective_min_count"],
                        "critic0_fit": selected["critic0_fit"],
                        "critic1_fit": selected["critic1_fit"],
                    },
                    "screen": audit,
                }
            else:
                raise ValueError(f"unknown rotating learner: {learner}")
            if audit["pass"]:
                row["scores"] = aipw_scores(nuisance, evaluation, split, r)
            by_split[split].append(row)
        n_raw, e_raw = folds[n_i], folds[e_i]
        for name, scores in baseline_score_arrays(n_raw, e_raw, transform).items():
            baseline_scores[name].append(scores)
        naive_parts.append(
            e_raw["Y"][e_raw["A"] == 1].mean() - e_raw["Y"][e_raw["A"] == 0].mean()
        )
    survivors = []
    fits = []
    for split, rows in by_split.items():
        passed_all = len(rows) == 4 and all(row["screen"]["pass"] for row in rows)
        public_rows = []
        for row in rows:
            public = {k: v for k, v in row.items() if k != "scores"}
            if "scores" in row:
                public["estimate"] = float(row["scores"].mean())
                public["score_se"] = float(row["scores"].std(ddof=1) / math.sqrt(len(row["scores"])))
                public["score_sha256"] = array_sha256(row["scores"])
            public_rows.append(public)
        fits.append({"split": list(split), "pass_all_four": passed_all, "rotations": public_rows})
        if passed_all:
            survivors.append((split, np.concatenate([row["scores"] for row in rows])))
    if survivors:
        matrix = np.column_stack([scores for _, scores in survivors])
        rho = common_radius(matrix, seed_base + 9)
        result = aggregate([split for split, _ in survivors], matrix.mean(axis=0), rho)
        pooled_median_all = float(np.median(matrix.mean(axis=0)))
    else:
        rho, result, pooled_median_all = None, aggregate([], np.array([]), 0.0), None
    retained_counts = {name: 0 for name in ("clean", "bad_singleton", "mixed", "S0")}
    for split, _ in survivors:
        retained_counts[split_category(split)] += 1
    component_counts = []
    for component in result["members"]:
        count = {name: 0 for name in retained_counts}
        for split in component:
            count[split_category(tuple(split))] += 1
        component_counts.append(count)
    return {
        "spec": asdict(spec),
        "total_n": total_n,
        "fold_n": n,
        "fold_seeds": [seed_base + j for j in range(4)],
        "fold_sha256": [role_sha256(fold) for fold in folds],
        "learner": f"{learner}_rotate4",
        "retention_rule": "screen_pass_in_all_four_rotations",
        "rotations": rotations,
        "rho": rho,
        "aggregation": result,
        "pooled_median_all_retained_ablation": pooled_median_all,
        "retained_counts_eval_only": retained_counts,
        "largest_component_counts_eval_only": component_counts,
        "baseline_estimates": {k: float(np.concatenate(v).mean()) for k, v in baseline_scores.items()}
        | {"naive": float(np.mean(naive_parts))},
        "split_fits": fits,
        "theory_scope": "cross_fitted_operational_extension_not_one_shot_conditional_guarantee",
    }


def run_e9_component_replicate(
    spec: SCMSpec, total_n: int, seed_base: int, *, grid_points: int = 101
) -> dict:
    """Paired hard4 decomposition with honest D/S and N/E swaps on four fixed folds.

    H11/H12 use one D/S fit; H21/H22 use both D0/S1 and D1/S0.
    Multiple representations are averaged within each unique evaluation row
    before different evaluation folds are concatenated.
    """
    if spec.treatment_link != "probit":
        raise ValueError("the E9 component ablation is frozen to the probit-link SCM-1")
    if total_n % 4:
        raise ValueError("total_n must divide exactly into four folds")
    n = total_n // 4
    folds = [generate_role(spec, n, seed_base + j) for j in range(4)]
    ds_pairs = ((0, 1), (1, 0))
    ne_pairs = ((2, 3), (3, 2))
    ds_fits: list[dict] = []
    for ds_index, (d_i, s_i) in enumerate(ds_pairs):
        transform = Transform.fit(folds[d_i], spec.observation)
        d = observed_view(folds[d_i], transform)
        s = observed_view(folds[s_i], transform)
        split_rows = {}
        for split_index, split in enumerate(ALL_SPLITS):
            selected = brier_select(
                d, split, seed_base + 1000 * ds_index + split_index, grid_points=grid_points
            )
            audit = brier_audit(s, split, selected)
            score_rows = []
            for n_i, e_i in ne_pairs:
                nuisance = observed_view(folds[n_i], transform, include_outcome=True)
                evaluation = observed_view(folds[e_i], transform, include_outcome=True)
                scores = aipw_scores(nuisance, evaluation, split, selected["r"])
                score_rows.append(scores)
            split_rows[split] = {"selected": selected, "screen": audit, "scores": score_rows}
        ds_fits.append({"D": d_i, "S": s_i, "transform": transform, "splits": split_rows})

    def make_arm(
        name: str, ds_indices: tuple[int, ...], ne_indices: tuple[int, ...], radius_seed_offset: int
    ) -> dict:
        survivors = []
        candidate_rows = []
        for split in ALL_SPLITS:
            passed = all(ds_fits[d]["splits"][split]["screen"]["pass"] for d in ds_indices)
            score_parts = []
            score_part_hashes = []
            for ne_index in ne_indices:
                part = average_score_vectors([
                    ds_fits[d]["splits"][split]["scores"][ne_index] for d in ds_indices
                ])
                score_parts.append(part)
                score_part_hashes.append(array_sha256(part))
            scores = np.concatenate(score_parts)
            row = {
                "split": list(split),
                "screen_pass": passed,
                "screen_by_ds": [ds_fits[d]["splits"][split]["screen"] for d in ds_indices],
                "r_by_ds": [float(ds_fits[d]["splits"][split]["selected"]["r"])
                            if ds_fits[d]["splits"][split]["selected"]["r"] is not None else None
                            for d in ds_indices],
                "estimate": float(scores.mean()),
                "score_part_sha256": score_part_hashes,
                "score_sha256": array_sha256(scores),
                "score_n": len(scores),
                "category_eval_only": split_category(split),
            }
            candidate_rows.append(row)
            if passed:
                survivors.append((split, scores))
        if survivors:
            matrix = np.column_stack([scores for _, scores in survivors])
            rho = common_radius(matrix, seed_base + radius_seed_offset)
            aggregation = aggregate([split for split, _ in survivors], matrix.mean(axis=0), rho)
        else:
            rho, aggregation = None, aggregate([], np.array([]), 0.0)
        ledgers = []
        for ne_index in ne_indices:
            n_i, e_i = ne_pairs[ne_index]
            ledgers.append(score_role_ledger(
                tuple(ds_fits[d]["D"] for d in ds_indices),
                tuple(ds_fits[d]["S"] for d in ds_indices), n_i, e_i, n,
            ))
        baseline_parts: dict[str, list[np.ndarray]] = {k: [] for k in ("X", "raw_XW", "oracle")}
        for ne_index in ne_indices:
            n_i, e_i = ne_pairs[ne_index]
            for baseline in baseline_parts:
                per_ds = [baseline_score_arrays(
                    folds[n_i], folds[e_i], ds_fits[d]["transform"]
                )[baseline] for d in ds_indices]
                baseline_parts[baseline].append(average_score_vectors(per_ds))
        baselines = {
            key: float(np.concatenate(parts).mean()) for key, parts in baseline_parts.items()
        }
        retained_counts = {key: 0 for key in ("clean", "bad_singleton", "mixed", "S0")}
        for split, _ in survivors:
            retained_counts[split_category(split)] += 1
        component_counts = []
        for component in aggregation["members"]:
            counts = {key: 0 for key in retained_counts}
            for split in component:
                counts[split_category(tuple(split))] += 1
            component_counts.append(counts)
        return {
            "name": name,
            "ds_pairs": [{"D": ds_fits[d]["D"], "S": ds_fits[d]["S"]} for d in ds_indices],
            "ne_pairs": [{"N": ne_pairs[i][0], "E": ne_pairs[i][1]} for i in ne_indices],
            "score_ledgers": ledgers,
            "unique_evaluation_rows": n * len(ne_indices),
            "rho": rho,
            "aggregation": aggregation,
            "retained_counts_eval_only": retained_counts,
            "largest_component_counts_eval_only": component_counts,
            "baseline_estimates": baselines,
            "candidate_rows": candidate_rows,
        }

    arms = {
        "H11": make_arm("H11", (0,), (0,), 9),
        "H12": make_arm("H12", (0,), (0, 1), 19),
        "H21": make_arm("H21", (0, 1), (0,), 29),
        "H22": make_arm("H22", (0, 1), (0, 1), 39),
    }
    return {
        "spec": asdict(spec),
        "total_n": total_n,
        "fold_n": n,
        "fold_seeds": [seed_base + j for j in range(4)],
        "fold_sha256": [role_sha256(fold) for fold in folds],
        "arms": arms,
        "scientific_logic": "unchanged_brier_screen_aipw_and_algorithm1",
        "theory_scope": "paired_operational_component_ablation_not_one_shot_theorem",
    }


def write_json_atomic(path: str | Path, payload: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable result: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
