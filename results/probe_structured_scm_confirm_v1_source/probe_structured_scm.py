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


def _fit_brier_probit(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Independently minimise empirical bounded-probit Brier risk."""
    design = np.column_stack([np.ones(len(a)), z])

    def risk(beta: np.ndarray) -> float:
        p = 0.1 + 0.8 * norm.cdf(np.clip(design @ beta, -8.0, 8.0))
        return float(np.mean((a - p) ** 2))

    fitted = minimize(risk, np.zeros(design.shape[1]), method="BFGS", options={"maxiter": 120})
    if not np.isfinite(fitted.fun):
        raise RuntimeError("Brier critic fit failed")
    return fitted.x


def _brier_rows(beta: np.ndarray, z: np.ndarray, a: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(a)), z])
    p = 0.1 + 0.8 * norm.cdf(np.clip(design @ beta, -8.0, 8.0))
    return (a - p) ** 2


def brier_select(
    data: dict[str, np.ndarray], split: tuple[int, ...], seed: int, grid_points: int = 41
) -> dict:
    """Restricted manuscript-objective encoder selection on an internal D split.

    q0 and q1 independently minimise their own training Brier risks.  The
    encoder is selected by the clipped nested-risk gap on untouched D-select.
    This is a restricted linear/probit critic class, not the neural experiment.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data["A"]))
    cut = (2 * len(order)) // 3
    fit, select = order[:cut], order[cut:]
    candidates: list[float | None] = [None] if 0 in split else list(np.linspace(-1.0, 1.0, grid_points))
    rows = []
    target = heldout(data, split)
    for r in candidates:
        z = representation(data, split, r)
        beta0 = _fit_brier_probit(z[fit], data["A"][fit])
        beta1 = _fit_brier_probit(np.column_stack([z[fit], target[fit]]), data["A"][fit])
        loss0 = _brier_rows(beta0, z[select], data["A"][select])
        loss1 = _brier_rows(beta1, np.column_stack([z[select], target[select]]), data["A"][select])
        signed_gap = float(np.mean(loss0 - loss1))
        rows.append(
            {
                "r": r,
                "signed_select_gap": signed_gap,
                "objective": max(signed_gap, 0.0),
                "beta0": beta0,
                "beta1": beta1,
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
        "signed_select_gap": selected["signed_select_gap"],
        "objective": selected["objective"],
        "grid_points": grid_points,
        "fit_n": len(fit),
        "select_n": len(select),
        "beta0": selected["beta0"],
        "beta1": selected["beta1"],
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
    """All-30 restricted-Brier selection/screen diagnostic, without ATE claims."""
    if spec.treatment_link != "probit":
        raise ValueError("restricted Brier critic is frozen to the probit-link settings")
    n = total_n // 4
    raw_d = generate_role(spec, n, seed_base)
    raw_s = generate_role(spec, n, seed_base + 1)
    transform = Transform.fit(raw_d, spec.observation)
    d, s = observed_view(raw_d, transform), observed_view(raw_s, transform)
    rows = []
    for i, split in enumerate(ALL_SPLITS):
        selected = brier_select(d, split, seed_base + 100 + i)
        audit = brier_audit(s, split, selected)
        rows.append(
            {
                "split": list(split),
                "r": selected["r"],
                "select_objective": selected["objective"],
                "signed_select_gap": selected["signed_select_gap"],
                "objective_min_count": selected["objective_min_count"],
                "audit": audit,
                "category_eval_only": split_category(split),
            }
        )
    return {
        "spec": asdict(spec),
        "total_n": total_n,
        "role_n": n,
        "used_roles": ["representation_internal_fit_select", "independent_screen"],
        "unused_roles": ["nuisance", "evaluation"],
        "learner": "restricted_parametric_empirical_brier",
        "rows": rows,
        "retained_counts_eval_only": {
            category: sum(row["audit"]["pass"] and row["category_eval_only"] == category for row in rows)
            for category in ("clean", "bad_singleton", "mixed", "S0")
        },
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


def run_rotating_replicate(spec: SCMSpec, total_n: int, seed_base: int) -> dict:
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
            root = moment_root(d, split)
            if 0 not in split and root["r"] is None:
                by_split[split].append({"root": root, "screen": {"pass": False, "reason": "no_root"}})
                continue
            audit = screen(s, split, root["r"])
            row = {"root": root, "screen": audit}
            if audit["pass"]:
                row["scores"] = aipw_scores(nuisance, evaluation, split, root["r"])
            by_split[split].append(row)
        n_raw, e_raw = folds[n_i], folds[e_i]
        k_n, k_e = transform.apply(n_raw), transform.apply(e_raw)
        maps = {
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
        for name, (z_n, z_e) in maps.items():
            baseline_scores[name].append(
                aipw_feature_scores(z_n, n_raw["A"], n_raw["Y"], z_e, e_raw["A"], e_raw["Y"])
            )
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
        "learner": "structured_observed_partial_correlation_rotate4",
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


def write_json_atomic(path: str | Path, payload: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable result: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
