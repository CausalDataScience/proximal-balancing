"""Immutable E9 five-gate scoring and hard4 baseline reconstruction."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

from probe_structured_scm import (
    SCMSpec,
    Transform,
    baseline_score_arrays,
    generate_role,
    role_sha256,
    source_sha256,
    write_json_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
TRUE = 1.0
DEFAULT_CELLS = {
    "noise_1.0_hard4": "results/probe_e9_scm1_noise_10_hard4.json",
    "noise_1.0_rotate4": "results/probe_e9_scm1_noise_10_rotate4.json",
    "noise_0.3_hard4": "results/probe_e9_scm1_noise_03_hard4.json",
    "noise_0.3_rotate4": "results/probe_e9_scm1_noise_03_rotate4.json",
}
GATES = {"mae": 0.05, "p90": 0.10, "return_rate": 0.95, "oracle_excess": 0.05}


def paired_upper(delta: np.ndarray, comparisons: int = 3, alpha: float = 0.05) -> float:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) < 2:
        return float("nan")
    quantile = tdist.ppf(1.0 - alpha / comparisons, len(delta) - 1)
    return float(delta.mean() + quantile * delta.std(ddof=1) / math.sqrt(len(delta)))


def reconstruct_hard4_baselines(payload: dict) -> list[dict]:
    """Regenerate exact historical folds, fail on hash mismatch, and score baselines only."""
    if not payload.get("complete") or payload.get("mode") != "hard4":
        raise ValueError("hard4 input must be complete")
    out = []
    for index, stored in enumerate(payload["rows"]):
        seed_base = payload["seed_start"] + 10 * index
        spec = SCMSpec(**stored["spec"])
        folds = [generate_role(spec, stored["role_n"], seed_base + j) for j in range(4)]
        hashes = [role_sha256(fold) for fold in folds]
        if hashes != stored["role_sha256"]:
            raise ValueError(f"historical fold hash mismatch at replicate {index}")
        transform = Transform.fit(folds[0], spec.observation)
        scores = baseline_score_arrays(folds[2], folds[3], transform)
        estimates = {name: float(value.mean()) for name, value in scores.items()}
        estimates["naive"] = float(
            folds[3]["Y"][folds[3]["A"] == 1].mean()
            - folds[3]["Y"][folds[3]["A"] == 0].mean()
        )
        out.append({"replicate": index, "seed_base": seed_base, "fold_sha256": hashes,
                    "baseline_estimates": estimates})
    return out


def five_gate_summary(payload: dict, hard_sidecar: list[dict] | None = None) -> dict:
    estimates, returns = [], []
    baselines = {key: [] for key in ("X", "raw_XW", "oracle")}
    for index, row in enumerate(payload["rows"]):
        unique = row["aggregation"]["return_kind"] == "unique_largest"
        returns.append(unique)
        estimates.append(float(row["aggregation"]["output"]) if unique else float("nan"))
        base = row.get("baseline_estimates")
        if base is None and hard_sidecar is not None:
            base = hard_sidecar[index]["baseline_estimates"]
        if base is None:
            raise ValueError("baseline estimates are required for all five gates")
        for key in baselines:
            baselines[key].append(float(base[key]))
    estimates = np.asarray(estimates)
    unique = np.isfinite(estimates)
    errors = np.abs(estimates[unique] - TRUE)
    metrics = {
        "replicates": len(estimates),
        "unique_returns": int(unique.sum()),
        "return_rate": float(np.mean(returns)),
        "mae_unique": float(errors.mean()) if len(errors) else None,
        "p90_unique": float(np.quantile(errors, 0.9)) if len(errors) else None,
    }
    comparisons = {}
    for key, values in baselines.items():
        base_error = np.abs(np.asarray(values)[unique] - TRUE)
        delta = errors - base_error
        comparisons[key] = {"mean_error_difference": float(delta.mean()),
                            "bonferroni_one_sided_upper": paired_upper(delta)}
    checks = {
        "mae": metrics["mae_unique"] is not None and metrics["mae_unique"] <= GATES["mae"],
        "p90": metrics["p90_unique"] is not None and metrics["p90_unique"] <= GATES["p90"],
        "return_rate": metrics["return_rate"] >= GATES["return_rate"],
        "beats_X": comparisons["X"]["bonferroni_one_sided_upper"] < 0.0,
        "beats_raw_XW": comparisons["raw_XW"]["bonferroni_one_sided_upper"] < 0.0,
        "oracle_excess": comparisons["oracle"]["bonferroni_one_sided_upper"] <= GATES["oracle_excess"],
    }
    return {"metrics": metrics, "paired": comparisons, "checks": checks,
            "all_frozen_empirical_benchmark_gates_pass": all(checks.values()),
            "boundary": "20-seed empirical benchmark verdict, not joint 95% population certification"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--hard-baseline-output", required=True)
    args = parser.parse_args()
    loaded = {name: json.loads((ROOT / path).read_text()) for name, path in DEFAULT_CELLS.items()}
    hard_sidecars = {}
    sidecar_rows = {}
    for name, payload in loaded.items():
        if name.endswith("hard4"):
            sidecar_rows[name] = reconstruct_hard4_baselines(payload)
            hard_sidecars[name] = sidecar_rows[name]
    inputs = {name: {"path": path, "sha256": source_sha256(ROOT / path)}
              for name, path in DEFAULT_CELLS.items()}
    source = Path(__file__).resolve()
    sources = {"analyze_e9.py": source_sha256(source),
               "probe_structured_scm.py": source_sha256(source.with_name("probe_structured_scm.py"))}
    sidecar = {"complete": True, "generated_at": datetime.now(timezone.utc).isoformat(),
               "source_sha256": sources, "inputs": inputs, "cells": sidecar_rows}
    write_json_atomic(args.hard_baseline_output, sidecar)
    summary = {
        "complete": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sources,
        "inputs": inputs,
        "gates": GATES,
        "cells": {name: five_gate_summary(payload, hard_sidecars.get(name))
                  for name, payload in loaded.items()},
        "production_mode": "rotate4_operational_crossfit",
        "hard4_scope": "one_shot_diagnostic",
    }
    write_json_atomic(args.output, summary)


if __name__ == "__main__":
    main()
