#!/usr/bin/env python3
"""Generator-only preflight. Does not train image representations."""
from __future__ import annotations

import argparse
import copy
import hashlib
import resource
import time
from pathlib import Path

import numpy as np
import torch

import sim5_digit_proxy as sim


def regression_adjustment(data: dict, controls: list[str]) -> dict:
    design = np.column_stack([np.ones(len(data["A"])), data["A"], *[data[k] for k in controls]])
    beta, _, rank, _ = np.linalg.lstsq(design, data["Y"], rcond=None)
    residual = data["Y"] - design @ beta
    inverse = np.linalg.pinv(design.T @ design)
    # HC0 sandwich standard error; the X-only regression is a diagnostic projection.
    meat = (design * residual[:, None]).T @ (design * residual[:, None])
    covariance = inverse @ meat @ inverse
    return {"estimate": float(beta[1]), "se_hc0": float(np.sqrt(covariance[1, 1])),
            "controls": controls, "rank": int(rank),
            "method": "OLS coefficient of A; X-only is a linear projection, not nonparametric adjustment"}


def preflight(n: int, seed: int) -> dict:
    started = time.monotonic()
    data = sim.generate_latents(n, seed)
    groups = [data["Y"][data["A"] == a] for a in (0, 1)]
    naive = float(groups[1].mean() - groups[0].mean())
    x_only = regression_adjustment(data, ["X"])
    oracle = regression_adjustment(data, ["X", "U_c_eval_only", "U_t_eval_only"])
    sources = [Path(__file__), Path(sim.__file__)]
    return {"complete": True, "phase": "generator_preflight", "n": n, "seed": seed,
            "spec": "Sim-5 digit proxy Revision 3; latent-first generator only; no pixels or representation learning",
            "true_ate": 1.0, "naive": naive, "x_only": x_only, "oracle": oracle,
            "simpson_reversal": naive < 0, "oracle_sanity": abs(oracle["estimate"] - 1) <= .03,
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "elapsed_seconds": time.monotonic() - started}


def _hash_array(values) -> str:
    arr = np.asarray(values)
    return hashlib.sha256(str((arr.shape, arr.dtype)).encode() + arr.tobytes()).hexdigest()


def fixed_split_smoke(seed: int, quick: bool, bank: dict | None = None,
                       budget_override: dict | None = None) -> dict:
    """One fixed split end-to-end. A quick run is an implementation diagnostic."""
    started = time.monotonic()
    sizes = ({"D_fit": 120, "D_select": 40, "D_audit": 40, "N": 160, "E": 400}
             if quick else dict(sim.FULL_SIZES))
    warm_budget = {"max_epochs": 2, "patience": 2} if quick else None
    critic_budget = {"max_epochs": 1, "patience": 1} if quick else None
    refit_budget = {"max_epochs": 2, "patience": 2} if quick else None
    nuisance_budget = {"max_epochs": 3, "patience": 2} if quick else None
    if budget_override is not None:
        if not quick:
            raise ValueError("budget overrides are allowed only in diagnostic quick mode")
        warm_budget = critic_budget = refit_budget = nuisance_budget = dict(budget_override)
    children = np.random.SeedSequence(seed).spawn(10)
    seeds = [int(s.generate_state(1)[0]) for s in children]
    latents = sim.generate_latents(sum(sizes.values()), seeds[0])
    bank = sim.load_mnist() if bank is None else bank
    images = sim.render_images(latents, seeds[1], bank=bank)
    view = sim.learner_view({**{k: latents[k] for k in ("X", "A", "Y")},
                             "images": images, "row_id": np.arange(len(images))})
    rows = sim.outer_rows(sizes, seeds[2])
    dfit = rows["D_fit"]
    cut = int(.8 * len(dfit))
    tr, va = dfit[:cut], dfit[cut:]
    generated_at = time.monotonic()
    encoder_bank = sim.fit_block_encoder_bank(view, tr, va, seeds[3], warm_budget)
    bank_at = time.monotonic()
    all_records, restarts, restart_work = [], [], []
    for restart in range(1 if quick else 2):
        warm = sim.fit_warm_fusion(view, encoder_bank, [1, 2, 3, 4], tr, va,
                                   seed=seeds[4] + restart, budget=warm_budget)
        balanced = sim.balance_updates(warm, view, tr, va, seed=seeds[5] + restart,
                                       critic_budget=critic_budget)
        selected = sim.select_checkpoint(balanced, view, tr, va, rows["D_select"],
                                          seed=seeds[6], budget=refit_budget, restart_id=restart)
        all_records.extend(selected["selection_records"])
        restarts.append({"restart_id": restart, "warm_digest": warm["warm_digest"],
                         "warm_fit": warm["fit"], "warm_parameter_change_l2": warm["representation_parameter_change_l2"],
                         "head_update_count": balanced["head_update_count"],
                         "critic_update_count": balanced["critic_update_count"],
                         "clipped_zero_count": balanced["clipped_zero_count"],
                         "zero_gradient_count": balanced["zero_gradient_count"],
                         "fusion_parameter_change_l2": balanced["fusion_parameter_change_l2"],
                         "updates": balanced["updates"], "refreshes": balanced["refreshes"],
                         "checkpoints": {str(k): {f: v for f, v in c.items() if f != "state"}
                                         for k, c in balanced["checkpoints"].items()},
                         "selection_records": selected["selection_records"]})
        restart_work.append((warm, balanced, selected))
    minimum = min(c["clipped_gap"] for c in all_records)
    chosen = min((c for c in all_records if c["clipped_gap"] <= minimum + 1e-6),
                 key=lambda c: (c["head_update_count"], c["restart_id"]))
    warm, balanced, selected = restart_work[chosen["restart_id"]]
    selected = dict(selected)
    selected["selected"] = chosen
    selected["model"] = copy.deepcopy(balanced["model"])
    selected["model"].load_state_dict(balanced["checkpoints"][chosen["head_update_count"]]["state"])
    selected_at = time.monotonic()
    audits = sim.audit_warm_final(selected, view, tr, va, rows["D_audit"], seeds[7], refit_budget)
    audited_at = time.monotonic()
    effects = sim.evaluate_warm_final(selected, view, rows["N"], rows["E"], seeds[8], nuisance_budget)
    for name in ("warm", "final"):
        evaluated = effects[name]["evaluation"]
        evaluated["score_sha256"] = _hash_array(evaluated.pop("scores"))
        evaluated["absolute_error"] = abs(evaluated["theta"] - 1.)
    joint = np.concatenate(list(rows.values()))
    checks = {"disjoint_outer_rows": len(np.unique(joint)) == len(joint),
              "all_100_head_updates": all(r["head_update_count"] == 100 for r in restarts),
              "all_real_checkpoints": all(set(r["checkpoints"]) == {"0", "25", "50", "100"} for r in restarts),
              "selected_digest_matches": chosen["state_sha256"] == sim.state_digest(selected["model"]),
              "warm_digest_matches": audits["warm"]["state_sha256"] == selected["warm_digest"],
              "independent_audit_seeds": audits["warm"]["seed"] != audits["final"]["seed"],
              "audit_did_not_change_selection": audits["selection_unchanged"],
              "finite_effects": all(np.isfinite(effects[k]["evaluation"]["theta"]) for k in effects)}
    sources = [Path(__file__), Path(sim.__file__)]
    prd = Path(__file__).resolve().parents[1] / "memo/2026-09-17-sim5-v2-digit-proxy-prd-v1.md"
    finished = time.monotonic()
    return {"complete": True, "phase": "fixed_split_smoke", "diagnostic_quick": quick,
            "implementation_pass": all(checks.values()), "checks": checks,
            "performance_pass": None, "interpretation": "single-split implementation smoke; not a performance or theorem verdict",
            "seed": seed, "child_seeds": seeds, "sizes": sizes, "heldout": [0], "retained": [1, 2, 3, 4],
            "mnist_sha256": bank.get("sha256", "synthetic_test_bank"),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "prd_sha256": hashlib.sha256(prd.read_bytes()).hexdigest(),
            "row_sha256": {k: _hash_array(v) for k, v in rows.items()},
            "internal_fit_row_sha256": _hash_array(tr), "internal_val_row_sha256": _hash_array(va),
            "bank_metadata": encoder_bank["metadata"], "restarts": restarts,
            "selected": chosen, "audits": audits, "effects": effects,
            "timing_seconds": {"generation": generated_at - started, "encoder_bank": bank_at - generated_at,
                               "warm_balance_selection": selected_at - bank_at, "audit": audited_at - selected_at,
                               "nuisance_evaluation": finished - audited_at, "total": finished - started},
            "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "torch_threads": torch.get_num_threads()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["preflight", "fixed-split-smoke"], default="preflight")
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=6_510_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = (preflight(args.n, args.seed) if args.phase == "preflight"
              else fixed_split_smoke(args.seed, args.quick))
    sim.write_json_atomic(args.output, result)
    if args.phase == "preflight":
        print(f"naive={result['naive']:.6f}; X-only OLS={result['x_only']['estimate']:.6f}; "
              f"oracle OLS={result['oracle']['estimate']:.6f}; seconds={result['elapsed_seconds']:.3f}")
    else:
        print(f"implementation_pass={result['implementation_pass']}; "
              f"warm={result['effects']['warm']['evaluation']['theta']:.6f}; "
              f"final={result['effects']['final']['evaluation']['theta']:.6f}; "
              f"seconds={result['timing_seconds']['total']:.3f}")


if __name__ == "__main__":
    main()
