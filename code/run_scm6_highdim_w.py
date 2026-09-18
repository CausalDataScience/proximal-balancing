"""SCM-6: the operational pipeline on 1,282-dimensional proxy blocks."""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

import minimal_suite_common as c
import probe_highdim_w as hd

SOURCES = ["minimal_suite_common.py", "probe_highdim_w.py", "run_scm6_highdim_w.py"]
PROTOCOL_PATH = "../results/scm6_highdim_protocol_v1.json"
PENALTY_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
FROZEN = {
    "protocol_id": "scm6-highdim-w-v1", "experiment_id": "SCM-6", "mode": "operational_rotate4",
    "total_n": 12000, "rows_per_fold": 3000, "emb_fit": 1200, "emb_cal": 300, "d_phi": 1500,
    "dim": hd.DIM, "raw_w_dim": 2 + 5 * hd.DIM, "nuisance_amplitude": hd.NUISANCE_AMPLITUDE,
    "r_grid_points": len(c.GRID), "penalty_grid": list(PENALTY_GRID),
    "screen_rule": "clipped gap at most twice its standard error, and fitted propensity inside [0.1, 0.9]",
    "retention": "a split is retained only if it passes in all four rotations",
    "dev_seeds": [94000000 + 1000 * r for r in range(5)],
    "confirm_seeds": [94100000 + 1000 * r for r in range(20)],
    "q_digests": {str(j): hd.orthogonal_digest(j) for j in range(5)},
    "gates": {"return_rate_min": 0.95, "mae_max": 0.05, "p90_max": 0.10,
              "oracle_excess_upper_max": 0.05, "channel_corr_median_min": 0.90,
              "bonferroni_contrasts": 8, "alpha": 0.05},
}


def freeze() -> dict:
    p = Path(PROTOCOL_PATH)
    if p.exists():
        return json.loads(p.read_text())
    proto = dict(FROZEN)
    proto["source_snapshot"] = c.source_snapshot(SOURCES)
    proto["frozen_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    proto["runs"] = [{"experiment_id": "SCM-6", "mode": "operational_rotate4", "phase": ph,
                      "seeds": FROZEN["dev_seeds"] if ph == "development" else FROZEN["confirm_seeds"]}
                     for ph in ("development", "confirmation")]
    return c.seal_protocol(p, proto)


def slice_rows(view: dict, idx: np.ndarray) -> dict:
    return {k: ([b[idx] for b in v] if k == "H" else v[idx]) for k, v in view.items()}


def one_replicate(seed: int) -> dict:
    t0 = time.time()
    data = hd.generate(FROZEN["total_n"], seed)
    view = hd.observed_view(data, include_outcome=True)
    folds = c.four_folds(FROZEN["total_n"], seed + 1)
    splits = c.oriented_splits()
    per_rot, ledger = [], []
    scores: dict[tuple, list] = {sp: [] for sp in splits}
    base_scores: dict[str, list] = {}
    passes: dict[tuple, int] = {sp: 0 for sp in splits}
    corrs = []
    for rot in c.rotations(folds):
        ledger.append({"rotation": rot.index, "roles": c.role_ledger(rot.roles),
                       "overlap": c.assert_roles_disjoint(rot.roles)})
        d_idx = rot.roles["D"]
        emb_fit = slice_rows(view, d_idx[:FROZEN["emb_fit"]])
        emb_cal = slice_rows(view, d_idx[FROZEN["emb_fit"]:FROZEN["emb_fit"] + FROZEN["emb_cal"]])
        phi = slice_rows(view, d_idx[FROZEN["emb_fit"] + FROZEN["emb_cal"]:])
        emb = hd.fit_embedding(emb_fit, emb_cal)
        if emb["fail_closed"]:
            return {"data_seed": seed, "return_kind": "no_return", "estimate": None,
                    "fail_closed": emb["fail_closed"], "complete": True,
                    "timing_seconds": time.time() - t0}
        corrs.append(hd.channel_correlation(emb, data, rot.roles["E"]))
        pcs = hd.blockwise_pca(emb_fit)
        s_rows, n_rows, e_rows = (slice_rows(view, rot.roles[k]) for k in ("S", "N", "E"))
        k_phi, k_s = hd.apply_embedding(emb, phi), hd.apply_embedding(emb, s_rows)
        k_n, k_e = hd.apply_embedding(emb, n_rows), hd.apply_embedding(emb, e_rows)
        rows = []
        for sp in splits:
            pen = hd.split_penalties(phi, k_phi, sp, PENALTY_GRID,
                                     c.optimizer_seed(FROZEN["protocol_id"], seed, rot.index,
                                                      c.split_id(sp), 0))
            sel = hd.select_r(phi, k_phi, sp, pen)
            scr = hd.screen(s_rows, k_s, sp, sel)
            rows.append({"split_id": c.split_id(sp), "r": sel["r"], "same_D_gap": sel["signed_gap"],
                         "objective_sha256": sel["objective_vector_sha256"],
                         "objective_min_count": sel["objective_min_count"],
                         "screen_gap": scr["gap"], "screen_se": scr["se"],
                         "overlap": [scr["p_min"], scr["p_max"]], "screen_pass": scr["pass"],
                         "penalty": pen["base"],
                         "evaluator_only": {"category": c.evaluator_category(sp)}})
            if scr["pass"]:
                passes[sp] += 1
                z_n = hd.representation(k_n, n_rows, sp, sel["r"])
                z_e = hd.representation(k_e, e_rows, sp, sel["r"])
                scores[sp].append(hd.aipw_from_features(z_n, n_rows["A"], n_rows["Y"],
                                                        z_e, e_rows["A"], e_rows["Y"]))
        for name, f_n in hd.baseline_features(n_rows, emb, pcs).items():
            f_e = hd.baseline_features(e_rows, emb, pcs)[name]
            base_scores.setdefault(name, []).append(
                hd.aipw_from_features(f_n, n_rows["A"], n_rows["Y"], f_e, e_rows["A"], e_rows["Y"]))
        base_scores.setdefault("oracle", []).append(hd.aipw_from_features(
            np.column_stack([n_rows["X"], data["U_eval_only"][rot.roles["N"]], n_rows["C"]]),
            n_rows["A"], n_rows["Y"],
            np.column_stack([e_rows["X"], data["U_eval_only"][rot.roles["E"]], e_rows["C"]]),
            e_rows["A"], e_rows["Y"]))
        per_rot.append(rows)

    retained = [sp for sp in splits if passes[sp] == 4]
    est = {sp: float(np.concatenate(scores[sp]).mean()) for sp in retained}
    baselines = {k: float(np.concatenate(v).mean()) for k, v in base_scores.items()}
    baselines["naive"] = float(view["Y"][view["A"] == 1].mean() - view["Y"][view["A"] == 0].mean())
    if retained:
        mat = np.column_stack([np.concatenate(scores[sp]) for sp in retained])
        rho = float(ps_common_radius(mat, seed + 5))
        agg = hd_aggregate([c.split_id(sp) for sp in retained],
                           np.array([est[sp] for sp in retained]), rho)
    else:
        rho, agg = None, {"return_kind": "no_return", "output": None, "members": []}
    return {"data_seed": seed, "fold_hashes": [c.array_sha256(f) for f in folds],
            "row_role_ledger": ledger,
            "representation_diagnostics": {"channel_correlation_per_rotation": corrs,
                                           "channel_correlation_median":
                                               float(np.median(np.array(corrs)))},
            "candidate_splits": [c.split_id(sp) for sp in splits],
            "retained_splits": [c.split_id(sp) for sp in retained],
            "candidate_estimates": {c.split_id(sp): est[sp] for sp in retained},
            "rho": rho, "component_membership": agg.get("members", []),
            "return_kind": agg["return_kind"], "estimate": agg["output"],
            "baselines": baselines, "per_rotation": per_rot, "complete": True,
            "timing_seconds": time.time() - t0}


def ps_common_radius(matrix: np.ndarray, seed: int) -> float:
    import probe_structured_scm as ps
    return ps.common_radius(matrix, seed)


def hd_aggregate(labels, estimates, rho):
    import probe_structured_scm as ps
    return ps.aggregate(labels, estimates, rho)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("development", "confirmation"), required=True)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    proto = freeze()
    seeds = FROZEN["dev_seeds"] if args.phase == "development" else FROZEN["confirm_seeds"]
    c.check_protocol(proto, {"experiment_id": "SCM-6", "mode": "operational_rotate4",
                             "phase": args.phase, "seeds": seeds}, PROTOCOL_PATH, SOURCES)
    if args.seeds:
        seeds = seeds[:args.seeds]
    root = c.result_root("SCM-6", "operational_rotate4", args.phase, PROTOCOL_PATH, SOURCES)
    print(f"SCM-6 | {args.phase} | raw W dim {FROZEN['raw_w_dim']} | n={FROZEN['total_n']} | "
          f"{len(seeds)} seeds")
    for s in seeds:
        cell = one_replicate(s)
        root["cells"].append(cell)
        print(f"  seed {s}: {cell['return_kind']} est={cell['estimate']} "
              f"retained={len(cell.get('retained_splits', []))} "
              f"corr={cell.get('representation_diagnostics', {}).get('channel_correlation_median')} "
              f"({cell['timing_seconds']:.0f}s)")
    root["complete"] = True
    c.write_json_new(args.out, root)
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
