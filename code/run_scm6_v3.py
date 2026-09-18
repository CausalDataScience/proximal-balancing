"""SCM-6-v3: full-target confirmation on 1,282-dimensional vector proxies.

What is new against the development runs.  The screen no longer compresses the held-out block: the
augmented critic sees all 256 coordinates through a residual branch that contains the base critic by
construction, and the pass rule is a one-sided upper confidence bound on the held-out risk difference.
Root recovery is reported but decides nothing.  In its place the run measures what the manuscript
actually claims: the returned effect, the screen's behaviour against the exactly known feasible set,
whether the true-target candidates form the unique largest component, and the causal target each
learned representation aims at, evaluated on a fresh 200,000-row sample.

Sampling.  Every confirmation seed is drawn once at n = 24,000 and the n = 12,000 arm is its prefix,
fold by fold, so the larger-n audit is a nested pair rather than a second dataset.

Rows inside the D fold at n = 12,000: 1,200 for the encoder, 300 for its calibration, 1,500 for the
coefficient search and the critics.  The encoder budget was set from an evaluator-only diagnostic on
development seeds: the learned target error at the exact root is 0.084 with 600 encoder rows, 0.040
with 1,200 and 0.027 with 1,800, and 1,800 would leave the search with 900 rows, which earlier
development showed is too few.  At n = 24,000 every allocation doubles.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import learned_target as lt
import minimal_suite_common as c
import probe_full_target as ft
import probe_highdim_w as hd
import probe_highdim_w_v2 as v2
import probe_structured_scm as ps
import run_scm6_v2_development as s6
import scm6_population as sp

SOURCES = ["minimal_suite_common.py", "probe_highdim_w.py", "probe_highdim_w_v2.py",
           "probe_full_target.py", "learned_target.py", "scm6_population.py", "run_scm6_v3.py"]
PROTOCOL_PATH = "../results/scm6_v3/protocol.json"
SHARD_DIR = "../results/scm6_v3/shards"
LARGEST_N = 24000
FROZEN = {
    "protocol_id": "scm6-v3-full-target-confirmation",
    "experiment_id": "SCM-6-v3", "mode": "operational_rotate4",
    "arms": {"confirmation": {"n": 12000, "seeds": [94200000 + 1000 * r for r in range(30)]},
             "larger_n": {"n": 24000, "seeds": [94200000 + 1000 * r for r in range(10)]}},
    "sampling": "one draw per seed at n = 24,000; the n = 12,000 arm is its fold-wise prefix",
    "d_fold_rows_at_12000": {"encoder_fit": 1200, "encoder_cal": 300, "search_and_critics": 1500},
    "selection": "cross-fitted nested Brier gap over the frozen r grid, ridge-probit critics, "
                 "compressed one-component target; the v2 development selector, unchanged",
    "screen": {"critics": "base MLP on Z; augmented = base + residual MLP on (Z, full 256-dim W_S), "
                          "hidden 64, SiLU, output in [0.05, 0.95], 2 restarts, early stopping on "
                          "validation Brier, lifted base always a checkpoint candidate",
               "statistic": "held-out Brier risk difference on the S fold",
               "rule": "max(gap, 0) + z_{1 - alpha/N} SE < tolerance, and no saturation",
               "tolerance": 2e-3, "alpha": 0.05, "N": 30,
               "population_window": [2.99e-5, 7.72e-3],
               "retention": "pass in all four rotations"},
    "aggregation": "common radius from the evaluation scores, link within 2 rho, median of the unique "
                   "largest component; a tie is no_return",
    "comparators": ["naive", "X", "raw_ridge", "learned_projection", "blockwise_pca", "oracle"],
    "learned_target_evaluator": {"n": 200000, "seed_rule": "424242 + data seed"},
    "decision_record": {
        "screen_tolerance": "2e-3, set after one development seed (94000000) showed the one-sided bound's "
                            "slack z*SE reaches 2e-3 to 4e-3 whenever the residual branch moves at all, so "
                            "the earlier 1e-3 retained only 2 of 8 balance-feasible splits across four "
                            "rotations. From the recorded U values, 2e-3 retains 4 of 8 and passes no "
                            "infeasible row (their minimum U was 3.36e-3). Inside the population window "
                            "(2.99e-5, 7.72e-3). Chosen by the user on 2026-09-14 from options 1e-3, 2e-3, "
                            "2.5e-3 before any confirmation seed was opened.",
        "learned_target_gate": "median 0.10, p90 0.15, relaxed from the reviewer's 0.05 / 0.10 by the "
                               "user's decision on 2026-09-14 after the same development seed gave "
                               "0.076 / 0.122 over the splits that target 1 (encoder error 0.040 at the "
                               "exact root with 1,200 encoder rows, plus selection error). This is a "
                               "threshold set with knowledge of one development value; it is recorded "
                               "here so the sealed protocol cannot be read as pre-specified in the "
                               "stricter sense. No confirmation seed had been run.",
    },
    "gates": {
        "performance": {"return_rate_min": 0.95, "mae_max": 0.05, "p90_max": 0.10,
                        "oracle_excess_upper_max": 0.05, "beats": ["X", "raw_ridge", "best_no_balance"],
                        "alpha": 0.05, "bonferroni_contrasts": 4},
        "mechanism": {"screen_tpr_min": 0.80, "screen_fpr_max": 0.20,
                      "unique_largest_rate_min": 0.90, "component_purity_min": 0.80,
                      "learned_target_error_median_max": 0.10,
                      "learned_target_error_p90_max": 0.15},
        "secondary_only": ["root_recovery", "median_root_error", "channel_correlation",
                           "selected_r_by_component", "runtime"]},
}


def freeze() -> dict:
    p = Path(PROTOCOL_PATH)
    if p.exists():
        return json.loads(p.read_text())
    proto = dict(FROZEN)
    proto["source_snapshot"] = c.source_snapshot(SOURCES)
    proto["frozen_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    proto["runs"] = [{"experiment_id": "SCM-6-v3", "mode": "operational_rotate4", "phase": "confirmation",
                      "arm": arm, "n": spec["n"], "seeds": spec["seeds"]}
                     for arm, spec in FROZEN["arms"].items()]
    return c.seal_protocol(p, proto)


def nested_draw(seed: int, n: int) -> tuple[dict, list[np.ndarray]]:
    """The largest draw for this seed, and folds cut to n as prefixes of the folds at the largest n."""
    data = hd.generate(LARGEST_N, seed)
    folds_big = c.four_folds(LARGEST_N, seed + 1)
    folds = c.nested_prefix(folds_big, n // 4)
    keep = np.sort(np.concatenate(folds))
    # re-index so the sliced data is a self-contained array set and folds address it directly
    pos = {row: i for i, row in enumerate(keep)}
    sliced = {k: ([b[keep] for b in v] if k == "H" else v[keep]) for k, v in data.items()}
    folds = [np.array([pos[r] for r in f]) for f in folds]
    return sliced, folds


def component_labels(agg: dict) -> list[str]:
    """Split ids of the largest component.  The aggregator explodes each string label into a list of
    characters, so ['S', '1', '2'] has to be joined back to 'S12' rather than re-encoded."""
    if not agg.get("members"):
        return []
    return ["".join(m) if not isinstance(m, str) else m for m in agg["members"][0]]


def evaluator_sample(seed: int) -> tuple[dict, dict]:
    big = hd.generate(FROZEN["learned_target_evaluator"]["n"], 424242 + seed)
    return big, hd.observed_view(big, include_outcome=True)


def one_replicate(seed: int, n: int, facts: dict, dev) -> dict:
    t0 = time.time()
    data, folds = nested_draw(seed, n)
    view = hd.observed_view(data, include_outcome=True)
    scale = n // 12000
    emb_fit, emb_cal = 1200 * scale, 300 * scale
    splits = c.oriented_splits()
    scr_spec = FROZEN["screen"]
    rows_out, corrs = [], []
    scores = {s: [] for s in splits}
    passes = {s: 0 for s in splits}
    base_scores: dict[str, list] = {}
    learned: dict[str, list] = {}
    _, big_view = evaluator_sample(seed)
    for rot in c.rotations(folds):
        c.assert_roles_disjoint(rot.roles)
        d_idx = rot.roles["D"]
        emb = hd.fit_embedding(s6.v1.slice_rows(view, d_idx[:emb_fit]),
                               s6.v1.slice_rows(view, d_idx[emb_fit:emb_fit + emb_cal]))
        if emb["fail_closed"]:
            return {"data_seed": seed, "n_total": n, "return_kind": "no_return", "estimate": None,
                    "fail_closed": emb["fail_closed"], "complete": True, "timing_seconds": time.time() - t0}
        corrs.append(float(np.median(hd.channel_correlation(emb, data, rot.roles["E"]))))
        phi = s6.v1.slice_rows(view, d_idx[emb_fit + emb_cal:])
        s_rows, n_rows, e_rows = (s6.v1.slice_rows(view, rot.roles[k]) for k in ("S", "N", "E"))
        k_phi, k_s = hd.apply_embedding(emb, phi), hd.apply_embedding(emb, s_rows)
        k_n, k_e = hd.apply_embedding(emb, n_rows), hd.apply_embedding(emb, e_rows)
        k_big = hd.apply_embedding(emb, big_view)
        pcs = hd.blockwise_pca(s6.v1.slice_rows(view, d_idx[:emb_fit]))
        feat_n, feat_e = hd.baseline_features(n_rows, emb, pcs), hd.baseline_features(e_rows, emb, pcs)
        for name in feat_n:
            base_scores.setdefault(name, []).append(hd.aipw_from_features(
                feat_n[name], n_rows["A"], n_rows["Y"], feat_e[name], e_rows["A"], e_rows["Y"]))
        base_scores.setdefault("oracle", []).append(hd.aipw_from_features(
            np.column_stack([n_rows["X"], data["U_eval_only"][rot.roles["N"]], n_rows["C"]]),
            n_rows["A"], n_rows["Y"],
            np.column_stack([e_rows["X"], data["U_eval_only"][rot.roles["E"]], e_rows["C"]]),
            e_rows["A"], e_rows["Y"]))
        for split in splits:
            sid = c.split_id(split)
            oseed = c.optimizer_seed(FROZEN["protocol_id"], seed, rot.index, sid, 0)
            proj = v2.fit_target_projection(phi, split, 1, oseed)
            pen = hd.split_penalties(phi, k_phi, split, s6.PENALTY_GRID, oseed)["base"]
            sel = v2.select_r_crossfit(phi, k_phi, split, pen, oseed, projection=proj)
            z_phi, t_phi = hd.representation(k_phi, phi, split, sel["r"]), hd.heldout(phi, split)
            z_s, t_s = hd.representation(k_s, s_rows, split, sel["r"]), hd.heldout(s_rows, split)
            base = ft.fit_base(z_phi, phi["A"], oseed, dev)
            aug = ft.fit_augmented(base, z_phi, t_phi, phi["A"], oseed, dev)
            nest = ft.nesting_check(base, aug, z_phi, t_phi, phi["A"], dev)
            scr = ft.screen_full_target(base, aug, z_s, t_s, s_rows["A"], scr_spec["tolerance"],
                                        scr_spec["alpha"], scr_spec["N"], dev)
            fact = facts[sid]
            row = {"rotation": rot.index, "split_id": sid, "r": sel["r"], "penalty": pen,
                   "selection_objective": sel["objective"],
                   "screen": {k: scr[k] for k in ("gap", "se", "z", "upper", "pass", "overlap_ok",
                                                  "saturated_fraction", "augmented_kept_lifted_base",
                                                  "squared_difference")},
                   "nesting": nest,
                   "critic_fit": {"base_val": base["info"]["best_val"],
                                  "augmented_val": aug["info"]["best_val"],
                                  "base_epochs": base["info"]["epochs_run"],
                                  "augmented_epochs": aug["info"]["epochs_run"]},
                   "evaluator_only": {"balance_root": fact["root"], "feasible": fact["feasible"],
                                      "targets_truth": fact["targets_truth"],
                                      "tau_at_root": fact["tau_at_root"],
                                      "root_error": (None if fact["root"] is None or sel["r"] is None
                                                     else abs(sel["r"] - fact["root"]))}}
            if scr["pass"]:
                passes[split] += 1
                scores[split].append(hd.aipw_from_features(
                    hd.representation(k_n, n_rows, split, sel["r"]), n_rows["A"], n_rows["Y"],
                    hd.representation(k_e, e_rows, split, sel["r"]), e_rows["A"], e_rows["Y"]))
                z_big = hd.representation(k_big, big_view, split, sel["r"])
                lt_out = lt.learned_target(z_big, big_view["A"], big_view["Y"], oseed, dev)
                row["evaluator_only"]["learned_target"] = lt_out["theta"]
                row["evaluator_only"]["learned_target_linear"] = lt_out["theta_linear"]
                learned.setdefault(sid, []).append(lt_out["theta"])
            rows_out.append(row)

    retained = [s for s in splits if passes[s] == 4]
    est = {c.split_id(s): float(np.concatenate(scores[s]).mean()) for s in retained}
    if retained:
        mat = np.column_stack([np.concatenate(scores[s]) for s in retained])
        rho = float(ps.common_radius(mat, seed + 5))
        agg = ps.aggregate([c.split_id(s) for s in retained],
                           np.array([est[c.split_id(s)] for s in retained]), rho)
    else:
        rho, agg = None, {"return_kind": "no_return", "output": None, "members": []}
    members = component_labels(agg)
    purity = (float(np.mean([facts[m]["targets_truth"] for m in members])) if members else None)
    baselines = {k: float(np.concatenate(v).mean()) for k, v in base_scores.items()}
    baselines["naive"] = float(view["Y"][view["A"] == 1].mean() - view["Y"][view["A"] == 0].mean())
    return {"data_seed": seed, "n_total": n, "rows": rows_out,
            "retained_splits": [c.split_id(s) for s in retained], "candidate_estimates": est,
            "rho": rho, "return_kind": agg["return_kind"], "estimate": agg["output"],
            "largest_component": members, "largest_component_purity": purity,
            "learned_target_by_split": {k: float(np.median(v)) for k, v in learned.items()},
            "baselines": baselines,
            "representation_diagnostics": {"channel_correlation_median": float(np.median(corrs))},
            "complete": True, "timing_seconds": time.time() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(FROZEN["arms"]), required=True)
    ap.add_argument("--seeds", type=int, default=None, help="run only the first k seeds of the arm")
    ap.add_argument("--only-seed", type=int, default=None)
    args = ap.parse_args()
    proto = freeze()
    arm = FROZEN["arms"][args.arm]
    c.check_protocol(proto, {"experiment_id": "SCM-6-v3", "mode": "operational_rotate4",
                             "phase": "confirmation", "arm": args.arm, "n": arm["n"],
                             "seeds": arm["seeds"]}, PROTOCOL_PATH, SOURCES)
    seeds = arm["seeds"][:args.seeds] if args.seeds else arm["seeds"]
    if args.only_seed is not None:
        seeds = [args.only_seed]
        if args.only_seed not in arm["seeds"]:
            raise SystemExit(f"seed {args.only_seed} is not in the frozen {args.arm} arm")
    facts = s6.truth()
    dev = ft.device()
    Path(SHARD_DIR).mkdir(parents=True, exist_ok=True)
    print(f"SCM-6-v3 | {args.arm} | n={arm['n']} | {len(seeds)} seeds | device {dev} | "
          f"protocol {proto[c.DIGEST_FIELD][:12]}")
    for seed in seeds:
        shard = Path(SHARD_DIR) / f"{args.arm}_seed{seed}_n{arm['n']}.json"
        if shard.exists():
            print(f"  seed {seed}: shard exists, skipping")
            continue
        cell = one_replicate(seed, arm["n"], facts, dev)
        cell["protocol"] = {"path": PROTOCOL_PATH, c.DIGEST_FIELD: proto[c.DIGEST_FIELD]}
        cell["source_snapshot"] = c.source_snapshot(SOURCES)
        c.write_json_new(shard, cell)
        print(f"  seed {seed}: {cell['return_kind']} est={cell['estimate']} "
              f"retained={len(cell.get('retained_splits', []))} "
              f"purity={cell.get('largest_component_purity')} "
              f"corr={cell.get('representation_diagnostics', {}).get('channel_correlation_median')} "
              f"({cell['timing_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
