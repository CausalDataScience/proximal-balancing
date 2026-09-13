"""P2 diagnostic: one coefficient, three objectives on the same data, and an optimizer that learns only r.

Design F with held-out block S = {1} and the representation family

    Z_r = (X, C, mean of the four kept measurements + r I),

so the whole representation is indexed by one number.  Balance holds exactly at the two roots of
r^2 - 3r + 1/k = 0 with k = 4.  The study has three parts, and they share data, preprocessing, critic
architecture and critic budget so that the only thing that differs is what selects r.

  population diagnosis   D^2(r) and tau_Z(r) from the structural equations, no sampling.
  fixed grid             the finite-sample objective on a pre-specified grid, scored two ways: with the true
                         conditional treatment probabilities, and with fitted critics.  Selection uses the
                         validation rows; the audit rows are reported separately as a post hoc check.
  single-coefficient     the ordinary training loop with a one-parameter encoder, same rows, same critics.
  learning               Its selected r is compared against the grid minimiser of the same objective.

That comparison separates four failure modes: a population objective that barely distinguishes r, a
finite-sample objective that cannot resolve good from bad, fitted critics that distort the objective, and an
optimizer that fails to find a good value the objective does reward.

The exact roots and the true probabilities are diagnostics.  They are never used to initialise the learner,
to pick a checkpoint, or to choose among restarts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import probe_dgp_ae as dgp
import provenance as prov
import probe_brier_neural as pbn
import staged_dgp_f as sf

S = (1,)
SEED = 20260912
GRID_LO, GRID_HI, GRID_N = -0.5, 1.5, 21      # fixed before any result was seen
R_INIT = 0.0                                   # fixed starting value, not a root
OBJECTIVE_TIE_TOL = 1e-12
PROTOCOL_ID = "dgp-f-r10-p0-v1"


def common_objective(gap: float) -> float:
    """The frozen manuscript empirical objective used to compare every candidate in both modes."""
    return max(float(gap), 0.0)


def attach_finite_candidate_regrets(candidates: list[dict]) -> None:
    """Attach select and audit regret relative to this finite, predeclared candidate collection."""
    if not candidates:
        raise ValueError("cannot compute regret for an empty candidate set")
    for split in ("select", "audit"):
        key = f"{split}_objective"
        vals = [float(c[key]) for c in candidates]
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"{key} must be finite")
        best = min(vals)
        for c, value in zip(candidates, vals):
            c[f"{split}_regret"] = float(value - best)


def select_candidate(candidates: list[dict], objective_key: str) -> dict:
    """Return the empirical minimiser set and the fixed truth-free point selection.

    Ties are objective values within ``OBJECTIVE_TIE_TOL`` of the observed minimum.  The operational point
    is the tied value with smallest absolute coefficient, then the smaller signed coefficient.  Outcome or
    population-truth fields never enter the rule.
    """
    if not candidates:
        raise ValueError("cannot select from an empty candidate set")
    vals = np.asarray([float(c[objective_key]) for c in candidates])
    if not np.all(np.isfinite(vals)):
        raise ValueError("candidate objectives must be finite")
    lo = float(vals.min())
    tied = [c for c, value in zip(candidates, vals) if value <= lo + OBJECTIVE_TIE_TOL]
    ordered = sorted(tied, key=lambda c: (abs(float(c["r"])), float(c["r"])))
    return {"minimum": lo, "n_tied": len(tied),
            "tie_r": sorted(float(c["r"]) for c in tied),
            "selected_r": float(ordered[0]["r"]), "selected": ordered[0]}


def values_of(rows: np.ndarray, data: dict) -> np.ndarray:
    """Realise coefficient rows over the noise vector as data columns."""
    noise = np.column_stack([
        data["U"], data["I1"], data["C"], (data["X"][:, 0] - data["U"]) / dgp.F_X_NOISE,
        *[data["M"][:, j] - data["U"] for j in range(5)],
        *[(data["blocks"][j]["v"][:, 1] - data["C"]) / dgp.F_C_NOISE for j in range(1, 5)],
        *[data["blocks"][j]["v"][:, 2] for j in range(1, 5)]])
    return noise @ rows.T


def true_conditional_probabilities(data: dict, r: float) -> tuple[np.ndarray, np.ndarray]:
    """P(A=1 | Z_r) and P(A=1 | T_S, Z_r) per row, from the structural equations.

    Each is a conditional expectation of 0.05 + 0.9 expit(V) given a Gaussian conditioning vector, so it is a
    one-dimensional quadrature around the conditional mean of V; that mean is itself a linear combination of
    the same independent noises, so its per-row value follows by realising its coefficient row.
    """
    maps = sf.f_linear_maps()
    Z = sf.z_map("family", S, r, maps)
    T = np.vstack([maps["X"][None, :]] + [maps[f"W{b}"] for b in S])
    out = []
    for rows in (Z, np.vstack([Z, T])):
        mean_row, resid = sf._conditional(maps["V"], rows)
        mu = values_of(mean_row[None, :], data)[:, 0]
        out.append(sf.smoothed_prob(mu, math.sqrt(resid), 40))
    return out[0], out[1]


def oracle_gap_rows(data: dict, r: float, idx: np.ndarray) -> dict:
    """Row-level Brier gap on `idx` using the true conditional probabilities as the two critics."""
    p0, p1 = true_conditional_probabilities(data, r)
    a = data["A"][idx]
    d = (a - p0[idx]) ** 2 - (a - p1[idx]) ** 2
    return {"audit_rows": np.asarray(idx, dtype=np.int64), "audit_gap_rows": d.astype(np.float64),
            "gap": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d)))}


def learn_coefficient(data: dict, rest: list[int], st: dict, d_fit: np.ndarray, d_val: np.ndarray,
                      opt_seed: int, data_seed: int = SEED) -> dict:
    """Run the ordinary training loop with the one-parameter encoder and record the coefficient trajectory."""
    specs = pbn.block_specs(data, rest)

    def factory(sp, d_x):
        return sf.CoefficientRepresentation(sp, d_x, r_init=R_INIT, st=st)

    saved = pbn.ENCODER_FACTORY
    trace: list[float] = []
    original_step = torch.optim.AdamW.step

    def traced_step(self, *a, **kw):
        out = original_step(self, *a, **kw)
        for group in self.param_groups:
            for q in group["params"]:
                if q.numel() == 1 and q.requires_grad and q.dim() == 0:
                    trace.append(float(q.detach()))
        return out

    pbn.ENCODER_FACTORY = factory
    torch.optim.AdamW.step = traced_step
    try:
        fitted = pbn.train_representation(data, S, d_fit, d_val, st, (data_seed, opt_seed))
    finally:
        torch.optim.AdamW.step = original_step
        pbn.ENCODER_FACTORY = saved
    r_hat = float(fitted["enc"].r.detach())
    return {"fitted": fitted, "r_hat": r_hat, "trace": trace[:2000],
            "n_learnable": sum(q.numel() for q in fitted["enc"].parameters() if q.requires_grad)}


def one_dataset(args, data_seed: int, run_id: str, rowdir: Path | None) -> dict:
    """Run the whole probe on one dataset and return every quantity, with row-level losses saved."""
    data = dgp.generate("F", args.n, replicate=data_seed, phase="pilot", coef=dgp.coefficients())
    # Four roles, so that the grid and the learner can select on the same rows without those rows ever having
    # fitted a critic parameter: `train` fits the critics, `stop` early stops the grid's critics, `select`
    # carries the objective that chooses r for both paths, `audit` is post hoc only.
    perm = np.random.default_rng(data_seed).permutation(args.n)
    q = args.n // 5
    d_train, d_stop, d_select, d_audit = perm[:2 * q], perm[2 * q:3 * q], perm[3 * q:4 * q], perm[4 * q:]
    sf.assert_disjoint("coefficient probe select", d_select, train=d_train, stop=d_stop, audit=d_audit)
    sf.assert_disjoint("coefficient probe audit", d_audit, train=d_train, stop=d_stop, select=d_select)
    st = pbn.build_standardizers(data, d_train)
    rest = [b for b in range(dgp.N_BLOCKS) if b not in set(S)]
    grid = [round(x, 6) for x in np.linspace(GRID_LO, GRID_HI, GRID_N)]

    def common_eval_for_r(r: float) -> dict:
        z = sf.f_representation(data, S, r)
        frozen = {"enc": sf.FrozenZ(z), "rest": rest,
                  "specs": (pbn.block_specs(data, rest), pbn.block_specs(data, list(S)))}
        return pbn.common_critic_evaluate(data, S, frozen, d_train, d_stop, d_select, d_audit, st,
                                          seed=data_seed + 7)

    curve = []
    for r in grid:
        b = sf.benchmark_rows(S, sf.z_map("family", S, r, sf.f_linear_maps()))
        common = common_eval_for_r(r)
        orc_sel, orc_aud = oracle_gap_rows(data, r, d_select), oracle_gap_rows(data, r, d_audit)
        if rowdir is not None:
            np.savez_compressed(rowdir / f"ds{data_seed}_r{r:+.3f}.npz",
                                select_rows=common["rows_select"], select_fitted=common["gap_rows_select"],
                                select_oracle=orc_sel["audit_gap_rows"], audit_rows=common["rows_audit"],
                                audit_fitted=common["gap_rows_audit"], audit_oracle=orc_aud["audit_gap_rows"])
        curve.append({"r": r, "population_D2": b["D2"], "tau_Z": b["tau_Z"],
                      "representation_bias": abs(b["tau_Z"] - dgp.TRUE_ATE),
                      "finite_sample_oracle_gap_select": orc_sel["gap"], "oracle_se_select": orc_sel["se"],
                      "finite_sample_oracle_gap_audit": orc_aud["gap"], "oracle_se_audit": orc_aud["se"],
                      "common_gap_select": common["gap_select"], "common_se_select": common["se_select"],
                      "common_gap_audit": common["gap_audit"], "common_se_audit": common["se_audit"],
                      "common_R0_select": common["R0_select"], "common_R1_select": common["R1_select"],
                      "common_R0_audit": common["R0_audit"], "common_R1_audit": common["R1_audit"],
                      "common_select_objective": common_objective(common["gap_select"]),
                      "common_audit_objective": common_objective(common["gap_audit"])})

    def minimisers(key: str) -> dict:
        values = [{"r": c["r"], "objective": (c[key] if key == "population_D2"
                                                else common_objective(c[key])), "row": c}
                  for c in curve]
        selected = select_candidate(values, "objective")
        tie = [q["row"] for q in values if q["r"] in selected["tie_r"]]
        point = selected["selected"]["row"]
        bias = [t["representation_bias"] for t in tie]
        return {"objective": key, "minimum": selected["minimum"], "n_tied": selected["n_tied"],
                "r_values": selected["tie_r"], "selected_r": selected["selected_r"],
                "selected_representation_bias": point["representation_bias"],
                "selected_success": bool(point["representation_bias"] <= 0.11),
                "bias_min": float(min(bias)), "bias_max": float(max(bias)),
                "bias_median": float(np.median(bias))}

    mins = {k: minimisers(k) for k in ("population_D2", "finite_sample_oracle_gap_select",
                                       "common_gap_select")}
    picks = []
    for os_ in range(args.opt_seeds):
        L = learn_coefficient(data, rest, st, d_train, d_stop, os_, data_seed)
        fitted = L["fitted"]
        pop = sf.benchmark_rows(S, sf.z_map("family", S, L["r_hat"], sf.f_linear_maps()))
        g = pbn.common_critic_evaluate(data, S, fitted, d_train, d_stop, d_select, d_audit, st,
                                       seed=data_seed + 7)
        obj = pbn.encoder_objective_report(data, S, fitted, d_train, st)
        ck = sf.save_checkpoint(f"coefprobe_ds{data_seed}_os{os_}", fitted, st,
                                {"S": S, "r_hat": L["r_hat"]}, run_id)
        picks.append({"data_seed": data_seed, "opt_seed": os_, "n_learnable": L["n_learnable"],
                      "r_hat": L["r_hat"], "r_init": R_INIT, "restart": fitted["restart"],
                      "epoch": fitted["epoch"], "encoder_updates": fitted["encoder_updates"],
                      "val_gap": fitted["val_gap"], "val_objective": fitted["val_objective"],
                      "population_D2": pop["D2"], "tau_Z": pop["tau_Z"],
                      "representation_bias": abs(pop["tau_Z"] - dgp.TRUE_ATE),
                      "common_evaluation": {"select_gap": g["gap_select"], "audit_gap": g["gap_audit"],
                                            "select_objective": common_objective(g["gap_select"]),
                                            "audit_objective": common_objective(g["gap_audit"]),
                                            "R0_select": g["R0_select"], "R1_select": g["R1_select"],
                                            "R0_audit": g["R0_audit"], "R1_audit": g["R1_audit"]},
                      "encoder_objective": obj, "trace_head": L["trace"][:20],
                      "trace_tail": L["trace"][-20:], "checkpoint": ck, "complete": True})

    candidates = []
    for c in curve:
        candidates.append({"candidate_id": f"grid:{c['r']:+.6f}", "source": "grid", "r": c["r"],
                           "select_objective": c["common_select_objective"],
                           "audit_objective": c["common_audit_objective"],
                           "representation_bias": c["representation_bias"]})
    for q in picks:
        ce = q["common_evaluation"]
        candidates.append({"candidate_id": f"learned:{q['opt_seed']}", "source": "learned",
                           "r": q["r_hat"], "select_objective": ce["select_objective"],
                           "audit_objective": ce["audit_objective"],
                           "representation_bias": q["representation_bias"]})
    attach_finite_candidate_regrets(candidates)
    by_id = {c["candidate_id"]: c for c in candidates}
    surrogate_min = min(float(q["val_objective"]) for q in picks)
    for q in picks:
        c = by_id[f"learned:{q['opt_seed']}"]
        q["common_evaluation"].update({"select_regret": c["select_regret"],
                                       "audit_regret": c["audit_regret"]})
        q["minibatch_surrogate"] = {"value": float(q["val_objective"]),
                                      "regret_among_optimizer_picks": float(q["val_objective"] - surrogate_min),
                                      "not_comparable_to_common_objective": True}
    selected = select_candidate(candidates, "select_objective")
    actual = selected["selected"]
    common_selection = {"candidate_set": "predeclared grid union learned r_hat",
                        "tie_tolerance": OBJECTIVE_TIE_TOL,
                        "tie_break": "smallest abs(r), then r", "n_tied": selected["n_tied"],
                        "tie_r": selected["tie_r"], "selected_candidate_id": actual["candidate_id"],
                        "selected_source": actual["source"], "selected_r": actual["r"],
                        "selected_representation_bias": actual["representation_bias"],
                        "selected_success": bool(actual["representation_bias"] <= 0.11),
                        "success_definition": "abs(tau_Z - 1) <= 0.11; diagnostic only"}
    return {"data_seed": data_seed, "curve": curve, "grid_minimisers": mins, "picks": picks,
            "common_candidate_selection": common_selection,
            "diagnosis": {"population": True, "true_probability": True,
                          "fitted_common": True, "optimizer_regret": True},
            "rows": {"train": sf.split_id(d_train), "stop": sf.split_id(d_stop),
                     "select": sf.split_id(d_select), "audit": sf.split_id(d_audit)},
            "n_rows": {"train": len(d_train), "stop": len(d_stop), "select": len(d_select),
                       "audit": len(d_audit)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--data-seeds", type=int, default=1)
    ap.add_argument("--save-rows", action="store_true",
                    help="write per-row fitted and oracle gaps so every standard error can be recomputed")
    ap.add_argument("--opt-seeds", type=int, default=5)
    ap.add_argument("--training-mode", choices=["variant", "manuscript"], default="manuscript")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--out", default="../results/dgp_f_coefficient_probe.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%S")
    pbn.Z_DIM = sf.LINEAR_Z_DIM
    pbn.TRAINING_MODE = args.training_mode
    sf.RUN_ID = run_id
    criteria_path = Path(__file__).with_name("frozen_criteria.json")
    criteria_sha = prov.file_sha256(criteria_path)
    source_manifest = prov.snapshot_run_sources(
        Path(args.out).with_suffix("") / f"run-{run_id}",
        {name: Path(__file__).with_name(name) for name in
         ("coefficient_probe.py", "staged_dgp_f.py", "probe_brier_neural.py", "probe_dgp_ae.py",
          "provenance.py", "frozen_criteria.json")})

    k = 4
    roots = [dgp.f_balance_coefficient(k), (3.0 + math.sqrt(9.0 - 4.0 / k)) / 2.0]
    print(f"design F coefficient probe | n={args.n} | mode {args.training_mode} | run {run_id} | "
          f"grid {GRID_N} points on [{GRID_LO}, {GRID_HI}] | r_init {R_INIT:+.3f} | "
          f"{args.data_seeds} data seed(s) x {args.opt_seeds} optimizer seed(s) | roots {roots[0]:.6f}, "
          f"{roots[1]:.6f} | code {source_manifest['bundle_sha256'][:16]}")
    print("the roots and the true probabilities are diagnostics; the learner never receives them")

    rowdir = None
    if args.save_rows:
        rowdir = Path(args.out).with_suffix("") / "rows"
        rowdir.mkdir(parents=True, exist_ok=True)
        print(f"per-row gaps will be written under {rowdir}")

    runs = []
    for ds in range(args.data_seeds):
        data_seed = SEED + ds
        r = one_dataset(args, data_seed, run_id, rowdir)
        runs.append(r)
        n = r["n_rows"]
        print(f"\n--- data seed {data_seed}: {n['train']} train, {n['stop']} stop, {n['select']} select, "
              f"{n['audit']} audit")
        rows = [{"r": f"{c['r']:+.3f}", "population D2": f"{c['population_D2']:.2e}",
                 "|tau_Z-1|": f"{c['representation_bias']:.4f}",
                 "oracle gap (select)": f"{c['finite_sample_oracle_gap_select']:+.2e}",
                 "oracle se": f"{c['oracle_se_select']:.1e}",
                 "common gap (select)": f"{c['common_gap_select']:+.2e}",
                 "common se": f"{c['common_se_select']:.1e}",
                 "common gap (audit)": f"{c['common_gap_audit']:+.2e}"} for c in r["curve"]]
        sf.report(rows, list(rows[0].keys()), "[probe] one grid, three objectives (selection on `select`)")
        for key, v in r["grid_minimisers"].items():
            print(f"[probe] grid minimiser of {key:32s}: {v['n_tied']} tied point(s) at r in "
                  f"{[f'{x:+.3f}' for x in v['r_values'][:6]]}, representation bias "
                  f"{v['bias_min']:.4f} to {v['bias_max']:.4f}; operational r "
                  f"{v['selected_r']:+.3f}, success={v['selected_success']}")
        for q in r["picks"]:
            print(f"[probe] learner ds{data_seed} os{q['opt_seed']}: {q['n_learnable']} learnable parameter, "
                  f"r_init {q['r_init']:+.3f} -> r_hat {q['r_hat']:+.5f}, val objective "
                  f"{q['val_objective']:.2e}, population D2 {q['population_D2']:.2e}, |tau_Z-1| "
                  f"{q['representation_bias']:.4f}, restart {q['restart']}")

    # ---- across data seeds: bias, variability, and how often each objective separates the exact root
    ladder_rows, agg = [], {}
    for name, key in (("population objective", "population_D2"),
                      ("finite sample, true probabilities", "finite_sample_oracle_gap_select"),
                      ("finite sample, common fitted critics", "common_gap_select")):
        b = [r["grid_minimisers"][key]["selected_representation_bias"] for r in runs]
        ties = [r["grid_minimisers"][key]["n_tied"] for r in runs]
        near = sum(1 for r in runs if r["grid_minimisers"][key]["selected_success"])
        agg[key] = {"bias_median_over_seeds": float(np.median(b)), "bias_min": float(min(b)),
                    "bias_max": float(max(b)), "ties": ties, "datasets_selecting_near_the_root": near,
                    "datasets": len(runs)}
        ladder_rows.append({"selected by": name, "median bias": f"{np.median(b):.4f}",
                            "min": f"{min(b):.4f}", "max": f"{max(b):.4f}",
                            "tied minima": str(ties),
                            "picks near the root": f"{near}/{len(runs)}"})
    learner_bias = [q["representation_bias"] for r in runs for q in r["picks"]]
    per_ds = [float(np.median([q["representation_bias"] for q in r["picks"]])) for r in runs]
    selected_bias = [float(r["common_candidate_selection"]["selected_representation_bias"]) for r in runs]
    select_regret = [float(q["common_evaluation"]["select_regret"]) for r in runs for q in r["picks"]]
    audit_regret = [float(q["common_evaluation"]["audit_regret"]) for r in runs for q in r["picks"]]
    ladder_rows.append({"selected by": "the one-parameter optimizer",
                        "median bias": f"{np.median(learner_bias):.4f}",
                        "min": f"{min(learner_bias):.4f}", "max": f"{max(learner_bias):.4f}",
                        "tied minima": "n/a",
                        "picks near the root": f"{sum(1 for q in learner_bias if q <= 0.11)}/"
                                               f"{len(learner_bias)}"})
    sf.report(ladder_rows, ["selected by", "median bias", "min", "max", "tied minima", "picks near the root"],
              f"[probe] representation bias by what selected r, over {len(runs)} dataset(s)")

    med = float(np.median(learner_bias))
    verdict = (f"finite-candidate empirical regret over grid union learned r_hat: median select "
               f"{np.median(select_regret):.6g}, median untouched-audit {np.median(audit_regret):.6g}; "
               "these are not global encoder optimisation bounds")
    print(f"\n[probe] {verdict}")

    header = prov.header(run_id=run_id, script="coefficient_probe.py", args=vars(args),
                         training_mode=args.training_mode, code_hash=source_manifest["bundle_sha256"][:16],
                         seeds={"data": [SEED + i for i in range(args.data_seeds)], "critic_offset": 7,
                                "optimizer": list(range(args.opt_seeds))},
                         extra={"protocol_id": PROTOCOL_ID, "criteria_sha256": criteria_sha,
                                "source_manifest": source_manifest,
                                "roots": roots, "grid": {"lo": GRID_LO, "hi": GRID_HI, "n": GRID_N},
                                "r_init": R_INIT, "row_dumps": str(rowdir) if rowdir else None,
                                "diagnostics_not_given_to_the_learner": ["exact roots",
                                                                        "true conditional probabilities"]})
    header["complete"] = True
    payload = {"provenance": header,
               "runs": runs, "across_datasets": agg,
               "learner": {"bias_per_dataset_median": per_ds, "bias_all": learner_bias,
                           "median": med, "min": float(min(learner_bias)), "max": float(max(learner_bias)),
                           "cell_success_definition": "abs(tau_Z - 1) <= 0.11",
                           "cell_successes": int(sum(x <= 0.11 for x in learner_bias)),
                           "cell_total": len(learner_bias), "selected_bias_per_dataset": selected_bias,
                           "selected_successes": int(sum(x <= 0.11 for x in selected_bias)),
                           "selected_total": len(selected_bias),
                           "finite_candidate_select_regret": select_regret,
                           "finite_candidate_audit_regret": audit_regret},
               "diagnosis_complete": all(all(v is True for v in r["diagnosis"].values()) for r in runs),
               "verdict": verdict}
    Path(args.out).write_text(json.dumps(payload, indent=1, default=float))
    print(f"[probe] written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
