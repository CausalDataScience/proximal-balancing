"""Checks for the staged design F study: data separation, checkpoint reproducibility, and the benchmark.

Run:  python3 test_staged_dgp_f.py
"""
from __future__ import annotations

import math
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import probe_dgp_ae as dgp
import probe_brier_neural as pbn
import staged_dgp_f as sf

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    torch.set_num_threads(2)
    pbn.ENCODER_FACTORY = sf.LinearRepresentation
    pbn.Z_DIM = sf.LINEAR_Z_DIM
    S = (1,)

    # ---- P0 completion condition 1: injected overlap is a hard failure, a clean split passes with zero
    fit, val, audit = sf.three_way(np.arange(3000), 5)
    counts = sf.assert_disjoint("clean", audit, fit=fit, val=val)
    check("P0.1 a clean 50/25/25 split has zero forbidden overlap", counts == {"fit": 0, "val": 0}, f"{counts}")
    check("P0.1 the three parts cover every row exactly once",
          len(set(fit) | set(val) | set(audit)) == 3000 and len(fit) + len(val) + len(audit) == 3000)
    raised = False
    try:
        sf.assert_disjoint("dirty", np.concatenate([audit, fit[:10]]), fit=fit, val=val)
    except ValueError as e:
        raised = "overlap" in str(e)
    check("P0.1 injecting ten training rows into the evaluation rows raises", raised)
    raised = False
    try:
        sf.evaluate_representation({}, S, np.zeros((3000, 3), np.float32), {}, 0, fit, val,
                                   np.concatenate([audit[:-5], val[:5]]))
    except ValueError:
        raised = True
    check("P0.1 the evaluator itself refuses overlapping audit rows before fitting anything", raised)

    # ---- P0 completion condition 2: a saved checkpoint reproduces the representation
    n = 3000
    data = dgp.generate("F", n, replicate=77, phase="pilot", coef=dgp.coefficients())
    idx = np.arange(n)
    L = sf.learn_representation(data, S, idx, data_seed=77, opt_seed=0)
    fitted, st = L["fitted"], L["st"]
    check("P0.2 the encoder's own audit rows are disjoint from its fit and val rows",
          sf.assert_disjoint("learn", L["audit"], fit=L["fit"], val=L["val"]) == {"fit": 0, "val": 0})
    z_before = sf.embed_rows(fitted["enc"], fitted["rest"], data, idx, st)
    import uuid as _uuid
    sf.CHECKPOINT_DIR = sf.Path("/private/tmp/claude-502/-Users-yonghanjung-paios/02dd4dd4-bfd1-480e-aa31-8cff60aa41a1/scratchpad/ckpt_test")
    rec0 = sf.save_checkpoint("test_ckpt", fitted, st, {}, "p" + _uuid.uuid4().hex[:8],
                              ("test", 0))
    check("P0 checkpoint provenance is independent of the consumer working directory",
          Path(rec0["path"]).is_absolute(), rec0["path"])
    saved = sf.load_checkpoint(rec0)
    enc2 = sf.LinearRepresentation(fitted["specs"][0], data["X"].shape[1])
    enc2.load_state_dict(saved["encoder"])
    z_after = sf.embed_rows(enc2, fitted["rest"], data, idx, saved["standardizers"])
    check("P0.2 reloading the checkpoint reproduces Z exactly", float(np.abs(z_before - z_after).max()) == 0.0,
          f"max |diff| = {np.abs(z_before - z_after).max():.1e}")
    enc0 = sf.LinearRepresentation(fitted["specs"][0], data["X"].shape[1])
    enc0.load_state_dict(saved["init"])
    z0 = sf.embed_rows(enc0, fitted["rest"], data, idx, st)
    check("P0.4 the stored initial state differs from the trained state (training happened)",
          float(np.abs(z0 - z_before).max()) > 1e-6)
    check("P0.6 the checkpoint records the optimizer seed and restart index",
          isinstance(saved["optimizer_seed"], int) and saved["restart"] in (0, 1, 2))
    check("P0 checkpoint ownership metadata is identical in the result record and saved payload",
          {k: rec0[k] for k in ("protocol_id", "run_id", "cell_key")}
          == saved["checkpoint_provenance"])

    # ---- F24 follow-up: the tolerance is on the distance improvement
    # signed change 0.02, distance change 0: the old rule accepted this, the new one must not.
    # The comparison itself now lives in `checkpoint_improves`, since the key is a pair (round 7, PRD-1).
    check("F24 a move from +0.010 to -0.010 is not an improvement",
          not pbn.checkpoint_improves(-0.010, 0.010))
    check("F24 a move from +0.010 to +0.005 is an improvement", pbn.checkpoint_improves(0.005, 0.010))

    # ---- F25: the decoupled fit trains every parameter of the augmented critic
    dec = pbn.decoupled_critic_fit(data, S, fitted, L["fit"], L["val"], L["audit"], st, 3)
    check("F25 the decoupled fit returns finite gaps on fit, val and audit",
          all(math.isfinite(dec[k]) for k in ("decoupled_gap_fit", "decoupled_gap_val", "decoupled_gap_audit")))
    src = open(pbn.__file__).read()
    check("F25 no parameter of the augmented critic is frozen in the decoupled fit",
          "requires_grad_(False)" not in src.split("def decoupled_critic_fit")[1].split("def audit_gap")[0])

    # ---- P1: the benchmark
    r_star = dgp.f_balance_coefficient(4)
    for lab, r in (("r*", r_star), ("second root", (3 + math.sqrt(9 - 1)) / 2)):
        b = sf.benchmark(S, "family", r, 40)
        check(f"P1 benchmark at {lab}: D2 below 1e-10 and tau_Z = 1 within 1e-8",
              b["D2"] < 1e-10 and abs(b["tau_Z"] - 1) < 1e-8, f"D2={b['D2']:.1e}, tau_Z={b['tau_Z']:.10f}")
    b40, b80 = sf.benchmark(S, "family", 0.0, 40), sf.benchmark(S, "family", 0.0, 80)
    check("P1 benchmark is stable when the quadrature order doubles",
          abs(b40["D2"] - b80["D2"]) / b80["D2"] < 1e-3 and abs(b40["tau_Z"] - b80["tau_Z"]) < 1e-4)
    bc = sf.benchmark(S, "constant", None, 60)
    check("P1 tau_Z of the constant representation equals the unadjusted contrast (about -0.76)",
          abs(bc["tau_Z"] + 0.760) < 0.01, f"tau_Z={bc['tau_Z']:.4f}")
    check("P1 the tower property E[m1|m0]=m0 holds in the benchmark covariance", b40["tower_defect"] < 1e-9)
    # the learned linear representation can be scored exactly, and scoring the exact map gives zero
    rows_exact = sf.z_map("family", S, r_star, sf.f_linear_maps())
    check("P1 the exact map scored through benchmark_rows gives D2 = 0", sf.benchmark_rows(S, rows_exact)["D2"] < 1e-10)
    pop = sf.benchmark_rows(S, sf.learned_z_rows(fitted, st))
    check("P1 a learned linear representation has a finite population D2 and tau_Z",
          math.isfinite(pop["D2"]) and math.isfinite(pop["tau_Z"]) and pop["D2"] >= 0, f"D2={pop['D2']:.2e} tau_Z={pop['tau_Z']:.4f}")
    # cross-check the exact scoring of a data-built family representation against z_map
    z_data = sf.f_representation(data, S, 0.5)
    lin = np.linalg.lstsq(np.column_stack([np.ones(n), data["X"][:, 0], data["C"], data["I1"], data["M"]]), z_data, rcond=None)[0]
    check("P1 the data-built family representation is an affine function of (X, C, I, M) to 1e-5",
          float(np.abs(np.column_stack([np.ones(n), data["X"][:, 0], data["C"], data["I1"], data["M"]]) @ lin - z_data).max()) < 1e-5)

    # ---- R1: the benchmark holds out T = (X, W_S), and conditioning is rank safe
    maps = sf.f_linear_maps()
    zc = np.vstack([maps["C"], maps["I"], maps["M0"]])
    d2c = sf.benchmark_rows(S, zc)["D2"]
    check("R1 a representation that drops X is scored against T = (X, W_S)",
          abs(d2c - 0.00812) < 2e-4, f"D2 = {d2c:.5f}, expected about 0.00812, not the 0.00333 of round 6")
    zr = sf.z_map("family", S, r_star, maps)
    base = sf.benchmark_rows(S, zr)
    dup = sf.benchmark_rows(S, np.vstack([zr, zr[0]]))
    lin = sf.benchmark_rows(S, np.array([[1., 2., 0.], [0., 1., 3.], [1., 0., 1.]]) @ zr)
    check("R1 adding a duplicate coordinate does not change D2 or tau_Z",
          dup["D2"] < 1e-10 and abs(dup["tau_Z"] - base["tau_Z"]) < 1e-8, f"D2={dup['D2']:.1e}")
    check("R1 an invertible linear reparametrisation does not change D2 or tau_Z",
          lin["D2"] < 1e-10 and abs(lin["tau_Z"] - base["tau_Z"]) < 1e-8, f"D2={lin['D2']:.1e}")
    tau_raw = sf.benchmark_rows(S, np.vstack([maps["X"][None, :]] + [maps[f"W{b}"] for b in range(5)]))["tau_Z"]
    tau_x = sf.benchmark_rows(S, maps["X"][None, :])["tau_Z"]
    check("R3 the raw and X adjustments do not target 1",
          abs(tau_raw - 0.635729) < 1e-4 and abs(tau_x + 0.335967) < 1e-4,
          f"tau_raw={tau_raw:.6f}, tau_X={tau_x:.6f}")

    # ---- R2: paired differences keep the evaluation sample in the uncertainty
    fake = {"audit_rows": np.arange(200), "audit_gap_rows": np.linspace(-0.01, 0.02, 200)}
    same = sf.paired_gap_difference(fake, fake)
    check("R2 identical row-level values give a zero paired difference and a zero standard error",
          same["mean"] == 0.0 and same["se"] == 0.0)
    shifted = {"audit_rows": np.arange(200), "audit_gap_rows": fake["audit_gap_rows"] + 0.005}
    sh = sf.paired_gap_difference(shifted, fake)
    check("R2 a constant row-level shift is recovered exactly with zero standard error",
          abs(sh["mean"] - 0.005) < 1e-12 and sh["se"] < 1e-12, f"mean={sh['mean']:.6f}")
    rng = np.random.default_rng(0)
    noisy = {"audit_rows": np.arange(200), "audit_gap_rows": fake["audit_gap_rows"] + rng.normal(0, 0.01, 200)}
    ns = sf.paired_gap_difference(noisy, fake)
    check("R2 differing row-level values give a strictly positive standard error", ns["se"] > 0)
    perm = rng.permutation(200)
    reordered = {"audit_rows": np.arange(200)[perm], "audit_gap_rows": noisy["audit_gap_rows"][perm]}
    check("R2 the pairing is by row id, so a reordered save gives the same answer",
          abs(sf.paired_gap_difference(reordered, fake)["mean"] - ns["mean"]) < 1e-12)
    raised = False
    try:
        sf.paired_gap_difference({"audit_rows": np.arange(5), "audit_gap_rows": np.zeros(5)}, fake)
    except ValueError:
        raised = True
    check("R2 pairing across different evaluation rows raises", raised)
    # PRD-4 problem A: seed averaging happens row by row, and the two uncertainties stay apart
    rowsA = np.arange(400)
    rng4 = np.random.default_rng(11)
    base = rng4.normal(0.001, 0.02, 400)
    seeds_a = [{"audit_rows": rowsA, "audit_gap_rows": base + rng4.normal(0, 0.001, 400)} for _ in range(5)]
    seeds_b = [{"audit_rows": rowsA, "audit_gap_rows": np.zeros(400)} for _ in range(5)]
    agg = sf.seed_averaged_paired_difference(seeds_a, seeds_b)
    manual = np.vstack([q["audit_gap_rows"] for q in seeds_a]).mean(0)
    check("PRD-4 the reported mean is the mean of the row-wise seed averages",
          abs(agg["mean"] - manual.mean()) < 1e-12)
    check("PRD-4 the evaluation standard error is sd of the row-wise seed averages over sqrt(n_audit)",
          abs(agg["se_evaluation"] - manual.std(ddof=1) / math.sqrt(400)) < 1e-12,
          f"{agg['se_evaluation']:.3e}")
    check("PRD-4 the evaluation standard error is not divided by the number of seeds",
          agg["se_evaluation"] > manual.std(ddof=1) / math.sqrt(400) / 2)
    ident = [{"audit_rows": rowsA, "audit_gap_rows": base} for _ in range(5)]
    agg2 = sf.seed_averaged_paired_difference(ident, seeds_b)
    check("PRD-4 identical seeds give zero seed spread but a positive evaluation standard error",
          agg2["sd_across_seeds"] == 0.0 and agg2["se_evaluation"] > 0,
          f"seed sd {agg2['sd_across_seeds']:.1e}, evaluation se {agg2['se_evaluation']:.3e}")
    check("PRD-4 the two uncertainties are returned as separate fields with a note on what each covers",
          "se_evaluation" in agg and "sd_across_seeds" in agg and "training-data" in agg["note"])
    raised = False
    try:
        sf.seed_averaged_paired_difference(seeds_a, seeds_b[:3])
    except ValueError:
        raised = True
    check("PRD-4 a mismatched number of seeds raises", raised)

    # PRD-4 problem B: the table builder refuses duplicate column names
    raised = False
    try:
        sf.report([{"a": 1, "b": 2}], ["a", "b", "b"], "dup")
    except ValueError:
        raised = True
    check("PRD-4 the table builder rejects a repeated column name", raised)
    raised = False
    try:
        sf.report([{"a": 1}], ["a", "missing"], "gap")
    except ValueError:
        raised = True
    check("PRD-4 the table builder rejects a column absent from the rows", raised)
    r = pbn.refit_critics(data, S, fitted, L["fit"], L["val"], L["audit"], st, 11)
    check("R2 refit_critics returns one loss per evaluation row for each critic",
          len(r["audit_rows"]) == len(L["audit"]) and len(r["audit_loss_base"]) == len(L["audit"])
          and len(r["audit_gap_rows"]) == len(L["audit"]))
    check("R2 the row-level gaps average to the reported gap",
          abs(float(np.mean(r["audit_gap_rows"])) - r["refit_gap_audit"]) < 1e-9)

    # ---- R4: the controlled design nests the training rows and fixes everything else
    pool = np.random.default_rng(3).permutation(48000)
    roles = sf.assign_roles(pool, 77)
    small = roles["fit_pool"][:int(round(len(roles["fit_pool"]) * 3000 / 48000))]
    check("R4 the small setting's training rows are a subset of the large setting's",
          set(small.tolist()) <= set(roles["fit_pool"].tolist()) and len(small) > 0,
          f"{len(small)} of {len(roles['fit_pool'])}")
    again = sf.assign_roles(pool, 77)
    check("R4 validation and audit rows do not depend on the training size",
          sf.split_id(roles["val"]) == sf.split_id(again["val"])
          and sf.split_id(roles["audit"]) == sf.split_id(again["audit"]))
    body = open(sf.__file__).read().split("def learn_representation")[1].split("\ndef ")[0]
    check("R4 the optimizer seed no longer contains the sample size",
          "len(idx_rep)" not in body and "(STAGE_SEED, opt_seed)" in body)

    # ---- R5: the two training modes differ in the documented way
    saved_mode = pbn.TRAINING_MODE
    pbn.TRAINING_MODE = "manuscript"
    check("R5 in manuscript mode a small positive gap loses to every non-positive gap",
          pbn.checkpoint_improves(-0.01, 0.001) and not pbn.checkpoint_improves(0.001, -0.01))
    check("R5 in manuscript mode the objective is the clipped gap",
          pbn.objective(-0.3) == 0.0 and pbn.objective(0.02) == 0.02)
    pbn.TRAINING_MODE = "variant"
    check("R5 in variant mode the ranking is distance from zero",
          pbn.checkpoint_improves(0.001, -0.01))
    pbn.TRAINING_MODE = saved_mode
    sh2 = pbn.critic_refit_risk_difference(data, S, fitted, L["fit"], L["val"], L["audit"], st, 5, restarts=2)
    check("R5 the refit risk difference reports both critics on all three splits",
          all(math.isfinite(sh2[f"{c}_excess_{n}"]) for c in ("R0", "R1") for n in ("fit", "val", "audit")),
          f"base audit {sh2['R0_excess_audit']:+.5f}, augmented audit {sh2['R1_excess_audit']:+.5f}")

    # ---- R6: checkpoints are run scoped, hashed, and never overwritten
    import uuid
    rid = "t" + uuid.uuid4().hex[:8]
    rec = sf.save_checkpoint("r6_test", fitted, st, {}, rid, ("r6", 0))
    check("R6 the checkpoint record carries a file digest and an initial-weight hash",
          len(rec["sha256"]) == 64 and len(rec["init_hash"]) == 16)
    check("R6 the digest verifies on load", sf.load_checkpoint(rec) is not None)
    raised = False
    try:
        sf.save_checkpoint("r6_test", fitted, st, {}, rid, ("r6", 0))
    except FileExistsError:
        raised = True
    check("R6 writing the same run id and tag twice raises instead of overwriting", raised)
    rec2 = sf.save_checkpoint("r6_test", fitted, st, {}, rid + "b", ("r6", 0))
    check("R6 a different run id gets a different path", rec2["path"] != rec["path"])
    bad = dict(rec); bad["sha256"] = "0" * 64
    raised = False
    try:
        sf.load_checkpoint(bad)
    except ValueError:
        raised = True
    check("R6 a mismatched digest raises on load", raised)

    # ---- PRD-3: the coefficient representation has exactly one trainable parameter and reproduces the family
    import coefficient_probe as cp
    rest3 = [b for b in range(dgp.N_BLOCKS) if b not in set(S)]
    specs3 = pbn.block_specs(data, rest3)
    enc_c = sf.CoefficientRepresentation(specs3, data["X"].shape[1], r_init=r_star, st=st)
    n_learn = sum(q.numel() for q in enc_c.parameters() if q.requires_grad)
    check("PRD-3 the coefficient encoder has exactly one trainable parameter", n_learn == 1, f"{n_learn}")
    with torch.no_grad():
        z_c = enc_c(pbn.make_parts(data, rest3, np.arange(n), st),
                    torch.as_tensor(st["X"](data["X"]))).numpy()
    check("PRD-3 it reproduces the reference family Z_r to float32 precision",
          float(np.abs(z_c - sf.f_representation(data, S, r_star)).max()) < 1e-5,
          f"max |diff| = {np.abs(z_c - sf.f_representation(data, S, r_star)).max():.1e}")
    with torch.no_grad():
        enc_c.r.fill_(0.7)
        z_d = enc_c(pbn.make_parts(data, rest3, np.arange(n), st),
                    torch.as_tensor(st["X"](data["X"]))).numpy()
    check("PRD-3 changing r moves only the third coordinate",
          float(np.abs(z_d[:, :2] - z_c[:, :2]).max()) < 1e-5 and float(np.abs(z_d[:, 2] - z_c[:, 2]).max()) > 1e-3)
    check("PRD-3 the grid and the starting value are fixed in the module, not chosen from a result",
          cp.R_INIT == 0.0 and cp.GRID_N == 21 and (cp.GRID_LO, cp.GRID_HI) == (-0.5, 1.5))
    check("PRD-3 the fixed starting value is not one of the exact roots",
          all(abs(cp.R_INIT - rt) > 1e-3 for rt in (r_star, (3 + math.sqrt(9 - 1)) / 2)))
    src = open(cp.__file__).read()
    learner = src.split("def learn_coefficient")[1].split("\ndef ")[0]
    check("PRD-3 the learner never sees the true probabilities or the exact roots",
          "true_conditional" not in learner and "f_balance_coefficient" not in learner
          and "roots" not in learner)

    # ---- PRD-5: every injected provenance fault must be refused, by the shared checker
    import copy as _copy
    import provenance as prov
    clean = {"provenance": {"run_id": "x", "code_hash": "abc", "script": "staged_dgp_f.py", "args": {},
                            "training_mode": "variant", "seeds": {}, "generated": "now", "complete": True},
             "stage4": {"cells": [{"data_seed": 1, "n_rep": 10, "opt_seed": 0, "complete": True,
                                   "checkpoint": {"path": "provenance.py",
                                                  "sha256": prov.file_sha256("provenance.py")}}]}}
    ckey = lambda c: (c["data_seed"], c["n_rep"], c["opt_seed"])
    cat = lambda d: d["stage4"]["cells"]
    check("PRD-5 a clean result file passes the shared checker",
          prov.verify(clean, "f", cell_key=ckey, cells_at=cat) == [])
    faults = [
        ("an unfinished run", lambda d: d["provenance"].update(complete=False), {}),
        ("an incomplete cell", lambda d: d["stage4"]["cells"][0].update(complete=False), {}),
        ("a duplicate cell", lambda d: d["stage4"]["cells"].append(_copy.deepcopy(d["stage4"]["cells"][0])), {}),
        ("a missing checkpoint", lambda d: d["stage4"]["cells"][0]["checkpoint"].update(path="nope.pt"), {}),
        ("a changed checksum", lambda d: d["stage4"]["cells"][0]["checkpoint"].update(sha256="0" * 64), {}),
        ("a checkpoint with no digest", lambda d: d["stage4"]["cells"][0].update(checkpoint="legacy.pt"), {}),
        ("a missing provenance field", lambda d: d["provenance"].pop("training_mode"), {}),
        ("the wrong training mode", lambda d: None, {"expect_mode": "manuscript"}),
        ("an absent expected cell", lambda d: None, {"expect_cells": {(1, 10, 0), (1, 10, 1)}}),
    ]
    for label, mutate, kw in faults:
        d = _copy.deepcopy(clean)
        mutate(d)
        out = prov.verify(d, "f", cell_key=ckey, cells_at=cat, **kw)
        check(f"PRD-5 the checker refuses {label}", bool(out), out[0] if out else "accepted")

    # ---- P0 hardening: exercise the actual verdict CLI, not only its helper functions.
    # The three artifacts form one protocol: both coefficient modes and the manuscript stage-4 run.
    def _cli(cmd):
        return subprocess.run([sys.executable, "-B", "verdicts.py", *cmd], cwd=Path(__file__).parent,
                              text=True, capture_output=True, timeout=30)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sources = {"coefficient_probe.py": Path(cp.__file__),
                   "staged_dgp_f.py": Path(sf.__file__),
                   "probe_brier_neural.py": Path(pbn.__file__),
                   "probe_dgp_ae.py": Path(dgp.__file__),
                   "provenance.py": Path(prov.__file__),
                   "frozen_criteria.json": Path(__file__).with_name("frozen_criteria.json")}

        def _header(mode, stem, stage4=False):
            selected_sources = ({k: v for k, v in sources.items() if k != "coefficient_probe.py"}
                                if stage4 else sources)
            manifest = prov.snapshot_run_sources(td / f"{stem}.artifacts", selected_sources)
            h = {"run_id": stem, "code_hash": manifest["bundle_sha256"][:16],
                 "script": "coefficient_probe.py",
                 "script_sha256": prov.file_sha256(cp.__file__), "args": {"data_seeds": 1, "opt_seeds": 1},
                 "training_mode": mode,
                 "seeds": {"data": list(range(20260912, 20260917)), "optimizer": list(range(5))},
                 "generated": "now", "complete": True,
                 "protocol_id": "dgp-f-r10-p0-v1",
                 "source_manifest": manifest,
                 "criteria_sha256": prov.file_sha256(Path(__file__).with_name("frozen_criteria.json"))}
            return h

        def _probe(mode, stem):
            runs = []
            for ds in range(20260912, 20260917):
                picks = []
                for os_ in range(5):
                    ck = td / f"{stem}-{ds}-{os_}.pt"
                    meta = {"protocol_id": "dgp-f-r10-p0-v1", "run_id": stem,
                            "cell_key": [ds, os_]}
                    torch.save({"checkpoint_provenance": meta}, ck)
                    picks.append({"data_seed": ds, "opt_seed": os_, "r_hat": 0.1,
                                  "representation_bias": 0.04,
                                  "common_evaluation": {"select_objective": 0.02,
                                                        "audit_objective": 0.03,
                                                        "select_regret": 0.01, "audit_regret": 0.02},
                                  "checkpoint": {"path": str(ck), "sha256": prov.file_sha256(ck), **meta},
                                  "complete": True})
                good = {"selected_success": True}
                runs.append({"data_seed": ds, "picks": picks,
                             "grid_minimisers": {"population_D2": good,
                                                 "finite_sample_oracle_gap_select": good,
                                                 "common_gap_select": good},
                             "diagnosis": {"population": True, "true_probability": True,
                                           "fitted_common": True, "optimizer_regret": True}})
            h = _header(mode, stem)
            h["args"] = {"n": 24000, "data_seeds": 5, "opt_seeds": 5,
                         "training_mode": mode}
            return {"provenance": h, "runs": runs,
                    "diagnosis_complete": True}

        def _stage4(stem="stage4"):
            h = _header("manuscript", stem, stage4=True)
            h["script"] = "staged_dgp_f.py"
            h["script_sha256"] = prov.file_sha256(sf.__file__)
            h["args"] = {"stage4_rep": [3000, 12000, 48000], "stage4_est": 32000,
                         "data_seeds": 3, "opt_seeds": 3, "fixed_preprocessing": True,
                         "stages": "4", "training_mode": "manuscript"}
            h["seeds"] = {"data": [20261211, 20261212, 20261213], "optimizer": [0, 1, 2]}
            cells = []
            for ds, err in ((20261211, 0.03), (20261212, 0.04), (20261213, 0.05)):
                for n_rep in (3000, 12000, 48000):
                    for os_ in range(3):
                        path = td / f"{stem}-{ds}-{n_rep}-{os_}.pt"
                        meta = {"protocol_id": "dgp-f-r10-p0-v1", "run_id": stem,
                                "cell_key": [ds, n_rep, os_]}
                        torch.save({"checkpoint_provenance": meta}, path)
                        cells.append({"data_seed": ds, "n_rep": n_rep, "opt_seed": os_,
                                      "err_representation": err, "complete": True,
                                      "split_ids": {"fit": f"fit-{ds}-{n_rep}",
                                                    "val": f"val-{ds}", "audit": f"audit-{ds}"},
                                      "estimation_split_id": f"est-{ds}",
                                      "checkpoint": {"path": str(path), "sha256": prov.file_sha256(path),
                                                     **meta}})
            by_seed = {str(ds): {"identical_initial_weights": True, "val_rows_fixed": True,
                                 "audit_rows_fixed": True, "estimation_rows_fixed": True}
                       for ds in (20261211, 20261212, 20261213)}
            return {"provenance": h, "stage4": {"sizes": [3000, 12000, 48000],
                                                   "data_seeds": 3, "opt_seeds": 3,
                                                   "controls": {"identical_initial_weights": True,
                                                                "val_rows_fixed": True,
                                                                "audit_rows_fixed": True,
                                                                "estimation_rows_fixed": True},
                                                   "controls_by_data_seed": by_seed,
                                                   "cells": cells}}

        pm, pv, s4 = _probe("manuscript", "probe-m"), _probe("variant", "probe-v"), _stage4()
        files = []
        for name, payload in (("pm.json", pm), ("pv.json", pv), ("s4.json", s4)):
            f = td / name; f.write_text(json.dumps(payload)); files.append(f)
        argv = ["--probe-manuscript", str(files[0]), "--probe-variant", str(files[1]),
                "--stage4-manuscript", str(files[2])]
        verdict_out = td / "verdict.json"
        good = _cli(argv + ["--out", str(verdict_out)])
        check("P0 CLI accepts only a complete manuscript/variant/stage4 protocol trio",
              good.returncode == 0 and "diagnostic_computation" in good.stdout and "PASS" in good.stdout,
              good.stdout + good.stderr)
        saved_verdict = json.loads(verdict_out.read_text())
        check("P0 verdict output binds its three inputs, evaluator source, criteria and generation time",
              saved_verdict["authoritative"] is True
              and all(x["sha256"] for x in saved_verdict["evaluation_provenance"]["inputs"].values())
              and saved_verdict["evaluation_provenance"]["evaluator"]["sha256"]
              and saved_verdict["evaluation_provenance"]["criteria"]["sha256"]
              and saved_verdict["evaluation_provenance"]["generated"])
        check("P0 verdict emits exactly four labelled diagnostic hypotheses",
              len(saved_verdict["hypotheses"]) == 4
              and saved_verdict["hypotheses"]["optimizer_misses_own_finite_candidate_minimum"]["status"]
                  == "NOT_SUPPORTED"
              and saved_verdict["hypotheses"]["critic_approximation_vs_sampling_attribution"]["status"]
                  == "UNRESOLVED")
        absent = _cli([])
        check("P0 CLI with no evidence is NOT_EVALUATED, never PASS",
              absent.returncode != 0 and "NOT_EVALUATED" in absent.stdout, absent.stdout + absent.stderr)

        def _tamper_source_manifest():
            Path(pm["provenance"]["source_manifest"]["path"]).write_text("{}")

        def _remove_source_role_coordinated():
            rec = pm["provenance"]["source_manifest"]
            path = Path(rec["path"])
            manifest = json.loads(path.read_text())
            manifest["sources"].pop("provenance.py")
            manifest["bundle_sha256"] = prov.source_bundle_digest(manifest["sources"])
            path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
            rec["sha256"] = prov.file_sha256(path)
            rec["bundle_sha256"] = manifest["bundle_sha256"]
            pm["provenance"]["code_hash"] = manifest["bundle_sha256"][:16]

        def _swap_probe_checkpoints():
            a, b = pm["runs"][0]["picks"][:2]
            for key in ("path", "sha256"):
                a["checkpoint"][key], b["checkpoint"][key] = b["checkpoint"][key], a["checkpoint"][key]

        def _swap_stage4_checkpoints():
            a, b = s4["stage4"]["cells"][:2]
            for key in ("path", "sha256"):
                a["checkpoint"][key], b["checkpoint"][key] = b["checkpoint"][key], a["checkpoint"][key]

        def _drift_second_data_seed_only():
            target = next(c for c in s4["stage4"]["cells"]
                          if c["data_seed"] == 20261212 and c["n_rep"] == 12000)
            target["split_ids"]["val"] = "second-seed-drift"

        def _shrink_probe_protocol():
            pm["provenance"]["seeds"] = {"data": [20260912], "optimizer": [0]}
            pm["provenance"]["args"].update(data_seeds=1, opt_seeds=1)
            pm["runs"] = [pm["runs"][0]]
            pm["runs"][0]["picks"] = [pm["runs"][0]["picks"][0]]

        def _shrink_stage4_protocol():
            s4["provenance"]["seeds"] = {"data": [20261211], "optimizer": [0]}
            s4["provenance"]["args"].update(stage4_rep=[48000], data_seeds=1, opt_seeds=1)
            s4["stage4"].update(sizes=[48000], data_seeds=1, opt_seeds=1)
            s4["stage4"]["cells"] = [next(c for c in s4["stage4"]["cells"]
                                                   if c["data_seed"] == 20261211
                                                   and c["n_rep"] == 48000 and c["opt_seed"] == 0)]

        corruptions = [
            ("wrong mode", lambda: pm["provenance"].update(training_mode="variant")),
            ("missing cell completeness", lambda: pv["runs"][0]["picks"][0].pop("complete")),
            ("criteria digest mismatch", lambda: s4["provenance"].update(criteria_sha256="0" * 64)),
            ("nested checkpoint tamper", lambda: Path(
                pm["runs"][0]["picks"][0]["checkpoint"]["path"]).write_bytes(b"changed")),
            ("two valid coefficient checkpoints swapped", _swap_probe_checkpoints),
            ("two valid stage4 checkpoints swapped", _swap_stage4_checkpoints),
            ("empty coefficient cells", lambda: pv["runs"][0]["picks"].clear()),
            ("missing stage4 cell", lambda: s4["stage4"]["cells"].pop()),
            ("duplicate stage4 cell", lambda: s4["stage4"]["cells"].append(
                _copy.deepcopy(s4["stage4"]["cells"][0]))),
            ("unexpected stage4 cell", lambda: s4["stage4"]["cells"].append(dict(
                _copy.deepcopy(s4["stage4"]["cells"][0]), opt_seed=9))),
            ("stage4 cell with no checkpoint", lambda: s4["stage4"]["cells"][0].pop("checkpoint")),
            ("false stage4 control", lambda: s4["stage4"]["controls"].update(val_rows_fixed=False)),
            ("second data seed split drift", _drift_second_data_seed_only),
            ("wrong protocol id", lambda: pm["provenance"].update(protocol_id="different")),
            ("wrong coefficient cell manifest", lambda: pm["provenance"]["seeds"].update(optimizer=[0])),
            ("wrong stage4 grid", lambda: s4["stage4"].update(sizes=[48000])),
            ("internally consistent but shrunken coefficient protocol", _shrink_probe_protocol),
            ("internally consistent but shrunken stage4 protocol", _shrink_stage4_protocol),
            ("tampered source manifest", _tamper_source_manifest),
            ("coordinated source-role removal", _remove_source_role_coordinated),
        ]
        for label, mutate in corruptions:
            slug = label.replace(" ", "-")
            pm, pv, s4 = _probe("manuscript", "probe-m2-" + slug), \
                         _probe("variant", "probe-v2-" + slug), _stage4("stage4-" + slug)
            mutate()
            for f, payload in zip(files, (pm, pv, s4)):
                f.write_text(json.dumps(payload))
            bad = _cli(argv)
            check(f"P0 CLI fails closed on {label}",
                  bad.returncode != 0 and "NOT_EVALUATED" in bad.stdout, bad.stdout + bad.stderr)

    tied = [{"r": -0.1, "objective": 0.2}, {"r": 0.1, "objective": 0.2},
            {"r": 0.0, "objective": 0.2}, {"r": 0.2, "objective": 0.3}]
    picked = cp.select_candidate(tied, "objective")
    check("P0 ties preserve the tied set and choose smallest |r|, then r",
          picked["tie_r"] == [-0.1, 0.0, 0.1] and picked["selected_r"] == 0.0)
    check("P0 the common objective is frozen across training modes",
          cp.common_objective(-0.1) == 0.0 and cp.common_objective(0.1) == 0.1)
    candidates = [{"r": 0.0, "select_objective": 0.03, "audit_objective": 0.04},
                  {"r": 0.1, "select_objective": 0.01, "audit_objective": 0.02}]
    cp.attach_finite_candidate_regrets(candidates)
    check("P0 finite-candidate regret uses the same candidate set on select and audit",
          abs(candidates[0]["select_regret"] - 0.02) < 1e-12
          and abs(candidates[0]["audit_regret"] - 0.02) < 1e-12
          and candidates[1]["select_regret"] == 0.0 and candidates[1]["audit_regret"] == 0.0)
    common = pbn.common_critic_evaluate(data, S, fitted, np.arange(1000), np.arange(1000, 1500),
                                        np.arange(1500, 2000), np.arange(2000, 3000), st, seed=991)
    check("P0 common evaluator scores select and untouched audit with independently stopped critics",
          all(k in common for k in ("R0_select", "R1_select", "gap_select", "R0_audit",
                                    "R1_audit", "gap_audit", "q0_stop_risk", "q1_stop_risk")))
    common_source = open(pbn.__file__).read().split("def common_critic_evaluate")[1].split("\ndef ")[0]
    check("P0 common evaluator early-stops q0 and q1 on their own Brier risks",
          "_fit_one(q0.parameters(), loss0" in common_source
          and "_fit_one(q1.parameters(), loss1" in common_source)
    h = __import__("verdicts").hierarchical_stage4(
        [{"data_seed": 1, "n_rep": 9, "opt_seed": 0, "err_representation": 0.01},
         {"data_seed": 1, "n_rep": 9, "opt_seed": 1, "err_representation": 0.19},
         {"data_seed": 2, "n_rep": 9, "opt_seed": 0, "err_representation": 0.09},
         {"data_seed": 2, "n_rep": 9, "opt_seed": 1, "err_representation": 0.11}], 9)
    check("P0 stage4 aggregates optimizer seeds inside each independent data seed",
          h["per_data_seed"] == [0.1, 0.1] and h["median"] == 0.1 and h["p90"] == 0.1, str(h))
    check("PRD-5 the checkpoint writer uses exclusive creation, not an existence check then a write",
          "O_EXCL" in open(sf.__file__).read())
    check("PRD-5 initial weights are compared within the same restart index, not across selected restarts",
          "c[\"restart\"]" in open(sf.__file__).read().split("init_ok")[0].split("groups = {}")[-1])

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + "; ".join(FAILURES))
        return 1
    print("all staged-study tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
