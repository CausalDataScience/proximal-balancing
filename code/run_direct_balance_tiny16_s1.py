"""Tiny16 direct image balancing on Shapes3D S1: Stages 0 to C, fail-closed, write-once.

    python3 -B run_direct_balance_tiny16_s1.py

  0  the mechanical tests (test_direct_balance_tiny16.py), recorded
  A1 does 16 x 16 keep the orientation?  a diagnostic reader on separate bank splits, label used here only
  A2 does the tiny critic device tell the GOOD control from the BAD one?
  B  the CNN, from random, forty updates, three-fold cross-fitted gap; checkpoint chosen by fresh critics on
     `select`; independent Tiny16 audit on `check`; the effect, after the choice is made
  C  the same representation frozen, audited by a fresh critic that reads the ORIGINAL 64 x 64 held-out image

Every gate is judged before the next stage runs.  Nothing is retuned after a look at a result.  The whole run
is capped at sixty minutes of wall clock and uses no GPU hour on DeltaAI.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import direct_balance_data as db
import direct_balance_eval as de
import direct_balance_net as dn
import direct_balance_run as dr
import direct_balance_tiny16 as t16
import family_v2_dgp as g
import family_v2_images as im
import family_v2_probe as pr
import run_family_v2 as rf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "direct_balance" / "tiny16_s1"
PRD_TEXT = "PRD: Tiny16 Direct Image Balancing Feasibility (review message of 2026-09-21 night; not a file)"
BENCH, SPLIT_NAME, SPLIT, SEED = "shapes3d", "S1", (0,), 96_900_800
THRESHOLD = 1.5e-3
WALL_CAP_S = 3_600
A1_GATE = {"r2_min": 0.90, "within_one_min": 0.95, "two_or_more_max": 0.01}
A1_TRAIN_IMAGES = 60_000
EFFECT_TOL = 0.15
SOURCES = ["direct_balance_data.py", "direct_balance_net.py", "direct_balance_eval.py", "direct_balance_tiny16.py",
           "run_direct_balance_tiny16_s1.py", "test_direct_balance_tiny16.py", "family_v2_dgp.py",
           "family_v2_probe.py", "family_v2_images.py", "probe_structured_scm.py"]
Z95 = 1.6448536269514722


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_arr(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_state(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode()); h.update(np.ascontiguousarray(state[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def write_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists; it is write-once")
    path.write_text(json.dumps(rf._jsonable(payload), indent=1, allow_nan=False) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "result_v1.json").exists():
        raise SystemExit("result_v1.json exists: the attempt was run once and is not rerun")
    t_wall = time.time()
    verdict = {"stage_reached": "0"}

    def finish(status: str, code: int) -> int:
        verdict["status"] = status
        verdict["seconds_total"] = round(time.time() - t_wall, 1)
        write_once(OUT / "result_v1.json", verdict)
        print(f"\n{status}\nwritten {OUT / 'result_v1.json'} | {verdict['seconds_total']}s", flush=True)
        return code

    def over() -> bool:
        return time.time() - t_wall > WALL_CAP_S

    # ------------------------------------------------------------------ data and protocol
    dev = dn.device()
    images, groups = db.load_bank(BENCH)
    data_full = g.generate(g.LEVELS[db.BASE_LEVEL], db.N, SEED)
    src = db.draw_images(data_full, groups, db.BENCHMARKS[BENCH], SEED)
    rows = db.split_rows(db.N, SEED)
    x, a, y = data_full["X"], data_full["A"], data_full["Y"]
    held, kept = db.image_columns(SPLIT)
    tiny = t16.TinyBatches(images, src, x, a, dev, standardise_rows=rows["represent"])
    torch.manual_seed(dr.purpose_seed(SEED, 0, 0))
    init_probe = dn.Representation(kept, x.shape[1], spec=t16.SPEC16)
    protocol = {"protocol_id": "direct-balance-tiny16-s1-v1", "prd": PRD_TEXT, "frozen_at_utc": rf.now(),
                "source_sha256": {f: sha(HERE / f) for f in SOURCES}, "benchmark": BENCH, "seed": SEED,
                "split": {"name": SPLIT_NAME, "held_images": held.tolist(), "kept_images": kept.tolist()},
                "rows_sha256": {k: sha_arr(v) for k, v in rows.items()}, "row_sizes": {k: int(len(v)) for k, v in rows.items()},
                "rows_disjoint": bool(len(np.unique(np.concatenate(list(rows.values())))) == db.N),
                "bank": {"split_rows_sha256": sha(im.BANK_DIR / "split_rows.npz"), "manifest_sha256": sha(im.BANK_DIR / "manifest.json")},
                "data_fingerprint_sha256": sha_arr(np.concatenate([a, y, src.ravel().astype(np.float64)])),
                "resize": tiny.report(), "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                                                      "numpy": np.__version__}, "device": str(dev),
                "network": {"tower": [type(m).__name__ for m in init_probe.towers[0].body], "feature_dim": dn.FEATURE_DIM,
                            "head_hidden": 64, "representation_dim": dn.REPRESENTATION_DIM,
                            "parameters": int(sum(p.numel() for p in init_probe.parameters()))},
                "initial_weights_sha256": sha_state(init_probe.state_dict()),
                "optimizer": {"name": "Adam", "lr": dn.TRAIN["lr_cnn"]}, "checkpoints": list(t16.CHECKPOINTS),
                "critic_budget": {"initial_steps": dn.TRAIN["critic_initial"], "steps_per_iteration": dn.TRAIN["critic_per_iter"],
                                  "folds": t16.FOLDS, "fresh_critic_check": de.CHECK},
                "threshold": THRESHOLD, "a1_gate": A1_GATE, "effect_tolerance": EFFECT_TOL, "wall_cap_s": WALL_CAP_S}
    write_once(OUT / "protocol_v1.json", protocol)
    del init_probe
    print(f"protocol frozen: {OUT / 'protocol_v1.json'} | device {dev} | resize {tiny.report()['unique_images']} images", flush=True)

    # ------------------------------------------------------------------ Stage 0
    tests = subprocess.run([sys.executable, "-B", str(HERE / "test_direct_balance_tiny16.py")], capture_output=True, text=True, cwd=HERE)
    s0 = {"returncode": tests.returncode, "passed": tests.returncode == 0 and "\nOK" in tests.stderr[-200:],
          "tail": tests.stderr[-400:], "utc": rf.now()}
    write_once(OUT / "stage0_tests.json", s0)
    print(f"stage 0: {'passed' if s0['passed'] else 'FAILED'}", flush=True)
    if not s0["passed"]:
        return finish("FAIL at stage 0: a mechanical check failed", 1)

    # ------------------------------------------------------------------ Stage A1: information
    verdict["stage_reached"] = "A1"
    splits = np.load(im.BANK_DIR / "split_rows.npz")
    rng_a1 = np.random.default_rng(SEED + 5_001)
    tr = np.sort(rng_a1.choice(splits["reader_train"], A1_TRAIN_IMAGES, replace=False))
    te = splits["reader_selection"]
    lv_tr = im.factor_index(tr)[:, im.ORIENTATION]
    lv_te = im.factor_index(te)[:, im.ORIENTATION]
    a1 = t16.information_check(images, tr, te, lv_tr, lv_te, dev, SEED + 5_002)
    a1["gate"] = {"r2": a1["r2"] >= A1_GATE["r2_min"], "within_one": a1["within_one_level"] >= A1_GATE["within_one_min"],
                  "two_or_more": a1["two_or_more_off"] <= A1_GATE["two_or_more_max"]}
    a1["passed"] = all(a1["gate"].values())
    a1["note"] = "orientation labels used here and nowhere in the main learner; bank splits disjoint from the causal bank"
    print(f"stage A1: R2 {a1['r2']:.4f} | within one level {a1['within_one_level']:.4f} | two or more off "
          f"{a1['two_or_more_off']:.4f} | {a1['seconds']}s -> {'passed' if a1['passed'] else 'FAILED'}", flush=True)

    # ------------------------------------------------------------------ Stage A2: the tiny device on fixed controls
    verdict["stage_reached"] = "A2"
    rr = rows["represent"]
    levels = np.column_stack([db.recorded(data_full, db.BENCHMARKS[BENCH])[k] for k in db.BLOCK_KEYS]).astype(np.float64)
    lv_mu, lv_sd = levels[rr].mean(0), levels[rr].std(0) + 1e-12
    lv = (levels - lv_mu) / lv_sd
    x_std = tiny.x.cpu().numpy().astype(np.float64)
    learned = pr.learn_representation(x_std[rr], lv[rr][:, kept], lv[rr][:, held], a[rr], 2, dr.purpose_seed(SEED, 0, 7))
    controls = {"good": np.column_stack([x_std, lv[:, kept] @ learned["b"].T]), "bad": np.column_stack([x_std, np.zeros((db.N, 2))])}
    a2 = {"good_source": {"learner": "family_v2_probe.learn_representation on exact levels of the kept images, fitted on represent rows",
                          "b": learned["b"].tolist(), "gap_fit": learned["gap_fit"], "level_mu": lv_mu.tolist(), "level_sd": lv_sd.tolist()}}
    for name, z_np in controls.items():
        z_t = torch.as_tensor(z_np, dtype=torch.float32, device=dev)
        chk = de.check_from_z(z_t[torch.as_tensor(rows["check"], device=dev)], tiny, rows["check"], SPLIT,
                              dr.purpose_seed(SEED, 0, 1), t16.SPEC16)
        chk["lower"] = chk["gap"] - (chk["upper"] - max(chk["gap"], 0.0))          # gap - z * se, same z
        a2[name] = chk
        print(f"stage A2 {name.upper()}: gap {chk['gap']:+.5f} se {chk['se']:.5f} upper {chk['upper']:.5f} lower {chk['lower']:+.5f}", flush=True)
    a2["gate"] = {"good_upper_at_most_t": a2["good"]["upper"] <= THRESHOLD, "bad_lower_above_t": a2["bad"]["lower"] > THRESHOLD}
    a2["passed"] = all(a2["gate"].values())
    write_once(OUT / "stageA_controls.json", {"A1": a1, "A2": a2, "utc": rf.now()})
    print(f"stage A2 gate: {a2['gate']}", flush=True)
    if not a1["passed"]:
        return finish("FAIL at A1: 16 x 16 does not keep the orientation to the pre-set standard", 1)
    if not a2["passed"]:
        return finish("FAIL at A2: the tiny critic device cannot tell the GOOD control from the BAD one", 1)
    if over():
        return finish("TIMEOUT before B", 1)

    # ------------------------------------------------------------------ Stage B
    verdict["stage_reached"] = "B"
    remaining = WALL_CAP_S - (time.time() - t_wall) - 600                        # leave ten minutes for C
    try:
        fit = t16.train_split_kfold(tiny, rows, SPLIT, dr.purpose_seed(SEED, 0, 0), x.shape[1], max_seconds=remaining,
                                    log=lambda m: print(m, flush=True))
    except TimeoutError as err:
        write_once(OUT / "stageB_training.json", {"status": "TIMEOUT", "reason": str(err)})
        return finish("TIMEOUT at B", 1)
    states = fit["checkpoint_states"]
    if sha_state(fit["initial_state"]) != protocol["initial_weights_sha256"]:
        write_once(OUT / "stageB_training.json", {"status": "INVALID", "reason": "initial weights differ from the protocol"})
        return finish("INVALID at B: initial weights differ from the frozen protocol", 1)
    sel = de.select_with_fresh_critics(states, tiny, rows, SPLIT, x.shape[1], dr.purpose_seed(SEED, 0, 5), t16.SPEC16)
    chosen = sel["chosen"]
    print(f"stage B: trained {fit['seconds']}s | steps taken {sum(h['stepped'] for h in fit['history'])}/{len(fit['history'])} | "
          f"fresh-critic selection on select -> checkpoint {chosen} | gaps "
          f"{{{', '.join(f'{k}: {v['gap']:+.4f}' for k, v in sorted(sel['table'].items()))}}}", flush=True)
    check_seed = dr.purpose_seed(SEED, 0, 1)
    audit, d_rows, est = {}, {}, {}
    u = data_full["U_eval_only"]
    for label, k in (("initial", 0), ("chosen", chosen)):
        rep_k = dn.load_representation(states[k], kept, x.shape[1], dev, t16.SPEC16)
        chk = de.check(rep_k, tiny, rows["check"], SPLIT, check_seed, t16.SPEC16, return_rows=True)
        d_rows[label] = chk.pop("rows_d")
        audit[label] = {"checkpoint": k, **chk}
        e = de.estimate(rep_k, tiny, a, y, rows, SPLIT, dr.purpose_seed(SEED, 0, 2)); e.pop("scores")
        zn_, ze_ = (dn.representation_values(rep_k, tiny, rows[r_], kept, grad=False).cpu().numpy().astype(np.float64)
                    for r_ in ("nuisance", "evaluate"))
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(zn_)), zn_]), u[rows["nuisance"]], rcond=None)[0]
        e["u_r2_from_Z_after_judging"] = float(1 - (u[rows["evaluate"]] - np.column_stack([np.ones(len(ze_)), ze_]) @ coef).var() / u[rows["evaluate"]].var())
        est[label] = e
        print(f"  {label} (ckpt {k}): Tiny16 audit gap {chk['gap']:+.5f} se {chk['se']:.5f} upper {chk['upper']:.5f} | "
              f"[after judging] theta {e['theta']:.3f} (sealed linear {e['theta_sealed_linear']:.3f}), U R2 {e['u_r2_from_Z_after_judging']:.3f}", flush=True)
    diff = d_rows["chosen"] - d_rows["initial"]
    paired = {"mean_change": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(diff))), "n": int(len(diff))}
    paired["one_sided_95_upper"] = paired["mean_change"] + Z95 * paired["se"]
    paired["significant_decrease"] = bool(paired["one_sided_95_upper"] < 0)
    err_init, err_chosen = abs(est["initial"]["theta"] - 1.0), abs(est["chosen"]["theta"] - 1.0)
    finite = all(np.isfinite(v) for v in (audit["chosen"]["gap"], audit["chosen"]["se"], est["chosen"]["theta"])) \
        and audit["chosen"]["finite"] and est["chosen"]["finite"]
    b_gate = {"chosen_after_zero": chosen > 0, "audit_upper_at_most_t": audit["chosen"]["upper"] <= THRESHOLD,
              "paired_decrease_significant": paired["significant_decrease"], "effect_within_tolerance": err_chosen <= EFFECT_TOL,
              "error_below_initial": err_chosen < err_init, "all_finite": finite}
    stage_b = {"training": {"seconds": fit["seconds"], "folds": fit["folds"], "fold_sizes": fit["fold_sizes"],
                            "history": fit["history"], "steps_taken": sum(h["stepped"] for h in fit["history"]),
                            "checkpoint_weight_sha256": {k: sha_state(v) for k, v in states.items()},
                            "checkpoint_select_training_critics": fit["checkpoint_select_training_critics"],
                            "checkpoint_fit_risks": fit["checkpoint_fit_risks"]},
               "selection": {"chosen": chosen, "table": {str(k): v for k, v in sel["table"].items()}, "device": "fresh neural critics, 16 x 16 held-out image, select rows"},
               "tiny16_audit": audit, "paired_change_chosen_minus_initial": paired, "effect": est, "gate": b_gate,
               "passed": all(b_gate.values()), "utc": rf.now()}
    write_once(OUT / "stageB_training.json", stage_b)
    torch.save(states, OUT / "checkpoints_v1.pt")
    print(f"stage B paired change: {paired['mean_change']:+.5f} (se {paired['se']:.5f}, one-sided 95% upper {paired['one_sided_95_upper']:+.5f}) | gate {b_gate}", flush=True)
    verdict["B"] = {"gate": b_gate, "passed": stage_b["passed"], "chosen": chosen, "theta": est["chosen"]["theta"]}

    # ------------------------------------------------------------------ Stage C: original-resolution audit
    verdict["stage_reached"] = "C"
    mixed = t16.TinyBatches(images, src, x, a, dev, standardise_rows=rows["represent"], held64=held)
    audit64 = {}
    for label, k in (("initial", 0), ("chosen", chosen)):
        rep_k = dn.load_representation(states[k], kept, x.shape[1], dev, t16.SPEC16)
        chk = de.check(rep_k, mixed, rows["check"], SPLIT, check_seed, dn.SPECS["shapes3d"])
        audit64[label] = {"checkpoint": k, **chk}
        print(f"stage C {label} (ckpt {k}): original-64 audit gap {chk['gap']:+.5f} se {chk['se']:.5f} upper {chk['upper']:.5f}", flush=True)
    c_gate = {"audit64_upper_at_most_t": audit64["chosen"]["upper"] <= THRESHOLD}
    write_once(OUT / "stageC_original64_audit.json", {"audit": audit64, "gate": c_gate, "passed": all(c_gate.values()), "utc": rf.now(),
                                                       "note": "representation from 16 x 16 retained images; only the fresh augmented critic reads 64 x 64; not used to choose"})
    verdict["C"] = {"gate": c_gate, "passed": all(c_gate.values())}
    if stage_b["passed"] and verdict["C"]["passed"]:
        final = "STRONG_FEASIBILITY_PASS"
    elif stage_b["passed"]:
        final = "TINY_PROXY_ONLY_PASS"
    else:
        final = "FAIL"
    verdict["final"] = final
    return finish(f"COMPLETED: {final}", 0 if final != "FAIL" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
