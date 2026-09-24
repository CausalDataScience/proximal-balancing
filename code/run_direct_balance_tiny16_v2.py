"""Tiny16 direct image balancing, version 2: the training critics generalise and the nesting is decided on rows
they did not fit (PRD v2, memo/2026-09-22-tiny16-v2-validated-critics-prd.md).

    python3 -B run_direct_balance_tiny16_v2.py --until A      stages 0 and A only: everything up to the experiment
    python3 -B run_direct_balance_tiny16_v2.py                 continue: stage B (the CNN), then C, then the verdict

Each stage writes its own file and is never rerun: a second invocation picks up after the last file present.
Nothing is retuned after a look at a result; the settings are frozen in protocol_v1.json before stage 0 runs.

The verdict is tiered rather than pass/fail, at the user's request:
    clear     chosen checkpoint after zero, independent audit upper <= t, paired decrease significant (one-sided
              95%), |tau - 1| <= 0.15, error below the untrained network's
    partial   chosen after zero, error below the untrained network's, and either the paired decrease is
              significant or the audit gap fell
    none      anything else
plus `strong` when `clear` also passes the original-resolution audit.
"""
from __future__ import annotations

import argparse
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
OUT = HERE.parent / "results" / "direct_balance" / "tiny16_v2"
PRD = HERE.parent / "memo" / "2026-09-22-tiny16-v2-validated-critics-prd.md"
BENCH, SPLIT_NAME, SPLIT, SEED = "shapes3d", "S1", (0,), 96_900_800
THRESHOLD, EFFECT_TOL = 1.5e-3, 0.15
ITERS, CHECKPOINTS = 100, (0, 10, 20, 40, 70, 100)
CAP_B_S, CAP_TOTAL_S = 1_800, 3_600
A1_GATE = {"r2_min": 0.90, "within_one_min": 0.95, "two_or_more_max": 0.01}
Z95 = 1.6448536269514722
SOURCES = ["direct_balance_data.py", "direct_balance_net.py", "direct_balance_eval.py", "direct_balance_tiny16.py",
           "run_direct_balance_tiny16_v2.py", "test_direct_balance_tiny16.py", "family_v2_dgp.py",
           "family_v2_probe.py", "family_v2_images.py", "probe_structured_scm.py"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_arr(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_state(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode()); h.update(np.ascontiguousarray(state[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def write_once(name: str, payload: dict) -> None:
    path = OUT / name
    if path.exists():
        raise SystemExit(f"{path} exists; it is write-once")
    path.write_text(json.dumps(rf._jsonable(payload), indent=1, allow_nan=False) + "\n")


def have(name: str) -> bool:
    return (OUT / name).exists()


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def main(until: str | None) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if have("result_v1.json"):
        raise SystemExit("result_v1.json exists: the attempt was completed once and is not rerun")
    t_wall = time.time()
    dev = dn.device()
    images, groups = db.load_bank(BENCH)
    data_full = g.generate(g.LEVELS[db.BASE_LEVEL], db.N, SEED)
    src = db.draw_images(data_full, groups, db.BENCHMARKS[BENCH], SEED)
    rows = db.split_rows(db.N, SEED)
    x, a, y = data_full["X"], data_full["A"], data_full["Y"]
    held, kept = db.image_columns(SPLIT)
    tiny = t16.TinyBatches(images, src, x, a, dev, standardise_rows=rows["represent"])

    # ------------------------------------------------------------------ protocol, frozen before anything runs
    if not have("protocol_v1.json"):
        torch.manual_seed(dr.purpose_seed(SEED, 0, 0))
        probe = dn.Representation(kept, x.shape[1], spec=t16.SPEC16)
        write_once("protocol_v1.json", {
            "protocol_id": "direct-balance-tiny16-v2", "prd": {"path": str(PRD.relative_to(HERE.parent)), "sha256": sha(PRD) if PRD.exists() else None},
            "frozen_at_utc": rf.now(), "source_sha256": {f: sha(HERE / f) for f in SOURCES},
            "benchmark": BENCH, "seed": SEED, "split": {"name": SPLIT_NAME, "held_images": held.tolist(), "kept_images": kept.tolist()},
            "rows_sha256": {k: sha_arr(v) for k, v in rows.items()}, "row_sizes": {k: int(len(v)) for k, v in rows.items()},
            "rows_disjoint": bool(len(np.unique(np.concatenate(list(rows.values())))) == db.N),
            "bank": {"split_rows_sha256": sha(im.BANK_DIR / "split_rows.npz"), "manifest_sha256": sha(im.BANK_DIR / "manifest.json")},
            "data_fingerprint_sha256": sha_arr(np.concatenate([a, y, src.ravel().astype(np.float64)])),
            "resize": tiny.report(), "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__},
            "device": str(dev), "network": {"tower": [type(m).__name__ for m in probe.towers[0].body], "feature_dim": dn.FEATURE_DIM,
                                             "representation_dim": dn.REPRESENTATION_DIM, "parameters": int(sum(p.numel() for p in probe.parameters()))},
            "initial_weights_sha256": sha_state(probe.state_dict()),
            "training": {"iters": ITERS, "checkpoints": list(CHECKPOINTS), "lr": dn.TRAIN["lr_cnn"], "folds": t16.FOLDS,
                         "critics": "validated: early-stopped on a held-back fifth of their fit rows; lifted base decided there",
                         "critic_budget": dn.CRITIC_V, "fresh_critic_check": de.CHECK},
            "threshold": THRESHOLD, "effect_tolerance": EFFECT_TOL, "a1_gate": A1_GATE,
            "verdict_tiers": "clear / partial / none, plus strong when clear also passes the 64 x 64 audit",
            "caps_s": {"B": CAP_B_S, "total": CAP_TOTAL_S}})
        del probe
        print(f"protocol frozen: {OUT / 'protocol_v1.json'}", flush=True)
    proto = load("protocol_v1.json")
    moved = [f for f, h in proto["source_sha256"].items() if sha(HERE / f) != h]
    if moved:
        raise SystemExit(f"sources changed since the protocol was frozen: {moved}")

    # ------------------------------------------------------------------ stage 0
    if not have("stage0_tests.json"):
        tests = subprocess.run([sys.executable, "-B", str(HERE / "test_direct_balance_tiny16.py")], capture_output=True, text=True, cwd=HERE)
        s0 = {"returncode": tests.returncode, "passed": tests.returncode == 0 and "\nOK" in tests.stderr[-200:], "tail": tests.stderr[-300:], "utc": rf.now()}
        write_once("stage0_tests.json", s0)
        print(f"stage 0: {'passed' if s0['passed'] else 'FAILED'}", flush=True)
    if not load("stage0_tests.json")["passed"]:
        return finish({"status": "FAIL at stage 0"}, 1)

    # ------------------------------------------------------------------ stage A
    if not have("stageA_controls.json"):
        splits = np.load(im.BANK_DIR / "split_rows.npz")
        tr = np.sort(np.random.default_rng(SEED + 5_001).choice(splits["reader_train"], 60_000, replace=False))
        te = splits["reader_selection"]
        a1 = t16.information_check(images, tr, te, im.factor_index(tr)[:, im.ORIENTATION], im.factor_index(te)[:, im.ORIENTATION], dev, SEED + 5_002)
        a1["gate"] = {"r2": a1["r2"] >= A1_GATE["r2_min"], "within_one": a1["within_one_level"] >= A1_GATE["within_one_min"],
                      "two_or_more": a1["two_or_more_off"] <= A1_GATE["two_or_more_max"]}
        a1["passed"] = all(a1["gate"].values())
        print(f"stage A1: R2 {a1['r2']:.4f} within-one {a1['within_one_level']:.4f} two-or-more {a1['two_or_more_off']:.5f} -> {'passed' if a1['passed'] else 'FAILED'}", flush=True)
        rr = rows["represent"]
        levels = np.column_stack([db.recorded(data_full, db.BENCHMARKS[BENCH])[k] for k in db.BLOCK_KEYS]).astype(np.float64)
        lv = (levels - levels[rr].mean(0)) / (levels[rr].std(0) + 1e-12)
        x_std = tiny.x.cpu().numpy().astype(np.float64)
        learned = pr.learn_representation(x_std[rr], lv[rr][:, kept], lv[rr][:, held], a[rr], 2, dr.purpose_seed(SEED, 0, 7))
        a2 = {"good_source": {"b": learned["b"].tolist(), "gap_fit": learned["gap_fit"], "fit_rows": "represent"}}
        for name, z_np in (("good", np.column_stack([x_std, lv[:, kept] @ learned["b"].T])), ("bad", np.column_stack([x_std, np.zeros((db.N, 2))]))):
            z_t = torch.as_tensor(z_np, dtype=torch.float32, device=dev)
            chk = de.check_from_z(z_t[torch.as_tensor(rows["check"], device=dev)], tiny, rows["check"], SPLIT, dr.purpose_seed(SEED, 0, 1), t16.SPEC16)
            chk["one_sided_95_lower"] = chk["gap"] - Z95 * chk["se"]
            a2[name] = chk
            print(f"stage A2 {name.upper()}: gap {chk['gap']:+.5f} se {chk['se']:.5f} upper {chk['upper']:.5f} one-sided-95 lower {chk['one_sided_95_lower']:+.5f}", flush=True)
        a2["flags"] = {"good_upper_at_most_t": a2["good"]["upper"] <= THRESHOLD, "bad_detected_one_sided_95": a2["bad"]["one_sided_95_lower"] > 0,
                       "device_discriminates": a2["bad"]["gap"] > 0 and a2["bad"]["gap"] > a2["good"]["gap"]}
        a2["hard_stop"] = not a2["flags"]["device_discriminates"]
        write_once("stageA_controls.json", {"A1": a1, "A2": a2, "utc": rf.now()})
        print(f"stage A2 flags: {a2['flags']} | hard stop: {a2['hard_stop']}", flush=True)
    sa = load("stageA_controls.json")
    if not sa["A1"]["passed"]:
        return finish({"status": "FAIL at A1: 16 x 16 lost the orientation"}, 1)
    if sa["A2"]["hard_stop"]:
        return finish({"status": "FAIL at A2: the device does not discriminate at all"}, 1)
    if until == "A":
        print(f"\nstopped before the experiment as asked: stages 0 and A are on disk in {OUT}; stage B has not run", flush=True)
        return 0

    # ------------------------------------------------------------------ stage B: the experiment
    if not have("stageB_training.json"):
        try:
            fit = t16.train_split_kfold(tiny, rows, SPLIT, dr.purpose_seed(SEED, 0, 0), x.shape[1], checkpoints=CHECKPOINTS,
                                        max_seconds=CAP_B_S, validated=True, log=lambda m: print(m, flush=True))
        except TimeoutError as err:
            write_once("stageB_training.json", {"status": "TIMEOUT", "reason": str(err)})
            return finish({"status": "TIMEOUT at B"}, 1)
        if sha_state(fit["initial_state"]) != proto["initial_weights_sha256"]:
            write_once("stageB_training.json", {"status": "INVALID", "reason": "initial weights differ from the protocol"})
            return finish({"status": "INVALID at B"}, 1)
        states = fit["checkpoint_states"]
        torch.save(states, OUT / "checkpoints_v1.pt")
        sel = de.select_with_fresh_critics(states, tiny, rows, SPLIT, x.shape[1], dr.purpose_seed(SEED, 0, 5), t16.SPEC16)
        chosen = sel["chosen"]
        print(f"stage B: {fit['seconds']}s | steps taken {sum(h['stepped'] for h in fit['history'])}/{len(fit['history'])} | lifted "
              f"{sum(any(c['kept_the_lifted_base'] for c in h['critics']) for h in fit['history'])} | fresh selection -> {chosen} | "
              f"gaps {{{', '.join(f'{k}: {v['gap']:+.4f}' for k, v in sorted(sel['table'].items()))}}}", flush=True)
        check_seed = dr.purpose_seed(SEED, 0, 1)
        audit, d_rows, est, u = {}, {}, {}, data_full["U_eval_only"]
        for label, k in (("initial", 0), ("chosen", chosen), ("last", max(states))):
            rep_k = dn.load_representation(states[k], kept, x.shape[1], dev, t16.SPEC16)
            chk = de.check(rep_k, tiny, rows["check"], SPLIT, check_seed, t16.SPEC16, return_rows=True)
            d_rows[label] = chk.pop("rows_d")
            audit[label] = {"checkpoint": k, **chk}
            e = de.estimate(rep_k, tiny, a, y, rows, SPLIT, dr.purpose_seed(SEED, 0, 2)); e.pop("scores")
            zn_, ze_ = (dn.representation_values(rep_k, tiny, rows[r_], kept, grad=False).cpu().numpy().astype(np.float64) for r_ in ("nuisance", "evaluate"))
            coef = np.linalg.lstsq(np.column_stack([np.ones(len(zn_)), zn_]), u[rows["nuisance"]], rcond=None)[0]
            e["u_r2_from_Z_after_judging"] = float(1 - (u[rows["evaluate"]] - np.column_stack([np.ones(len(ze_)), ze_]) @ coef).var() / u[rows["evaluate"]].var())
            est[label] = e
            print(f"  {label:8s} (ckpt {k:3d}): audit gap {chk['gap']:+.5f} se {chk['se']:.5f} upper {chk['upper']:.5f} | "
                  f"[after judging] theta {e['theta']:.3f}, U R2 {e['u_r2_from_Z_after_judging']:.3f}", flush=True)
        diff = d_rows["chosen"] - d_rows["initial"]
        paired = {"mean_change": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(diff))), "n": int(len(diff))}
        paired["one_sided_95_upper"] = paired["mean_change"] + Z95 * paired["se"]
        paired["significant_decrease"] = bool(paired["one_sided_95_upper"] < 0)
        err_i, err_c = abs(est["initial"]["theta"] - 1.0), abs(est["chosen"]["theta"] - 1.0)
        flags = {"chosen_after_zero": chosen > 0, "audit_upper_at_most_t": audit["chosen"]["upper"] <= THRESHOLD,
                 "audit_gap_fell": audit["chosen"]["gap"] < audit["initial"]["gap"], "paired_decrease_significant": paired["significant_decrease"],
                 "effect_within_tolerance": err_c <= EFFECT_TOL, "error_below_initial": err_c < err_i,
                 "all_finite": bool(audit["chosen"]["finite"] and est["chosen"]["finite"])}
        if flags["chosen_after_zero"] and flags["error_below_initial"] and flags["all_finite"]:
            tier = "clear" if (flags["audit_upper_at_most_t"] and flags["paired_decrease_significant"] and flags["effect_within_tolerance"]) \
                else ("partial" if (flags["paired_decrease_significant"] or flags["audit_gap_fell"]) else "none")
        else:
            tier = "none"
        write_once("stageB_training.json", {"training": {k: v for k, v in fit.items() if k not in ("initial_state", "checkpoint_states")},
                                            "checkpoint_weight_sha256": {k: sha_state(v) for k, v in states.items()},
                                            "selection": {"chosen": chosen, "table": {str(k): v for k, v in sel["table"].items()}},
                                            "audit_tiny16": audit, "paired_change_chosen_minus_initial": paired, "effect": est,
                                            "flags": flags, "tier": tier, "utc": rf.now()})
        print(f"stage B: paired change {paired['mean_change']:+.5f} (se {paired['se']:.5f}) | flags {flags} | tier {tier}", flush=True)
    sb = load("stageB_training.json")
    if sb.get("status") in ("TIMEOUT", "INVALID"):
        return finish({"status": sb["status"]}, 1)

    # ------------------------------------------------------------------ stage C: original-resolution audit
    if not have("stageC_original64_audit.json"):
        states = torch.load(OUT / "checkpoints_v1.pt")
        mixed = t16.TinyBatches(images, src, x, a, dev, standardise_rows=rows["represent"], held64=held)
        audit64 = {}
        for label, k in (("initial", 0), ("chosen", sb["selection"]["chosen"])):
            rep_k = dn.load_representation(states[k], kept, x.shape[1], dev, t16.SPEC16)
            chk = de.check(rep_k, mixed, rows["check"], SPLIT, dr.purpose_seed(SEED, 0, 1), dn.SPECS["shapes3d"])
            audit64[label] = {"checkpoint": k, **chk}
            print(f"stage C {label} (ckpt {k}): original-64 audit gap {chk['gap']:+.5f} upper {chk['upper']:.5f}", flush=True)
        write_once("stageC_original64_audit.json", {"audit": audit64, "passed": audit64["chosen"]["upper"] <= THRESHOLD, "utc": rf.now()})
    sc = load("stageC_original64_audit.json")
    tier = sb["tier"]
    final = "strong" if tier == "clear" and sc["passed"] else tier
    return finish({"status": f"COMPLETED: {final}", "tier": tier, "original64_passed": sc["passed"], "chosen": sb["selection"]["chosen"],
                   "theta": sb["effect"]["chosen"]["theta"], "flags": sb["flags"]}, 0 if final != "none" else 1)


def finish(verdict: dict, code: int) -> int:
    verdict["finished_utc"] = rf.now()
    write_once("result_v1.json", verdict)
    print(f"\n{verdict['status']}\nwritten {OUT / 'result_v1.json'}", flush=True)
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", choices=["A"], default=None, help="stop after stage A: everything but the experiment")
    raise SystemExit(main(ap.parse_args().until))
