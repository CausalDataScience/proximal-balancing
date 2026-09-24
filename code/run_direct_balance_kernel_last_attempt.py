"""The last attempt at direct image balancing: one setting, one split, one data set, under an hour on the Mac.

    python3 -B run_direct_balance_kernel_last_attempt.py

Stages, each with its own cap; a stage that overruns ends the attempt, it never widens into the next.
  A  implementation, gradient and measurement checks; the GOOD/BAD contrast through the kernel device (10 min)
  B  the CNN, from the same random start as every rehearsal, at most forty updates                   (30 min)
  C  checkpoint choice by the existing fresh neural critics; the separate check fold; the estimate     (15 min)
  D  the verdict and the record                                                                       (5 min)

The PRD's rules: nothing is retuned after a look at the result; success starts no further run; a mechanical error
found afterwards voids the result rather than earning a rerun.  Exact levels and the latent variable are read by
the generator and by the after-the-fact diagnostics only; the CNN path never sees them.
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
import direct_balance_kernel as dk
import direct_balance_net as dn
import direct_balance_run as dr
import family_v2_dgp as g
import family_v2_probe as pr
import run_family_v2 as rf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "direct_balance" / "kernel_last_attempt"
BENCH, SPLIT_NAME, SPLIT, SEED = "shapes3d", "S1", (0,), 96_900_800
THRESHOLD = 1.5e-3
CAPS_S = {"A": 600, "B": 1_800, "C": 900, "D": 300}
ITERS, CHECKPOINTS, LR = 40, (0, 10, 20, 30, 40), 1e-3
FEATURE_SEED_OFFSET, WIDTH_SEED_OFFSET = 4_001, 4_002       # their own generators; the CNN's draw is untouched
SOURCES = ["direct_balance_data.py", "direct_balance_net.py", "direct_balance_eval.py", "direct_balance_kernel.py",
           "run_direct_balance_kernel_last_attempt.py", "test_direct_balance_kernel.py", "family_v2_dgp.py",
           "family_v2_probe.py", "family_v2_images.py", "probe_structured_scm.py"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_state(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode()); h.update(np.ascontiguousarray(state[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


class Clock:
    def __init__(self) -> None:
        self.t0 = time.time()
        self.stage_start = self.t0

    def start(self, stage: str) -> None:
        self.stage_start = time.time()
        print(f"\n=== stage {stage} ({CAPS_S[stage] // 60} min cap) at +{self.stage_start - self.t0:.0f}s", flush=True)

    def over(self, stage: str) -> bool:
        return time.time() - self.stage_start > CAPS_S[stage]

    def elapsed(self) -> float:
        return round(time.time() - self.stage_start, 1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "result_v1.json"
    if result_path.exists():
        raise SystemExit(f"{result_path} exists: the last attempt was run once and is not rerun")
    clock = Clock()
    rec = {"artifact": "direct_balance_kernel_last_attempt", "benchmark": BENCH, "split": SPLIT_NAME, "seed": SEED,
           "threshold": THRESHOLD, "caps_s": CAPS_S, "started_utc": rf.now(),
           "source_sha256": {f: sha(HERE / f) for f in SOURCES}, "torch": torch.__version__, "numpy": np.__version__,
           "settings": {"iters": ITERS, "checkpoints": list(CHECKPOINTS), "lr": LR, "features": dk.D_FEATURES,
                        "lambda": dk.LAMBDA, "eps": dk.EPS, "sigma_z": dk.SIGMA_Z, "sd_floor": dk.SD_FLOOR,
                        "clip": list(dk.CLIP), "pairs_for_width": dk.PAIRS_FOR_WIDTH}, "status": "running"}

    def save(status: str) -> None:
        rec["status"] = status
        rec["seconds_total"] = round(time.time() - clock.t0, 1)
        result_path.write_text(json.dumps(rf._jsonable(rec), indent=1, allow_nan=False) + "\n")

    # ------------------------------------------------------------------ data, identical to every rehearsal
    dev, spec = dn.device(), dn.SPECS[BENCH]
    images, groups = db.load_bank(BENCH)
    data_full = g.generate(g.LEVELS[db.BASE_LEVEL], db.N, SEED)
    src = db.draw_images(data_full, groups, db.BENCHMARKS[BENCH], SEED)
    rows = db.split_rows(db.N, SEED)
    x, a, y = data_full["X"], data_full["A"], data_full["Y"]
    batches = dn.Batches(images, src, x, a, dev, cache_rows=np.arange(db.N), standardise_rows=rows["represent"])
    held, kept = db.image_columns(SPLIT)
    every = np.concatenate(list(rows.values()))
    rec["rows"] = {k: v.tolist() for k, v in rows.items()}
    rec["row_sets_disjoint"] = bool(len(np.unique(every)) == db.N)
    rec["data_fingerprint_sha256"] = hashlib.sha256(a.tobytes() + y.tobytes() + src.tobytes()).hexdigest()
    rec["device"] = str(dev)
    rr = rows["represent"]
    a_rep = torch.as_tensor(a[rr], dtype=torch.float64)

    # ------------------------------------------------------------------ A
    clock.start("A")
    tests = subprocess.run([sys.executable, "-B", str(HERE / "test_direct_balance_kernel.py")], capture_output=True, text=True)
    unit_ok = tests.returncode == 0 and "\nOK" in tests.stderr[-400:]      # unittest ends with "OK" or "OK (skipped=k)"
    rec["A"] = {"unit_tests": {"returncode": tests.returncode, "tail": tests.stderr[-300:]}}
    print(f"  unit tests: {'ok' if unit_ok else 'FAILED'}", flush=True)
    width = dk.pixel_width(batches, rr, held, SEED + WIDTH_SEED_OFFSET)
    print(f"  pixel kernel width: {width['sigma_w']:.4f} on {width['n_pix']} pixels (quantiles {width['distance_quantiles']})", flush=True)
    if not (width["finite"] and width["sigma_w"] > 0):
        rec["A"]["width"] = width; save("STOPPED: pixel width not usable"); return 1
    feats = dk.Features(2 + dn.REPRESENTATION_DIM, width["n_pix"], width["sigma_w"], SEED + FEATURE_SEED_OFFSET, dev)
    pw_rep = feats.pixel_projection(batches, rr, held)
    rec["A"].update({"width": width, "features": feats.record()})

    # GOOD: the sealed numeric learner on the exact recorded levels of the kept images, fitted on represent rows only
    levels = np.column_stack([db.recorded(data_full, db.BENCHMARKS[BENCH])[k] for k in db.BLOCK_KEYS]).astype(np.float64)
    lv_mu, lv_sd = levels[rr].mean(0), levels[rr].std(0) + 1e-12
    lv = (levels - lv_mu) / lv_sd
    x_std = batches.x.cpu().numpy().astype(np.float64)
    learned = pr.learn_representation(x_std[rr], lv[rr][:, kept], lv[rr][:, held], a[rr], 2, dr.purpose_seed(SEED, 0, 7))
    z_good = torch.as_tensor(np.column_stack([x_std, lv[:, kept] @ learned["b"].T]), dtype=torch.float64)
    z_bad = torch.as_tensor(np.column_stack([x_std, np.zeros((db.N, 2))]), dtype=torch.float64)
    rec["A"]["good_candidate"] = {"source": "family_v2_probe.learn_representation on exact levels of the kept images",
                                  "fit_rows": "represent", "b": learned["b"].tolist(), "gap_fit": learned["gap_fit"],
                                  "level_mu": lv_mu.tolist(), "level_sd": lv_sd.tolist(), "x_mu": batches.x_mu.tolist(),
                                  "x_sd": batches.x_sd.tolist()}
    contrast = {}
    rep_idx = torch.as_tensor(rr)
    for name, z_all in (("good", z_good), ("bad", z_bad)):
        mu, sd = dk.h_stats(z_all[rep_idx])
        chk = dk.kernel_check(z_all[rep_idx], pw_rep, a_rep, feats, mu, sd, THRESHOLD)
        chk.pop("rows_d")
        contrast[name] = chk
        print(f"  {name.upper():4s} through the kernel device on the represent fold: gap {chk['gap']:+.5f} se {chk['se']:.5f} "
              f"upper {chk['upper']:.5f} lower {chk['lower']:+.5f} | q0 {chk['risk_base']:.4f} q1 {chk['risk_augmented']:.4f}", flush=True)
    rec["A"]["contrast_represent_fold"] = contrast
    a_ok = {"unit_tests": unit_ok, "good_passes_upper": contrast["good"]["passes_upper"],
            "bad_clearly_imbalanced": contrast["bad"]["clearly_imbalanced"],
            "row_sets_disjoint": rec["row_sets_disjoint"]}
    rec["A"]["gate"] = a_ok
    rec["A"]["seconds"] = clock.elapsed()
    print(f"  A gate: {a_ok}", flush=True)
    if not all(a_ok.values()):
        save("STOPPED at A: the kernel setting cannot tell the good candidate from the bad one" if
             not (a_ok["good_passes_upper"] and a_ok["bad_clearly_imbalanced"]) else "STOPPED at A: mechanical check failed")
        print(f"written {result_path}"); return 1
    if clock.over("A"):
        save("TIMEOUT at A"); return 1

    # ------------------------------------------------------------------ B
    clock.start("B")
    torch.manual_seed(dr.purpose_seed(SEED, 0, 0))                 # the same random start as every rehearsal
    rep = dn.Representation(kept, x.shape[1], spec=spec).to(dev)
    init_sha = sha_state(rep.state_dict())
    opt = torch.optim.Adam(rep.parameters(), lr=LR)
    states, history = {}, []
    with torch.no_grad():
        z0 = dk.representation_all(rep, batches, rr, kept)
        mu0, sd0 = dk.h_stats(z0)
    states[0] = {"state": {k: v.detach().cpu().clone() for k, v in rep.state_dict().items()}, "h_mean": mu0.tolist(), "h_sd": sd0.tolist()}
    timed_out = False
    for it in range(1, ITERS + 1):
        if clock.over("B"):
            timed_out = True
            break
        t1 = time.time()
        step = dk.two_stage_step(rep, batches, rr, kept, pw_rep, a_rep, feats, opt)
        step.update(iteration=it, seconds=round(time.time() - t1, 1))
        history.append(step)
        if it in CHECKPOINTS:
            states[it] = {"state": {k: v.detach().cpu().clone() for k, v in rep.state_dict().items()},
                          "h_mean": step["h_mean"], "h_sd": step["h_sd"]}
        print(f"  iter {it:2d}: signed gap {step['signed_gap']:+.5f} | q0 {step['risk_base']:.4f} q1 {step['risk_augmented']:.4f}"
              f" | lifted {step['lifted_base_used']} | stepped {step['stepped']} | dL/dZ {step['dLdZ_norm']:.2e} grad {step['grad_norm']:.2e}"
              f" first conv {step['first_conv_grad_norm']:.2e} | cond {step['cond_augmented']:.1e} | {step['seconds']}s", flush=True)
    rec["B"] = {"initial_weights_sha256": init_sha, "history": history, "iterations_done": len(history),
                "steps_taken": sum(h["stepped"] for h in history), "lifted_base_used": sum(h["lifted_base_used"] for h in history),
                "checkpoints_saved": sorted(states), "timed_out": timed_out, "seconds": clock.elapsed(),
                "checkpoint_weight_sha256": {k: sha_state(v["state"]) for k, v in states.items()},
                "checkpoint_h_stats": {k: {"mean": v["h_mean"], "sd": v["h_sd"]} for k, v in states.items()}}
    torch.save({k: v["state"] for k, v in states.items()}, OUT / "checkpoints_v1.pt")
    if timed_out:
        save("TIMEOUT at B"); print(f"written {result_path}"); return 1

    # ------------------------------------------------------------------ C
    clock.start("C")
    sel = de.select_with_fresh_critics({k: v["state"] for k, v in states.items()}, batches, rows, SPLIT, x.shape[1],
                                       dr.purpose_seed(SEED, 0, 5), spec)
    chosen = sel["chosen"]
    print(f"  selection by fresh neural critics on select: {chosen} | gaps {{{', '.join(f'{k}: {v['gap']:+.4f}' for k, v in sorted(sel['table'].items()))}}}", flush=True)
    check_seed = dr.purpose_seed(SEED, 0, 1)
    final, d_rows = {}, {}
    for label, k in (("before", 0), ("after", chosen)):
        rep_k = dn.load_representation(states[k]["state"], kept, x.shape[1], dev, spec)
        chk = de.check(rep_k, batches, rows["check"], SPLIT, check_seed, spec, return_rows=True)
        d_rows[label] = chk.pop("rows_d")
        # the kernel device on the same check rows, for the "only the training score improved" case
        z_chk = dk.representation_all(rep_k, batches, rows["check"], kept)
        pw_chk = feats.pixel_projection(batches, rows["check"], held)
        kchk = dk.kernel_check(z_chk, pw_chk, torch.as_tensor(a[rows["check"]], dtype=torch.float64), feats,
                               torch.as_tensor(states[k]["h_mean"], dtype=torch.float64),
                               torch.as_tensor(states[k]["h_sd"], dtype=torch.float64), THRESHOLD)
        kchk.pop("rows_d")
        est = de.estimate(rep_k, batches, a, y, rows, SPLIT, dr.purpose_seed(SEED, 0, 2)); est.pop("scores")
        u = data_full["U_eval_only"]
        zn_, ze_ = (dk.representation_all(rep_k, batches, rows[r_], kept).numpy() for r_ in ("nuisance", "evaluate"))
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(zn_)), zn_]), u[rows["nuisance"]], rcond=None)[0]
        u_r2 = float(1 - (u[rows["evaluate"]] - np.column_stack([np.ones(len(ze_)), ze_]) @ coef).var() / u[rows["evaluate"]].var())
        final[label] = {"checkpoint": k, "neural_check": chk, "kernel_check": kchk,
                        "after_judging": {"estimate": est, "u_r2_from_Z": u_r2}}
        print(f"  {label} (ckpt {k}): neural check gap {chk['gap']:+.5f} upper {chk['upper']:.5f} | kernel check gap {kchk['gap']:+.5f} "
              f"upper {kchk['upper']:.5f} | [after judging] theta {est['theta']:.3f}, U R2 {u_r2:.3f}", flush=True)
    diff = d_rows["after"] - d_rows["before"]
    paired = {"mean_change": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(diff))), "n": int(len(diff))}
    paired["z"] = paired["mean_change"] / paired["se"] if paired["se"] > 0 else 0.0
    rec["C"] = {"selection": {"chosen": chosen, "table": {str(k): v for k, v in sel["table"].items()}},
                "final": final, "paired_change_after_minus_before": paired, "seconds": clock.elapsed()}
    print(f"  paired change (after - before) on the check rows: {paired['mean_change']:+.5f} (se {paired['se']:.5f}, z {paired['z']:+.2f})", flush=True)

    # ------------------------------------------------------------------ D
    clock.start("D")
    after = final["after"]
    both_identical = after["neural_check"]["gap"] == 0.0 and after["neural_check"]["se"] == 0.0
    verdict = {"IMPLEMENTATION_PASS": bool(all(a_ok.values()) and all(h["cond_augmented"] < 1e12 for h in history)
                                           and all(np.isfinite(h["signed_gap"]) for h in history)),
               "BALANCE_LEARNING_PASS": bool(chosen > 0 and after["neural_check"]["upper"] <= THRESHOLD
                                             and paired["mean_change"] < 0 and -paired["mean_change"] > 2 * paired["se"]),
               "EFFECT_SANITY_PASS": bool(abs(after["after_judging"]["estimate"]["theta"] - 1.0) <= 0.10),
               "critics_identical_after": both_identical}
    if both_identical and verdict["BALANCE_LEARNING_PASS"]:
        verdict["BALANCE_LEARNING_PASS"] = "INCONCLUSIVE"
    verdict["FINAL"] = "PASS" if (verdict["IMPLEMENTATION_PASS"] and verdict["BALANCE_LEARNING_PASS"] is True
                                  and verdict["EFFECT_SANITY_PASS"]) else "FAIL"
    rec["verdict"] = verdict
    rec["D"] = {"seconds": clock.elapsed()}
    save("COMPLETED")
    print(f"\nverdict: {verdict}\nwritten {result_path} | {rec['seconds_total']}s total", flush=True)
    return 0 if verdict["FINAL"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
