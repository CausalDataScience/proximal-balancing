"""Two arms on Shapes3D S1: a random start and an autoencoder start, otherwise the same balancing experiment.

    python3 -B run_direct_balance_ae_warmstart.py

Order (PRD section 10): data and autoencoder, then arm R, then arm AE, then BOTH selections are locked, and only
then the independent checks and the effect estimates of the initial, selected and final states of each arm, and
last the actual-update diagnostics.  A sixty-minute monotonic cap covers all of it; a stage that runs out of time
saves what it has and says so.  Nothing stops because a number looks bad.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import direct_balance_ae_warmstart as ae
import direct_balance_data as db
import direct_balance_eval as de
import direct_balance_net as dn
import direct_balance_tiny16 as t16
import direct_balance_v3 as v3
import family_v2_dgp as g
import family_v2_images as im
import run_family_v2 as rf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "direct_balance" / "ae_warmstart_v1"
PRD = HERE.parent / "memo" / "2026-09-22-tiny16-autoencoder-warmstart-prd-v1.md"
BENCH, SPLIT_NAME, SPLIT = "shapes3d", "S1", (0,)
THRESHOLD, EFFECT_MARKER = 1.5e-3, 0.15
BUDGET_S = {"data_and_ae": 600, "arm_R": 720, "arm_AE": 720, "evaluation": 960, "diagnostics": 600}
TOTAL_S = 3600
Z_UPPER = 2.9354506971231146                      # Phi^{-1}(1 - 0.05/30), the descriptive marker of earlier runs
MIN_FREE_GIB = 2.0
SOURCES = ["direct_balance_ae_warmstart.py", "run_direct_balance_ae_warmstart.py",
           "test_direct_balance_ae_warmstart.py", "direct_balance_v3.py", "direct_balance_tiny16.py",
           "direct_balance_net.py", "direct_balance_eval.py", "direct_balance_data.py",
           "family_v2_dgp.py", "family_v2_probe.py", "family_v2_images.py"]
READ_ONLY = ["direct_balance_v3.py", "direct_balance_tiny16.py", "direct_balance_net.py",
             "direct_balance_eval.py", "direct_balance_data.py"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_arr(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_once(name: str, payload) -> None:
    path = OUT / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)      # fail closed if it already exists
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(rf._jsonable(payload), indent=1, allow_nan=False) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "summary.json").exists():
        raise SystemExit("summary.json exists: this experiment was run once and is not rerun")
    if shutil.disk_usage(OUT).free / 2 ** 30 < MIN_FREE_GIB:
        raise SystemExit(f"less than {MIN_FREE_GIB} GiB free; stopping without deleting anything")
    clock = ae.Deadline(TOTAL_S)
    log = lambda m: print(m, flush=True)
    ledger, status = {}, {}

    dev = dn.device()
    images, groups = db.load_bank(BENCH)
    data_full = g.generate(g.LEVELS[db.BASE_LEVEL], db.N, ae.DATA_SEED)
    src = db.draw_images(data_full, groups, db.BENCHMARKS[BENCH], ae.DATA_SEED)
    rows = db.split_rows(db.N, ae.DATA_SEED)
    x, a, y = data_full["X"], data_full["A"], data_full["Y"]
    held, kept = db.image_columns(SPLIT)
    tiny = t16.TinyBatches(images, src, x, a, dev, standardise_rows=rows["represent"])
    ids = ae.image_id_split(src, rows["represent"])

    tests = subprocess.run([sys.executable, "-B", str(HERE / "test_direct_balance_ae_warmstart.py")],
                           capture_output=True, text=True, cwd=HERE)
    write_once("protocol.json", {
        "protocol_id": "direct-balance-ae-warmstart-v1", "frozen_at_utc": rf.now(),
        "status": "development and diagnosis on data already looked at many times; not a confirmation",
        "prd": {"path": str(PRD.relative_to(HERE.parent)), "sha256": sha(PRD)},
        "source_sha256": {f: sha(HERE / f) for f in SOURCES},
        "read_only_sources": {f: sha(HERE / f) for f in READ_ONLY},
        "unit_tests": {"returncode": tests.returncode, "stdout_tail": tests.stdout[-400:], "stderr_tail": tests.stderr[-400:]},
        "benchmark": BENCH, "data_seed": ae.DATA_SEED,
        "split": {"name": SPLIT_NAME, "held_images": held.tolist(), "kept_images": kept.tolist()},
        "rows_sha256": {k: sha_arr(v) for k, v in rows.items()}, "row_sizes": {k: int(len(v)) for k, v in rows.items()},
        "rows_disjoint": bool(len(np.unique(np.concatenate(list(rows.values())))) == db.N),
        "source_index_sha256": sha_arr(src), "data_fingerprint_sha256": sha_arr(np.concatenate([a, y])),
        "bank": {"split_rows_sha256": sha(im.BANK_DIR / "split_rows.npz"), "manifest_sha256": sha(im.BANK_DIR / "manifest.json")},
        "ae_image_ids": {"fit": int(len(ids["fit"])), "val": int(len(ids["val"])), "all": int(len(ids["all"])),
                         "fit_sha256": sha_arr(ids["fit"]), "val_sha256": sha_arr(ids["val"]),
                         "overlap": int(len(np.intersect1d(ids["fit"], ids["val"])))},
        "preprocessing": tiny.report(), "seed_purposes": ae.PURPOSE,
        "constants": {"ae": ae.AE, "critic": ae.CRITIC, "train": {**ae.TRAIN, "checkpoints": list(ae.TRAIN["checkpoints"])},
                      "select": ae.SELECT, "check": de.CHECK, "nuisance": de.NUISANCE,
                      "actual_update_ordinals": list(ae.ACTUAL_UPDATE_ORDINALS), "sd_floor": ae.SD_FLOOR},
        "markers_not_gates": {"threshold": THRESHOLD, "effect": EFFECT_MARKER, "z_for_upper": Z_UPPER},
        "budget_s": {**BUDGET_S, "total": TOTAL_S},
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__},
        "device": str(dev)})
    log(f"protocol written | unit tests returncode {tests.returncode} | device {dev} | "
        f"AE images fit {len(ids['fit'])} val {len(ids['val'])}")

    # ------------------------------------------------------------------ the autoencoder
    t0 = time.monotonic()
    ae_out = ae.train_autoencoder(tiny, ids, ae.Deadline(min(BUDGET_S["data_and_ae"], clock.left())), log=log)
    ledger["data_and_ae"] = round(time.monotonic() - t0, 1)
    write_once("ae.json", {k: v for k, v in ae_out.items() if k != "state"})
    torch.save(ae_out["state"], OUT / "ae_encoder.pt")
    log(f"  AE: {ae_out['epochs_completed']} epochs completed, chosen epoch {ae_out['chosen_epoch']}, "
        f"val mse {ae_out['chosen_val_mse']:.5f}, {ae_out['seconds']}s | usable {ae_out['usable']}")

    # ------------------------------------------------------------------ the two arms
    arms = {}
    for arm in ("R", "AE"):
        t0 = time.monotonic()
        if arm == "AE" and not ae_out["usable"]:
            status[arm] = "skipped: no usable autoencoder initialisation"
            log(f"  [{arm}] {status[arm]}")
            continue
        rep = ae.build_representation(kept, x.shape[1], dev, arm, ae_out["state"] if arm == "AE" else None)
        init_sha = ae.sha_state(rep.state_dict())
        budget = ae.Deadline(min(BUDGET_S[f"arm_{arm}"], clock.left() - BUDGET_S["evaluation"] * 0.5))
        out = ae.run_arm(arm, rep, tiny, rows, SPLIT, kept, held, budget, log=log)
        out["initial_weights_sha256"] = init_sha
        arms[arm] = out
        ledger[f"arm_{arm}"] = round(time.monotonic() - t0, 1)
        status[arm] = out["status"]
        torch.save({"checkpoints": out["checkpoint_states"], "diag": out["diag"]}, OUT / f"weights_{arm}.pt")
        log(f"  [{arm}] {out['status']}: {out['iterations_done']}/{out['iterations_planned']} iterations, "
            f"{out['actual_updates']} actual updates, {out['seconds']}s")

    # ------------------------------------------------------------------ lock both selections, then look
    t0 = time.monotonic()
    picks = {arm: ae.select(out, tiny, rows, SPLIT, kept, held, x.shape[1], dev, log=log) for arm, out in arms.items()}
    write_once("selection.json", {"locked_utc": rf.now(),
                                  **{arm: {"chosen": p["chosen"], "table": p["table"],
                                           "checkpoint_sha256": {str(k): ae.sha_state(v) for k, v in arms[arm]["checkpoint_states"].items()}}
                                     for arm, p in picks.items()}})
    log(f"  selections locked: {{{', '.join(f'{k}: {v['chosen']}' for k, v in picks.items())}}}")

    mixed = t16.TinyBatches(images, src, x, a, dev, standardise_rows=rows["represent"], held64=held)
    u = data_full["U_eval_only"]
    evaluation = {}
    for arm, out in arms.items():
        chosen, last = picks[arm]["chosen"], max(out["checkpoint_states"])
        seen, rec = {}, {}
        for label, k in (("initial", 0), ("selected", chosen), ("final", last)):
            key = ae.sha_state(out["checkpoint_states"][k]) + json.dumps(out["checkpoint_normalisers"][k], sort_keys=True)
            if key in seen:
                rec[label] = {"checkpoint": k, "same_state_as": seen[key]}
                log(f"  [{arm}] {label:8s} (ckpt {k:3d}): identical to {seen[key]}")
                continue
            seen[key] = label
            w = v3.load_normalised(out["checkpoint_states"][k], kept, x.shape[1], dev, t16.SPEC16,
                                   out["checkpoint_normalisers"][k])
            c16 = de.check(w, tiny, rows["check"], SPLIT, ae.seed_for("check16"), t16.SPEC16, return_rows=True)
            d16 = c16.pop("rows_d")
            c64 = de.check(w, mixed, rows["check"], SPLIT, ae.seed_for("check64"), dn.SPECS["shapes3d"])
            est = de.estimate(w, tiny, a, y, rows, SPLIT, ae.seed_for("nuisance"))
            scores = est.pop("scores")
            zn_, ze_ = (dn.representation_values(w, tiny, rows[r_], kept, grad=False).cpu().numpy().astype(np.float64)
                        for r_ in ("nuisance", "evaluate"))
            coef = np.linalg.lstsq(np.column_stack([np.ones(len(zn_)), zn_]), u[rows["nuisance"]], rcond=None)[0]
            rec[label] = {"checkpoint": k,
                          "check16": {**c16, "upper": max(c16["gap"], 0.0) + Z_UPPER * c16["se"]},
                          "check64": {**c64, "upper": max(c64["gap"], 0.0) + Z_UPPER * c64["se"]},
                          "effect": {"theta": est["theta"], "score_se": float(scores.std(ddof=1) / np.sqrt(len(scores))),
                                     "abs_error": abs(est["theta"] - 1.0), "theta_sealed_linear": est["theta_sealed_linear"],
                                     "outside_clip_before": est["outside_clip_before"]},
                          "u_r2_from_Z_after_lock": float(1 - (u[rows["evaluate"]] - np.column_stack([np.ones(len(ze_)), ze_]) @ coef).var() / u[rows["evaluate"]].var()),
                          "rows_d_mean": float(d16.mean())}
            rec[label]["_rows_d"] = d16
            log(f"  [{arm}] {label:8s} (ckpt {k:3d}): 16 gap {c16['gap']:+.5f} se {c16['se']:.5f} upper "
                f"{rec[label]['check16']['upper']:.5f} | 64 gap {c64['gap']:+.5f} | theta {est['theta']:+.3f} "
                f"(|err| {abs(est['theta'] - 1.0):.3f}) | U R2 {rec[label]['u_r2_from_Z_after_lock']:.3f}")
        paired = {}
        base = rec["initial"].get("_rows_d")
        for label in ("selected", "final"):
            other = rec[label].get("_rows_d")
            if base is None or other is None:
                continue
            diff = other - base
            se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
            paired[label] = {"mean_change": float(diff.mean()), "se": se, "n": int(len(diff)),
                             "ci95": [float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)]}
        for label in rec:
            rec[label].pop("_rows_d", None)
        evaluation[arm] = {"selection": picks[arm], "states": rec, "paired_vs_initial": paired}
    write_once("evaluation.json", evaluation)
    ledger["evaluation"] = round(time.monotonic() - t0, 1)

    # ------------------------------------------------------------------ actual-update diagnostics
    t0 = time.monotonic()
    diag = {}
    for arm, out in arms.items():
        budget = ae.Deadline(min(BUDGET_S["diagnostics"] / max(len(arms), 1), clock.left() - 60))
        diag[arm] = ae.actual_update_diagnostics(out, tiny, rows, SPLIT, kept, held, x.shape[1], dev, budget, log=log)
    write_once("diagnostics.json", diag)
    ledger["diagnostics"] = round(time.monotonic() - t0, 1)

    # ------------------------------------------------------------------ the six-row table
    table = []
    for arm in ("R", "AE"):
        if arm not in evaluation:
            continue
        for label in ("initial", "selected", "final"):
            s = evaluation[arm]["states"][label]
            if "same_state_as" in s:
                same = evaluation[arm]["states"][s["same_state_as"]]
                row = {"arm": arm, "state": label, "checkpoint": s["checkpoint"], "same_state_as": s["same_state_as"],
                       **{k: same[k] for k in ("check16", "check64", "effect", "u_r2_from_Z_after_lock")}}
            else:
                row = {"arm": arm, "state": label, "checkpoint": s["checkpoint"], **{k: s[k] for k in ("check16", "check64", "effect", "u_r2_from_Z_after_lock")}}
            row["cumulative_actual_updates"] = (0 if label == "initial" else arms[arm]["actual_updates"])
            row["markers"] = {"upper16_at_most_threshold": row["check16"]["upper"] <= THRESHOLD,
                              "abs_error_at_most_marker": row["effect"]["abs_error"] <= EFFECT_MARKER}
            table.append(row)
    write_once("summary.json", {
        "artifact": "direct_balance_ae_warmstart_summary", "generated_utc": rf.now(),
        "status": "development and diagnosis, not a confirmation",
        "arms": {arm: {"status": status.get(arm), "iterations_planned": out["iterations_planned"],
                       "iterations_done": out["iterations_done"], "actual_updates": out["actual_updates"],
                       "stopped": out["stopped"], "selected": picks[arm]["chosen"]}
                 for arm, out in arms.items()},
        "autoencoder": {"epochs_completed": ae_out["epochs_completed"], "chosen_epoch": ae_out["chosen_epoch"],
                        "chosen_val_mse": ae_out["chosen_val_mse"], "usable": ae_out["usable"]},
        "table": table, "paired_vs_initial": {arm: evaluation[arm]["paired_vs_initial"] for arm in evaluation},
        "actual_update_diagnostics": {arm: [{k: v for k, v in d.items() if k != "with_fresh_critics"}
                                            | {"fresh": d.get("with_fresh_critics", {}).get("change_raw",
                                                                                            d.get("with_fresh_critics", {}).get("status"))}
                                            for d in diag[arm]] for arm in diag},
        "time_ledger_s": {**ledger, "total": round(TOTAL_S - clock.left(), 1), "cap": TOTAL_S},
        "read_only_sources_unchanged": all(sha(HERE / f) == json.loads((OUT / "protocol.json").read_text())["read_only_sources"][f] for f in READ_ONLY)})
    log("\n" + json.dumps([{k: r[k] for k in ("arm", "state", "checkpoint", "cumulative_actual_updates")}
                           | {"gap16": round(r["check16"]["gap"], 5), "upper16": round(r["check16"]["upper"], 5),
                              "gap64": round(r["check64"]["gap"], 5), "theta": round(r["effect"]["theta"], 3),
                              "abs_error": round(r["effect"]["abs_error"], 3)} for r in table], indent=1))
    log(f"time ledger {ledger} | written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
