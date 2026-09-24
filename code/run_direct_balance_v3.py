"""Two runs of direct image balancing on Shapes3D S1: A fixes the validation-row reuse, B adds the
representation standardisation.  Both are evaluated to the end; neither is gated on a statistical threshold.

    python3 -B run_direct_balance_v3.py

Stages are write-once and resumable: a second invocation continues after the last file present.  Every run
reports what it actually did (updates taken and skipped), what the critics looked like, how the representation
was scaled, what the independent check and the effect estimate say, and whether each saved update's decrease
survives refitting the critics.  The effect tolerance and the screen threshold are printed as markers, not as
gates: nothing here stops because performance is poor.

These data have been looked at many times, so this is development and diagnosis, never a confirmation.
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
import direct_balance_v3 as v3
import family_v2_dgp as g
import family_v2_images as im
import run_family_v2 as rf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "direct_balance" / "v3_ab"
BENCH, SPLIT_NAME, SPLIT, SEED = "shapes3d", "S1", (0,), 96_900_800
THRESHOLD, EFFECT_TOL = 1.5e-3, 0.15
CAP_PER_ARM_S, CAP_TOTAL_S = 1_500, 3_600
Z95 = 1.6448536269514722
ARMS = (("A", False), ("B", True))
SOURCES = ["direct_balance_data.py", "direct_balance_net.py", "direct_balance_eval.py", "direct_balance_tiny16.py",
           "direct_balance_v3.py", "run_direct_balance_v3.py", "test_direct_balance_v3.py",
           "test_direct_balance_tiny16.py", "family_v2_dgp.py", "family_v2_probe.py", "family_v2_images.py"]


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


def evaluate(arm: str, out: dict, tiny, data_full, rows, kept, held, x, a, y, dev, log) -> dict:
    """The three representations every run must report: untrained, the one the selection fold chose, and the last."""
    states, stats = out["checkpoint_states"], out["checkpoint_normalisers"]
    last = max(states)
    wrapped = {k: v3.load_normalised(states[k], kept, x.shape[1], dev, t16.SPEC16, stats[k]) for k in states}
    # the selection must see the same Z the critics saw, so it runs on the wrapped representations directly
    table = {}
    for k, w in wrapped.items():
        table[k] = de.fresh_gap(w, tiny, rows["represent"], rows["select"], SPLIT, dr.purpose_seed(SEED, 0, 5), t16.SPEC16)
    chosen = min(sorted(table), key=lambda k: (table[k]["clipped_gap"], k))
    log(f"  [{arm}] selection on the select fold: {chosen} | "
        f"gaps {{{', '.join(f'{k}: {v['gap']:+.4f}' for k, v in sorted(table.items()))}}}")
    check_seed = dr.purpose_seed(SEED, 0, 1)
    u = data_full["U_eval_only"]
    audit, d_rows, est = {}, {}, {}
    mixed = t16.TinyBatches(im.load_images(), tiny.src, x, a, dev, standardise_rows=rows["represent"], held64=held)
    audit64 = {}
    for label, k in (("initial", 0), ("chosen", chosen), ("last", last)):
        w = wrapped[k]
        chk = de.check(w, tiny, rows["check"], SPLIT, check_seed, t16.SPEC16, return_rows=True)
        d_rows[label] = chk.pop("rows_d")
        audit[label] = {"checkpoint": k, **chk}
        e = de.estimate(w, tiny, a, y, rows, SPLIT, dr.purpose_seed(SEED, 0, 2)); e.pop("scores")
        zn_, ze_ = (dn.representation_values(w, tiny, rows[r_], kept, grad=False).cpu().numpy().astype(np.float64)
                    for r_ in ("nuisance", "evaluate"))
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(zn_)), zn_]), u[rows["nuisance"]], rcond=None)[0]
        e["u_r2_from_Z_after_judging"] = float(1 - (u[rows["evaluate"]] - np.column_stack([np.ones(len(ze_)), ze_]) @ coef).var() / u[rows["evaluate"]].var())
        e["abs_error"] = abs(e["theta"] - 1.0)
        est[label] = e
        w64 = v3.load_normalised(states[k], kept, x.shape[1], dev, t16.SPEC16, stats[k])
        audit64[label] = {"checkpoint": k, **de.check(w64, mixed, rows["check"], SPLIT, check_seed, dn.SPECS["shapes3d"])}
        log(f"  [{arm}] {label:7s} (ckpt {k:3d}): 16x16 audit gap {chk['gap']:+.5f} se {chk['se']:.5f} upper "
            f"{chk['upper']:.5f} | 64x64 audit gap {audit64[label]['gap']:+.5f} | theta {e['theta']:+.3f} "
            f"(|err| {e['abs_error']:.3f}) | U R2 {e['u_r2_from_Z_after_judging']:.3f}")
    paired = {}
    for label in ("chosen", "last"):
        diff = d_rows[label] - d_rows["initial"]
        p = {"mean_change": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(diff))), "n": int(len(diff))}
        p["one_sided_95_upper"] = p["mean_change"] + Z95 * p["se"]
        p["decrease_significant"] = bool(p["one_sided_95_upper"] < 0)
        paired[label] = p
    return {"selection": {"chosen": chosen, "table": {str(k): v for k, v in table.items()}},
            "audit_16": audit, "audit_64": audit64, "effect": est, "paired_vs_initial": paired,
            "markers": {"threshold": THRESHOLD, "effect_tolerance": EFFECT_TOL,
                        "audit_upper_at_most_t": audit["chosen"]["upper"] <= THRESHOLD,
                        "audit_gap_fell": audit["chosen"]["gap"] < audit["initial"]["gap"],
                        "effect_within_tolerance": est["chosen"]["abs_error"] <= EFFECT_TOL,
                        "error_below_initial": est["chosen"]["abs_error"] < est["initial"]["abs_error"]}}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t_wall = time.time()
    log = lambda m: print(m, flush=True)
    dev = dn.device()
    images, groups = db.load_bank(BENCH)
    data_full = g.generate(g.LEVELS[db.BASE_LEVEL], db.N, SEED)
    src = db.draw_images(data_full, groups, db.BENCHMARKS[BENCH], SEED)
    rows = db.split_rows(db.N, SEED)
    x, a, y = data_full["X"], data_full["A"], data_full["Y"]
    held, kept = db.image_columns(SPLIT)
    tiny = t16.TinyBatches(images, src, x, a, dev, standardise_rows=rows["represent"])

    if not (OUT / "protocol_v1.json").exists():
        tests = subprocess.run([sys.executable, "-B", str(HERE / "test_direct_balance_v3.py")], capture_output=True, text=True, cwd=HERE)
        torch.manual_seed(dr.purpose_seed(SEED, 0, 0))
        probe = dn.Representation(kept, x.shape[1], spec=t16.SPEC16)
        plan = v3.fold_plan(rows, dr.purpose_seed(SEED, 0, 0))
        write_once("protocol_v1.json", {
            "protocol_id": "direct-balance-v3-ab", "frozen_at_utc": rf.now(), "status": "development and diagnosis, not confirmation",
            "arms": {"A": "fixed validation rows and the exact whole-fold gradient", "B": "A plus the representation standardisation"},
            "source_sha256": {f: sha(HERE / f) for f in SOURCES}, "unit_tests": {"returncode": tests.returncode, "tail": tests.stderr[-200:]},
            "benchmark": BENCH, "seed": SEED, "split": {"name": SPLIT_NAME, "held_images": held.tolist(), "kept_images": kept.tolist()},
            "rows_sha256": {k: sha_arr(v) for k, v in rows.items()}, "row_sizes": {k: int(len(v)) for k, v in rows.items()},
            "fold_plan": v3.plan_report(plan), "data_fingerprint_sha256": sha_arr(np.concatenate([a, y, src.ravel().astype(np.float64)])),
            "initial_weights_sha256": sha_state(probe.state_dict()), "resize": tiny.report(),
            "training": {"iters": v3.ITERS, "checkpoints": list(v3.CHECKPOINTS), "lr": dn.TRAIN["lr_cnn"],
                         "critic": v3.CRITIC, "diagnostic_pairs": [list(p) for p in v3.DIAG_PAIRS],
                         "diagnostic_refit_steps": v3.DIAG_REFIT_STEPS, "sd_floor": v3.SD_FLOOR},
            "markers_not_gates": {"threshold": THRESHOLD, "effect_tolerance": EFFECT_TOL},
            "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__},
            "device": str(dev), "caps_s": {"per_arm": CAP_PER_ARM_S, "total": CAP_TOTAL_S}})
        del probe
        log(f"protocol frozen | unit tests returncode {tests.returncode} | device {dev}")
    proto = json.loads((OUT / "protocol_v1.json").read_text())
    moved = [f for f, h in proto["source_sha256"].items() if sha(HERE / f) != h]
    if moved:
        raise SystemExit(f"sources changed since the protocol was frozen: {moved}")
    if proto["fold_plan"]["leaks"]:
        raise SystemExit(f"the fold plan leaks: {proto['fold_plan']['leaks']}")

    for arm, standardise in ARMS:
        name = f"arm{arm}.json"
        if (OUT / name).exists():
            log(f"arm {arm}: already on disk")
            continue
        if time.time() - t_wall > CAP_TOTAL_S:
            write_once(name, {"status": "TIMEOUT before this arm"}); break
        log(f"\n=== arm {arm}: {proto['arms'][arm]}")
        try:
            out = v3.train(tiny, rows, SPLIT, dr.purpose_seed(SEED, 0, 0), x.shape[1], t16.SPEC16, standardise,
                           max_seconds=CAP_PER_ARM_S, log=log)
        except TimeoutError as err:
            write_once(name, {"status": "TIMEOUT", "reason": str(err)}); continue
        if sha_state(out["initial_state"]) != proto["initial_weights_sha256"]:
            write_once(name, {"status": "INVALID", "reason": "initial weights differ from the protocol"}); continue
        torch.save({"checkpoints": out["checkpoint_states"], "diag": out["diag_states"]}, OUT / f"weights_{arm}.pt")
        log(f"  [{arm}] trained {out['seconds']}s | updates taken {out['steps_taken']}/{len(out['history'])} | "
            f"lifted base kept {sum(any(c['kept_the_lifted_base'] for c in h['critics']) for h in out['history'])}")
        ev = evaluate(arm, out, tiny, data_full, rows, kept, held, x, a, y, dev, log)
        log(f"  [{arm}] refit diagnostic:")
        diag = v3.refit_diagnostic(out, tiny, rows, SPLIT, x.shape[1], t16.SPEC16, standardise,
                                   dr.purpose_seed(SEED, 0, 9), log=log)
        write_once(name, {"arm": arm, "standardise": standardise, "status": "COMPLETED",
                          "training": {"seconds": out["seconds"], "steps_taken": out["steps_taken"],
                                       "history": out["history"], "plan_report": out["plan_report"],
                                       "checkpoint_normalisers": out["checkpoint_normalisers"],
                                       "checkpoint_critics": out["checkpoint_critics"],
                                       "checkpoint_weight_sha256": {k: sha_state(v) for k, v in out["checkpoint_states"].items()}},
                          "evaluation": ev, "refit_diagnostic": diag, "utc": rf.now()})
        log(f"  [{arm}] written {OUT / name}")

    if not (OUT / "summary_v1.json").exists() and all((OUT / f"arm{a_}.json").exists() for a_, _ in ARMS):
        arms = {a_: json.loads((OUT / f"arm{a_}.json").read_text()) for a_, _ in ARMS}
        summary = {"artifact": "direct_balance_v3_ab_summary", "generated_utc": rf.now(),
                   "status": "development and diagnosis, not confirmation", "seconds_total": round(time.time() - t_wall, 1),
                   "arms": {}}
        for a_, rec in arms.items():
            if rec.get("status") != "COMPLETED":
                summary["arms"][a_] = {"status": rec.get("status")}; continue
            ev, tr = rec["evaluation"], rec["training"]
            summary["arms"][a_] = {
                "updates_taken": tr["steps_taken"], "updates_planned": len(tr["history"]),
                "chosen": ev["selection"]["chosen"],
                "audit16": {k: {"gap": v["gap"], "se": v["se"], "upper": v["upper"]} for k, v in ev["audit_16"].items()},
                "audit64": {k: {"gap": v["gap"], "upper": v["upper"]} for k, v in ev["audit_64"].items()},
                "effect": {k: {"theta": v["theta"], "abs_error": v["abs_error"], "u_r2": v["u_r2_from_Z_after_judging"]}
                           for k, v in ev["effect"].items()},
                "paired_vs_initial": ev["paired_vs_initial"], "markers": ev["markers"],
                "h_scale_last": tr["history"][-1]["scale"] if tr["history"] else None,
                "refit_diagnostic": {"both_fell": sum(d["both_fell"] for d in rec["refit_diagnostic"]),
                                     "only_fixed_fell": sum(d["only_the_fixed_critics_fell"] for d in rec["refit_diagnostic"]),
                                     "of": len(rec["refit_diagnostic"])}}
        write_once("summary_v1.json", summary)
        log("\n" + json.dumps(summary["arms"], indent=1)[:3000])
        log(f"written {OUT / 'summary_v1.json'} | {summary['seconds_total']}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
