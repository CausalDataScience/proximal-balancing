"""Push the residual discrepancy to zero: keep training until critics that are refitted from scratch, with a
large budget, can no longer beat the base critic.

    python3 -B run_direct_balance_push.py [R|AE]

The warm-start run showed the representation freezing after a handful of updates.  The freeze is not a decision
that balance was reached: the pooled cross-fitted gap on 4,800 rows has a standard error of about 0.0016, so a
true residual of 0.004 can read as -0.001, the clip makes the objective flat there, and the loop then repeats
the same measurement forever because nothing moves.

So the stall is contested rather than obeyed.  When the pooled gap is not positive, every critic is thrown away
and refitted FROM SCRATCH with a much larger budget on the same fixed fold rows.  If those stronger critics find
a positive gap, the representation takes its step and training continues.  Only when the strong critics agree,
several times in a row, does the run call it converged.  The step rule itself is unchanged: a step is taken only
against a positive pooled gap, and a negative gap is never minimised.

The check fold is not touched here.  Progress is watched on the selection fold, and the independent check and
the effect estimate are computed once at the end, on the initial state, the state the selection fold picks, and
the last state.
"""
from __future__ import annotations

import hashlib
import json
import os
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
OUT = HERE.parent / "results" / "direct_balance" / "push_v1"
BENCH, SPLIT, SPLIT_NAME = "shapes3d", (0,), "S1"
PUSH = {"iters": 300, "escape_steps": 600, "stall_limit": 4, "checkpoint_every": 25,
        "monitor_every": 25, "time_cap_s": 2400}
THRESHOLD, EFFECT_MARKER = 1.5e-3, 0.15
Z_UPPER = 2.9354506971231146
SOURCES = ["run_direct_balance_push.py", "direct_balance_ae_warmstart.py", "direct_balance_v3.py",
           "direct_balance_tiny16.py", "direct_balance_net.py", "direct_balance_eval.py", "direct_balance_data.py"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_once(name: str, payload) -> None:
    fd = os.open(OUT / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(rf._jsonable(payload), indent=1, allow_nan=False) + "\n")


def main(arm: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "result.json").exists():
        raise SystemExit("result.json exists; this run is write-once")
    clock = ae.Deadline(PUSH["time_cap_s"])
    log = lambda m: print(m, flush=True)
    dev = dn.device()
    images, groups = db.load_bank(BENCH)
    data_full = g.generate(g.LEVELS[db.BASE_LEVEL], db.N, ae.DATA_SEED)
    src = db.draw_images(data_full, groups, db.BENCHMARKS[BENCH], ae.DATA_SEED)
    rows = db.split_rows(db.N, ae.DATA_SEED)
    x, a, y = data_full["X"], data_full["A"], data_full["Y"]
    held, kept = db.image_columns(SPLIT)
    tiny = t16.TinyBatches(images, src, x, a, dev, standardise_rows=rows["represent"])

    ae_state = None
    if arm == "AE":
        path = HERE.parent / "results" / "direct_balance" / "ae_warmstart_v1" / "ae_encoder.pt"
        if not path.exists():
            raise SystemExit(f"no autoencoder at {path}; run the warm-start experiment first")
        ae_state = torch.load(path, map_location=dev)
    rep = ae.build_representation(kept, x.shape[1], dev, arm, ae_state)
    init_state = {k: v.detach().cpu().clone() for k, v in rep.state_dict().items()}
    plan = ae.fold_plan(rows)
    opt = torch.optim.Adam(rep.parameters(), lr=ae.TRAIN["lr"])
    with torch.no_grad():
        norm = v3.Normaliser.fit(dn.representation_values(rep, tiny, plan["represent"], kept, grad=False),
                                 enabled=True).frozen()
    states = [None] * plan["folds"]

    def refit(steps: int, iteration: int, old, fresh: bool):
        pairs = []
        for i in range(plan["folds"]):
            b, a_, states[i] = ae.fit_critics(rep, norm, tiny, plan["fit_sets"][i], plan["val_sets"][i], kept, held,
                                              steps, i, iteration, None if fresh else states[i],
                                              None if fresh else old, clock)
            pairs.append((b, a_))
        return pairs

    write_once("protocol.json", {"protocol_id": "direct-balance-push-v1", "arm": arm, "frozen_at_utc": rf.now(),
                                 "status": "development and diagnosis, not a confirmation",
                                 "source_sha256": {f: sha(HERE / f) for f in SOURCES},
                                 "data_seed": ae.DATA_SEED, "split": SPLIT_NAME, "push": PUSH,
                                 "rows": {k: int(len(v)) for k, v in rows.items()},
                                 "initial_weights_sha256": ae.sha_state(init_state), "device": str(dev),
                                 "rule": "a step is taken only against a positive pooled gap; when the gap is not "
                                         "positive the critics are refitted from scratch with a larger budget and "
                                         "the gap is read again; a negative gap is never minimised"})
    log(f"push arm {arm} on {dev} | initial weights {ae.sha_state(init_state)[:12]}")

    pairs = refit(ae.CRITIC["initial_steps"], 0, None, True)
    t0 = time.monotonic()
    history, saved, monitor = [], {0: {"state": init_state, "normaliser": norm.record()}}, {}
    actual, stalls, escapes, escapes_that_worked = 0, 0, 0, 0
    stop_reason = None
    for it in range(1, PUSH["iters"] + 1):
        if clock.over():
            stop_reason = f"time cap at iteration {it}"
            break
        step, norm_next = ae.update(rep, tiny, plan, pairs, kept, held, opt, norm)
        escaped = False
        if not step["stepped"]:
            escapes += 1
            strong = refit(PUSH["escape_steps"], it, None, True)          # throw the critics away, fit big ones
            step2, norm2 = ae.update(rep, tiny, plan, strong, kept, held, opt, norm)
            step2["escape_attempt"] = True
            if step2["stepped"]:
                escapes_that_worked += 1
                escaped, stalls = True, 0
                step, norm_next, pairs = step2, norm2, strong
            else:
                stalls += 1
                step = dict(step, escape_raw_gap=step2["raw_signed_gap"], escape_failed=True)
                norm_next, pairs = norm2, strong
        else:
            stalls = 0
        if step["stepped"]:
            actual += 1
        old_norm, norm = norm, norm_next
        if not escaped:
            pairs = refit(ae.CRITIC["steps_per_iter"], it, old_norm, False)
        history.append(dict(step, iteration=it, actual_update=actual if step["stepped"] else None,
                            stalls=stalls, escaped=escaped,
                            critics=[st["info"] for st in states], seconds=round(time.monotonic() - t0, 1)))
        if it % PUSH["checkpoint_every"] == 0 or it == PUSH["iters"]:
            saved[it] = {"state": {k: v.detach().cpu().clone() for k, v in rep.state_dict().items()},
                         "normaliser": norm.record()}
        if it % PUSH["monitor_every"] == 0 and not clock.over():
            w = v3.load_normalised({k: v.detach().cpu().clone() for k, v in rep.state_dict().items()}, kept,
                                   x.shape[1], dev, t16.SPEC16, norm.record())
            m = de.fresh_gap(w, tiny, plan["represent"], rows["select"], SPLIT, ae.seed_for("selection", 0, it), t16.SPEC16)
            monitor[it] = {"gap": m["gap"], "se": m["se"], "clipped_gap": m["clipped_gap"]}
            log(f"  monitor at {it}: select-fold fresh gap {m['gap']:+.5f} (se {m['se']:.5f})")
        if it <= 5 or it % 10 == 0 or escaped or step.get("escape_failed"):
            log(f"  iter {it:3d}: raw gap {step['raw_signed_gap']:+.5f} stepped {step['stepped']} "
                f"(actual {actual}) escaped {escaped} stalls {stalls} | h sd "
                f"{np.round(step['scale']['h_sd'], 3).tolist()} | {time.monotonic() - t0:.0f}s")
        if stalls >= PUSH["stall_limit"]:
            stop_reason = f"strong fresh critics found no positive gap {stalls} times in a row (iteration {it})"
            break
    done = len(history)
    last = max(saved)
    if last != done and done > 0:
        saved[done] = {"state": {k: v.detach().cpu().clone() for k, v in rep.state_dict().items()},
                       "normaliser": norm.record()}
    log(f"training stopped: {stop_reason or 'planned iterations complete'} | {done} iterations, {actual} actual "
        f"updates, {escapes} escape attempts of which {escapes_that_worked} freed a step | {time.monotonic() - t0:.0f}s")
    torch.save({k: v["state"] for k, v in saved.items()}, OUT / "checkpoints.pt")
    write_once("training.json", {"arm": arm, "iterations_done": done, "actual_updates": actual,
                                 "escape_attempts": escapes, "escapes_that_freed_a_step": escapes_that_worked,
                                 "stop_reason": stop_reason, "monitor_on_select": monitor,
                                 "history": history, "seconds": round(time.monotonic() - t0, 1)})

    # ------------------------------------------------------------------ lock the choice, then look
    table = {}
    for k in sorted(saved):
        w = v3.load_normalised(saved[k]["state"], kept, x.shape[1], dev, t16.SPEC16, saved[k]["normaliser"])
        table[k] = de.fresh_gap(w, tiny, plan["represent"], rows["select"], SPLIT, ae.seed_for("selection", 0, k), t16.SPEC16)
    chosen = None
    for k in sorted(table):
        if chosen is None or table[k]["clipped_gap"] < table[chosen]["clipped_gap"] - ae.SELECT["tolerance"]:
            chosen = k
    log(f"selection locked: {chosen} | gaps {{{', '.join(f'{k}: {v['gap']:+.4f}' for k, v in sorted(table.items()))}}}")
    write_once("selection.json", {"chosen": chosen, "table": {str(k): v for k, v in table.items()},
                                  "locked_utc": rf.now()})

    mixed = t16.TinyBatches(images, src, x, a, dev, standardise_rows=rows["represent"], held64=held)
    u = data_full["U_eval_only"]
    out, d_rows = {}, {}
    for label, k in (("initial", 0), ("selected", chosen), ("final", max(saved))):
        w = v3.load_normalised(saved[k]["state"], kept, x.shape[1], dev, t16.SPEC16, saved[k]["normaliser"])
        c16 = de.check(w, tiny, rows["check"], SPLIT, ae.seed_for("check16"), t16.SPEC16, return_rows=True)
        d_rows[label] = c16.pop("rows_d")
        c64 = de.check(w, mixed, rows["check"], SPLIT, ae.seed_for("check64"), dn.SPECS["shapes3d"])
        est = de.estimate(w, tiny, a, y, rows, SPLIT, ae.seed_for("nuisance"))
        scores = est.pop("scores")
        zn_, ze_ = (dn.representation_values(w, tiny, rows[r_], kept, grad=False).cpu().numpy().astype(np.float64)
                    for r_ in ("nuisance", "evaluate"))
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(zn_)), zn_]), u[rows["nuisance"]], rcond=None)[0]
        out[label] = {"checkpoint": k,
                      "check16": {**c16, "upper": max(c16["gap"], 0.0) + Z_UPPER * c16["se"]},
                      "check64": {**c64, "upper": max(c64["gap"], 0.0) + Z_UPPER * c64["se"]},
                      "effect": {"theta": est["theta"], "abs_error": abs(est["theta"] - 1.0),
                                 "score_se": float(scores.std(ddof=1) / np.sqrt(len(scores))),
                                 "theta_sealed_linear": est["theta_sealed_linear"]},
                      "u_r2_from_Z_after_lock": float(1 - (u[rows["evaluate"]] - np.column_stack([np.ones(len(ze_)), ze_]) @ coef).var() / u[rows["evaluate"]].var())}
        log(f"  {label:8s} (ckpt {k:3d}): 16 gap {c16['gap']:+.5f} se {c16['se']:.5f} upper "
            f"{out[label]['check16']['upper']:.5f} | 64 gap {c64['gap']:+.5f} | theta {est['theta']:+.3f} "
            f"(|err| {out[label]['effect']['abs_error']:.3f}) | U R2 {out[label]['u_r2_from_Z_after_lock']:.3f}")
    paired = {}
    for label in ("selected", "final"):
        diff = d_rows[label] - d_rows["initial"]
        se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        paired[label] = {"mean_change": float(diff.mean()), "se": se,
                         "ci95": [float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)]}
    write_once("result.json", {"artifact": "direct_balance_push", "arm": arm, "generated_utc": rf.now(),
                               "status": "development and diagnosis, not a confirmation",
                               "training": {"iterations_done": done, "actual_updates": actual,
                                            "escape_attempts": escapes, "escapes_that_freed_a_step": escapes_that_worked,
                                            "stop_reason": stop_reason},
                               "selection": {"chosen": chosen}, "states": out, "paired_vs_initial": paired,
                               "markers": {"threshold": THRESHOLD, "effect": EFFECT_MARKER}})
    log(f"paired change on the check rows: {json.dumps(paired)}")
    log(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "AE"))
