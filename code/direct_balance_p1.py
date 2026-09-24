"""P1 of the direct image balancing experiment: does the machinery do what it claims, before any full run
(PRD memo/2026-09-21-direct-image-balancing-prd-v1.md, section 8, stage P1).

    python3 -B direct_balance_p1.py run

Five questions, on two pre-specified splits and nothing else:

  A  positive control   on a representation known to be imbalanced, does the augmented critic find the extra
                        treatment information the held-out images carry?
  B  negative control   with the treatment shuffled, so that nothing predicts it, is the false signal small?
  C  gradient           when the fold's gap is positive, does the gradient reach the convolutional layers, not
                        just the head sitting on top of them?
  D  movement           do those weights actually change after the step?
  E  learning           over the twenty outer iterations, does the training gap fall without the independent gap
                        rising?

A and C are blocking: the PRD says that if the gradient does not reach the body, or the critic cannot read a
positive control, the full run does not start.

The two splits are fixed in advance and chosen for contrast.  S1 holds out one block and is balanceable in
SCM-1, so its gap should be drivable towards zero.  S5 holds out the fifth block and is NOT balanceable, so its
gap should not go away.  Neither is a result about the method: with one data set and one seed this is a check
that the code does what it says.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import direct_balance_data as db
import direct_balance_net as dn
import family_v2_dgp as g
import family_v2_images as im

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "direct_balance"
OUT = RESULTS / "p1_checks_v2.json"          # v1 stays on disk; see the note in `run`
PRD = HERE.parent / "memo" / "2026-09-21-direct-image-balancing-prd-v1.md"
SOURCES = ["direct_balance_data.py", "direct_balance_net.py", "direct_balance_eval.py", "direct_balance_p1.py"]
P1_SEED = 96_200_000                       # a range no earlier run has used
SPLITS = {"S1": (0,), "S5": (4,)}          # pre-specified: balanceable, and not balanceable
THRESHOLD = 1.5e-3                         # the development threshold of PRD section 6


def setup(seed: int, shuffle_treatment: bool = False):
    dev = dn.device()
    images = im.load_images()
    groups = db.image_bank()
    data = g.generate(g.LEVELS[db.BASE_LEVEL], db.N, seed)
    src = db.draw_images(data, groups, im.LEVELS, seed)
    rows = db.split_rows(db.N, seed)
    a = data["A"]
    if shuffle_treatment:
        a = np.random.default_rng(seed + 55).permutation(a)
    cached = np.concatenate([rows["represent"], rows["select"], rows["check"]])
    return dev, images, src, data["X"], a, rows, dn.Batches(images, src, data["X"], a, dev, cache_rows=cached,
                                                            standardise_rows=rows["represent"])


def conv_parameters(rep: dn.Representation):
    return [(n, p) for n, p in rep.named_parameters() if ".body." in n and n.endswith("weight") and p.dim() == 4]


def control(split, shuffle: bool, seed: int) -> dict:
    """Fit both critics against an untrained representation and read the gap on both folds.

    An untrained representation is the imbalanced case by construction: it cannot have removed anything, so the
    held-out images still carry the latent state.  With the treatment shuffled, nothing carries it, and whatever
    gap remains is what the fitting procedure invents.
    """
    dev, images, src, x, a, rows, batches = setup(seed, shuffle_treatment=shuffle)
    held, kept = db.image_columns(split)
    torch.manual_seed(seed)
    rep = dn.Representation(kept, x.shape[1]).to(dev)
    rng = np.random.default_rng(seed)
    base, aug, _ = dn.fit_critics(rep, batches, rows["represent"], kept, held, dn.TRAIN["critic_initial"], rng, None)
    on_train = dn.signed_gap(rep, base, aug, batches, rows["represent"], kept, held)
    on_select = dn.signed_gap(rep, base, aug, batches, rows["select"], kept, held)
    return {"treatment": "shuffled" if shuffle else "as generated",
            "fitted_fold": {k: round(v, 6) for k, v in on_train.items()},
            "independent_fold": {k: round(v, 6) for k, v in on_select.items()}}


def gradient_and_movement(split, seed: int) -> dict:
    """One update, watched: where the gradient lands and what moves."""
    dev, images, src, x, a, rows, batches = setup(seed)
    held, kept = db.image_columns(split)
    torch.manual_seed(seed)
    rep = dn.Representation(kept, x.shape[1]).to(dev)
    rng = np.random.default_rng(seed)
    base, aug, _ = dn.fit_critics(rep, batches, rows["represent"], kept, held, dn.TRAIN["critic_initial"], rng, None)
    before = {n: p.detach().clone() for n, p in rep.named_parameters()}
    opt = torch.optim.Adam(rep.parameters(), lr=dn.TRAIN["lr_cnn"])
    step = dn.cnn_step(rep, [(rows["represent"], base, aug)], batches, kept, held, opt)
    convs = conv_parameters(rep)
    grads = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in convs}
    moved = {n: float((p.detach() - before[n]).abs().max()) for n, p in convs}
    first_layer = [n for n, _ in convs if n.endswith("body.0.weight")]
    critics_frozen = all(p.requires_grad for p in list(base.parameters()) + list(aug.parameters()))
    return {"signed_gap": round(step["signed_gap"], 6), "stepped": step["stepped"],
            "grad_norm_all": step["grad_norm"],
            "conv_grad_norms": {n: round(v, 8) if v is not None else None for n, v in grads.items()},
            "conv_max_weight_change": {n: round(v, 8) for n, v in moved.items()},
            "deepest_layer_from_the_loss": first_layer,
            "grad_reaches_first_conv": all(grads[n] is not None and grads[n] > 1e-10 for n in first_layer),
            "first_conv_moved": all(moved[n] > 0 for n in first_layer),
            "critics_left_trainable_afterwards": critics_frozen}


def learning(split, seed: int) -> dict:
    """The full fit, judged where it cannot flatter itself.

    The first version of this check compared the chosen checkpoint with checkpoint zero on the `select` fold.
    The checkpoint is chosen as the smallest gap on that fold, zero included, so the comparison could not fail,
    and it used the critics the representation had been trained against.  Now the network before balancing and
    the chosen one are each handed to FRESH critics on the `check` fold, which selected nothing.
    """
    import direct_balance_eval as de
    dev, images, src, x, a, rows, batches = setup(seed)
    out = dn.train_split(batches, rows, split, seed, x.shape[1])
    _, kept = db.image_columns(split)
    fresh = {}
    for arm, state in (("before", out["initial_state"]), ("after", out["chosen_state"])):
        rep = dn.load_representation(state, kept, x.shape[1], dev)
        got = de.check(rep, batches, rows["check"], split, seed + 1)
        fresh[arm] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in got.items()}
    curve = [{"iteration": h["iteration"], "signed_gap_on_the_training_fold": round(h["signed_gap"], 6),
              "stepped": h["stepped"], "kept_the_lifted_base": any(c["kept_the_lifted_base"] for c in h["critics"])}
             for h in out["history"]]
    select = {k: round(v["clipped_gap"], 6) for k, v in out["checkpoint_select_gap"].items()}
    return {"split": out["split"], "kept_images": out["kept_images"], "held_images": out["held_images"],
            "chosen_checkpoint": out["chosen_checkpoint"], "pixel_cache": batches.cache_report(),
            "seconds": out["seconds"], "steps_taken": sum(h["stepped"] for h in out["history"]),
            "times_the_lifted_base_was_kept": sum(any(c["kept_the_lifted_base"] for c in h["critics"]) for h in out["history"]),
            "training_gap_first": round(out["history"][0]["signed_gap"], 6),
            "training_gap_last": round(out["history"][-1]["signed_gap"], 6),
            "select_gap_by_checkpoint_training_critics": select,
            "fresh_critics_on_the_check_fold": fresh,
            "fresh_gap_fell": fresh["after"]["gap"] < fresh["before"]["gap"],
            "curve": curve}


def run(out_path: Path = OUT) -> int:
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; P1 is write-once")
    import run_family_v2 as rf
    t0 = time.time()
    report = {"artifact": "direct_balance_p1_checks", "generated_utc": rf.now(), "seed": P1_SEED,
              "n": db.N, "splits": {k: db.split_name(v) for k, v in SPLITS.items()}, "threshold": THRESHOLD,
              "prd_sha256": rf.sha256(PRD), "source_sha256": {f: rf.sha256(HERE / f) for f in SOURCES},
              "settings": dn.TRAIN, "checks": {}}

    print("A. positive control: an untrained representation on S1", flush=True)
    pos = control(SPLITS["S1"], shuffle=False, seed=P1_SEED)
    print(f"   {pos['independent_fold']}", flush=True)
    print("B. negative control: the same, with the treatment shuffled", flush=True)
    neg = control(SPLITS["S1"], shuffle=True, seed=P1_SEED)
    print(f"   {neg['independent_fold']}", flush=True)
    print("C, D. one update on S1: where the gradient lands and what moves", flush=True)
    grad = gradient_and_movement(SPLITS["S1"], P1_SEED)
    print(f"   gap {grad['signed_gap']} stepped {grad['stepped']} | reaches the first conv "
          f"{grad['grad_reaches_first_conv']} | it moved {grad['first_conv_moved']}", flush=True)
    curves = {}
    for name, split in SPLITS.items():
        print(f"E. twenty iterations on {name} ({'balanceable' if name == 'S1' else 'not balanceable'})", flush=True)
        curves[name] = learning(split, P1_SEED)
        c = curves[name]
        f = c["fresh_critics_on_the_check_fold"]
        print(f"   training gap {c['training_gap_first']:.2e} -> {c['training_gap_last']:.2e} | steps "
              f"{c['steps_taken']} | lifted base kept {c['times_the_lifted_base_was_kept']} times | FRESH critics: "
              f"gap {f['before']['gap']:+.2e} -> {f['after']['gap']:+.2e}, upper {f['before']['upper']:.2e} -> "
              f"{f['after']['upper']:.2e} | {c['seconds']}s", flush=True)

    report["checks"] = {"positive_control": pos, "negative_control": neg, "gradient_and_movement": grad,
                        "learning": curves}
    gate = {
        "critic_reads_the_positive_control": pos["independent_fold"]["signed_gap"] >= THRESHOLD,
        "false_signal_is_small": neg["independent_fold"]["clipped_gap"] <= THRESHOLD,
        "positive_control_clears_the_false_signal": (pos["independent_fold"]["signed_gap"]
                                                     > 5 * max(neg["independent_fold"]["clipped_gap"], 1e-9)),
        "gradient_reaches_the_first_convolution": grad["grad_reaches_first_conv"],
        "those_weights_move": grad["first_conv_moved"],
        "fresh_critic_gap_falls_on_the_balanceable_split": curves["S1"]["fresh_gap_fell"]}
    report["gate"] = gate
    report["blocking"] = ["critic_reads_the_positive_control", "gradient_reaches_the_first_convolution"]
    report["pass"] = all(gate.values())
    report["blocking_pass"] = all(gate[k] for k in report["blocking"])
    report["gpu_minutes"] = round((time.time() - t0) / 60, 1)
    report["supersedes"] = ("p1_checks_v1.json, whose learning check could not fail (it compared the chosen "
                            "checkpoint with checkpoint zero on the fold that chose it) and whose critics saw h "
                            "without X.  v1 is kept on disk as the record of what was run.")
    report["contrast_note"] = ("S1 is balanceable in SCM-1 and S5 is not, so S5's gap is expected to stay above "
                               "S1's.  One data set and one seed: this is a check on the code, not evidence "
                               "about the method.")
    report["device"] = str(dn.device())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(f"\ngate: {json.dumps(gate, indent=1)}")
    print(f"{'PASS' if report['pass'] else 'FAIL ' + str([k for k, v in gate.items() if not v])} | "
          f"blocking {'ok' if report['blocking_pass'] else 'NOT ok'} | {report['gpu_minutes']} GPU minutes")
    print(f"written {out_path}")
    return 0 if report["blocking_pass"] else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "run":
        raise SystemExit("usage: direct_balance_p1.py run")
    raise SystemExit(run(Path(sys.argv[2]) if len(sys.argv) > 2 else OUT))
