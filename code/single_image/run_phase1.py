"""Phase 1 of the single-image PROBE plan: phantom, stage 1 on fold R, level A and comparators on the rest.

    cd research/papers/single_proxy_balancing/code
    PYTHONDONTWRITEBYTECODE=1 nice -n 10 python3 -B -m single_image.run_phase1 --n 24000 --sigma 0.85 --seed 0

Writes results/single_image/phase1/phase1_<variant>_n<n>_sigma<sigma>_seed<seed>.json and the reader and
masked-autoencoder weights under results/single_image/phase1/weights/.  Refuses to overwrite unless
--overwrite.  --quick shortens every training loop (tests and smoke checks only).
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "2"

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
PROJECT = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import torch  # noqa: E402

from single_image import folds, level_a, mae as mae_mod, nets, oracle, phantom as ph  # noqa: E402
from single_image import reader as rd, regions as rg, stage1  # noqa: E402

DEFAULT_OUT = PROJECT / "results" / "single_image" / "phase1"
PLAN = PROJECT / "memo" / "2026-10-08-single-image-probe-plan-v1.md"


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage2_cost(device: torch.device, n_d: int, n_c: int, log=print) -> dict:
    """Micro-benchmark of the stage-2 building blocks on this machine and the resulting cost estimate.

    Assumptions (plan section 7): per split and per rotation, 30 rounds; a round = A-step (q_1 ladder
    level 3, a crop CNN with Z appended, about 10 epochs over D with early stopping; q_0 is a tiny MLP on
    Z and is ignored) + B-step (50 encoder steps at batch 256) + encoder forward passes over D and C to
    refresh Z and the risks.  Five singleton splits, five rotations of D, C, P, S, E."""
    nets.seed_all(0)
    z = torch.randn(256, 2, 64, 64, device=device)
    enc = nets.RegionEncoder(nets.ConvTrunk(), d_out=2).to(device)
    opt = torch.optim.Adam(enc.parameters(), 1e-4)

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()

    for _ in range(3):
        loss = (enc(z) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    sync(); t0 = time.time()
    for _ in range(10):
        loss = (enc(z) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    sync(); step_enc = (time.time() - t0) / 10
    with torch.no_grad():
        zz = torch.randn(1024, 2, 64, 64, device=device)
        enc(zz); sync(); t0 = time.time()
        for _ in range(3):
            enc(zz)
        sync(); fwd_per_img = (time.time() - t0) / (3 * 1024)
    crop = torch.randn(128, 2, *rg.crop_size(), device=device)
    net = rd.RegionReaderNet(8).to(device)
    head = torch.nn.Linear(8 + 2, 1).to(device)
    zc = torch.randn(128, 2, device=device)
    opt2 = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), 1e-3)
    for _ in range(3):
        loss = head(torch.cat([net(crop), zc], 1)).pow(2).mean(); opt2.zero_grad(); loss.backward(); opt2.step()
    sync(); t0 = time.time()
    for _ in range(20):
        loss = head(torch.cat([net(crop), zc], 1)).pow(2).mean(); opt2.zero_grad(); loss.backward(); opt2.step()
    sync(); step_q1 = (time.time() - t0) / 20
    a_step = 10 * int(np.ceil(n_d / 128)) * step_q1
    b_step = 50 * step_enc
    refresh = (n_d + n_c) * fwd_per_img
    per_round = a_step + b_step + refresh
    per_split_rotation = 30 * per_round
    per_dataset = 5 * 5 * per_split_rotation
    out = {"device": str(device), "encoder_train_step_b256_s": step_enc, "encoder_forward_per_image_s": fwd_per_img,
           "q1_crop_cnn_step_b128_s": step_q1, "n_D": n_d, "n_C": n_c,
           "round_s": {"A_step": a_step, "B_step": b_step, "refresh_Z": refresh, "total": per_round},
           "per_split_per_rotation_30_rounds_min": per_split_rotation / 60,
           "per_dataset_5_splits_5_rotations_h": per_dataset / 3600,
           "full_grid_4_sigma_x_20_reps_h_single_process": 80 * per_dataset / 3600,
           "parameters_encoder": nets.count_parameters(enc)}
    log(f"  stage-2 estimate: {per_round:.1f} s/round, {per_split_rotation / 60:.1f} min per split-rotation, "
        f"{per_dataset / 3600:.2f} h per data set, {80 * per_dataset / 3600:.0f} h for 4 sigma x 20 reps")
    return out


def run(args) -> dict:
    if getattr(args, "diagnostic_only", False):
        return run_diagnostic(args)
    try:
        os.nice(10)
    except OSError:
        pass
    nets.setup_torch(2)
    device = nets.pick_device(args.device)
    log = print
    t_all = time.time()
    timing = {}
    cfg = ph.PhantomConfig(n=args.n, sigma=args.sigma, seed=args.seed, variant=args.variant,
                          image_only=getattr(args, "image_only", False))
    stem = f"phase1_{cfg.variant}_n{cfg.n}_sigma{cfg.sigma:g}_seed{cfg.seed}" + \
           ("_image_only" if cfg.image_only else "") + (f"_{args.tag}" if args.tag else "")
    out_dir = Path(args.out_dir)
    out_path = out_dir / f"{stem}.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; refuse before training or saving weights")
    log(f"phase 1: n={args.n} sigma={args.sigma} seed={args.seed} variant={args.variant} device={device}")

    t0 = time.time()
    data = ph.generate(cfg)
    timing["generate"] = round(time.time() - t0, 2)
    groups = folds.phase1_groups(cfg.n, cfg.seed)
    R, rest = groups["R"], groups["rest"]
    log(f"  generated {cfg.n} images in {timing['generate']} s; fold R {len(R)}, rest {len(rest)}")

    quick = args.quick
    s1 = stage1.run_stage1(data.images[R], seed=cfg.seed, device=device,
                           reader_epochs=4 if quick else args.reader_epochs,
                           mae_epochs=2 if quick else args.mae_epochs,
                           n_boot=50 if quick else args.n_boot, log=log)
    timing["stage1"] = s1["timing_s"]
    bank = s1["bank"]

    t0 = time.time()
    psi_rest = bank.transform(data.images[rest], device)
    timing["readers_apply_rest"] = round(time.time() - t0, 1)
    d = data.subset(rest)

    t0 = time.time()
    evald = rd.evaluation_diagnostics(psi_rest, d.U, d.L, d.G, d.I)
    chi_rest = s1["summaries"].transform(d.images)
    f_rest, share_rest = rd.first_pcs(psi_rest, [chi_rest[:, k, 0] for k in range(5)])
    nb = 50 if quick else args.n_boot
    factor_rest = rd.one_factor_diagnostic(f_rest, n_boot=nb, seed=cfg.seed + 1)
    factor_rest["pc1_variance_share"] = share_rest
    factor_true_L = rd.one_factor_diagnostic(d.L, n_boot=nb, seed=cfg.seed + 2)
    # supplementary: the same diagnostic on each reader's second principal component
    f2_rest, share2 = rd.first_pcs(psi_rest, [chi_rest[:, k, 0] for k in range(5)], component=1)
    factor_rest_pc2 = rd.one_factor_diagnostic(f2_rest, n_boot=nb, seed=cfg.seed + 3)
    factor_rest_pc2["pc_variance_share"] = share2

    def corr(f):
        return {"U": [float(np.corrcoef(f[:, k], d.U)[0, 1]) for k in range(5)],
                "G": [float(np.corrcoef(f[:, k], d.G)[0, 1]) for k in range(5)],
                "L_k": [float(np.corrcoef(f[:, k], d.L[:, k])[0, 1]) for k in range(5)]}
    pc1_corr = corr(f_rest)
    factor_rest_pc2["correlations_eval_only"] = corr(f2_rest)
    timing["diagnostics_rest"] = round(time.time() - t0, 1)
    log("  reader R^2(L_k): " + ", ".join(f"{evald['per_region'][str(k)]['r2_L_k']:.3f}" for k in range(1, 6))
        + f" | R^2(U | all psi) {evald['r2_U_all_psi']:.3f} (all true L {evald['r2_U_all_true_L']:.3f})"
        + f" | AUC(I | psi_5) {evald['per_region']['5']['auc_I']:.3f}")

    pseed = level_a.probe_seed(cfg.seed)
    views = {"level_A_readers": [psi_rest[k] for k in range(5)],
             "oracle_reader_L": [d.L[:, [k]] for k in range(5)]}
    if not args.no_oracle_device:
        views["oracle_reader_L_device_in_W5"] = [d.L[:, [k]] for k in range(4)] + [np.column_stack([d.L[:, 4], d.I])]
    probes, comps = {}, {}
    for nm, blocks in views.items():
        view = level_a.make_view(d.X, blocks, d.A, d.Y)
        t0 = time.time()
        probes[nm] = level_a.run_probe(view, pseed, f"SINGLE_IMAGE_{nm.upper()}")
        timing[f"probe_{nm}"] = round(time.time() - t0, 1)
        comps[nm] = level_a.comparators(view, d.U, pseed)
        timing[f"comparators_{nm}"] = comps[nm]["seconds"]
        p = probes[nm]
        log(f"  PROBE[{nm}] {timing[f'probe_{nm}']} s: v3 {p['v3']['estimate']} ({p['v3']['n_retained']} kept, "
            f"{p['v3']['n_invalid_kept']} invalid) sealed {p['sealed']['estimate']} | naive {comps[nm]['naive']:+.3f} "
            f"X {comps[nm]['aipw_X']:+.3f} X+blocks {comps[nm]['aipw_X_blocks']:+.3f} X+U {comps[nm]['aipw_X_U']:+.3f}")

    t0 = time.time()
    pop = oracle.population(cfg, n_mc=100_000 if quick else args.n_mc)
    timing["oracle_population"] = round(time.time() - t0, 1)

    t0 = time.time()
    n_grp = int(round((cfg.n - len(R)) / 5))
    cost = stage2_cost(device, n_d=n_grp, n_c=n_grp, log=log)
    timing["stage2_benchmark"] = round(time.time() - t0, 1)

    weights = {}
    if not args.no_weights:
        wdir = out_dir / "weights"
        rp, mp = wdir / f"{stem}_readers.pt", wdir / f"{stem}_mae.pt"
        if not args.overwrite and (rp.exists() or (s1["mae"] is not None and mp.exists())):
            raise SystemExit("checkpoint exists; refuse to replace original weights")
        meta = {"config": ph.config_dict(cfg), "fold_R_size": int(len(R)), "stem": stem}
        bank.save(rp, meta)
        if s1["mae"] is not None:
            mae_mod.save(mp, s1["mae"], meta)
        weights = {"readers": str(rp.relative_to(PROJECT)) if rp.is_relative_to(PROJECT) else str(rp),
                   "mae": str(mp.relative_to(PROJECT)) if mp.is_relative_to(PROJECT) else str(mp)}

    mae_res = s1["mae"]
    timing["total"] = round(time.time() - t_all, 1)
    tau = cfg.tau
    table = {"tau": tau}
    for nm in views:
        table[f"probe_v3[{nm}]"] = probes[nm]["v3"]["estimate"]
        table[f"probe_sealed[{nm}]"] = probes[nm]["sealed"]["estimate"]
    for k in ("naive", "aipw_X", "aipw_X_U"):
        table[k] = comps["level_A_readers"][k]
    table["aipw_X_readers"] = comps["level_A_readers"]["aipw_X_blocks"]
    table["aipw_X_true_L"] = comps["oracle_reader_L"]["aipw_X_blocks"]

    result = {
        "artifact": "single_image_phase1", "generated_unix": time.time(), "plan": str(PLAN.relative_to(PROJECT)),
        "plan_sha256": _sha(PLAN), "config": ph.config_dict(cfg), "quick": quick,
        "feature_mode": "image_only" if cfg.image_only else "legacy_X",
        "scope": "Stage1 plus fixed-reader rank-2 baseline; Stage2 alternating encoder not run",
        "threshold_reanalysis": {nm: {str(t): level_a.rethreshold(probes[nm], t)
                                      for t in (.005, .01, .02)} for nm in probes},
        "machine": {"platform": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__,
                    "device": str(device), "threads": torch.get_num_threads()},
        "folds": {"R": int(len(R)), "rest": int(len(rest)), "rule": "folds.phase1_groups (25% R); PROBE's own "
                  "four rotating folds inside the rest"},
        "regions": {"names": rg.REGION_NAMES, "pixels": [int((rg.label_map() == k).sum()) for k in range(1, 6)],
                    "adjacency": sorted(rg.ADJACENCY), "non_adjacent": {k: rg.non_adjacent(k) for k in range(1, 6)},
                    "device_region": rg.DEVICE_REGION, "crop_size": rg.crop_size()},
        "data_summary": {"P_A": float(data.A.mean()), "P_I": float(data.I.mean()),
                         "mean_blob_count": data.counts.mean(0).tolist(),
                         "capped_count_share": float((data.counts == ph.BMAX).mean()),
                         "true_tau": tau, "sample_ate_rest": float((d.Y1 - d.Y0).mean())},
        "estimates": table,
        "errors_vs_tau": {k: (None if v is None else v - tau) for k, v in table.items() if k != "tau"},
        "stage1": {"summaries": {"fingerprint": s1["summaries"].fingerprint(), "n_fit": s1["summaries"].n_fit,
                                 "thresholds": s1["summaries"].thresholds.tolist()},
                   "reader_attempts": s1["attempts"], "d_psi_used": bank.d_psi,
                   "factor_R": s1["factor_R"],
                   "mae": None if mae_res is None else {k: v for k, v in mae_res.items() if k != "model"}},
        "reader_eval_rest": {**evald, "pc1_correlations_eval_only": pc1_corr},
        "factor_rest_psi": factor_rest, "factor_rest_psi_pc2": factor_rest_pc2, "factor_rest_true_L": factor_true_L,
        "omega_truth": 1.0 / cfg.sigma ** 2,
        "level_A": probes, "comparators": comps, "oracle_population": pop,
        "stage2_cost_estimate": cost, "weights": weights, "timing_s": timing,
        "sealed_source_sha256": level_a.sealed_hashes(),
        "source_sha256": {p.name: _sha(p) for p in sorted(HERE.glob("*.py"))},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace it")
    out_path.write_text(json.dumps(result, indent=1, default=_json_default, allow_nan=False) + "\n")
    log(f"  written {out_path} (total {timing['total']} s)")
    log("  estimates vs tau = 1: " + json.dumps({k: (None if v is None else round(v, 4)) for k, v in table.items()}))
    return result


def run_diagnostic(args) -> dict:
    """Preserve the old experiment; inspect it and run one new image-only locked-Z audit.

    Reuse its A/Y-free reader checkpoint. No checkpoint is saved and no alternating encoder/grid is run.
    Oracle probabilities below belong only to an explicitly constructed true-L posterior reference.
    """
    output = Path(args.out_dir) / "phase1_diagnostic_v2.json"
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; diagnostic refuses to overwrite")
    previous_pilot = None
    if output.exists():
        prior = json.loads(output.read_text())
        previous_pilot = {"sha256": _sha(output), "seconds": prior["seconds"],
                          "source_sha256": prior["source_sha256"],
                          "Z_sha256": prior["fixed_learned_Z_audit"]["Z_sha256"],
                          "q0_prediction_sha256": prior["fixed_learned_Z_audit"]["q0_prediction_sha256"],
                          "q0": prior["fixed_learned_Z_audit"]["q0"],
                          "contrasts": prior["fixed_learned_Z_audit"]["contrasts"],
                          "selected_fits": prior["fixed_learned_Z_audit"]["folds"]}
    started = time.time()
    nets.setup_torch(2)
    if args.device not in ("cpu", "auto"):
        raise ValueError("the small diagnostic is CPU only")
    cfg = ph.PhantomConfig(n=args.n, sigma=args.sigma, seed=args.seed, variant=args.variant, image_only=True)
    legacy_path, reader_path = Path(args.legacy_result), Path(args.reader_checkpoint)
    old = json.loads(legacy_path.read_text())
    original_hashes = {str(legacy_path): _sha(legacy_path), str(reader_path): _sha(reader_path)}
    bank = rd.ReaderBank.load(reader_path)
    meta_cfg = bank.report["checkpoint_meta"].get("config", {})
    for key in ("n", "sigma", "seed", "variant"):
        if meta_cfg.get(key) != getattr(cfg, key):
            raise ValueError(f"checkpoint {key} does not match the generated images/fold R")
    print(f"diagnostic v2: image-only n={cfg.n}, CPU, same stored image-only reader initialization", flush=True)
    data = ph.generate(cfg)
    groups = folds.phase1_groups(cfg.n, cfg.seed)
    rest = groups["rest"]
    d = data.subset(rest)
    psi = bank.transform(d.images, torch.device("cpu"))
    if d.X.shape != (len(rest), 0) or cfg.treat_x != 0 or cfg.out_x != 0:
        raise AssertionError("image-only input and SCM must both omit X")
    view = level_a.make_view(d.X, psi, d.A, d.Y)
    pseed = level_a.probe_seed(cfg.seed)
    locked = level_a.lock_reader_representation(view, pseed, args.diagnostic_rows)
    sr = locked["screen_rows"]
    crops = rg.region_crops(nets.standardise(d.images[sr], bank.stats), 5)
    audit = level_a.fresh_critic_contrast(locked["Z"], locked["reader_T"], d.A[sr], raw_crop=crops,
                                        oracle_i=d.I[sr], threshold=args.balance_threshold, seed=pseed,
                                        epochs=args.critic_epochs, raw_epochs=args.raw_critic_epochs)
    audit["locked_encoder"] = locked["metadata"]
    audit["patient_rows_sha256"] = level_a.array_sha(rest[sr])
    audit["inputs"] = {"q0": "locked retained-image representation Z only",
                       "reader": "Z and heldout region-5 compressed reader",
                       "raw_image": "Z, same reader candidate, and heldout raw region-5 crop/mask",
                       "oracle_I_positive_control": "same reader candidate plus true I; evaluation control"}
    if previous_pilot is not None:
        invariance = {"Z": audit["Z_sha256"] == previous_pilot["Z_sha256"],
                      "q0": audit["q0_prediction_sha256"] == previous_pilot["q0_prediction_sha256"],
                      "reader_q1": audit["contrasts"]["reader"]["q1_prediction_sha256"] ==
                                   previous_pilot["contrasts"]["reader"]["q1_prediction_sha256"],
                      "oracle_I_q1": audit["contrasts"]["oracle_I_positive_control"]["q1_prediction_sha256"] ==
                                     previous_pilot["contrasts"]["oracle_I_positive_control"]["q1_prediction_sha256"]}
        if not all(invariance.values()):
            raise RuntimeError("raw revision changed an allegedly fixed comparison branch")
        audit["previous_pilot_fixed_branch_hash_checks"] = invariance
    print("  learned-Z fresh reader/raw/I contrast complete", flush=True)
    # A separate reference, not the learned rank-2 B: Z=m(L1..L4), T=(L5,I).
    m_keep, v_keep = oracle._posterior(d.L[sr], (0, 1, 2, 3), cfg.sigma)
    m_all, v_all = oracle._posterior(d.L[sr], (0, 1, 2, 3, 4), cfg.sigma)
    x_zero = np.zeros(len(sr))
    e = oracle._p_given_post(m_keep, v_keep, x_zero, None, cfg)
    p = oracle._p_given_post(m_all, v_all, x_zero, d.I[sr], cfg)
    reference = level_a.fresh_critic_contrast(m_keep[:, None], np.column_stack([d.L[sr, 4], d.I[sr]]), d.A[sr],
                                            threshold=args.balance_threshold, seed=pseed + 900,
                                            epochs=args.critic_epochs, reference_e=e, reference_p=p)
    reference["contrasts"]["true_L_and_I"] = reference["contrasts"].pop("reader")
    reference["representation"] = "fixed posterior mean of true L1..L4; no I in Z, T=(L5,true I)"
    reference["oracle_scope"] = "epsilon and D2 apply only to this reference; unavailable for learned reader/raw Z"
    reader_comps = level_a.comparators(view, d.U, pseed)
    l_view = level_a.make_view(d.X, [d.L[:, [k]] for k in range(5)], d.A, d.Y)
    l_comps = level_a.comparators(l_view, d.U, pseed)
    pop = oracle.population(cfg, n_mc=args.diagnostic_mc)
    # Same immutable old JSON, different cutoffs/rules; this remains the old X-input experiment.
    reanalysis = {nm: {rule: {str(t): level_a.rethreshold(rec, t, rule)
                             for t in (.005, .01, .02)} for rule in ("all_rotations", "pooled")}
                  for nm, rec in old["level_A"].items()}
    for path, before in original_hashes.items():
        if _sha(Path(path)) != before:
            raise RuntimeError(f"original artifact changed: {path}")
    result = {"artifact": "single_image_phase1_diagnostic_v2", "generated_unix": time.time(),
              "config": ph.config_dict(cfg), "feature_mode": "image_only", "balance_threshold_D2": args.balance_threshold,
              "scope": "legacy report/screen repair and one new fixed-encoder audit; Stage2 and causal recovery unresolved",
              "development_pilot": True, "scoring_reused_after_architecture_revision": previous_pilot is not None,
              "architecture_revision_reason": "raw CNN failed internal validation; add fixed image-only high-pass features",
              "previous_diagnostic_pilot": previous_pilot,
              "machine": {"device": "cpu", "threads": torch.get_num_threads(), "python": sys.version.split()[0],
                          "torch": torch.__version__, "platform": platform.platform()},
              "protocol": {"R": len(groups["R"]), "rest": len(rest), "split": "S5", "rotation": 0,
                           "critic_epochs_max": args.critic_epochs, "raw_critic_epochs_max": args.raw_critic_epochs,
                           "scoring_checkpoint_selection": False, "new_weights_saved": False,
                           "raw_model_selection": "internal validation only; max epochs from the revision protocol",
                           "raw_feature_config": {"names": list(rd.RAW_TEXTURE_FEATURES),
                                                  "blur_sigma": rd.RAW_TEXTURE_BLUR_SIGMA,
                                                  "excluded_boundary_pixels": rd.RAW_TEXTURE_BOUNDARY_PIXELS},
                           "reader_pretraining": "original A/Y-free fold-R checkpoint, same n/seed/sigma/variant",
                           "image_draws": "legacy W draws preserved; only X causes/input removed"},
              "legacy_reanalysis": {"source": str(legacy_path), "source_sha256": original_hashes[str(legacy_path)],
                                    "feature_mode": "legacy_X", "estimates": old["estimates"],
                                    "oracle_population_targets": old["oracle_population"]["targets"],
                                    "bounded_overlap_certified": False,
                                    "S5_screens": {nm: {"estimate": rec["per_split"]["S5"]["estimate"],
                                                         "gap": rec["per_split"]["S5"]["pooled_gap"],
                                                         "pooled_upper": rec["per_split"]["S5"]["pooled_upper"],
                                                         "max_rotation_upper": max(s["upper"] for s in
                                                                                    rec["per_split"]["S5"]["screen"])}
                                                    for nm, rec in old["level_A"].items()},
                                    "scenario_labels": {"level_A_readers": "actual compressed readers",
                                                        "oracle_reader_L": "true L only, no device input",
                                                        "oracle_reader_L_device_in_W5": "true L plus true I positive control"},
                                    "thresholds": reanalysis},
              "fixed_learned_Z_audit": audit, "true_L_posterior_reference_audit": reference,
              "finite_fitted_AIPW_comparators": {"no_feature": reader_comps["aipw_X"],
                                                "all_image_readers": reader_comps["aipw_X_blocks"],
                                                "all_true_L_control": l_comps["aipw_X_blocks"],
                                                "oracle_U_control": reader_comps["aipw_X_U"],
                                                "naive": reader_comps["naive"],
                                                "scope": "finite legacy bounded-AIPW comparator; not population targets or corrected PROBE output"},
              "oracle_population_reference": pop, "seconds": round(time.time() - started, 2),
              "original_artifacts_unchanged_sha256": original_hashes,
              "sealed_source_sha256": level_a.sealed_hashes(),
              "source_sha256": {p.name: _sha(p) for p in sorted(HERE.glob("*.py"))}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1, default=_json_default, allow_nan=False) + "\n")
    print(f"  written {output}; {result['seconds']} seconds", flush=True)
    return result


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--sigma", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", choices=("gaussian", "nonlinear"), default="gaussian")
    ap.add_argument("--device", default="auto", help="auto (MPS if available, else CPU), cpu or mps")
    ap.add_argument("--reader-epochs", type=int, default=60)
    ap.add_argument("--mae-epochs", type=int, default=40)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-mc", type=int, default=oracle.N_MC)
    ap.add_argument("--no-oracle-device", action="store_true")
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--image-only", action="store_true", help="remove X causes and X predictor input together")
    ap.add_argument("--diagnostic-only", action="store_true", help="CPU locked-Z reader/raw/I contrast; no saved weights")
    ap.add_argument("--balance-threshold", type=float, default=.01)
    ap.add_argument("--critic-epochs", type=int, default=30)
    ap.add_argument("--raw-critic-epochs", type=int, default=60)
    ap.add_argument("--diagnostic-rows", type=int, default=4000)
    ap.add_argument("--diagnostic-mc", type=int, default=20000)
    ap.add_argument("--legacy-result", default=str(DEFAULT_OUT / "phase1_gaussian_n24000_sigma0.85_seed0.json"))
    ap.add_argument("--reader-checkpoint", default=str(DEFAULT_OUT / "weights" /
                                                     "phase1_gaussian_n24000_sigma0.85_seed0_readers.pt"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    run(parse(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
