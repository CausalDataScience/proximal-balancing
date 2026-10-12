"""Stage 1 (plan section 6): everything learned from fold-R images alone.

    summaries   FixedSummaries fitted on R and frozen (section 6.3)
    readers     psi_1..psi_5 trained on R (section 6.4), sufficiency check, d_psi raised to 16 if it fails
    factor      tetrad / one-factor diagnostic on R and Omega_k
    mae         masked autoencoder on R (section 6.2)

`run_stage1` takes the fold-R image array and nothing else.  There is no argument through which a
treatment, an outcome or a latent could enter, which is the construction the tests check.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import mae as mae_mod
from . import reader as rd
from .summaries import FixedSummaries


def run_stage1(images_r: np.ndarray, seed: int, device: torch.device, reader_epochs: int = 60,
               mae_epochs: int = 40, n_boot: int = 500, auto_raise: bool = True, train_mae: bool = True,
               log=print) -> dict:
    if not isinstance(images_r, np.ndarray) or images_r.ndim != 3:
        raise TypeError("run_stage1 takes the (n, 64, 64) fold-R image array and nothing else")
    timing = {}
    t0 = time.time()
    summ = FixedSummaries().fit(images_r)
    timing["summaries_fit"] = round(time.time() - t0, 2)
    log(f"  summaries fitted on R ({images_r.shape[0]} images), fingerprint {summ.fingerprint()}")

    t0 = time.time()
    bank = rd.train_readers(images_r, summ, seed, device, d_psi=rd.D_PSI, max_epochs=reader_epochs, log=log)
    timing["readers_train"] = round(time.time() - t0, 1)
    t0 = time.time()
    suff = rd.sufficiency_check(bank, images_r, summ, seed)
    timing["sufficiency_check"] = round(time.time() - t0, 1)
    attempts = [{"d_psi": bank.d_psi, "sufficiency": suff, "train_report": bank.report}]
    log("  sufficiency (loss psi / loss g_512): "
        + ", ".join(f"{k}: {v['ratio_psi_over_g']:.3f}" for k, v in suff.items() if k != "all_accept"))
    if auto_raise and not suff["all_accept"]:
        log("  sufficiency failed for some region: retraining every reader at d_psi = 16")
        t0 = time.time()
        bank = rd.train_readers(images_r, summ, seed, device, d_psi=16, max_epochs=reader_epochs, log=log)
        suff = rd.sufficiency_check(bank, images_r, summ, seed)
        timing["readers_train_d16"] = round(time.time() - t0, 1)
        attempts.append({"d_psi": 16, "sufficiency": suff, "train_report": bank.report})

    t0 = time.time()
    psi_r = bank.transform(images_r, device)
    chi_r = summ.transform(images_r)
    f_r, share = rd.first_pcs(psi_r, [chi_r[:, k, 0] for k in range(5)])
    factor = rd.one_factor_diagnostic(f_r, n_boot=n_boot, seed=seed)
    factor["pc1_variance_share"] = share
    timing["factor_diagnostic"] = round(time.time() - t0, 1)
    log(f"  tetrads within 2 SE: {factor['n_within_2se']}/{factor['n_tetrads']} (max |z| "
        f"{factor['max_abs_z']:.2f}); Omega_FA = {np.round(factor['omega_fa'], 3).tolist()}")

    mae_res = None
    if train_mae:
        t0 = time.time()
        mae_res = mae_mod.train_mae(images_r, seed, device, epochs=mae_epochs, log=log)
        timing["mae_train"] = round(time.time() - t0, 1)
        log(f"  MAE val hidden-pixel MSE {mae_res['val_masked_mse']:.4f} (mean image "
            f"{mae_res['reference_mean_image_mse']:.4f}, zero {mae_res['reference_zero_mse']:.4f})")
    return {"summaries": summ, "bank": bank, "attempts": attempts, "factor_R": factor, "mae": mae_res,
            "timing_s": timing, "n_R": int(images_r.shape[0])}
