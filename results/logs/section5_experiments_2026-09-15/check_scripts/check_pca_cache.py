"""Cached per-block PCA against the previous per-split recomputation: identical numbers, less time."""
import time, json, numpy as np, torch
import minimal_suite_common as c, probe_image_w as im, probe_full_target as ft, run_scm7_v3 as r7


def main():
    torch.set_num_threads(2)
    dev = ft.device()
    lat, folds = r7.nested_latents(95000000, 12000)
    cache = r7.SeedCache(lat, dev)
    rot = c.rotations(folds)[0]
    d = rot.roles["D"]; phi_idx, s_idx = d[1500:], rot.roles["S"]
    im.TRAIN.update({"epochs": 3, "early_stopping": False, "scale_min": 1e-12})
    enc = im.fit_encoder(lat, d[:1200], d[1200:1500], seed=95000000)
    assert enc["fail_closed"] is None
    k_all = im.apply_encoder(enc, lat, np.arange(12000))
    splits = c.oriented_splits()
    t = time.time(); par = r7.run_selection(splits, phi_idx, s_idx, cache, k_all, 95000000, 0, workers=8); tp = time.time() - t
    # the previous path recomputed the projection per split; reproduce it for one heavy split and compare
    phi_blocks = r7.s7.Blocks(lat, phi_idx)
    ref = {}
    for split in [(1,), (1, 2), (1, 2, 3, 4)]:
        proj = r7.s7.fit_image_projection(phi_blocks, split, r7.s7.TARGET_COMPONENTS)
        oseed = c.optimizer_seed(r7.FROZEN["protocol_id"], 95000000, 0, c.split_id(split), 0)
        num_phi = cache.rows(phi_idx)
        z_ref = r7.hd.representation(k_all[phi_idx], num_phi, split, 0.0)
        pen = r7.hd.choose_penalty(lambda: r7.hd.Scaler(z_ref)(z_ref), num_phi["A"], r7.s7.PENALTY_GRID, oseed)
        sel = r7.s7.gap_and_screen(phi_blocks, k_all[phi_idx], r7.s7.Blocks(lat, s_idx), k_all[s_idx], split, proj, pen, oseed)
        ref[split] = (pen, sel, proj["digest"])
    same = all(par[s][1]["r"] == ref[s][1]["r"] and par[s][0] == ref[s][0]
               and par[s][1]["objective"] == ref[s][1]["objective"]
               and par[s][1]["target_digest"] == ref[s][2] for s in ref)
    print(f"cached-PCA parallel(8): {tp:.0f}s for 30 splits | identical to per-split recomputation on 3 probes: {same}")
    for s in ref:
        print(f"  {c.split_id(s):6s} r {par[s][1]['r']} vs {ref[s][1]['r']} | obj diff {par[s][1]['objective'] - ref[s][1]['objective']:.1e} | digest same {par[s][1]['target_digest'] == ref[s][2]}")


if __name__ == "__main__":
    main()
