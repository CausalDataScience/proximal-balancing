"""Does the parallel r search reproduce the serial one exactly, and how much faster is it?"""
import time, numpy as np, torch
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
    assert enc["fail_closed"] is None, enc["fail_closed"]
    k_all = im.apply_encoder(enc, lat, np.arange(12000))
    splits = c.oriented_splits()
    t = time.time(); par = r7.run_selection(splits, phi_idx, s_idx, cache, k_all, 95000000, 0, workers=8); tp = time.time() - t
    print(f"parallel(8) {tp:.0f}s", flush=True)
    t = time.time(); serial = r7.run_selection(splits, phi_idx, s_idx, cache, k_all, 95000000, 0, workers=1); ts = time.time() - t
    same_r = sum(serial[s][1]["r"] == par[s][1]["r"] for s in splits)
    same_pen = sum(serial[s][0] == par[s][0] for s in splits)
    max_obj = max(abs(serial[s][1]["objective"] - par[s][1]["objective"]) for s in splits)
    max_d2 = max(abs(serial[s][1]["screen_d2"] - par[s][1]["screen_d2"]) for s in splits)
    print(f"serial {ts:.0f}s | parallel(8) {tp:.0f}s | speedup {ts/tp:.1f}x")
    print(f"identical r: {same_r}/30 | identical penalty: {same_pen}/30 | max |objective diff| {max_obj:.2e} | max |d2 diff| {max_d2:.2e}")


if __name__ == "__main__":
    main()
