"""Does the encoder train on the accelerator now, and does it land where the CPU run landed?"""
import time, numpy as np, torch
import minimal_suite_common as c, probe_image_w as im


def main():
    torch.set_num_threads(4)
    lat, = [im.latents(3000, 95000000)]
    fit, cal, ev = np.arange(1200), np.arange(1200, 1500), np.arange(1500, 3000)
    print("device chosen:", im._device())
    t = time.time(); enc = im.fit_encoder(lat, fit, cal, seed=95000000); dt = time.time() - t
    assert enc["fail_closed"] is None, enc["fail_closed"]
    dev = enc["model"].parameters()[0].device
    corr = im.channel_correlation(enc, lat, ev)
    print(f"trained on {dev} in {dt:.0f}s | stopped at epoch {enc['stopped_at_epoch']} | "
          f"channel corr median {np.median(corr):.4f} per block {[round(x, 3) for x in corr]}")
    k = im.apply_encoder(enc, lat, ev)
    print("apply_encoder shape", k.shape, "| finite:", np.isfinite(k).all())
    print("reference: the CPU qualification run on the same seed gave channel corr 0.947 (n=12000 rows);"
          " a 3,000-row fixture will differ somewhat but must be in the same range")


if __name__ == "__main__":
    main()
