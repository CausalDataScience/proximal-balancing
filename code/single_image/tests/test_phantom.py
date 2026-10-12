import numpy as np
import pytest

from single_image import folds, phantom as ph, regions as rg


def test_determinism_and_shapes():
    cfg = ph.PhantomConfig(n=120, seed=3)
    a, b = ph.generate(cfg), ph.generate(cfg)
    assert a.images.shape == (120, 64, 64) and a.images.dtype == np.float32
    assert a.X.shape == (120, 1) and a.L.shape == (120, 5) and a.counts.shape == (120, 5)
    for k in ("images", "X", "A", "Y", "U", "L", "G", "I", "propensity", "Y0", "Y1", "counts"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    c = ph.generate(ph.PhantomConfig(n=120, seed=4))
    assert not np.array_equal(a.images, c.images)
    assert set(np.unique(a.A)) <= {0.0, 1.0} and set(np.unique(a.I)) <= {0.0, 1.0}
    assert np.allclose(a.Y1 - a.Y0, 1.0)
    assert np.allclose(a.Y, np.where(a.A > 0.5, a.Y1, a.Y0))


def test_latent_equations():
    d = ph.generate(ph.PhantomConfig(n=20000, seed=1, sigma=0.85))
    resid = d.L - d.U[:, None]
    assert abs(resid.std() - 0.85) < 0.02
    assert abs(d.I.mean() - 0.3) < 0.02
    lin = 1.5 * d.U + 1.0 * d.I + 0.2 * d.X[:, 0]
    assert np.allclose(d.propensity, __import__("scipy").special.ndtr(lin))


def test_masks_are_a_partition_with_five_regions():
    lab = rg.label_map()
    assert lab.shape == (64, 64)
    assert set(np.unique(lab).tolist()) == {0, 1, 2, 3, 4, 5}
    m = rg.masks()
    assert m.shape == (5, 64, 64)
    assert np.all(m.sum(0) <= 1.0)                                   # disjoint
    assert np.array_equal(m.sum(0) > 0, lab > 0)                     # union = lungs
    assert all(m[k].sum() >= 150 for k in range(5))                  # every lobe is substantial
    assert np.all(rg.body_mask()[lab > 0])                           # lungs inside the body


def test_adjacency_is_fixed_and_matches_masks():
    assert rg.computed_adjacency() == rg.ADJACENCY == frozenset({(1, 2), (2, 3), (4, 5)})
    assert rg.non_adjacent(1) == (3, 4, 5) and rg.non_adjacent(5) == (1, 2, 3)
    assert all(len(rg.non_adjacent(k)) >= 2 for k in range(1, 6))


def test_device_only_in_region_5_and_only_when_I_is_1():
    cfg = ph.PhantomConfig(n=300, seed=7)
    with_dev = ph.generate(cfg)
    without = ph.generate(ph.with_changes(cfg, device_strength=0.0))
    diff = with_dev.images - without.images
    # every other draw is unchanged when the device is switched off
    assert np.array_equal(with_dev.L, without.L) and np.array_equal(with_dev.A, without.A)
    outside5 = rg.label_map() != rg.DEVICE_REGION
    assert np.all(diff[:, outside5] == 0.0)
    assert np.all(diff[with_dev.I == 0] == 0.0)
    on = with_dev.I == 1
    assert on.sum() > 50
    assert np.all(diff[on].reshape(on.sum(), -1).max(1) > 0.3)      # the line is visible for every I = 1


def test_region_crops_see_only_their_region():
    d = ph.generate(ph.PhantomConfig(n=10, seed=2))
    for k in range(1, 6):
        c = rg.region_crops(d.images, k)
        assert c.shape[1:] == (2, *rg.crop_size())
        assert np.isclose(c[:, 1].sum(), 10 * rg.masks()[k - 1].sum())
        assert np.all(c[:, 0][c[:, 1] == 0] == 0.0)
    v = rg.holdout_view(d.images, (5,))
    assert np.all(v[:, 0][:, rg.label_map() == 5] == 0.0) and np.all(v[:, 1][:, rg.label_map() == 5] == 0.0)


def test_nonlinear_variant_flag():
    g = ph.generate(ph.PhantomConfig(n=200, seed=5))
    nl = ph.generate(ph.PhantomConfig(n=200, seed=5, variant="nonlinear"))
    assert np.array_equal(g.U, nl.U) and np.array_equal(g.A, nl.A)
    assert np.allclose(nl.Y0 - g.Y0, 0.5 * g.U ** 2)
    assert not np.array_equal(g.images, nl.images)
    with pytest.raises(ValueError):
        ph.PhantomConfig(n=10, variant="other")


def test_folds():
    g = folds.phase1_groups(2000, 0)
    assert len(g["R"]) == 500 and len(np.intersect1d(g["R"], g["rest"])) == 0
    assert np.array_equal(np.sort(np.concatenate([g["R"], g["rest"]])), np.arange(2000))
    s = folds.six_groups(2000, 0)
    assert np.array_equal(s["R"], g["R"])
    allg = np.concatenate(s["groups"])
    assert len(np.unique(allg)) == 1500 and np.array_equal(np.sort(allg), g["rest"])
    r0, r1 = folds.roles(s["groups"], 0), folds.roles(s["groups"], 1)
    assert np.array_equal(r0["C"], r1["D"])


def test_image_only_contract_removes_causes_and_feature_without_changing_W():
    legacy_cfg = ph.PhantomConfig(n=128, seed=19)
    legacy = ph.generate(legacy_cfg)
    cfg = ph.with_changes(legacy_cfg, image_only=True)
    image_only = ph.generate(cfg)
    explicit_zero = ph.generate(ph.with_changes(legacy_cfg, treat_x=0., out_x=0.))
    assert cfg.treat_x == cfg.out_x == 0.
    assert image_only.X.shape == (128, 0)
    assert np.array_equal(image_only.images, legacy.images)
    for key in ("A", "Y", "Y0", "Y1", "propensity"):
        assert np.array_equal(getattr(image_only, key), getattr(explicit_zero, key)), key
    assert np.allclose(image_only.propensity, __import__("scipy").special.ndtr(1.5 * image_only.U + image_only.I))
