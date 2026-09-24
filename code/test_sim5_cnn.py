"""Generator tests for the Sim5 CNN PRD, section 14.1.  Model, sampling and theorem-ledger tests arrive
with the files they test.  Seeds here are unit-test seeds, not data seeds."""
import hashlib

import numpy as np
import pytest

import sim5_cnn_dgp as g

TEST_SEED = 11


def test_bank_has_twenty_exemplars_per_digit_in_source_order():
    b = g.bank()
    raw = np.load(g.MNIST_PATH)
    assert b["templates"].shape == (200, 28, 28)
    assert np.array_equal(np.bincount(b["digit"]), np.full(10, 20))
    for d in range(10):
        idx = b["source_index"][b["digit"] == d]
        assert np.array_equal(idx, np.flatnonzero(raw["y_train"] == d)[:20])
    assert len(set(b["template_sha256"])) == 200
    assert hashlib.sha256(g.MNIST_PATH.read_bytes()).hexdigest() == g.MNIST_SHA256


def test_severity_style_and_effect_are_centred():
    st = g.states()
    assert abs(st["r"].mean()) < 1e-12 and abs(st["s"].mean()) < 1e-12
    assert abs(st["tau_u"].mean() - 1.0) < 1e-12
    assert np.abs(st["r"]).max() == pytest.approx(1.0) and np.abs(st["s"]).max() == pytest.approx(1.0)


def test_masks_are_disjoint_and_cover_every_pixel_once():
    m = g.block_masks()
    assert m.shape == (64, 28, 28)
    assert np.array_equal(m.sum(axis=0), np.ones((28, 28), dtype=int))
    sides = {int(m[k].any(axis=1).sum()) for k in range(64)} | {int(m[k].any(axis=0).sum()) for k in range(64)}
    assert sides <= {3, 4}


def test_propensity_overlap_and_learner_view():
    data = g.generate(4000, TEST_SEED)
    assert data["p_eval_only"].min() >= 0.1 and data["p_eval_only"].max() <= 0.9
    view = g.learner_view(data)
    assert set(view) == {"X", "W", "A", "Y"}
    assert view["W"].shape == (4000, 28, 28) and view["X"].shape == (4000, 4)
    assert not any(k in view for k in g.LATENT_KEYS)


def test_pixel_noise_is_bounded_and_background_is_state_free():
    data = g.generate(3000, TEST_SEED)
    tpl = g.bank()["templates"][data["U_eval_only"]]
    xi = (data["W"].astype(np.float64) - 0.1) / 0.8 - tpl
    assert np.abs(xi).max() <= g.NOISE_HALF_WIDTH + 1e-6
    common_zero = (g.bank()["templates"] == 0).all(axis=0)
    bg = data["W"][:, common_zero]
    assert bg.min() >= 0.06 - 1e-6 and bg.max() <= 0.14 + 1e-6
    low, high = data["D_eval_only"] < 5, data["D_eval_only"] >= 5
    assert abs(bg[low].mean() - bg[high].mean()) < 5e-4


def test_seed_replay_is_bitwise_identical_and_images_do_not_shift_other_streams():
    a, b = g.generate(500, TEST_SEED), g.generate(500, TEST_SEED)
    assert all(np.array_equal(a[k], b[k]) for k in ("X", "W", "A", "Y", "U_eval_only"))
    c = g.generate(500, TEST_SEED, images=False)
    assert c["W"] is None and all(np.array_equal(a[k], c[k]) for k in ("X", "A", "Y", "U_eval_only"))


def test_population_values_match_the_prd():
    pop = g.population_summary(order=16)
    assert pop["E_tau_u"] == pytest.approx(1.0, abs=1e-12)
    assert pop["naive"] == pytest.approx(-0.5, abs=1e-12)
    assert pop["c_solved"] == pytest.approx(g.C_OUTCOME, abs=1e-4)
    assert pop["theta_X"] == pytest.approx(-0.53656, abs=1e-3)


def test_generator_matches_population_naive_contrast():
    data = g.generate(400_000, TEST_SEED, images=False)
    a, y = data["A"], data["Y"]
    naive = y[a == 1].mean() - y[a == 0].mean()
    assert naive == pytest.approx(-0.5, abs=0.05)
    assert data["tau_u_eval_only"].mean() == pytest.approx(1.0, abs=0.005)


# ----------------------------------------------------------------------------- sampling, honesty, aggregation
import torch

import sim5_cnn_algorithm1 as a1
import sim5_cnn_models as nm
import run_sim5_cnn as rs


def test_split_draw_is_unique_nonempty_proper_and_replayable():
    words = a1.draw_splits(128, 12345)
    assert len(words) == len(set(words)) == 128
    assert all(0 < w < a1.FULL for w in words)
    assert words == a1.draw_splits(128, 12345)
    sizes = [len(a1.held_out_blocks(w)) for w in words]
    assert 20 < np.mean(sizes) < 44                     # sanity only, not a uniformity certificate


def test_keep_mask_complements_the_held_out_blocks():
    masks = g.block_masks()
    w = a1.draw_splits(1, 7)[0]
    keep = a1.keep_mask(w, masks)
    held = np.zeros_like(keep)
    for k in a1.held_out_blocks(w):
        held |= masks[k]
    assert np.array_equal(keep, ~held)


def test_row_roles_are_disjoint_and_cover_the_sample():
    r = rs.roles({"n_D": 60, "n_N": 50, "n_E": 90}, 3)
    assert len(set(r["D"]) | set(r["N"]) | set(r["E"])) == 200
    assert set(r["D_fit"]) | set(r["D_val"]) == set(r["D"]) and not set(r["D_fit"]) & set(r["D_val"])
    assert set(r["N_fit"]) | set(r["N_val"]) == set(r["N"]) and not set(r["N_fit"]) & set(r["N_val"])


def test_representation_ignores_held_out_pixels():
    torch.manual_seed(0)
    trunk, head = nm.Trunk(), nm.Head()
    masks = g.block_masks()
    w_word = a1.draw_splits(1, 99)[0]
    keep = torch.from_numpy(a1.keep_mask(w_word, masks))
    w = torch.rand(8, 28, 28); x = torch.randn(8, 4)
    w2 = w.clone(); w2[:, ~keep] = torch.rand(int((~keep).sum()))
    with torch.no_grad():
        z1, z2 = head(trunk(w, keep), x), head(trunk(w2, keep), x)
    assert torch.equal(z1, z2)


def test_lifted_augmented_critic_reproduces_the_base():
    torch.manual_seed(1)
    z, h = torch.randn(50, 20), torch.randn(50, 30)
    base = nm.BaseCritic(z)
    aug = nm.AugCritic(z, h).lift(base)
    with torch.no_grad():
        assert torch.allclose(base(z), aug(z, h))


def test_aggregation_edges():
    out = a1.aggregate(["a", "b"], [0.0, 0.2], 0.1)                  # distance exactly 2 rho links
    assert out["return_kind"] == "unique_largest" and out["component_sizes"] == [2]
    out = a1.aggregate(list("abc"), [0.0, 0.15, 0.3], 0.08)           # chain a-b-c
    assert out["component_sizes"] == [3] and out["output"] == pytest.approx(0.15)
    out = a1.aggregate(list("abcd"), [0.0, 0.01, 1.0, 1.01], 0.05)    # tie
    assert out["return_kind"] == "tied_largest" and out["output"] is None and len(out["candidates"]) == 2
    assert a1.aggregate([], [], 0.1)["return_kind"] == "no_return"
    assert a1.aggregate(["a"], [float("nan")], 0.1)["return_kind"] == "no_return"
    assert a1.aggregate(["a"], [0.3], float("inf"))["return_kind"] == "no_return"


def test_aipw_score_matches_the_definition():
    e, m0, m1 = np.array([0.5, 0.2]), np.array([0.0, 1.0]), np.array([1.0, 2.0])
    a, y = np.array([1.0, 0.0]), np.array([3.0, 0.0])
    want = np.array([1.0 + (3 - 1) / 0.5, 1.0 - (0 - 1) / 0.8])
    assert np.allclose(a1.aipw_scores(e, m0, m1, a, y), want)
