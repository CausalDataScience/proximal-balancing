"""Checks for the WSC design used by wsc_probe.py."""
from __future__ import annotations

import pytest

import probe_j2 as pj
import wsc_probe as ws


@pytest.fixture(scope="module")
def loaded():
    return ws.load_view()


def test_the_august_design_is_reused_exactly(loaded):
    view, meta = loaded
    assert (meta["n"], meta["treated"]) == (1_105, 288)
    assert view["X"].shape == (1_105, 14)
    assert meta["rct_benchmark"] == pytest.approx(0.760, abs=0.0005)


def test_the_seven_blocks_follow_the_instruments(loaded):
    view, meta = loaded
    assert list(meta["block_sizes"]) == ws.BLOCK_ORDER
    assert meta["block_sizes"] == {"big5": 44, "amas": 9, "bdi": 13, "gses": 6, "math_pre": 12,
                                   "vocab_pre": 24, "mcs_pre": 6}
    assert sum(view[f"W{j + 1}"].shape[1] for j in range(7)) == 114


def test_seven_blocks_give_126_candidates():
    assert len(pj.all_splits(7)) == 126


def test_no_post_treatment_item_enters_a_block(loaded):
    import wsc_population_ate_pilot as wp
    names = wp.load_data()["proxy_names"]
    assert not any("post" in n.lower() for n in names)


def test_the_exact_answer_weights_the_two_preference_groups():
    import wsc_exact_benchmark as web
    b = web.compute()
    assert b["weights"] == {"prefer_math": 288, "prefer_vocabulary": 817}
    assert b["estimate"] == pytest.approx(0.787, abs=0.001)
