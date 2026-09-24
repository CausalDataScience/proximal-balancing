"""Checks for run_kspc_comparator.py (PRD v2)."""
from __future__ import annotations

import numpy as np
import pytest

import run_kspc_comparator as rk


def test_the_task_lists_match_the_final_experiments():
    counts = {e: len(rk.tasks(e)) for e in ("E1", "E2", "E3", "E5", "E6", "E7")}
    assert counts == {"E1": 60, "E2": 20, "E3": 1, "E5": 10, "E6": 10, "E7": 9}


def test_the_integrity_check_rejects_data_that_differ_from_the_sealed_shard():
    task = rk.tasks("E5")[0]
    view, _ = rk.build_view(task)
    rk.integrity(view, task)                                   # the rebuilt data pass
    view = dict(view)
    rng = np.random.default_rng(0)
    view["W1"] = view["W1"] + 0.05 * rng.normal(size=view["W1"].shape)   # a change the estimators can see
    with pytest.raises(RuntimeError, match="integrity"):
        rk.integrity(view, task)


def test_the_generator_code_still_hashes_as_every_frozen_protocol_recorded():
    """Closes the gap the COCA check leaves (a constant added to Y)."""
    import kspc_comparator_integrity as ci
    assert ci.check()["pass"]
