"""Checks for the planted RHC extension (extend_rhc_planted.py)."""
from __future__ import annotations

import extend_rhc_planted as ext
import run_rhc_planted as rrp


def test_the_extension_is_fixed_and_disjoint_from_every_earlier_replicate():
    assert ext.EXTENSION == tuple(range(21, 51))
    assert ext.DESIGNATED == tuple(range(11, 21))
    assert not set(ext.EXTENSION) & set(ext.DESIGNATED)
    assert not set(ext.EXTENSION) & set(rrp.CONFIRM)          # the development replicates 1-10


def test_the_extension_runs_exactly_the_code_the_protocol_froze():
    ext.assert_same_code()                                   # raises SystemExit if any file differs


def test_the_frozen_expectation_covers_every_runner_source():
    assert set(ext.frozen_expectation()) == set(rrp.SOURCES)
