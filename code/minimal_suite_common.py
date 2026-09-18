"""Shared machinery for the minimal manuscript-ready suite: E12, E11-v2, SCM-6 and SCM-7.

Everything here serves one rule: a number may enter a manuscript table only if the file it came from can be
shown to have been produced by the frozen protocol, on the row roles the protocol declares, without any
evaluator-only quantity reaching a learner.  The helpers below make each of those checkable rather than
assumed, and every one of them fails closed.

Two execution modes are kept apart on purpose.

  literal_one_shot     one D, one N, one E, exactly as Algorithm 1 is written.  Only E11-v2 uses it, and only
                       it may be described as an audit of the one-shot finite-sample theorem.
  operational_rotate4  four honest rotations of the same four folds, averaged.  E12, SCM-6 and SCM-7 use it.
                       It is an efficiency variant and is never called a direct application of that theorem.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
MODES = ("literal_one_shot", "operational_rotate4")
PHASES = ("development", "confirmation")
ROLE_NAMES = ("D", "S", "N", "E")
N_BLOCKS = 5
GRID = tuple(np.round(np.linspace(-1.0, 1.0, 101), 12))
TIE_TOL = 1e-12
GAP_FLOOR = -1e-10
OVERLAP_RANGE = (0.1, 0.9)


# ----------------------------------------------------------------------------- hashing and provenance
def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(a: np.ndarray) -> str:
    h = hashlib.sha256()
    v = np.ascontiguousarray(a)
    h.update(str(v.dtype).encode()); h.update(json.dumps(v.shape).encode()); h.update(v.tobytes())
    return h.hexdigest()


def source_snapshot(files: list[str]) -> dict:
    """Per-file digests plus one bundle digest, so a single value pins the whole code state."""
    here = Path(__file__).parent
    per = {f: file_sha256(here / f) for f in sorted(files)}
    bundle = hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()
    return {"bundle_sha256": bundle, "files": per}


def environment() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                timeout=10, cwd=str(Path(__file__).parent)).stdout.strip() or None
    except Exception:
        commit = None
    try:
        import torch
        torch_v = torch.__version__
    except Exception:
        torch_v = None
    return {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch_v,
            "platform": platform.platform(), "git_commit": commit}


def optimizer_seed(protocol_id: str, data_seed: int, rotation: int, split_id: str, restart_id: int) -> int:
    """The protocol's deterministic optimizer-seed rule, separate from the data seed."""
    key = f"{protocol_id}|{data_seed}|{rotation}|{split_id}|{restart_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


# ----------------------------------------------------------------------------- writing results
def write_json_new(path: str | Path, payload: dict) -> None:
    """Write once, atomically, with no NaN and no silent overwrite of an existing result."""
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"{p} already exists; a result path is written once")
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, allow_nan=False, default=_json_default)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        if not np.isfinite(v):
            raise ValueError("a non-finite value reached the result file")
        return v
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"{type(o)} is not serialisable")


DIGEST_FIELD = "payload_sha256"


def canonical_digest(payload: dict) -> str:
    """Digest of everything in the payload except the digest field itself.

    A file cannot contain a hash of its own bytes, which is why the earlier self_sha256 guard compared a
    value that was never written and therefore never fired.  Hashing the canonical serialisation of the
    payload minus that one field breaks the circularity: the value can be stamped in at freeze time and
    recomputed at check time from the parsed object.

    This detects any later edit to a frozen protocol, including an edit that also rewrites the digest to a
    stale value.  It does not defend against someone who edits a field and recomputes the digest with this
    same function.  The result file records the digest it ran under, so that case is caught by comparing
    the protocol against the results that cite it.
    """
    body = {k: v for k, v in payload.items() if k != DIGEST_FIELD}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=_json_default)
    return hashlib.sha256(text.encode()).hexdigest()


def seal_protocol(path: str | Path, payload: dict) -> dict:
    """Stamp the canonical digest into a protocol and write it once.  Every runner freezes through this."""
    sealed = {k: v for k, v in payload.items() if k != DIGEST_FIELD}
    sealed[DIGEST_FIELD] = canonical_digest(sealed)
    write_json_new(path, sealed)
    return sealed


def archive_superseded(path: str | Path, reason: str, replacement: str, tag: str) -> Path:
    """Move a frozen artifact aside under a name that records why, without ever overwriting an archive.

    A file can be superseded more than once for different reasons, so the tag is part of the name.  An
    earlier archive was destroyed by reusing the plain `_superseded` suffix twice, which is why this
    refuses to write over one.
    """
    src = Path(path)
    dst = src.with_name(f"{src.stem}_superseded_{tag}{src.suffix}")
    if dst.exists():
        raise FileExistsError(f"{dst} already records a superseded version; choose another tag")
    payload = json.loads(src.read_text())
    payload["superseded"] = {"reason": reason, "replacement": replacement, "tag": tag}
    write_json_new(dst, payload)
    src.unlink()
    return dst


def result_root(experiment_id: str, mode: str, phase: str, protocol_path: str, sources: list[str]) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    return {"schema_version": SCHEMA_VERSION, "experiment_id": experiment_id, "mode": mode, "phase": phase,
            "complete": False,
            "protocol": {"path": str(protocol_path), "sha256": file_sha256(protocol_path),
                         DIGEST_FIELD: json.loads(Path(protocol_path).read_text()).get(DIGEST_FIELD)},
            "source_snapshot": source_snapshot(sources), "environment": environment(),
            "cells": [], "summary": {}, "verdict": "NOT_EVALUATED"}


def check_protocol(protocol: dict, requested: dict, protocol_path: str, sources: list[str]) -> None:
    """Refuse to run unless the command matches a frozen run and the code still hashes as recorded."""
    snap = source_snapshot(sources)
    frozen = protocol.get("source_snapshot", {}).get("files", {})
    for name, digest in frozen.items():
        if snap["files"].get(name) != digest:
            raise SystemExit(f"source {name} differs from the frozen protocol")
    runs = protocol.get("runs")
    if runs is None or requested not in runs:
        raise SystemExit(f"the command does not match a frozen run: {requested}")
    recorded = protocol.get(DIGEST_FIELD)
    if recorded is None:
        raise SystemExit(f"{protocol_path} carries no {DIGEST_FIELD}: it was not sealed and cannot be "
                         f"checked for changes made after freezing")
    if canonical_digest(protocol) != recorded:
        raise SystemExit(f"{protocol_path} does not match its own {DIGEST_FIELD}: it changed after it "
                         f"was frozen")


# ----------------------------------------------------------------------------- roles and splits
@dataclass(frozen=True)
class Rotation:
    index: int
    roles: dict[str, np.ndarray]


def four_folds(n_total: int, seed: int) -> list[np.ndarray]:
    """Four disjoint folds of equal size, as row indices into one generated sample."""
    if n_total % 4:
        raise ValueError("the total sample must divide into four equal folds")
    order = np.random.default_rng(seed).permutation(n_total)
    per = n_total // 4
    return [np.sort(order[k * per:(k + 1) * per]) for k in range(4)]


def rotations(folds: list[np.ndarray]) -> list[Rotation]:
    """Rotation k assigns D = F_k, S = F_{k+1}, N = F_{k+2}, E = F_{k+3}, indices modulo four."""
    out = []
    for k in range(4):
        roles = {name: folds[(k + i) % 4] for i, name in enumerate(ROLE_NAMES)}
        assert_roles_disjoint(roles)
        out.append(Rotation(k, roles))
    return out


def assert_roles_disjoint(roles: dict[str, np.ndarray]) -> dict:
    """Stop the run if any row serves two roles in the same rotation."""
    counts = {}
    names = list(roles)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = len(np.intersect1d(roles[a], roles[b]))
            counts[f"{a}&{b}"] = shared
            if shared:
                raise ValueError(f"roles {a} and {b} share {shared} rows")
    return counts


def role_ledger(roles: dict[str, np.ndarray]) -> list[dict]:
    return [{"role": k, "n": int(len(v)), "rows_sha256": array_sha256(v)} for k, v in roles.items()]


def nested_prefix(folds: list[np.ndarray], per_fold: int) -> list[np.ndarray]:
    """The smaller sample sizes take a prefix of each fold, so they nest inside the larger ones."""
    if any(per_fold > len(f) for f in folds):
        raise ValueError("requested more rows per fold than the fold holds")
    return [f[:per_fold] for f in folds]


def oriented_splits(n_blocks: int = N_BLOCKS) -> list[tuple[int, ...]]:
    """Every nonempty proper subset of the blocks, held out in turn.  None is removed by a truth label."""
    out = []
    for mask in range(1, 2 ** n_blocks - 1):
        out.append(tuple(j for j in range(n_blocks) if mask >> j & 1))
    return sorted(out, key=lambda s: (len(s), s))


def split_id(split: tuple[int, ...]) -> str:
    return "S" + "".join(str(j) for j in split)


def evaluator_category(split: tuple[int, ...]) -> str:
    """A label for the evaluation tables only.  Never reaches selection, screening or aggregation."""
    if 0 in split:
        return "block0_heldout"
    if split == (4,):
        return "bad_singleton"
    return "mixed" if 4 in split else "clean"
