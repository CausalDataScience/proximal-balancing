"""One place that says what every result file must carry, and checks it.

Round 7 review, PRD-5.  Checkpoint writing and reading were hardened, but the summariser, the rescorer and
the coefficient probe each reached results by their own route and skipped the checks.  Every producer now
builds its header through `header()` and every consumer calls `verify()` before it reports a number.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import shutil
from pathlib import Path

REQUIRED = ("run_id", "code_hash", "script", "args", "training_mode", "seeds", "generated", "complete")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def snapshot_run_sources(artifact_dir: str | Path, sources: dict[str, str | Path]) -> dict:
    """Copy exact run-start source bytes into a new directory and return a hashed manifest record.

    The directory is created exclusively.  Existing historical evidence is therefore never overwritten.
    `sources` is keyed by a stable role, and must include ``frozen_criteria.json``.
    """
    root = Path(artifact_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    if "frozen_criteria.json" not in sources:
        raise ValueError("the run snapshot must include frozen_criteria.json")
    records = {}
    for i, (role, source) in enumerate(sorted(sources.items())):
        src = Path(source).resolve()
        if not src.is_file():
            raise FileNotFoundError(src)
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in role)
        dst = root / f"{i:02d}-{safe}"
        shutil.copyfile(src, dst)
        records[role] = {"original_path": str(src), "snapshot_path": dst.name,
                         "sha256": file_sha256(dst)}
    bundle = hashlib.sha256()
    for role, record in sorted(records.items()):
        bundle.update(role.encode()); bundle.update(record["sha256"].encode())
    body = {"schema_version": 1, "bundle_sha256": bundle.hexdigest(), "sources": records}
    manifest = root / "source-manifest.json"
    manifest.write_text(json.dumps(body, indent=1, sort_keys=True))
    return {"path": str(manifest), "sha256": file_sha256(manifest),
            "bundle_sha256": body["bundle_sha256"]}


def _verify_source_manifest(prov: dict, path: str, expected_criteria_sha256: str | None) -> list[str]:
    problems = []
    rec = prov.get("source_manifest")
    if not isinstance(rec, dict) or not rec.get("path") or not rec.get("sha256"):
        return ["provenance has no complete source_manifest record"]
    try:
        manifest_path = Path(rec["path"])
        if file_sha256(manifest_path) != rec["sha256"]:
            return [f"source manifest {manifest_path} has changed since run start"]
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError, TypeError) as e:
        return [f"source manifest cannot be read: {e}"]
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        return ["source manifest has no sources"]
    if rec.get("bundle_sha256") != manifest.get("bundle_sha256"):
        problems.append("source_manifest bundle digest does not match its manifest")
    if prov.get("code_hash") != str(manifest.get("bundle_sha256", ""))[:16]:
        problems.append("provenance code_hash does not match the snapshotted source bundle")
    for role, src in sources.items():
        if not isinstance(src, dict) or not src.get("snapshot_path") or not src.get("sha256"):
            problems.append(f"source manifest role {role!r} is incomplete")
            continue
        snap = manifest_path.parent / src["snapshot_path"]
        try:
            if file_sha256(snap) != src["sha256"]:
                problems.append(f"snapshot for {role!r} has changed")
        except OSError:
            problems.append(f"snapshot for {role!r} is missing")
    criteria = sources.get("frozen_criteria.json")
    if not isinstance(criteria, dict):
        problems.append("source manifest does not contain frozen_criteria.json")
    else:
        recorded = prov.get("criteria_sha256")
        if recorded != criteria.get("sha256"):
            problems.append("provenance criteria_sha256 does not match the snapshotted criteria")
        if expected_criteria_sha256 is not None and recorded != expected_criteria_sha256:
            problems.append("snapshotted criteria differ from the frozen criteria used by this consumer")
    script = prov.get("script")
    script_record = sources.get(script)
    if not isinstance(script_record, dict):
        problems.append(f"source manifest does not contain producing script {script!r}")
    elif prov.get("script_sha256") != script_record.get("sha256"):
        problems.append("provenance script_sha256 does not match the snapshotted producing script")
    return problems


def current_source_status(payload: dict) -> dict[str, str]:
    """Compare current source bytes with a valid historical snapshot without invalidating it."""
    prov = payload.get("provenance") or {}
    rec = prov.get("source_manifest") or {}
    try:
        mp = Path(rec["path"])
        manifest = json.loads(mp.read_text())
    except (OSError, KeyError, TypeError, ValueError):
        return {"manifest": "INVALID"}
    out = {}
    for role, src in manifest.get("sources", {}).items():
        try:
            out[role] = ("CURRENT_MATCH" if file_sha256(src["original_path"]) == src["sha256"]
                         else "CURRENT_DIFFERS")
        except (OSError, KeyError, TypeError):
            out[role] = "CURRENT_MISSING"
    return out


def header(run_id: str, script: str, args: dict, training_mode: str, seeds: dict, code_hash: str,
           inputs: dict | None = None, extra: dict | None = None) -> dict:
    """Build the provenance block.  `inputs` maps a role to a source file that gets hashed and recorded."""
    h = {"run_id": run_id, "code_hash": code_hash, "script": script,
         "script_sha256": file_sha256(Path(__file__).with_name(script)), "args": dict(args),
         "training_mode": training_mode, "seeds": dict(seeds), "git_commit": git_commit(),
         "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "complete": False}
    if inputs:
        h["inputs"] = {k: {"path": str(v), "sha256": file_sha256(v)} for k, v in inputs.items()}
    if extra:
        h.update(extra)
    return h


def verify(payload: dict, path: str, expect_mode: str | None = None, check_checkpoints: bool = True,
           expect_cells: set | None = None, cell_key=None, cells_at=None,
           require_manifest: bool = False, require_checkpoints: bool = False,
           expected_criteria_sha256: str | None = None) -> list[str]:
    """Return every reason this result file must not be summarised.  An empty list means it is usable."""
    problems = []
    prov = payload.get("provenance")
    if not isinstance(prov, dict):
        return [f"{path} has no provenance block"]
    for k in REQUIRED:
        if k not in prov:
            problems.append(f"provenance is missing '{k}'")
    if not prov.get("complete"):
        problems.append("the run did not finish (provenance.complete is false)")
    if expect_mode is not None and prov.get("training_mode") != expect_mode:
        problems.append(f"training_mode is {prov.get('training_mode')!r}, expected {expect_mode!r}")
    if require_manifest:
        problems.extend(_verify_source_manifest(prov, path, expected_criteria_sha256))
    for role, src in (prov.get("inputs") or {}).items():
        try:
            if file_sha256(src["path"]) != src["sha256"]:
                problems.append(f"input '{role}' at {src['path']} has changed since this result was written")
        except OSError:
            problems.append(f"input '{role}' at {src['path']} is missing")

    cells = []
    if cells_at:
        try:
            raw_cells = cells_at(payload)
            if not isinstance(raw_cells, list) or not raw_cells:
                problems.append("the required cell collection is missing or empty")
            else:
                cells = list(raw_cells)
        except (KeyError, TypeError, ValueError) as e:
            problems.append(f"the required cell collection cannot be read: {e}")
    if cells:
        incomplete = [c for c in cells if c.get("complete") is not True]
        if incomplete:
            problems.append(f"{len(incomplete)} cells are marked incomplete")
        if cell_key is not None:
            keys = [cell_key(c) for c in cells]
            dup = {k for k in keys if keys.count(k) > 1}
            if dup:
                problems.append(f"duplicate cells: {sorted(dup)[:5]}")
            if expect_cells is not None:
                missing = expect_cells - set(keys)
                extra = set(keys) - expect_cells
                if missing:
                    problems.append(f"{len(missing)} expected cells are absent, e.g. {sorted(missing)[:3]}")
                if extra:
                    problems.append(f"{len(extra)} unexpected cells present, e.g. {sorted(extra)[:3]}")
        if check_checkpoints:
            seen = {}
            for c in cells:
                ck = c.get("checkpoint")
                if ck is None:
                    if require_checkpoints:
                        problems.append("a required cell has no checkpoint")
                    continue
                if not isinstance(ck, dict) or "sha256" not in ck:
                    problems.append("a cell has a checkpoint without a digest; it predates hashed "
                                    "checkpoints and cannot be certified retroactively")
                    continue
                if ck["path"] in seen:
                    problems.append(f"two cells share the checkpoint {ck['path']}")
                seen[ck["path"]] = True
                try:
                    if file_sha256(ck["path"]) != ck["sha256"]:
                        problems.append(f"checkpoint {ck['path']} has changed since the result was written")
                except OSError:
                    problems.append(f"checkpoint {ck['path']} is missing")
    return problems


def require(payload: dict, path: str, **kw) -> None:
    problems = verify(payload, path, **kw)
    if problems:
        raise SystemExit(f"refusing to use {path}:\n  - " + "\n  - ".join(problems))
