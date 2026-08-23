#!/usr/bin/env python3
"""Verify an allowlisted single-proxy-balancing artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("kmu", "zika", "benchmark_suite_tabular", "benchmark_suite_image")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path, manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    files = manifest.get("files", [])
    if not files:
        raise ValueError("Manifest contains no files")

    checked = []
    for record in files:
        relative = record["path"]
        lowered = relative.lower()
        if any(term in lowered for term in FORBIDDEN):
            raise ValueError(f"Rejected artifact entered manifest: {relative}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        expected = record["sha256"]
        if observed != expected:
            raise ValueError(f"Hash mismatch for {relative}: {observed} != {expected}")
        checked.append(relative)

    return {
        "status": "verified",
        "root": str(root),
        "manifest": str(manifest_path),
        "files_checked": len(checked),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest = (args.manifest or root / "code" / "final_experiments" / "manifest.json").resolve()
    print(json.dumps(verify(root, manifest), indent=2))


if __name__ == "__main__":
    main()
