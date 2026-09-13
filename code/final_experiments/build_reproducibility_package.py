#!/usr/bin/env python3
"""Build the standalone, allowlisted reproducibility package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = Path("/Users/yonghanjung/Dropbox/Personal/Research/Code/single-proxy-balancing")
MANUSCRIPT_EXCLUDES = {
    ".aux", ".bbl", ".bcf", ".blg", ".fdb_latexmk", ".fls", ".log", ".out",
    ".run.xml", ".synctex.gz", ".toc",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source_root: Path, target_root: Path, relative: str, target_relative: str | None = None) -> None:
    source = source_root / relative
    target = target_root / (target_relative or relative)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_manuscript(source_root: Path, target_root: Path) -> None:
    source_dir = source_root / "manuscript"
    for source in sorted(source_dir.rglob("*")):
        if not source.is_file():
            continue
        if any(source.name.endswith(suffix) for suffix in MANUSCRIPT_EXCLUDES):
            continue
        relative = source.relative_to(source_root)
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def package_readme() -> str:
    return """# Single Proxy Balancing: Reproducibility Package

This package contains only the experiments accepted for the manuscript:

1. honest approximate SCM1 and SCM2;
2. exact-assumption Tracks A and B;
3. finite-state certificate verification; and
4. the descriptive WSC observational benchmark.

KMU, Zika, exploratory high-dimensional/image suites, and superseded pilots are intentionally excluded.

## Quick verification

```bash
python3 verify_package.py
python3 code/final_experiments/make_manuscript_figures.py --root . --check-only
```

To regenerate the figures:

```bash
python3 code/final_experiments/make_manuscript_figures.py --root .
```

To compile the manuscript:

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The WSC randomized contrast is a noisy reference, not causal ground truth. Its ten repetitions are algorithmic data splits on one fixed data set, not Monte Carlo data-generating repetitions.
"""


def environment_note() -> str:
    return """# Environment

The final artifacts were validated on macOS with Python 3 and the versions pinned in `requirements.txt`.
The manuscript additionally requires a LaTeX installation providing `latexmk` and the packages loaded by `manuscript/main.tex`.
No dependency was added for packaging. PDF and PNG figures are generated with a noninteractive Matplotlib backend.
The WSC source files are exposed under `data/wsc/` and mirrored under their original `materials/real_world_data/` paths so that the frozen analysis scripts run without modification.
"""


def requirements() -> str:
    return """matplotlib==3.11.0
numpy==2.4.6
pandas==3.0.3
scikit-learn==1.9.0
torch==2.13.0
"""


def gitignore() -> str:
    return """__pycache__/
*.py[cod]
manuscript/*.aux
manuscript/*.bbl
manuscript/*.blg
manuscript/*.fdb_latexmk
manuscript/*.fls
manuscript/*.log
manuscript/*.out
manuscript/*.toc
"""


def verifier_wrapper() -> str:
    return """#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import json

root = Path(__file__).resolve().parent
module_path = root / "code" / "final_experiments" / "verify_manifest.py"
spec = importlib.util.spec_from_file_location("spb_verify_manifest", module_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print(json.dumps(module.verify(root, root / "MANIFEST.json"), indent=2))
"""


def build(source_root: Path, target_root: Path) -> dict[str, object]:
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError(f"Target exists and is not empty: {target_root}")
    target_root.mkdir(parents=True, exist_ok=True)

    canonical_manifest_path = source_root / "code" / "final_experiments" / "manifest.json"
    with canonical_manifest_path.open("r", encoding="utf-8") as handle:
        canonical_manifest = json.load(handle)

    for record in canonical_manifest["files"]:
        if record.get("package", False):
            copy_file(source_root, target_root, record["path"], record.get("package_path"))
            if record.get("package_path") and record["path"].startswith("materials/real_world_data/"):
                copy_file(source_root, target_root, record["path"])

    copy_manuscript(source_root, target_root)
    copy_file(source_root, target_root, "code/final_experiments/make_manuscript_figures.py")
    copy_file(source_root, target_root, "code/final_experiments/verify_manifest.py")
    copy_file(source_root, target_root, "code/final_experiments/manifest.json")
    (target_root / "code" / "__init__.py").touch()
    (target_root / "code" / "final_experiments" / "__init__.py").touch()

    write_text(target_root / "README.md", package_readme())
    write_text(target_root / "ENVIRONMENT.md", environment_note())
    write_text(target_root / "requirements.txt", requirements())
    write_text(target_root / ".gitignore", gitignore())
    write_text(target_root / "verify_package.py", verifier_wrapper())

    records = []
    for path in sorted(target_root.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json" or ".git" in path.parts:
            continue
        records.append({"path": str(path.relative_to(target_root)), "sha256": sha256(path)})
    package_manifest = {
        "schema_version": 1,
        "scope": "accepted final single-proxy-balancing reproducibility package",
        "files": records,
    }
    write_text(target_root / "MANIFEST.json", json.dumps(package_manifest, indent=2) + "\n")
    return {"target": str(target_root), "files": len(records)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build(args.source_root.resolve(), args.target.resolve()), indent=2))


if __name__ == "__main__":
    main()
