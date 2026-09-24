"""After every KSPC comparator task has run: write the summary, freeze every result file with sha256,
rebuild the two main figures with the KSPC rows, and copy them into the paper's figure folder.

    python3 -B finalize_kspc_comparator.py
"""
from __future__ import annotations

import datetime
import glob
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results" / "kspc_comparator"


def main() -> int:
    import run_kspc_comparator as rk
    expected = {e: len(rk.tasks(e)) for e in ("E1", "E2", "E3", "E5", "E6", "E7")}
    have = {e: len(list((OUT / e).glob("*.json"))) for e in expected}
    if have != expected:
        raise SystemExit(f"incomplete: have {have}, expected {expected}")
    if not (OUT / "summary_v1.json").exists():
        rk.summarise()
    freeze = OUT / "FROZEN_v1.json"
    if not freeze.exists():
        files = sorted(glob.glob(str(OUT / "E*" / "*.json"))) + [str(OUT / n) for n in
                 ("summary_v1.json", "gates_v1.json", "gate2_seed_diagnostic_v1.json", "code_integrity_v1.json", "binary_x_diagnostic_v1.json")]
        files += [str(HERE / f) for f in ("run_kspc_comparator.py", "kspc_wrapper.py", "kspc_comparator_integrity.py",
                                          "test_kspc_comparator.py")]
        files += [str(ROOT / "memo" / "2026-09-22-kspc-comparator-prd-v2.md")]
        rec = {"artifact": "kspc_comparator_freeze",
               "frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               "rule": "the recorded KSPC comparator results; not edited, any rerun writes new files",
               "note": "run_kspc_comparator.py had two docstring-only edits while the runs were in progress, so "
                       "shards record one of three hashes for it; no line of code changed",
               "sha256": {str(Path(f).relative_to(ROOT)): hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}
        freeze.write_text(json.dumps(rec, indent=1) + "\n")
        print(f"frozen {len(files)} files")
    for script, stem, target in (("make_section5_v2_figures.py", "section5-v2-estimates", "probe-estimates.pdf"),
                                 ("make_prospective_figure.py", "realdata-prospective", "probe-realdata.pdf")):
        subprocess.run([sys.executable, "-B", script], cwd=HERE, check=True)
        shutil.copy(ROOT / "figures" / "section5" / f"{stem}.pdf", ROOT / "ICLR" / "paper" / "figures" / target)
        print(f"copied {stem}.pdf -> ICLR/paper/figures/{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
