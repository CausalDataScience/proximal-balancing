"""SCM under the v3 rule: designed against merged discovered blocks (2026-10-20, memo section 43).

The benchmarks use the v3 decision (noise_scaled.decide); SCM had been compared under the sealed tolerance, which was
set on the designed blocks.  Here both block splits are judged both ways on the same data sets:
    E1 seeds (20 per level): designed from results/extension_v1/E1 (screen statistics stored there), merged from
        the rerun in RESULTS_DIR/results_v3_e1 (run_discovered_blocks.py --merged)
    confirmation seeds (20 per level): both from RESULTS_DIR/results_v3_confirm (run_confirmation_v1.py)
Paired differences of absolute errors (merged minus designed) with standard errors.

Usage: compare_v3_scm_v1.py E1_DIR CONFIRM_DIR
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import noise_scaled as ns  # noqa: E402

E1 = CODE.parent / "results" / "extension_v1" / "E1"


def paired(a: np.ndarray, b: np.ndarray) -> str:
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean():+.4f} (se {se:.4f}, t {d.mean() / se:+.2f})"


def report(name: str, rows: list[dict]) -> None:
    for rule in ("sealed", "v3"):
        ok = [r for r in rows if r[f"m_{rule}"] is not None and r[f"d_{rule}"] is not None]
        em = np.array([abs(r[f"m_{rule}"] - 1.0) for r in ok]); ed = np.array([abs(r[f"d_{rule}"] - 1.0) for r in ok])
        print(f"  {name:13s} {rule:6s} {len(ok)}/{len(rows)} returned: MAE merged {em.mean():.4f} designed {ed.mean():.4f}; "
              f"merged minus designed {paired(em, ed)}")


def main(argv: list[str]) -> int:
    e1_dir, confirm_dir = Path(argv[1]), Path(argv[2])
    for level in ("SCM-1", "SCM-2", "SCM-3"):
        print(level)
        rows = []
        for f in sorted(glob.glob(str(e1_dir / f"merged_{level}_seed*.json"))):
            m = json.loads(Path(f).read_text())
            d = json.loads(Path(glob.glob(str(E1 / f"{level}_seed{m['seed']}*.json"))[0]).read_text())
            rows.append({"m_sealed": m["estimate"], "m_v3": m["v3"]["output"], "d_sealed": d["probe"]["estimate"],
                         "d_v3": ns.decide(d["probe"], d["n"], d["task"]["seed"])["output"]})
        if rows:
            report("E1 seeds", rows)
        rows = []
        for f in sorted(glob.glob(str(confirm_dir / f"confirm_{level}_seed*.json"))):
            c = json.loads(Path(f).read_text())
            rows.append({"m_sealed": c["merged"]["estimate"], "m_v3": c["merged"]["v3"]["output"],
                         "d_sealed": c["designed"]["estimate"], "d_v3": c["designed"]["v3"]["output"]})
        if rows:
            report("confirm seeds", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
