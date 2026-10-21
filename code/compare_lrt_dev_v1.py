"""Development comparison for the likelihood-ratio refinement (2026-10-20, memo section 43).

On the 80 SCM-2 and SCM-3 data sets already seen (E1 seeds and confirmation seeds): MAE of PROBE with the merged
discovered blocks refined with penalty 3.84 (run_discovered_blocks.py --merged --penalty 3.84), against the stored
runs of the same data sets with the angular refinement (penalty none) and with the designed blocks, all under the
sealed tolerance.  Paired differences of absolute errors with standard errors.

Usage: compare_lrt_dev_v1.py LRT_DIR
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results" / "discovered_blocks_v1"


def angular_and_designed(level: str, seed: int) -> tuple[float, float]:
    e1 = RES / "deltaai_merged" / f"merged_{level}_seed{seed}.json"
    if e1.exists():
        d = json.loads(e1.read_text())
        return d["estimate"], d["designed_estimate"]
    d = json.loads((RES / "deltaai_confirm" / f"confirm_{level}_seed{seed}.json").read_text())
    return d["merged"]["estimate"], d["designed"]["estimate"]


def paired(a: np.ndarray, b: np.ndarray) -> str:
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean():+.4f} (se {se:.4f}, t {d.mean() / se:+.2f})"


def main(argv: list[str]) -> int:
    folder = Path(argv[1])
    for level in ("SCM-2", "SCM-3"):
        rows = []
        for f in sorted(glob.glob(str(folder / f"lrt_{level}_seed*.json"))):
            d = json.loads(Path(f).read_text())
            ang, des = angular_and_designed(level, d["seed"])
            pure = all(sum(v > 0 for v in c["from"].values()) == 1 for c in d["blocks"])
            rows.append((d["estimate"], ang, des, pure))
        if not rows:
            continue
        ok = [r for r in rows if None not in r[:3]]
        e = {k: np.array([abs(r[i] - 1.0) for r in ok]) for i, k in enumerate(("lrt", "angular", "designed"))}
        print(f"{level}: {len(rows)} data sets ({len(ok)} returned by all three), blocks pure in {sum(r[3] for r in rows)}")
        print(f"   MAE lrt {e['lrt'].mean():.4f}  angular {e['angular'].mean():.4f}  designed {e['designed'].mean():.4f}")
        print(f"   lrt minus angular {paired(e['lrt'], e['angular'])};  lrt minus designed {paired(e['lrt'], e['designed'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
