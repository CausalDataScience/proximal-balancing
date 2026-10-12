"""Pool realimg_v2 result files: python summarize_v2.py <tag>  (e.g. r5, r10)."""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[3] / "results/single_image/pneumoniamnist_v1"
tag = sys.argv[1] if len(sys.argv) > 1 else "r5"


def ms(v):
    v = np.asarray(v, float)
    return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"))


summary = {}
for udef in ("lin", "logit"):
    p = OUT / f"v2_{udef}_{tag}.jsonl"
    if not p.exists():
        continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for c in sorted({r["scenario"] for r in rows}):
        rr = [r for r in rows if r["scenario"] == c]
        cell = {
            "n": len(rr),
            "naive": ms([r["naive"] for r in rr]),
            "oracle": ms([r["oracle"] for r in rr]),
            "CNN_all": ms([r["CNN_all"] for r in rr]),
            "CNN_all_clipped_frac": ms([r["CNN_all_clipped_frac"] for r in rr]),
            "all_blocks_no_balance": ms([r["all_blocks_no_balance"] for r in rr]),
            "PROBE": ms([r["PROBE"]["theta"] for r in rr]),
            "PROBE_minus_CNN_all": ms([r["PROBE"]["vs_CNN_all"] for r in rr]),
            "PROBE_flag_counts": {f"W{j}": sum(1 for r in rr if j in r["flagged"]) for j in range(1, 6)},
            "PROBE_no_flag": sum(1 for r in rr if not r["flagged"]),
            "ref_CNN_all_minus_W5": ms([r["ref_CNN_all_minus_W5"]["theta"] for r in rr]),
            "gap_mean_by_block": {f"W{j}": float(np.mean([r["blocks"][f"W{j}"]["gap"] for r in rr])) for j in range(1, 6)},
            "rmse": {k: float(np.sqrt(np.mean((np.array([f(r) for r in rr]) - 1.0) ** 2)))
                     for k, f in (("CNN_all", lambda r: r["CNN_all"]), ("PROBE", lambda r: r["PROBE"]["theta"]),
                                  ("all_blocks", lambda r: r["all_blocks_no_balance"]), ("oracle", lambda r: r["oracle"]))},
        }
        summary[f"{udef} D{c}"] = cell
        print(f"== {udef} D{c} (n={len(rr)})")
        for k in ("naive", "oracle", "CNN_all", "all_blocks_no_balance", "PROBE", "ref_CNN_all_minus_W5"):
            m, se = cell[k]
            print(f"  {k:24s} {m:.3f} (MC SE {se:.3f})")
        m, se = cell["PROBE_minus_CNN_all"]
        print(f"  PROBE - CNN_all          {m:+.3f} (MC SE {se:.3f}) | clipped CNN_all {cell['CNN_all_clipped_frac'][0]:.2f} | "
              f"flags {cell['PROBE_flag_counts']} none {cell['PROBE_no_flag']} | gaps {dict((k, round(v, 4)) for k, v in cell['gap_mean_by_block'].items())}")
        print(f"  RMSE {dict((k, round(v, 3)) for k, v in cell['rmse'].items())}")
(OUT / f"v2_summary_{tag}.json").write_text(json.dumps(summary, indent=1))
