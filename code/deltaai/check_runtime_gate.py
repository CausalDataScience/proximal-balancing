"""The runtime gate that has to pass before confirmation seeds are spent.

Warm seed at n = 12,000 under 20 minutes, peak GPU memory under 100 GB, all thirty splits and four
rotations present, every nesting check true, no fail-closed encoder.  The first qualification seed is
the cold one and is excluded from the timing rule.
"""
import glob
import json
import sys

paths = sorted(glob.glob(sys.argv[1]))
cells = [json.load(open(p)) for p in paths]
if len(cells) < 2:
    sys.exit(f"need two qualification shards, found {len(cells)}")
warm = cells[1:]
problems = []
for q in warm:
    if q["timing_seconds"] > 1200:
        problems.append(f"seed {q['data_seed']}: {q['timing_seconds']:.0f}s exceeds the 1200s warm limit")
    if (q.get("peak_gpu_gb") or 0) > 100:
        problems.append(f"seed {q['data_seed']}: peak GPU {q['peak_gpu_gb']:.1f} GB exceeds 100 GB")
for q in cells:
    if "fail_closed" in q:
        problems.append(f"seed {q['data_seed']}: encoder failed closed {q['fail_closed']}")
        continue
    if len(q["rows"]) != 120:
        problems.append(f"seed {q['data_seed']}: {len(q['rows'])} candidate rows, expected 120")
    if any(not r["nesting"]["nested"] for r in q["rows"]):
        problems.append(f"seed {q['data_seed']}: a nesting check failed")
    if not all(d["pass"] for d in q["representation_diagnostics"]["discrimination"]):
        problems.append(f"seed {q['data_seed']}: the discrimination check failed")
for q in cells:
    print(f"seed {q['data_seed']}: {q['timing_seconds']:.0f}s, peak GPU {q.get('peak_gpu_gb')}, "
          f"timing {{ {', '.join(f'{k}: {v:.0f}' for k, v in q['timing'].items())} }}")
if problems:
    print("RUNTIME GATE FAILED"); [print("  " + p) for p in problems]; sys.exit(1)
print("runtime gate passed; confirmation may open")
