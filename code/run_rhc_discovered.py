"""PROBE on RHC with blocks discovered from the proxy alone (2026-10-19, memo section 43).

The same data, seed and threshold as the stored RHC run (run_rhc.py: pr.run_dataset at GRID[2] with rd.SEED), and
the same report as the figure: the v3 noise-scaled decision (noise_scaled.decide) and the sampling spread of its
aggregate (rhc_aggregate_uncertainty.py, 20,000 draws, same seed).  Three block splits of the ten physiological
measurements are run:
    designed  the hand grouping by clinical system (rd.BLOCKS); a check that this wrapper reproduces the stored run
    spectral  discover_blocks.discover_blocks, from the ten measurements alone (no A, no Y)
    refined   discover_blocks.discover_blocks_refined, the same followed by the k-subspaces refinement
There is no ground truth on RHC.

Output: results/rhc/discovered_blocks_v1.json (write-once).
    python3 -B run_rhc_discovered.py
With the argument `merged`: one more variant, discover_blocks.discover_blocks_merged run through
probe_any_blocks.configure(J), written to results/rhc/discovered_blocks_v2_merged.json (write-once).
    python3 -B run_rhc_discovered.py merged
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import family_v2_dgp as g  # noqa: E402
import family_v2_probe as pr  # noqa: E402
import noise_scaled as ns  # noqa: E402
import probe_structured_scm as ps  # noqa: E402
import rhc_data as rd  # noqa: E402
import run_rhc as rr  # noqa: E402
import probe_any_blocks as pab  # noqa: E402
from discover_blocks import discover_blocks, discover_blocks_merged, discover_blocks_refined  # noqa: E402

OUT = CODE.parent / "results" / "rhc" / "discovered_blocks_v1.json"
DRAWS, DRAW_SEED = 20_000, 98_600_000          # as in rhc_aggregate_uncertainty.py


def spread(probe: dict, decision: dict) -> dict | None:
    """rhc_aggregate_uncertainty.main, on one run: resample the retained estimates and re-aggregate."""
    if decision["output"] is None:
        return None
    names, splits = probe["split_names"], pr.all_splits()
    keep = [names.index(nm) for nm in decision["retained"]]
    cov = np.array(probe["score_crossproduct"])[np.ix_(keep, keep)] / (rd.N * rd.N)
    centre = np.array([probe["per_split"][names[i]]["estimate"] for i in keep])
    sample = np.random.default_rng(DRAW_SEED).multivariate_normal(centre, cov, size=DRAWS)
    outputs = [ps.aggregate([splits[i] for i in keep], row, decision["rho"])["output"] for row in sample]
    outputs = np.array([o for o in outputs if o is not None])
    return {"returned": int(len(outputs)), "of": DRAWS, "mean": float(outputs.mean()), "sd": float(outputs.std(ddof=1)),
            "q025": float(np.quantile(outputs, 0.025)), "q975": float(np.quantile(outputs, 0.975))}


def main() -> int:
    global OUT
    merged = len(sys.argv) > 1 and sys.argv[1] == "merged"
    if merged:
        OUT = OUT.with_name("discovered_blocks_v2_merged.json")
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; write-once")
    data = rd.load()
    view, names = data["view"], data["proxy_names"]
    w = np.column_stack([view[f"W{j + 1}"] for j in range(5)])
    splits = {"designed": [np.array([names.index(c) for c in b]) for b in rd.BLOCKS],
              "spectral": discover_blocks(w, 5, seed=rd.SEED),
              "refined": discover_blocks_refined(w, 5, seed=rd.SEED)}
    if merged:
        splits = {"merged": discover_blocks_merged(w, 5, seed=rd.SEED)}
        pab.configure(len(splits["merged"]))
    out = {"seed": rd.SEED, "threshold": rr.GRID[2], "draws": DRAWS, "draw_seed": DRAW_SEED, "variants": {}}
    for tag, groups in splits.items():
        t0 = time.time()
        dview = {k: view[k] for k in ("X", "A", "Y")}
        for j, idx in enumerate(groups):
            dview[f"W{j + 1}"] = w[:, idx]
        lv = g.Level(name="RHC", index=0, d_x=data["d_x"], d_blocks=tuple(len(idx) for idx in groups), k=2, variant="real")
        probe = pr.run_dataset(dview, lv, rr.GRID[2], rd.SEED)
        stored = {"per_split": {nm: {"estimate": rec["estimate"], "screen": [x["screen"] for x in rec["rotations"]]}
                                for nm, rec in probe["records"].items()},
                  "split_names": probe["split_names"], "score_crossproduct": probe["score_crossproduct"]}
        v3 = ns.decide(stored, rd.N, rd.SEED)
        rec = {"blocks": [[names[i] for i in idx] for idx in groups], "probe": stored,
               "v3": {k: v3[k] for k in ("floor", "n_retained", "retained", "rho", "return_kind", "output")},
               "spread": spread(stored, v3), "seconds": round(time.time() - t0, 1)}
        out["variants"][tag] = rec
        s = rec["spread"]
        print(f"{tag}: blocks {rec['blocks']}; v3 kept {v3['n_retained']} {v3['retained']}; output "
              f"{'none' if v3['output'] is None else f'{v3['output']:.4f}'}"
              f"{'' if s is None else f', 95% ({s['q025']:.3f}, {s['q975']:.3f})'}; {rec['seconds']} s", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "x") as handle:
        handle.write(json.dumps(out, allow_nan=False) + "\n")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
