"""PROBE on SCM-1 to SCM-3 with a random split of the proxy into blocks (2026-10-19, author's question: what if the
block split is not known in advance?).

For one data set: the same data, seed, fold seed and sealed tolerance as the extension run E1
(results/extension_v1/E1/<level>_seed<seed>.json), but the columns of W = (W_1, ..., W_5) are pooled, permuted at
random (seed + 11), and cut into five blocks of the designed sizes, so every block has the width PROBE saw before.
PROBE (family_v2_probe.run_dataset) is unchanged.  The evaluator records, for every random block, how many of its
columns come from each designed block; a random block is clean when it has no column of W_4 or W_5.

Output: results/random_blocks_v1/<level>_seed<seed>.json.  Usage: run_random_blocks.py LEVEL N_DATASETS
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import family_v2_dgp as g  # noqa: E402
import family_v2_probe as pr  # noqa: E402

N = 24_000
RES = CODE.parent / "results"
OUT = RES / "random_blocks_v1"


def random_view(view: dict, sizes: tuple[int, ...], seed: int) -> tuple[dict, list[dict]]:
    cols = np.column_stack([view[f"W{j + 1}"] for j in range(g.N_BLOCKS)])
    origin = np.concatenate([np.full(view[f"W{j + 1}"].shape[1], j + 1) for j in range(g.N_BLOCKS)])
    perm = np.random.default_rng(seed + 11).permutation(cols.shape[1])
    cuts = np.cumsum((0,) + tuple(sizes))
    out = {k: view[k] for k in ("X", "A", "Y")}
    comp = []
    for j in range(g.N_BLOCKS):
        idx = perm[cuts[j]:cuts[j + 1]]
        out[f"W{j + 1}"] = cols[:, idx]
        counts = {f"W{b}": int((origin[idx] == b).sum()) for b in range(1, g.N_BLOCKS + 1)}
        comp.append({"columns": idx.tolist(), "from": counts, "clean": counts["W4"] == 0 and counts["W5"] == 0})
    return out, comp


def main(argv: list[str]) -> int:
    level_name, n_sets = argv[1], int(argv[2])
    lv = g.level(level_name)
    OUT.mkdir(parents=True, exist_ok=True)
    designed = sorted(glob.glob(str(RES / "extension_v1" / "E1" / f"{level_name}_seed*.json")))[:n_sets]
    for path in designed:
        ref = json.loads(Path(path).read_text())
        seed, tol = ref["task"]["seed"], ref["probe"]["tolerance"]
        out_path = OUT / f"{level_name}_seed{seed}.json"
        if out_path.exists():
            continue
        t0 = time.time()
        view = g.learner_view(g.generate(lv, N, seed))
        rview, comp = random_view(view, lv.d_blocks, seed)
        res = pr.run_dataset(rview, lv, tol, seed + 7)
        rec = {"level": level_name, "seed": seed, "tolerance": tol, "blocks": comp,
               "n_clean_blocks": sum(c["clean"] for c in comp),
               "random": {k: res[k] for k in ("estimate", "return_kind", "retained", "members", "component_sizes", "rho")},
               "random_split_estimates": {k: v["estimate"] for k, v in res["records"].items()},
               "random_split_pass": {k: v["pass_all"] for k, v in res["records"].items()},
               "designed": {"estimate": ref["probe"]["estimate"], "return_kind": ref["probe"]["return_kind"],
                            "retained": ref["probe"]["retained"]},
               "seconds": round(time.time() - t0, 1)}
        out_path.write_text(json.dumps(rec, indent=1))
        est = rec["random"]["estimate"]
        print(f"{level_name} seed {seed}: clean blocks {rec['n_clean_blocks']}, retained {len(res['retained'])}, "
              f"random {'none' if est is None else f'{est:.3f}'} vs designed {ref['probe']['estimate']:.3f} "
              f"({rec['seconds']} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
