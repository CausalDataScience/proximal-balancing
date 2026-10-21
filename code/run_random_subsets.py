"""PROBE on SCM-1 and SCM-2 when the split of the proxy into blocks is not known (2026-10-19, author's question).

Every coordinate of W is its own block, W_1, ..., W_m (m = 6 in SCM-1, 80 in SCM-2), and PROBE draws its held-out
sets at random from the subsets of {1, ..., m}, as the Sample step of Algorithm 1 does:
  uniform  a split budget of 30 held-out sets drawn uniformly without replacement from the nonempty proper subsets
           (all 62 when m = 6 would exceed the budget, so 30 of them are drawn there too);
  single   30 held-out sets of one coordinate each (all 6 in SCM-1), a contrast with small held-out sets.
Everything else is family_v2_probe unchanged: the same data, fold seed, sealed tolerance as the extension run E1
(results/extension_v1/E1/<level>_seed<seed>.json), the nested Brier screen, AIPW, and the aggregation by the largest
agreeing component.  With single coordinates there is no block to reduce, so each coordinate is standardised on the
representation fold.  The evaluator records, for each held-out set, how many of its coordinates come from each
designed block; a held-out set is clean when it has no coordinate of W_4 or W_5.

Output: results/random_subsets_v1/<mode>_<level>_seed<seed>.json.  Usage: run_random_subsets.py MODE LEVEL N_DATASETS
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
import probe_structured_scm as ps  # noqa: E402

N = 24_000
BUDGET = 30
RES = CODE.parent / "results"
OUT = RES / "random_subsets_v1"


def sample_splits(m: int, mode: str, rng) -> list[tuple[int, ...]]:
    if mode == "single":
        return [(int(i),) for i in sorted(rng.choice(m, size=min(BUDGET, m), replace=False))]
    seen, out = set(), []
    while len(out) < min(BUDGET, 2 ** m - 2):
        s = tuple(int(i) for i in np.flatnonzero(rng.random(m) < 0.5))
        if 0 < len(s) < m and s not in seen:
            seen.add(s); out.append(s)
    return out


def run(view: dict, origin: np.ndarray, lv, tol: float, seed: int, splits) -> dict:
    t0 = time.time()
    x_all, a_all, y_all = view["X"], view["A"], view["Y"]
    w_all = np.column_stack([view[f"W{j + 1}"] for j in range(g.N_BLOCKS)])
    n, m = w_all.shape
    fold_rows = pr.folds(n, seed)
    scores = np.zeros((n, len(splits)))
    recs = [{"rotations": []} for _ in splits]
    for rot in range(4):
        rows = {k: fold_rows[v] for k, v in pr.roles(rot).items()}
        d = rows["D"]
        mx, sx = x_all[d].mean(0), x_all[d].std(0) + 1e-12
        mw, sw = w_all[d].mean(0), w_all[d].std(0) + 1e-12
        x = {r: (x_all[i] - mx) / sx for r, i in rows.items()}
        w = {r: (w_all[i] - mw) / sw for r, i in rows.items()}
        a = {r: a_all[i] for r, i in rows.items()}
        y = {r: y_all[i] for r, i in rows.items()}
        for s_idx, split in enumerate(splits):
            held = list(split); kept = [j for j in range(m) if j not in split]
            learned = pr.learn_representation(x["D"], w["D"][:, kept], w["D"][:, held], a["D"], lv.k, seed + 1_000 * rot + s_idx)
            b = learned["b"]
            scr = pr.screen(x["S"], w["S"][:, kept], w["S"][:, held], a["S"], b, tol)
            f = {r: np.column_stack([np.ones(len(a[r])), x[r], w[r][:, kept] @ b.T]) for r in ("N", "E")}
            psi = pr.aipw(f["N"], a["N"], y["N"], f["E"], a["E"], y["E"])
            scores[rows["E"], s_idx] = psi
            recs[s_idx]["rotations"].append({"pass": bool(scr["pass"]), "theta": float(psi.mean())})
    out = []
    for split, rec in zip(splits, recs):
        counts = {f"W{b}": int((origin[list(split)] == b).sum()) for b in range(1, g.N_BLOCKS + 1)}
        out.append({"held_out": list(split), "size": len(split), "from": counts,
                    "clean": counts["W4"] == 0 and counts["W5"] == 0,
                    "pass_all": all(r["pass"] for r in rec["rotations"]),
                    "estimate": float(np.mean([r["theta"] for r in rec["rotations"]]))})
    retained = [i for i, r in enumerate(out) if r["pass_all"]]
    if retained:
        rho = float(ps.common_radius(scores[:, retained], seed + 5))
        agg = ps.aggregate([splits[i] for i in retained], np.array([out[i]["estimate"] for i in retained]), rho)
    else:
        rho, agg = None, {"return_kind": "no_return", "output": None, "members": []}
    return {"splits": out, "retained": retained, "rho": rho, "estimate": agg["output"], "return_kind": agg["return_kind"],
            "members": [list(map(list, c)) for c in agg.get("members", [])] if agg.get("members") else [],
            "seconds": round(time.time() - t0, 1)}


def main(argv: list[str]) -> int:
    mode, level_name, n_sets = argv[1], argv[2], int(argv[3])
    lv = g.level(level_name)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in sorted(glob.glob(str(RES / "extension_v1" / "E1" / f"{level_name}_seed*.json")))[:n_sets]:
        ref = json.loads(Path(path).read_text())
        seed, tol = ref["task"]["seed"], ref["probe"]["tolerance"]
        out_path = OUT / f"{mode}_{level_name}_seed{seed}.json"
        if out_path.exists():
            continue
        view = g.learner_view(g.generate(lv, N, seed))
        origin = np.concatenate([np.full(view[f"W{j + 1}"].shape[1], j + 1) for j in range(g.N_BLOCKS)])
        splits = sample_splits(len(origin), mode, np.random.default_rng(seed + 13))
        res = run(view, origin, lv, tol, seed + 7, splits)
        res.update({"mode": mode, "level": level_name, "seed": seed, "tolerance": tol, "m": int(len(origin)),
                    "designed": {"estimate": ref["probe"]["estimate"], "retained": ref["probe"]["retained"]}})
        out_path.write_text(json.dumps(res, indent=1))
        sp = res["splits"]; kept = [sp[i] for i in res["retained"]]
        est = res["estimate"]
        print(f"{mode} {level_name} seed {seed}: clean held-out sets {sum(s['clean'] for s in sp)}/{len(sp)}, "
              f"retained {len(kept)} ({sum(s['clean'] for s in kept)} clean), PROBE {'none' if est is None else f'{est:.3f}'} "
              f"vs designed blocks {ref['probe']['estimate']:.3f} ({res['seconds']} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
