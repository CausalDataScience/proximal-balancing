"""PROBE for a proxy with J blocks, J other than five; the paper uses it for the two-block kernel benchmark.

`core.run_dataset` is written for the five-block family, so it cannot be called on a data set with another
number of blocks.  Everything that computes a number is still the code of `core`: the preprocessing rule,
`learn_representation`, `screen`, `aipw`, `common_radius` and `aggregate` are imported and used unchanged.
What this file supplies is the loop over a different number of blocks, and the preprocessing class rewritten
for J blocks with the same rule copied line for line.

With J blocks there are T = 2^J - 2 candidate splits and z = Phi^{-1}(1 - 0.05 / T).  With two blocks there are
exactly two candidates, "hold out W1" and "hold out W2", so
    z = Phi^{-1}(1 - 0.05 / 2)   rather than the five-block family's 1 - 0.05 / 30.

Three readings of the screen are reported side by side, because at two candidates they differ and none of
them is obviously the right one:

    significance   retain S when gap_S - z SE_S <= 0, that is when there is no evidence of imbalance.
                   No free threshold, and it is the reading this file treats as primary.
    noise floor    the noise-floor rule, threshold = median over the candidates of z SE.  With two candidates the
                   median is their average, so exactly the more precisely estimated one passes whenever
                   both gaps are zero.  It is degenerate here and is reported to show that.
    both           the gap, the standard error and the estimate of each candidate, with no rule applied.

Overlap is recorded rather than assumed: each candidate's record says whether every base-critic prediction lay
in [0.05, 0.95] in all four rotations.  None of the three readings uses it.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from . import aggregation as ps
from . import core as pr

ALPHA = pr.ALPHA


class PrepJ:
    """The sealed `Prep` rule for an arbitrary number of blocks: standardise X; reduce a block with more
    than two coordinates to its principal directions above the Marchenko-Pastur edge, at most P; then
    standardise.  Fitted on the representation fold alone."""

    def __init__(self, view: dict[str, np.ndarray], rows: np.ndarray, n_blocks: int):
        x = view["X"][rows]
        self.x_mu, self.x_sd = x.mean(0), x.std(0) + 1e-12
        self.n_blocks = n_blocks
        self.blocks = []
        for j in range(n_blocks):
            w = view[f"W{j + 1}"][rows]
            mu = w.mean(0)
            if w.shape[1] > 2:
                _, sv, vt = np.linalg.svd(w - mu, full_matrices=False)
                eig = sv ** 2 / len(rows)
                proj = vt[:pr.spike_count(eig, len(rows), w.shape[1])].T
            else:
                proj = np.eye(w.shape[1])
            scores = (w - mu) @ proj
            self.blocks.append((mu, proj, scores.std(0) + 1e-12))

    def x(self, view, rows):
        return (view["X"][rows] - self.x_mu) / self.x_sd

    def block(self, view, rows, j):
        mu, proj, sd = self.blocks[j]
        return ((view[f"W{j + 1}"][rows] - mu) @ proj) / sd


def all_splits(n_blocks: int) -> list[tuple[int, ...]]:
    import itertools
    return [s for r in range(1, n_blocks) for s in itertools.combinations(range(n_blocks), r)]


def split_name(split) -> str:
    return "S" + "".join(str(j + 1) for j in split)


def run(view: dict, k: int, seed: int, n_blocks: int = 2) -> dict:
    """One pass of Algorithm 1 over the candidates of a J-block proxy, with the sealed pieces."""
    n = len(view["A"])
    fold_rows = pr.folds(n, seed)
    splits = all_splits(n_blocks)
    z = float(norm.ppf(1.0 - ALPHA / len(splits)))
    scores = np.zeros((n, len(splits)))
    records = {split_name(s): {"rotations": []} for s in splits}
    for rot in range(4):
        rl = pr.roles(rot)
        rows = {key: fold_rows[v] for key, v in rl.items()}
        prep = PrepJ(view, rows["D"], n_blocks)
        feats = {role: {"x": prep.x(view, r), "blocks": [prep.block(view, r, j) for j in range(n_blocks)]}
                 for role, r in rows.items()}
        a = {role: view["A"][r] for role, r in rows.items()}
        y = {role: view["Y"][r] for role, r in rows.items()}
        for s_idx, split in enumerate(splits):
            kept = [j for j in range(n_blocks) if j not in split]
            cat = lambda role, idx: np.column_stack([feats[role]["blocks"][j] for j in idx])
            learned = pr.learn_representation(feats["D"]["x"], cat("D", kept), cat("D", split), a["D"], k,
                                              seed + 1_000 * rot + s_idx)
            b = learned["b"]
            scr = pr.screen(feats["S"]["x"], cat("S", kept), cat("S", split), a["S"], b, tolerance=np.inf)
            f = {role: np.column_stack([np.ones(len(a[role])), feats[role]["x"], cat(role, kept) @ b.T])
                 for role in ("N", "E")}
            psi = pr.aipw(f["N"], a["N"], y["N"], f["E"], a["E"], y["E"])
            scores[rows["E"], s_idx] = psi
            records[split_name(split)]["rotations"].append(
                {"rotation": rot, "gap_fit": learned["gap_fit"], "screen": scr, "theta": float(psi.mean())})
    names = [split_name(s) for s in splits]
    for name in names:
        rots = records[name]["rotations"]
        records[name]["estimate"] = float(np.mean([r["theta"] for r in rots]))
        gaps = np.array([r["screen"]["gap"] for r in rots])
        ses = np.array([r["screen"]["se"] for r in rots])
        records[name]["pooled_gap"] = float(gaps.mean())
        records[name]["pooled_se"] = float(np.sqrt((ses ** 2).sum()) / len(ses))
        records[name]["overlap_all_rotations"] = bool(all(r["screen"]["overlap"] for r in rots))

    def aggregate_on(keep_names):
        idx = [names.index(nm) for nm in keep_names]
        if not idx:
            return {"return_kind": "no_return", "output": None, "rho": None, "retained": []}
        rho = float(ps.common_radius(scores[:, idx], seed + 5))
        values = np.array([records[names[i]]["estimate"] for i in idx])
        agg = ps.aggregate([splits[i] for i in idx], values, rho)
        return {"return_kind": agg["return_kind"], "output": agg["output"], "rho": rho,
                "retained": list(keep_names)}

    significance = [nm for nm in names
                    if records[nm]["pooled_gap"] - z * records[nm]["pooled_se"] <= 0.0]
    uppers = {nm: max(records[nm]["pooled_gap"], 0.0) + z * records[nm]["pooled_se"] for nm in names}
    floor = float(np.median([z * records[nm]["pooled_se"] for nm in names]))
    noise_floor = [nm for nm in names if uppers[nm] <= floor]

    return {"n": n, "k": k, "seed": seed, "n_blocks": n_blocks, "z": z, "split_names": names,
            "records": {nm: {key: records[nm][key] for key in
                             ("estimate", "pooled_gap", "pooled_se", "overlap_all_rotations")}
                        for nm in names},
            "per_candidate_upper": uppers, "noise_floor_threshold": floor,
            "readings": {"significance": {"rule": "retain S when pooled gap - z SE <= 0", **aggregate_on(significance)},
                         "noise_floor": {"rule": "retain S when max(gap,0) + z SE <= median_S z SE; with "
                                                 "two candidates the median is their average, so this is "
                                                 "degenerate and is reported for contrast only",
                                         **aggregate_on(noise_floor)},
                         "both": {"rule": "no rule applied", **aggregate_on(names)}},
            "score_crossproduct": ((scores - scores.mean(0)).T @ (scores - scores.mean(0))).tolist()}
