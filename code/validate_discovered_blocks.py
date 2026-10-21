"""Validation of discover_blocks on SCM-1 to SCM-3 at the full sample size (2026-10-19, memo section 43).

For each data set (the first N_SETS seeds of E1 per level, n = 24,000): discover 5 blocks from W alone, then score
them with the generator's own structure, which no learner sees.  Block j of the design is [info_j, noise] Q_j^T, so
coordinate c of W_j carries Q_j[c, 0] M_j (and, in W_5, also Q_j[c, 1] I), with M_j = U + 0.85 eps_j and
M_4 = U - 3 I + 0.85 eps_4.  For a discovered block B:
  u_energy  = sum over its coordinates of the squared loading on U         (a designed block has 1)
  i_energy  = sum over its coordinates of the squared loading on I         (treatment-only content; 0 for W_1..W_3)
  read_i_share = over the principal components of B's population covariance that PROBE's per-block reduction keeps,
               those above the Marchenko-Pastur edge sigma^2 (1 + sqrt(|B| / n_D))^2 with n_D = 6,000 rows and sigma^2
               the mean noise variance of B's coordinates, the largest share of signal variance that comes from I
               (the first version looked at the leading component only and let a block of W_5 coordinates pass whose
               second component, also above the edge, is almost all I)
A discovered block is called clean when read_i_share < 0.01.  Also reported: the designed block each discovered block
mostly comes from, and its purity.

Output: results/discovered_blocks_v1/validation_discovery.json and a printed table.
Usage: validate_discovered_blocks.py N_SETS
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
from discover_blocks import discover_blocks  # noqa: E402

RES = CODE.parent / "results"
LAT = ["U", "I", "e0", "e1", "e2", "e3", "e4"]          # independent standard normal latents behind W


def loadings(level) -> tuple[np.ndarray, np.ndarray]:
    """Rows: coordinates of W in order W_1..W_5; columns: LAT.  Also the origin block of every coordinate."""
    q = g.rotations(level)
    rows, origin = [], []
    for j in range(g.N_BLOCKS):
        qj = q[f"W{j + 1}"]
        for c in range(qj.shape[0]):
            v = np.zeros(len(LAT))
            m_idx = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}[j]          # W_1..W_4 carry M_1..M_4, W_5 carries M_0
            v[0] += qj[c, 0]; v[2 + m_idx] += g.PROXY_SD * qj[c, 0]
            if j == 3:
                v[1] += g.BAD_I * qj[c, 0]
            if j == 4:
                v[1] += qj[c, 1]
            rows.append(v); origin.append(j + 1)
    return np.array(rows), np.array(origin)


def score_block(idx: np.ndarray, load: np.ndarray, noise_var: np.ndarray) -> dict:
    lb = load[idx]
    cov = lb @ lb.T + np.diag(noise_var[idx])
    lam, vec = np.linalg.eigh(cov)
    edge = noise_var[idx].mean() * (1 + np.sqrt(len(idx) / 6000)) ** 2
    read = [k for k in range(len(lam)) if lam[k] > edge] or [len(lam) - 1]
    shares = []
    for k in read:
        pc = vec[:, k] @ lb
        shares.append(float(pc[1] ** 2 / (pc ** 2).sum()))
    return {"u_energy": float((lb[:, 0] ** 2).sum()), "i_energy": float((lb[:, 1] ** 2).sum()),
            "n_read": len(read), "read_i_share": max(shares)}


def main(argv: list[str]) -> int:
    n_sets = int(argv[1])
    out = {"levels": {}}
    for lvn in ("SCM-1", "SCM-2", "SCM-3"):
        lv = g.level(lvn)
        load, origin = loadings(lv)
        # noise variance of each coordinate: the rotated pure-noise coordinates
        q = g.rotations(lv); nv = []
        for j in range(g.N_BLOCKS):
            qj = q[f"W{j + 1}"]; c_info = len(g.block_info(lv)[j])
            nv += list((g.NOISE_SD ** 2) * (qj[:, c_info:] ** 2).sum(1))
        noise_var = np.array(nv)
        designed = [score_block(np.flatnonzero(origin == b), load, noise_var) for b in range(1, 6)]
        sets = []
        for f in sorted(glob.glob(str(RES / "extension_v1" / "E1" / f"{lvn}_seed*.json")))[:n_sets]:
            seed = json.loads(Path(f).read_text())["task"]["seed"]
            t0 = time.time()
            view = g.learner_view(g.generate(lv, 24_000, seed))
            w = np.column_stack([view[f"W{j + 1}"] for j in range(g.N_BLOCKS)])
            groups = discover_blocks(w, g.N_BLOCKS, seed=seed)
            blocks = []
            for idx in groups:
                counts = np.bincount(origin[idx], minlength=6)[1:]
                sc = score_block(idx, load, noise_var)
                blocks.append({"size": int(len(idx)), "from": counts.tolist(), "major": int(counts.argmax() + 1),
                               "purity": float(counts.max() / len(idx)), "columns": idx.tolist(), **sc, "clean": sc["read_i_share"] < 0.01})
            sets.append({"seed": seed, "blocks": blocks, "seconds": round(time.time() - t0, 1)})
            clean = [b for b in blocks if b["clean"]]
            print(f"{lvn} seed {seed}: clean blocks {len(clean)} (u_energy {', '.join(f'{b['u_energy']:.2f}' for b in clean)}; "
                  f"max read I share {max(b['read_i_share'] for b in clean) if clean else float('nan'):.4f}; from {[b['from'] for b in clean]}); "
                  f"others read I share {', '.join(f'{b['read_i_share']:.3f}' for b in blocks if not b['clean'])}", flush=True)
        out["levels"][lvn] = {"designed": designed, "sets": sets}
    path = RES / "discovered_blocks_v1" / "validation_discovery.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print("designed blocks (u_energy, pc I share):",
          {k: [(round(b["u_energy"], 2), round(b["read_i_share"], 3)) for b in v["designed"]] for k, v in out["levels"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
