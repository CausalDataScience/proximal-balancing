"""v9: the paper's Algorithm 1 on the X-ray benchmark, next to whole-image adjustment and the block audit used in
the main-text single-image table (memo/2026-10-10-pneumoniamnist-benchmark-v1.md, section 34).

Scenarios (64 px, U = cross-fitted classifier logit, hidden; tau = 1):
  V0  clean.
  V3  treatment response in W5, exactly as realimg_v5_sup.py (same seeds, so reps 0-4 redraw the main-table data).
  V7  device collider in W5: the V1 tube line, drawn with P(device) = expit(-2 + 2A + 1.5U): treated patients and
      sicker patients carry it more often.  It is post-treatment and a collider (A -> device <- U).
Per block and cross-fit half, the v5 supervised encoder gives 128 pooled features.
  * whole-image adjustment and the block audit use PCA 16 per block (80 features), as in the main table;
  * Algorithm 1 is the sealed family_v2_probe.run_dataset through level_a.run_probe on the raw 128 features per block
    (the sealed Prep keeps <= 4 principal components per block; rank-2 balancing representation; 30 splits; four
    rotations; honest AIPW; plurality median).  Reported under the noise-floor rule (the paper's real-data
    benchmarks), the fixed threshold 0.0015 (the paper's synthetic settings), and the relaxed rule (all rotations
    upper <= 0.005, median of retained), with every split's estimate and pooled gap, and the sealed-Prep all-block
    adjustment without balance (the compression diagnostic).
Resumable: results/single_image/pneumoniamnist_v1/v9algo1_<tag>.jsonl, one line per (scenario, rep).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import norm

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import realimg_benchmark as rb  # noqa: E402
import realimg_v3 as v3  # noqa: E402
import realimg_v4 as v4  # noqa: E402
import realimg_v5_sup as v5  # noqa: E402
from realimg_benchmark import pr, nets, level_a  # noqa: E402

RELAXED = 0.005
DEVICE = (-2.0, 2.0, 1.5)        # intercept, A, U on the logit scale (memo section 34, fixed before running)


def draw(c, r, x, u, m):
    n = len(x); rng = np.random.default_rng(20265000 + 100 * c + r)
    a = (rng.random(n) < 0.1 + 0.8 * norm.cdf(1.5 * u)).astype(float)
    y = a + u + rng.standard_normal(n)
    img = x.copy(); true_blocks = []; extra = {}
    if c == 3:
        rr = u - 1.5 * a + 0.5 * rng.standard_normal(n); rt = np.clip((rr - rr.mean()) / rr.std(), -2.5, 2.5)
        yy, xx = np.mgrid[0:64, 0:64]
        blob = (np.exp(-((yy - 20) ** 2 + (xx - 31.5) ** 2) / (2 * 2.0 ** 2)) * m[4]).astype(np.float32)
        img = np.clip(img + 0.25 * rt[:, None, None].astype(np.float32) * blob[None], 0, 1); true_blocks = [4]
    if c == 7:
        dev = rng.random(n) < expit(DEVICE[0] + DEVICE[1] * a + DEVICE[2] * u)
        line = np.zeros((64, 64), np.float32); line[4:45, 31:33] = 0.35
        img[dev] = np.clip(img[dev] + line[None] * m[4][None], 0, 1); true_blocks = [4]
        extra = {"device_rate": float(dev.mean()), "device_rate_A0": float(dev[a == 0].mean()),
                 "device_rate_A1": float(dev[a == 1].mean())}
    return a, y, img, true_blocks, extra


def done_pairs(out):
    if not out.exists():
        return set()
    return {(json.loads(l)["scenario"], json.loads(l)["rep"]) for l in out.read_text().splitlines() if l.strip()}


def run(scenarios, rep_start, reps, tag, deadline=None, smoke=False):
    out = rb.OUT / f"v9algo1_{tag}.jsonl"; done = done_pairs(out)
    todo = [(c, r) for c in scenarios for r in range(rep_start, reps) if (c, r) not in done]
    if not todo:
        rb.log(f"v9algo1[{tag}]: nothing left"); return
    x, label = rb.load(); m = rb.masks(); _, stats, _ = rb.get_mae(x)
    u = np.load(rb.OUT / "u_logit_crossfit.npy"); u = (u - u.mean()) / u.std()
    if smoke:
        keep = np.random.default_rng(0).permutation(len(x))[:800]; x, u = x[keep], u[keep]; v5.EPOCHS = 1
    n = len(x); last = None
    for c, r in todo:
        if deadline is not None:
            est = 1.25 * last if last else 900.0
            if time.time() + est > deadline:
                rb.log(f"v9algo1[{tag}]: deadline rule, stopping before V{c} r{r}"); return
        t0 = time.time()
        a, y, img, true_blocks, extra = draw(c, r, x, u, m)
        seed = 15000 + 100 * c + r
        raw = [v5.supervised_block_features(img, stats, m[k], a, y, seed + 10 * k) for k in range(5)]
        feats = [v4.pca16(f) for f in raw]
        t_feat = time.time() - t0
        fold_rows = pr.folds(n, seed); roles = [pr.roles(rot) for rot in range(4)]
        pix = [img[:, m[j]].astype(np.float64) for j in range(5)]
        tfeat = [[v3.t_features(pix[j], fold_rows[roles[rot]["D"]]) for rot in range(4)] for j in range(5)]
        Z = lambda hidden: np.column_stack([feats[j] for j in range(5) if j not in hidden])

        def aipw(F):
            sc = np.zeros(n)
            for rot in range(4):
                rn, re = fold_rows[roles[rot]["N"]], fold_rows[roles[rot]["E"]]
                sc[re] = pr.aipw(np.column_stack([np.ones(len(rn)), F[rn]]), a[rn], y[rn], np.column_stack([np.ones(len(re)), F[re]]), a[re], y[re])
            return sc

        def audit(hidden):
            o = {}
            for j in range(5):
                if j in hidden: continue
                F = Z(list(hidden) + [j]); gs, ses = [], []
                for rot in range(4):
                    rs_ = fold_rows[roles[rot]["S"]]
                    sc = pr.screen(np.zeros((len(rs_), 0)), F[rs_], tfeat[j][rot](rs_), a[rs_], np.eye(F.shape[1]), v4.DELTA2); gs.append(sc["gap"]); ses.append(sc["se"])
                o[j] = (float(np.mean(gs)), float(np.sqrt(np.mean(np.square(ses)) / 4)))
            return o

        gaps = audit([]); med = float(np.median([g for g, _ in gaps.values()]))
        H = [j for j, (g, s) in gaps.items() if g > med + 2 * s]
        Hs, cur = [], gaps
        while len(Hs) < 2:
            m_ = float(np.median([g for g, _ in cur.values()])); cand = [(g, j) for j, (g, s) in cur.items() if g > m_ + 2 * s and g > v4.DELTA2]
            if not cand: break
            Hs.append(max(cand)[1]); cur = audit(Hs)
        s_all, s_probe, s_seq, s_w5, s_orc = aipw(Z([])), aipw(Z(H)), aipw(Z(Hs)), aipw(Z([4])), aipw(u[:, None])
        F = Z([]); fold = np.random.default_rng(1).permutation(n) % 4; pred = np.zeros(n); X = np.column_stack([np.ones(n), F])
        for j in range(4):
            tr, te = fold != j, fold == j; b = np.linalg.solve(X[tr].T @ X[tr] + 1e-2 * np.eye(X.shape[1]), X[tr].T @ u[tr]); pred[te] = X[te] @ b
        r2 = float(1 - np.mean((u - pred) ** 2) / np.var(u))
        # ---- Algorithm 1, sealed code, raw 128 features per block
        t1 = time.time()
        view = level_a.make_view(np.zeros((n, 0)), raw, a, y)
        probe = level_a.run_probe(view, seed, "REALIMG_V9")
        est = {nm: rr_["estimate"] for nm, rr_ in probe["per_split"].items()}
        relaxed = level_a.rethreshold(probe, RELAXED, "all_rotations")["retained"]
        prep_widths = pr.Prep(view, fold_rows[roles[0]["D"]]).widths()
        algo1 = {"noise_floor": probe["v3"]["estimate"], "noise_floor_kind": probe["v3"]["return_kind"],
                 "noise_floor_retained": probe["v3"]["retained"], "noise_floor_value": probe["v3"]["floor"],
                 "fixed_t": probe["sealed"]["estimate"], "fixed_t_kind": probe["sealed"]["return_kind"],
                 "fixed_t_retained": probe["sealed"]["retained"],
                 "relaxed": (float(np.median([est[s] for s in relaxed])) if relaxed else None), "relaxed_retained": relaxed,
                 "median_all_splits": probe["median_all_splits"], "per_split_theta": est,
                 "per_split_gap": {nm: rr_["pooled_gap"] for nm, rr_ in probe["per_split"].items()},
                 "per_split_upper": {nm: rr_["pooled_upper"] for nm, rr_ in probe["per_split"].items()},
                 "prep_widths_rot0": prep_widths, "prep_all_blocks_no_balance": float(rb.all_blocks_prep(view, seed)),
                 "seconds": time.time() - t1}
        rec = {"scenario": c, "rep": r, "seed": seed, "n": n, "smoke": smoke, "true_blocks": [j + 1 for j in true_blocks], **extra,
               "R2_U_from_all": r2, "naive": float(y[a == 1].mean() - y[a == 0].mean()), "oracle": float(s_orc.mean()),
               "all_blocks": float(s_all.mean()), "audit_PROBE": float(s_probe.mean()), "H": [j + 1 for j in H],
               "audit_seq_floor": float(s_seq.mean()), "H_seq_floor": [j + 1 for j in Hs], "W5_hidden": float(s_w5.mean()),
               "gaps": {f"W{j + 1}": gaps[j] for j in range(5)}, "algo1": algo1,
               "seconds_features": t_feat, "seconds": time.time() - t0, "finished_at": datetime.now().isoformat(timespec="seconds")}
        with out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        last = rec["seconds"]
        nf = algo1["noise_floor"]; ft = algo1["fixed_t"]; rl = algo1["relaxed"]
        fmt = lambda v: "none" if v is None else f"{v:.3f}"
        rb.log(f"v9algo1[{tag}] V{c} r{r}: R2 {r2:.3f} oracle {rec['oracle']:.3f} all {rec['all_blocks']:.3f} W5hid {rec['W5_hidden']:.3f} "
               f"| audit {rec['audit_PROBE']:.3f} H{rec['H']} seq {rec['audit_seq_floor']:.3f} H{rec['H_seq_floor']} "
               f"| Algo1 floor {fmt(nf)} ({len(algo1['noise_floor_retained'])} kept) fixed {fmt(ft)} ({len(algo1['fixed_t_retained'])}) "
               f"relaxed {fmt(rl)} ({len(relaxed)}) prep-all {algo1['prep_all_blocks_no_balance']:.3f} widths {prep_widths} "
               f"| {algo1['seconds']:.0f}s algo1, {rec['seconds']:.0f}s total")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=lambda s: [int(v) for v in s.split(",")], default=[3, 0, 7])
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="r5")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--group", default="")
    a = ap.parse_args()
    nets.setup_torch(2)
    if a.smoke:
        run([0, 3, 7], 0, 1, "smoke", smoke=True)
    else:
        run(a.scenarios, a.rep_start, a.reps, a.tag, deadline=a.deadline)
