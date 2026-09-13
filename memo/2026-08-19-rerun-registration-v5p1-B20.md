# Registration v5.1: B=20 Replication Extension (frozen before execution)

Date: 2026-08-19. User directive: rerun both registration-v5 external
experiments with B=20 repetitions. Everything else is identical to v5
(sha256 3f7fa8eb...): same DGPs, arms, estimators, seeds formula, n,
steps, decision rules. Because the per-repetition seed formula is
unchanged, repetitions 0-9 (Park) and 0-3 (dSprites) reproduce the v5
runs exactly and the new repetitions extend them; the original v5 result
files are preserved and the B=20 runs write to new files.

- T1 Park cross-world: reps 20 per world (40 runs), output
  results/external_park_crossworld_B20.json. Frozen readout: the same
  2x2 table, plus two-sided clarity on the our-world ordering
  (park vs constrained) at 20 pairs.
- T2 dSprites-as-X: reps 20 per lambda (40 runs), output
  results/external_dsprites_image_B20.json. Frozen decision rule
  unchanged: per lambda, win rate >= 0.70 and one-sided Wilcoxon
  p < 0.05 for an advantage claim; now adequately powered (20 pairs).
