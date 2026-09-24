# PRD: more replicates for the final experiments (v1)

2026-09-23 CDT. Draft for the author's approval; nothing runs before section 0 is answered. ICLR deadline
2026-09-26 06:59 CDT, about 78 hours away.

## 0. Decisions for the author

| | decision | recommendation |
|---|---|---|
| D1 | The counts and identifiers of section 2 | yes |
| D2 | Extensions of E1 and E2 run without CEVAE and without the pixel CNN, which take 70 to 90 percent of each data set's time; those two rows keep their 20 sealed dots and say so | yes |
| D3 | E7 (ACIC): download more realizations of the ACIC 2016 data, or leave E7 at its nine | leave at nine; E7 is exploratory |
| D4 | Every added replicate is reported, favourable or not, as an extension of the frozen set | required |

## 1. Why, and the risk

The figures draw 20 dots per row for E1, E2 and E8 and 9 or 10 for E5 to E7. More replicates tighten every
interval. The same tightening can turn a tie into a clear result either way: on the planted RHC design,
30 more replicates turned PROBE's tie with "adjust for X and all of W" into a clear loss. On IHDP (E6) PROBE
is third with an interval that includes a tie; more replicates will settle it, possibly against PROBE.

## 2. What is added

Every extension uses the frozen code, checked against each protocol's recorded hashes before it runs, and
identifiers fixed here before any is drawn. The sealed files are not touched.

| | now | added | identifiers | seed rule, fixed here |
|---|---|---|---|---|
| E1 SCM-1, 2, 3 | 20 each | +30 each | rep 20 to 49 | 99_500_000 + 100_000 × level index + 1_000 × rep |
| E2 SCM-4 | 20 | +30 | rep 20 to 49 | the same rule, level index 3 |
| E3 RHC | 1 | none | one real data set; more points would be resampling, not data | |
| E5 Twins | 10 | +40 | replicates 21 to 60 | the runner's own rule, `REPLICATE_SEED` + replicate |
| E6 IHDP | 10 | +40 | realizations 22 to 61 of the 100 in the npz | the runner's own rule |
| E7 ACIC | 9 | none unless D3 | the local copy holds realizations 1 to 10 only | |
| E8 KSPC benchmark | 20 | +30 | replicates 21 to 50 | `kspc_design.SEED` + replicate |

The E1 and E2 seed rule is new because the sealed rule, 1_000 per replicate inside 10_000 per level, would
collide with the next level's seeds beyond replicate 9.

**Methods on each added replicate.** Everything the sealed replicate reports, with the D2 exceptions: PROBE
under the experiment's own rule (the sealed threshold for E1 and E2, v3 for E5 and E6), adjust for X, adjust
for X and all of W, both 2SLS orientations, COCA on the clean and the contaminated block, the oracle, and
KSPC with covariates on both blocks as in the comparator runs. E8 repeats its own method list.

## 3. Cost, two workers at `nice 10`

| | per replicate, without CEVAE and pixel CNN | added | wall clock |
|---|---|---|---|
| E1 SCM-1 | about 5 minutes | 30 | about 1.3 h |
| E1 SCM-2 | about 10 minutes | 30 | about 2.5 h |
| E1 SCM-3 | about 30 minutes | 30 | about 7.5 h |
| E2 SCM-4 | about 8 minutes, images included | 30 | about 2 h |
| E5 Twins | about 3.5 minutes | 40 | about 1.2 h |
| E6 IHDP | about 1 minute | 40 | about 20 minutes |
| E8 | about 30 seconds | 30 | about 10 minutes |

About 15 hours in all. SCM-3 alone is half of it; if time runs short it is the one to cut to +10, a choice
made now and not after seeing results.

## 4. Reporting

- New write-once shards beside the sealed ones, under `results/extension_v1/<experiment>/`, each labelled
  "extension, run after the protocol was frozen".
- Each figure draws the sealed and the added replicates together, with the count per row; CEVAE and the
  pixel CNN keep 20 and are marked so.
- `EXPERIMENT_INDEX.md` reports each experiment's numbers on the sealed set, on the extension, and pooled.
  The sealed-set numbers stay the headline where a protocol named them.

## 5. Acceptance

- Code hashes match every protocol before any run; every added replicate has a result file or a recorded
  failure; nothing sealed changes; all tests pass; the index and both figures are updated.

## Status, 2026-09-23: not run

The author froze E1 to E8 as they are, for writing the paper (`results/FINAL_FREEZE_v1.json`). This
extension is not approved and nothing in it has run.
