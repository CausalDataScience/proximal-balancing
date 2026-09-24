# PRD: more replicates for the final experiments, on DeltaAI (v2)

2026-09-23 CDT. Supersedes v1 (`2026-09-23-replicate-extension-prd-v1.md`, never run). The author asked for this
PRD and for the work to start ("일단 PRD 쓰고 시작해"). ICLR deadline 2026-09-26 06:59 CDT.

Every action on DeltaAI that writes there (copying files, building the environment, submitting a job) waits
for the author's explicit go-ahead on the exact command, given right before it runs.

## 0. Decisions

| | decision | by |
|---|---|---|
| D1 | Fill each experiment toward 100 points where the data allow | author |
| D2 | Run on NCSA DeltaAI with any parallelism; the total charge stays within 10 GPU-hours | author |
| D3 | CEVAE and the pixel CNN are kept on the added replicates. The fallback in section 5 is fixed now, before any timing is known | this PRD |
| D4 | E7 (ACIC) stays at its 9 realizations; more would need a download, and E7 is exploratory | this PRD |
| D5 | Every added replicate is reported, whatever it shows | required |
| D6 | Figures draw sealed and added points together with the count per row; tables report the pooled numbers, and the sealed-only numbers stay in the appendix | this PRD; the author may change it only before any added result exists |

## 1. What stays untouched

`results/FINAL_FREEZE_v1.json` (271 files) and `results/FINAL_FREEZE_v1_addendum_1.json` (25 files) must verify
before and after every step. No frozen file is edited, including `EXPERIMENT_INDEX.md`. Added results go to
`results/extension_v1/<experiment>/`; the index gets a new companion file and the freeze gets
`FINAL_FREEZE_v1_addendum_2.json`.

## 2. What is added

Identifiers and seeds are fixed here, before any is drawn.

| | sealed | added | identifiers | seed | frozen function called |
|---|---|---|---|---|---|
| E1 SCM-1, 2, 3 | 20 each | 80 each | rep 20 to 99 | $99{,}700{,}000 + 100{,}000\,i + 1{,}000\,\text{rep}$, $i = 0, 1, 2$ | `run_family_v2.run_task`, thresholds 0.0015, 0.0018, 0.0016 |
| E2 SCM-4 | 20 | 80 | rep 20 to 99 | the same rule with $i = 3$ | `run_scm4.run_task`, threshold 0.0017 |
| E3 RHC bootstrap | 20 | 80 | resample 21 to 100 | $99{,}600{,}000 + b$ | `rhc_bootstrap.run_one` |
| E5 Twins | 10 | 90 | replicate 21 to 110 | the runner's rule, $98{,}210{,}000 + 1{,}000\,r + 7$ | `run_twins.run_replicate` |
| E6 IHDP | 10 | 79 | realization 22 to 100 | the runner's rule, $98{,}400{,}000 + 1{,}000\,r + 7$ | `run_ihdp_acic.run_replicate` |
| E8 KSPC benchmark | 20 | 80 | replicate 21 to 100 | the runner's rules: data $99{,}300{,}000 + r$, analysis $99{,}300{,}000 + 1{,}000\,r$ | `run_kspc_design.run_replicate` |
| KSPC rows | | 489 | every added E1, E2, E5, E6 data set | subsample $99{,}400{,}000 + \text{identifier}$ | `run_kspc_comparator.run_task` |

IHDP reaches 89, not 100: of the 100 realizations on disk, 1 was opened for the design check, 2 to 11 were run
under the first rule, and 12 to 21 are E6. A hundred would need the 1,000-realization file.

**Seeds.** The E1 and E2 range, 99,720,000 to 100,099,000, contains no seed constant used anywhere in `code/`.
Two facts found while fixing it:

- The sealed E1 rule, $97{,}200{,}000 + 10{,}000\,i + 1{,}000\,\text{rep}$, gives the same seed to SCM-1
  replicates 10 to 19 and SCM-2 replicates 0 to 9, and likewise to SCM-2 10 to 19 and SCM-3 0 to 9.
  `family_v2_dgp.generate` seeds from the seed alone, so each such pair shares its random draws. Results within
  a level are unaffected; a comparison across levels is paired on those replicates. The extension index
  discloses this; nothing sealed changes, and the new rule has no overlap.
- The frozen runners' own rules for E5, E6 and E8 reach numbers other experiments used (Twins $r \ge 90$ enters
  the planted-RHC range, IHDP $r = 100$ equals ACIC realization 0's seed, E8 $r = 100$ equals the KSPC subsample
  base). These are different data sets in different experiments and are never paired; the runners are not
  edited to avoid it.

**Two facts about the frozen code, found while checking hashes.** Every source file matches the hash its
experiment's frozen record holds, with two recorded exceptions:

- `run_kspc_comparator.py` differs from the hash in the sealed KSPC rows. It was edited after those rows were
  written (the docstring's claim about shift invariance was corrected), and the earlier version is not in git.
  A sealed row rerun through the current file must reproduce bit for bit (step A).
- `v3_fresh_scorer.py`, which scored E5 and E6, was changed at 16:54 CDT on 2026-09-22, 17 minutes after the v3
  protocol was frozen and 11 seconds after its correction (`v3_confirmation_protocol_v2.json`) was frozen, so
  that it reads the corrected protocol. The sealed E5 and E6 summaries were written at 17:14 by the current file.
  The added replicates are scored by the same file.

## 3. Methods on each added replicate

Exactly what the frozen runner computes on its sealed replicates, through the same function. PROBE uses the
experiment's own rule: the sealed threshold for E1 and E2, the v3 rule for E3, E5 and E6 (scored by
`v3_fresh_scorer`), E8's own J = 2 rule. Comparators are those each runner already computes (adjust for X,
adjust for X and all W, both 2SLS orientations, COCA on the clean and the contaminated block, CEVAE, the
oracle; on SCM-4 also the pixel CNN; on E8 its own list), plus KSPC with covariates on W1 and W4 as in the
comparator runs.

## 4. Compute on DeltaAI

**Charge.** DeltaAI charges the largest of a job's GPU, CPU and memory shares of a node: one GPU share is one
GPU, 72 cores or about 110 GB, for one hour. Our 2026-09-14 jobs (1 GPU, 120 GB) were billed 1.082 per hour.
CPU work is therefore charged too: 72 cores for an hour cost one GPU-hour.

**Devices.** The frozen code has no CUDA path. PROBE, the baselines, CEVAE, COCA and KSPC run on CPU cores,
one thread each where the runner sets one. On the Mac the SCM-4 reader and pixel CNN ran on Apple's GPU, and
the pixel CNN stops at a 2,400-second cap (2 of the 20 sealed runs hit it while the Mac was loaded; the other
18 took 4 to 6 minutes). On one CPU core it would hit the cap routinely, so the driver sets
`family_v2_images.device` to the job's CUDA device by assignment at run time. No file changes; the reader then
also runs on CUDA.

**Driver.** A new `code/run_extension_v1.py`: it verifies every source hash against the frozen protocols,
freezes `results/extension_v1/protocol_v1.json` (tasks, seeds, hashes, this PRD's hash) before any task runs,
redirects each runner's output attribute to `results/extension_v1/<experiment>/`, and runs tasks by explicit
identifier only. Tests in `code/test_extension_v1.py`.

**Environment and files.** A new virtual environment on DeltaAI with numpy 2.4.6, scipy, scikit-learn 1.9.0,
pandas 3.0.3 and a CUDA build of torch; versions recorded. Copied: `code/`, the memos the runners hash, the
frozen protocols, and the data (the Shapes3D cache of 5.9 GB with the reader files, Twins, IHDP, RHC, MNIST, the
KSPC authors' code), with a sha256 manifest checked on both ends.

## 5. Steps, budget and stop rules

| step | where | charge ceiling |
|---|---|---|
| A. Driver and tests; sealed items rerun through the driver must reproduce bit for bit | Mac | 0 |
| B. Smoke job: rerun sealed SCM-1 rep 0, SCM-4 rep 6, Twins 11, IHDP 12, E8 1, RHC resample 1 and two KSPC rows into a scratch folder; record time and peak memory per task | 1 GPU share, 60-minute limit | 1.0 |
| C. Main job, longest tasks (SCM-3) first, the rest in identifier order | 2 GPU shares (2 GPUs, 144 cores, at most 220 GB), 4-hour limit | 8.0 |
| D. Reserve for rerunning failed tasks | | 1.0 |

The time limits make the ceilings hard: the most the account can be charged is 10.0 GPU-hours. The expected
charge is about 5 to 7.

**Gate after B.** The data must be identical to the sealed data (COCA on W1 and W4 within $10^{-8}$ relative),
and PROBE must return the same kind of answer with the same retained set and an estimate within $10^{-6}$.
CEVAE and the pixel CNN train differently on different hardware; their differences from the sealed values are
recorded, not gated. A failed gate stops the extension and it is reported.

**Fallback, fixed now.** From step B's times, project step C's charge. If it exceeds 8.0, drop CEVAE from the
added E1 replicates; if it still exceeds 8.0, cut SCM-3 to the count that fits, keeping replicates 20, 21, ...
in order. If C ends at its limit with tasks unfinished, the reserve reruns them in the same order until it is
spent; what remains is reported as not run.

## 6. Reporting

- Each added shard is labelled "extension, run on DeltaAI after the protocol was frozen".
- A new `EXPERIMENT_INDEX_EXTENSION_v1.md` reports every experiment on the sealed set, on the extension and
  pooled, with the seed disclosure of section 2. New figure files; the frozen figures stay.
- Hardware differences are stated: added replicates ran on Arm Neoverse V2 cores and an H100 GPU, the sealed
  ones on an Apple M4.

## 7. Acceptance

Every planned task has a shard or a recorded failure; both freeze records verify; the smoke gate passed; the
charge from `sacct` is within 10 GPU-hours; tests pass; the extension index, figures and
`FINAL_FREEZE_v1_addendum_2.json` exist.

## Amendment 1, 2026-09-23 about 11:10 CDT, before any added E3 result from DeltaAI exists

The author's decision: on RHC, Proximal 2SLS is reported only as Cui et al. specify it, one row. Cui et al.
(2022, section 6) set Z = (pafi1, paco21), W = (ph1, hema1), put "the remaining 67 variables in X", which include
the other six physiological measures, and use a linear outcome bridge (their equation 15). The frozen runner's
2SLS kept those six measures out of X, because PROBE receives them as candidate proxies; that version is no
longer reported. `code/rhc_p2sls_cui.py` computes the specified version on the observed data (−1.641) and on
all 100 resamples, drawn by the same call and seeds as `rhc_bootstrap.py`; recomputing the runner's own 2SLS on
resamples 1 to 20 equals the sealed shards exactly, which pins the resampling. Output
`results/rhc_p2sls_cui/v1.json`. Our covariate set drops three variables more than half missing (cat2, adld3p,
urin1), so X is close to Cui et al.'s but not identical; their proximal OR is −1.80. The synthetic and
benchmark experiments keep their 2SLS rows, since there the roles and covariates are set by the design.

## Amendment 2, 2026-09-23, presentation only

The author removed Cui et al.'s published standard doubly robust estimate (−1.17) from the RHC figure; the
published proximal DR (−1.66) stays. Frozen records that contain it (`probe-rhc.pdf`, `probe-rhc-bootstrap.pdf`,
`EXPERIMENT_INDEX.md`, and the pre-registered reproduction gate of E3) are history and are not edited.

## Amendment 3, 2026-09-23 about 12:40 CDT, after the gate of step B failed

The author looked at the comparison (the same data sets on the Mac and on DeltaAI against the spread of the
sealed data sets, `scratchpad` figure `mac_vs_deltaai.png`) and decided to continue on DeltaAI: "거의
똑같은데? 이 정도는 무시하고 계속 진행". This replaces the step B gate after it was seen to fail, and the
extension index states so. The replacement, fixed before any further DeltaAI run:

- the rebuilt data equal the sealed data (COCA within $10^{-8}$ relative), except SCM-4, whose reader runs
  on CUDA (within $10^{-3}$);
- on every rerun sealed item, PROBE's reported output differs from the sealed value by less than half the
  standard deviation of that experiment's sealed PROBE outputs (SCM-1 0.0113, SCM-4 0.0126, Twins 0.0046,
  IHDP 0.247, E3 0.133, E8 0.0059);
- every task finishes without error, and the SCM-4 pixel CNN finishes within its cap.

Step B's items meet the first two (the closest call is SCM-4: a gap of 0.0093 against 0.0126). What remains: E3 and E8, which did not
run, and SCM-4's pixel CNN. The fix for the latter: each job copies the project tree, without the virtual
environment, to the node's memory (`/dev/shm`) and runs there, so images are read from memory; results are
copied back to `/work` every five minutes and at the end. No computation changes. Budget: step B used 1.0;
a recheck of E2, E3, E6 and E8 at most 0.5; the main job at most 8.0; at most 9.5 in all.

## Status

- 2026-09-23 10:05 CDT, step A passed. The driver `code/run_extension_v1.py` reran eight sealed items on the
  Mac (E6 12, E8 1, E3 resample 1, E5 11, SCM-4 rep 6, SCM-1 rep 0, and the sealed KSPC rows of SCM-1 rep 0
  and IHDP 12); all eight shards are identical to the sealed ones, bookkeeping fields aside. SCM-1 rep 0 took
  6.8 minutes, against 17.7 minutes when it was sealed. `code/summarise_extension_v1.py` reproduces every
  sealed number in `EXPERIMENT_INDEX.md`; 13 tests pass.
- 10:24 CDT, protocol frozen: `results/extension_v1/protocol_v1.json`, 649 tasks, sha256 `cb6b02d54c4d7c7f`.
- Files on DeltaAI (`/work/hdd/bgtp/yjung6/probe_ext_v1/`): 314 files, 5.9 GB, every sha256 verified there.
  Environment: Python 3.11.9, numpy 2.4.6, scipy 1.17.1 (1.18.0 has no build for Python 3.11), scikit-learn
  1.9.0, pandas 3.0.3, torch 2.14.0+cu130 (`venv_freeze.txt`).
- Step B, job 3198134 (10:43 to 11:44 CDT): stopped at its 60-minute limit, charged 1.0 GPU-hour. **The gate
  failed**, so the extension stops here, as section 5 fixes. The data rebuilt on DeltaAI are identical to the
  sealed data (COCA within $10^{-14}$ relative) except SCM-4, whose reader ran on CUDA instead of Apple's GPU
  (COCA $6.5\times10^{-5}$ relative). PROBE's estimate is not reproduced to $10^{-6}$ on any item: SCM-1 rep 0
  moved by $8.3\times10^{-3}$, SCM-4 rep 6 by $9.3\times10^{-3}$, Twins 11 (v3 output) by $1.3\times10^{-3}$,
  IHDP 12 by $1.9\times10^{-4}$, and IHDP 12's retained set changed. The source is the iterative fit inside
  each rotation: on identical data a rotation's estimate moved by up to 0.035 (SCM-1, split S23, rotation 2).
  E3 and E8 did not run (a data file, `NSW_AFDC_CS.dta`, read when `two_proxy_data` is imported, was not
  copied). The SCM-4 pixel CNN passed its 2,400-second cap because the images were read at random from the
  shared disk (reading took 540 s against 31 s on the Mac). The same items rerun on the Mac reproduce bit for
  bit (step A). Total charge so far: 1.0 GPU-hour. Waiting for the author's decision.
- 12:40 CDT, the author decided to continue (amendment 3). Sent `NSW_AFDC_CS.dta` and the staged job scripts
  (hashes verified on DeltaAI).
- Recheck, job 3198540, 15.5 minutes, 0.26 GPU-hours: E2 (SCM-4 rep 6), E3 resample 1, E6 12 and E8 1 all pass
  the replacement gate (`results/extension_v1/gate_recheck.json`); SCM-4's image reading fell from 540 to 38 s
  and its pixel CNN finished in 13 s (0.711 against the sealed 0.721). Step B under the same gate:
  `gate_step_b.json` (SCM-1, Twins, IHDP pass; SCM-4 failed only on the pixel CNN cap, now fixed).
- Step C submitted: job 3198612, two GPU shares, 4-hour limit. Charge so far 1.26 GPU-hours.
- 13:09 CDT, job 3198612 cancelled after 40:52 (1.36 GPU-hours): 100 CPU-only tasks (95 E1, 5 E6) failed with
  "CUDA error: out of memory" from `cuDevicePrimaryCtxRetain`. The job's GPUs were visible to every task, and
  each CPU task opened a CUDA context while seeding torch 2.14; about a hundred contexts filled GPU memory. No
  computation ran on a GPU in those tasks. Kept: every shard written before the cancel (E1 145, E2 16, E3 35,
  E5 6, E6 37, E8 41; KSPC rows 202), all parse; no partial file. Fix: the job hides the GPUs from CPU tasks
  (`CUDA_VISIBLE_DEVICES=""`); the driver gives each GPU task device 0 or 1, as before.
- Resubmitted as job 3198776, 3.5-hour limit, at most 7.0 GPU-hours; the charge cannot pass 1.0 + 0.26 +
  1.36 + 7.0 = 9.62 GPU-hours.
- Job 3198776 ran 13:49 to 15:10 CDT (1:21:16, 2.71 GPU-hours) and finished every remaining task but one:
  IHDP identifier 100, which does not exist (the file indexes realizations 0 to 99; the plan was off by one,
  and realization 0 was never used and was not run). Retrieved and checked: 648 added data sets and 488 KSPC
  rows, all parse, every E1 and E2 shard points to the frozen protocol. Total charge 5.33 GPU-hours.
- Written: `results/extension_v1/summary_v1.json`, the four `figures/section5/extension-*` figures,
  `EXPERIMENT_INDEX_EXTENSION_v1.md`, and the freeze record `results/FINAL_FREEZE_v1_addendum_2.json`.
