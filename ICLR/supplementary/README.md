# PROBE: supplementary code

This package contains two things and nothing else:

1. the implementation of PROBE (PROxy Blockwise Exclusion), the method of the paper, with `PROBE.py` as its entry point;
2. the code that generates or prepares every data set used in the paper's experiments.

It contains no experiment runners, no stored results and no plotting code. The modules under `probe_lib/` and the
generators and loaders under `data/` are copied from the code that produced the reported results. The copies differ
only in import statements, file paths and comments, and in leaving out three things PROBE does not use: an
interpreter check, an unused overlap report, and true and false positive rates that read the known valid splits.
`PROBE.py`, `data/make_dataset.py` and `data/scm4.py` are thin wrappers around that code, written for this package.
On the computer that ran the paper's original experiments, this package reproduces the stored estimates exactly for
the same data and seeds (checked on one data set each of SCM-1, SCM-4, Twins, IHDP, ACIC, RHC and the kernel
benchmark).

## Files

| File | What it is |
|---|---|
| `PROBE.py` | Entry point: the function `probe(...)` and a command-line interface. |
| `probe_lib/core.py` | Learning, screening and AIPW estimation for one split, and the loop over splits and fold rotations for five proxy blocks. |
| `probe_lib/aggregation.py` | The linking radius (Gaussian multiplier bootstrap) and the median of the largest agreeing component. |
| `probe_lib/pooled_screen.py` | The noise-floor screen rule used for Twins, IHDP, ACIC and RHC. |
| `probe_lib/j_blocks.py` | The loop over splits for a number of blocks other than five (used for the two-block kernel benchmark). |
| `data/make_dataset.py` | Command line: writes any data set of the paper as an `.npz` file that `PROBE.py` reads. |
| `data/scm_family.py` | Generator of SCM-1, SCM-2 and SCM-3. |
| `data/shapes3d.py` | SCM-4 image bank: checks, cache, bank split, recording rule, and the image reader network. |
| `data/scm4.py` | SCM-4 data sets: renders SCM-1's proxy coordinates as Shapes3D images and reads them back. |
| `data/reader_frozen.pt` | Frozen weights of the image reader used for every SCM-4 data set (3.1 MB). |
| `data/hidden_proxy_recipe.py` | The recipe that turns IHDP and ACIC into proxy problems (Twins uses the same mechanism in `data/twins.py`). |
| `data/twins.py` | Twins loader and replicate generator. |
| `data/ihdp_acic.py` | IHDP and ACIC 2016 loaders and replicate generators. |
| `data/rhc.py` | Loader of the right heart catheterization (RHC) study. |
| `data/kspc_benchmark.py` | The kernel single-proxy benchmark of Xu and Gretton with a binary treatment. |
| `requirements.txt` | Python packages and the versions used. |

## Installation

The code was run with Python 3.14.7 and the package versions in `requirements.txt`.

```
pip install -r requirements.txt
```

`numpy` and `scipy` are all that PROBE itself needs. `pandas` is needed for Twins, IHDP and ACIC; `torch` and `h5py`
only for SCM-4; `scikit-learn` only for an optional diagnostic in `data/hidden_proxy_recipe.py`. Everything runs on a
CPU. The SCM-4 image reader uses Apple's MPS device when it is available and the CPU otherwise.

All commands below are run from the package root.

## Quick start

Generate one small SCM-1 data set (6,000 units; the paper uses 24,000) and run PROBE on 10 of its 30 splits:

```
python data/make_dataset.py scm --level SCM-1 --n 6000 --seed 1 --out example.npz
python PROBE.py example.npz --rule noise_floor --m 10 --split-seed 0
```

This takes about half a minute on one CPU and prints

```
PROBE on example.npz: n = 6000, J = 5 blocks, 10 of T = 30 splits searched, rule noise_floor, seed 8, 26.5 s
  retained 2: S13 S3
  linking radius rho = 0.100956; component sizes [2]
  estimate of the average treatment effect: 0.978074
  true effect stored in the file (not used by PROBE): 1.000000
```

Without `--m`, all 30 splits are searched, as in the paper (estimate 0.962977 for this data set). At this small
sample size the estimate is noisier than at the paper's 24,000 units. The running time depends on the machine.

## PROBE.py

### What it computes

The input is a data set of n units with covariates X, a proxy divided into J prespecified blocks W1, ..., WJ, a binary
treatment A and an outcome Y. For every split S (a nonempty proper subset of the blocks, held out; T = 2^J - 2 splits
in all), PROBE

1. **learns** a representation Z = (X, V B^T) of the kept blocks V on the representation fold. B has rank k = 2 and
   is chosen to minimise the empirical Brier discrepancy: the Brier risk of a base critic that predicts A from Z minus
   the Brier risk of an augmented critic that predicts A from (X, W_S, Z) (Definition 4 of the paper);
2. **screens** the split on a separate screen fold with freshly fitted critics (rules below);
3. **estimates** the effect with the honest AIPW score: a propensity clipped to [0.05, 0.95] and outcome regressions
   for each arm, fitted on the nuisance fold and evaluated on the evaluation fold.

The retained splits are linked when their estimates differ by at most 2 rho, and PROBE returns the median of the unique
largest connected component. A tie for the largest component, or an empty retained set, is a declared failure: the
estimate is `None`.

### How the code relates to Algorithm 1

These are the implementation choices described in the appendix of the paper; the code makes all of them.

- **Four folds, rotated.** The rows are split at random into four folds of nearly equal size. Each fold serves once
  as the representation fold, once as the screen fold, once as the nuisance fold and once as the evaluation fold. A
  split's estimate is the plain mean of its four rotations.
- **Encoder class.** X is standardised. A block with more than two coordinates is reduced to its principal directions
  above the Marchenko-Pastur edge (at least one, at most four), fitted on the representation fold only. B acts on the
  kept blocks after this step, so the class is linear and never uses a label that identifies a block. B is fitted by
  L-BFGS-B (at most 60 iterations) from two starts, and the start with the lower discrepancy is kept.
- **Critics and propensity.** Bounded probit models, P(A = 1) = 0.1 + 0.8 Phi(index), with ridge penalty 1e-4. The
  augmented critic nests the base critic. The outcome regressions are ridge least squares fits, one per arm.
- **Screen statistic.** On the screen fold, both critics are refitted by row parity (even rows fit, odd rows score,
  then the reverse). The statistic is the mean out-of-fold paired Brier difference, `gap`, with standard error `SE`.
  The overlap check requires every base-critic prediction on that fold to lie in [0.05, 0.95]; with the bounded link
  it always holds, and it is recorded.
- **Linking radius.** rho is the 95th percentile of max over splits of |xi| over 5,000 draws of xi ~ N(0, Sigma), where
  Sigma is the centred cross-product of the evaluation-fold AIPW scores of the retained splits divided by n^2.
- **Split budget.** Algorithm 1 draws m of the T splits uniformly without replacement. Every experiment in the paper
  used m = T, which is the default. With `m` below T, the m splits are drawn with `split_seed`, independently of the
  data, and only they enter the screen and the aggregation. All T splits are still fitted with the same seeds, so a
  split's result does not depend on m and the running time does not shrink with m.

### Screen rules

In every rule z = Phi^-1(1 - 0.05/T), which is 2.935 for the T = 30 splits of five blocks.

| Rule | Blocks | Retain split S when | Used in the paper for |
|---|---|---|---|
| `threshold` | J = 5 | in every one of the four rotations, max(gap, 0) + z SE < t and the overlap check passes | SCM-1 to SCM-4, with t = 0.0015, 0.0018, 0.0016, 0.0017 |
| `noise_floor` | any J | max(mean gap, 0) + z SE <= the median of z SE over the splits, where the mean is over the four rotations and SE = sqrt(sum of the four SE^2) / 4 | Twins, IHDP, ACIC, RHC |
| `significance` | J other than 5 | mean gap - z SE <= 0, with the same pooled mean and SE | kernel benchmark (J = 2) |

The thresholds of the `threshold` rule were calibrated at n = 24,000 and usually retain no split at much smaller n;
use `noise_floor` there.

### Python interface

```python
from PROBE import probe

result = probe(view, rule="noise_floor", threshold=None, seed=0, k=2, m=None, split_seed=0)
```

- `view`: a dict of numpy arrays with one row per unit: `X` (n x d, d may be 0), `W1`, ..., `WJ` (n x d_j, J >= 2),
  `A` (0 or 1) and `Y`.
- `seed` fixes the fold split, the random start of each encoder fit and the bootstrap of rho.
- `k` is the rank of B (2 in the paper). `m` and `split_seed` set the split budget; `None` means all T splits.

The result is a dict with `estimate` (a float, or `None` on failure), `return_kind` (`unique_largest`,
`tied_largest` or `no_return`), `retained` (names of the retained splits; `S13` holds out W1 and W3), `rho`,
`floor` (noise-floor rule), `splits_searched`, and `per_split`: for every split its estimate, pooled gap, pooled
standard error and overlap record. `return_records=True` adds every rotation-level record.

### Command line

```
python PROBE.py DATA.npz [--rule threshold|noise_floor|significance] [--threshold t] [--seed s]
                         [--k 2] [--m m] [--split-seed s] [--out result.json] [--records]
```

`DATA.npz` holds the arrays `X`, `W1`, ..., `WJ`, `A` and `Y`. Without `--seed`, PROBE uses the file's `probe_seed`
if there is one (files written by `data/make_dataset.py` store the seed used in the paper) and 0 otherwise.

## Data

`data/make_dataset.py` writes one data set per call as an `.npz` file with the learner's view (`X`, `W1`, ..., `WJ`,
`A`, `Y`) and two values PROBE does not use as data: `tau`, the true average effect when it is known, and
`probe_seed`. Raw files are read from `raw/<name>/` under the package root by default, or from the path given on the
command line. No raw data are included; download them from the sources below.

| Setting | Command | What it produces | Raw data |
|---|---|---|---|
| SCM-1, SCM-2, SCM-3 | `scm --level SCM-1 --rep r` | n = 24,000; X with 2, 8 or 32 columns; five blocks of (1, 1, 1, 1, 2) columns in SCM-1, 16 each in SCM-2 and 256 each in SCM-3; true effect 1 | none |
| SCM-4 | `scm4-prepare`, then `scm4 --rep r` | SCM-1 at n = 24,000 with each of the six proxy coordinates shown as a Shapes3D image and read back by the frozen reader; blocks (1, 1, 1, 1, 2); true effect 1 | Shapes3D |
| Twins | `twins --replicate r` | n = 11,984; 42 covariates; gestational age hidden as U; proxies and treatment generated from it; real potential outcomes, so the effect is known exactly | Twins |
| IHDP | `ihdp --replicate r` | n = 747; 24 covariates; the sixth covariate hidden as U; exact effect per replicate | IHDP |
| ACIC 2016 | `acic --replicate r` | n = 4,802; 78 covariates after dummy coding; `x_44` hidden as U; exact effect per realization | ACIC 2016 |
| RHC | `rhc` | n = 5,735; 55 covariate columns; ten physiological measurements in five blocks of two; treatment RHC, outcome days survived within 30; no known effect | RHC |
| Kernel benchmark | `kspc --replicate r` | n = 3,000; no covariates; W1 = exp(U) + noise, W2 = an MNIST image (784 pixels) of the digit floor(5U + 5); binary treatment; true effect 1 | MNIST |

For Twins, IHDP and ACIC, the recipe hides one real covariate as U, draws a treatment-only cause I, and generates five
proxy blocks and the treatment from them by the mechanism of SCM-1; the outcomes are the benchmark's own potential
outcomes, so the effect is known exactly.

### SCM-4 image pipeline

1. Download `3dshapes.h5` (267,573,662 bytes, MD5 `099a2078d58cec4daad0702c55d06868`) into `raw/shapes3d/`.
2. `python data/make_dataset.py scm4-prepare` checks the file, writes a 5.9 GB uint8 image cache and the fixed split of
   the 480,000 images into 200,000 reader-training, 40,000 selection and 240,000 causal-bank images. It needs `h5py`
   and about 12 GB of free disk space.
3. `python data/make_dataset.py scm4 --rep r --out scm4.npz` builds a data set with `data/reader_frozen.pt`.
4. Optional: `python data/make_dataset.py scm4-train-reader --out reader.pt` trains a reader by the procedure in the
   appendix (four convolutions and two linear layers, six epochs, epoch chosen on the selection images), to check how
   the frozen weights were made. A retrained reader can give slightly different readings, so use the frozen weights to
   rebuild the paper's data sets.

### Where to get the raw data

The sources and terms below are the ones recorded with the data used for the paper. Please check each source for its
current terms.

| Name | Put in | Source | Terms as recorded |
|---|---|---|---|
| Shapes3D | `raw/shapes3d/3dshapes.h5` | https://storage.googleapis.com/3d-shapes/3dshapes.h5 (the 3D Shapes data set) | no license recorded with our copy; see the data set's page |
| Twins | `raw/twins/twin_pairs_{X,T,Y}_3years_samesex.csv` | https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/TWINS | no license file in that repository when retrieved; derived from public-use NBER twin birth records; cite Louizos et al. (2017) and Almond, Chay and Lee (2005) |
| IHDP | `raw/ihdp/ihdp_npci_1-100.{train,test}.npz` | https://www.fredjo.com/files/ | no license stated on the distribution page; semi-synthetic benchmark of Hill (2011), distributed by Johansson, Shalit and Sontag (2016) |
| ACIC 2016 | `raw/acic2016/x.csv`, `zymu_1.csv` to `zymu_10.csv` | https://github.com/BiomedSciAI/causallib, folder `causallib/datasets/data/acic_challenge_2016` (commit 267e0c72df81ea8264ece46496107f99a599b60d) | Community Data License Agreement, Sharing, version 1.0 (the license file of that folder); cite Dorie et al. (2019) |
| RHC | `raw/rhc/rhc.csv` | https://hbiostat.org/data/repo/rhc.csv | no standard license; the site permits use with citation of the original paper, Connors et al. (1996) |
| MNIST | `raw/mnist/mnist.npz` | https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz (Keras packaging) | Creative Commons Attribution-Share Alike 3.0, as stated in the Keras documentation |

SHA-256 of the files used for the paper:

```
6ec6263e678f67205d584ebda083173c1ec44fb46ea7cfaafe3c87a31f612b86  twins/twin_pairs_X_3years_samesex.csv
fbe6eb1cad27794c61cbbd8b11b72a26e96549e59d64fb88b07b30aaf2bae07d  twins/twin_pairs_T_3years_samesex.csv
d03428d23ed308673f9151e3934aa7712df27c1fed244ed44ecc80a98a60836b  twins/twin_pairs_Y_3years_samesex.csv
750697c71b4f8d7a3aafff771b56a4ac4cd83ec649bf69afb04f8a5aee41a240  ihdp/ihdp_npci_1-100.train.npz
a70a8acbcc4e8deb677cc9bf9e9dabeb17caaa37cdbb1d7ba06be7ffb929c41c  ihdp/ihdp_npci_1-100.test.npz
0d6387ad45d23e54b11cbf967248fb68ad5e578db80d193208b8b4c98b1ed0c1  acic2016/x.csv
002fad96bb55d54edec08a0af46ef4f8ad69e0cfe6294ab48f351f22976a6e76  acic2016/zymu_1.csv
bc3a03bb461bb1629add09c01dfc77053cf36e8faf8e48a7645b35ddb4005d1c  acic2016/zymu_2.csv
8090562d3fcf5a3b02ff5493592a2880b95b106e1c12e2c3a6e6c28ae8468ff6  acic2016/zymu_3.csv
61b405c545a451b7e2ce31614150834f94aa4b27069e2755876ed6e11cfee5fd  acic2016/zymu_4.csv
e350ac76d0b8e83f45f7f04a4b76b9a32a0d53098452075617e61ce62ad0463a  acic2016/zymu_5.csv
a059950f0b51056660364eca14524a3d3254f93c608d8a1df9240c2c4179a7b5  acic2016/zymu_6.csv
ac102c3f34680ff901c30fb95985297eea0f20efaf02108f287ceec60b6a2a1f  acic2016/zymu_7.csv
bbcde6ad666640adcd39fd50cab00865d22b5c463ce9db6d0940d8eccd5afcf3  acic2016/zymu_8.csv
02663b776fa87cd3d91ee7871e98c1b4d63579957b12ef9bb1ecf367b7afc366  acic2016/zymu_9.csv
c8ee8ca537ef4315a0ee5753bd01d792a1b3cdacb1fd69e1288b2bb89944d1b7  acic2016/zymu_10.csv
9ef4ab578be4b40ad5d97d3a7e08ffdc1f9f76aeeefee51b4996e4221556f8e8  rhc/rhc.csv
731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1  mnist/mnist.npz
62e85167c38e1d6cee562f8618c5f5aadafb80d1328470fde94a652b2498a4fb  data/reader_frozen.pt (included)
```

## Running the paper's configurations

The defaults of `make_dataset.py` are the paper's seeds, and `PROBE.py` uses the seed stored in the file.

| Setting | Data sets in the tables | Data | PROBE |
|---|---|---|---|
| SCM-1 | `--rep` 0 to 19 | `scm --level SCM-1 --rep r` | `--rule threshold --threshold 0.0015` |
| SCM-2 | `--rep` 0 to 19 | `scm --level SCM-2 --rep r` | `--rule threshold --threshold 0.0018` |
| SCM-3 | `--rep` 0 to 19 | `scm --level SCM-3 --rep r` | `--rule threshold --threshold 0.0016` |
| SCM-4 | `--rep` 0 to 19 | `scm4 --rep r` | `--rule threshold --threshold 0.0017` |
| Twins | replicates 11 to 20 | `twins --replicate r` | `--rule noise_floor` |
| IHDP | replicates 12 to 21 | `ihdp --replicate r` | `--rule noise_floor` |
| ACIC 2016 | realizations 2 to 10 | `acic --replicate r` | `--rule noise_floor` |
| RHC | the one data set | `rhc` | `--rule noise_floor` |
| Kernel benchmark | replicates 1 to 20 | `kspc --replicate r` | `--rule significance` |

The figures of the paper also include data sets added later with seeds fixed in advance: SCM-1 to SCM-4 data sets
20 to 99, whose data seed 99700000 + 100000 i + 1000 rep (i = 0, 1, 2, 3 for SCM-1 to SCM-4) is passed as `--seed`,
Twins replicates 21 to 110, IHDP replicates 22 to 99 and kernel benchmark replicates 21 to 100.

For example, the first SCM-1 data set of the paper:

```
python data/make_dataset.py scm --level SCM-1 --rep 0 --out scm1_rep0.npz
python PROBE.py scm1_rep0.npz --rule threshold --threshold 0.0015
```

One data set takes from a few seconds (kernel benchmark) to about five minutes (SCM-1 at n = 24,000, Twins) on one
CPU. Results are deterministic for fixed seeds on a given machine. Other hardware or numerical libraries can change
the last digits of intermediate fits, and through them the estimates slightly.
