# The KSPC benchmark: results

2026-09-22 23:00 CDT. `results/kspc_design/replicate_{1..20}.json`, `results/kspc_design/summary_v1.json`,
figure `figures/section5/kspc-benchmark.pdf`. Design fixed in `code/kspc_design.py` before any run.

## What was run

The authors' code, commit a5283d62 (`materials/external_code/KernelSingleProxy/PROVENANCE.md`), unchanged,
called through `code/kspc_wrapper.py`. Two things the wrapper adds, both declared in its docstring:

1. An empty stand-in for `torchvision`, which the authors' package imports only to download MNIST. Our
   MNIST is the local `materials/benchmarks/mnist/mnist.npz`, so the download is never used.
2. A bandwidth rule for a binary treatment. The authors set the Gaussian bandwidth to the median squared
   pairwise distance; for a binary treatment that median is zero and every kernel entry divides by zero.
   Only then, the median of the positive distances is used, which is 1 for a binary treatment. The
   authors' first experiment gives identical numbers with and without this rule.

**Sanity gate, passed.** On the authors' first experiment (continuous treatment, n = 1,000, their own
generator), the mean squared error of the structural function is 0.093 for SKPV and 0.149 for SPMMR, against
0.487 for a regression of Y on A alone.

**The benchmark.** The authors' generator with one change: the treatment is binary,
$P(A = 1 \mid U) = 0.1 + 0.8\{1 + \mathrm{erf}(U)\}/2$, built on their own index $\mathrm{erf}(U)$. The outcome
stays deterministic, $Y = \sin(\pi U/2) + A - 0.3$, so the effect is exactly 1. PROBE gets the authors' two
proxies as its two blocks: $W_1 = e^{U} + N(0, 0.1^2)$ and $W_2$ an MNIST image of the digit
$\lfloor 5U + 5 \rfloor$. KSPC gets one proxy at a time, with the authors' configuration for each. n = 3,000,
20 replicates.

## Result

| method | MAE | mean estimate |
|---|---|---|
| oracle (adjusts for U) | 0.004 | 0.998 |
| KSPC (SPMMR), given W1 | **0.007** | 0.994 |
| COCA, given W1 | 0.017 | 0.984 |
| KSPC (SKPV), given W1 | 0.023 | 0.977 |
| 2SLS, W2 treats | 0.025 | 1.025 |
| **PROBE** | **0.054** | 1.054 |
| 2SLS, W1 treats | 0.100 | 1.018 |
| adjust for all of W | 0.104 | 1.104 |
| COCA, given the image | 0.140 | 0.921 |
| KSPC (SKPV), given the image | 0.273 | 1.273 |
| KSPC (SPMMR), given the image | 0.555 | 1.555 |
| naive | 0.617 | 1.617 |

## Reading

On its own benchmark, with the low-dimensional proxy it was designed around, KSPC is the most accurate
method after the oracle, and it beats PROBE by a factor of 2 to 8. Handed the image instead, it is 5 to 10
times worse than PROBE. PROBE is not told which proxy to trust and lands between the two: twice as
accurate as adjusting for everything, less accurate than KSPC with the right proxy. An analyst who picked
KSPC's proxy at random would expect a mean absolute error of 0.15 (SKPV) or 0.28 (SPMMR), against PROBE's
0.054.

The honest claim is therefore about not knowing which proxy to use, not about beating KSPC when the right
proxy is known.
