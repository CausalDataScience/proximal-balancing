"""Kernel single proxy control (Xu and Gretton, arXiv 2308.04585) as a comparator, using the authors' code
unchanged from materials/external_code/KernelSingleProxy (commit a5283d62, see PROVENANCE.md).

KSPC takes one proxy W, the treatment A and the outcome Y, and returns the structural function
f(a) = E[Y(a)].  For a binary treatment the average effect is f(1) - f(0).  It takes no covariates, so on a
data set with X it is run without X and reported as such.  Its identifying condition is a deterministic
outcome; on data with outcome noise it runs outside that condition, and every caption says so.

Hyperparameters are the authors' own: `configs/skpv.json` (the regularisation grid, both stages on all
the data) for a low-dimensional proxy and `configs/skpv_mnist.json` for an image proxy; likewise for SPMMR.
The authors' sample split calls scikit-learn without a seed, so the numpy global seed is set before each
fit to make runs reproducible.

    python3 -B kspc_wrapper.py        # the sanity gate on the authors' own first experiment
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KSPC_ROOT = HERE.parent / "materials" / "external_code" / "KernelSingleProxy"
if str(KSPC_ROOT) not in sys.path:
    sys.path.insert(0, str(KSPC_ROOT))

# The authors' package imports torchvision only to download MNIST.  That download is never used here (our
# images come from materials/benchmarks/mnist/mnist.npz), and torchvision is not installed, so an empty
# stand-in is registered for exactly the two names it imports.  The authors' files are not edited.
if "torchvision" not in sys.modules:
    import types
    _tv = types.ModuleType("torchvision")
    _tv.datasets = types.SimpleNamespace(MNIST=None)
    _tv.transforms = types.SimpleNamespace()
    sys.modules["torchvision"] = _tv
    sys.modules["torchvision.datasets"] = _tv.datasets
    sys.modules["torchvision.transforms"] = _tv.transforms

from src.data.data_class import SingleProxyTrainDataSet  # noqa: E402
from src.models.SKPV.skpv_model import SKPVModel  # noqa: E402
from src.models.SPMMR.spmmr_model import SPMMRModel  # noqa: E402
from src.utils import kernel_func as _kf  # noqa: E402

# The authors' Gaussian kernel sets its bandwidth to the median squared pairwise distance.  For a binary
# treatment more than half of those distances are zero, the median is zero, and every kernel entry divides
# by zero.  Only in that case, the median of the positive distances is used instead; for a binary treatment
# that is 1, so k(0, 1) = exp(-1) and any function on {0, 1} is representable.  Continuous inputs, whose
# median is positive, are untouched.  The authors' file is not edited; this wraps its method.
_authors_fit = _kf.GaussianKernel.fit


def _fit_with_positive_median(self, data, scale=1.0, **kwargs):
    _authors_fit(self, data, scale=scale, **kwargs)
    if not self.sigma > 0:
        from scipy.spatial.distance import cdist
        d = cdist(data, data, "sqeuclidean")
        self.sigma = (float(np.median(d[d > 0])) if np.any(d > 0) else 1.0) * scale


_kf.GaussianKernel.fit = _fit_with_positive_median


def _config(name: str) -> dict:
    return json.loads((KSPC_ROOT / "configs" / f"{name}.json").read_text())["model"]


# ----------------------------------------------------------------------------- Appendix A, our implementation
#
# Xu and Gretton, Appendix A, Theorems A.6 and A.7: with an observed confounder X every kernel matrix is
# multiplied elementwise by a kernel on X.
#   A.6  B = (K_AA * K_YY * K_XX + n lam I)^{-1} (K_{A A.} * K_{Y Y.} * K_{X X.}),
#        M = K_{A. A.} * (B' K_WW B) * K_{X. X.},  alpha = (M + m eta I)^{-1} y.,
#        h(a, w, x) = alpha' {k_A.(a) * (B' k_W(w)) * k_X.(x)}.
#   A.7  L = K_AA * K_WW * K_XX,  G = K_AA * K_YY * K_XX,  alpha = sqrt(G) (sqrt(G) L sqrt(G) + n^2 eta I)^{-1} sqrt(G) y.
# Steps the appendix leaves implicit, decided here and nowhere else:
#   f(a) is the empirical average of h(a, W_i, X_i) over the observed pairs (W_i, X_i), the joint version of
#       the authors' own no-X code, which averages over the observed W_i;
#   the regularisation search and the validation score of each method gain the same factor K_XX;
#   X is standardised and gets the authors' Gaussian kernel with the median bandwidth;
#   every other line, including the order of random calls, follows the authors' fit() so that a constant X
#       reproduces their estimator exactly (gate 1 below).
# The authors' files are not edited: these classes subclass theirs.

from sklearn.model_selection import train_test_split  # noqa: E402

from src.models.SKPV.skpv_model import get_kernel_func as _skpv_kernels  # noqa: E402
from src.models.SPMMR.spmmr_model import get_kernel_func as _spmmr_kernels  # noqa: E402


def _split(n: int, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """The authors' split_train_data, returning indices, with the same random call."""
    if ratio < 0.0:
        idx = np.arange(n)
        return idx, idx
    first, second = train_test_split(np.arange(n), train_size=ratio)
    return first, second


class SKPVWithX(SKPVModel):
    def fit_x(self, a, w, y, x, data_name: str):
        i1, i2 = _split(len(a), self.split_ratio)
        kt, kw, ky = _skpv_kernels(data_name)
        kx = _kf.GaussianKernel()
        self.treatment_kernel_func, self.proxy_kernel_func, self.outcome_kernel_func = kt, kw, ky
        n1, n2 = len(i1), len(i2)
        self.train_data_2nd = SingleProxyTrainDataSet(treatment=a[i2], outcome=y[i2], proxy=w[i2])
        kt.fit(a, scale=self.scale); kw.fit(w, scale=self.scale); ky.fit(y, scale=self.scale); kx.fit(x, scale=self.scale)
        KA11, KY11, KW11, KX11 = (k.cal_kernel_mat(v[i1], v[i1]) for k, v in ((kt, a), (ky, y), (kw, w), (kx, x)))
        K = KA11 * KY11 * KX11
        KA12, KY12, KX12 = (k.cal_kernel_mat(v[i1], v[i2]) for k, v in ((kt, a), (ky, y), (kx, x)))
        K2 = KA12 * KY12 * KX12
        KA22, KX22 = kt.cal_kernel_mat(a[i2], a[i2]), kx.cal_kernel_mat(x[i2], x[i2])
        if self.lam1 is None:
            self.tune_lam1(K, KW11)
        B = np.linalg.solve(K + self.lam1 * n1 * np.eye(n1), K2)
        B_tilde = np.linalg.solve(K + self.lam1 * n1 * np.eye(n1), K)
        M = KA22 * (B.T @ KW11 @ B) * KX22
        M_tilde = KA12.T * (B_tilde.T @ KW11) * KX12.T
        self.w = np.mean((B.T @ KW11) * KX12.T, axis=1, keepdims=True)
        if self.lam2 is None:
            self.tune_lam2(M, M_tilde, y[i2], y[i1])
        self.alpha = np.linalg.solve(M + self.lam2 * n2 * np.eye(n2), y[i2])
        return self


class SPMMRWithX(SPMMRModel):
    def fit_x(self, a, w, y, x, data_name: str):
        kt, kw, ky = _spmmr_kernels(data_name)
        kx = _kf.GaussianKernel()
        self.treatment_kernel_func, self.proxy_kernel_func, self.outcome_kernel_func = kt, kw, ky
        kt.fit(a, scale=self.scale); kw.fit(w, scale=self.scale); ky.fit(y, scale=self.scale); kx.fit(x, scale=self.scale)
        iv = None
        idx = np.arange(len(a))
        if self.val_ratio > 0:
            iv, idx = _split(len(a), self.val_ratio)
        self.train_data = SingleProxyTrainDataSet(treatment=a[idx], outcome=y[idx], proxy=w[idx])
        n = len(idx)
        KAA, KYY, KWW, KXX = (k.cal_kernel_mat(v[idx], v[idx]) for k, v in ((kt, a), (ky, y), (kw, w), (kx, x)))
        G = KAA * KYY * KXX
        L = KAA * KWW * KXX
        D, V = np.linalg.eigh(G)
        sqG = V @ np.diag(np.sqrt(np.maximum(D, 0.0))) @ V.T
        if self.lam is None:
            k_test = kt.cal_kernel_mat(a[idx], a[iv]) * kw.cal_kernel_mat(w[idx], w[iv]) * kx.cal_kernel_mat(x[idx], x[iv])
            Wv = kt.cal_kernel_mat(a[iv], a[iv]) * ky.cal_kernel_mat(y[iv], y[iv]) * kx.cal_kernel_mat(x[iv], x[iv])
            Kq, kyv = sqG @ L @ sqG, sqG @ y[idx]
            scores = {}
            for lam in np.logspace(np.log10(self.lam_min), np.log10(self.lam_max), self.n_lam_search):
                al = sqG @ np.linalg.solve(Kq + lam * n * n * np.eye(n), kyv)
                err = y[iv] - k_test.T.dot(al)
                scores[lam] = float(np.asarray(err.T.dot(Wv.dot(err)) + lam * n * n * al.T.dot(L.dot(al))).item())
            self.lam = min(scores, key=scores.get)
        self.alpha = sqG @ np.linalg.solve(sqG @ L @ sqG + self.lam * n * n * np.eye(n), sqG @ y[idx])
        self.proxy_mean_vec = np.mean(KWW * KXX, axis=1, keepdims=True)
        return self


def kspc_effect(a: np.ndarray, w: np.ndarray, y: np.ndarray, *, method: str = "skpv", image: bool = False,
                seed: int = 0, grid: np.ndarray | None = None, x: np.ndarray | None = None) -> dict:
    """f(1) - f(0) for a binary treatment, or f on `grid` for a continuous one."""
    a, w, y = (np.asarray(v, float).reshape(len(v), -1) for v in (a, w, y))
    data = SingleProxyTrainDataSet(treatment=a, outcome=y, proxy=w)
    name = {"skpv": "skpv_mnist" if image else "skpv", "spmmr": "spmmr_mnist" if image else "spmmr"}[method]
    params = {k: v for k, v in _config(name).items() if k != "name"}
    np.random.seed(seed)
    if x is None:
        model = SKPVModel(**params) if method == "skpv" else SPMMRModel(**params)
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(data, "mnist" if image else "synthetic")
    else:
        x = np.asarray(x, float).reshape(len(a), -1)
        x = (x - x.mean(0)) / (x.std(0) + 1e-12)
        model = SKPVWithX(**params) if method == "skpv" else SPMMRWithX(**params)
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit_x(a, w, y, x, "mnist" if image else "synthetic")
    if grid is not None:
        return {"f": model.predict(np.asarray(grid, float).reshape(-1, 1)).ravel(), "config": name}
    f = model.predict(np.array([[0.0], [1.0]])).ravel()
    return {"ate": float(f[1] - f[0]), "f0": float(f[0]), "f1": float(f[1]), "config": name}


def sanity_gate(n: int = 1000, seed: int = 42) -> dict:
    """The authors' first experiment, generated by their own code: KSPC must beat a regression of Y on A
    alone, which ignores the confounding, on the mean squared error of the structural function."""
    from src.data.generate_synthetic import generate_synthetic_test_data, generate_synthetic_train_data
    train, test = generate_synthetic_train_data(n, seed=seed), generate_synthetic_test_data()
    out = {}
    for method in ("skpv", "spmmr"):
        f = kspc_effect(train.treatment, train.proxy, train.outcome, method=method, seed=seed, grid=test.treatment)["f"]
        out[method] = float(np.mean((f - test.structural.ravel()) ** 2))
    from sklearn.kernel_ridge import KernelRidge
    kr = KernelRidge(alpha=1e-3, kernel="rbf", gamma=1.0).fit(train.treatment, train.outcome.ravel())
    out["regression_on_A_only"] = float(np.mean((kr.predict(test.treatment) - test.structural.ravel()) ** 2))
    out["pass"] = bool(max(out["skpv"], out["spmmr"]) < 0.5 * out["regression_on_A_only"])
    return out


def gate_reduction(n: int = 800, seed: int = 7) -> dict:
    """PRD v2 gate 1: with X constant, the covariate version must equal the authors' estimator."""
    import kspc_design as kd
    d = kd.generate(1)
    a, w, y = d["A"][:n], d["W1"][:n], d["Y"][:n]
    out = {}
    for method in ("skpv", "spmmr"):
        ref = kspc_effect(a, w, y, method=method, seed=seed)
        cov = kspc_effect(a, w, y, method=method, seed=seed, x=np.zeros((n, 1)))
        out[method] = max(abs(ref["f0"] - cov["f0"]), abs(ref["f1"] - cov["f1"]))
    out["pass"] = bool(max(out["skpv"], out["spmmr"]) < 1e-8)
    return out


def gate_validity(n: int = 1000, seed: int = 42) -> dict:
    """PRD v2 gate 2: the authors' first experiment with an observed confounder X added; the covariate
    version must at least halve the structural-function error of the version that ignores X."""
    from scipy import special
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, 1))
    u = rng.uniform(-1.0, 1.0, (n, 1))
    a = special.erf(u) + 0.5 * x + rng.normal(0.0, 0.1, (n, 1))
    w = np.exp(u) + rng.normal(0.0, 0.1, (n, 1))
    y = np.sin(np.pi * u / 2.0) + a ** 2 - 0.3 + 0.5 * x
    grid = np.linspace(-1.0, 1.0, 100)
    truth = grid ** 2 - 0.3
    out = {}
    for method in ("skpv", "spmmr"):
        f0 = kspc_effect(a, w, y, method=method, seed=seed, grid=grid)["f"]
        fx = kspc_effect(a, w, y, method=method, seed=seed, grid=grid, x=x)["f"]
        out[method] = {"mse_without_x": float(np.mean((f0 - truth) ** 2)), "mse_with_x": float(np.mean((fx - truth) ** 2))}
    out["pass"] = bool(all(v["mse_with_x"] <= 0.5 * v["mse_without_x"] for v in out.values() if isinstance(v, dict)))
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gates":
        import run_family_v2 as rf
        out_path = HERE.parent / "results" / "kspc_comparator" / "gates_v1.json"
        if out_path.exists():
            raise SystemExit(f"{out_path} exists; write-once")
        g1, g2 = gate_reduction(), gate_validity()
        rec = {"artifact": "kspc_comparator_gates", "generated_utc": rf.now(), "prd": "memo/2026-09-22-kspc-comparator-prd-v2.md",
               "gate_1_reduction": g1, "gate_2_validity": g2, "pass": bool(g1["pass"] and g2["pass"]),
               "source_sha256": {"kspc_wrapper.py": rf.sha256(HERE / "kspc_wrapper.py")}}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "x") as handle:
            handle.write(json.dumps(rec, indent=1) + "\n")
        print(json.dumps(rec, indent=1))
    else:
        print(json.dumps(sanity_gate(), indent=1))
