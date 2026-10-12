"""Region readers psi_k (plan section 6.4) and their image-only diagnostics.

psi_k reads only region k: the input is the bounding-box crop of the standardised image with every pixel
outside region k set to 0, plus the region-k indicator (regions.region_crops).  Output d_psi = 8.  It is
trained on fold R only, from images only, to predict the fixed summaries chi_l of the regions NOT adjacent
to k through a linear head theta_k:

    L_k(psi_k) = min_theta  mean_{i in R}  sum_{l in N(k)} || chi_l(W_{l,i}) - theta_kl^T psi_k(W_{k,i}) ||^2

(trained jointly with theta by Adam; reported per target coordinate).  Fold R is split 80/20 into a
training part and a validation part used for early stopping and for the sufficiency check.

Sufficiency check: ridge prediction of the same targets on the validation part from psi_k (8 numbers)
versus from g_k, 512 fixed random convolutional features of the same crop (no pre-trained weights are
available offline, so g_k is pre-training-free).  Accept d_psi = 8 when loss(psi) <= 1.05 loss(g); else
retrain every reader at d_psi = 16.  The region's own 16 summaries chi_k are reported as a second reference.

One-factor diagnostic: first principal component of each psi_k, the 15 tetrads of their 5 x 5 covariance,
bootstrap standard errors, |tetrad| <= 2 SE flagged as consistent; Omega_k = lambda_k^2 / noise_k from a
one-factor fit.  These are diagnostics, not certificates.

Nothing in this module receives A or Y: every training entry point takes an image array and the frozen
summaries, and refuses anything else.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion, gaussian_filter
from torch import nn

from . import nets
from . import regions as rg
from .summaries import D_CHI, FixedSummaries

D_PSI = 8
RIDGE_ALPHAS = np.logspace(-4, 4, 17)
SUFFICIENCY_RATIO = 1.05
N_RANDOM_FEATURES = 512
RAW_TEXTURE_FEATURES = ("highpass_mean_square", "abs_highpass_q90", "abs_highpass_q95", "abs_highpass_q99",
                        "positive_highpass_q95", "positive_highpass_q99")
RAW_TEXTURE_BLUR_SIGMA = 1.0
RAW_TEXTURE_BOUNDARY_PIXELS = 3


def raw_texture_features(crops: np.ndarray) -> np.ndarray:
    """Fixed image-only features. No treatment/outcome/latent/device labels select these features.

    Subtract a mask-normalized local blur. Additive global brightness cancels in the residual.
    Exclude a fixed 3-pixel mask boundary before taking predetermined moments and quantiles.
    The features can expose fine image structure but are not a causal proxy certificate.
    """
    if not isinstance(crops, np.ndarray) or crops.ndim != 4 or crops.shape[1] != 2:
        raise TypeError("raw_texture_features takes only (n, 2, height, width) image/mask crops")
    image = np.asarray(crops[:, 0], float)
    mask = crops[:, 1] > .5
    denom = gaussian_filter(mask.astype(float), sigma=(0, RAW_TEXTURE_BLUR_SIGMA, RAW_TEXTURE_BLUR_SIGMA),
                            mode="constant")
    smooth = gaussian_filter(image * mask, sigma=(0, RAW_TEXTURE_BLUR_SIGMA, RAW_TEXTURE_BLUR_SIGMA),
                             mode="constant") / np.maximum(denom, 1e-12)
    residual = image - smooth
    interior = binary_erosion(mask, structure=np.ones((1, 3, 3)), iterations=RAW_TEXTURE_BOUNDARY_PIXELS)
    if np.any(interior.sum((1, 2)) == 0):
        raise ValueError("crop mask needs a nonempty interior after the fixed boundary exclusion")
    out = []
    for values in (residual[j][interior[j]] for j in range(len(image))):
        out.append([np.mean(values ** 2), *np.quantile(np.abs(values), (.90, .95, .99)),
                    *np.quantile(np.maximum(values, 0.), (.95, .99))])
    return np.asarray(out, dtype=np.float32)


class RegionReaderNet(nn.Module):
    def __init__(self, d_psi: int = D_PSI, width: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, 1, 1), nn.GELU(),
            nn.Conv2d(16, width, 3, 2, 1), nn.GELU(),
            nn.Conv2d(width, width, 3, 1, 1), nn.GELU(),
        )
        self.fc = nn.Sequential(nn.Linear(2 * width, 32), nn.GELU(), nn.Linear(32, d_psi))

    def forward(self, x):
        f = self.conv(x)
        m = nn.functional.avg_pool2d(x[:, 1:2], 2)
        mean = (f * m).sum((2, 3)) / (m.sum((2, 3)) + 1e-6)
        mx = (f * m).amax((2, 3))
        return self.fc(torch.cat([mean, mx], 1))


def _crops(z_images: np.ndarray, k: int) -> np.ndarray:
    return rg.region_crops(z_images, k)


def _predict(net: nn.Module, crops: np.ndarray, device, batch: int = 1024) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for c0 in range(0, crops.shape[0], batch):
            out.append(net(torch.from_numpy(crops[c0:c0 + batch]).to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def _train_one(crops_tr, y_tr, crops_val, y_val, d_psi, seed, device, max_epochs, patience, batch, lr, log):
    nets.seed_all(seed)
    rng = np.random.default_rng([seed, 11])
    net = RegionReaderNet(d_psi).to(device)
    head = nn.Linear(d_psi, y_tr.shape[1]).to(device)
    params = list(net.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    xv, yv = torch.from_numpy(crops_val).to(device), torch.from_numpy(y_val).to(device)
    best, best_state, bad, hist = np.inf, None, 0, []
    steps = int(np.ceil(len(y_tr) / batch))
    for ep in range(max_epochs):
        net.train(); head.train()
        order = rng.permutation(len(y_tr))
        run = 0.0
        for b in range(steps):
            idx = order[b * batch:(b + 1) * batch]
            x = torch.from_numpy(crops_tr[idx]).to(device)
            y = torch.from_numpy(y_tr[idx]).to(device)
            loss = ((head(net(x)) - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss.detach())
        net.eval(); head.eval()
        with torch.no_grad():
            val = float(((head(net(xv)) - yv) ** 2).mean())
        hist.append({"epoch": ep + 1, "train": run / steps, "val": val})
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = ({k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    return net, head, hist, best


@dataclass
class ReaderBank:
    nets: list                      # five RegionReaderNet, on CPU after training
    d_psi: int
    stats: tuple[float, float]      # image standardisation from fold R (training part)
    report: dict = field(default_factory=dict)

    def transform(self, images: np.ndarray, device=torch.device("cpu")) -> list[np.ndarray]:
        z = nets.standardise(images, self.stats)
        out = []
        for k in range(1, rg.J + 1):
            net = self.nets[k - 1].to(device)
            out.append(_predict(net, _crops(z, k), device))
            self.nets[k - 1] = net.cpu()
        return out

    def save(self, path: Path, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dicts": [n.state_dict() for n in self.nets], "d_psi": self.d_psi,
                    "stats": self.stats, "meta": meta}, path)

    @staticmethod
    def load(path: Path) -> "ReaderBank":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        bank = []
        for sd in ck["state_dicts"]:
            net = RegionReaderNet(ck["d_psi"])
            net.load_state_dict(sd)
            bank.append(net)
        return ReaderBank(bank, ck["d_psi"], tuple(ck["stats"]), {"checkpoint_meta": ck.get("meta", {})})


def reader_targets(chi: np.ndarray, k: int) -> np.ndarray:
    """Concatenated standardised summaries of the regions non-adjacent to k: (n, 16 |N(k)|)."""
    return np.concatenate([chi[:, l - 1] for l in rg.non_adjacent(k)], axis=1).astype(np.float32)


def r_split(n_r: int, seed: int, val_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng([seed, 10]).permutation(n_r)
    n_val = int(round(val_frac * n_r))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def train_readers(images_r: np.ndarray, summaries: FixedSummaries, seed: int, device: torch.device,
                  d_psi: int = D_PSI, max_epochs: int = 60, patience: int = 6, batch: int = 128,
                  lr: float = 2e-3, log=print) -> ReaderBank:
    """Train psi_1..psi_5 on fold-R images.  Only an image array and the frozen summaries come in."""
    if not isinstance(images_r, np.ndarray) or images_r.ndim != 3:
        raise TypeError("train_readers takes a (n, 64, 64) image array and nothing else")
    if not summaries.frozen:
        raise RuntimeError("summaries must be fitted on fold R and frozen first")
    t0 = time.time()
    tr, val = r_split(images_r.shape[0], seed)
    stats = (float(images_r[tr].mean()), float(images_r[tr].std()))
    z = nets.standardise(images_r, stats)
    chi = summaries.transform(images_r)
    bank, rep = [], {"regions": {}}
    for k in range(1, rg.J + 1):
        crops = _crops(z, k)
        y = reader_targets(chi, k)
        net, head, hist, best = _train_one(crops[tr], y[tr], crops[val], y[val], d_psi, seed * 100 + k, device,
                                           max_epochs, patience, batch, lr, log)
        bank.append(net.cpu())
        rep["regions"][str(k)] = {"targets": list(rg.non_adjacent(k)), "n_targets": int(y.shape[1]),
                                  "epochs_run": len(hist), "best_val_mse_per_coordinate": best,
                                  "final_train_mse_per_coordinate": hist[-1]["train"]}
        log(f"    reader psi_{k} (d_psi={d_psi}): targets N({k})={rg.non_adjacent(k)}, "
            f"{len(hist)} epochs, val MSE/coord {best:.4f}")
    rep.update({"d_psi": d_psi, "n_train": int(len(tr)), "n_val": int(len(val)), "device": str(device),
                "seconds": round(time.time() - t0, 1), "parameters_per_reader": nets.count_parameters(bank[0])})
    return ReaderBank(bank, d_psi, stats, rep)


# ----------------------------------------------------------------------------- sufficiency

def random_conv_features(crops: np.ndarray, seed: int = 4242) -> np.ndarray:
    """g_k: 512 fixed random features, 32 random 5 x 5 filters on the two-channel crop, ReLU, 4 x 4 average
    pooling grid (32 x 16 = 512).  Fixed seed, never trained."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(32, 2, 5, 5, generator=gen) / np.sqrt(50.0)
    b = 0.1 * torch.randn(32, generator=gen)
    out = []
    with torch.no_grad():
        for c0 in range(0, crops.shape[0], 2048):
            x = torch.from_numpy(crops[c0:c0 + 2048])
            f = torch.relu(nn.functional.conv2d(x, w, b, padding=2))
            out.append(nn.functional.adaptive_avg_pool2d(f, (4, 4)).flatten(1).numpy())
    return np.concatenate(out).astype(np.float64)


def _ridge_loss(f_tr, y_tr, f_val, y_val) -> float:
    from sklearn.linear_model import RidgeCV
    mu, sd = f_tr.mean(0), f_tr.std(0)
    keep = sd > 1e-6 * max(1.0, float(np.abs(mu).max()))    # drop features that are constant on R-train
    f_tr, f_val, mu, sd = f_tr[:, keep], f_val[:, keep], mu[keep], sd[keep]
    model = RidgeCV(alphas=RIDGE_ALPHAS).fit((f_tr - mu) / sd, y_tr)
    return float(np.mean((model.predict((f_val - mu) / sd) - y_val) ** 2))


def sufficiency_check(bank: ReaderBank, images_r: np.ndarray, summaries: FixedSummaries, seed: int) -> dict:
    tr, val = r_split(images_r.shape[0], seed)
    psi = bank.transform(images_r)
    z = nets.standardise(images_r, bank.stats)
    chi = summaries.transform(images_r)
    out = {}
    for k in range(1, rg.J + 1):
        y = reader_targets(chi, k).astype(np.float64)
        g = random_conv_features(_crops(z, k))
        own = chi[:, k - 1].astype(np.float64)
        lp = _ridge_loss(psi[k - 1][tr], y[tr], psi[k - 1][val], y[val])
        lg = _ridge_loss(g[tr], y[tr], g[val], y[val])
        lo = _ridge_loss(own[tr], y[tr], own[val], y[val])
        out[str(k)] = {"loss_psi": lp, "loss_random_conv_512": lg, "loss_own_chi_16": lo,
                       "target_variance": float(y[val].var(0).mean()),
                       "ratio_psi_over_g": lp / lg, "accept": bool(lp <= SUFFICIENCY_RATIO * lg)}
    out["all_accept"] = all(v["accept"] for k, v in out.items() if k != "all_accept")
    return out


# ----------------------------------------------------------------------------- one-factor diagnostic

def first_pcs(psi_blocks: list[np.ndarray], anchors: list[np.ndarray],
              component: int = 0) -> tuple[np.ndarray, list[float]]:
    """Standardised principal component (`component` = 0 is the first, the plan's diagnostic) of each
    block, signed to correlate positively with an image-only anchor (the region's own mean intensity for
    the first component; for later components the sign is fixed by the largest loading)."""
    cols, share = [], []
    for p, a in zip(psi_blocks, anchors):
        s = (p - p.mean(0)) / (p.std(0) + 1e-12)
        _, sv, vt = np.linalg.svd(s, full_matrices=False)
        v = vt[component]
        f = s @ v
        flip = np.corrcoef(f, a)[0, 1] < 0 if component == 0 else v[np.argmax(np.abs(v))] < 0
        if flip:
            f = -f
        cols.append((f - f.mean()) / f.std())
        share.append(float(sv[component] ** 2 / (sv ** 2).sum()))
    return np.column_stack(cols), share


def tetrads(cov: np.ndarray) -> tuple[np.ndarray, list[str]]:
    vals, labels = [], []
    for i, j, k, l in itertools.combinations(range(cov.shape[0]), 4):
        for (a, b), (c, d), (e, f), (g, h) in (((i, j), (k, l), (i, k), (j, l)),
                                               ((i, j), (k, l), (i, l), (j, k)),
                                               ((i, k), (j, l), (i, l), (j, k))):
            vals.append(cov[a, b] * cov[c, d] - cov[e, f] * cov[g, h])
            labels.append(f"s{a + 1}{b + 1}s{c + 1}{d + 1}-s{e + 1}{f + 1}s{g + 1}{h + 1}")
    return np.array(vals), labels


def one_factor_omega(f: np.ndarray) -> dict:
    """Omega_k = lambda_k^2 / noise_k from (i) a one-factor maximum-likelihood fit and (ii) the triad
    median lambda_k^2 = median_{l,m} s_kl s_km / s_lm, both on standardised columns."""
    from sklearn.decomposition import FactorAnalysis
    s = (f - f.mean(0)) / f.std(0)
    fa = FactorAnalysis(n_components=1, random_state=0).fit(s)
    lam2 = fa.components_[0] ** 2
    om_fa = lam2 / fa.noise_variance_
    c = np.corrcoef(s, rowvar=False)
    om_tri = []
    for k in range(c.shape[0]):
        others = [x for x in range(c.shape[0]) if x != k]
        tri = [c[k, l] * c[k, m] / c[l, m] for l, m in itertools.combinations(others, 2) if abs(c[l, m]) > 1e-6]
        l2 = float(np.clip(np.median(tri), 1e-6, 1 - 1e-6))
        om_tri.append(l2 / (1 - l2))
    return {"omega_fa": om_fa.tolist(), "omega_triad": om_tri, "loadings_sq_fa": lam2.tolist()}


def one_factor_diagnostic(f: np.ndarray, n_boot: int, seed: int) -> dict:
    cov = np.cov(f, rowvar=False)
    tv, labels = tetrads(cov)
    om = one_factor_omega(f)
    rng = np.random.default_rng([seed, 21])
    boot_t, boot_o = [], []
    n = f.shape[0]
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_t.append(tetrads(np.cov(f[idx], rowvar=False))[0])
        boot_o.append(one_factor_omega(f[idx])["omega_fa"])
    se = np.array(boot_t).std(0, ddof=1)
    z = tv / se
    return {"n": int(n), "n_boot": n_boot, "cov": cov.tolist(), "tetrads": tv.tolist(), "tetrad_labels": labels,
            "tetrad_se": se.tolist(), "tetrad_z": z.tolist(), "n_within_2se": int(np.sum(np.abs(z) <= 2.0)),
            "n_tetrads": int(len(tv)), "max_abs_z": float(np.max(np.abs(z))),
            "one_factor_consistent": bool(np.all(np.abs(z) <= 2.0)),
            **om, "omega_fa_se": np.array(boot_o).std(0, ddof=1).tolist()}


# ----------------------------------------------------------------------------- evaluation-only diagnostics

def _cv_r2(f: np.ndarray, y: np.ndarray) -> float:
    from sklearn.linear_model import LinearRegression
    n = len(y)
    half = np.arange(n) % 2 == 0
    pred = np.empty(n)
    for a, b in ((half, ~half), (~half, half)):
        pred[b] = LinearRegression().fit(f[a], y[a]).predict(f[b])
    return float(1.0 - np.mean((y - pred) ** 2) / y.var())


def _cv_auc(f: np.ndarray, y: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    n = len(y)
    half = np.arange(n) % 2 == 0
    pred = np.empty(n)
    mu, sd = f.mean(0), f.std(0) + 1e-12
    s = (f - mu) / sd
    for a, b in ((half, ~half), (~half, half)):
        pred[b] = LogisticRegression(max_iter=2000).fit(s[a], y[a]).decision_function(s[b])
    return float(roc_auc_score(y, pred))


def evaluation_diagnostics(psi_blocks: list[np.ndarray], U, L, G, I) -> dict:
    """Uses the generator's truths; evaluation only, never fed back.  Two-fold cross-fitted linear R^2
    (and AUC for the device indicator)."""
    out = {"per_region": {}}
    for k in range(1, rg.J + 1):
        p = psi_blocks[k - 1]
        r2u = _cv_r2(p, U)
        out["per_region"][str(k)] = {"r2_L_k": _cv_r2(p, L[:, k - 1]), "r2_U": r2u, "r2_G": _cv_r2(p, G),
                                     "omega_from_r2_U": r2u / (1.0 - r2u), "auc_I": _cv_auc(p, I)}
    allp = np.column_stack(psi_blocks)
    out["r2_U_all_psi"] = _cv_r2(allp, U)
    out["r2_U_all_true_L"] = _cv_r2(L, U)
    out["r2_U_single_true_L"] = [float(_cv_r2(L[:, [k]], U)) for k in range(rg.J)]
    out["auc_I_all_psi"] = _cv_auc(allp, I)
    return out
