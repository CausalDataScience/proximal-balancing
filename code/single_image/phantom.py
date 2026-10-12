"""Gaussian lung phantom for single-image PROBE (plan sections 10.1 and 10.3).

Latent and observed variables, one patient per row (one image per patient):

    U ~ N(0, 1)                     severity
    L_k = U + sigma eps_k           lesion burden of lobe k, eps_k iid N(0, 1), k = 1..5
    G ~ N(0, 1)                     global brightness: 0.3 G is added to every pixel
    I ~ Bern(0.3)                   device: a thin line drawn only inside lobe 5
    X ~ N(0, 1)                     the one observed covariate
    A = 1{1.5 U + 1.0 I + 0.2 X + eps_A > 0}
    Y = tau A + h(U) + 0.3 X + eps_Y,   h(U) = U (Gaussian) or U + 0.5 U^2 (nonlinear variant), tau = 1

Rendering of lobe k (visibility v_k = L_k, or tanh(L_k) in the nonlinear variant):

    count     ~ Poisson(exp(log 5 + 0.45 v_k)), capped at 28     (5 blobs at v = 0)
    opacity   = 1.5 sigmoid(v_k) exp(0.35 xi_b)                   (per-blob jitter xi_b ~ N(0, 1))
    shape     isotropic Gaussian, sd ~ U(1.0, 2.4) px, centre uniform over the lobe's pixels
    lesion    0.6 (1 - exp(-sum_b opacity_b g_b)) inside the lobe only (semi-transparent accumulation)

Image = tissue base (lung 0.12, body 0.55, outside 0) + lesions + 0.3 G + device + N(0, 0.04^2) noise.
Count, opacity and coverage all rise with L_k, while random positions, sizes, opacity jitter, overlap and
saturation keep the burden from being read exactly.  Every random component has its own generator from
one SeedSequence, so switching one component off (for example the device) leaves all other draws intact.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.special import expit, ndtr

from . import regions as rg

GRID = rg.GRID
J = rg.J
BMAX = 28                     # blob slots per lobe; counts above are capped (P < 0.5% at sigma = 0.85)
CHUNK = 1000                  # rendering chunk (fixed, so output depends only on the seed and n)
STREAMS = ("latent", "noise_ay", "count", "blob", "opacity", "device", "pixel")


@dataclass(frozen=True)
class PhantomConfig:
    n: int
    sigma: float = 0.85
    seed: int = 0
    variant: str = "gaussian"          # "gaussian" or "nonlinear" (tanh visibility and U + 0.5 U^2 outcome)
    image_only: bool = False          # True removes X from both the SCM and the feature input
    tau: float = 1.0
    p_device: float = 0.3
    treat_u: float = 1.5
    treat_i: float = 1.0
    treat_x: float = 0.2
    out_x: float = 0.3
    global_coef: float = 0.3
    lung_base: float = 0.12
    body_base: float = 0.55
    lesion_gain: float = 0.6
    count_c0: float = math.log(5.0)
    count_c1: float = 0.45
    opacity_base: float = 1.5
    opacity_jitter: float = 0.35
    size_lo: float = 1.0
    size_hi: float = 2.4
    device_strength: float = 0.8
    device_halflength: float = 7.0
    device_width: float = 0.45
    pixel_noise: float = 0.04

    def __post_init__(self):
        if self.variant not in ("gaussian", "nonlinear"):
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.image_only:
            object.__setattr__(self, "treat_x", 0.0)
            object.__setattr__(self, "out_x", 0.0)

    @property
    def saturating_visibility(self) -> bool:
        return self.variant == "nonlinear"

    @property
    def nonlinear_outcome(self) -> bool:
        return self.variant == "nonlinear"


@dataclass
class PhantomData:
    images: np.ndarray        # (n, 64, 64) float32
    X: np.ndarray             # (n, 1) legacy; (n, 0) in image-only mode
    A: np.ndarray             # (n,) float {0, 1}
    Y: np.ndarray             # (n,)
    # evaluation-only truths: never passed to a learner
    U: np.ndarray
    L: np.ndarray             # (n, 5)
    G: np.ndarray
    I: np.ndarray             # (n,) float {0, 1}
    propensity: np.ndarray    # P(A = 1 | U, I, X)
    Y0: np.ndarray
    Y1: np.ndarray
    counts: np.ndarray        # (n, 5) blob counts actually drawn
    config: PhantomConfig

    @property
    def tau(self) -> float:
        return float(self.config.tau)

    def subset(self, rows: np.ndarray) -> "PhantomData":
        f = {k: getattr(self, k)[rows] for k in ("images", "X", "A", "Y", "U", "L", "G", "I", "propensity",
                                                 "Y0", "Y1", "counts")}
        return PhantomData(**f, config=self.config)


def _streams(seed: int) -> dict[str, np.random.Generator]:
    ss = np.random.SeedSequence([int(seed), 0x51A6E])
    return {name: np.random.default_rng(s) for name, s in zip(STREAMS, ss.spawn(len(STREAMS)))}


def outcome_shape(u: np.ndarray, cfg: PhantomConfig) -> np.ndarray:
    return u + 0.5 * u ** 2 if cfg.nonlinear_outcome else u


def visibility(lk: np.ndarray, cfg: PhantomConfig) -> np.ndarray:
    return np.tanh(lk) if cfg.saturating_visibility else lk


def propensity(u, i, x, cfg: PhantomConfig) -> np.ndarray:
    return ndtr(cfg.treat_u * u + cfg.treat_i * i + cfg.treat_x * x)


def _base_image(cfg: PhantomConfig) -> np.ndarray:
    base = np.where(rg.body_mask(), cfg.body_base, 0.0)
    base = np.where(rg.lung_mask(), cfg.lung_base, base)
    return base.astype(np.float32)


def _device_layer(rows_center: np.ndarray, angle: np.ndarray, cfg: PhantomConfig) -> np.ndarray:
    """(m, 64, 64) thin line through each centre at each angle, restricted to lobe 5."""
    yy, xx = np.mgrid[0:GRID, 0:GRID].astype(np.float32) + 0.5
    cy = rows_center[:, 0, None, None].astype(np.float32)
    cx = rows_center[:, 1, None, None].astype(np.float32)
    dy, dx = np.sin(angle)[:, None, None].astype(np.float32), np.cos(angle)[:, None, None].astype(np.float32)
    along = (xx - cx) * dx + (yy - cy) * dy
    across = -(xx - cx) * dy + (yy - cy) * dx
    line = np.exp(-0.5 * (across / cfg.device_width) ** 2) * (np.abs(along) <= cfg.device_halflength)
    return (cfg.device_strength * line * rg.masks()[rg.DEVICE_REGION - 1]).astype(np.float32)


def render(L: np.ndarray, G: np.ndarray, I: np.ndarray, cfg: PhantomConfig,
           gens: dict[str, np.random.Generator]) -> tuple[np.ndarray, np.ndarray]:
    """Images (n, 64, 64) float32 and blob counts (n, 5)."""
    n = L.shape[0]
    vis = visibility(L, cfg)
    counts = np.minimum(gens["count"].poisson(np.exp(cfg.count_c0 + cfg.count_c1 * vis)), BMAX)
    base = _base_image(cfg)
    masks = rg.masks()
    pix = [rg.region_pixels(k).astype(np.float32) for k in range(1, J + 1)]
    grid = np.arange(GRID, dtype=np.float32) + 0.5
    images = np.empty((n, GRID, GRID), dtype=np.float32)
    for c0 in range(0, n, CHUNK):
        c1 = min(n, c0 + CHUNK)
        m = c1 - c0
        lesion = np.zeros((m, GRID, GRID), dtype=np.float32)
        for k in range(J):
            # blob geometry: fixed number of draws per (image, lobe), whatever the count
            pick = gens["blob"].integers(0, len(pix[k]), size=(m, BMAX))
            jitter = gens["blob"].random((m, BMAX, 2), dtype=np.float32)
            centre = pix[k][pick] + jitter                                   # (m, B, 2) row, col
            sd = cfg.size_lo + (cfg.size_hi - cfg.size_lo) * gens["blob"].random((m, BMAX), dtype=np.float32)
            xi = gens["opacity"].standard_normal((m, BMAX), dtype=np.float32)
            alpha = (cfg.opacity_base * expit(vis[c0:c1, k])[:, None] * np.exp(cfg.opacity_jitter * xi))
            alpha = alpha * (np.arange(BMAX)[None, :] < counts[c0:c1, k][:, None])
            gy = np.exp(-0.5 * ((grid[None, None, :] - centre[..., 0:1]) / sd[..., None]) ** 2)  # (m, B, 64)
            gx = np.exp(-0.5 * ((grid[None, None, :] - centre[..., 1:2]) / sd[..., None]) ** 2)
            dens = np.matmul((alpha[..., None] * gy).transpose(0, 2, 1).astype(np.float32),
                             gx.astype(np.float32))                          # (m, 64, 64)
            lesion += masks[k] * (cfg.lesion_gain * (1.0 - np.exp(-dens)))
        img = base[None] + lesion + (cfg.global_coef * G[c0:c1])[:, None, None].astype(np.float32)
        dev_rows = np.flatnonzero(I[c0:c1] > 0.5)
        # device geometry is drawn for every row so that I does not shift any other draw
        pick = gens["device"].integers(0, len(pix[rg.DEVICE_REGION - 1]), size=m)
        jitter = gens["device"].random((m, 2), dtype=np.float32)
        angle = gens["device"].random(m) * np.pi
        if len(dev_rows) and cfg.device_strength != 0.0:
            centre = pix[rg.DEVICE_REGION - 1][pick[dev_rows]] + jitter[dev_rows]
            img[dev_rows] += _device_layer(centre, angle[dev_rows], cfg)
        img += cfg.pixel_noise * gens["pixel"].standard_normal((m, GRID, GRID), dtype=np.float32)
        images[c0:c1] = img
    return images, counts


def generate(cfg: PhantomConfig) -> PhantomData:
    gens = _streams(cfg.seed)
    n = cfg.n
    lat = gens["latent"]
    U = lat.standard_normal(n)
    eps = lat.standard_normal((n, J))
    G = lat.standard_normal(n)
    I = (lat.random(n) < cfg.p_device).astype(float)
    X = lat.standard_normal((n, 1))
    L = U[:, None] + cfg.sigma * eps
    noise = gens["noise_ay"]
    eps_a = noise.standard_normal(n)
    eps_y = noise.standard_normal(n)
    A = (cfg.treat_u * U + cfg.treat_i * I + cfg.treat_x * X[:, 0] + eps_a > 0).astype(float)
    Y0 = outcome_shape(U, cfg) + cfg.out_x * X[:, 0] + eps_y
    Y1 = Y0 + cfg.tau
    Y = np.where(A > 0.5, Y1, Y0)
    images, counts = render(L, G, I, cfg, gens)
    # Preserve the original RNG draw sequence and rendered W when only the X causes are removed.
    X_features = np.empty((n, 0), dtype=float) if cfg.image_only else X
    return PhantomData(images=images, X=X_features, A=A, Y=Y, U=U, L=L, G=G, I=I,
                       propensity=propensity(U, I, X[:, 0], cfg), Y0=Y0, Y1=Y1, counts=counts, config=cfg)


def config_dict(cfg: PhantomConfig) -> dict:
    return asdict(cfg)


def with_changes(cfg: PhantomConfig, **kw) -> PhantomConfig:
    return replace(cfg, **kw)
