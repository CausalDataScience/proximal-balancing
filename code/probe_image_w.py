"""SCM-7B: the shared latent SCM rendered as constant-mass spatial deformations of MNIST templates.

Each proxy block is a 96x96 image.  The latent channel K_j moves the template horizontally by
8 tanh(K_j / 3); an independent uniform shift moves it vertically and carries no information.  The splat
is bilinear and conserves both total mass and the first moment, so in real arithmetic the channel is
recovered exactly from the horizontal centroid.  That inverse is evaluator-only: the learner is given the
image and nothing else, not the constants 3 and 8, not the digit label, not the template identity.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import norm

CANVAS = 96
CENTRE = 48.0
SHIFT_SCALE = 8.0
CHANNEL_SCALE = 3.0
STYLE_RANGE = 8.0
N_BLOCKS = 5
MNIST_PATH = Path(__file__).resolve().parent.parent / "materials" / "benchmarks" / "mnist" / "mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
PROXY_SD = 0.85
BAD_I = -0.6
TREATMENT_I = 2.5
OUTCOME_U = 2.5

TOLERANCES = {
    "mass_error_max": 1e-9,
    "centroid_error_max": 1e-8,
    "inverse_k_error_max": 1e-6,
    "boundary_margin_min": 2.0,
    "tanh_saturation_max": 1.0 - 1e-6,
}


@lru_cache(maxsize=1)
def bank() -> dict:
    """The frozen template bank: L1-normalised train images, their centroids, and per-class indices.

    The digest is checked on every load, so a changed or replaced file stops the run rather than silently
    altering every image in the study.
    """
    digest = hashlib.sha256(MNIST_PATH.read_bytes()).hexdigest()
    if digest != MNIST_SHA256:
        raise RuntimeError(f"MNIST bank digest {digest} does not match the frozen {MNIST_SHA256}")
    raw = np.load(MNIST_PATH)
    images = raw["x_train"].astype(np.float64)
    labels = raw["y_train"].astype(np.int64)
    mass = images.sum(axis=(1, 2))
    if not np.all(mass > 0):
        raise RuntimeError("a training template has zero mass")
    templates = images / mass[:, None, None]
    coords = np.arange(28.0)
    centroid_p = templates.sum(axis=2) @ coords
    centroid_q = templates.sum(axis=1) @ coords
    return {"templates": templates, "labels": labels, "centroid_p": centroid_p,
            "centroid_q": centroid_q, "sha256": digest,
            "class_index": [np.flatnonzero(labels == d) for d in range(10)]}


def render(k: np.ndarray, digit: np.ndarray, template: np.ndarray, style: np.ndarray) -> np.ndarray:
    """Bilinear constant-mass splat of the chosen templates onto the 96x96 canvas.

    No clipping, no quantisation, no pixel noise and no renormalisation afterwards.  Any of those would
    break the moment identity that makes the channel exactly recoverable.
    """
    b = bank()
    idx = b["class_index"]
    rows = np.array([idx[d][t] for d, t in zip(digit, template)])
    tpl = b["templates"][rows]
    n = len(rows)
    grid = np.arange(28.0)
    x = CENTRE + (grid[None, :] - b["centroid_p"][rows][:, None]) + (
        SHIFT_SCALE * np.tanh(k / CHANNEL_SCALE))[:, None]
    y = CENTRE + (grid[None, :] - b["centroid_q"][rows][:, None]) + style[:, None]
    a, bb = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    alpha, beta = x - a, y - bb
    if a.min() < 0 or bb.min() < 0 or a.max() + 1 >= CANVAS or bb.max() + 1 >= CANVAS:
        raise RuntimeError("a rendered template would fall outside the canvas")
    flat = np.zeros(n * CANVAS * CANVAS)
    base = (np.arange(n) * CANVAS * CANVAS)[:, None, None]
    for da, wx in ((0, 1.0 - alpha), (1, alpha)):
        for db, wy in ((0, 1.0 - beta), (1, beta)):
            weights = tpl * wx[:, :, None] * wy[:, None, :]
            index = base + (a + da)[:, :, None] * CANVAS + (bb + db)[:, None, :]
            flat += np.bincount(index.ravel(), weights=weights.ravel(), minlength=flat.size)
    return flat.reshape(n, CANVAS, CANVAS)


def horizontal_centroid(images: np.ndarray) -> np.ndarray:
    """First moment along the canvas's first axis, which the plan writes as the x direction.

    That axis is MNIST's row index, so in the usual way of looking at a digit the channel moves it up
    and down rather than left and right.  The name follows the plan's notation, not the screen.
    """
    return images.sum(axis=2) @ np.arange(float(CANVAS))


def invert_channel(images: np.ndarray) -> np.ndarray:
    """Evaluator-only inverse K = 3 atanh{(centroid_x - 48) / 8}, used for the information check alone."""
    return CHANNEL_SCALE * np.arctanh((horizontal_centroid(images) - CENTRE) / SHIFT_SCALE)


def latents(n: int, seed: int) -> dict:
    """The shared latent SCM plus the rendering nuisances, which carry no information about A or Y."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 9))
    u, i_cause, c_sev, e_x = z[:, :4].T
    x = u + 2.0 * e_x
    k = u[:, None] + PROXY_SD * z[:, 4:9]
    k[:, 4] += BAD_I * i_cause
    index = u + TREATMENT_I * i_cause + 0.2 * x + 0.3 * c_sev
    p = 0.1 + 0.8 * norm.cdf(index)
    a = rng.binomial(1, p).astype(float)
    y = a - OUTCOME_U * u + 0.2 * x - 0.5 * c_sev + 0.3 * rng.standard_normal(n)
    sizes = np.array([len(v) for v in bank()["class_index"]])
    digit = rng.integers(0, 10, size=(n, N_BLOCKS))
    template = (rng.random((n, N_BLOCKS)) * sizes[digit]).astype(np.int64)
    style = rng.uniform(-STYLE_RANGE, STYLE_RANGE, size=(n, N_BLOCKS))
    return {"X": x, "I": i_cause, "C": c_sev, "A": a, "Y": y, "U_eval_only": u, "K_eval_only": k,
            "p_eval_only": p, "digit": digit, "template": template, "style": style, "n": n}


def block(lat: dict, j: int, rows: np.ndarray | None = None) -> np.ndarray:
    """Render block j for the requested rows.  Images are produced on demand and never all held at once."""
    sel = slice(None) if rows is None else rows
    return render(lat["K_eval_only"][sel, j], lat["digit"][sel, j], lat["template"][sel, j],
                  lat["style"][sel, j])


def exact_channel_check(n: int = 4000, seed: int = 515151) -> dict:
    """The information-preservation gate, whose tolerances are frozen above before any pilot runs."""
    lat = latents(n, seed)
    report = {"n": n, "mnist_sha256": bank()["sha256"], "per_block": []}
    for j in range(N_BLOCKS):
        images = block(lat, j)
        k_true = lat["K_eval_only"][:, j]
        target = CENTRE + SHIFT_SCALE * np.tanh(k_true / CHANNEL_SCALE)
        report["per_block"].append({
            "block": j,
            "mass_error": float(np.abs(images.sum(axis=(1, 2)) - 1.0).max()),
            "centroid_error": float(np.abs(horizontal_centroid(images) - target).max()),
            "inverse_k_error": float(np.abs(invert_channel(images) - k_true).max()),
            "tanh_max": float(np.abs(np.tanh(k_true / CHANNEL_SCALE)).max()),
            "support_min": float(np.min(np.flatnonzero(images.sum(axis=(0, 2)) > 0))),
            "support_max": float(np.max(np.flatnonzero(images.sum(axis=(0, 2)) > 0))),
        })
    worst = {k: max(d[k] for d in report["per_block"])
             for k in ("mass_error", "centroid_error", "inverse_k_error", "tanh_max")}
    margin = min(min(d["support_min"] for d in report["per_block"]),
                 CANVAS - 1 - max(d["support_max"] for d in report["per_block"]))
    report["worst"] = worst | {"boundary_margin": margin}
    report["gates"] = {
        "mass": worst["mass_error"] <= TOLERANCES["mass_error_max"],
        "centroid": worst["centroid_error"] <= TOLERANCES["centroid_error_max"],
        "inverse_k": worst["inverse_k_error"] <= TOLERANCES["inverse_k_error_max"],
        "boundary": margin >= TOLERANCES["boundary_margin_min"],
        "tanh_tail": worst["tanh_max"] <= TOLERANCES["tanh_saturation_max"],
    }
    report["exact_channel"] = all(report["gates"].values())
    return report


# ----------------------------------------------------------------------------- image encoder
TRAIN = {"epochs": 50, "batch": 128, "lr": 3e-4, "weight_decay": 1e-4, "trunk_width": (16, 32, 64),
         "pool": 4, "embed": 32, "fit_rows": 600, "cal_rows": 150, "scale_min": 1e-3,
         "val_fraction": 0.25, "early_stopping": True}


def _torch():
    import torch
    return torch


class Trunk:
    """One 96x96 convolutional trunk shared by every block, with a separate linear head per block.

    The head predicts X, not K_j.  The learner is never given the constants that invert the renderer, nor
    the digit label, nor the template identity, so the only route from image to channel is the spatial
    deformation itself.
    """

    def __init__(self, seed: int) -> None:
        torch = _torch()
        torch.manual_seed(seed)
        nn = torch.nn
        w = TRAIN["trunk_width"]
        self.net = nn.Sequential(
            nn.Conv2d(1, w[0], 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(w[0], w[1], 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(w[1], w[2], 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((TRAIN["pool"], TRAIN["pool"])), nn.Flatten(),
            nn.Linear(w[2] * TRAIN["pool"] ** 2, TRAIN["embed"]), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(TRAIN["embed"], 1) for _ in range(N_BLOCKS)])
        self.mu = 0.0
        self.sd = 1.0

    def parameters(self):
        return list(self.net.parameters()) + list(self.heads.parameters())

    def to(self, device):
        """Move both sub-modules; the standardisation constants are Python floats and need no move."""
        self.net.to(device)
        self.heads.to(device)
        return self

    def standardise(self, images: np.ndarray) -> None:
        self.mu, self.sd = float(images.mean()), float(max(images.std(), 1e-12))

    def forward(self, images, j: int):
        torch = _torch()
        x = (images - self.mu) / self.sd
        return self.heads[j](self.net(x.reshape(-1, 1, CANVAS, CANVAS))).squeeze(-1)


def _device():
    """CUDA, then Apple MPS, then CPU.  The qualification run on a GH200 node showed the encoder training
    on the host CPU at the same speed as a laptop, with 1.3 GB of GPU memory in use: tensors were created
    without a device.  Only the arithmetic device changes here; seeds, batches, order and the stopping
    rule are untouched, so results agree to floating-point reduction order."""
    torch = _torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_encoder(lat: dict, fit_rows: np.ndarray, cal_rows: np.ndarray, seed: int) -> dict:
    """Train the shared trunk on the fit rows, then fix each block's centre and scale on the calibration rows.

    The two row sets are disjoint, so the direction and the scale never share information.  A block whose
    calibration covariance is degenerate or non-finite fails closed and the replicate returns nothing.
    """
    torch = _torch()
    torch.manual_seed(seed)
    fit_images = np.stack([block(lat, j, fit_rows) for j in range(N_BLOCKS)])
    cal_images = np.stack([block(lat, j, cal_rows) for j in range(N_BLOCKS)])
    dev = _device()
    model = Trunk(seed)
    model.standardise(fit_images)
    model.to(dev)
    fit_t = torch.tensor(fit_images, dtype=torch.float32, device=dev)
    x_t = torch.tensor(lat["X"][fit_rows], dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    generator = torch.Generator().manual_seed(seed + 1)
    # A held-out slice of the embedding rows decides when to stop.  Training longer keeps driving the
    # fitting loss down while the learned feature stops tracking the latent channel, so the epoch count
    # cannot be fixed in advance.  The split is inside D_emb and uses only X, which is observed.
    n_fit = len(fit_rows)
    n_val = int(round(TRAIN["val_fraction"] * n_fit)) if TRAIN["early_stopping"] else 0
    perm = torch.randperm(n_fit, generator=torch.Generator().manual_seed(seed + 2))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    history, val_history = [], []
    best = {"epoch": -1, "loss": float("inf"), "state": None}
    for epoch in range(TRAIN["epochs"]):
        order = tr_idx[torch.randperm(len(tr_idx), generator=generator)]
        epoch_loss = 0.0
        for start in range(0, len(order), TRAIN["batch"]):
            batch = order[start:start + TRAIN["batch"]].to(dev)
            opt.zero_grad()
            loss = sum(((model.forward(fit_t[j][batch], j) - x_t[batch]) ** 2).mean()
                       for j in range(N_BLOCKS))
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * len(batch)
        history.append(epoch_loss / len(order))
        if n_val:
            with torch.no_grad():
                vi = val_idx.to(dev)
                v = float(sum(((model.forward(fit_t[j][vi], j) - x_t[vi]) ** 2).mean()
                              for j in range(N_BLOCKS)))
            val_history.append(v)
            if v < best["loss"]:
                best = {"epoch": epoch, "loss": v,
                        "state": {k: t.detach().clone() for k, t in
                                  (list(model.net.state_dict().items())
                                   + [(f"head{i}.{k}", t) for i in range(N_BLOCKS)
                                      for k, t in model.heads[i].state_dict().items()])}}
    if n_val and best["state"] is not None:
        model.net.load_state_dict({k: v for k, v in best["state"].items()
                                   if not k.startswith("head")})
        for i in range(N_BLOCKS):
            model.heads[i].load_state_dict(
                {k.split(".", 1)[1]: v for k, v in best["state"].items()
                 if k.startswith(f"head{i}.")})
    with torch.no_grad():
        cal_t = torch.tensor(cal_images, dtype=torch.float32, device=dev)
        preds = [model.forward(cal_t[j], j).cpu().numpy().astype(np.float64) for j in range(N_BLOCKS)]
    centres, scales = [], []
    for j, pred in enumerate(preds):
        scale = float(np.cov(pred, lat["X"][cal_rows], ddof=0)[0, 1])
        if not np.isfinite(scale) or abs(scale) <= TRAIN["scale_min"]:
            return {"fail_closed": f"block {j}: calibration covariance {scale:.3e}", "history": history}
        centres.append(float(pred.mean()))
        scales.append(scale)
    return {"model": model, "centres": centres, "scales": scales, "fail_closed": None,
            "history": history, "final_loss": history[-1], "val_history": val_history,
            "stopped_at_epoch": best["epoch"], "best_val_loss": best["loss"],
            "early_stopping": bool(n_val)}


def apply_encoder(enc: dict, lat: dict, rows: np.ndarray) -> np.ndarray:
    """K-hat for every block on the requested rows, using the frozen trunk, centre and scale."""
    torch = _torch()
    dev = enc["model"].parameters()[0].device
    out = []
    with torch.no_grad():
        for j in range(N_BLOCKS):
            preds = []
            imgs = block(lat, j, rows)
            for s0 in range(0, len(rows), 512):
                images = torch.tensor(imgs[s0:s0 + 512], dtype=torch.float32, device=dev)
                preds.append(enc["model"].forward(images, j).cpu().numpy().astype(np.float64))
            pred = np.concatenate(preds)
            out.append((pred - enc["centres"][j]) / enc["scales"][j])
    return np.column_stack(out)


def channel_correlation(enc: dict, lat: dict, rows: np.ndarray) -> list[float]:
    """Evaluator-only: how well the learned channel tracks the true K_j on held-out rows."""
    k_hat = apply_encoder(enc, lat, rows)
    k_true = lat["K_eval_only"][rows]
    return [float(np.corrcoef(k_hat[:, j], k_true[:, j])[0, 1]) for j in range(N_BLOCKS)]


# ----------------------------------------------------------------------------- comparator channels
def image_statistics(images: np.ndarray, bins: int = 8) -> np.ndarray:
    """Generic summaries a practitioner would try before reaching for a network.

    Both centroids, total mass, both second central moments, the L2 norm, and a coarse intensity
    histogram.  Fixing the total mass at one removes the mean-brightness shortcut but not every
    intensity shortcut, because subpixel interpolation still couples the norm to the shift, which is
    exactly what this baseline measures.
    """
    n = len(images)
    coords = np.arange(float(CANVAS))
    mass = images.sum(axis=(1, 2))
    m0, m1 = images.sum(axis=2), images.sum(axis=1)
    c0, c1 = m0 @ coords, m1 @ coords
    v0 = m0 @ (coords ** 2) - c0 ** 2
    v1 = m1 @ (coords ** 2) - c1 ** 2
    l2 = np.sqrt((images ** 2).sum(axis=(1, 2)))
    flat = images.reshape(n, -1)
    edges = np.linspace(0.0, float(flat.max()) + 1e-12, bins + 1)
    hist = np.stack([((flat >= edges[b]) & (flat < edges[b + 1])).sum(axis=1) for b in range(bins)],
                    axis=1).astype(np.float64)
    return np.column_stack([c0, c1, mass, v0, v1, l2, hist])


def fit_statistic_channel(fit_stats: np.ndarray, fit_x: np.ndarray, cal_stats: np.ndarray,
                          cal_x: np.ndarray) -> dict | None:
    """The same centre-and-scale rule the network head gets, applied to a statistics predictor.

    The known inverse constants are never used.  A degenerate calibration covariance fails closed, so a
    baseline that cannot recover the channel abstains rather than returning a wildly scaled one.
    """
    design = np.column_stack([np.ones(len(fit_stats)), fit_stats])
    coef = np.linalg.lstsq(design, fit_x, rcond=None)[0]
    pred_cal = np.column_stack([np.ones(len(cal_stats)), cal_stats]) @ coef
    scale = float(np.cov(pred_cal, cal_x, ddof=0)[0, 1])
    if not np.isfinite(scale) or abs(scale) <= TRAIN["scale_min"]:
        return None
    return {"coef": coef, "centre": float(pred_cal.mean()), "scale": scale}


def apply_statistic_channel(fitted: dict, stats: np.ndarray) -> np.ndarray:
    pred = np.column_stack([np.ones(len(stats)), stats]) @ fitted["coef"]
    return (pred - fitted["centre"]) / fitted["scale"]
