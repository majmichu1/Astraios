"""Total Generalized Variation (TGV²) Denoising — GPU-accelerated.

TGV² is a second-order variational denoising method that outperforms
Total Variation for astrophotography because it preserves smooth
gradients (nebula cores, sky background) without the staircasing
artefacts of TV. Solved via the Chambolle-Pock primal-dual algorithm.

Reference:
  Knoll F, Bredies K, Pock T, Stollberger R.
  "Second order total generalized variation (TGV) for MRI."
  Magn Reson Med. 2011;65(2):480-491.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch

from cosmica.core.device_manager import get_device_manager

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


@dataclass
class TGVParams:
    """Parameters for TGV² denoising.

    Attributes
    ----------
    strength : float
        Overall denoising strength. Higher = smoother. Range 0.1–2.0.
        Internally maps to the regularisation parameter λ.
    alpha0 : float
        TGV weight on the first-order term (penalises ∇u−p).
        Controls how much gradient smoothing is allowed. Default 1.0.
    alpha1 : float
        TGV weight on the second-order term (penalises E(p)).
        Controls how much curvature is allowed. Default 2.0.
    n_iter : int
        Number of primal-dual iterations. 100 is fast, 300 gives full
        convergence. Default 150.
    """
    strength: float = 0.5       # maps to 1/lambda (0.1 gentle … 2.0 heavy)
    alpha0: float = 1.0
    alpha1: float = 2.0
    n_iter: int = 150


# ── helpers (operate on (B, H, W) tensors for batch-channel processing) ──────

import torch.nn.functional as _F


def _grad(u: torch.Tensor) -> torch.Tensor:
    """Forward-difference gradient → shape (..., H, W, 2).

    Uses Neumann (reflective) boundary: edge differences are zero.
    """
    *_, h, w = u.shape
    u_pad = _F.pad(u, (0, 1, 0, 1), mode="replicate")
    dy = u_pad[..., 1:, :w] - u
    dx = u_pad[..., :h, 1:] - u
    return torch.stack([dy, dx], dim=-1)


def _div(p: torch.Tensor) -> torch.Tensor:
    """Adjoint of _grad (backward-difference divergence). p: (..., H, W, 2)."""
    d0 = torch.zeros_like(p[..., 0])
    d0[..., :-1, :] = p[..., :-1, 0] - p[..., 1:, 0]
    d0[..., -1, :] = -p[..., -1, 0]
    d1 = torch.zeros_like(p[..., 1])
    d1[..., :, :-1] = p[..., :, :-1, 1] - p[..., :, 1:, 1]
    d1[..., :, -1] = -p[..., :, -1, 1]
    return d0 + d1


def _sym_grad(p: torch.Tensor) -> torch.Tensor:
    """Symmetrised gradient of vector field p → (..., H, W, 3).

    Components: (∂y py, ∂x px, ½(∂x py + ∂y px))
    Neumann boundaries: edge differences are zero.
    """
    py = p[..., 0]
    px = p[..., 1]
    py_pad = _F.pad(py, (0, 0, 0, 1), mode="replicate")
    px_pad = _F.pad(px, (0, 1, 0, 0), mode="replicate")
    pyx_pad = _F.pad(py, (0, 1, 0, 0), mode="replicate")
    pxy_pad = _F.pad(px, (0, 0, 0, 1), mode="replicate")

    pyy = py_pad[1:, :] - py
    pxx = px_pad[:, 1:] - px
    pxy = 0.5 * (pyx_pad[:, 1:] - py + pxy_pad[1:, :] - px)
    return torch.stack([pyy, pxx, pxy], dim=-1)


def _div_sym(e: torch.Tensor) -> torch.Tensor:
    """Adjoint of _sym_grad. e: (..., H, W, 3) → (..., H, W, 2)."""
    e0 = e[..., 0]
    e1 = e[..., 1]
    e2 = e[..., 2]
    d0 = torch.zeros_like(e0)
    d0[..., :-1, :] = e0[..., :-1, :] - e0[..., 1:, :]
    d0[..., -1, :] = -e0[..., -1, :]
    d0[..., :, :-1] += 0.5 * (e2[..., :, :-1] - e2[..., :, 1:])
    d0[..., :, -1] += -0.5 * e2[..., :, -1]
    d1 = torch.zeros_like(e1)
    d1[..., :, :-1] = e1[..., :, :-1] - e1[..., :, 1:]
    d1[..., :, -1] = -e1[..., :, -1]
    d1[..., :-1, :] += 0.5 * (e2[..., :-1, :] - e2[..., 1:, :])
    d1[..., -1, :] += -0.5 * e2[..., -1, :]
    return torch.stack([d0, d1], dim=-1)


def _proj_ball(v: torch.Tensor, r: float) -> torch.Tensor:
    """Project v onto the pointwise L² ball of radius r."""
    norm = v.norm(dim=-1, keepdim=True).clamp_min(r)
    return v * (r / norm)


def _tgv_channel(f: torch.Tensor, params: TGVParams,
                 progress: ProgressCallback | None, ch: int, n_ch: int) -> torch.Tensor:
    """Run TGV² on a single (H, W) channel tensor, returns same shape."""
    lam = 1.0 / max(params.strength, 1e-6)

    # Step sizes from Chambolle-Pock theory (L² norm of forward op ≤ sqrt(12))
    L = 12.0 ** 0.5
    sigma = 1.0 / L
    tau   = 1.0 / L

    H, W = f.shape
    u     = f.clone()
    u_bar = f.clone()
    p     = torch.zeros(H, W, 2,  device=f.device, dtype=f.dtype)
    xi    = torch.zeros(H, W, 2,  device=f.device, dtype=f.dtype)
    eta   = torch.zeros(H, W, 3,  device=f.device, dtype=f.dtype)

    report_every = max(1, params.n_iter // 10)

    for k in range(params.n_iter):
        # ── dual updates ────────────────────────────────────────────
        xi  = _proj_ball(xi  + sigma * (_grad(u_bar) - p),   params.alpha1)
        eta = _proj_ball(eta + sigma * _sym_grad(p),          params.alpha0)

        u_old = u
        p_old = p

        # ── primal updates ───────────────────────────────────────────
        u = (u + tau * _div(xi) + tau * lam * f) / (1.0 + tau * lam)
        p = p + tau * (xi + _div_sym(eta))

        # ── over-relaxation ──────────────────────────────────────────
        u_bar = 2.0 * u - u_old
        # (p_bar used implicitly in next xi update via u_bar)

        if progress is not None and k % report_every == 0:
            frac = (ch * params.n_iter + k) / (n_ch * params.n_iter)
            progress(frac, f"TGV denoising channel {ch+1}/{n_ch} iter {k}/{params.n_iter}")

    return u.clamp(0.0, 1.0)


def tgv_denoise(
    data: "np.ndarray",
    params: TGVParams | None = None,
    progress: ProgressCallback | None = None,
) -> "np.ndarray":
    """Apply TGV² denoising to an astrophotography image.

    Parameters
    ----------
    data : ndarray
        Float32 [0,1]. Shape (H,W) or (C,H,W).
    params : TGVParams, optional
    progress : callable, optional

    Returns
    -------
    ndarray
        Denoised image, same shape and dtype as input.
    """
    import numpy as np

    if params is None:
        params = TGVParams()

    dm = get_device_manager()
    device = dm.device

    is_mono = data.ndim == 2
    arr = data[np.newaxis] if is_mono else data          # (C, H, W)
    n_ch = arr.shape[0]

    out_channels = []
    for ch in range(n_ch):
        f = torch.from_numpy(arr[ch].astype(np.float32)).to(device)
        with torch.no_grad():
            r = _tgv_channel(f, params, progress, ch, n_ch)
        out_channels.append(r.cpu().numpy())

    if progress:
        progress(1.0, "TGV denoising complete")

    result = np.stack(out_channels, axis=0)              # (C, H, W)
    if is_mono:
        result = result[0]
    return result.astype(np.float32)
