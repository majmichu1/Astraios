"""Chroma (colour) noise reduction.

The eye is far more sensitive to luminance detail than to colour detail, and
OSC / one-shot-colour stacks carry most of their objectionable noise in the
*colour*, not the luminance — the blotchy red/green/magenta speckle over the
background. So denoise the colour hard while leaving the luminance (and its
sharpness) almost untouched.

Splits the image into luminance + per-channel chroma (channel − luminance),
smooths the chroma on the GPU, and recombines. Mono images and the luminance
itself are returned unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from astraios.core.device_manager import get_device_manager

__all__ = ["chroma_denoise"]


def _gaussian_blur_gpu(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a 2D tensor on its current device."""
    radius = max(1, int(round(sigma * 3)))
    x = torch.arange(-radius, radius + 1, device=t.device, dtype=t.dtype)
    k = torch.exp(-(x**2) / (2.0 * sigma * sigma))
    k = k / k.sum()
    t4 = t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    kx = k.view(1, 1, 1, -1)
    ky = k.view(1, 1, -1, 1)
    t4 = F.pad(t4, (radius, radius, 0, 0), mode="reflect")
    t4 = F.conv2d(t4, kx)
    t4 = F.pad(t4, (0, 0, radius, radius), mode="reflect")
    t4 = F.conv2d(t4, ky)
    return t4.squeeze(0).squeeze(0)


def _median_gpu(t: torch.Tensor, ksize: int) -> torch.Tensor:
    """ksize x ksize median filter via shifted copies (kills isolated spots)."""
    r = ksize // 2
    h, w = t.shape
    p = F.pad(t.unsqueeze(0).unsqueeze(0), (r, r, r, r), mode="reflect")[0, 0]
    shifts = [p[dy:dy + h, dx:dx + w]
              for dy in range(ksize) for dx in range(ksize)]
    return torch.stack(shifts, dim=0).median(dim=0).values


@torch.no_grad()
def chroma_denoise(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Reduce colour noise while preserving luminance detail.

    Works on the chroma (channel minus luminance): a **median** filter first to
    remove isolated colour speckle/spots, then a light Gaussian to smooth what
    remains. The median is the important part — a pure Gaussian (the old
    behaviour) only *spreads* colour noise into soft low-amplitude blobs that
    show up on pixel-peeping, whereas the median actually removes the outliers.

    Parameters
    ----------
    image : ndarray
        ``(C, H, W)`` float32 in ``[0, 1]``. Needs ``C >= 3``; mono is returned
        unchanged.
    strength : float
        0 = no-op, 1 = moderate, higher = stronger colour cleaning (larger median
        window + Gaussian radius).

    Returns
    -------
    ndarray
        Same shape/dtype, colour-denoised.
    """
    if image.ndim != 3 or image.shape[0] < 3 or strength <= 0:
        return image.astype(np.float32, copy=False)

    dm = get_device_manager()
    t = torch.as_tensor(image, dtype=torch.float32, device=dm.device)
    r, g, b = t[0], t[1], t[2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    ksize = 5 if strength >= 1.5 else 3
    # Median removes the spots, so the Gaussian only needs to gently smooth.
    sigma = max(1.0, 1.5 + 2.5 * float(strength))

    out = t.clone()
    for c in range(3):
        chroma = t[c] - lum
        try:
            chroma = _median_gpu(chroma, ksize)  # remove colour spots/outliers
        except (RuntimeError, MemoryError):
            if dm.is_gpu:  # huge image OOM — skip median, Gaussian still helps
                torch.cuda.empty_cache()
        chroma = _gaussian_blur_gpu(chroma, sigma)
        out[c] = (lum + chroma).clamp(0.0, 1.0)
    # Extra channels (e.g. an L plane in LRGB) are left untouched.

    return out.detach().cpu().numpy().astype(np.float32)
