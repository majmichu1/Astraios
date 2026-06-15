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

from cosmica.core.device_manager import get_device_manager

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


@torch.no_grad()
def chroma_denoise(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Reduce colour noise while preserving luminance detail.

    Parameters
    ----------
    image : ndarray
        ``(C, H, W)`` float32 in ``[0, 1]``. Needs ``C >= 3``; mono is returned
        unchanged.
    strength : float
        0 = no-op, 1 = moderate, higher = stronger colour smoothing. Controls
        the chroma blur radius.

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

    # Median-style speckle knock-down then a smooth blur. Radius scales with
    # strength; colour structure in astro images is broad, so this is safe.
    sigma = max(1.5, 2.0 + 5.0 * float(strength))

    out = t.clone()
    for c in range(3):
        chroma = t[c] - lum
        chroma_s = _gaussian_blur_gpu(chroma, sigma)
        out[c] = (lum + chroma_s).clamp(0.0, 1.0)
    # Extra channels (e.g. an L plane in LRGB) are left untouched.

    return out.detach().cpu().numpy().astype(np.float32)
