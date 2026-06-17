"""Vignette Correction — synthetic flat correction for uncalibrated images.

GPU-accelerated via the device manager.
Generates a radial falloff model from the image center and divides it out,
compensating for optical vignetting without requiring a real flat frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)


@dataclass
class VignetteParams:
    """Parameters for synthetic vignette correction."""

    strength: float = 1.0
    center_x: float = 0.5
    center_y: float = 0.5
    radius: float = 1.0
    falloff: float = 2.0


def correct_vignette(
    image: np.ndarray,
    params: VignetteParams | None = None,
    mask: Mask | None = None,
) -> np.ndarray:
    """Apply synthetic flat-field vignette correction — GPU-accelerated."""
    if params is None:
        params = VignetteParams()

    if params.strength == 0.0:
        return image.copy()

    if image.ndim == 2:
        h, w = image.shape
    elif image.ndim == 3:
        _, h, w = image.shape
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    dm = get_device_manager()
    if dm.is_gpu:
        return _vignette_gpu(image, params, mask, h, w, dm)
    else:
        return _vignette_cpu(image, params, mask, h, w)


@torch.no_grad()
def _vignette_gpu(
    image: np.ndarray,
    params: VignetteParams,
    mask: Mask | None,
    h: int, w: int,
    dm,
) -> np.ndarray:
    """GPU vignette correction."""
    cx = params.center_x * (w - 1)
    cy = params.center_y * (h - 1)
    max_dist = np.sqrt(max(cx, w - 1 - cx) ** 2 + max(cy, h - 1 - cy) ** 2)
    max_dist = max(max_dist * params.radius, 1e-10)

    # Build distance map on GPU
    y_coords = torch.arange(h, device=dm.device, dtype=torch.float32)
    x_coords = torch.arange(w, device=dm.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    norm_dist = dist / max_dist
    correction = 1.0 + params.strength * torch.pow(norm_dist, params.falloff)

    t_img = dm.from_numpy(image)
    if image.ndim == 2:
        result = t_img * correction
    else:
        result = t_img * correction.unsqueeze(0)

    result = torch.clamp(result, 0.0, 1.0).cpu().numpy().astype(np.float32)
    return apply_mask(image, result, mask)


def _vignette_cpu(
    image: np.ndarray,
    params: VignetteParams,
    mask: Mask | None,
    h: int, w: int,
) -> np.ndarray:
    """CPU vignette correction (fallback)."""
    cx = params.center_x * (w - 1)
    cy = params.center_y * (h - 1)
    max_dist = np.sqrt(max(cx, w - 1 - cx) ** 2 + max(cy, h - 1 - cy) ** 2)
    max_dist = max(max_dist * params.radius, 1e-10)

    y_coords = np.arange(h, dtype=np.float32)
    x_coords = np.arange(w, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords)

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    norm_dist = dist / max_dist
    correction = 1.0 + params.strength * np.power(norm_dist, params.falloff)

    original = image
    if image.ndim == 2:
        result = image * correction
    else:
        result = image * correction[np.newaxis, :, :]

    result = np.clip(result, 0.0, 1.0).astype(np.float32)
    return apply_mask(original, result, mask)
