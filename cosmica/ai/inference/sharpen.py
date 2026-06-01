"""AI Sharpen — deep learning sharpening using U-Net.

Same tiled inference pattern as AI denoise, trained for deblurring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from cosmica.ai.inference.tiled import tiled_inference
from cosmica.ai.models.unet import UNet
from cosmica.core.device_manager import get_device_manager
from cosmica.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class AISharpenParams:
    """Parameters for AI sharpening."""

    strength: float = 1.0  # 0-1, blend between original and sharpened
    tile_size: int = 512
    overlap: int = 64


def ai_sharpen(
    data: np.ndarray,
    model: torch.nn.Module | None = None,
    params: AISharpenParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback | None = None,
) -> np.ndarray:
    """Sharpen an image using a trained U-Net model.

    Parameters
    ----------
    data : ndarray
        Image data, shape (H, W) or (C, H, W), float32 in [0, 1].
    model : nn.Module, optional
        Trained U-Net model. If None, falls back to traditional Richardson-Lucy.
    params : AISharpenParams, optional
        Processing parameters.
    mask : Mask, optional
        Processing mask.
    progress : callable, optional
        Progress callback.

    Returns
    -------
    ndarray
        Sharpened image.
    """
    if params is None:
        params = AISharpenParams()
    if progress is None:
        progress = _noop_progress

    if model is None:
        # No trained model — fall back to traditional Richardson-Lucy
        log.info("No AI model available, falling back to Richardson-Lucy deconvolution")
        return _traditional_sharpen(data, params, mask, progress)

    dm = get_device_manager()
    model = model.to(dm.device)
    original = data.copy()

    if data.ndim == 2:
        result = tiled_inference(
            data, model,
            tile_size=params.tile_size,
            overlap=params.overlap,
            progress=progress,
        )
    else:
        result = np.empty_like(data)
        n_ch = data.shape[0]
        for ch in range(n_ch):
            def ch_progress(frac, msg, c=ch):
                total_frac = (c + frac) / n_ch
                progress(total_frac, f"Channel {c + 1}/{n_ch}: {msg}")

            result[ch] = tiled_inference(
                data[ch], model,
                tile_size=params.tile_size,
                overlap=params.overlap,
                progress=ch_progress,
            )

    if params.strength < 1.0:
        result = data * (1 - params.strength) + result * params.strength

    result = np.clip(result, 0, 1).astype(np.float32)
    return apply_mask(original, result, mask)


def _traditional_sharpen(
    data: np.ndarray,
    params: AISharpenParams,
    mask: Mask | None,
    progress: ProgressCallback,
) -> np.ndarray:
    """Traditional Richardson-Lucy deconvolution as fallback when no AI model is available."""
    from cosmica.core.deconvolution import DeconvolutionParams, richardson_lucy

    # Estimate PSF FWHM from strength parameter
    fwhm = 1.0 + params.strength * 3.0  # 1.0 to 4.0 pixels
    dp = DeconvolutionParams(
        psf_fwhm=fwhm,
        iterations=20,
        regularization=0.005,
        deringing=True,
        deringing_amount=0.5,
    )
    return richardson_lucy(data, params=dp, mask=mask, progress=progress)


def _create_default_model() -> UNet:
    """Create a default U-Net model (untrained, for testing)."""
    return UNet(in_channels=1, out_channels=1, base_features=16, depth=3)
