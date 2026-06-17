"""Background Neutralization — shift sky background to neutral zero.

Measures the background level per channel using the darkest percentile of
pixels (robust against stars and nebulosity) and subtracts it so the sky
background is flat at 0.  Equivalent to PixInsight's BackgroundNeutralization
in statistical mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(f: float, m: str) -> None:
    pass


@dataclass
class BackgroundNeutralizationParams:
    """Parameters for background neutralization."""

    percentile: float = 2.0    # darkest X% used to estimate sky background
    amount: float = 1.0        # blend strength (0 = no change, 1 = full correction)
    protect_bright: float = 0.5  # ignore pixels above this fraction of the maximum value when sampling


def background_neutralization(
    image: np.ndarray,
    params: BackgroundNeutralizationParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Subtract per-channel sky background so the background level is ~0.

    Parameters
    ----------
    image : ndarray
        Float32 image, shape (H, W) or (C, H, W), values in [0, 1].
    params : BackgroundNeutralizationParams, optional
        Parameters.
    mask : Mask, optional
        Selective processing mask.
    progress : callable
        Progress callback.

    Returns
    -------
    ndarray
        Background-neutralized image, clipped to [0, 1].
    """
    if params is None:
        params = BackgroundNeutralizationParams()

    original = image.copy()
    progress(0.1, "Measuring background levels…")

    def _sky_level(channel: np.ndarray) -> float:
        """Estimate sky background from pixels that are genuinely dark sky.

        Excludes bright pixels (stars, nebulosity) before taking the
        percentile so diffuse emission doesn't inflate the estimate.
        """
        flat = channel.ravel()
        if params.protect_bright < 1.0:
            cap = float(flat.max() * params.protect_bright)
            flat = flat[flat < cap]
        if flat.size == 0:
            return 0.0
        return float(np.percentile(flat, params.percentile))

    if image.ndim == 2:
        bg = _sky_level(image)
        log.info("BackgroundNeutralization: mono bg=%.5f", bg)
        correction = image - bg * params.amount
        result = np.clip(correction, 0, 1).astype(np.float32)
    else:
        n_ch = image.shape[0]
        result = image.copy()
        for ch in range(n_ch):
            progress(0.1 + 0.8 * ch / n_ch, f"Neutralizing channel {ch + 1}/{n_ch}…")
            bg = _sky_level(image[ch])
            log.info("BackgroundNeutralization: ch%d bg=%.5f", ch, bg)
            result[ch] = np.clip(image[ch] - bg * params.amount, 0, 1)

    progress(0.95, "Applying mask…")
    result = apply_mask(original, result, mask)
    progress(1.0, "Background neutralization complete")
    return result
