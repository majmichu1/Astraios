"""WaveScale Dark Enhance — multiscale faint dark-structure enhancement.

Builds a "darkness mask" from the negative dips of mid-frequency à-trous
wavelet detail scales (places where the signal falls below its local
background trend — dust lanes, faint tidal streams, dim outer halo), then
iteratively boosts those same mid-frequency dips so faint structure becomes
visible without blowing out already-bright regions.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from astraios.core.masks import Mask, apply_mask
from astraios.core.wavelets import wavelet_decompose, wavelet_reconstruct

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(f: float, m: str) -> None:
    pass


@dataclass
class WaveScaleDarkEnhanceParams:
    """Parameters for WaveScale Dark Enhance faint-structure enhancement."""

    n_scales: int = 6
    """Number of wavelet detail scales analyzed/reconstructed (2-10). More
    scales let the darkness mask and the boost reach broader faint structure;
    fewer keeps it confined to small-scale dips."""

    boost_factor: float = 5.0
    """Strength of the dark-detail boost (0.10-10.00). 1.0 disables the
    enhancement entirely (exact no-op); higher values dig deeper into faint
    structure at the risk of amplifying noise."""

    mask_gamma: float = 1.0
    """Gamma applied to the darkness-weighting mask (0.10-10.00). Higher
    concentrates enhancement on only the faintest dips; lower spreads it
    into brighter mid-tones too."""

    iterations: int = 2
    """Number of enhancement passes (1-10). The darkness mask is recomputed
    from the updated image at the start of each pass, letting the effect
    compound on structure it just revealed."""

    decay_rate: float = 0.5
    """Per-scale falloff base: detail scale i's boost is weighted by
    decay_rate**i, tapering the enhancement on coarser (larger-structure)
    scales. The finest scale (i=0) is always left untouched (skipped) to
    avoid amplifying pixel-level noise."""


def _gaussian_blur(data: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with reflect edge handling (matches SASpro's fallback)."""
    if sigma <= 0:
        return data
    return cv2.GaussianBlur(
        data, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
    )


def _darkness_mask(l_norm: np.ndarray, n_scales: int, gamma: float) -> np.ndarray:
    """Weight mask emphasizing faint/dark mid-scale structure.

    Built from the negative part (dips below the local trend) of wavelet
    detail scales 1-3, averaged, normalized to its own peak, gamma-shaped,
    smoothed, and pushed through a gentle S-curve so mid-tones of the mask
    are boosted rather than a hard cutoff.

    Parameters
    ----------
    l_norm : ndarray
        2D lightness/luminance array normalized to [0, 1].
    n_scales : int
        Number of wavelet scales to decompose into.
    gamma : float
        Mask gamma.

    Returns
    -------
    ndarray
        2D mask, float32 in [0, 1].
    """
    scales = wavelet_decompose(l_norm, n_scales=n_scales)
    sel = scales[1:4]
    if not sel:
        return np.zeros_like(l_norm, dtype=np.float32)

    neg = [np.clip(-p, 0, None) for p in sel]
    combined = np.mean(neg, axis=0).astype(np.float32)
    denom = float(np.max(combined) + 1e-8)
    m = combined / denom
    if gamma != 1.0:
        m = np.power(m, gamma, dtype=np.float32)
    m = _gaussian_blur(m, sigma=3.0).astype(np.float32)
    m = np.clip(1.5 * m - 0.5 * (m * m), 0.0, 1.0).astype(np.float32)
    return m


def _enhance_once(
    l_data: np.ndarray,
    params: WaveScaleDarkEnhanceParams,
    hi: float,
) -> np.ndarray:
    """Run one darkness-mask + boost pass on a 2D array in [0, hi]."""
    dmask = _darkness_mask(
        np.clip(l_data / hi, 0.0, 1.0), params.n_scales, params.mask_gamma
    )

    scales = wavelet_decompose(l_data, n_scales=params.n_scales)
    residual = scales[-1]
    planes = scales[:-1]
    for i in range(len(planes)):
        if i == 0:
            continue  # skip highest-frequency (noise-dominated) scale
        decay = params.decay_rate**i
        neg = np.clip(-planes[i], 0, None)
        enhancement = neg * dmask * (params.boost_factor - 1.0) * decay
        planes[i] = planes[i] - enhancement

    out = np.clip(wavelet_reconstruct([*planes, residual]), 0.0, hi)
    return out.astype(np.float32, copy=False)


def _to_hwc_rgb(data: np.ndarray) -> np.ndarray:
    """Convert (C, H, W) channels-first color data to contiguous (H, W, 3) RGB."""
    return np.ascontiguousarray(np.transpose(data[:3], (1, 2, 0)).astype(np.float32, copy=False))


def _to_chw(rgb: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) RGB back to contiguous (C, H, W)."""
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))


def apply_dark_enhance(
    data: np.ndarray,
    params: WaveScaleDarkEnhanceParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Enhance faint dark structure via iterative wavelet boosting.

    For mono images the boost runs directly on the [0, 1] pixel data. For
    color images it runs on the CIE L*a*b* lightness channel only, preserving
    chrominance.

    Parameters
    ----------
    data : ndarray
        Image data, shape (H, W) mono or (C, H, W) color, float32 in [0, 1].
    params : WaveScaleDarkEnhanceParams, optional
        Processing parameters.
    mask : Mask, optional
        Processing mask (final blend: processed * mask + original * (1 - mask)).
    progress : callable, optional
        Progress callback ``(fraction, message)``.

    Returns
    -------
    ndarray
        Processed image, same shape and dtype as input.
    """
    if params is None:
        params = WaveScaleDarkEnhanceParams()

    n_iter = max(1, params.iterations)
    progress(0.0, "WaveScale Dark Enhance: analyzing dark structure…")

    if data.ndim == 2:
        l_data = np.clip(data, 0.0, 1.0).astype(np.float32, copy=True)
        for it in range(n_iter):
            progress(0.1 + 0.8 * it / n_iter, f"WaveScale Dark Enhance: pass {it + 1}/{n_iter}…")
            l_data = _enhance_once(l_data, params, hi=1.0)
        result = l_data
    else:
        rgb = _to_hwc_rgb(data)
        lab = cv2.cvtColor(np.clip(rgb, 0.0, 1.0), cv2.COLOR_RGB2LAB)
        l_data = lab[..., 0].astype(np.float32, copy=True)
        for it in range(n_iter):
            progress(0.1 + 0.8 * it / n_iter, f"WaveScale Dark Enhance: pass {it + 1}/{n_iter}…")
            l_data = _enhance_once(l_data, params, hi=100.0)
        lab[..., 0] = l_data
        rgb_out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        result = _to_chw(np.clip(rgb_out, 0.0, 1.0))

    result = np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)
    progress(1.0, "WaveScale Dark Enhance complete")
    return apply_mask(data, result, mask)
