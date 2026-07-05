"""WaveScale HDR — multiscale wavelet-based dynamic range compression.

Decomposes the CIE L*a*b* lightness channel of a (typically already
stretched) image into à-trous wavelet scales, then boosts each detail scale
by a factor that grows with local brightness and decays across scales. This
lets bright regions (a galaxy core, a bloated star field) reveal fine
structure instead of clipping to a flat white, while the darker background
is left comparatively untouched.

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
class WaveScaleHDRParams:
    """Parameters for WaveScale HDR dynamic-range compression.

    Fields mirror Seti Astro Suite Pro's WaveScale HDR dialog controls, plus
    ``decay_rate`` and ``dim_gamma`` which SASpro hardcodes/auto-computes but
    exposes as real knobs in its underlying compute function.
    """

    n_scales: int = 5
    """Number of wavelet detail scales the luminance is split into (2-10).
    More scales let the effect reach larger structures (broad spiral arms,
    big nebula gradients); fewer keeps it focused on fine detail."""

    compression_factor: float = 1.5
    """Strength of the local-contrast boost applied to bright regions
    (0.10-5.00). 1.0 = gentle uniform lift; below 1.0 suppresses detail in
    bright areas; above ~2.0 can produce halos and a crunchy look."""

    mask_gamma: float = 5.0
    """Gamma shaping the brightness-derived steering mask: mask =
    (L / 100)^gamma (0.10-10.00). Higher concentrates the effect on only the
    brightest regions (e.g. a galaxy core); lower spreads it into fainter
    areas too, which also lifts background noise."""

    decay_rate: float = 0.5
    """Per-scale falloff base: detail scale i is weighted by decay_rate**i,
    so the compression boost is strongest on the finest detail scale and
    tapers off on coarser (larger-structure) scales."""

    dim_gamma: float | None = None
    """Optional override for the post-reconstruction highlight-taming gamma
    curve. None = auto (1.0 + n_scales * 0.2), SASpro's default behavior."""


def _mask_from_luminance(l0: np.ndarray, gamma: float) -> np.ndarray:
    """Brightness-derived steering mask: (L / 100)^gamma, clipped to [0, 1]."""
    m = np.clip(l0 / 100.0, 0.0, 1.0).astype(np.float32)
    if gamma != 1.0:
        m = np.power(m, gamma, dtype=np.float32)
    return m


def _split_channels(data: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a contiguous (H, W, 3) RGB float32 view, plus whether mono."""
    if data.ndim == 2:
        rgb = np.stack([data, data, data], axis=-1)
        return np.ascontiguousarray(rgb.astype(np.float32, copy=False)), True
    rgb = np.transpose(data[:3], (1, 2, 0))
    return np.ascontiguousarray(rgb.astype(np.float32, copy=False)), False


def _merge_channels(rgb: np.ndarray, was_mono: bool) -> np.ndarray:
    """Invert `_split_channels`: (H, W, 3) -> (H, W) mono or (C, H, W) color."""
    if was_mono:
        return np.mean(rgb, axis=-1, dtype=np.float32)
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))


def apply_wavescale_hdr(
    data: np.ndarray,
    params: WaveScaleHDRParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Compress dynamic range with a wavelet-based local-contrast boost.

    Parameters
    ----------
    data : ndarray
        Image data, shape (H, W) mono or (C, H, W) color, float32 in [0, 1].
    params : WaveScaleHDRParams, optional
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
        params = WaveScaleHDRParams()

    progress(0.0, "WaveScale HDR: converting to Lab…")
    rgb, was_mono = _split_channels(data)
    lab = cv2.cvtColor(np.clip(rgb, 0.0, 1.0), cv2.COLOR_RGB2LAB)
    l0 = lab[..., 0].astype(np.float32, copy=True)

    progress(0.15, "WaveScale HDR: decomposing luminance…")
    scales = wavelet_decompose(l0, n_scales=params.n_scales)
    planes, residual = scales[:-1], scales[-1]

    hdr_mask = _mask_from_luminance(l0, params.mask_gamma)

    progress(0.4, "WaveScale HDR: boosting scales…")
    for i, plane in enumerate(planes):
        decay = params.decay_rate**i
        scale = (1.0 + (params.compression_factor - 1.0) * hdr_mask * decay) * 2.0
        planes[i] = plane * scale

    lr = wavelet_reconstruct([*planes, residual])

    # Midtones alignment: keep the median lightness where it started, so the
    # compression only redistributes contrast rather than shifting exposure.
    med0 = float(np.median(l0))
    med1 = float(np.median(lr)) or 1.0
    lr = np.clip(lr * (med0 / med1), 0.0, 100.0)

    lab[..., 0] = lr
    progress(0.75, "WaveScale HDR: converting back to RGB…")
    rgb_out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    dim_gamma = params.dim_gamma if params.dim_gamma is not None else 1.0 + params.n_scales * 0.2
    rgb_out = np.power(np.clip(rgb_out, 0.0, 1.0), dim_gamma, dtype=np.float32)

    result = _merge_channels(rgb_out, was_mono)
    result = np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)

    progress(1.0, "WaveScale HDR complete")
    return apply_mask(data, result, mask)
