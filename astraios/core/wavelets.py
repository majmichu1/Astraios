"""Wavelets — GPU-accelerated a trous wavelet decomposition and reconstruction.

Uses PyTorch conv2d for GPU acceleration. Falls back to CPU when no GPU.
The B3 spline kernel is used for the a trous algorithm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(f: float, m: str) -> None:
    pass

# B3 spline 1D kernel for a trous wavelet transform
_B3_KERNEL_1D = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0


_MAX_WAVELET_SCALES = 8


def _atrous_kernel_2d(scale: int) -> np.ndarray:
    """Create 2D a trous B3 spline kernel at given scale.

    The kernel is an upsampled version of the B3 spline where
    zeros are inserted between coefficients (a trous = with holes).
    """
    k1d = _B3_KERNEL_1D
    step = 2**scale
    size = (len(k1d) - 1) * step + 1
    padded = np.zeros(size, dtype=np.float32)
    for i, v in enumerate(k1d):
        padded[i * step] = v
    return np.outer(padded, padded)


def _smooth_gpu(data_t: torch.Tensor, kernel_t: torch.Tensor) -> torch.Tensor:
    """Apply 2D smoothing convolution on GPU using torch conv2d."""
    # data_t: (1, 1, H, W), kernel_t: (1, 1, kH, kW)
    # Replicate (edge-clamp) padding, NOT conv2d's default zero-padding: zeros
    # pull the smoothed value toward 0 at the borders, darkening the residual
    # edge and leaving compensating artifacts in the detail scales (visible once
    # thresholded). Replicate keeps the telescoping reconstruction exact.
    pad_h = kernel_t.shape[2] // 2
    pad_w = kernel_t.shape[3] // 2
    padded = F.pad(data_t, (pad_w, pad_w, pad_h, pad_h), mode="replicate")
    return F.conv2d(padded, kernel_t)


@dataclass
class WaveletParams:
    """Parameters for wavelet processing."""

    n_scales: int = 4  # number of wavelet scales
    scale_weights: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0]
    )
    residual_weight: float = 1.0
    # Per-scale noise thresholds for soft thresholding (0 = disabled).
    # Length must match n_scales (padded with 0 if shorter).
    noise_thresholds: list[float] = field(default_factory=list)


def wavelet_decompose(
    data: np.ndarray,
    n_scales: int = 4,
) -> list[np.ndarray]:
    """Decompose a 2D array into wavelet scales + residual using a trous.

    Parameters
    ----------
    data : ndarray
        Single-channel 2D array, float32.
    n_scales : int
        Number of wavelet detail scales (capped at _MAX_WAVELET_SCALES).

    n_scales = min(n_scales, _MAX_WAVELET_SCALES)

    Returns
    -------
    list[ndarray]
        List of (n_scales + 1) arrays: detail scales [0..n_scales-1] and
        the residual (smooth) at index n_scales.
    """
    dm = get_device_manager()
    device = dm.device

    current = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    scales = []

    with torch.no_grad():
        for s in range(n_scales):
            kernel_np = _atrous_kernel_2d(s)
            kernel_t = torch.from_numpy(kernel_np).unsqueeze(0).unsqueeze(0).to(device)
            smoothed = _smooth_gpu(current, kernel_t)
            detail = current - smoothed
            scales.append(detail.squeeze().cpu().numpy())
            current = smoothed
            del kernel_t, detail, smoothed  # free VRAM each scale

        # Residual (low-frequency content)
        scales.append(current.squeeze().cpu().numpy())
        del current

    return scales


def wavelet_reconstruct(scales: list[np.ndarray]) -> np.ndarray:
    """Reconstruct an image from wavelet scales.

    Parameters
    ----------
    scales : list[ndarray]
        List from wavelet_decompose: detail scales + residual.

    Returns
    -------
    ndarray
        Reconstructed 2D array.
    """
    result = np.zeros_like(scales[0])
    for s in scales:
        result = result + s
    return result


def wavelet_sharpen(
    data: np.ndarray,
    params: WaveletParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Sharpen or smooth image using per-scale wavelet weights.

    Weights > 1.0 sharpen that scale, < 1.0 smooth it.

    Parameters
    ----------
    data : ndarray
        Image data, shape (H, W) or (C, H, W), float32 in [0, 1].
    params : WaveletParams, optional
        Processing parameters.
    mask : Mask, optional
        Processing mask.

    Returns
    -------
    ndarray
        Processed image.
    """
    if params is None:
        params = WaveletParams()

    original = data.copy()

    # Ensure scale_weights length matches n_scales
    weights = list(params.scale_weights)
    while len(weights) < params.n_scales:
        weights.append(1.0)

    thresholds = list(params.noise_thresholds)
    while len(thresholds) < params.n_scales:
        thresholds.append(0.0)

    def _soft_threshold(coeff: np.ndarray, thr: float) -> np.ndarray:
        """Soft thresholding: shrink coefficients toward zero by thr."""
        if thr <= 0:
            return coeff
        sign = np.sign(coeff)
        return sign * np.maximum(np.abs(coeff) - thr, 0)

    def _process_channel(ch: np.ndarray) -> np.ndarray:
        scales = wavelet_decompose(ch, n_scales=params.n_scales)
        for i in range(params.n_scales):
            scales[i] = _soft_threshold(scales[i], thresholds[i])
            scales[i] = scales[i] * weights[i]
        scales[-1] = scales[-1] * params.residual_weight
        return np.clip(wavelet_reconstruct(scales), 0, 1).astype(np.float32)

    if data.ndim == 2:
        progress(0.0, "Wavelet processing…")
        result = _process_channel(data)
        progress(1.0, "Wavelet complete")
    else:
        n_ch = data.shape[0]
        result = np.empty_like(data)
        for ch in range(n_ch):
            progress(ch / n_ch, f"Wavelet ch {ch + 1}/{n_ch}…")
            result[ch] = _process_channel(data[ch])
        progress(1.0, "Wavelet complete")

    return apply_mask(original, result, mask)
