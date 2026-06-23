"""Noise Reduction — wavelet (GPU) and NLM (CPU) denoising.

Wavelet denoising uses the GPU-accelerated a trous wavelet transform
from wavelets.py (same algorithm PixInsight uses) with BayesShrink
thresholding — runs entirely on GPU when available.

NLM (OpenCV) remains CPU-based since OpenCV's fastNlMeansDenoising
is already optimized native code and GPU NLM would require custom kernels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import cv2
import numpy as np
import torch

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask
from astraios.core.wavelets import wavelet_decompose

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


class DenoiseMethod(Enum):
    NLM = auto()
    WAVELET = auto()
    TGV = auto()
    MEDIAN = auto()


@dataclass
class DenoiseParams:
    """Parameters for noise reduction."""

    method: DenoiseMethod = DenoiseMethod.WAVELET
    strength: float = 0.5
    detail_preservation: float = 0.5
    chrominance_only: bool = False
    # NLM-specific
    nlm_h: float = 10.0
    nlm_template_size: int = 7
    nlm_search_size: int = 21
    # Wavelet-specific
    wavelet: str = "db4"
    wavelet_levels: int = 4
    # Median-specific
    median_kernel: int = 3
    # TGV-specific
    tgv_n_iter: int = 150


def _estimate_noise_sigma(wavelet_scales: list[np.ndarray]) -> float:
    """Estimate noise sigma from the finest wavelet scale using MAD."""
    finest = wavelet_scales[0]
    median_abs = float(np.median(np.abs(finest)))
    return median_abs / 0.6745 if median_abs > 0 else 1e-6


def measure_noise(image: np.ndarray) -> tuple[float, float]:
    """Measure the noise level and SNR of an image.

    Uses the patch-based estimator from :mod:`astraios.core.mure_denoise`.
    Astraios stores colour images channel-first ``(C, H, W)`` whereas the
    estimator expects mono ``(H, W)``; we therefore estimate per channel and
    average, which both fixes the layout mismatch and gives a single number
    to drive the UI.

    Args:
        image: ``(H, W)`` or ``(C, H, W)`` float32 in ``[0, 1]``.

    Returns:
        ``(sigma, snr)`` — noise standard deviation (in ``[0, 1]`` units,
        averaged across channels) and the corresponding signal-to-noise ratio.
    """
    from astraios.core.mure_denoise import estimate_noise, snr_estimate

    if image.ndim == 2:
        sigma = float(estimate_noise(image))
    else:
        sigmas = [float(estimate_noise(image[c])) for c in range(image.shape[0])]
        sigma = float(np.mean(sigmas)) if sigmas else 0.0
    return sigma, snr_estimate(image, sigma)


def recommend_strength(image: np.ndarray) -> tuple[float, float, float]:
    """Recommend a denoise *Amount* (0–1) from the measured noise level.

    The mapping is empirical: cleaner images get gentler denoising, noisier
    ones get stronger. It gives the user a sensible, data-driven starting
    point rather than a fixed default — they remain free to adjust.

    Returns:
        ``(strength, sigma, snr)`` so callers can also surface the measurement.
    """
    sigma, snr = measure_noise(image)
    strength = float(np.clip(0.2 + sigma * 20.0, 0.15, 0.9))
    return strength, sigma, snr


def _denoise_wavelet_gpu(
    image: np.ndarray,
    params: DenoiseParams,
    device: torch.device,
) -> np.ndarray:
    """GPU wavelet denoising using a trous decomposition + BayesShrink."""
    n_scales = min(params.wavelet_levels, 6)
    threshold_scale = params.strength * 3.0

    channels = [image] if image.ndim == 2 else [image[ch] for ch in range(image.shape[0])]
    results = []
    for ch_data in channels:
        scales_np = wavelet_decompose(ch_data, n_scales=n_scales)
        sigma_noise = _estimate_noise_sigma(scales_np)
        results.append(
            _wavelet_threshold_scale(
                ch_data, scales_np, sigma_noise, threshold_scale,
                params, n_scales, device,
            )
        )
    if image.ndim == 2:
        return results[0]
    return np.stack(results, axis=0)


def _wavelet_threshold_scale(
    original: np.ndarray,
    scales: list[np.ndarray],
    sigma_noise: float,
    threshold_scale: float,
    params: DenoiseParams,
    n_scales: int,
    device: torch.device,
) -> np.ndarray:
    """Apply soft thresholding to wavelet scales on GPU and reconstruct."""
    dm = get_device_manager()
    denoised_scales = [scales[0]]
    for level in range(1, n_scales):
        level_factor = 1.0 - (level - 1) / max(n_scales - 1, 1)
        effective_factor = level_factor * (1.0 - params.detail_preservation * 0.8)

        d = dm.from_numpy(scales[level])
        sigma_d = max(float(d.std().cpu()), 1e-10)
        sigma_signal = max(float((max(sigma_d**2 - sigma_noise**2, 0)) ** 0.5), 1e-10)
        thresh = float((sigma_noise**2 / sigma_signal) * threshold_scale * effective_factor)

        denoised = torch.where(d.abs() > thresh, d.sign() * (d.abs() - thresh), torch.zeros_like(d))
        denoised_scales.append(dm.to_cpu(denoised))

    result = np.sum(denoised_scales, axis=0)
    return np.clip(result, 0, 1).astype(np.float32)


def _denoise_nlm_channel(channel: np.ndarray, params: DenoiseParams) -> np.ndarray:
    """Apply OpenCV Non-Local Means denoising to a single channel (CPU)."""
    img_u8 = np.clip(channel * 255, 0, 255).astype(np.uint8)
    h = params.nlm_h * params.strength
    denoised = cv2.fastNlMeansDenoising(
        img_u8,
        None,
        h=h,
        templateWindowSize=params.nlm_template_size,
        searchWindowSize=params.nlm_search_size,
    )
    return denoised.astype(np.float32) / 255.0


def _denoise_median_channel(channel: np.ndarray, params: DenoiseParams) -> np.ndarray:
    """Apply median filter denoising to a single channel (CPU).

    `strength` scales the kernel size (3..15, odd).
    `detail_preservation` blends result with original (1=full original, 0=full median).
    """
    k = max(3, int(round(3 + 12 * params.strength)))
    if k % 2 == 0:
        k += 1
    img_u8 = np.clip(channel * 255, 0, 255).astype(np.uint8)
    denoised_u8 = cv2.medianBlur(img_u8, k)
    denoised = denoised_u8.astype(np.float32) / 255.0
    if params.detail_preservation > 0:
        blend = params.detail_preservation
        denoised = denoised * (1.0 - blend) + channel * blend
    return np.clip(denoised, 0, 1).astype(np.float32)


def _denoise_tgv(image: np.ndarray, params: DenoiseParams, progress: ProgressCallback) -> np.ndarray:
    """Apply TGV² denoising. Delegates to astraios.core.tgv_denoise.tgv_denoise."""
    from astraios.core.tgv_denoise import TGVParams, tgv_denoise as _tgv

    tgv_params = TGVParams(
        strength=max(0.01, params.strength * 0.5),
        n_iter=params.tgv_n_iter,
    )
    return _tgv(image, tgv_params, progress=progress)


def denoise(
    image: np.ndarray,
    params: DenoiseParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    if params is None:
        params = DenoiseParams()

    # No defensive copy: every path below builds a new `result` and the denoise
    # backends never mutate `image` (verified across methods), so apply_mask can
    # read `image` directly — saving a full copy per call (per channel in the
    # smart processor).
    dm = get_device_manager()
    use_gpu = dm.is_gpu and params.method == DenoiseMethod.WAVELET

    if use_gpu:
        progress(0.1, "Denoising on GPU...")
        result = _denoise_wavelet_gpu(image, params, dm.device)
        progress(1.0, "Noise reduction complete")
        return apply_mask(image, result, mask)

    if params.method == DenoiseMethod.TGV:
        progress(0.1, "Running TGV denoise...")
        try:
            result = _denoise_tgv(image, params, progress)
        except Exception as e:
            progress(0.5, f"TGV failed ({type(e).__name__}), falling back to wavelet...")
            fallback = DenoiseParams(
                method=DenoiseMethod.WAVELET,
                strength=params.strength,
                detail_preservation=params.detail_preservation,
                chrominance_only=params.chrominance_only,
            )
            return denoise(image, fallback, mask=mask, progress=progress)
        progress(1.0, "Noise reduction complete")
        return apply_mask(image, result, mask)

    if params.method == DenoiseMethod.MEDIAN:
        denoise_fn = _denoise_median_channel
    elif params.method == DenoiseMethod.NLM:
        denoise_fn = _denoise_nlm_channel
    else:
        denoise_fn = _denoise_wavelet_gpu_cpu

    if image.ndim == 2:
        progress(0.1, "Denoising mono image...")
        result = denoise_fn(image, params)
    elif params.chrominance_only and image.shape[0] >= 3:
        progress(0.1, "Denoising chrominance only...")
        result = _denoise_chrominance_only(image, params, denoise_fn, progress)
    else:
        result = np.empty_like(image)
        n_ch = image.shape[0]
        for ch in range(n_ch):
            progress(ch / n_ch, f"Denoising channel {ch + 1}/{n_ch}...")
            result[ch] = denoise_fn(image[ch], params)

    progress(1.0, "Noise reduction complete")
    return apply_mask(image, result, mask)


def _denoise_wavelet_gpu_cpu(channel: np.ndarray, params: DenoiseParams) -> np.ndarray:
    """CPU fallback using pywt (same algorithm as before)."""
    import pywt

    coeffs = pywt.wavedec2(channel, params.wavelet, level=params.wavelet_levels)
    detail_finest = coeffs[-1]
    hh = detail_finest[2]
    sigma_noise = float(np.median(np.abs(hh)) / 0.6745)
    threshold_scale = params.strength * 3.0

    denoised_coeffs = [coeffs[0]]
    for level_idx, detail in enumerate(coeffs[1:], 1):
        level_factor = 1.0 - (level_idx - 1) / max(params.wavelet_levels, 1)
        effective_factor = level_factor * (1.0 - params.detail_preservation * 0.8)
        thresholded = []
        for d in detail:
            sigma_d = max(float(np.std(d)), 1e-10)
            sigma_signal = max(float((max(sigma_d**2 - sigma_noise**2, 0)) ** 0.5), 1e-10)
            thresh = (sigma_noise**2 / sigma_signal) * threshold_scale * effective_factor
            thresholded.append(pywt.threshold(d, thresh, mode='soft'))
        denoised_coeffs.append(tuple(thresholded))

    result = pywt.waverec2(denoised_coeffs, params.wavelet)
    result = result[:channel.shape[0], :channel.shape[1]]
    return np.clip(result, 0, 1).astype(np.float32)


def _denoise_chrominance_only(
    image: np.ndarray,
    params: DenoiseParams,
    denoise_fn,
    progress: ProgressCallback,
) -> np.ndarray:
    r, g, b = image[0], image[1], image[2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = b - y
    cr = r - y

    progress(0.3, "Denoising Cb channel...")
    cb_dn = denoise_fn(cb + 0.5, params) - 0.5
    progress(0.6, "Denoising Cr channel...")
    cr_dn = denoise_fn(cr + 0.5, params) - 0.5

    result = np.empty_like(image)
    result[0] = np.clip(y + cr_dn, 0, 1)
    result[1] = np.clip(y - 0.2126 / 0.7152 * cr_dn - 0.0722 / 0.7152 * cb_dn, 0, 1)
    result[2] = np.clip(y + cb_dn, 0, 1)

    if image.shape[0] > 3:
        result[3:] = image[3:]

    return result
