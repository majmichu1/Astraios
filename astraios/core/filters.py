"""Image Filters — classic sharpening and noise reduction for astrophotography.

Provides Unsharp Mask (sharpening) and Median Filter (noise reduction).
All images are float32 numpy arrays in [0, 1] range.
Mono images have shape (H, W), color images have shape (C, H, W).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np
import scipy.ndimage
import torch

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unsharp Mask
# ---------------------------------------------------------------------------


@dataclass
class UnsharpMaskParams:
    """Parameters for unsharp mask sharpening.

    Attributes
    ----------
    radius : float
        Gaussian blur sigma in pixels.  Larger values sharpen coarser detail.
    amount : float
        Sharpening strength (0 = no effect, 2 = very strong).
    threshold : float
        Luminance threshold in [0, 1].  Differences below this value are not
        sharpened, which helps avoid amplifying noise.
    """

    radius: float = 2.0
    amount: float = 0.5
    threshold: float = 0.0


def _make_gaussian_kernel_1d(sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 1D Gaussian kernel for separable convolution."""
    sigma = max(sigma, 0.5)
    ksize = int(np.ceil(sigma * 3)) * 2 + 1
    ksize = max(ksize, 3)
    x = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


@torch.no_grad()
def _gaussian_blur_gpu(
    channel: np.ndarray,
    sigma: float,
    dm,
) -> np.ndarray:
    """GPU-accelerated Gaussian blur using separable 1D convolutions with reflect padding."""
    try:
        device = dm.device
        t_img = torch.from_numpy(channel).unsqueeze(0).unsqueeze(0).to(device)
        k1d = _make_gaussian_kernel_1d(sigma, device)
        pad = k1d.shape[0] // 2

        # Reflect-pad to avoid zero-padding edge artifacts on uniform regions
        t_padded = torch.nn.functional.pad(t_img, (pad, pad, pad, pad), mode="reflect")

        # Horizontal pass (no extra padding needed — already padded)
        kh = k1d.reshape(1, 1, 1, k1d.shape[0])
        blurred = torch.nn.functional.conv2d(t_padded, kh, padding=0)

        # Vertical pass
        kv = k1d.reshape(1, 1, k1d.shape[0], 1)
        blurred = torch.nn.functional.conv2d(blurred, kv, padding=0)

        result = blurred.squeeze().cpu().numpy()
        return result.astype(np.float32)
    except RuntimeError:
        log.debug("GPU blur OOM, falling back to CPU")
        ksize = int(np.ceil(sigma * 3)) * 2 + 1
        return cv2.GaussianBlur(channel, (ksize, ksize), sigma)


def _unsharp_mask_channel(
    channel: np.ndarray,
    params: UnsharpMaskParams,
) -> np.ndarray:
    """Apply unsharp mask to a single 2-D channel.

    Uses GPU-accelerated Gaussian blur when available for faster processing.

    Parameters
    ----------
    channel : ndarray
        2-D float32 array with values in [0, 1].
    params : UnsharpMaskParams
        Sharpening parameters.

    Returns
    -------
    ndarray
        Sharpened channel, clipped to [0, 1].
    """
    dm = get_device_manager()
    if dm.device.type != "cpu":
        blurred = _gaussian_blur_gpu(channel, params.radius, dm)
    else:
        ksize = int(np.ceil(params.radius * 3)) * 2 + 1
        blurred = cv2.GaussianBlur(channel, (ksize, ksize), params.radius)

    diff = channel - blurred

    # Apply threshold: only sharpen where |diff| exceeds the threshold.
    if params.threshold > 0.0:
        mask_below = np.abs(diff) < params.threshold
        diff[mask_below] = 0.0

    result = channel + params.amount * diff
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def unsharp_mask(
    image: np.ndarray,
    params: UnsharpMaskParams | None = None,
    mask: Mask | None = None,
) -> np.ndarray:
    """Apply unsharp mask sharpening to an image.

    The classic sharpening algorithm: subtract a blurred copy from the
    original to isolate high-frequency detail, then add that detail back
    with a controllable strength.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W), float32 values in [0, 1].
    params : UnsharpMaskParams, optional
        Sharpening parameters.  If ``None``, sensible defaults are used.
    mask : Mask, optional
        If provided, the sharpening effect is blended with the original
        image according to the mask (1.0 = fully sharpened, 0.0 = original).

    Returns
    -------
    ndarray
        Sharpened image with the same shape and dtype as the input.
    """
    if params is None:
        params = UnsharpMaskParams()

    log.debug(
        "Unsharp mask: radius=%.2f, amount=%.2f, threshold=%.4f",
        params.radius,
        params.amount,
        params.threshold,
    )

    original = image.copy()

    if image.ndim == 2:
        log.debug("Processing mono image %s", image.shape)
        result = _unsharp_mask_channel(image, params)
    else:
        n_ch = image.shape[0]
        log.debug("Processing %d-channel image %s", n_ch, image.shape)
        result = np.empty_like(image)
        for ch in range(n_ch):
            result[ch] = _unsharp_mask_channel(image[ch], params)

    return apply_mask(original, result, mask)


# ---------------------------------------------------------------------------
# Median Filter
# ---------------------------------------------------------------------------


@dataclass
class MedianFilterParams:
    """Parameters for median filtering.

    Attributes
    ----------
    kernel_size : int
        Size of the square median kernel.  Must be a positive odd integer.
    """

    kernel_size: int = 3


def _validate_kernel_size(kernel_size: int) -> int:
    """Ensure the kernel size is a positive odd integer.

    If *kernel_size* is even it is incremented by one so that the kernel
    is always centred on the target pixel.

    Returns
    -------
    int
        Validated (odd) kernel size.
    """
    if kernel_size < 1:
        log.warning("kernel_size=%d is less than 1, clamping to 1", kernel_size)
        kernel_size = 1
    if kernel_size % 2 == 0:
        kernel_size += 1
        log.warning("kernel_size must be odd; adjusted to %d", kernel_size)
    return kernel_size


def median_filter(
    image: np.ndarray,
    params: MedianFilterParams | None = None,
    mask: Mask | None = None,
) -> np.ndarray:
    """Apply a median filter to an image for noise reduction.

    Replaces each pixel with the median of the surrounding neighbourhood,
    which is effective at removing salt-and-pepper noise and hot pixels
    commonly found in astrophotography subs.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W), float32 values in [0, 1].
    params : MedianFilterParams, optional
        Filter parameters.  If ``None``, a 3x3 kernel is used.
    mask : Mask, optional
        If provided, the filtered result is blended with the original
        image according to the mask (1.0 = fully filtered, 0.0 = original).

    Returns
    -------
    ndarray
        Filtered image with the same shape and dtype as the input.
    """
    if params is None:
        params = MedianFilterParams()

    ksize = _validate_kernel_size(params.kernel_size)

    log.debug("Median filter: kernel_size=%d", ksize)

    original = image.copy()

    if image.ndim == 2:
        log.debug("Processing mono image %s", image.shape)
        result = scipy.ndimage.median_filter(image, size=ksize).astype(np.float32)
    else:
        n_ch = image.shape[0]
        log.debug("Processing %d-channel image %s", n_ch, image.shape)
        result = np.empty_like(image)
        for ch in range(n_ch):
            result[ch] = scipy.ndimage.median_filter(
                image[ch], size=ksize,
            ).astype(np.float32)

    return apply_mask(original, result, mask)


# ---------------------------------------------------------------------------
# Convolution (blur)
# ---------------------------------------------------------------------------


class ConvolutionKernel(Enum):
    GAUSSIAN = auto()  # smooth Gaussian blur (radius = sigma)
    BOX = auto()       # uniform box / mean blur


@dataclass
class ConvolutionParams:
    """Parameters for a convolution (blur).

    Attributes
    ----------
    kernel : ConvolutionKernel
        ``GAUSSIAN`` (sigma = ``radius``) or ``BOX`` (mean over a
        ``2*radius+1`` window).
    radius : float
        Blur radius in pixels.
    amount : float
        Blend with the original in ``[0, 1]``: 1 = full blur, 0 = unchanged.
    """

    kernel: ConvolutionKernel = ConvolutionKernel.GAUSSIAN
    radius: float = 2.0
    amount: float = 1.0


def _blur_channel(channel: np.ndarray, params: ConvolutionParams) -> np.ndarray:
    if params.radius <= 0:
        return channel.astype(np.float32, copy=True)
    if params.kernel == ConvolutionKernel.BOX:
        ksize = int(round(params.radius)) * 2 + 1
        return cv2.blur(channel, (ksize, ksize)).astype(np.float32)
    dm = get_device_manager()
    if dm.device.type != "cpu":
        return _gaussian_blur_gpu(np.ascontiguousarray(channel), params.radius, dm)
    ksize = int(np.ceil(params.radius * 3)) * 2 + 1
    return cv2.GaussianBlur(channel, (ksize, ksize), params.radius).astype(np.float32)


def convolve(
    image: np.ndarray,
    params: ConvolutionParams | None = None,
    mask: Mask | None = None,
) -> np.ndarray:
    """Convolve (blur) an image with a Gaussian or box kernel — GPU-accelerated.

    Useful for softening, feathering masks, or building synthetic PSFs. Handles
    mono ``(H, W)`` and colour ``(C, H, W)`` float32 ``[0, 1]`` images.
    """
    if params is None:
        params = ConvolutionParams()

    original = image.copy()
    if image.ndim == 2:
        blurred = _blur_channel(image, params)
    else:
        blurred = np.empty_like(image)
        for ch in range(image.shape[0]):
            blurred[ch] = _blur_channel(image[ch], params)

    amount = float(np.clip(params.amount, 0.0, 1.0))
    result = original * (1.0 - amount) + blurred * amount
    result = np.clip(result, 0.0, 1.0).astype(np.float32)
    return apply_mask(original, result, mask)
