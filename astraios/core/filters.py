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

    # apply_mask only reads the original, and ``image`` is never mutated here
    # (results go to fresh buffers), so we can pass ``image`` directly and skip
    # an ~880MB defensive copy on large frames.
    if image.ndim == 2:
        log.debug("Processing mono image %s", image.shape)
        result = _unsharp_mask_channel(image, params)
    else:
        n_ch = image.shape[0]
        log.debug("Processing %d-channel image %s", n_ch, image.shape)
        result = np.empty_like(image)
        for ch in range(n_ch):
            result[ch] = _unsharp_mask_channel(image[ch], params)

    return apply_mask(image, result, mask)


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


# --------------------------------------------------------------------------
# Fast Gaussian blur
# --------------------------------------------------------------------------

_GAUSS_BAND_BYTES = 256 * 1024 * 1024
"""Peak transient budget for the banded GPU path.

A previous GPU port stacked ksize^2 shifted copies of a channel, reached ~7GB
on a 73MP frame, and the caller caught the OOM and silently skipped the step
-- so the tool quietly stopped working on exactly the large images that needed
it. Bounding the transient here keeps that from repeating.
"""


def gaussian_kernel_1d(sigma: float, truncate: float = 4.0) -> tuple[np.ndarray, int]:
    """The same 1D Gaussian weights scipy.ndimage.gaussian_filter builds.

    Computed in float64 and returned as float32, matching scipy's radius rule
    ``int(truncate * sigma + 0.5)`` so the two agree on kernel extent as well
    as on weights.
    """
    radius = int(truncate * float(sigma) + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    k /= k.sum()
    return k.astype(np.float32), radius


def _gaussian_blur_gpu_2d(plane: np.ndarray, sigma: float, truncate: float) -> np.ndarray:
    """Separable Gaussian on the GPU, matching scipy's mode='reflect'.

    scipy's "reflect" is numpy's "symmetric" (d c b a | a b c d); torch's
    F.pad has no symmetric mode and its "reflect" is scipy's "mirror", so the
    padding is done with numpy before the transfer rather than trusting a
    same-named option to mean the same thing.

    Not bit-identical to scipy: scipy accumulates the weighted sum in float64
    while this accumulates in float32, and a GPU sums in a different order
    besides. The difference is around 1e-5 of full scale, which is under one
    16-bit level. :func:`gaussian_blur` decides when that trade is worth
    making.
    """
    import torch
    import torch.nn.functional as F

    from astraios.core.device_manager import get_device_manager

    k_np, radius = gaussian_kernel_1d(sigma, truncate)
    device = get_device_manager().device

    padded = np.pad(plane, radius, mode="symmetric")
    h_pad, w_pad = padded.shape

    kt = torch.from_numpy(k_np.copy()).to(device)
    k_v = kt.view(1, 1, -1, 1)
    k_h = kt.view(1, 1, 1, -1)

    # Band the vertical pass so the transient stays bounded on large frames.
    # Every output row still sees the same window, so banding changes nothing
    # about the result; a test asserts that against the single-shot path.
    rows_budget = max(1, _GAUSS_BAND_BYTES // max(1, w_pad * 4))
    band = max(1, min(plane.shape[0], rows_budget))

    out_rows = []
    with torch.no_grad():
        for start in range(0, plane.shape[0], band):
            stop = min(start + band, plane.shape[0])
            chunk = padded[start:stop + 2 * radius]          # halo included
            t = torch.from_numpy(np.ascontiguousarray(chunk)).to(device)
            t = t.view(1, 1, *t.shape)
            t = F.conv2d(t, k_v)                             # vertical
            t = F.conv2d(t, k_h)                             # horizontal
            out_rows.append(t[0, 0].cpu().numpy())
            del t

    return np.concatenate(out_rows, axis=0).astype(np.float32)


def gaussian_blur(
    image: np.ndarray,
    sigma: float,
    *,
    truncate: float = 4.0,
    prefer_gpu: bool = True,
) -> np.ndarray:
    """Gaussian blur matching ``scipy.ndimage.gaussian_filter(mode="reflect")``.

    Uses the GPU when one is present and the work is large enough to pay for
    the transfer, otherwise scipy. The GPU path is not bit-identical to scipy
    (see :func:`_gaussian_blur_gpu_2d`); it agrees to about 1e-5 of full scale,
    below one 16-bit level.

    A small blur is left on the CPU on purpose. Measured on Apple Silicon
    earlier in this project, a tool can be 0.35x on the GPU at 256px and 1.79x
    at 1024px: below some size the transfer costs more than the arithmetic
    saves, and a blanket "GPU is faster" would make small images slower.
    """
    sigma = float(sigma)
    if sigma <= 0:
        return image.astype(np.float32, copy=True)

    radius = int(truncate * sigma + 0.5)
    pixels = int(np.prod(image.shape[-2:]))
    # Work scales with pixels * kernel taps; below this there is nothing to win.
    worth_gpu = prefer_gpu and pixels * (2 * radius + 1) > 20_000_000

    if worth_gpu:
        try:
            from astraios.core.device_manager import get_device_manager

            if get_device_manager().is_gpu:
                if image.ndim == 2:
                    return _gaussian_blur_gpu_2d(image, sigma, truncate)
                return np.stack([
                    _gaussian_blur_gpu_2d(image[c], sigma, truncate)
                    for c in range(image.shape[0])
                ]).astype(np.float32)
        except Exception as exc:  # pragma: no cover - falls back below
            # Never let a GPU problem silently drop the blur: say so, then do
            # the work on the CPU anyway.
            log.warning("GPU gaussian blur unavailable (%s); using scipy", exc)

    if image.ndim == 2:
        return scipy.ndimage.gaussian_filter(
            image, sigma=sigma, truncate=truncate, mode="reflect"
        ).astype(np.float32)
    return np.stack([
        scipy.ndimage.gaussian_filter(
            image[c], sigma=sigma, truncate=truncate, mode="reflect"
        )
        for c in range(image.shape[0])
    ]).astype(np.float32)


def _median_2d(plane: np.ndarray, ksize: int) -> np.ndarray:
    """Median-filter one 2D plane, matching scipy's default border exactly.

    OpenCV's medianBlur is around 200x faster than scipy.ndimage at a 3x3
    kernel and 88x at 5x5 (120ms vs 0.6ms, 351ms vs 4.0ms on a 1024x1024
    float32 plane here), but it replicates edge pixels while scipy's default
    reflects them. Left alone that quietly rewrites the image border: 678
    pixels differ on a 256x256 frame at ksize=5.

    Padding with numpy's "symmetric" mode first reproduces scipy's "reflect"
    (the two libraries use the same word for different things), so the crop
    comes back bit-identical. cv2 only accepts float32 at ksize 3 and 5, so
    anything larger keeps the scipy path.
    """
    if ksize in (3, 5):
        pad = ksize // 2
        padded = np.pad(plane, pad, mode="symmetric")
        return cv2.medianBlur(padded, ksize)[pad:-pad, pad:-pad]
    return scipy.ndimage.median_filter(plane, size=ksize).astype(np.float32)


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

    # ``image`` is read-only here (results go to fresh buffers and apply_mask
    # only reads the original), so skip the defensive copy.
    if image.ndim == 2:
        log.debug("Processing mono image %s", image.shape)
        result = _median_2d(image, ksize).astype(np.float32)
    else:
        n_ch = image.shape[0]
        log.debug("Processing %d-channel image %s", n_ch, image.shape)
        result = np.empty_like(image)
        for ch in range(n_ch):
            result[ch] = _median_2d(image[ch], ksize)

    return apply_mask(image, result, mask)


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

    # ``image`` is never mutated below, so use it directly instead of copying.
    if image.ndim == 2:
        blurred = _blur_channel(image, params)
    else:
        blurred = np.empty_like(image)
        for ch in range(image.shape[0]):
            blurred[ch] = _blur_channel(image[ch], params)

    amount = float(np.clip(params.amount, 0.0, 1.0))
    result = image * (1.0 - amount) + blurred * amount
    result = np.clip(result, 0.0, 1.0).astype(np.float32)
    return apply_mask(image, result, mask)
