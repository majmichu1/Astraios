"""Channel alignment — corrects R/G/B channel misregistration (chromatic aberration)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from cosmica.core.device_manager import (
    get_device_manager,  # noqa: F401  # part of Cosmica GPU convention
)

log = logging.getLogger(__name__)

MIN_PIXELS_FOR_GPU = 256 * 256


@dataclass
class ChannelMatchParams:
    """Parameters for channel alignment.

    Attributes:
        reference_channel: Which channel to align to: "R", "G", "B", or "Mean".
        method: "fft", "ecc", or "auto" (star-based if possible).
        max_translation: Maximum pixel shift to search.
        zoom_out: Downsample factor for speed (1 = full res).
    """

    reference_channel: str = "G"
    method: str = "auto"
    max_translation: int = 50
    zoom_out: int = 4


def align_channels(
    img: NDArray,
    params: ChannelMatchParams | None = None,
) -> NDArray:
    """Align R/G/B channels of an RGB image.

    Args:
        img: (H, W, 3) float32 image in [0, 1] or linear range.
        params: Alignment parameters. If *None*, defaults are used.

    Returns:
        (H, W, 3) with channels aligned to reference, clipped to [0, 1].

    Raises:
        ValueError: If the image is not (H, W, 3).
    """
    if params is None:
        params = ChannelMatchParams()

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"Expected (H, W, 3) RGB image, got shape {img.shape}"
        )

    h, w = img.shape[:2]
    channels = {"R": img[:, :, 0], "G": img[:, :, 1], "B": img[:, :, 2]}

    if params.reference_channel not in ("R", "G", "B", "Mean"):
        raise ValueError(
            f"reference_channel must be 'R', 'G', 'B', or 'Mean', got {params.reference_channel!r}"
        )

    if params.reference_channel == "Mean":
        reference = (channels["R"] + channels["G"] + channels["B"]) / 3.0
    else:
        reference = channels[params.reference_channel]

    result = img.copy()
    zoom_out = max(1, params.zoom_out)

    small_w = max(1, w // zoom_out)
    small_h = max(1, h // zoom_out)

    if zoom_out > 1:
        ref_small = cv2.resize(
            reference, (small_w, small_h), interpolation=cv2.INTER_AREA
        )
    else:
        ref_small = reference

    # Decide method
    use_ecc = False
    if params.method == "ecc":
        use_ecc = True
    elif params.method == "auto":
        total_pixels = h * w
        if total_pixels >= MIN_PIXELS_FOR_GPU:
            local_std = np.std(reference[::4, ::4])
            if local_std > 0.01:
                use_ecc = True

    max_shift = max(1, params.max_translation // zoom_out)
    ch_idx_map = {"R": 0, "G": 1, "B": 2}

    for ch_name in ("R", "G", "B"):
        if ch_name == params.reference_channel or params.reference_channel == "Mean":
            continue

        moving = channels[ch_name]
        idx = ch_idx_map[ch_name]

        if zoom_out > 1:
            mov_small = cv2.resize(
                moving, (small_w, small_h), interpolation=cv2.INTER_AREA
            )
        else:
            mov_small = moving

        if use_ecc:
            try:
                dy, dx, _scale, _angle = _align_channel_ecc(mov_small, ref_small, max_shift)
                log.info(
                    "ECC alignment %s: (%.2f, %.2f) px, scale=%.4f, angle=%.2f deg",
                    ch_name, dy, dx, _scale, _angle,
                )
            except Exception as exc:
                log.warning(
                    "ECC failed for %s: %s; falling back to FFT", ch_name, exc,
                )
                dy, dx = _align_channel_fft(mov_small, ref_small, max_shift)
        else:
            dy, dx = _align_channel_fft(mov_small, ref_small, max_shift)
            log.info(
                "FFT alignment %s: (%.2f, %.2f) px", ch_name, dy, dx,
            )

        result[:, :, idx] = _apply_shift(moving, -dy * zoom_out, -dx * zoom_out)

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _align_channel_fft(
    moving: NDArray,
    reference: NDArray,
    max_shift: int = 50,
) -> tuple[float, float]:
    """Compute (dy, dx) shift of *moving* relative to *reference* via FFT.

    Uses ``skimage.registration.phase_cross_correlation`` with sub-pixel
    refinement. The returned shift describes how much *moving* is displaced
    relative to *reference*.

    Args:
        moving: (H, W) float32 image to be aligned.
        reference: (H, W) float32 reference image.
        max_shift: Maximum allowed shift in pixels; larger values are clamped.

    Returns:
        (dy, dx) in pixels.
    """
    from skimage.registration import phase_cross_correlation

    try:
        shift, _error, _ = phase_cross_correlation(
            reference,
            moving,
            upsample_factor=10,
            overlap_ratio=0.3,
            normalization="phase",
        )
    except Exception as exc:
        log.warning("phase_cross_correlation failed: %s", exc)
        return 0.0, 0.0

    dy, dx = float(shift[0]), float(shift[1])

    if abs(dy) > max_shift or abs(dx) > max_shift:
        log.warning(
            "FFT shift (%.2f, %.2f) exceeds max_shift=%d; clamping",
            dy, dx, max_shift,
        )
        dy = max(-max_shift, min(max_shift, dy))
        dx = max(-max_shift, min(max_shift, dx))

    return dy, dx


def _align_channel_ecc(
    moving: NDArray,
    reference: NDArray,
    max_shift: int = 50,
) -> tuple[float, float, float, float]:
    """Compute affine (dy, dx, scale, rotation) via OpenCV ECC.

    ``cv2.findTransformECC`` estimates the warp that aligns *moving* to
    *reference*. The returned values describe the displacement and
    deformation of *moving* relative to *reference*.

    Args:
        moving: (H, W) float32 image to be aligned.
        reference: (H, W) float32 reference image.
        max_shift: Maximum allowed translation in pixels; larger values are clamped.

    Returns:
        (dy, dx, scale, angle_deg).

    Raises:
        RuntimeError: If ECC alignment fails.
    """
    h, w = moving.shape
    if h < 16 or w < 16:
        raise RuntimeError(f"Image too small for ECC: {moving.shape}")

    # Convert to float32 if needed
    ref_f32 = reference.astype(np.float32)
    mov_f32 = moving.astype(np.float32)

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        50,
        1e-4,
    )

    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_f32,
            mov_f32,
            warp_matrix,
            cv2.MOTION_AFFINE,
            criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
    except cv2.error as exc:
        raise RuntimeError(f"ECC alignment failed: {exc}") from exc

    # Extract affine parameters
    # warp_matrix = [[a, b, tx], [c, d, ty]]
    a, b, tx = float(warp_matrix[0, 0]), float(warp_matrix[0, 1]), float(warp_matrix[0, 2])
    c, d, ty = float(warp_matrix[1, 0]), float(warp_matrix[1, 1]), float(warp_matrix[1, 2])

    dx = tx
    dy = ty

    scale_x = np.sqrt(a * a + c * c)
    scale_y = np.sqrt(b * b + d * d)
    scale = (scale_x + scale_y) / 2.0
    angle_deg = float(np.degrees(np.arctan2(c, a)))

    if abs(dy) > max_shift or abs(dx) > max_shift:
        log.warning(
            "ECC shift (%.2f, %.2f) exceeds max_shift=%d; clamping",
            dy, dx, max_shift,
        )
        dy = max(-max_shift, min(max_shift, dy))
        dx = max(-max_shift, min(max_shift, dx))

    return dy, dx, scale, angle_deg


def _apply_shift(img: NDArray, dy: float, dx: float) -> NDArray:
    """Apply sub-pixel translation via OpenCV warpAffine.

    Applies a shift that *corrects* a detected misalignment of (dy, dx)
    by translating the image by (-dy, -dx).

    Args:
        img: (H, W) float32 image.
        dy: Detected y-shift to correct (pixels).
        dx: Detected x-shift to correct (pixels).

    Returns:
        (H, W) float32 shifted image, same shape.
    """
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return img.copy()

    h, w = img.shape[:2]
    mat = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float64)

    return cv2.warpAffine(
        img,
        mat,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    ).astype(np.float32)
