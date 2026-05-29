"""Image Stacking Engine — registration, rejection, and integration.

Alignment uses proven library algorithms (skimage, OpenCV) with GPU
acceleration on top. Rejection uses astropy SigmaClip and scipy for
all statistical methods. No custom reimplementations of known algorithms.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from astropy.stats import SigmaClip
from skimage.registration import phase_cross_correlation as _skimage_pcc

from cosmica.core.device_manager import get_device_manager
from cosmica.core.gpu_stars import (
    Star,
    compose_affine_transforms,
    detect_stars_gpu,
    estimate_transform_gpu,
    match_stars_gpu,
    warp_image_gpu,
)
from cosmica.core.image_io import FrameType, ImageData, load_image
from cosmica.core.star_detection import (
    align_image as _cpu_align_image,
)
from cosmica.core.star_detection import (
    detect_stars as _cpu_detect_stars,
)
from cosmica.core.star_detection import (
    find_transform as _cpu_find_transform,
)
from cosmica.core.star_detection import (
    find_transform_triangle as _cpu_find_transform_triangle,
)

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


def _highpass_for_detection(t_img):
    """Apply large-scale background subtraction before star detection.

    Removes LP glow, Hα nebulosity, and vignetting gradients that would
    swamp the noise threshold and prevent star detection.

    Uses a pyramid approach: downsample 8× → Gaussian (sigma=25, equivalent
    to σ≈200 at full scale) → upsample → subtract.  ~50× faster than a
    direct 1001px kernel on 4000px images, same quality.

    ``t_img`` shape: ``(H, W)`` or ``(C, H, W)``, float32, any device.
    Returns same shape on same device.
    """
    try:
        import torch.nn.functional as F

        if t_img.dim() == 2:
            inp = t_img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            squeeze_2d = True
        elif t_img.dim() == 3:
            inp = t_img.unsqueeze(0)               # (1, C, H, W)
            squeeze_2d = False
        else:
            inp = t_img
            squeeze_2d = False

        H, W = inp.shape[-2], inp.shape[-1]
        C = inp.shape[-3]

        # Pyramid: work at 1/8 resolution
        scale = 8
        sh = max(16, H // scale)
        sw = max(16, W // scale)
        small = F.interpolate(inp, size=(sh, sw), mode="bilinear", align_corners=False)

        # Gaussian at small scale — sigma=25 here ≈ sigma=200 at original scale
        sigma = 25.0
        radius = min(int(sigma * 3), sh // 2 - 1, sw // 2 - 1)
        if radius < 1:
            return t_img
        kernel_size = 2 * radius + 1
        xs = torch.arange(kernel_size, dtype=torch.float32, device=inp.device) - radius
        g = torch.exp(-0.5 * (xs / sigma) ** 2)
        g = g / g.sum()

        kH = g.view(1, 1, 1, kernel_size).expand(C, 1, 1, kernel_size)
        kV = g.view(1, 1, kernel_size, 1).expand(C, 1, kernel_size, 1)
        bg_small = F.conv2d(F.pad(small, (radius, radius, 0, 0), mode="reflect"), kH, groups=C)
        bg_small = F.conv2d(F.pad(bg_small, (0, 0, radius, radius), mode="reflect"), kV, groups=C)

        bg = F.interpolate(bg_small, size=(H, W), mode="bilinear", align_corners=False)
        mean_val = inp.mean()
        hp = (inp - bg + mean_val).clamp(0.0, 1.0)

        if squeeze_2d:
            return hp.squeeze(0).squeeze(0)
        return hp.squeeze(0)
    except Exception:
        return t_img  # fallback: return original unchanged


# ---------------------------------------------------------------------------
# Configuration Enums
# ---------------------------------------------------------------------------


class RejectionMethod(Enum):
    NONE = auto()
    SIGMA_CLIP = auto()           # astropy SigmaClip — standard, proven
    WINSORIZED_SIGMA = auto()     # winsorize extremes, then sigma clip (Siril)
    LINEAR_FIT = auto()           # normalization + sigma clip (PixInsight)
    PERCENTILE_CLIP = auto()      # reject top/bottom N% per pixel (PixInsight)
    ESD = auto()                  # Generalized ESD test (PixInsight)
    MIN_MAX = auto()              # reject N lowest + N highest per pixel (Siril)


class IntegrationMethod(Enum):
    AVERAGE = auto()
    MEDIAN = auto()
    WEIGHTED_AVERAGE = auto()  # per-frame SNR/FWHM weights via StackingParams.frame_weights


class RegistrationMode(Enum):
    STAR_1_PASS = auto()      # GPU star detection → RANSAC → warp (single pass)
    STAR_2_PASS = auto()      # as above + refinement pass with tight radius
    TRIANGLE = auto()         # triangle-similarity matching (robust, handles large rotations)
    FFT_TRANSLATION = auto()  # FFT phase cross-correlation (translation only)
    COMET = auto()            # Track comet nucleus centroid; align by translation only


class NormalizationMethod(Enum):
    NONE = auto()
    ADDITIVE = auto()          # shift frames to match reference median
    MULTIPLICATIVE = auto()    # scale frames to match reference median
    ADDITIVE_SCALING = auto()  # Siril default: scale + shift via MAD (robust)


@dataclass
class StackingParams:
    rejection: RejectionMethod = RejectionMethod.SIGMA_CLIP
    integration: IntegrationMethod = IntegrationMethod.AVERAGE
    registration_mode: RegistrationMode = RegistrationMode.STAR_1_PASS
    normalization: NormalizationMethod = NormalizationMethod.ADDITIVE_SCALING
    kappa_low: float = 3.0
    kappa_high: float = 3.0
    max_iterations: int = 5
    winsorize_cutoff: float = 1.5   # sigmas for Winsorized clipping
    percentile_low: float = 10.0    # % to reject at low end (PERCENTILE_CLIP)
    percentile_high: float = 10.0   # % to reject at high end (PERCENTILE_CLIP)
    min_max_reject: int = 1         # frames to reject at each extreme (MIN_MAX)
    upsample_factor: int = 10       # sub-pixel refinement for FFT alignment
    use_gpu: bool = True            # prefer GPU; falls back to CPU automatically
    comet_nucleus_radius: int = 15  # search radius for nucleus peak (COMET mode)
    reference_frame_index: int = -1 # -1 = auto (highest variance), ≥0 = explicit index
    frame_weights: list[float] | None = None  # per-frame weights for WEIGHTED_AVERAGE


@dataclass
class StackResult:
    image: ImageData
    n_frames: int
    rejection_map: np.ndarray | None = None
    total_rejected: int = 0


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_stack(
    stack: np.ndarray,
    method: NormalizationMethod = NormalizationMethod.ADDITIVE_SCALING,
) -> np.ndarray:
    """Normalize background levels across frames.

    Prevents sigma clipping from treating brightness differences as outliers.
    Equivalent to Siril's normalization / PixInsight's Linear Fit.

    Parameters
    ----------
    stack : ndarray, shape (N, H, W) or (N, C, H, W)
    method : NormalizationMethod
    """
    n = stack.shape[0]
    if n < 2 or method == NormalizationMethod.NONE:
        return stack

    # Build broadcast shape: (N, 1, 1, ...) matching extra trailing dims
    extra_dims = stack.ndim - 1  # number of non-N dimensions
    broadcast = (slice(None),) + (None,) * extra_dims  # e.g. [:, None, None] or [:, None, None, None]

    flat = stack.reshape(n, -1)
    ref_med = float(np.median(flat[0]))

    if method == NormalizationMethod.ADDITIVE:
        frame_meds = np.median(flat, axis=1)
        shifts = ref_med - frame_meds
        return stack + shifts[broadcast]

    if method == NormalizationMethod.MULTIPLICATIVE:
        frame_meds = np.median(flat, axis=1)
        safe_meds = np.where(np.abs(frame_meds) > 1e-8, frame_meds, 1.0)
        scales = ref_med / safe_meds
        return stack * scales[broadcast]

    # ADDITIVE_SCALING (default): robust MAD-based scale + shift
    ref_mad = float(np.median(np.abs(flat[0] - ref_med))) * 1.4826
    frame_meds = np.median(flat, axis=1)
    frame_mads = np.median(np.abs(flat - frame_meds[:, None]), axis=1) * 1.4826

    if ref_mad < 1e-6:
        # Reference has no structure: offset-only correction
        shifts = ref_med - frame_meds
        log.debug("Normalization: offset-only (reference has no structure)")
        return stack + shifts[broadcast]

    safe_mads = np.maximum(frame_mads, 1e-8)
    scales = ref_mad / safe_mads
    shifts = ref_med - scales * frame_meds
    log.debug("Normalization: scale range [%.4f, %.4f]", scales.min(), scales.max())
    return stack * scales[broadcast] + shifts[broadcast]


def normalize_stack_linear_fit(stack: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for normalize_stack(ADDITIVE_SCALING)."""
    return normalize_stack(stack, NormalizationMethod.ADDITIVE_SCALING)


def _compute_frame_norm_factors(
    paths: "list",
    method: NormalizationMethod,
    sample_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame scale/shift from center crops — mirrors ``normalize_stack()``."""
    n = len(paths)
    scales = np.ones(n, dtype=np.float32)
    shifts = np.zeros(n, dtype=np.float32)
    if n < 2 or method == NormalizationMethod.NONE:
        return scales, shifts

    flats = np.stack([_sample_frame_crop(p, sample_size).ravel() for p in paths])
    ref_med = float(np.median(flats[0]))
    frame_meds = np.median(flats, axis=1)

    if method == NormalizationMethod.ADDITIVE:
        shifts = (ref_med - frame_meds).astype(np.float32)
        return scales, shifts

    if method == NormalizationMethod.MULTIPLICATIVE:
        safe_meds = np.where(np.abs(frame_meds) > 1e-8, frame_meds, 1.0)
        scales = (ref_med / safe_meds).astype(np.float32)
        return scales, shifts

    ref_mad = float(np.median(np.abs(flats[0] - ref_med))) * 1.4826
    frame_mads = np.median(np.abs(flats - frame_meds[:, None]), axis=1) * 1.4826

    if ref_mad < 1e-6:
        shifts = (ref_med - frame_meds).astype(np.float32)
        return scales, shifts

    safe_mads = np.maximum(frame_mads, 1e-8)
    scales = (ref_mad / safe_mads).astype(np.float32)
    shifts = (ref_med - scales * frame_meds).astype(np.float32)
    return scales, shifts


# ---------------------------------------------------------------------------
# Pixel Rejection Helpers
# ---------------------------------------------------------------------------


def _reject_sigma_clip(
    stack: np.ndarray, kappa_low: float, kappa_high: float, max_iterations: int
) -> np.ma.MaskedArray:
    """Astropy sigma clipping — the proven, standard approach."""
    sigclip = SigmaClip(sigma_lower=kappa_low, sigma_upper=kappa_high, maxiters=max_iterations)
    return sigclip(stack, axis=0)


def _reject_winsorized_sigma(
    stack: np.ndarray,
    kappa_low: float,
    kappa_high: float,
    max_iterations: int,
    winsorize_cutoff: float,
) -> np.ma.MaskedArray:
    """Winsorized sigma clipping (Siril).

    Clips the most extreme values to get robust location/scale estimates,
    then rejects pixels that deviate beyond kappa sigmas from those estimates.
    """
    from scipy.stats.mstats import winsorize

    n = stack.shape[0]
    limit = min(0.40, winsorize_cutoff / max(n, 1))
    limit = max(0.01, limit)

    # Winsorize to get robust mean/std estimates
    wins = np.asarray(winsorize(stack, limits=[limit, limit], axis=0))
    wins_mean = wins.mean(axis=0)
    wins_std = wins.std(axis=0, ddof=1) + 1e-10

    # Initial rejection mask
    hi_dev = (stack - wins_mean[None]) / wins_std[None]
    lo_dev = (wins_mean[None] - stack) / wins_std[None]
    mask = (hi_dev > kappa_high) | (lo_dev > kappa_low)

    # Iterate to refine: recompute stats on remaining values
    for _ in range(max_iterations - 1):
        remaining = np.where(mask, np.nan, stack)
        new_mean = np.nanmean(remaining, axis=0)
        new_std = np.nanstd(remaining, axis=0, ddof=1) + 1e-10
        hi_dev = (stack - new_mean[None]) / new_std[None]
        lo_dev = (new_mean[None] - stack) / new_std[None]
        new_mask = (hi_dev > kappa_high) | (lo_dev > kappa_low)
        if not np.any(new_mask & ~mask):
            break
        mask = new_mask

    return np.ma.array(stack, mask=mask)


def _reject_percentile_clip(
    stack: np.ndarray, percentile_low: float, percentile_high: float
) -> np.ma.MaskedArray:
    """Reject per-pixel values outside the specified percentile range (PixInsight)."""
    lo = np.percentile(stack, percentile_low, axis=0)
    hi = np.percentile(stack, 100.0 - percentile_high, axis=0)
    mask = (stack < lo[None]) | (stack > hi[None])
    return np.ma.array(stack, mask=mask)


def _reject_esd(
    stack: np.ndarray,
    alpha: float = 0.05,
    max_outliers: int | None = None,
) -> np.ma.MaskedArray:
    """Generalized ESD (Extreme Studentised Deviate) rejection (PixInsight).

    Rosner's test: iteratively finds the most extreme value per pixel and
    tests against the t-distribution critical value. More statistically
    rigorous than fixed-kappa sigma clipping.

    Parameters
    ----------
    stack : ndarray (N, H, W)
    alpha : float
        Significance level (lower = less aggressive).
    max_outliers : int
        Maximum outliers to test for. Defaults to N//3.
    """
    from scipy.stats import t as t_dist

    n = stack.shape[0]
    if max_outliers is None:
        max_outliers = max(1, n // 3)

    mask = np.zeros_like(stack, dtype=bool)
    h, w = stack.shape[1], stack.shape[2]
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]

    for i in range(1, max_outliers + 1):
        ni = n - (i - 1)
        if ni < 3:
            break

        # Critical value (scalar, same for all pixels)
        p = 1.0 - alpha / (2.0 * (n - i + 1))
        t_crit = float(t_dist.ppf(p, max(ni - 2, 1)))
        denom = (ni - i - 1 + t_crit**2) * (ni - i + 1)
        lambda_i = (ni - i) * t_crit / (denom**0.5 + 1e-10)

        # Per-pixel stats excluding already-masked values
        vals = np.where(mask, np.nan, stack)
        mu = np.nanmean(vals, axis=0)
        sig = np.nanstd(vals, axis=0, ddof=1) + 1e-10

        abs_dev = np.abs(vals - mu[None]) / sig[None]
        abs_dev = np.where(mask, -1.0, abs_dev)

        # Frame with largest deviation per pixel
        max_frame = np.argmax(abs_dev, axis=0)  # (H, W)
        max_dev = abs_dev[max_frame, rows, cols]

        outlier_pixels = max_dev > lambda_i
        if not np.any(outlier_pixels):
            break

        py, px = np.where(outlier_pixels)
        mask[max_frame[py, px], py, px] = True

    return np.ma.array(stack, mask=mask)


def _reject_min_max(stack: np.ndarray, n_reject: int) -> np.ma.MaskedArray:
    """Reject the n_reject lowest and highest values per pixel (Siril).

    Parameters
    ----------
    n_reject : int
        Number of frames to reject at each extreme.
    """
    n = stack.shape[0]
    if n_reject * 2 >= n:
        n_reject = max(1, (n - 1) // 2)

    sorted_indices = np.argsort(stack, axis=0)  # (N, H, W)
    mask = np.zeros_like(stack, dtype=bool)

    # Mark n_reject lowest and n_reject highest frames per pixel
    for k in range(n_reject):
        lo_frames = sorted_indices[k]          # (H, W)
        hi_frames = sorted_indices[n - 1 - k]  # (H, W)
        h, w = stack.shape[1], stack.shape[2]
        ys, xs = np.mgrid[0:h, 0:w]
        mask[lo_frames, ys, xs] = True
        mask[hi_frames, ys, xs] = True

    return np.ma.array(stack, mask=mask)


def _get_mask(masked_data: np.ma.MaskedArray, shape: tuple) -> np.ndarray:
    """Extract boolean mask from a MaskedArray, or zeros if nomask."""
    if masked_data.mask is np.ma.nomask:
        return np.zeros(shape, dtype=bool)
    return np.asarray(masked_data.mask)


def _integrate(
    masked_data: np.ma.MaskedArray,
    method: IntegrationMethod,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Combine pixel values after rejection."""
    if method == IntegrationMethod.MEDIAN:
        return np.ma.median(masked_data, axis=0).data
    if method == IntegrationMethod.WEIGHTED_AVERAGE and weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        return np.ma.average(masked_data, axis=0, weights=w).data.astype(np.float32)
    return np.ma.mean(masked_data, axis=0).data


# ---------------------------------------------------------------------------
# Affine Transform Utilities
# ---------------------------------------------------------------------------


def _invert_affine_2x3(m: np.ndarray) -> np.ndarray:
    """Invert a 2×3 affine transform matrix via homogeneous coordinates.

    PyTorch affine_grid expects output→input mapping (inverse transform).
    RANSAC / estimateAffinePartial2D returns the forward transform (src→dst).
    OpenCV warpAffine inverts internally; torch does not — so we invert explicitly.
    """
    m3 = np.vstack([m, [0.0, 0.0, 1.0]])
    return np.linalg.inv(m3)[:2].astype(np.float32)


# ---------------------------------------------------------------------------
# FFT Alignment (GPU + CPU paths)
# ---------------------------------------------------------------------------


def _apply_shift_cpu(data: np.ndarray, row_shift: float, col_shift: float) -> np.ndarray:
    """Apply sub-pixel translation using OpenCV Lanczos4 (CPU)."""
    mat = np.float64([[1, 0, col_shift], [0, 1, row_shift]])
    if data.ndim == 3:
        h, w = data.shape[1], data.shape[2]
        return np.stack([
            cv2.warpAffine(data[c], mat, (w, h), flags=cv2.INTER_LANCZOS4,
                           borderMode=cv2.BORDER_CONSTANT)
            for c in range(data.shape[0])
        ])
    h, w = data.shape
    return cv2.warpAffine(data, mat, (w, h), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT)


def _fft_align_frames(
    images: list[ImageData],
    progress: ProgressCallback,
    upsample_factor: int,
    use_gpu: bool,
) -> list[ImageData]:
    """Align via FFT phase cross-correlation (translation only).

    GPU path: torch.fft with sub-pixel bicubic refinement.
    CPU path: skimage.registration.phase_cross_correlation (proven, same as Siril).
    Both apply shifts using sub-pixel interpolation (not np.roll).
    """
    n = len(images)
    if n <= 1:
        return list(images)

    # Reference: highest variance after normalisation — vectorised over (N, H, W)
    normalized = np.stack([img.data / (img.data.max() + 1e-10) for img in images], axis=0)
    ref_idx = int(np.var(normalized, axis=tuple(range(1, normalized.ndim))).argmax())
    ref_img = images[ref_idx]

    log.info("FFT alignment: reference frame #%d", ref_idx + 1)

    aligned = [None] * n
    aligned[ref_idx] = ref_img

    dm = get_device_manager()
    gpu_available = use_gpu and dm.device.type != "cpu"

    # Grayscale for correlation
    def to_gray(data: np.ndarray) -> np.ndarray:
        return data.mean(axis=0) if data.ndim == 3 else data

    ref_gray = to_gray(ref_img.data)

    if gpu_available:
        ref_tensor = dm.from_numpy(ref_gray.astype(np.float32))

    for i in range(n):
        if i == ref_idx:
            continue
        progress(0.1 + 0.9 * (i / (n - 1)), f"FFT aligning frame {i + 1}/{n}...")
        tgt = images[i]
        tgt_gray = to_gray(tgt.data)

        if gpu_available:
            tgt_tensor = dm.from_numpy(tgt_gray.astype(np.float32))
            row_shift, col_shift = _gpu_fft_shift(ref_tensor, tgt_tensor, upsample_factor)
            # Apply with GPU warp
            full_tensor = dm.from_numpy(tgt.data.astype(np.float32))
            mat = np.array([[1, 0, col_shift], [0, 1, row_shift]], dtype=np.float32)
            warped_tensor = warp_image_gpu(full_tensor, mat, mode="bicubic")
            shifted_data = dm.to_cpu(warped_tensor).numpy().astype(np.float32)
            if shifted_data.ndim == 3 and shifted_data.shape[0] == 1:
                shifted_data = shifted_data[0]
        else:
            shift, _, _ = _skimage_pcc(ref_gray, tgt_gray, upsample_factor=upsample_factor)
            row_shift, col_shift = float(shift[0]), float(shift[1])
            shifted_data = _apply_shift_cpu(tgt.data, row_shift, col_shift)

        log.debug("Frame %d: row=%.3f col=%.3f", i + 1, row_shift, col_shift)
        aligned[i] = ImageData(
            data=shifted_data.astype(np.float32),
            header=tgt.header.copy(),
            frame_type=tgt.frame_type,
        )

    progress(1.0, "FFT alignment complete")
    return [img for img in aligned if img is not None]


def _gpu_fft_shift(
    ref: torch.Tensor,
    target: torch.Tensor,
    upsample_factor: int = 10,
) -> tuple[float, float]:
    """GPU phase cross-correlation returning (row_shift, col_shift).

    Fixed implementation: correct peak index calculation and single-pass
    sub-pixel refinement without double division.
    """
    rows, cols = ref.shape[-2], ref.shape[-1]
    new_rows, new_cols = 2 * rows, 2 * cols

    # Zero-pad to 2x size (avoids circular convolution)
    ref_padded = torch.zeros(new_rows, new_cols, device=ref.device, dtype=ref.dtype)
    tgt_padded = torch.zeros(new_rows, new_cols, device=target.device, dtype=target.dtype)
    ref_padded[:rows, :cols] = ref
    tgt_padded[:rows, :cols] = target

    ref_fft = torch.fft.fft2(ref_padded)
    tgt_fft = torch.fft.fft2(tgt_padded)

    norm = (ref_fft.abs() * tgt_fft.abs()).clamp(min=1e-10)
    cross = (tgt_fft * ref_fft.conj()) / norm
    result = torch.fft.fftshift(torch.fft.ifft2(cross).real)

    # Integer peak — FIXED: divide by new_cols (not new_rows)
    peak_flat = torch.argmax(result.flatten()).item()
    peak_row = peak_flat // new_cols
    peak_col = peak_flat % new_cols

    # Coarse shift relative to centre
    shift_row = float(peak_row - new_rows // 2)
    shift_col = float(peak_col - new_cols // 2)

    # Sub-pixel refinement via bicubic upsampling around the peak
    if upsample_factor > 1:
        zoom_r = 5
        r_min = max(0, peak_row - zoom_r)
        r_max = min(new_rows, peak_row + zoom_r + 1)
        c_min = max(0, peak_col - zoom_r)
        c_max = min(new_cols, peak_col + zoom_r + 1)

        region = result[r_min:r_max, c_min:c_max]
        reg_cols = c_max - c_min

        upsampled = functional.interpolate(
            region.unsqueeze(0).unsqueeze(0),
            scale_factor=upsample_factor,
            mode="bicubic",
            align_corners=True,
        ).squeeze()

        up_flat = torch.argmax(upsampled.flatten()).item()
        up_row = up_flat // (reg_cols * upsample_factor)
        up_col = up_flat % (reg_cols * upsample_factor)

        # Convert back to original coordinates (no double division)
        refined_row = r_min + up_row / upsample_factor
        refined_col = c_min + up_col / upsample_factor

        # Replace coarse shift with refined shift
        shift_row = refined_row - new_rows // 2
        shift_col = refined_col - new_cols // 2

    # Clamp to image bounds
    shift_row = max(-rows, min(rows, shift_row))
    shift_col = max(-cols, min(cols, shift_col))

    return float(shift_row), float(shift_col)


# ---------------------------------------------------------------------------
# Comet Nucleus Tracking
# ---------------------------------------------------------------------------


def _find_comet_nucleus(
    data: np.ndarray,
    search_radius: int = 15,
) -> tuple[float, float]:
    """Find the comet nucleus position in an image.

    Strategy:
    1. Work on the 2-D luminance channel.
    2. Apply a broad Gaussian blur to suppress stars and noise.
    3. Find the brightest pixel (nucleus peak).
    4. Compute centroid within `search_radius` pixels for sub-pixel accuracy.

    Returns (cx, cy) — column, row centroid.
    """
    # Get 2-D luminance
    if data.ndim == 3:
        gray = data.mean(axis=0).astype(np.float32)
    else:
        gray = data.astype(np.float32)

    h, w = gray.shape
    # Blur out stars: σ ≈ 4 px is enough to smooth point sources
    blur_k = max(3, (search_radius // 2) * 2 + 1)
    try:
        import cv2 as _cv2
        blurred = _cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    except Exception:
        blurred = gray  # cv2 should always be present

    # Peak pixel
    peak_flat = int(np.argmax(blurred))
    py, px = divmod(peak_flat, w)

    # Centroid in a box around peak
    r = search_radius
    y0, y1 = max(0, py - r), min(h, py + r + 1)
    x0, x1 = max(0, px - r), min(w, px + r + 1)
    region = gray[y0:y1, x0:x1].astype(np.float64)

    region_min = region.min()
    region = region - region_min  # subtract local bg
    total = region.sum()
    if total < 1e-10:
        return float(px), float(py)

    ys, xs = np.mgrid[y0:y1, x0:x1]
    cx = float((xs * region).sum() / total)
    cy = float((ys * region).sum() / total)
    return cx, cy


def _comet_align_frames(
    images: list[ImageData],
    params: StackingParams,
    progress: ProgressCallback,
) -> list[ImageData]:
    """Align frames by tracking the comet nucleus centroid.

    All frames are shifted so their nuclei coincide with the nucleus
    in the reference frame. No rotation or scale correction is applied.
    """
    n = len(images)
    progress(0.0, "Finding comet nucleus in reference frame…")

    # Reference = first frame (temporal ordering; the nucleus must be visible)
    ref_img = images[0]
    ref_cx, ref_cy = _find_comet_nucleus(ref_img.data, params.comet_nucleus_radius)
    log.info("Comet nucleus (reference): (%.1f, %.1f)", ref_cx, ref_cy)

    aligned: list[ImageData] = [ref_img]

    for i in range(1, n):
        progress(i / n, f"Comet alignment: frame {i + 1}/{n}…")
        frame = images[i]
        cx, cy = _find_comet_nucleus(frame.data, params.comet_nucleus_radius)
        dx = ref_cx - cx
        dy = ref_cy - cy
        log.debug("Frame %d nucleus: (%.1f, %.1f), shift: (%.1f, %.1f)", i + 1, cx, cy, dx, dy)

        # Apply translation: use GPU (warp_image_gpu) when available, else OpenCV
        d = frame.data
        dm = get_device_manager()
        if dm.device.type != "cpu":
            # warp_image_gpu expects the inverse (output→input) affine matrix.
            # Forward shift (dx_col, dy_row) → inverse matrix has (-dx, -dy).
            M_inv = np.float32([[1, 0, -dx], [0, 1, -dy]])
            t_frame = torch.from_numpy(d).to(dm.device)
            with torch.no_grad():
                warped_t = warp_image_gpu(t_frame, M_inv, mode="bilinear")
            shifted = warped_t.cpu().numpy()
            del t_frame, warped_t
        else:
            import cv2 as _cv2
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            if d.ndim == 2:
                h_d, w_d = d.shape
                shifted = _cv2.warpAffine(d, M, (w_d, h_d), flags=_cv2.INTER_LINEAR,
                                          borderMode=_cv2.BORDER_CONSTANT, borderValue=0.0)
            else:
                c_d, h_d, w_d = d.shape
                shifted = np.stack([
                    _cv2.warpAffine(d[ch], M, (w_d, h_d), flags=_cv2.INTER_LINEAR,
                                    borderMode=_cv2.BORDER_CONSTANT, borderValue=0.0)
                    for ch in range(c_d)
                ], axis=0)

        aligned.append(ImageData(
            data=shifted.astype(np.float32),
            header=frame.header.copy(),
            frame_type=frame.frame_type,
        ))

    progress(1.0, f"Comet alignment complete: {n} frames")
    return aligned


# ---------------------------------------------------------------------------
# Star-Based Alignment (GPU + CPU paths)
# ---------------------------------------------------------------------------


def align_frames(
    images: list[ImageData],
    params: StackingParams | None = None,
    progress: ProgressCallback = _noop_progress,
) -> list[ImageData]:
    """Align images using star detection and matching.

    GPU path (preferred): detect_stars_gpu → match_stars_gpu → RANSAC → warp_image_gpu.
    CPU fallback: detect_stars → find_transform (OpenCV RANSAC) → align_image.
    Both 1-pass and 2-pass modes are supported.

    For FFT_TRANSLATION mode, delegates to _fft_align_frames.
    """
    if params is None:
        params = StackingParams()

    n = len(images)
    if n == 0:
        raise ValueError("No images to align")
    if n == 1:
        return [images[0]]

    if params.registration_mode == RegistrationMode.FFT_TRANSLATION:
        return _fft_align_frames(
            images, progress, params.upsample_factor, params.use_gpu
        )

    if params.registration_mode == RegistrationMode.COMET:
        return _comet_align_frames(images, params, progress)

    dm = get_device_manager()
    gpu_available = params.use_gpu and dm.device.type != "cpu"

    # Select reference frame
    progress(0.0, "Selecting reference frame...")
    if params.reference_frame_index == -2:
        ref_idx = n - 1
        log.info("Star alignment: using last frame (#%d) as reference", ref_idx + 1)
    elif params.reference_frame_index >= 0:
        ref_idx = max(0, min(params.reference_frame_index, n - 1))
        log.info("Star alignment: using user-specified reference frame #%d", ref_idx + 1)
    else:
        ref_idx = int(np.argmax([np.var(img.data) for img in images]))
        log.info("Star alignment: auto-selected reference frame #%d (highest variance)", ref_idx + 1)
    progress(0.1, f"Reference: frame #{ref_idx + 1}")

    ref_img = images[ref_idx]
    aligned = [None] * n
    aligned[ref_idx] = ref_img

    if gpu_available:
        t_ref = dm.from_numpy(ref_img.data)
        ref_stars = detect_stars_gpu(_highpass_for_detection(t_ref))
        log.info("Reference: %d stars detected (GPU)", len(ref_stars))
        ref_sf = None
    else:
        ref_sf = _cpu_detect_stars(ref_img.data)
        ref_stars = None
        log.info("Reference: %d stars detected (CPU)", len(ref_sf))

    two_pass = params.registration_mode == RegistrationMode.STAR_2_PASS
    use_triangle = params.registration_mode == RegistrationMode.TRIANGLE

    # Pre-detect ref stars on CPU for triangle mode (always CPU, no GPU path needed)
    if use_triangle:
        ref_sf = _cpu_detect_stars(ref_img.data)
        gpu_available = False  # triangle mode always uses CPU path

    for i in range(n):
        if i == ref_idx:
            continue
        progress(0.1 + 0.8 * (i / n), f"Aligning frame {i + 1}/{n}...")
        frame = images[i]

        if frame.data.shape != ref_img.data.shape:
            log.warning("Frame %d shape mismatch, skipping", i + 1)
            aligned[i] = frame
            continue

        if gpu_available:
            t_img = dm.from_numpy(frame.data)
            tgt_stars = detect_stars_gpu(_highpass_for_detection(t_img))
            matches = match_stars_gpu(ref_stars, tgt_stars, max_dist=100.0)
            transform = estimate_transform_gpu(matches)

            if transform is None:
                log.warning("Frame %d: GPU transform failed, using CPU fallback", i + 1)
                ref_sf2 = _cpu_detect_stars(ref_img.data)
                tgt_sf2 = _cpu_detect_stars(frame.data)
                transform = _cpu_find_transform(ref_sf2, tgt_sf2)
                if transform is None:
                    aligned[i] = frame
                    continue
                warped_data = _cpu_align_image(frame.data, transform, ref_img.data.shape)
                aligned[i] = ImageData(
                    data=warped_data.astype(np.float32),
                    header=frame.header.copy(),
                    frame_type=frame.frame_type,
                )
                continue

            # CRITICAL: transform maps target→ref (forward), but affine_grid in
            # warp_image_gpu expects output(ref)→input(target) (inverse).
            # Must invert. OpenCV warpAffine handles this automatically; torch does not.
            if two_pass:
                # First pass: warp to ref-space with inverted transform
                transform_inv_coarse = _invert_affine_2x3(transform)
                warped_coarse = warp_image_gpu(t_img, transform_inv_coarse, mode="bilinear")
                re_stars = detect_stars_gpu(_highpass_for_detection(warped_coarse))
                re_matches = match_stars_gpu(ref_stars, re_stars, max_dist=20.0)
                refine_transform = estimate_transform_gpu(re_matches)
                if refine_transform is not None:
                    # Compose forward transforms: refine ∘ transform
                    transform = compose_affine_transforms(transform, refine_transform)

            transform_inv = _invert_affine_2x3(transform)
            final = warp_image_gpu(t_img, transform_inv, mode="bicubic")
            res = dm.to_cpu(final).numpy().astype(np.float32)
            if res.ndim == 3 and res.shape[0] == 1:
                res = res[0]
            del t_img, final  # free GPU tensors immediately after warp

        else:
            # CPU path: triangle-matching or standard nearest-neighbor
            tgt_sf = _cpu_detect_stars(frame.data)
            if use_triangle:
                transform = _cpu_find_transform_triangle(ref_sf, tgt_sf)
            else:
                transform = _cpu_find_transform(ref_sf, tgt_sf)
            if transform is None:
                log.warning("Frame %d: no transform found, skipping", i + 1)
                aligned[i] = frame
                continue

            if two_pass:
                warped_coarse = _cpu_align_image(frame.data, transform, ref_img.data.shape)
                tgt_sf2 = _cpu_detect_stars(warped_coarse)
                refine = _cpu_find_transform(ref_sf, tgt_sf2)
                if refine is not None:
                    transform = compose_affine_transforms(transform, refine)

            res = _cpu_align_image(frame.data, transform, ref_img.data.shape).astype(np.float32)

        aligned[i] = ImageData(
            data=res, header=frame.header.copy(), frame_type=frame.frame_type
        )

    # Free GPU reference tensor and flush allocator cache
    if gpu_available:
        del t_ref
        gc.collect()
        get_device_manager().empty_cache()

    progress(1.0, f"Alignment complete: {n} frames")
    return [img for img in aligned if img is not None]


# ---------------------------------------------------------------------------
# Path-based alignment (low-memory — peak RAM ~2 frames)
# ---------------------------------------------------------------------------


def _fast_variance_at(path) -> float:
    """Return variance of the center 256×256 crop without loading the full file.

    Uses FITS memmap to read only the required section — typically 50-100×
    faster than ``load_image`` for large files.  The raw (possibly integer)
    pixel values are sufficient for relative variance comparison; BZERO/BSCALE
    scaling preserves the ordering so we don't need to apply them.

    Falls back to ``load_image`` (full frame) on any error.
    """
    import astropy.io.fits as _fits

    try:
        # memmap=False so BZERO/BSCALE are applied correctly (memmap=True refuses
        # when these keywords are present, which is the case for most camera FITS).
        # We load only the center 256×256 crop via section slicing — astropy reads
        # only the required rows from disk, so this is still much faster than a
        # full load even without memmap.
        with _fits.open(str(path), memmap=False, lazy_load_hdus=True) as hdul:
            # Find first HDU with image data
            hdu = next((h for h in hdul if h.data is not None), hdul[0])
            sh = hdu.shape
            h, w = sh[-2], sh[-1]
            cy, cx = h // 2, w // 2
            half = 128
            r0, r1 = max(0, cy - half), min(h, cy + half)
            c0, c1 = max(0, cx - half), min(w, cx + half)
            if hdu.data.ndim == 2:
                crop = np.asarray(hdu.data[r0:r1, c0:c1], dtype=np.float32)
            else:
                crop = np.asarray(hdu.data[0, r0:r1, c0:c1], dtype=np.float32)
            return float(np.var(crop))
    except Exception as _e:
        log.warning("_fast_variance_at: FITS crop failed for %s (%s), falling back to full load", path, _e)
        img = load_image(str(path))
        arr = img.data
        del img
        sh = arr.shape[-2], arr.shape[-1]
        cy, cx = sh[0] // 2, sh[1] // 2
        half = 128
        crop = arr[..., max(0, cy - half): cy + half, max(0, cx - half): cx + half]
        return float(np.var(crop))


def _interp_for_detection(t: torch.Tensor, scale: float) -> torch.Tensor:
    """Bilinear downscale a (H,W) or (C,H,W) tensor for star detection.

    F.interpolate needs 4-D input (N,C,H,W), so we handle both mono and
    colour by adding/removing the necessary batch+channel dims.
    """
    if t.dim() == 2:                         # (H, W)  — mono Bayer frame
        inp = t.unsqueeze(0).unsqueeze(0)    # → (1, 1, H, W)
        out = functional.interpolate(inp, scale_factor=scale, mode="bilinear", align_corners=False)
        return out.squeeze(0).squeeze(0)     # → (H', W')
    else:                                    # (C, H, W)  — colour frame
        inp = t.unsqueeze(0)                 # → (1, C, H, W)
        out = functional.interpolate(inp, scale_factor=scale, mode="bilinear", align_corners=False)
        return out.squeeze(0)               # → (C, H', W')


def align_from_paths(
    paths: list,
    output_dir,
    params: StackingParams | None = None,
    progress: ProgressCallback = _noop_progress,
    filename_prefix: str = "aligned",
) -> list:
    """Align frames directly from disk, writing results to output_dir immediately.

    Peak RAM usage: ~2 frames (reference + current frame being processed).
    Suitable for large datasets (100+ frames) that would OOM with align_frames().

    Parameters
    ----------
    paths : list of str or Path
        Input FITS file paths.
    output_dir : str or Path
        Directory to write aligned FITS files.  Created if absent.
    params : StackingParams, optional
    progress : callable

    Returns
    -------
    list of Path
        Paths of written aligned files, same order as input.
    """
    import gc
    from pathlib import Path

    if params is None:
        params = StackingParams()

    paths = [Path(p) for p in paths]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(paths)
    if n == 0:
        raise ValueError("align_from_paths: no paths provided")

    dm = get_device_manager()
    gpu_available = params.use_gpu and dm.device.type != "cpu"

    # ── Select reference frame index ─────────────────────────────────────────
    progress(0.0, "Selecting reference frame...")
    if params.reference_frame_index == -2:
        ref_idx = n - 1
        log.info("align_from_paths: using last frame (#%d) as reference", ref_idx + 1)
    elif params.reference_frame_index >= 0:
        ref_idx = max(0, min(params.reference_frame_index, n - 1))
        log.info("align_from_paths: using user-specified frame #%d", ref_idx + 1)
    else:
        # Auto: sample center crop of each frame to find highest variance.
        # Fast path: use FITS memmap to read only a 256×256 center section
        # (~5–20 ms per file vs ~1–2 s for a full load), so scanning all
        # frames is affordable even for 200+ frame datasets.
        scan_indices = list(range(n))
        log.info(
            "align_from_paths: scanning all %d frames for best reference (center-crop)…",
            n,
        )
        best_var, ref_idx = -1.0, n // 2  # default to middle frame if all samples fail
        for i in scan_indices:
            p = paths[i]
            try:
                v = _fast_variance_at(p)
                if v > best_var:
                    best_var, ref_idx = v, i
            except Exception as exc:
                log.warning("align_from_paths: failed to sample %s: %s", p.name, exc)
        log.info("align_from_paths: auto-selected reference frame #%d", ref_idx + 1)
    progress(0.05, f"Reference: frame #{ref_idx + 1}")

    # ── Load reference frame fully ────────────────────────────────────────────
    # Debayer OSC frames BEFORE warping so the affine transform operates on
    # proper R/G/B channels, not raw Bayer mosaic pixels.  Warping a Bayer
    # mosaic with bicubic/Lanczos interpolation mixes pixels of different
    # filter colours (R with neighbouring G and B), corrupting the CFA pattern
    # and producing severe colour fringing on stars after debayering.
    # Industry standard (Siril, PixInsight) is to debayer first, then align.
    ref_img = load_image(str(paths[ref_idx]), debayer=True)
    ref_shape = ref_img.data.shape

    # Detection scale: run star detection on 1/2 resolution.
    # 1/4 makes stars 1-2px (killed by min_distance pooling); 1/2 keeps them 2-4px.
    # Star positions scaled back to full-res before matching; warp on full frame.
    _DETECT_SCALE = 2

    # Star detection always needs a 2D luminance view.
    _ref_for_detection = ref_img.data
    if _ref_for_detection.ndim == 3:
        _ref_for_detection = _ref_for_detection.mean(axis=0).astype(np.float32)

    t_ref = None
    if gpu_available:
        # Upload luminance-only for detection (saves VRAM; warp uses full color tensor later).
        t_ref_lum = dm.from_numpy(_ref_for_detection)
        t_ref_small = _interp_for_detection(t_ref_lum, 1.0 / _DETECT_SCALE)
        ref_stars_small = detect_stars_gpu(_highpass_for_detection(t_ref_small))
        del t_ref_small, t_ref_lum
        ref_stars = [
            Star(x=s.x * _DETECT_SCALE, y=s.y * _DETECT_SCALE, flux=s.flux, fwhm=s.fwhm * _DETECT_SCALE)
            for s in ref_stars_small
        ]
    else:
        ref_stars = None
        ref_sf = _cpu_detect_stars(_ref_for_detection)

    two_pass = params.registration_mode == RegistrationMode.STAR_2_PASS
    use_triangle = params.registration_mode == RegistrationMode.TRIANGLE
    if use_triangle:
        # Triangle mode: always CPU, no GPU path
        ref_sf = _cpu_detect_stars(_ref_for_detection)
        gpu_available = False

    # ── Write reference frame aligned (it's already aligned to itself) ────────
    out_paths: list = [None] * n
    ref_out = output_dir / f"{filename_prefix}_{ref_idx + 1:05d}.fits"
    _write_aligned_fits(ref_img, ref_out)
    out_paths[ref_idx] = ref_out
    log.info("align_from_paths: reference saved → %s", ref_out.name)

    # ── Align all other frames — prefetch IO hides disk latency ─────────────
    # While the GPU processes frame i, a background thread loads frame i+1.
    # This overlaps IO (slow) with GPU compute, typically 1.5-2× faster.
    from concurrent.futures import Future, ThreadPoolExecutor as _TPE

    non_ref = [i for i in range(n) if i != ref_idx]
    gc_interval = 10  # run gc every N frames instead of every frame

    def _load(idx: int):
        """Load one frame from disk; returns (idx, ImageData|None).

        debayer=True: debayer OSC frames before warping so that the affine
        transform operates on proper R/G/B channels.  Warping a raw Bayer
        mosaic with bicubic/Lanczos interpolation mixes R/G/B neighbours,
        corrupting the CFA pattern and producing colour fringing on stars.
        """
        try:
            return idx, load_image(str(paths[idx]), debayer=True)
        except Exception as exc:
            log.warning("align_from_paths: failed to load %s: %s", paths[idx].name, exc)
            return idx, None

    # Separate write pool so FITS writes overlap with GPU processing of the next frame.
    # max_workers=1 keeps writes sequential and memory usage bounded to ~2 frames.
    write_pool = _TPE(max_workers=1)

    with _TPE(max_workers=1) as pool:
        # Kick off the first prefetch before the loop starts
        prefetch: Future | None = pool.submit(_load, non_ref[0]) if non_ref else None

        for loop_pos, i in enumerate(non_ref):
            frac = 0.05 + 0.90 * (loop_pos / len(non_ref))
            progress(frac, f"Aligning frame {i + 1}/{n} ({loop_pos + 1}/{len(non_ref)})...")

            # Collect the prefetched frame (blocks only until IO finishes)
            _, frame = prefetch.result()

            # Prefetch the next frame while GPU processes this one
            next_pos = loop_pos + 1
            if next_pos < len(non_ref):
                prefetch = pool.submit(_load, non_ref[next_pos])
            else:
                prefetch = None

            if frame is None:
                continue  # load failed, skip

            if frame.data.shape != ref_shape:
                log.warning("align_from_paths: frame %d shape mismatch, copying as-is", i + 1)
                out_p = output_dir / f"{filename_prefix}_{i + 1:05d}.fits"
                _write_aligned_fits(frame, out_p)
                out_paths[i] = out_p
                del frame
                continue

            # Luminance view for star detection (works for both mono and RGB frames).
            frame_lum = frame.data.mean(axis=0).astype(np.float32) if frame.data.ndim == 3 else frame.data

            t_img = None
            if gpu_available:
                t_img = dm.from_numpy(frame.data)

                # Detect on luminance at half-res — saves VRAM; warp uses full color t_img.
                t_lum = dm.from_numpy(frame_lum)
                t_small = _interp_for_detection(t_lum, 1.0 / _DETECT_SCALE)
                tgt_stars_small = detect_stars_gpu(_highpass_for_detection(t_small))
                del t_small, t_lum
                tgt_stars = [
                    Star(x=s.x * _DETECT_SCALE, y=s.y * _DETECT_SCALE, flux=s.flux, fwhm=s.fwhm * _DETECT_SCALE)
                    for s in tgt_stars_small
                ]
                matches = match_stars_gpu(ref_stars, tgt_stars, max_dist=100.0)
                transform = estimate_transform_gpu(matches)

                if transform is not None and two_pass:
                    t_inv_coarse = _invert_affine_2x3(transform)
                    warped_c = warp_image_gpu(t_img, t_inv_coarse, mode="bilinear")
                    # Compute luminance of coarsely warped frame for refinement detection
                    warped_lum = warped_c.mean(dim=0, keepdim=False) if warped_c.dim() == 3 else warped_c
                    wc_small = _interp_for_detection(warped_lum, 1.0 / _DETECT_SCALE)
                    re_stars_small = detect_stars_gpu(_highpass_for_detection(wc_small))
                    del wc_small, warped_lum, warped_c
                    re_stars = [
                        Star(x=s.x * _DETECT_SCALE, y=s.y * _DETECT_SCALE, flux=s.flux, fwhm=s.fwhm * _DETECT_SCALE)
                        for s in re_stars_small
                    ]
                    re_matches = match_stars_gpu(ref_stars, re_stars, max_dist=20.0)
                    refine = estimate_transform_gpu(re_matches)
                    if refine is not None:
                        transform = compose_affine_transforms(transform, refine)

                if transform is not None:
                    transform_inv = _invert_affine_2x3(transform)
                    final = warp_image_gpu(t_img, transform_inv, mode="bicubic")
                    res = dm.to_cpu(final).numpy().astype(np.float32)
                    if res.ndim == 3 and res.shape[0] == 1:
                        res = res[0]
                else:
                    log.warning("Frame %d: GPU transform failed, trying CPU", i + 1)
                    ref_sf2 = _cpu_detect_stars(_ref_for_detection)
                    tgt_sf2 = _cpu_detect_stars(frame_lum)
                    cpu_t = _cpu_find_transform(ref_sf2, tgt_sf2)
                    if cpu_t is None:
                        log.warning("Frame %d: CPU fallback also failed, copying as-is", i + 1)
                        res = frame.data
                    else:
                        res = _cpu_align_image(frame.data, cpu_t, ref_shape).astype(np.float32)
            else:
                tgt_sf = _cpu_detect_stars(frame_lum)
                if use_triangle:
                    transform = _cpu_find_transform_triangle(ref_sf, tgt_sf)
                else:
                    transform = _cpu_find_transform(ref_sf, tgt_sf)
                if transform is None:
                    log.warning("Frame %d: no transform found, copying as-is", i + 1)
                    res = frame.data
                else:
                    if two_pass:
                        warped_c = _cpu_align_image(frame.data, transform, ref_shape)
                        warped_lum = (
                            warped_c.mean(axis=0).astype(np.float32)
                            if warped_c.ndim == 3 else warped_c
                        )
                        tgt_sf2 = _cpu_detect_stars(warped_lum)
                        refine = _cpu_find_transform(ref_sf, tgt_sf2)
                        if refine is not None:
                            transform = compose_affine_transforms(transform, refine)
                    res = _cpu_align_image(frame.data, transform, ref_shape).astype(np.float32)

            aligned_img = ImageData(data=res, header=frame.header.copy(), frame_type=frame.frame_type)
            out_p = output_dir / f"{filename_prefix}_{i + 1:05d}.fits"
            write_pool.submit(_write_aligned_fits, aligned_img, out_p)
            out_paths[i] = out_p

            del frame, res
            # aligned_img kept alive by write_pool until the write thread finishes
            del aligned_img
            if t_img is not None:
                del t_img
            if (loop_pos + 1) % gc_interval == 0:
                gc.collect()

    # Drain all pending FITS writes before returning
    write_pool.shutdown(wait=True)

    del ref_img
    gc.collect()
    if gpu_available:
        get_device_manager().empty_cache()

    result = [p for p in out_paths if p is not None]
    progress(1.0, f"Path-based alignment complete: {len(result)}/{n} frames saved")
    log.info("align_from_paths: %d/%d frames aligned → %s", len(result), n, output_dir)
    return result


def _write_aligned_fits(img: ImageData, out_path) -> None:
    """Write an ImageData to a FITS file for aligned output."""
    from pathlib import Path

    from astropy.io import fits as _fits

    out_path = Path(out_path)
    data = img.data
    if data.ndim == 3:
        # (C, H, W) — save as multi-extension or transpose to (H, W, C)?
        # Store as (C, H, W) NAXIS3=C, NAXIS2=H, NAXIS1=W (FITS convention for cubes)
        pass  # astropy handles ndim=3 natively
    hdr = _fits.Header()
    if img.header:
        for k, v in img.header.items():
            try:
                hdr[k] = v
            except Exception:
                pass
    hdr["ALIGNED"] = True
    hdu = _fits.PrimaryHDU(data=data.astype(np.float32), header=hdr)
    hdu.writeto(str(out_path), overwrite=True)


# ---------------------------------------------------------------------------
# GPU rejection helpers (used by stack_from_paths tiled path)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _gpu_sigma_clip(
    stack: torch.Tensor,
    kappa_low: float,
    kappa_high: float,
    max_iter: int = 5,
) -> tuple[torch.Tensor, int]:
    """Iterative sigma clipping on GPU. stack shape: (N, ...).
    Returns (mean of kept pixels, n_rejected).
    """
    mask = torch.ones_like(stack, dtype=torch.bool)  # True = keep

    for _ in range(max_iter):
        count = mask.sum(dim=0, keepdim=True).float().clamp(min=1)
        mean = (stack * mask).sum(dim=0, keepdim=True) / count
        # Variance of kept pixels
        diff = stack - mean
        var = (diff ** 2 * mask).sum(dim=0, keepdim=True) / count
        std = var.sqrt().clamp(min=1e-8)

        new_mask = mask & (diff >= -kappa_low * std) & (diff <= kappa_high * std)
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask

    count = mask.sum(dim=0).float().clamp(min=1)
    result = (stack * mask).sum(dim=0) / count
    n_rejected = int((~mask).sum().item())
    return result, n_rejected


@torch.no_grad()
def _gpu_percentile_clip(
    stack: torch.Tensor,
    pct_low: float,
    pct_high: float,
) -> tuple[torch.Tensor, int]:
    """Percentile clip on GPU — reject out-of-range pixels, mean survivors."""
    lo = torch.quantile(stack, pct_low / 100.0, dim=0)
    hi = torch.quantile(stack, 1.0 - pct_high / 100.0, dim=0)
    mask = (stack >= lo.unsqueeze(0)) & (stack <= hi.unsqueeze(0))
    count = mask.sum(dim=0).float().clamp(min=1)
    result = (stack * mask).sum(dim=0) / count
    n_rejected = int((~mask).sum().item())
    return result, n_rejected


@torch.no_grad()
def _gpu_min_max(
    stack: torch.Tensor,
    n_reject: int,
) -> tuple[torch.Tensor, int]:
    """Min/Max rejection on GPU — reject n_reject lowest and highest per pixel."""
    n = stack.shape[0]
    if n_reject * 2 >= n:
        n_reject = max(1, (n - 1) // 2)

    sorted_s, _ = stack.sort(dim=0)
    kept = sorted_s[n_reject: n - n_reject]
    if kept.shape[0] == 0:
        kept = sorted_s
    result = kept.mean(dim=0)
    n_rejected = 2 * n_reject * int(stack[0].numel())
    return result, n_rejected


# ---------------------------------------------------------------------------
# Tiled stacking from file paths (memory-efficient)
# ---------------------------------------------------------------------------


def _open_fits_for_tiles(path: "Path"):
    """Open FITS for tile reads; disable memmap when BZERO/BSCALE are present."""
    from astropy.io import fits as _fits

    hdr = _fits.getheader(str(path), ext=0)
    use_memmap = "BZERO" not in hdr and "BSCALE" not in hdr
    return _fits.open(str(path), memmap=use_memmap)


def _load_fits_tile(path: "Path", y0: int, y1: int) -> np.ndarray:
    """Load a horizontal tile [y0:y1, :] from a FITS file via memmap.

    Returns float32 array of shape (C, tile_h, w) or (tile_h, w).
    Only the requested rows are read from disk.
    """
    from cosmica.core.image_io import _normalize_fits_tile

    with _open_fits_for_tiles(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                d = hdu.data
                header = dict(hdu.header)
                if d.ndim == 2:
                    tile = np.array(d[y0:y1, :])
                elif d.ndim == 3:
                    tile = np.array(d[:, y0:y1, :])
                else:
                    tile = np.array(d)
                return _normalize_fits_tile(tile, header)
    raise ValueError(f"No image data in {path}")


def _sample_frame_crop(path: "Path", sample_size: int = 256) -> np.ndarray:
    """Return a normalized float32 center crop from a FITS file."""
    from cosmica.core.image_io import _normalize_fits_tile

    with _open_fits_for_tiles(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                d = hdu.data
                header = dict(hdu.header)
                if d.ndim == 2:
                    h, w = d.shape
                    y0 = max(0, h // 2 - sample_size // 2)
                    x0 = max(0, w // 2 - sample_size // 2)
                    crop = np.array(d[y0 : y0 + sample_size, x0 : x0 + sample_size])
                elif d.ndim == 3:
                    _, h, w = d.shape
                    y0 = max(0, h // 2 - sample_size // 2)
                    x0 = max(0, w // 2 - sample_size // 2)
                    crop = np.array(d[0, y0 : y0 + sample_size, x0 : x0 + sample_size])
                else:
                    return np.zeros((sample_size, sample_size), dtype=np.float32)
                return _normalize_fits_tile(crop, header)
    return np.zeros((sample_size, sample_size), dtype=np.float32)


def _sample_frame_background(path: "Path", sample_size: int = 256) -> float:
    """Estimate frame background by sampling the center of a FITS file."""
    return float(np.median(_sample_frame_crop(path, sample_size)))


def _get_fits_shape(path: "Path") -> tuple:
    """Return the data shape (C,H,W) or (H,W) from a FITS header (no pixel read)."""
    from astropy.io import fits as _fits

    with _fits.open(str(path), memmap=True) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return hdu.data.shape
    raise ValueError(f"No image data in {path}")


def stack_from_paths(
    paths: "list[Path]",
    params: "StackingParams | None" = None,
    progress: ProgressCallback = _noop_progress,
) -> "StackResult":
    """Stack frames from FITS file paths using tiled processing.

    Only one horizontal tile-strip is in memory at a time.  For N=50 frames
    at 3096×2080 in float32, peak RAM is ~300 MB regardless of frame count,
    versus ~4 GB for the in-memory path.

    Normalization factors are computed from a small center crop (fast, ≈1 MB
    per file), then applied per-tile as each strip is loaded.

    Parameters
    ----------
    paths : list[Path]
        Paths to aligned FITS files to stack.
    params : StackingParams, optional
        Stacking configuration (rejection, normalization, etc.).
    progress : callable
        Progress callback (fraction, message).
    """
    from pathlib import Path as _Path

    if params is None:
        params = StackingParams()

    n = len(paths)
    if n == 0:
        raise ValueError("No paths provided")
    if n == 1:
        img = load_image(str(paths[0]))
        return StackResult(image=img, n_frames=1)

    # 1. Determine output shape from first file
    progress(0.0, "Reading file headers…")
    ref_shape = _get_fits_shape(paths[0])
    if len(ref_shape) == 2:
        h, w = ref_shape
        n_channels = 1
        is_color = False
    else:
        n_channels, h, w = ref_shape
        is_color = True

    # Detect Bayer (OSC raw) — shape is 2D but camera is color
    bayer_pat = None
    try:
        from astropy.io import fits as _fits
        with _fits.open(str(paths[0]), memmap=True) as _hdul:
            for _hdu in _hdul:
                if _hdu.data is not None:
                    for _kw in ("BAYERPAT", "COLORTYP", "CFA-PAT", "BAYER"):
                        _bp = str(_hdu.header.get(_kw, "")).strip().upper()
                        if _bp in ("RGGB", "BGGR", "GRBG", "GBRG"):
                            bayer_pat = _bp
                    break
    except Exception:
        pass

    color_desc = f"color (C,H,W)" if is_color else (
        f"OSC Bayer ({bayer_pat})" if bayer_pat else "mono"
    )
    log.info(
        "Stack from paths: %d files, shape=%s, %s",
        n, ref_shape, color_desc,
    )

    # 2. Per-frame normalization (same logic as normalize_stack on full frames)
    scales = np.ones(n, dtype=np.float32)
    shifts = np.zeros(n, dtype=np.float32)
    if params.normalization != NormalizationMethod.NONE:
        progress(0.02, "Computing normalization factors…")
        scales, shifts = _compute_frame_norm_factors(paths, params.normalization)
        log.debug(
            "Normalization scales: [%.4f, %.4f] shifts: [%.4f, %.4f]",
            scales.min(), scales.max(), shifts.min(), shifts.max(),
        )

    # 3. Compute tile height so that (N × n_channels × tile_h × W × 4) ≤ ~350 MB RAM
    # GPU path moves each tile to VRAM after assembly, so RAM budget is per-tile only
    dm = get_device_manager()
    use_gpu = params.use_gpu and dm.device.type != "cpu"
    target_bytes = 350 * 1024 * 1024
    bytes_per_row = n * n_channels * w * 4
    tile_h = max(1, min(h, target_bytes // max(bytes_per_row, 1)))
    n_tiles = (h + tile_h - 1) // tile_h
    log.info(
        "Tiled stacking: tile_h=%d rows, %d tiles, ~%d MB/tile, GPU=%s",
        tile_h, n_tiles, (bytes_per_row * tile_h) // (1024 * 1024), use_gpu,
    )

    # 4. Allocate output
    if is_color:
        output = np.zeros((n_channels, h, w), dtype=np.float32)
    else:
        output = np.zeros((h, w), dtype=np.float32)
    total_rejected = 0

    # Pre-normalise per-frame weights once (reused every tile)
    _tile_weights_np: np.ndarray | None = None
    _tile_weights_t: "torch.Tensor | None" = None
    if params.frame_weights is not None:
        if len(params.frame_weights) != n:
            raise ValueError(
                f"frame_weights length {len(params.frame_weights)} != number of frames {n}"
            )
        _tile_weights_np = np.asarray(params.frame_weights, dtype=np.float32)
        _tile_weights_np = _tile_weights_np / _tile_weights_np.sum()
        if use_gpu:
            _tile_weights_t = torch.from_numpy(_tile_weights_np).to(dm.device)

    # 5. Process tiles
    _logged_cpu_rejection = False
    for tile_idx in range(n_tiles):
        y0 = tile_idx * tile_h
        y1 = min(y0 + tile_h, h)

        progress(
            0.05 + 0.90 * (tile_idx / n_tiles),
            f"Stacking tile {tile_idx + 1}/{n_tiles} (rows {y0}–{y1})…",
        )

        # Load tile from each file — disk → RAM one frame at a time
        tile_frames = []
        for i, p in enumerate(paths):
            try:
                tile = _load_fits_tile(p, y0, y1)
            except Exception as exc:
                raise RuntimeError(f"Failed to read tile from frame {i}: {p}") from exc
            tile = tile * scales[i] + shifts[i]
            tile_frames.append(tile)

        # Stack → (N, [C,] tile_h, W)
        tile_stack = np.array(tile_frames, dtype=np.float32)
        del tile_frames

        gpu_rejections = {
            RejectionMethod.SIGMA_CLIP,
            RejectionMethod.LINEAR_FIT,
            RejectionMethod.PERCENTILE_CLIP,
            RejectionMethod.MIN_MAX,
            RejectionMethod.NONE,
        }
        if use_gpu and params.rejection in gpu_rejections:
            # GPU path: move tile to VRAM, do rejection+integration on GPU
            t = torch.from_numpy(tile_stack).to(dm.device)
            del tile_stack

            if params.rejection in (RejectionMethod.SIGMA_CLIP, RejectionMethod.LINEAR_FIT):
                result_t, n_rej = _gpu_sigma_clip(t, params.kappa_low, params.kappa_high, params.max_iterations)
            elif params.rejection == RejectionMethod.PERCENTILE_CLIP:
                result_t, n_rej = _gpu_percentile_clip(t, params.percentile_low, params.percentile_high)
            elif params.rejection == RejectionMethod.MIN_MAX:
                result_t, n_rej = _gpu_min_max(t, params.min_max_reject)
            else:  # NONE
                if (params.integration == IntegrationMethod.WEIGHTED_AVERAGE
                        and _tile_weights_t is not None):
                    # Broadcast weights over spatial dims: (N,) → (N, 1, ..., 1)
                    w_view = _tile_weights_t.view(-1, *([1] * (t.dim() - 1)))
                    result_t = (t * w_view).sum(dim=0)
                else:
                    result_t = t.mean(dim=0)
                n_rej = 0

            total_rejected += n_rej
            tile_result = result_t.cpu().numpy()
            del t, result_t
            if use_gpu and tile_idx % 10 == 9:
                dm.empty_cache()
        else:
            if params.rejection == RejectionMethod.WINSORIZED_SIGMA and not _logged_cpu_rejection:
                log.info("Tiled stack: winsorized sigma rejection on CPU")
                _logged_cpu_rejection = True
            elif params.rejection == RejectionMethod.ESD and not _logged_cpu_rejection:
                log.info("Tiled stack: ESD rejection on CPU")
                _logged_cpu_rejection = True
            # CPU fallback (winsorized, ESD, and when GPU disabled)
            if params.rejection == RejectionMethod.SIGMA_CLIP:
                masked = _reject_sigma_clip(tile_stack, params.kappa_low, params.kappa_high, params.max_iterations)
            elif params.rejection == RejectionMethod.WINSORIZED_SIGMA:
                masked = _reject_winsorized_sigma(tile_stack, params.kappa_low, params.kappa_high, params.max_iterations, params.winsorize_cutoff)
            elif params.rejection == RejectionMethod.LINEAR_FIT:
                masked = _reject_sigma_clip(tile_stack, params.kappa_low, params.kappa_high, params.max_iterations)
            elif params.rejection == RejectionMethod.PERCENTILE_CLIP:
                masked = _reject_percentile_clip(tile_stack, params.percentile_low, params.percentile_high)
            elif params.rejection == RejectionMethod.ESD:
                masked = _reject_esd(tile_stack)
            elif params.rejection == RejectionMethod.MIN_MAX:
                masked = _reject_min_max(tile_stack, params.min_max_reject)
            else:
                masked = np.ma.array(tile_stack, mask=False)

            tile_mask = _get_mask(masked, tile_stack.shape)
            total_rejected += int(np.sum(tile_mask))
            tile_result = _integrate(masked, params.integration, weights=_tile_weights_np)
            del tile_stack, masked

        if is_color:
            output[:, y0:y1, :] = tile_result
        else:
            output[y0:y1, :] = tile_result

    output = np.clip(output, 0, 1).astype(np.float32)

    # Auto-debayer if raw Bayer mosaic detected
    if bayer_pat and not is_color:
        progress(0.97, f"Debayering stacked result ({bayer_pat})…")
        from cosmica.core.debayer import debayer as _debayer
        output = _debayer(output, pattern=bayer_pat, method="vng")
        log.info("Auto-debayered stacked result: %s → %s", bayer_pat, output.shape)

    progress(1.0, f"Stacking complete: {n} files, {total_rejected} pixels rejected")
    log.info("Tiled stacking complete: %d frames, %d rejected", n, total_rejected)

    # Release GPU memory back to OS; PyTorch caches it otherwise
    if use_gpu:
        gc.collect()
        dm.empty_cache()

    # Load header from first file for metadata
    ref_img = load_image(str(paths[0]))
    result_image = ImageData(
        data=output,
        header=ref_img.header.copy(),
        frame_type=FrameType.RESULT,
    )
    return StackResult(image=result_image, n_frames=n, total_rejected=total_rejected)


# ---------------------------------------------------------------------------
# Main Stacking Function
# ---------------------------------------------------------------------------


def stack_images(
    images: list[ImageData],
    params: StackingParams | None = None,
    align: bool = True,
    progress: ProgressCallback = _noop_progress,
) -> StackResult:
    """Stack images: align → normalize → reject outliers → integrate.

    Parameters
    ----------
    images : list[ImageData]
        Input frames (calibrated light frames).
    params : StackingParams, optional
        Stacking configuration.
    align : bool
        If True, align frames before stacking.
    progress : callable
        Progress callback (fraction, message).
    """
    if params is None:
        params = StackingParams()

    n = len(images)
    if n == 0:
        raise ValueError("No images")
    if n == 1:
        return StackResult(image=images[0], n_frames=1)

    # 1. Alignment
    if align:
        aligned_images = align_frames(
            images, params, progress=lambda f, m: progress(f * 0.3, m)
        )
    else:
        aligned_images = list(images)

    # 2. Build numpy stack (N, H, W)
    progress(0.30, "Loading frames into memory...")
    data_stack = np.array([img.data for img in aligned_images], dtype=np.float32)

    # 3. Normalization
    progress(0.35, "Normalizing background levels...")
    data_stack = normalize_stack(data_stack, params.normalization)

    # 4. Rejection
    progress(0.50, f"Running {params.rejection.name}...")

    total_rejected = 0
    mask = None

    shape = data_stack.shape

    if params.rejection == RejectionMethod.SIGMA_CLIP:
        masked_data = _reject_sigma_clip(
            data_stack, params.kappa_low, params.kappa_high, params.max_iterations
        )
        mask = _get_mask(masked_data, shape)
        total_rejected = int(np.sum(mask))

    elif params.rejection == RejectionMethod.WINSORIZED_SIGMA:
        masked_data = _reject_winsorized_sigma(
            data_stack, params.kappa_low, params.kappa_high,
            params.max_iterations, params.winsorize_cutoff,
        )
        mask = _get_mask(masked_data, shape)
        total_rejected = int(np.sum(mask))

    elif params.rejection == RejectionMethod.LINEAR_FIT:
        # Linear fit = normalization (already done) + sigma clip
        masked_data = _reject_sigma_clip(
            data_stack, params.kappa_low, params.kappa_high, params.max_iterations
        )
        mask = _get_mask(masked_data, shape)
        total_rejected = int(np.sum(mask))

    elif params.rejection == RejectionMethod.PERCENTILE_CLIP:
        masked_data = _reject_percentile_clip(
            data_stack, params.percentile_low, params.percentile_high
        )
        mask = _get_mask(masked_data, shape)
        total_rejected = int(np.sum(mask))

    elif params.rejection == RejectionMethod.ESD:
        masked_data = _reject_esd(data_stack)
        mask = _get_mask(masked_data, shape)
        total_rejected = int(np.sum(mask))

    elif params.rejection == RejectionMethod.MIN_MAX:
        masked_data = _reject_min_max(data_stack, params.min_max_reject)
        mask = _get_mask(masked_data, shape)
        total_rejected = int(np.sum(mask))

    else:
        # NONE: no rejection
        masked_data = np.ma.array(data_stack, mask=False)

    # 5. Integration
    progress(0.85, "Integrating frames...")
    _weights = np.asarray(params.frame_weights, dtype=np.float32) if params.frame_weights else None
    result = _integrate(masked_data, params.integration, weights=_weights)

    # 6. Finalize
    result = np.clip(result, 0, 1).astype(np.float32)

    ref_img = aligned_images[0]
    result_image = ImageData(
        data=result,
        header=ref_img.header.copy(),
        frame_type=FrameType.RESULT,
    )

    progress(1.0, f"Stacking complete: {n} frames, {total_rejected} pixels rejected")
    log.info("Stacking complete: %d frames, %d rejected", n, total_rejected)

    return StackResult(
        image=result_image,
        n_frames=n,
        rejection_map=mask,
        total_rejected=total_rejected,
    )
