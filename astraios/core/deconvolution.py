"""Deconvolution — Richardson-Lucy deconvolution with GPU acceleration.

Sharpens images by reversing PSF blurring using iterative RL algorithm
with FFT convolution and optional total variation regularization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask
from astraios.core.star_detection import Star, detect_stars

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

# Sobel kernels for edge-aware deringing
_SOBEL_X = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).view(1, 1, 3, 3)


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class DeconvolutionParams:
    """Parameters for Richardson-Lucy deconvolution."""

    psf_fwhm: float = 3.0  # PSF FWHM in pixels
    iterations: int = 50  # number of RL iterations
    regularization: float = 0.001  # TV regularization strength (0 = off)
    deringing: bool = True  # apply deringing protection
    deringing_amount: float = 0.5  # deringing strength (0-1)


def _create_gaussian_psf(fwhm: float, size: int | None = None) -> np.ndarray:
    """Create a 2D Gaussian PSF kernel.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum in pixels.
    size : int, optional
        Kernel size. If None, computed from FWHM.

    Returns
    -------
    ndarray
        Normalized PSF kernel.
    """
    sigma = fwhm / 2.3548  # FWHM to sigma
    if size is None:
        size = int(np.ceil(fwhm * 3)) | 1  # ensure odd
        size = max(size, 3)

    center = size // 2
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    kernel = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def _fft_convolve_2d(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolve a 2D image with a kernel using FFT.

    Parameters
    ----------
    image : Tensor
        2D image tensor.
    kernel : Tensor
        2D kernel tensor (will be zero-padded to image size).

    Returns
    -------
    Tensor
        Convolution result, same size as image.
    """
    h, w = image.shape
    return _fft_convolve_with_kernel_fft(image, _padded_kernel_fft(kernel, h, w), h, w)


def _padded_kernel_fft(kernel: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """rfft2 of ``kernel`` zero-padded to ``(h, w)`` and centred at the origin.

    Precompute this once for a constant kernel (e.g. the PSF across all
    Richardson-Lucy iterations) instead of rebuilding it every convolution.
    """
    kh, kw = kernel.shape
    padded_kernel = torch.zeros(h, w, device=kernel.device, dtype=kernel.dtype)
    padded_kernel[:kh, :kw] = kernel
    padded_kernel = torch.roll(padded_kernel, (-(kh // 2), -(kw // 2)), dims=(0, 1))
    return torch.fft.rfft2(padded_kernel)


def _fft_convolve_with_kernel_fft(
    image: torch.Tensor, kern_fft: torch.Tensor, h: int, w: int
) -> torch.Tensor:
    """FFT-convolve ``image`` with a kernel whose rfft2 is already computed."""
    return torch.fft.irfft2(torch.fft.rfft2(image) * kern_fft, s=(h, w))


def _tv_regularization(image: torch.Tensor, weight: float) -> torch.Tensor:
    """Compute total variation regularization term.

    Returns a correction factor to multiply with the RL update.
    """
    # Gradient magnitude
    dy = torch.diff(image, dim=0, prepend=image[:1, :])
    dx = torch.diff(image, dim=1, prepend=image[:, :1])
    grad_mag = torch.sqrt(dx**2 + dy**2 + 1e-10)

    # Divergence of normalized gradient
    nx = dx / grad_mag
    ny = dy / grad_mag
    div_x = torch.diff(nx, dim=1, append=nx[:, -1:])
    div_y = torch.diff(ny, dim=0, append=ny[-1:, :])
    divergence = div_x + div_y

    # Regularization factor
    reg = 1.0 / (1.0 - weight * divergence)
    return torch.clamp(reg, 0.5, 2.0)


def richardson_lucy(
    image: np.ndarray,
    params: DeconvolutionParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Apply Richardson-Lucy deconvolution to an image.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W), values in [0, 1].
    params : DeconvolutionParams, optional
        Deconvolution parameters.
    mask : Mask, optional
        If provided, only deconvolve within the mask.
    progress : callable
        Progress callback.

    Returns
    -------
    ndarray
        Deconvolved image.
    """
    if params is None:
        params = DeconvolutionParams()

    dm = get_device_manager()
    # no copy: op never mutates the input; apply_mask reads image directly
    psf_np = _create_gaussian_psf(params.psf_fwhm)

    if image.ndim == 2:
        result = _rl_channel(image, psf_np, params, dm, progress, 0, 1)
    else:
        result = np.empty_like(image)
        n_ch = image.shape[0]
        for ch in range(n_ch):
            result[ch] = _rl_channel(
                image[ch], psf_np, params, dm, progress, ch, n_ch
            )

    result = np.clip(result, 0, 1)
    return apply_mask(image, result, mask)


def _rl_channel(
    channel: np.ndarray,
    psf: np.ndarray,
    params: DeconvolutionParams,
    dm,
    progress: ProgressCallback,
    ch_idx: int,
    n_channels: int,
) -> np.ndarray:
    """Run RL deconvolution on a single channel."""
    ch_offset = ch_idx / n_channels
    ch_scale = 1.0 / n_channels

    try:
        t_img = torch.from_numpy(channel.astype(np.float32)).to(dm.device)
        t_psf = torch.from_numpy(psf.astype(np.float32)).to(dm.device)
    except RuntimeError:
        log.warning("GPU OOM for deconvolution, falling back to CPU")
        t_img = torch.from_numpy(channel.astype(np.float32))
        t_psf = torch.from_numpy(psf.astype(np.float32))

    # Flipped PSF for correlation step
    t_psf_flip = torch.flip(t_psf, [0, 1])

    # RL iterations — no_grad prevents 2-4 GB autograd graph accumulation on 4K images
    estimate = t_img.detach().clone()
    # The PSF is constant across iterations, so transform it once instead of
    # rebuilding + FFTing the padded kernel on every convolution (halves the
    # per-iteration FFT work).
    h, w = t_img.shape
    psf_fft = _padded_kernel_fft(t_psf, h, w)
    psf_flip_fft = _padded_kernel_fft(t_psf_flip, h, w)
    with torch.no_grad():
        for i in range(params.iterations):
            frac = ch_offset + ch_scale * (i / params.iterations)
            progress(frac, f"Deconvolution ch{ch_idx + 1} iter {i + 1}/{params.iterations}")

            # Convolution of estimate with PSF. The intermediates are fresh
            # tensors owned by this loop, so the clamp/div/mul run in place —
            # bit-identical math, ~3 fewer full-frame buffers live per iteration.
            blurred = _fft_convolve_with_kernel_fft(estimate, psf_fft, h, w)
            blurred.clamp_(min=1e-10)

            # Ratio (reuses the blurred buffer)
            ratio = torch.div(t_img, blurred, out=blurred)

            # Correlation with flipped PSF
            correction = _fft_convolve_with_kernel_fft(ratio, psf_flip_fft, h, w)

            # TV regularization
            if params.regularization > 0:
                tv_factor = _tv_regularization(estimate, params.regularization)
                correction.mul_(tv_factor)

            # Update
            estimate.mul_(correction)
            estimate.clamp_(0, 1)

    # Deringing: blend result back toward the original near edges and where
    # ringing oscillations are detected.  The `deringing_amount` slider always
    # has a visible effect because it also targets high-contrast edges (the
    # most likely ringing locations), not just pixels that already ring.
    with torch.no_grad():
        if params.deringing and params.deringing_amount > 0:
            ks = 9
            pad = ks // 2
            orig_4d = t_img.unsqueeze(0).unsqueeze(0)

            # ── Edge mask ─────────────────────────────────────────────
            # Sobel-like edge response ensures the slider works even when
            # RL converges cleanly (no oscillations).
            edge_x = torch.abs(F.conv2d(orig_4d, _SOBEL_X.to(t_img.device), padding=1))
            edge_y = torch.abs(F.conv2d(orig_4d, _SOBEL_Y.to(t_img.device), padding=1))
            edge_mask = torch.clamp(edge_x + edge_y, 0, 1).squeeze()

            # ── Ringing detection mask ────────────────────────────────
            local_max = F.max_pool2d(orig_4d, kernel_size=ks, stride=1, padding=pad).squeeze()
            local_min = -F.max_pool2d(-orig_4d, kernel_size=ks, stride=1, padding=pad).squeeze()
            overshoot = torch.clamp(estimate - local_max, 0, 1)
            undershoot = torch.clamp(local_min - estimate, 0, 1)
            ringing_mask = torch.clamp(overshoot + undershoot, 0, 1)
            ringing_mask = F.max_pool2d(
                ringing_mask.unsqueeze(0).unsqueeze(0), kernel_size=7, stride=1, padding=3
            ).squeeze()

            # ── Composite mask ────────────────────────────────────────
            artifact_mask = torch.clamp(ringing_mask + edge_mask * 0.3, 0, 1)

            blend = artifact_mask * params.deringing_amount
            estimate = estimate * (1 - blend) + t_img * blend

    result = estimate.cpu().numpy() if estimate.device.type != "cpu" else estimate.numpy()
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Spatially-Varying PSF Deconvolution
# ---------------------------------------------------------------------------


@dataclass
class ZonePSF:
    """PSF measurement for a single image zone."""

    row: int  # zone grid row
    col: int  # zone grid col
    center_y: float  # zone center y in pixels
    center_x: float  # zone center x in pixels
    fwhm: float  # measured FWHM in this zone
    n_stars: int  # number of stars used
    ellipticity: float  # measured ellipticity


@dataclass
class SpatialDeconvParams:
    """Parameters for spatially-varying PSF deconvolution.

    Attributes
    ----------
    grid_zones : int
        Number of zones along each axis (NxN grid). 3 means 3x3 = 9 zones.
    iterations : int
        RL iterations per zone.
    regularization : float
        TV regularization strength.
    deringing : bool
        Apply deringing protection.
    deringing_amount : float
        Deringing strength.
    fallback_fwhm : float
        FWHM to use for zones with too few stars.
    min_stars_per_zone : int
        Minimum stars required to trust a zone's PSF measurement.
    blend_radius_fraction : float
        Fraction of zone size used for blending overlap between zones.
        0.2 means 20% overlap on each side.
    """

    grid_zones: int = 3
    iterations: int = 30
    regularization: float = 0.001
    deringing: bool = True
    deringing_amount: float = 0.5
    fallback_fwhm: float = 3.0
    min_stars_per_zone: int = 3
    blend_radius_fraction: float = 0.25


def _fit_star_fwhm(
    gray: np.ndarray,
    star: Star,
    cutout_radius: int = 10,
) -> float | None:
    """Quick radial FWHM measurement from a star cutout.

    Faster than full 2D Gaussian fit — measures the half-max radius
    from the radial profile.
    """
    h, w = gray.shape
    ix, iy = int(round(star.x)), int(round(star.y))
    r = cutout_radius

    if ix - r < 0 or ix + r >= w or iy - r < 0 or iy + r >= h:
        return None

    cutout = gray[iy - r : iy + r + 1, ix - r : ix + r + 1]
    peak = float(cutout.max())
    bg = float(np.percentile(cutout, 10))
    half_max = (peak + bg) / 2.0

    if peak - bg < 0.02:
        return None

    # Radial profile from center
    size = 2 * r + 1
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - r) ** 2.0 + (yy - r) ** 2.0)
    # Find the distance where the profile crosses half-max
    # by checking concentric rings
    for radius in np.arange(0.5, r, 0.5):
        ring = (dist >= radius - 0.5) & (dist < radius + 0.5)
        if ring.sum() == 0:
            continue
        ring_mean = float(cutout[ring].mean())
        if ring_mean <= half_max:
            return float(radius * 2.0)  # diameter = FWHM

    return None


def measure_zone_psf(
    gray: np.ndarray,
    zone_y0: int,
    zone_y1: int,
    zone_x0: int,
    zone_x1: int,
    fallback_fwhm: float = 3.0,
    min_stars: int = 3,
) -> tuple[float, int, float]:
    """Measure PSF in a specific image zone.

    Returns (fwhm, n_stars_used, ellipticity).
    """
    zone = gray[zone_y0:zone_y1, zone_x0:zone_x1]
    if zone.size == 0:
        return fallback_fwhm, 0, 0.0

    try:
        sf = detect_stars(zone, max_stars=80, sigma_threshold=5.0)
    except Exception:
        return fallback_fwhm, 0, 0.0

    # Filter to bright unsaturated stars
    candidates = [
        s for s in sf.stars
        if 0.2 <= s.flux <= 0.95 and s.roundness < 0.5
    ]

    if len(candidates) < min_stars:
        # Relax constraints
        candidates = [s for s in sf.stars if s.flux >= 0.1 and s.flux <= 0.98]

    if len(candidates) < 1:
        return fallback_fwhm, 0, 0.0

    fwhm_values = []
    for star in candidates[:50]:
        fwhm = _fit_star_fwhm(zone, star)
        if fwhm is not None and 1.0 < fwhm < 20.0:
            fwhm_values.append(fwhm)

    if len(fwhm_values) < 1:
        return fallback_fwhm, 0, 0.0

    arr = np.array(fwhm_values)
    # Sigma-clip
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = max(mad * 1.4826, 0.3)
    good = arr[np.abs(arr - med) < 2.5 * sigma]
    if len(good) < 1:
        good = arr

    fwhm_result = float(np.median(good))
    n_used = len(good)

    # Rough ellipticity from star roundness values
    roundness_vals = [s.roundness for s in candidates[:len(good)]]
    ellipticity = float(np.median(roundness_vals)) if roundness_vals else 0.0

    return fwhm_result, n_used, ellipticity


def _build_blend_weights(
    h: int,
    w: int,
    grid_zones: int,
    blend_fraction: float,
) -> list[list[np.ndarray]]:
    """Build smooth blending weight maps for each zone.

    Each zone gets a weight map that is 1.0 in the center and tapers
    smoothly to 0.0 at the edges, allowing neighboring zones to blend.
    """
    zone_h = h / grid_zones
    zone_w = w / grid_zones

    weights = []
    for zy in range(grid_zones):
        row_weights = []
        for zx in range(grid_zones):
            cy = (zy + 0.5) * zone_h
            cx = (zx + 0.5) * zone_w

            # The weight is separable (taper in y times taper in x), so build
            # 1-D tapers and take the outer product. This avoids the two full
            # H×W meshgrid arrays per zone — on a 65 MP image those alone were
            # ~4.7 GB of transient allocations.
            yy = np.arange(h, dtype=np.float32)
            xx = np.arange(w, dtype=np.float32)
            dy = np.abs(yy - cy) / (zone_h * (0.5 + blend_fraction))  # (H,)
            dx = np.abs(xx - cx) / (zone_w * (0.5 + blend_fraction))  # (W,)

            # Cosine taper — 1.0 at center, falls to 0 at edges
            wy = np.where(dy <= 1.0, 0.5 * (1.0 + np.cos(np.pi * np.clip(dy, 0, 1))), 0.0)
            wx = np.where(dx <= 1.0, 0.5 * (1.0 + np.cos(np.pi * np.clip(dx, 0, 1))), 0.0)

            weight = (wy[:, None] * wx[None, :]).astype(np.float32)
            row_weights.append(weight)
        weights.append(row_weights)

    # Normalize so weights sum to 1.0 at each pixel
    total = np.zeros((h, w), dtype=np.float32)
    for zy in range(grid_zones):
        for zx in range(grid_zones):
            total += weights[zy][zx]

    total = np.maximum(total, 1e-10)
    for zy in range(grid_zones):
        for zx in range(grid_zones):
            weights[zy][zx] /= total

    return weights


def richardson_lucy_spatial(
    image: np.ndarray,
    params: SpatialDeconvParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Apply spatially-varying Richardson-Lucy deconvolution.

    Divides the image into a grid of zones, measures the PSF in each zone
    from local stars, deconvolves the full image once per zone with that
    zone's PSF, and blends results using smooth cosine-tapered weights.

    This handles field curvature, coma, and astigmatism that cause the PSF
    to vary across the field — the main cause of edge ringing in global
    deconvolution.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W), float32 in [0, 1].
    params : SpatialDeconvParams, optional
        Spatial deconvolution parameters.
    mask : Mask, optional
        If provided, deconvolution is blended through the mask.
    progress : callable
        Progress callback.

    Returns
    -------
    ndarray
        Deconvolved image.
    """
    if params is None:
        params = SpatialDeconvParams()

    dm = get_device_manager()
    # no copy: op never mutates the input; apply_mask reads image directly

    # Get grayscale for PSF measurement
    if image.ndim == 3:
        gray = np.mean(image, axis=0).astype(np.float32)
        h, w = image.shape[1], image.shape[2]
    else:
        gray = image.astype(np.float32)
        h, w = image.shape

    nz = params.grid_zones
    zone_h = h / nz
    zone_w = w / nz

    log.info(
        "Spatial deconvolution: %dx%d grid, %d iters, image %dx%d",
        nz, nz, params.iterations, w, h,
    )

    # Phase 1: Measure PSF in each zone
    progress(0.0, "Measuring PSF across field...")
    zone_psfs: list[list[ZonePSF]] = []
    for zy in range(nz):
        row = []
        for zx in range(nz):
            y0 = int(zy * zone_h)
            y1 = int(min((zy + 1) * zone_h, h))
            x0 = int(zx * zone_w)
            x1 = int(min((zx + 1) * zone_w, w))

            fwhm, n_stars, ell = measure_zone_psf(
                gray, y0, y1, x0, x1,
                fallback_fwhm=params.fallback_fwhm,
                min_stars=params.min_stars_per_zone,
            )

            zone = ZonePSF(
                row=zy, col=zx,
                center_y=(y0 + y1) / 2.0,
                center_x=(x0 + x1) / 2.0,
                fwhm=fwhm,
                n_stars=n_stars,
                ellipticity=ell,
            )
            row.append(zone)
            log.info(
                "Zone [%d,%d]: FWHM=%.2f px, %d stars, ellipticity=%.3f",
                zy, zx, fwhm, n_stars, ell,
            )
        zone_psfs.append(row)

    # Fill zones with too few stars using neighbor interpolation
    for zy in range(nz):
        for zx in range(nz):
            if zone_psfs[zy][zx].n_stars < params.min_stars_per_zone:
                neighbor_fwhms = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = zy + dy, zx + dx
                        if 0 <= ny < nz and 0 <= nx < nz and (dy != 0 or dx != 0):
                            if zone_psfs[ny][nx].n_stars >= params.min_stars_per_zone:
                                neighbor_fwhms.append(zone_psfs[ny][nx].fwhm)
                if neighbor_fwhms:
                    interp_fwhm = float(np.median(neighbor_fwhms))
                    log.info(
                        "Zone [%d,%d]: interpolated FWHM=%.2f from %d neighbors",
                        zy, zx, interp_fwhm, len(neighbor_fwhms),
                    )
                    zone_psfs[zy][zx] = ZonePSF(
                        row=zy, col=zx,
                        center_y=zone_psfs[zy][zx].center_y,
                        center_x=zone_psfs[zy][zx].center_x,
                        fwhm=interp_fwhm,
                        n_stars=0,
                        ellipticity=zone_psfs[zy][zx].ellipticity,
                    )

    # Phase 2: Build blend weights
    blend_weights = _build_blend_weights(h, w, nz, params.blend_radius_fraction)

    # Phase 3: Deconvolve per zone and blend
    total_zones = nz * nz
    n_ch = image.shape[0] if image.ndim == 3 else 1

    if image.ndim == 2:
        channels = [image]
    else:
        channels = [image[c] for c in range(n_ch)]

    # Zone FWHMs are quantized (the radial-profile fit steps by 0.5 px), so
    # many zones share the exact same PSF. The RL run depends only on
    # (channel, fwhm), so deconvolving once per UNIQUE fwhm and reusing the
    # result is bit-identical to the per-zone runs while skipping most of the
    # work (e.g. 64 zones with 5 unique PSFs = 5 runs instead of 64). Cached
    # results are freed after their last use to bound peak memory.
    from collections import Counter

    fwhm_uses = Counter(
        zone_psfs[zy][zx].fwhm for zy in range(nz) for zx in range(nz)
    )
    log.info(
        "Spatial deconvolution: %d zones share %d unique PSFs",
        total_zones, len(fwhm_uses),
    )

    result_channels = []
    for ch_idx, channel in enumerate(channels):
        accumulated = np.zeros((h, w), dtype=np.float32)
        remaining = dict(fwhm_uses)
        deconv_cache: dict[float, np.ndarray] = {}

        for zy in range(nz):
            for zx in range(nz):
                zone_idx = zy * nz + zx
                overall_frac = (ch_idx * total_zones + zone_idx) / (n_ch * total_zones)
                progress(
                    0.1 + 0.9 * overall_frac,
                    f"Deconvolving ch{ch_idx + 1} zone [{zy},{zx}] "
                    f"FWHM={zone_psfs[zy][zx].fwhm:.1f}px",
                )

                zone_fwhm = zone_psfs[zy][zx].fwhm

                if zone_fwhm in deconv_cache:
                    deconv = deconv_cache[zone_fwhm]
                else:
                    zone_psf_np = _create_gaussian_psf(zone_fwhm)
                    zone_params = DeconvolutionParams(
                        psf_fwhm=zone_fwhm,
                        iterations=params.iterations,
                        regularization=params.regularization,
                        deringing=params.deringing,
                        deringing_amount=params.deringing_amount,
                    )
                    # Deconvolve full channel with this zone's PSF
                    deconv = _rl_channel(
                        channel, zone_psf_np, zone_params, dm,
                        _noop_progress, 0, 1,
                    )
                    if remaining[zone_fwhm] > 1:
                        deconv_cache[zone_fwhm] = deconv

                # Accumulate weighted result (same per-zone order as before)
                accumulated += deconv * blend_weights[zy][zx]

                remaining[zone_fwhm] -= 1
                if remaining[zone_fwhm] == 0:
                    deconv_cache.pop(zone_fwhm, None)

        result_channels.append(np.clip(accumulated, 0, 1).astype(np.float32))

    if image.ndim == 2:
        result = result_channels[0]
    else:
        result = np.stack(result_channels, axis=0)

    progress(1.0, "Spatial deconvolution complete")
    return apply_mask(image, result, mask)
