"""Background Extraction — automated gradient removal.

Independently implements polynomial surface fitting for background modeling.
Samples the background at grid points (CPU, fast on sparse data), fits a
polynomial surface (CPU lstsq, tiny matrix), then evaluates and smooths the
model on the full image grid (GPU via PyTorch for large images).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

log = logging.getLogger(__name__)


@dataclass
class BackgroundParams:
    grid_size: int = 8  # NxN sample grid
    box_size: int = 32  # sample box size in pixels
    polynomial_order: int = 3  # polynomial surface degree
    sigma_clip: float = 2.5  # reject bright samples (stars)
    per_pixel_sigma: float = 2.5  # within-sample pixel rejection (PixInsight Tolerance)
    smoothing: float = 0.5  # Gaussian smoothing of the model (fraction of image size)
    manual_points: list[tuple[int, int]] = field(default_factory=list)  # user-placed sample points
    object_aware: bool = False  # if True, use exclusion_mask to protect target signal
    exclusion_mask: np.ndarray | None = field(default=None, repr=False)  # (H,W) float32 mask, 1=exclude from sampling


def extract_background(
    data: np.ndarray,
    params: BackgroundParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and subtract background gradient.

    Input: float32 (H, W) for mono or (C, H, W) for color.
    Returns: (corrected_image, background_model)
    """
    if params is None:
        params = BackgroundParams()

    if data.ndim == 3:
        n_ch = data.shape[0]
        corrected = np.empty_like(data)
        bg_model = np.empty_like(data)
        for ch in range(n_ch):
            corrected[ch], bg_model[ch] = _extract_single_channel(data[ch], params)
        return corrected, bg_model
    else:
        return _extract_single_channel(data, params)


def _extract_single_channel(
    channel: np.ndarray,
    params: BackgroundParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract background from a single channel."""
    h, w = channel.shape

    # Generate sample points
    sample_points = _generate_sample_points(h, w, params)

    # Measure background level at each sample point
    samples_x = []
    samples_y = []
    samples_val = []

    for py, px in sample_points:
        # Object-aware: skip points inside the exclusion mask
        if (params.object_aware and params.exclusion_mask is not None
                and 0 <= py < h and 0 <= px < w
                and params.exclusion_mask[py, px] > 0.5):
            continue

        val = _measure_sample(channel, py, px, params.box_size, params.per_pixel_sigma)
        if val is not None:
            samples_x.append(px)
            samples_y.append(py)
            samples_val.append(val)

    if len(samples_val) < 6:
        log.warning("Too few background samples (%d), returning original", len(samples_val))
        return channel.copy(), np.zeros_like(channel)

    samples_x = np.array(samples_x, dtype=np.float64)
    samples_y = np.array(samples_y, dtype=np.float64)
    samples_val = np.array(samples_val, dtype=np.float64)

    # Sigma-clip to reject star-contaminated samples
    median_val = np.median(samples_val)
    mad = np.median(np.abs(samples_val - median_val))
    sigma = max(mad * 1.4826, 1e-10)  # MAD to sigma conversion

    keep = np.abs(samples_val - median_val) < params.sigma_clip * sigma
    samples_x = samples_x[keep]
    samples_y = samples_y[keep]
    samples_val = samples_val[keep]

    log.info("Background samples: %d kept, %d rejected", keep.sum(), (~keep).sum())

    if len(samples_val) < 3:
        log.warning("Too few samples after clipping, returning original")
        return channel.copy(), np.zeros_like(channel)

    # Auto-reduce polynomial degree if there aren't enough samples.
    # A degree-d surface has (d+1)(d+2)/2 terms; need at least that many samples.
    n_samples = len(samples_val)
    poly_degree = params.polynomial_order
    while poly_degree > 1:
        min_required = (poly_degree + 1) * (poly_degree + 2) // 2
        if n_samples >= min_required:
            break
        poly_degree -= 1
    if poly_degree < params.polynomial_order:
        log.warning(
            "Background: reduced polynomial degree %d→%d "
            "(need %d samples for degree %d, have %d)",
            params.polynomial_order, poly_degree,
            (params.polynomial_order + 1) * (params.polynomial_order + 2) // 2,
            params.polynomial_order, n_samples,
        )

    # Normalize coordinates to [-1, 1]
    x_norm = (samples_x / w) * 2 - 1
    y_norm = (samples_y / h) * 2 - 1

    # Fit polynomial surface (CPU, tiny matrix — fast)
    coeffs = _fit_polynomial_surface(x_norm, y_norm, samples_val, poly_degree)

    # Evaluate model over full image — GPU if available, else CPU
    bg_model = _evaluate_polynomial_gpu(h, w, coeffs, poly_degree)

    # Optional smoothing (GPU Gaussian)
    if params.smoothing > 0:
        smooth_sigma = params.smoothing * min(h, w) / 10
        bg_model = _gaussian_smooth_gpu(bg_model, smooth_sigma)

    bg_model = bg_model.astype(np.float32)

    # Subtract and re-normalize (GPU path for large images)
    corrected = _subtract_and_floor_gpu(channel, bg_model)

    return corrected, bg_model


def create_object_exclusion_mask(
    image_shape: tuple[int, int],
    objects: list[dict],
    plate_scale: float,
    margin_factor: float = 1.3,
) -> np.ndarray:
    """Create an exclusion mask from known object positions.

    Parameters
    ----------
    image_shape : (H, W)
        Image dimensions.
    objects : list of dict
        Each dict has 'center_x', 'center_y' (pixel coords) and
        'radius_arcmin' (angular radius of the object).
    plate_scale : float
        Arcseconds per pixel.
    margin_factor : float
        Expand object radius by this factor for safety margin.

    Returns
    -------
    ndarray
        Float32 mask, 1.0 inside objects, 0.0 outside.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.ogrid[0:h, 0:w]

    for obj in objects:
        cx = obj["center_x"]
        cy = obj["center_y"]
        # Convert angular radius to pixels
        radius_px = (obj["radius_arcmin"] * 60.0 / plate_scale) * margin_factor
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        mask[dist_sq <= radius_px**2] = 1.0

    return mask


def _generate_sample_points(
    h: int, w: int, params: BackgroundParams
) -> list[tuple[int, int]]:
    """Generate grid sample points plus any manual points."""
    points = []
    margin = params.box_size
    grid = params.grid_size

    ys = np.linspace(margin, h - margin, grid).astype(int)
    xs = np.linspace(margin, w - margin, grid).astype(int)

    for y in ys:
        for x in xs:
            points.append((int(y), int(x)))

    points.extend(params.manual_points)
    return points


def _measure_sample(
    channel: np.ndarray, cy: int, cx: int, box_size: int,
    rejection_sigma: float = 2.5,
) -> float | None:
    """Measure the background level in a box centered at (cy, cx).

    Two-stage rejection (similar to PixInsight DBE Tolerance):
    1. Darkest-fraction heuristic: use only darkest 50% of pixels.
    2. Per-pixel sigma rejection within the selected pixels: iteratively
       reject pixels deviating more than rejection_sigma from the median.

    This removes both star contamination and hot/cold pixel outliers
    within each sample box, producing more robust background estimates.
    """
    h, w = channel.shape
    half = box_size // 2
    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)

    box = channel[y0:y1, x0:x1]
    if box.size == 0:
        return None

    # Stage 1: Darkest 50% to avoid bright stars
    sorted_vals = np.sort(box.ravel())
    n = len(sorted_vals)
    dark_half = sorted_vals[: n // 2]
    if len(dark_half) < 4:
        return None

    # Stage 2: Per-pixel sigma rejection within darkest half
    vals = dark_half.astype(np.float64)
    for _ in range(3):  # up to 3 iterations
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        sigma = max(mad * 1.4826, 1e-10)
        mask = np.abs(vals - med) < rejection_sigma * sigma
        if mask.sum() < 3:
            return float(med)
        vals = vals[mask]

    return float(np.median(vals))


def _fit_polynomial_surface(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, order: int
) -> np.ndarray:
    """Fit a 2D polynomial surface to scattered data points."""
    terms = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            terms.append(x**i * y**j)
    A = np.column_stack(terms)
    result, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    return result


@torch.no_grad()
def _evaluate_polynomial_gpu(h: int, w: int, coeffs: np.ndarray, order: int) -> np.ndarray:
    """Evaluate 2D polynomial over a full (H, W) grid, using GPU when available.

    Coordinates are normalised to [-1, 1].  Returns a float32 numpy array.
    """
    try:
        import torch

        from cosmica.core.device_manager import get_device_manager
        device = get_device_manager().device

        # Build coordinate grids in float32 on device
        ys = torch.linspace(-1.0, 1.0, h, dtype=torch.float32, device=device)
        xs = torch.linspace(-1.0, 1.0, w, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W) each

        # Pre-compute powers: x_pows[i] = xx**i, y_pows[j] = yy**j
        max_pow = order + 1
        x_pows = [None] * max_pow
        y_pows = [None] * max_pow
        x_pows[0] = torch.ones_like(xx)
        y_pows[0] = torch.ones_like(yy)
        for p in range(1, max_pow):
            x_pows[p] = x_pows[p - 1] * xx
            y_pows[p] = y_pows[p - 1] * yy

        result = torch.zeros(h, w, dtype=torch.float32, device=device)
        idx = 0
        for i in range(order + 1):
            for j in range(order + 1 - i):
                result += float(coeffs[idx]) * x_pows[i] * y_pows[j]
                idx += 1

        return result.cpu().numpy()

    except Exception:
        # CPU fallback — still faster than before because we use float32
        ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)   # (H, W) float32

        max_pow = order + 1
        x_pows = [None] * max_pow
        y_pows = [None] * max_pow
        x_pows[0] = np.ones((h, w), dtype=np.float32)
        y_pows[0] = np.ones((h, w), dtype=np.float32)
        for p in range(1, max_pow):
            x_pows[p] = x_pows[p - 1] * xx
            y_pows[p] = y_pows[p - 1] * yy

        result = np.zeros((h, w), dtype=np.float32)
        idx = 0
        for i in range(order + 1):
            for j in range(order + 1 - i):
                result += np.float32(coeffs[idx]) * x_pows[i] * y_pows[j]
                idx += 1
        return result


@torch.no_grad()
def _subtract_and_floor_gpu(channel: np.ndarray, bg_model: np.ndarray) -> np.ndarray:
    """Subtract background model only — preserve linear data for stretch/export."""
    try:
        import torch

        from cosmica.core.device_manager import get_device_manager
        device = get_device_manager().device

        t = torch.from_numpy(channel).to(device=device, dtype=torch.float32)
        m = torch.from_numpy(bg_model).to(device=device, dtype=torch.float32)
        return (t - m).cpu().numpy().astype(np.float32)

    except Exception:
        return (channel - bg_model).astype(np.float32)


@torch.no_grad()
def _gaussian_smooth_gpu(model: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to a 2D model array, using GPU when available."""
    if sigma < 0.5:
        return model
    try:
        import math

        import torch
        import torch.nn.functional as F

        from cosmica.core.device_manager import get_device_manager
        device = get_device_manager().device

        # Build 1D Gaussian kernel
        radius = int(math.ceil(sigma * 3))
        size = 2 * radius + 1
        kernel_1d = torch.arange(size, dtype=torch.float32, device=device) - radius
        kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)
        kernel_1d /= kernel_1d.sum()

        # Separable 2D via two 1D convolutions
        t = torch.from_numpy(model).to(device=device, dtype=torch.float32)
        t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        kh = kernel_1d.view(1, 1, -1, 1)
        kw = kernel_1d.view(1, 1, 1, -1)
        t = F.conv2d(t, kh, padding=(radius, 0))
        t = F.conv2d(t, kw, padding=(0, radius))

        return t.squeeze().cpu().numpy()

    except Exception:
        from scipy import ndimage
        return ndimage.gaussian_filter(model.astype(np.float64), sigma=sigma).astype(np.float32)
