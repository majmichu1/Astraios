"""Automatic Background Extraction — polynomial and RBF-based gradient removal.

Models and removes large-scale background gradients (light pollution,
vignetting, sky glow) from astrophotography images.

Default model: 2-D polynomial surface (degree 2).  Polynomials extrapolate
smoothly everywhere and have no edge-extrapolation artefacts.  RBF
(Radial Basis Function) is available as an alternative for complex,
non-polynomial gradients but is more sensitive to sampling density at
the image boundaries.

Requires scipy >= 1.7 for ``scipy.interpolate.RBFInterpolator``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)


@dataclass
class ABEParams:
    """Parameters for Automatic Background Extraction.

    Attributes
    ----------
    grid_size : int
        Number of sample points along each axis (NxN grid).
    box_size : int
        Side length in pixels of the measurement box at each sample point.
    sigma_clip : float
        Sigma-clipping threshold for rejecting star-contaminated samples.
        Samples whose measured value exceeds ``median + sigma_clip * MAD``
        are discarded.
    sample_percentile : float
        Within each measurement box, use this percentile of pixel values
        (rather than the median of the darkest half).  Lower values (5–15)
        are more resistant to nebulosity contamination.  Default 10 ≈ using
        the ~10th-brightest pixel in each box.
    darkest_fraction : float
        After measuring all grid samples, keep only the darkest
        ``darkest_fraction`` of them for the polynomial/RBF fit.  Set to
        1.0 to use all surviving samples.  0.5 means only the 50% darkest
        samples drive the fit, protecting the model against nebula-
        contaminated points that passed sigma-clipping.
    model_type : str
        Background model type: ``"polynomial"`` (default) or ``"rbf"``.
    polynomial_degree : int
        Degree of the 2-D polynomial surface (1=plane, 2=quadratic,
        3=cubic).
    rbf_kernel : str
        Kernel function for ``RBFInterpolator``.
    rbf_smoothing : float
        Smoothing parameter for the RBF interpolator.
    correction_mode : str
        ``"subtraction"`` subtracts the model; ``"division"`` divides.
    iterations : int
        Number of refinement iterations.
    """

    grid_size: int = 10
    box_size: int = 48
    sigma_clip: float = 2.0
    sample_percentile: float = 10.0   # percentile within each box
    darkest_fraction: float = 0.6     # keep darkest 60% of samples; rejects nebula-contaminated center boxes
    model_type: str = "polynomial"
    polynomial_degree: int = 4        # degree 4 captures cos⁴ vignette falloff at edges
    rbf_kernel: str = "thin_plate_spline"
    rbf_smoothing: float = 0.5
    correction_mode: str = "subtraction"  # subtraction for gradients/light-pollution; division for multiplicative optical vignetting
    iterations: int = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def abe_extract(
    data: np.ndarray,
    params: ABEParams | None = None,
    mask: Mask | None = None,
    progress=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and remove background using polynomial or RBF interpolation.

    Parameters
    ----------
    data : ndarray
        Input image as float32 in [0, 1].  Shape ``(H, W)`` for mono or
        ``(C, H, W)`` for color.
    params : ABEParams, optional
        Extraction parameters.  Uses defaults when *None*.
    mask : Mask, optional
        Selective-processing mask.  Where the mask is 0 the original
        pixel values are preserved; where 1 the corrected values are
        used.

    Returns
    -------
    corrected : ndarray
        Background-corrected image, same shape and dtype as *data*,
        clipped to [0, 1].
    background_model : ndarray
        The estimated background model, same shape as *data*.
    """
    if params is None:
        params = ABEParams()
    _prog = progress if callable(progress) else (lambda f, m: None)

    log.info(
        "ABE: model=%s, grid=%d, box=%d, mode=%s, iters=%d",
        params.model_type,
        params.grid_size,
        params.box_size,
        params.correction_mode,
        params.iterations,
    )

    if data.ndim == 3:
        n_channels = data.shape[0]
        corrected = np.empty_like(data)
        bg_model = np.empty_like(data)
        for ch in range(n_channels):
            _prog(ch / n_channels, f"ABE channel {ch + 1}/{n_channels}…")
            log.info("ABE: processing channel %d/%d", ch + 1, n_channels)
            corrected[ch], bg_model[ch] = _extract_channel(data[ch], params)
        _prog(1.0, "ABE complete")
    else:
        _prog(0.0, "ABE processing…")
        corrected, bg_model = _extract_channel(data, params)
        _prog(1.0, "ABE complete")

    corrected = apply_mask(data, corrected, mask)
    return corrected, bg_model


# ---------------------------------------------------------------------------
# Single-channel pipeline
# ---------------------------------------------------------------------------


def _extract_channel(
    channel: np.ndarray,
    params: ABEParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full ABE pipeline on a single (H, W) channel."""
    h, w = channel.shape

    if params.correction_mode == "subtraction":
        total_bg = np.zeros((h, w), dtype=np.float32)
    else:
        total_bg = np.ones((h, w), dtype=np.float32)

    working = channel.copy()

    for it in range(params.iterations):
        log.info("ABE iteration %d/%d", it + 1, params.iterations)

        points, values = _sample_background(working, params)

        n_terms = (params.polynomial_degree + 1) * (params.polynomial_degree + 2) // 2
        if len(values) < max(n_terms, 6):
            log.warning(
                "ABE: only %d samples survived clipping (need %d) at iteration %d — "
                "skipping further iterations",
                len(values),
                n_terms,
                it + 1,
            )
            break

        if params.model_type == "rbf":
            iter_bg = _build_rbf_model(points, values, h, w, params)
        else:
            iter_bg = _build_poly_model(points, values, h, w, params)

        if params.correction_mode == "division":
            # Normalize the model to its mean so division flattens spatial
            # variation without changing overall image brightness.
            # Raw background values (0.2–0.5) would amplify by 2–5×; normalised
            # values (~1.0 at the bright point, <1 at dark corners) only correct
            # the relative illumination falloff, keeping the mean level constant.
            model_mean = float(np.mean(iter_bg))
            if model_mean > 1e-7:
                norm_bg = iter_bg / model_mean
            else:
                norm_bg = np.ones_like(iter_bg)
            safe_norm = np.maximum(norm_bg, 1e-4)
            working = np.clip(working / safe_norm, 0.0, 1.0).astype(np.float32)
            total_bg *= safe_norm
        else:
            working = np.clip(working - iter_bg, 0.0, 1.0).astype(np.float32)
            total_bg += iter_bg

    return working, total_bg


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _generate_grid_points(
    h: int,
    w: int,
    grid_size: int,
    box_size: int,
) -> np.ndarray:
    """Return an (N, 2) array of (row, col) sample-point coordinates.

    Includes a ring of edge samples (just inside the image boundary) so
    the model is constrained by real data near the borders and does not
    extrapolate freely there.
    """
    margin = box_size // 2
    ys = np.linspace(margin, h - margin - 1, grid_size).astype(int)
    xs = np.linspace(margin, w - margin - 1, grid_size).astype(int)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    interior = np.column_stack([yy.ravel(), xx.ravel()])

    # Edge ring: sample close to all four borders using a smaller box.
    # These constrain the polynomial/RBF at the image boundary so it
    # doesn't extrapolate to wrong values and leave bright residuals.
    edge_m = max(2, box_size // 4)
    n_edge = max(4, grid_size)
    edge_ys = np.linspace(edge_m, h - edge_m - 1, n_edge).astype(int)
    edge_xs = np.linspace(edge_m, w - edge_m - 1, n_edge).astype(int)
    edge_pts = []
    for ey in edge_ys:
        edge_pts.append((ey, edge_m))            # left border
        edge_pts.append((ey, w - edge_m - 1))   # right border
    for ex in edge_xs:
        edge_pts.append((edge_m, ex))            # top border
        edge_pts.append((h - edge_m - 1, ex))   # bottom border

    edge_arr = np.array(edge_pts, dtype=int)
    return np.vstack([interior, edge_arr])


def _measure_background_at(
    channel: np.ndarray,
    row: int,
    col: int,
    box_size: int,
    percentile: float = 10.0,
) -> float | None:
    """Measure background level in a box centred at (*row*, *col*).

    Uses a low percentile (default 10th) rather than the median so that
    sample boxes partially occupied by nebulosity still yield a sky-level
    estimate.  A high sigma-clip and ``darkest_fraction`` in the caller
    then discard any boxes where even the 10th percentile is lifted by
    diffuse emission.
    """
    h, w = channel.shape
    half = box_size // 2
    r0 = max(0, row - half)
    r1 = min(h, row + half)
    c0 = max(0, col - half)
    c1 = min(w, col + half)

    box = channel[r0:r1, c0:c1]
    if box.size == 0:
        return None

    return float(np.percentile(box.ravel(), percentile))


def _sample_background(
    channel: np.ndarray,
    params: ABEParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate grid samples, sigma-clip to reject stars, then keep only
    the darkest fraction to prevent nebulosity from biasing the model.

    Two-pass rejection:
    1. Sigma-clip: drops boxes where even the low-percentile value is lifted
       by bright stars or compact emission.
    2. Darkest-fraction filter: among remaining samples, keep only the
       ``darkest_fraction`` darkest ones.  On a nebula field the brightest
       samples are nebula-contaminated; discarding them makes the polynomial
       fit to actual sky rather than to diffuse emission.
    """
    h, w = channel.shape
    grid_pts = _generate_grid_points(h, w, params.grid_size, params.box_size)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for row, col in grid_pts:
        val = _measure_background_at(
            channel, int(row), int(col), params.box_size,
            percentile=params.sample_percentile,
        )
        if val is not None:
            rows.append(int(row))
            cols.append(int(col))
            vals.append(val)

    if len(vals) == 0:
        log.warning("ABE: no valid background samples obtained")
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)

    points = np.column_stack([rows, cols]).astype(np.float64)
    values = np.asarray(vals, dtype=np.float64)

    # Pass 1: sigma-clip (reject star-bright boxes)
    median_val = np.median(values)
    mad = np.median(np.abs(values - median_val))
    threshold = median_val + params.sigma_clip * mad
    keep = values <= threshold
    points, values = points[keep], values[keep]

    # Pass 2: darkest-fraction filter (protect nebula signal)
    if params.darkest_fraction < 1.0 and len(values) > 4:
        n_keep = max(4, int(len(values) * params.darkest_fraction))
        order = np.argsort(values)
        keep2 = order[:n_keep]
        points, values = points[keep2], values[keep2]

    log.info(
        "ABE sampling: %d measured → %d after sigma-clip+darkest-fraction",
        len(vals),
        len(values),
    )

    return points, values


# ---------------------------------------------------------------------------
# Polynomial background model
# ---------------------------------------------------------------------------


def _build_poly_model(
    points: np.ndarray,
    values: np.ndarray,
    h: int,
    w: int,
    params: ABEParams,
) -> np.ndarray:
    """Fit a 2-D polynomial surface to the sample points.

    The fit is performed on the CPU (tiny matrix), and the full-image
    evaluation is offloaded to the GPU via ``_evaluate_polynomial_gpu``
    from ``astraios.core.background`` (GPU→CPU tensor ops; falls back to
    CPU numpy automatically if GPU is unavailable).

    Coordinate convention matches ``_evaluate_polynomial_gpu``:
      x = col (normalised), y = row (normalised).
    Basis: x^i · y^j for 0 ≤ i, 0 ≤ j, i+j ≤ degree.

    Polynomial surfaces extrapolate smoothly by construction — no edge
    artefacts regardless of the sample distribution.
    """
    from astraios.core.background import _evaluate_polynomial_gpu

    degree = params.polynomial_degree

    # Normalise — x=col, y=row to match _evaluate_polynomial_gpu convention
    x_n = points[:, 1] / max(w - 1, 1) * 2.0 - 1.0   # col → x
    y_n = points[:, 0] / max(h - 1, 1) * 2.0 - 1.0   # row → y

    # Build Vandermonde: basis is x^i * y^j (same ordering as GPU evaluator)
    def _vander(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
        cols = []
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                cols.append(x_arr ** i * y_arr ** j)
        return np.column_stack(cols)

    A = _vander(x_n, y_n)

    # Least-squares fit (CPU, tiny matrix — microseconds)
    coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)

    # Evaluate over full image on GPU (or CPU fallback)
    bg_full = _evaluate_polynomial_gpu(h, w, coeffs, degree)

    # Only clamp from below: a negative model would *add* to the image when
    # subtracted, creating bright halos.  Do NOT clamp from above — a tight
    # upper bound clips the polynomial at image corners where the true
    # background can be higher than anywhere in the sample interior, leaving
    # a bright residual (the halo the user sees).
    bg_full = np.maximum(bg_full, 0.0)

    log.info(
        "ABE poly model: degree=%d, bg range [%.5f, %.5f]",
        degree,
        float(bg_full.min()),
        float(bg_full.max()),
    )
    return bg_full.astype(np.float32)


# ---------------------------------------------------------------------------
# RBF background model (alternative)
# ---------------------------------------------------------------------------

_DOWNSAMPLE_STEP = 8  # evaluate RBF every Nth pixel, then upscale


def _add_boundary_anchors(
    points: np.ndarray,
    values: np.ndarray,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend the sample set with anchor points at the image boundary.

    Anchors prevent wild RBF extrapolation beyond the convex hull of the
    measurement grid by extending that hull to the full image extent.
    Each anchor takes the value of the nearest real sample point.
    """
    if len(points) == 0:
        return points, values

    n_edge = 5

    anchors: list[tuple[float, float]] = []
    for r in (0.0, float(h - 1)):
        for c in (0.0, float(w - 1)):
            anchors.append((r, c))

    for frac in np.linspace(0, 1, n_edge + 2)[1:-1]:
        anchors.append((0.0,           frac * (w - 1)))
        anchors.append((float(h - 1),  frac * (w - 1)))
        anchors.append((frac * (h - 1), 0.0))
        anchors.append((frac * (h - 1), float(w - 1)))

    anchors_arr = np.array(anchors, dtype=np.float64)
    anchor_vals: list[float] = []
    for ry, cx in anchors_arr:
        dists = np.sqrt((points[:, 0] - ry) ** 2 + (points[:, 1] - cx) ** 2)
        anchor_vals.append(float(values[int(np.argmin(dists))]))

    all_points = np.vstack([points, anchors_arr])
    all_values = np.concatenate([values, np.array(anchor_vals, dtype=np.float64)])
    return all_points, all_values


def _build_rbf_model(
    points: np.ndarray,
    values: np.ndarray,
    h: int,
    w: int,
    params: ABEParams,
) -> np.ndarray:
    """Fit an RBF to the sample points and evaluate over the full image.

    For efficiency the RBF is evaluated on a coarse grid (every
    ``_DOWNSAMPLE_STEP`` pixels) and then upscaled to the full resolution
    with bicubic interpolation.

    Boundary anchor points are added before fitting so that the RBF hull
    covers the full image extent, reducing edge-extrapolation artefacts.
    """
    step = _DOWNSAMPLE_STEP

    aug_points, aug_values = _add_boundary_anchors(points, values, h, w)

    rbf = RBFInterpolator(
        aug_points,
        aug_values,
        kernel=params.rbf_kernel,
        smoothing=params.rbf_smoothing,
    )

    coarse_rows = np.arange(0, h, step)
    coarse_cols = np.arange(0, w, step)
    cr, cc = np.meshgrid(coarse_rows, coarse_cols, indexing="ij")
    eval_pts = np.column_stack([cr.ravel(), cc.ravel()])

    coarse_vals = rbf(eval_pts).reshape(cr.shape)

    coarse_f32 = coarse_vals.astype(np.float32)
    bg_full = cv2.resize(coarse_f32, (w, h), interpolation=cv2.INTER_CUBIC)

    # Same rationale as _build_poly_model: no upper clamp.
    bg_full = np.nan_to_num(bg_full, nan=0.0)
    bg_full = np.maximum(bg_full, 0.0)

    return bg_full.astype(np.float32)
