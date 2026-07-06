"""Linear Fit — match one image's pixel levels onto a reference image's.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro

Porting note: SASpro ships two different "linear fit" implementations and
this module reconciles them into one robust API:

    - ``pro/linear_fit.py`` (``linear_fit_rgb`` / ``linear_fit_mono_to_ref``)
      is scale-only: it matches per-channel *medians* by a single
      multiplicative ratio, with no intercept and no outlier rejection.
    - ``pro/continuum_subtract.py``'s internal ``_fit_ab`` helper does a
      genuine slope+intercept ``numpy.linalg.lstsq`` regression (used there
      to record the white-balance recipe as an affine map), but is not
      sigma-clipped and subsamples only for speed.

Neither one alone matches "robust/sigma-clipped least squares" — SASpro's
own code doesn't sigma-clip a linear fit anywhere. This port implements the
general case properly (slope *and* intercept, with iterative sigma-clipped
rejection of outlier pixel pairs before the final fit), which is a superset
of both SASpro behaviors: `sigma_clip=False` reproduces `_fit_ab`'s plain
least squares, and forcing `slope` to the median ratio would reproduce the
old scale-only tool. Astropy's `sigma_clip` is used per project convention
(never hand-roll sigma clipping).

GPU decision: this is a scalar regression over (optionally subsampled)
pixel pairs — a few numpy reductions per iteration. No GPU benefit; no
``device_manager`` usage, consistent with other light-math core modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clip
from numpy.typing import NDArray

log = logging.getLogger(__name__)


@dataclass
class LinearFitParams:
    """Parameters for linear fit.

    Attributes:
        per_channel: For color (C, H, W) images, fit each channel
            independently (True) or pool all channels into one global
            slope/intercept (False).
        sigma_clip: Iteratively reject outlier pixel pairs (e.g. stars,
            hot pixels, saturated regions) before the final fit.
        sigma: Sigma-clip rejection threshold, in standard deviations of
            the fit residual.
        max_iters: Maximum sigma-clip / refit iterations.
        sample_size: Random subsample cap on the number of pixel pairs used
            for the fit (for speed on large images); None uses every pixel.
            SASpro's own `_fit_ab` helper caps at 100,000 pixels.
        clip_output: Clip the mapped result to [clip_min, clip_max].
        clip_min: Output floor.
        clip_max: Output ceiling.
    """

    per_channel: bool = True
    sigma_clip: bool = True
    sigma: float = 3.0
    max_iters: int = 5
    sample_size: int | None = 100_000
    clip_output: bool = True
    clip_min: float = 0.0
    clip_max: float = 1.0


def _fit_pair(
    x: NDArray, y: NDArray, params: LinearFitParams, seed: int
) -> tuple[float, float]:
    """Slope+intercept least squares mapping `x` onto `y`, with optional
    iterative sigma-clipped outlier rejection.

    Returns
    -------
    (slope, intercept) : tuple[float, float]
    """
    if params.sample_size is not None and x.size > params.sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.size, params.sample_size, replace=False)
        x, y = x[idx], y[idx]

    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)

    keep = np.ones(x64.shape, dtype=bool)
    slope, intercept = 1.0, 0.0

    for _ in range(max(1, params.max_iters)):
        xs, ys = x64[keep], y64[keep]
        if xs.size < 2 or np.ptp(xs) < 1e-12:
            # Degenerate (too few points or no variance): fall back to a
            # pure offset so the reference median is still matched.
            slope = 1.0
            intercept = float(np.median(ys) - np.median(xs)) if xs.size else 0.0
            break

        design = np.vstack([xs, np.ones_like(xs)]).T
        (slope, intercept), *_rest = np.linalg.lstsq(design, ys, rcond=None)

        if not params.sigma_clip:
            break

        residuals = y64 - (slope * x64 + intercept)
        clipped = sigma_clip(residuals, sigma=params.sigma, maxiters=1, masked=True)
        new_keep = ~np.asarray(clipped.mask, dtype=bool)
        if new_keep.sum() < 2 or np.array_equal(new_keep, keep):
            keep = new_keep if new_keep.sum() >= 2 else keep
            break
        keep = new_keep

    return float(slope), float(intercept)


def compute_linear_fit(
    image: NDArray,
    reference: NDArray,
    params: LinearFitParams | None = None,
) -> tuple[float, float] | tuple[NDArray, NDArray]:
    """Compute the slope+intercept mapping `image` onto `reference`'s levels.

    Parameters
    ----------
    image : ndarray
        Source image, shape (H, W) mono or (C, H, W) color.
    reference : ndarray
        Reference image, same shape as `image`.
    params : LinearFitParams, optional

    Returns
    -------
    (slope, intercept)
        Python floats for mono images, or for color images with
        `params.per_channel=True`, 1D float64 arrays of length C (one
        slope/intercept per channel). With `per_channel=False`, a single
        pair of floats fit across all channels pooled together.
    """
    if params is None:
        params = LinearFitParams()

    image = np.asarray(image)
    reference = np.asarray(reference)
    if image.shape != reference.shape:
        raise ValueError(
            f"image shape {image.shape} does not match reference shape {reference.shape}"
        )

    if image.ndim == 3 and params.per_channel:
        n = image.shape[0]
        slopes = np.empty(n, dtype=np.float64)
        intercepts = np.empty(n, dtype=np.float64)
        for c in range(n):
            slopes[c], intercepts[c] = _fit_pair(
                image[c].ravel(), reference[c].ravel(), params, seed=c
            )
        return slopes, intercepts

    return _fit_pair(image.ravel(), reference.ravel(), params, seed=0)


def linear_fit(
    image: NDArray,
    reference: NDArray,
    params: LinearFitParams | None = None,
) -> NDArray:
    """Map `image`'s pixel levels onto `reference`'s via a linear fit.

    Fits `reference ≈ slope * image + intercept` (robustly, with optional
    sigma-clipped outlier rejection — see `LinearFitParams`) and returns
    `slope * image + intercept`, i.e. `image` rescaled to sit on
    `reference`'s levels. Useful for matching channel backgrounds/scales
    before combination, or matching one frame to another.

    Parameters
    ----------
    image : ndarray
        Image to remap, shape (H, W) mono or (C, H, W) color, float32.
    reference : ndarray
        Reference image, same shape as `image`.
    params : LinearFitParams, optional

    Returns
    -------
    ndarray
        `image` mapped onto `reference`'s levels, same shape and dtype
        float32.
    """
    if params is None:
        params = LinearFitParams()

    image = np.asarray(image, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)

    if image.ndim == 3 and params.per_channel:
        slopes, intercepts = compute_linear_fit(image, reference, params)
        out = np.empty_like(image)
        for c in range(image.shape[0]):
            out[c] = image[c] * slopes[c] + intercepts[c]
    else:
        slope, intercept = compute_linear_fit(image, reference, params)
        out = image * slope + intercept

    if params.clip_output:
        out = np.clip(out, params.clip_min, params.clip_max)

    return out.astype(np.float32, copy=False)
