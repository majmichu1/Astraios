"""Reference-image prior — fetch a survey cutout of the target.

Uses the CDS **hips2fits** service to render a DSS2 (or other HiPS survey)
cutout *at the user's frame geometry* (centre, field of view, rotation taken
from the plate-solve WCS). Because we request the field directly, the cutout
comes back approximately pixel-aligned with the user's frame — no reprojection
needed. The aligned reference is then thresholded into a **shape-accurate object
mask** (the object's real outline, not just the catalog ellipse).

Entirely best-effort and optional: every function returns ``None`` on any
failure (offline, service down, no WCS, parse error), so callers fall back to
the elliptical object mask / whole-image behaviour.
"""

from __future__ import annotations

import io
import logging
import urllib.parse
import urllib.request

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["fetch_reference_image", "reference_object_mask"]

_HIPS2FITS = "https://alasky.u-strasbg.fr/hips-image-services/hips2fits"
_DEFAULT_SURVEY = "CDS/P/DSS2/red"
_MAX_DIM = 400  # cap the fetched cutout size — a soft mask doesn't need full res


def fetch_reference_image(
    ra_deg: float,
    dec_deg: float,
    fov_deg: float,
    width: int,
    height: int,
    rotation_deg: float = 0.0,
    survey: str = _DEFAULT_SURVEY,
    timeout: float = 20.0,
) -> np.ndarray | None:
    """Fetch a survey cutout at the given field geometry, normalised to [0, 1].

    Parameters mirror the user's frame: ``ra_deg``/``dec_deg`` centre, ``fov_deg``
    angular width, ``width``/``height`` pixel dimensions, ``rotation_deg`` field
    rotation. The cutout is downsampled to at most ``_MAX_DIM`` on the long axis.

    Returns a float32 ``(h, w)`` array in [0, 1] (downsampled), or ``None``.
    """
    if fov_deg <= 0 or width <= 0 or height <= 0:
        return None

    scale = min(1.0, _MAX_DIM / float(max(width, height)))
    w = max(16, int(round(width * scale)))
    h = max(16, int(round(height * scale)))

    query = urllib.parse.urlencode({
        "hips": survey,
        "ra": f"{ra_deg:.6f}",
        "dec": f"{dec_deg:.6f}",
        "fov": f"{fov_deg:.6f}",
        "width": w,
        "height": h,
        "rotation_angle": f"{rotation_deg:.3f}",
        "projection": "TAN",
        "format": "fits",
    })
    url = f"{_HIPS2FITS}?{query}"

    try:
        from astropy.io import fits

        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
        with fits.open(io.BytesIO(raw)) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
    except Exception as exc:
        log.info("Reference image fetch failed: %s", exc)
        return None

    if data.ndim != 2 or data.size == 0 or not np.any(np.isfinite(data)):
        return None

    # Robust normalise to [0, 1] (survey data has arbitrary ADU scale + NaNs).
    finite = data[np.isfinite(data)]
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.5))
    if hi <= lo:
        return None
    out = (np.nan_to_num(data, nan=lo) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def reference_object_mask(
    ra_deg: float,
    dec_deg: float,
    fov_deg: float,
    width: int,
    height: int,
    rotation_deg: float = 0.0,
    survey: str = _DEFAULT_SURVEY,
    timeout: float = 20.0,
    sigma_above_bg: float = 2.5,
) -> np.ndarray | None:
    """Build a soft object mask from a survey reference, sized to (height, width).

    Fetches the reference, thresholds it above its own background (median +
    ``sigma_above_bg``·MAD), smooths to a soft mask, and upsamples to the user's
    frame size. Returns float32 ``(height, width)`` in [0, 1], or ``None``.
    """
    ref = fetch_reference_image(
        ra_deg, dec_deg, fov_deg, width, height, rotation_deg, survey, timeout
    )
    if ref is None:
        return None

    try:
        from scipy import ndimage

        med = float(np.median(ref))
        mad = float(np.median(np.abs(ref - med)))
        thresh = med + sigma_above_bg * max(mad * 1.4826, 1e-4)
        binary = (ref > thresh).astype(np.float32)
        if float(np.mean(binary)) < 1e-4:
            return None  # nothing stood out — don't claim a mask

        # Keep only SIGNIFICANT CONNECTED structure (the real object) and drop
        # scattered noise specks — otherwise a near-blank/noisy reference yields
        # a bogus speckled "mask". A real object is a contiguous region.
        lbl, n = ndimage.label(binary > 0)
        if n == 0:
            return None
        areas = ndimage.sum(np.ones_like(lbl, dtype=np.float64), lbl,
                            index=np.arange(1, n + 1))
        min_area = max(20.0, ref.size * 0.003)
        keep = np.nonzero(areas >= min_area)[0] + 1
        if keep.size == 0:
            return None
        binary = np.isin(lbl, keep).astype(np.float32)

        # Soft mask: dilate a touch to bridge star gaps in the object, then blur.
        binary = ndimage.maximum_filter(binary, size=3)
        soft = ndimage.gaussian_filter(binary, sigma=max(ref.shape) * 0.01)
        soft = np.clip(soft / max(float(soft.max()), 1e-6), 0.0, 1.0)

        # Upsample to the user's frame resolution.
        zoom = (height / soft.shape[0], width / soft.shape[1])
        full = ndimage.zoom(soft, zoom, order=1)
        full = np.clip(full, 0.0, 1.0).astype(np.float32)
        # zoom can be off by a pixel — pad/crop to the exact size.
        full = full[:height, :width]
        if full.shape != (height, width):
            fixed = np.zeros((height, width), np.float32)
            fixed[:full.shape[0], :full.shape[1]] = full
            full = fixed
        return full
    except Exception as exc:
        log.info("Reference mask build failed: %s", exc)
        return None
