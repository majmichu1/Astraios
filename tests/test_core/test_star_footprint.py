from __future__ import annotations

import numpy as np

from astraios.core.star_detection import Star, detect_stars
from astraios.core.star_reduction import create_star_mask


def _field(amp: float = 0.95, fwhm: float = 3.0) -> np.ndarray:
    """160x160 flat sky (0.02) with one Gaussian star at (80, 80)."""
    h = w = 160
    yy, xx = np.mgrid[0:h, 0:w]
    s = fwhm / 2.3548
    img = np.full((h, w), 0.02, np.float32)
    img += (amp * np.exp(-(((xx - 80) ** 2 + (yy - 80) ** 2) / (2 * s * s)))).astype(np.float32)
    return np.clip(img, 0, 1)


def test_footprint_uses_area_when_present() -> None:
    """Diameter is computed from the stored area, not the FWHM."""
    star = Star(x=0, y=0, flux=1, fwhm=3.0, area=np.pi * 16)
    assert abs(star.footprint_diameter - 8.0) < 1e-6


def test_footprint_falls_back_to_fwhm_without_area() -> None:
    """Without an area, the FWHM is the diameter proxy."""
    star = Star(x=0, y=0, flux=1, fwhm=3.0)
    assert star.footprint_diameter == 3.0


def test_detect_stars_fills_area() -> None:
    """Detection stores the thresholded blob area in the star."""
    stars = detect_stars(_field(), sigma_threshold=5.0).stars
    assert len(stars) == 1
    assert stars[0].area > 0


def test_footprint_is_wider_than_fwhm_for_a_bright_star() -> None:
    """A bright star's visible light spreads past its half-maximum width."""
    stars = detect_stars(_field(), sigma_threshold=5.0).stars
    assert stars[0].footprint_diameter > stars[0].fwhm * 1.5


def test_star_mask_covers_the_core() -> None:
    """The star mask is strong at a bright star's core, not thinned by the blur."""
    # Regression: sizing mask blobs from the PSF FWHM left ~0.15 here after blurring.
    mask = create_star_mask(_field()).data
    assert mask[80, 80] > 0.25
