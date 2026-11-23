from __future__ import annotations

import numpy as np

from astraios.core.diffraction_spikes import DiffractionSpikeParams, _select_stars
from astraios.core.star_detection import detect_stars


def _field() -> np.ndarray:
    """200x200 mono sky (0.02) with one bright Gaussian star (FWHM 3 px) at (100, 100)."""
    h = w = 200
    yy, xx = np.mgrid[0:h, 0:w]
    s = 3.0 / 2.3548
    img = np.full((h, w), 0.02, np.float32)
    img += (0.9 * np.exp(-(((xx - 100) ** 2 + (yy - 100) ** 2) / (2 * s * s)))).astype(np.float32)
    return np.clip(img, 0, 1)


def test_spike_radius_follows_the_star_footprint() -> None:
    """Spike radius is half the star's visible footprint, not half its PSF FWHM."""
    img = _field()
    img_rgb = np.stack([img, img, img], axis=-1)
    star = detect_stars(img, sigma_threshold=5.0).stars[0]
    picked = _select_stars(img_rgb, img, DiffractionSpikeParams())
    assert len(picked) == 1
    assert abs(picked[0].radius - star.footprint_diameter / 2.0) < 1e-6
    # A bright star's footprint is well beyond its FWHM; sizing from fwhm shrank spikes.
    assert picked[0].radius > star.fwhm / 2.0 * 1.5
