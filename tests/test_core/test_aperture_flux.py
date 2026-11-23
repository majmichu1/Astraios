from __future__ import annotations

import numpy as np

from astraios.core.color_calibration import _aperture_flux


def _star(sky: float, colour: tuple[float, float, float]) -> np.ndarray:
    """(3, 80, 80) image: flat sky plus one Gaussian star (FWHM 3 px) at (40, 40)."""
    h = w = 80
    yy, xx = np.mgrid[0:h, 0:w]
    s = 3.0 / 2.3548
    psf = np.exp(-(((xx - 40) ** 2 + (yy - 40) ** 2) / (2 * s * s)))
    img = np.full((3, h, w), sky, np.float32)
    for c in range(3):
        img[c] += (0.6 * colour[c] * psf).astype(np.float32)
    return img


def test_sky_does_not_pull_colour_toward_grey() -> None:
    """Star colour ratios come out right on a bright neutral sky."""
    flux = _aperture_flux(_star(0.08, (1.0, 0.8, 0.6)), 40.0, 40.0, 6)
    assert flux is not None
    assert abs(flux[1] / flux[0] - 0.8) < 0.01
    assert abs(flux[2] / flux[0] - 0.6) < 0.01


def test_result_is_independent_of_sky_level() -> None:
    """Raising the sky changes nothing once it is subtracted."""
    dark = _aperture_flux(_star(0.0, (1.0, 0.8, 0.6)), 40.0, 40.0, 6)
    bright = _aperture_flux(_star(0.1, (1.0, 0.8, 0.6)), 40.0, 40.0, 6)
    assert dark is not None and bright is not None
    assert np.allclose(dark, bright, rtol=1e-3)


def test_returns_none_when_the_star_is_off_the_frame() -> None:
    """No box pixels inside the image means no measurement."""
    assert _aperture_flux(_star(0.05, (1.0, 1.0, 1.0)), -50.0, -50.0, 4) is None
