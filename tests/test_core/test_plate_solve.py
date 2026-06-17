"""Tests for plate solving."""

import pytest
import numpy as np

from astraios.core import plate_solve as ps
from astraios.core.plate_solve import PlateSolveParams, PlateSolveResult, plate_solve


def _star_image():
    """Create a synthetic image with detectable stars."""
    image = np.ones((200, 200), dtype=np.float32) * 0.05
    yy, xx = np.mgrid[0:200, 0:200]
    positions = [
        (30, 30), (170, 30), (100, 100), (30, 170), (170, 170),
        (60, 80), (140, 60), (80, 140), (150, 150), (40, 120),
    ]
    for sx, sy in positions:
        dist_sq = (xx - sx) ** 2 + (yy - sy) ** 2
        star = 0.8 * np.exp(-dist_sq / (2 * 3.0**2))
        image += star.astype(np.float32)
    return np.clip(image, 0, 1)


class TestPlateSolve:
    def test_returns_result(self):
        image = _star_image()
        result = plate_solve(image)
        assert isinstance(result, PlateSolveResult)

    def test_finds_stars(self):
        image = _star_image()
        result = plate_solve(image)
        assert result.n_stars_matched > 0

    def test_with_scale_hint(self):
        image = _star_image()
        params = PlateSolveParams(scale_hint=1.0)
        with pytest.raises(NotImplementedError):
            plate_solve(image, params)

    def test_empty_image_fails(self):
        image = np.ones((100, 100), dtype=np.float32) * 0.05
        result = plate_solve(image)
        assert not result.success

    def test_geometry_estimation(self):
        image = _star_image()
        result = plate_solve(image)
        if result.success:
            # Rotation should be a finite number
            assert np.isfinite(result.rotation)


class TestExternalSolverAdapters:
    """The array-based astap/net wrappers delegate to star_catalog and
    convert its dict into a PlateSolveResult with a canonical wcs_header."""

    def test_astap_delegates_and_converts_fits_header(self, monkeypatch):
        img = np.zeros((64, 64), dtype=np.float32)
        fake = {
            "ra": 10.0, "dec": 41.0, "scale": 2.0, "rotation": 5.0,
            "wcs_header": {
                "CRVAL1": 10.0, "CRVAL2": 41.0,
                "CD1_1": 2.0 / 3600, "CD2_2": 2.0 / 3600,
                "CRPIX1": 32, "CRPIX2": 32,
            },
        }
        from astraios.core import star_catalog
        monkeypatch.setattr(star_catalog, "plate_solve_astap", lambda *a, **k: fake)
        result = ps.plate_solve_astap(img)
        assert result.success
        assert abs(result.ra_center - 10.0) < 1e-6
        assert "ra_center" in result.wcs_header  # canonical format for consumers

    def test_conversion_handles_flat_dict_and_none(self):
        img = np.zeros((32, 32), dtype=np.float32)
        flat = {"ra": 5.0, "dec": -3.0, "scale": 1.5, "rotation": 0.0, "wcs_header": {}}
        result = ps._result_from_solver_dict(flat, img)
        assert result.success and abs(result.pixel_scale - 1.5) < 1e-6
        assert ps._result_from_solver_dict(None, img).success is False

    def test_net_requires_api_key(self):
        img = np.zeros((32, 32), dtype=np.float32)
        assert ps.plate_solve_astrometry_net(img, api_key=None).success is False
