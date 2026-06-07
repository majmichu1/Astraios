"""Tests for plate solving."""

import pytest
import numpy as np

from cosmica.core.plate_solve import PlateSolveParams, PlateSolveResult, plate_solve


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
