"""Tests for photometric color calibration."""

import numpy as np

from astraios.core.color_calibration import (
    ColorCalibrationParams,
    ColorCalibrationResult,
    color_calibrate,
)


def _color_image_with_cast():
    """Create a color image with a blue cast."""
    rng = np.random.default_rng(42)
    image = np.zeros((3, 200, 200), dtype=np.float32)
    image[0] = 0.2 + rng.normal(0, 0.02, (200, 200))  # R dim
    image[1] = 0.25 + rng.normal(0, 0.02, (200, 200))  # G slightly brighter
    image[2] = 0.35 + rng.normal(0, 0.02, (200, 200))  # B brightest (blue cast)

    # Add some stars
    yy, xx = np.mgrid[0:200, 0:200]
    for sx, sy in [(50, 50), (150, 50), (100, 100), (50, 150), (150, 150)]:
        dist_sq = (xx - sx) ** 2 + (yy - sy) ** 2
        star = 0.7 * np.exp(-dist_sq / (2 * 3.0**2))
        for ch in range(3):
            image[ch] += star

    return np.clip(image, 0, 1).astype(np.float32)


class TestColorCalibration:
    def test_produces_result(self):
        image = _color_image_with_cast()
        result = color_calibrate(image)
        assert isinstance(result, ColorCalibrationResult)
        assert result.data.shape == image.shape
        assert len(result.correction_factors) == 3

    def test_correction_factors_valid(self):
        image = _color_image_with_cast()
        result = color_calibrate(image)
        # All factors should be positive and <= 1.0 (normalized)
        for f in result.correction_factors:
            assert 0 < f <= 1.01

    def test_background_neutralization(self):
        image = _color_image_with_cast()
        params = ColorCalibrationParams(neutralize_background=True)
        result = color_calibrate(image, params)
        # Background should be more neutral after calibration
        bg_r = result.data[0, :10, :10].mean()
        bg_g = result.data[1, :10, :10].mean()
        bg_b = result.data[2, :10, :10].mean()
        spread = max(bg_r, bg_g, bg_b) - min(bg_r, bg_g, bg_b)
        # Original spread is about 0.15, should be reduced
        assert spread < 0.1

    def test_mono_unchanged(self):
        image = np.ones((100, 100), dtype=np.float32) * 0.5
        result = color_calibrate(image)
        np.testing.assert_array_equal(result.data, image)

    def test_custom_reference(self):
        image = _color_image_with_cast()
        params = ColorCalibrationParams(
            white_reference="custom",
            custom_rgb=(1.0, 0.9, 0.8),
        )
        result = color_calibrate(image, params)
        assert result.correction_factors == (1.0, 0.9, 0.8)
