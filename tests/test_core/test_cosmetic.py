"""Tests for cosmetic correction."""

import numpy as np
import pytest

from astraios.core.cosmetic import CosmeticParams, CosmeticResult, cosmetic_correction
from astraios.core.masks import Mask


class TestCosmeticCorrection:
    """Tests for the cosmetic correction module."""

    def test_no_defects(self):
        """Clean image should have minimal corrections."""
        image = np.random.rand(100, 100).astype(np.float32) * 0.1 + 0.1
        result = cosmetic_correction(image)
        assert isinstance(result, CosmeticResult)
        assert result.data.shape == image.shape

    def test_hot_pixel_detection(self):
        """Should detect and correct bright outlier pixels."""
        image = np.ones((100, 100), dtype=np.float32) * 0.1
        # Add hot pixels
        image[50, 50] = 0.95
        image[30, 30] = 0.90
        result = cosmetic_correction(image, CosmeticParams(hot_sigma=3.0))
        assert result.hot_pixels > 0
        # Hot pixels should be replaced with neighbor median
        assert result.data[50, 50] < 0.5

    def test_dead_pixel_detection(self):
        """Should detect zero-value pixels."""
        image = np.ones((100, 100), dtype=np.float32) * 0.2
        image[50, 50] = 0.0
        image[30, 30] = 0.0
        result = cosmetic_correction(image, CosmeticParams(detect_dead=True))
        assert result.dead_pixels >= 2
        # Dead pixels should be filled
        assert result.data[50, 50] > 0.0

    def test_color_image(self):
        """Should work with multi-channel images."""
        image = np.random.rand(3, 100, 100).astype(np.float32) * 0.1 + 0.1
        image[0, 50, 50] = 0.95  # hot pixel in red
        result = cosmetic_correction(image)
        assert result.data.shape == (3, 100, 100)

    def test_mask_support(self):
        """Mask should limit corrections to masked areas."""
        image = np.ones((100, 100), dtype=np.float32) * 0.1
        image[50, 50] = 0.95  # hot pixel inside mask
        image[20, 20] = 0.95  # hot pixel outside mask

        mask_data = np.zeros((100, 100), dtype=np.float32)
        mask_data[40:60, 40:60] = 1.0  # only center
        mask = Mask(data=mask_data)

        result = cosmetic_correction(image, mask=mask)
        # Pixel at (50, 50) should be corrected (inside mask)
        assert result.data[50, 50] < 0.5
        # Pixel at (20, 20) should be uncorrected (outside mask)
        assert result.data[20, 20] > 0.8

    def test_total_corrected(self):
        """total_corrected should sum all defect types."""
        image = np.ones((100, 100), dtype=np.float32) * 0.1
        image[50, 50] = 0.95
        image[30, 30] = 0.0
        result = cosmetic_correction(image)
        assert result.total_corrected == result.hot_pixels + result.cold_pixels + result.dead_pixels
