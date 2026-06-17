"""Tests for histogram transformation."""

import numpy as np

from astraios.core.histogram_transform import HistogramTransformParams, histogram_transform
from astraios.core.masks import Mask


class TestHistogramTransform:
    """Tests for histogram transformation."""

    def test_neutral_params(self):
        """Neutral params (bp=0, mt=0.5, wp=1) should be near identity."""
        image = np.random.rand(100, 100).astype(np.float32)
        params = HistogramTransformParams(black_point=0.0, midtone=0.5, white_point=1.0)
        result = histogram_transform(image, params)
        np.testing.assert_array_almost_equal(result, image, decimal=3)

    def test_black_point_clips(self):
        """Raising black point should darken shadows."""
        image = np.linspace(0, 1, 10000).reshape(100, 100).astype(np.float32)
        params = HistogramTransformParams(black_point=0.3)
        result = histogram_transform(image, params)
        # Values below 0.3 should be clipped to 0
        assert result[:30, :].mean() < 0.01

    def test_white_point_clips(self):
        """Lowering white point should brighten highlights."""
        image = np.linspace(0, 1, 10000).reshape(100, 100).astype(np.float32)
        params = HistogramTransformParams(white_point=0.5, midtone=0.5)
        result = histogram_transform(image, params)
        # Values above 0.5 should be at or near 1.0
        assert result[55:, :].mean() > 0.9

    def test_midtone_brightens(self):
        """Low midtone value should brighten the image."""
        image = np.ones((100, 100), dtype=np.float32) * 0.3
        params = HistogramTransformParams(midtone=0.15)
        result = histogram_transform(image, params)
        assert result.mean() > image.mean()

    def test_color_image(self):
        """Should work with multi-channel images."""
        image = np.random.rand(3, 50, 50).astype(np.float32)
        result = histogram_transform(image)
        assert result.shape == (3, 50, 50)

    def test_mask_support(self):
        """Mask should protect unmasked regions."""
        image = np.ones((100, 100), dtype=np.float32) * 0.5
        params = HistogramTransformParams(midtone=0.1)

        mask_data = np.zeros((100, 100), dtype=np.float32)
        mask_data[50:, :] = 1.0
        mask = Mask(data=mask_data)

        result = histogram_transform(image, params, mask=mask)
        # Top half unchanged
        np.testing.assert_allclose(result[:50, :].mean(), 0.5, atol=0.01)
        # Bottom half brightened
        assert result[50:, :].mean() > 0.6
