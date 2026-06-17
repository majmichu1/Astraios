"""Tests for local contrast enhancement."""

import numpy as np

from astraios.core.local_contrast import LocalContrastParams, local_contrast_enhance


class TestLocalContrast:
    def test_mono_enhancement(self):
        data = np.random.rand(100, 100).astype(np.float32) * 0.5
        result = local_contrast_enhance(data)
        assert result.shape == (100, 100)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_color_enhancement(self):
        data = np.random.rand(3, 100, 100).astype(np.float32) * 0.5
        result = local_contrast_enhance(data)
        assert result.shape == (3, 100, 100)

    def test_output_in_range(self):
        data = np.random.rand(100, 100).astype(np.float32)
        params = LocalContrastParams(clip_limit=5.0)
        result = local_contrast_enhance(data, params)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_amount_blending(self):
        data = np.random.rand(100, 100).astype(np.float32) * 0.5
        full = local_contrast_enhance(data, LocalContrastParams(amount=1.0))
        half = local_contrast_enhance(data, LocalContrastParams(amount=0.5))
        # Half amount should be closer to original than full
        diff_full = np.mean(np.abs(full - data))
        diff_half = np.mean(np.abs(half - data))
        assert diff_half <= diff_full + 0.01

    def test_mask_support(self):
        from astraios.core.masks import Mask
        data = np.random.rand(100, 100).astype(np.float32) * 0.5
        mask_data = np.zeros((100, 100), dtype=np.float32)
        mask_data[:50] = 1.0
        mask = Mask(data=mask_data)
        result = local_contrast_enhance(data, mask=mask)
        # Bottom half should be unchanged
        np.testing.assert_allclose(result[50:], data[50:], atol=1e-5)

    def test_tile_size_parameter(self):
        data = np.random.rand(100, 100).astype(np.float32) * 0.5
        r1 = local_contrast_enhance(data, LocalContrastParams(tile_size=4))
        r2 = local_contrast_enhance(data, LocalContrastParams(tile_size=16))
        # Different tile sizes should produce different results
        assert not np.allclose(r1, r2, atol=1e-4)
