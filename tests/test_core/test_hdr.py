"""Tests for HDR composition."""

import numpy as np
import pytest

from astraios.core.hdr import HDRMethod, HDRParams, hdr_compose


class TestHDRCompose:
    def test_mertens_fusion(self):
        # Two exposures: dark and bright
        dark = np.random.rand(50, 50).astype(np.float32) * 0.3
        bright = np.random.rand(50, 50).astype(np.float32) * 0.7 + 0.3
        result = hdr_compose([dark, bright])
        assert result.shape == (50, 50)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_weighted_average(self):
        dark = np.ones((50, 50), dtype=np.float32) * 0.2
        bright = np.ones((50, 50), dtype=np.float32) * 0.8
        params = HDRParams(method=HDRMethod.WEIGHTED_AVERAGE)
        result = hdr_compose([dark, bright], params)
        assert result.shape == (50, 50)
        # Result should be between the two
        assert 0.2 < result.mean() < 0.8

    def test_color_images(self):
        dark = np.random.rand(3, 50, 50).astype(np.float32) * 0.3
        bright = np.random.rand(3, 50, 50).astype(np.float32) * 0.7 + 0.3
        result = hdr_compose([dark, bright])
        assert result.shape == (3, 50, 50)

    def test_three_exposures(self):
        low = np.random.rand(50, 50).astype(np.float32) * 0.2
        mid = np.random.rand(50, 50).astype(np.float32) * 0.3 + 0.3
        high = np.random.rand(50, 50).astype(np.float32) * 0.3 + 0.7
        result = hdr_compose([low, mid, high])
        assert result.shape == (50, 50)

    def test_too_few_images_raises(self):
        single = np.random.rand(50, 50).astype(np.float32)
        with pytest.raises(ValueError):
            hdr_compose([single])

    def test_output_in_range(self):
        imgs = [np.random.rand(50, 50).astype(np.float32) for _ in range(3)]
        result = hdr_compose(imgs)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
