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


class TestTonemapOperators:
    def test_drago_maps_black_to_black(self):
        """Regression: the Drago curve started at log10(2) ~= 0.30, lifting
        all blacks by 30% and compressing the image into [0.30, 1]."""
        from astraios.core.hdr_operators import tonemap_drago

        data = np.linspace(0.0, 100.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        tone = tonemap_drago(data)
        assert tone.min() < 0.05, f"black lifted to {tone.min()}"
        assert abs(tone.max() - 1.0) < 1e-5
        assert np.all(np.diff(tone.ravel()) >= -1e-6)  # monotonic
        assert np.all(np.isfinite(tone))

    def test_drago_color(self):
        from astraios.core.hdr_operators import tonemap_drago

        data = np.random.rand(3, 32, 32).astype(np.float32) * 10.0
        tone = tonemap_drago(data)
        assert tone.shape == data.shape
        assert tone.min() >= 0.0 and tone.max() <= 1.0

    def test_reinhard_in_range(self):
        from astraios.core.hdr_operators import tonemap_reinhard

        data = np.random.rand(32, 32).astype(np.float32) * 50.0
        tone = tonemap_reinhard(data)
        assert tone.min() >= 0.0 and tone.max() <= 1.0

    def test_core_blend_color_shape(self):
        """Regression: the core mask was blurred over (C, H, W) and then
        broadcast via [np.newaxis, ...], inflating color output to
        (C, C, H, W) and crashing downstream transposes."""
        from astraios.core.hdr_operators import tonemap_core_blend

        data = np.random.rand(3, 32, 32).astype(np.float32) * 0.8 + 0.1
        data[:, 16, 16] = 1.0  # bright core pixel
        tone = tonemap_core_blend(data)
        assert tone.shape == data.shape
