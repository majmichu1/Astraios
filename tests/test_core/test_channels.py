"""Tests for channel operations."""

import numpy as np

from astraios.core.channels import (
    combine_channels,
    extract_luminance,
    hsl_to_rgb,
    replace_channel,
    replace_luminance,
    rgb_to_hsl,
    split_channels,
)


class TestSplitCombine:
    def test_split_color(self):
        image = np.random.rand(3, 50, 50).astype(np.float32)
        channels = split_channels(image)
        assert len(channels) == 3
        for ch in channels:
            assert ch.shape == (50, 50)

    def test_split_mono(self):
        image = np.random.rand(50, 50).astype(np.float32)
        channels = split_channels(image)
        assert len(channels) == 1

    def test_combine_roundtrip(self):
        image = np.random.rand(3, 50, 50).astype(np.float32)
        channels = split_channels(image)
        combined = combine_channels(channels)
        np.testing.assert_array_almost_equal(combined, image)

    def test_combine_single(self):
        channel = np.random.rand(50, 50).astype(np.float32)
        combined = combine_channels([channel])
        np.testing.assert_array_almost_equal(combined, channel)


class TestExtractLuminance:
    def test_color(self):
        image = np.ones((3, 50, 50), dtype=np.float32) * 0.5
        lum = extract_luminance(image)
        assert lum.shape == (50, 50)
        np.testing.assert_allclose(lum.mean(), 0.5, atol=0.01)

    def test_mono(self):
        image = np.ones((50, 50), dtype=np.float32) * 0.3
        lum = extract_luminance(image)
        np.testing.assert_allclose(lum.mean(), 0.3, atol=0.01)

    def test_weighted(self):
        image = np.zeros((3, 50, 50), dtype=np.float32)
        image[1] = 1.0  # green only
        lum = extract_luminance(image)
        np.testing.assert_allclose(lum.mean(), 0.7152, atol=0.01)


class TestReplaceChannel:
    def test_replace_red(self):
        image = np.zeros((3, 50, 50), dtype=np.float32)
        new_red = np.ones((50, 50), dtype=np.float32)
        result = replace_channel(image, 0, new_red)
        np.testing.assert_array_equal(result[0], 1.0)
        np.testing.assert_array_equal(result[1], 0.0)

    def test_original_unchanged(self):
        image = np.zeros((3, 50, 50), dtype=np.float32)
        new_red = np.ones((50, 50), dtype=np.float32)
        replace_channel(image, 0, new_red)
        np.testing.assert_array_equal(image[0], 0.0)  # original not modified


class TestReplaceLuminance:
    def test_replace_luminance(self):
        image = np.ones((3, 50, 50), dtype=np.float32) * 0.5
        new_lum = np.ones((50, 50), dtype=np.float32) * 0.8
        result = replace_luminance(image, new_lum)
        result_lum = extract_luminance(result)
        np.testing.assert_allclose(result_lum.mean(), 0.8, atol=0.05)

    def test_mono_returns_luminance(self):
        image = np.ones((50, 50), dtype=np.float32) * 0.3
        new_lum = np.ones((50, 50), dtype=np.float32) * 0.7
        result = replace_luminance(image, new_lum)
        np.testing.assert_allclose(result.mean(), 0.7, atol=0.01)


class TestHSLConversion:
    def test_roundtrip(self):
        image = np.random.rand(3, 50, 50).astype(np.float32)
        hsl = rgb_to_hsl(image)
        rgb = hsl_to_rgb(hsl)
        np.testing.assert_array_almost_equal(rgb, image, decimal=4)

    def test_pure_red_hue(self):
        image = np.zeros((3, 10, 10), dtype=np.float32)
        image[0] = 1.0
        hsl = rgb_to_hsl(image)
        # H should be near 0 for red
        np.testing.assert_allclose(hsl[0].mean(), 0.0, atol=0.01)
        # S should be 1
        np.testing.assert_allclose(hsl[1].mean(), 1.0, atol=0.01)

    def test_gray_has_zero_saturation(self):
        image = np.ones((3, 10, 10), dtype=np.float32) * 0.5
        hsl = rgb_to_hsl(image)
        np.testing.assert_allclose(hsl[1].mean(), 0.0, atol=0.01)
