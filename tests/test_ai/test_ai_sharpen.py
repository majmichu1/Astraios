"""Tests for AI sharpen inference."""

import numpy as np
import pytest

from astraios.ai.inference.sharpen import AISharpenParams, ai_sharpen
from astraios.ai.models.unet import UNet
from astraios.core.masks import Mask


def _small_model():
    """Create a small untrained model for fast tests."""
    return UNet(in_channels=1, out_channels=1, base_features=8, depth=2)


def _blurry_mono(h=64, w=64):
    """Create a blurry mono image with a soft edge."""
    img = np.zeros((h, w), dtype=np.float32)
    img[:, w // 2:] = 0.8
    # Simulate blur with a gradient transition
    ramp_width = 8
    center = w // 2
    for i in range(ramp_width):
        col = center - ramp_width // 2 + i
        if 0 <= col < w:
            img[:, col] = 0.8 * (i / ramp_width)
    return img


def _blurry_color(c=3, h=64, w=64):
    """Create a blurry color image (C, H, W)."""
    rng = np.random.default_rng(42)
    return np.clip(rng.uniform(0.2, 0.6, (c, h, w)), 0, 1).astype(np.float32)


class TestAISharpenParams:
    def test_defaults(self):
        params = AISharpenParams()
        assert params.strength == 1.0
        assert params.tile_size == 512
        assert params.overlap == 64


class TestAISharpen:
    def test_output_shape_mono(self):
        model = _small_model()
        data = _blurry_mono()
        result = ai_sharpen(data, model=model, params=AISharpenParams(tile_size=64, overlap=16))
        assert result.shape == data.shape

    def test_output_shape_color(self):
        model = _small_model()
        data = _blurry_color()
        result = ai_sharpen(data, model=model, params=AISharpenParams(tile_size=64, overlap=16))
        assert result.shape == data.shape

    def test_output_range(self):
        model = _small_model()
        data = _blurry_mono()
        result = ai_sharpen(data, model=model, params=AISharpenParams(tile_size=64, overlap=16))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype(self):
        model = _small_model()
        data = _blurry_mono()
        result = ai_sharpen(data, model=model, params=AISharpenParams(tile_size=64, overlap=16))
        assert result.dtype == np.float32

    def test_strength_zero_returns_original(self):
        """With strength=0, the result should equal the original."""
        model = _small_model()
        data = _blurry_mono()
        params = AISharpenParams(strength=0.0, tile_size=64, overlap=16)
        result = ai_sharpen(data, model=model, params=params)
        np.testing.assert_array_almost_equal(result, data, decimal=5)

    def test_strength_blending(self):
        """Partial strength should blend between original and fully sharpened."""
        model = _small_model()
        data = _blurry_mono()

        full = ai_sharpen(data, model=model, params=AISharpenParams(strength=1.0, tile_size=64, overlap=16))
        half = ai_sharpen(data, model=model, params=AISharpenParams(strength=0.5, tile_size=64, overlap=16))

        dist_to_orig = np.mean(np.abs(half - data))
        dist_full_to_orig = np.mean(np.abs(full - data))
        assert dist_to_orig <= dist_full_to_orig + 1e-5

    def test_mask_application(self):
        """Mask should protect areas where mask=0."""
        model = _small_model()
        data = _blurry_mono(h=64, w=64)

        mask_data = np.zeros((64, 64), dtype=np.float32)
        mask_data[32:, :] = 1.0
        mask = Mask(data=mask_data)

        result = ai_sharpen(data, model=model, params=AISharpenParams(tile_size=64, overlap=16), mask=mask)
        np.testing.assert_array_almost_equal(result[:32, :], data[:32, :])

    def test_no_model_uses_default(self):
        """When model=None, a default model should be created internally."""
        data = _blurry_mono(h=64, w=64)
        params = AISharpenParams(tile_size=64, overlap=16)
        result = ai_sharpen(data, model=None, params=params)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_progress_callback(self):
        model = _small_model()
        data = _blurry_mono()
        calls = []
        def progress(frac, msg):
            calls.append((frac, msg))
        ai_sharpen(data, model=model, params=AISharpenParams(tile_size=64, overlap=16), progress=progress)
        assert len(calls) > 0
