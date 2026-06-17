"""Tests for AI denoise inference."""

import numpy as np
import pytest

from astraios.ai.inference.denoise import AIDenoiseParams, ai_denoise
from astraios.ai.models.unet import UNet
from astraios.core.masks import Mask


def _small_model():
    """Create a small untrained model for fast tests."""
    return UNet(in_channels=1, out_channels=1, base_features=8, depth=2)


def _noisy_mono(h=64, w=64, noise_level=0.1):
    """Create a noisy mono image."""
    rng = np.random.default_rng(42)
    clean = np.full((h, w), 0.4, dtype=np.float32)
    noise = rng.normal(0, noise_level, (h, w)).astype(np.float32)
    return np.clip(clean + noise, 0, 1)


def _noisy_color(c=3, h=64, w=64, noise_level=0.05):
    """Create a noisy color image (C, H, W)."""
    rng = np.random.default_rng(42)
    clean = np.full((c, h, w), 0.4, dtype=np.float32)
    noise = rng.normal(0, noise_level, (c, h, w)).astype(np.float32)
    return np.clip(clean + noise, 0, 1)


class TestAIDenoiseParams:
    def test_defaults(self):
        params = AIDenoiseParams()
        assert params.strength == 1.0
        assert params.tile_size == 256
        assert params.overlap == 32


class TestAIDenoise:
    def test_output_shape_mono(self):
        model = _small_model()
        data = _noisy_mono()
        result = ai_denoise(data, model=model, params=AIDenoiseParams(tile_size=64, overlap=16))
        assert result.shape == data.shape

    def test_output_shape_color(self):
        model = _small_model()
        data = _noisy_color()
        result = ai_denoise(data, model=model, params=AIDenoiseParams(tile_size=64, overlap=16))
        assert result.shape == data.shape

    def test_output_range(self):
        model = _small_model()
        data = _noisy_mono()
        result = ai_denoise(data, model=model, params=AIDenoiseParams(tile_size=64, overlap=16))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype(self):
        model = _small_model()
        data = _noisy_mono()
        result = ai_denoise(data, model=model, params=AIDenoiseParams(tile_size=64, overlap=16))
        assert result.dtype == np.float32

    def test_strength_zero_returns_original(self):
        """With strength=0, the result should equal the original."""
        model = _small_model()
        data = _noisy_mono()
        params = AIDenoiseParams(strength=0.0, tile_size=64, overlap=16)
        result = ai_denoise(data, model=model, params=params)
        np.testing.assert_array_almost_equal(result, data, decimal=5)

    def test_strength_blending(self):
        """Partial strength should blend between original and fully denoised."""
        model = _small_model()
        data = _noisy_mono()

        full = ai_denoise(data, model=model, params=AIDenoiseParams(strength=1.0, tile_size=64, overlap=16))
        half = ai_denoise(data, model=model, params=AIDenoiseParams(strength=0.5, tile_size=64, overlap=16))

        # half should be between data and full (in terms of distance)
        dist_to_orig = np.mean(np.abs(half - data))
        dist_full_to_orig = np.mean(np.abs(full - data))
        # With partial strength, distance to original should be smaller than full strength
        assert dist_to_orig <= dist_full_to_orig + 1e-5

    def test_mask_application(self):
        """Mask should protect areas where mask=0."""
        model = _small_model()
        data = _noisy_mono(h=64, w=64)

        mask_data = np.zeros((64, 64), dtype=np.float32)
        mask_data[32:, :] = 1.0  # Only process bottom half
        mask = Mask(data=mask_data)

        result = ai_denoise(data, model=model, params=AIDenoiseParams(tile_size=64, overlap=16), mask=mask)

        # Top half should be unchanged (protected by mask=0)
        np.testing.assert_array_almost_equal(result[:32, :], data[:32, :])

    def test_no_model_uses_default(self):
        """When model=None, a default model should be created internally."""
        data = _noisy_mono(h=64, w=64)
        params = AIDenoiseParams(tile_size=64, overlap=16)
        result = ai_denoise(data, model=None, params=params)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_progress_callback(self):
        model = _small_model()
        # Image must be larger than tile_size to exercise the tiled path (and progress)
        data = _noisy_mono(h=128, w=128)
        calls = []
        def progress(frac, msg):
            calls.append((frac, msg))
        ai_denoise(data, model=model, params=AIDenoiseParams(tile_size=64, overlap=16), progress=progress)
        assert len(calls) > 0
