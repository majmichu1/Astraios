"""Tests for tiled inference."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from astraios.ai.inference.tiled import _create_blend_weight, tiled_inference
from astraios.ai.models.unet import UNet


def _identity_model():
    """Create a small model that approximates identity (returns input)."""
    model = UNet(in_channels=1, out_channels=1, base_features=8, depth=2)
    model.eval()
    return model


def _mono_image(h=100, w=100, value=0.5):
    """Create a mono test image."""
    return np.full((h, w), value, dtype=np.float32)


def _gradient_image(h=100, w=100):
    """Create a mono image with a horizontal gradient 0 to 1."""
    ramp = np.linspace(0, 1, w, dtype=np.float32)
    return np.broadcast_to(ramp, (h, w)).copy()


class TestCreateBlendWeight:
    def test_shape(self):
        w = _create_blend_weight(tile_size=64, overlap=16)
        assert w.shape == (64, 64)

    def test_center_is_one(self):
        """Center of the blend weight should be 1.0."""
        w = _create_blend_weight(tile_size=64, overlap=16)
        assert w[32, 32] == pytest.approx(1.0)

    def test_corners_near_zero(self):
        """Corners should be near zero due to cosine ramp."""
        w = _create_blend_weight(tile_size=64, overlap=16)
        assert w[0, 0] < 0.01

    def test_symmetric(self):
        """Blend weight should be symmetric along both axes."""
        w = _create_blend_weight(tile_size=64, overlap=16)
        np.testing.assert_array_almost_equal(w, w[::-1, :])
        np.testing.assert_array_almost_equal(w, w[:, ::-1])

    def test_no_overlap(self):
        """With overlap=0, blend weight should be all ones."""
        w = _create_blend_weight(tile_size=64, overlap=0)
        np.testing.assert_array_almost_equal(w, np.ones((64, 64), dtype=np.float32))

    def test_values_in_range(self):
        w = _create_blend_weight(tile_size=128, overlap=32)
        assert w.min() >= 0.0
        assert w.max() <= 1.0


class TestTiledInference:
    def test_output_shape_matches_input(self):
        model = _identity_model()
        data = _mono_image(h=100, w=100)
        result = tiled_inference(data, model, tile_size=64, overlap=16)
        assert result.shape == data.shape

    def test_small_image(self):
        """Image smaller than tile_size should still work."""
        model = _identity_model()
        data = _mono_image(h=32, w=32)
        result = tiled_inference(data, model, tile_size=64, overlap=16)
        assert result.shape == (32, 32)

    def test_output_in_range(self):
        model = _identity_model()
        data = _gradient_image(h=64, w=64)
        result = tiled_inference(data, model, tile_size=64, overlap=16)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype(self):
        model = _identity_model()
        data = _mono_image(h=64, w=64)
        result = tiled_inference(data, model, tile_size=64, overlap=16)
        assert result.dtype == np.float32

    def test_progress_callback_called(self):
        """Progress callback should be called with fraction and message."""
        model = _identity_model()
        data = _mono_image(h=64, w=64)
        calls = []
        def progress(frac, msg):
            calls.append((frac, msg))
        tiled_inference(data, model, tile_size=64, overlap=16, progress=progress)
        # Should have at least the final call at 1.0
        assert len(calls) > 0
        assert calls[-1][0] == 1.0

    def test_large_image_multiple_tiles(self):
        """Larger image should produce multiple tiles."""
        model = _identity_model()
        data = _mono_image(h=200, w=200)
        result = tiled_inference(data, model, tile_size=64, overlap=16)
        assert result.shape == (200, 200)

    def test_non_square_image(self):
        model = _identity_model()
        data = _mono_image(h=64, w=128)
        result = tiled_inference(data, model, tile_size=64, overlap=16)
        assert result.shape == (64, 128)
