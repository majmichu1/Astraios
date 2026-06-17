"""Tests for drizzle integration."""

import numpy as np
import pytest

from astraios.core.drizzle import DrizzleParams, DrizzleResult, drizzle_integrate


class TestDrizzle:
    def test_single_frame_upscales(self):
        data = np.random.rand(20, 20).astype(np.float32) * 0.5
        params = DrizzleParams(scale=2, use_gpu=False)
        result = drizzle_integrate(
            [data],
            transforms=[np.eye(2, 3, dtype=np.float32)],
            params=params,
        )
        assert isinstance(result, DrizzleResult)
        assert result.data.shape == (40, 40)
        assert result.output_scale == 2

    def test_output_in_range(self):
        data = np.random.rand(20, 20).astype(np.float32)
        params = DrizzleParams(scale=2, use_gpu=False)
        result = drizzle_integrate(
            [data],
            transforms=[np.eye(2, 3, dtype=np.float32)],
            params=params,
        )
        assert result.data.min() >= 0.0
        assert result.data.max() <= 1.0

    def test_weight_map_positive(self):
        data = np.random.rand(20, 20).astype(np.float32)
        params = DrizzleParams(scale=2, use_gpu=False)
        result = drizzle_integrate(
            [data],
            transforms=[np.eye(2, 3, dtype=np.float32)],
            params=params,
        )
        assert result.weight_map.max() > 0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            drizzle_integrate([])

    def test_color_image(self):
        data = np.random.rand(3, 20, 20).astype(np.float32)
        params = DrizzleParams(scale=2, use_gpu=False)
        result = drizzle_integrate(
            [data],
            transforms=[np.eye(2, 3, dtype=np.float32)],
            params=params,
        )
        assert result.data.shape == (3, 40, 40)

    def test_multi_frame_identity_transforms(self):
        """Multiple frames with identity transforms should average correctly."""
        data = np.full((20, 20), 0.5, dtype=np.float32)
        images = [data.copy() for _ in range(3)]
        transforms = [np.eye(2, 3, dtype=np.float32)] * 3
        params = DrizzleParams(scale=2, use_gpu=False)
        result = drizzle_integrate(images, transforms=transforms, params=params)
        assert result.data.shape == (40, 40)
        assert result.n_frames == 3
        # Output should be close to 0.5
        np.testing.assert_allclose(result.data[result.weight_map > 0].mean(), 0.5, atol=0.05)

    def test_multi_frame_with_translation(self):
        """Frames with sub-pixel translations should produce valid combined output."""
        base = np.zeros((30, 30), dtype=np.float32)
        base[15, 15] = 1.0  # bright star at centre
        images = [base.copy() for _ in range(4)]
        # Small translations (sub-pixel at 2× scale = 1 output pixel)
        transforms = [
            np.array([[1, 0, 0.0], [0, 1, 0.0]], dtype=np.float32),
            np.array([[1, 0, 0.5], [0, 1, 0.0]], dtype=np.float32),
            np.array([[1, 0, 0.0], [0, 1, 0.5]], dtype=np.float32),
            np.array([[1, 0, 0.5], [0, 1, 0.5]], dtype=np.float32),
        ]
        params = DrizzleParams(scale=2, use_gpu=False)
        result = drizzle_integrate(images, transforms=transforms, params=params)
        assert result.data.shape == (60, 60)
        assert result.n_frames == 4
        assert result.data.max() > 0

    def test_scale_1_passthrough(self):
        """Scale=1 should produce same-size output."""
        data = np.random.rand(25, 25).astype(np.float32)
        params = DrizzleParams(scale=1, use_gpu=False)
        result = drizzle_integrate(
            [data], transforms=[np.eye(2, 3, dtype=np.float32)], params=params
        )
        assert result.data.shape == (25, 25)
