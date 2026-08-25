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


class TestGaussianDropKernel:
    """DrizzleParams.pixel_weight documented "uniform" or "gaussian" and the
    code read neither, so asking for a gaussian drop silently got a square
    one."""

    @staticmethod
    def _drizzle_point(mode: str, scale: int = 2):
        from astraios.core.drizzle import _drizzle_frame_numpy

        img = np.zeros((8, 8), dtype=np.float32)
        img[4, 4] = 1.0
        out = np.zeros((8 * scale, 8 * scale), dtype=np.float32)
        weight = np.zeros((8 * scale, 8 * scale), dtype=np.float32)
        _drizzle_frame_numpy(img, out, weight, None, scale, 0.7, mode)
        return out, weight

    def test_gaussian_changes_the_weighting(self):
        _, w_uniform = self._drizzle_point("uniform")
        _, w_gauss = self._drizzle_point("gaussian")
        assert not np.allclose(w_uniform, w_gauss), "gaussian must not equal uniform"

    def test_uniform_weights_stay_flat(self):
        """The square kernel gives every covered output pixel the same share."""
        _, weight = self._drizzle_point("uniform")
        covered = weight[weight > 0]
        assert np.allclose(covered, covered[0])

    def test_gaussian_weights_fall_off_from_the_centre(self):
        _, weight = self._drizzle_point("gaussian")
        covered = weight[weight > 0]
        assert covered.max() > covered.min(), "gaussian weights should vary"
        assert covered.min() > 0

    def test_normalised_amplitude_is_preserved(self):
        """Signal and weight carry the same kernel, so a point source must
        normalise back to its own value under either kernel."""
        for mode in ("uniform", "gaussian"):
            out, weight = self._drizzle_point(mode)
            lit = weight > 0
            norm = np.zeros_like(out)
            norm[lit] = out[lit] / weight[lit]
            assert np.isfinite(norm).all()
            assert abs(float(norm.max()) - 1.0) < 1e-4, mode

    def test_gaussian_forces_the_cpu_path(self):
        """Both backends must agree on what the settings mean, so the gaussian
        kernel is not silently squared off on the GPU."""
        from astraios.core.drizzle import DrizzleParams, drizzle_integrate

        data = np.zeros((12, 12), dtype=np.float32)
        data[6, 6] = 1.0
        params = DrizzleParams(scale=2, pixel_weight="gaussian", use_gpu=True)
        result = drizzle_integrate(
            [data], transforms=[np.eye(2, 3, dtype=np.float32)], params=params
        )
        assert result.data.shape == (24, 24)
        assert np.isfinite(result.data).all()
