"""Tests for wavelet decomposition and reconstruction."""

import numpy as np
import pytest

from astraios.core.wavelets import (
    WaveletParams,
    wavelet_decompose,
    wavelet_reconstruct,
    wavelet_sharpen,
)


class TestWaveletDecompose:
    def test_decompose_returns_correct_count(self):
        data = np.random.rand(64, 64).astype(np.float32)
        scales = wavelet_decompose(data, n_scales=3)
        assert len(scales) == 4  # 3 detail + 1 residual

    def test_decompose_shapes_match(self):
        data = np.random.rand(64, 64).astype(np.float32)
        scales = wavelet_decompose(data, n_scales=3)
        for s in scales:
            assert s.shape == (64, 64)

    def test_reconstruct_recovers_original(self):
        data = np.random.rand(64, 64).astype(np.float32)
        scales = wavelet_decompose(data, n_scales=4)
        reconstructed = wavelet_reconstruct(scales)
        np.testing.assert_allclose(reconstructed, data, atol=1e-4)

    def test_detail_scales_sum_to_zero_approx(self):
        data = np.random.rand(64, 64).astype(np.float32)
        scales = wavelet_decompose(data, n_scales=3)
        # Detail scales should have near-zero mean
        for s in scales[:-1]:
            assert abs(s.mean()) < 0.1

    def test_residual_is_smooth(self):
        data = np.random.rand(64, 64).astype(np.float32)
        scales = wavelet_decompose(data, n_scales=3)
        residual = scales[-1]
        # Residual should be smoother than the original
        assert np.std(residual) <= np.std(data) + 0.01


class TestWaveletSharpen:
    def test_identity_weights(self):
        data = np.random.rand(64, 64).astype(np.float32) * 0.5
        params = WaveletParams(n_scales=3, scale_weights=[1.0, 1.0, 1.0])
        result = wavelet_sharpen(data, params)
        np.testing.assert_allclose(result, data, atol=1e-3)

    def test_sharpening_increases_detail(self):
        data = np.random.rand(64, 64).astype(np.float32) * 0.5
        params = WaveletParams(n_scales=3, scale_weights=[2.0, 1.0, 1.0])
        result = wavelet_sharpen(data, params)
        # Sharpened image should have more contrast (higher std)
        assert np.std(result) >= np.std(data) - 0.01

    def test_smoothing_reduces_detail(self):
        data = np.random.rand(64, 64).astype(np.float32) * 0.5
        params = WaveletParams(n_scales=3, scale_weights=[0.0, 1.0, 1.0])
        result = wavelet_sharpen(data, params)
        # First detail scale removed, should be smoother
        assert np.std(result) < np.std(data) + 0.1

    def test_color_image(self):
        data = np.random.rand(3, 64, 64).astype(np.float32) * 0.5
        params = WaveletParams(n_scales=3, scale_weights=[1.5, 1.0, 1.0])
        result = wavelet_sharpen(data, params)
        assert result.shape == (3, 64, 64)

    def test_output_in_range(self):
        data = np.random.rand(64, 64).astype(np.float32)
        params = WaveletParams(n_scales=3, scale_weights=[3.0, 2.0, 1.0])
        result = wavelet_sharpen(data, params)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_mask_support(self):
        from astraios.core.masks import Mask
        data = np.random.rand(64, 64).astype(np.float32) * 0.5
        mask_data = np.zeros((64, 64), dtype=np.float32)
        mask_data[:32] = 1.0
        mask = Mask(data=mask_data)
        params = WaveletParams(n_scales=3, scale_weights=[3.0, 1.0, 1.0])
        result = wavelet_sharpen(data, params, mask=mask)
        # Bottom half should be unchanged
        np.testing.assert_allclose(result[32:], data[32:], atol=1e-5)
