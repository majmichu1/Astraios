"""Tests for WaveScale Dark Enhance (multiscale faint-structure enhancement)."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
import torch

from astraios.core.device_manager import Backend, get_device_manager
from astraios.core.masks import Mask
from astraios.core.wavescale_dark_enhance import (
    WaveScaleDarkEnhanceParams,
    apply_dark_enhance,
)


@contextlib.contextmanager
def _forced_cpu():
    """Temporarily force the device manager singleton onto the CPU backend."""
    dm = get_device_manager()
    orig_device, orig_backend = dm._device, dm._backend
    dm._device = torch.device("cpu")
    dm._backend = Backend.CPU
    try:
        yield
    finally:
        dm._device = orig_device
        dm._backend = orig_backend


def _faint_dust_lane(size: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic faint dark band (dust-lane-like) on a flat mid-tone background."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy, cx = size / 2, size / 2
    band = np.abs(xx - cx) < size * 0.2
    dip = -0.03 * np.exp(-(((xx - cx) / (size * 0.12)) ** 2))
    img = np.clip(0.35 + dip, 0.0, 1.0).astype(np.float32)
    interior = band & (np.abs(yy - cy) < size * 0.2)
    return img, interior


class TestWaveScaleDarkEnhanceShapesAndRanges:
    def test_mono_shape_preserved(self):
        data = np.random.rand(64, 64).astype(np.float32)
        result = apply_dark_enhance(data)
        assert result.shape == (64, 64)
        assert result.dtype == np.float32

    def test_color_shape_preserved(self):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        result = apply_dark_enhance(data)
        assert result.shape == (3, 64, 64)
        assert result.dtype == np.float32

    def test_output_finite_and_in_range_mono(self):
        data = np.random.rand(80, 80).astype(np.float32)
        params = WaveScaleDarkEnhanceParams(boost_factor=8.0, iterations=3)
        result = apply_dark_enhance(data, params)
        assert np.isfinite(result).all()
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_finite_and_in_range_color(self):
        data = np.random.rand(3, 80, 80).astype(np.float32)
        params = WaveScaleDarkEnhanceParams(boost_factor=8.0, iterations=3)
        result = apply_dark_enhance(data, params)
        assert np.isfinite(result).all()
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @pytest.mark.parametrize("n_scales", [2, 3, 6, 8, 10])
    def test_every_scale_count_runs_mono(self, n_scales):
        data = np.random.rand(96, 96).astype(np.float32)
        result = apply_dark_enhance(data, WaveScaleDarkEnhanceParams(n_scales=n_scales))
        assert result.shape == data.shape
        assert np.isfinite(result).all()

    @pytest.mark.parametrize("n_scales", [2, 3, 6, 8, 10])
    def test_every_scale_count_runs_color(self, n_scales):
        data = np.random.rand(3, 96, 96).astype(np.float32)
        result = apply_dark_enhance(data, WaveScaleDarkEnhanceParams(n_scales=n_scales))
        assert result.shape == data.shape
        assert np.isfinite(result).all()

    @pytest.mark.parametrize("iterations", [1, 2, 5, 10])
    def test_every_iteration_count_runs(self, iterations):
        data = np.random.rand(64, 64).astype(np.float32)
        result = apply_dark_enhance(data, WaveScaleDarkEnhanceParams(iterations=iterations))
        assert result.shape == data.shape
        assert np.isfinite(result).all()


class TestWaveScaleDarkEnhanceIdentity:
    """boost_factor=1.0 zeroes the enhancement term unconditionally, so the
    output must equal the input regardless of image content or iterations.
    """

    def test_identity_mono(self):
        data = np.random.rand(64, 64).astype(np.float32)
        params = WaveScaleDarkEnhanceParams(boost_factor=1.0, iterations=3)
        result = apply_dark_enhance(data, params)
        np.testing.assert_allclose(result, data, atol=1e-4)

    def test_identity_color(self):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        params = WaveScaleDarkEnhanceParams(boost_factor=1.0, iterations=1)
        result = apply_dark_enhance(data, params)
        np.testing.assert_allclose(result, data, atol=1e-2)


class TestWaveScaleDarkEnhanceFaintStructure:
    def test_dip_deepens_mono(self):
        img, interior = _faint_dust_lane()
        result = apply_dark_enhance(img, WaveScaleDarkEnhanceParams())

        depth_before = img[interior].max() - img[interior].min()
        depth_after = result[interior].max() - result[interior].min()
        assert depth_after > depth_before

        std_before = img[interior].std()
        std_after = result[interior].std()
        assert std_after > std_before

    def test_dip_deepens_color(self):
        img, interior = _faint_dust_lane()
        data = np.stack([img, img, img], axis=0)
        result = apply_dark_enhance(data, WaveScaleDarkEnhanceParams())

        depth_before = img[interior].max() - img[interior].min()
        depth_after = result[0][interior].max() - result[0][interior].min()
        assert depth_after > depth_before


class TestWaveScaleDarkEnhanceMask:
    def test_mask_protects_region(self):
        data = np.random.rand(3, 64, 64).astype(np.float32) * 0.4 + 0.1
        mask_data = np.zeros((64, 64), dtype=np.float32)
        mask_data[32:] = 1.0
        mask = Mask(data=mask_data)
        result = apply_dark_enhance(
            data, WaveScaleDarkEnhanceParams(boost_factor=8.0), mask=mask
        )
        np.testing.assert_allclose(result[:, :32], data[:, :32], atol=1e-6)


class TestWaveScaleDarkEnhanceGPUCPUAgreement:
    def test_gpu_and_cpu_agree(self):
        dm = get_device_manager()
        if not dm.is_gpu:
            pytest.skip("No GPU available in this environment")

        rng = np.random.default_rng(7)
        data = rng.random((3, 256, 256)).astype(np.float32)
        params = WaveScaleDarkEnhanceParams()

        gpu_result = apply_dark_enhance(data.copy(), params)
        with _forced_cpu():
            cpu_result = apply_dark_enhance(data.copy(), params)

        np.testing.assert_allclose(gpu_result, cpu_result, atol=2e-3)
