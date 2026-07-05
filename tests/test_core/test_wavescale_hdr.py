"""Tests for WaveScale HDR (wavelet-based dynamic range compression)."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
import torch

from astraios.core.device_manager import Backend, get_device_manager
from astraios.core.masks import Mask
from astraios.core.wavescale_hdr import WaveScaleHDRParams, apply_wavescale_hdr


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


def _bright_core_with_ripple(size: int = 200) -> np.ndarray:
    """Synthetic HDR-ish scene: smooth bright core with faint fine ripple detail.

    The core brightness (~0.78 peak) sits well below saturation and the ripple
    amplitude is small relative to the core, mimicking a stretched image where
    fine structure in a bright region has been compressed to low contrast.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy, cx = size / 2, size / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    falloff = np.exp(-((r / (size * 0.27)) ** 2))
    bg = 0.12
    core = 0.65 * falloff
    ripple = 0.015 * np.sin(xx / 3.0) * np.sin(yy / 3.0) * falloff
    img = np.clip(bg + core + ripple, 0.0, 1.0).astype(np.float32)
    return img, r


class TestWaveScaleHDRShapesAndRanges:
    def test_mono_shape_preserved(self):
        data = np.random.rand(64, 64).astype(np.float32)
        result = apply_wavescale_hdr(data)
        assert result.shape == (64, 64)
        assert result.dtype == np.float32

    def test_color_shape_preserved(self):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        result = apply_wavescale_hdr(data)
        assert result.shape == (3, 64, 64)
        assert result.dtype == np.float32

    def test_output_finite_and_in_range_mono(self):
        data = np.random.rand(80, 80).astype(np.float32)
        params = WaveScaleHDRParams(compression_factor=3.0, mask_gamma=0.5)
        result = apply_wavescale_hdr(data, params)
        assert np.isfinite(result).all()
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_finite_and_in_range_color(self):
        data = np.random.rand(3, 80, 80).astype(np.float32)
        params = WaveScaleHDRParams(compression_factor=3.0, mask_gamma=0.5)
        result = apply_wavescale_hdr(data, params)
        assert np.isfinite(result).all()
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @pytest.mark.parametrize("n_scales", [2, 3, 5, 8, 10])
    def test_every_scale_count_runs_mono(self, n_scales):
        data = np.random.rand(96, 96).astype(np.float32)
        result = apply_wavescale_hdr(data, WaveScaleHDRParams(n_scales=n_scales))
        assert result.shape == data.shape
        assert np.isfinite(result).all()

    @pytest.mark.parametrize("n_scales", [2, 3, 5, 8, 10])
    def test_every_scale_count_runs_color(self, n_scales):
        data = np.random.rand(3, 96, 96).astype(np.float32)
        result = apply_wavescale_hdr(data, WaveScaleHDRParams(n_scales=n_scales))
        assert result.shape == data.shape
        assert np.isfinite(result).all()

    def test_dim_gamma_override_used(self):
        data = np.random.rand(64, 64).astype(np.float32) * 0.5 + 0.3
        auto = apply_wavescale_hdr(data, WaveScaleHDRParams(dim_gamma=None))
        forced = apply_wavescale_hdr(data, WaveScaleHDRParams(dim_gamma=1.0))
        assert not np.allclose(auto, forced, atol=1e-4)


class TestWaveScaleHDRIdentity:
    """A perfectly flat image has zero wavelet detail, so with dim_gamma
    pinned to 1.0 (bypassing the highlight-taming curve) the compression has
    nothing to act on and the output must equal the input.
    """

    def test_identity_on_flat_image_mono(self):
        flat = np.full((64, 64), 0.4, dtype=np.float32)
        params = WaveScaleHDRParams(dim_gamma=1.0)
        result = apply_wavescale_hdr(flat, params)
        np.testing.assert_allclose(result, flat, atol=5e-3)

    def test_identity_on_flat_image_color(self):
        flat = np.full((3, 64, 64), 0.4, dtype=np.float32)
        params = WaveScaleHDRParams(dim_gamma=1.0)
        result = apply_wavescale_hdr(flat, params)
        np.testing.assert_allclose(result, flat, atol=5e-3)


class TestWaveScaleHDRBrightCoreDetail:
    def test_core_local_contrast_increases(self):
        img, r = _bright_core_with_ripple()
        data = np.stack([img, img, img], axis=0)
        result = apply_wavescale_hdr(data, WaveScaleHDRParams())

        interior = r < (img.shape[0] * 0.11)  # well inside the core, away from any edge effects
        std_before = img[interior].std()
        std_after = result[0][interior].std()
        assert std_after > std_before

        gy0, gx0 = np.gradient(img)
        grad_before = np.sqrt(gy0**2 + gx0**2)[interior].mean()
        gy1, gx1 = np.gradient(result[0])
        grad_after = np.sqrt(gy1**2 + gx1**2)[interior].mean()
        assert grad_after > grad_before

        # The bright core should not have been driven entirely into flat
        # clipping — that would defeat the point of "revealing" detail. A
        # few pixels right at the profile's peak may still saturate.
        clipped_fraction = np.mean(result[0][interior] >= 1.0 - 1e-6)
        assert clipped_fraction < 0.1


class TestWaveScaleHDRMask:
    def test_mask_protects_region(self):
        data = np.random.rand(3, 64, 64).astype(np.float32) * 0.6 + 0.2
        mask_data = np.zeros((64, 64), dtype=np.float32)
        mask_data[32:] = 1.0
        mask = Mask(data=mask_data)
        result = apply_wavescale_hdr(data, WaveScaleHDRParams(compression_factor=3.0), mask=mask)
        np.testing.assert_allclose(result[:, :32], data[:, :32], atol=1e-6)


class TestWaveScaleHDRGPUCPUAgreement:
    def test_gpu_and_cpu_agree(self):
        dm = get_device_manager()
        if not dm.is_gpu:
            pytest.skip("No GPU available in this environment")

        rng = np.random.default_rng(42)
        data = rng.random((3, 256, 256)).astype(np.float32)
        params = WaveScaleHDRParams()

        gpu_result = apply_wavescale_hdr(data.copy(), params)
        with _forced_cpu():
            cpu_result = apply_wavescale_hdr(data.copy(), params)

        np.testing.assert_allclose(gpu_result, cpu_result, atol=2e-3)
