"""Tests for texture & clarity local-contrast enhancement."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
import torch

from astraios.core.device_manager import Backend, get_device_manager
from astraios.core.masks import Mask
from astraios.core.texture_clarity import TextureClarityParams, apply_texture_clarity


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


def _textured_scene(size: int = 200, seed: int = 7) -> np.ndarray:
    """Smooth gradient with fine ripple detail baked in — good local-contrast target."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 0.3 + 0.4 * (xx / size)
    ripple = 0.03 * np.sin(xx / 2.0) * np.cos(yy / 2.0)
    noise = rng.normal(0, 0.005, size=(size, size)).astype(np.float32)
    return np.clip(base + ripple + noise, 0.0, 1.0).astype(np.float32)


def _laplacian_std(img: np.ndarray) -> float:
    lap = (
        -4 * img[1:-1, 1:-1]
        + img[:-2, 1:-1]
        + img[2:, 1:-1]
        + img[1:-1, :-2]
        + img[1:-1, 2:]
    )
    return float(np.std(lap))


class TestIdentity:
    def test_default_params_mono_identity(self):
        data = _textured_scene()
        result = apply_texture_clarity(data)
        np.testing.assert_allclose(result, data, atol=1e-6)

    def test_default_params_color_identity(self):
        data = np.stack([_textured_scene(seed=i) for i in range(3)], axis=0)
        result = apply_texture_clarity(data)
        np.testing.assert_allclose(result, data, atol=1e-5)

    def test_zero_amounts_identity_explicit(self):
        data = _textured_scene()
        params = TextureClarityParams(texture_amount=0.0, clarity_amount=0.0)
        result = apply_texture_clarity(data, params)
        np.testing.assert_allclose(result, data, atol=1e-6)


class TestShapes:
    def test_mono_shape_preserved(self):
        data = _textured_scene()
        result = apply_texture_clarity(data, TextureClarityParams(texture_amount=0.5))
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_color_shape_preserved(self):
        data = np.stack([_textured_scene(seed=i) for i in range(3)], axis=0)
        result = apply_texture_clarity(data, TextureClarityParams(clarity_amount=0.5))
        assert result.shape == data.shape

    def test_output_in_range(self):
        data = _textured_scene()
        params = TextureClarityParams(texture_amount=1.0, clarity_amount=1.0)
        result = apply_texture_clarity(data, params)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_alpha_channel_passthrough(self):
        rgb = np.stack([_textured_scene(seed=i) for i in range(3)], axis=0)
        alpha = np.full((1, 200, 200), 0.42, dtype=np.float32)
        data = np.concatenate([rgb, alpha], axis=0)
        result = apply_texture_clarity(data, TextureClarityParams(texture_amount=0.5))
        np.testing.assert_allclose(result[3], alpha[0], atol=1e-5)


class TestLocalContrastIncrease:
    def test_texture_increases_local_contrast_without_median_shift(self):
        data = _textured_scene()
        params = TextureClarityParams(texture_amount=1.0, texture_radius=1.0)
        result = apply_texture_clarity(data, params)

        assert _laplacian_std(result) > _laplacian_std(data) * 1.1
        assert abs(float(np.median(result)) - float(np.median(data))) < 0.02

    def test_clarity_increases_local_contrast_without_median_shift(self):
        data = _textured_scene()
        params = TextureClarityParams(clarity_amount=0.8, clarity_radius=3.0)
        result = apply_texture_clarity(data, params)

        assert _laplacian_std(result) > _laplacian_std(data) * 1.05
        assert abs(float(np.median(result)) - float(np.median(data))) < 0.02

    def test_negative_texture_smooths(self):
        data = _textured_scene()
        params = TextureClarityParams(texture_amount=-0.8, texture_radius=1.0)
        result = apply_texture_clarity(data, params)
        assert _laplacian_std(result) < _laplacian_std(data)


class TestMaskSupport:
    def test_mask_protects_region(self):
        data = _textured_scene()
        mask_data = np.zeros_like(data)
        mask_data[100:] = 1.0
        mask = Mask(data=mask_data)
        params = TextureClarityParams(texture_amount=0.9, clarity_amount=0.9)
        result = apply_texture_clarity(data, params, mask=mask)
        np.testing.assert_allclose(result[:100], data[:100], atol=1e-6)
        assert not np.allclose(result[100:], data[100:], atol=1e-6)


class TestClarityMidtoneMask:
    def test_mask_strength_zero_affects_shadows_and_highlights(self):
        size = 64
        data = np.zeros((size, size), dtype=np.float32)
        data[:, : size // 2] = 0.02  # near-black region
        data[:, size // 2 :] = 0.98  # near-white region
        # add a touch of local texture so clarity has something to act on
        data += 0.01 * np.sin(np.arange(size))[None, :].astype(np.float32)
        data = np.clip(data, 0.0, 1.0)

        midtone_only = apply_texture_clarity(
            data, TextureClarityParams(clarity_amount=0.9, clarity_radius=1.0, mask_strength=1.0)
        )
        everywhere = apply_texture_clarity(
            data, TextureClarityParams(clarity_amount=0.9, clarity_radius=1.0, mask_strength=0.0)
        )
        # With mask_strength=1 (classic clarity), shadow/highlight extremes barely move;
        # with mask_strength=0 they are free to move. The two results should differ.
        assert not np.allclose(midtone_only, everywhere, atol=1e-4)


class TestGPUCPUAgreement:
    def test_gpu_and_cpu_agree_texture(self):
        dm = get_device_manager()
        if not dm.is_gpu:
            pytest.skip("No GPU available in this environment")

        data = _textured_scene()
        params = TextureClarityParams(texture_amount=0.6, texture_radius=1.5)

        gpu_result = apply_texture_clarity(data.copy(), params)
        with _forced_cpu():
            cpu_result = apply_texture_clarity(data.copy(), params)

        np.testing.assert_allclose(gpu_result, cpu_result, atol=5e-3)

    def test_gpu_and_cpu_agree_clarity(self):
        dm = get_device_manager()
        if not dm.is_gpu:
            pytest.skip("No GPU available in this environment")

        data = _textured_scene()
        params = TextureClarityParams(clarity_amount=0.6, clarity_radius=2.0)

        gpu_result = apply_texture_clarity(data.copy(), params)
        with _forced_cpu():
            cpu_result = apply_texture_clarity(data.copy(), params)

        # Bilateral filter GPU/CPU implementations differ (tiled unfold vs cv2),
        # so allow a looser but still meaningful tolerance.
        np.testing.assert_allclose(gpu_result, cpu_result, atol=3e-2)

    def test_gpu_and_cpu_agree_large_radius_downsample_path(self):
        dm = get_device_manager()
        if not dm.is_gpu:
            pytest.skip("No GPU available in this environment")

        data = _textured_scene(size=256)
        params = TextureClarityParams(clarity_amount=0.5, clarity_radius=8.0)

        gpu_result = apply_texture_clarity(data.copy(), params)
        with _forced_cpu():
            cpu_result = apply_texture_clarity(data.copy(), params)

        np.testing.assert_allclose(gpu_result, cpu_result, atol=4e-2)
