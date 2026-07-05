"""Tests for selective color / selective luminance adjustments."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
import torch

from astraios.core.device_manager import Backend, get_device_manager
from astraios.core.masks import Mask
from astraios.core.selective_adjust import (
    SelectiveColorParams,
    SelectiveLumaParams,
    apply_selective_color,
    apply_selective_luma,
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


def _hsv_to_rgb_patch(h_deg: float, s: float, v: float) -> tuple[float, float, float]:
    """Minimal standalone HSV->RGB for building synthetic test patches."""
    c = v * s
    hp = (h_deg / 60.0) % 6.0
    x = c * (1 - abs(hp % 2 - 1))
    m = v - c
    table = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)]
    r, g, b = table[int(hp) % 6]
    return r + m, g + m, b + m


def _multi_hue_image(size: int = 20) -> tuple[np.ndarray, dict[str, tuple[slice, slice]]]:
    """A (3, 4*size, size) image with four solid-color quadrant patches:
    red, green, blue, yellow (top to bottom), each fully saturated.
    """
    hues = {"red": 0.0, "green": 120.0, "blue": 240.0, "yellow": 60.0}
    img = np.zeros((3, size * 4, size), dtype=np.float32)
    regions: dict[str, tuple[slice, slice]] = {}
    for i, (name, h) in enumerate(hues.items()):
        r, g, b = _hsv_to_rgb_patch(h, 1.0, 0.8)
        rows = slice(i * size, (i + 1) * size)
        cols = slice(0, size)
        img[0, rows, cols] = r
        img[1, rows, cols] = g
        img[2, rows, cols] = b
        regions[name] = (rows, cols)
    return img.astype(np.float32), regions


def _lum_stripe_image(size: int = 20) -> tuple[np.ndarray, dict[str, tuple[slice, slice]]]:
    """A (3, 4*size, size) grayscale-stripe image at luminance 0.1/0.4/0.6/0.9."""
    levels = {"dark": 0.1, "low_mid": 0.4, "high_mid": 0.6, "bright": 0.9}
    img = np.zeros((3, size * 4, size), dtype=np.float32)
    regions: dict[str, tuple[slice, slice]] = {}
    for i, (name, v) in enumerate(levels.items()):
        rows = slice(i * size, (i + 1) * size)
        cols = slice(0, size)
        img[:, rows, cols] = v
        regions[name] = (rows, cols)
    return img.astype(np.float32), regions


# ---------------------------------------------------------------------
# Selective Color
# ---------------------------------------------------------------------


class TestSelectiveColorIdentity:
    def test_default_params_mono_identity(self):
        data = np.random.default_rng(0).random((40, 40)).astype(np.float32)
        result = apply_selective_color(data)
        np.testing.assert_allclose(result, data, atol=1e-5)

    def test_default_params_color_identity(self):
        data, _ = _multi_hue_image()
        result = apply_selective_color(data)
        np.testing.assert_allclose(result, data, atol=1e-5)

    def test_zero_adjustments_identity_any_hue_range(self):
        data, _ = _multi_hue_image()
        params = SelectiveColorParams(hue_ranges=[(0.0, 360.0)], min_chroma=0.0)
        result = apply_selective_color(data, params)
        np.testing.assert_allclose(result, data, atol=1e-5)


class TestSelectiveColorShapes:
    def test_color_shape_preserved(self):
        data, _ = _multi_hue_image()
        result = apply_selective_color(data, SelectiveColorParams(red=0.2))
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_mono_shape_preserved(self):
        data = np.random.default_rng(1).random((40, 40)).astype(np.float32)
        result = apply_selective_color(data, SelectiveColorParams(luminance=0.2))
        assert result.shape == data.shape

    def test_alpha_channel_passthrough(self):
        rgb, _ = _multi_hue_image()
        alpha = np.full((1, *rgb.shape[1:]), 0.5, dtype=np.float32)
        data = np.concatenate([rgb, alpha], axis=0)
        result = apply_selective_color(data, SelectiveColorParams(red=0.3))
        np.testing.assert_allclose(result[3], alpha[0], atol=1e-5)

    def test_output_in_range(self):
        data, _ = _multi_hue_image()
        params = SelectiveColorParams(red=1.0, luminance=1.0, contrast=1.0)
        result = apply_selective_color(data, params)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestSelectiveColorTargeting:
    def test_only_targeted_hue_patch_changes(self):
        data, regions = _multi_hue_image()
        # Target Red only, tight band, no feather, gate out low chroma.
        params = SelectiveColorParams(
            hue_ranges=[(340.0, 360.0), (0.0, 15.0)],
            smooth_deg=2.0,
            min_chroma=0.2,
            red=-0.3,
            green=0.3,
        )
        result = apply_selective_color(data, params)

        red_rows, red_cols = regions["red"]
        assert not np.allclose(
            result[:, red_rows, red_cols], data[:, red_rows, red_cols], atol=1e-4
        )

        for name in ("green", "blue", "yellow"):
            rows, cols = regions[name]
            np.testing.assert_allclose(result[:, rows, cols], data[:, rows, cols], atol=1e-4)

    def test_invert_range_flips_selection(self):
        data, regions = _multi_hue_image()
        params = SelectiveColorParams(
            hue_ranges=[(340.0, 360.0), (0.0, 15.0)],
            smooth_deg=2.0,
            min_chroma=0.2,
            invert_range=True,
            red=0.3,
        )
        result = apply_selective_color(data, params)

        red_rows, red_cols = regions["red"]
        np.testing.assert_allclose(
            result[:, red_rows, red_cols], data[:, red_rows, red_cols], atol=1e-4
        )
        # At least one other hue patch should now be affected.
        changed = any(
            not np.allclose(result[:, r, c], data[:, r, c], atol=1e-4)
            for name, (r, c) in regions.items()
            if name != "red"
        )
        assert changed


class TestSelectiveColorMaskSupport:
    def test_mask_protects_region(self):
        data, regions = _multi_hue_image()
        mask_data = np.zeros(data.shape[1:], dtype=np.float32)
        red_rows, red_cols = regions["red"]
        # Protect the red patch itself via the external mask, even though the
        # hue selection targets it.
        mask = Mask(data=mask_data)
        params = SelectiveColorParams(
            hue_ranges=[(340.0, 360.0), (0.0, 15.0)], smooth_deg=2.0, min_chroma=0.2, red=0.5
        )
        result = apply_selective_color(data, params, mask=mask)
        np.testing.assert_allclose(result, data, atol=1e-5)


class TestSelectiveColorGPUCPUAgreement:
    def test_gpu_and_cpu_agree(self):
        dm = get_device_manager()
        if not dm.is_gpu:
            pytest.skip("No GPU available in this environment")

        data, _ = _multi_hue_image(size=48)
        params = SelectiveColorParams(
            hue_ranges=[(200.0, 270.0)],
            smooth_deg=15.0,
            min_chroma=0.05,
            edge_blur=3.0,
            cyan=0.2,
            red=-0.1,
            luminance=0.1,
            contrast=0.2,
            chroma=0.2,
        )

        gpu_result = apply_selective_color(data.copy(), params)
        with _forced_cpu():
            cpu_result = apply_selective_color(data.copy(), params)

        np.testing.assert_allclose(gpu_result, cpu_result, atol=5e-3)

    def test_gpu_and_cpu_agree_saturation_mode(self):
        dm = get_device_manager()
        if not dm.is_gpu:
            pytest.skip("No GPU available in this environment")

        data, _ = _multi_hue_image(size=48)
        params = SelectiveColorParams(
            hue_ranges=[(40.0, 70.0)],
            use_chroma_mode=False,
            saturation=0.4,
        )

        gpu_result = apply_selective_color(data.copy(), params)
        with _forced_cpu():
            cpu_result = apply_selective_color(data.copy(), params)

        np.testing.assert_allclose(gpu_result, cpu_result, atol=5e-3)


# ---------------------------------------------------------------------
# Selective Luminance
# ---------------------------------------------------------------------


class TestSelectiveLumaIdentity:
    def test_default_params_mono_identity(self):
        data = np.random.default_rng(2).random((40, 40)).astype(np.float32)
        result = apply_selective_luma(data)
        np.testing.assert_allclose(result, data, atol=1e-5)

    def test_default_params_color_identity(self):
        data, _ = _lum_stripe_image()
        result = apply_selective_luma(data)
        np.testing.assert_allclose(result, data, atol=1e-5)

    def test_zero_adjustments_identity_full_band(self):
        data, _ = _lum_stripe_image()
        params = SelectiveLumaParams(lo=0.0, hi=1.0)
        result = apply_selective_luma(data, params)
        np.testing.assert_allclose(result, data, atol=1e-5)


class TestSelectiveLumaShapes:
    def test_color_shape_preserved(self):
        data, _ = _lum_stripe_image()
        result = apply_selective_luma(data, SelectiveLumaParams(luminance=0.1))
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_mono_shape_preserved(self):
        data = np.random.default_rng(3).random((40, 40)).astype(np.float32)
        result = apply_selective_luma(data, SelectiveLumaParams(luminance=0.1))
        assert result.shape == data.shape

    def test_output_in_range(self):
        data, _ = _lum_stripe_image()
        params = SelectiveLumaParams(lo=0.3, hi=0.7, luminance=1.0, contrast=1.0)
        result = apply_selective_luma(data, params)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestSelectiveLumaTargeting:
    def test_only_targeted_band_changes(self):
        data, regions = _lum_stripe_image()
        params = SelectiveLumaParams(lo=0.35, hi=0.65, smooth=0.02, edge_blur=0.0, luminance=0.2)
        result = apply_selective_luma(data, params)

        for name in ("low_mid", "high_mid"):
            rows, cols = regions[name]
            assert not np.allclose(result[:, rows, cols], data[:, rows, cols], atol=1e-4)

        for name in ("dark", "bright"):
            rows, cols = regions[name]
            np.testing.assert_allclose(result[:, rows, cols], data[:, rows, cols], atol=1e-4)

    def test_invert_flips_selection(self):
        data, regions = _lum_stripe_image()
        params = SelectiveLumaParams(
            lo=0.35, hi=0.65, smooth=0.02, edge_blur=0.0, invert=True, luminance=0.2
        )
        result = apply_selective_luma(data, params)

        for name in ("low_mid", "high_mid"):
            rows, cols = regions[name]
            np.testing.assert_allclose(result[:, rows, cols], data[:, rows, cols], atol=1e-4)
        for name in ("dark", "bright"):
            rows, cols = regions[name]
            assert not np.allclose(result[:, rows, cols], data[:, rows, cols], atol=1e-4)


class TestSelectiveLumaMaskSupport:
    def test_mask_protects_region(self):
        data, regions = _lum_stripe_image()
        mask = Mask(data=np.zeros(data.shape[1:], dtype=np.float32))
        params = SelectiveLumaParams(lo=0.35, hi=0.65, smooth=0.02, edge_blur=0.0, luminance=0.5)
        result = apply_selective_luma(data, params, mask=mask)
        np.testing.assert_allclose(result, data, atol=1e-5)


class TestSelectiveLumaGPUCPUAgreement:
    def test_gpu_and_cpu_agree(self):
        dm = get_device_manager()
        if not dm.is_gpu:
            pytest.skip("No GPU available in this environment")

        data, _ = _lum_stripe_image(size=48)
        params = SelectiveLumaParams(
            lo=0.2, hi=0.6, smooth=0.05, edge_blur=4.0, luminance=0.1, contrast=0.3, chroma=0.2
        )

        gpu_result = apply_selective_luma(data.copy(), params)
        with _forced_cpu():
            cpu_result = apply_selective_luma(data.copy(), params)

        np.testing.assert_allclose(gpu_result, cpu_result, atol=5e-3)

    def test_gpu_and_cpu_agree_negative_contrast(self):
        dm = get_device_manager()
        if not dm.is_gpu:
            pytest.skip("No GPU available in this environment")

        data, _ = _lum_stripe_image(size=48)
        params = SelectiveLumaParams(lo=0.0, hi=1.0, contrast=-0.4)

        gpu_result = apply_selective_luma(data.copy(), params)
        with _forced_cpu():
            cpu_result = apply_selective_luma(data.copy(), params)

        np.testing.assert_allclose(gpu_result, cpu_result, atol=5e-3)
