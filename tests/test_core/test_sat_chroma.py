"""Tests for the Saturation / Chroma hue-curve tool."""

import numpy as np
import pytest

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask
from astraios.core.sat_chroma import (
    SatChromaMode,
    SatChromaParams,
    _apply_chroma_lab_cpu,
    _apply_chroma_lab_gpu,
    _apply_saturation_hsv_cpu,
    _apply_saturation_hsv_gpu,
    _build_lut,
    apply_sat_chroma,
)


def _hue_patch_image():
    """(3, 120, 60) image with three saturated color blocks: red, green, blue."""
    img = np.zeros((3, 120, 60), dtype=np.float32)
    # Red patch (hue ~0)
    img[0, 0:40, :] = 0.8
    img[1, 0:40, :] = 0.2
    img[2, 0:40, :] = 0.2
    # Green patch (hue ~120)
    img[0, 40:80, :] = 0.2
    img[1, 40:80, :] = 0.8
    img[2, 40:80, :] = 0.2
    # Blue patch (hue ~240)
    img[0, 80:120, :] = 0.2
    img[1, 80:120, :] = 0.2
    img[2, 80:120, :] = 0.8
    return img


def _boost_red_params(mode=SatChromaMode.SATURATION_HSV, boost=3.0):
    return SatChromaParams(
        mode=mode,
        curve_points=[
            (0.0, boost),
            (60.0, 1.0),
            (120.0, 1.0),
            (180.0, 1.0),
            (240.0, 1.0),
            (300.0, 1.0),
            (360.0, boost),
        ],
        strength=1.0,
    )


class TestIdentity:
    def test_flat_curve_is_near_identity(self):
        img = _hue_patch_image()
        result = apply_sat_chroma(img, SatChromaParams())
        np.testing.assert_allclose(result, img, atol=1e-3)

    def test_zero_strength_desaturates_toward_gray_not_error(self):
        img = _hue_patch_image()
        params = SatChromaParams(strength=0.0)
        result = apply_sat_chroma(img, params)
        assert result.shape == img.shape
        assert np.isfinite(result).all()


class TestHueSelectivity:
    def test_hsv_boost_only_affects_target_hue(self):
        img = _hue_patch_image()
        params = _boost_red_params(SatChromaMode.SATURATION_HSV)
        result = apply_sat_chroma(img, params)

        red_diff = np.abs(result[:, 0:40, :] - img[:, 0:40, :]).mean()
        green_diff = np.abs(result[:, 40:80, :] - img[:, 40:80, :]).mean()
        blue_diff = np.abs(result[:, 80:120, :] - img[:, 80:120, :]).mean()

        assert red_diff > 0.03
        assert green_diff < 0.01
        assert blue_diff < 0.01

    def test_chroma_lab_boost_only_affects_target_hue(self):
        img = _hue_patch_image()
        params = _boost_red_params(SatChromaMode.CHROMA_LAB)
        result = apply_sat_chroma(img, params)

        red_diff = np.abs(result[:, 0:40, :] - img[:, 0:40, :]).mean()
        green_diff = np.abs(result[:, 40:80, :] - img[:, 40:80, :]).mean()
        blue_diff = np.abs(result[:, 80:120, :] - img[:, 80:120, :]).mean()

        assert red_diff > 0.02
        assert green_diff < 0.01
        assert blue_diff < 0.01


class TestChromaDenoise:
    def test_low_strength_reduces_chroma_noise(self):
        rng = np.random.default_rng(42)
        h, w = 80, 80
        base = np.full((3, h, w), 0.5, dtype=np.float32)
        noise = rng.normal(0.0, 0.08, size=(3, h, w)).astype(np.float32)
        noisy = np.clip(base + noise, 0.0, 1.0)

        def chroma_energy(a):
            lum = a.mean(axis=0, keepdims=True)
            return np.std(a - lum)

        before = chroma_energy(noisy)
        params = SatChromaParams(mode=SatChromaMode.CHROMA_LAB, strength=0.2)
        result = apply_sat_chroma(noisy, params)
        after = chroma_energy(result)

        assert after < before * 0.6


class TestMonoHandling:
    def test_mono_is_noop(self):
        mono = np.random.default_rng(0).random((50, 50)).astype(np.float32)
        result = apply_sat_chroma(mono, SatChromaParams())
        np.testing.assert_array_equal(result, mono)


class TestMaskBlend:
    def test_mask_protects_original(self):
        img = _hue_patch_image()
        mask_data = np.zeros(img.shape[1:], dtype=np.float32)
        mask_data[0:40, :] = 1.0  # only red patch is editable
        mask = Mask(data=mask_data)
        params = _boost_red_params(SatChromaMode.SATURATION_HSV)
        result = apply_sat_chroma(img, params, mask=mask)

        # Outside the mask (green/blue patches) must be untouched exactly.
        np.testing.assert_array_equal(result[:, 40:, :], img[:, 40:, :])


class TestGpuCpuAgreement:
    def test_hsv_gpu_matches_cpu(self):
        dm = get_device_manager()
        if not dm.is_gpu:
            pytest.skip("no GPU available")
        img = _hue_patch_image()
        params = _boost_red_params(SatChromaMode.SATURATION_HSV)
        lut = _build_lut(params)
        cpu_out = _apply_saturation_hsv_cpu(img, lut)
        gpu_out = _apply_saturation_hsv_gpu(img, lut, dm)
        assert np.abs(cpu_out - gpu_out).mean() < 0.03

    def test_lab_gpu_matches_cpu(self):
        dm = get_device_manager()
        if not dm.is_gpu:
            pytest.skip("no GPU available")
        img = _hue_patch_image()
        params = _boost_red_params(SatChromaMode.CHROMA_LAB)
        lut = _build_lut(params)
        cpu_out = _apply_chroma_lab_cpu(img, lut)
        gpu_out = _apply_chroma_lab_gpu(img, lut, dm)
        assert np.abs(cpu_out - gpu_out).mean() < 0.03
