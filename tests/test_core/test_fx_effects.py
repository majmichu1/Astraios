"""Tests for FX effects (Orton glow, soft focus, bloom, vignette, grain, split tone)."""

from __future__ import annotations

import numpy as np
import pytest

from astraios.core.device_manager import get_device_manager
from astraios.core.fx_effects import (
    BlendMode,
    FXEffect,
    FXParams,
    apply_fx,
)
from astraios.core.masks import Mask


def _mono_image(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = rng.rand(h, w).astype(np.float32) * 0.6 + 0.1
    img[h // 2 - 2 : h // 2 + 2, w // 2 - 2 : w // 2 + 2] = 0.98
    return img


def _color_image(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = rng.rand(3, h, w).astype(np.float32) * 0.6 + 0.1
    img[:, h // 2 - 2 : h // 2 + 2, w // 2 - 2 : w // 2 + 2] = 0.98
    return img


ALL_EFFECTS = list(FXEffect)


class TestFXParamsDefaults:
    def test_defaults_construct(self):
        p = FXParams()
        assert p.effect == FXEffect.ORTON_GLOW
        assert p.blend_mode == BlendMode.SCREEN
        assert p.grain_mono is True


@pytest.mark.parametrize("effect", ALL_EFFECTS)
class TestFXEffectsRunOnBothShapes:
    def test_mono_runs_and_is_finite_and_in_range(self, effect):
        img = _mono_image()
        params = FXParams(effect=effect)
        out = apply_fx(img, params)
        assert out.shape == img.shape
        assert out.dtype == np.float32
        assert np.isfinite(out).all()
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_color_runs_and_is_finite_and_in_range(self, effect):
        img = _color_image()
        params = FXParams(effect=effect)
        out = apply_fx(img, params)
        assert out.shape == img.shape
        assert out.dtype == np.float32
        assert np.isfinite(out).all()
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestNoOpParams:
    def test_orton_glow_zero_opacity_is_noop(self):
        img = _color_image()
        params = FXParams(effect=FXEffect.ORTON_GLOW, opacity=0.0)
        out = apply_fx(img, params)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_soft_focus_zero_opacity_is_noop(self):
        img = _mono_image()
        params = FXParams(effect=FXEffect.SOFT_FOCUS, opacity=0.0)
        out = apply_fx(img, params)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_bloom_zero_opacity_is_noop(self):
        img = _color_image()
        params = FXParams(effect=FXEffect.BLOOM, opacity=0.0)
        out = apply_fx(img, params)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_vignette_zero_amount_is_noop(self):
        img = _mono_image()
        params = FXParams(effect=FXEffect.VIGNETTE, vignette_amount=0.0)
        out = apply_fx(img, params)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_film_grain_zero_intensity_is_noop(self):
        img = _color_image()
        params = FXParams(effect=FXEffect.FILM_GRAIN, grain_intensity=0.0)
        out = apply_fx(img, params)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_split_tone_zero_strength_is_noop(self):
        img = _color_image()
        params = FXParams(effect=FXEffect.SPLIT_TONE, tone_strength=0.0)
        out = apply_fx(img, params)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_split_tone_mono_is_noop(self):
        """Split tone requires color; mono input should pass through unchanged."""
        img = _mono_image()
        params = FXParams(effect=FXEffect.SPLIT_TONE, tone_strength=0.8)
        out = apply_fx(img, params)
        np.testing.assert_allclose(out, img, atol=1e-6)


class TestOrtonGlow:
    def test_brightens_image_on_average(self):
        img = _color_image()
        out = apply_fx(img, FXParams(effect=FXEffect.ORTON_GLOW, opacity=0.8))
        assert out.mean() >= img.mean()

    def test_blend_modes_differ(self):
        img = _color_image()
        out_screen = apply_fx(
            img, FXParams(effect=FXEffect.ORTON_GLOW, opacity=0.8, blend_mode=BlendMode.SCREEN)
        )
        out_soft = apply_fx(
            img,
            FXParams(effect=FXEffect.ORTON_GLOW, opacity=0.8, blend_mode=BlendMode.SOFT_LIGHT),
        )
        assert not np.allclose(out_screen, out_soft)

    def test_highlight_protect_reduces_clipped_region_change(self):
        img = _color_image()
        img[:, :8, :8] = 1.0  # fully clipped patch
        out_no_protect = apply_fx(
            img, FXParams(effect=FXEffect.ORTON_GLOW, opacity=0.8, highlight_protect=0.0)
        )
        out_protect = apply_fx(
            img, FXParams(effect=FXEffect.ORTON_GLOW, opacity=0.8, highlight_protect=1.0)
        )
        diff_no_protect = np.abs(out_no_protect[:, :8, :8] - img[:, :8, :8]).mean()
        diff_protect = np.abs(out_protect[:, :8, :8] - img[:, :8, :8]).mean()
        assert diff_protect <= diff_no_protect + 1e-6


class TestBloom:
    def test_threshold_changes_output(self):
        img = _color_image()
        low = apply_fx(img, FXParams(effect=FXEffect.BLOOM, bloom_threshold=0.1, opacity=0.9))
        high = apply_fx(img, FXParams(effect=FXEffect.BLOOM, bloom_threshold=0.95, opacity=0.9))
        assert not np.allclose(low, high)


class TestVignette:
    def test_darkens_edges_more_than_center(self):
        img = np.full((3, 100, 100), 0.6, dtype=np.float32)
        out = apply_fx(img, FXParams(effect=FXEffect.VIGNETTE, vignette_amount=0.8))
        center = out[:, 50, 50].mean()
        corner = out[:, 0, 0].mean()
        assert corner < center


class TestFilmGrain:
    def test_deterministic_with_fixed_seed(self):
        img = _color_image()
        params = FXParams(effect=FXEffect.FILM_GRAIN, grain_intensity=0.5)
        out1 = apply_fx(img, params)
        out2 = apply_fx(img, params)
        np.testing.assert_array_equal(out1, out2)

    def test_mono_grain_flag_produces_equal_channel_deltas(self):
        # Mid-range, flat image so intensity*noise never clips at 0/1 — clipping
        # would make per-channel deltas diverge even for shared mono noise.
        img = np.full((3, 64, 64), 0.5, dtype=np.float32)
        params = FXParams(effect=FXEffect.FILM_GRAIN, grain_intensity=0.2, grain_mono=True)
        out = apply_fx(img, params)
        delta = out - img
        np.testing.assert_allclose(delta[0], delta[1], atol=1e-5)
        np.testing.assert_allclose(delta[1], delta[2], atol=1e-5)

    def test_color_grain_flag_produces_independent_channel_deltas(self):
        img = np.full((3, 64, 64), 0.5, dtype=np.float32)
        params = FXParams(effect=FXEffect.FILM_GRAIN, grain_intensity=0.2, grain_mono=False)
        out = apply_fx(img, params)
        delta = out - img
        assert not np.allclose(delta[0], delta[1], atol=1e-5)


class TestSplitTone:
    def test_hue_changes_output(self):
        img = _color_image()
        a = apply_fx(
            img, FXParams(effect=FXEffect.SPLIT_TONE, tone_strength=0.7, shadow_hue=0.0)
        )
        b = apply_fx(
            img, FXParams(effect=FXEffect.SPLIT_TONE, tone_strength=0.7, shadow_hue=180.0)
        )
        assert not np.allclose(a, b)


class TestMaskSupport:
    def test_mask_zero_region_unchanged(self):
        img = _color_image(h=64, w=64)
        mask_data = np.zeros((64, 64), dtype=np.float32)
        mask_data[32:, :] = 1.0
        mask = Mask(data=mask_data)
        out = apply_fx(
            img, FXParams(effect=FXEffect.ORTON_GLOW, opacity=0.9), mask=mask
        )
        np.testing.assert_allclose(out[:, :32, :], img[:, :32, :], atol=1e-5)

    def test_mask_none_equals_full_mask(self):
        img = _mono_image()
        params = FXParams(effect=FXEffect.BLOOM, opacity=0.7)
        out_none = apply_fx(img, params, mask=None)
        full_mask = Mask(data=np.ones_like(img))
        out_full = apply_fx(img, params, mask=full_mask)
        np.testing.assert_allclose(out_none, out_full, atol=1e-5)


class TestProgressCallback:
    def test_progress_called(self):
        calls = []

        def progress(frac, msg):
            calls.append((frac, msg))

        img = _mono_image()
        apply_fx(img, FXParams(effect=FXEffect.SOFT_FOCUS, opacity=0.5), progress=progress)
        assert len(calls) >= 2
        assert calls[-1][0] == pytest.approx(1.0)


@pytest.mark.skipif(not get_device_manager().is_gpu, reason="No GPU available")
class TestGPUCPUAgreement:
    """Blur-based effects (Orton Glow, Soft Focus, Bloom) have both GPU and CPU
    Gaussian blur paths; verify they agree within a small numerical tolerance."""

    @pytest.mark.parametrize(
        "effect",
        [FXEffect.ORTON_GLOW, FXEffect.SOFT_FOCUS, FXEffect.BLOOM],
    )
    def test_gpu_and_cpu_blur_agree(self, effect, monkeypatch):
        from astraios.core import fx_effects as mod

        img = _color_image(h=256, w=256)
        params = FXParams(effect=effect, opacity=0.8, blur_radius=12.0)

        monkeypatch.setattr(mod, "GPU_PIXEL_THRESHOLD", 1)
        out_gpu = apply_fx(img, params)

        monkeypatch.setattr(mod, "GPU_PIXEL_THRESHOLD", 10**12)
        out_cpu = apply_fx(img, params)

        np.testing.assert_allclose(out_gpu, out_cpu, atol=2e-2)
