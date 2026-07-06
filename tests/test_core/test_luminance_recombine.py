"""Tests for luminance recombine (LRGB/narrowband finishing step)."""

from __future__ import annotations

import numpy as np
import pytest

from astraios.core.luminance_recombine import (
    _LUMA_REC709,
    LUMA_PROFILES,
    LuminanceRecombineParams,
    compute_luminance,
    recombine_luminance,
    recombine_luminance_linear_scale,
    resolve_luma_profile_weights,
)
from astraios.core.masks import Mask


def _make_rgb(seed=0, size=32):
    rng = np.random.default_rng(seed)
    return (rng.random((3, size, size)).astype(np.float32) * 0.6 + 0.2)


class TestComputeLuminance:
    def test_mono_passthrough(self):
        mono = np.random.default_rng(0).random((16, 16)).astype(np.float32)
        out = compute_luminance(mono)
        np.testing.assert_allclose(out, mono)

    def test_rec709_matches_manual_weighting(self):
        rgb = _make_rgb()
        out = compute_luminance(rgb, method="rec709")
        expected = (
            _LUMA_REC709[0] * rgb[0] + _LUMA_REC709[1] * rgb[1] + _LUMA_REC709[2] * rgb[2]
        )
        np.testing.assert_allclose(out, np.clip(expected, 0, 1), atol=1e-6)

    def test_equal_method_is_mean(self):
        rgb = _make_rgb()
        out = compute_luminance(rgb, method="equal")
        np.testing.assert_allclose(out, rgb.mean(axis=0), atol=1e-6)

    def test_max_method(self):
        rgb = _make_rgb()
        out = compute_luminance(rgb, method="max")
        np.testing.assert_allclose(out, rgb.max(axis=0), atol=1e-6)

    def test_median_method(self):
        rgb = _make_rgb()
        out = compute_luminance(rgb, method="median")
        np.testing.assert_allclose(out, np.median(rgb, axis=0), atol=1e-6)

    def test_explicit_weights_override_method(self):
        rgb = _make_rgb()
        w = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        out = compute_luminance(rgb, method="rec709", weights=w)
        np.testing.assert_allclose(out, rgb[0], atol=1e-6)

    def test_snr_requires_noise_sigma(self):
        rgb = _make_rgb()
        with pytest.raises(ValueError):
            compute_luminance(rgb, method="snr")

    def test_snr_with_noise_sigma(self):
        rgb = _make_rgb()
        sigma = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        out = compute_luminance(rgb, method="snr", noise_sigma=sigma)
        # Equal sigma -> equal weights -> equivalent to "equal" method.
        np.testing.assert_allclose(out, rgb.mean(axis=0), atol=1e-5)


class TestResolveLumaProfile:
    def test_default_is_rec709(self):
        method, w, name = resolve_luma_profile_weights(None)
        assert method == "rec709"
        np.testing.assert_allclose(w, _LUMA_REC709)
        assert name is None

    def test_alias_resolves(self):
        method, _w, _name = resolve_luma_profile_weights("rec.709")
        assert method == "rec709"

    def test_unknown_key_falls_back_to_rec709(self):
        method, w, name = resolve_luma_profile_weights("totally-unknown-key")
        assert method == "rec709"
        np.testing.assert_allclose(w, _LUMA_REC709)
        assert name is None

    def test_sensor_profile_returns_custom_weights_and_name(self):
        key = "sensor:Sony IMX571 (ASI2600/QHY268)"
        method, w, name = resolve_luma_profile_weights(key)
        assert method == "rec709"
        assert w is not None
        assert w.shape == (3,)
        assert name == "Sony IMX571 (ASI2600/QHY268)"

    def test_all_profiles_have_valid_weights_or_none(self):
        for key, prof in LUMA_PROFILES.items():
            w = prof["weights"]
            if w is not None:
                assert w.shape == (3,), key


class TestRecombineLinearScale:
    def test_identity_when_new_l_equals_own_luminance(self):
        rgb = _make_rgb()
        y = compute_luminance(rgb, method="rec709")
        out = recombine_luminance_linear_scale(rgb, y, pedestal=0.0, blend=1.0)
        np.testing.assert_allclose(out, rgb, atol=1e-4)

    def test_recombined_luminance_matches_new_l(self):
        rgb = _make_rgb()
        new_l = np.full((32, 32), 0.5, dtype=np.float32)
        out = recombine_luminance_linear_scale(rgb, new_l, pedestal=0.0, blend=1.0)
        measured = compute_luminance(out, method="rec709")
        # Away from clipping, the recombined luminance should closely match new_L.
        interior = (out.min(axis=0) > 0.01) & (out.max(axis=0) < 0.99)
        np.testing.assert_allclose(measured[interior], new_l[interior], atol=1e-3)

    def test_chroma_direction_preserved_without_pedestal(self):
        rgb = _make_rgb()
        new_l = np.full((32, 32), 0.4, dtype=np.float32)
        out = recombine_luminance_linear_scale(rgb, new_l, pedestal=0.0, blend=1.0)
        # Per-pixel channel ratios should be unchanged (pure scaling), where non-degenerate.
        eps = 1e-4
        ratio_before = rgb[0] / (rgb[1] + eps)
        ratio_after = out[0] / (out[1] + eps)
        interior = (out.max(axis=0) < 0.99) & (out.min(axis=0) > 1e-3)
        np.testing.assert_allclose(ratio_after[interior], ratio_before[interior], rtol=1e-2)

    def test_blend_zero_returns_original(self):
        rgb = _make_rgb()
        new_l = np.full((32, 32), 0.9, dtype=np.float32)
        out = recombine_luminance_linear_scale(rgb, new_l, blend=0.0)
        np.testing.assert_allclose(out, rgb, atol=1e-5)

    def test_pedestal_clamped_to_sane_range(self):
        rgb = _make_rgb()
        new_l = np.full((32, 32), 0.5, dtype=np.float32)
        out_a = recombine_luminance_linear_scale(rgb, new_l, pedestal=10.0)
        out_b = recombine_luminance_linear_scale(rgb, new_l, pedestal=0.5)
        np.testing.assert_allclose(out_a, out_b, atol=1e-5)

    def test_resizes_mismatched_luminance(self):
        rgb = _make_rgb(size=32)
        small_l = np.full((16, 16), 0.6, dtype=np.float32)
        out = recombine_luminance_linear_scale(rgb, small_l)
        assert out.shape == rgb.shape

    def test_rejects_mono_target(self):
        mono = np.random.default_rng(0).random((16, 16)).astype(np.float32)
        with pytest.raises(ValueError):
            recombine_luminance_linear_scale(mono, np.full((16, 16), 0.5, dtype=np.float32))

    def test_gpu_and_cpu_paths_agree(self):
        rgb = _make_rgb(size=48)
        new_l = np.random.default_rng(1).random((48, 48)).astype(np.float32)
        out_cpu = recombine_luminance_linear_scale(rgb, new_l, use_gpu=False)
        out_gpu = recombine_luminance_linear_scale(rgb, new_l, use_gpu=True)
        np.testing.assert_allclose(out_cpu, out_gpu, rtol=1e-3, atol=1e-4)


class TestRecombineLuminance:
    def test_mono_luma_used_directly(self):
        rgb = _make_rgb()
        new_l = np.full((32, 32), 0.5, dtype=np.float32)
        out = recombine_luminance(rgb, new_l, LuminanceRecombineParams(pedestal=0.0))
        measured = compute_luminance(out, method="rec709")
        interior = (out.min(axis=0) > 0.01) & (out.max(axis=0) < 0.99)
        np.testing.assert_allclose(measured[interior], new_l[interior], atol=1e-3)

    def test_rgb_luma_source_derives_luminance(self):
        rgb = _make_rgb()
        luma_source_rgb = _make_rgb(seed=99)
        out = recombine_luminance(
            rgb, luma_source_rgb, LuminanceRecombineParams(luma_method="rec709", pedestal=0.0)
        )
        expected_l = compute_luminance(luma_source_rgb, method="rec709")
        measured = compute_luminance(out, method="rec709")
        interior = (out.min(axis=0) > 0.01) & (out.max(axis=0) < 0.99)
        np.testing.assert_allclose(measured[interior], expected_l[interior], atol=1e-3)

    def test_rejects_non_rgb_target(self):
        mono = np.random.default_rng(0).random((16, 16)).astype(np.float32)
        with pytest.raises(ValueError):
            recombine_luminance(mono, np.full((16, 16), 0.5, dtype=np.float32))

    def test_explicit_luma_weights_override_profile(self):
        rgb = _make_rgb()
        new_l = np.full((32, 32), 0.5, dtype=np.float32)
        params = LuminanceRecombineParams(luma_method="rec601", luma_weights=[1.0, 0.0, 0.0])
        # Should not raise, and should run the "equal-ish" red-only weighting path.
        out = recombine_luminance(rgb, new_l, params)
        assert out.shape == rgb.shape

    def test_saturation_boost_zero_is_noop_before_recombine(self):
        rgb = _make_rgb()
        new_l = compute_luminance(rgb, method="rec709")
        params = LuminanceRecombineParams(saturation_boost=0.0, pedestal=0.0)
        out_a = recombine_luminance(rgb, new_l, params)
        np.testing.assert_allclose(out_a, rgb, atol=1e-4)

    def test_mask_zero_returns_original(self):
        rgb = _make_rgb()
        new_l = np.full((32, 32), 0.9, dtype=np.float32)
        mask = Mask(data=np.zeros((32, 32), dtype=np.float32))
        out = recombine_luminance(rgb, new_l, mask=mask)
        np.testing.assert_allclose(out, rgb, atol=1e-5)

    def test_mask_one_returns_full_result(self):
        rgb = _make_rgb()
        new_l = np.full((32, 32), 0.9, dtype=np.float32)
        mask = Mask(data=np.ones((32, 32), dtype=np.float32))
        out_masked = recombine_luminance(rgb, new_l, mask=mask)
        out_unmasked = recombine_luminance(rgb, new_l)
        np.testing.assert_allclose(out_masked, out_unmasked, atol=1e-5)

    def test_output_dtype_and_range(self):
        rgb = _make_rgb()
        new_l = np.random.default_rng(2).random((32, 32)).astype(np.float32)
        out = recombine_luminance(rgb, new_l)
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0
