"""Regressions for the August 2026 audit: each test reproduces a confirmed
defect first, then pins the fix."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from astraios.core.device_manager import get_device_manager


def _rng(seed=0):
    return np.random.default_rng(seed)


class TestResizeNearest:
    def test_colour_nearest_does_not_raise(self):
        from astraios.core.transforms import InterpolationMethod, ResizeParams, resize

        img = _rng().random((3, 40, 60)).astype(np.float32)
        out = resize(img, ResizeParams(scale=0.5, interpolation=InterpolationMethod.NEAREST))
        assert out.shape == (3, 20, 30)


class TestReinhard:
    def test_black_pixels_with_full_light_adapt_are_finite(self):
        from astraios.core.hdr_operators import ReinhardParams, tonemap_reinhard

        img = _rng(1).random((32, 32)).astype(np.float32)
        img[:8] = 0.0
        out = tonemap_reinhard(img, ReinhardParams(light_adapt=1.0))
        assert np.isfinite(out).all()
        assert (out[:8] == 0).all()

    def test_drago_saturation_is_read(self):
        from astraios.core.hdr_operators import DragoParams, tonemap_drago

        img = _rng(2).random((3, 16, 16)).astype(np.float32)
        base = tonemap_drago(img, DragoParams())
        desat = tonemap_drago(img, DragoParams(saturation=0.0))
        assert np.allclose(desat[0], desat[1]) and np.allclose(desat[1], desat[2])
        assert not np.allclose(base, desat)


class TestMonoCubeAndRGBA:
    def test_ensure_rgb_handles_single_channel_cube(self):
        from astraios.core.selective_adjust import _ensure_rgb_chw

        assert _ensure_rgb_chw(np.zeros((1, 8, 8), np.float32)).shape == (3, 8, 8)

    def test_colour_film_grain_on_rgba_and_mono_cube(self):
        from astraios.core.fx_effects import FXParams, _fx_film_grain

        p = FXParams(grain_intensity=0.5, grain_mono=False)
        rgba = _rng(3).random((8, 8, 4)).astype(np.float32)
        assert _fx_film_grain(rgba, p, None).shape == (8, 8, 4)
        mono = _rng(4).random((8, 8, 1)).astype(np.float32)
        assert _fx_film_grain(mono, p, None).shape == (8, 8, 1)


class TestMorphologyIterations:
    def test_gpu_matches_cv2_for_close_with_iterations(self):
        from astraios.core.morphology import MorphologyParams, MorphOp, _morphology_cpu, _morphology_gpu

        img = (_rng(5).random((48, 48)) > 0.6).astype(np.float32)
        params = MorphologyParams(operation=MorphOp.CLOSE, kernel_size=3, iterations=3)
        import cv2

        kernel = np.ones((3, 3), np.uint8)
        cpu = _morphology_cpu(img, kernel, cv2.MORPH_CLOSE, 3)
        gpu = _morphology_gpu(img, MorphOp.CLOSE, 3, 3)
        inner = (slice(4, -4), slice(4, -4))  # away from the border modes
        assert np.array_equal(cpu[inner], gpu[inner]), params


class TestGHSLinked:
    def test_linked_keeps_channel_ratios_unlinked_does_not(self):
        from astraios.core.stretch import GHSParams, generalized_hyperbolic_stretch

        base = _rng(6).random((24, 24)).astype(np.float32) * 0.5
        img = np.stack([base, base * 0.5, base * 0.25]).astype(np.float32)
        linked = generalized_hyperbolic_stretch(img, GHSParams(D=3.0, linked=True))
        unlinked = generalized_hyperbolic_stretch(img, GHSParams(D=3.0, linked=False))
        # Unlinked renormalises each channel to its own max, so every channel
        # peaks at 1.0; linked shares one scale, so the dim channels stay dim.
        assert unlinked[2].max() == pytest.approx(1.0, abs=1e-5)
        assert linked[2].max() < linked[0].max()


class TestLinearFitRejection:
    def test_rejects_the_outlier_and_keeps_a_sky_trend(self):
        from astraios.core.stacking import _reject_linear_fit

        # 20 frames: with fewer, a single end-of-stack outlier has enough
        # leverage on the fit to hide inside 3 sigma of its own residual.
        n = 20
        trend = np.linspace(0.20, 0.30, n, dtype=np.float32)  # sky brightening
        stack = np.repeat(trend[:, None, None], 16, axis=1).repeat(16, axis=2)
        stack = stack + _rng(7).normal(0, 0.002, stack.shape).astype(np.float32)
        stack[5, 8, 8] = 0.9  # a satellite hit
        m = _reject_linear_fit(stack, 3.0, 3.0, 5)
        assert m.mask[5, 8, 8]
        assert m.mask.sum() < 0.05 * stack.size

    def test_gpu_twin_agrees_with_numpy(self):
        from astraios.core.stacking import _gpu_linear_fit_clip, _reject_linear_fit

        stack = _rng(8).random((9, 10, 10)).astype(np.float32) * 0.1
        stack[2, 3, 3] = 1.0
        stack[7, 5, 5] = 0.0
        cpu = _reject_linear_fit(stack, 2.5, 2.5, 5)
        keep_t, n_rej = _gpu_linear_fit_clip(get_device_manager().from_numpy(stack), 2.5, 2.5, 5)
        keep = keep_t.cpu().numpy()
        assert np.array_equal(~cpu.mask, keep)
        assert n_rej == int(cpu.mask.sum())


class TestGPUSigmaClipEvenCount:
    def test_even_frame_count_matches_astropy(self):
        from astraios.core.stacking import _gpu_sigma_clip, _reject_sigma_clip

        stack = _rng(9).normal(0.3, 0.01, (8, 12, 12)).astype(np.float32)
        stack[3, 4, 4] = 0.8
        stack[6, 7, 7] = 0.02
        cpu = _reject_sigma_clip(stack, 3.0, 3.0, 5)
        keep_t, _ = _gpu_sigma_clip(get_device_manager().from_numpy(stack), 3.0, 3.0, 5)
        assert np.array_equal(~np.ma.getmaskarray(cpu), keep_t.cpu().numpy())


class TestDrizzleGPUFootprint:
    def test_gpu_and_cpu_frames_agree(self):
        from astraios.core.drizzle import _drizzle_frame_gpu, _drizzle_frame_numpy

        img = _rng(10).random((20, 24)).astype(np.float32)
        tf = np.array([[1.0, 0.0, 0.3], [0.0, 1.0, -0.4]], dtype=np.float32)
        scale, shrink = 2, 0.7
        out_h, out_w = 20 * scale, 24 * scale
        out_c = np.zeros((out_h, out_w), np.float32)
        w_c = np.zeros((out_h, out_w), np.float32)
        _drizzle_frame_numpy(img, out_c, w_c, tf, scale, shrink, "uniform")
        dm = get_device_manager()
        out_g = torch.zeros((out_h, out_w), device=dm.device)
        w_g = torch.zeros((out_h, out_w), device=dm.device)
        _drizzle_frame_gpu(img, out_g, w_g, tf, scale, shrink)
        assert np.array_equal(w_c, w_g.cpu().numpy())
        assert np.allclose(out_c, out_g.cpu().numpy(), atol=2e-6)


class TestSmallFixes:
    def test_luminance_recombine_equal_round_trips(self):
        from astraios.core.luminance_recombine import (
            LuminanceRecombineParams, compute_luminance, recombine_luminance,
        )

        rgb = _rng(11).random((3, 12, 12)).astype(np.float32) * 0.8
        lum = compute_luminance(rgb, method="equal")
        out = recombine_luminance(rgb, lum, LuminanceRecombineParams(luma_method="equal"))
        assert np.allclose(out, rgb, atol=1e-4)

    def test_wavescale_dark_mask_ignores_residual_for_two_scales(self):
        from astraios.core.wavescale_dark_enhance import _darkness_mask

        img = _rng(12).random((64, 64)).astype(np.float32)
        m2 = _darkness_mask(img, 2, 1.0)
        assert m2.shape == img.shape and np.isfinite(m2).all()

    def test_arcsinh_linked_black_point_keeps_ratios(self):
        from astraios.core.stretch import ArcsinhStretchParams, arcsinh_stretch

        base = _rng(13).random((16, 16)).astype(np.float32) * 0.5 + 0.1
        img = np.stack([base, base * 0.5, base * 0.5]).astype(np.float32)
        out = arcsinh_stretch(img, ArcsinhStretchParams(stretch_factor=5.0, black_point=0.05, linked=True))
        ratio = out[1] / np.maximum(out[0], 1e-6)
        expected = (img[1] - 0.05) / np.maximum(img[0] - 0.05, 1e-6)
        sel = (img[0] - 0.05) > 0.05
        assert np.allclose(ratio[sel], expected[sel], atol=1e-3)
