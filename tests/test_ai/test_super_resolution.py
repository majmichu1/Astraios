"""Tests for super-resolution upscaling.

This feature is wired into the Tools panel but had no test coverage. On a
stock install (no basicsr, no Real-ESRGAN weights) it falls back to a
denoise + interpolation enlarge; these tests pin that behaviour and the
scale/shape/colour contract so a regression cannot ship silently.
"""

import numpy as np
import pytest

from astraios.ai.inference.super_resolution import (
    MODEL_URLS,
    SuperResParams,
    _upscale_classic,
    upscale,
)


def _img(h=48, w=64):
    rng = np.random.default_rng(0)
    return np.clip(rng.random((h, w)).astype(np.float32) * 0.5 + 0.1, 0, 1)


class TestUpscaleContract:
    def test_mono_doubles_dimensions(self):
        out = upscale(_img(48, 64), SuperResParams(scale=2, tile_size=0))
        assert out.shape == (96, 128)

    def test_mono_quadruples_dimensions(self):
        out = upscale(_img(32, 40), SuperResParams(scale=4, tile_size=0))
        assert out.shape == (128, 160)

    def test_color_preserves_channels(self):
        rng = np.random.default_rng(1)
        img = np.clip(rng.random((3, 40, 50)).astype(np.float32), 0, 1)
        out = upscale(img, SuperResParams(scale=2, tile_size=0))
        assert out.shape == (3, 80, 100)

    def test_output_stays_in_range(self):
        out = upscale(_img(), SuperResParams(scale=2, tile_size=0))
        assert np.isfinite(out).all()
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_dtype_preserved(self):
        img = _img().astype(np.float32)
        out = upscale(img, SuperResParams(scale=2, tile_size=0))
        assert out.dtype == np.float32

    def test_deterministic(self):
        img = _img()
        a = upscale(img, SuperResParams(scale=2, tile_size=0, pre_denoise=False))
        b = upscale(img, SuperResParams(scale=2, tile_size=0, pre_denoise=False))
        assert np.array_equal(a, b)


class TestClassicUpscalers:
    @pytest.mark.parametrize("method", ["bicubic", "lanczos"])
    def test_explicit_classic_methods(self, method):
        out = upscale(_img(40, 40), SuperResParams(scale=2, model=method, tile_size=0))
        assert out.shape == (80, 80)

    def test_classic_helper_mono_and_color(self):
        assert _upscale_classic(_img(20, 30), 2, "bicubic").shape == (40, 60)
        c = np.stack([_img(20, 30)] * 3)
        assert _upscale_classic(c, 3, "lanczos").shape == (3, 60, 90)


class TestModelUrls:
    """The download URLs must point at assets that actually exist. The old
    RealESRGAN_x2.pth / _x4.pth names 404'd; the real assets are the 'plus'
    checkpoints."""

    def test_urls_use_the_real_plus_checkpoints(self):
        assert "RealESRGAN_x2plus.pth" in MODEL_URLS["real_esrgan_x2.pth"]
        assert "RealESRGAN_x4plus.pth" in MODEL_URLS["real_esrgan_x4.pth"]

    def test_no_url_points_at_the_dead_v0220_names(self):
        for url in MODEL_URLS.values():
            assert "RealESRGAN_x2.pth" not in url
            assert "RealESRGAN_x4.pth" not in url


class TestFallbackDoesNotHitNetwork:
    def test_upscale_never_downloads_on_a_stock_install(self, monkeypatch):
        """Without basicsr the AI path must use the local simple upsampler and
        never reach the download code. Patch urllib.request.urlretrieve
        globally so any attempt to fetch weights fails the test."""
        import urllib.request

        def _boom(*a, **k):
            raise AssertionError("super-resolution tried to hit the network")

        monkeypatch.setattr(urllib.request, "urlretrieve", _boom)
        out = upscale(_img(), SuperResParams(scale=2, tile_size=0))
        assert out.shape == (96, 128)

    def test_missing_basicsr_yields_the_simple_upsampler(self):
        """Guards the documented fallback: no basicsr -> _SimpleUpsampler,
        so 'Super-Resolution' is honest interpolation rather than a no-op or
        a crash."""
        import importlib.util

        from astraios.ai.inference.super_resolution import _load_model

        model = _load_model(2)
        # A model object is always returned (never None on the no-basicsr
        # path); if basicsr is genuinely absent it is the simple upsampler.
        if importlib.util.find_spec("basicsr") is None:
            assert type(model).__name__ == "_SimpleUpsampler"
        else:
            assert model is not None
