"""Tests for super-resolution upscaling.

This feature is wired into the Tools panel. The RRDBNet architecture is now
vendored (astraios.ai.models.rrdbnet) rather than imported from basicsr, so
the neural path is reachable on a stock install once the pinned weights are
cached; without cached weights it still degrades to interpolation.

These tests pin the scale/shape/colour contract, the fallback, and the
checkpoint key names the vendored architecture has to keep matching.
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


@pytest.fixture
def isolated_weights(tmp_path, monkeypatch):
    """Point the weight cache at an empty dir and make downloading impossible.

    Both halves matter. Without the empty dir a developer who has already
    cached real weights runs a different test from CI; without the blocked
    download the suite would fetch 64 MB from GitHub on every run.
    """
    import astraios.ai.inference.super_resolution as sr

    monkeypatch.setattr(sr, "MODEL_DIR", tmp_path / "models")

    def _blocked(*a, **k):
        raise AssertionError("super-resolution tried to hit the network")

    monkeypatch.setattr(sr, "_download_weights", _blocked)
    return sr


class TestVendoredArchitecture:
    """RRDBNet is vendored rather than imported from basicsr, so these guard
    the thing vendoring can get wrong: silently diverging from the released
    checkpoints and reconstructing garbage."""

    @pytest.mark.parametrize("scale,expect", [(2, 32), (4, 64)])
    def test_scale_factor_is_honoured(self, scale, expect):
        import torch

        from astraios.ai.models.rrdbnet import RRDBNet

        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale)
        with torch.no_grad():
            out = model(torch.rand(1, 3, 16, 16))
        assert out.shape == (1, 3, expect, expect)

    def test_parameter_names_match_the_official_checkpoints(self):
        """The released .pth files are keyed by these names. If a refactor
        renames a layer the weights stop loading, and the loader's
        strict=False would hide that as a silently random model."""
        from astraios.ai.models.rrdbnet import RRDBNet

        keys = set(RRDBNet(num_in_ch=3, num_out_ch=3, scale=4).state_dict())
        for expected in (
            "conv_first.weight",
            "conv_body.weight",
            "conv_up1.weight",
            "conv_up2.weight",
            "conv_hr.weight",
            "conv_last.weight",
            "body.0.rdb1.conv1.weight",
            "body.22.rdb3.conv5.bias",
        ):
            assert expected in keys, f"missing checkpoint key {expected}"

    def test_block_count_matches_the_released_depth(self):
        from astraios.ai.models.rrdbnet import RRDBNet

        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4)
        assert len(model.body) == 23


class TestFallbackDoesNotHitNetwork:
    def test_upscale_falls_back_rather_than_downloading(self, isolated_weights):
        """With no cached weights the tool must degrade to interpolation, not
        stall a click on a 64 MB fetch inside the test suite."""
        out = upscale(_img(), SuperResParams(scale=2, tile_size=0))
        assert out.shape == (96, 128)

    def test_load_model_returns_none_when_weights_are_unavailable(self, isolated_weights):
        """None is the contract _upscale_ai checks before falling back to
        Lanczos. Returning a random-weight model instead would produce
        confident nonsense."""
        assert isolated_weights._load_model(2) is None
