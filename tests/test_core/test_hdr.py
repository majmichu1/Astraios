"""Tests for HDR composition."""

import numpy as np
import pytest

from astraios.core.hdr import HDRMethod, HDRParams, hdr_compose


class TestHDRCompose:
    def test_mertens_fusion(self):
        # Two exposures: dark and bright
        dark = np.random.rand(50, 50).astype(np.float32) * 0.3
        bright = np.random.rand(50, 50).astype(np.float32) * 0.7 + 0.3
        result = hdr_compose([dark, bright])
        assert result.shape == (50, 50)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_weighted_average(self):
        dark = np.ones((50, 50), dtype=np.float32) * 0.2
        bright = np.ones((50, 50), dtype=np.float32) * 0.8
        params = HDRParams(method=HDRMethod.WEIGHTED_AVERAGE)
        result = hdr_compose([dark, bright], params)
        assert result.shape == (50, 50)
        # Result should be between the two
        assert 0.2 < result.mean() < 0.8

    def test_color_images(self):
        dark = np.random.rand(3, 50, 50).astype(np.float32) * 0.3
        bright = np.random.rand(3, 50, 50).astype(np.float32) * 0.7 + 0.3
        result = hdr_compose([dark, bright])
        assert result.shape == (3, 50, 50)

    def test_three_exposures(self):
        low = np.random.rand(50, 50).astype(np.float32) * 0.2
        mid = np.random.rand(50, 50).astype(np.float32) * 0.3 + 0.3
        high = np.random.rand(50, 50).astype(np.float32) * 0.3 + 0.7
        result = hdr_compose([low, mid, high])
        assert result.shape == (50, 50)

    def test_too_few_images_raises(self):
        single = np.random.rand(50, 50).astype(np.float32)
        with pytest.raises(ValueError):
            hdr_compose([single])

    def test_output_in_range(self):
        imgs = [np.random.rand(50, 50).astype(np.float32) for _ in range(3)]
        result = hdr_compose(imgs)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestTonemapOperators:
    def test_drago_maps_black_to_black(self):
        """Regression: the Drago curve started at log10(2) ~= 0.30, lifting
        all blacks by 30% and compressing the image into [0.30, 1]."""
        from astraios.core.hdr_operators import tonemap_drago

        data = np.linspace(0.0, 100.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        tone = tonemap_drago(data)
        assert tone.min() < 0.05, f"black lifted to {tone.min()}"
        assert abs(tone.max() - 1.0) < 1e-5
        assert np.all(np.diff(tone.ravel()) >= -1e-6)  # monotonic
        assert np.all(np.isfinite(tone))

    def test_drago_color(self):
        from astraios.core.hdr_operators import tonemap_drago

        data = np.random.rand(3, 32, 32).astype(np.float32) * 10.0
        tone = tonemap_drago(data)
        assert tone.shape == data.shape
        assert tone.min() >= 0.0 and tone.max() <= 1.0

    def test_reinhard_in_range(self):
        from astraios.core.hdr_operators import tonemap_reinhard

        data = np.random.rand(32, 32).astype(np.float32) * 50.0
        tone = tonemap_reinhard(data)
        assert tone.min() >= 0.0 and tone.max() <= 1.0

    def test_core_blend_color_shape(self):
        """Regression: the core mask was blurred over (C, H, W) and then
        broadcast via [np.newaxis, ...], inflating color output to
        (C, C, H, W) and crashing downstream transposes."""
        from astraios.core.hdr_operators import tonemap_core_blend

        data = np.random.rand(3, 32, 32).astype(np.float32) * 0.8 + 0.1
        data[:, 16, 16] = 1.0  # bright core pixel
        tone = tonemap_core_blend(data)
        assert tone.shape == data.shape


class TestReinhardAdaptation:
    """light_adapt and color_adapt were declared on ReinhardParams and never
    read, so the operator silently ignored two of its three settings. These
    pin the behaviour now that they are implemented."""

    @staticmethod
    def _hdr_field():
        rng = np.random.default_rng(7)
        img = np.clip(rng.random((3, 48, 64)).astype(np.float32) * 0.6 + 0.05, 0, 1)
        img[:, 20:24, 30:36] = 3.0      # a core well above the display range
        return img

    def test_defaults_are_the_old_global_curve(self):
        """The adaptation term must be exactly 1.0 at the defaults, so
        existing projects re-run to the same pixels."""
        from astraios.core.hdr_operators import ReinhardParams, tonemap_reinhard

        img = self._hdr_field()
        got = tonemap_reinhard(img, ReinhardParams())

        expected = np.empty_like(img)
        for ch in range(3):
            s = img[ch]
            l_white = max(float(np.max(s)), 1e-10)
            expected[ch] = np.clip(s * (1.0 + s / (l_white * l_white)) / (1.0 + s), 0.0, 1.0)
        assert np.array_equal(got, expected)

    def test_light_adapt_rescues_a_clipped_core(self):
        """The point of the setting: a blown core comes back below clipping
        without dragging the background with it."""
        from astraios.core.hdr_operators import ReinhardParams, tonemap_reinhard

        img = self._hdr_field()
        core = (slice(None), slice(20, 24), slice(30, 36))

        flat = tonemap_reinhard(img, ReinhardParams(light_adapt=0.0))
        adapted = tonemap_reinhard(img, ReinhardParams(light_adapt=0.5))

        assert (flat[core] >= 0.999).all(), "core should clip with global adaptation"
        assert (adapted[core] < 0.999).all(), "light_adapt should pull it back"

        bg = (slice(None), slice(0, 10), slice(0, 10))
        assert abs(float(adapted[bg].mean()) - float(flat[bg].mean())) < 0.01

    def test_color_adapt_desaturates(self):
        from astraios.core.hdr_operators import ReinhardParams, tonemap_reinhard

        img = self._hdr_field()
        keep = tonemap_reinhard(img, ReinhardParams(light_adapt=1.0, color_adapt=0.0))
        drop = tonemap_reinhard(img, ReinhardParams(light_adapt=1.0, color_adapt=1.0))

        spread = lambda a: float(np.abs(a.max(axis=0) - a.min(axis=0)).mean())  # noqa: E731
        assert spread(drop) < spread(keep) / 2

    def test_mono_and_out_of_range_inputs_are_handled(self):
        from astraios.core.hdr_operators import ReinhardParams, tonemap_reinhard

        img = self._hdr_field()
        mono = tonemap_reinhard(img[0], ReinhardParams(light_adapt=1.0, color_adapt=1.0))
        assert mono.shape == img[0].shape
        assert np.isfinite(mono).all()
        assert mono.min() >= 0.0 and mono.max() <= 1.0

    def test_adaptation_settings_are_clamped(self):
        """Out-of-range values must not produce a negative divisor."""
        from astraios.core.hdr_operators import ReinhardParams, tonemap_reinhard

        img = self._hdr_field()
        out = tonemap_reinhard(img, ReinhardParams(light_adapt=5.0, color_adapt=-3.0))
        assert np.isfinite(out).all()
        assert out.min() >= 0.0 and out.max() <= 1.0
