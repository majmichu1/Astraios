"""Tests for channel alignment — channel_match.py."""

import numpy as np
import pytest

from astraios.core.channel_match import (
    ChannelMatchParams,
    _align_channel_ecc,
    _align_channel_fft,
    _apply_shift,
    align_channels,
)


def _rgb(h: int = 100, w: int = 100) -> np.ndarray:
    """Create an (H, W, 3) float32 test image with distinct channels."""
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:, :, 0] = 0.2
    img[:, :, 1] = 0.5
    img[:, :, 2] = 0.8
    return img


def _gradient_rgb(h: int = 100, w: int = 100) -> np.ndarray:
    """Create an (H, W, 3) image with identical spatial structure per channel."""
    yy, xx = np.mgrid[:h, :w]
    base = (np.sin(xx * 0.1) * np.cos(yy * 0.08) + 1.0) / 2.0
    base = base.astype(np.float32) * 0.5 + 0.25
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:, :, 0] = base
    img[:, :, 1] = base
    img[:, :, 2] = base
    return img


# ---------------------------------------------------------------------------
# _apply_shift
# ---------------------------------------------------------------------------


class TestApplyShift:
    def test_zero_shift_returns_copy(self):
        img = np.random.rand(50, 50).astype(np.float32)
        result = _apply_shift(img, 0.0, 0.0)
        np.testing.assert_array_equal(result, img)
        assert result is not img

    def test_integer_shift(self):
        img = np.zeros((50, 50), dtype=np.float32)
        img[10, 10] = 1.0
        result = _apply_shift(img, -5.0, -5.0)
        assert result[15, 15] == pytest.approx(1.0, abs=0.5)

    def test_preserves_shape(self):
        img = np.random.rand(60, 40).astype(np.float32)
        result = _apply_shift(img, 3.0, -2.0)
        assert result.shape == (60, 40)
        assert result.dtype == np.float32

    def test_negative_positive_shift(self):
        img = np.zeros((30, 30), dtype=np.float32)
        img[5, 5] = 1.0
        result = _apply_shift(img, -3.0, 0.0)
        assert result[8, 5] == pytest.approx(1.0, abs=0.5)
        result2 = _apply_shift(img, 3.0, 0.0)
        assert result2[2, 5] == pytest.approx(1.0, abs=0.5)


# ---------------------------------------------------------------------------
# _align_channel_fft
# ---------------------------------------------------------------------------


class TestAlignChannelFFT:
    def test_identity_shift(self):
        img = np.random.rand(64, 64).astype(np.float32)
        dy, dx = _align_channel_fft(img, img, max_shift=50)
        assert abs(dy) < 0.5
        assert abs(dx) < 0.5

    def test_known_integer_shift(self):
        ref = np.random.rand(64, 64).astype(np.float32)
        shifted = _apply_shift(ref, -3.0, 2.0)
        dy, dx = _align_channel_fft(shifted, ref, max_shift=50)
        assert abs(dy + 3.0) < 1.0
        assert abs(dx - 2.0) < 1.0

    def test_max_shift_clamp(self):
        ref = np.random.rand(32, 32).astype(np.float32)
        shifted = _apply_shift(ref, -100.0, 0.0)
        dy, dx = _align_channel_fft(shifted, ref, max_shift=10)
        assert abs(dy) <= 10.0

    def test_uniform_image_does_not_crash(self):
        ref = np.ones((32, 32), dtype=np.float32) * 0.5
        moving = ref.copy()
        dy, dx = _align_channel_fft(moving, ref)
        assert isinstance(dy, float)
        assert isinstance(dx, float)


# ---------------------------------------------------------------------------
# _align_channel_ecc
# ---------------------------------------------------------------------------


class TestAlignChannelECC:
    def test_identity(self):
        img = np.random.rand(64, 64).astype(np.float32)
        dy, dx, scale, angle = _align_channel_ecc(img, img, max_shift=50)
        assert abs(dy) < 1.0
        assert abs(dx) < 1.0
        assert abs(scale - 1.0) < 0.05
        assert abs(angle) < 1.0

    def test_small_shift(self):
        # Use structured Gaussian blobs for ECC convergence
        np.random.seed(42)
        ref = np.zeros((64, 64), dtype=np.float32)
        for _ in range(10):
            y, x = np.random.randint(10, 54, size=2)
            ref[y - 2:y + 2, x - 2:x + 2] = np.random.rand()
        ref = np.clip(ref, 0, 1).astype(np.float32)
        shifted = _apply_shift(ref, -2.5, 1.5)
        dy, dx, scale, angle = _align_channel_ecc(shifted, ref, max_shift=50)
        assert abs(dy - 2.5) < 1.5
        assert abs(dx + 1.5) < 1.5
        assert abs(scale - 1.0) < 0.1
        assert abs(angle) < 5.0

    def test_too_small_raises(self):
        ref = np.ones((8, 8), dtype=np.float32)
        mov = ref.copy()
        with pytest.raises(RuntimeError, match="too small"):
            _align_channel_ecc(mov, ref)


# ---------------------------------------------------------------------------
# align_channels (integration)
# ---------------------------------------------------------------------------


class TestAlignChannels:
    def test_identity_image(self):
        img = _rgb(100, 100)
        result = align_channels(img)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.float32

    def test_green_reference_unchanged(self):
        img = _rgb(100, 100)
        result = align_channels(img)
        np.testing.assert_array_equal(result[:, :, 1], img[:, :, 1])

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            align_channels(np.zeros((100, 100), dtype=np.float32))

    def test_invalid_reference_channel_raises(self):
        img = _rgb(50, 50)
        with pytest.raises(ValueError, match="reference_channel"):
            align_channels(img, ChannelMatchParams(reference_channel="X"))

    def test_mean_reference(self):
        img = _rgb(50, 50)
        params = ChannelMatchParams(reference_channel="Mean")
        result = align_channels(img, params)
        assert result.shape == (50, 50, 3)

    def test_fft_method_explicit(self):
        img = _gradient_rgb(64, 64)
        params = ChannelMatchParams(method="fft", zoom_out=1)
        result = align_channels(img, params)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.float32

    def test_known_shift_fft(self):
        """Shift R and verify alignment reduces error (crop border for edge effects)."""
        img = _gradient_rgb(128, 128)
        shifted_img = img.copy()
        shifted_img[:, :, 0] = _apply_shift(img[:, :, 0], -3.0, 2.0)

        params = ChannelMatchParams(method="fft", zoom_out=1, max_translation=20)
        result = align_channels(shifted_img, params)

        # Crop 10 px border to exclude interpolation edge effects
        err_before = np.abs(shifted_img[10:-10, 10:-10, 0] - shifted_img[10:-10, 10:-10, 1]).mean()
        err_after = np.abs(result[10:-10, 10:-10, 0] - result[10:-10, 10:-10, 1]).mean()
        assert err_after < err_before * 0.9

    def test_known_shift_ecc(self):
        """ECC method should also reduce alignment error."""
        img = _gradient_rgb(64, 64)
        shifted_img = img.copy()
        shifted_img[:, :, 0] = _apply_shift(img[:, :, 0], -2.0, 1.0)

        params = ChannelMatchParams(method="ecc", zoom_out=1, max_translation=20)
        result = align_channels(shifted_img, params)

        err_before = np.abs(shifted_img[:, :, 0] - shifted_img[:, :, 1]).mean()
        err_after = np.abs(result[:, :, 0] - result[:, :, 1]).mean()
        assert err_after < err_before + 0.02

    def test_clip_output_range(self):
        img = np.ones((50, 50, 3), dtype=np.float32) * 0.5
        result = align_channels(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_none_params_uses_defaults(self):
        img = _rgb(50, 50)
        result_none = align_channels(img)
        result_default = align_channels(img, ChannelMatchParams())
        np.testing.assert_array_almost_equal(result_none, result_default)

    def test_zoom_out_produces_valid_result(self):
        """zoom_out > 1 should still produce a valid aligned result."""
        img = _gradient_rgb(128, 128)
        params = ChannelMatchParams(zoom_out=4, method="fft")
        result = align_channels(img, params)
        assert result.shape == (128, 128, 3)
        assert result.dtype == np.float32

    def test_all_channels_aligned_to_green(self):
        """After alignment with G reference, R and B should match G closely."""
        img = _gradient_rgb(64, 64)
        result = align_channels(img)
        r_err = np.abs(result[:, :, 0] - result[:, :, 1]).mean()
        b_err = np.abs(result[:, :, 2] - result[:, :, 1]).mean()
        assert r_err < 0.15
        assert b_err < 0.15
