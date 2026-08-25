"""Tests for image filters — unsharp mask and median filter."""

import numpy as np
import pytest

from astraios.core.filters import (
    MedianFilterParams,
    UnsharpMaskParams,
    median_filter,
    unsharp_mask,
)
from astraios.core.masks import Mask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mono(h: int = 100, w: int = 100, value: float = 0.5) -> np.ndarray:
    """Create a mono float32 image of shape (H, W)."""
    return np.full((h, w), value, dtype=np.float32)


def _color(h: int = 100, w: int = 100) -> np.ndarray:
    """Create a 3-channel float32 image of shape (C, H, W) with distinct channels."""
    img = np.empty((3, h, w), dtype=np.float32)
    img[0] = 0.2
    img[1] = 0.5
    img[2] = 0.8
    return img


def _noisy_mono(h: int = 100, w: int = 100, seed: int = 42) -> np.ndarray:
    """Create a noisy mono image with salt-and-pepper outliers."""
    rng = np.random.RandomState(seed)
    img = np.full((h, w), 0.3, dtype=np.float32)
    # Add salt-and-pepper noise at ~2% of pixels.
    n_noisy = int(h * w * 0.02)
    ys = rng.randint(0, h, n_noisy)
    xs = rng.randint(0, w, n_noisy)
    img[ys[:n_noisy // 2], xs[:n_noisy // 2]] = 1.0  # salt
    img[ys[n_noisy // 2:], xs[n_noisy // 2:]] = 0.0  # pepper
    return img


# ---------------------------------------------------------------------------
# Unsharp Mask
# ---------------------------------------------------------------------------


class TestUnsharpMask:
    """Tests for unsharp_mask()."""

    def test_basic_sharpening_mono(self):
        """Unsharp mask with positive amount should alter the image."""
        img = _mono(100, 100, value=0.5)
        # Add a soft feature.
        img[45:55, 45:55] = 0.8
        params = UnsharpMaskParams(radius=2.0, amount=1.0, threshold=0.0)
        result = unsharp_mask(img, params)
        assert result.shape == (100, 100)
        assert result.dtype == np.float32
        # The edge between 0.5 and 0.8 should be enhanced (overshoot).
        assert result.max() > img.max() - 0.01
        # Result should differ from input.
        assert not np.allclose(result, img)

    def test_basic_sharpening_color(self):
        img = _color(100, 100)
        img[:, 45:55, 45:55] = 0.9
        params = UnsharpMaskParams(radius=2.0, amount=1.0, threshold=0.0)
        result = unsharp_mask(img, params)
        assert result.shape == (3, 100, 100)
        assert result.dtype == np.float32
        assert not np.allclose(result, img)

    def test_amount_zero_is_identity(self):
        """amount=0 should return the original image unchanged."""
        img = np.random.RandomState(0).rand(100, 100).astype(np.float32)
        result = unsharp_mask(img, UnsharpMaskParams(amount=0.0))
        np.testing.assert_array_almost_equal(result, img)

    def test_result_clipped_to_01(self):
        """Output should always be in [0, 1]."""
        img = np.random.RandomState(1).rand(100, 100).astype(np.float32)
        params = UnsharpMaskParams(radius=3.0, amount=5.0, threshold=0.0)
        result = unsharp_mask(img, params)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_threshold_gating(self):
        """High threshold should suppress sharpening of low-contrast detail."""
        img = _mono(100, 100, value=0.5)
        img[50, 50] = 0.52  # tiny bump, well below threshold
        params_no_thresh = UnsharpMaskParams(radius=2.0, amount=2.0, threshold=0.0)
        params_high_thresh = UnsharpMaskParams(radius=2.0, amount=2.0, threshold=0.1)
        result_no = unsharp_mask(img, params_no_thresh)
        result_hi = unsharp_mask(img, params_high_thresh)
        # With high threshold, the tiny bump should be left alone.
        diff_no = abs(float(result_no[50, 50]) - img[50, 50])
        diff_hi = abs(float(result_hi[50, 50]) - img[50, 50])
        assert diff_hi <= diff_no

    def test_uniform_image_unchanged(self):
        """A perfectly uniform image has no detail to sharpen."""
        img = _mono(100, 100, value=0.4)
        params = UnsharpMaskParams(radius=3.0, amount=2.0, threshold=0.0)
        result = unsharp_mask(img, params)
        np.testing.assert_array_almost_equal(result, img, decimal=5)

    def test_mask_full_applies_sharpening(self):
        """A mask of all 1.0 should give the same result as no mask."""
        img = _mono(100, 100, value=0.5)
        img[45:55, 45:55] = 0.8
        params = UnsharpMaskParams(radius=2.0, amount=1.5)
        mask = Mask(data=np.ones((100, 100), dtype=np.float32))
        result_mask = unsharp_mask(img, params, mask=mask)
        result_none = unsharp_mask(img, params, mask=None)
        np.testing.assert_array_almost_equal(result_mask, result_none)

    def test_mask_zero_preserves_original(self):
        """A mask of all 0.0 should return the original image."""
        img = _mono(100, 100, value=0.5)
        img[45:55, 45:55] = 0.8
        params = UnsharpMaskParams(radius=2.0, amount=2.0)
        mask = Mask(data=np.zeros((100, 100), dtype=np.float32))
        result = unsharp_mask(img, params, mask=mask)
        np.testing.assert_array_almost_equal(result, img)

    def test_mask_partial_blends(self):
        """Half-mask should blend sharpened and original regions."""
        img = _mono(100, 100, value=0.5)
        img[45:55, :] = 0.8
        params = UnsharpMaskParams(radius=2.0, amount=2.0)
        mask_data = np.zeros((100, 100), dtype=np.float32)
        mask_data[:, 50:] = 1.0  # right half processed
        mask = Mask(data=mask_data)
        result = unsharp_mask(img, params, mask=mask)
        # Left half (mask=0) should equal original.
        np.testing.assert_array_almost_equal(result[:, :50], img[:, :50])
        # Right half (mask=1) should be sharpened (differ from original).
        assert not np.allclose(result[48, 50:], img[48, 50:])

    def test_mask_with_color_image(self):
        img = _color(100, 100)
        img[:, 45:55, 45:55] = 0.9
        params = UnsharpMaskParams(radius=2.0, amount=1.0)
        mask = Mask(data=np.ones((100, 100), dtype=np.float32) * 0.5)
        result = unsharp_mask(img, params, mask=mask)
        assert result.shape == (3, 100, 100)

    def test_none_params_uses_defaults(self):
        img = _mono(100, 100, value=0.5)
        img[50, 50] = 0.9
        result = unsharp_mask(img, None)
        assert result.shape == (100, 100)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Median Filter
# ---------------------------------------------------------------------------


class TestMedianFilter:
    """Tests for median_filter()."""

    def test_basic_noise_removal_mono(self):
        """Median filter should reduce salt-and-pepper noise."""
        img = _noisy_mono(100, 100)
        params = MedianFilterParams(kernel_size=3)
        result = median_filter(img, params)
        assert result.shape == (100, 100)
        assert result.dtype == np.float32
        # The standard deviation should drop because noise is smoothed.
        assert result.std() < img.std()

    def test_basic_noise_removal_color(self):
        img = _color(100, 100)
        # Inject salt noise in each channel.
        rng = np.random.RandomState(7)
        for c in range(3):
            ys = rng.randint(0, 100, 50)
            xs = rng.randint(0, 100, 50)
            img[c, ys, xs] = 1.0
        params = MedianFilterParams(kernel_size=3)
        result = median_filter(img, params)
        assert result.shape == (3, 100, 100)
        assert result.dtype == np.float32

    def test_uniform_image_unchanged(self):
        """A uniform image should pass through median filter unchanged."""
        img = _mono(100, 100, value=0.4)
        result = median_filter(img, MedianFilterParams(kernel_size=3))
        np.testing.assert_array_almost_equal(result, img)

    def test_even_kernel_becomes_odd(self):
        """Even kernel_size should be incremented to the next odd number."""
        img = _noisy_mono(100, 100)
        # kernel_size=4 should behave like kernel_size=5.
        result4 = median_filter(img, MedianFilterParams(kernel_size=4))
        result5 = median_filter(img, MedianFilterParams(kernel_size=5))
        np.testing.assert_array_equal(result4, result5)

    def test_kernel_size_1_is_identity(self):
        """A 1x1 median filter should not change anything."""
        img = np.random.RandomState(3).rand(100, 100).astype(np.float32)
        result = median_filter(img, MedianFilterParams(kernel_size=1))
        np.testing.assert_array_almost_equal(result, img)

    def test_larger_kernel_more_smoothing(self):
        """Larger kernels should produce more smoothing."""
        img = _noisy_mono(100, 100)
        result3 = median_filter(img, MedianFilterParams(kernel_size=3))
        result7 = median_filter(img, MedianFilterParams(kernel_size=7))
        # Larger kernel should yield lower standard deviation.
        assert result7.std() <= result3.std()

    def test_mask_zero_preserves_original(self):
        """A zero mask should return the original noisy image."""
        img = _noisy_mono(100, 100)
        mask = Mask(data=np.zeros((100, 100), dtype=np.float32))
        result = median_filter(img, MedianFilterParams(kernel_size=5), mask=mask)
        np.testing.assert_array_almost_equal(result, img)

    def test_mask_full_applies_filter(self):
        """A full mask should give the same result as no mask."""
        img = _noisy_mono(100, 100)
        mask = Mask(data=np.ones((100, 100), dtype=np.float32))
        result_mask = median_filter(img, MedianFilterParams(kernel_size=3), mask=mask)
        result_none = median_filter(img, MedianFilterParams(kernel_size=3), mask=None)
        np.testing.assert_array_almost_equal(result_mask, result_none)

    def test_mask_partial_blends(self):
        """Partial mask should protect some regions."""
        img = _noisy_mono(100, 100)
        mask_data = np.zeros((100, 100), dtype=np.float32)
        mask_data[:, 50:] = 1.0
        mask = Mask(data=mask_data)
        result = median_filter(img, MedianFilterParams(kernel_size=5), mask=mask)
        # Left half (mask=0) should equal original.
        np.testing.assert_array_almost_equal(result[:, :50], img[:, :50])
        # Right half (mask=1) should be filtered (differ from original
        # because noise was present).
        assert not np.array_equal(result[:, 50:], img[:, 50:])

    def test_mask_with_color_image(self):
        img = _color(100, 100)
        mask = Mask(data=np.ones((100, 100), dtype=np.float32) * 0.5)
        result = median_filter(img, MedianFilterParams(kernel_size=3), mask=mask)
        assert result.shape == (3, 100, 100)

    def test_none_params_uses_defaults(self):
        img = _mono(80, 80)
        result = median_filter(img, None)
        assert result.shape == (80, 80)
        assert result.dtype == np.float32

    def test_result_stays_in_range(self):
        """Median filter should not produce values outside [0, 1]."""
        img = np.random.RandomState(9).rand(100, 100).astype(np.float32)
        result = median_filter(img, MedianFilterParams(kernel_size=5))
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestGaussianBlurFastPath:
    """gaussian_blur matches scipy's gaussian_filter(mode='reflect') and runs
    the large kernels on the GPU.

    It is deliberately NOT bit-identical there: scipy accumulates the weighted
    sum in float64 while the GPU accumulates in float32 and in a different
    order. The agreement is about 1e-5 of full scale, under one 16-bit level.
    These tests pin that bound so it cannot quietly widen.
    """

    #: Measured worst case is 1.15e-05 (0.75 of a 16-bit level) at sigma 51.
    #: The bound is set a little above that, not at a round number pulled from
    #: the air, so a real regression trips it.
    TOLERANCE = 3e-5

    @staticmethod
    def _scene(size=256):
        rng = np.random.default_rng(7)
        yy, xx = np.mgrid[0:size, 0:size]
        base = 0.08 + 0.05 * (xx / size) + rng.normal(0, 0.004, (size, size))
        for _ in range(20):
            cy, cx = rng.uniform(10, size - 10, 2)
            base += 0.6 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 3.0)
        return np.clip(base, 0, 1).astype(np.float32)

    @pytest.mark.parametrize("sigma", [1.0, 4.0, 12.0, 30.0])
    def test_agrees_with_scipy(self, sigma):
        import scipy.ndimage

        from astraios.core.filters import gaussian_blur

        img = self._scene()
        ref = scipy.ndimage.gaussian_filter(img, sigma=sigma, mode="reflect")
        got = gaussian_blur(img, sigma)
        assert got.shape == img.shape
        assert np.max(np.abs(ref - got)) < self.TOLERANCE

    def test_small_sigma_stays_exact(self):
        """Below the size threshold the CPU path runs, and scipy compared with
        scipy is exact. A tolerance would hide a wrong branch here."""
        import scipy.ndimage

        from astraios.core.filters import gaussian_blur

        img = self._scene(128)
        ref = scipy.ndimage.gaussian_filter(img, sigma=1.5, mode="reflect").astype(np.float32)
        assert np.array_equal(ref, gaussian_blur(img, 1.5))

    def test_colour_matches_per_channel_scipy(self):
        import scipy.ndimage

        from astraios.core.filters import gaussian_blur

        mono = self._scene()
        img = np.stack([mono, mono * 0.9, mono * 0.8]).astype(np.float32)
        ref = np.stack([
            scipy.ndimage.gaussian_filter(img[c], sigma=10.0, mode="reflect")
            for c in range(3)
        ])
        got = gaussian_blur(img, 10.0)
        assert got.shape == img.shape
        assert np.max(np.abs(ref - got)) < self.TOLERANCE

    def test_banding_does_not_change_the_result(self):
        """The VRAM guard splits the work into row bands with halos. Every
        output row still sees the same window, so banding must be exactly
        invisible -- a previous GPU port OOMed at 73MP and the caller silently
        skipped the step, which is what the banding exists to prevent."""
        from astraios.core.device_manager import get_device_manager

        if not get_device_manager().is_gpu:
            pytest.skip("no GPU available")

        import astraios.core.filters as filt

        img = self._scene(512)
        full = filt._gaussian_blur_gpu_2d(img, 12.0, 4.0)
        original = filt._GAUSS_BAND_BYTES
        try:
            filt._GAUSS_BAND_BYTES = 64 * 1024      # force many tiny bands
            banded = filt._gaussian_blur_gpu_2d(img, 12.0, 4.0)
        finally:
            filt._GAUSS_BAND_BYTES = original
        assert np.array_equal(full, banded)

    def test_zero_sigma_is_a_copy(self):
        from astraios.core.filters import gaussian_blur

        img = self._scene(64)
        out = gaussian_blur(img, 0.0)
        assert np.array_equal(out, img)
        assert out is not img

    def test_kernel_matches_scipys_radius_rule(self):
        from scipy.ndimage import _filters

        from astraios.core.filters import gaussian_kernel_1d

        for sigma in (1.0, 7.5, 30.0):
            k, radius = gaussian_kernel_1d(sigma)
            expected_radius = int(4.0 * sigma + 0.5)
            assert radius == expected_radius
            ref = _filters._gaussian_kernel1d(sigma, 0, expected_radius)[::-1]
            assert np.max(np.abs(k - ref.astype(np.float32))) < 1e-7
