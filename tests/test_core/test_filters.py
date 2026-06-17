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
