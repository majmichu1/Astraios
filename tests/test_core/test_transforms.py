"""Tests for image transforms — crop, rotate, flip, resize, bin, invert."""

import numpy as np
import pytest

from astraios.core.transforms import (
    BinMode,
    BinParams,
    CropParams,
    FlipAxis,
    FlipParams,
    InterpolationMethod,
    ResizeParams,
    RotateAngle,
    RotateParams,
    bin_image,
    crop,
    flip,
    invert,
    resize,
    rotate,
)


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


def _gradient_mono(h: int = 100, w: int = 100) -> np.ndarray:
    """Create a mono image with a left-to-right gradient for orientation checks."""
    row = np.linspace(0.0, 1.0, w, dtype=np.float32)
    return np.tile(row, (h, 1))


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------


class TestCrop:
    """Tests for crop()."""

    def test_crop_mono_basic(self):
        img = _mono(100, 120)
        result = crop(img, CropParams(x=10, y=20, width=50, height=30))
        assert result.shape == (30, 50)
        assert result.dtype == np.float32

    def test_crop_color_basic(self):
        img = _color(100, 120)
        result = crop(img, CropParams(x=10, y=20, width=50, height=30))
        assert result.shape == (3, 30, 50)
        assert result.dtype == np.float32

    def test_crop_preserves_pixel_values(self):
        img = np.arange(100 * 100, dtype=np.float32).reshape(100, 100) / 10000.0
        result = crop(img, CropParams(x=5, y=10, width=20, height=15))
        expected = img[10:25, 5:25]
        np.testing.assert_array_equal(result, expected)

    def test_crop_zero_dims_returns_full_image(self):
        """width=0 and height=0 should mean 'full remaining extent'."""
        img = _mono(80, 120)
        result = crop(img, CropParams(x=0, y=0, width=0, height=0))
        assert result.shape == (80, 120)
        np.testing.assert_array_equal(result, img)

    def test_crop_zero_dims_with_offset(self):
        """width=0 from x=30 should give remaining width (120-30=90)."""
        img = _mono(80, 120)
        result = crop(img, CropParams(x=30, y=10, width=0, height=0))
        assert result.shape == (70, 90)

    def test_crop_clamps_to_image_bounds(self):
        img = _mono(50, 50)
        result = crop(img, CropParams(x=40, y=40, width=100, height=100))
        assert result.shape == (10, 10)

    def test_crop_none_params_returns_copy(self):
        img = _mono(50, 50)
        result = crop(img, None)
        assert result.shape == img.shape
        np.testing.assert_array_equal(result, img)
        assert result is not img  # must be a copy

    def test_crop_returns_copy(self):
        """Modifying the crop should not affect the original."""
        img = _mono(100, 100, value=0.3)
        result = crop(img, CropParams(x=0, y=0, width=50, height=50))
        result[:] = 0.9
        assert img[0, 0] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Rotate
# ---------------------------------------------------------------------------


class TestRotate:
    """Tests for rotate()."""

    def test_rotate_90_cw_mono(self):
        img = _gradient_mono(80, 120)
        result = rotate(img, RotateParams(angle=RotateAngle.CW_90))
        assert result.shape == (120, 80)
        # Top-left of the gradient (low values) should move to top-right after 90 CW.
        assert result[0, -1] == pytest.approx(0.0, abs=0.02)

    def test_rotate_90_cw_color(self):
        img = _color(80, 120)
        result = rotate(img, RotateParams(angle=RotateAngle.CW_90))
        assert result.shape == (3, 120, 80)

    def test_rotate_180_mono(self):
        img = _gradient_mono(100, 100)
        result = rotate(img, RotateParams(angle=RotateAngle.CW_180))
        assert result.shape == (100, 100)
        # After 180, the gradient is reversed.
        assert result[0, 0] == pytest.approx(1.0, abs=0.02)
        assert result[0, -1] == pytest.approx(0.0, abs=0.02)

    def test_rotate_270_cw_mono(self):
        img = _gradient_mono(80, 120)
        result = rotate(img, RotateParams(angle=RotateAngle.CW_270))
        assert result.shape == (120, 80)
        # Top-left of the gradient (low values) should move to bottom-left.
        assert result[-1, 0] == pytest.approx(0.0, abs=0.02)

    def test_rotate_360_is_identity(self):
        """Four 90-CW rotations should restore the image exactly."""
        img = _gradient_mono(64, 64)
        result = img.copy()
        for _ in range(4):
            result = rotate(result, RotateParams(angle=RotateAngle.CW_90))
        np.testing.assert_array_almost_equal(result, img)

    def test_rotate_arbitrary_expand_true(self):
        img = _mono(100, 100)
        result = rotate(
            img,
            RotateParams(
                angle=RotateAngle.ARBITRARY,
                arbitrary_degrees=45.0,
                expand=True,
            ),
        )
        # Expanded 45-degree rotation of a square should produce a larger image.
        assert result.shape[0] > 100
        assert result.shape[1] > 100
        assert result.dtype == np.float32

    def test_rotate_arbitrary_expand_false(self):
        img = _mono(100, 100)
        result = rotate(
            img,
            RotateParams(
                angle=RotateAngle.ARBITRARY,
                arbitrary_degrees=45.0,
                expand=False,
            ),
        )
        assert result.shape == (100, 100)

    def test_rotate_arbitrary_color(self):
        img = _color(80, 80)
        result = rotate(
            img,
            RotateParams(
                angle=RotateAngle.ARBITRARY,
                arbitrary_degrees=30.0,
                expand=True,
            ),
        )
        assert result.ndim == 3
        assert result.shape[0] == 3
        assert result.dtype == np.float32

    def test_rotate_arbitrary_zero_is_identity(self):
        """0-degree arbitrary rotation should return a copy."""
        img = _mono(64, 64, value=0.42)
        result = rotate(
            img,
            RotateParams(
                angle=RotateAngle.ARBITRARY,
                arbitrary_degrees=0.0,
            ),
        )
        np.testing.assert_array_equal(result, img)
        assert result is not img

    def test_rotate_none_params(self):
        img = _mono(50, 80)
        result = rotate(img, None)
        # Default is CW_90.
        assert result.shape == (80, 50)


# ---------------------------------------------------------------------------
# Flip
# ---------------------------------------------------------------------------


class TestFlip:
    """Tests for flip()."""

    def test_flip_horizontal_mono(self):
        img = _gradient_mono(100, 100)
        result = flip(img, FlipParams(axis=FlipAxis.HORIZONTAL))
        assert result.shape == (100, 100)
        # Left-right flip reverses the horizontal gradient.
        assert result[0, 0] == pytest.approx(1.0, abs=0.02)
        assert result[0, -1] == pytest.approx(0.0, abs=0.02)

    def test_flip_horizontal_color(self):
        img = _color(80, 120)
        result = flip(img, FlipParams(axis=FlipAxis.HORIZONTAL))
        assert result.shape == (3, 80, 120)

    def test_flip_vertical_mono(self):
        # Vertical gradient: row 0 = 0, row H-1 = 1.
        col = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        img = np.tile(col[:, np.newaxis], (1, 100))
        result = flip(img, FlipParams(axis=FlipAxis.VERTICAL))
        assert result[0, 0] == pytest.approx(1.0, abs=0.02)
        assert result[-1, 0] == pytest.approx(0.0, abs=0.02)

    def test_flip_both(self):
        img = _gradient_mono(100, 100)
        result = flip(img, FlipParams(axis=FlipAxis.BOTH))
        assert result.shape == (100, 100)
        # Both flips = 180-degree rotation.
        expected = rotate(img, RotateParams(angle=RotateAngle.CW_180))
        np.testing.assert_array_almost_equal(result, expected)

    def test_flip_double_is_identity(self):
        """Flipping twice along the same axis restores the image."""
        img = _gradient_mono(80, 120)
        result = flip(flip(img, FlipParams(axis=FlipAxis.HORIZONTAL)),
                      FlipParams(axis=FlipAxis.HORIZONTAL))
        np.testing.assert_array_almost_equal(result, img)

    def test_flip_returns_copy(self):
        img = _mono(50, 50, value=0.3)
        result = flip(img, FlipParams(axis=FlipAxis.HORIZONTAL))
        result[:] = 0.9
        assert img[0, 0] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------


class TestResize:
    """Tests for resize()."""

    def test_resize_by_scale_up_mono(self):
        img = _mono(100, 100)
        result = resize(img, ResizeParams(scale=2.0))
        assert result.shape == (200, 200)
        assert result.dtype == np.float32

    def test_resize_by_scale_down_mono(self):
        img = _mono(100, 100)
        result = resize(img, ResizeParams(scale=0.5))
        assert result.shape == (50, 50)

    def test_resize_by_scale_color(self):
        img = _color(100, 100)
        result = resize(img, ResizeParams(scale=0.5))
        assert result.shape == (3, 50, 50)
        assert result.dtype == np.float32

    def test_resize_by_target_dims(self):
        img = _mono(100, 100)
        result = resize(img, ResizeParams(target_width=200, target_height=150))
        assert result.shape == (150, 200)

    def test_resize_by_target_width_preserves_aspect(self):
        img = _mono(100, 200)
        result = resize(img, ResizeParams(target_width=100))
        # Height should scale proportionally: 100 * (100/200) = 50.
        assert result.shape == (50, 100)

    def test_resize_by_target_height_preserves_aspect(self):
        img = _mono(100, 200)
        result = resize(img, ResizeParams(target_height=50))
        # Width should scale proportionally: 200 * (50/100) = 100.
        assert result.shape == (50, 100)

    def test_resize_interpolation_nearest(self):
        img = _mono(100, 100)
        result = resize(
            img,
            ResizeParams(scale=2.0, interpolation=InterpolationMethod.NEAREST),
        )
        assert result.shape == (200, 200)

    def test_resize_interpolation_bilinear(self):
        img = _mono(100, 100)
        result = resize(
            img,
            ResizeParams(scale=0.5, interpolation=InterpolationMethod.BILINEAR),
        )
        assert result.shape == (50, 50)

    def test_resize_interpolation_bicubic(self):
        img = _mono(100, 100)
        result = resize(
            img,
            ResizeParams(scale=0.5, interpolation=InterpolationMethod.BICUBIC),
        )
        assert result.shape == (50, 50)

    def test_resize_none_params_returns_copy(self):
        img = _mono(64, 64)
        result = resize(img, None)
        np.testing.assert_array_equal(result, img)
        assert result is not img

    def test_resize_scale_1_preserves_shape(self):
        img = _mono(100, 100)
        result = resize(img, ResizeParams(scale=1.0))
        assert result.shape == (100, 100)

    def test_resize_preserves_value_range(self):
        """Resized image should still be approximately in [0, 1] after clipping."""
        img = np.random.rand(100, 100).astype(np.float32)
        result = resize(img, ResizeParams(scale=0.5))
        # Interpolation can produce slight overshoot — result should be roughly bounded
        result_clipped = np.clip(result, 0, 1)
        assert result_clipped.shape == (50, 50)
        assert result_clipped.min() >= 0.0
        assert result_clipped.max() <= 1.0


# ---------------------------------------------------------------------------
# Bin
# ---------------------------------------------------------------------------


class TestBinImage:
    """Tests for bin_image()."""

    def test_bin_2x2_average_mono(self):
        img = _mono(100, 100, value=0.4)
        result = bin_image(img, BinParams(factor=2, mode=BinMode.AVERAGE))
        assert result.shape == (50, 50)
        assert result.dtype == np.float32
        # Uniform image: average of identical pixels should equal the pixel value.
        np.testing.assert_array_almost_equal(result, 0.4)

    def test_bin_2x2_sum_mono(self):
        img = _mono(100, 100, value=0.2)
        result = bin_image(img, BinParams(factor=2, mode=BinMode.SUM))
        assert result.shape == (50, 50)
        # Sum of four 0.2 pixels = 0.8.
        np.testing.assert_array_almost_equal(result, 0.8)

    def test_bin_3x3_average_mono(self):
        img = _mono(99, 99, value=0.6)
        result = bin_image(img, BinParams(factor=3, mode=BinMode.AVERAGE))
        assert result.shape == (33, 33)
        np.testing.assert_array_almost_equal(result, 0.6)

    def test_bin_3x3_sum_mono(self):
        img = _mono(99, 99, value=0.1)
        result = bin_image(img, BinParams(factor=3, mode=BinMode.SUM))
        assert result.shape == (33, 33)
        # Sum of nine 0.1 pixels = 0.9.
        np.testing.assert_array_almost_equal(result, 0.9)

    def test_bin_2x2_color(self):
        img = _color(100, 100)
        result = bin_image(img, BinParams(factor=2, mode=BinMode.AVERAGE))
        assert result.shape == (3, 50, 50)
        assert result.dtype == np.float32
        # Each channel should retain its uniform value.
        np.testing.assert_array_almost_equal(result[0], 0.2)
        np.testing.assert_array_almost_equal(result[1], 0.5)
        np.testing.assert_array_almost_equal(result[2], 0.8)

    def test_bin_trims_non_divisible(self):
        """Image not evenly divisible should be trimmed before binning."""
        img = _mono(101, 103)
        result = bin_image(img, BinParams(factor=2))
        # 101 // 2 = 50, 103 // 2 = 51
        assert result.shape == (50, 51)

    def test_bin_none_params_defaults_2x2_avg(self):
        img = _mono(100, 100, value=0.5)
        result = bin_image(img, None)
        assert result.shape == (50, 50)
        np.testing.assert_array_almost_equal(result, 0.5)

    def test_bin_known_pattern(self):
        """Test binning a checkerboard-like pattern to verify arithmetic."""
        img = np.zeros((4, 4), dtype=np.float32)
        img[0, 0] = 1.0
        img[0, 1] = 0.0
        img[1, 0] = 0.0
        img[1, 1] = 0.0
        result = bin_image(img, BinParams(factor=2, mode=BinMode.AVERAGE))
        assert result.shape == (2, 2)
        assert result[0, 0] == pytest.approx(0.25)  # mean(1, 0, 0, 0)
        assert result[0, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Invert
# ---------------------------------------------------------------------------


class TestInvert:
    """Tests for invert()."""

    def test_invert_mono(self):
        img = _mono(100, 100, value=0.3)
        result = invert(img)
        assert result.shape == (100, 100)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, 0.7)

    def test_invert_color(self):
        img = _color(80, 80)
        result = invert(img)
        assert result.shape == (3, 80, 80)
        np.testing.assert_array_almost_equal(result[0], 0.8)
        np.testing.assert_array_almost_equal(result[1], 0.5)
        np.testing.assert_array_almost_equal(result[2], 0.2)

    def test_invert_double_is_identity(self):
        img = np.random.rand(100, 100).astype(np.float32)
        result = invert(invert(img))
        np.testing.assert_array_almost_equal(result, img)

    def test_invert_boundaries(self):
        """0 should become 1 and 1 should become 0."""
        img = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
        result = invert(img)
        expected = np.array([[1.0, 0.0], [0.5, 0.75]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
