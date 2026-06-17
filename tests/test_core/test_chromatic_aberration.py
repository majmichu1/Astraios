"""Tests for chromatic aberration correction."""

import numpy as np
import pytest

from astraios.core.chromatic_aberration import CAParams, correct_chromatic_aberration
from astraios.core.masks import Mask


class TestCAParams:
    """Tests for CAParams defaults and construction."""

    def test_defaults(self):
        """Default params should have auto_detect=True and zero shifts."""
        p = CAParams()
        assert p.auto_detect is True
        assert p.red_shift_x == 0.0
        assert p.red_shift_y == 0.0
        assert p.blue_shift_x == 0.0
        assert p.blue_shift_y == 0.0
        assert p.max_correction == 3.0

    def test_custom_params(self):
        """Should accept custom parameter values."""
        p = CAParams(
            auto_detect=False,
            red_shift_x=1.5,
            red_shift_y=-0.3,
            blue_shift_x=-1.0,
            blue_shift_y=0.7,
            max_correction=5.0,
        )
        assert p.auto_detect is False
        assert p.red_shift_x == 1.5
        assert p.blue_shift_y == 0.7
        assert p.max_correction == 5.0


class TestCorrectChromaticAberrationRequiresColor:
    """Tests that CA correction enforces color image requirements."""

    def test_mono_image_raises(self):
        """Mono (H, W) image should raise ValueError."""
        image = np.full((64, 64), 0.5, dtype=np.float32)
        with pytest.raises(ValueError, match="colour image"):
            correct_chromatic_aberration(image)

    def test_two_channel_raises(self):
        """Image with fewer than 3 channels should raise ValueError."""
        image = np.full((2, 64, 64), 0.5, dtype=np.float32)
        with pytest.raises(ValueError, match="C >= 3"):
            correct_chromatic_aberration(image)

    def test_three_channels_accepted(self):
        """3-channel image should be accepted without error."""
        image = np.full((3, 64, 64), 0.5, dtype=np.float32)
        result = correct_chromatic_aberration(image)
        assert result.shape == (3, 64, 64)

    def test_four_channels_accepted(self):
        """4-channel image (e.g. LRGB) should also be accepted."""
        image = np.full((4, 64, 64), 0.5, dtype=np.float32)
        result = correct_chromatic_aberration(image)
        assert result.shape == (4, 64, 64)


class TestManualShifts:
    """Tests for manual CA correction (auto_detect=False)."""

    def test_known_red_shift_corrected(self):
        """A known red channel shift should be corrected by the function."""
        h, w = 64, 64
        # Create a color image with a bright vertical stripe in the center
        image = np.full((3, h, w), 0.1, dtype=np.float32)
        # Green channel: stripe at column 32
        image[1, :, 30:34] = 0.9
        # Red channel: same stripe shifted right by 2 pixels
        image[0, :, 32:36] = 0.9
        # Blue channel: same as green
        image[2, :, 30:34] = 0.9

        params = CAParams(
            auto_detect=False,
            red_shift_x=2.0,  # red is shifted +2 in x
            red_shift_y=0.0,
            blue_shift_x=0.0,
            blue_shift_y=0.0,
        )
        result = correct_chromatic_aberration(image, params)

        # After correction, the red channel stripe should be realigned closer
        # to the green channel stripe (centered around col 30-33).
        # The red channel at the original green position should now be bright.
        assert result[0, 32, 31] > image[0, 32, 31], (
            "Red channel should be brighter at green stripe position after correction"
        )

    def test_known_blue_shift_corrected(self):
        """A known blue channel shift should be corrected."""
        h, w = 64, 64
        image = np.full((3, h, w), 0.1, dtype=np.float32)
        # Green and red: stripe at column 32
        image[0, :, 30:34] = 0.9
        image[1, :, 30:34] = 0.9
        # Blue channel: shifted left by 1.5 pixels
        image[2, :, 28:32] = 0.9

        params = CAParams(
            auto_detect=False,
            red_shift_x=0.0,
            red_shift_y=0.0,
            blue_shift_x=-1.5,
            blue_shift_y=0.0,
        )
        result = correct_chromatic_aberration(image, params)

        # Blue should now be shifted back toward the green position
        assert result[2, 32, 32] > image[2, 32, 32], (
            "Blue channel should be brighter at green stripe position after correction"
        )

    def test_negligible_shifts_return_copy(self):
        """When all shifts are below 0.01, should return a copy unchanged."""
        image = np.random.RandomState(42).rand(3, 32, 32).astype(np.float32)
        params = CAParams(
            auto_detect=False,
            red_shift_x=0.005,
            red_shift_y=0.001,
            blue_shift_x=-0.003,
            blue_shift_y=0.002,
        )
        result = correct_chromatic_aberration(image, params)
        np.testing.assert_array_equal(result, image)
        assert result is not image

    def test_zero_shifts_return_copy(self):
        """Explicitly zero shifts should return an unchanged copy."""
        image = np.random.RandomState(7).rand(3, 32, 32).astype(np.float32)
        params = CAParams(
            auto_detect=False,
            red_shift_x=0.0,
            red_shift_y=0.0,
            blue_shift_x=0.0,
            blue_shift_y=0.0,
        )
        result = correct_chromatic_aberration(image, params)
        np.testing.assert_array_equal(result, image)

    def test_green_channel_unchanged(self):
        """Green channel (reference) should never be shifted."""
        image = np.random.RandomState(99).rand(3, 32, 32).astype(np.float32)
        params = CAParams(
            auto_detect=False,
            red_shift_x=1.5,
            red_shift_y=0.5,
            blue_shift_x=-1.0,
            blue_shift_y=-0.5,
        )
        result = correct_chromatic_aberration(image, params)
        # Green channel should be identical to the original
        np.testing.assert_array_equal(result[1], image[1])

    def test_shifts_capped_at_max_correction(self):
        """Shifts exceeding max_correction should be clamped."""
        image = np.full((3, 32, 32), 0.5, dtype=np.float32)
        # Add a distinct feature to detect shift magnitude
        image[:, 15:17, 15:17] = 0.9

        params_big = CAParams(
            auto_detect=False,
            red_shift_x=10.0,
            max_correction=2.0,
        )
        params_at_cap = CAParams(
            auto_detect=False,
            red_shift_x=2.0,
            max_correction=2.0,
        )
        result_big = correct_chromatic_aberration(image, params_big)
        result_cap = correct_chromatic_aberration(image, params_at_cap)

        # Both should produce the same result since 10.0 is clamped to 2.0
        np.testing.assert_array_almost_equal(result_big, result_cap)


class TestAutoDetect:
    """Tests for auto-detection mode."""

    def test_auto_detect_no_stars_returns_copy(self):
        """Uniform image with no stars should have negligible shifts => copy."""
        image = np.full((3, 64, 64), 0.3, dtype=np.float32)
        params = CAParams(auto_detect=True)
        result = correct_chromatic_aberration(image, params)
        # No stars to detect means zero offsets => negligible => unchanged copy
        np.testing.assert_array_equal(result, image)

    def test_auto_detect_aligned_image_returns_copy(self):
        """Image with identical channel content should have zero shifts."""
        rng = np.random.RandomState(123)
        plane = rng.rand(64, 64).astype(np.float32) * 0.3 + 0.1
        image = np.stack([plane, plane, plane], axis=0)
        params = CAParams(auto_detect=True)
        result = correct_chromatic_aberration(image, params)
        # Identical channels => negligible offsets => copy
        np.testing.assert_array_equal(result, image)


class TestOutputProperties:
    """Tests for general output properties."""

    def test_output_dtype_float32(self):
        """Output should always be float32."""
        image = np.random.RandomState(1).rand(3, 32, 32).astype(np.float32)
        params = CAParams(auto_detect=False, red_shift_x=1.0)
        result = correct_chromatic_aberration(image, params)
        assert result.dtype == np.float32

    def test_output_clipped_to_unit_range(self):
        """All output values should be in [0, 1]."""
        image = np.random.RandomState(2).rand(3, 32, 32).astype(np.float32)
        params = CAParams(auto_detect=False, red_shift_x=2.0, blue_shift_x=-2.0)
        result = correct_chromatic_aberration(image, params)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_shape_preserved(self):
        """Output shape should match input shape."""
        image = np.random.RandomState(3).rand(3, 48, 64).astype(np.float32)
        params = CAParams(auto_detect=False, red_shift_x=1.0)
        result = correct_chromatic_aberration(image, params)
        assert result.shape == image.shape


class TestChromaticAberrationMask:
    """Tests for mask support in CA correction."""

    def test_mask_limits_correction(self):
        """Mask=0 areas should remain unchanged from the original."""
        image = np.random.RandomState(10).rand(3, 64, 64).astype(np.float32)
        mask_data = np.zeros((64, 64), dtype=np.float32)
        mask_data[32:, :] = 1.0  # only bottom half corrected
        mask = Mask(data=mask_data)

        params = CAParams(auto_detect=False, red_shift_x=2.0)
        result = correct_chromatic_aberration(image, params, mask=mask)

        # Top half (mask=0) should be identical to original
        np.testing.assert_array_almost_equal(result[:, :32, :], image[:, :32, :])

    def test_full_mask_same_as_no_mask(self):
        """A mask of all 1.0 should produce the same result as no mask."""
        image = np.random.RandomState(11).rand(3, 32, 32).astype(np.float32)
        mask_data = np.ones((32, 32), dtype=np.float32)
        mask = Mask(data=mask_data)

        params = CAParams(auto_detect=False, red_shift_x=1.5, blue_shift_y=-0.5)
        result_mask = correct_chromatic_aberration(image, params, mask=mask)
        result_none = correct_chromatic_aberration(image, params, mask=None)

        np.testing.assert_array_almost_equal(result_mask, result_none)

    def test_none_params_uses_defaults(self):
        """Passing params=None should behave like default CAParams."""
        image = np.full((3, 32, 32), 0.4, dtype=np.float32)
        result = correct_chromatic_aberration(image, params=None)
        # Default is auto_detect=True on uniform image => negligible => copy
        np.testing.assert_array_equal(result, image)
