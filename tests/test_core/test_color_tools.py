"""Tests for color tools (SCNR and color adjustment)."""

import pytest
import numpy as np

from astraios.core.color_tools import (
    ColorAdjustParams,
    SCNRMethod,
    SCNRParams,
    color_adjust,
    scnr,
    _rgb_to_hsv,
    _hsv_to_rgb,
)
from astraios.core.masks import Mask


class TestSCNR:
    """Tests for Subtractive Chromatic Noise Reduction."""

    def test_reduces_green_excess(self):
        """SCNR should reduce green where it exceeds neutral."""
        image = np.zeros((3, 100, 100), dtype=np.float32)
        image[0] = 0.3  # R
        image[1] = 0.7  # G (excess green)
        image[2] = 0.3  # B
        result = scnr(image, SCNRParams(amount=1.0))
        # Green should be reduced
        assert result[1].mean() < 0.7

    def test_no_change_balanced(self):
        """SCNR should not change already balanced colors."""
        image = np.ones((3, 100, 100), dtype=np.float32) * 0.5
        result = scnr(image, SCNRParams(amount=1.0))
        np.testing.assert_array_almost_equal(result, image, decimal=3)

    def test_maximum_neutral_method(self):
        """Maximum neutral should use max(R, B) as reference."""
        image = np.zeros((3, 100, 100), dtype=np.float32)
        image[0] = 0.5  # R
        image[1] = 0.8  # G
        image[2] = 0.3  # B
        params = SCNRParams(method=SCNRMethod.MAXIMUM_NEUTRAL, amount=1.0)
        result = scnr(image, params)
        # Green should be brought down toward max(R, B) = 0.5
        assert result[1].mean() < image[1].mean()

    def test_amount_controls_strength(self):
        """Lower amount should produce less correction."""
        image = np.zeros((3, 100, 100), dtype=np.float32)
        image[0] = 0.3
        image[1] = 0.8
        image[2] = 0.3
        r_full = scnr(image, SCNRParams(amount=1.0))
        r_half = scnr(image, SCNRParams(amount=0.5))
        # Half amount should change less
        diff_full = abs(r_full[1].mean() - image[1].mean())
        diff_half = abs(r_half[1].mean() - image[1].mean())
        assert diff_half < diff_full

    def test_mono_image_raises(self):
        """Mono images should raise ValueError — SCNR is color-only."""
        image = np.ones((100, 100), dtype=np.float32) * 0.5
        with pytest.raises(ValueError, match="color image"):
            scnr(image)

    def test_mask_support(self):
        """Mask should restrict SCNR to masked area."""
        image = np.zeros((3, 100, 100), dtype=np.float32)
        image[0] = 0.3
        image[1] = 0.8
        image[2] = 0.3

        mask_data = np.zeros((100, 100), dtype=np.float32)
        mask_data[50:, :] = 1.0
        mask = Mask(data=mask_data)

        result = scnr(image, SCNRParams(amount=1.0), mask=mask)
        # Top half should be unchanged
        np.testing.assert_allclose(result[1, :50, :].mean(), 0.8, atol=0.01)
        # Bottom half should have reduced green
        assert result[1, 50:, :].mean() < 0.7


class TestRGBHSVConversion:
    """Tests for RGB <-> HSV conversion."""

    def test_roundtrip(self):
        """RGB -> HSV -> RGB should be identity."""
        image = np.random.rand(3, 50, 50).astype(np.float32)
        hsv = _rgb_to_hsv(image)
        rgb = _hsv_to_rgb(hsv)
        np.testing.assert_array_almost_equal(rgb, image, decimal=4)

    def test_pure_red(self):
        """Pure red should have H=0, S=1, V=1."""
        image = np.zeros((3, 10, 10), dtype=np.float32)
        image[0] = 1.0  # Red = 1
        hsv = _rgb_to_hsv(image)
        np.testing.assert_allclose(hsv[0].mean(), 0.0, atol=1)  # H near 0
        np.testing.assert_allclose(hsv[1].mean(), 1.0, atol=0.01)  # S = 1
        np.testing.assert_allclose(hsv[2].mean(), 1.0, atol=0.01)  # V = 1

    def test_gray(self):
        """Gray should have S=0."""
        image = np.ones((3, 10, 10), dtype=np.float32) * 0.5
        hsv = _rgb_to_hsv(image)
        np.testing.assert_allclose(hsv[1].mean(), 0.0, atol=0.01)


class TestColorAdjust:
    """Tests for color adjustment."""

    def test_identity(self):
        """Default params should be near identity."""
        image = np.random.rand(3, 50, 50).astype(np.float32)
        result = color_adjust(image)
        np.testing.assert_array_almost_equal(result, image, decimal=3)

    def test_desaturate(self):
        """Saturation < 1 should desaturate."""
        image = np.zeros((3, 50, 50), dtype=np.float32)
        image[0] = 1.0  # pure red
        params = ColorAdjustParams(saturation=0.0)
        result = color_adjust(image, params)
        # Should be gray
        assert abs(result[0].mean() - result[1].mean()) < 0.05

    def test_saturate(self):
        """Saturation > 1 should increase saturation."""
        image = np.zeros((3, 50, 50), dtype=np.float32)
        image[0] = 0.8
        image[1] = 0.4
        image[2] = 0.4
        params = ColorAdjustParams(saturation=2.0)
        result = color_adjust(image, params)
        # Red channel should be more different from G/B
        orig_diff = abs(image[0].mean() - image[1].mean())
        new_diff = abs(result[0].mean() - result[1].mean())
        assert new_diff > orig_diff

    def test_hue_shift(self):
        """Hue shift should change the dominant color."""
        image = np.zeros((3, 50, 50), dtype=np.float32)
        image[0] = 1.0  # pure red
        params = ColorAdjustParams(hue_shift=120)  # shift to green
        result = color_adjust(image, params)
        # Green should now be dominant
        assert result[1].mean() > result[0].mean()

    def test_vibrance(self):
        """Vibrance should boost desaturated colors more."""
        image = np.zeros((3, 50, 50), dtype=np.float32)
        image[0] = 0.5
        image[1] = 0.45
        image[2] = 0.5
        params = ColorAdjustParams(vibrance=0.5)
        result = color_adjust(image, params)
        # Result should be more saturated than input
        orig_hsv = _rgb_to_hsv(image)
        result_hsv = _rgb_to_hsv(result)
        assert result_hsv[1].mean() > orig_hsv[1].mean()

    def test_mono_unchanged(self):
        """Mono images should pass through unchanged."""
        image = np.ones((100, 100), dtype=np.float32) * 0.5
        result = color_adjust(image)
        np.testing.assert_array_equal(result, image)
