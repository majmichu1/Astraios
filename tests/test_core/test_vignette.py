"""Tests for vignette correction."""

import numpy as np
import pytest

from astraios.core.vignette import VignetteParams, correct_vignette
from astraios.core.masks import Mask


class TestVignetteParams:
    """Tests for VignetteParams defaults and construction."""

    def test_defaults(self):
        """Default params should have standard values."""
        p = VignetteParams()
        assert p.strength == 1.0
        assert p.center_x == 0.5
        assert p.center_y == 0.5
        assert p.radius == 1.0
        assert p.falloff == 2.0

    def test_custom_params(self):
        """Should accept custom parameter values."""
        p = VignetteParams(strength=0.5, center_x=0.3, center_y=0.7,
                           radius=0.8, falloff=3.0)
        assert p.strength == 0.5
        assert p.center_x == 0.3
        assert p.center_y == 0.7
        assert p.radius == 0.8
        assert p.falloff == 3.0


class TestCorrectVignetteMono:
    """Tests for vignette correction on mono (H, W) images."""

    def test_output_shape_matches_input(self):
        """Output should have the same shape as the input."""
        image = np.full((64, 64), 0.5, dtype=np.float32)
        result = correct_vignette(image, VignetteParams(strength=1.0))
        assert result.shape == image.shape
        assert result.dtype == np.float32

    def test_output_clipped_to_unit_range(self):
        """All output values should be in [0, 1]."""
        image = np.full((64, 64), 0.8, dtype=np.float32)
        result = correct_vignette(image, VignetteParams(strength=2.0))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_corners_brighter_after_correction(self):
        """Vignette correction should brighten corner pixels relative to center."""
        # Simulate a vignetted image: darker corners, brighter center
        h, w = 64, 64
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        # Vignetted image: center=0.8, corners dimmed
        image = (0.8 * (1.0 - 0.5 * (dist / max_dist) ** 2)).astype(np.float32)

        result = correct_vignette(image, VignetteParams(strength=1.0, falloff=2.0))

        # Corner pixel (0,0) should be brighter after correction
        assert result[0, 0] > image[0, 0]
        # Center pixel should be approximately the same (correction ~1 at center)
        center_y, center_x = h // 2, w // 2
        assert result[center_y, center_x] == pytest.approx(
            image[center_y, center_x], abs=0.05
        )

    def test_strength_zero_returns_copy(self):
        """Strength=0 should return an unchanged copy."""
        image = np.random.RandomState(42).rand(32, 32).astype(np.float32)
        result = correct_vignette(image, VignetteParams(strength=0.0))
        np.testing.assert_array_equal(result, image)
        # Verify it is a copy, not the same object
        assert result is not image

    def test_none_params_uses_defaults(self):
        """Passing params=None should behave like default VignetteParams."""
        image = np.full((32, 32), 0.5, dtype=np.float32)
        result_none = correct_vignette(image, params=None)
        result_default = correct_vignette(image, VignetteParams())
        np.testing.assert_array_almost_equal(result_none, result_default)

    def test_center_pixel_least_affected(self):
        """The center pixel should receive the smallest correction."""
        image = np.full((64, 64), 0.5, dtype=np.float32)
        result = correct_vignette(image, VignetteParams(strength=1.0))

        # At the center, correction factor is 1 + 0 = 1, so value unchanged
        cy, cx = 32, 32
        assert result[cy, cx] == pytest.approx(0.5, abs=0.01)

    def test_higher_falloff_steeper_edges(self):
        """Higher falloff should produce a steeper correction gradient at edges."""
        image = np.full((64, 64), 0.5, dtype=np.float32)

        result_low = correct_vignette(
            image, VignetteParams(strength=1.0, falloff=1.0)
        )
        result_high = correct_vignette(
            image, VignetteParams(strength=1.0, falloff=4.0)
        )

        # At center both should be similar (1.0 correction)
        # At edges, falloff=1 gives more correction at moderate distances
        # because d^1 > d^4 for d < 1. So at mid-distance, low falloff
        # corrects MORE than high falloff.
        mid_y, mid_x = 0, 32  # top-center edge (moderate distance)
        assert result_low[mid_y, mid_x] > result_high[mid_y, mid_x]

    def test_different_falloff_values(self):
        """Correction with falloff=2 and falloff=3 should produce different results."""
        image = np.full((64, 64), 0.6, dtype=np.float32)
        result_a = correct_vignette(image, VignetteParams(falloff=2.0))
        result_b = correct_vignette(image, VignetteParams(falloff=3.0))
        # The two results should not be identical
        assert not np.allclose(result_a, result_b)


class TestCorrectVignetteColor:
    """Tests for vignette correction on color (C, H, W) images."""

    def test_output_shape_matches_color_input(self):
        """Output shape should match (C, H, W) input."""
        image = np.full((3, 64, 64), 0.5, dtype=np.float32)
        result = correct_vignette(image, VignetteParams(strength=1.0))
        assert result.shape == (3, 64, 64)
        assert result.dtype == np.float32

    def test_corners_brighter_all_channels(self):
        """All color channels should have brighter corners after correction."""
        h, w = 64, 64
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        plane = (0.7 * (1.0 - 0.4 * (dist / max_dist) ** 2)).astype(np.float32)
        image = np.stack([plane, plane * 0.9, plane * 0.8], axis=0)

        result = correct_vignette(image, VignetteParams(strength=1.0))

        for ch in range(3):
            assert result[ch, 0, 0] > image[ch, 0, 0], f"Channel {ch} corner not brighter"

    def test_same_correction_across_channels(self):
        """All channels should get the same correction factor (uniform input)."""
        image = np.full((3, 32, 32), 0.4, dtype=np.float32)
        result = correct_vignette(image, VignetteParams(strength=1.0))
        # All channels started equal, so they should end equal
        np.testing.assert_array_almost_equal(result[0], result[1])
        np.testing.assert_array_almost_equal(result[1], result[2])


class TestCorrectVignetteMask:
    """Tests for mask support in vignette correction."""

    def test_mask_limits_correction(self):
        """Mask=0 areas should remain unchanged."""
        image = np.full((64, 64), 0.5, dtype=np.float32)
        mask_data = np.zeros((64, 64), dtype=np.float32)
        mask_data[32:, :] = 1.0  # only bottom half corrected
        mask = Mask(data=mask_data)

        result = correct_vignette(image, VignetteParams(strength=1.5), mask=mask)

        # Top-left corner: mask=0 => must equal original
        assert result[0, 0] == pytest.approx(image[0, 0], abs=1e-6)
        # Bottom-left corner: mask=1 => correction applied (divided by >1 => lower)
        assert result[63, 0] != pytest.approx(image[63, 0], abs=1e-3)

    def test_full_mask_same_as_no_mask(self):
        """A mask of all 1.0 should produce the same result as no mask."""
        image = np.full((32, 32), 0.6, dtype=np.float32)
        mask_data = np.ones((32, 32), dtype=np.float32)
        mask = Mask(data=mask_data)

        result_mask = correct_vignette(image, VignetteParams(strength=1.0), mask=mask)
        result_none = correct_vignette(image, VignetteParams(strength=1.0), mask=None)

        np.testing.assert_array_almost_equal(result_mask, result_none)


class TestCorrectVignetteEdgeCases:
    """Edge case tests for vignette correction."""

    def test_invalid_ndim_raises(self):
        """Should raise ValueError for non-2D/3D input."""
        image = np.zeros((2, 3, 4, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="Unexpected image shape"):
            correct_vignette(image)

    def test_off_center_vignette(self):
        """Off-center correction should shift the brightness pattern."""
        image = np.full((64, 64), 0.5, dtype=np.float32)
        # Center shifted to top-left quadrant
        params = VignetteParams(strength=1.0, center_x=0.25, center_y=0.25)
        result = correct_vignette(image, params)

        # The vignette center is at (0.25, 0.25) — top-left area
        # Top-left should be closer to vignette center → less correction applied
        # Bottom-right should be farther → more correction (brighter result)
        # So bottom-right should be brighter than top-left
        assert result[63, 63] > result[0, 0]

    def test_large_strength(self):
        """High strength should still produce valid output in [0, 1]."""
        image = np.full((32, 32), 0.3, dtype=np.float32)
        result = correct_vignette(image, VignetteParams(strength=2.0))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_small_image(self):
        """Should work on very small images."""
        image = np.full((4, 4), 0.5, dtype=np.float32)
        result = correct_vignette(image, VignetteParams(strength=1.0))
        assert result.shape == (4, 4)
        assert result.dtype == np.float32
