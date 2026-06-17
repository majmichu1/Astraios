"""Tests for the mask system."""

import numpy as np
import pytest

from astraios.core.masks import (
    Mask,
    MaskType,
    apply_mask,
    binarize_mask,
    blur_mask,
    combine_masks,
    create_luminance_mask,
    create_range_mask,
    grow_mask,
    invert_mask,
    shrink_mask,
)


class TestMask:
    """Tests for the Mask dataclass."""

    def test_create_mask(self):
        data = np.ones((100, 100), dtype=np.float32) * 0.5
        mask = Mask(data=data, name="Test", mask_type=MaskType.MANUAL)
        assert mask.height == 100
        assert mask.width == 100
        assert mask.shape == (100, 100)
        assert mask.name == "Test"

    def test_mask_clips_values(self):
        data = np.array([[1.5, -0.5], [0.5, 2.0]], dtype=np.float32)
        mask = Mask(data=data)
        assert mask.data.max() <= 1.0
        assert mask.data.min() >= 0.0

    def test_to_display(self):
        data = np.ones((50, 60), dtype=np.float32)
        mask = Mask(data=data)
        display = mask.to_display()
        assert display.shape == (50, 60, 3)
        assert display.dtype == np.uint8
        assert display.max() == 255


class TestApplyMask:
    """Tests for apply_mask."""

    def test_no_mask_returns_processed(self):
        original = np.zeros((100, 100), dtype=np.float32)
        processed = np.ones((100, 100), dtype=np.float32)
        result = apply_mask(original, processed, None)
        np.testing.assert_array_equal(result, processed)

    def test_full_mask_returns_processed(self):
        original = np.zeros((100, 100), dtype=np.float32)
        processed = np.ones((100, 100), dtype=np.float32)
        mask = Mask(data=np.ones((100, 100), dtype=np.float32))
        result = apply_mask(original, processed, mask)
        np.testing.assert_array_almost_equal(result, processed)

    def test_zero_mask_returns_original(self):
        original = np.ones((100, 100), dtype=np.float32) * 0.3
        processed = np.ones((100, 100), dtype=np.float32) * 0.9
        mask = Mask(data=np.zeros((100, 100), dtype=np.float32))
        result = apply_mask(original, processed, mask)
        np.testing.assert_array_almost_equal(result, original)

    def test_half_mask_blends(self):
        original = np.zeros((100, 100), dtype=np.float32)
        processed = np.ones((100, 100), dtype=np.float32)
        mask = Mask(data=np.ones((100, 100), dtype=np.float32) * 0.5)
        result = apply_mask(original, processed, mask)
        np.testing.assert_array_almost_equal(result, 0.5)

    def test_mask_with_color_image(self):
        original = np.zeros((3, 100, 100), dtype=np.float32)
        processed = np.ones((3, 100, 100), dtype=np.float32)
        mask_data = np.ones((100, 100), dtype=np.float32) * 0.5
        mask = Mask(data=mask_data)
        result = apply_mask(original, processed, mask)
        assert result.shape == (3, 100, 100)
        np.testing.assert_array_almost_equal(result, 0.5)


class TestCreateLuminanceMask:
    """Tests for create_luminance_mask."""

    def test_full_range(self):
        image = np.random.rand(100, 100).astype(np.float32)
        mask = create_luminance_mask(image, low=0.0, high=1.0)
        assert mask.mask_type == MaskType.LUMINANCE
        assert mask.data.shape == (100, 100)
        # Full range should select everything
        assert mask.data.mean() > 0.9

    def test_narrow_range(self):
        # Create an image with known distribution
        image = np.linspace(0, 1, 10000).reshape(100, 100).astype(np.float32)
        mask = create_luminance_mask(image, low=0.4, high=0.6)
        # Middle 20% should be selected, edges not
        center_row = mask.data[45:55, :]
        assert center_row.mean() > 0.5
        edge_top = mask.data[:10, :]
        assert edge_top.mean() < 0.3

    def test_color_image(self):
        image = np.random.rand(3, 100, 100).astype(np.float32)
        mask = create_luminance_mask(image, low=0.2, high=0.8)
        assert mask.data.shape == (100, 100)


class TestCreateRangeMask:
    """Tests for create_range_mask."""

    def test_mono_image(self):
        image = np.linspace(0, 1, 10000).reshape(100, 100).astype(np.float32)
        mask = create_range_mask(image, low=0.5, high=1.0)
        assert mask.mask_type == MaskType.RANGE
        # Bottom half should be 0, top half should be 1
        assert mask.data[:45, :].sum() < mask.data[55:, :].sum()

    def test_specific_channel(self):
        image = np.zeros((3, 100, 100), dtype=np.float32)
        image[0] = 0.8  # Red channel bright
        image[1] = 0.2  # Green channel dark
        mask = create_range_mask(image, channel=0, low=0.5, high=1.0)
        assert mask.data.mean() == 1.0  # All red pixels are bright

        mask = create_range_mask(image, channel=1, low=0.5, high=1.0)
        assert mask.data.mean() == 0.0  # All green pixels are dark


class TestInvertMask:
    """Tests for invert_mask."""

    def test_invert(self):
        data = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
        mask = Mask(data=data, name="Original")
        inverted = invert_mask(mask)
        np.testing.assert_array_almost_equal(inverted.data, 1.0 - data)
        assert "Inverted" in inverted.name

    def test_double_invert_restores(self):
        data = np.random.rand(50, 50).astype(np.float32)
        mask = Mask(data=data)
        double_inv = invert_mask(invert_mask(mask))
        np.testing.assert_array_almost_equal(double_inv.data, data)


class TestCombineMasks:
    """Tests for combine_masks."""

    def test_multiply(self):
        m1 = Mask(data=np.ones((50, 50), dtype=np.float32) * 0.5)
        m2 = Mask(data=np.ones((50, 50), dtype=np.float32) * 0.5)
        result = combine_masks([m1, m2], mode="multiply")
        np.testing.assert_array_almost_equal(result.data, 0.25)

    def test_max(self):
        m1 = Mask(data=np.ones((50, 50), dtype=np.float32) * 0.3)
        m2 = Mask(data=np.ones((50, 50), dtype=np.float32) * 0.7)
        result = combine_masks([m1, m2], mode="max")
        np.testing.assert_array_almost_equal(result.data, 0.7)

    def test_min(self):
        m1 = Mask(data=np.ones((50, 50), dtype=np.float32) * 0.3)
        m2 = Mask(data=np.ones((50, 50), dtype=np.float32) * 0.7)
        result = combine_masks([m1, m2], mode="min")
        np.testing.assert_array_almost_equal(result.data, 0.3)

    def test_screen(self):
        m1 = Mask(data=np.ones((50, 50), dtype=np.float32) * 0.5)
        m2 = Mask(data=np.ones((50, 50), dtype=np.float32) * 0.5)
        result = combine_masks([m1, m2], mode="screen")
        # screen: 1 - (1-0.5)*(1-0.5) = 0.75
        np.testing.assert_array_almost_equal(result.data, 0.75)

    def test_single_mask(self):
        m = Mask(data=np.ones((50, 50), dtype=np.float32) * 0.42)
        result = combine_masks([m])
        np.testing.assert_array_almost_equal(result.data, 0.42)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            combine_masks([])


class TestBlurMask:
    """Tests for blur_mask."""

    def test_blur_smooths_edges(self):
        data = np.zeros((100, 100), dtype=np.float32)
        data[40:60, 40:60] = 1.0
        mask = Mask(data=data)
        blurred = blur_mask(mask, radius=5.0)
        # Edges should now have intermediate values
        assert 0.0 < blurred.data[40, 39] < 1.0
        # Center should still be high
        assert blurred.data[50, 50] > 0.9

    def test_zero_blur_unchanged(self):
        data = np.random.rand(50, 50).astype(np.float32)
        mask = Mask(data=data)
        blurred = blur_mask(mask, radius=0.0)
        np.testing.assert_array_almost_equal(blurred.data, data)


class TestBinarizeMask:
    """Tests for binarize_mask."""

    def test_binarize(self):
        data = np.array([[0.3, 0.6], [0.5, 0.8]], dtype=np.float32)
        mask = Mask(data=data)
        binary = binarize_mask(mask, threshold=0.5)
        expected = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        np.testing.assert_array_equal(binary.data, expected)


class TestGrowShrinkMask:
    """Tests for grow_mask and shrink_mask."""

    def test_grow_expands(self):
        data = np.zeros((100, 100), dtype=np.float32)
        data[50, 50] = 1.0
        mask = Mask(data=data)
        grown = grow_mask(mask, pixels=3)
        # The grown mask should have more non-zero pixels
        assert grown.data.sum() > data.sum()

    def test_shrink_contracts(self):
        data = np.zeros((100, 100), dtype=np.float32)
        data[40:60, 40:60] = 1.0
        mask = Mask(data=data)
        shrunk = shrink_mask(mask, pixels=3)
        assert shrunk.data.sum() < data.sum()
        # Center should still be 1
        assert shrunk.data[50, 50] == 1.0

    def test_zero_pixels_unchanged(self):
        data = np.random.rand(50, 50).astype(np.float32)
        mask = Mask(data=data)
        grown = grow_mask(mask, pixels=0)
        np.testing.assert_array_almost_equal(grown.data, data)
