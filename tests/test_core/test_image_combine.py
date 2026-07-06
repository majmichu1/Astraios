"""Tests for pixel-arithmetic image combine (ported from SASpro image_combine.py)."""

import numpy as np
import pytest

from astraios.core.image_combine import (
    CombineOperation,
    ImageCombineParams,
    combine_images,
)


def _mono(val, shape=(24, 32)):
    return np.full(shape, val, dtype=np.float32)


def _color(val, shape=(3, 24, 32)):
    return np.full(shape, val, dtype=np.float32)


class TestArithmeticOpsMono:
    def test_add(self):
        params = ImageCombineParams(operation=CombineOperation.ADD)
        out = combine_images(_mono(0.3), _mono(0.2), params)
        np.testing.assert_allclose(out, _mono(0.5), atol=1e-5)

    def test_subtract(self):
        out = combine_images(
            _mono(0.7), _mono(0.2), ImageCombineParams(operation=CombineOperation.SUBTRACT)
        )
        np.testing.assert_allclose(out, _mono(0.5), atol=1e-5)

    def test_average_is_mean(self):
        a, b = _mono(0.2), _mono(0.8)
        out = combine_images(a, b, ImageCombineParams(operation=CombineOperation.AVERAGE))
        np.testing.assert_allclose(out, _mono(0.5), atol=1e-5)

    def test_multiply(self):
        out = combine_images(
            _mono(0.5), _mono(0.4), ImageCombineParams(operation=CombineOperation.MULTIPLY)
        )
        np.testing.assert_allclose(out, _mono(0.2), atol=1e-5)

    def test_divide(self):
        out = combine_images(
            _mono(0.4), _mono(0.5), ImageCombineParams(operation=CombineOperation.DIVIDE)
        )
        np.testing.assert_allclose(out, _mono(0.8), atol=1e-3)

    def test_min_max(self):
        a, b = _mono(0.3), _mono(0.7)
        out_min = combine_images(a, b, ImageCombineParams(operation=CombineOperation.MIN))
        out_max = combine_images(a, b, ImageCombineParams(operation=CombineOperation.MAX))
        np.testing.assert_allclose(out_min, _mono(0.3), atol=1e-5)
        np.testing.assert_allclose(out_max, _mono(0.7), atol=1e-5)

    def test_difference(self):
        out = combine_images(
            _mono(0.9), _mono(0.35), ImageCombineParams(operation=CombineOperation.DIFFERENCE)
        )
        np.testing.assert_allclose(out, _mono(0.55), atol=1e-5)

    def test_screen_brightens(self):
        a, b = _mono(0.4), _mono(0.5)
        out = combine_images(a, b, ImageCombineParams(operation=CombineOperation.SCREEN))
        expected = 1 - (1 - 0.4) * (1 - 0.5)
        np.testing.assert_allclose(out, _mono(expected), atol=1e-5)
        assert out.mean() > a.mean()

    def test_overlay(self):
        # A <= 0.5 branch: 2*A*B
        out_low = combine_images(
            _mono(0.3), _mono(0.4), ImageCombineParams(operation=CombineOperation.OVERLAY)
        )
        np.testing.assert_allclose(out_low, _mono(2 * 0.3 * 0.4), atol=1e-5)
        # A > 0.5 branch: 1-2*(1-A)*(1-B)
        out_high = combine_images(
            _mono(0.6), _mono(0.4), ImageCombineParams(operation=CombineOperation.OVERLAY)
        )
        np.testing.assert_allclose(out_high, _mono(1 - 2 * 0.4 * 0.6), atol=1e-5)

    def test_blend_crossfade(self):
        a, b = _mono(0.0), _mono(1.0)
        out = combine_images(
            a, b, ImageCombineParams(operation=CombineOperation.BLEND, weight_a=0.75, weight_b=0.25)
        )
        np.testing.assert_allclose(out, _mono(0.25), atol=1e-5)


class TestColor:
    def test_add_color(self):
        params = ImageCombineParams(operation=CombineOperation.ADD)
        out = combine_images(_color(0.3), _color(0.2), params)
        assert out.shape == (3, 24, 32)
        np.testing.assert_allclose(out, _color(0.5), atol=1e-5)

    def test_mono_broadcasts_against_color(self):
        a = _color(0.4)
        b = _mono(0.5)  # (H, W)
        out = combine_images(a, b, ImageCombineParams(operation=CombineOperation.MULTIPLY))
        assert out.shape == (3, 24, 32)
        np.testing.assert_allclose(out, _color(0.2), atol=1e-5)

    def test_color_broadcasts_against_mono_base(self):
        a = _mono(0.5)  # (H, W)
        b = _color(0.4)
        out = combine_images(a, b, ImageCombineParams(operation=CombineOperation.MULTIPLY))
        assert out.shape == (3, 24, 32)
        np.testing.assert_allclose(out, _color(0.2), atol=1e-5)

    def test_both_mono_stays_mono(self):
        params = ImageCombineParams(operation=CombineOperation.ADD)
        out = combine_images(_mono(0.3), _mono(0.4), params)
        assert out.ndim == 2


class TestWeights:
    def test_weighted_average(self):
        a, b = _mono(1.0), _mono(0.0)
        out = combine_images(
            a, b, ImageCombineParams(operation=CombineOperation.AVERAGE, weight_a=3.0, weight_b=1.0)
        )
        # (3*1.0 + 1*0.0) / (3+1) = 0.75
        np.testing.assert_allclose(out, _mono(0.75), atol=1e-5)

    def test_weighted_add(self):
        a, b = _mono(0.2), _mono(0.3)
        out = combine_images(
            a, b, ImageCombineParams(operation=CombineOperation.ADD, weight_a=2.0, weight_b=1.0)
        )
        # clipped: min(1.0, 2*0.2 + 1*0.3) = min(1.0, 0.7) = 0.7
        np.testing.assert_allclose(out, _mono(0.7), atol=1e-5)


class TestClipAndRescale:
    def test_clip_default_clamps_to_unit_range(self):
        params = ImageCombineParams(operation=CombineOperation.ADD)
        out = combine_images(_mono(0.9), _mono(0.9), params)
        assert out.max() <= 1.0 and out.min() >= 0.0
        np.testing.assert_allclose(out, _mono(1.0), atol=1e-5)

    def test_clip_false_leaves_raw_values(self):
        out = combine_images(
            _mono(0.9), _mono(0.9), ImageCombineParams(operation=CombineOperation.ADD, clip=False)
        )
        np.testing.assert_allclose(out, _mono(1.8), atol=1e-5)

    def test_rescale_normalizes_to_unit_range(self):
        a = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
        b = np.zeros_like(a)
        out = combine_images(
            a, b, ImageCombineParams(operation=CombineOperation.ADD, weight_a=2.0, rescale=True)
        )
        assert np.isclose(out.min(), 0.0, atol=1e-5)
        assert np.isclose(out.max(), 1.0, atol=1e-5)


class TestShapeErrors:
    def test_spatial_size_mismatch_raises(self):
        with pytest.raises(ValueError):
            combine_images(_mono(0.5, (20, 20)), _mono(0.5, (30, 30)))

    def test_incompatible_channel_counts_raise(self):
        a = np.full((2, 10, 10), 0.5, dtype=np.float32)
        b = np.full((3, 10, 10), 0.5, dtype=np.float32)
        with pytest.raises(ValueError):
            combine_images(a, b)

    def test_bad_ndim_raises(self):
        with pytest.raises(ValueError):
            combine_images(np.zeros((2, 3, 4, 5), dtype=np.float32), _mono(0.5))
