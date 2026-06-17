"""Tests for image blend modes."""

import numpy as np

from astraios.core.image_blend import BlendMode, BlendParams, blend_images
from astraios.core.masks import Mask


def _img(val, shape=(3, 32, 40)):
    return np.full(shape, val, np.float32)


class TestBlendModes:
    def test_screen_brightens_and_matches_formula(self):
        a, b = _img(0.4), _img(0.5)
        out = blend_images(a, b, BlendParams(mode=BlendMode.SCREEN))
        expected = 1 - (1 - 0.4) * (1 - 0.5)
        np.testing.assert_allclose(out, np.full_like(a, expected), atol=1e-5)
        assert out.mean() > a.mean()  # screen never darkens

    def test_multiply_darkens(self):
        a, b = _img(0.6), _img(0.5)
        out = blend_images(a, b, BlendParams(mode=BlendMode.MULTIPLY))
        np.testing.assert_allclose(out, _img(0.3), atol=1e-5)

    def test_lighten_darken_add_subtract(self):
        a, b = _img(0.3), _img(0.7)
        assert np.allclose(blend_images(a, b, BlendParams(mode=BlendMode.LIGHTEN)), _img(0.7), atol=1e-5)
        assert np.allclose(blend_images(a, b, BlendParams(mode=BlendMode.DARKEN)), _img(0.3), atol=1e-5)
        assert np.allclose(blend_images(a, b, BlendParams(mode=BlendMode.ADD)), _img(1.0), atol=1e-5)
        assert np.allclose(blend_images(b, a, BlendParams(mode=BlendMode.SUBTRACT)), _img(0.4), atol=1e-5)

    def test_difference_and_average(self):
        a, b = _img(0.8), _img(0.2)
        assert np.allclose(blend_images(a, b, BlendParams(mode=BlendMode.DIFFERENCE)), _img(0.6), atol=1e-5)
        assert np.allclose(blend_images(a, b, BlendParams(mode=BlendMode.AVERAGE)), _img(0.5), atol=1e-5)

    def test_opacity_interpolates(self):
        a, b = _img(0.0), _img(1.0)
        out = blend_images(a, b, BlendParams(mode=BlendMode.NORMAL, opacity=0.25))
        np.testing.assert_allclose(out, _img(0.25), atol=1e-5)

    def test_result_clipped(self):
        out = blend_images(_img(0.9), _img(0.9), BlendParams(mode=BlendMode.ADD))
        assert out.max() <= 1.0 and out.min() >= 0.0


class TestShapeHandling:
    def test_mono_layer_broadcasts_to_color_base(self):
        base = _img(0.4, (3, 16, 16))
        layer = _img(0.5, (16, 16))  # mono
        out = blend_images(base, layer, BlendParams(mode=BlendMode.SCREEN))
        assert out.shape == base.shape

    def test_blend_layer_resized_to_base(self):
        base = _img(0.3, (3, 32, 32))
        layer = _img(0.6, (3, 8, 8))  # smaller -> resized
        out = blend_images(base, layer, BlendParams(mode=BlendMode.SCREEN))
        assert out.shape == base.shape

    def test_mono_base_preserved(self):
        base = _img(0.3, (24, 24))
        layer = _img(0.5, (24, 24))
        out = blend_images(base, layer, BlendParams(mode=BlendMode.SCREEN))
        assert out.shape == (24, 24)

    def test_mask_protects_region(self):
        base = _img(0.2, (3, 16, 16))
        layer = _img(0.9, (3, 16, 16))
        mask_data = np.zeros((16, 16), np.float32)
        mask_data[8:, :] = 1.0
        out = blend_images(base, layer, BlendParams(mode=BlendMode.SCREEN), mask=Mask(data=mask_data))
        np.testing.assert_allclose(out[:, :8, :], base[:, :8, :], atol=1e-6)
