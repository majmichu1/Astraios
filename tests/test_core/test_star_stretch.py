"""Tests for the colour-preserving star stretch."""

import numpy as np

from astraios.core.masks import Mask
from astraios.core.star_stretch import StarStretchParams, star_stretch


def _star_image(h=64, w=64):
    """Dim, colourful 'stars' on a dark background."""
    rng = np.random.default_rng(0)
    img = np.full((3, h, w), 0.02, np.float32)
    for _ in range(20):
        y, x = int(rng.integers(4, h - 4)), int(rng.integers(4, w - 4))
        color = rng.random(3).astype(np.float32) * 0.25 + 0.05
        img[:, y - 1:y + 2, x - 1:x + 2] += color[:, None, None]
    return np.clip(img, 0, 1)


class TestStarStretch:
    def test_brightens_dim_stars(self):
        img = _star_image()
        out = star_stretch(img, StarStretchParams(amount=0.5))
        assert out.shape == img.shape
        assert out.min() >= 0.0 and out.max() <= 1.0
        # A stretch must lift the mean of the (mostly dim) image.
        assert out.mean() > img.mean()

    def test_preserves_hue_ratios(self):
        # A grey-free coloured pixel should keep its channel ordering after the
        # colour-preserving stretch (no boost).
        img = np.zeros((3, 8, 8), np.float32)
        img[0] = 0.30  # R brightest
        img[1] = 0.15
        img[2] = 0.05  # B dimmest
        out = star_stretch(img, StarStretchParams(amount=0.4, color_boost=1.0))
        assert out[0].mean() > out[1].mean() > out[2].mean()

    def test_color_boost_increases_saturation(self):
        img = _star_image()
        base = star_stretch(img, StarStretchParams(amount=0.4, color_boost=1.0))
        boosted = star_stretch(img, StarStretchParams(amount=0.4, color_boost=2.0))

        def mean_chroma(a):
            lum = a.mean(axis=0, keepdims=True)
            return float(np.mean(np.abs(a - lum)))

        assert mean_chroma(boosted) > mean_chroma(base)

    def test_mono_image_supported(self):
        mono = np.clip(np.full((32, 32), 0.05, np.float32)
                       + np.random.default_rng(1).random((32, 32)) * 0.1, 0, 1).astype(np.float32)
        out = star_stretch(mono, StarStretchParams(amount=0.5, color_boost=2.0))
        assert out.shape == mono.shape
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_mask_protects_region(self):
        img = _star_image()
        mask_data = np.zeros((64, 64), np.float32)  # convention: (H, W)
        mask_data[32:, :] = 1.0
        out = star_stretch(img, StarStretchParams(amount=0.6), mask=Mask(data=mask_data))
        np.testing.assert_allclose(out[:, :32, :], img[:, :32, :], atol=1e-6)
