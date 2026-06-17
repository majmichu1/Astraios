"""Tests for star reduction and star mask generation."""

import numpy as np

from astraios.core.star_reduction import (
    StarReductionParams,
    create_star_mask,
    reduce_stars,
)
from astraios.core.masks import MaskType


def _star_image(n_stars=5):
    """Create a synthetic image with bright stars."""
    image = np.ones((200, 200), dtype=np.float32) * 0.05
    yy, xx = np.mgrid[0:200, 0:200]
    positions = [(40, 40), (160, 40), (100, 100), (40, 160), (160, 160)]
    for sx, sy in positions[:n_stars]:
        dist_sq = (xx - sx) ** 2 + (yy - sy) ** 2
        star = 0.9 * np.exp(-dist_sq / (2 * 3.0**2))
        image += star.astype(np.float32)
    return np.clip(image, 0, 1)


class TestCreateStarMask:
    def test_creates_mask(self):
        image = _star_image()
        mask = create_star_mask(image)
        assert mask.mask_type == MaskType.STAR
        assert mask.data.shape == (200, 200)

    def test_stars_are_bright_in_mask(self):
        image = _star_image()
        mask = create_star_mask(image)
        # Star positions should have high mask values
        assert mask.data[40, 40] > 0.3
        assert mask.data[100, 100] > 0.3

    def test_background_is_dark(self):
        image = _star_image()
        mask = create_star_mask(image)
        # Far from any star should be near zero
        assert mask.data[0, 0] < 0.1

    def test_color_image(self):
        mono = _star_image()
        color = np.stack([mono, mono * 0.8, mono * 0.6], axis=0)
        mask = create_star_mask(color)
        assert mask.data.shape == (200, 200)


class TestReduceStars:
    def test_reduces_star_brightness(self):
        image = _star_image()
        result = reduce_stars(image, params=StarReductionParams(amount=1.0, iterations=3))
        # Stars should be dimmer
        assert result[100, 100] < image[100, 100]

    def test_preserves_background(self):
        image = _star_image()
        result = reduce_stars(image)
        # Background far from stars should be similar
        bg_orig = image[0:10, 0:10].mean()
        bg_result = result[0:10, 0:10].mean()
        assert abs(bg_orig - bg_result) < 0.05

    def test_amount_controls_strength(self):
        image = _star_image()
        r_low = reduce_stars(image, params=StarReductionParams(amount=0.2))
        r_high = reduce_stars(image, params=StarReductionParams(amount=1.0))
        # More reduction with higher amount
        assert r_high[100, 100] < r_low[100, 100]

    def test_color_image(self):
        mono = _star_image()
        color = np.stack([mono, mono, mono], axis=0)
        result = reduce_stars(color)
        assert result.shape == (3, 200, 200)

    def test_output_in_range(self):
        image = _star_image()
        result = reduce_stars(image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
