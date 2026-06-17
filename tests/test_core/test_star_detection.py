"""Tests for the star detection module."""

import numpy as np
import pytest

from astraios.core.star_detection import (
    Star,
    StarField,
    align_image,
    detect_stars,
    find_transform,
)


def _make_star_image(
    width: int = 200,
    height: int = 200,
    star_positions: list[tuple[int, int]] | None = None,
    brightness: float = 0.9,
    sigma: float = 2.0,
) -> np.ndarray:
    """Create a synthetic image with Gaussian stars."""
    image = np.random.rand(height, width).astype(np.float32) * 0.01  # faint background
    if star_positions is None:
        star_positions = [(50, 50), (150, 50), (100, 100), (50, 150), (150, 150)]

    yy, xx = np.mgrid[0:height, 0:width]
    for sx, sy in star_positions:
        dist_sq = (xx - sx) ** 2 + (yy - sy) ** 2
        star = brightness * np.exp(-dist_sq / (2 * sigma**2))
        image += star.astype(np.float32)

    return np.clip(image, 0, 1)


class TestStarDetection:
    """Tests for detect_stars."""

    def test_detect_basic_stars(self):
        image = _make_star_image()
        result = detect_stars(image)
        assert isinstance(result, StarField)
        assert len(result) >= 3  # should find most of the 5 stars

    def test_returns_positions(self):
        image = _make_star_image()
        result = detect_stars(image)
        pos = result.positions
        assert pos.ndim == 2
        assert pos.shape[1] == 2

    def test_returns_fluxes(self):
        image = _make_star_image()
        result = detect_stars(image)
        fluxes = result.fluxes
        assert len(fluxes) == len(result)
        assert all(f > 0 for f in fluxes)

    def test_sorted_by_brightness(self):
        image = _make_star_image()
        result = detect_stars(image)
        if len(result) > 1:
            fluxes = result.fluxes
            for i in range(len(fluxes) - 1):
                assert fluxes[i] >= fluxes[i + 1]

    def test_max_stars_limit(self):
        # Create image with many stars
        positions = [(x, y) for x in range(20, 180, 20) for y in range(20, 180, 20)]
        image = _make_star_image(star_positions=positions)
        result = detect_stars(image, max_stars=5)
        assert len(result) <= 5

    def test_color_image(self):
        mono = _make_star_image()
        color = np.stack([mono, mono * 0.8, mono * 0.6], axis=0)
        result = detect_stars(color)
        assert len(result) >= 3

    def test_empty_image(self):
        image = np.zeros((100, 100), dtype=np.float32)
        result = detect_stars(image)
        assert len(result) == 0
        assert result.positions.shape == (0, 2)

    def test_image_dimensions_stored(self):
        image = _make_star_image(width=300, height=200)
        result = detect_stars(image)
        assert result.image_width == 300
        assert result.image_height == 200


class TestStarFieldProperties:
    """Tests for StarField dataclass."""

    def test_star_properties(self):
        star = Star(x=10.5, y=20.3, flux=0.8, fwhm=3.2, roundness=0.1)
        assert star.x == 10.5
        assert star.y == 20.3
        assert star.flux == 0.8
        assert star.fwhm == 3.2
        assert star.roundness == 0.1

    def test_empty_starfield(self):
        sf = StarField(stars=[], image_width=100, image_height=100)
        assert len(sf) == 0
        assert sf.positions.shape == (0, 2)
        assert sf.fluxes.shape == (0,)


class TestFindTransform:
    """Tests for find_transform."""

    def test_identity_transform(self):
        image = _make_star_image()
        stars = detect_stars(image)
        # Same stars should give near-identity transform
        transform = find_transform(stars, stars)
        if transform is not None:
            # Should be close to identity: [[1, 0, 0], [0, 1, 0]]
            np.testing.assert_array_almost_equal(
                transform[:, :2], np.eye(2), decimal=1
            )

    def test_shifted_stars(self):
        positions1 = [(50, 50), (150, 50), (100, 100), (50, 150), (150, 150)]
        positions2 = [(55, 53), (155, 53), (105, 103), (55, 153), (155, 153)]
        img1 = _make_star_image(star_positions=positions1)
        img2 = _make_star_image(star_positions=positions2)
        sf1 = detect_stars(img1)
        sf2 = detect_stars(img2)
        transform = find_transform(sf1, sf2)
        # Should find a valid transform
        assert transform is not None
        assert transform.shape == (2, 3)

    def test_too_few_stars(self):
        sf1 = StarField(stars=[Star(x=10, y=10, flux=1)], image_width=100, image_height=100)
        sf2 = StarField(stars=[Star(x=20, y=20, flux=1)], image_width=100, image_height=100)
        transform = find_transform(sf1, sf2)
        assert transform is None

    def test_accepts_ndarray(self):
        pts1 = np.array([[50, 50], [150, 50], [100, 100], [50, 150]], dtype=np.float32)
        pts2 = pts1 + 5  # small shift
        transform = find_transform(pts1, pts2)
        assert transform is not None


class TestAlignImage:
    """Tests for align_image."""

    def test_identity_align(self):
        image = _make_star_image()
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        aligned = align_image(image, identity, image.shape)
        np.testing.assert_array_almost_equal(aligned, image, decimal=4)

    def test_color_image_align(self):
        mono = _make_star_image()
        color = np.stack([mono, mono, mono], axis=0)
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        aligned = align_image(color, identity, color.shape)
        assert aligned.shape == color.shape

    def test_translation_align(self):
        image = _make_star_image()
        # Translate by (10, 5) pixels
        transform = np.array([[1, 0, 10], [0, 1, 5]], dtype=np.float32)
        aligned = align_image(image, transform, image.shape)
        assert aligned.shape == image.shape
        # Border should be zero (from BORDER_CONSTANT)
        assert aligned[0, 0] == 0.0
