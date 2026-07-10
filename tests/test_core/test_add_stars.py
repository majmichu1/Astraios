"""Tests for Add Stars (ported from SASpro pro/add_stars.py)."""

import numpy as np
import pytest

from astraios.core.add_stars import AddStarsBlendMode, AddStarsParams, add_stars
from astraios.core.masks import Mask


def _mono(val, shape=(24, 32)):
    return np.full(shape, val, dtype=np.float32)


def _color(val, shape=(3, 24, 32)):
    return np.full(shape, val, dtype=np.float32)


def _with_star(img, y=10, x=12, value=0.95):
    out = img.copy()
    if out.ndim == 2:
        out[y, x] = value
    else:
        out[:, y, x] = value
    return out


class TestScreenBlendMono:
    def test_star_present_and_brighter(self):
        starless = _mono(0.1)
        stars = np.zeros((24, 32), dtype=np.float32)
        stars[10, 12] = 0.9

        out = add_stars(starless, stars, AddStarsParams(blend_mode=AddStarsBlendMode.SCREEN))

        # Star pixel present and brighter than the starless background.
        assert out[10, 12] > starless[10, 12]
        assert out[10, 12] > 0.9
        # Background away from the star is essentially unchanged (stars == 0 there).
        np.testing.assert_allclose(out[0, 0], starless[0, 0], atol=1e-6)

    def test_matches_screen_formula_numerically(self):
        rng = np.random.default_rng(0)
        starless = rng.random((24, 32), dtype=np.float32)
        stars = rng.random((24, 32), dtype=np.float32)

        out = add_stars(
            starless, stars, AddStarsParams(blend_mode=AddStarsBlendMode.SCREEN, amount=1.0)
        )
        expected = 1.0 - (1.0 - starless) * (1.0 - stars)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_amount_zero_returns_starless_unchanged(self):
        starless = _mono(0.2)
        stars = _with_star(np.zeros((24, 32), dtype=np.float32))

        out = add_stars(starless, stars, AddStarsParams(amount=0.0))
        np.testing.assert_allclose(out, starless, atol=1e-6)

    def test_partial_amount_interpolates(self):
        starless = _mono(0.2)
        stars = _mono(0.6)
        out = add_stars(
            starless, stars, AddStarsParams(blend_mode=AddStarsBlendMode.SCREEN, amount=0.5)
        )
        base = starless + stars - starless * stars
        expected = 0.5 * starless + 0.5 * base
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_result_stays_in_unit_range(self):
        starless = _mono(0.9)
        stars = _mono(0.95)
        out_screen = add_stars(
            starless, stars, AddStarsParams(blend_mode=AddStarsBlendMode.SCREEN)
        )
        out_add = add_stars(starless, stars, AddStarsParams(blend_mode=AddStarsBlendMode.ADD))
        assert out_screen.min() >= 0.0 and out_screen.max() <= 1.0
        assert out_add.min() >= 0.0 and out_add.max() <= 1.0
        # ADD of two bright layers should actually clip (proves clipping happens).
        assert np.isclose(out_add.max(), 1.0, atol=1e-5)


class TestAddBlend:
    def test_add_matches_formula(self):
        starless = _mono(0.3)
        stars = _mono(0.2)
        out = add_stars(starless, stars, AddStarsParams(blend_mode=AddStarsBlendMode.ADD))
        np.testing.assert_allclose(out, _mono(0.5), atol=1e-5)


class TestColorPath:
    def test_color_plus_color(self):
        starless = _color(0.1)
        stars = _with_star(np.zeros((3, 24, 32), dtype=np.float32))
        out = add_stars(starless, stars, AddStarsParams(blend_mode=AddStarsBlendMode.SCREEN))
        assert out.shape == (3, 24, 32)
        assert np.all(out[:, 10, 12] > 0.9)

    def test_mono_stars_onto_color_starless_broadcasts(self):
        starless = _color(0.15)
        mono_stars = _with_star(np.zeros((24, 32), dtype=np.float32))
        out = add_stars(starless, mono_stars, AddStarsParams(blend_mode=AddStarsBlendMode.SCREEN))
        assert out.shape == (3, 24, 32)
        # Star should show up identically in every channel (broadcast).
        np.testing.assert_allclose(out[0, 10, 12], out[1, 10, 12], atol=1e-6)
        np.testing.assert_allclose(out[1, 10, 12], out[2, 10, 12], atol=1e-6)
        assert out[0, 10, 12] > 0.9


class TestMaskProtection:
    def test_protected_region_unchanged(self):
        starless = _mono(0.1)
        stars = _with_star(np.zeros((24, 32), dtype=np.float32), y=10, x=12, value=0.9)

        mask_data = np.ones((24, 32), dtype=np.float32)
        mask_data[:12, :] = 0.0  # protect the top half, including the star row
        mask = Mask(data=mask_data)

        out = add_stars(
            starless, stars, AddStarsParams(blend_mode=AddStarsBlendMode.SCREEN), mask=mask
        )
        # Protected region (top half, where the star lives) stays as starless.
        np.testing.assert_allclose(out[:12, :], starless[:12, :], atol=1e-6)
        # Unprotected region still gets the star treatment where applicable
        # (here it's flat starless everywhere else, so just confirm no crash
        # and range sanity).
        assert out.shape == starless.shape


class TestShapeMismatch:
    def test_spatial_mismatch_raises(self):
        starless = _mono(0.2, shape=(24, 32))
        stars = _mono(0.5, shape=(10, 10))
        with pytest.raises(ValueError):
            add_stars(starless, stars)

    def test_incompatible_channel_counts_raise(self):
        starless = _color(0.2, shape=(3, 24, 32))
        stars = np.zeros((2, 24, 32), dtype=np.float32)
        with pytest.raises(ValueError):
            add_stars(starless, stars)
