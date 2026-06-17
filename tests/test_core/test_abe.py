"""Tests for Automatic Background Extraction (ABE)."""

import numpy as np
import pytest

from astraios.core.abe import ABEParams, abe_extract
from astraios.core.masks import Mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mono_with_gradient(h: int = 200, w: int = 200) -> np.ndarray:
    """Create a mono image with a smooth linear gradient (simulating light pollution).

    The gradient goes from ~0.3 at the top to ~0.7 at the bottom, which
    represents a typical sky-glow pattern.
    """
    col = np.linspace(0.3, 0.7, h, dtype=np.float32)
    return np.tile(col[:, np.newaxis], (1, w))


def _color_with_gradient(h: int = 200, w: int = 200) -> np.ndarray:
    """Create a 3-channel image where each channel has a different gradient."""
    img = np.empty((3, h, w), dtype=np.float32)
    img[0] = np.tile(np.linspace(0.2, 0.5, h, dtype=np.float32)[:, None], (1, w))
    img[1] = np.tile(np.linspace(0.3, 0.6, h, dtype=np.float32)[:, None], (1, w))
    img[2] = np.tile(np.linspace(0.1, 0.4, h, dtype=np.float32)[:, None], (1, w))
    return img


def _flat_mono(h: int = 200, w: int = 200, value: float = 0.3) -> np.ndarray:
    """Create a perfectly flat mono image (no gradient)."""
    return np.full((h, w), value, dtype=np.float32)


# Use small grid and single iteration for speed in tests.
# darkest_fraction=1.0: test images are pure gradients with no nebula, so every
# sample is valid and the full dataset should drive the polynomial fit.
_FAST_PARAMS = ABEParams(
    grid_size=8,
    box_size=16,
    sigma_clip=3.0,
    rbf_kernel="thin_plate_spline",
    rbf_smoothing=1.0,
    correction_mode="subtraction",
    darkest_fraction=1.0,
    iterations=1,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestABEExtract:
    """Tests for abe_extract()."""

    def test_returns_tuple_of_two(self):
        """abe_extract should return (corrected, background_model)."""
        img = _mono_with_gradient()
        corrected, bg_model = abe_extract(img, _FAST_PARAMS)
        assert isinstance(corrected, np.ndarray)
        assert isinstance(bg_model, np.ndarray)

    def test_output_shapes_mono(self):
        img = _mono_with_gradient()
        corrected, bg_model = abe_extract(img, _FAST_PARAMS)
        assert corrected.shape == img.shape
        assert bg_model.shape == img.shape
        assert corrected.dtype == np.float32

    def test_output_shapes_color(self):
        img = _color_with_gradient()
        corrected, bg_model = abe_extract(img, _FAST_PARAMS)
        assert corrected.shape == (3, 200, 200)
        assert bg_model.shape == (3, 200, 200)
        assert corrected.dtype == np.float32

    def test_gradient_removal_subtraction(self):
        """Subtracting the background should flatten the gradient significantly."""
        img = _mono_with_gradient()
        params = ABEParams(
            grid_size=8,
            box_size=16,
            sigma_clip=3.0,
            rbf_smoothing=1.0,
            correction_mode="subtraction",
            iterations=2,
        )
        corrected, bg_model = abe_extract(img, params)

        # The original image has a strong vertical gradient.
        original_range = img.max() - img.min()  # ~0.4
        corrected_range = corrected.max() - corrected.min()

        # After background extraction, the range should be much smaller.
        assert corrected_range < original_range * 0.5

    def test_gradient_removal_division(self):
        """Division mode should also flatten the gradient."""
        img = _mono_with_gradient()
        params = ABEParams(
            grid_size=8,
            box_size=16,
            sigma_clip=3.0,
            rbf_smoothing=1.0,
            correction_mode="division",
            iterations=2,
        )
        corrected, bg_model = abe_extract(img, params)

        original_range = img.max() - img.min()
        corrected_range = corrected.max() - corrected.min()
        assert corrected_range < original_range * 0.5

    def test_corrected_clipped_to_01(self):
        """Corrected image should be clipped to [0, 1]."""
        img = _mono_with_gradient()
        corrected, _ = abe_extract(img, _FAST_PARAMS)
        assert corrected.min() >= 0.0
        assert corrected.max() <= 1.0

    def test_background_model_is_smooth(self):
        """The background model should be smoother than a noisy image."""
        rng = np.random.RandomState(42)
        img = _mono_with_gradient() + rng.normal(0, 0.02, (200, 200)).astype(np.float32)
        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        _, bg_model = abe_extract(img, _FAST_PARAMS)
        # Background model std should be less than the noisy input std.
        assert bg_model.std() < img.std()

    def test_flat_image_minimal_correction(self):
        """A flat image should produce near-zero correction in subtraction mode."""
        img = _flat_mono(200, 200, value=0.3)
        params = ABEParams(
            grid_size=8,
            box_size=16,
            sigma_clip=3.0,
            rbf_smoothing=1.0,
            correction_mode="subtraction",
            iterations=1,
        )
        corrected, bg_model = abe_extract(img, params)
        # The background model should be approximately uniform at ~0.3.
        assert bg_model.mean() == pytest.approx(0.3, abs=0.1)
        # The corrected image should be close to zero (original - background).
        assert corrected.mean() < 0.15

    def test_subtraction_vs_division_both_flatten(self):
        """Both modes should reduce gradient, though results differ."""
        img = _mono_with_gradient()
        original_std = img.std()

        params_sub = ABEParams(
            grid_size=8, box_size=16, sigma_clip=3.0,
            rbf_smoothing=1.0, correction_mode="subtraction", iterations=1,
        )
        params_div = ABEParams(
            grid_size=8, box_size=16, sigma_clip=3.0,
            rbf_smoothing=1.0, correction_mode="division", iterations=1,
        )

        corrected_sub, _ = abe_extract(img, params_sub)
        corrected_div, _ = abe_extract(img, params_div)

        # Both should reduce the standard deviation.
        assert corrected_sub.std() < original_std
        assert corrected_div.std() < original_std

        # The two modes should produce different results.
        assert not np.allclose(corrected_sub, corrected_div, atol=0.01)

    def test_color_gradient_removal(self):
        """ABE should flatten gradients independently per channel."""
        img = _color_with_gradient()
        params = ABEParams(
            grid_size=8, box_size=16, sigma_clip=3.0,
            rbf_smoothing=1.0, correction_mode="subtraction", iterations=2,
        )
        corrected, bg_model = abe_extract(img, params)

        for ch in range(3):
            orig_range = img[ch].max() - img[ch].min()
            corr_range = corrected[ch].max() - corrected[ch].min()
            assert corr_range < orig_range * 0.5

    def test_mask_zero_preserves_original(self):
        """A zero mask should return the original image unchanged."""
        img = _mono_with_gradient()
        mask = Mask(data=np.zeros((200, 200), dtype=np.float32))
        corrected, bg_model = abe_extract(img, _FAST_PARAMS, mask=mask)
        np.testing.assert_array_almost_equal(corrected, img)

    def test_mask_full_applies_correction(self):
        """A full mask should give the same result as no mask."""
        img = _mono_with_gradient()
        mask = Mask(data=np.ones((200, 200), dtype=np.float32))
        corrected_mask, _ = abe_extract(img, _FAST_PARAMS, mask=mask)
        corrected_none, _ = abe_extract(img, _FAST_PARAMS, mask=None)
        np.testing.assert_array_almost_equal(corrected_mask, corrected_none)

    def test_mask_partial_blends(self):
        """Partial mask should blend corrected and original regions."""
        img = _mono_with_gradient()
        mask_data = np.zeros((200, 200), dtype=np.float32)
        mask_data[:, 100:] = 1.0  # right half processed
        mask = Mask(data=mask_data)
        corrected, _ = abe_extract(img, _FAST_PARAMS, mask=mask)
        # Left half (mask=0) should equal original.
        np.testing.assert_array_almost_equal(corrected[:, :100], img[:, :100])
        # Right half (mask=1) should differ from original.
        assert not np.allclose(corrected[:, 100:], img[:, 100:])

    def test_mask_with_color_image(self):
        img = _color_with_gradient()
        mask = Mask(data=np.ones((200, 200), dtype=np.float32) * 0.5)
        corrected, bg_model = abe_extract(img, _FAST_PARAMS, mask=mask)
        assert corrected.shape == (3, 200, 200)

    def test_none_params_uses_defaults(self):
        img = _mono_with_gradient()
        corrected, bg_model = abe_extract(img, None)
        assert corrected.shape == img.shape
        assert bg_model.shape == img.shape

    def test_multiple_iterations_refine(self):
        """More iterations should produce a better gradient removal."""
        img = _mono_with_gradient()
        params_1 = ABEParams(
            grid_size=8, box_size=16, sigma_clip=3.0,
            rbf_smoothing=1.0, correction_mode="subtraction", iterations=1,
        )
        params_3 = ABEParams(
            grid_size=8, box_size=16, sigma_clip=3.0,
            rbf_smoothing=1.0, correction_mode="subtraction", iterations=3,
        )
        corrected_1, _ = abe_extract(img, params_1)
        corrected_3, _ = abe_extract(img, params_3)
        # More iterations should yield a flatter result (lower std).
        assert corrected_3.std() <= corrected_1.std() + 0.01

    def test_division_background_model_positive(self):
        """In division mode, the background model should be all positive values."""
        img = _mono_with_gradient()
        params = ABEParams(
            grid_size=8, box_size=16, sigma_clip=3.0,
            rbf_smoothing=1.0, correction_mode="division", iterations=1,
        )
        _, bg_model = abe_extract(img, params)
        # The model starts at ones and is multiplied by per-iteration models,
        # which are derived from positive background samples.
        assert bg_model.min() > 0.0
