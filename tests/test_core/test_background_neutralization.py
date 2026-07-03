"""Tests for background neutralization (astraios/core/background_neutralization.py)."""

import numpy as np

from astraios.core.background_neutralization import (
    BackgroundNeutralizationParams,
    background_neutralization,
)
from astraios.core.masks import Mask, MaskType


def _color_cast_image(h=120, w=140, bg=(0.05, 0.08, 0.12), seed=1):
    """Build a color image with a deliberate per-channel sky background color
    cast, plus a bright signal patch (like stars) so protect_bright has real
    bright pixels to exclude from the background estimate.
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((3, h, w), dtype=np.float32)
    for c in range(3):
        img[c] = bg[c] + rng.normal(0, 0.003, (h, w)).astype(np.float32)
    img[:, 50:70, 50:70] += 0.6  # bright "star" patch, well above protect_bright cutoff
    return np.clip(img, 0, 1).astype(np.float32)


class TestColorCastEqualization:
    def test_equalizes_per_channel_background_medians(self):
        img = _color_cast_image()
        sky_before = [float(np.median(img[c, :20, :20])) for c in range(3)]
        spread_before = max(sky_before) - min(sky_before)
        assert spread_before > 0.05  # sanity: the cast is real

        out = background_neutralization(img)
        sky_after = [float(np.median(out[c, :20, :20])) for c in range(3)]
        spread_after = max(sky_after) - min(sky_after)

        # The deliberate color cast in the background must be almost entirely removed.
        assert spread_after < spread_before * 0.05

    def test_amount_scales_correction_strength(self):
        img = _color_cast_image()
        sky_before = np.array([np.median(img[c, :20, :20]) for c in range(3)])

        half = background_neutralization(img, BackgroundNeutralizationParams(amount=0.5))
        sky_half = np.array([np.median(half[c, :20, :20]) for c in range(3)])

        full = background_neutralization(img, BackgroundNeutralizationParams(amount=1.0))
        sky_full = np.array([np.median(full[c, :20, :20]) for c in range(3)])

        # Half-strength correction should land between the original and full correction.
        assert np.all(sky_half > sky_full)
        assert np.all(sky_half < sky_before)

    def test_amount_zero_is_a_no_op(self):
        img = _color_cast_image()
        out = background_neutralization(img, BackgroundNeutralizationParams(amount=0.0))
        assert np.allclose(out, img)

    def test_bright_signal_is_not_erased(self):
        img = _color_cast_image()
        out = background_neutralization(img)
        # The bright patch should remain clearly brighter than background after correction.
        signal_mean = float(np.mean(out[:, 55:65, 55:65]))
        bg_mean = float(np.mean(out[:, :20, :20]))
        assert signal_mean > bg_mean + 0.3


class TestMonoImage:
    def test_mono_background_is_pulled_toward_zero(self):
        rng = np.random.RandomState(2)
        h, w = 100, 100
        mono = np.full((h, w), 0.08, dtype=np.float32)
        mono += rng.normal(0, 0.002, (h, w)).astype(np.float32)
        mono[40:60, 40:60] += 0.7  # bright signal so protect_bright has something to exclude
        mono = np.clip(mono, 0, 1).astype(np.float32)

        out = background_neutralization(mono)
        assert out.shape == mono.shape
        sky_before = float(np.median(mono[:20, :20]))
        sky_after = float(np.median(out[:20, :20]))
        assert sky_after < sky_before * 0.5

    def test_flat_mono_image_with_no_bright_pixels_is_unchanged(self):
        """Regression/behavior-lock: when the whole image sits above the
        protect_bright*max cutoff (no pixel is ever "bright" relative to a flat
        field), _sky_level's bright-pixel filter excludes every pixel, the
        background estimate collapses to 0.0, and the image passes through
        unmodified. This is arguably a bug — protect_bright assumes background
        is near zero relative to peak signal, which fails for flat fields with
        no strong point sources (astraios/core/background_neutralization.py:74-79) —
        but it is documented here as current, tested behavior.
        """
        rng = np.random.RandomState(3)
        mono = 0.15 + rng.normal(0, 0.005, (80, 80)).astype(np.float32)
        mono = np.clip(mono, 0, 1).astype(np.float32)

        out = background_neutralization(mono)
        assert np.array_equal(out, mono)


class TestMaskIntegration:
    def test_mask_protects_original_pixels(self):
        img = _color_cast_image()
        mask_data = np.zeros(img.shape[-2:], dtype=np.float32)
        mask_data[:, :70] = 1.0  # only left half gets processed
        mask = Mask(data=mask_data, name="left-half", mask_type=MaskType.MANUAL)

        out = background_neutralization(img, mask=mask)

        # Right half (protected) must equal the original image exactly.
        assert np.array_equal(out[:, :, 70:], img[:, :, 70:])
        # Left half (processed) should differ from the original.
        assert not np.array_equal(out[:, :, :70], img[:, :, :70])


class TestInputValidation:
    def test_output_dtype_and_range(self):
        img = _color_cast_image()
        out = background_neutralization(img)
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0
        assert out.shape == img.shape
