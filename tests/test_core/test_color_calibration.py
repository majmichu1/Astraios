"""Tests for photometric color calibration."""

import numpy as np

from astraios.core.color_calibration import (
    ColorCalibrationParams,
    ColorCalibrationResult,
    color_calibrate,
)


def _color_image_with_cast():
    """Create a color image with a blue cast."""
    rng = np.random.default_rng(42)
    image = np.zeros((3, 200, 200), dtype=np.float32)
    image[0] = 0.2 + rng.normal(0, 0.02, (200, 200))  # R dim
    image[1] = 0.25 + rng.normal(0, 0.02, (200, 200))  # G slightly brighter
    image[2] = 0.35 + rng.normal(0, 0.02, (200, 200))  # B brightest (blue cast)

    # Add some stars
    yy, xx = np.mgrid[0:200, 0:200]
    for sx, sy in [(50, 50), (150, 50), (100, 100), (50, 150), (150, 150)]:
        dist_sq = (xx - sx) ** 2 + (yy - sy) ** 2
        star = 0.7 * np.exp(-dist_sq / (2 * 3.0**2))
        for ch in range(3):
            image[ch] += star

    return np.clip(image, 0, 1).astype(np.float32)


class TestColorCalibration:
    def test_produces_result(self):
        image = _color_image_with_cast()
        result = color_calibrate(image)
        assert isinstance(result, ColorCalibrationResult)
        assert result.data.shape == image.shape
        assert len(result.correction_factors) == 3

    def test_correction_factors_valid(self):
        image = _color_image_with_cast()
        result = color_calibrate(image)
        # All factors should be positive and <= 1.0 (normalized)
        for f in result.correction_factors:
            assert 0 < f <= 1.01

    def test_background_neutralization(self):
        image = _color_image_with_cast()
        params = ColorCalibrationParams(neutralize_background=True)
        result = color_calibrate(image, params)
        # Background should be more neutral after calibration
        bg_r = result.data[0, :10, :10].mean()
        bg_g = result.data[1, :10, :10].mean()
        bg_b = result.data[2, :10, :10].mean()
        spread = max(bg_r, bg_g, bg_b) - min(bg_r, bg_g, bg_b)
        # Original spread is about 0.15, should be reduced
        assert spread < 0.1

    def test_mono_unchanged(self):
        image = np.ones((100, 100), dtype=np.float32) * 0.5
        result = color_calibrate(image)
        np.testing.assert_array_equal(result.data, image)

    def test_custom_reference(self):
        image = _color_image_with_cast()
        params = ColorCalibrationParams(
            white_reference="custom",
            custom_rgb=(1.0, 0.9, 0.8),
        )
        result = color_calibrate(image, params)
        assert result.correction_factors == (1.0, 0.9, 0.8)


class TestBackgroundStaysNeutral:
    """Regression: white balance is multiplicative and background
    neutralization is additive, so neutralizing first and applying gains
    afterwards scales the background straight back off neutral.

    The old order left a *magenta* background that got worse the stronger the
    original cast was: a 25% green cast came out at -15% green, a 45% cast at
    -23%, both worse than doing nothing at all.
    """

    @staticmethod
    def _scene(green_cast=1.0, seed=0):
        rng = np.random.default_rng(seed)
        h = w = 160
        yy, xx = np.mgrid[0:h, 0:w]
        img = np.full((3, h, w), 0.004, np.float32)
        neb = 0.012 * np.exp(-(((xx - 80) ** 2 + (yy - 78) ** 2) / (2 * 32.0**2)))
        for c, amp in enumerate((1.0, 0.55, 0.5)):
            img[c] += (neb * amp).astype(np.float32)
        for _ in range(25):
            x, y = rng.integers(12, h - 12, 2)
            star = rng.uniform(0.02, 0.3) * np.exp(
                -(((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.8**2))
            )
            img = img + star.astype(np.float32)
        img = img + rng.normal(0, 0.0015, img.shape)
        img[1] *= green_cast
        return np.clip(img, 0, 1).astype(np.float32)

    @staticmethod
    def _background_green_excess(image):
        lum = image.mean(axis=0)
        sel = lum < np.percentile(lum, 40)
        ch = [float(image[c][sel].mean()) for c in range(3)]
        neutral = (ch[0] + ch[2]) / 2.0
        return (ch[1] - neutral) / max(neutral, 1e-9)

    def test_background_ends_near_neutral_for_any_cast(self):
        for cast in (1.10, 1.25, 1.45):
            img = self._scene(green_cast=cast)
            out = color_calibrate(
                img, ColorCalibrationParams(neutralize_background=True)
            ).data
            before = abs(self._background_green_excess(img))
            after = abs(self._background_green_excess(out))
            assert after < 0.05, (
                f"cast {cast}: background left at {after:.4f} green excess"
            )
            assert after < before, f"cast {cast}: calibration made the cast worse"

    def test_correction_does_not_degrade_with_a_stronger_cast(self):
        """The old bug scaled with the input: the worse the cast, the worse
        the magenta residual it left behind."""
        residuals = []
        for cast in (1.10, 1.25, 1.45):
            out = color_calibrate(
                self._scene(green_cast=cast),
                ColorCalibrationParams(neutralize_background=True),
            ).data
            residuals.append(abs(self._background_green_excess(out)))
        assert max(residuals) < 0.05
        # and they should all be about the same, not growing with the cast
        assert max(residuals) - min(residuals) < 0.02

    def test_star_colour_is_corrected_too(self):
        img = self._scene(green_cast=1.35)
        out = color_calibrate(
            img, ColorCalibrationParams(neutralize_background=True)
        ).data

        def star_green_excess(a):
            lum = a.mean(axis=0)
            sel = lum > np.percentile(lum, 99.5)
            ch = [float(a[c][sel].mean()) for c in range(3)]
            neutral = (ch[0] + ch[2]) / 2.0
            return (ch[1] - neutral) / max(neutral, 1e-9)

        assert abs(star_green_excess(out)) < abs(star_green_excess(img)) * 0.25
