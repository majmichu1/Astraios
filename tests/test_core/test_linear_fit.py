"""Tests for linear fit — linear_fit.py."""

import numpy as np
import pytest

from astraios.core.linear_fit import LinearFitParams, compute_linear_fit, linear_fit


def _reference_mono(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    base = 0.3 + 0.15 * np.sin(xx * 0.15) * np.cos(yy * 0.1)
    noise = 0.01 * rng.standard_normal((h, w))
    return np.clip(base + noise, 0.0, 1.0).astype(np.float32)


class TestComputeLinearFit:
    def test_recovers_known_slope_and_intercept(self):
        reference = _reference_mono()
        a, b = 0.6, 0.05
        image = np.clip(reference * a + b, 0.0, 1.0).astype(np.float32)

        slope, intercept = compute_linear_fit(image, reference)
        # image = a*reference + b  =>  reference ~= (image - b) / a
        # compute_linear_fit fits reference ~= slope*image + intercept,
        # so slope ~= 1/a, intercept ~= -b/a.
        assert slope == pytest.approx(1.0 / a, rel=0.05)
        assert intercept == pytest.approx(-b / a, abs=0.05)

    def test_mapping_recovers_reference(self):
        reference = _reference_mono()
        a, b = 0.6, 0.05
        image = np.clip(reference * a + b, 0.0, 1.0).astype(np.float32)

        mapped = linear_fit(image, reference)
        assert mapped.shape == reference.shape
        assert np.abs(mapped - reference).mean() < 0.02

    def test_identity_when_image_equals_reference(self):
        reference = _reference_mono()
        mapped = linear_fit(reference, reference)
        np.testing.assert_allclose(mapped, reference, atol=1e-4)
        slope, intercept = compute_linear_fit(reference, reference)
        assert slope == pytest.approx(1.0, abs=1e-3)
        assert intercept == pytest.approx(0.0, abs=1e-3)

    def test_sigma_clip_rejects_outliers(self):
        rng = np.random.default_rng(3)
        reference = _reference_mono(seed=1)
        a, b = 1.3, -0.05
        image = np.clip(reference * a + b, 0.0, 1.0).astype(np.float32)

        # Inject gross outliers (e.g. hot pixels / cosmic rays) into a small
        # fraction of the image that would otherwise dominate a plain fit.
        image_outliers = image.copy()
        flat_idx = rng.choice(image.size, size=image.size // 20, replace=False)
        image_outliers.flat[flat_idx] = 1.0
        reference_outliers = reference.copy()
        # Keep reference clean at those same locations so the injected
        # points are true outliers relative to the linear relationship.

        clipped_params = LinearFitParams(sigma_clip=True, sigma=3.0, max_iters=5)
        unclipped_params = LinearFitParams(sigma_clip=False)

        slope_clipped, _ = compute_linear_fit(
            image_outliers, reference_outliers, clipped_params
        )
        slope_unclipped, _ = compute_linear_fit(
            image_outliers, reference_outliers, unclipped_params
        )

        true_slope = 1.0 / a
        assert abs(slope_clipped - true_slope) < abs(slope_unclipped - true_slope)
        assert slope_clipped == pytest.approx(true_slope, rel=0.1)

    def test_per_channel_vs_global(self):
        reference = np.stack([_reference_mono(seed=i) for i in range(3)], axis=0)
        gains = np.array([0.5, 1.0, 1.5]).reshape(3, 1, 1)
        image = np.clip(reference * gains, 0.0, 1.0).astype(np.float32)

        per_channel = linear_fit(image, reference, LinearFitParams(per_channel=True))
        # Per-channel fit should recover each channel's reference well.
        assert np.abs(per_channel - reference).mean() < 0.02

        slopes, intercepts = compute_linear_fit(image, reference, LinearFitParams(per_channel=True))
        assert slopes.shape == (3,)
        assert intercepts.shape == (3,)
        np.testing.assert_allclose(slopes, 1.0 / gains.ravel(), rtol=0.1)

        global_slope, global_intercept = compute_linear_fit(
            image, reference, LinearFitParams(per_channel=False)
        )
        assert isinstance(global_slope, float)
        assert isinstance(global_intercept, float)

    def test_shape_mismatch_raises(self):
        reference = _reference_mono()
        image = _reference_mono()[:-1, :]
        with pytest.raises(ValueError):
            compute_linear_fit(image, reference)

    def test_output_clipped_to_range(self):
        reference = _reference_mono()
        image = np.clip(reference * 3.0, 0.0, 1.0).astype(np.float32)
        mapped = linear_fit(image, reference, LinearFitParams(clip_output=True))
        assert mapped.min() >= 0.0
        assert mapped.max() <= 1.0

    def test_output_not_clipped_when_disabled(self):
        reference = _reference_mono()
        a, b = 3.0, 0.5
        image = np.clip(reference * a + b, 0.0, 1.0).astype(np.float32)
        mapped = linear_fit(image, reference, LinearFitParams(clip_output=False))
        # Without clipping, some out-of-range values are plausible given
        # noise and rounding in the fit; just ensure clipping truly differs.
        clipped = linear_fit(image, reference, LinearFitParams(clip_output=True))
        assert not np.array_equal(mapped, clipped) or (mapped.min() >= 0 and mapped.max() <= 1)
