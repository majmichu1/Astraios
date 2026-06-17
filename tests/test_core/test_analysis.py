"""Tests for image analysis tools."""

import numpy as np
import pytest

from astraios.core.analysis.aperture_photometry import (
    PhotometryParams,
    PhotometryResult,
    _annulus_stats,
    _aperture_sum,
    _detect_sources,
    run_photometry,
)
from astraios.core.analysis.fwhm_map import (
    FWHMMapResult,
    _detect_stars_zone,
    _measure_fwhm_vectorized,
    compute_fwhm_map,
)
from astraios.core.analysis.tilt_analysis import (
    TiltAnalysisParams,
    TiltAnalysisResult,
    _measure_star_shape,
    analyze_tilt,
)

# =============================================================================
# FWHM Map tests
# =============================================================================

class TestGaussianFit2D:
    """Tests for _gaussian_fit_2d."""

    def test_fit_on_synthetic_star(self):
        """Should measure FWHM from a synthetic Gaussian star."""
        size = 21
        cx, cy = 10, 10
        sigma = 2.0
        yy, xx = np.mgrid[0:size, 0:size]
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        g = g.astype(np.float64)
        ys, xs, fwhms = _measure_fwhm_vectorized(g, threshold=0.1)
        expected_fwhm = sigma * 2.355
        assert len(fwhms) > 0
        assert fwhms[0] == pytest.approx(expected_fwhm, rel=0.3)

    def test_fit_returns_empty_on_flat(self):
        """Should return empty arrays for a flat (no-signal) patch."""
        flat = np.full((11, 11), 0.5, dtype=np.float64)
        ys, xs, fwhms = _measure_fwhm_vectorized(flat, threshold=0.9)
        assert len(fwhms) == 0

    def test_fit_returns_empty_on_zero(self):
        """Should return empty arrays for zero-valued patch."""
        zero = np.zeros((11, 11), dtype=np.float64)
        ys, xs, fwhms = _measure_fwhm_vectorized(zero, threshold=0.0)
        assert len(fwhms) == 0


class TestDetectStarsZone:
    """Tests for _detect_stars_zone."""

    def test_detects_synthetic_star(self):
        """Should detect a single bright synthetic star."""
        zone = np.full((40, 40), 0.01, dtype=np.float64)
        sigma = 1.5
        cy, cx = 20, 20
        yy, xx = np.mgrid[0:40, 0:40]
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        zone += g * 0.5
        stars = _detect_stars_zone(zone, threshold_sigma=3.0)
        assert len(stars) >= 1

    def test_empty_zone(self):
        """Should return empty list for a uniform zone."""
        zone = np.full((20, 20), 0.01, dtype=np.float64)
        stars = _detect_stars_zone(zone, threshold_sigma=10.0)
        assert len(stars) == 0


class TestComputeFWHMMap:
    """Tests for compute_fwhm_map."""

    def test_returns_result_object(self):
        """Should return a FWHMMapResult with correct structure."""
        image = np.full((64, 64), 0.01, dtype=np.float32)
        cy, cx = 32, 32
        yy, xx = np.mgrid[0:64, 0:64]
        sigma = 2.0
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        image += g.astype(np.float32) * 0.5

        result = compute_fwhm_map(image)
        assert isinstance(result, FWHMMapResult)
        assert result.fwhm_map.shape == (8, 8)
        assert result.star_counts.shape == (8, 8)

    def test_rgb_input(self):
        """Should handle (C, H, W) input by computing luminance."""
        h, w = 32, 32
        image = np.zeros((3, h, w), dtype=np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        sigma = 2.0
        g = np.exp(-((xx - w // 2) ** 2 + (yy - h // 2) ** 2) / (2 * sigma ** 2))
        image[0] = g * 0.5
        image[1] = g * 0.3
        image[2] = g * 0.1
        result = compute_fwhm_map(image)
        assert isinstance(result, FWHMMapResult)
        assert result.fwhm_map.shape == (8, 8)

    def test_tilt_detection_trigger(self):
        """Tilt should be detected when max > 1.5x min FWHM."""
        fwhm_map = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 2.5],
            [1.0, 1.0, 1.0],
        ])
        star_counts = np.ones((3, 3), dtype=np.int32) * 5
        result = FWHMMapResult(
            fwhm_map=fwhm_map,
            star_counts=star_counts,
            mean_fwhm=float(np.mean(fwhm_map)),
            std_fwhm=float(np.std(fwhm_map)),
            max_fwhm=2.5,
            min_fwhm=1.0,
            tilt_detected=2.5 > 1.5 * 1.0,
            tilt_angle=0.0,
        )
        assert result.tilt_detected is True

    def test_no_tilt_when_uniform(self):
        """Tilt should not be detected when FWHM is uniform."""
        fwhm_map = np.full((3, 3), 2.0)
        star_counts = np.ones((3, 3), dtype=np.int32) * 5
        result = FWHMMapResult(
            fwhm_map=fwhm_map,
            star_counts=star_counts,
            mean_fwhm=2.0,
            std_fwhm=0.0,
            max_fwhm=2.0,
            min_fwhm=2.0,
            tilt_detected=2.0 > 1.5 * 2.0,
            tilt_angle=0.0,
        )
        assert result.tilt_detected is False


# =============================================================================
# Aperture Photometry tests
# =============================================================================

class TestApertureSum:
    """Tests for _aperture_sum."""

    def test_sum_centered_aperture(self):
        """Should sum all values within the aperture radius."""
        image = np.zeros((20, 20), dtype=np.float64)
        image[5:15, 5:15] = 1.0
        s = _aperture_sum(image, 10.0, 10.0, 5.0)
        assert s > 0

    def test_sum_outside_bounds(self):
        """Should handle aperture centre near the image edge."""
        image = np.ones((10, 10), dtype=np.float64)
        s = _aperture_sum(image, 0.0, 0.0, 3.0)
        assert s > 0


class TestAnnulusStats:
    """Tests for _annulus_stats."""

    def test_annulus_on_uniform(self):
        """Should return median close to the uniform value."""
        image = np.full((30, 30), 0.5, dtype=np.float64)
        med, std = _annulus_stats(image, 15.0, 15.0, 5.0, 10.0)
        assert med == pytest.approx(0.5, abs=0.01)
        assert std == pytest.approx(0.0, abs=0.01)

    def test_annulus_empty(self):
        """Should return zeros for an annulus that extends beyond the image."""
        image = np.ones((10, 10), dtype=np.float64)
        med, std = _annulus_stats(image, 5.0, 5.0, 100.0, 200.0)
        assert med == 0.0
        assert std == 0.0


class TestDetectSources:
    """Tests for _detect_sources."""

    def test_detects_bright_spot(self):
        """Should detect a bright region above threshold."""
        image = np.zeros((20, 20), dtype=np.float64)
        image[8:12, 8:12] = 1.0
        x, y = _detect_sources(image, 0.5, 10)
        assert len(x) >= 1
        assert len(y) >= 1

    def test_respects_max_sources(self):
        """Should not return more than max_sources."""
        image = np.ones((20, 20), dtype=np.float64)
        x, y = _detect_sources(image, 0.5, 3)
        assert len(x) <= 3
        assert len(y) <= 3


class TestRunPhotometry:
    """Tests for run_photometry."""

    def test_returns_result_object(self):
        """Should return a PhotometryResult."""
        image = np.zeros((64, 64), dtype=np.float32)
        image[30:35, 30:35] = 0.8
        result = run_photometry(image)
        assert isinstance(result, PhotometryResult)

    def test_handles_empty_image(self):
        """Should not crash on an empty image."""
        image = np.full((32, 32), 0.01, dtype=np.float32)
        result = run_photometry(image)
        assert isinstance(result, PhotometryResult)

    def test_rgb_input(self):
        """Should handle (C, H, W) RGB input."""
        image = np.zeros((3, 32, 32), dtype=np.float32)
        image[:, 14:18, 14:18] = 0.8
        result = run_photometry(image)
        assert isinstance(result, PhotometryResult)

    def test_custom_params(self):
        """Should accept custom PhotometryParams."""
        params = PhotometryParams(
            aperture_radius=5.0,
            annulus_inner=8.0,
            annulus_outer=12.0,
            detection_threshold=3.0,
            max_sources=100,
        )
        image = np.zeros((64, 64), dtype=np.float32)
        image[30:35, 30:35] = 0.9
        result = run_photometry(image, params)
        assert isinstance(result, PhotometryResult)


# =============================================================================
# Tilt Analysis tests
# =============================================================================

class TestMeasureStarShape:
    """Tests for _measure_star_shape."""

    def test_circular_star(self):
        """Should measure low ellipticity for a round Gaussian."""
        size = 21
        sigma = 2.0
        yy, xx = np.mgrid[0:size, 0:size]
        g = np.exp(-((xx - 10) ** 2 + (yy - 10) ** 2) / (2 * sigma ** 2))
        g = g.astype(np.float64) * 0.8 + 0.01
        shape = _measure_star_shape(g, 10, 10, size=10)
        assert shape is not None
        ell, angle = shape
        assert ell < 0.3

    def test_elongated_star(self):
        """Should measure higher ellipticity for an elongated Gaussian."""
        size = 21
        yy, xx = np.mgrid[0:size, 0:size]
        g = np.exp(-((xx - 10) ** 2 / (2 * 1.0 ** 2) + (yy - 10) ** 2 / (2 * 4.0 ** 2)))
        g = g.astype(np.float64) * 0.8 + 0.01
        shape = _measure_star_shape(g, 10, 10, size=10)
        assert shape is not None
        ell, angle = shape
        assert ell > 0.3

    def test_returns_none_on_flat(self):
        """Should return None for a flat region."""
        flat = np.full((11, 11), 0.5, dtype=np.float64)
        assert _measure_star_shape(flat, 5, 5, size=5) is None


class TestTiltAnalysisParams:
    """Tests for TiltAnalysisParams defaults."""

    def test_default_params(self):
        """Default params should have sensible values."""
        params = TiltAnalysisParams()
        assert params.grid_rows == 8
        assert params.grid_cols == 8
        assert params.min_stars_per_zone == 2


class TestAnalyzeTilt:
    """Tests for analyze_tilt."""

    def test_returns_result_object(self):
        """Should return a TiltAnalysisResult."""
        image = np.full((64, 64), 0.01, dtype=np.float32)
        yy, xx = np.mgrid[0:64, 0:64]
        sigma = 2.0
        g = np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * sigma ** 2))
        image += g.astype(np.float32) * 0.5
        result = analyze_tilt(image)
        assert isinstance(result, TiltAnalysisResult)

    def test_all_maps_have_correct_shape(self):
        """All output maps should match the grid dimensions."""
        image = np.full((64, 64), 0.01, dtype=np.float32)
        yy, xx = np.mgrid[0:64, 0:64]
        sigma = 2.0
        g = np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * sigma ** 2))
        image += g.astype(np.float32) * 0.5
        result = analyze_tilt(image, TiltAnalysisParams(grid_rows=4, grid_cols=4))
        assert result.ellipticity_map.shape == (4, 4)
        assert result.angle_map.shape == (4, 4)
        assert result.fwhm_map.shape == (4, 4)

    def test_rgb_input(self):
        """Should handle (C, H, W) RGB input."""
        image = np.zeros((3, 32, 32), dtype=np.float32)
        yy, xx = np.mgrid[0:32, 0:32]
        g = np.exp(-((xx - 16) ** 2 + (yy - 16) ** 2) / (2 * 2.0 ** 2))
        image[0] = g.astype(np.float32) * 0.5
        image[1] = g.astype(np.float32) * 0.3
        image[2] = g.astype(np.float32) * 0.1
        result = analyze_tilt(image)
        assert isinstance(result, TiltAnalysisResult)

    def test_summary_is_string(self):
        """The summary field should be a non-empty string."""
        image = np.full((48, 48), 0.01, dtype=np.float32)
        yy, xx = np.mgrid[0:48, 0:48]
        sigma = 2.0
        g = np.exp(-((xx - 24) ** 2 + (yy - 24) ** 2) / (2 * sigma ** 2))
        image += g.astype(np.float32) * 0.5
        result = analyze_tilt(image)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_coma_detection_flag(self):
        """coma_detected should be a boolean."""
        image = np.full((64, 64), 0.01, dtype=np.float32)
        yy, xx = np.mgrid[0:64, 0:64]
        g = np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 2.0 ** 2))
        image += g.astype(np.float32) * 0.5
        result = analyze_tilt(image)
        assert isinstance(result.coma_detected, bool)
        assert isinstance(result.astigmatism_detected, bool)
        assert isinstance(result.tilt_detected, bool)
